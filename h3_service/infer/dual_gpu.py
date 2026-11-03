"""Two model owners with bounded task overlap and serialized per-GPU work."""
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from h3_service.infer.engine import Engine, GenerationError, LeaseLost
from h3_service.infer.resident import ResidentEngine, atomic
from h3_service.infer.manifest import MODEL_ID

LOG = logging.getLogger(__name__)


class DualGpuEngine(Engine):
    def __init__(self, settings, progress=None):
        super().__init__(settings, progress)
        self.owners = {}
        self.locks = {'condition': threading.Lock(), 'generate': threading.Lock()}
        self.healthy = False

    def preflight(self, cancelled, **kwargs):
        if not self.settings.sr_model.is_file():
            raise FileNotFoundError('H3_SR_MODEL missing; run deploy.sh with H3_MODEL_LIFETIME=dual_gpu')
        import hashlib
        from h3_service.infer.download_sr import SHA256
        if hashlib.sha256(self.settings.sr_model.read_bytes()).hexdigest() != SHA256:
            raise ValueError('H3_SR_MODEL checkpoint SHA256 mismatch')
        kwargs['probe_models'] = False
        subprocess.run([sys.executable, '-c',
            'import torch; assert torch.cuda.device_count() == 2, "dual_gpu requires two distinct accessible CUDA devices"'],
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': self.settings.condition_gpu + ',' + self.settings.generate_gpu},
            check=True, timeout=60)
        super().preflight(cancelled, **kwargs)
        try:
            for role, gpu in [('condition', self.settings.condition_gpu), ('generate', self.settings.generate_gpu)]:
                owner = ResidentEngine(self.settings, self.progress)
                self.owners[role] = owner
                owner.ipc = tempfile.TemporaryDirectory(prefix=role + '-', dir=self.settings.work_dir)
                environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu,
                    'H3_MODEL_LIFETIME': 'dual_gpu', 'H3_PROCESS_ROLE': role,
                    'H3_MODEL_DIR': str(self.settings.model_dir.resolve()),
                    'H3_EROS_FILE': str(self.settings.eros_file.resolve()),
                    'H3_TRANSFORMER_CACHE': str(self.settings.transformer_cache.resolve()),
                    'H3_SR_MODEL': str(self.settings.sr_model.resolve()),
                    'H3_CONVERTED_CACHE': '1' if self.settings.converted_cache else '0',
                    'H3_RESIDENT_MIN_RAM_GIB': str(self.settings.resident_min_ram_gib),
                    'H3_DUAL_MIN_FREE_GIB': str(self.settings.dual_min_free_gib),
                    'H3_ATTENTION': self.settings.attention,
                    'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
                owner.process = subprocess.Popen(
                    [sys.executable, '-u', '-m', 'h3_service.infer.resident', owner.ipc.name],
                    env=environment, cwd=str(Path(__file__).resolve().parents[2]),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
                self.progress(f'dual GPU loading role={role} gpu={gpu} pid={owner.process.pid}')
                owner.wait_reply(Path(owner.ipc.name) / 'ready.json', cancelled)
                owner.healthy = True
            self.healthy = True
        except BaseException:
            self.close()
            raise

    def phase_call(self, role, mode, job, cancelled, stage):
        task = Path(job['directory']).name.split('-', 1)[0]
        queued = time.monotonic()
        stage(mode + '_waiting_gpu')
        lock = self.locks[role]
        while not lock.acquire(timeout=0.25):
            if cancelled.is_set() or not self.healthy:
                raise LeaseLost('dual GPU wait cancelled')
        try:
            if cancelled.is_set() or not self.healthy:
                raise LeaseLost('dual GPU execution cancelled')
            wait_s = time.monotonic() - queued
            stage({'condition': 'conditioning', 'generate': 'denoise_decode', 'upscale': 'upscaling'}[mode])
            owner = self.owners[role]
            gpu = self.settings.condition_gpu if role == 'condition' else self.settings.generate_gpu
            owner.progress = lambda message: LOG.info('task=%s gpu=%s role=%s phase=%s %s', task, gpu, role, mode, message)
            ident = uuid.uuid4().hex
            started = time.monotonic()
            atomic(Path(owner.ipc.name) / 'command.json', {'id': ident, 'mode': mode, 'job': job})
            reply = Path(owner.ipc.name) / (ident + '.json')
            result = owner.wait_reply(reply, cancelled)
            reply.unlink()
            result.update(wall_s=time.monotonic() - started, gpu_wait_s=wait_s, gpu=gpu)
            LOG.info('phase complete task=%s phase=%s gpu=%s timing=%s', task, mode, gpu, json.dumps(result))
            return result
        except BaseException:
            self.healthy = False
            raise
        finally:
            lock.release()

    def generate(self, job, directory, cancelled, stage):
        job = {**job, 'directory': str(directory.resolve())}
        timings = {}
        timings['condition'] = self.phase_call('condition', 'condition', job, cancelled, stage)
        timings['generate'] = self.phase_call('generate', 'generate', job, cancelled, stage)
        return directory / 'output.mp4', {'model': MODEL_ID, 'timing': timings}

    def upscale(self, source, directory, cancelled, stage):
        output = directory / 'upscaled.mp4'
        metrics = self.phase_call('condition', 'upscale', {'directory': str(directory.resolve()),
            'source': str(source.resolve()), 'output': str(output.resolve())}, cancelled, stage)
        return output, metrics

    def close(self):
        self.healthy = False
        for owner in self.owners.values():
            owner.close()
