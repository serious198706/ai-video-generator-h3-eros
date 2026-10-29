"""Resident model owner, isolated from the SQS process; private filesystem IPC."""
import gc
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from h3_service.infer.engine import Engine, GenerationError, LeaseLost
from h3_service.infer.manifest import MODEL_ID


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False))
    temporary.chmod(0o600)
    temporary.replace(path)


def available_memory(meminfo=Path('/proc/meminfo'), root=Path('/sys/fs/cgroup')):
    """Account for the cgroup limit and reclaimable inactive file cache on Linux."""
    info = {line.split(':')[0]: int(line.split()[1]) * 1024
            for line in meminfo.read_text().splitlines() if line.split()[1].isdigit()}
    available = info['MemAvailable']
    if (root / 'memory.max').exists() and (root / 'memory.max').read_text().strip() != 'max':
        limit = int((root / 'memory.max').read_text())
        current = int((root / 'memory.current').read_text())
        stats = dict(line.split() for line in (root / 'memory.stat').read_text().splitlines())
        available = min(available, max(0, limit - current + int(stats.get('inactive_file', '0'))))
    return available


class ResidentModels:
    def __init__(self, settings, emit):
        from h3_service.infer.cpu_budget import configure_environment, configure_torch
        configure_environment()
        import torch
        emit(f'CPU intra-op thread budget: {configure_torch()}')
        from h3_service.infer.blocks import load_pipeline
        from h3_service.infer.eros import apply_eros
        from h3_service.infer.converted_cache import load_generator
        self.settings, self.emit = settings, emit
        self.pipes = {}
        # Initial budget is deliberately conservative and configurable; no silent cold fallback.
        self.check_memory(settings.resident_min_ram_gib)
        for mode, conditioner in [('condition', True), ('generate', False)]:
            emit(f'loading resident {mode} into CPU memory')
            if conditioner:
                pipe = load_pipeline(settings.model_dir, conditioner=True, progress=emit)
            else:
                pipe, self.conversion = load_generator(settings, emit, load_pipeline, apply_eros)
                pipe.transformer.set_attention_backend(settings.attention)
            self.pipes[mode] = pipe
            self.check_memory(16)
            emit(f'probing resident {mode} GPU placement')
            if torch.cuda.mem_get_info()[0] / 1024**3 < settings.min_free_gib:
                raise RuntimeError('insufficient free GPU memory')
            pipe.to('cuda')
            torch.cuda.synchronize()
            self.offload(pipe)
        emit('resident models ready on CPU; Eros conversion completed once')

    def check_memory(self, required):
        free = available_memory() / 1024**3
        self.emit(f'RAM effective available={free:.1f} GiB, required headroom={required:.1f} GiB')
        if free < required:
            raise RuntimeError('insufficient effective RAM for resident models')

    def offload(self, pipe):
        import torch
        self.emit('returning model weights to CPU')
        pipe.to('cpu')
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

    def run(self, mode, job):
        from h3_service.infer.phase import execute
        self.check_memory(16)
        pipe = self.pipes[mode]
        if mode == 'generate':
            atomic(Path(job['directory']) / 'conversion.json', self.conversion)
        # execute's state/tensors go out of scope before offload; on error the process exits.
        result = execute(mode, job, self.settings, progress=self.emit, loaded_pipe=pipe)
        started = time.monotonic()
        self.offload(pipe)
        result.update(offload_s=time.monotonic() - started, resident=True)
        return result


def serve(directory):
    from h3_service.config import Settings
    progress = directory / 'progress.json'
    def emit(stage):
        atomic(progress, {'stage': stage})
    reply = directory / 'ready.json'
    try:
        emit('importing resident model dependencies')
        models = ResidentModels(Settings(), emit)
        atomic(reply, {'ok': True})
        command = directory / 'command.json'
        while True:
            if not command.exists():
                time.sleep(0.1)
                continue
            request = json.loads(command.read_text())
            command.unlink()
            reply = directory / (request['id'] + '.json')
            result = models.run(request['mode'], request['job'])
            atomic(reply, {'ok': True, 'result': result})
    except Exception as exc:
        import traceback
        atomic(reply, {'ok': False, 'error': type(exc).__name__,
                       'trace': [{'file': Path(f.filename).name, 'line': f.lineno, 'function': f.name}
                                 for f in traceback.extract_tb(exc.__traceback__)]})
        return 1


class ResidentEngine(Engine):
    def __init__(self, settings, progress=None):
        super().__init__(settings, progress)
        self.process = None
        self.ipc = None
        self.healthy = False

    def close(self):
        self.healthy = False
        proc = self.process
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        self.process = None
        if self.ipc:
            self.ipc.cleanup()
            self.ipc = None

    def wait_reply(self, path, cancelled):
        started = last = time.monotonic()
        previous = None
        while True:
            if cancelled.is_set():
                raise LeaseLost('resident execution cancelled')
            if path.exists():
                response = json.loads(path.read_text())
                if not response['ok']:
                    self.progress(f'Resident process failure: {response}')
                    raise GenerationError('resident_failed')
                return response.get('result')
            if self.process.poll() is not None:
                raise GenerationError('resident_process_exited')
            if time.monotonic() - started > self.settings.phase_timeout:
                raise GenerationError('phase_timeout')
            status = Path(self.ipc.name) / 'progress.json'
            stage = json.loads(status.read_text())['stage'] if status.exists() else 'starting Python'
            current = time.monotonic()
            if stage != previous or current - last >= 10:
                self.progress(f'resident: {stage}; elapsed={current - started:.1f}s, pid={self.process.pid}')
                previous, last = stage, current
            cancelled.wait(0.25)

    def preflight(self, cancelled, **kwargs):
        if self.process is not None:
            raise RuntimeError('resident engine already initialized')
        # Hashes and runtime checked once, then load the actual reusable objects.
        kwargs['probe_models'] = False
        super().preflight(cancelled, **kwargs)
        self.ipc = tempfile.TemporaryDirectory(prefix='resident-', dir=self.settings.work_dir)
        environment = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                       'H3_MODEL_DIR': str(self.settings.model_dir.resolve()),
                       'H3_EROS_FILE': str(self.settings.eros_file.resolve()),
                       'H3_ATTENTION': self.settings.attention,
                       'H3_MIN_FREE_GIB': str(self.settings.min_free_gib),
                       'H3_RESIDENT_MIN_RAM_GIB': str(self.settings.resident_min_ram_gib)}
        environment.update(H3_TRANSFORMER_CACHE=str(self.settings.transformer_cache.resolve()),
                           H3_CONVERTED_CACHE='1' if self.settings.converted_cache else '0',
                           H3_VERIFY_MODE=self.settings.verify_mode)
        try:
            self.process = subprocess.Popen([sys.executable, '-u', '-m', 'h3_service.infer.resident', self.ipc.name],
                cwd=str(Path(__file__).resolve().parents[2]), env=environment,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            self.wait_reply(Path(self.ipc.name) / 'ready.json', cancelled)
            self.healthy = True
            self.progress(f'Resident models ready on CPU; pid={self.process.pid}; subsequent jobs reuse loaded weights')
        except BaseException:
            self.close()
            raise

    def generate(self, job, directory, cancelled, stage):
        if not self.healthy:
            raise GenerationError('resident_not_ready')
        timings = {}
        try:
            for mode, label in [('condition', 'conditioning'), ('generate', 'denoise_decode')]:
                stage(label)
                ident = uuid.uuid4().hex
                started = time.monotonic()
                atomic(Path(self.ipc.name) / 'command.json', {'id': ident, 'mode': mode,
                       'job': {**job, 'directory': str(directory.resolve())}})
                reply = Path(self.ipc.name) / (ident + '.json')
                timings[mode] = self.wait_reply(reply, cancelled)
                reply.unlink()
                timings[mode]['wall_s'] = time.monotonic() - started
            return directory / 'output.mp4', {'model': MODEL_ID, 'timing': timings}
        except BaseException:
            self.close()
            raise


if __name__ == '__main__':
    sys.exit(serve(Path(sys.argv[1])))
