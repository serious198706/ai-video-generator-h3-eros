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
        from h3_service.infer.cpu_weights import CpuWeights, validate_cuda
        self.settings, self.emit = settings, emit
        self.pipes = {}
        self.cpu_weights = {}
        self.active_mode = None
        self.last_release = {}
        self.dual_int8 = settings.model_lifetime == 'dual_int8'
        self.quantization = {}
        # Initial budget is deliberately conservative and configurable; no silent cold fallback.
        self.check_memory(settings.resident_min_ram_gib)
        if self.dual_int8:
            from importlib.metadata import version
            emit(f'experimental dual INT8WO mode; torchao={version("torchao")}')
        else:
            emit('validating CPU parameter restore on CUDA before loading large models')
            emit('CPU parameter restore CUDA check: ' + validate_cuda())
        for mode, conditioner in [('condition', True), ('generate', False)]:
            emit(f'loading resident {mode} into CPU memory')
            if conditioner:
                pipe = load_pipeline(settings.model_dir, conditioner=True, progress=emit)
            else:
                pipe, self.conversion = load_generator(settings, emit, load_pipeline, apply_eros)
                pipe.transformer.set_attention_backend(settings.attention)
            self.pipes[mode] = pipe
            if self.dual_int8:
                from h3_service.infer.quantization import quantize_int8wo
                module = pipe.text_encoder if conditioner else pipe.transformer
                component = 'conditioner' if conditioner else 'generator'
                self.quantization[mode] = quantize_int8wo(module, component, emit)
            else:
                self.cpu_weights[mode] = CpuWeights(pipe)
                emit(f'CPU parameter storage retained mode={mode} GiB={self.cpu_weights[mode].bytes / 1024**3:.2f}')
            self.check_memory(16)
            if not self.dual_int8:
                emit(f'probing resident {mode} GPU placement')
                if torch.cuda.mem_get_info()[0] / 1024**3 < settings.min_free_gib:
                    raise RuntimeError('insufficient free GPU memory')
                from h3_service.infer.transfer import place_on_gpu
                place_on_gpu(pipe, emit)
                self.offload(pipe, mode)
        if self.dual_int8:
            from h3_service.infer.transfer import place_on_gpu
            for mode in ('condition', 'generate'):
                emit(f'placing quantized {mode} permanently on GPU')
                place_on_gpu(self.pipes[mode], emit)
                free, total = torch.cuda.mem_get_info()
                emit(f'dual INT8WO placement mode={mode} allocated_gib={torch.cuda.memory_allocated() / 1024**3:.2f} '
                     f'reserved_gib={torch.cuda.memory_reserved() / 1024**3:.2f} free_gib={free / 1024**3:.2f} '
                     f'total_gib={total / 1024**3:.2f}')
            free, _ = torch.cuda.mem_get_info()
            if free / 1024**3 < settings.dual_min_free_gib:
                emit(f'dual INT8WO rejected: free_gib={free / 1024**3:.2f} '
                     f'required_runtime_headroom_gib={settings.dual_min_free_gib:.2f}')
                raise RuntimeError('dual INT8WO models fit but leave insufficient runtime VRAM headroom')
            self.active_mode = 'dual_int8'
            emit('dual INT8WO models ready on GPU; no per-task model placement')
        else:
            emit('resident models ready on CPU; Eros conversion completed once')

    def check_memory(self, required):
        free = available_memory() / 1024**3
        self.emit(f'RAM effective available={free:.1f} GiB, required headroom={required:.1f} GiB')
        if free < required:
            raise RuntimeError('insufficient effective RAM for resident models')

    def offload(self, pipe, mode):
        import torch
        self.emit(f'restoring retained CPU weights mode={mode}; releasing GPU parameter copies')
        started = time.monotonic()
        torch.cuda.synchronize()
        sync_s = time.monotonic() - started
        release = self.cpu_weights[mode].restore()
        torch.cuda.synchronize()
        started = time.monotonic()
        gc.collect()
        gc_s = time.monotonic() - started
        started = time.monotonic()
        torch.cuda.empty_cache()
        self.last_release = {**release, 'synchronize_s': sync_s, 'gc_s': gc_s,
                             'empty_cache_s': time.monotonic() - started}
        self.emit(f'GPU release complete mode={mode} timing={json.dumps(self.last_release)}')
        self.active_mode = None

    def run(self, mode, job):
        from h3_service.infer.phase import execute
        self.check_memory(16)
        if self.dual_int8:
            pipe = self.pipes[mode]
            if mode == 'generate':
                atomic(Path(job['directory']) / 'conversion.json', self.conversion)
                atomic(Path(job['directory']) / 'quantization.json', self.quantization)
            result = execute(mode, job, self.settings, progress=self.emit, loaded_pipe=pipe,
                             already_on_gpu=True)
            result.update(offload_s=0.0, previous_model_release_s=0.0, release={},
                          previous_model_release={}, gpu_retained=True, resident=True,
                          quantization='dual_int8wo')
            return result
        eviction_s = 0.0
        eviction_details = {}
        if self.active_mode is not None and self.active_mode != mode:
            started = time.monotonic()
            self.offload(self.pipes[self.active_mode], self.active_mode)
            eviction_s = time.monotonic() - started
            eviction_details = dict(self.last_release)
        pipe = self.pipes[mode]
        if mode == 'generate':
            atomic(Path(job['directory']) / 'conversion.json', self.conversion)
        # execute's state/tensors go out of scope before offload; on error the process exits.
        result = execute(mode, job, self.settings, progress=self.emit, loaded_pipe=pipe,
                         already_on_gpu=self.active_mode == mode)
        self.active_mode = mode
        started = time.monotonic()
        if mode == 'condition':
            self.offload(pipe, mode)
            offload_s = time.monotonic() - started
        else:
            offload_s = 0.0
            self.emit('video ready; generator stays on GPU until the next conditioning phase')
        result.update(offload_s=offload_s, previous_model_release_s=eviction_s,
                      release=dict(self.last_release) if mode == 'condition' else {},
                      previous_model_release=eviction_details, gpu_retained=mode == 'generate', resident=True)
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
                       'H3_RESIDENT_MIN_RAM_GIB': str(self.settings.resident_min_ram_gib),
                       'H3_MODEL_LIFETIME': self.settings.model_lifetime,
                       'H3_DUAL_MIN_FREE_GIB': str(self.settings.dual_min_free_gib)}
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
                self.progress(f'phase timing phase={mode} metrics={json.dumps(timings[mode], sort_keys=True)}')
            return directory / 'output.mp4', {'model': MODEL_ID, 'timing': timings}
        except BaseException:
            self.close()
            raise


if __name__ == '__main__':
    sys.exit(serve(Path(sys.argv[1])))
