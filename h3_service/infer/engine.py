import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import importlib.metadata
import logging
from pathlib import Path
from contextlib import nullcontext
from h3_service.infer.manifest import verify, MODEL_ID


class LeaseLost(RuntimeError):
    pass


class GenerationError(RuntimeError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


class Engine:
    def __init__(self, settings, progress=None):
        self.settings = settings
        self.progress = progress or (lambda message: logging.getLogger(__name__).info('%s', message))

    def phase(self, mode, directory, cancelled, job=None):
        report = directory / f'{mode}-report.json'
        progress_file = directory / f'{mode}-progress.json'
        progress_file.unlink(missing_ok=True)
        report.unlink(missing_ok=True)
        command = [sys.executable, '-u', '-m', 'h3_service.infer.phase', mode, '--report', str(report),
                   '--progress-file', str(progress_file)]
        if job:
            command += ['--job', str(job)]
        environment = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                       'H3_MODEL_DIR': str(self.settings.model_dir.resolve()),
                       'H3_EROS_FILE': str(self.settings.eros_file.resolve()),
                       'H3_ATTENTION': self.settings.attention, 'H3_MIN_FREE_GIB': str(self.settings.min_free_gib)}
        # Dependency logs can contain request inputs; keep phase stdout/stderr out of general logs.
        proc = subprocess.Popen(command, env=environment, cwd=str(Path(__file__).resolve().parents[2]),
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        deadline = time.monotonic() + self.settings.phase_timeout
        started = last_notice = time.monotonic()
        last_stage = None
        self.progress(f'{mode}: subprocess started, pid={proc.pid}; waiting for Python imports')
        try:
            while proc.poll() is None:
                if cancelled.is_set():
                    raise LeaseLost('execution ownership lost')
                if time.monotonic() > deadline:
                    raise GenerationError('phase_timeout')
                stage = None
                if progress_file.exists():
                    try:
                        stage = json.loads(progress_file.read_text()).get('stage')
                    except (OSError, ValueError):
                        pass  # Atomic replace normally prevents this; tolerate transient filesystem reads.
                current = time.monotonic()
                if (stage and stage != last_stage) or current - last_notice >= 10:
                    self.progress(f'{mode}: {stage or last_stage or "Python imports/startup"}; '
                                  f'elapsed={current - started:.1f}s, pid={proc.pid}')
                    last_stage, last_notice = stage or last_stage, current
                cancelled.wait(0.25)
            if cancelled.is_set():
                raise LeaseLost('execution ownership lost')
            if proc.returncode:
                if report.exists():
                    diagnostic = json.loads(report.read_text())
                    logging.getLogger(__name__).error('phase=%s failure=%s', mode, diagnostic)
                raise GenerationError('conditioning_failed' if 'condition' in mode else 'generate_failed')
            result = json.loads(report.read_text())
            self.progress(f'{mode}: completed in {time.monotonic() - started:.1f}s')
            return result
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()

    def preflight(self, cancelled, *, probe_models=True, diagnostic_dir=None):
        self.progress('Preflight: checking dependencies and ffprobe')
        if not shutil.which('ffprobe'):
            raise RuntimeError('ffprobe is required')
        from h3_service.infer.manifest import DIFFUSERS_REVISION
        installed = importlib.metadata.distribution('diffusers')
        origin = json.loads(installed.read_text('direct_url.json') or '{}')
        if origin.get('vcs_info', {}).get('commit_id') != DIFFUSERS_REVISION:
            raise RuntimeError('diffusers commit mismatch; install requirements-gpu.txt in the H3 venv')
        for name, version in [('torch', '2.11.0+cu130'), ('torchvision', '0.26.0+cu130'),
                              ('transformers', '5.8.0'), ('accelerate', '1.14.0')]:
            if importlib.metadata.version(name) != version:
                raise RuntimeError(f'{name} version mismatch')
        self.settings.work_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        if diagnostic_dir is not None:
            diagnostic_dir = Path(diagnostic_dir)
            diagnostic_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        context = nullcontext(diagnostic_dir) if diagnostic_dir is not None else tempfile.TemporaryDirectory(dir=self.settings.work_dir)
        with context as temp:
            self.phase('probe-runtime', Path(temp), cancelled)
            if self.settings.verify_mode not in {'cached', 'full'}:
                raise ValueError('H3_VERIFY_MODE must be cached or full')
            verify(self.settings, progress=self.progress, cancelled=cancelled,
                   cache_path=self.settings.work_dir / 'weight-verification.json' if self.settings.verify_mode == 'cached' else None)
            if probe_models:
                for mode in ('probe-condition', 'probe-generate'):
                    self.phase(mode, Path(temp), cancelled)
            else:
                self.progress('Preflight: separate model-load probes omitted; actual generation loads and validates both models')

    def generate(self, job, directory, cancelled, stage):
        file = directory / 'job.json'
        file.write_text(json.dumps({**job, 'directory': str(directory.resolve())}))
        file.chmod(0o600)
        timings = {}
        for mode, label in [('condition', 'conditioning'), ('generate', 'denoise_decode')]:
            stage(label)
            start = time.monotonic()
            timings[mode] = self.phase(mode, directory, cancelled, file)
            timings[mode]['wall_s'] = time.monotonic() - start
            self.progress(f'phase timing phase={mode} metrics={json.dumps(timings[mode], sort_keys=True)}')
        return directory / 'output.mp4', {'model': MODEL_ID, 'timing': timings}
