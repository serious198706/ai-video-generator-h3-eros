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
from h3_service.infer.manifest import verify, MODEL_ID


class LeaseLost(RuntimeError):
    pass


class GenerationError(RuntimeError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


class Engine:
    def __init__(self, settings):
        self.settings = settings

    def phase(self, mode, directory, cancelled, job=None):
        report = directory / f'{mode}-report.json'
        command = [sys.executable, '-m', 'h3_service.infer.phase', mode, '--report', str(report)]
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
        try:
            while proc.poll() is None:
                if cancelled.is_set():
                    raise LeaseLost('execution ownership lost')
                if time.monotonic() > deadline:
                    raise GenerationError('phase_timeout')
                cancelled.wait(0.25)
            if cancelled.is_set():
                raise LeaseLost('execution ownership lost')
            if proc.returncode:
                if report.exists():
                    diagnostic = json.loads(report.read_text())
                    logging.getLogger(__name__).error('phase=%s failure=%s', mode, diagnostic)
                raise GenerationError('conditioning_failed' if 'condition' in mode else 'generate_failed')
            return json.loads(report.read_text())
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()

    def preflight(self, cancelled):
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
        with tempfile.TemporaryDirectory(dir=self.settings.work_dir) as temp:
            self.phase('probe-runtime', Path(temp), cancelled)
            verify(self.settings)
            for mode in ('probe-condition', 'probe-generate'):
                self.phase(mode, Path(temp), cancelled)

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
        return directory / 'output.mp4', {'model': MODEL_ID, 'timing': timings}
