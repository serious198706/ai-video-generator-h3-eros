import contextvars
import hashlib
import json
import logging
import math
import os
import threading
import time
from pathlib import Path
from h3_service.infer.engine import Engine, GenerationError, LeaseLost
from h3_service.infer.manifest import fingerprint

LOG = logging.getLogger(__name__)
request_context = contextvars.ContextVar('simulation_request')
# Placeholder container header only; deliberately contains no playable video frames.
PLACEHOLDER = b'\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2'


class SimulatedEngine(Engine):
    def __init__(self, settings, progress=None):
        super().__init__(settings, progress)
        self.profile = os.getenv('H3_SIM_PROFILE', '6000-maxq')
        self.scale = float(os.getenv('H3_SIM_TIME_SCALE', '1'))
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError('H3_SIM_TIME_SCALE must be finite and positive')
        profiles = json.loads(Path(__file__).with_name('profiles.json').read_text())
        if self.profile not in profiles:
            raise ValueError('H3_SIM_PROFILE must be 6000-s or 6000-maxq')
        self.table = profiles[self.profile]
        self.locks = {'condition': threading.Lock(), 'generate': threading.Lock()}
        self.report_lock = threading.Lock()
        self.healthy = False

    @staticmethod
    def fingerprint(request, seed):
        return hashlib.sha256(('simulation-v1:' + fingerprint(request, seed)).encode()).hexdigest()

    def preflight(self, cancelled, **kwargs):
        if cancelled.is_set():
            raise LeaseLost('simulation cancelled')
        self.settings_image_size()
        self.healthy = True
        self.progress(f'SIMULATION ready profile={self.profile} time_scale={self.scale}; no GPU, torch or model files used')

    def recipe(self):
        request = request_context.get()
        key = f'{request.duration:g}/{request.resolution}'
        if key not in self.table or request.steps != 4 or request.audio:
            raise GenerationError('simulation_requires_5_10_15s_720p_1080p_4steps_no_audio')
        return self.table[key]

    def phase(self, role, name, cancelled, stage):
        recipe = self.recipe()
        queued = time.monotonic()
        stage(name + '_waiting_gpu')
        lock = self.locks[role]
        while not lock.acquire(timeout=.1):
            if cancelled.is_set() or not self.healthy:
                raise LeaseLost('simulation GPU wait cancelled')
        try:
            wait_s = time.monotonic() - queued
            stage({'condition': 'conditioning', 'generate': 'denoise_decode', 'upscale': 'upscaling'}[name])
            started = time.monotonic()
            remaining = recipe[name] * self.scale
            while remaining > 0:
                if cancelled.wait(min(remaining, 10)) or not self.healthy:
                    raise LeaseLost('simulation cancelled')
                remaining = recipe[name] * self.scale - (time.monotonic() - started)
                if remaining > 0:
                    LOG.info('simulation progress phase=%s remaining_s=%.1f', name, remaining)
            result = {'wall_s': time.monotonic() - started, 'gpu_wait_s': wait_s,
                      'gpu': '0' if role == 'condition' else '1',
                      'baseline_s': recipe[name], 'simulation': True}
            LOG.info('simulation phase complete phase=%s wall_s=%.3f gpu_wait_s=%.3f baseline_s=%.3f',
                     name, result['wall_s'], wait_s, recipe[name])
            return result
        finally:
            lock.release()

    def generate(self, job, directory, cancelled, stage):
        timing = {'condition': self.phase('condition', 'condition', cancelled, stage),
                  'generate': self.phase('generate', 'generate', cancelled, stage)}
        path = directory / 'output.mp4'
        path.write_bytes(PLACEHOLDER)
        return path, {'model': 'simulation-' + self.profile, 'simulation': True, 'timing': timing}

    def upscale(self, source, directory, cancelled, stage):
        timing = self.phase('condition', 'upscale', cancelled, stage)
        output = directory / 'upscaled.mp4'
        output.write_bytes(PLACEHOLDER)
        width, height = self.settings_image_size()
        timing.update(width=1080 if width <= height else 1920, height=1080 if height <= width else 1920)
        return output, timing

    @staticmethod
    def settings_image_size():
        try:
            width, height = (int(x) for x in os.getenv('H3_SIM_IMAGE_SIZE', '1080x1920').lower().split('x'))
        except ValueError:
            raise ValueError('H3_SIM_IMAGE_SIZE must be WIDTHxHEIGHT') from None
        if not (1 <= width <= 4096 and 1 <= height <= 4096):
            raise ValueError('simulation image dimensions must be between 1 and 4096')
        return width, height

    def close(self):
        self.healthy = False
