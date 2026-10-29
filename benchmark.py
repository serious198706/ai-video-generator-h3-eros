"""Real local generation; no SQS, S3 or callbacks, no placeholder mode."""
import argparse
import json
import platform
import secrets
import subprocess
import sys
import threading
import time
from pathlib import Path
from h3_service.config import Settings
from h3_service.contracts.models import GenerateRequest, frame_count
from h3_service.infer.engine import Engine
from h3_service.infer.manifest import sha256, MODEL_ID, BASE_REVISION, DIFFUSERS_REVISION
from h3_service.media.images import prepare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--last-image', type=Path)
    parser.add_argument('--prompt-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration', type=float, default=5)
    parser.add_argument('--resolution', default='540p')
    parser.add_argument('--steps', type=int, default=6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--audio', action='store_true')
    args = parser.parse_args()
    request = GenerateRequest(image=str(args.image), prompt=args.prompt_file.read_text(), duration=args.duration,
                              resolution=args.resolution, steps=args.steps, seed=args.seed, audio=args.audio)
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    output = args.output.resolve()
    settings = Settings()
    engine = Engine(settings)
    cancelled = threading.Event()
    report = {'model': MODEL_ID, 'base_revision': BASE_REVISION, 'diffusers_revision': DIFFUSERS_REVISION,
              'python': platform.python_version(), 'image_sha256': sha256(args.image),
              'prompt_sha256': sha256(args.prompt_file), 'status': 'running'}
    try:
        started = time.monotonic()
        engine.preflight(cancelled)
        report['preflight_s'] = time.monotonic() - started
        first, last, width, height = prepare(args.image, args.last_image, request.resolution, output)
        job = {'prompt': request.prompt, 'first': str(first), 'last': str(last) if last else None,
               'width': width, 'height': height, 'num_frames': frame_count(request.duration),
               'steps': request.steps, 'seed': request.seed, 'audio': request.audio}
        report['resolved'] = {k: v for k, v in job.items() if k not in ('prompt', 'first', 'last')}
        _, metrics = engine.generate(job, output, cancelled, lambda stage: print(stage, flush=True))
        report.update(status='succeeded', **metrics)
        return 0
    except Exception as exc:
        report.update(status='failed', error=type(exc).__name__)
        print(f'Benchmark failed: {type(exc).__name__}', file=sys.stderr)
        return 1
    finally:
        # The temporary prompt and embedding files are not part of the benchmark artifact.
        for name in ('job.json', 'conditioning.safetensors'):
            (output / name).unlink(missing_ok=True)
        (output / 'report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    sys.exit(main())
