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
    parser.add_argument('--output', type=Path, required=True, help='new output directory, not an MP4 filename')
    parser.add_argument('--load-probes', action='store_true', help='also load both models as separate preflight probes (loads twice)')
    parser.add_argument('--resident', action='store_true', help='use the server resident model lifecycle')
    parser.add_argument('--runs', type=int, default=1, help='consecutive generations; >1 requires --resident')
    parser.add_argument('--duration', type=float, default=5)
    parser.add_argument('--resolution', default='540p')
    parser.add_argument('--steps', type=int, default=6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--audio', action='store_true')
    args = parser.parse_args()
    if args.runs < 1 or (args.runs > 1 and not args.resident):
        parser.error('--runs must be positive; multiple runs require --resident')
    total_started = time.monotonic()

    def log(message, *, error=False):
        elapsed = time.monotonic() - total_started
        print(f'[{elapsed:8.1f}s] {message}', file=sys.stderr if error else sys.stdout, flush=True)

    image = args.image.expanduser().resolve()
    last_image = args.last_image.expanduser().resolve() if args.last_image else None
    prompt_file = args.prompt_file.expanduser().resolve()
    output = args.output.expanduser().resolve()
    report = {'model': MODEL_ID, 'base_revision': BASE_REVISION, 'diffusers_revision': DIFFUSERS_REVISION,
              'python': platform.python_version(), 'status': 'running'}
    output_created = False
    engine = None
    try:
        log('H3 benchmark starting')
        log(f'Model: {MODEL_ID}')
        log(f'Input image: {image}')
        log(f'Last image: {last_image or "(none)"}')
        log(f'Prompt file: {prompt_file}')
        log(f'Output directory: {output}')
        log(f'Request: resolution={args.resolution}, duration={args.duration}s, steps={args.steps}, '
            f'seed={args.seed}, audio={args.audio}')
        request = GenerateRequest(image=str(image), prompt=prompt_file.read_text(), duration=args.duration,
                                  resolution=args.resolution, steps=args.steps, seed=args.seed, audio=args.audio)
        output.mkdir(parents=True, exist_ok=False, mode=0o700)
        output_created = True
        if output.suffix.lower() == '.mp4':
            log('Note: --output is a directory; the video will be written to ' + str(output / 'output.mp4'))
        settings = Settings()
        if args.resident:
            from h3_service.infer.resident import ResidentEngine
            engine = ResidentEngine(settings, progress=log)
        else:
            engine = Engine(settings, progress=log)
        cancelled = threading.Event()
        report.update(image_sha256=sha256(image), prompt_sha256=sha256(prompt_file))

        log('Preflight started: runtime/dependency checks, full weight hashes; '
            f'separate model-load probes={args.load_probes}')
        started = time.monotonic()
        engine.preflight(cancelled, probe_models=args.load_probes, diagnostic_dir=output / 'preflight')
        report['preflight_s'] = time.monotonic() - started
        log(f'Preflight passed in {report["preflight_s"]:.1f}s')

        log('Preparing input image(s)')
        first, last, width, height = prepare(image, last_image, request.resolution, output)
        job = {'prompt': request.prompt, 'first': str(first), 'last': str(last) if last else None,
               'width': width, 'height': height, 'num_frames': frame_count(request.duration),
               'steps': request.steps, 'seed': request.seed, 'audio': request.audio}
        report['resolved'] = {k: v for k, v in job.items() if k not in {'prompt', 'first', 'last'}}
        log(f'Resolved: canvas={width}x{height}, frames={job["num_frames"]}, fps=24, '
            f'actual_duration={job["num_frames"] / 24:.3f}s')

        stage_name = None
        stage_started = None

        def on_stage(stage):
            nonlocal stage_name, stage_started
            now = time.monotonic()
            if stage_name is not None:
                log(f'Stage finished: {stage_name} ({now - stage_started:.1f}s)')
            stage_name, stage_started = stage, now
            log(f'Stage started: {stage}; resident={args.resident}')

        measurements = []
        for run in range(args.runs):
            stage_name = stage_started = None
            run_directory = output if args.runs == 1 else output / f'run-{run + 1:02d}'
            run_directory.mkdir(exist_ok=True, mode=0o700)
            log(f'Generation {run + 1}/{args.runs}; resident={args.resident}')
            run_started = time.monotonic()
            video_path, metrics = engine.generate(job, run_directory, cancelled, on_stage)
            metrics['generation_wall_s'] = time.monotonic() - run_started
            if stage_name is not None:
                log(f'Stage finished: {stage_name} ({time.monotonic() - stage_started:.1f}s)')
            log(f'Generation {run + 1}/{args.runs} complete in {metrics["generation_wall_s"]:.3f}s; '
                f'phase timing={json.dumps(metrics.get("timing", {}), sort_keys=True)}')
            measurements.append(metrics)
            report['runs'] = measurements
        for phase, timing in metrics.get('timing', {}).items():
            log(f'{phase} timing: load={timing.get("load_s", 0):.1f}s, '
                f'compute_export={timing.get("compute_export_s", 0):.1f}s, '
                f'wall={timing.get("wall_s", 0):.1f}s, '
                f'peak_allocated={timing.get("peak_allocated_gib", 0):.2f} GiB, '
                f'peak_reserved={timing.get("peak_reserved_gib", 0):.2f} GiB')
        report.update(status='succeeded', **metrics)
        log(f'Generation succeeded: {video_path} ({video_path.stat().st_size / 1024**2:.1f} MiB)')
        return 0
    except Exception as exc:
        report.update(status='failed', error=type(exc).__name__)
        log(f'Benchmark failed: {type(exc).__name__}; partial artifacts retained in {output}', error=True)
        return 1
    finally:
        if engine is not None:
            close = getattr(engine, 'close', None)
            if close:
                close()
        report['total_s'] = time.monotonic() - total_started
        # The temporary prompt and embedding files are not part of the benchmark artifact.
        if output_created:
            for folder in [output, *output.glob('run-*')]:
                for name in ('job.json', 'conditioning.safetensors'):
                    (folder / name).unlink(missing_ok=True)
            report_path = output / 'report.json'
            report_path.write_text(json.dumps(report, indent=2))
            log(f'Report written: {report_path}')
        log(f'Benchmark finished: status={report["status"]}, total={report["total_s"]:.1f}s')


if __name__ == '__main__':
    sys.exit(main())
