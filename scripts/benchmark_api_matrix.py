"""Nine real Lambda API jobs, sequential execution and resumable timing reports."""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, report):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.chmod(0o600)
    temporary.replace(path)
    fields = ['duration', 'resolution', 'task_id', 'status', 'end_to_end_s',
              'observed_queue_s', 'generation_s', 'upscale_s', 'width', 'height',
              'actual_duration', 'error']
    with path.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(report['cases'])


def summarize(case, state):
    case['status'] = state.get('status', 'unknown')
    case['stage'] = state.get('stage')
    case['error'] = state.get('error')
    case['timing'] = state.get('timing', {})
    result = state.get('result') or {}
    for name in ('width', 'height', 'actual_duration', 'num_frames', 'audio_present'):
        if name in result:
            case[name] = result[name]
    case['actual_duration'] = result.get('actual_duration', result.get('duration'))
    case['video_url'] = state.get('video_url')
    timing = case['timing']
    if 'generation_s' in timing:
        case['generation_s'] = timing.get('conditioning_s', 0) + timing['generation_s']
        case['upscale_s'] = timing.get('upscale_s')
    else:
        case['generation_s'] = sum(
            (timing.get(phase) or {}).get('wall_s', 0) for phase in ('condition', 'generate')
        ) or None
        case['upscale_s'] = (timing.get('upscale') or {}).get('wall_s')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api-base', default=os.getenv('H3_TEST_API_BASE', ''),
                        help='existing Lambda URL, including any API Gateway stage; no /v1 suffix')
    parser.add_argument('--image-url', required=True, help='public HTTPS image, identical for all nine jobs')
    prompts = parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument('--prompt-file', type=Path)
    prompts.add_argument('--prompt')
    parser.add_argument('--output', required=True, type=Path, help='report directory; reuse to resume')
    parser.add_argument('--steps', type=int, default=4, choices=range(4, 9))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--audio', action='store_true')
    parser.add_argument('--durations', nargs='+', type=int, choices=(5, 10, 15), default=[5, 10, 15])
    parser.add_argument('--resolutions', nargs='+', choices=('480p', '720p', '1080p'),
                        default=['480p', '720p', '1080p'])
    parser.add_argument('--poll-seconds', type=float, default=5)
    parser.add_argument('--task-timeout', type=float, default=3600)
    args = parser.parse_args()
    if not args.api_base.startswith('https://') or not args.image_url.startswith('https://'):
        parser.error('--api-base and --image-url must use HTTPS')
    if args.poll_seconds <= 0 or args.task_timeout <= 0:
        parser.error('poll interval and task timeout must be positive')
    prompt = (args.prompt if args.prompt is not None else args.prompt_file.read_text()).strip()
    if not prompt:
        parser.error('prompt file is empty')
    identity = hashlib.sha256(json.dumps(
        [args.api_base, args.image_url, prompt, args.steps, args.seed, args.audio],
        ensure_ascii=False,
    ).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    report_path = args.output / 'report.json'
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report.get('input_fingerprint') != identity:
            parser.error('existing report uses different inputs; choose a new output directory')
    else:
        report = {'created_at': utc(), 'input_fingerprint': identity,
                  'steps': args.steps, 'seed': args.seed, 'audio': args.audio,
                  'cases': [{'duration': duration, 'resolution': resolution, 'status': 'not_submitted'}
                            for duration in args.durations for resolution in args.resolutions]}
        save(report_path, report)
    session = requests.Session()
    # Optional API key comes from the environment and is never stored in reports.
    if os.getenv('H3_TEST_API_KEY'):
        session.headers['x-api-key'] = os.environ['H3_TEST_API_KEY']
    try:
        for index, case in enumerate(report['cases'], 1):
            label = f"[{index}/{len(report['cases'])}] {case['duration']}s/{case['resolution']}"
            if case['status'] in {'succeeded', 'failed'}:
                print(f"{label} already {case['status']}; skipped", flush=True)
                continue
            if not case.get('task_id'):
                if case['status'] == 'submission_uncertain':
                    raise RuntimeError(f'{label}: prior submission uncertain; locate task ID before retrying')
                case.update(status='submission_uncertain', submitted_at=utc())
                save(report_path, report)
                print(f'{label} submitting', flush=True)
                # A POST timeout may mean accepted. Never automatically retry POST.
                response = session.post(args.api_base.rstrip('/') + '/v1/generate', json={
                    'image': args.image_url, 'prompt': prompt, 'duration': case['duration'],
                    'resolution': case['resolution'], 'steps': args.steps,
                    'seed': args.seed, 'audio': args.audio,
                }, timeout=(10, 60), allow_redirects=False)
                if response.status_code != 202:
                    raise RuntimeError(f'{label}: submission HTTP {response.status_code}; inspect API before retry')
                data = response.json()
                task_id = data.get('task_id') or data.get('id')
                if not isinstance(task_id, str) or len(task_id) != 32 or any(c not in '0123456789abcdef' for c in task_id):
                    raise RuntimeError('submission response missing a valid task ID')
                case.update(task_id=task_id, status='queued')
                save(report_path, report)
            print(f"{label} polling task={case['task_id']}", flush=True)
            deadline = time.monotonic() + args.task_timeout
            previous = None
            while time.monotonic() < deadline:
                try:
                    response = session.get(args.api_base.rstrip('/') + '/v1/tasks/' + case['task_id'],
                                           timeout=(10, 30), allow_redirects=False)
                    response.raise_for_status()
                    state = response.json()
                except (requests.RequestException, ValueError) as exc:
                    print(f'{label} poll failed type={type(exc).__name__}; retrying GET', flush=True)
                    time.sleep(args.poll_seconds)
                    continue
                summarize(case, state)
                elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(case['submitted_at'])).total_seconds()
                if state.get('status') in {'pending', 'generating', 'succeeded'} and 'observed_queue_s' not in case:
                    case['observed_queue_s'] = round(elapsed, 3)
                phase = (case['status'], case.get('stage'))
                if phase != previous:
                    print(f'{label} status={phase[0]} stage={phase[1]} elapsed={elapsed:.1f}s', flush=True)
                    previous = phase
                if case['status'] in {'succeeded', 'failed'}:
                    case.update(completed_at=utc(), end_to_end_s=round(elapsed, 3))
                    save(report_path, report)
                    break
                save(report_path, report)
                time.sleep(args.poll_seconds)
            else:
                raise TimeoutError(f'{label}: poll timeout; task remains active; rerun with same output to resume')
        print('\nduration resolution status       total_s generation_s upscale_s', flush=True)
        for case in report['cases']:
            print(f"{case['duration']:>8} {case['resolution']:>10} {case['status']:<12} "
                  f"{case.get('end_to_end_s', '-')} {case.get('generation_s', '-')} {case.get('upscale_s', '-')}")
        return 0 if all(case['status'] == 'succeeded' for case in report['cases']) else 1
    except (Exception, KeyboardInterrupt) as exc:
        report['last_error'] = type(exc).__name__
        save(report_path, report)
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    finally:
        session.close()
        print(f'Reports: {report_path} and {report_path.with_suffix(".csv")}', flush=True)


if __name__ == '__main__':
    sys.exit(main())
