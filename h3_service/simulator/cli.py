"""Read-only AWS readiness checks and local queue latency reports."""
import argparse
import csv
import json
import math
import time
import urllib.request
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from h3_service.config import Settings


def task_rows(root, run=None, start=None):
    latest = {}
    for path in root.glob('*/*.json'):
        if path.name == 'summary.json':
            continue
        item = json.loads(path.read_text())
        simulation = item.get('simulation', {})
        if not simulation.get('enabled'):
            continue
        if run and item.get('producer', {}).get('run_id') != run:
            continue
        received = datetime.fromisoformat(simulation['received_at'])
        if start and received < start:
            continue
        ident = item['task']
        if ident in latest and latest[ident]['finished_at'] >= item['finished_at']:
            continue
        latest[ident] = item
    rows = []
    for ident, item in latest.items():
        from h3_service.status import compact_timing
        sim = item['simulation']
        rows.append({'task_id': ident, 'status': item['status'], 'error': item.get('error'),
                     'duration': item['request']['duration'], 'resolution': item['request']['resolution'],
                     'profile': sim['profile'], 'time_scale': sim['time_scale'],
                     'submitted_at': sim.get('queue_sent_at'), 'received_at': sim['received_at'],
                     'generation_started_at': sim.get('generation_started_at'),
                     'finished_at': item['finished_at'], 'sqs_deleted': item['sqs_deleted'],
                     **item.get('producer', {}), **item.get('queue_timing', {}),
                     **(item.get('timing') or compact_timing(item['generation'])), 'task_total_s': item['total_s']})
    return sorted(rows, key=lambda row: row.get('submitted_at') or row['received_at'])


def aggregate(rows, expected=0):
    def stats(field):
        values = sorted(row[field] for row in rows if row.get(field) is not None)
        if not values:
            return {'min': None, 'p50': None, 'p95': None, 'max': None}
        return {'min': values[0], 'p50': values[math.ceil(len(values) * .5) - 1],
                'p95': values[math.ceil(len(values) * .95) - 1], 'max': values[-1]}
    return {'expected': expected, 'completed': len(rows),
            'succeeded': sum(r['status'] == 'succeeded' for r in rows),
            'failed': sum(r['status'] != 'succeeded' for r in rows),
            'messages_deleted': sum(bool(r['sqs_deleted']) for r in rows),
            'queue_s': stats('queue_s'), 'wait_until_generation_s': stats('wait_until_generation_s'),
            'submitted_to_result_s': stats('submitted_to_result_s'),
            'profiles': sorted({r['profile'] for r in rows}),
            'time_scales': sorted({r['time_scale'] for r in rows})}


def save(root, rows, summary):
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    from h3_service.infer.resident import atomic
    atomic(root / 'report.json', {'summary': summary, 'tasks': rows})
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        temporary = root / 'report.csv.tmp'
        with temporary.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        temporary.chmod(0o600)
        temporary.replace(root / 'report.csv')


def submit(settings, args):
    from h3_service.media.cloud import Store, clients, now
    from h3_service.contracts.models import GenerateRequest
    from h3_service.infer.resident import atomic
    settings.validate_cloud()
    s3, queue = clients(settings)
    store = Store(settings, s3)
    combinations = [(duration, resolution) for duration in args.durations for resolution in args.resolutions]
    tasks = []
    for i in range(args.count):
        duration, resolution = combinations[i % len(combinations)]
        tasks.append({'task_id': uuid.uuid4().hex, 'duration': duration, 'resolution': resolution,
                      'submission': 'not_sent'})
    batch = {'created_at': now(), 'tasks': tasks, 'queue_url': settings.queue_url}
    directory = args.output or settings.work_dir / 'batches' / (datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    manifest = directory / 'batch.json'
    atomic(manifest, batch)
    bodies = {}
    try:
        # Persist all queued statuses before publishing. No missing-object probe or ListBucket needed.
        for task in tasks:
            ident = task['task_id']
            request = GenerateRequest(image=args.image_url, prompt='Animate the scene.', duration=task['duration'],
                                      resolution=task['resolution'], steps=4, seed=42, audio=False)
            wire = {'task_id': ident, **request.model_dump()}
            store.write({'task_id': ident, 'request': wire, 'status': 'queued', 'created_at': now(),
                         'simulation': {'enabled': True, 'placeholder': True}})
            bodies[ident] = json.dumps(wire)
        for index in range(0, len(tasks), 10):
            group = tasks[index:index + 10]
            for task in group:
                task['submission'] = 'uncertain'
            atomic(manifest, batch)
            response = queue.send_message_batch(QueueUrl=settings.queue_url,
                Entries=[{'Id': t['task_id'], 'MessageBody': bodies[t['task_id']]} for t in group])
            success = {item['Id'] for item in response.get('Successful', [])}
            failed = {item['Id']: item['Code'] for item in response.get('Failed', [])}
            for task in group:
                if task['task_id'] in success:
                    task['submission'] = 'submitted'
                elif task['task_id'] in failed:
                    task.update(submission='failed', error=failed[task['task_id']])
            atomic(manifest, batch)
            if len(success) != len(group):
                raise RuntimeError('partial_batch_submission')
    except Exception as exc:
        batch['error'] = type(exc).__name__
        atomic(manifest, batch)
        print(f'Submission incomplete type={type(exc).__name__}; inspect {manifest}; uncertain sends are never auto-retried.', flush=True)
        return 1
    print(f'Submitted {len(tasks)} simulation tasks. Manifest: {manifest}', flush=True)
    print(f'./sim.sh report --batch {manifest} --watch', flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'report', 'submit'])
    parser.add_argument('--expected', type=int, default=50)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--run', help='run_id from ./sim.sh status; otherwise all simulator runs are included')
    parser.add_argument('--batch', type=Path, help='batch.json from sim.sh submit; isolates exactly those task IDs')
    parser.add_argument('--since', help='ISO timestamp to isolate a batch')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--timeout', type=float, default=8 * 3600)
    parser.add_argument('--count', type=int, default=50)
    parser.add_argument('--durations', nargs='+', type=int, choices=(5, 10, 15), default=[5, 10, 15])
    parser.add_argument('--resolutions', nargs='+', choices=('720p', '1080p'), default=['720p', '1080p'])
    parser.add_argument('--image-url', default='https://d2ud1xskuh00y0.cloudfront.net/test1.jpg')
    args = parser.parse_args()
    settings = Settings()
    if args.action == 'submit':
        if args.count < 1 or args.count > 10000:
            parser.error('--count must be between 1 and 10000')
        return submit(settings, args)
    if args.action == 'check':
        from h3_service.aws_setup import verify_credentials, verify_queue
        from h3_service.media.cloud import clients
        settings.validate_cloud()
        verify_credentials(settings.region)
        _, queue = clients(settings)
        verify_queue(settings, queue)
        attributes = queue.get_queue_attributes(QueueUrl=settings.queue_url, AttributeNames=[
            'ApproximateNumberOfMessages', 'ApproximateNumberOfMessagesNotVisible'])['Attributes']
        print(json.dumps({'bucket': settings.bucket, 'status_prefix': settings.status_prefix,
                          'video_prefix': settings.prefix, 'queue_url': settings.queue_url,
                          'region': settings.region, 'queue': attributes,
                          'note': 'No message received/deleted. No object written. S3 permissions checked when used.'}, indent=2))
        return 0
    if args.expected < 1 or args.timeout <= 0:
        parser.error('--expected and --timeout must be positive')
    start = datetime.fromisoformat(args.since.replace('Z', '+00:00')) if args.since else None
    if start and start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    output = args.output or settings.work_dir / 'queue-report'
    batch_ids = None
    if args.batch:
        batch = json.loads(args.batch.read_text())
        batch_ids = {task['task_id'] for task in batch['tasks']}
        args.expected = len(batch_ids)
    run = args.run
    if not run and not batch_ids:
        try:
            with urllib.request.urlopen('http://127.0.0.1:' + os.getenv('H3_PORT', '8021') + '/health', timeout=2) as response:
                health = json.load(response)
                if health.get('simulation'):
                    run = health['producer']['run_id']
        except (OSError, ValueError, KeyError):
            pass
        if not run:
            history = task_rows(settings.work_dir / 'reports')
            if history:
                run = max(history, key=lambda row: row['finished_at'])['run_id']
    print(f'Report run={run or "not-yet-known"}; use --since to isolate a batch within a run.', flush=True)
    deadline = time.monotonic() + args.timeout
    previous = None
    last_notice = 0
    while True:
        rows = task_rows(settings.work_dir / 'reports', run, start)
        if batch_ids is not None:
            rows = [row for row in rows if row['task_id'] in batch_ids]
        summary = aggregate(rows, args.expected)
        summary['run_id'] = run
        save(output, rows, summary)
        if summary != previous or time.monotonic() - last_notice > 30:
            print(json.dumps(summary, ensure_ascii=False), flush=True)
            previous = summary
            last_notice = time.monotonic()
        if not args.watch or len(rows) >= args.expected:
            print(f'Report: {output.resolve() / "report.json"}; CSV: {output.resolve() / "report.csv"}', flush=True)
            return 0 if not args.watch or summary['failed'] == 0 else 1
        if time.monotonic() >= deadline:
            print('Report wait timed out; tasks may still be running. Rerun with identical filters to resume.', flush=True)
            return 1
        time.sleep(5)


if __name__ == '__main__':
    raise SystemExit(main())
