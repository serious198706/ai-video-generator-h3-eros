"""Filter structured logs and export sanitized task diagnostics, offline."""
import argparse
import json
import os
import re
import tarfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from h3_service.config import ROOT, Settings


def since(value):
    if not value:
        return None
    now = datetime.now().astimezone()
    if value == 'today':
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    match = re.fullmatch(r'(\d+) (hour|minute|day)s? ago', value)
    if match:
        return now - timedelta(**{match[2] + 's': int(match[1])})
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    return parsed if parsed.tzinfo else parsed.astimezone()


def selected(record, args, start):
    if start and datetime.fromisoformat(record['ts']) < start:
        return False
    if args.errors and record.get('level') not in {'ERROR', 'CRITICAL'}:
        return False
    for key, option in [('task_id', args.task), ('worker_id', args.worker), ('run_id', args.run)]:
        if option and record.get(key) != option:
            return False
    return True


def records(paths):
    for path in paths:
        try:
            with path.open() as stream:
                for line in stream:
                    try:
                        record = json.loads(line)
                        if isinstance(record, dict) and 'ts' in record:
                            yield record
                    except ValueError:
                        continue
        except FileNotFoundError:
            continue


def follow(path):
    """Finish the open inode before switching to a replacement after rotation."""
    stream = None
    try:
        while True:
            if stream is None:
                try:
                    stream = path.open()
                except FileNotFoundError:
                    time.sleep(.5)
                    continue
            position = stream.tell()
            line = stream.readline()
            if line.endswith('\n'):
                try:
                    record = json.loads(line)
                    if isinstance(record, dict) and 'ts' in record:
                        yield record
                except ValueError:
                    pass
                continue
            stream.seek(position)
            try:
                current = path.stat()
                if current.st_ino != os.fstat(stream.fileno()).st_ino:
                    stream.close()
                    stream = None
                    continue
                if current.st_size < position:
                    stream.seek(0)
            except FileNotFoundError:
                pass
            time.sleep(.5)
    finally:
        if stream is not None:
            stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task')
    parser.add_argument('--worker')
    parser.add_argument('--run')
    parser.add_argument('--errors', action='store_true')
    parser.add_argument('--since', help='today, 1 hour ago, or ISO datetime; unzoned input uses machine timezone')
    parser.add_argument('--summary', action='store_true')
    parser.add_argument('--follow', action='store_true')
    parser.add_argument('--export', type=Path, help='write a sanitized .tar.gz for --task')
    args = parser.parse_args()
    if args.task and not re.fullmatch('[0-9a-f]{32}', args.task):
        parser.error('--task must be a 32-character task ID')
    if args.follow and (args.summary or args.export):
        parser.error('--follow cannot be combined with --summary or --export')
    if args.export and not args.task:
        parser.error('--export requires --task')
    start = since(args.since)
    root = Settings().work_dir / 'reports'
    directory = Path(os.getenv('H3_LOG_DIR', str(ROOT / 'logs')))
    paths = sorted(directory.glob('h3.jsonl*'), key=lambda p: p.stat().st_mtime)
    if args.summary:
        for path in sorted(root.glob(f'{args.task or "*"}/*.json'), key=lambda p: p.stat().st_mtime):
            if path.name == 'summary.json' and len(list(path.parent.glob('*.json'))) > 1:
                continue
            item = json.loads(path.read_text())
            record = {'ts': item.get('finished_at', datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()),
                      'task_id': item['task'], 'level': 'ERROR' if item.get('error') else 'INFO', **item.get('producer', {})}
            if selected(record, args, start):
                from h3_service.status import compact_timing
                print(json.dumps({**record, 'status': item['status'], 'error': item['error'],
                    'timing': compact_timing(item['generation']), 'task_total_s': item['total_s']}, ensure_ascii=False))
        return 0
    def output(record):
        if selected(record, args, start):
            print(json.dumps(record, ensure_ascii=False), flush=True)
    if args.export:
        import io
        args.export.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(args.export, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as stream, tarfile.open(fileobj=stream, mode='w:gz') as archive:
            for path in sorted((root / args.task).glob('*.json')):
                archive.add(path, arcname='reports/' + path.name)
            data = ''.join(json.dumps(r, ensure_ascii=False) + '\n'
                           for r in records(paths) if selected(r, args, start)).encode()
            info = tarfile.TarInfo('task.jsonl')
            info.size, info.mode = len(data), 0o600
            archive.addfile(info, io.BytesIO(data))
        print(args.export.resolve())
        return 0
    current = directory / 'h3.jsonl'
    for record in records([p for p in paths if not args.follow or p != current]):
        output(record)
    if args.follow:
        try:
            for record in follow(current):
                output(record)
        except KeyboardInterrupt:
            return 0
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
