"""Reproduce task image preparation without GPU, writes to S3, or queue operations."""
import argparse
import json
import tempfile
from pathlib import Path
from h3_service.config import Settings
from h3_service.contracts.models import task_id, normalize_stored_request
from h3_service.media.cloud import clients, Store
from h3_service.media.images import prepare
from h3_service.net.http import download
from h3_service.tasks.worker import input_failure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-id', type=task_id, required=True)
    args = parser.parse_args()
    settings = Settings()
    s3, _ = clients(settings)
    state = Store(settings, s3).read(args.task_id)
    if state is None:
        raise RuntimeError('task status missing')
    request = normalize_stored_request(state['request'], args.task_id)
    try:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = root / 'first.download'
            last = root / 'last.download' if request['last_image'] else None
            for label, url, destination in [('first', request['image'], first), ('last', request['last_image'], last)]:
                if url:
                    print(json.dumps({'task': args.task_id, 'stage': 'download_start', 'frame': label}), flush=True)
                    download(url, destination, settings.image_hosts)
                    print(json.dumps({'stage': 'download_complete', 'frame': label, 'bytes': destination.stat().st_size}), flush=True)
            _, _, width, height = prepare(first, last, request['resolution'], root)
            print(json.dumps({'stage': 'images_prepared', 'width': width, 'height': height}))
        return 0
    except Exception as exc:
        print(input_failure(exc))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
