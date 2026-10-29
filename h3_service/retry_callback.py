"""Administrative callback replay for a terminal job; stop its worker before using."""
import argparse
import sys
import threading
from h3_service.config import Settings
from h3_service.contracts.models import task_id
from h3_service.media.cloud import Store, clients
from h3_service.media.webhook import notify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task_id', type=task_id)
    args = parser.parse_args()
    settings = Settings()
    settings.validate_cloud()
    s3, _ = clients(settings)
    store = Store(settings, s3)
    state = store.read(args.task_id)
    if not state or state['status'] not in {'succeeded', 'failed'}:
        raise ValueError('task is not terminal')
    outcome = notify(state, settings, threading.Event())
    state['callback'] = outcome
    store.write(state)
    print(outcome)
    return 0 if outcome in {'delivered', 'not_requested'} else 1


if __name__ == '__main__':
    sys.exit(main())
