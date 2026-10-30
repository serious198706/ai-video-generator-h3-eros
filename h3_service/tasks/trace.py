"""One request record and one timing summary per delivery, including failure paths."""
import hashlib
import json
import logging
import time
import traceback
from pathlib import Path
from contextlib import contextmanager
from urllib.parse import urlsplit
from h3_service.aws_diagnostics import details

LOG = logging.getLogger('h3_service.tasks.worker')


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def source(value):
    if not value:
        return None
    try:
        parsed = urlsplit(value)
    except ValueError:
        return {'scheme': 'invalid', 'sha256': digest(value)}
    return {'scheme': parsed.scheme, 'host': parsed.hostname, 'sha256': digest(value),
            'has_query': bool(parsed.query)}


class TaskTrace:
    def __init__(self):
        self.started = time.monotonic()
        self.ident = None
        self.status = 'unvalidated'
        self.error = None
        self.deleted = False
        self.callback = None
        self.request = {}
        self.times = {}
        self.metrics = {}
        self.active = None
        self.failed_stage = None

    def accept(self, ident, request, receive_count):
        self.ident = ident
        self.status = 'received'
        self.request = {k: getattr(request, k) for k in ('duration', 'resolution', 'steps', 'seed', 'quality', 'audio')}
        self.request.update(prompt_chars=len(request.prompt), prompt_sha256=digest(request.prompt),
                            negative_prompt_chars=len(request.negative_prompt or ''),
                            image=source(request.image), last_image=source(request.last_image),
                            webhook=source(request.webhook_url), receive_count=receive_count)
        LOG.info('request task=%s params=%s', ident, json.dumps(self.request, sort_keys=True))

    @contextmanager
    def measure(self, label):
        started = time.monotonic()
        previous = self.active
        self.active = label
        outcome = 'succeeded'
        LOG.info('stage start task=%s stage=%s task_elapsed_s=%.3f', self.ident, label, started - self.started)
        try:
            yield
        except BaseException:
            outcome = 'failed'
            self.failed_stage = self.failed_stage or label
            raise
        finally:
            elapsed = time.monotonic() - started
            self.times[label] = self.times.get(label, 0.0) + elapsed
            LOG.info('stage end task=%s stage=%s outcome=%s elapsed_s=%.3f task_elapsed_s=%.3f',
                     self.ident, label, outcome, elapsed, time.monotonic() - self.started)
            self.active = previous

    def failure(self, exc):
        self.error = self.error or type(exc).__name__
        diagnostic = details(exc)
        diagnostic['trace'] = [{'file': Path(frame.filename).name, 'function': frame.name, 'line': frame.lineno}
                               for frame in traceback.extract_tb(exc.__traceback__)[-8:]]
        LOG.error('task exception task=%s failed_stage=%s details=%s',
                  self.ident, self.failed_stage, json.dumps(diagnostic))

    def finish(self):
        summary = {'task': self.ident, 'status': self.status, 'error': self.error,
                   'failed_stage': self.failed_stage, 'sqs_deleted': self.deleted,
                   'callback': self.callback, 'total_s': round(time.monotonic() - self.started, 3),
                   'stages_s': {k: round(v, 3) for k, v in self.times.items()},
                   'generation': self.metrics, 'request': self.request}
        LOG.info('timing task=%s status=%s error=%s total_s=%.3f sqs_deleted=%s callback=%s details=%s',
                 self.ident, self.status, self.error, summary['total_s'], self.deleted, self.callback,
                 json.dumps(summary, sort_keys=True, allow_nan=False))
