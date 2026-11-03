"""Non-secret provenance, shared by all task threads in one server run."""
import contextvars
import os
import re
import socket
import subprocess
import uuid
from functools import lru_cache
from h3_service.config import ROOT

task_context = contextvars.ContextVar('task_context', default={})


def label(value):
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', str(value))[:96]


@lru_cache(maxsize=1)
def producer():
    instance = os.getenv('CONTAINER_ID', '')
    worker = os.getenv('H3_WORKER_ID') or ('vast-' + instance if instance else 'host-' + socket.gethostname())
    revision = os.getenv('H3_REVISION')
    if not revision:
        try:
            revision = subprocess.check_output(['git', 'rev-parse', '--short=12', 'HEAD'],
                cwd=ROOT, stderr=subprocess.DEVNULL, text=True, timeout=3).strip()
        except (OSError, subprocess.SubprocessError):
            revision = 'unknown'
    return {'worker_id': label(worker), 'run_id': uuid.uuid4().hex[:16], 'revision': label(revision)}


def attempt():
    return {**producer(), 'attempt_id': uuid.uuid4().hex}
