import json
import logging
import os
import re
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from h3_service.config import ROOT
from h3_service.identity import producer, task_context


class PrivateRotatingHandler(RotatingFileHandler):
    def _open(self):
        stream = super()._open()
        os.chmod(self.baseFilename, 0o600)
        return stream


class ContextFilter(logging.Filter):
    def filter(self, record):
        record.provenance = {**producer(), **task_context.get()}
        record.worker_id = record.provenance['worker_id']
        record.run_id = record.provenance['run_id']
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record):
        message = record.getMessage()
        context = dict(record.provenance)
        task = re.search(r'\btask=([0-9a-f]{32})\b', message)
        if task:
            context['task_id'] = task.group(1)
        return json.dumps({'ts': datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            'level': record.levelname, 'logger': record.name, 'thread': record.threadName,
            **context, 'message': message}, ensure_ascii=False, allow_nan=False)


def setup_logging():
    directory = Path(os.getenv('H3_LOG_DIR', str(ROOT / 'logs')))
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(directory, 0o700)
    size = int(os.getenv('H3_LOG_MAX_MB', '20')) * 1024**2
    backups = int(os.getenv('H3_LOG_BACKUPS', '10'))
    text = PrivateRotatingHandler(directory / 'h3.log', maxBytes=size, backupCount=backups, encoding='utf-8')
    structured = PrivateRotatingHandler(directory / 'h3.jsonl', maxBytes=size, backupCount=backups, encoding='utf-8')
    structured.setFormatter(JsonFormatter())
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s [%(threadName)s] worker=%(worker_id)s run=%(run_id)s %(message)s')
    console = logging.StreamHandler()
    for handler in (text, console):
        handler.setFormatter(formatter)
    for handler in (text, console, structured):
        handler.addFilter(ContextFilter())
    logging.basicConfig(level=os.getenv('H3_LOG_LEVEL', 'INFO').upper(), handlers=[console, text, structured], force=True)
    logging.getLogger('boto3').setLevel(logging.WARNING)
    logging.getLogger('botocore').setLevel(logging.WARNING)
