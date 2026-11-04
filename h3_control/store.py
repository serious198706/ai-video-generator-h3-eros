"""DynamoDB state, atomic admission, leases, and a durable SQS outbox.

The small admission ledger is deliberately bounded. Every change uses a versioned
transaction; no Scan, GSI, distributed counter repair, or DynamoDB TTL for live work.
"""
import hashlib
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone

import boto3
from boto3.dynamodb.types import TypeDeserializer, TypeSerializer
from botocore.config import Config

LOG = logging.getLogger(__name__)
TERMINAL = {'succeeded', 'failed'}
SER = TypeSerializer()
DES = TypeDeserializer()
# GPU1 seconds measured on the two real hosts. GPU0 conditioning/SR overlaps.
PROFILES = {
    '6000-s': {5: (51.033, 48.531, 6.505), 10: (126.587, 126.838, 12.009), 15: (213.144, 214.879, 16.761)},
    '6000-maxq': {5: (69.068, 67.569, 11.012), 10: (174.416, 174.923, 21.024), 15: (300.311, 300.801, 30.033)},
}


class CapacityExceeded(Exception):
    def __init__(self, estimate, limit, reason='estimated_completion_exceeded'):
        self.detail = {'code': 'LOCAL_CAPACITY_EXCEEDED', 'estimated_completion_seconds': estimate,
                       'admission_limit_seconds': limit, 'reason': reason, 'fallback': 'a2e'}
        super().__init__(reason)


class Conflict(Exception):
    pass


class LeaseLost(Exception):
    pass


def utc():
    return datetime.now(timezone.utc).isoformat()


def encode(item):
    return {k: SER.serialize(v) for k, v in item.items()}


def decode(item):
    return {k: DES.deserialize(v) for k, v in item.items()}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


class Control:
    def __init__(self, table, region='us-east-1', namespace='h3', client=None, profile='6000-s',
                 limit=780, max_tasks=18, worker_stale=90, lease_seconds=180, retention=30 * 86400):
        self.table, self.namespace = table, namespace
        self.profile, self.limit, self.max_tasks = profile, int(limit), int(max_tasks)
        self.worker_stale, self.lease_seconds, self.retention = int(worker_stale), int(lease_seconds), int(retention)
        if profile not in PROFILES or min(self.limit, self.max_tasks, self.worker_stale, self.lease_seconds, self.retention) <= 0:
            raise ValueError('invalid H3 control configuration')
        self.client = client or boto3.client('dynamodb', region_name=region,
            endpoint_url=os.getenv('H3_DYNAMODB_ENDPOINT') or None,
            config=Config(connect_timeout=5, read_timeout=10, max_pool_connections=30,
                          retries={'mode': 'standard', 'total_max_attempts': 3}))

    @classmethod
    def from_env(cls, region=None):
        table = os.getenv('H3_DYNAMODB_TABLE', '').strip()
        if not table:
            return None
        return cls(table, region or os.getenv('AWS_REGION') or os.getenv('AWS_DEFAULT_REGION') or 'us-east-1',
                   namespace=os.getenv('H3_CONTROL_NAMESPACE', 'h3'),
                   profile=os.getenv('H3_CAPACITY_PROFILE', '6000-s'),
                   limit=os.getenv('H3_ADMISSION_LIMIT_SECONDS', '780'),
                   max_tasks=os.getenv('H3_ADMISSION_MAX_TASKS', '18'),
                   worker_stale=os.getenv('H3_WORKER_STALE_SECONDS', '90'),
                   lease_seconds=os.getenv('H3_TASK_LEASE_SECONDS', '180'),
                   retention=os.getenv('H3_TASK_RETENTION_SECONDS', str(30 * 86400)))

    def key(self, ident=None):
        return {'pk': f'{self.namespace}#TASK#{ident}' if ident else f'{self.namespace}#POOL', 'sk': 'STATE'}

    def get(self, ident=None):
        item = self.client.get_item(TableName=self.table, Key=encode(self.key(ident)), ConsistentRead=True).get('Item')
        return decode(item) if item else None

    def pool(self):
        item = self.get()
        return item or {**self.key(), 'version': 0, 'data': canonical({'workers': {}, 'tasks': {}})}

    def put_operation(self, item, version):
        condition = 'attribute_not_exists(pk)' if version is None else '#v = :v'
        args = {'TableName': self.table, 'Item': encode(item), 'ConditionExpression': condition}
        if version is not None:
            args.update(ExpressionAttributeNames={'#v': 'version'}, ExpressionAttributeValues=encode({':v': version}))
        return {'Put': args}

    def commit(self, pool, data, task=None, old_task=None):
        old_version = int(pool['version'])
        record = {**self.key(), 'version': old_version + 1, 'data': canonical(data)}
        operations = [self.put_operation(record, old_version if old_version else None)]
        if task is not None:
            operations.append(self.put_operation(task, int(old_task['version']) if old_task else None))
        # A new token is appropriate here: this exact transaction is retried internally
        # by botocore; a later CAS iteration has a different snapshot and token.
        try:
            self.client.transact_write_items(TransactItems=operations, ClientRequestToken=uuid.uuid4().hex)
        except self.client.exceptions.TransactionCanceledException as exc:
            codes = [r.get('Code') for r in exc.response.get('CancellationReasons', [])]
            if codes and all(c in {'None', 'ConditionalCheckFailed', 'TransactionConflict'} for c in codes):
                return False
            raise
        return True

    def change_pool(self, change):
        for _ in range(30):
            item = self.pool()
            data = json.loads(item['data'])
            result = change(data)
            if self.commit(item, data):
                return result
        raise RuntimeError('control ledger contention; retry request')

    def heartbeat(self, worker_id, run_id, ready):
        def change(data):
            now = time.time()
            data['workers'] = {k: v for k, v in data['workers'].items() if now - v['seen'] < self.worker_stale * 4}
            previous = data['workers'].get(worker_id)
            if previous and previous['ready'] and previous['run_id'] != run_id and now - previous['seen'] < self.worker_stale:
                raise Conflict('worker_id already used by another live process')
            data['workers'][worker_id] = {'run_id': run_id, 'ready': bool(ready), 'seen': now}
        self.change_pool(change)

    def cost(self, request):
        duration = request.get('duration', 5)
        resolution = request.get('resolution')
        if duration not in (5, 10, 15) or resolution not in {'720p', '1080p'} or request.get('audio') or request.get('steps') != 4:
            raise ValueError('admission supports 5/10/15s, 720p/1080p, steps=4, audio=false')
        native, high, upscale = PROFILES[self.profile][duration]
        return (high if resolution == '1080p' else native) + 2, (upscale if resolution == '1080p' else 0) + 10

    def estimate(self, data, work, tail):
        now = time.time()
        workers = {k: v for k, v in data['workers'].items() if v['ready'] and now - v['seen'] < self.worker_stale}
        if not workers:
            raise CapacityExceeded(None, self.limit, 'no_ready_workers')
        loads = {k: 0.0 for k in workers}
        queued = []
        for entry in data['tasks'].values():
            remaining = entry['work']
            owner = entry.get('worker_id')
            # Only fresh, current-run generation progress may reduce reserved work.
            current = workers.get(owner, {})
            if (entry.get('stage') == 'denoise_decode' and current.get('run_id') == entry.get('run_id')
                    and now - entry.get('seen', 0) < self.worker_stale):
                remaining = max(5.0, remaining - (now - entry['stage_started']))
            if (entry.get('stage') in {'upscaling', 'upload', 'complete'}
                    and current.get('run_id') == entry.get('run_id')
                    and now - entry.get('seen', 0) < self.worker_stale):
                remaining = 0
            if owner in loads and current.get('run_id') == entry.get('run_id'):
                loads[owner] += remaining
            else:
                queued.append((entry['submitted'], remaining))
        for _, remaining in sorted(queued):
            worker = min(loads, key=loads.get)
            loads[worker] += remaining
        return round(min(loads.values()) + work + tail)

    def admit(self, request, queue_url, idempotency_key=None):
        if idempotency_key is not None and not 1 <= len(idempotency_key) <= 128:
            raise ValueError('Idempotency-Key must contain 1..128 characters')
        ident = (hashlib.sha256((self.namespace + ':' + idempotency_key).encode()).hexdigest()[:32]
                 if idempotency_key else uuid.uuid4().hex)
        request = {k: v for k, v in request.items() if k != 'task_id'}
        digest = hashlib.sha256(canonical(request).encode()).hexdigest()
        work, tail = self.cost(request)
        for _ in range(30):
            previous = self.get(ident)
            if previous:
                if previous['request_hash'] != digest:
                    raise Conflict('Idempotency-Key reused with different request')
                return json.loads(previous['state']), False
            pool = self.pool()
            data = json.loads(pool['data'])
            estimate = self.estimate(data, work, tail)
            if len(data['tasks']) >= self.max_tasks:
                raise CapacityExceeded(estimate, self.limit, 'unfinished_task_limit')
            if estimate > self.limit:
                raise CapacityExceeded(estimate, self.limit)
            state = {'task_id': ident, 'request': request, 'status': 'queued', 'stage': 'queued',
                     'created_at': utc(), 'updated_at': utc(), 'estimated_completion_seconds': estimate}
            task = {**self.key(ident), 'version': 1, 'request_hash': digest, 'state': canonical(state),
                    'queue_url': queue_url, 'message': canonical({**request, 'task_id': ident})}
            if len(canonical(task).encode()) > 300 * 1024:
                raise ValueError('task record too large')
            data['tasks'][ident] = {'work': work, 'tail': tail, 'submitted': time.time(), 'sent': False}
            if self.commit(pool, data, task):
                return state, True
        raise RuntimeError('admission contention; retry with the same Idempotency-Key')

    def read(self, ident):
        item = self.get(ident)
        return json.loads(item['state']) if item else None

    def describe(self, ident):
        item = self.get(ident)
        if not item:
            return None
        state = json.loads(item['state'])
        if state['status'] in {'pending', 'generating'} and int(item.get('lease_until', 0)) < time.time():
            state.update(status='stalled', error='worker_unresponsive')
        return state

    def dispatch(self, sqs, ident=None):
        data = json.loads(self.pool()['data'])
        pending = [ident] if ident else [k for k, v in data['tasks'].items() if not v['sent']]
        for task_id in pending:
            entry = data['tasks'].get(task_id)
            if not entry or entry['sent']:
                continue
            task = self.get(task_id)
            if task is None:
                raise RuntimeError('outbox task missing')
            if json.loads(task['state'])['status'] in TERMINAL:
                continue
            # On a timeout, leave sent=false. A later dispatch can safely duplicate.
            sqs.send_message(QueueUrl=task['queue_url'], MessageBody=task['message'])
            def mark(pool, task_id=task_id):
                if task_id in pool['tasks']:
                    pool['tasks'][task_id]['sent'] = True
            self.change_pool(mark)
            LOG.info('outbox dispatched task=%s', task_id)

    def claim(self, ident, owner, worker_id, run_id):
        for _ in range(30):
            task = self.get(ident)
            if task is None:
                raise Conflict('task not admitted through DynamoDB API')
            if task.get('lease_owner') != owner and int(task.get('lease_until', 0)) > time.time():
                return False
            pool = self.pool()
            data = json.loads(pool['data'])
            updated = {**task, 'version': int(task['version']) + 1, 'lease_owner': owner,
                       'lease_until': int(time.time()) + self.lease_seconds}
            claimed_state = json.loads(task['state'])
            claimed_state['attempt_count'] = int(claimed_state.get('attempt_count', 0)) + 1
            updated['state'] = canonical(claimed_state)
            if ident in data['tasks']:
                data['tasks'][ident].update(worker_id=worker_id, run_id=run_id, seen=time.time(), stage='claimed', sent=True)
            if self.commit(pool, data, updated, task):
                return True
        raise RuntimeError('claim contention')

    def write(self, state, owner):
        ident = state['task_id']
        for _ in range(30):
            task = self.get(ident)
            if not task or task.get('lease_owner') != owner or int(task.get('lease_until', 0)) <= time.time():
                raise LeaseLost('task lease no longer owned')
            old_state = json.loads(task['state'])
            if old_state['status'] in TERMINAL and state['status'] not in TERMINAL:
                raise LeaseLost('terminal state cannot regress')
            pool = self.pool()
            data = json.loads(pool['data'])
            updated = {**task, 'version': int(task['version']) + 1, 'state': canonical(state),
                       'lease_until': int(time.time()) + self.lease_seconds}
            if state['status'] in TERMINAL:
                updated['expires_at'] = int(time.time()) + self.retention
                updated.pop('message', None)
                updated.pop('queue_url', None)
                data['tasks'].pop(ident, None)
            elif ident in data['tasks']:
                entry = data['tasks'][ident]
                stage = state.get('stage', 'unknown')
                if entry.get('stage') != stage:
                    entry['stage_started'] = time.time()
                entry.update(stage=stage, seen=time.time())
            if self.commit(pool, data, updated, task):
                return
        raise RuntimeError('state write contention')
