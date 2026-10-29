import json
import logging
import secrets
import tempfile
import threading
import time
from pathlib import Path
from pydantic import ValidationError
from h3_service.contracts.models import Message, frame_count, normalize_stored_request, validation_summary
from h3_service.infer.engine import GenerationError, LeaseLost
from h3_service.infer.manifest import fingerprint
from h3_service.media.cloud import now
from h3_service.media.images import prepare
from h3_service.media.webhook import notify
from h3_service.net.http import download, resolve
from h3_service.aws_diagnostics import details

LOG = logging.getLogger(__name__)
TERMINAL = {'succeeded', 'failed'}


class Worker:
    """One consumer, one task. On uncertain lease/status ownership, stop accepting work."""
    def __init__(self, settings, store, sqs, engine):
        self.settings, self.store, self.sqs, self.engine = settings, store, sqs, engine
        self.stop = threading.Event()
        self.cancelled = threading.Event()
        self.ready = False
        self.error = None
        self.current = None
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self.run, name='h3-worker', daemon=True)
        self.thread.start()

    def shutdown(self):
        self.ready = False
        self.stop.set()
        self.cancelled.set()
        if self.thread:
            self.thread.join(timeout=45)

    def run(self):
        try:
            # Also protect direct uvicorn launches that bypass start.sh.
            from h3_service.aws_setup import verify_credentials, verify_queue
            verify_credentials(self.settings.region)
            verify_queue(self.settings, self.sqs)
            self.engine.preflight(self.cancelled)
            self.ready = True
            LOG.info('worker ready; model initialization complete; waiting for SQS tasks')
            while not self.stop.is_set():
                try:
                    messages = self.sqs.receive_message(QueueUrl=self.settings.queue_url, MaxNumberOfMessages=1,
                        WaitTimeSeconds=20, VisibilityTimeout=self.settings.visibility,
                        MessageSystemAttributeNames=['ApproximateReceiveCount']).get('Messages', [])
                    for message in messages:
                        if not self.stop.is_set():
                            self.handle(message)
                except Exception as exc:
                    LOG.error('task/queue failure details=%s; message retained', json.dumps(details(exc)))
                    if not getattr(self.engine, 'healthy', True):
                        self.error = 'resident_engine_unavailable'
                        self.stop.set()
                    self.stop.wait(5)
                if not getattr(self.engine, 'healthy', True):
                    self.error = 'resident_engine_unavailable'
                    self.stop.set()
        except Exception as exc:
            self.error = type(exc).__name__
            LOG.error('worker initialization failed details=%s', json.dumps(details(exc)))
        finally:
            self.ready = False
            close = getattr(self.engine, 'close', None)
            if close:
                close()

    def check(self):
        if self.cancelled.is_set():
            raise LeaseLost('worker stopped or lease uncertain')

    def handle(self, message):
        # Invalid envelopes are never silently deleted: SQS redrive policy must move them to a DLQ.
        try:
            envelope = Message.model_validate_json(message['Body'])
        except ValidationError as exc:
            LOG.error('SQS message rejected; fields=%s; input values omitted; message retained for DLQ',
                      json.dumps(validation_summary(exc)))
            raise
        ident, request = envelope.task_id, envelope.request
        receipt = message['ReceiptHandle']
        state = self.store.read(ident)
        requested = request.model_dump()
        if state:
            try:
                stored_request = normalize_stored_request(state.get('request'), ident)
            except ValidationError as exc:
                LOG.error('stored task request rejected task=%s fields=%s', ident, json.dumps(validation_summary(exc)))
                raise
            if stored_request != requested:
                raise ValueError('task identity/request conflict')
        if state is None:
            wire = json.loads(message['Body'])
            state = {'task_id': ident, 'request': wire if 'request' not in wire else requested,
                     'status': 'pending', 'created_at': now()}
        state.setdefault('seed', request.seed if request.seed is not None else secrets.randbelow(2**63))
        self.current = ident
        lock = threading.RLock()
        done = threading.Event()

        def persist(**changes):
            self.check()
            with lock:
                state.update(changes)
                self.store.write(state)

        def heartbeat():
            while not done.wait(self.settings.heartbeat):
                try:
                    self.sqs.change_message_visibility(QueueUrl=self.settings.queue_url, ReceiptHandle=receipt,
                                                       VisibilityTimeout=self.settings.visibility)
                    with lock:
                        if state['status'] not in TERMINAL:
                            self.store.write(state)
                except Exception as exc:
                    LOG.error('heartbeat failed task=%s type=%s', ident, type(exc).__name__)
                    self.error = 'lease_or_status_heartbeat_failed'
                    self.ready = False
                    self.cancelled.set()
                    self.stop.set()
                    return

        heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        heartbeat_thread.start()
        try:
            if state['status'] not in TERMINAL:
                persist(status='pending', stage='recovery')  # seed durable before any generation
                stamp = fingerprint(requested, state['seed'])
                result = self.store.recover(ident, stamp)
                if result is not None:
                    persist(status='succeeded', stage='complete', result=result, video_url=self.store.video_url(ident))
                elif int(message.get('Attributes', {}).get('ApproximateReceiveCount', '1')) >= self.settings.max_receive:
                    persist(status='failed', error='poisoned', stage='complete')
                else:
                    self.execute(request, state, persist, stamp)
            self.check()
            if state.get('callback') not in {'delivered', 'not_requested', 'exhausted'}:
                outcome = notify(state, self.settings, self.cancelled)
                persist(callback=outcome)
            self.check()
            self.sqs.delete_message(QueueUrl=self.settings.queue_url, ReceiptHandle=receipt)
        finally:
            done.set()
            heartbeat_thread.join(timeout=40)
            self.current = None

    def execute(self, request, state, persist, stamp):
        ident = state['task_id']
        self.settings.work_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        with tempfile.TemporaryDirectory(prefix=ident + '-', dir=self.settings.work_dir) as temp:
            directory = Path(temp)
            persist(status='generating', stage='download')
            try:
                # Validate callback before spending GPU time; queued messages are not trusted.
                if request.webhook_url:
                    resolve(request.webhook_url, self.settings.webhook_hosts, webhook=True)
                first, last = directory / 'first.download', directory / 'last.download' if request.last_image else None
                for url, path in [(request.image, first), (request.last_image, last)]:
                    if url:
                        for attempt in range(3):
                            try:
                                download(url, path, self.settings.image_hosts)
                                break
                            except Exception:
                                self.check()
                                if attempt == 2:
                                    raise
                                self.cancelled.wait(2**attempt)
                first, last, width, height = prepare(first, last, request.resolution, directory)
            except LeaseLost:
                raise
            except Exception:
                persist(status='failed', stage='complete', error='download_failed')
                return
            frames = frame_count(request.duration)
            job = {'prompt': request.prompt, 'first': str(first), 'last': str(last) if last else None,
                   'width': width, 'height': height, 'num_frames': frames,
                   'steps': request.steps, 'seed': state['seed'], 'audio': request.audio}
            persist(resolved={k: v for k, v in job.items() if k not in {'prompt', 'first', 'last'}})
            try:
                output, metrics = self.engine.generate(job, directory, self.cancelled, lambda label: persist(stage=label))
            except GenerationError as exc:
                persist(status='failed', stage='complete', error=exc.code)
                return
            self.check()
            result = {'seed': state['seed'], 'width': width, 'height': height, 'num_frames': frames, 'fps': 24,
                      'actual_duration': frames / 24, 'audio_present': request.audio, 'model': metrics['model']}
            persist(stage='upload', timing=metrics['timing'])
            # Transport exceptions propagate: leave message unacked for recovery/retry.
            self.store.upload(ident, output, result, stamp)
            persist(status='succeeded', stage='complete', result=result, video_url=self.store.video_url(ident))
