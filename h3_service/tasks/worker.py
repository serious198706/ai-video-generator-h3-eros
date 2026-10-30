import json
import logging
import secrets
import tempfile
import threading
import time
import traceback
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
from h3_service.tasks.trace import TaskTrace

LOG = logging.getLogger(__name__)
TERMINAL = {'succeeded', 'failed'}


def input_failure(exc):
    result = {'type': type(exc).__name__, 'trace': [
        {'file': Path(f.filename).name, 'function': f.name, 'line': f.lineno}
        for f in traceback.extract_tb(exc.__traceback__)[-5:]]}
    from h3_service.net.http import UrlError
    if isinstance(exc, UrlError):
        result.update(reason=exc.code, http_status=exc.http_status)
    causes = []
    cause = exc.__cause__ or exc.__context__
    for _ in range(4):
        if cause is None:
            break
        causes.append(type(cause).__name__)
        cause = cause.__cause__ or cause.__context__
    result['cause_types'] = causes
    return json.dumps(result)


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
        self.stage = 'initializing'
        self.received_messages = 0
        self.deleted_messages = 0
        self.last_task = None

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
            self.stage = 'idle'
            last_idle = time.monotonic()
            while not self.stop.is_set():
                try:
                    polled = time.monotonic()
                    messages = self.sqs.receive_message(QueueUrl=self.settings.queue_url, MaxNumberOfMessages=1,
                        WaitTimeSeconds=20, VisibilityTimeout=self.settings.visibility,
                        MessageSystemAttributeNames=['ApproximateReceiveCount']).get('Messages', [])
                    if messages:
                        LOG.info('SQS receive complete count=%s elapsed_s=%.3f', len(messages), time.monotonic() - polled)
                    elif time.monotonic() - last_idle >= 60:
                        LOG.info('SQS polling active; no messages returned; received=%s deleted=%s',
                                 self.received_messages, self.deleted_messages)
                        last_idle = time.monotonic()
                    for message in messages:
                        if not self.stop.is_set():
                            self.handle(message)
                except Exception as exc:
                    LOG.error('task/queue failure task=%s stage=%s details=%s; message retained',
                              self.current, self.stage, json.dumps(details(exc)))
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
        trace = TaskTrace()
        try:
            return self._handle(message, trace)
        except BaseException as exc:
            if trace.status not in TERMINAL:
                trace.status = 'cancelled' if isinstance(exc, LeaseLost) else 'retry_pending'
            trace.failed_stage = trace.failed_stage or self.stage
            trace.failure(exc)
            raise
        finally:
            trace.finish()

    def _handle(self, message, trace):
        started = trace.started
        self.received_messages += 1
        self.stage = 'validate_message'
        LOG.info('SQS message received receive_count=%s body_bytes=%s; validating',
                 message.get('Attributes', {}).get('ApproximateReceiveCount', '1'),
                 len(message.get('Body', '').encode()))
        # Invalid envelopes are never silently deleted: SQS redrive policy must move them to a DLQ.
        try:
            with trace.measure('validation'):
                envelope = Message.model_validate_json(message['Body'])
        except ValidationError as exc:
            LOG.error('SQS message rejected; fields=%s; input values omitted; message retained for DLQ',
                      json.dumps(validation_summary(exc)))
            raise
        ident, request = envelope.task_id, envelope.request
        trace.accept(ident, request, message.get('Attributes', {}).get('ApproximateReceiveCount', '1'))
        LOG.info('task accepted task=%s resolution=%s duration=%s steps=%s audio=%s callback_requested=%s',
                 ident, request.resolution, request.duration, request.steps, request.audio, bool(request.webhook_url))
        receipt = message['ReceiptHandle']
        self.stage = 'read_status'
        LOG.info('status read start task=%s', ident)
        with trace.measure('status_read'):
            state = self.store.read(ident)
        LOG.info('status read complete task=%s status=%s stage=%s', ident,
                 state.get('status') if state else 'missing', state.get('stage') if state else None)
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
        trace.request['seed'] = state['seed']
        self.current = ident
        lock = threading.RLock()
        done = threading.Event()

        def persist(**changes):
            self.check()
            with lock:
                state.update(changes)
                with trace.measure('status_write'):
                    self.store.write(state)
                trace.status, trace.error = state['status'], state.get('error')
                if state.get('seed') is not None:
                    trace.request['seed'] = state['seed']
                self.stage = state.get('stage', self.stage)
                LOG.info('status persisted task=%s status=%s stage=%s error=%s',
                         ident, state['status'], state.get('stage'), state.get('error'))

        def heartbeat():
            while not done.wait(self.settings.heartbeat):
                try:
                    self.sqs.change_message_visibility(QueueUrl=self.settings.queue_url, ReceiptHandle=receipt,
                                                       VisibilityTimeout=self.settings.visibility)
                    LOG.info('SQS visibility renewed task=%s timeout_s=%s stage=%s',
                             ident, self.settings.visibility, self.stage)
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
                # A newly queued task has no video yet. Only inspect video metadata after a
                # persisted upload attempt, avoiding a guaranteed missing-object GET per task.
                recover_upload = bool(state.get('upload_started')) or state.get('stage') == 'upload'
                persist(status='pending', stage='recovery', upload_started=recover_upload)
                stamp = fingerprint(requested, state['seed'])
                if recover_upload:
                    LOG.info('upload recovery check start task=%s', ident)
                with trace.measure('upload_recovery'):
                    result = self.store.recover(ident, stamp) if recover_upload else None
                if result is not None:
                    LOG.info('upload recovery found existing video task=%s; generation skipped', ident)
                    persist(status='succeeded', stage='complete', result=result, video_url=self.store.video_url(ident))
                elif int(message.get('Attributes', {}).get('ApproximateReceiveCount', '1')) >= self.settings.max_receive:
                    LOG.warning('task receive limit reached task=%s limit=%s; marking failed without generation', ident, self.settings.max_receive)
                    persist(status='failed', error='poisoned', stage='complete')
                else:
                    self.execute(request, state, persist, stamp, trace)
            else:
                LOG.info('terminal task redelivered task=%s status=%s; generation skipped', ident, state['status'])
            trace.status, trace.error = state['status'], state.get('error')
            self.check()
            if state.get('callback') not in {'delivered', 'not_requested', 'exhausted'}:
                LOG.info('callback processing start task=%s', ident)
                with trace.measure('callback'):
                    outcome = notify(state, self.settings, self.cancelled)
                persist(callback=outcome)
                LOG.info('callback processing complete task=%s outcome=%s', ident, outcome)
            else:
                LOG.info('callback already resolved task=%s outcome=%s', ident, state['callback'])
            self.check()
            trace.callback = state.get('callback')
            self.stage = 'delete_message'
            LOG.info('SQS delete start task=%s status=%s callback=%s', ident, state['status'], state.get('callback'))
            with trace.measure('sqs_delete'):
                self.sqs.delete_message(QueueUrl=self.settings.queue_url, ReceiptHandle=receipt)
            trace.deleted = True
            self.deleted_messages += 1
            self.last_task = {'task_id': ident, 'status': state['status'], 'error': state.get('error'),
                              'elapsed_s': round(time.monotonic() - started, 3)}
            LOG.info('task complete; SQS message deleted task=%s status=%s error=%s elapsed_s=%.3f',
                     ident, state['status'], state.get('error'), time.monotonic() - started)
        except BaseException:
            trace.failed_stage = trace.failed_stage or self.stage
            raise
        finally:
            done.set()
            heartbeat_thread.join(timeout=40)
            self.current = None
            self.stage = 'idle' if self.ready else 'not_ready'

    def execute(self, request, state, persist, stamp, trace):
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
                                LOG.info('image download start task=%s frame=%s attempt=%s', ident,
                                         'first' if path == first else 'last', attempt + 1)
                                transfer_started = time.monotonic()
                                with trace.measure('download_first' if path == first else 'download_last'):
                                    download(url, path, self.settings.image_hosts)
                                LOG.info('image download complete task=%s frame=%s elapsed_s=%.3f bytes=%s',
                                         ident, 'first' if path == first else 'last', time.monotonic() - transfer_started, path.stat().st_size)
                                trace.failed_stage = None  # A successful retry resolves earlier attempt failures.
                                break
                            except Exception as exc:
                                LOG.warning('image download failed task=%s attempt=%s details=%s', ident, attempt + 1, input_failure(exc))
                                self.check()
                                if attempt == 2:
                                    raise
                                self.cancelled.wait(2**attempt)
                with trace.measure('image_prepare'):
                    first, last, width, height = prepare(first, last, request.resolution, directory)
                LOG.info('images prepared task=%s width=%s height=%s last_frame=%s', ident, width, height, bool(last))
            except LeaseLost:
                raise
            except Exception as exc:
                LOG.error('task input/download failed task=%s details=%s; no generation', ident, input_failure(exc))
                trace.failed_stage = trace.failed_stage or 'download_or_prepare'
                persist(status='failed', stage='complete', error='download_failed')
                return
            frames = frame_count(request.duration)
            job = {'prompt': request.prompt, 'first': str(first), 'last': str(last) if last else None,
                   'width': width, 'height': height, 'num_frames': frames,
                   'steps': request.steps, 'seed': state['seed'], 'audio': request.audio}
            persist(resolved={k: v for k, v in job.items() if k not in {'prompt', 'first', 'last'}})
            try:
                LOG.info('generation start task=%s seed=%s frames=%s steps=%s', ident, state['seed'], frames, request.steps)
                generation_started = time.monotonic()
                def on_stage(label):
                    LOG.info('generation stage start task=%s stage=%s', ident, label)
                    persist(stage=label)
                with trace.measure('generation'):
                    output, metrics = self.engine.generate(job, directory, self.cancelled, on_stage)
                trace.metrics = metrics.get('timing', {})
                LOG.info('generation complete task=%s elapsed_s=%.3f timing=%s', ident,
                         time.monotonic() - generation_started, json.dumps(metrics.get('timing', {})))
            except GenerationError as exc:
                LOG.error('generation failed task=%s error=%s', ident, exc.code)
                persist(status='failed', stage='complete', error=exc.code)
                return
            self.check()
            result = {'seed': state['seed'], 'width': width, 'height': height, 'num_frames': frames, 'fps': 24,
                      'actual_duration': frames / 24, 'audio_present': request.audio, 'model': metrics['model']}
            persist(stage='upload', timing=metrics['timing'], upload_started=True)
            # Transport exceptions propagate: leave message unacked for recovery/retry.
            self.stage = 'upload'
            upload_started = time.monotonic()
            LOG.info('S3 video upload start task=%s', ident)
            with trace.measure('upload'):
                self.store.upload(ident, output, result, stamp)
            LOG.info('S3 video upload complete task=%s elapsed_s=%.3f', ident, time.monotonic() - upload_started)
            persist(status='succeeded', stage='complete', result=result, video_url=self.store.video_url(ident))
