import logging
import uuid
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from h3_service.config import Settings
from h3_service.contracts.models import GenerateRequest, Message, task_id
from h3_service.media.cloud import Store, clients, now
from h3_service.net.http import resolve, UrlError

LOG = logging.getLogger(__name__)


def create_app(settings=None, store=None, sqs=None):
    settings = settings or Settings()
    app = FastAPI(title='H3 Queue API', docs_url=None, redoc_url=None)
    dependencies = {'store': store, 'sqs': sqs}

    def deps():
        try:
            settings.validate_cloud()
            if dependencies['store'] is None:
                s3, queue = clients(settings)
                dependencies.update(store=Store(settings, s3), sqs=queue)
            return dependencies['store'], dependencies['sqs']
        except Exception:
            raise HTTPException(503, 'service not configured') from None

    @app.post('/v1/generate')
    def submit(body: GenerateRequest):
        try:
            resolve(body.image, settings.image_hosts)
            if body.last_image:
                resolve(body.last_image, settings.image_hosts)
            if body.webhook_url:
                resolve(body.webhook_url, settings.webhook_hosts, webhook=True)
        except UrlError as exc:
            raise HTTPException(400, str(exc)) from None
        store, queue = deps()
        ident = uuid.uuid4().hex
        state = {'task_id': ident, 'request': body.model_dump(), 'status': 'queued', 'created_at': now()}
        try:
            if settings.max_queue > 0:
                size = int(queue.get_queue_attributes(QueueUrl=settings.queue_url,
                    AttributeNames=['ApproximateNumberOfMessages'])['Attributes']['ApproximateNumberOfMessages'])
                if size >= settings.max_queue:
                    raise HTTPException(429, 'queue is full')
            # Fail closed: never knowingly enqueue a task whose initial state cannot be persisted.
            store.write(state)
            queue.send_message(QueueUrl=settings.queue_url,
                               MessageBody=Message(task_id=ident, request=body).model_dump_json())
        except HTTPException:
            raise
        except Exception as exc:
            # SendMessage timeouts can mean accepted; do not stamp failed and suppress actual work.
            LOG.error('enqueue outcome uncertain task=%s type=%s', ident, type(exc).__name__)
            raise HTTPException(503, {'error': 'enqueue_unconfirmed', 'task_id': ident}) from None
        return JSONResponse({'id': ident, 'task_id': ident, 'status': 'queued'}, status_code=202)

    @app.get('/v1/tasks/{ident}')
    def describe(ident: str):
        try:
            task_id(ident)
        except ValueError:
            raise HTTPException(422, 'invalid task_id') from None
        store, _ = deps()
        try:
            state = store.read(ident)
        except Exception:
            raise HTTPException(503, 'status unavailable') from None
        if state is None:
            return {'id': ident, 'task_id': ident, 'status': 'unknown'}
        response = {**state, 'id': ident, **state.get('result', {})}
        if state['status'] in {'pending', 'generating'} and settings.stall_seconds:
            try:
                age = (datetime.now(timezone.utc) - datetime.fromisoformat(state['updated_at'])).total_seconds()
                if age > settings.stall_seconds:
                    response.update(status='stalled', error='worker_unresponsive')
            except (KeyError, ValueError, TypeError):
                raise HTTPException(503, 'invalid status timestamp') from None
        return response

    @app.get('/health')
    def health():
        return {'ok': True, 'queue_configured': bool(settings.queue_url)}

    return app


app = create_app()
