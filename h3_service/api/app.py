import logging
import fcntl
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from h3_service.config import Settings
from h3_service.infer.engine import Engine
from h3_service.infer.resident import ResidentEngine
from h3_service.media.cloud import Store, clients
from h3_service.tasks.worker import Worker
from h3_service.log import setup_logging


@asynccontextmanager
async def lifespan(app):
    setup_logging()
    settings = Settings()
    settings.validate_cloud()
    settings.work_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (settings.work_dir / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        s3, sqs = clients(settings)
        engine_class = ResidentEngine if settings.model_lifetime in {'resident', 'dual_int8'} else Engine
        worker = Worker(settings, Store(settings, s3), sqs, engine_class(settings))
        app.state.worker = worker
        worker.start()
        try:
            yield
        finally:
            worker.shutdown()


app = FastAPI(title='H3 GPU Worker', lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.get('/health')
def health():
    worker = app.state.worker
    return {'ok': True, 'model_ready': worker.ready, 'task_id': worker.current, 'error': worker.error,
            'task_stage': worker.stage, 'received_messages': worker.received_messages,
            'deleted_messages': worker.deleted_messages, 'last_task': worker.last_task}


@app.get('/ready')
def ready():
    if not app.state.worker.ready:
        raise HTTPException(503, app.state.worker.error or 'model preflight in progress')
    return {'ok': True}
