import logging
import fcntl
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from h3_service.config import Settings
from h3_service.infer.engine import Engine
from h3_service.media.cloud import Store, clients
from h3_service.tasks.worker import Worker


@asynccontextmanager
async def lifespan(app):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
    settings = Settings()
    settings.validate_cloud()
    settings.work_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (settings.work_dir / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        s3, sqs = clients(settings)
        worker = Worker(settings, Store(settings, s3), sqs, Engine(settings))
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
    return {'ok': True, 'model_ready': worker.ready, 'task_id': worker.current, 'error': worker.error}


@app.get('/ready')
def ready():
    if not app.state.worker.ready:
        raise HTTPException(503, app.state.worker.error or 'model preflight in progress')
    return {'ok': True}
