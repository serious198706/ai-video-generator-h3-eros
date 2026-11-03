import fcntl
import logging
import os
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from h3_service.config import Settings
from h3_service.identity import producer
from h3_service.log import setup_logging
from h3_service.media.cloud import Store, clients
from h3_service.simulator.engine import SimulatedEngine
from h3_service.simulator.worker import SimulationWorker


@asynccontextmanager
async def lifespan(app):
    if os.getenv('H3_SIMULATION') != '1':
        raise RuntimeError('Use ./sim.sh start to explicitly launch the simulation service')
    setup_logging()
    settings = Settings()
    if settings.model_lifetime != 'dual_gpu':
        raise ValueError('simulator requires the dual_gpu scheduling contract')
    settings.validate_cloud()
    settings.work_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (settings.work_dir / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        s3, sqs = clients(settings)
        worker = SimulationWorker(settings, Store(settings, s3), sqs, SimulatedEngine(settings))
        app.state.worker = worker
        logging.getLogger(__name__).warning('SIMULATION using real SQS/S3; producer=%s bucket=%s status_prefix=%s',
                                            producer(), settings.bucket, settings.status_prefix)
        worker.start()
        try:
            yield
        finally:
            worker.shutdown()


app = FastAPI(title='H3 Local Latency Simulator', lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.get('/health')
def health():
    worker = app.state.worker
    return {'ok': not bool(worker.error), 'simulation': True, 'profile': worker.engine.profile,
            'time_scale': worker.engine.scale, 'producer': producer(), 'model_ready': worker.ready,
            'accepting_tasks': worker.ready and worker.accepting and not worker.stop.is_set(),
            'error': worker.error, 'uptime_s': round(time.monotonic() - worker.started, 1),
            'active_tasks': getattr(worker, 'active_tasks', []), 'resource_warnings': worker.resource_warnings,
            'received_messages': worker.received_messages, 'deleted_messages': worker.deleted_messages,
            'last_task': worker.last_task}


@app.get('/ready')
def ready():
    worker = app.state.worker
    if not worker.ready or not worker.accepting:
        raise HTTPException(503, worker.error or 'simulation not ready')
    return {'ok': True, 'simulation': True}


if __name__ == '__main__':
    from h3_service.serve import main
    raise SystemExit(main(app))
