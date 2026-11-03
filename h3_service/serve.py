"""Managed Uvicorn entry point: failed workers must fail the service process."""
import asyncio
import json
import logging
import os
import time
from pathlib import Path


def restart_budget(path, *, failure=False, now=None):
    now = time.time() if now is None else now
    try:
        history = json.loads(path.read_text())
    except FileNotFoundError:
        history = []
    history = [t for t in history if now - t < int(os.getenv('H3_RESTART_WINDOW', '3600'))]
    if failure:
        history.append(now)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    from h3_service.infer.resident import atomic
    atomic(path, history)
    return len(history) < int(os.getenv('H3_RESTART_LIMIT', '3'))


def fatal_reason(worker, startup_timeout):
    if worker.error:
        return worker.error
    if worker.thread and not worker.thread.is_alive() and not worker.stop.is_set():
        return 'worker_thread_exited'
    if worker.ready and not worker.model_processes_alive():
        return 'resident_process_exited'
    if not worker.ready and time.monotonic() - worker.started > startup_timeout:
        return 'initialization_timeout'
    return None


async def run(app=None):
    import uvicorn
    if app is None:
        from h3_service.api.app import app
    server = uvicorn.Server(uvicorn.Config(app, host=os.getenv('H3_BIND', '127.0.0.1'),
        port=int(os.getenv('H3_PORT', '8001')), access_log=False, workers=1))
    task = asyncio.create_task(server.serve())
    failed = False
    last_cleanup = 0
    last_resources = 0
    try:
        while not task.done():
            await asyncio.sleep(2)
            if server.should_exit:
                continue
            worker = getattr(app.state, 'worker', None)
            if worker is None:
                continue
            if time.monotonic() - last_resources > 30:
                from h3_service.ops.resources import warnings
                current = warnings(worker.settings)
                if current != worker.resource_warnings:
                    logging.getLogger(__name__).warning('resource admission changed warnings=%s', current)
                worker.resource_warnings = current
                worker.accepting = not bool(current)
                last_resources = time.monotonic()
            if time.monotonic() - last_cleanup > 3600:
                from h3_service.ops.retention import cleanup_reports
                try:
                    cleanup_reports(worker.settings.work_dir / 'reports')
                except OSError as exc:
                    logging.getLogger(__name__).warning('report retention failed errno=%s', exc.errno)
                last_cleanup = time.monotonic()
            reason = fatal_reason(worker, int(os.getenv('H3_STARTUP_TIMEOUT', '3600')))
            if reason:
                logging.getLogger(__name__).error('process guard stopping unhealthy service reason=%s', reason)
                worker.error, worker.ready = reason, False
                worker.cancelled.set()
                worker.stop.set()
                server.should_exit = True
                failed = True
        await task
        return 1 if failed or not server.started else 0
    finally:
        if not task.done():
            server.should_exit = True
            await task


def main(app=None):
    os.umask(0o077)
    from h3_service.config import Settings
    managed = os.getenv('H3_MANAGED') == '1'
    budget = Settings().work_dir / 'service' / 'failures.json'
    if managed and not restart_budget(budget):
        control = './' + Path(os.getenv('H3_SERVICE_ENTRY', 'service.sh')).name
        print(f'Automatic restart limit reached. Inspect logs, fix the cause, then {control} reset.', flush=True)
        return 78
    try:
        code = asyncio.run(run(app))
    except Exception as exc:
        logging.getLogger(__name__).error('service entry failed type=%s', type(exc).__name__)
        code = 1
    if managed and code:
        restart_budget(budget, failure=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
