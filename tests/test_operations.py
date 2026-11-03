import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from h3_service.config import Settings
from h3_service.identity import producer, task_context
from h3_service.ops.logs import selected, since
from h3_service.ops.retention import cleanup_reports
from h3_service.ops.service import configuration
from h3_service.serve import fatal_reason, restart_budget
from h3_service.status import compact_timing
from tests import test_worker_api


class OperationsTests(unittest.TestCase):
    def test_port_conflict_fails_before_any_worker_or_model_start(self):
        import asyncio
        import socket
        from h3_service.serve import run
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0)); listener.listen()
            with patch.dict(os.environ, {'H3_PORT': str(listener.getsockname()[1])}), \
                 patch('h3_service.api.app.Worker.start') as start:
                self.assertEqual(asyncio.run(run()), 1)
                start.assert_not_called()

    def test_actual_uvicorn_runner_exits_nonzero_when_worker_initialization_fails(self):
        import asyncio
        from h3_service.api.app import app
        from h3_service.serve import run
        import logging
        with tempfile.TemporaryDirectory() as temp:
            settings = Settings(work_dir=Path(temp), model_lifetime='cold')
            def start(worker):
                worker.error = 'FileNotFoundError'
            handlers = logging.getLogger().handlers[:]
            try:
                with patch('h3_service.api.app.Settings', return_value=settings), \
                     patch.object(Settings, 'validate_cloud'), \
                     patch('h3_service.api.app.clients', return_value=(MagicMock(), MagicMock())), \
                     patch('h3_service.api.app.Worker.start', start), \
                     patch('h3_service.api.app.setup_logging'), \
                     patch.dict(os.environ, {'H3_PORT': '0'}):
                    self.assertEqual(asyncio.run(run()), 1)
            finally:
                logging.getLogger().handlers = handlers

    def test_supervisor_real_start_stop_and_terminal_failure_without_models(self):
        with tempfile.TemporaryDirectory() as temp:
            # Keep socket paths short on macOS as well as Linux.
            root = Path(temp)
            start = root / 'start.sh'
            start.write_text('echo started >> starts; exec sleep 30\n')
            config = root / 'supervisord.conf'
            config.write_text(configuration(root, root, Path(sys.executable), 1).replace('startsecs=5', 'startsecs=0'))
            daemon = subprocess.Popen([sys.executable, '-m', 'supervisor.supervisord', '-n', '-c', str(config)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            ctl = [sys.executable, '-m', 'supervisor.supervisorctl', '-c', str(config)]
            def control(*args):
                return subprocess.run(ctl + list(args), capture_output=True, text=True, timeout=15)
            try:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    if (root / 'starts').exists():
                        break
                    time.sleep(.1)
                self.assertTrue((root / 'starts').exists())
                self.assertIn('RUNNING', control('status', 'h3').stdout)
                self.assertEqual(control('stop', 'h3').returncode, 0)
                self.assertIn('STOPPED', control('status', 'h3').stdout)
                # Exit 78 must not trigger infinite automatic restarts.
                start.write_text('echo exhausted >> starts; exit 78\n')
                control('start', 'h3')
                time.sleep(.5)
                self.assertIn('EXITED', control('status', 'h3').stdout)
                self.assertEqual((root / 'starts').read_text().count('exhausted'), 1)
            finally:
                control('shutdown')
                try:
                    daemon.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    daemon.kill(); daemon.wait()

    def test_private_supervisor_config_parses_and_caps_restarts(self):
        from supervisor.options import ServerOptions
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / 'supervisord.conf'
            config.write_text(configuration(root, root, Path(sys.executable), 900))
            options = ServerOptions()
            options.realize(['-c', str(config)])
            process = options.process_group_configs[0].process_configs[0]
            self.assertEqual(process.stopwaitsecs, 960)
            self.assertEqual(process.exitcodes, [0, 78])
            self.assertTrue(process.killasgroup)
            self.assertNotIn('inet_http_server', config.read_text())
            self.assertNotIn('AWS_', config.read_text())

    def test_restart_budget_survives_process_restart_and_expires(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'failures.json'
            with patch.dict(os.environ, {'H3_RESTART_LIMIT': '3', 'H3_RESTART_WINDOW': '100'}):
                self.assertTrue(restart_budget(path, now=1))
                restart_budget(path, failure=True, now=2)
                restart_budget(path, failure=True, now=3)
                self.assertFalse(restart_budget(path, failure=True, now=4))
                self.assertFalse(restart_budget(path, now=5))
                self.assertTrue(restart_budget(path, now=105))

    def test_guard_detects_idle_owner_exit_and_initialization_failure(self):
        worker = SimpleNamespace(error=None, thread=SimpleNamespace(is_alive=lambda: True),
            stop=threading.Event(), ready=True, model_processes_alive=lambda: False, started=time.monotonic())
        self.assertEqual(fatal_reason(worker, 3600), 'resident_process_exited')
        worker.ready = False
        self.assertIsNone(fatal_reason(worker, 3600))
        worker.error = 'FileNotFoundError'
        self.assertEqual(fatal_reason(worker, 3600), 'FileNotFoundError')
        worker.error = None
        worker.started -= 4000
        self.assertEqual(fatal_reason(worker, 3600), 'initialization_timeout')

    def test_graceful_stop_allows_inflight_completion_before_cancellation(self):
        from h3_service.tasks.worker import Worker
        worker = Worker(Settings(shutdown_grace=1), MagicMock(), MagicMock(), MagicMock())
        completed = threading.Event()
        def job():
            worker.stop.wait(1)
            if not worker.cancelled.is_set():
                completed.set()
        worker.thread = threading.Thread(target=job)
        worker.thread.start()
        worker.shutdown()
        self.assertTrue(completed.is_set())
        self.assertFalse(worker.cancelled.is_set())

    def test_time_contract_excludes_wait_and_diagnostic_fields(self):
        metrics = {'condition': {'wall_s': 1, 'gpu_wait_s': 10},
                   'generate': {'wall_s': 20, 'gpu_wait_s': 30, 'peak_reserved_gib': 80},
                   'upscale': {'wall_s': 3, 'gpu_wait_s': 5}}
        result = compact_timing(metrics)
        self.assertEqual(result['processing_s'], 24)
        self.assertEqual(result['gpu_wait_s'], 45)
        self.assertNotIn('peak_reserved_gib', json.dumps(result))

    def test_benchmark_understands_new_compact_and_old_nested_timings(self):
        from scripts.benchmark_api_matrix import summarize
        case = {}
        summarize(case, {'status': 'succeeded', 'timing': {'conditioning_s': 1, 'generation_s': 20, 'upscale_s': 3}})
        self.assertEqual(case['generation_s'], 21)
        self.assertEqual(case['upscale_s'], 3)
        summarize(case, {'status': 'succeeded', 'timing': {'condition': {'wall_s': 1}, 'generate': {'wall_s': 20}}})
        self.assertEqual(case['generation_s'], 21)

    def test_log_export_contains_only_requested_task_and_uses_private_permissions(self):
        import tarfile
        from h3_service.ops.logs import main
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); logs = root / 'logs'; logs.mkdir()
            task = 'a' * 32
            rows = [{'ts': '2026-11-03T00:00:00+00:00', 'task_id': ident, 'message': 'completed'}
                    for ident in (task, 'b' * 32)]
            (logs / 'h3.jsonl').write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
            out = root / 'task.tar.gz'
            with patch.dict(os.environ, {'H3_LOG_DIR': str(logs), 'H3_WORK_DIR': str(root)}), \
                 patch('sys.argv', ['logs', '--task', task, '--export', str(out)]):
                self.assertEqual(main(), 0)
            self.assertEqual(out.stat().st_mode & 0o777, 0o600)
            with tarfile.open(out) as archive:
                data = archive.extractfile('task.jsonl').read().decode()
            self.assertIn(task, data)
            self.assertNotIn('b' * 32, data)

    def test_follow_reads_tail_of_old_file_then_new_file_after_rotation(self):
        from h3_service.ops.logs import follow
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'h3.jsonl'
            def row(number):
                return json.dumps({'ts': '2026-11-03T00:00:00+00:00', 'number': number}) + '\n'
            path.write_text(row(1) + row(2))
            stream = follow(path)
            try:
                self.assertEqual(next(stream)['number'], 1)
                path.rename(path.with_suffix('.jsonl.1'))
                path.write_text(row(3))
                self.assertEqual(next(stream)['number'], 2)
                self.assertEqual(next(stream)['number'], 3)
            finally:
                stream.close()

    def test_disk_pressure_pauses_admission_without_raising_or_killing_models(self):
        from h3_service.ops.resources import warnings
        with patch('h3_service.ops.resources.shutil.disk_usage', return_value=SimpleNamespace(free=1024)), \
             patch('h3_service.ops.resources.os.path.exists', return_value=False):
            self.assertEqual(warnings(Settings()), ['disk_space_low'])

    def test_identity_and_json_log_context_do_not_expose_credentials(self):
        from h3_service.log import ContextFilter, JsonFormatter
        import logging
        try:
            producer.cache_clear()
            with patch.dict(os.environ, {'CONTAINER_ID': '12345', 'H3_REVISION': 'abc',
                                        'AWS_SECRET_ACCESS_KEY': 'do-not-log'}, clear=True):
                self.assertEqual(producer()['worker_id'], 'vast-12345')
                record = logging.LogRecord('h3', logging.INFO, '', 1, 'task=%s completed', ('a' * 32,), None)
                token = task_context.set({'attempt_id': 'attempt-A'})
                try:
                    ContextFilter().filter(record)
                finally:
                    task_context.reset(token)
                item = json.loads(JsonFormatter().format(record))
                self.assertEqual(item['task_id'], 'a' * 32)
                self.assertEqual(item['attempt_id'], 'attempt-A')
                self.assertNotIn('do-not-log', json.dumps(item))
        finally:
            producer.cache_clear()

    def test_filters_require_task_worker_run_and_timestamp(self):
        args = Namespace(task='a' * 32, worker='vast-123', run='run-1', errors=True)
        record = {'ts': '2026-11-03T06:00:00+00:00', 'task_id': args.task,
                  'worker_id': args.worker, 'run_id': args.run, 'level': 'ERROR'}
        self.assertTrue(selected(record, args, since('2026-11-03T05:00:00Z')))
        self.assertFalse(selected(record, args, since('2026-11-03T07:00:00Z')))
        record['run_id'] = 'run-2'
        self.assertFalse(selected(record, args, None))

    def test_report_retention_never_removes_fresh_attempts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / ('a' * 32)
            folder.mkdir()
            old, fresh = folder / 'old.json', folder / 'new.json'
            old.write_text('{}'); fresh.write_text('{}')
            os.utime(old, (1, 1))
            cleanup_reports(root, now=time.time())
            self.assertFalse(old.exists())
            self.assertTrue(fresh.exists())


class ProvenanceRecoveryTests(unittest.TestCase):
    setUp = test_worker_api.WorkerTests.setUp
    run_job = test_worker_api.WorkerTests.run_job

    def test_public_status_is_compact_and_metadata_keeps_provenance(self):
        def generate(job, directory, cancelled, stage):
            return directory / 'unused.mp4', {'model': 'test', 'timing': {
                'condition': {'wall_s': 1, 'peak_reserved_gib': 40},
                'generate': {'wall_s': 20, 'gpu_wait_s': 5, 'stages': {'denoise_s': 18}}}}
        self.engine.generate = generate
        self.run_job()
        state = self.store.states[test_worker_api.IDENT]
        self.assertEqual(state['timing']['processing_s'], 21)
        self.assertEqual(state['timing']['gpu_wait_s'], 5)
        self.assertNotIn('peak_reserved_gib', json.dumps(state))
        self.assertNotIn('resolved', state)
        self.assertEqual(self.store.videos[test_worker_api.IDENT][0]['producer'], state['producer'])

    def test_failed_status_write_recovers_original_producer_on_another_machine(self):
        self.store.fail_terminal = True
        origin = {'worker_id': 'machine-A', 'run_id': 'run-A', 'revision': 'rev-A', 'attempt_id': 'attempt-A'}
        with patch('h3_service.tasks.worker.attempt', return_value=origin):
            with self.assertRaises(OSError):
                self.run_job()
        self.store.fail_terminal = False
        other = {**origin, 'worker_id': 'machine-B', 'attempt_id': 'attempt-B'}
        with patch('h3_service.tasks.worker.attempt', return_value=other):
            self.run_job()
        state = self.store.states[test_worker_api.IDENT]
        self.assertEqual(state['producer']['worker_id'], 'machine-A')
        self.assertNotIn('producer', state['result'])
        self.assertNotIn('resolved', state)
        self.assertNotIn('upload_started', state)
        self.assertEqual(self.engine.calls, 1)
        reports = Path(self.temp.name) / 'reports' / test_worker_api.IDENT
        self.assertTrue((reports / 'attempt-A.json').exists())
        self.assertTrue((reports / 'attempt-B.json').exists())


@unittest.skipUnless(sys.platform == 'linux', 'Linux parent-death signal')
class ParentGuardTests(unittest.TestCase):
    def test_model_child_dies_when_owning_process_is_killed(self):
        with tempfile.TemporaryDirectory() as temp:
            pidfile = Path(temp) / 'pid'
            child = ('from h3_service.infer.parent_guard import protect_parent; protect_parent(); '
                     f'from pathlib import Path; import os,time; Path({str(pidfile)!r}).write_text(str(os.getpid())); time.sleep(30)')
            code = ('import subprocess,os,time,sys; '
                    f'subprocess.Popen([sys.executable,"-c",{child!r}],env={{**os.environ,"H3_OWNER_PID":str(os.getpid())}}); time.sleep(30)')
            parent = subprocess.Popen([sys.executable, '-c', code])
            try:
                deadline = time.monotonic() + 5
                while not pidfile.exists() and time.monotonic() < deadline:
                    time.sleep(.05)
                self.assertTrue(pidfile.exists())
                child_pid = int(pidfile.read_text())
                parent.kill(); parent.wait()
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    status = Path(f'/proc/{child_pid}/stat')
                    if not status.exists() or status.read_text().split()[2] == 'Z':
                        break
                    time.sleep(.05)
                else:
                    os.kill(child_pid, signal.SIGKILL)
                    self.fail('model subprocess survived owning process SIGKILL')
            finally:
                if parent.poll() is None:
                    parent.kill(); parent.wait()
