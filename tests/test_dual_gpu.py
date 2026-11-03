import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import replace
from h3_service.config import Settings
from h3_service.infer.dual_gpu import DualGpuEngine
from h3_service.contracts.models import Message
from tests import test_worker_api


class DualGpuLockTests(unittest.TestCase):
    def test_dispatcher_keeps_two_task_contexts_in_flight(self):
        from types import SimpleNamespace
        from h3_service.tasks.worker import Worker
        queue = MagicMock()
        settings = Settings(model_lifetime='dual_gpu')
        parent = Worker(settings, MagicMock(), queue, SimpleNamespace(healthy=True))
        requests = [Message(task_id=char * 32, request={'image': 'x', 'prompt': 'scene'}).model_dump_json()
                    for char in ('a', 'b')]
        remaining = list(requests)
        queue.receive_message.side_effect = lambda **kw: {'Messages': [
            {'Body': remaining.pop(0), 'ReceiptHandle': 'r'}] if remaining else []}
        both_entered = threading.Event()
        release = threading.Event()
        mutex = threading.Lock()
        counts = {'entered': 0, 'finished': 0}
        def handle(raw):
            with mutex:
                counts['entered'] += 1
                if counts['entered'] == 2:
                    both_entered.set()
            release.wait(3)
            with mutex:
                counts['finished'] += 1
                if counts['finished'] == 2:
                    parent.stop.set()
        def child(*args):
            return SimpleNamespace(handle=handle, stage='test', deleted_messages=0, last_task=None)
        with patch('h3_service.tasks.worker.Worker', side_effect=child):
            thread = threading.Thread(target=parent.run_pipeline)
            thread.start()
            self.assertTrue(both_entered.wait(2))
            release.set()
            thread.join(4)
            self.assertFalse(thread.is_alive())
        self.assertEqual(counts, {'entered': 2, 'finished': 2})

    def test_condition_and_sr_share_lock_but_generator_overlaps(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            engine = DualGpuEngine(Settings())
            engine.healthy = True
            entered = threading.Event()
            release = threading.Event()
            generated = threading.Event()
            calls = []
            def wait_reply(path, cancelled):
                mode = calls[-1]
                if mode == 'condition':
                    entered.set()
                    if not release.wait(3):
                        raise RuntimeError('test release timeout')
                else:
                    generated.set()
                return {}
            for role in ('condition', 'generate'):
                owner = MagicMock()
                owner.ipc.name = temp
                owner.wait_reply.side_effect = wait_reply
                engine.owners[role] = owner
            errors = []
            def invoke(role, mode):
                try:
                    engine.phase_call(role, mode, {'directory': temp}, threading.Event(), lambda s: None)
                except Exception as exc:
                    errors.append(exc)
            def send(path, request):
                calls.append(request['mode'])
                (root / (request['id'] + '.json')).touch()
            with patch('h3_service.infer.dual_gpu.atomic', side_effect=send):
                first = threading.Thread(target=invoke, args=('condition', 'condition'))
                first.start()
                self.assertTrue(entered.wait(2))
                second = threading.Thread(target=invoke, args=('generate', 'generate'))
                second.start()
                self.assertTrue(generated.wait(2))
                self.assertFalse(engine.locks['condition'].acquire(blocking=False))
                release.set()
                first.join(3)
                second.join(3)
            self.assertFalse(errors)
            self.assertCountEqual(calls, ['condition', 'generate'])


class LocalUpscaleWorkerTests(unittest.TestCase):
    run_job = test_worker_api.WorkerTests.run_job

    def setUp(self):
        test_worker_api.WorkerTests.setUp(self)
        self.worker.settings = replace(self.settings, model_lifetime='dual_gpu')
        self.request.resolution = '1080p'
        self.message['Body'] = Message(task_id=test_worker_api.IDENT, request=self.request).model_dump_json()
        self.engine.upscale = MagicMock(return_value=(Path(self.temp.name) / 'up.mp4',
            {'width': 1080, 'height': 1920, 'wall_s': 42, 'gpu_wait_s': 1, 'gpu': '0'}))

    def test_local_sr_uploads_only_final_and_returns_clean_result(self):
        with patch.object(self.store, 'upload', wraps=self.store.upload) as upload:
            self.run_job()
        state = self.store.states[test_worker_api.IDENT]
        self.assertEqual(state['status'], 'succeeded')
        self.assertEqual(state['result']['width'], 1080)
        self.assertNotIn('wall_s', state['result'])
        self.assertNotIn('upscale_handoff', state)
        self.assertFalse(self.queue.sent)
        self.assertNotIn('native', upload.call_args.kwargs)
        self.assertTrue((Path(self.temp.name) / 'reports' / test_worker_api.IDENT / 'summary.json').exists())
        self.run_job()
        self.assertEqual(self.engine.calls, 1)
        self.engine.upscale.assert_called_once()

    def test_final_upload_recovery_does_not_repeat_generation_or_sr(self):
        self.store.fail_terminal = True
        with self.assertRaises(OSError):
            self.run_job()
        self.store.fail_terminal = False
        self.run_job()
        self.assertEqual(self.engine.calls, 1)
        self.engine.upscale.assert_called_once()
