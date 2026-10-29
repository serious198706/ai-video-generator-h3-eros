import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from h3_service.config import Settings
from h3_service.infer.engine import Engine, LeaseLost
from h3_service.media.webhook import notify
from h3_service.contracts.models import GenerateRequest, Message
from h3_service.tasks.worker import Worker
from tests.fakes import MemoryStore, Queue, Engine as FakeEngine


class ExecutionTests(unittest.TestCase):
    def test_cancel_kills_phase_before_return(self):
        proc = MagicMock(pid=1234, returncode=None)
        proc.poll.return_value = None
        cancelled = threading.Event()
        cancelled.set()
        with tempfile.TemporaryDirectory() as temp, patch('subprocess.Popen', return_value=proc), patch('os.killpg') as kill:
            with self.assertRaises(LeaseLost):
                Engine(Settings()).phase('generate', Path(temp), cancelled)
            kill.assert_called_once()
            proc.wait.assert_called_once()

    def test_heartbeat_failure_never_acks(self):
        ident = 'c' * 32
        request = GenerateRequest(image='https://example.com/i', prompt='forest')
        message = {'Body': Message(task_id=ident, request=request).model_dump_json(), 'ReceiptHandle': 'receipt'}
        queue, store = Queue(), MemoryStore()
        queue.fail_renew = True
        with tempfile.TemporaryDirectory() as temp:
            settings = Settings(work_dir=Path(temp), heartbeat=0.01)
            worker = Worker(settings, store, queue, FakeEngine())
            store.states[ident] = {'task_id': ident, 'request': request.model_dump(),
                                   'status': 'generating', 'upload_started': True}

            def wait_for_failure(*args):
                self.assertTrue(worker.cancelled.wait(2))
                return None

            with patch.object(store, 'recover', side_effect=wait_for_failure), self.assertRaises(LeaseLost):
                worker.handle(message)
            self.assertFalse(queue.deleted)
            self.assertTrue(worker.stop.is_set())
            self.assertFalse(worker.ready)

    def test_webhook_signature_is_over_sent_bytes(self):
        import hashlib
        import hmac
        state = {'task_id': 'd' * 32, 'status': 'failed', 'error': 'generate_failed',
                 'request': GenerateRequest(image='https://example.com/i', prompt='forest',
                     webhookUrl='https://callback.example.com/x').model_dump()}
        response = MagicMock(status=204)
        pool = MagicMock()
        with patch('h3_service.media.webhook.request', return_value=(pool, response)) as transport:
            result = notify(state, Settings(webhook_secret='test-only-key'), threading.Event())
        self.assertEqual(result, 'delivered')
        kwargs = transport.call_args.kwargs
        expected = hmac.new(b'test-only-key', kwargs['body'], hashlib.sha256).hexdigest()
        self.assertEqual(kwargs['headers']['X-Wan-Signature'], 'sha256=' + expected)
        self.assertEqual(json.loads(kwargs['body'])['error'], 'generate_failed')
        response.close.assert_called_once()
        pool.close.assert_called_once()
