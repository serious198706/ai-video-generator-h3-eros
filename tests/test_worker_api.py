import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from PIL import Image
from h3_service.config import Settings
from h3_service.contracts.models import GenerateRequest, Message
from h3_service.tasks.worker import Worker
from lambda_api.app import create_app
from tests.fakes import MemoryStore, Queue, Engine

IDENT = 'a' * 32


def download_fake(url, path, hosts):
    Image.new('RGB', (200, 300), 'green').save(path, format='PNG')


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = Settings(bucket='test', queue_url='queue', public_base='https://cdn.example.com', work_dir=Path(self.temp.name))
        self.store, self.queue, self.engine = MemoryStore(), Queue(), Engine()
        self.worker = Worker(self.settings, self.store, self.queue, self.engine)
        self.request = GenerateRequest(image='https://example.com/image', prompt='a calm forest')
        self.message = {'Body': Message(task_id=IDENT, request=self.request).model_dump_json(), 'ReceiptHandle': 'r',
                        'Attributes': {'ApproximateReceiveCount': '1'}}

    def run_job(self):
        with patch('h3_service.tasks.worker.download', side_effect=download_fake):
            self.worker.handle(self.message)

    def test_complete_and_duplicate(self):
        self.run_job()
        self.assertEqual(self.store.states[IDENT]['status'], 'succeeded')
        self.message['Attributes']['ApproximateReceiveCount'] = '9'
        self.run_job()
        self.assertEqual(self.engine.calls, 1)
        self.assertEqual(self.store.states[IDENT]['status'], 'succeeded')

    def test_fresh_queued_job_does_not_get_missing_video(self):
        with patch.object(self.store, 'recover', side_effect=AssertionError('fresh task must not GET a video')):
            self.run_job()
        self.assertEqual(self.store.states[IDENT]['status'], 'succeeded')

    def test_upload_recovery_after_terminal_write_failure(self):
        self.store.fail_terminal = True
        with self.assertRaises(OSError):
            self.run_job()
        self.assertFalse(self.queue.deleted)
        seed = self.store.states[IDENT]['seed']
        self.store.fail_terminal = False
        self.run_job()
        self.assertEqual(self.engine.calls, 1)
        self.assertEqual(self.store.states[IDENT]['result']['seed'], seed)
        self.assertEqual(len(self.queue.deleted), 1)

    def test_read_error_does_not_generate(self):
        self.store.fail_read = True
        with self.assertRaises(OSError):
            self.run_job()
        self.assertEqual(self.engine.calls, 0)
        self.assertFalse(self.queue.deleted)

    def test_callback_retries_without_regeneration(self):
        self.request.webhook_url = 'https://callback.example.com/job'
        self.message['Body'] = Message(task_id=IDENT, request=self.request).model_dump_json()
        with patch('h3_service.tasks.worker.resolve'), patch('h3_service.tasks.worker.notify', side_effect=OSError):
            with self.assertRaises(OSError):
                self.run_job()
        with patch('h3_service.tasks.worker.notify', return_value='delivered'):
            self.run_job()
        self.assertEqual(self.engine.calls, 1)
        self.assertEqual(self.store.states[IDENT]['callback'], 'delivered')

    def test_poison_valid_message(self):
        self.message['Attributes']['ApproximateReceiveCount'] = '5'
        self.run_job()
        self.assertEqual(self.store.states[IDENT]['error'], 'poisoned')
        self.assertEqual(self.engine.calls, 0)

    def test_malformed_envelope_retained_for_dlq(self):
        self.message['Body'] = '{}'
        with self.assertRaises(ValueError):
            self.run_job()
        self.assertFalse(self.queue.deleted)

    def test_identity_conflict(self):
        self.run_job()
        self.request.prompt = 'different request'
        self.message['Body'] = Message(task_id=IDENT, request=self.request).model_dump_json()
        with self.assertRaises(ValueError):
            self.run_job()
        self.assertEqual(self.engine.calls, 1)


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.store, self.queue = MemoryStore(), Queue()
        settings = Settings(bucket='test', queue_url='queue', public_base='https://cdn.example.com')
        self.client = TestClient(create_app(settings, self.store, self.queue))

    def test_enqueue_and_query(self):
        with patch('lambda_api.app.resolve'):
            response = self.client.post('/v1/generate', json={'image': 'https://example.com/i', 'prompt': 'forest'})
        self.assertEqual(response.status_code, 202)
        ident = response.json()['task_id']
        self.assertEqual(self.client.get('/v1/tasks/' + ident).json()['status'], 'queued')
        self.assertEqual(len(self.queue.sent), 1)

    def test_uncertain_send_preserves_queued(self):
        self.queue.fail_send = True
        with patch('lambda_api.app.resolve'):
            response = self.client.post('/v1/generate', json={'image': 'https://example.com/i', 'prompt': 'forest'})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(next(iter(self.store.states.values()))['status'], 'queued')

    def test_unsupported_does_not_enqueue(self):
        response = self.client.post('/v1/generate', json={'image': 'https://example.com/i', 'prompt': 'forest', 'resolution': '1080p'})
        self.assertEqual(response.status_code, 422)
        self.assertFalse(self.queue.sent)

    def test_status_read_error_is_503(self):
        self.store.fail_read = True
        self.assertEqual(self.client.get('/v1/tasks/' + IDENT).status_code, 503)

    def test_unknown(self):
        self.assertEqual(self.client.get('/v1/tasks/' + IDENT).json()['status'], 'unknown')
