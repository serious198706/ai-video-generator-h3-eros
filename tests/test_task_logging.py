import unittest
from unittest.mock import patch
from tests import test_worker_api


class TaskLoggingTests(unittest.TestCase):
    setUp = test_worker_api.WorkerTests.setUp
    run_job = test_worker_api.WorkerTests.run_job

    def test_success_logs_each_external_step_in_order_without_inputs(self):
        with self.assertLogs('h3_service', level='INFO') as captured:
            self.run_job()
        logs = '\n'.join(captured.output)
        expected = ['SQS message received', 'task accepted', 'status read start', 'status read complete',
                    'image download start', 'image download complete', 'images prepared', 'generation start',
                    'generation stage start', 'generation complete', 'S3 video upload start',
                    'S3 video upload complete', 'callback processing start', 'callback processing complete',
                    'SQS delete start', 'task complete; SQS message deleted']
        cursor = -1
        for phrase in expected:
            cursor = logs.index(phrase, cursor + 1)
        self.assertNotIn(self.request.prompt, logs)
        self.assertNotIn(self.request.image, logs)
        self.assertEqual(self.worker.received_messages, 1)
        self.assertEqual(self.worker.deleted_messages, 1)
        self.assertEqual(self.worker.last_task['status'], 'succeeded')

    def test_failed_download_logs_error_and_no_generation_then_terminal_ack(self):
        from h3_service.net.http import UrlError
        with patch('h3_service.tasks.worker.download', side_effect=UrlError('image download rejected', http_status=403)), \
                patch.object(self.worker.cancelled, 'wait'), self.assertLogs('h3_service', level='INFO') as captured:
            self.worker.handle(self.message)
        logs = '\n'.join(captured.output)
        self.assertIn('"http_status": 403', logs)
        self.assertIn('download_failed', logs)
        self.assertNotIn('generation start', logs)
        self.assertIn('SQS message deleted', logs)
        self.assertEqual(self.worker.last_task['error'], 'download_failed')
