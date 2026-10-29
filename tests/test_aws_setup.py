import os
import unittest
from unittest.mock import patch, MagicMock
from botocore.exceptions import NoCredentialsError
from h3_service.aws_setup import bootstrap, verify_credentials


class AwsSetupTests(unittest.TestCase):
    def test_queue_access_failure_prevents_loading(self):
        from h3_service.tasks.worker import Worker
        from h3_service.config import Settings
        engine, queue = MagicMock(), MagicMock()
        queue.get_queue_attributes.side_effect = RuntimeError('access denied')
        worker = Worker(Settings(queue_url='test-queue'), MagicMock(), queue, engine)
        with patch('h3_service.aws_setup.verify_credentials'):
            worker.run()
        engine.preflight.assert_not_called()
        self.assertFalse(worker.ready)

    def test_existing_chain_validated_without_prompt(self):
        with patch('boto3.Session') as session, patch('getpass.getpass') as prompt:
            bootstrap('us-east-1')
            prompt.assert_not_called()
            session.return_value.client.return_value.get_caller_identity.assert_called_once()

    def test_missing_noninteractive_fails(self):
        with patch('boto3.Session') as session, patch('sys.stdin.isatty', return_value=False):
            session.return_value.get_credentials.return_value = None
            with self.assertRaises(NoCredentialsError):
                bootstrap('us-east-1')

    def test_hidden_input_is_only_in_environment_and_stale_token_removed(self):
        with patch.dict(os.environ, {'AWS_SESSION_TOKEN': 'test-old'}), \
                patch('boto3.Session') as session, patch('sys.stdin.isatty', return_value=True), \
                patch('getpass.getpass', side_effect=['test-id', 'test-secret', '']), \
                patch('h3_service.aws_setup.verify_credentials') as verify:
            session.return_value.get_credentials.return_value = None
            bootstrap('us-east-1')
            self.assertEqual(os.environ['AWS_ACCESS_KEY_ID'], 'test-id')
            self.assertNotIn('AWS_SESSION_TOKEN', os.environ)
            verify.assert_called_once_with('us-east-1')

    def test_worker_stops_before_model_loading_when_credentials_missing(self):
        from h3_service.tasks.worker import Worker
        from h3_service.config import Settings
        engine = MagicMock()
        worker = Worker(Settings(), MagicMock(), MagicMock(), engine)
        with patch('h3_service.aws_setup.verify_credentials', side_effect=NoCredentialsError):
            worker.run()
        engine.preflight.assert_not_called()
        self.assertFalse(worker.ready)
        self.assertEqual(worker.error, 'NoCredentialsError')
