import importlib.util
import io
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from h3_service.config import Settings
from h3_service.media.cloud import Store
from h3_service.media.webhook import payload
from tests import test_worker_api
from tests import test_wan_messages


class MemoryS3:
    exceptions = SimpleNamespace(NoSuchKey=KeyError)
    def __init__(self):
        self.objects = {}
    def put_object(self, **kwargs):
        self.objects[kwargs['Key']] = kwargs['Body']
    def get_object(self, **kwargs):
        return {'Body': io.BytesIO(self.objects[kwargs['Key']])}


class WanIntegrationTests(unittest.TestCase):
    def test_original_config_names_and_independent_status_location(self):
        with patch.dict(os.environ, {'SQS_QUEUE_URL': 'queue', 'WAN22_S3_BUCKET': 'bucket',
                'WAN22_S3_PREFIX': 'video/wan22/', 'STATUS_PREFIX': 'video/wan22/',
                'H3_S3_PREFIX': 'video/h3/', 'WAN22_S3_PUBLIC_BASE_URL': 'https://cdn.example.com',
                'WAN22_WEBHOOK_SECRET': 'test-key'}, clear=True):
            settings = Settings()
            settings.validate_cloud()
            store = Store(settings, MemoryS3())
            ident = test_worker_api.IDENT
            self.assertEqual(settings.queue_url, 'queue')
            self.assertEqual(settings.webhook_secret, 'test-key')
            self.assertEqual(store.key(ident), f'video/wan22/status/{ident}.json')
            self.assertEqual(store.key(ident, True), f'video/h3/{ident}.mp4')

    def test_existing_wan_lambda_can_read_h3_state_and_callback_fields_match(self):
        path = Path(__file__).resolve().parents[2] / 'server/lambda_api/status.py'
        if not path.exists():
            self.skipTest('original sibling Wan source unavailable')
        s3 = MemoryS3()
        with patch.dict(os.environ, {'S3_BUCKET': 'bucket', 'STATUS_PREFIX': 'video/wan22/'}, clear=True), \
                patch('boto3.client', return_value=s3):
            spec = importlib.util.spec_from_file_location('original_wan_status', path)
            legacy = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(legacy)
            ident = test_worker_api.IDENT
            state = {'task_id': ident, 'status': 'succeeded', 'request': test_wan_messages.WanProtocolTests().payload(),
                'seed': 42, 'video_url': 'https://cdn.example.com/video/h3/result.mp4'}
            Store(Settings(bucket='bucket'), s3).write(state)
            result = legacy.describe(ident)
            self.assertEqual(result['status'], 'succeeded')
            self.assertEqual(result['video_url'], state['video_url'])
            self.assertEqual(result['request'], state['request'])
            self.assertEqual(set(payload(state)), {'id', 'task_id', 'status', 'video_url', 'error', 'seed', 'duration', 'resolution'})
