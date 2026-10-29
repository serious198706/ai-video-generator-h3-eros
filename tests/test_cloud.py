import io
import json
import unittest
import boto3
from botocore.response import StreamingBody
from botocore.stub import Stubber
from h3_service.config import Settings
from h3_service.media.cloud import Store

IDENT = 'b' * 32


class CloudTests(unittest.TestCase):
    def setUp(self):
        # Fake local test credentials only; no credential discovery or network.
        self.client = boto3.client('s3', region_name='us-east-1', aws_access_key_id='testing', aws_secret_access_key='testing')
        self.stub = Stubber(self.client)
        self.stub.activate()
        self.addCleanup(self.stub.deactivate)
        self.store = Store(Settings(bucket='test', public_base='https://cdn.example.com'), self.client)

    def test_missing_is_none(self):
        self.stub.add_client_error('get_object', service_error_code='NoSuchKey', http_status_code=404,
                                  expected_params={'Bucket': 'test', 'Key': f'video/wan22/status/{IDENT}.json'})
        self.assertIsNone(self.store.read(IDENT))
        self.stub.assert_no_pending_responses()

    def test_access_denied_is_not_missing(self):
        self.stub.add_client_error('get_object', service_error_code='AccessDenied', http_status_code=403)
        with self.assertRaises(Exception):
            self.store.read(IDENT)

    def test_closes_stream(self):
        body = json.dumps({'task_id': IDENT, 'status': 'queued'}).encode()
        raw = io.BytesIO(body)
        stream = StreamingBody(raw, len(body))
        self.stub.add_response('get_object', {'Body': stream})
        self.assertEqual(self.store.read(IDENT)['status'], 'queued')
        self.assertTrue(raw.closed)

    def test_recovery_requires_matching_fingerprint(self):
        raw = io.BytesIO(b'x')
        self.stub.add_response('get_object', {'Body': StreamingBody(raw, 1),
            'Metadata': {'task-id': IDENT, 'fingerprint': 'other', 'h3-result': '{}'}})
        with self.assertRaises(ValueError):
            self.store.recover(IDENT, 'wanted')
        self.assertTrue(raw.closed)
