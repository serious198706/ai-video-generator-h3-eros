import unittest
from unittest.mock import MagicMock
from botocore.exceptions import ClientError
from h3_service.aws_diagnostics import details, resource_call


class AwsDiagnosticsTests(unittest.TestCase):
    def test_denied_get_preserves_action_resource_and_request_id_without_secret_message(self):
        error = ClientError({'Error': {'Code': 'AccessDenied', 'Message': 'private input'},
                             'ResponseMetadata': {'HTTPStatusCode': 403, 'RequestId': 'test-request'}}, 'GetObject')
        client = MagicMock()
        client.get_object.side_effect = error
        with self.assertRaises(ClientError) as raised:
            resource_call(client, 'get_object', Bucket='test', Key='video/wan22/test.mp4')
        result = details(raised.exception)
        self.assertEqual(result['operation'], 'GetObject')
        self.assertEqual(result['resource']['key'], 'video/wan22/test.mp4')
        self.assertEqual(result['request_id'], 'test-request')
        self.assertNotIn('private input', str(result))
