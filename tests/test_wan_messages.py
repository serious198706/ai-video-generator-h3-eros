import json
import unittest
from pydantic import ValidationError
from tests import test_worker_api
IDENT = test_worker_api.IDENT
from h3_service.contracts.models import Message, normalize_stored_request, validation_summary


class WanProtocolTests(unittest.TestCase):
    def payload(self):
        return {'task_id': IDENT, 'image_url': 'https://example.com/image', 'prompt': 'a calm forest',
                'negative_prompt': '', 'last_image_url': None, 'duration': 5, 'resolution': None,
                'webhook_url': None, 'seed': None, 'steps': None, 'quality': None, 'audio': False}

    def test_actual_wan_lambda_wire_shape(self):
        message = Message.model_validate(self.payload())
        self.assertEqual(message.request.image, 'https://example.com/image')
        self.assertEqual(message.request.steps, 6)
        self.assertEqual(message.request.resolution, '540p')
        self.assertEqual(normalize_stored_request(self.payload(), IDENT), message.request.model_dump())

    def test_reported_720p_four_step_request(self):
        data = {**self.payload(), 'resolution': '720p', 'steps': 4}
        message = Message.model_validate_json(json.dumps(data))
        self.assertEqual(message.request.duration, 5)
        self.assertEqual(message.request.resolution, '720p')
        self.assertEqual(message.request.steps, 4)
        self.assertFalse(message.request.audio)

    def test_image_alias_conflict_rejected(self):
        with self.assertRaises(ValidationError):
            Message.model_validate({**self.payload(), 'image': 'different'})

    def test_unsupported_parameters_still_rejected_and_logs_redacted(self):
        data = {**self.payload(), 'negative_prompt': 'private text', 'quality': 6}
        with self.assertRaises(ValidationError) as raised:
            Message.model_validate(data)
        summary = json.dumps(validation_summary(raised.exception))
        self.assertIn('quality', summary)
        self.assertNotIn('private text', summary)
        self.assertNotIn('example.com', summary)


class WanWorkerTests(unittest.TestCase):
    setUp = test_worker_api.WorkerTests.setUp
    run_job = test_worker_api.WorkerTests.run_job

    def test_flat_message_with_flat_existing_s3_request_generates_once(self):
        data = WanProtocolTests().payload()
        self.message['Body'] = json.dumps(data)
        self.store.states[IDENT] = {'task_id': IDENT, 'status': 'queued', 'request': data}
        self.run_job()
        self.assertEqual(self.store.states[IDENT]['status'], 'succeeded')
        self.assertEqual(self.store.states[IDENT]['request'], data)
        self.run_job()
        self.assertEqual(self.engine.calls, 1)
