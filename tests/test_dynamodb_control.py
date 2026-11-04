import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import boto3
try:
    from moto import mock_aws
except ImportError:
    mock_aws = None

from h3_control.store import Control, CapacityExceeded, Conflict, LeaseLost


@unittest.skipIf(mock_aws is None, 'install moto[dynamodb,sqs,s3] for DynamoDB integration tests')
class DynamoDBControlTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'AWS_ACCESS_KEY_ID': 'testing', 'AWS_SECRET_ACCESS_KEY': 'testing',
            'AWS_DEFAULT_REGION': 'us-east-1', 'AWS_EC2_METADATA_DISABLED': 'true',
            'H3_DYNAMODB_TABLE': 'h3-control-test', 'H3_CONTROL_NAMESPACE': 'test',
            'H3_SIM_TIME_SCALE': '.00001', 'H3_SIM_PROFILE': '6000-s', 'H3_SIM_IMAGE_SIZE': '32x48'})
        self.env.start(); self.addCleanup(self.env.stop)
        self.mock = mock_aws(); self.mock.start(); self.addCleanup(self.mock.stop)
        client = boto3.client('dynamodb', region_name='us-east-1')
        client.create_table(TableName='h3-control-test', BillingMode='PAY_PER_REQUEST',
            KeySchema=[{'AttributeName': 'pk', 'KeyType': 'HASH'}, {'AttributeName': 'sk', 'KeyType': 'RANGE'}],
            AttributeDefinitions=[{'AttributeName': 'pk', 'AttributeType': 'S'}, {'AttributeName': 'sk', 'AttributeType': 'S'}])
        self.control = Control.from_env()
        self.sqs = boto3.client('sqs', region_name='us-east-1')
        self.queue = self.sqs.create_queue(QueueName='control-test')['QueueUrl']
        self.request = {'image': 'https://example.com/a.jpg', 'prompt': 'Animate the scene.',
                        'duration': 5, 'resolution': '720p', 'steps': 4, 'audio': False}

    def ready(self, count=3):
        for i in range(count):
            self.control.heartbeat(f'worker-{i}', f'run-{i}', True)

    def test_no_ready_workers_rejects_without_creating_task(self):
        with self.assertRaises(CapacityExceeded) as error:
            self.control.admit(self.request, self.queue, 'key')
        self.assertEqual(error.exception.detail['reason'], 'no_ready_workers')
        self.assertEqual(json.loads(self.control.pool()['data'])['tasks'], {})

    def test_admission_cap_idempotency_and_conflicting_request(self):
        self.ready()
        for i in range(18):
            self.control.admit(self.request, self.queue, str(i))
        with self.assertRaises(CapacityExceeded):
            self.control.admit(self.request, self.queue, 'overflow')
        state, created = self.control.admit(self.request, self.queue, '0')
        self.assertFalse(created)
        with self.assertRaises(Conflict):
            self.control.admit({**self.request, 'duration': 10}, self.queue, '0')
        self.assertEqual(len(json.loads(self.control.pool()['data'])['tasks']), 18)

    def test_long_jobs_rejected_before_count_cap_and_stale_worker_not_capacity(self):
        self.ready()
        for i in range(9):
            self.control.admit({**self.request, 'duration': 15}, self.queue, str(i))
        with self.assertRaises(CapacityExceeded):
            self.control.admit({**self.request, 'duration': 15}, self.queue, 'extra')
        self.control.change_pool(lambda pool: [w.update(seen=0) for w in pool['workers'].values()])
        with self.assertRaises(CapacityExceeded) as error:
            self.control.admit(self.request, self.queue)
        self.assertEqual(error.exception.detail['reason'], 'no_ready_workers')

    def test_cas_rejects_stale_snapshot_without_creating_orphan(self):
        self.ready()
        old = self.control.pool()
        data = json.loads(old['data'])
        self.control.admit(self.request, self.queue, 'first')
        self.assertFalse(self.control.commit(old, data))
        self.assertEqual(len(json.loads(self.control.pool()['data'])['tasks']), 1)

    def test_sqs_timeout_after_acceptance_replays_without_second_generation(self):
        self.ready()
        state, _ = self.control.admit(self.request, self.queue, 'timeout')
        real_send = self.sqs.send_message
        def uncertain(**kwargs):
            real_send(**kwargs)
            raise TimeoutError('response lost')
        with patch.object(self.sqs, 'send_message', side_effect=uncertain):
            with self.assertRaises(TimeoutError):
                self.control.dispatch(self.sqs)
        self.control.dispatch(self.sqs)
        messages = self.sqs.receive_message(QueueUrl=self.queue, MaxNumberOfMessages=10)['Messages']
        self.assertEqual(len(messages), 2)
        ident = state['task_id']
        self.assertTrue(self.control.claim(ident, 'attempt-a', 'worker-0', 'run-0'))
        self.assertFalse(self.control.claim(ident, 'attempt-b', 'worker-1', 'run-1'))
        self.assertEqual(self.control.read(ident)['attempt_count'], 1)

    def test_expired_owner_fenced_and_terminal_release_atomic(self):
        self.ready()
        state, _ = self.control.admit(self.request, self.queue, 'lease')
        ident = state['task_id']
        self.assertTrue(self.control.claim(ident, 'old', 'worker-0', 'run-0'))
        self.control.client.update_item(TableName=self.control.table,
            Key={'pk': {'S': f'test#TASK#{ident}'}, 'sk': {'S': 'STATE'}},
            UpdateExpression='SET lease_until = :zero', ExpressionAttributeValues={':zero': {'N': '0'}})
        self.assertTrue(self.control.claim(ident, 'new', 'worker-1', 'run-1'))
        with self.assertRaises(LeaseLost):
            self.control.write({**state, 'status': 'succeeded'}, 'old')
        self.control.write({**state, 'status': 'succeeded', 'video_url': 'https://cdn/x.mp4'}, 'new')
        self.assertEqual(json.loads(self.control.pool()['data'])['tasks'], {})
        item = self.control.get(ident)
        self.assertIn('expires_at', item)
        self.assertNotIn('message', item)
        with self.assertRaises(LeaseLost):
            self.control.write(state, 'new')

    def test_duplicate_live_worker_ids_fail_closed(self):
        self.ready()
        with self.assertRaises(Conflict):
            self.control.heartbeat('worker-0', 'different-run', True)

    def test_existing_worker_end_to_end_real_state_and_immutable_s3(self):
        from h3_service.config import Settings
        from h3_service.media.cloud import Store
        from h3_service.simulator.engine import SimulatedEngine, PLACEHOLDER
        from h3_service.simulator.worker import SimulationWorker
        self.ready()
        s3 = boto3.client('s3', region_name='us-east-1')
        s3.create_bucket(Bucket='h3-control-output')
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(bucket='h3-control-output', queue_url=self.queue,
                public_base='https://cdn.invalid', model_lifetime='dual_gpu', work_dir=Path(directory))
            store = Store(settings, s3)
            engine = SimulatedEngine(settings)
            worker = SimulationWorker(settings, store, self.sqs, engine)
            engine.preflight(worker.cancelled)
            worker.ready = True
            state, _ = self.control.admit({**self.request, 'resolution': '1080p'}, self.queue, 'full')
            self.control.dispatch(self.sqs)
            raw = self.sqs.receive_message(QueueUrl=self.queue)['Messages'][0]
            worker.handle(raw)
            result = self.control.read(state['task_id'])
            self.assertEqual(result['status'], 'succeeded')
            self.assertTrue(result['result']['simulation'])
            self.assertIn(result['video_object_key'], result['video_url'])
            self.assertTrue(result['video_object_key'].endswith('.mp4'))
            with s3.get_object(Bucket=settings.bucket, Key=result['video_object_key'])['Body'] as body:
                self.assertEqual(body.read(), PLACEHOLDER)
            # No status JSON gets written to S3 in DynamoDB mode.
            keys = s3.list_objects_v2(Bucket=settings.bucket)['Contents']
            self.assertEqual(len(keys), 1)
            self.assertEqual(json.loads(self.control.pool()['data'])['tasks'], {})


if __name__ == '__main__':
    unittest.main()
