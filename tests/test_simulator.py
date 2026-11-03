import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from h3_service.config import Settings
from h3_service.contracts.models import GenerateRequest
from h3_service.media.cloud import Store, clients, now
from h3_service.simulator.cli import aggregate, task_rows
from h3_service.simulator.engine import SimulatedEngine, request_context, PLACEHOLDER
from h3_service.simulator.worker import SimulationWorker
from h3_service.infer.engine import LeaseLost
from h3_service.infer.manifest import fingerprint


class SimulatorTests(unittest.TestCase):
    def test_simulator_import_and_initialization_never_import_model_dependencies(self):
        code = '''import builtins
original = builtins.__import__
def checked(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'diffusers', 'transformers', 'torchao'}:
        raise AssertionError('model dependency imported: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = checked
from h3_service.simulator.app import app
from h3_service.simulator.engine import SimulatedEngine
from h3_service.config import Settings
from threading import Event
engine = SimulatedEngine(Settings())
engine.preflight(Event())
assert engine.healthy
'''
        subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, timeout=15)

    def test_profiles_and_simulation_fingerprint_are_separate_from_real_output(self):
        with patch.dict(os.environ, {'H3_SIM_PROFILE': '6000-maxq'}):
            engine = SimulatedEngine(Settings())
            self.assertAlmostEqual(sum(engine.table['15/1080p'].values()), 331.585, places=3)
            request = GenerateRequest(image='https://example.com/image', prompt='scene', steps=4).model_dump()
            self.assertNotEqual(engine.fingerprint(request, 42), fingerprint(request, 42))

    def test_generate_sleeps_under_generator_lock_and_cancellation_does_not_upload(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'H3_SIM_TIME_SCALE': '.001'}):
            engine = SimulatedEngine(Settings())
            cancelled = threading.Event(); engine.preflight(cancelled)
            token = request_context.set(GenerateRequest(image='x', prompt='scene', steps=4, duration=5, resolution='720p'))
            try:
                stages = []
                output, metrics = engine.generate({}, Path(temp), cancelled, stages.append)
                self.assertEqual(output.read_bytes(), PLACEHOLDER)
                self.assertTrue(metrics['simulation'])
                self.assertIn('generate_waiting_gpu', stages)
                self.assertIn('denoise_decode', stages)
                cancelled.set()
                with self.assertRaises(LeaseLost):
                    engine.generate({}, Path(temp), cancelled, stages.append)
            finally:
                request_context.reset(token)

    def test_reports_measure_queue_and_wait_to_generate_separately(self):
        rows = [{'status': 'succeeded', 'sqs_deleted': True, 'queue_s': i,
                 'wait_until_generation_s': i + 20, 'submitted_to_result_s': i + 40,
                 'profile': '6000-maxq', 'time_scale': 1} for i in range(50)]
        result = aggregate(rows, 50)
        self.assertEqual(result['queue_s']['max'], 49)
        self.assertEqual(result['wait_until_generation_s']['max'], 69)
        self.assertEqual(result['submitted_to_result_s']['max'], 89)
        self.assertEqual(result['messages_deleted'], 50)


try:
    from moto import mock_aws
except ImportError:
    mock_aws = None


@unittest.skipIf(mock_aws is None, 'install moto[s3,sqs,sts] for the 50-job AWS protocol test')
class SimulatorQueueTests(unittest.TestCase):
    def test_fifty_jobs_real_runner_sqs_s3_protocol_without_external_aws_or_models(self):
        with mock_aws(), tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {
            'AWS_ACCESS_KEY_ID': 'testing', 'AWS_SECRET_ACCESS_KEY': 'testing',
            'AWS_DEFAULT_REGION': 'us-east-1', 'AWS_EC2_METADATA_DISABLED': 'true',
            'H3_SIM_TIME_SCALE': '.0001', 'H3_SIM_PROFILE': '6000-maxq',
            'H3_SIM_IMAGE_SIZE': '32x48', 'H3_AWS_ENDPOINT': ''}):
            import boto3
            s3 = boto3.client('s3', region_name='us-east-1')
            sqs = boto3.client('sqs', region_name='us-east-1')
            s3.create_bucket(Bucket='h3-simulation-test')
            url = sqs.create_queue(QueueName='h3-simulation-test')['QueueUrl']
            settings = Settings(bucket='h3-simulation-test', queue_url=url, public_base='https://cdn.invalid',
                prefix='simulation/', status_prefix='simulation/', model_lifetime='dual_gpu',
                work_dir=Path(temp), shutdown_grace=10, max_receive=5)
            store = Store(settings, s3)
            from h3_service.simulator.cli import submit
            args = SimpleNamespace(count=50, durations=[5, 10, 15], resolutions=['720p', '1080p'],
                                   image_url='https://not-fetched.invalid/image.jpg', output=Path(temp) / 'batch')
            self.assertEqual(submit(settings, args), 0)
            batch = json.loads((args.output / 'batch.json').read_text())
            self.assertTrue(all(t['submission'] == 'submitted' for t in batch['tasks']))
            engine = SimulatedEngine(settings)
            worker = SimulationWorker(settings, store, sqs, engine)
            worker.start()
            try:
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline and worker.deleted_messages < 50 and not worker.error:
                    time.sleep(.1)
                self.assertIsNone(worker.error)
                self.assertEqual(worker.deleted_messages, 50)
            finally:
                worker.shutdown()
            for task in batch['tasks']:
                ident = task['task_id']
                state = store.read(ident)
                self.assertEqual(state['status'], 'succeeded')
                self.assertTrue(state['simulation']['enabled'])
                self.assertTrue(state['result']['simulation'])
                self.assertIn('wait_until_generation_s', state['timing'])
                self.assertGreaterEqual(state['timing']['queue_s'], 0)
                response = s3.get_object(Bucket=settings.bucket, Key=store.key(ident, True))
                with response['Body'] as stream:
                    self.assertEqual(stream.read(), PLACEHOLDER)
            rows = task_rows(Path(temp) / 'reports')
            from h3_service.contracts.models import normalize_stored_request
            sample = store.read(batch['tasks'][0]['task_id'])
            recovered = store.recover(sample['task_id'], engine.fingerprint(
                normalize_stored_request(sample['request'], sample['task_id']), sample['seed']))
            self.assertTrue(recovered['simulation'])
            result = aggregate(rows, 50)
            self.assertEqual(result['succeeded'], 50)
            self.assertEqual(result['messages_deleted'], 50)
            self.assertGreater(result['queue_s']['max'], result['queue_s']['min'])
            automatic = json.loads((Path(temp) / 'auto-report' / 'report.json').read_text())
            self.assertEqual(automatic['summary']['messages_deleted'], 50)
            self.assertEqual(len(sqs.receive_message(QueueUrl=url).get('Messages', [])), 0)
