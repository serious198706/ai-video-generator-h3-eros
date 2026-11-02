import copy
import json
import unittest
from dataclasses import replace
from unittest.mock import patch

from h3_service.contracts.models import Message
from tests import test_worker_api


IDENT = test_worker_api.IDENT


class UpscaleHandoffTests(unittest.TestCase):
    run_job = test_worker_api.WorkerTests.run_job

    def setUp(self):
        test_worker_api.WorkerTests.setUp(self)
        self.worker.settings = replace(self.settings, upscale_queue_url='upscale-queue')
        self.request.resolution = '1080p'
        self.message['Body'] = Message(task_id=IDENT, request=self.request).model_dump_json()

    # These cases exercise the new handoff without inheriting unrelated worker assertions.
    def test_handoff_native_and_no_callback(self):
        with patch('h3_service.tasks.worker.notify') as callback:
            self.run_job()
        state = self.store.states[IDENT]
        self.assertEqual(state['status'], 'generating')
        self.assertEqual(state['stage'], 'upscale_queued')
        self.assertEqual((state['result']['width'], state['result']['height']), (768, 1344))
        self.assertNotIn('video_url', state)
        self.assertEqual(json.loads(self.queue.sent[0]['MessageBody'])['target_height'], 1920)
        callback.assert_not_called()

    def test_send_failure_retries_without_generation_or_status_overwrite(self):
        self.queue.fail_send = True
        with self.assertRaises(OSError):
            self.run_job()
        self.assertFalse(self.queue.deleted)
        self.queue.fail_send = False
        self.store.states[IDENT]['stage'] = 'upscaling'
        saved = copy.deepcopy(self.store.states[IDENT])
        self.run_job()
        self.assertEqual(self.engine.calls, 1)
        self.assertEqual(self.store.states[IDENT], saved)
        self.assertEqual(len(self.queue.deleted), 1)

    def test_native_upload_recovered_after_status_failure(self):
        original = self.store.write
        def write(state):
            if state.get('upscale_handoff'):
                raise OSError('status unavailable')
            original(state)
        with patch.object(self.store, 'write', side_effect=write):
            with self.assertRaises(OSError):
                self.run_job()
        self.run_job()
        self.assertEqual(self.engine.calls, 1)
        self.assertEqual(self.store.states[IDENT]['stage'], 'upscale_queued')


