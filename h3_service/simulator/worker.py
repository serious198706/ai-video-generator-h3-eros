import time
import os
import logging
from datetime import datetime, timezone
from PIL import Image
from h3_service.tasks.worker import Worker
from h3_service.simulator.engine import request_context


class SimulationWorker(Worker):
    def handle(self, message):
        try:
            return super().handle(message)
        finally:
            # Each delivery report is already persisted by the real runner.
            # A separate output avoids racing the interactive report command.
            from h3_service.simulator.cli import task_rows, aggregate, save
            from h3_service.identity import producer
            try:
                with self.engine.report_lock:
                    rows = task_rows(self.settings.work_dir / 'reports', run=producer()['run_id'])
                    summary = aggregate(rows, int(os.getenv('H3_SIM_EXPECTED_TASKS', '50')))
                    summary['run_id'] = producer()['run_id']
                    save(self.settings.work_dir / 'auto-report', rows, summary)
            except (OSError, ValueError, KeyError) as exc:
                logging.getLogger(__name__).error('simulation aggregate report failed type=%s', type(exc).__name__)

    def new_child(self):
        return SimulationWorker(self.settings, self.store, self.sqs, self.engine)

    def download_input(self, url, path):
        # User image URLs are recorded by the normal safe trace, never fetched.
        image = Image.new('RGB', self.engine.settings_image_size(), '#404040')
        try:
            image.save(path, format='PNG')
        finally:
            image.close()

    def execute(self, request, state, persist, stamp, trace):
        token = request_context.set(request)
        try:
            return super().execute(request, state, persist, stamp, trace)
        finally:
            request_context.reset(token)

    def decorate_state(self, state, trace):
        simulation = getattr(trace, 'simulation', None)
        if simulation is None:
            received = trace.received_at
            sent = datetime.fromtimestamp(int(trace.sqs_sent_ms) / 1000, timezone.utc) if trace.sqs_sent_ms else None
            simulation = {'enabled': True, 'profile': self.engine.profile, 'time_scale': self.engine.scale,
                          'received_at': received.isoformat(), 'queue_sent_at': sent.isoformat() if sent else None,
                          'placeholder': True, 'image_source': 'local_placeholder'}
            trace.simulation = simulation
            trace.queue_timing = {'queue_s': round(max(0, (received - sent).total_seconds()), 3)} if sent else {}
        elapsed = time.monotonic() - trace.started
        if state.get('stage') == 'denoise_decode' and 'generation_started_at' not in simulation:
            simulation['generation_started_at'] = datetime.now(timezone.utc).isoformat()
            if 'queue_s' in trace.queue_timing:
                trace.queue_timing['wait_until_generation_s'] = round(trace.queue_timing['queue_s'] + elapsed, 3)
        if state['status'] in {'succeeded', 'failed'} and 'finished_at' not in simulation:
            simulation['finished_at'] = datetime.now(timezone.utc).isoformat()
            if 'queue_s' in trace.queue_timing:
                trace.queue_timing['submitted_to_result_s'] = round(trace.queue_timing['queue_s'] + elapsed, 3)
        state['simulation'] = dict(simulation)
        state.setdefault('timing', {}).update(trace.queue_timing)
        trace.annotations = {'simulation': dict(simulation), 'queue_timing': dict(trace.queue_timing),
                             'timing': dict(state.get('timing', {}))}
