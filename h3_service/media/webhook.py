import hashlib
import hmac
import json
import time
import logging
from h3_service.net.http import request
from h3_service.contracts.models import normalize_stored_request

LOG = logging.getLogger(__name__)


def payload(state):
    result = state.get('result', {})
    requested = normalize_stored_request(state['request'], state['task_id'])
    return {'id': state['task_id'], 'task_id': state['task_id'], 'status': state['status'],
            'error': state.get('error'), 'video_url': state.get('video_url'),
            'duration': requested['duration'], 'resolution': requested['resolution'],
            'seed': result.get('seed', state.get('seed'))}


def notify(state, settings, cancelled):
    url = normalize_stored_request(state['request'], state['task_id']).get('webhook_url')
    if not url:
        LOG.info('webhook not requested task=%s', state['task_id'])
        return 'not_requested'
    body = json.dumps(payload(state), ensure_ascii=False, separators=(',', ':')).encode()
    headers = {'Content-Type': 'application/json'}
    if settings.webhook_secret:
        headers['X-Wan-Signature'] = 'sha256=' + hmac.new(settings.webhook_secret.encode(), body, hashlib.sha256).hexdigest()
    for index in range(3):
        if cancelled.is_set():
            return 'pending'
        try:
            LOG.info('webhook attempt task=%s attempt=%s/3 status=%s', state['task_id'], index + 1, state['status'])
            pool, response = request(url, hosts=settings.webhook_hosts, webhook=True, method='POST', body=body, headers=headers)
            try:
                LOG.info('webhook response task=%s attempt=%s http_status=%s', state['task_id'], index + 1, response.status)
                if 200 <= response.status < 300:
                    return 'delivered'
                if response.status < 500 and response.status != 429:
                    return 'exhausted'
            finally:
                response.close()
                pool.close()
        except Exception as exc:
            LOG.warning('webhook transport failed task=%s attempt=%s type=%s', state['task_id'], index + 1, type(exc).__name__)
            # No URL or body in logs; the durable callback outcome is the diagnostic.
            pass
        if index < 2:
            cancelled.wait(2**index)
    return 'exhausted'
