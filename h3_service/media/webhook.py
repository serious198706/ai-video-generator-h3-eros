import hashlib
import hmac
import json
import time
from h3_service.net.http import request
from h3_service.contracts.models import normalize_stored_request


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
        return 'not_requested'
    body = json.dumps(payload(state), ensure_ascii=False, separators=(',', ':')).encode()
    headers = {'Content-Type': 'application/json'}
    if settings.webhook_secret:
        headers['X-Wan-Signature'] = 'sha256=' + hmac.new(settings.webhook_secret.encode(), body, hashlib.sha256).hexdigest()
    for index in range(3):
        if cancelled.is_set():
            return 'pending'
        try:
            pool, response = request(url, hosts=settings.webhook_hosts, webhook=True, method='POST', body=body, headers=headers)
            try:
                if 200 <= response.status < 300:
                    return 'delivered'
                if response.status < 500 and response.status != 429:
                    return 'exhausted'
            finally:
                response.close()
                pool.close()
        except Exception:
            # No URL or body in logs; the durable callback outcome is the diagnostic.
            pass
        if index < 2:
            cancelled.wait(2**index)
    return 'exhausted'
