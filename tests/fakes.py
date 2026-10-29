import copy
from h3_service.media.cloud import now


class MemoryStore:
    def __init__(self):
        self.states = {}
        self.videos = {}
        self.fail_read = False
        self.fail_terminal = False

    def read(self, ident):
        if self.fail_read:
            raise OSError('storage unavailable')
        return copy.deepcopy(self.states.get(ident))

    def write(self, state):
        if self.fail_terminal and state['status'] == 'succeeded':
            raise OSError('terminal unavailable')
        state['updated_at'] = now()
        self.states[state['task_id']] = copy.deepcopy(state)

    def recover(self, ident, stamp):
        if ident not in self.videos:
            return None
        result, fingerprint = self.videos[ident]
        if fingerprint != stamp:
            raise ValueError('conflict')
        return copy.deepcopy(result)

    def upload(self, ident, path, result, stamp):
        self.videos[ident] = (copy.deepcopy(result), stamp)

    def video_url(self, ident):
        return 'https://cdn.example.com/video/h3/' + ident + '.mp4'


class Queue:
    def __init__(self):
        self.deleted = []
        self.sent = []
        self.fail_send = False
        self.fail_renew = False

    def delete_message(self, **kwargs):
        self.deleted.append(kwargs)

    def send_message(self, **kwargs):
        if self.fail_send:
            raise OSError('uncertain send')
        self.sent.append(kwargs)

    def change_message_visibility(self, **kwargs):
        if self.fail_renew:
            raise OSError('lease lost')


class Engine:
    def __init__(self):
        self.calls = 0

    def generate(self, job, directory, cancelled, stage):
        self.calls += 1
        stage('conditioning')
        stage('denoise_decode')
        # Test-only fake; no production dry-run branch.
        return directory / 'unused.mp4', {'model': 'test-model', 'timing': {}}
