import json
from datetime import datetime, timezone
from urllib.parse import quote
import boto3
from botocore.config import Config
from h3_service.contracts.models import task_id


def now():
    return datetime.now(timezone.utc).isoformat()


def clients(settings):
    session = boto3.Session(region_name=settings.region)
    config = Config(connect_timeout=5, read_timeout=30,
                    retries={'total_max_attempts': 3, 'mode': 'standard'})
    options = {'config': config}
    if settings.endpoint:
        options['endpoint_url'] = settings.endpoint
    return session.client('s3', **options), session.client('sqs', **options)


class Store:
    def __init__(self, settings, s3):
        self.settings, self.s3 = settings, s3

    def key(self, ident, video=False):
        task_id(ident)
        tail = f'{ident}.mp4' if video else f'status/{ident}.json'
        return self.settings.prefix.strip('/') + '/' + tail

    def read(self, ident):
        try:
            response = self.s3.get_object(Bucket=self.settings.bucket, Key=self.key(ident))
        except self.s3.exceptions.NoSuchKey:
            return None
        with response['Body'] as stream:
            state = json.loads(stream.read(256 * 1024))
        if not isinstance(state, dict) or state.get('task_id') != ident:
            raise ValueError('invalid persisted state')
        return state

    def write(self, state):
        state['updated_at'] = now()
        self.s3.put_object(Bucket=self.settings.bucket, Key=self.key(state['task_id']),
                           Body=json.dumps(state, ensure_ascii=False, allow_nan=False).encode(),
                           ContentType='application/json', CacheControl='no-store')

    def upload(self, ident, path, result, fingerprint):
        # Metadata committed atomically with the MP4 enables recovery after status-write failure.
        metadata = {'h3-result': json.dumps(result, separators=(',', ':'), allow_nan=False),
                    'task-id': ident, 'fingerprint': fingerprint}
        if sum(len(k) + len(v.encode()) for k, v in metadata.items()) > 1900:
            raise ValueError('result metadata too large')
        self.s3.upload_file(str(path), self.settings.bucket, self.key(ident, True),
                            ExtraArgs={'ContentType': 'video/mp4', 'Metadata': metadata})

    def recover(self, ident, fingerprint):
        # GET has a typed NoSuchKey error, unlike HEAD's generic 404.
        try:
            response = self.s3.get_object(Bucket=self.settings.bucket, Key=self.key(ident, True), Range='bytes=0-0')
        except self.s3.exceptions.NoSuchKey:
            return None
        response['Body'].close()
        metadata = response.get('Metadata', {})
        if metadata.get('task-id') != ident or metadata.get('fingerprint') != fingerprint:
            raise ValueError('existing video does not match this job')
        result = json.loads(metadata['h3-result'])
        if not isinstance(result, dict) or not {'seed', 'num_frames', 'width', 'height', 'audio_present'} <= result.keys():
            raise ValueError('invalid recovery metadata')
        return result

    def video_url(self, ident):
        return self.settings.public_base.rstrip('/') + '/' + quote(self.key(ident, True), safe='/')
