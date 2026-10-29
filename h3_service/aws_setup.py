"""Hidden credential input and STS validation; optional explicitly requested .env storage."""
import getpass
import os
import sys
import shlex
import tempfile
from pathlib import Path
import boto3
from botocore.config import Config
from botocore.exceptions import NoCredentialsError


def verify_credentials(region):
    session = boto3.Session(region_name=region)
    if session.get_credentials() is None:
        raise NoCredentialsError()
    session.client('sts', config=Config(connect_timeout=5, read_timeout=10,
        retries={'total_max_attempts': 2, 'mode': 'standard'})).get_caller_identity()


def verify_queue(settings, sqs=None):
    if not settings.queue_url:
        return  # Credential setup is also usable before machine-specific queue configuration.
    if sqs is None:
        sqs = boto3.Session(region_name=settings.region).client('sqs', config=Config(
            connect_timeout=5, read_timeout=10, retries={'total_max_attempts': 2, 'mode': 'standard'}))
    sqs.get_queue_attributes(QueueUrl=settings.queue_url, AttributeNames=['QueueArn'])


def save_credentials(path, region):
    """Persist only explicitly supplied static credentials, after successful STS validation."""
    values = {k: os.environ[k] for k in ('AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN') if os.environ.get(k)}
    if not {'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY'} <= values.keys():
        return False  # Profiles/roles keep their credential provider and refresh mechanism.
    values['AWS_REGION'] = region
    removed = set(values) | {'AWS_SESSION_TOKEN', 'AWS_DEFAULT_REGION'}
    lines = path.read_text().splitlines() if path.exists() else []
    kept = [line for line in lines if line.strip().removeprefix('export ').split('=', 1)[0].strip() not in removed]
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.aws-env-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as output:
            output.write('\n'.join(kept + [k + '=' + shlex.quote(v) for k, v in values.items()]) + '\n')
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temp, 0o600)
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)
    return True


def bootstrap(region, *, force=False, save=None):
    # Vast has no EC2 instance role; avoid unnecessary metadata credential probes.
    os.environ.setdefault('AWS_EC2_METADATA_DISABLED', 'true')
    session = boto3.Session(region_name=region)
    credentials = None if force else session.get_credentials()
    if credentials is None:
        if not sys.stdin.isatty():
            raise NoCredentialsError()
        print('AWS credentials required for SQS and S3; inputs are hidden.', flush=True)
        key = getpass.getpass('AWS Access Key ID: ').strip()
        secret = getpass.getpass('AWS Secret Access Key: ').strip()
        token = getpass.getpass('AWS Session Token (empty for long-lived keys): ').strip()
        if not key or not secret:
            raise NoCredentialsError()
        os.environ.update(AWS_ACCESS_KEY_ID=key, AWS_SECRET_ACCESS_KEY=secret)
        os.environ.pop('AWS_SESSION_TOKEN', None)
        if token:
            os.environ['AWS_SESSION_TOKEN'] = token
    verify_credentials(region)
    return save_credentials(save, region) if save is not None else False


def main():
    arguments = sys.argv[1:]
    save = None
    if arguments and arguments[0] == '--save':
        save = Path(arguments[1])
        arguments = arguments[2:]
    force = bool(arguments and arguments[0] == '--force')
    if force:
        arguments.pop(0)
    if arguments and arguments[0] == '--':
        arguments.pop(0)
    from h3_service.config import Settings
    try:
        saved = bootstrap(Settings().region, force=force, save=save)
        verify_queue(Settings())
    except Exception as exc:
        code = getattr(exc, 'response', {}).get('Error', {}).get('Code', type(exc).__name__)
        print(f'AWS credential/queue check failed ({code}); models were not loaded. '
              'Supply credentials through the hidden prompt, AWS profile or process environment.', file=sys.stderr)
        return 1
    print('AWS STS and configured queue checks passed; S3 access is checked when used.', flush=True)
    if not arguments:
        print('Credentials saved (mode 600).' if saved else 'Credential provider validated; no file written.', flush=True)
        return 0
    os.execvpe(arguments[0], arguments, os.environ)


if __name__ == '__main__':
    sys.exit(main())
