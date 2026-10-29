"""Read-only AWS probe. Never receives/deletes SQS messages or writes S3 objects."""
import argparse
import json
from h3_service.config import Settings
from h3_service.contracts.models import task_id
from h3_service.media.cloud import clients, Store
from h3_service.aws_diagnostics import details, resource_call


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-id', type=task_id, required=True, help='a REAL task id returned by the existing Lambda')
    parser.add_argument('--video', action='store_true', help='also read video metadata; use for a known completed task')
    args = parser.parse_args()
    settings = Settings()
    settings.validate_cloud()
    s3, sqs = clients(settings)
    store = Store(settings, s3)
    print(json.dumps({'bucket': settings.bucket, 'status_prefix': settings.status_prefix,
                      'video_prefix': settings.prefix, 'region': settings.region}))
    success = True
    try:
        sqs.get_queue_attributes(QueueUrl=settings.queue_url, AttributeNames=['QueueArn'])
        print('SQS GetQueueAttributes: OK (does not prove ReceiveMessage permission)')
    except Exception as exc:
        success = False
        print(json.dumps({'check': 'queue', **details(exc)}))
    ident = args.task_id
    for video in ((False, True) if args.video else (False,)):
        key = store.key(ident, video)
        try:
            obj = resource_call(s3, 'get_object', Bucket=settings.bucket, Key=key, Range='bytes=0-0')
            obj['Body'].close()
            print(json.dumps({'check': 'video' if video else 'status', 'key': key, 'result': 'readable'}))
        except s3.exceptions.NoSuchKey:
            print(json.dumps({'check': 'video' if video else 'status', 'key': key, 'result': 'missing (404); check the task id/prefix'}))
        except Exception as exc:
            success = False
            print(json.dumps({'check': 'video' if video else 'status', **details(exc)}))
    print('No model loaded. No message received/deleted. No object written. PutObject permissions not tested.')
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
