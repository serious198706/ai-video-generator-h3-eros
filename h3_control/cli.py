"""Read-only configuration check and explicit durable outbox repair."""
import argparse
import json
import os
import sys

import boto3
from botocore.config import Config

from h3_control.store import Control


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['check', 'status', 'dispatch'])
    args = parser.parse_args()
    control = Control.from_env()
    if control is None:
        parser.error('H3_DYNAMODB_TABLE is required')
    if args.command == 'check':
        table = control.client.describe_table(TableName=control.table)['Table']
        keys = {v['AttributeName']: v['KeyType'] for v in table['KeySchema']}
        types = {v['AttributeName']: v['AttributeType'] for v in table['AttributeDefinitions']}
        if keys != {'pk': 'HASH', 'sk': 'RANGE'} or types.get('pk') != 'S' or types.get('sk') != 'S':
            raise ValueError('table requires string partition key pk and string sort key sk')
        control.pool()
        print(json.dumps({'table': control.table, 'namespace': control.namespace, 'status': table['TableStatus'],
                          'keys': keys, 'profile': control.profile, 'limit_seconds': control.limit,
                          'max_tasks': control.max_tasks, 'permissions': 'DescribeTable/GetItem only checked'}))
    elif args.command == 'status':
        data = json.loads(control.pool()['data'])
        print(json.dumps({'workers': data['workers'], 'unfinished': len(data['tasks']),
                          'dispatch_pending': sum(not t['sent'] for t in data['tasks'].values())}, indent=2))
    else:
        sqs = boto3.client('sqs', region_name=os.getenv('AWS_REGION', 'us-east-1'),
                           endpoint_url=os.getenv('H3_AWS_ENDPOINT') or None,
                           config=Config(connect_timeout=5, read_timeout=10,
                                         retries={'mode': 'standard', 'total_max_attempts': 3}))
        control.dispatch(sqs)
        print('Durable outbox dispatch completed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
