"""Structured AWS error details, excluding request bodies, URLs and credential values."""
from botocore.exceptions import ClientError


def details(exc):
    error = exc
    for _ in range(4):
        if isinstance(error, ClientError):
            response = error.response
            return {'operation': error.operation_name,
                    'code': response.get('Error', {}).get('Code'),
                    'http_status': response.get('ResponseMetadata', {}).get('HTTPStatusCode'),
                    'request_id': response.get('ResponseMetadata', {}).get('RequestId'),
                    'resource': getattr(error, 'h3_resource', None)}
        error = error.__cause__ or error.__context__
        if error is None:
            break
    return {'type': type(exc).__name__}


def resource_call(client, method, **kwargs):
    try:
        return getattr(client, method)(**kwargs)
    except ClientError as exc:
        # Context for the top-level retry handler; no access-denied fallback or deletion.
        exc.h3_resource = {'bucket': kwargs.get('Bucket'), 'key': kwargs.get('Key')}
        raise
