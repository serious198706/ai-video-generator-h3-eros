"""Validate a captured SQS JSON file locally without loading models or touching AWS."""
import json
import sys
from pathlib import Path
from pydantic import ValidationError
from h3_service.contracts.models import Message, validation_summary


def main():
    try:
        envelope = Message.model_validate_json(Path(sys.argv[1]).read_text())
        print(json.dumps({'valid': True, 'task_id': envelope.task_id, 'resolution': envelope.request.resolution,
                          'steps': envelope.request.steps, 'duration': envelope.request.duration}))
        return 0
    except ValidationError as exc:
        print(json.dumps({'valid': False, 'fields': validation_summary(exc)}))
        return 1


if __name__ == '__main__':
    sys.exit(main())
