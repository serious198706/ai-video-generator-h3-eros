import math
import re
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def task_id(value: str) -> str:
    if not re.fullmatch(r'[0-9a-f]{32}', value):
        raise ValueError('invalid task_id')
    return value


def frame_count(seconds: float) -> int:
    frames = math.ceil(seconds * 24)
    return frames + (5 - frames) % 17


class GenerateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra='forbid', allow_inf_nan=False)
    image: str = Field(min_length=1, max_length=8192)
    prompt: str = Field(min_length=1, max_length=12000)
    last_image: str | None = Field(default=None, alias='lastImage', max_length=8192)
    webhook_url: str | None = Field(default=None, alias='webhookUrl', max_length=8192)
    duration: float = Field(default=5, ge=2, le=14)
    resolution: Literal['540p', '720p'] = '540p'
    steps: int = Field(default=6, ge=4, le=8, strict=True)
    seed: int | None = Field(default=None, ge=0, le=2**63-1, strict=True)
    audio: bool = False
    negative_prompt: str | None = Field(default=None, alias='negativePrompt')
    quality: None = None

    @field_validator('image', 'prompt', 'last_image', 'webhook_url', mode='before')
    @classmethod
    def strip(cls, value):
        return value.strip() if isinstance(value, str) else value

    @field_validator('audio', mode='before')
    @classmethod
    def audio_null(cls, value):
        if value is None:
            return False
        if not isinstance(value, bool):
            raise ValueError('audio must be a boolean')
        return value

    @field_validator('negative_prompt')
    @classmethod
    def no_negative(cls, value):
        if value and value.strip():
            raise ValueError('negativePrompt is unsupported by H3')
        return None


class Message(BaseModel):
    model_config = ConfigDict(extra='forbid')
    schema_version: Literal[1] = 1
    task_id: str
    request: GenerateRequest

    @field_validator('task_id')
    @classmethod
    def valid_id(cls, value):
        return task_id(value)

    @model_validator(mode='before')
    @classmethod
    def wan_flat_message(cls, value):
        if not isinstance(value, dict) or 'request' in value:
            return value
        fields = dict(value)
        envelope = {'task_id': fields.pop('task_id', None)}
        if 'schema_version' in fields:
            envelope['schema_version'] = fields.pop('schema_version')
        for old, new in [('image_url', 'image'), ('last_image_url', 'last_image')]:
            if old in fields:
                if new in fields and fields[new] != fields[old]:
                    raise ValueError('conflicting image aliases')
                fields[new] = fields.pop(old)
        # Wan serializes optional defaults as null; H3 resolves these explicitly.
        for name in ('duration', 'resolution', 'steps'):
            if fields.get(name, 'present') is None:
                fields.pop(name)
        envelope['request'] = fields
        return envelope


def normalize_stored_request(value, ident):
    if not isinstance(value, dict):
        raise ValueError('stored request must be an object')
    if 'task_id' in value:
        message = Message.model_validate(value)
        if message.task_id != ident:
            raise ValueError('stored request task identity conflict')
    else:
        message = Message.model_validate({'task_id': ident, **value})
    return message.request.model_dump()


def validation_summary(exc):
    """No input values, user-provided messages/context, prompts or signed URLs."""
    known = set(GenerateRequest.model_fields) | {'request', 'task_id', 'schema_version',
        'negativePrompt', 'lastImage', 'webhookUrl', 'image_url', 'last_image_url'}
    return [{'field': '.'.join(str(part) if part in known or isinstance(part, int) else '<unknown-field>'
                              for part in error['loc']), 'type': error['type']}
            for error in exc.errors(include_input=False, include_context=False, include_url=False)]
