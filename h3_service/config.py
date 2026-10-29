"""Environment only; importing configuration never creates AWS/GPU clients."""
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    region: str = field(default_factory=lambda: os.getenv('AWS_REGION', 'us-east-1'))
    bucket: str = field(default_factory=lambda: os.getenv('H3_S3_BUCKET', ''))
    prefix: str = field(default_factory=lambda: os.getenv('H3_S3_PREFIX', 'video/h3/'))
    public_base: str = field(default_factory=lambda: os.getenv('H3_PUBLIC_BASE_URL', ''))
    queue_url: str = field(default_factory=lambda: os.getenv('H3_QUEUE_URL', ''))
    endpoint: str = field(default_factory=lambda: os.getenv('H3_AWS_ENDPOINT', ''))
    image_hosts: tuple = field(default_factory=lambda: tuple(filter(None, os.getenv('H3_IMAGE_HOSTS', '').split(','))))
    webhook_hosts: tuple = field(default_factory=lambda: tuple(filter(None, os.getenv('H3_WEBHOOK_HOSTS', '').split(','))))
    webhook_secret: str = field(default_factory=lambda: os.getenv('H3_WEBHOOK_SECRET', ''))
    work_dir: Path = field(default_factory=lambda: Path(os.getenv('H3_WORK_DIR', str(ROOT / 'data'))))
    model_dir: Path = field(default_factory=lambda: Path(os.getenv('H3_MODEL_DIR', str(ROOT / 'models/base'))))
    eros_file: Path = field(default_factory=lambda: Path(os.getenv('H3_EROS_FILE', str(ROOT / 'models/eros/10Eros_Max_h3_TURBO-hybrid_beta5.safetensors'))))
    attention: str = field(default_factory=lambda: os.getenv('H3_ATTENTION', '_native_cudnn'))
    visibility: int = 600
    heartbeat: int = 60
    max_receive: int = 5
    max_queue: int = field(default_factory=lambda: int(os.getenv('H3_MAX_QUEUE', '0')))
    stall_seconds: int = field(default_factory=lambda: int(os.getenv('H3_STALL_SECONDS', '2700')))
    phase_timeout: int = field(default_factory=lambda: int(os.getenv('H3_PHASE_TIMEOUT', '3600')))
    min_free_gib: float = field(default_factory=lambda: float(os.getenv('H3_MIN_FREE_GIB', '88')))
    resident_min_ram_gib: float = field(default_factory=lambda: float(os.getenv('H3_RESIDENT_MIN_RAM_GIB', '160')))
    model_lifetime: str = field(default_factory=lambda: os.getenv('H3_MODEL_LIFETIME', 'resident'))

    def validate_cloud(self):
        if self.model_lifetime not in {'resident', 'cold'}:
            raise ValueError('H3_MODEL_LIFETIME must be resident or cold')
        if not self.bucket or not self.queue_url or not self.public_base.startswith('https://'):
            raise ValueError('H3_S3_BUCKET, H3_QUEUE_URL and HTTPS H3_PUBLIC_BASE_URL are required')
        if not self.prefix.strip('/'):
            raise ValueError('H3_S3_PREFIX must be nonempty')
        if not 0 < self.heartbeat < self.visibility / 2:
            raise ValueError('invalid heartbeat/visibility settings')
