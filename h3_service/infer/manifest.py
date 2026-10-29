import hashlib
import json
from pathlib import Path

BASE_REPO = 'MiniMaxAI/MiniMax-H3'
BASE_REVISION = '42ed227ee7df40d41602854ae760620d6eb651fe'
EROS_REPO = 'TenStrip/10Eros-Max'
EROS_REVISION = '5794cf604068adad01d9232db1c4e55964d3cbfb'
EROS_NAME = '10Eros_Max_h3_TURBO-hybrid_beta5.safetensors'
EROS_SHA256 = '098f138f3e899d03821dd8296c8db3db6fa16371e50fcdd38661d347fd9a1dfa'
DIFFUSERS_REVISION = '665f578278365ea4a3318cb8c9b66ce6c01204b9'
SPACE_REVISION = '157975f3eb07b3d17a1601f5f3f8dbf0949758ab'
MODEL_ID = '10eros-beta5-' + EROS_SHA256[:12]
COMPONENTS = ('transformer', 'vae', 'audio_vae', 'text_encoder', 'tokenizer', 'processor', 'scheduler', 'audio_scheduler')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(request, seed):
    payload = {'request': request, 'seed': seed, 'model': MODEL_ID,
               'base': BASE_REVISION, 'diffusers': DIFFUSERS_REVISION, 'contract': 1}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def verify(settings):
    if sha256(settings.eros_file) != EROS_SHA256:
        raise ValueError('Eros SHA-256 mismatch')
    stamp = json.loads((settings.model_dir / 'h3-manifest.json').read_text())
    if stamp.get('revision') != BASE_REVISION:
        raise ValueError('base revision mismatch; use download_models')
    files = stamp.get('files', {})
    if not files or 'modular_model_index.json' not in files:
        raise ValueError('missing base file hashes; run download_models')
    root = settings.model_dir.resolve()
    for relative, expected in files.items():
        path = root / relative
        if not path.resolve().is_relative_to(root) or sha256(path) != expected:
            raise ValueError(f'base file hash mismatch: {relative}')
    for component in COMPONENTS:
        if not (settings.model_dir / component).is_dir():
            raise ValueError(f'missing component: {component}')
