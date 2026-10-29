import hashlib
import json
import time
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


def sha256(path, progress=None, cancelled=None):
    digest = hashlib.sha256()
    total = Path(path).stat().st_size
    read = 0
    started = last = time.monotonic()
    if progress:
        progress(0, total, 0.0)
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            if cancelled is not None and cancelled.is_set():
                raise InterruptedError('weight verification cancelled')
            digest.update(chunk)
            read += len(chunk)
            current = time.monotonic()
            if progress and current - last >= 5:
                progress(read, total, current - started)
                last = current
    if progress:
        progress(read, total, time.monotonic() - started)
    return digest.hexdigest()


def fingerprint(request, seed):
    payload = {'request': request, 'seed': seed, 'model': MODEL_ID,
               'base': BASE_REVISION, 'diffusers': DIFFUSERS_REVISION, 'contract': 1}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def verify(settings, progress=None, cancelled=None):
    emit = progress or (lambda message: None)
    stamp = json.loads((settings.model_dir / 'h3-manifest.json').read_text())
    if stamp.get('revision') != BASE_REVISION:
        raise ValueError('base revision mismatch; use download_models')
    files = stamp.get('files', {})
    if not files or 'modular_model_index.json' not in files:
        raise ValueError('missing base file hashes; run download_models')
    root = settings.model_dir.resolve()
    checks = [(settings.eros_file, EROS_SHA256, 'Eros/' + settings.eros_file.name)]
    for relative, expected in files.items():
        path = root / relative
        if not path.resolve().is_relative_to(root):
            raise ValueError('invalid base manifest path')
        checks.append((path, expected, relative))
    total = sum(path.stat().st_size for path, _, _ in checks)
    completed = 0
    emit(f'Weight hashes: {len(checks)} files, {total / 1024**3:.2f} GiB to read')
    for index, (path, expected, label) in enumerate(checks, 1):
        def on_bytes(read, size, elapsed):
            rate = read / max(elapsed, 0.001) / 1024**2
            emit(f'Hash {index}/{len(checks)} {label}: {read / 1024**3:.2f}/{size / 1024**3:.2f} GiB; '
                 f'total {(completed + read) / 1024**3:.2f}/{total / 1024**3:.2f} GiB; {rate:.1f} MiB/s')
        if sha256(path, progress=on_bytes, cancelled=cancelled) != expected:
            raise ValueError(f'weight hash mismatch: {label}')
        completed += path.stat().st_size
    emit('All weight hashes verified')
    for component in COMPONENTS:
        if not (settings.model_dir / component).is_dir():
            raise ValueError(f'missing component: {component}')
