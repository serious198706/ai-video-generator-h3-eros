"""Immutable local Diffusers checkpoint of the converted Eros transformer."""
import fcntl
import json
import os
import shutil
import tempfile
from pathlib import Path
from h3_service.infer.manifest import BASE_REVISION, DIFFUSERS_REVISION, EROS_SHA256, sha256

VERSION = {'base': BASE_REVISION, 'diffusers': DIFFUSERS_REVISION, 'eros': EROS_SHA256, 'converter': 1}


def stat_signature(path):
    s = path.stat()
    return [s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_ino, s.st_dev]


def check_cache(directory, full=False):
    manifest = json.loads((directory / 'h3-conversion.json').read_text())
    if manifest['version'] != VERSION or not manifest['files']:
        raise ValueError('converted cache version mismatch')
    for name, record in manifest['files'].items():
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError('invalid converted cache path')
        if full or stat_signature(path) != record['stat']:
            if sha256(path) != record['sha256']:
                raise ValueError('converted cache hash mismatch')
    return manifest['conversion']


def load_generator(settings, emit, loader, converter):
    if not settings.converted_cache:
        pipe = loader(settings.model_dir, conditioner=False, progress=emit)
        return pipe, converter(pipe.transformer, settings.eros_file, progress=emit)
    root = settings.transformer_cache
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = root / ('eros-' + EROS_SHA256[:12] + '-v1')
    with (root / '.conversion.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # The exclusive lock proves no other converter is using these staging directories.
        for stale in root.glob('.conversion-*'):
            if stale.is_dir():
                shutil.rmtree(stale)
        if target.exists():
            try:
                conversion = check_cache(target, full=settings.verify_mode == 'full')
            except (OSError, ValueError, KeyError):
                emit('Converted checkpoint invalid or incompatible; rebuilding')
            else:
                emit('Loading verified converted Eros checkpoint; CPU conversion skipped')
                return loader(settings.model_dir, conditioner=False, progress=emit, transformer_path=target), conversion
        pipe = loader(settings.model_dir, conditioner=False, progress=emit)
        conversion = converter(pipe.transformer, settings.eros_file, progress=emit)
        emit('Saving converted Eros checkpoint once for later restarts (requires ~62 GiB disk)')
        staging = Path(tempfile.mkdtemp(prefix='.conversion-', dir=root))
        try:
            pipe.transformer.save_pretrained(staging, safe_serialization=True, max_shard_size='5GB')
            files = {}
            for path in sorted(staging.iterdir()):
                if path.is_file():
                    files[path.name] = {'sha256': sha256(path), 'stat': stat_signature(path)}
            if 'config.json' not in files or not any(n.endswith('.safetensors') for n in files):
                raise ValueError('converted checkpoint incomplete')
            (staging / 'h3-conversion.json').write_text(json.dumps({'version': VERSION, 'files': files, 'conversion': conversion}))
            if target.exists():
                shutil.rmtree(target)
            os.replace(staging, target)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return pipe, conversion
