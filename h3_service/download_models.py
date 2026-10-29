"""Explicit download command; serving never downloads or follows moving revisions."""
import json
import sys
from pathlib import Path
from h3_service.config import Settings
from h3_service.infer.manifest import (
    BASE_REPO, BASE_REVISION, EROS_REPO, EROS_REVISION, EROS_NAME, EROS_SHA256, COMPONENTS, sha256, verify,
)


def main():
    from huggingface_hub import snapshot_download, hf_hub_download
    settings = Settings()
    if (settings.model_dir / 'h3-manifest.json').exists():
        try:
            verify(settings, progress=print, cache_path=settings.work_dir / 'weight-verification.json')
            print('Pinned models already present and verified; no download needed.')
            return 0
        except (OSError, ValueError):
            print('Existing model verification failed; repairing the pinned snapshot.')
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(BASE_REPO, revision=BASE_REVISION, local_dir=settings.model_dir,
                      allow_patterns=['modular_model_index.json', 'LICENSE', *[c + '/*' for c in COMPONENTS]])
    downloaded = hf_hub_download(EROS_REPO, EROS_NAME, revision=EROS_REVISION, local_dir=settings.eros_file.parent)
    if Path(downloaded).resolve() != settings.eros_file.resolve():
        raise ValueError('H3_EROS_FILE must retain the pinned filename')
    if sha256(downloaded) != EROS_SHA256:
        raise ValueError('downloaded Eros hash mismatch')
    # Component specs embed Hub paths. Resolve these to this pinned local snapshot explicitly.
    index_path = settings.model_dir / 'modular_model_index.json'
    index = json.loads(index_path.read_text())
    for name in COMPONENTS:
        spec = index[name]
        spec[2]['pretrained_model_name_or_path'] = str(settings.model_dir.resolve())
        spec[2]['revision'] = None
    index.pop('transformer_ref', None)
    index_path.write_text(json.dumps(index, indent=2))
    print('Hashing local base components for the immutable manifest.', flush=True)
    hashes = {str(p.relative_to(settings.model_dir)): sha256(p)
              for p in sorted(settings.model_dir.rglob('*'))
              if p.is_file() and '.cache' not in p.parts and p.name != 'h3-manifest.json'}
    (settings.model_dir / 'h3-manifest.json').write_text(json.dumps({'revision': BASE_REVISION, 'files': hashes}, indent=2))
    print('Pinned model files downloaded and hashed. GPU validation has not run.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
