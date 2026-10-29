"""Read public pinned metadata and compare real Eros header shapes to a meta-device H3.

No model data or GPU is loaded. This is not a numeric or video-quality test.
"""
import argparse
import json
import struct
import time
import urllib.request
from pathlib import Path
import torch
from h3_service.infer.manifest import BASE_REPO, BASE_REVISION, EROS_REPO, EROS_REVISION, EROS_NAME, DIFFUSERS_REVISION
from h3_service.infer.eros import map_tensor, adaln_target, SKIP, KEEP


def fetch(url, *, header=False):
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={'Range': 'bytes=0-1048575'} if header else {})
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read(1048576)
            if header:
                size = struct.unpack('<Q', data[:8])[0]
                if size > len(data) - 8:
                    raise ValueError('header exceeds bounded fetch')
                return json.loads(data[8:8 + size])
            return json.loads(data)
        except OSError:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from diffusers import MiniMaxH3Transformer3DModel
    from h3_service.infer.blocks import ConditionerBlocks, GeneratorBlocks
    config = fetch(f'https://huggingface.co/{BASE_REPO}/resolve/{BASE_REVISION}/transformer/config.json')
    header = fetch(f'https://huggingface.co/{EROS_REPO}/resolve/{EROS_REVISION}/{EROS_NAME}', header=True)
    with torch.device('meta'):
        model = MiniMaxH3Transformer3DModel.from_config(config)
    params = dict(model.named_parameters())
    inner = model.config.num_attention_heads * model.config.attention_head_dim
    written = set()
    for key, spec in header.items():
        if key == '__metadata__' or key in SKIP:
            continue
        prefix = adaln_target(key)
        if prefix:
            if key.endswith('.bias'):
                continue
            pairs = [(prefix + '.weight', (spec['shape'][0], header['adaln_basis']['shape'][1])),
                     (prefix + '.bias', (spec['shape'][0],))]
        else:
            pairs = [(k, tuple(v.shape)) for k, v in map_tensor(key, torch.empty(spec['shape'], device='meta'), inner)]
        for target, shape in pairs:
            if target in written or target not in params or tuple(params[target].shape) != shape:
                raise ValueError(f'header mapping mismatch: {target}')
            written.add(target)
    if set(params) - written != KEEP:
        raise ValueError('unexpected retained parameters')
    report = {'kind': 'header_shape_check_only', 'base_revision': BASE_REVISION,
              'eros_revision': EROS_REVISION, 'diffusers_expected_revision': DIFFUSERS_REVISION,
              'source_tensors': len(header) - int('__metadata__' in header),
              'mapped': len(written), 'retained': sorted(KEEP), 'target_total': len(params),
              'conditioner_components': [c.name for c in ConditionerBlocks().expected_components],
              'generator_components': [c.name for c in GeneratorBlocks().expected_components],
              'status': 'passed', 'full_weights_loaded': False, 'gpu_generation_tested': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
