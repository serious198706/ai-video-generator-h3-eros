"""Reference-tree -> Diffusers conversion; see third_party/SOURCES.md."""
import torch
from safetensors import safe_open

SKIP = {'rope.inv_freq', 'adaln_basis', 'adaln_mean', 'adaln_t_table'}
KEEP = {f'time_embedder.linear_{i}.{leaf}' for i in (1, 2) for leaf in ('weight', 'bias')}
DIRECT = {'video_patch_proj': 'proj_in', 'audio_patch_proj': 'audio_proj_in',
          'condition_proj': 'context_embedder', 'final_layer.video_out': 'proj_out',
          'final_layer.audio_out': 'audio_proj_out', 'final_layer.norm': 'norm_out.norm'}


def adaln_target(name):
    if name.startswith('final_layer.adaln_proj.linear.'):
        return 'norm_out.linear'
    if name.startswith('blocks.') and '.adaln_proj.linear.' in name:
        return name.replace('blocks.', 'transformer_blocks.', 1).rsplit('.', 1)[0]
    return None


def rebuild_adaln(weight, bias, basis, mean):
    expanded = weight.float() @ basis.float()
    return expanded, bias.float() - expanded @ mean.float()


def map_tensor(name, tensor, inner_dim):
    stem, _, leaf = name.rpartition('.')
    if stem in DIRECT:
        return [(DIRECT[stem] + '.' + leaf, tensor)]
    target = name.replace('token_refiner.blocks.', 'token_refiner.refiner_blocks.', 1)
    if name.startswith('blocks.'):
        target = name.replace('blocks.', 'transformer_blocks.', 1)
    if target.endswith('.attn.qkv_proj.weight'):
        if tensor.shape[0] != inner_dim * 3:
            raise ValueError('invalid QKV shape')
        prefix = target.removesuffix('qkv_proj.weight')
        return [(prefix + 'to_' + k + '.weight', part) for k, part in zip('qkv', tensor.split(inner_dim, dim=0))]
    if target.endswith('.mlp.fc1.weight'):
        if tensor.shape[0] % 2:
            raise ValueError('invalid gate/value shape')
        gate, value = tensor.chunk(2, dim=0)
        return [(target.replace('.mlp.fc1.weight', '.ff.net.0.proj.weight'), torch.cat([value, gate]))]
    for old, new in [('.mlp.fc2.weight', '.ff.net.2.weight'), ('.attn.out_proj.weight', '.attn.to_out.0.weight'),
                     ('.attn.q_norm.weight', '.attn.norm_q.weight'), ('.attn.k_norm.weight', '.attn.norm_k.weight')]:
        if target.endswith(old):
            target = target.removesuffix(old) + new
            break
    return [(target, tensor)]


@torch.no_grad()
def apply_eros(transformer, path, progress=None):
    params = dict(transformer.named_parameters())
    written, skipped = set(), set()
    inner = transformer.config.num_attention_heads * transformer.config.attention_head_dim

    def put(key, tensor):
        if key in written or key in KEEP or key not in params:
            raise ValueError(f'unexpected or duplicate target: {key}')
        param = params[key]
        if param.shape != tensor.shape or not torch.isfinite(tensor).all():
            raise ValueError(f'invalid tensor: {key}')
        converted = tensor.to(param.dtype)
        if not torch.isfinite(converted).all():
            raise ValueError(f'conversion overflow: {key}')
        param.copy_(converted)
        written.add(key)

    with safe_open(str(path), framework='pt', device='cpu') as source:
        keys = set(source.keys())
        if not SKIP <= keys:
            raise ValueError('missing reference auxiliaries')
        basis, mean = source.get_tensor('adaln_basis'), source.get_tensor('adaln_mean')
        for index, name in enumerate(sorted(keys), 1):
            if progress and (index == 1 or index % 16 == 0):
                progress(f'Eros conversion: source tensor {index}/{len(keys)}, {len(written)} target parameters copied')
            if name in SKIP:
                skipped.add(name)
                continue
            prefix = adaln_target(name)
            if prefix:
                if name.endswith('.bias'):
                    if name.removesuffix('.bias') + '.weight' not in keys:
                        raise ValueError('orphan AdaLN bias')
                    continue
                if not name.endswith('.weight'):
                    raise ValueError('unknown AdaLN tensor')
                weight, bias = rebuild_adaln(source.get_tensor(name), source.get_tensor(name.removesuffix('.weight') + '.bias'), basis, mean)
                put(prefix + '.weight', weight)
                put(prefix + '.bias', bias)
            else:
                for key, tensor in map_tensor(name, source.get_tensor(name), inner):
                    put(key, tensor)
    remaining = set(params) - written
    if remaining != KEEP:
        raise ValueError(f'unexpected untouched parameters: {sorted(remaining ^ KEEP)}')
    if progress:
        progress(f'Eros conversion complete: {len(written)} replaced, {len(remaining)} retained')
    return {'replaced': len(written), 'written': sorted(written), 'retained': sorted(remaining), 'skipped': sorted(skipped)}
