"""Experimental TorchAO INT8 weight-only conversion for dual GPU residency."""
import time


GENERATOR_BF16_PREFIXES = (
    'transformer_blocks.0.', 'transformer_blocks.1.',
    'transformer_blocks.48.', 'transformer_blocks.49.',
)


def int8wo_config():
    """TorchAO 0.17 v2 expresses group size through granularity, not group_size."""
    from torchao.quantization import Int8WeightOnlyConfig, PerGroup
    return Int8WeightOnlyConfig(granularity=PerGroup(128), version=2)


def int8wo_filter(component, module, fqn):
    """Keep output/tied and H3 boundary layers in BF16; quantize other Linear weights."""
    import torch
    if not isinstance(module, torch.nn.Linear):
        return False
    # PerGroup(128) requires the input dimension to be exactly divisible. Qwen3-VL
    # contains a few projection shapes which are not; retaining those in BF16 is
    # safer than padding weights or weakening every other layer to per-row INT8.
    if module.weight.ndim != 2 or module.weight.shape[-1] % 128:
        return False
    if component == 'conditioner':
        return not (fqn == 'lm_head' or fqn.startswith('lm_head.'))
    if component == 'generator':
        return not fqn.startswith(GENERATOR_BF16_PREFIXES)
    raise ValueError('unknown quantization component')


def quantize_int8wo(module, component, emit):
    """Quantize in CPU memory before either large model is placed on the GPU."""
    import torch
    from torchao.quantization import quantize_
    incompatible = [fqn for fqn, child in module.named_modules()
                    if isinstance(child, torch.nn.Linear) and child.weight.shape[-1] % 128]
    targets = [fqn for fqn, child in module.named_modules()
               if int8wo_filter(component, child, fqn)]
    if not targets:
        raise RuntimeError(f'no INT8WO targets found for {component}')
    emit(f'INT8WO quantization start component={component} linear_targets={len(targets)} '
         f'bf16_incompatible={len(incompatible)}')
    started = time.monotonic()
    quantize_(module, int8wo_config(),
              filter_fn=lambda child, fqn: int8wo_filter(component, child, fqn))
    elapsed = time.monotonic() - started
    # TorchAO weights retain their logical BF16 dtype; verify their tensor subclass changed.
    unchanged = [fqn for fqn, child in module.named_modules()
                 if fqn in targets and type(child.weight) is torch.nn.Parameter]
    if unchanged:
        raise RuntimeError(f'INT8WO left ordinary weights in {component}: {unchanged[:3]}')
    report = {'scheme': 'int8wo-group128-v2', 'component': component,
              'linear_quantized': len(targets), 'elapsed_s': elapsed,
              'bf16_incompatible': incompatible,
              'bf16_boundary_blocks': list(GENERATOR_BF16_PREFIXES) if component == 'generator' else [],
              'lm_head_bf16': component == 'conditioner'}
    emit(f'INT8WO quantization complete component={component} linear_quantized={len(targets)} '
         f'bf16_incompatible={len(incompatible)} '
         f'elapsed_s={elapsed:.1f}')
    return report
