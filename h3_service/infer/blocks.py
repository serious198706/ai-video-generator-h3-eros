"""Local fl2va split, following H3EROS at the revision in SOURCES.md."""
from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3TextEncoderStep
from diffusers.modular_pipelines.minimax_h3.decoders import MiniMaxH3AfterDenoiseStep
from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import (
    MiniMaxH3AutoResizeStep, MiniMaxH3AutoKeyframeVaeEncoderStep,
    MiniMaxH3CoreDenoiseStep, MiniMaxH3DecodeStep, _generation_outputs,
)
from diffusers.modular_pipelines.modular_pipeline import SequentialPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import OutputParam
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MiniMaxH3ModularPipeline


class ConditionerBlocks(SequentialPipelineBlocks):
    model_name = 'minimax-h3'
    block_classes = [MiniMaxH3AutoResizeStep, MiniMaxH3TextEncoderStep]
    block_names = ['resize', 'text_encoder']

    @property
    def description(self):
        return 'Local keyframe and text conditioning.'

    @property
    def outputs(self):
        return [OutputParam.template('prompt_embeds'), OutputParam('text_token_tags'),
                OutputParam('height', type_hint=int), OutputParam('width', type_hint=int)]


class GeneratorBlocks(SequentialPipelineBlocks):
    model_name = 'minimax-h3'
    block_classes = [MiniMaxH3AutoResizeStep, MiniMaxH3AutoKeyframeVaeEncoderStep,
                     MiniMaxH3CoreDenoiseStep, MiniMaxH3AfterDenoiseStep, MiniMaxH3DecodeStep]
    block_names = ['resize', 'vae_encoder', 'denoise', 'after_denoise', 'decode']

    @property
    def description(self):
        return 'Local generation using separately encoded conditioning.'

    @property
    def outputs(self):
        return _generation_outputs()


def load_pipeline(model_dir, *, conditioner, progress=None):
    import torch
    emit = progress or (lambda message: None)
    # Same explicit extension as upstream Space; short clips remain GPU-unvalidated.
    MiniMaxH3ModularPipeline.min_duration = property(lambda self: 2.0)
    blocks = ConditionerBlocks() if conditioner else GeneratorBlocks()
    emit('initializing local pipeline configuration')
    pipe = blocks.init_pipeline(str(model_dir))
    for component in blocks.expected_components:
        if component.default_creation_method == 'from_pretrained':
            emit(f'loading component from disk: {component.name}')
            pipe.load_components(names=[component.name], dtype=torch.bfloat16, local_files_only=True)
    # Upstream load_components logs and continues on individual load errors.
    missing = [c.name for c in blocks.expected_components if getattr(pipe, c.name, None) is None]
    if missing:
        raise RuntimeError('components failed to load: ' + ', '.join(missing))
    if not conditioner:
        for name in ('vae', 'audio_vae'):
            if any(p.is_floating_point() and p.dtype != torch.float32 for p in getattr(pipe, name).parameters()):
                raise RuntimeError(f'{name} must retain FP32 weights')
    return pipe
