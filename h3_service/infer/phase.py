"""One GPU phase per process. Exit releases CUDA ownership before the next phase."""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path


def validate_wire(embeds, tags):
    import torch
    if (embeds.dtype != torch.bfloat16 or embeds.ndim != 3 or embeds.shape[0] != 1
            or embeds.shape[2] != 5120 or not 0 < embeds.shape[1] <= 32768
            or tags.dtype != torch.int64 or tags.shape != (embeds.shape[1],)
            or not torch.isfinite(embeds).all() or not ((tags == 0) | (tags == 1)).all()):
        raise ValueError('invalid conditioner output')


def execute(mode, job, settings, progress=None, loaded_pipe=None):
    emit = progress or (lambda message: None)
    emit('importing runtime dependencies')
    if mode == 'probe-runtime':
        from h3_service.check_runtime import check_runtime
        emit('checking driver, CUDA BF16 and cuDNN kernels')
        return check_runtime(settings.attention)
    import torch
    from safetensors.torch import save_file
    from safetensors import safe_open
    from h3_service.infer.blocks import load_pipeline
    from h3_service.infer.eros import apply_eros
    from h3_service.media.images import read_image
    from h3_service.media.video import export_video

    emit('checking CUDA availability and free memory')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; no placeholder fallback')
    torch.cuda.set_device(0)
    free, total = torch.cuda.mem_get_info()
    if free / 1024**3 < settings.min_free_gib:
        raise RuntimeError('insufficient free GPU memory')
    torch.cuda.reset_peak_memory_stats()
    conditioner = mode in ('condition', 'probe-condition')
    started = time.monotonic()
    pipe = loaded_pipe
    if pipe is None:
        pipe = load_pipeline(settings.model_dir, conditioner=conditioner, progress=emit)
    if not conditioner and loaded_pipe is None:
        emit('converting Eros weights on CPU')
        report = apply_eros(pipe.transformer, settings.eros_file, progress=emit)
        if job:
            Path(job['directory'], 'conversion.json').write_text(json.dumps(report))
        pipe.transformer.set_attention_backend(settings.attention)
    emit('moving loaded model to GPU')
    pipe.to('cuda')
    torch.cuda.synchronize()
    loaded = time.monotonic() - started
    if mode.startswith('probe-'):
        return {'load_s': loaded, 'total_gib': total / 1024**3}
    directory = Path(job['directory'])
    width, height, num_frames = job['width'], job['height'], job['num_frames']
    common = {'image': read_image(job['first']),
              'last_image': read_image(job['last']) if job['last'] else None,
              'height': height, 'width': width}
    started = time.monotonic()
    with torch.inference_mode():
        if conditioner:
            emit('encoding prompt and keyframe images')
            state = pipe(prompt=job['prompt'], **common)
            embeds, tags = state.get('prompt_embeds').cpu().contiguous(), state.get('text_token_tags').cpu().contiguous()
            validate_wire(embeds, tags)
            if (int(state.get('width')), int(state.get('height'))) != (width, height):
                raise ValueError('conditioner canvas changed')
            save_file({'prompt_embeds': embeds, 'text_token_tags': tags}, str(directory / 'conditioning.safetensors'),
                      {'width': str(width), 'height': str(height), 'num_frames': str(num_frames), 'schema': '1'})
        else:
            emit('reading conditioning tensors')
            with safe_open(str(directory / 'conditioning.safetensors'), framework='pt', device='cpu') as source:
                metadata = source.metadata()
                if metadata != {'width': str(width), 'height': str(height), 'num_frames': str(num_frames), 'schema': '1'}:
                    raise ValueError('conditioning plan mismatch')
                embeds, tags = source.get_tensor('prompt_embeds'), source.get_tensor('text_token_tags')
            validate_wire(embeds, tags)
            emit('running denoising and video/audio decoding')
            state = pipe(prompt_embeds=embeds.to('cuda'), text_token_tags=tags,
                         num_frames=num_frames, num_inference_steps=job['steps'], output_type='np',
                         generator=torch.Generator('cpu').manual_seed(job['seed']), **common)
            emit('validating frames/audio and encoding MP4')
            export_video(state.get('videos')[0], state.get('audio')[0].cpu(), state.get('sampling_rate'),
                         directory / 'output.mp4', width, height, num_frames, job['audio'])
    torch.cuda.synchronize()
    return {'load_s': loaded if loaded_pipe is None else 0.0, 'transfer_s': loaded if loaded_pipe is not None else 0.0, 'compute_export_s': time.monotonic() - started,
            'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
            'peak_reserved_gib': torch.cuda.max_memory_reserved() / 1024**3}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['condition', 'generate', 'probe-runtime', 'probe-condition', 'probe-generate'])
    parser.add_argument('--job', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--progress-file', type=Path)
    args = parser.parse_args()
    from h3_service.config import Settings
    def emit(stage):
        if args.progress_file:
            temporary = args.progress_file.with_suffix('.tmp')
            temporary.write_text(json.dumps({'stage': stage}))
            temporary.replace(args.progress_file)
    try:
        result = execute(args.mode, json.loads(args.job.read_text()) if args.job else None, Settings(), progress=emit)
        args.report.write_text(json.dumps(result, allow_nan=False))
        return 0
    except Exception as exc:
        # Never echo prompts, signed URLs or full exception messages into service logs.
        frames = [{'file': Path(f.filename).name, 'line': f.lineno, 'function': f.name}
                  for f in traceback.extract_tb(exc.__traceback__)]
        args.report.write_text(json.dumps({'error': type(exc).__name__, 'trace': frames}))
        print(f'H3 {args.mode} failed: {type(exc).__name__}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
