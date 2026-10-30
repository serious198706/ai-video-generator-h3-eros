"""Scoped GPU block wall timings. Nested totals are not additive."""
import time
from contextlib import contextmanager


@contextmanager
def generation_blocks(progress, timings):
    import torch
    from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import MiniMaxH3CoreDenoiseStep
    from diffusers.modular_pipelines.minimax_h3.decoders import MiniMaxH3VideoDecodeStep, MiniMaxH3AudioDecodeStep
    originals = []

    def wrap(original, label):
        def measured(*args, **kwargs):
            progress(f'{label}: started')
            torch.cuda.synchronize()
            started = time.monotonic()
            outcome = 'failed'
            try:
                result = original(*args, **kwargs)
                torch.cuda.synchronize()
                outcome = 'succeeded'
                return result
            finally:
                elapsed = time.monotonic() - started
                timings[label + '_s'] = timings.get(label + '_s', 0.0) + elapsed
                progress(f'{label}: {outcome}, elapsed_s={elapsed:.3f}')
        return measured

    try:
        for cls, label in [(MiniMaxH3CoreDenoiseStep, 'denoise'),
                           (MiniMaxH3VideoDecodeStep, 'video_decode'), (MiniMaxH3AudioDecodeStep, 'audio_decode')]:
            inherited = '__call__' not in cls.__dict__
            original = cls.__call__
            originals.append((cls, original, inherited))
            cls.__call__ = wrap(original, label)
        yield
    finally:
        for cls, original, inherited in reversed(originals):
            if inherited:
                delattr(cls, '__call__')
            else:
                cls.__call__ = original
