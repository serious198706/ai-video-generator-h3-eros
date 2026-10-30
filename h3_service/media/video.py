import json
import subprocess
import time
from fractions import Fraction
import numpy as np


def export_video(frames, audio, sample_rate, path, width, height, num_frames, include_audio,
                 progress=None):
    import av
    from h3_service.infer.cpu_budget import thread_budget
    emit = progress or (lambda message: None)
    frames = np.asarray(frames)
    if frames.shape != (num_frames, height, width, 3) or frames.dtype.kind not in 'fu':
        raise ValueError('invalid video frames')
    if frames.dtype.kind == 'u' and frames.dtype != np.uint8:
        raise ValueError('invalid video pixel dtype')
    waveform = audio.numpy() if hasattr(audio, 'numpy') else np.asarray(audio)
    if waveform.ndim != 2 or waveform.shape[0] != 2 or not np.isfinite(waveform).all():
        raise ValueError('invalid audio')
    if sample_rate != 32000 or abs(waveform.shape[-1] / sample_rate - num_frames / 24) > 0.15:
        raise ValueError('audio duration or rate mismatch')
    metrics = {'audio_peak': float(np.max(np.abs(waveform))), 'audio_rms': float(np.sqrt(np.mean(waveform**2))),
               'audio_clip_fraction': float(np.mean(np.abs(waveform) >= 1))}
    if include_audio and metrics['audio_rms'] < 1e-7:
        raise ValueError('unexpected silent soundtrack')
    timing = {'frame_prepare_s': 0.0, 'video_encode_s': 0.0}
    threads = thread_budget()
    emit(f'MP4 encode started frames={num_frames} threads={threads} codec=libx264 preset=medium crf=23')
    started = time.monotonic()
    nonzero = False
    # Only frame-sized conversion buffers: no full-video zeros/ones/masks or uint8 copies.
    scratch = np.empty((height, width, 3), dtype=np.float32) if frames.dtype.kind == 'f' else None
    try:
        with av.open(str(path), mode='w') as container:
            video_stream = container.add_stream('libx264', rate=24)
            video_stream.width, video_stream.height = width, height
            video_stream.pix_fmt = 'yuv420p'
            video_stream.codec_context.thread_count = threads
            video_stream.options = {'preset': 'medium', 'crf': '23'}
            audio_stream = None
            if include_audio:
                audio_stream = container.add_stream('aac', rate=sample_rate)
                audio_stream.layout = 'stereo'
                audio_stream.bit_rate = 192000
                audio_stream.codec_context.thread_count = 1
            for index, source in enumerate(frames):
                stage_started = time.monotonic()
                if not np.isfinite(source).all():
                    raise ValueError('invalid video frames')
                nonzero = nonzero or bool(source.any())
                if scratch is not None:
                    if source.min() < 0 or source.max() > 1:
                        raise ValueError('invalid video pixel range')
                    np.multiply(source, 255, out=scratch)
                    np.rint(scratch, out=scratch)
                    pixels = scratch.astype(np.uint8)
                else:
                    pixels = np.ascontiguousarray(source)
                frame = av.VideoFrame.from_ndarray(pixels, format='rgb24')
                frame.pts, frame.time_base = index, Fraction(1, 24)
                timing['frame_prepare_s'] += time.monotonic() - stage_started
                stage_started = time.monotonic()
                for packet in video_stream.encode(frame):
                    container.mux(packet)
                timing['video_encode_s'] += time.monotonic() - stage_started
                if (index + 1) % 24 == 0 or index + 1 == num_frames:
                    emit(f'MP4 encode frames={index + 1}/{num_frames} elapsed_s={time.monotonic() - started:.3f}')
            if not nonzero:
                raise ValueError('invalid video frames')
            stage_started = time.monotonic()
            for packet in video_stream.encode():
                container.mux(packet)
            timing['video_encode_s'] += time.monotonic() - stage_started
            stage_started = time.monotonic()
            if audio_stream is not None:
                samples = np.ascontiguousarray(np.clip(waveform, -1, 1), dtype=np.float32)
                for offset in range(0, samples.shape[1], 1024):
                    frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(samples[:, offset:offset + 1024]),
                                                      format='fltp', layout='stereo')
                    frame.sample_rate, frame.pts, frame.time_base = sample_rate, offset, Fraction(1, sample_rate)
                    for packet in audio_stream.encode(frame):
                        container.mux(packet)
                for packet in audio_stream.encode():
                    container.mux(packet)
            timing['audio_encode_s'] = time.monotonic() - stage_started
        timing['encode_mux_s'] = time.monotonic() - started
        emit(f'MP4 container validation started; encode timing={json.dumps(timing)}')
        started = time.monotonic()
        validate_container(path, width, height, num_frames, include_audio)
        timing['container_validation_s'] = time.monotonic() - started
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    path.with_suffix('.metrics.json').write_text(json.dumps({**metrics, 'export_timing': timing}))
    emit(f'MP4 export complete timing={json.dumps(timing)}')
    return timing


def validate_container(path, width, height, num_frames, include_audio):
    from h3_service.infer.cpu_budget import thread_budget
    proc = subprocess.run(['ffprobe', '-v', 'error', '-threads', str(thread_budget()),
                           '-count_frames', '-show_streams', '-of', 'json', str(path)],
                          capture_output=True, check=True, timeout=120)
    streams = json.loads(proc.stdout)['streams']
    videos = [s for s in streams if s['codec_type'] == 'video']
    audios = [s for s in streams if s['codec_type'] == 'audio']
    if len(videos) != 1 or bool(audios) != include_audio:
        raise ValueError('unexpected output tracks')
    video = videos[0]
    if (video['width'], video['height'], int(video['nb_read_frames'])) != (width, height, num_frames):
        raise ValueError('encoded video plan mismatch')
    if Fraction(video['avg_frame_rate']) != 24:
        raise ValueError('encoded FPS mismatch')
    if audios and (len(audios) != 1 or abs(float(audios[0]['duration']) - num_frames / 24) > 0.15):
        raise ValueError('encoded audio duration mismatch')
