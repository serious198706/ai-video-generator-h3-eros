import json
import subprocess
import numpy as np


def export_video(frames, audio, sample_rate, path, width, height, num_frames, include_audio):
    from diffusers.utils import encode_video
    frames = np.asarray(frames)
    if frames.shape != (num_frames, height, width, 3) or not np.isfinite(frames).all() or not frames.any():
        raise ValueError('invalid video frames')
    waveform = audio.numpy() if hasattr(audio, 'numpy') else np.asarray(audio)
    if waveform.ndim != 2 or waveform.shape[0] != 2 or not np.isfinite(waveform).all():
        raise ValueError('invalid audio')
    if sample_rate != 32000 or abs(waveform.shape[-1] / sample_rate - num_frames / 24) > 0.15:
        raise ValueError('audio duration or rate mismatch')
    metrics = {'audio_peak': float(np.max(np.abs(waveform))), 'audio_rms': float(np.sqrt(np.mean(waveform**2))),
               'audio_clip_fraction': float(np.mean(np.abs(waveform) >= 1))}
    if include_audio and metrics['audio_rms'] < 1e-7:
        raise ValueError('unexpected silent soundtrack')
    encode_video(frames, fps=24, output_path=str(path),
                 audio=audio if include_audio else None, audio_sample_rate=sample_rate if include_audio else None)
    validate_container(path, width, height, num_frames, include_audio)
    path.with_suffix('.metrics.json').write_text(json.dumps(metrics))


def validate_container(path, width, height, num_frames, include_audio):
    from fractions import Fraction
    proc = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-show_streams', '-of', 'json', str(path)],
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
