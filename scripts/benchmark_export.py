"""Synthetic export-only check: no weights, CUDA, queue or AWS access."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from h3_service.media.video import export_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True, help='new output directory')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    width, height, count = 768, 1344, 124
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    frame = np.empty((height, width, 3), dtype=np.float32)
    frame[..., 0], frame[..., 1], frame[..., 2] = x, y, (x + y) / 2
    # Non-contiguous, read-only view also exercises frame-wise conversion.
    frames = np.broadcast_to(frame, (count, height, width, 3))
    audio = np.zeros((2, round(count / 24 * 32000)), dtype=np.float32)
    started = time.monotonic()
    timing = export_video(frames, audio, 32000, args.output / 'synthetic.mp4',
                          width, height, count, False, progress=lambda message: print(message, flush=True))
    report = {'synthetic': True, 'width': width, 'height': height, 'num_frames': count,
              'wall_s': time.monotonic() - started, 'timing': timing}
    path = args.output / 'report.json'
    path.write_text(json.dumps(report, indent=2))
    path.chmod(0o600)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
