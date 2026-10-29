import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from h3_service.media.video import export_video, validate_container


@unittest.skipUnless(shutil.which('ffprobe'), 'ffprobe is required for real container tests')
class VideoTests(unittest.TestCase):
    def test_real_audio_and_silent_mp4(self):
        frames = np.full((56, 64, 64, 3), 0.4, dtype=np.float32)
        samples = round(56 / 24 * 32000)
        wave = torch.sin(torch.arange(samples) * 2 * torch.pi * 220 / 32000) * 0.1
        audio = torch.stack([wave, wave])
        with tempfile.TemporaryDirectory() as temp:
            for enabled in (True, False):
                path = Path(temp) / f'{enabled}.mp4'
                export_video(frames, audio, 32000, path, 64, 64, 56, enabled)
                self.assertGreater(path.stat().st_size, 0)
                with self.assertRaises(ValueError):
                    validate_container(path, 64, 64, 56, not enabled)

    def test_black_video_rejected(self):
        with self.assertRaisesRegex(ValueError, 'video frames'):
            export_video(np.zeros((56, 64, 64, 3)), torch.zeros(2, 100), 32000, Path('unused'), 64, 64, 56, True)
