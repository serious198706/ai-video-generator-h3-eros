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
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'invalid.mp4'
            with self.assertRaisesRegex(ValueError, 'video frames'):
                export_video(np.zeros((56, 64, 64, 3)), torch.zeros(2, round(56 / 24 * 32000)),
                             32000, path, 64, 64, 56, False)
            self.assertFalse(path.exists())

    def test_invalid_late_frame_removes_partial_container(self):
        frames = np.full((56, 64, 64, 3), 0.4, dtype=np.float32)
        frames[-1, 0, 0, 0] = np.nan
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'invalid.mp4'
            with self.assertRaisesRegex(ValueError, 'video frames'):
                export_video(frames, torch.zeros(2, round(56 / 24 * 32000)),
                             32000, path, 64, 64, 56, False)
            self.assertFalse(path.exists())

    def test_framewise_float_and_uint8_conversion_preserve_rgb_and_frame_count(self):
        import av
        pixels = np.zeros((56, 64, 64, 3), dtype=np.uint8)
        pixels[:] = [192, 96, 32]
        with tempfile.TemporaryDirectory() as temp:
            for frames in (pixels, pixels.astype(np.float32) / 255):
                path = Path(temp) / 'colors.mp4'
                timing = export_video(frames, torch.zeros(2, round(56 / 24 * 32000)),
                                      32000, path, 64, 64, 56, False)
                self.assertIn('container_validation_s', timing)
                with av.open(str(path)) as container:
                    decoded = list(container.decode(video=0))
                self.assertEqual(len(decoded), 56)
                np.testing.assert_allclose(decoded[0].to_ndarray(format='rgb24').mean(axis=(0, 1)),
                                           [192, 96, 32], atol=4)
