from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .geometry import standard_target
from .video import RawVideoWriter, mux_audio, probe

LOG = logging.getLogger(__name__)


class UpscaleEngine:
    def __init__(self, settings):
        self.settings = settings
        self.descriptor = None
        self.device = None
        self.dtype = None

    def load(self) -> None:
        import spandrel
        import spandrel_extra_arches
        import torch

        if self.descriptor is not None:
            return
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required")
        spandrel_extra_arches.install(ignore_duplicates=True)
        started = time.monotonic()
        descriptor = spandrel.ModelLoader().load_from_file(str(self.settings.model))
        if not isinstance(descriptor, spandrel.ImageModelDescriptor):
            raise TypeError(f"not an image SR model: {type(descriptor).__name__}")
        self.device = torch.device("cuda")
        descriptor = descriptor.to(self.device).eval()
        if descriptor.supports_half:
            descriptor = descriptor.half()
        self.descriptor = descriptor
        parameter = next(descriptor.model.parameters())
        self.dtype = parameter.dtype
        LOG.info(
            "upscale model ready model=%s device=%s dtype=%s load_s=%.3f",
            self.settings.model.name,
            torch.cuda.get_device_name(0),
            self.dtype,
            time.monotonic() - started,
        )

    def run(
        self,
        source: Path,
        output: Path,
        *,
        target_width: int | None = None,
        target_height: int | None = None,
        preserve_audio: bool = True,
        progress: Callable[[str], None] | None = None,
    ) -> dict:
        import torch

        self.load()
        info = probe(source)
        if target_width is None or target_height is None:
            target_width, target_height = standard_target(info.width, info.height)
        if target_width % 2 or target_height % 2:
            raise ValueError("target dimensions must be even")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        output.parent.mkdir(parents=True, exist_ok=True)
        video_only = output.with_suffix(".video-only.mp4")
        video_only.unlink(missing_ok=True)
        output.unlink(missing_ok=True)
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise FileNotFoundError(f"cannot open input video: {source}")
        writer = RawVideoWriter(
            video_only, target_width, target_height, info.fps,
            crf=self.settings.crf, preset=self.settings.preset,
        )
        model_s = 0.0
        frames = 0
        started = time.monotonic()
        try:
            with torch.inference_mode():
                while True:
                    ok, bgr = capture.read()
                    if not ok:
                        break
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    frame_started = time.monotonic()
                    result = self._frame(rgb, target_width, target_height)
                    model_s += time.monotonic() - frame_started
                    writer.write(result)
                    frames += 1
                    if progress and (frames == 1 or frames % 30 == 0 or frames == info.frames):
                        progress(f"frame {frames}/{info.frames}")
            writer.close()
            if frames != info.frames:
                raise RuntimeError(f"decoded frame count mismatch: expected={info.frames} actual={frames}")
            mux_started = time.monotonic()
            mux_audio(video_only, source, output, preserve_audio)
            mux_s = time.monotonic() - mux_started
        except BaseException:
            writer.abort()
            output.unlink(missing_ok=True)
            raise
        finally:
            capture.release()
            video_only.unlink(missing_ok=True)
        result_info = probe(output)
        if result_info.width != target_width or result_info.height != target_height:
            output.unlink(missing_ok=True)
            raise RuntimeError("encoded output dimensions do not match target")
        if result_info.frames != frames:
            output.unlink(missing_ok=True)
            raise RuntimeError("encoded output frame count mismatch")
        if preserve_audio and info.audio and not result_info.audio:
            output.unlink(missing_ok=True)
            raise RuntimeError("source audio was lost")
        wall_s = time.monotonic() - started
        return {
            "source_width": info.width,
            "source_height": info.height,
            "width": target_width,
            "height": target_height,
            "frames": frames,
            "fps": info.fps,
            "duration": result_info.duration,
            "audio_present": result_info.audio,
            "model_s": model_s,
            "mux_s": mux_s,
            "wall_s": wall_s,
            "frames_per_second": frames / wall_s,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
            "resize": self.settings.resize,
            "tile": self.settings.tile,
        }

    def _frame(self, rgb: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        if self.settings.tile:
            return self._frame_tiled(rgb, target_width, target_height)
        return self._frame_full(rgb, target_width, target_height)

    def _tensor(self, rgb: np.ndarray):
        import torch

        return (
            torch.from_numpy(np.ascontiguousarray(rgb))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=self.device, dtype=self.dtype)
            .div_(255)
        )

    @staticmethod
    def _crop_tensor(tensor, target_width: int, target_height: int):
        height, width = tensor.shape[-2:]
        target_ratio = target_width / target_height
        source_ratio = width / height
        if source_ratio > target_ratio:
            crop_width = max(1, round(height * target_ratio))
            left = (width - crop_width) // 2
            return tensor[..., :, left:left + crop_width]
        crop_height = max(1, round(width / target_ratio))
        top = (height - crop_height) // 2
        return tensor[..., top:top + crop_height, :]

    @staticmethod
    def _finish_cpu(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        height, width = image.shape[:2]
        target_ratio = target_width / target_height
        if width / height > target_ratio:
            crop_width = max(1, round(height * target_ratio))
            left = (width - crop_width) // 2
            image = image[:, left:left + crop_width]
        else:
            crop_height = max(1, round(width / target_ratio))
            top = (height - crop_height) // 2
            image = image[top:top + crop_height]
        if image.shape[:2] != (target_height, target_width):
            image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        return np.ascontiguousarray(image)

    def _frame_full(self, rgb: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        import torch
        import torch.nn.functional as functional

        output = self.descriptor(self._tensor(rgb)).float().clamp_(0, 1)
        if self.settings.resize == "gpu_bicubic":
            output = self._crop_tensor(output, target_width, target_height)
            output = functional.interpolate(
                output, size=(target_height, target_width), mode="bicubic",
                align_corners=False, antialias=True,
            ).clamp_(0, 1)
            return (
                output.mul_(255).round_().to(torch.uint8)
                .squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()
            )
        image = (
            output.mul_(255).round_().to(torch.uint8)
            .squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()
        )
        return self._finish_cpu(image, target_width, target_height)

    def _frame_tiled(self, rgb: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        import torch

        height, width = rgb.shape[:2]
        tile, pad = self.settings.tile, self.settings.tile_pad
        canvas = None
        scale = None
        for top in range(0, height, tile):
            for left in range(0, width, tile):
                bottom, right = min(top + tile, height), min(left + tile, width)
                padded_top, padded_left = max(0, top - pad), max(0, left - pad)
                padded_bottom, padded_right = min(height, bottom + pad), min(width, right + pad)
                patch = rgb[padded_top:padded_bottom, padded_left:padded_right]
                output = self.descriptor(self._tensor(patch)).float().clamp_(0, 1)
                patch_scale = output.shape[-1] // patch.shape[1]
                if patch_scale < 1 or output.shape[-2] != patch.shape[0] * patch_scale:
                    raise RuntimeError("unexpected model scale")
                if canvas is None:
                    scale = patch_scale
                    canvas = np.empty((height * scale, width * scale, 3), dtype=np.uint8)
                if patch_scale != scale:
                    raise RuntimeError("model scale changed between tiles")
                y0, x0 = (top - padded_top) * scale, (left - padded_left) * scale
                y1, x1 = y0 + (bottom - top) * scale, x0 + (right - left) * scale
                core = (
                    output[..., y0:y1, x0:x1].mul_(255).round_().to(torch.uint8)
                    .squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()
                )
                canvas[top * scale:bottom * scale, left * scale:right * scale] = core
        return self._finish_cpu(canvas, target_width, target_height)


