from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int
    duration: float
    audio: bool


def ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError("ffmpeg not found") from exc


def ffprobe() -> str:
    found = shutil.which("ffprobe")
    if not found:
        raise RuntimeError("ffprobe not found; install the ffmpeg package")
    return found


def probe(path: str | Path) -> VideoInfo:
    command = [
        ffprobe(), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    videos = [item for item in streams if item.get("codec_type") == "video"]
    if len(videos) != 1:
        raise ValueError("input must contain exactly one video stream")
    video = videos[0]
    rate = video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1"
    numerator, denominator = (float(part) for part in rate.split("/", 1))
    fps = numerator / denominator if denominator else 0
    frames = int(video.get("nb_frames") or 0)
    duration = float(video.get("duration") or payload.get("format", {}).get("duration") or 0)
    if frames < 1 and fps > 0 and duration > 0:
        frames = round(fps * duration)
    info = VideoInfo(
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=fps,
        frames=frames,
        duration=duration,
        audio=any(item.get("codec_type") == "audio" for item in streams),
    )
    if info.width < 2 or info.height < 2 or info.fps <= 0 or info.frames < 1:
        raise ValueError(f"invalid input video metadata: {info}")
    return info


class RawVideoWriter:
    def __init__(self, path: Path, width: int, height: int, fps: float, *, crf: int, preset: str):
        self.path = path
        self.process = subprocess.Popen(
            [
                ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                "-r", f"{fps:.8f}", "-i", "-", "-an", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-preset", preset, "-crf", str(crf),
                "-movflags", "+faststart", str(path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame) -> None:
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin unavailable")
        self.process.stdin.write(frame.tobytes())

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        error = self.process.stderr.read().decode(errors="replace") if self.process.stderr else ""
        code = self.process.wait()
        if code != 0 or not self.path.is_file() or self.path.stat().st_size < 64:
            raise RuntimeError(f"ffmpeg encode failed code={code}: {error[-2000:]}")

    def abort(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()
        self.path.unlink(missing_ok=True)


def mux_audio(video_only: Path, source: Path, output: Path, preserve_audio: bool) -> None:
    output.unlink(missing_ok=True)
    if not preserve_audio:
        video_only.replace(output)
        return
    result = subprocess.run(
        [
            ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(video_only), "-i", str(source),
            "-map", "0:v:0", "-map", "1:a?", "-c", "copy",
            "-map_metadata", "1", "-movflags", "+faststart", str(output),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not output.is_file() or output.stat().st_size < 64:
        output.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg audio mux failed code={result.returncode}: {result.stderr[-2000:]}")


