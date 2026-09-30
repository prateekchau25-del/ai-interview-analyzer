"""FFmpeg helpers: split an uploaded recording into analysis-ready audio and video."""
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class MediaError(RuntimeError):
    pass


def _ffmpeg_bin(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise MediaError(f"{name} not found on PATH. Install FFmpeg (winget install Gyan.FFmpeg).")
    return path


@dataclass
class PreparedMedia:
    audio_path: Path | None  # 16 kHz mono WAV
    video_path: Path | None  # original file if it has a video stream
    duration_sec: float
    has_video: bool
    has_audio: bool


def probe(path: Path) -> dict:
    out = subprocess.run(
        [_ffmpeg_bin("ffprobe"), "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=False,
    )
    if out.returncode != 0:
        raise MediaError(f"ffprobe failed: {out.stderr.strip()[:300]}")
    return json.loads(out.stdout)


def extract_audio(src: Path, dst: Path, sample_rate: int = 16000) -> Path:
    cmd = [_ffmpeg_bin("ffmpeg"), "-y", "-v", "error", "-i", str(src),
           "-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(dst)]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise MediaError(f"audio extraction failed: {res.stderr.strip()[:300]}")
    return dst


def prepare(src: Path, work_dir: Path) -> PreparedMedia:
    info = probe(src)
    streams = info.get("streams", [])
    has_video = any(s.get("codec_type") == "video" for s in streams)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    duration = float(info.get("format", {}).get("duration") or 0.0)

    work_dir.mkdir(parents=True, exist_ok=True)
    audio_path = extract_audio(src, work_dir / "audio.wav") if has_audio else None

    if duration <= 0 and audio_path:
        # WebM from browsers often lacks a duration header; read it from the decoded WAV.
        import soundfile as sf

        duration = sf.info(str(audio_path)).duration

    return PreparedMedia(
        audio_path=audio_path,
        video_path=src if has_video else None,
        duration_sec=round(duration, 2),
        has_video=has_video,
        has_audio=has_audio,
    )
