"""Segment selection and log-mel computation for ingested YouTube audio."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def compute_segment(
    duration: float | None,
    offset_ratio: float = 0.35,
    segment_seconds: float = 30.0,
    min_duration: float | None = None,
) -> tuple[float, float] | None:
    """Return (start, end) in seconds, or None if the song is too short.

    start = duration * offset_ratio. If start + segment_seconds overflows the
    song, the window is shifted back so it still lasts segment_seconds.
    """
    if min_duration is None:
        min_duration = segment_seconds
    if duration is None or duration < max(min_duration, segment_seconds):
        return None
    start = duration * offset_ratio
    end = start + segment_seconds
    if end > duration:
        end = float(duration)
        start = end - segment_seconds
    return start, end


def get_ffmpeg() -> str:
    """ffmpeg on PATH, else the binary bundled with imageio-ffmpeg."""
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
    except ImportError as exc:
        raise RuntimeError(
            "ffmpeg not found. Install it or run: pip install imageio-ffmpeg"
        ) from exc
    # yt-dlp only recognises a binary literally named "ffmpeg[.exe]".
    link = Path(tempfile.gettempdir()) / "mgc_ffmpeg" / f"ffmpeg{exe.suffix}"
    if not link.exists() or link.stat().st_size != exe.stat().st_size:
        link.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(exe, link)
    # yt-dlp's partial-download check looks only at PATH, not at ffmpeg_location.
    os.environ["PATH"] = str(link.parent) + os.pathsep + os.environ.get("PATH", "")
    return str(link)


def decode_audio(path: Path, sample_rate: int, ffmpeg: str | None = None,
                 start: float | None = None, duration: float | None = None) -> np.ndarray:
    """Decode any audio file to mono float32 at ``sample_rate`` via ffmpeg.

    ``start``/``duration`` (s) crop the decoded range (used for full-length downloads).
    """
    cmd = [ffmpeg or get_ffmpeg(), "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-i", str(path), "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode(errors='replace').strip()}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def audio_length(path: Path, ffmpeg: str | None = None) -> float:
    """Duration (s) of an audio file, measured by decoding it (no ffprobe needed)."""
    return len(decode_audio(path, 8000, ffmpeg)) / 8000


def audio_to_mel(
    path: Path,
    dataset_cfg,
    segment_seconds: float,
    tolerance: float = 0.5,
    ffmpeg: str | None = None,
    start: float | None = None,
) -> np.ndarray:
    """Decode a downloaded segment and return its (n_mels, T) log-mel.

    Uses the same mel code as the FMA cache, so shape and scale match.
    The waveform is cropped/padded to exactly ``segment_seconds``; a deficit
    larger than ``tolerance`` seconds is an error.
    """
    import torch
    from data.preprocess import waveform_to_melspec

    sr = dataset_cfg.audio.sample_rate
    wav = decode_audio(path, sr, ffmpeg, start,
                       segment_seconds if start is not None else None)
    target = int(round(segment_seconds * sr))
    if len(wav) < target - int(tolerance * sr):
        raise ValueError(f"segment too short: {len(wav) / sr:.1f}s < {segment_seconds}s")
    wav = wav[:target]
    if len(wav) < target:
        wav = np.pad(wav, (0, target - len(wav)))

    mel = waveform_to_melspec(torch.from_numpy(wav).unsqueeze(0), sr, dataset_cfg)
    expected_t = target // dataset_cfg.mel.hop_length + 1
    if mel.shape != (dataset_cfg.mel.n_mels, expected_t):
        raise ValueError(f"unexpected mel shape {mel.shape}, expected "
                         f"({dataset_cfg.mel.n_mels}, {expected_t})")
    return mel
