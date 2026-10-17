"""Extract log-mel spectrograms from MP3 clips and cache them as .npy files."""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

log = logging.getLogger(__name__)


def is_corrupt(audio_path: Path) -> bool:
    """Return True if the file cannot be decoded by torchaudio or librosa."""
    try:
        import torchaudio
        torchaudio.load(str(audio_path))
        return False
    except Exception:
        pass
    try:
        import librosa
        librosa.load(str(audio_path), sr=None, mono=True, duration=1.0)
        return False
    except Exception:
        return True


def waveform_to_melspec(waveform, orig_sr: int, cfg) -> np.ndarray:
    """Mix to mono, resample to cfg.audio.sample_rate and return the log-mel.

    Args:
        waveform: torch.Tensor of shape (channels, samples).
    Returns:
        np.ndarray of shape (n_mels, T), dtype float32.
    """
    import torch
    import torchaudio.transforms as T

    sr = cfg.audio.sample_rate

    # Mix to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample
    if int(orig_sr) != sr:
        waveform = T.Resample(orig_freq=int(orig_sr), new_freq=sr)(waveform)

    mel_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=cfg.mel.n_fft,
        hop_length=cfg.mel.hop_length,
        n_mels=cfg.mel.n_mels,
        f_min=float(cfg.mel.f_min),
        f_max=float(cfg.mel.f_max),
        power=float(cfg.mel.power),
    )
    db_transform = T.AmplitudeToDB(stype="power", top_db=float(cfg.mel.top_db))

    with torch.no_grad():
        mel = mel_transform(waveform)   # (1, n_mels, T)
        mel = db_transform(mel)

    return mel.squeeze(0).numpy().astype(np.float32)  # (n_mels, T)


def compute_melspec(audio_path: Path, cfg) -> np.ndarray:
    """Load an MP3, resample to mono @ cfg.audio.sample_rate, return log-mel spectrogram.

    Returns:
        np.ndarray of shape (n_mels, T), dtype float32.
    """
    import torch
    import torchaudio

    try:
        waveform, orig_sr = torchaudio.load(str(audio_path))
    except Exception:
        import librosa
        y, orig_sr = librosa.load(str(audio_path), sr=None, mono=True)
        waveform = torch.from_numpy(y).unsqueeze(0)
        orig_sr = int(orig_sr)

    return waveform_to_melspec(waveform, int(orig_sr), cfg)


def _process_track(
    track_id: int,
    audio_path: Path,
    cache_path: Path,
    cfg,
) -> tuple[int, bool, str]:
    """Worker unit: process one track. Returns (track_id, success, error_msg)."""
    try:
        if cache_path.exists():
            return track_id, True, ""
        if is_corrupt(audio_path):
            return track_id, False, "corrupt"
        spec = compute_melspec(audio_path, cfg)
        np.save(cache_path, spec)
        return track_id, True, ""
    except Exception as exc:
        return track_id, False, str(exc)


def preprocess_dataset(cfg) -> None:
    """Compute spectrograms for every valid track and write to cache_dir/<id>.npy.

    Corrupt files are logged to cfg.corrupt_log and skipped.
    Uses ThreadPoolExecutor for parallel I/O + computation.
    """
    # Fail loudly: is_corrupt() swallows exceptions, so a missing dependency
    # would otherwise mark every track as corrupt.
    try:
        import librosa  # noqa: F401
        import torchaudio  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            f"Missing audio dependency ({exc}). Run: pip install -r requirements.txt"
        ) from exc

    metadata_dir = Path(cfg.metadata_dir)
    root_dir = Path(cfg.root_dir)
    cache_dir = Path(cfg.cache_dir)
    corrupt_log = Path(cfg.corrupt_log)

    cache_dir.mkdir(parents=True, exist_ok=True)
    corrupt_log.parent.mkdir(parents=True, exist_ok=True)

    tracks = pd.read_csv(metadata_dir / "tracks.csv", index_col=0, header=[0, 1])
    subset_key = cfg.name.replace("fma_", "")
    tracks = tracks[tracks[("set", "subset")] == subset_key]

    work: list[tuple[int, Path, Path]] = []
    for track_id in tracks.index:
        tid_str = f"{track_id:06d}"
        audio_path = root_dir / tid_str[:3] / f"{tid_str}.mp3"
        if audio_path.exists():
            work.append((track_id, audio_path, cache_dir / f"{track_id}.npy"))

    log.info("Processing %d tracks → %s", len(work), cache_dir)

    n_workers = min(os.cpu_count() or 4, 8)
    corrupt_ids: list[int] = []

    with (
        ThreadPoolExecutor(max_workers=n_workers) as pool,
        tqdm(total=len(work), desc="Preprocessing", unit="track") as pbar,
    ):
        futures = {
            pool.submit(_process_track, tid, ap, cp, cfg): tid
            for tid, ap, cp in work
        }
        for fut in as_completed(futures):
            track_id, ok, msg = fut.result()
            if not ok:
                corrupt_ids.append(track_id)
                log.debug("Skipped %d: %s", track_id, msg)
            pbar.update(1)

    with open(corrupt_log, "w") as fh:
        for tid in sorted(corrupt_ids):
            fh.write(f"{tid}\n")

    log.info(
        "Done. %d ok, %d corrupt/missing (see %s)",
        len(work) - len(corrupt_ids),
        len(corrupt_ids),
        corrupt_log,
    )
