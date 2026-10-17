"""PyTorch Dataset for FMA: random crop (train) or sliding windows (eval)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.custom import custom_enabled, load_custom_items
from data.stats import load_stats

log = logging.getLogger(__name__)


def _load_tracks(metadata_dir: Path, dataset_name: str) -> pd.DataFrame:
    """Load tracks.csv and filter to the requested FMA subset."""
    tracks = pd.read_csv(
        metadata_dir / "tracks.csv",
        index_col=0,
        header=[0, 1],
    )
    subset_key = dataset_name.replace("fma_", "")
    return tracks[tracks[("set", "subset")] == subset_key]


def _load_corrupt_ids(log_path: Path) -> set[int]:
    if not log_path.exists():
        return set()
    ids: set[int] = set()
    with open(log_path) as fh:
        for line in fh:
            line = line.strip()
            if line.isdigit():
                ids.add(int(line))
    return ids


def sliding_windows(spec: np.ndarray, window_frames: int, hop_frames: int) -> list[np.ndarray]:
    """Cut ``spec`` (n_mels, T) into ``window_frames``-wide windows with ``hop_frames`` stride.

    Short clips are reflect-padded to one window; a final partial window is
    aligned to the end of the clip.
    """
    T, n = spec.shape[1], window_frames
    if T < n:
        return [np.pad(spec, ((0, 0), (0, n - T)), mode="reflect")]

    windows: list[np.ndarray] = []
    start = 0
    while start + n <= T:
        windows.append(spec[:, start : start + n])
        start += hop_frames

    # Final partial window, only if frames remain beyond the last window's end.
    if start - hop_frames + n < T:
        windows.append(spec[:, T - n : T])
    return windows


class FMADataset(Dataset):
    """FMA spectrogram dataset.

    Args:
        cfg:       Dataset config node (OmegaConf).
        split:     ``"training"``, ``"validation"``, or ``"test"``.
        mode:      ``"train"`` → single random crop per sample.
                   ``"eval"``  → list of sliding-window crops per sample.
        transform: Optional callable applied to each ``(1, n_mels, T)`` tensor.
    """

    def __init__(
        self,
        cfg,
        split: Literal["training", "validation", "test"],
        mode: Literal["train", "eval"] = "train",
        transform: Callable | None = None,
    ) -> None:
        self.cfg = cfg
        self.split = split
        self.mode = mode
        self.transform = transform

        metadata_dir = Path(cfg.metadata_dir)
        cache_dir = Path(cfg.cache_dir)

        # Load and filter tracks
        tracks = _load_tracks(metadata_dir, cfg.name)
        tracks = tracks[tracks[("set", "split")] == split]

        # Remove corrupt files
        corrupt_ids = _load_corrupt_ids(Path(cfg.corrupt_log))
        tracks = tracks[~tracks.index.isin(corrupt_ids)]

        custom_items = (
            [it for it in load_custom_items(cfg.custom) if it["split"] == split]
            if custom_enabled(cfg) else []
        )

        # Build genre → index map
        if cfg.genres is not None:
            genres = [str(g) for g in cfg.genres]
        else:
            genres = sorted(
                set(tracks[("track", "genre_top")].dropna().unique().tolist())
                | {it["genre"] for it in custom_items}
            )
        self.genre_to_idx: dict[str, int] = self.build_genre_map(genres)

        # Keep only tracks with a known genre AND an existing cache file
        has_genre = tracks[("track", "genre_top")].isin(self.genre_to_idx)
        has_cache = tracks.index.map(
            lambda tid: (cache_dir / f"{tid}.npy").exists()
        )
        tracks = tracks[has_genre & has_cache]

        # One flat list of samples: FMA tracks (int ids) then custom songs (str ids).
        self.track_ids: list = tracks.index.tolist()
        self._paths: list[Path] = [cache_dir / f"{tid}.npy" for tid in self.track_ids]
        self._labels: list[int] = [
            self.genre_to_idx[str(g)] for g in tracks[("track", "genre_top")]
        ]
        for it in custom_items:
            if it["genre"] in self.genre_to_idx:
                self.track_ids.append(it["track_id"])
                self._paths.append(it["path"])
                self._labels.append(self.genre_to_idx[it["genre"]])
        self.n_custom = len(self.track_ids) - len(tracks)

        # Frame counts derived from config
        sr  = int(cfg.audio.sample_rate)
        hop = int(cfg.mel.hop_length)
        self.crop_frames   = int(cfg.audio.train_crop_duration * sr / hop) + 1
        self.window_frames = int(cfg.audio.eval_window_duration * sr / hop) + 1
        self.hop_frames    = int(cfg.audio.eval_window_hop * sr / hop)

        # Normalisation statistics
        self.mean, self.std = load_stats(cfg)

        log.info(
            "FMADataset [%s | %s]: %d tracks (%d custom), %d classes",
            split, mode, len(self.track_ids), self.n_custom, len(self.genre_to_idx),
        )

    # ── Dataset protocol ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.track_ids)

    def __getitem__(self, idx: int):
        """Return ``(tensor, label)`` in train mode or ``([tensors], label)`` in eval mode."""
        label = self._labels[idx]
        spec = np.load(self._paths[idx])   # (n_mels, T)

        if self.mode == "train":
            crop   = self._random_crop(spec)
            tensor = self._to_tensor(crop)
            if self.transform is not None:
                tensor = self.transform(tensor)
            return tensor, label

        # eval: list of window tensors
        windows = [self._to_tensor(w) for w in self._sliding_windows(spec)]
        return windows, label

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _to_tensor(self, spec: np.ndarray) -> torch.Tensor:
        """Normalise and add channel dim: (n_mels, T) → (1, n_mels, T)."""
        t = torch.from_numpy(spec.copy()).float()
        t = (t - self.mean) / (self.std + 1e-6)
        return t.unsqueeze(0)

    def _random_crop(self, spec: np.ndarray) -> np.ndarray:
        """Return a random ``crop_frames``-wide slice of spec."""
        T = spec.shape[1]
        n = self.crop_frames
        if T <= n:
            return np.pad(spec, ((0, 0), (0, n - T)), mode="reflect")
        start = np.random.randint(0, T - n + 1)
        return spec[:, start : start + n]

    def _sliding_windows(self, spec: np.ndarray) -> list[np.ndarray]:
        """Return all ``window_frames``-wide windows with ``hop_frames`` stride."""
        return sliding_windows(spec, self.window_frames, self.hop_frames)

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def build_genre_map(genres: list[str]) -> dict[str, int]:
        """Map genre names to integer indices in sorted (alphabetical) order."""
        return {g: i for i, g in enumerate(sorted(genres))}
