"""Compute and persist per-channel mean/std from the training split only."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

log = logging.getLogger(__name__)

_STATS_FILE = "stats.json"


def compute_stats(cache_dir: Path, train_ids: list[int]) -> tuple[float, float]:
    """Return (mean, std) over all spectrogram frames of the training split.

    Uses Chan's parallel algorithm (online, per-file) for numerical stability
    without loading all files into memory at once.
    """
    cache_dir = Path(cache_dir)
    n: int = 0
    mean: float = 0.0
    M2: float = 0.0

    for track_id in tqdm(train_ids, desc="Computing stats", unit="track"):
        path = cache_dir / f"{track_id}.npy"
        if not path.exists():
            continue
        spec = np.load(path).astype(np.float64)
        batch = spec.ravel()
        b_n = len(batch)
        b_mean = float(batch.mean())
        b_var = float(batch.var())   # population variance

        # Chan's formula for combining two sets of statistics
        delta = b_mean - mean
        new_n = n + b_n
        mean = (n * mean + b_n * b_mean) / new_n
        M2 += b_var * b_n + delta ** 2 * n * b_n / new_n
        n = new_n

    if n < 2:
        raise RuntimeError(
            f"Not enough training data to compute stats (found {n} frames). "
            "Run preprocess_dataset first."
        )

    std = float(np.sqrt(M2 / n))
    log.info("Stats from %d frames: mean=%.4f  std=%.4f", n, mean, std)
    return float(mean), std


def save_stats(cfg, mean: float, std: float) -> None:
    """Write mean/std to {cache_dir}/stats.json."""
    stats_path = Path(cfg.cache_dir) / _STATS_FILE
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as fh:
        json.dump({"mean": mean, "std": std}, fh)
    log.info("Stats saved → %s", stats_path)


def load_stats(cfg) -> tuple[float, float]:
    """Load pre-computed stats from {cache_dir}/stats.json.

    Raises RuntimeError if the file does not exist (prepare_data not yet run).
    """
    stats_path = Path(cfg.cache_dir) / _STATS_FILE
    if not stats_path.exists():
        # Allow fallback to values already set in the config (e.g. in tests)
        m, s = cfg.stats.mean, cfg.stats.std
        if m is not None and s is not None:
            return float(m), float(s)
        raise RuntimeError(
            f"Stats file not found: {stats_path}\n"
            "Run `python scripts/prepare_data.py` to compute them."
        )
    with open(stats_path) as fh:
        data = json.load(fh)
    return float(data["mean"]), float(data["std"])
