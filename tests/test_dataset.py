"""Tests for FMA dataset loading, splits, and spectrogram shapes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf


# ── Synthetic dataset fixture ─────────────────────────────────────────────────

def _make_mini_dataset(tmp_path: Path, n_train: int = 4, n_val: int = 2, n_test: int = 2):
    """Create a minimal synthetic FMA-like dataset in tmp_path.

    Writes:
    - fma_metadata/tracks.csv  (two-row MultiIndex header, FMA format)
    - cache/fma_small/<id>.npy  (synthetic spectrograms)
    - cache/fma_small/stats.json

    Returns:
        cfg:     OmegaConf dataset config
        records: list of dicts with track metadata
    """
    genres = ["Electronic", "Folk", "Pop", "Rock"]

    records = []
    track_id = 1
    for split, count in [("training", n_train), ("validation", n_val), ("test", n_test)]:
        for i in range(count):
            records.append({
                "id": track_id,
                "subset": "small",
                "split": split,
                "genre_top": genres[i % len(genres)],
                "artist_id": track_id * 100,
            })
            track_id += 1

    # ── Write tracks.csv with two-row MultiIndex header ───────────────────────
    metadata_dir = tmp_path / "fma_metadata"
    metadata_dir.mkdir()

    lines = [
        ",set,set,track,artist\n",
        ",subset,split,genre_top,id\n",
    ]
    for r in records:
        lines.append(f"{r['id']},{r['subset']},{r['split']},{r['genre_top']},{r['artist_id']}\n")
    (metadata_dir / "tracks.csv").write_text("".join(lines))

    # ── Write cached spectrograms ─────────────────────────────────────────────
    cache_dir = tmp_path / "cache" / "fma_small"
    cache_dir.mkdir(parents=True)

    rng = np.random.default_rng(0)
    for r in records:
        spec = rng.standard_normal((128, 3001)).astype(np.float32)
        np.save(cache_dir / f"{r['id']}.npy", spec)

    # ── Write pre-computed stats ──────────────────────────────────────────────
    (cache_dir / "stats.json").write_text(json.dumps({"mean": -30.0, "std": 20.0}))

    cfg = OmegaConf.create({
        "name": "fma_small",
        "root_dir": str(tmp_path / "fma_small"),
        "metadata_dir": str(metadata_dir),
        "cache_dir": str(cache_dir),
        "corrupt_log": str(cache_dir / "corrupt_files.log"),
        "num_classes": 4,
        "genres": genres,
        "audio": {
            "sample_rate": 16000,
            "train_crop_duration": 10.0,
            "eval_window_duration": 10.0,
            "eval_window_hop": 5.0,
        },
        "mel": {
            "n_mels": 128,
            "n_fft": 400,
            "hop_length": 160,
        },
        "stats": {"mean": None, "std": None},
    })

    return cfg, records


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_genre_map_stable():
    """Genre → index mapping must be deterministic (alphabetical order)."""
    from data.dataset import FMADataset

    genres = ["Rock", "Pop", "Folk", "Electronic"]
    gmap = FMADataset.build_genre_map(genres)

    assert gmap["Electronic"] == 0
    assert gmap["Folk"]       == 1
    assert gmap["Pop"]        == 2
    assert gmap["Rock"]       == 3

    # Second call with same input must return identical mapping
    assert gmap == FMADataset.build_genre_map(genres)


def test_train_sample_shape(tmp_path):
    """Train-mode __getitem__ must return tensor (1, 128, crop_T) and a valid label."""
    from data.dataset import FMADataset

    cfg, _ = _make_mini_dataset(tmp_path)
    ds = FMADataset(cfg, split="training", mode="train")

    assert len(ds) == 4

    spec, label = ds[0]

    assert spec.ndim == 3,              "Expected 3-D tensor (C, F, T)"
    assert spec.shape[0] == 1,          "Expected 1 channel"
    assert spec.shape[1] == 128,        "Expected 128 mel bins"
    assert spec.shape[2] > 0,           "Expected non-zero time frames"
    assert 0 <= label < cfg.num_classes


def test_eval_mode_returns_windows(tmp_path):
    """Eval mode must return a list of ≥1 windows, each shape (1, 128, window_T)."""
    from data.dataset import FMADataset

    cfg, _ = _make_mini_dataset(tmp_path)
    ds = FMADataset(cfg, split="validation", mode="eval")

    windows, label = ds[0]

    assert isinstance(windows, list),       "Eval mode must return a list"
    assert len(windows) >= 1,               "Expected ≥1 window per clip"
    assert windows[0].shape[0] == 1,        "Expected 1 channel"
    assert windows[0].shape[1] == 128,      "Expected 128 mel bins"
    assert 0 <= label < cfg.num_classes

    # All windows must have the same shape
    shapes = [w.shape for w in windows]
    assert len(set(shapes)) == 1, f"Inconsistent window shapes: {shapes}"


def test_corrupt_files_excluded(tmp_path):
    """Tracks listed in corrupt_log must not appear in the dataset."""
    from data.dataset import FMADataset

    cfg, records = _make_mini_dataset(tmp_path)
    corrupt_id = records[0]["id"]

    # Write corrupt log before constructing the dataset
    Path(cfg.corrupt_log).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.corrupt_log).write_text(f"{corrupt_id}\n")

    ds = FMADataset(cfg, split="training", mode="train")

    assert corrupt_id not in ds.track_ids, (
        f"Corrupt track {corrupt_id} should have been excluded"
    )
    assert len(ds) == 3   # 4 training tracks minus 1 corrupt


def test_splits_are_disjoint(tmp_path):
    """Track IDs must be strictly disjoint across train / val / test splits."""
    from data.dataset import FMADataset

    cfg, _ = _make_mini_dataset(tmp_path)

    train_ids = set(FMADataset(cfg, "training",   "train").track_ids)
    val_ids   = set(FMADataset(cfg, "validation",  "train").track_ids)
    test_ids  = set(FMADataset(cfg, "test",        "train").track_ids)

    assert len(train_ids) > 0
    assert len(val_ids)   > 0
    assert len(test_ids)  > 0

    assert train_ids.isdisjoint(val_ids),  "Train and val share track IDs"
    assert train_ids.isdisjoint(test_ids), "Train and test share track IDs"
    assert val_ids.isdisjoint(test_ids),   "Val and test share track IDs"
