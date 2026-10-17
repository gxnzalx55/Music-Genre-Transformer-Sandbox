"""Tests for merging ingested custom songs into the dataset."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from tests.test_dataset import _make_mini_dataset


def _write_custom(root: Path, songs: list[tuple[str, str]], outliers: tuple[str, ...] = ()):
    """songs: (track_id, genre). Writes manifest.csv, mels/ and optionally review.csv."""
    (root / "mels").mkdir(parents=True)
    with open(root / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["track_id", "source", "title", "channel", "duration", "start", "end",
                    "path", "genre", "playlist"])
        for tid, genre in songs:
            np.save(root / "mels" / f"{tid}.npy", np.zeros((128, 3001), dtype=np.float32))
            w.writerow([tid, "spotify", "t", "a", 100, 35, 65, f"mels/{tid}.npy", genre, "u"])
    if outliers:
        with open(root / "review.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["track_id", "genre", "status"])
            for tid, _ in songs:
                w.writerow([tid, "x", "outlier" if tid in outliers else "ok"])


def _cfg(tmp_path, songs, outliers=(), min_tracks=2):
    cfg, _ = _make_mini_dataset(tmp_path)
    _write_custom(tmp_path / "custom", songs, outliers)
    cfg.custom = {"enabled": True, "dir": str(tmp_path / "custom"),
                  "split": {"training": 0.8, "validation": 0.1, "test": 0.1},
                  "exclude_outliers": True, "min_tracks": min_tracks}
    return cfg


def test_assign_split_is_stable_and_roughly_proportional():
    from data.custom import assign_split

    frac = {"training": 0.8, "validation": 0.1, "test": 0.1}
    splits = [assign_split(f"id{i}", frac) for i in range(2000)]
    assert splits == [assign_split(f"id{i}", frac) for i in range(2000)]
    for name, expected in frac.items():
        assert abs(splits.count(name) / 2000 - expected) < 0.03


def test_expand_adds_genres_and_num_classes(tmp_path):
    from data.custom import expand_dataset_cfg

    songs = [(f"r{i}", "Reggaeton") for i in range(3)] + [(f"p{i}", "Pop") for i in range(3)]
    cfg = expand_dataset_cfg(_cfg(tmp_path, songs))
    assert sorted(cfg.genres) == ["Electronic", "Folk", "Pop", "Reggaeton", "Rock"]
    assert cfg.num_classes == 5            # "Pop" merged with the existing class


def test_small_genres_and_outliers_are_excluded(tmp_path):
    from data.custom import load_custom_items

    songs = [(f"r{i}", "Reggaeton") for i in range(4)] + [("j0", "Jazz")]
    cfg = _cfg(tmp_path, songs, outliers=("r0",), min_tracks=2)
    items = load_custom_items(cfg.custom)
    assert {it["genre"] for it in items} == {"Reggaeton"}      # Jazz: 1 song < min_tracks
    assert "r0" not in {it["track_id"] for it in items}


def test_dataset_merges_custom_songs_per_split(tmp_path):
    from data.custom import expand_dataset_cfg, load_custom_items
    from data.dataset import FMADataset

    songs = [(f"r{i}", "Reggaeton") for i in range(40)]
    cfg = expand_dataset_cfg(_cfg(tmp_path, songs, min_tracks=2))
    expected = {s: {it["track_id"] for it in load_custom_items(cfg.custom) if it["split"] == s}
                for s in ("training", "validation", "test")}
    seen = []
    for split in ("training", "validation", "test"):
        ds = FMADataset(cfg, split, mode="train")
        custom_ids = {t for t in ds.track_ids if isinstance(t, str)}
        assert custom_ids == expected[split]
        assert ds.n_custom == len(expected[split])
        seen.append(custom_ids)
        for i in range(len(ds)):
            x, label = ds[i]
            assert 0 <= label < cfg.num_classes
        reggaeton = ds.genre_to_idx["Reggaeton"]
        assert sum(lbl == reggaeton for lbl in ds._labels) == len(expected[split])
    assert not (seen[0] & seen[1]) and not (seen[0] & seen[2]) and not (seen[1] & seen[2])


def test_custom_disabled_leaves_dataset_unchanged(tmp_path):
    from data.dataset import FMADataset

    cfg = _cfg(tmp_path, [(f"r{i}", "Reggaeton") for i in range(5)])
    cfg.custom.enabled = False
    ds = FMADataset(cfg, "training", mode="train")
    assert ds.n_custom == 0 and all(isinstance(t, int) for t in ds.track_ids)
