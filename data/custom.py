"""Custom songs ingested with scripts/ingest_playlist.py, merged into the FMA dataset."""
from __future__ import annotations

import csv
import hashlib
import logging
from pathlib import Path

log = logging.getLogger(__name__)

SPLITS = ("training", "validation", "test")


def custom_enabled(cfg) -> bool:
    custom = cfg.get("custom")
    return bool(custom is not None and custom.get("enabled", False))


def assign_split(track_id: str, fractions) -> str:
    """Stable split from a hash of the id: a song never changes split when more are added."""
    u = int(hashlib.sha1(track_id.encode("utf-8")).hexdigest()[:8], 16) / 0x100000000
    test = float(fractions["test"])
    val = float(fractions["validation"])
    if u < test:
        return "test"
    if u < test + val:
        return "validation"
    return "training"


def load_custom_items(custom_cfg) -> list[dict]:
    """Usable custom songs: labelled, file present, not flagged as outlier, genre big enough.

    Each item is ``{"track_id", "path", "genre", "split"}``.
    """
    root = Path(custom_cfg.dir)
    manifest = root / "manifest.csv"
    if not manifest.exists():
        log.warning("custom dataset enabled but %s does not exist", manifest)
        return []

    excluded: set[str] = set()
    review = root / "review.csv"
    if custom_cfg.get("exclude_outliers", True) and review.exists():
        with open(review, newline="", encoding="utf-8") as fh:
            excluded = {r["track_id"] for r in csv.DictReader(fh) if r["status"] == "outlier"}

    items: list[dict] = []
    with open(manifest, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            genre = row["genre"].strip()
            path = root / row["path"]
            if not genre or row["track_id"] in excluded or not path.exists():
                continue
            items.append({"track_id": row["track_id"], "path": path, "genre": genre,
                          "split": assign_split(row["track_id"], custom_cfg.split)})

    counts: dict[str, int] = {}
    for it in items:
        counts[it["genre"]] = counts.get(it["genre"], 0) + 1
    min_tracks = int(custom_cfg.get("min_tracks", 30))
    small = {g: n for g, n in counts.items() if n < min_tracks}
    if small:
        log.warning("custom genres ignored (< %d songs): %s", min_tracks, small)
    return [it for it in items if it["genre"] not in small]


def custom_genres(cfg) -> list[str]:
    return sorted({it["genre"] for it in load_custom_items(cfg.custom)})


def expand_dataset_cfg(dataset_cfg):
    """Add the custom genres to ``dataset_cfg.genres`` and update ``num_classes`` (in place)."""
    if not custom_enabled(dataset_cfg):
        return dataset_cfg
    genres = sorted(set(str(g) for g in dataset_cfg.genres) | set(custom_genres(dataset_cfg)))
    dataset_cfg.genres = genres
    dataset_cfg.num_classes = len(genres)
    return dataset_cfg
