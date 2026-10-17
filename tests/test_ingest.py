"""Tests for the YouTube ingestion helpers (segment maths, mel parity)."""
from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from data.ingest.segment import audio_to_mel, compute_segment


def test_normal_song_starts_at_35_percent():
    start, end = compute_segment(200.0)
    assert start == pytest.approx(70.0)
    assert end == pytest.approx(100.0)


def test_short_song_is_discarded():
    assert compute_segment(29.9) is None
    assert compute_segment(None) is None


def test_end_beyond_duration_shifts_back():
    # 35 % of 40 s = 14 s -> 44 s > 40 s, so the window moves back to 10-40 s
    start, end = compute_segment(40.0)
    assert end == pytest.approx(40.0)
    assert end - start == pytest.approx(30.0)
    assert start == pytest.approx(10.0)


def test_exactly_30s_song():
    assert compute_segment(30.0) == (0.0, 30.0)


def test_custom_ratio_and_length():
    start, end = compute_segment(100.0, offset_ratio=0.5, segment_seconds=10.0)
    assert (start, end) == (50.0, 60.0)


def test_audio_to_mel_matches_fma_pipeline(tmp_path):
    """Same file through compute_melspec (FMA path) and audio_to_mel (ingest path)."""
    import soundfile as sf
    from data.preprocess import compute_melspec

    cfg = OmegaConf.load("configs/dataset/fma_small.yaml")
    sr = 16000
    t = np.arange(30 * sr) / sr
    wav = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = tmp_path / "tone.wav"
    sf.write(path, wav, sr)

    mel = audio_to_mel(path, cfg, 30.0)
    ref = compute_melspec(path, cfg)
    assert mel.shape == ref.shape == (128, 3001)
    assert mel.dtype == np.float32
    np.testing.assert_allclose(mel, ref, atol=1e-3)


def test_store_migrates_old_manifest_and_saves_genre(tmp_path):
    import csv

    import numpy as np

    from scripts.ingest_playlist import MANIFEST_FIELDS, Store

    out = tmp_path / "custom"
    (out / "mels").mkdir(parents=True)
    np.save(out / "mels" / "old.npy", np.zeros((2, 2), dtype=np.float32))
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        fh.write("track_id,source,title,channel,duration,start,end,path\n"
                 "old,spotify,T,A,100.0,35.0,65.0,mels/old.npy\n")

    store = Store(out)
    assert store.genres == {"old": ""}
    assert "old" in store.done

    store.save({"track_id": "new", "source": "spotify", "title": "T2", "channel": "B",
                "duration": 90.0, "start": 31.5, "end": 61.5, "path": "mels/new.npy",
                "genre": "Reggaeton", "playlist": "url"}, np.zeros((2, 2), dtype=np.float32))
    with open(out / "manifest.csv", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert reader.fieldnames == MANIFEST_FIELDS
    assert [(r["track_id"], r["genre"]) for r in rows] == [("old", ""), ("new", "Reggaeton")]


def test_flag_outliers_catches_off_class_song():
    from data.ingest.labeling import flag_outliers

    rng = np.random.default_rng(0)
    centre = rng.normal(size=32)
    good = centre + 0.1 * rng.normal(size=(30, 32))
    bad = -centre[None] + 0.1 * rng.normal(size=(1, 32))
    _, z = flag_outliers(np.vstack([good, bad]), z_thr=3.0)
    assert z[-1] > 3.0
    assert (z[:-1] > 3.0).sum() <= 1
