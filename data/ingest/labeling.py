"""Label quality control: flag songs whose audio embedding is far from its genre's centre."""
from __future__ import annotations

import numpy as np


def flag_outliers(emb: np.ndarray, z_thr: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Cosine distance of each row of ``emb`` (N, D) to the class centroid, and a robust z-score.

    The z-score uses median / MAD, so a few mislabelled songs do not inflate the spread
    that is used to judge them. Returns ``(distance, z)``; outlier means ``z > z_thr``.
    """
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    centre = np.median(emb, axis=0)
    centre /= np.linalg.norm(centre) + 1e-12
    dist = 1.0 - emb @ centre
    med = np.median(dist)
    mad = np.median(np.abs(dist - med)) * 1.4826
    z = (dist - med) / (mad + 1e-12)
    return dist, z
