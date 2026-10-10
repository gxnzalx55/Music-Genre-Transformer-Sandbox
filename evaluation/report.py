"""Save confusion matrix plot and per-class classification report."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def save_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    """Write a seaborn heatmap of cm to output_path as PNG."""
    raise NotImplementedError


def save_classification_report(
    metrics: dict,
    class_names: list[str],
    output_path: Path,
) -> None:
    """Write per-class precision/recall/F1/support to output_path as plain text."""
    raise NotImplementedError
