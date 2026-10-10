"""Accuracy, macro-F1, confusion matrix; sliding-window prediction averaging."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate_with_windows(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict:
    """Run inference with sliding-window logit averaging per clip.

    Returns dict with keys:
        accuracy       (float)
        macro_f1       (float)
        per_class      (dict genre → {precision, recall, f1, support})
        confusion_matrix (np.ndarray, shape (num_classes, num_classes))
    """
    raise NotImplementedError


def aggregate_window_logits(window_logits: list[np.ndarray]) -> np.ndarray:
    """Average softmax probabilities over windows → final predicted class index."""
    raise NotImplementedError
