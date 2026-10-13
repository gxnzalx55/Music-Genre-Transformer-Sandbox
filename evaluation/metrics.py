"""Accuracy, macro-F1, confusion matrix; sliding-window prediction averaging."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader


def eval_collate(batch: list) -> list:
    """Collate for ``FMADataset(mode="eval")``: keep ``(windows, label)`` items as a list."""
    return batch


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


@torch.no_grad()
def evaluate_with_windows(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict:
    """Run inference with sliding-window logit averaging per clip.

    ``loader`` must iterate an eval-mode ``FMADataset`` with
    ``collate_fn=eval_collate``: each batch is a list of ``(windows, label)``.

    Returns dict with keys:
        accuracy       (float)
        macro_f1       (float)
        per_class      (dict class_index → {precision, recall, f1, support})
        confusion_matrix (np.ndarray, shape (num_classes, num_classes))
    """
    model.eval()
    use_amp = device.type == "cuda"
    preds, targets = [], []

    for batch in loader:
        # One forward pass over all windows of all clips in the batch.
        counts = [len(windows) for windows, _ in batch]
        x = torch.cat([torch.stack(windows) for windows, _ in batch]).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(x).float().cpu().numpy()

        start = 0
        for n, (_, label) in zip(counts, batch):
            probs = aggregate_window_logits(list(logits[start : start + n]))
            preds.append(int(probs.argmax()))
            targets.append(int(label))
            start += n

    labels = list(range(num_classes))
    p, r, f1, support = precision_recall_fscore_support(
        targets, preds, labels=labels, zero_division=0
    )
    present = support > 0  # macro-F1 over classes that occur in the split
    return {
        "accuracy": float(np.mean(np.array(preds) == np.array(targets))),
        "macro_f1": float(f1[present].mean()),
        "per_class": {
            i: {
                "precision": float(p[i]), "recall": float(r[i]),
                "f1": float(f1[i]), "support": int(support[i]),
            }
            for i in labels
        },
        "confusion_matrix": confusion_matrix(targets, preds, labels=labels),
    }


def aggregate_window_logits(window_logits: list[np.ndarray]) -> np.ndarray:
    """Average softmax probabilities over windows → clip-level probability vector.

    The predicted class is ``result.argmax()``.
    """
    return _softmax(np.stack(window_logits)).mean(axis=0)
