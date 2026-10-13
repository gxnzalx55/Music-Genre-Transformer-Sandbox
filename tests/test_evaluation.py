"""Tests for window aggregation, windowed evaluation and report writing."""
import numpy as np
import torch
import torch.nn as nn

from data.dataset import sliding_windows
from evaluation.metrics import aggregate_window_logits, eval_collate, evaluate_with_windows
from evaluation.report import save_classification_report, save_confusion_matrix


def test_aggregate_averages_probabilities():
    # A confident window outweighs two weak ones for the other class.
    logits = [np.array([10.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0])]
    probs = aggregate_window_logits(logits)
    assert probs.shape == (2,)
    assert np.isclose(probs.sum(), 1.0)
    assert probs.argmax() == 0


def test_sliding_windows_cover_clip():
    spec = np.zeros((4, 25))
    windows = sliding_windows(spec, window_frames=10, hop_frames=5)
    assert all(w.shape == (4, 10) for w in windows)
    assert len(windows) == 4  # starts 0, 5, 10, 15
    assert sliding_windows(np.zeros((4, 3)), 10, 5)[0].shape == (4, 10)  # short clip padded


def test_evaluate_with_windows_and_reports(tmp_path):
    class Fixed(nn.Module):
        """Predicts class 1 whenever the window mean is positive, else class 0."""
        def forward(self, x):
            s = x.mean(dim=(1, 2, 3))
            return torch.stack([-s, s], dim=1)

    pos = [torch.ones(1, 8, 8)] * 3
    neg = [-torch.ones(1, 8, 8)] * 2
    items = [(pos, 1), (neg, 0), (pos, 0)]  # last one is a deliberate error
    loader = [items]  # a loader yielding one eval_collate-style batch

    m = evaluate_with_windows(Fixed(), loader, torch.device("cpu"), num_classes=2)
    assert np.isclose(m["accuracy"], 2 / 3)
    assert m["confusion_matrix"].tolist() == [[1, 1], [0, 1]]
    assert m["per_class"][1]["support"] == 1

    save_confusion_matrix(m["confusion_matrix"], ["a", "b"], tmp_path / "cm.png")
    save_classification_report(m, ["a", "b"], tmp_path / "report.txt")
    assert (tmp_path / "cm.png").stat().st_size > 0
    assert "accuracy" in (tmp_path / "report.txt").read_text()
    assert eval_collate(items) is items
