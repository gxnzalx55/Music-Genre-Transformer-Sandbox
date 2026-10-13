"""Save confusion matrix plot and per-class classification report."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402


def save_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    """Write a seaborn heatmap of cm to output_path as PNG.

    Cells show raw counts; colour is the row-normalised recall so classes of
    different sizes are comparable.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    size = max(6, 0.9 * len(class_names))
    fig, ax = plt.subplots(figsize=(size + 1, size))
    sns.heatmap(
        norm, annot=cm, fmt="d", cmap="Blues", vmin=0, vmax=1,
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={"label": "Fraction of true class"}, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_classification_report(
    metrics: dict,
    class_names: list[str],
    output_path: Path,
) -> None:
    """Write per-class precision/recall/F1/support to output_path as plain text."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width = max(len(n) for n in class_names + ["macro avg"])

    lines = [f"{'':<{width}}  {'precision':>9}  {'recall':>9}  {'f1-score':>9}  {'support':>7}"]
    for i, name in enumerate(class_names):
        m = metrics["per_class"][i]
        lines.append(
            f"{name:<{width}}  {m['precision']:>9.3f}  {m['recall']:>9.3f}  "
            f"{m['f1']:>9.3f}  {m['support']:>7d}"
        )
    lines.append("")
    lines.append(f"accuracy: {metrics['accuracy']:.4f}")
    lines.append(f"macro-F1: {metrics['macro_f1']:.4f}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
