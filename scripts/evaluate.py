"""CLI: evaluate a checkpoint on the FMA test set."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate model on FMA test set.")
    p.add_argument("--checkpoint", required=True,
                   help="Path to .pt checkpoint file")
    p.add_argument("--dataset",    required=True,
                   help="Dataset config YAML")
    p.add_argument("--split", default="test", choices=["validation", "test"],
                   help="Split to evaluate on")
    p.add_argument("--output-dir", default="outputs/evaluation",
                   help="Directory to write confusion_matrix.png and report.txt")
    return p.parse_args()


def main() -> None:
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    from data.dataset import FMADataset
    from evaluation.metrics import eval_collate, evaluate_with_windows
    from evaluation.report import save_classification_report, save_confusion_matrix
    from models.factory import load_model_from_checkpoint
    from training.utils import get_device

    args = parse_args()
    device = get_device()
    model, _ = load_model_from_checkpoint(args.checkpoint, device)

    dataset_cfg = OmegaConf.load(args.dataset)
    ds = FMADataset(dataset_cfg, args.split, mode="eval")
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=eval_collate)

    class_names = sorted(ds.genre_to_idx, key=ds.genre_to_idx.get)
    metrics = evaluate_with_windows(model, loader, device, num_classes=len(class_names))

    out = Path(args.output_dir)
    save_confusion_matrix(metrics["confusion_matrix"], class_names, out / "confusion_matrix.png")
    save_classification_report(metrics, class_names, out / "classification_report.txt")
    print(f"{args.split}: accuracy {metrics['accuracy']:.4f} | macro-F1 {metrics['macro_f1']:.4f}")
    print(f"Wrote results to {out}")


if __name__ == "__main__":
    main()
