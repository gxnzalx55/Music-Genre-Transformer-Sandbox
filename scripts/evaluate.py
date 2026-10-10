"""CLI: evaluate a checkpoint on the FMA test set."""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate model on FMA test set.")
    p.add_argument("--checkpoint", required=True,
                   help="Path to .pt checkpoint file")
    p.add_argument("--dataset",    required=True,
                   help="Dataset config YAML")
    p.add_argument("--output-dir", default="outputs/evaluation",
                   help="Directory to write confusion_matrix.png and report.txt")
    return p.parse_args()


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
