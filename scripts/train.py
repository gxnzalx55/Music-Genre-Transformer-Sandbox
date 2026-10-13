"""CLI: train the music genre classifier."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train music genre classifier.")
    p.add_argument("--base",    default="configs/base.yaml",
                   help="Base config YAML")
    p.add_argument("--dataset", required=True,
                   help="Dataset config YAML (e.g. configs/dataset/fma_small.yaml)")
    p.add_argument("--model",   required=True,
                   help="Model config YAML (e.g. configs/model/ast_scratch.yaml)")
    p.add_argument("overrides", nargs="*",
                   help="OmegaConf dotlist overrides, e.g. training.batch_size=32")
    return p.parse_args()


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
