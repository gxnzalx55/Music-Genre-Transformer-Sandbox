"""CLI: predict top-K genres for a single audio file."""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Predict music genre from an audio file."
    )
    p.add_argument("audio_file", help="Path to .mp3 / .wav / .flac")
    p.add_argument("--checkpoint", required=True,
                   help="Path to .pt checkpoint file")
    p.add_argument("--dataset",    default="configs/dataset/fma_small.yaml")
    p.add_argument("--model",      default="configs/model/ast_scratch.yaml")
    p.add_argument("--top-k", type=int, default=3,
                   help="Number of top genres to display")
    return p.parse_args()


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
