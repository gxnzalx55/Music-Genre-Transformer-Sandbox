"""CLI: predict top-K genres for a single audio file."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Predict music genre from an audio file."
    )
    p.add_argument("audio_file", help="Path to .mp3 / .wav / .flac, or a cached log-mel .npy")
    p.add_argument("--checkpoint", required=True,
                   help="Path to .pt checkpoint file")
    p.add_argument("--dataset",    default="configs/dataset/fma_small.yaml")
    p.add_argument("--model",      default="configs/model/ast_scratch.yaml")
    p.add_argument("--top-k", type=int, default=3,
                   help="Number of top genres to display")
    return p.parse_args()


def main() -> None:
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    from data.dataset import sliding_windows
    from data.preprocess import compute_melspec
    from data.stats import load_stats
    from evaluation.metrics import aggregate_window_logits
    from models.factory import load_model_from_checkpoint
    from training.utils import get_device

    args = parse_args()
    device = get_device()
    # --model is unused: the architecture comes from the checkpoint's stored config.
    model, ckpt_cfg = load_model_from_checkpoint(args.checkpoint, device)
    dataset_cfg = OmegaConf.load(args.dataset)
    genres = sorted(ckpt_cfg.dataset.genres)

    audio_path = Path(args.audio_file)
    if audio_path.suffix == ".npy":
        spec = np.load(audio_path).astype(np.float32)                    # (n_mels, T)
    else:
        spec = compute_melspec(audio_path, dataset_cfg)                  # (n_mels, T)
    sr, hop = dataset_cfg.audio.sample_rate, dataset_cfg.mel.hop_length
    windows = sliding_windows(
        spec,
        int(dataset_cfg.audio.eval_window_duration * sr / hop) + 1,
        int(dataset_cfg.audio.eval_window_hop * sr / hop),
    )
    mean, std = load_stats(dataset_cfg)
    x = torch.from_numpy(np.stack(windows)).float()
    x = ((x - mean) / (std + 1e-6)).unsqueeze(1).to(device)              # (W, 1, n_mels, T)

    with torch.no_grad():
        logits = model(x).float().cpu().numpy()
    probs = aggregate_window_logits(list(logits))

    for rank, idx in enumerate(np.argsort(probs)[::-1][: args.top_k], start=1):
        print(f"{rank}. {genres[idx]:<15} {probs[idx]:.3f}")


if __name__ == "__main__":
    main()
