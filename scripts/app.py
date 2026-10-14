"""Gradio demo: upload an audio file and see the predicted genres."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gradio genre classifier demo.")
    p.add_argument("--checkpoint", default="checkpoints/pretrained_15ep_AST-HuggingFace/best.pt")
    p.add_argument("--dataset", default="configs/dataset/fma_small.yaml")
    p.add_argument("--share", action="store_true", help="Create a public Gradio link")
    return p.parse_args()


def main() -> None:
    import gradio as gr
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
    model, ckpt_cfg = load_model_from_checkpoint(args.checkpoint, device)
    model.eval()
    cfg = OmegaConf.load(args.dataset)
    genres = sorted(ckpt_cfg.dataset.genres)
    mean, std = load_stats(cfg)
    sr, hop = cfg.audio.sample_rate, cfg.mel.hop_length

    def predict(path: str) -> dict[str, float]:
        if not path:
            return {}
        spec = compute_melspec(Path(path), cfg)
        windows = sliding_windows(
            spec,
            int(cfg.audio.eval_window_duration * sr / hop) + 1,
            int(cfg.audio.eval_window_hop * sr / hop),
        )
        x = torch.from_numpy(np.stack(windows)).float()
        x = ((x - mean) / (std + 1e-6)).unsqueeze(1).to(device)
        with torch.no_grad():
            logits = model(x).float().cpu().numpy()
        probs = aggregate_window_logits(list(logits))
        return {g: float(p) for g, p in zip(genres, probs)}

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Audio(type="filepath", label="Audio (mp3/wav/flac)"),
        outputs=gr.Label(num_top_classes=len(genres), label="Genre"),
        title="Music genre classifier (AST, FMA small)",
        flagging_mode="never",
    )
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
