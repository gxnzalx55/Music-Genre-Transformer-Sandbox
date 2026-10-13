"""Build the correct model from config."""
from __future__ import annotations

import torch.nn as nn


def build_model(model_cfg, dataset_cfg) -> nn.Module:
    """Instantiate AudioSpectrogramTransformer or PretrainedAST.

    Dispatches on model_cfg.type ∈ {"scratch", "pretrained"}.
    """
    num_classes = dataset_cfg.num_classes
    if model_cfg.type == "scratch":
        from models.ast_scratch import AudioSpectrogramTransformer

        audio, mel = dataset_cfg.audio, dataset_cfg.mel
        # Same frame count as FMADataset.crop_frames.
        frames = int(audio.train_crop_duration * audio.sample_rate / mel.hop_length) + 1
        return AudioSpectrogramTransformer(model_cfg, num_classes, (mel.n_mels, frames))
    if model_cfg.type == "pretrained":
        from models.ast_pretrained import PretrainedAST

        return PretrainedAST(model_cfg, num_classes)
    raise ValueError(f"Unknown model type: {model_cfg.type!r}")
