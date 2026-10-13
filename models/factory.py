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


def load_model_from_checkpoint(path, device="cpu"):
    """Rebuild a model from a ``Trainer`` checkpoint and load its weights.

    Returns ``(model_in_eval_mode, cfg)`` where ``cfg`` is the training config
    stored in the checkpoint.
    """
    import torch
    from omegaconf import OmegaConf

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = OmegaConf.create(ckpt["cfg"])
    model = build_model(cfg.model, cfg.dataset)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), cfg
