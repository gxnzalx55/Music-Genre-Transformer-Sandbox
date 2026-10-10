"""Fine-tuning wrapper for a HuggingFace ASTForAudioClassification checkpoint."""
from __future__ import annotations

import torch
import torch.nn as nn


class PretrainedAST(nn.Module):
    """Loads a pretrained HF AST model and replaces its head for FMA genres.

    Args:
        cfg:         Model config node (hf_model_id, freeze_layers, etc.).
        num_classes: Number of target classes.
    """

    def __init__(self, cfg, num_classes: int) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, freq, time) → logits: (B, num_classes)."""
        raise NotImplementedError

    def get_param_groups(self, base_lr: float) -> list[dict]:
        """Return two param groups: encoder at base_lr * multiplier, head at base_lr."""
        raise NotImplementedError
