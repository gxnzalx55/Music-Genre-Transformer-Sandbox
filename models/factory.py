"""Build the correct model from config."""
from __future__ import annotations

import torch.nn as nn


def build_model(model_cfg, dataset_cfg) -> nn.Module:
    """Instantiate AudioSpectrogramTransformer or PretrainedAST.

    Dispatches on model_cfg.type ∈ {"scratch", "pretrained"}.
    """
    raise NotImplementedError
