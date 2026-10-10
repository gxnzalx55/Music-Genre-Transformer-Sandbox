"""Early stopping with best-checkpoint saving to disk."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Args:
        patience:         Number of epochs to wait after last improvement.
        min_delta:        Minimum change to qualify as an improvement.
        mode:             "max" (higher is better) or "min".
        checkpoint_path:  Where to save the best model state_dict.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = "max",
        checkpoint_path: Path | None = None,
    ) -> None:
        raise NotImplementedError

    def step(self, metric: float, model: nn.Module) -> bool:
        """Update state; save checkpoint if improved.

        Returns True when training should stop.
        """
        raise NotImplementedError

    def load_best(self, model: nn.Module) -> nn.Module:
        """Load best saved weights into model and return it."""
        raise NotImplementedError
