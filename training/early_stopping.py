"""Early stopping with best-checkpoint saving to disk."""
from __future__ import annotations

import copy
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
                          If None, the best weights are kept in memory.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = "max",
        checkpoint_path: Path | None = None,
    ) -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.best: float | None = None
        self.counter = 0
        self._best_state: dict | None = None

    def _improved(self, metric: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "max":
            return metric > self.best + self.min_delta
        return metric < self.best - self.min_delta

    def step(self, metric: float, model: nn.Module) -> bool:
        """Update state; save checkpoint if improved.

        Returns True when training should stop.
        """
        if self._improved(metric):
            self.best = metric
            self.counter = 0
            if self.checkpoint_path is not None:
                self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), self.checkpoint_path)
            else:
                self._best_state = copy.deepcopy(model.state_dict())
            return False
        self.counter += 1
        return self.counter >= self.patience

    def load_best(self, model: nn.Module) -> nn.Module:
        """Load best saved weights into model and return it."""
        if self.checkpoint_path is not None:
            state = torch.load(self.checkpoint_path, map_location="cpu")
        elif self._best_state is not None:
            state = self._best_state
        else:
            raise RuntimeError("No best checkpoint available: step() never improved.")
        model.load_state_dict(state)
        return model
