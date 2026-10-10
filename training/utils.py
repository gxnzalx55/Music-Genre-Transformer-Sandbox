"""Device selection, reproducibility seeds, config loading, parameter counting."""
from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig


def get_device() -> torch.device:
    """Return best available device: CUDA → MPS → CPU.

    Prints device name and available VRAM (CUDA only).
    """
    raise NotImplementedError


def seed_everything(seed: int = 42) -> None:
    """Fix Python, NumPy, and PyTorch (CPU + CUDA) random seeds."""
    raise NotImplementedError


def load_config(
    base: str,
    dataset: str,
    model: str,
    overrides: list[str] | None = None,
) -> DictConfig:
    """Merge base + dataset + model YAML configs; apply optional CLI dotlist overrides.

    Later files override earlier ones; CLI overrides take highest priority.
    """
    raise NotImplementedError


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    raise NotImplementedError
