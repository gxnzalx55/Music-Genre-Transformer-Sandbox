"""Device selection, reproducibility seeds, config loading, parameter counting."""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf


def get_device() -> torch.device:
    """Return best available device: CUDA → MPS → CPU.

    Prints device name and available VRAM (CUDA only).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(device)
        print(f"Device: {props.name} ({props.total_memory / 1024**3:.1f} GB VRAM)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: Apple MPS")
    else:
        device = torch.device("cpu")
        print("Device: CPU")
    return device


def seed_everything(seed: int = 42) -> None:
    """Fix Python, NumPy, and PyTorch (CPU + CUDA) random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(
    base: str,
    dataset: str,
    model: str,
    overrides: list[str] | None = None,
) -> DictConfig:
    """Merge base + dataset + model YAML configs; apply optional CLI dotlist overrides.

    Later files override earlier ones; CLI overrides take highest priority.
    The result has the layout ``cfg.<base keys>``, ``cfg.dataset`` and ``cfg.model``.
    A null ``model.num_classes`` is filled from ``dataset.num_classes``.
    """
    cfg = OmegaConf.merge(
        OmegaConf.load(base),
        {"dataset": OmegaConf.load(dataset), "model": OmegaConf.load(model)},
    )
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    if cfg.model.get("num_classes") is None:
        cfg.model.num_classes = cfg.dataset.num_classes
    return cfg


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
