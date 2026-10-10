"""Linear warmup + cosine decay LR scheduler."""
from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def build_scheduler(
    optimizer: Optimizer,
    cfg,
    steps_per_epoch: int,
) -> LRScheduler:
    """Build a SequentialLR: linear warmup then cosine decay.

    Warmup runs for cfg.training.scheduler.warmup_epochs * steps_per_epoch steps.
    """
    raise NotImplementedError
