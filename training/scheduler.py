"""Linear warmup + cosine decay LR scheduler."""
from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
)


def build_scheduler(
    optimizer: Optimizer,
    cfg,
    steps_per_epoch: int,
) -> LRScheduler:
    """Build a SequentialLR: linear warmup then cosine decay.

    Warmup runs for cfg.training.scheduler.warmup_epochs * steps_per_epoch steps.
    The scheduler is meant to be stepped once per optimizer step.
    """
    sched_cfg = cfg.training.scheduler
    total_steps = cfg.training.epochs * steps_per_epoch
    warmup_steps = min(int(sched_cfg.warmup_epochs * steps_per_epoch), total_steps - 1)

    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(total_steps - warmup_steps, 1),
        eta_min=sched_cfg.eta_min,
    )
    if warmup_steps <= 0:
        return cosine
    warmup = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])
