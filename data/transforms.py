"""SpecAugment and Mixup augmentations for log-mel spectrograms."""
from __future__ import annotations

import torch
import torch.nn as nn


class SpecAugment(nn.Module):
    """Time and frequency masking (Park et al., 2019).

    Wraps torchaudio TimeMasking + FrequencyMasking applied N times each.
    """

    def __init__(
        self,
        time_mask_param: int = 80,
        freq_mask_param: int = 27,
        num_time_masks: int = 2,
        num_freq_masks: int = 2,
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, freq, time) or (1, freq, time) → same shape."""
        raise NotImplementedError


def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply Mixup to a batch.

    Returns:
        x_mixed, y_a, y_b, lam
        loss = lam * CE(pred, y_a) + (1 - lam) * CE(pred, y_b)
    """
    raise NotImplementedError
