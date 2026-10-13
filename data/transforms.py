"""SpecAugment and Mixup augmentations for log-mel spectrograms."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _mask_axis(x: torch.Tensor, max_width: int, axis: int) -> torch.Tensor:
    """Zero one random band per sample along ``axis`` (-2 = freq, -1 = time).

    ``x`` is (B, ..., freq, time); each sample gets its own band, of width
    uniform in [0, max_width] (clipped to the axis length).
    """
    size = x.shape[axis]
    max_width = min(max_width, size)
    if max_width <= 0:
        return x
    b = x.shape[0]
    width = torch.randint(0, max_width + 1, (b,), device=x.device)
    start = (torch.rand(b, device=x.device) * (size - width + 1).float()).long()
    idx = torch.arange(size, device=x.device)
    band = (idx[None, :] >= start[:, None]) & (idx[None, :] < (start + width)[:, None])
    shape = [b] + [1] * (x.dim() - 1)
    shape[axis] = size
    return x.masked_fill(band.reshape(shape), 0.0)


class SpecAugment(nn.Module):
    """Time and frequency masking (Park et al., 2019).

    Applies N time masks and N frequency masks, drawn independently per
    sample. Masked values are set to 0 (the mean after normalisation).
    Only active in training mode.
    """

    def __init__(
        self,
        time_mask_param: int = 80,
        freq_mask_param: int = 27,
        num_time_masks: int = 2,
        num_freq_masks: int = 2,
    ) -> None:
        super().__init__()
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, freq, time) or (1, freq, time) → same shape."""
        if not self.training:
            return x
        unbatched = x.dim() == 3
        if unbatched:
            x = x.unsqueeze(0)
        for _ in range(self.num_freq_masks):
            x = _mask_axis(x, self.freq_mask_param, axis=-2)
        for _ in range(self.num_time_masks):
            x = _mask_axis(x, self.time_mask_param, axis=-1)
        return x.squeeze(0) if unbatched else x


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
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    perm = torch.randperm(x.size(0), device=x.device)
    x_mixed = lam * x + (1.0 - lam) * x[perm]
    return x_mixed, y, y[perm], lam
