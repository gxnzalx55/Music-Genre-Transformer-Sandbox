"""Audio Spectrogram Transformer trained from scratch."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """Split spectrogram into non-overlapping patches and project to embed_dim.

    Uses a Conv2d with kernel=stride=patch_size for efficient patch extraction.
    """

    def __init__(
        self,
        patch_size: tuple[int, int],
        in_channels: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """x: (B, C, H, W) → tokens: (B, N, embed_dim), grid: (h_patches, w_patches)."""
        raise NotImplementedError


class DropPath(nn.Module):
    """Stochastic depth regularisation (Huang et al., 2016)."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TransformerEncoderBlock(nn.Module):
    """Pre-norm Transformer block using F.scaled_dot_product_attention (Flash Attention)."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, D) → (B, N, D)."""
        raise NotImplementedError


class AudioSpectrogramTransformer(nn.Module):
    """AST: PatchEmbed → positional emb → [CLS] → Transformer encoder → head.

    Args:
        cfg:         Model config node.
        num_classes: Number of output classes (from dataset config).
        spec_size:   (freq_bins, time_frames) of the input spectrogram.
    """

    def __init__(self, cfg, num_classes: int, spec_size: tuple[int, int]) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, freq, time) → logits: (B, num_classes)."""
        raise NotImplementedError
