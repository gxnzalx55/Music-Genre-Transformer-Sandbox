"""Audio Spectrogram Transformer trained from scratch."""
from __future__ import annotations

import math

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
        self.patch_size = tuple(patch_size)
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """x: (B, C, H, W) → tokens: (B, N, embed_dim), grid: (h_patches, w_patches)."""
        x = self.proj(x)  # (B, D, h, w)
        grid = (x.shape[2], x.shape[3])
        return x.flatten(2).transpose(1, 2), grid


class DropPath(nn.Module):
    """Stochastic depth regularisation (Huang et al., 2016)."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.dim() - 1)).bernoulli_(keep)
        return x * mask / keep


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
        if embed_dim % num_heads:
            raise ValueError(f"embed_dim {embed_dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.norm1 = nn.LayerNorm(embed_dim)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path)

    def _attn(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, d // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each (B, heads, N, head_dim)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout if self.training else 0.0
        )
        return self.proj_drop(self.proj(out.transpose(1, 2).reshape(b, n, d)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, D) → (B, N, D)."""
        x = x + self.drop_path(self._attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


def _sinusoidal_embedding(n: int, dim: int) -> torch.Tensor:
    pos = torch.arange(n, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(n, dim)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: dim // 2])
    return pe


class AudioSpectrogramTransformer(nn.Module):
    """AST: PatchEmbed → positional emb → [CLS] → Transformer encoder → head.

    Args:
        cfg:         Model config node.
        num_classes: Number of output classes (from dataset config).
        spec_size:   (freq_bins, time_frames) of the input spectrogram.
    """

    def __init__(self, cfg, num_classes: int, spec_size: tuple[int, int]) -> None:
        super().__init__()
        d = cfg.embed_dim
        self.patch_embed = PatchEmbed(tuple(cfg.patch_size), cfg.in_channels, d)
        ph, pw = cfg.patch_size
        num_patches = (spec_size[0] // ph) * (spec_size[1] // pw)
        if num_patches == 0:
            raise ValueError(f"spec_size {spec_size} smaller than patch_size {tuple(cfg.patch_size)}")

        self.use_cls = bool(cfg.cls_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d)) if self.use_cls else None
        n_tokens = num_patches + int(self.use_cls)
        if cfg.pos_embed == "learned":
            self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, d))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        elif cfg.pos_embed == "sinusoidal":
            self.register_buffer("pos_embed", _sinusoidal_embedding(n_tokens, d)[None])
        else:
            raise ValueError(f"Unknown pos_embed: {cfg.pos_embed!r}")
        if self.use_cls:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_drop = nn.Dropout(cfg.dropout)
        # Linearly increasing stochastic-depth rate across blocks.
        dpr = torch.linspace(0, cfg.drop_path_rate, cfg.depth).tolist()
        self.blocks = nn.ModuleList(
            TransformerEncoderBlock(
                d, cfg.num_heads, cfg.mlp_ratio, cfg.qkv_bias,
                cfg.dropout, cfg.attn_dropout, dpr[i],
            )
            for i in range(cfg.depth)
        )
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, num_classes)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, freq, time) → logits: (B, num_classes)."""
        x, _ = self.patch_embed(x)
        if self.use_cls:
            x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        if x.shape[1] != self.pos_embed.shape[1]:
            raise ValueError(
                f"Input yields {x.shape[1]} tokens but model was built for "
                f"{self.pos_embed.shape[1]}; check spec_size."
            )
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        pooled = x[:, 0] if self.use_cls else x.mean(dim=1)
        return self.head(pooled)
