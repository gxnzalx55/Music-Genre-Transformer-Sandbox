"""Fine-tuning wrapper for a HuggingFace ASTForAudioClassification checkpoint."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PretrainedAST(nn.Module):
    """Loads a pretrained HF AST model and replaces its head for FMA genres.

    The HF model has fixed positional embeddings for ``max_length`` time frames
    (1024), so inputs are zero-padded / cropped along time to that length.

    Args:
        cfg:         Model config node (hf_model_id, freeze_layers, etc.).
        num_classes: Number of target classes.
    """

    def __init__(self, cfg, num_classes: int) -> None:
        super().__init__()
        from transformers import ASTForAudioClassification

        self.hf = ASTForAudioClassification.from_pretrained(cfg.hf_model_id)
        hidden = self.hf.config.hidden_size
        self.max_length = self.hf.config.max_length
        self.lr_multiplier_encoder = float(cfg.lr_multiplier_encoder)
        self.hf.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(hidden, num_classes),
        )

        n_freeze = int(cfg.freeze_layers)
        if n_freeze > 0:
            ast = self.hf.audio_spectrogram_transformer
            for p in ast.embeddings.parameters():
                p.requires_grad = False
            # transformers 4.x: encoder.layer; 5.x: layers
            blocks = ast.layers if hasattr(ast, "layers") else ast.encoder.layer
            for layer in blocks[:n_freeze]:
                for p in layer.parameters():
                    p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, freq, time) → logits: (B, num_classes)."""
        x = x.squeeze(1).transpose(1, 2)  # (B, time, freq), as HF expects
        t = x.shape[1]
        if t < self.max_length:
            x = F.pad(x, (0, 0, 0, self.max_length - t))
        elif t > self.max_length:
            x = x[:, : self.max_length]
        return self.hf(input_values=x).logits

    def get_param_groups(self, base_lr: float) -> list[dict]:
        """Return two param groups: encoder at base_lr * multiplier, head at base_lr."""
        encoder = [p for p in self.hf.audio_spectrogram_transformer.parameters() if p.requires_grad]
        head = list(self.hf.classifier.parameters())
        return [
            {"params": encoder, "lr": base_lr * self.lr_multiplier_encoder},
            {"params": head, "lr": base_lr},
        ]
