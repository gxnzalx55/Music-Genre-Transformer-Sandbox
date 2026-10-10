"""Shared pytest fixtures."""
import pytest
from omegaconf import OmegaConf


@pytest.fixture
def tiny_cfg():
    """Minimal config for fast unit tests (no real data required)."""
    return OmegaConf.create({
        "dataset": {
            "name": "fma_small",
            "num_classes": 8,
            "genres": [
                "Electronic", "Experimental", "Folk", "Hip-Hop",
                "Instrumental", "International", "Pop", "Rock",
            ],
            "audio": {
                "sample_rate": 16000,
                "train_crop_duration": 10.0,
                "eval_window_duration": 10.0,
                "eval_window_hop": 5.0,
            },
            "mel": {
                "n_mels": 128,
                "n_fft": 400,
                "hop_length": 160,
                "f_min": 0,
                "f_max": 8000,
            },
            "stats": {"mean": 0.0, "std": 1.0},
        },
        "model": {
            "type": "scratch",
            "patch_size": [16, 16],
            "in_channels": 1,
            "embed_dim": 64,
            "depth": 2,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "qkv_bias": True,
            "dropout": 0.0,
            "attn_dropout": 0.0,
            "drop_path_rate": 0.0,
            "cls_token": True,
            "pos_embed": "learned",
        },
        "training": {
            "epochs": 5,
            "batch_size": 4,
            "amp": {"enabled": False, "dtype": "float32"},
            "compile": False,
            "gradient_clip_val": 1.0,
            "optimizer": {"type": "adamw", "lr": 1e-3, "weight_decay": 0.0},
            "augmentation": {
                "spec_augment": {"enabled": False},
                "mixup": {"enabled": False},
            },
        },
    })
