"""Sanity check: model must overfit a single synthetic batch."""
import pytest


def test_overfit_single_batch(tiny_cfg, tmp_path):
    """Train on one batch of 8 synthetic samples for 200 steps.

    Expects train accuracy ≥ 0.95 — validates the full forward/backward chain.
    """
    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader, TensorDataset

    from models.ast_scratch import AudioSpectrogramTransformer
    from training.trainer import Trainer
    from training.utils import seed_everything

    seed_everything(0)
    cfg = OmegaConf.merge(tiny_cfg, {"training": {"epochs": 200, "batch_size": 8},
                                       "paths": {"checkpoint_dir": str(tmp_path)}})
    x = torch.randn(8, 1, 128, 64)
    y = torch.arange(8)
    loader = DataLoader(TensorDataset(x, y), batch_size=8)

    model = AudioSpectrogramTransformer(cfg.model, 8, (128, 64))
    best = Trainer(model, loader, loader, cfg).fit()

    assert best["val_acc"] >= 0.95
