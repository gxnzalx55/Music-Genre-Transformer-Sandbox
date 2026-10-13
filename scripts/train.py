"""CLI: train the music genre classifier."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train music genre classifier.")
    p.add_argument("--base",    default="configs/base.yaml",
                   help="Base config YAML")
    p.add_argument("--dataset", required=True,
                   help="Dataset config YAML (e.g. configs/dataset/fma_small.yaml)")
    p.add_argument("--model",   required=True,
                   help="Model config YAML (e.g. configs/model/ast_scratch.yaml)")
    p.add_argument("overrides", nargs="*",
                   help="OmegaConf dotlist overrides, e.g. training.batch_size=32")
    return p.parse_args()


def main() -> None:
    import logging

    from torch.utils.data import DataLoader

    from data.dataset import FMADataset
    from models.factory import build_model
    from training.trainer import Trainer
    from training.utils import count_parameters, get_device, load_config, seed_everything

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    cfg = load_config(args.base, args.dataset, args.model, args.overrides)
    seed_everything(cfg.seed)
    device = get_device()

    # Validation also uses single random crops (mode="train"); the final,
    # deterministic window-averaged evaluation is done by scripts/evaluate.py.
    train_ds = FMADataset(cfg.dataset, "training", mode="train")
    val_ds = FMADataset(cfg.dataset, "validation", mode="train")

    t = cfg.training
    workers = int(t.num_workers)
    loader_kwargs = dict(
        batch_size=int(t.batch_size),
        num_workers=workers,
        pin_memory=bool(t.pin_memory) and device.type == "cuda",
    )
    if workers > 0:
        loader_kwargs.update(
            persistent_workers=bool(t.persistent_workers),
            prefetch_factor=int(t.prefetch_factor),
        )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    model = build_model(cfg.model, cfg.dataset).to(device)
    print(f"Model: {cfg.model.name} | {count_parameters(model) / 1e6:.2f} M trainable params")
    print(f"Data: {len(train_ds)} train / {len(val_ds)} val clips")

    best = Trainer(model, train_loader, val_loader, cfg).fit()
    print("Best validation metrics:", {k: round(v, 4) for k, v in best.items()})


if __name__ == "__main__":
    main()
