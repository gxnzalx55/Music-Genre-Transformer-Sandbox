"""Main training loop: forward, backward, AMP, logging, checkpointing."""
from __future__ import annotations

import contextlib
import logging
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data.transforms import SpecAugment, mixup_batch
from training.early_stopping import EarlyStopping
from training.scheduler import build_scheduler

log = logging.getLogger(__name__)

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class Trainer:
    """Encapsulates the training and validation loops.

    Handles:
    - BF16 autocast (CUDA Blackwell)
    - torch.compile(backend='inductor')
    - SpecAugment + Mixup augmentation
    - Gradient clipping
    - TensorBoard (and optional W&B) logging
    - EarlyStopping + top-k checkpoint saving

    Optional config sections (scheduler, early_stopping, logging, checkpoint,
    paths) fall back to defaults when absent, so minimal configs work in tests.

    Args:
        model:        The model to train (already on the correct device).
        train_loader: DataLoader for the training split.
        val_loader:   DataLoader for the validation split.
        cfg:          Merged OmegaConf config.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        tcfg = cfg.training
        self.device = next(model.parameters()).device

        # ── Mixed precision ────────────────────────────────────────────────
        amp = tcfg.get("amp", {})
        self.amp_enabled = bool(amp.get("enabled", False)) and self.device.type == "cuda"
        self.amp_dtype = _DTYPES[amp.get("dtype", "bfloat16")]
        # float16 needs loss scaling; bfloat16 does not.
        self.scaler = torch.amp.GradScaler(
            enabled=self.amp_enabled and self.amp_dtype == torch.float16
        )

        # ── Optimizer / scheduler ──────────────────────────────────────────
        ocfg = tcfg.optimizer
        if ocfg.type.lower() != "adamw":
            raise ValueError(f"Unsupported optimizer: {ocfg.type!r}")
        self.accum = int(tcfg.get("accumulate_grad_batches", 1))
        param_groups = (
            model.get_param_groups(ocfg.lr)
            if hasattr(model, "get_param_groups")
            else self._default_param_groups(ocfg)
        )
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=ocfg.lr,
            betas=tuple(ocfg.get("betas", (0.9, 0.999))),
            eps=ocfg.get("eps", 1e-8),
            weight_decay=ocfg.weight_decay,
        )
        self.steps_per_epoch = math.ceil(len(train_loader) / self.accum)
        self.scheduler = (
            build_scheduler(self.optimizer, cfg, self.steps_per_epoch)
            if tcfg.get("scheduler") is not None
            else None
        )

        # ── Augmentation ───────────────────────────────────────────────────
        aug = tcfg.get("augmentation", {})
        sa = aug.get("spec_augment", {})
        self.spec_augment = (
            SpecAugment(
                sa.get("time_mask_param", 80), sa.get("freq_mask_param", 27),
                sa.get("num_time_masks", 2), sa.get("num_freq_masks", 2),
            ).train()
            if sa.get("enabled", False) else None
        )
        mx = aug.get("mixup", {})
        self.mixup_alpha = float(mx.get("alpha", 0.3)) if mx.get("enabled", False) else 0.0

        # ── torch.compile (CUDA only; the uncompiled model is what we save) ─
        self._forward = model
        if tcfg.get("compile", False) and self.device.type == "cuda":
            self._forward = torch.compile(model, backend="inductor")

        # ── Checkpointing / early stopping / logging ───────────────────────
        ccfg = cfg.get("checkpoint", {})
        self.monitor = ccfg.get("monitor", "val_acc")
        self.mode = ccfg.get("mode", "max")
        self.save_top_k = int(ccfg.get("save_top_k", 3))
        paths = cfg.get("paths", {})
        run = f"{cfg.get('experiment_name', 'run')}_{cfg.model.get('name', cfg.model.type)}"
        self.ckpt_dir = Path(paths.get("checkpoint_dir", "checkpoints")) / run
        self.top_k: list[tuple[float, Path]] = []

        ecfg = tcfg.get("early_stopping", {})
        self.early_stopping = EarlyStopping(
            patience=ecfg.get("patience", tcfg.epochs),
            min_delta=ecfg.get("min_delta", 0.001),
            mode=self.mode,
            checkpoint_path=None,  # best weights are written by _save_checkpoint
        )

        lcfg = cfg.get("logging", {})
        self.log_every = int(lcfg.get("log_every_n_steps", 10))
        self.writer = None
        if lcfg.get("tensorboard", False):
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(Path(paths.get("log_dir", "logs")) / run)
            except ImportError:
                log.warning("tensorboard not installed — TensorBoard logging disabled")
        self.wandb = None
        if lcfg.get("wandb", False):
            import wandb

            self.wandb = wandb.init(
                project=lcfg.get("wandb_project", "music_genre_clf"),
                name=run,
                config=OmegaConf.to_container(cfg, resolve=True),
            )
        self.global_step = 0

    def _default_param_groups(self, ocfg) -> list[dict]:
        """Weight decay on matrices only; none on biases, norms, [CLS] or pos-embeds."""
        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            is_matrix = p.ndim >= 2 and "pos_embed" not in name and "cls_token" not in name
            (decay if is_matrix else no_decay).append(p)
        return [
            {"params": decay, "weight_decay": ocfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self) -> dict:
        """Run the full training loop.

        Returns best validation metrics dict.
        """
        best: dict = {}
        best_score: float | None = None
        for epoch in range(1, int(self.cfg.training.epochs) + 1):
            t0 = time.time()
            train_m = self._train_epoch(epoch)
            val_m = self._val_epoch(epoch)
            metrics = {**train_m, **val_m, "epoch": epoch}

            log.info(
                "epoch %d/%d  train_loss %.4f train_acc %.4f | val_loss %.4f "
                "val_acc %.4f val_f1 %.4f | %.0fs",
                epoch, self.cfg.training.epochs, train_m["train_loss"],
                train_m["train_acc"], val_m["val_loss"], val_m["val_acc"],
                val_m["val_f1"], time.time() - t0,
            )
            self._log_scalars(metrics, epoch)

            score = metrics[self.monitor]
            if best_score is None or (
                score > best_score if self.mode == "max" else score < best_score
            ):
                best_score, best = score, metrics
                self._save_checkpoint(self.ckpt_dir.parent / "best.pt", metrics)
            self._save_topk(score, epoch, metrics)

            if self.early_stopping.step(score, self.model):
                log.info("Early stopping at epoch %d", epoch)
                break

        self.early_stopping.load_best(self.model)
        if self.writer is not None:
            self.writer.close()
        if self.wandb is not None:
            self.wandb.finish()
        return best

    # ── Epoch loops ───────────────────────────────────────────────────────────

    def _autocast(self):
        if not self.amp_enabled:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        clip = float(self.cfg.training.get("gradient_clip_val", 0) or 0)
        total_loss = total_correct = total_n = 0.0
        n_batches = len(self.train_loader)
        self.optimizer.zero_grad(set_to_none=True)

        for i, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            if self.spec_augment is not None:
                x = self.spec_augment(x)
            y_b, lam = y, 1.0
            if self.mixup_alpha > 0:
                x, y, y_b, lam = mixup_batch(x, y, self.mixup_alpha)

            with self._autocast():
                logits = self._forward(x)
            logits = logits.float()
            loss = lam * F.cross_entropy(logits, y) + (1 - lam) * F.cross_entropy(logits, y_b)
            self.scaler.scale(loss / self.accum).backward()

            if (i + 1) % self.accum == 0 or i + 1 == n_batches:
                if clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    self.scheduler.step()
                self.global_step += 1
                if self.writer is not None and self.global_step % self.log_every == 0:
                    self.writer.add_scalar("train/loss_step", loss.item(), self.global_step)
                    self.writer.add_scalar(
                        "train/lr", self.optimizer.param_groups[-1]["lr"], self.global_step
                    )

            pred = logits.argmax(1)
            bs = x.shape[0]
            total_loss += loss.item() * bs
            total_correct += (
                lam * (pred == y).sum() + (1 - lam) * (pred == y_b).sum()
            ).item()
            total_n += bs

        return {
            "train_loss": total_loss / total_n,
            "train_acc": total_correct / total_n,
        }

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> dict:
        self.model.eval()
        total_loss = total_n = 0.0
        preds, targets = [], []
        for x, y in self.val_loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            with self._autocast():
                logits = self._forward(x)
            logits = logits.float()
            total_loss += F.cross_entropy(logits, y, reduction="sum").item()
            total_n += x.shape[0]
            preds.append(logits.argmax(1))
            targets.append(y)

        pred, target = torch.cat(preds), torch.cat(targets)
        num_classes = logits.shape[1]
        cm = torch.zeros(num_classes, num_classes, device=pred.device)
        cm.index_put_((target, pred), torch.ones_like(target, dtype=cm.dtype), accumulate=True)
        tp = cm.diag()
        precision = tp / cm.sum(0).clamp(min=1)
        recall = tp / cm.sum(1).clamp(min=1)
        f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
        present = cm.sum(1) > 0  # macro-F1 over classes that occur in val
        return {
            "val_loss": total_loss / total_n,
            "val_acc": (tp.sum() / cm.sum()).item(),
            "val_f1": f1[present].mean().item(),
        }

    # ── Logging / checkpoints ─────────────────────────────────────────────────

    def _log_scalars(self, metrics: dict, epoch: int) -> None:
        if self.writer is not None:
            for k, v in metrics.items():
                if k != "epoch":
                    self.writer.add_scalar(k.replace("_", "/", 1), v, epoch)
        if self.wandb is not None:
            self.wandb.log(metrics, step=epoch)

    def _save_checkpoint(self, path: Path, metrics: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "cfg": OmegaConf.to_container(self.cfg, resolve=True),
                "epoch": metrics["epoch"],
                "metrics": metrics,
            },
            path,
        )

    def _save_topk(self, score: float, epoch: int, metrics: dict) -> None:
        if self.save_top_k <= 0:
            return
        path = self.ckpt_dir / f"epoch{epoch:03d}_{self.monitor}{score:.4f}.pt"
        self._save_checkpoint(path, metrics)
        self.top_k.append((score, path))
        self.top_k.sort(key=lambda t: t[0], reverse=(self.mode == "max"))
        while len(self.top_k) > self.save_top_k:
            _, worst = self.top_k.pop()
            worst.unlink(missing_ok=True)
