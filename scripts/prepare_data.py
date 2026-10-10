"""CLI: download FMA archives and extract log-mel spectrograms to cache."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download and preprocess FMA dataset.")
    p.add_argument("--dataset", required=True,
                   help="Path to dataset YAML config (e.g. configs/dataset/fma_small.yaml)")
    p.add_argument("--skip-download",   action="store_true",
                   help="Skip download — archives already present")
    p.add_argument("--skip-preprocess", action="store_true",
                   help="Skip spectrogram extraction — cache already present")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.dataset)

    metadata_dir = Path(cfg.metadata_dir)
    subset = cfg.name.replace("fma_", "")

    # ── 1. Download ───────────────────────────────────────────────────────────
    if not args.skip_download:
        from data.download import download_fma, extract_archives, verify_checksums

        log.info("=== Step 1/4: Download ===")
        download_fma(metadata_dir.parent, subset=subset)

        log.info("=== Step 2/4: Verify checksums ===")
        if not verify_checksums(metadata_dir.parent, subset=subset):
            log.error("Checksum verification failed. Re-download the archives.")
            sys.exit(1)

        log.info("=== Step 3/4: Extract archives ===")
        extract_archives(metadata_dir.parent)
    else:
        log.info("Skipping download.")

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    if not args.skip_preprocess:
        from data.preprocess import preprocess_dataset

        log.info("=== Step 4/4: Compute spectrograms ===")
        preprocess_dataset(cfg)
    else:
        log.info("Skipping preprocessing.")

    # ── 3. Normalisation stats (train split only) ─────────────────────────────
    import pandas as pd
    from data.stats import compute_stats, save_stats

    log.info("=== Computing normalisation stats (train split) ===")
    tracks = pd.read_csv(
        metadata_dir / "tracks.csv",
        index_col=0,
        header=[0, 1],
    )
    train_ids: list[int] = tracks[
        (tracks[("set", "subset")] == subset) &
        (tracks[("set", "split")] == "training")
    ].index.tolist()

    mean, std = compute_stats(Path(cfg.cache_dir), train_ids)
    save_stats(cfg, mean, std)

    log.info(
        "prepare_data complete — mean=%.4f std=%.4f\n"
        "Next: python scripts/train.py --dataset %s --model configs/model/ast_scratch.yaml",
        mean, std, args.dataset,
    )


if __name__ == "__main__":
    main()
