"""CLI: flag songs of data/custom whose audio does not fit the genre of their playlist."""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Embed every ingested song with the fine-tuned AST and flag, per genre, "
                    "the ones far from the class centre. Writes review.csv next to the manifest.")
    p.add_argument("--data", default="data/custom")
    p.add_argument("--checkpoint",
                   default="checkpoints/pretrained_scale05_AST-HuggingFace/best.pt")
    p.add_argument("--dataset", default="configs/dataset/fma_small.yaml")
    p.add_argument("--z", type=float, default=3.0, help="Robust z-score above which a song is flagged")
    p.add_argument("--min-tracks", type=int, default=10,
                   help="Genres with fewer songs are reported but not judged")
    return p.parse_args()


def main() -> None:
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    from data.dataset import sliding_windows
    from data.ingest.labeling import flag_outliers
    from data.stats import load_stats
    from models.factory import load_model_from_checkpoint
    from training.utils import get_device

    args = parse_args()
    root = Path(args.data)
    with open(root / "manifest.csv", newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["genre"]]
    if not rows:
        sys.exit("No labelled songs in the manifest (ingest with --genre first).")

    device = get_device()
    model, _ = load_model_from_checkpoint(args.checkpoint, device)
    dataset_cfg = OmegaConf.load(args.dataset)
    mean, std = load_stats(dataset_cfg)
    sr, hop = dataset_cfg.audio.sample_rate, dataset_cfg.mel.hop_length
    win = int(dataset_cfg.audio.eval_window_duration * sr / hop) + 1
    step = int(dataset_cfg.audio.eval_window_hop * sr / hop)

    captured: dict = {}
    # The classifier's input is the pooled AST representation (before LayerNorm/head).
    model.hf.classifier.register_forward_hook(lambda m, i, o: captured.__setitem__("x", i[0]))

    embs = []
    for r in rows:
        spec = np.load(root / r["path"]).astype(np.float32)
        x = torch.from_numpy(np.stack(sliding_windows(spec, win, step))).float()
        x = ((x - mean) / (std + 1e-6)).unsqueeze(1).to(device)
        with torch.no_grad():
            model(x)
        embs.append(captured["x"].float().mean(0).cpu().numpy())
    embs = np.stack(embs)

    by_genre = defaultdict(list)
    for i, r in enumerate(rows):
        by_genre[r["genre"]].append(i)

    out_rows, summary = [], []
    for genre, idx in sorted(by_genre.items()):
        if len(idx) < args.min_tracks:
            status = {i: "not_judged" for i in idx}
            dist = z = {i: "" for i in idx}
            summary.append((genre, len(idx), None))
        else:
            d, zz = flag_outliers(embs[idx], args.z)
            dist = {i: f"{d[k]:.4f}" for k, i in enumerate(idx)}
            z = {i: f"{zz[k]:.2f}" for k, i in enumerate(idx)}
            status = {i: ("outlier" if zz[k] > args.z else "ok") for k, i in enumerate(idx)}
            summary.append((genre, len(idx), int((zz > args.z).sum())))
        for i in idx:
            r = rows[i]
            out_rows.append({"track_id": r["track_id"], "genre": genre, "title": r["title"],
                             "channel": r["channel"], "distance": dist[i], "z": z[i],
                             "status": status[i]})
    out_rows.sort(key=lambda r: (r["genre"], -(float(r["z"]) if r["z"] else -1e9)))

    review = root / "review.csv"
    with open(review, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)

    for genre, n, flagged in summary:
        note = (f"{flagged} outliers" if flagged is not None
                else f"not judged (< {args.min_tracks} songs)")
        print(f"{genre:<20} {n:>4} songs  {note}")
    print(f"Review file: {review}")


if __name__ == "__main__":
    main()
