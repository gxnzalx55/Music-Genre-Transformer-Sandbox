# Music Genre Classification — Audio Spectrogram Transformer

Train and evaluate an Audio Spectrogram Transformer (AST) on the
[FMA dataset](https://github.com/mdeff/fma).  Two model variants:

- **AST from scratch** — custom patch-embed + Transformer encoder (9.67 M parameters).
- **AST pretrained** — fine-tuning `MIT/ast-finetuned-audioset-10-10-0.4593`
  from HuggingFace.

See [`INFORME.md`](INFORME.md) for a detailed report of the implementation and results.

## Requirements

- Python 3.11+ (developed on 3.13)
- NVIDIA GPU with CUDA 12.8+ (developed on RTX 5070 12 GB / Blackwell, Windows 11)

## Installation

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# 2. Install PyTorch with CUDA 12.8 (required for Blackwell / RTX 50xx).
#    Do this BEFORE requirements.txt, otherwise pip may install a CPU-only build.
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# 3. Install remaining dependencies
pip install -r requirements.txt
```

Check that the GPU is visible: `python -c "import torch; print(torch.cuda.is_available())"`
must print `True`.  The code was tested with `torch 2.11.0+cu128` and `transformers 5.19`
(4.x is also supported).

## Usage

All commands are run from the project root.

### 1 — Download and preprocess

```bash
python scripts/prepare_data.py --dataset configs/dataset/fma_small.yaml
```

This will:
- Download `fma_small.zip` (~7.7 GB) and `fma_metadata.zip` (~0.4 GB).  The mirror does
  not publish `checksums.txt`, so checksum verification is skipped with a warning.
- Extract the archives (a few minutes).
- Compute log-mel spectrograms and cache them as `.npy` files in `data/cache/fma_small/`
  (~2 minutes).  Undecodable MP3s are listed in `data/cache/fma_small/corrupt_files.log`
  (6 of 8000 for FMA small).
- Compute the train-set mean/std and save them to `data/cache/fma_small/stats.json`
  (not into the YAML).

Useful flags: `--skip-download` and `--skip-preprocess`.  The script exits with a
non-zero code on failure, e.g. when audio dependencies are missing.

### 2 — Train

```bash
# AST from scratch
python scripts/train.py \
    --dataset configs/dataset/fma_small.yaml \
    --model   configs/model/ast_scratch.yaml

# Fine-tune pretrained AST (HuggingFace) — downloads ~350 MB on first use
python scripts/train.py \
    --dataset configs/dataset/fma_small.yaml \
    --model   configs/model/ast_pretrained.yaml \
    training.batch_size=16

# Override any config value from the CLI
python scripts/train.py \
    --dataset configs/dataset/fma_small.yaml \
    --model   configs/model/ast_scratch.yaml \
    training.batch_size=32 training.epochs=30 experiment_name=my_run
```

Notes:
- **Pretrained model: use `training.batch_size=16` on a 12 GB GPU.**  With 32 the peak
  memory (~15 GB) exceeds VRAM and every step becomes ~200× slower.
- `training.compile` is `false` by default because `torch.compile` needs Triton, which is
  not available on Windows.  Set `training.compile=true` on Linux for extra speed.
- Set `experiment_name=...` for each run: checkpoints and logs go to
  `checkpoints/<experiment_name>_<model name>/` and `logs/<experiment_name>_<model name>/`.
- Validation during training uses one random crop per clip; the final, deterministic
  evaluation with sliding windows is done by `evaluate.py`.

Each run keeps the top-3 checkpoints of its experiment and writes the best one to
`checkpoints/best.pt`.  **`best.pt` is shared by all experiments and is overwritten by each
new run**, so copy it (or use the per-experiment checkpoint) if you want to keep it.

TensorBoard logs are written to `logs/`.  Launch with:

```bash
tensorboard --logdir logs
```

### 3 — Evaluate

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best.pt \
    --dataset    configs/dataset/fma_small.yaml \
    --output-dir outputs/evaluation
```

Evaluates on the test split (`--split validation` is also available) using 10 s sliding
windows with probability averaging per clip.  Produces `confusion_matrix.png` and
`classification_report.txt` in the output directory and prints accuracy and macro-F1.
The model architecture is rebuilt from the config stored inside the checkpoint.

### 4 — Predict a single file

```bash
python scripts/predict.py path/to/song.mp3 \
    --checkpoint checkpoints/best.pt \
    --top-k 3
```

Output example:
```
1. Hip-Hop         0.414
2. Electronic      0.269
3. International   0.098
```

`--dataset` (default `configs/dataset/fma_small.yaml`) provides the mel settings and
normalisation statistics.  `--model` is accepted but unused: the architecture comes from
the checkpoint.

## Switching to FMA medium (16 genres)

Replace `--dataset configs/dataset/fma_small.yaml` with
`--dataset configs/dataset/fma_medium.yaml` in every command above and re-run
`prepare_data.py` first to build the medium cache.  **This path has not been tested**;
only FMA small has been run end to end.

## Project structure

```
music_genre_clf/
├── configs/           YAML hyperparameters (base + dataset + model)
├── data/              Download, preprocessing, Dataset, augmentations
├── models/            AST scratch, HuggingFace wrapper, factory, checkpoint loading
├── training/          Trainer, scheduler, early stopping, utilities
├── evaluation/        Metrics, confusion matrix, classification report
├── scripts/           CLI entry points
├── tests/             pytest suite (run with `python -m pytest`)
└── INFORME.md         Detailed implementation and results report
```

Datasets, caches, checkpoints, logs and outputs are git-ignored.

## Results (FMA small, 8 genres, RTX 5070)

| Model | Training | Val. accuracy | Test accuracy | Test macro-F1 |
|---|---|---|---|---|
| AST scratch | 50 epochs, ~11 min | 53.5 % | **49.6 %** | 0.485 |
| AST pretrained (HF) | 15 epochs, ~38 min | 63.3 % | not yet evaluated | — |

The pretrained model reached 54 % validation accuracy after a single epoch and plateaued
around 62–63 % from epoch 7, with mild overfitting afterwards.  Validation and test
accuracy are different measurements, so the two rows are not directly comparable until
the pretrained model is evaluated on the test split.  Both models are below the usual
literature figures; see `INFORME.md` for limitations and suggested next steps.
