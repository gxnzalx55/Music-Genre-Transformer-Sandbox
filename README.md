# Music Genre Classification — Audio Spectrogram Transformer

Train and evaluate an Audio Spectrogram Transformer (AST) on the
[FMA dataset](https://github.com/mdeff/fma).  Two model variants:

- **AST from scratch** — custom patch-embed + Transformer encoder.
- **AST pretrained** — fine-tuning `MIT/ast-finetuned-audioset-10-10-0.4593`
  from HuggingFace.

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA 12.8+ (developed on RTX 5070 / Blackwell)

## Installation

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# 2. Install PyTorch with CUDA 12.8 (required for Blackwell / RTX 50xx)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# 3. Install remaining dependencies
pip install -r requirements.txt
```

## Usage

### 1 — Download and preprocess

```bash
python scripts/prepare_data.py --dataset configs/dataset/fma_small.yaml
```

This will:
- Download `fma_small.zip` (~7.2 GB) and `fma_metadata.zip`.
- Extract and detect corrupt MP3s (logged to `data/cache/fma_small/corrupt_files.log`).
- Compute log-mel spectrograms and cache them as `.npy` files.
- Compute train-set mean/std and save them back into the dataset config.

### 2 — Train

```bash
# AST from scratch
python scripts/train.py \
    --dataset configs/dataset/fma_small.yaml \
    --model   configs/model/ast_scratch.yaml

# Fine-tune pretrained AST (HuggingFace)
python scripts/train.py \
    --dataset configs/dataset/fma_small.yaml \
    --model   configs/model/ast_pretrained.yaml

# Override any config value from the CLI
python scripts/train.py \
    --dataset configs/dataset/fma_small.yaml \
    --model   configs/model/ast_scratch.yaml \
    training.batch_size=32 training.epochs=30
```

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

Produces `confusion_matrix.png` and `classification_report.txt` in
`outputs/evaluation/`.

### 4 — Predict a single file

```bash
python scripts/predict.py path/to/song.mp3 \
    --checkpoint checkpoints/best.pt \
    --top-k 3
```

Output example:
```
1. Rock          0.612
2. Pop           0.198
3. Folk          0.087
```

## Switching to FMA medium (16 genres)

Replace `--dataset configs/dataset/fma_small.yaml` with
`--dataset configs/dataset/fma_medium.yaml` in every command above.
Re-run `prepare_data.py` first to build the medium cache.

## Project structure

```
music_genre_clf/
├── configs/           YAML hyperparameters (base + dataset + model)
├── data/              Download, preprocessing, Dataset, augmentations
├── models/            AST scratch, HuggingFace wrapper, factory
├── training/          Trainer, scheduler, early stopping, utilities
├── evaluation/        Metrics, confusion matrix, classification report
├── scripts/           CLI entry points
└── tests/             pytest suite
```

## Expected results (FMA small, RTX 5070)

| Model | Test accuracy | Training time |
|---|---|---|
| AST scratch | ~55–65 % | ~30–45 min |
| AST pretrained (HF) | ~75–85 % | ~20–30 min |
