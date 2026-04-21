# Fourier Neural Operator (FNO) Implementation

Modular Fourier Neural Operator training and evaluation pipeline for PDE operator learning, with synthetic datasets (Burgers, Darcy, Navier-Stokes) and FD-Bench/PDEBench-style HDF5 ingestion.

## What This Repository Includes

- FNO implementations:
  - `ours`: custom architecture in `src/FNO/model.py`
  - `original`: wrapper around `neuraloperator` in `src/FNO/original_model.py`
- Data system based on registry APIs in `src/data/registry.py`
- Reproducible CLI scripts for generation, inspection, training, evaluation, and visualization
- FD-Bench-compatible data ingestion from local `.h5/.hdf5` files

## Quick Start

### 1. Environment Setup

```bash
# From repository root
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 2. Minimal End-to-End Run (Burgers)

```bash
# Optional data pre-generation
# If this is not run, data will be generated on-the-fly during training
python -m scripts.generate_data --datasets burgers --burgers_samples 200

# Inspect shape/channel metadata
python -m scripts.inspect_dataset --dataset burgers --n_samples 200 --subsample 8

# Train (wandb.run_name is required)
python -m scripts.train_fno \
  wandb.run_name=burgers_quick \
  wandb.mode=disabled \
  data.dataset=burgers \
  data.n_samples=200 \
  training.epochs=5

# Evaluate checkpoint
python -m scripts.evaluate_fno \
  --checkpoint_a models/burgers_quick.pth \
  --n_samples 200 \
  --n_plots 3
```

## Repository Map

```
fno_implementation/
├── configs/
│   └── train_fno.yaml                 # Hydra config defaults for training
├── scripts/
│   ├── train_fno.py                   # Train and save best checkpoint
│   ├── evaluate_fno.py                # Evaluate one or two checkpoints
│   ├── generate_data.py               # Generate/cache raw synthetic datasets
│   ├── inspect_dataset.py             # Print dataset sample/shape metadata
│   ├── visualize_data.py              # Plot raw dataset examples/stats
│   └── download_fd_bench_data.py      # List/check FD-Bench HF presets
├── src/
│   ├── FNO/
│   │   ├── model.py                   # Custom FNO components and model
│   │   ├── original_model.py          # neuraloperator-backed wrapper model
│   │   └── train.py                   # Training loop and checkpoint writing
│   ├── data/
│   │   ├── registry.py                # Dataset registration and split APIs
│   │   ├── providers_builtin.py       # Built-in dataset registrations
│   │   ├── datasets.py                # Dataset builders used by train/eval
│   │   ├── data_visualization.py      # Plotting utilities
│   │   └── sources/
│   │       ├── synthetic.py           # Burgers/Darcy/Navier-Stokes generators
│   │       └── fd_bench.py            # FD-Bench/PDEBench ingestion utilities
│   ├── eval/
│   │   ├── eval_framework.py          # Checkpoint compatibility + orchestration
│   │   └── evaluate.py                # Metrics and prediction plotting
│   └── utils.py                       # Seeding, checkpoint loading, helpers
├── generated_data/                    # Generated/cached HDF5 data
├── models/                            # Saved checkpoints
├── plots/                             # Evaluation plots
└── outputs/                           # Hydra run output folders
```

## Data and Registry System

All scripts use `src.data.registry`.

- Built-in registered datasets:
  - `burgers` (train/eval + raw generation)
  - `darcy` (train/eval + raw generation)
  - `navier_stokes` (raw generation only)
  - `fd_bench_<preset>` entries (train/eval from HuggingFace FD-Bench source)
- Core registry capabilities:
  - discover available datasets via `list_datasets(...)`
  - build train/test tensors via `build_train_test_split_from_args(...)`
  - generate raw data via `generate_raw_data(...)`

## Script Reference

### `scripts/generate_data.py`

Generates raw synthetic data and caches it under `generated_data/`.

```bash
python -m scripts.generate_data --datasets burgers darcy
python -m scripts.generate_data --datasets navier_stokes --navier_stokes_samples 50
```

Notes:
- `--dataset_kwargs_json` is passed to the raw generator for each selected dataset.
- Default dataset list includes synthetic datasets available for raw generation.

### `scripts/inspect_dataset.py`

Builds train/test splits through the registry and prints sample tensors and metadata.

```bash
python -m scripts.inspect_dataset --dataset burgers --n_samples 1000 --subsample 8
python -m scripts.inspect_dataset --dataset darcy --n_samples 400 --subsample 2
```

For HuggingFace FD-Bench presets:

```bash
python -m scripts.inspect_dataset \
  --dataset fd_bench_ns0 \
  --n_samples 100 \
  --dataset_kwargs_json '{"shuffle":false,"random_time":false}'
```

### `scripts/visualize_data.py`

Plots samples/statistics from raw generators.

```bash
python -m scripts.visualize_data --datasets burgers darcy --n_samples 100 --n_plots 5
python -m scripts.visualize_data --datasets navier_stokes --n_samples 30 --n_plots_navier 3
```

### `scripts/train_fno.py`

Hydra-driven training entrypoint.

Important behavior:
- `wandb.run_name` is required. If omitted, training exits with an error.
- Checkpoint path is: `model.models_folder/<wandb.run_name>.pth`.
- `wandb.mode=disabled` allows local runs without online logging.

```bash
# Burgers (custom implementation)
python -m scripts.train_fno \
  wandb.run_name=fno_burgers \
  wandb.mode=disabled \
  data.dataset=burgers \
  data.n_samples=1200 \
  data.subsample=8

# Darcy
python -m scripts.train_fno \
  wandb.run_name=fno_darcy \
  wandb.mode=disabled \
  data.dataset=darcy \
  model.modes=[12,12] \
  data.subsample=2

# Original neuraloperator-backed model
python -m scripts.train_fno \
  wandb.run_name=fno_original_burgers \
  wandb.mode=disabled \
  model.implementation=original \
  data.dataset=burgers
```

### `scripts/evaluate_fno.py`

Evaluates one or two checkpoints.

```bash
# Single model
python -m scripts.evaluate_fno --checkpoint_a models/fno_burgers.pth --n_plots 5

# Pairwise comparison
python -m scripts.evaluate_fno \
  --checkpoint_a models/fno_burgers.pth \
  --checkpoint_b models/fno_original_burgers.pth \
  --name_a "Ours" \
  --name_b "Original" \
  --n_plots 5
```

Backward-compatible aliases are supported:
- `--checkpoint` for `--checkpoint_a`
- `--original_checkpoint` for `--checkpoint_b`

Compatibility rule when using two checkpoints:
- dataset metadata must match (`dataset`, `subsample`, `dim`, `out_channels`, `dataset_kwargs`)

### `scripts/download_fd_bench_data.py`

Lists and checks FD-Bench HuggingFace presets (project-local downloads are disabled).

```bash
# List available presets
python -m scripts.download_fd_bench_data \
  --list

# Check one preset and optional access check
python -m scripts.download_fd_bench_data \
  --dataset ns0 \
  --check_access
```

## FD-Bench / PDEBench Compatibility Guide

The FD-Bench ingestion path in `src/data/sources/fd_bench.py` is HuggingFace-only.

- Source collection: `https://huggingface.co/collections/RuoyanLi1/fd-bench`
- Registered datasets follow `fd_bench_<preset>` names (for example `fd_bench_ns0`, `fd_bench_tgv`).
- FD-Bench entries are build-only datasets in the registry (no raw local generation).

### Training with FD-Bench

```bash
python -m scripts.train_fno \
  wandb.run_name=fdbench_run \
  wandb.mode=disabled \
  data.dataset=fd_bench_ns0 \
  data.n_samples=500 \
  data.subsample=2 \
  data.dataset_kwargs.hf_cache_dir=... \
  data.dataset_kwargs.temporal_subsample=1 \
  data.dataset_kwargs.batch_subsample=1 \
  data.dataset_kwargs.normalize=false \
  data.dataset_kwargs.shuffle=true \
  data.dataset_kwargs.random_time=true
```

You can also set `FNO_HF_CACHE_DIR` in your shell,
which is used as a fallback when `data.dataset_kwargs.hf_cache_dir` is not provided.

FD-Bench channel metadata is fixed per preset and registered in `FD_BENCH_DATASET_PRESETS`.

## Checkpoint Format

Saved checkpoints contain:

```python
{
    "model_state_dict": {...},
    "hyperparameters": {
        "modes": [...],
        "width": int,
        "layers": int,
        "in_channels": int,
        "out_channels": int,
        "dim": int,
        "dataset": str,
        "subsample": int,
        "implementation": "ours" | "original",
    },
}
```

These metadata are used to reconstruct the model and dataset settings during evaluation.

## Troubleshooting

- Error: `wandb.run_name must be set`
  - Fix: pass `wandb.run_name=<your_name>` in the train command.
- Error: `HuggingFace datasets is required for FD-Bench ingestion`
  - Fix: `pip install datasets`.
- Error: `Checkpoints are incompatible`
  - Fix: compare only checkpoints trained with matching dataset metadata.
- Unexpected duplicate registration warning for built-in datasets:
  - This warning is harmless and happens when registration is called multiple times in one process.

## References

- Fourier Neural Operator for Parametric PDEs: https://arxiv.org/abs/2010.08895
- Original implementation: https://github.com/zongyi-li/fourier_neural_operator
- FD-Bench: https://anonymous.4open.science/r/FD-Bench-15BC
