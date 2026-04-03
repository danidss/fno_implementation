# Fourier Neural Operator (FNO) Implementation

A clean, modular implementation of Fourier Neural Operators for learning solution operators of parametric PDEs. This repository includes training, evaluation, data generation, and visualization utilities.

## Overview

This implementation supports:
- **1D PDEs**: Burgers' equation
- **2D PDEs**: Darcy Flow (training/evaluation), Navier-Stokes (data generation/visualization)
- **Comparison** with the original FNO implementation from [Zongyi Li et al.](https://github.com/zongyi-li/fourier_neural_operator)

## Installation

### Requirements
- Python 3.10+
- PyTorch with CUDA support (optional, CPU works too)
- Dependencies listed in `requirements.txt`

### Setup

```bash
# Clone and navigate to the directory
cd fno_implementation

# Create virtual environment (Optional but recommended)
python -m venv .env
source .env/bin/activate  # On Windows: .env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
fno_implementation/
├── scripts/                    # Runnable entrypoints (CLI + main logic)
│   ├── train_fno.py           # Train FNO models
│   ├── evaluate_fno.py        # Evaluate and compare models
│   ├── generate_data.py       # Generate PDE datasets
│   ├── visualize_data.py      # Visualize datasets
│   └── inspect_dataset.py     # Inspect dataset properties
│
├── src/                        # Core functionality
│   ├── FNO/                   # FNO model and training
│   │   ├── model.py           # Core FNO architecture (1D, 2D, 3D variants)
│   │   ├── original_model.py  # Wrapper around neuraloperator.models.FNO
│   │   └── train.py           # Training loop (reusable function)
│   │
│   ├── data/                  # Data generation and loading
│   │   ├── generate_data.py   # PDE solvers (Burgers, Darcy, NS)
│   │   ├── pytorch_data.py    # PyTorch dataset classes
│   │   └── data_visualization.py  # Plotting utilities
│   │
│   ├── eval/                  # Evaluation framework
│   │   ├── eval_framework.py  # Checkpoint loading & eval orchestration
│   │   └── evaluate.py        # Metrics and comparison logic
│   │
│   └── utils.py               # Utilities (checkpointing, seeding, etc.)
│
├── generated_data/            # Cached datasets (HDF5 format)
├── models/                    # Trained checkpoints
└── plots/                     # Generated evaluation plots
```

## Datasets

### Available PDEs

**Burgers' Equation (1D)**
- Initial condition: Gaussian Random Field
- Physics: Nonlinear advection-diffusion
- Default: 1000 training samples, 8192 spatial points
- Resolution-invariant: Model trained at 8192 points generalizes to any resolution

**Darcy Flow (2D)**
- Random permeability field → pressure solution
- Physics: Elliptic PDE (steady-state flow through porous media)
- Default: 1000 training samples, 421×421 grid
- Dataset split: 80% train, 20% test

**Navier-Stokes (2D)**
- Vorticity formulation
- Physics: Time-evolving incompressible flow
- Default: 100 training samples, outputs time evolution at multiple steps
- 256×256 spatial resolution

### Data Format
All datasets are cached in HDF5 format under `generated_data/` with automatic incremental generation—request 1000 samples, then later request 1500, and only 500 new samples are generated and appended.

## Usage

All scripts accept command-line arguments. Use `--help` for detailed options.

### 1. Generate Datasets

```bash
# Generate all default datasets
python -m scripts.generate_data

# Generate specific datasets with custom sample counts
python -m scripts.generate_data --datasets burgers darcy \
  --burgers_samples 2000 --darcy_samples 1500
```

**Output**: `generated_data/burgers_*.h5`, `generated_data/darcy_*.h5`

### 2. Inspect Dataset Properties

```bash
# Check dataset shapes and statistics
python -m scripts.inspect_dataset --dataset burgers --n_samples 1000

python -m scripts.inspect_dataset --dataset darcy --n_samples 1000 --subsample 2
```

**Output**: Prints input/output shapes, train/test split info, number of channels, spatial dimension.

### 3. Visualize Datasets

```bash
# Visualize Darcy Flow and Burgers' equation
python -m scripts.visualize_data --datasets darcy burgers --n_samples 100

# Visualize only Navier-Stokes with 50 samples
python -m scripts.visualize_data --datasets navier_stokes --n_samples 50
```

**Output**: Interactive Matplotlib windows for samples, statistics, and comparisons.

### 4. Train FNO

Training uses **Hydra** for configuration and **Weights & Biases** for experiment tracking.

```bash
# Default training (defined in configs/train_fno.yaml)
python -m scripts.train_fno

# Override hyperparameters via command line
python -m scripts.train_fno \
  data.dataset=darcy \
  model.modes=8 \
  model.width=32 \
  training.epochs=1000 \
  training.learning_rate=0.0005 \
  wandb.project=fno-darcy-runs
```

**Key Parameters (YAML path):**
- `data.dataset`: {burgers, darcy} - PDE to solve
- `data.n_samples`: Total samples to generate/use
- `model.implementation`: {ours, original}
- `model.modes`: Fourier modes to retain
- `model.width`: Hidden channel width
- `training.batch_size`: Batch size
- `training.epochs`: Number of epochs
- `wandb.mode`: Set to `disabled` for offline runs

**Output**: Best model checkpoint saved in `model.models_folder` with filename `<wandb.run_name>.pth` (default: `models/fno_burgers.pth`).

### 5. Evaluate Models

```bash
# Evaluate trained model (our implementation)
python -m scripts.evaluate_fno --checkpoint_a models/fno_burgers.pth --n_plots 5

# Compare with original FNO implementation
python -m scripts.evaluate_fno \
  --checkpoint_a models/fno_burgers.pth \
  --checkpoint_b models/fno_original_burgers.pth \
  --n_plots 10
```

`--checkpoint` and `--original_checkpoint` are still supported as backward-compatible aliases.

**Output**: 
- Console: Parameter count, inference time, relative L2 error, MSE, MAE
- PNG plots: Ground truth vs predictions, difference maps, side-by-side comparisons

## Model Architecture

### Spectral Convolution

The core operation replaces standard convolution in the Fourier domain:

```
Input: (batch, channels, spatial...)
  ↓
FFT (1D/2D/3D)
  ↓
Truncate to specified Fourier modes
  ↓
Learnable linear transform (channel-wise)
  ↓
Inverse FFT
  ↓
Output: (batch, channels, spatial...)
```

### FNO Layers

Each FNO layer combines:
1. **Spectral convolution** (Fourier-domain operator)
2. **Local convolution** (spatial domain, coeff-wise)
3. **Normalization + Nonlinearity** (BatchNorm + ReLU)
4. **Skip connection** (residual link)

The model automatically handles variable input resolutions from a single trained instance.

## Checkpoint Format

Trained models are saved with both weights and hyperparameters:

```python
{
    "model_state_dict": {...},
    "hyperparameters": {
        "dim": 1,
        "modes": 16,
        "width": 64,
        "layers": 4,
        "in_channels": 2,
      "out_channels": 1,
        "dataset": "burgers",
      "subsample": 1,
      "implementation": "ours"
    }
}
```

This allows automatic model reconstruction during evaluation.

## Typical Workflow

1. **Generate data** (optional, as it will be generated on-the-fly if missing):
   ```bash
   python -m scripts.generate_data --burgers_samples 1000
   ```

2. **Inspect data** before training:
   ```bash
   python -m scripts.inspect_dataset --dataset burgers
   ```

3. **Train model**:
   ```bash
   python -m scripts.train_fno --dataset burgers --epochs 500
   ```

4. **Evaluate**:
   ```bash
  python -m scripts.evaluate_fno --checkpoint_a models/fno_burgers.pth --n_plots 5
   ```

5. **Compare with original** (optional):
   ```bash
   python -m scripts.evaluate_fno \
     --checkpoint_a models/fno_burgers.pth \
     --checkpoint_b models/fno_original_burgers.pth
   ```

## References

- **FNO Paper**: [Fourier Neural Operator for Parametric PDEs](https://arxiv.org/abs/2010.08895) (Zongyi Li et al., ICLR 2021)
- **Original Implementation**: [GitHub - zongyi-li/fourier_neural_operator](https://github.com/zongyi-li/fourier_neural_operator)
