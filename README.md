# XRePIT / RePIT-Framework

**Residual-guided AI–CFD hybrid framework for stable and scalable fluid simulations.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2510.21804-b31b1b.svg)](https://arxiv.org/abs/2510.21804)

This repository is the official implementation of the **XRePIT** method introduced in:

> **Residual-guided AI-CFD hybrid method enables stable and scalable simulations: from 2D benchmarks to 3D applications**  
> Shilaj Baral, Youngkyu Lee, Sangam Khanal, Joongoo Jeon  
> *arXiv preprint* arXiv:2510.21804, 2025  
> [[Paper]](https://arxiv.org/abs/2510.21804)

XRePIT alternates between an OpenFOAM CFD solver and a data-driven neural network (FVMN or FNO), using physics residuals to decide when to switch. It achieves **up to 4.98× wall-clock speedup** over CFD-alone while keeping thermal field errors around 10⁻³ and velocity errors below 10⁻² m s⁻¹, over 10,000+ timesteps in both 2D and 3D natural convection problems.

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
  - [Conda (recommended)](#1-conda-recommended)
  - [pip](#2-pip)
  - [Docker](#3-docker)
  - [OpenFOAM](#4-openfoam)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Models](#models)
- [Visualization](#visualization)
- [Running the Tests](#running-the-tests)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Overview

RePIT-Framework automates the **XRePIT hybrid ML–CFD loop**:

```
┌─────────────────────────────────────────────────┐
│  1. Run OpenFOAM solver for N timesteps          │
│     (buoyantFoam, natural convection)            │
└─────────┬───────────────────────────────────────┘
          │  Convert fields → NumPy arrays
          ▼
┌─────────────────────────────────────────────────┐
│  2. Build FVMNDataset                            │
│     Normalize · add stencil features · apply BCs│
└─────────┬───────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────┐
│  3. Train ML model                               │
│     Full training (first segment) or            │
│     transfer learning with layer-freezing        │
└─────────┬───────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────┐
│  4. Autoregressive ML prediction                 │
│     Continue until mass-residual > ε_th          │
│     Save predictions as .npy files               │
└─────────┬───────────────────────────────────────┘
          │  Convert back to OpenFOAM format
          └──────────► repeat until t_end
```

The **scaled continuity residual** (mass residual relative to an OpenFOAM reference) acts as the switching signal: once the neural network diverges beyond a threshold ε_th, control returns to the solver.

---

## Key Results

| Case | Dimension | Model | Speedup (ψ) | ΔT error |
|------|-----------|-------|-------------|----------|
| Case A | 2D | FVMN | 2.05× | ~10⁻³ K |
| Case A | 2D | FVFNO | 1.44× | ~10⁻³ K |
| Case A | 3D | FVMN | **3.00×** | ~10⁻³ K |
| Case B | 2D | FVMN | 2.24× | ~10⁻³ K |
| Case C | 2D | FVMN | 2.17× | ~10⁻³ K |

---

## Repository Layout

```
repitframework/
├── repitframework/            # Main Python package
│   ├── config.py              # BaseConfig, TrainingConfig, NaturalConvectionConfig
│   ├── runner.py              # hybrid_train_predict() — main entry point
│   ├── trainer.py             # BaseHybridTrainer (training loop + checkpointing)
│   ├── predictor.py           # BaseHybridPredictor (autoregressive rollout)
│   ├── model_selector.py      # Factory for models, optimizers, schedulers
│   ├── plot_utils.py          # All visualization utilities (incl. plot_everything)
│   ├── utils.py               # Timer, state dict helpers
│   ├── foamResetFramework.py  # Reset OpenFOAM case to initial state
│   ├── DataLoader/
│   │   └── loader.py          # train_val_split()
│   ├── Dataset/
│   │   ├── baseline.py        # BaseDataset (generic CFD → PyTorch dataset)
│   │   ├── fvmn.py            # FVMNDataset (stencil features + hard BCs)
│   │   └── utils.py           # normalize, add_feature, hard_constraint_bc
│   ├── Metrics/
│   │   ├── ResidualNaturalConvection.py   # residual_mass, residual_momentum, residual_heat
│   │   └── OperatorEmbeddings.py
│   ├── Models/
│   │   ├── FVMN/              # Finite Volume Machine Network (MLP, node-assigned)
│   │   └── NeuralOperator/    # FNO1D, FNO2D, FVFNO1D, FVFNO2D
│   ├── OpenFOAM/
│   │   ├── utils.py           # OpenfoamUtils: run solver, parse fields, save numpy
│   │   ├── numpyToFoam.py     # Write predicted arrays back to OpenFOAM format
│   │   └── NusseltNumber/     # Custom OpenFOAM function object (C++)
│   └── Solvers/               # Ready-to-run OpenFOAM case templates
│       ├── natural_convection_case1/       # 2D cavity, ΔT = 19.6 K
│       ├── natural_convection_case1_3D/    # 3D cavity
│       ├── natural_convection_case2/       # 2D cavity, ΔT = 39.6 K
│       └── natural_convection_case3/       # 2D cavity, ΔT = 59.6 K
├── runner.py                  # Top-level entry point
├── random/                    # Analysis notebooks and scripts
│   ├── nature.ipynb           # Reproduction of all paper figures
│   ├── train_cylinderFNO.py   # Standalone FNO training example
│   ├── order_of_accuracy.py   # Grid convergence index (GCI) study
│   └── plot_residuals.py      # Residual time-series visualization
├── tests/                     # Unit tests
├── Dockerfile                 # Reproducible GPU + OpenFOAM image
├── docker-compose.yml
├── environment.yml            # Exact conda environment
├── setup.py
├── pyproject.toml
└── LICENSE.md
```

---

## Installation

### System Requirements

- **OS**: Linux (Ubuntu 22.04 recommended)
- **Python**: ≥ 3.12
- **GPU**: CUDA-capable GPU recommended (CPU fallback available)
- **OpenFOAM**: v12 (for CFD solver integration; not required for ML-only usage)

---

### 1. Conda (recommended)

```bash
# Clone the repository
git clone https://github.com/POSTECH-NINE/repitframework.git
cd repitframework

# Create environment from the exact specification used in the paper
conda env create -f environment.yml
conda activate repit_env

# Install the package in editable mode
pip install -e .
```

Minimal conda environment (if you prefer a lighter install):

```bash
conda create -n repit_env python=3.12 -y
conda activate repit_env
pip install -e .
```

---

### 2. pip

```bash
git clone https://github.com/POSTECH-NINE/repitframework.git
cd repitframework
pip install -e .
```

This installs all dependencies listed in `pyproject.toml`:
`numpy`, `pandas`, `torch`, `matplotlib`, `seaborn`, `imageio`, `tqdm`, `einops`, `Ofpp`.

---

### 3. Docker

The provided Docker image bundles PyTorch (CUDA 12.1) and OpenFOAM 12:

```bash
# Build the image
docker build -t repitframework:latest .

# Or pull a prebuilt image (if available)
docker pull shilaj/repitframework-v1.0:latest

# Run with GPU support, mounting your data directory
docker run -d --name repit \
    --gpus all \
    -p 8888:8888 \
    -v "$(pwd):/home/ninelab/repitframework" \
    repitframework:latest

# Enter the container
docker exec -it repit /bin/bash
```

---

### 4. OpenFOAM

OpenFOAM 12 is required to run the CFD solver stages. Inside the container it is installed automatically. For a bare-metal install on Ubuntu:

```bash
wget -q -O - https://dl.openfoam.org/gpg.key | apt-key add -
add-apt-repository http://dl.openfoam.org/ubuntu
apt-get update && apt-get install -y openfoam12
echo "source /opt/openfoam12/etc/bashrc" >> ~/.bashrc
source ~/.bashrc
```

---

## Quick Start

### Run the full hybrid loop

From the command line:

```bash
python runner.py
```

OR from your own script:

```python
from repitframework.config import NaturalConvectionConfig, OpenfoamConfig
from runner import hybrid_train_predict

# Use defaults: 2D natural convection Case A, FVMN model
training_config = NaturalConvectionConfig()
openfoam_config  = OpenfoamConfig()

hybrid_train_predict(
    training_config,
    openfoam_config,
    saved_model_name=None,        # None → train from scratch
    initial_training_epochs=5000,
    transfer_learning_epochs=2,
)
```
To reset the simulation and enable clean restart: 

```bash
python repitframework/foamResetFramework.py
```

### ML-only inference (no OpenFOAM required)

```python
from pathlib import Path
from repitframework.config import NaturalConvectionConfig
from repitframework.Dataset import FVMNDataset
from repitframework.DataLoader import train_val_split
from repitframework.trainer import BaseHybridTrainer
from repitframework.predictor import BaseHybridPredictor

config = NaturalConvectionConfig()

# Build dataset from pre-saved .npy files in config.assets_dir
dataset = FVMNDataset(
    start_time=10.0,
    end_time=10.03,
    time_step=0.01,
    dataset_dir=config.assets_dir,
    vars_list=config.get_variables(),
    extended_vars_list=config.extend_variables(),
    dims=config.data_dim,
    round_to=config.round_to,
    grid_x=config.grid_x,
    grid_y=config.grid_y,
    grid_z=config.grid_z,
    grid_step=config.grid_step,
    output_dims=config.output_dims,
    do_normalize=True,
    left_wall_temperature=config.left_wall_temperature,
    right_wall_temperature=config.right_wall_temperature,
    do_feature_selection=True,
)

train_loader, val_loader = train_val_split(dataset, batch_size=10000)

trainer = BaseHybridTrainer(config)
trainer.fit(train_loader, val_loader)

predictor = BaseHybridPredictor(training_config=config)
end_time = predictor.predict(prediction_start_time=10.03, model=trainer.model)
print(f"ML prediction ran until t = {end_time:.2f} s")
```

---

## Configuration

All experiment parameters live in `repitframework/config.py`. The main class is `NaturalConvectionConfig` (extends `TrainingConfig`):

```python
from repitframework.config import NaturalConvectionConfig

config = NaturalConvectionConfig()

# Key parameters and their defaults:
config.model_type            # "fvmn"  (or "fvfno2d")
config.epochs                # 5000    (initial training)
config.learning_rate         # 1e-3
config.batch_size            # 10000
config.residual_threshold    # 5.0     (ε_th — switching criterion)
config.do_feature_selection  # True    (stencil neighbourhood features)
config.layers_to_freeze      # 1       (layers frozen during transfer learning)
config.grid_x                # 200
config.grid_y                # 200
config.data_dim              # 2       (1 / 2 / 3)
config.write_interval        # 0.01    (OpenFOAM output interval, seconds)

# To switch to Case B (ΔT = 39.6 K):
from pathlib import Path
config.solver_dir = config.root_dir / "Solvers" / "natural_convection_case2"
```

---

## Models

### FVMN — Finite Volume Machine Network

A node-assigned MLP that learns to map stencil features (center + 2d+1 neighbours for d-dimensional grids) to the next-step field increment. Supports 1D, 2D, and 3D grids.

```python
from repitframework.Models import FVMNetwork

model = FVMNetwork(
    vars_list=["U_x", "U_y", "T"],
    hidden_layers=3,
    hidden_size=398,
    activation=torch.nn.ReLU,
    dropout=0.2,
)
```

### Fourier Neural Operators (FNO / FVFNO)

Spectral-domain operator learning. Available variants: `FNO1D`, `FNO2D`, `FVFNO1D`, `FVFNO2D`.

```python
# Select via config
config.model_type = "fvfno2d"
config.model_kwargs = {"modes": 12, "width": 32}
```

### Model selector

Use `model_selector.py` to swap models without changing the training loop:

```python
from repitframework.model_selector import ModelSelector
model = ModelSelector("fvmn", config.model_kwargs)
```

---

## Visualization

`repitframework/plot_utils.py` provides publication-quality figures (Nature journal style).

### All-in-one

```python
from pathlib import Path
from repitframework.plot_utils import plot_everything

plot_everything(
    prediction_dir=Path("repitframework/Assets/natural_convection_case1"),
    ground_truth_dir=Path("repitframework/Assets/natural_convection_case1_backup"),
    pred_time_list=[10.05, 10.06, 10.07, 10.08, 10.09, 10.10],
    save_dir=Path("repitframework/plots"),
    grid_shape=(200, 200),
    residual_threshold=5.0,
)
```

This generates in `save_dir/`:
- `field_comparison.png` — side-by-side temperature & velocity fields
- `l2_errors.png` — relative L2 error over time for each variable
- `mae_errors.png` — mean/max absolute error over time
- `streamlines.png` — overlaid streamlines (CFD vs prediction)
- `spectral_analysis.png` — power spectrum (CFD vs prediction)
- `residual_divergence.png` — log-scale continuity residual over time

### Individual plots

```python
from repitframework.plot_utils import (
    still_comparisons,       # Side-by-side field snapshots
    plot_MAE,                # Max / mean absolute error
    plot_L2_error,           # Relative L2 error
    plot_streamlines_comparison,  # Streamline overlay
    plot_spectral_analysis,  # FFT energy spectrum
    plot_residual_change,    # Scaled residual vs time
    save_loss,               # Training / validation loss curve
    make_animation,          # GIF of field evolution
)

still_comparisons(
    prediction_dir="path/to/predictions",
    ground_truth_dir="path/to/ground_truth",
    time_list=[20.0, 60.0, 110.0],
)
```

---

## Running the Tests

```bash
# From the repository root
python -m pytest tests/ -v
```

The test suite covers dataset utilities, OpenFOAM parsing helpers, and model components.

---

## Citation

If you use this code or the XRePIT method in your research, please cite:

```bibtex
@article{baral2025xrepit,
  title   = {Residual-guided {AI}-{CFD} hybrid method enables stable and scalable simulations:
             from {2D} benchmarks to {3D} applications},
  author  = {Baral, Shilaj and Lee, Youngkyu and Khanal, Sangam and Jeon, Joongoo},
  journal = {arXiv preprint arXiv:2510.21804},
  year    = {2025},
  url     = {https://arxiv.org/abs/2510.21804}
}
```

The XRePIT method builds on the original RePIT algorithm:

```bibtex
@article{jeon2022repit,
  title   = {Residual-based physics-informed transfer learning: A hybrid method for
             accelerating long-term {CFD} simulations via deep learning},
  author  = {Jeon, Joongoo and others},
  journal = {arXiv preprint arXiv:2206.06817},
  year    = {2022},
  url     = {https://arxiv.org/abs/2206.06817}
}
```

---

## License

This project is released under the [MIT License](LICENSE.md).

---

## Contact

- **Author**: Shilaj Baral — shilaj@postech.ac.kr
- **Lab**: NINELAB, Jeonbuk National University (POSTECH)
- **Issues / PRs**: [github.com/POSTECH-NINE/repitframework](https://github.com/POSTECH-NINE/repitframework)

Contributions, bug reports, and feature requests are welcome — please see [CONTRIBUTING.md](CONTRIBUTING.md).
