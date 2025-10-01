# RePIT-Framework

Automation framework for hybrid ML–CFD (Machine Learning — Computational Fluid Dynamics) cross-computation. RePIT-Framework is an extension of the RePIT algorithm introduced by J. Jeon et al. (see citation below). It provides utilities to prepare FVM/CFD data, integrate ML models (Neural Operators, FVMN, etc.), run hybrid RePIT-style training/inference, and interoperate with OpenFOAM pipelines.

- Author: NINELAB
- Package name: repitframework (version 1.0)
- Python: >= 3.13
- License: MIT

Table of contents
- Overview
- Key features
- Quick start (clone + install)
- Installation (conda / pip / docker)
- Running examples
- Project layout
- Development notes (testing, contribution, reproducibility)
- Citation & license
- Contact

Overview
--------
RePIT-Framework is intended to automate ML–CFD workflows:
- Convert OpenFOAM results to numpy datasets and structured inputs for ML.
- Provide dataset classes (FVMN dataset, baseline dataset) and utilities for normalization and feature engineering.
- Implement baseline ML models and neural operators (e.g., FNO, FVMN variants) and a trainer that manages checkpointing, scheduler/optimizer interaction, and hybrid training logic.
- Utilities to run, parse, and visualize CFD solutions and to orchestrate combined ML–CFD experiments.

Key features
------------
- Dataset utilities and format conversion (OpenFOAM <-> numpy).
- Model implementations: Fourier Neural Operator (FNO), FVMN and other baseline networks.
- Trainer that handles checkpointing, metric tracking, resuming training, and custom loss pipelines.
- Visualization scripts (3D viz, probe plots) and notebooks for result analysis and plotting.
- Docker image to reproduce a full environment (PyTorch + CUDA + Miniconda + utilities).
- OpenFOAM utilities (Ofpp-backed helpers) to run solvers and parse results programmatically.

Quick start
-----------
1. Clone repository:
  ```
  git clone git@github.com:JBNU-NINE/repitframework.git
  cd repitframework
  ```

2. Local editable install (recommended for development):
  `python -m pip install -e .`

3. Or use the provided Docker image (recommended for reproducibility, see Docker section).

Installation
------------
1) System requirements
- Linux (Ubuntu recommended for OpenFOAM compatibility)
- CUDA drivers for GPU workflows (if using GPU Docker or native GPU training)
- Python >= 3.13

2) Install via conda (example)
- Create and activate a conda env (the Dockerfile uses Miniconda internally and a conda environment for reproducibility):
```
  conda create -n repit_env python=3.13 -y
  conda activate repit_env
```

- Install dependencies (minimal):
` pip install numpy pandas torch imageio tqdm Ofpp `

Notes:
- setup.py lists dependencies: numpy, pandas, Ofpp, torch, imageio, tqdm.
- Some parts of the framework interact with OpenFOAM and may require system-level OpenFOAM installation.

3) Using Docker (the quickest reproducible route)
- Pull prebuilt image (if available):
  `docker pull shilaj/repitframework-v1.0:latest`

- Run container (example):
```
 docker run -d --name repitframework --gpus all -p 8888:8888 -v "/path/on/your/host:/home/ninelab/repitframework" shilaj/repitframework-v1.0:latest
```

- Exec into it:
 `docker exec -it repitframework /bin/bash`

- The provided Dockerfile uses base image: pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel, and creates a non-root user `ninelab` and installs Miniconda, system dependencies, and project code.

OpenFOAM
--------
- Several pipeline utilities assume OpenFOAM is installed on the host or inside the container.
- Example apt-install snippet:
  ```
  wget -q -O - https://dl.openfoam.org/gpg.key | apt-key add - && \
  add-apt-repository http://dl.openfoam.org/ubuntu && \
  apt-get update && apt-get install -y openfoam12 && \
  echo "source /opt/openfoam12/etc/bashrc" >> /home/$USER/.bashrc && \
  chown -R $USER:$USER /opt/openfoam12 /home/$USER
  ```

Usage examples
--------------
1) Convert/OpenFOAM parsing
- The repo contains OpenFOAM utilities in repitframework/OpenFOAM/utils.py that parse field data to numpy and provide helpers to run solvers programmatically. Typical usage:
  `python -c "from repitframework.OpenFOAM.utils import OpenfoamUtils; ..."`

2) Dataset creation
- Build datasets for training via the dataset classes in repitframework/Dataset (FVMNDataset, BaseDataset). Example pseudocode:
```
  from repitframework.Dataset.fvmn import FVMNDataset

  dataset = FVMNDataset(start_time, 
                      end_time, 
                      grid_step, 
                      dataset_dir, 
                      do_normalize=True, ...)
```

3) Training
- The hybrid trainer is at repitframework/runner.py and supports saving best models and resuming from checkpoints. Usage pattern:
  - Prepare config (metrics, model params, dataset paths)
  - Initialize OpenFOAMUtils
  - Initialize trainer (BaseHybridTrainer or subclass)
  - Initialize predictor (BaseHybridPredictor or subclass)
  - openfoam.run_solver(...)
  - trainer.fit(train_loader, val_loader)
  - predictor.predict(...)
  - REPEAT

<!-- 4) 3D visualization
- A script random/vis_3d.py provides interactive visualization of meshes with scalar slices and velocity glyphs. Example (adjust to the concrete CLI of the script):
  python random/vis_3d.py --input path/to/mesh.vtk --scalar T --vector U -->

4) Notebooks & plotting
- There are notebooks in random/ (*.ipynb) that showcase plotting routines and analyses (probe time-series, number of ML steps vs simulation time, publication-ready figure styling).

Project layout (high-level)
---------------------------
- **repitframework/** 
  - **OpenFOAM/**       # OpenFOAM helpers (Ofpp usage, parse->numpy)
  - **Dataset/**         # Dataset classes, FVMN dataset, baseline
  - **Models/**          # Neural Operator implementations (FNO), FVMN models
  - *runner.py*       # Base hybrid trainer
  - *config.py*        # Configuration classes (OpenFOAM and experiment settings)
  - *plot_utils.py*    # Plotting and metrics loading helpers
- **random/**             # scripts and notebooks for visualization & analysis
- *Dockerfile*          # Reproducible image building recipe
- *setup.py*            # Python packaging metadata (install_requires lists deps)
- *LICENSE.md*         # MIT License

Development notes
-----------------
- Python version: Codebase declares python_requires='>=3.13' in setup.py — ensure your env matches.
- Tests: Add unit tests for dataset readers, normalization routines, and trainer checkpointing to improve CI coverage.
- CI/CD: Consider GitHub Actions for linting, unit tests, and building/pushing the Docker image. 
- Reproducibility: Save normalization stats (means/stds) with dataset creation. The BaseDataset includes a normalization routine to read/write those stats.
- Contribution: Please follow conventional commits, open PRs against main, and include a short description of experiments, config, and expected behavior.

Citation
--------
If you use RePIT-Framework in research, please cite the RePIT algorithm paper:
- J. Jeon et al., "Residual-based physics-informed transfer learning: A hybrid method for accelerating long-term CFD simulations via deep learning", https://arxiv.org/abs/2206.06817 

License
-------
This project is released under the MIT License. See LICENSE.md.

Contact
-------
- Author: NINELAB
- Repo: https://github.com/JBNU-NINE/repitframework