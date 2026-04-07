# Contributing to XRePIT / RePIT-Framework

Thank you for your interest in contributing! This document covers the workflow for reporting bugs, proposing features, and submitting pull requests.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Report a Bug](#how-to-report-a-bug)
- [How to Request a Feature](#how-to-request-a-feature)
- [Development Setup](#development-setup)
- [Coding Conventions](#coding-conventions)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Running Tests](#running-tests)
- [Citing the Paper](#citing-the-paper)

---

## Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/) Code of Conduct v2.1. Please be respectful and constructive in all interactions.

---

## How to Report a Bug

1. Search existing [GitHub Issues](https://github.com/POSTECH-NINE/repitframework/issues) to avoid duplicates.
2. Open a new issue with the label **bug** and include:
   - A minimal reproducible example.
   - The exact error message and full traceback.
   - Your environment: OS, Python version, PyTorch version, OpenFOAM version (if applicable).

---

## How to Request a Feature

1. Check whether a similar request already exists in the issues.
2. Open an issue with the label **enhancement** describing:
   - The motivation or use-case.
   - A sketch of the proposed API or behaviour.

---

## Development Setup

```bash
# 1. Fork and clone
git clone https://github.com/<your-username>/repitframework.git
cd repitframework

# 2. Create a feature branch
git checkout -b feat/my-feature

# 3. Create and activate the conda environment
conda env create -f environment.yml
conda activate repit_env

# 4. Install in editable mode
pip install -e .
```

---

## Coding Conventions

- **Style**: Follow [PEP 8](https://pep8.org/).  Use 4-space indentation (tabs in existing files are retained for consistency).
- **Type hints**: Add type hints to new functions and method signatures.
- **Docstrings**: NumPy-style docstrings for all public functions and classes.
- **No breaking changes**: Do not alter existing public APIs without a deprecation notice.
- **Tests**: Add or update tests under `tests/` for any new functionality.

---

## Submitting a Pull Request

1. Ensure your branch is up to date with `main`:
   ```bash
   git fetch origin
   git rebase origin/main
   ```
2. Run the test suite locally (see below) and make sure all tests pass.
3. Open a pull request against `main` with:
   - A clear title (e.g., `fix: handle missing prediction_metrics.ndjson gracefully`).
   - A description explaining **what** changed and **why**.
   - References to related issues (e.g., `Closes #42`).

We aim to review PRs within one week.

---

## Running Tests

```bash
# From the repository root
python -m pytest tests/ -v
```

The test suite covers:
- Dataset utilities (`tests/test_utils.py`)
- FVMN dataset construction (`tests/test_fvmn.py`)
- OpenFOAM parsing helpers (`tests/test_OpenFOAM.py`)

---

## Citing the Paper

If your contribution relates to the XRePIT method, please reference the paper in your PR description:

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
