# Soccer Prediction Project Technical Documentation

## Overview
The Soccer Prediction Project is designed to predict soccer match draws and goal patterns using an ensemble of machine learning models. The system is optimized for CPU-only environments, ensuring high-precision model predictions with robust experiment tracking via MLflow.

## Development Environment
- **Operating System:** Windows 11
- **Python Version:** >=3.9.19
- **Package Manager:** uv
- **Build Backend:** hatchling with uv-dynamic-versioning
- **Hardware:** CPU-only (explicitly configured with tree_method='hist' for XGBoost and device='cpu' for all training tasks)
- **Environment Variables:**
  - `PYTHONHASHSEED=19` (for reproducibility)
  - `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=4` (for parallel computations)
  - `TF_ENABLE_ONEDNN_OPTS=0` (to disable oneDNN optimizations and ensure numerical consistency)

## Project Structure
The project utilizes a standard `src` layout and includes the following key directories:

- **/src:** Contains the core Python package code.
  - **/models:** Implementations of base ML models (`/models/StackedEnsemble`) and the ensemble logic (`/models/ensemble`).
  - **/utils:** Utility functions for logging, MLflow, feature engineering, etc.
  - **/predictors:** Prediction service logic.
  - **/backend:** FastAPI backend code (if applicable).
- **/docs:** Project documentation files (Markdown format).
- **/devtools:** Helper scripts for development workflows (e.g., `lint.py`).
- **/data:** Data files (raw, processed, prediction).
- **/logs:** Application and experiment logs.
- **/tests:** (Optional) Unit and integration tests.
- **pyproject.toml:** Defines project metadata, dependencies, and tool configurations (build system, ruff, pytest, etc.).
- **Makefile:** Provides shortcuts for common development tasks (install, lint, test, clean, build).
- **mkdocs.yml:** Configuration file for the MkDocs documentation site.

## Core Technologies and Dependencies

- **Package Management:**
  - `uv`: Used for dependency management, installation, and virtual environments.
- **Build System:**
  - `hatchling`: Modern build backend used for creating distributable packages.
  - `uv-dynamic-versioning`: Determines package version dynamically from Git tags.
- **Linting & Formatting:**
  - `ruff`: Fast linter and formatter used for maintaining code quality.
  - `basedpyright`: (Optional, configured in pyproject.toml) Type checker.
- **Testing Framework:**
  - `pytest`: Used for running automated tests.
- **Documentation:**
  - `mkdocs`: Static site generator for project documentation.
  - `mkdocs-material`: (Optional, configured in dev dependencies) Theme for MkDocs.
- **Machine Learning Libraries:**
  - XGBoost, LightGBM, TabNet, scikit-learn (configured as previously).
- **Experiment Tracking:**
  - MLflow (configured as previously).
- **Data Processing:**
  - Pandas and NumPy.
- **Utilities:**
  - Joblib, Custom logging (`ExperimentLogger`).

## Configuration and Environment Management

- **Virtual Environment:** Managed using `uv`. Create/activate using standard `uv venv` commands.
- **Installation:** Install dependencies using the `Makefile` command `make install` (which runs `uv sync --all-extras --dev`) or directly with `uv sync` / `uv pip install -e ".[dev]"`.
- **Reproducibility:** Fixed seeds and controlled environment variables ensure consistency (settings remain the same).

## Development Workflow

Common tasks are automated via the `Makefile`:
- `make install`: Set up the environment and install dependencies.
- `make lint`: Run `ruff` checks and formatting (via `devtools/lint.py`).
- `make test`: Run automated tests using `pytest` (via `uv run pytest`).
- `make clean`: Remove build artifacts and cache directories.
- `make build`: Build the package distribution files.

## MLflow Integration

- Experiments are tracked with MLflow, with models registered using timestamp-based registry names.
- Parameters, metrics, and artifacts (including feature importance and analysis reports) are logged for reproducibility.
- Base models are loaded from MLflow with specific run IDs:
  ```python
  self.xgb_run_id = '30402608b8dc4c899d675e5b56c48c01'
  self.lgb_run_id = '8312e6c4f0184ed9afb56f87c10f45a0'
  self.tabnet_run_id = '46e86bfb663e4548a1a91360f9827de7'
  self.rf_run_id = 'cbfda1f197654fd2bdcb610a73cf8fad'
  ```

## Precision-Focused Weighting and Threshold Optimization

- **Dynamic Weighting:** The system calculates weights for each base model based on validation performance, focusing on precision.
- **Threshold Optimization:** Thresholds are tuned to achieve a target precision (≥50%) while maintaining minimum recall requirements (≥25%).
- **Vectorized Operations:** Performance-optimized functions use vectorized operations for efficiency.

## Testing and Reproducibility

- **Reproducible Results:** Fixed seeds and deterministic operations are enforced to maintain consistency across runs.
- **Testing:** Execute tests using `make test` or `uv run pytest`.

## Future Enhancements and Optimizations

- **Model Extensions:** Integration of additional base models and exploration of deeper neural network architectures.
- **GPU Support:** While currently optimized for CPU, future updates may incorporate GPU-based training.
- **Data Validation:** Continued improvements to data ingestion and anomaly detection mechanisms in the preprocessing pipeline.

## Model Execution

To run the ensemble model (note the path change due to `src` layout):

```bash
# Ensure you are in the project root directory (TheDrawCode)
python -m src.models.ensemble.run_ensemble --extra_model random_forest --meta_learner_type lgb --target_precision 0.5 --required_recall 0.25
```

This documentation serves as a comprehensive guide to the technical implementation of the Soccer Prediction Project.