# 🥅 Soccer Prediction Project v2.3

![GitHub License](https://img.shields.io/github/license/ronyka77/TheDrawCode)
![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)
![uv](https://img.shields.io/badge/pkg%20manager-uv-blue)
![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)
![CPU Optimized](https://img.shields.io/badge/CPU-optimized-brightgreen.svg)
![MLflow Tracking](https://img.shields.io/badge/MLflow-integrated-blue.svg)

A machine learning system that accurately predicts soccer match draws and goal patterns using ensemble methods and advanced feature engineering techniques, with a focus on high-precision results for betting applications.

## 📋 Table of Contents

- [Key Features](#-key-features)
- [Project Architecture](#-project-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Development Workflow](#-development-workflow)
- [Model Pipeline](#-model-pipeline)
- [Configuration](#-configuration)
- [Extending the Model](#-extending-the-model)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Key Features

- **Ensemble Model Architecture**: Combines XGBoost, TabNet, LightGBM, and Random Forest for robust predictions.
- **Precision-Focused Weighting**: Optimized weights based on each model's precision performance.
- **Vectorized Threshold Optimization**: Efficient threshold tuning for precision-recall balance.
- **CPU-Only Optimization**: Explicitly configured for deterministic CPU-based training.
- **Reproducible Results**: Comprehensive seed setting and environment variable control.
- **MLflow Integration**: Pre-trained model loading and versioned model registration.
- **Modern Tooling**: Uses `uv` for package management, `hatchling` for builds, `ruff` for linting/formatting, and `Makefile` for task automation.
- **`src` Layout**: Follows standard Python project structure.

## 🏗 Project Architecture

The system employs a multi-stage ensemble approach. Core Python code resides in the `src/` directory. Key components include:

- **Data Ingestion & Preprocessing** (`src/utils`, `data/`)
- **Feature Engineering** (`src/utils/advanced_goal_features.py`)
- **Base Model Training** (`src/models/StackedEnsemble/base/`)
- **Ensemble Logic** (`src/models/ensemble/`)
- **Prediction Service** (`src/predictors/`)
- **Backend API** (`src/backend/` - if applicable)
- **Documentation** (`docs/`, `mkdocs.yml`)
- **Development Tools** (`devtools/`, `Makefile`, `pyproject.toml`)

_(See `docs/architecture.md` and `docs/technical.md` for more details)_.

## 🚀 Installation

### Prerequisites

- Python >=3.11,<4.0
- Git
- `uv` package manager (See [uv installation](https://github.com/astral-sh/uv#installation))
- `make` (Optional, for using the Makefile. See [Makefile Setup Note](#makefile-setup-note))
- Windows 11 (Tested on)

### Setup

```bash
# Clone the repository (replace with your actual URL)
git clone https://github.com/ronyka77/TheDrawCode.git
cd TheDrawCode

# Create and activate virtual environment using uv
uv venv
# On Windows (cmd/powershell)
.venv\Scripts\activate
# On Linux/macOS
# source .venv/bin/activate

# Install dependencies using the Makefile (recommended)
make install
# OR install directly using uv
# uv sync --all-extras --dev

# Set up environment variables for reproducibility (Windows example)
# Add these to your environment activation or run them in your session
# $env:PYTHONHASHSEED=19
# $env:TF_ENABLE_ONEDNN_OPTS=0
# $env:OMP_NUM_THREADS=4
# $env:MKL_NUM_THREADS=4
# $env:OPENBLAS_NUM_THREADS=4
# $env:NUMEXPR_NUM_THREADS=4
# $env:VECLIB_MAXIMUM_THREADS=4
```

### Makefile Setup Note

The `Makefile` provides convenient shortcuts. If `make` is not installed on your system (common on Windows), you can either install it (e.g., via Chocolatey: `choco install make`) or run the corresponding commands from the `Makefile` directly (e.g., run `uv sync --all-extras --dev` instead of `make install`).

### Verification

Verify your installation by running the tests:

```bash
make test
# OR directly with uv
# uv run pytest
```

## 📊 Usage

### Basic Prediction Example

```python
# Ensure your virtual environment is activated
# Run scripts from the project root directory

# Example assumes prepare_data function exists and loads data appropriately
# Note the 'src.' prefix due to the src-layout

from src.models.ensemble.ensemble_model import EnsembleModel # Adjust filename if needed
from src.utils.logger import ExperimentLogger
import pandas as pd

# Initialize logger
logger = ExperimentLogger(experiment_name="soccer_prediction")

# Load dataset (replace with your data loading)
# data = pd.read_csv("path/to/matches.csv")
# X_train, y_train, X_test, y_test = prepare_data(data)

# # Initialize and train ensemble model (example)
# model = EnsembleModel(
#     logger=logger,
#     meta_learner_type='lgb',
#     dynamic_weighting=True,
#     target_precision=0.50,
#     required_recall=0.25,
#     extra_base_model_type='random_forest'
# )

# # Train the model (example)
# # results = model.train(X_train, y_train, X_test, y_test)

# # Make predictions (example)
# # predictions = model.predict(X_test)
# # probabilities = model.predict_proba(X_test)

# # print(f"Optimized threshold: {model.optimal_threshold}")
print("Usage example needs actual data loading and training steps.")
```

### Running Training via Script

```bash
# Run from project root
python -m src.models.ensemble.run_ensemble --extra_model random_forest --meta_learner_type lgb --target_precision 0.5 --required_recall 0.25
```

### Viewing Experiments

```bash
# Ensure MLflow server is configured or use local tracking
mlflow ui --port 5000 --backend-store-uri sqlite:///mlflow.db # Example using local SQLite
```

Navigate to `http://localhost:5000` in your browser.

## 🚀 Development Workflow

Use the `Makefile` for common tasks:

- `make install`: Install dependencies.
- `make lint`: Run `ruff` check and format.
- `make test`: Run `pytest` tests.
- `make clean`: Remove cache and build artifacts.
- `make build`: Build the package.

## 🧪 Model Pipeline

The system follows this workflow:

1.  **Data Preparation**: Feature engineering and validation.
2.  **Base Model Loading**: Loading pre-trained models from MLflow (from `src/models/ensemble/ensemble_model.py`).
    ```python
    # Example run IDs used by the system
    xgb_run_id = '30402608b8dc4c899d675e5b56c48c01'
    # ... other model run IDs ...
    ```
3.  **Dynamic Weighting**: Calculating weights (`src/models/ensemble/weights.py`).
4.  **Meta-Feature Creation**.
5.  **Meta-Learner Training**.
6.  **Threshold Optimization** (`src/models/ensemble/thresholds.py`).
7.  **Model Registration**: Registering the final model.

## ⚙️ Configuration

Configuration is managed via:

- **`pyproject.toml`**: Project metadata, dependencies, build settings, tool configurations (ruff, pytest).
- **Environment Variables**: For reproducibility and runtime settings (see Installation section).
- **Model Parameters**: Passed during `EnsembleModel` initialization or via `run_ensemble` script arguments.

_(See `docs/technical.md` for details on specific settings like reproducibility seeds and base model parameters)._

## 🧩 Extending the Model

### Adding New Base Models

1.  Implement the model in `src/models/StackedEnsemble/base/`.
2.  Add the model type to `extra_base_model_type` options in `src/models/ensemble/ensemble_model.py`.
3.  Update the `load_models_from_mlflow` method in `src/models/ensemble/ensemble_model.py`.
4.  Register a new MLflow run ID for your trained model.

## 🔧 Troubleshooting

### Common Issues

-   **Import Errors (`ModuleNotFoundError`)**: Ensure you run scripts from the project root directory (`TheDrawCode`) or have installed the package correctly (`make install` or `uv pip install -e .`). Verify the `src` layout is correct.
-   **`make` not found**: See [Makefile Setup Note](#makefile-setup-note).
-   **MLflow Model Loading Errors**: Check model registration and signatures in MLflow.
-   **TensorFlow Numerical Differences**: Ensure `TF_ENABLE_ONEDNN_OPTS=0` is set.
-   **Memory Issues**: Reduce batch sizes or feature counts.
-   **TabNet CPU Core Configuration**: Ensure threading environment variables are set and PyTorch threads are configured if issues persist.

_(See `docs/technical.md` for more details on specific configurations)._

## 👥 Contributing

Contributions are welcome!

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/your-feature`).
3.  Make your changes.
4.  Ensure code quality: Run `make lint` and `make test`.
5.  Commit your changes (`git commit -am 'Add some feature'`).
6.  Push to the branch (`git push origin feature/your-feature`).
7.  Open a Pull Request.

Please adhere to PEP 8, include docstrings/comments, add tests, and use type hints.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

*For more detailed documentation, run `mkdocs serve` and view the site locally, or refer to the files in the `docs/` directory.*