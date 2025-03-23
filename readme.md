# 🥅 Soccer Prediction Project v2.2

![GitHub](https://img.shields.io/github/license/username/soccer-prediction)
![Python Version](https://img.shields.io/badge/python-3.9-blue.svg)
![CPU Optimized](https://img.shields.io/badge/CPU-optimized-brightgreen.svg)

A machine learning system that accurately predicts soccer match draws and goal patterns using ensemble methods and advanced feature engineering techniques, with a focus on high-precision results for betting applications.

## 📋 Table of Contents

- [Key Features](#-key-features)
- [Project Architecture](#-project-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Pipeline](#-model-pipeline)
- [Configuration](#-configuration)
- [Extending the Model](#-extending-the-model)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Key Features

- **Ensemble Model Architecture**: Combines XGBoost, TabNet, LightGBM, and optional extra models for robust predictions
- **High-Precision Focus**: Optimized for precision with customizable threshold tuning
- **CPU-Only Optimization**: Designed to run efficiently without GPU requirements
- **Reproducible Results**: Fixed seeds and deterministic operations for consistent outcomes
- **MLflow Integration**: Comprehensive experiment tracking with model versioning
- **Advanced Feature Engineering**: Soccer-specific feature development for improved accuracy

## 🏗 Project Architecture

![System Architecture](https://via.placeholder.com/800x400?text=Soccer+Prediction+Architecture)

The system employs a multi-stage ensemble approach with the following components:

- **Data Ingestion & Preprocessing**: Data loading, cleaning, and validation
- **Feature Engineering**: Soccer-specific feature extraction
- **Base Model Training**: Training of XGBoost, TabNet, and LightGBM models
- **Hyperparameter Tuning**: Optimization using Optuna with persistent storage
- **Ensemble Learning**: Stacking approach with dynamic weighting
- **Threshold Optimization**: Fine-tuning for high precision predictions
- **Prediction Service**: Deployment for real-time predictions

## 🚀 Installation

### Prerequisites

- Python 3.9+
- Windows 11
- MongoDB (for data storage)

### Setup

```bash
# Clone the repository
git clone https://github.com/username/soccer-prediction.git
cd soccer-prediction

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment for reproducibility
set PYTHONHASHSEED=19
set TF_ENABLE_ONEDNN_OPTS=0
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
set OPENBLAS_NUM_THREADS=4
```

### Verification

Verify your installation with the following command:

```bash
python -m python_tests.test_environment
```

## 📊 Usage

### Basic Prediction

```python
from models.ensemble.ensemble_model import EnsembleModel
from utils.logger import ExperimentLogger
import pandas as pd

# Initialize logger
logger = ExperimentLogger(experiment_name="soccer_prediction")

# Load dataset
data = pd.read_csv("path/to/matches.csv")
X_train, y_train, X_test, y_test = prepare_data(data)

# Initialize and train ensemble model
model = EnsembleModel(
    logger=logger,
    calibrate=True,
    meta_learner_type='xgb',
    target_precision=0.50,
    required_recall=0.25,
    X_train=X_train
)

# Train the model
results = model.train(X_train, y_train, X_test, y_test)

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

print(f"Optimized threshold: {model.optimal_threshold}")
```

### Running with MLflow Tracking

```python
from models.ensemble.run_ensemble import run_ensemble

# Run ensemble with MLflow tracking
model = run_ensemble(
    extra_base_model_type='random_forest',
    meta_learner_type='xgb',
    calibrate=True, 
    dynamic_weighting=True,
    target_precision=0.50,
    required_recall=0.25,
    experiment_name="ensemble_experiment"
)
```

### Viewing Experiments

```bash
# Start MLflow UI
mlflow ui --port 5000
```

Then navigate to `http://localhost:5000` in your browser.

## 🧪 Model Pipeline

The system follows this workflow:

1. **Data Preparation**: Feature engineering and validation
2. **Base Model Training**: Training primary models (XGBoost, TabNet, LightGBM)
3. **Probability Calibration**: Optional sigmoid/isotonic calibration
4. **Meta-Feature Creation**: Converting base model predictions to meta-features
5. **Meta-Learner Training**: Training a model to combine base predictions
6. **Threshold Optimization**: Fine-tuning prediction threshold for precision/recall balance

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TF_ENABLE_ONEDNN_OPTS` | Enable/disable TensorFlow oneDNN optimizations | `0` |
| `OMP_NUM_THREADS` | Number of OpenMP threads | `4` |
| `MKL_NUM_THREADS` | Number of MKL threads | `4` |
| `OPENBLAS_NUM_THREADS` | Number of OpenBLAS threads | `4` |
| `PYTHONHASHSEED` | Python hash seed for reproducibility | `19` |
| `NUMEXPR_NUM_THREADS` | Number of NumExpr threads | `4` |
| `VECLIB_MAXIMUM_THREADS` | Number of VecLib threads | `4` |

### Primary Base Models

The ensemble includes the following primary base models:

1. **XGBoost**: CPU-optimized with `tree_method='hist'` and `device='cpu'`
2. **TabNet**: Neural network configuration with parameters:
   - learning_rate: 0.02196
   - n_d: 11
   - n_a: 16
   - n_steps: 9
   - gamma: 1.8
   - lambda_sparse: 2.48893e-05
   - momentum: 0.95
   - mask_type: 'entmax'
3. **LightGBM**: Configured with binary objective and optimized parameters

### Extra Model Options

Additional models that can be used as extra base models:
- CatBoost (moved from primary to extra options)
- RandomForest
- SVM
- MLP

### Model Parameters

The ensemble model accepts the following parameters:

```python
EnsembleModel(
    logger=None,                    # Logger instance
    calibrate=False,                # Whether to calibrate probabilities
    calibration_method="sigmoid",   # Calibration method
    individual_thresholding=False,  # Use individual thresholds
    meta_learner_type="xgb",        # Meta-learner type
    dynamic_weighting=True,         # Use dynamic weighting
    extra_base_model_type="mlp",    # Extra model type
    sampling_strategy=0.7,          # Sampling strategy
    complexity_penalty=0.01,        # Complexity penalty
    target_precision=0.50,          # Target precision
    required_recall=0.25,           # Required recall
    X_train=None                    # Training features
)
```

## 🧩 Extending the Model

### Adding New Base Models

To add a new model type to the ensemble:

1. Implement the model in `models/StackedEnsemble/base/`
2. Add the model type to `extra_base_model_type` options
3. Update the initialization logic in `EnsembleModel.__init__()`

Example for adding a new model type:

```python
# In ensemble_model.py, within __init__ method
if self.extra_base_model_type == 'your_new_model':
    self.model_extra = YourNewModel(
        param1=value1,
        param2=value2,
        random_state=19
    )
    self.logger.info("Extra base model initialized as YourNewModel.")
```

## 🔧 Troubleshooting

### Common Issues

#### TensorFlow Numerical Differences

**Problem**: Different numerical results on different machines due to oneDNN optimizations.

**Solution**: Disable oneDNN optimizations by setting `TF_ENABLE_ONEDNN_OPTS=0` before importing TensorFlow:

```python
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
```

#### Memory Issues with Large Models

**Problem**: Out of memory errors when training models on large datasets.

**Solution**: Reduce batch size, limit feature count, or use chunked processing:

```python
# Reduce batch size for neural networks
model = EnsembleModel(
    extra_base_model_type='mlp',
    batch_size=32  # Smaller batch size
)
```

#### TabNet CPU Core Configuration

**Problem**: TabNet using more CPU cores than allocated.

**Solution**: Explicitly configure thread limits for TabNet:

```python
import os
import torch

# Set environment variables
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"

# Configure PyTorch threads
torch.set_num_threads(4)
torch.set_num_interop_threads(4)

# When initializing TabNet
tabnet_params = {
    'device_name': 'cpu',
    'num_workers': 4
}
```

## 👥 Contributing

We welcome contributions to improve the prediction system! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

Please ensure your code follows our coding standards:
- Pass all tests
- Follow PEP 8 guidelines
- Include proper documentation
- Use type hints
- Handle errors appropriately

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*For more detailed documentation, please refer to the [docs](docs/) directory.*

(づ｡◕‿‿◕｡)づ