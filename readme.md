# 🥅 Soccer Prediction Project v2.3

![GitHub](https://img.shields.io/github/license/username/soccer-prediction)
![Python Version](https://img.shields.io/badge/python-3.9-blue.svg)
![CPU Optimized](https://img.shields.io/badge/CPU-optimized-brightgreen.svg)
![MLflow Tracking](https://img.shields.io/badge/MLflow-integrated-blue.svg)

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

- **Ensemble Model Architecture**: Combines XGBoost, TabNet, LightGBM, and Random Forest for robust predictions
- **Precision-Focused Weighting**: Optimized weights based on each model's precision performance
- **Vectorized Threshold Optimization**: Efficient threshold tuning for precision-recall balance
- **CPU-Only Optimization**: Explicitly configured for deterministic CPU-based training
- **Reproducible Results**: Comprehensive seed setting and environment variable control
- **MLflow Integration**: Pre-trained model loading and versioned model registration
- **Advanced Feature Engineering**: Soccer-specific feature development for improved accuracy

## 🏗 Project Architecture

![System Architecture](https://via.placeholder.com/800x400?text=Soccer+Prediction+Architecture)

The system employs a multi-stage ensemble approach with the following components:

- **Data Ingestion & Preprocessing**: Data loading, cleaning, and validation
- **Feature Engineering**: Soccer-specific feature extraction
- **Base Model Training**: Training of XGBoost, TabNet, LightGBM, and Random Forest models
- **Dynamic Weighting**: Precision-focused weighting for model contributions
- **Meta-Learning**: Stacking approach with optimized meta-learner
- **Threshold Optimization**: Vectorized optimization for high precision with minimum recall
- **Prediction Service**: Deployment with MLflow model registry integration

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
set NUMEXPR_NUM_THREADS=4
set VECLIB_MAXIMUM_THREADS=4
```

### Verification

Verify your installation with the following command:

```bash
python -m python_tests.test_environment
```

## 📊 Usage

### Basic Prediction

```python
from models.ensemble.ensemble_model_0324 import EnsembleModel
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
    meta_learner_type='lgb',
    dynamic_weighting=True,
    target_precision=0.50,
    required_recall=0.25,
    extra_base_model_type='random_forest'
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
    meta_learner_type='lgb',
    calibrate=False, 
    dynamic_weighting=True,
    target_precision=0.50,
    required_recall=0.25,
    experiment_name="ensemble_model_improved"
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
2. **Base Model Loading**: Loading pre-trained models from MLflow with specific run IDs:
   ```python
   # Example run IDs used by the system
   xgb_run_id = '30402608b8dc4c899d675e5b56c48c01'
   lgb_run_id = '8312e6c4f0184ed9afb56f87c10f45a0'
   tabnet_run_id = '46e86bfb663e4548a1a91360f9827de7'
   rf_run_id = 'cbfda1f197654fd2bdcb610a73cf8fad'
   ```
3. **Dynamic Weighting**: Calculating precision-focused weights using `compute_precision_focused_weights_optimized`
4. **Meta-Feature Creation**: Converting base model predictions to meta-features
5. **Meta-Learner Training**: Training a model to combine base predictions
6. **Threshold Optimization**: Vectorized threshold tuning for precision/recall balance using `tune_threshold_for_precision_optimized`
7. **Model Registration**: Registering the final model with timestamp-based naming (`ensemble_YYYYMMDD_HHMM`)

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

### Reproducibility Configuration

```python
# Reproducibility settings
SEED = 19
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

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
4. **Random Forest**: Used as the default extra model option

### Extra Model Options

Additional models that can be used as extra base models:
- CatBoost
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
    meta_learner_type="lgb",        # Meta-learner type (lgb, xgb)
    dynamic_weighting=True,         # Use dynamic weighting
    extra_base_model_type="random_forest", # Extra model type
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
3. Update the `load_models_from_mlflow` method in `ensemble_model_0324.py`
4. Register a new run ID for your model

Example for adding a new model type:

```python
# In ensemble_model_0324.py, add a new run_id
self.new_model_run_id = 'your_mlflow_run_id'

# Update load_models_from_mlflow method to load your new model
try:
    new_model_uri = f"runs:/{self.new_model_run_id}/{model_path}"
    self.model_extra = mlflow.sklearn.load_model(new_model_uri)
    # Extract feature signature
    # ...
except Exception as e:
    self.logger.error(f"Failed to load new model: {str(e)}")
    raise ValueError(f"Failed to load new model: {str(e)}")
```

## 🔧 Troubleshooting

### Common Issues

#### MLflow Model Loading Errors

**Problem**: Errors when loading models from MLflow with messages about missing signatures or features.

**Solution**: Ensure that models are properly registered with signatures in MLflow:

```python
# When saving a model to MLflow, include signature
signature = mlflow.models.infer_signature(
    input_example,
    model.predict(input_example)
)
mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    signature=signature
)
```

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