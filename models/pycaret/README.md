# Soccer Prediction with PyCaret

This module implements a complete soccer prediction pipeline using PyCaret, with a focus on predicting draws in soccer matches.

## Architecture

The implementation follows this workflow:

```
┌────────────────┐    ┌────────────────┐    ┌───────────────┐
│  data_module   │───→│feature_engineering│──→│model_training │
└────────────────┘    └────────────────┘    └───────────────┘
                                                    │
        ┌─────────────────────────────────────┬────┼────┐
        ▼                                     ▼         ▼
┌────────────────┐                  ┌────────────────┐  ┌────────────────┐
│   calibration  │←─────────────────│ threshold_utils│  │mlflow_module   │
└────────────────┘                  └────────────────┘  └────────────────┘
        │                                   │
        └───────────────────┬──────────────┘
                            ▼
                  ┌────────────────┐
                  │confidence_filtering│
                  └────────────────┘
```

## Components

- **data_module.py**: Handles data loading and preprocessing
- **feature_engineering.py**: Implements feature engineering techniques
- **model_training.py**: Core module for training and evaluating models
- **calibration.py**: Implements probability calibration techniques
- **threshold_utils.py**: Optimizes prediction thresholds
- **confidence_filtering.py**: Filters predictions based on confidence
- **mlflow_module.py**: Handles experiment tracking with MLflow
- **main.py**: Orchestrates the entire workflow
- **predict.py**: Implements batch prediction functionality
- **run_experiments.py**: Runs multiple experiments with different configurations

## Usage

### Training a Model

```bash
python models/pycaret/main.py --experiment-name "soccer_prediction_v1" --target-precision 0.45
```

Options:
- `--target-precision`: Target precision to achieve (default: 0.40)
- `--min-recall`: Minimum recall to maintain (default: 0.25)
- `--experiment-name`: MLflow experiment name (default: "soccer_prediction")
- `--skip-feature-engineering`: Skip feature engineering step
- `--skip-calibration`: Skip probability calibration step
- `--skip-confidence-filtering`: Skip confidence filtering step
- `--model-type`: Type of model to train ("single" or "ensemble", default: "ensemble")
- `--include-models`: Comma-separated list of models to include (default: "xgboost,lightgbm,rf,et,catboost")
- `--n-iter`: Number of iterations for hyperparameter tuning (default: 50)
- `--output-dir`: Directory to save models and results (default: "models/saved")

### Making Predictions

```bash
python models/pycaret/predict.py --model-path "models/saved/ensemble_20230215_1245" --data-path "data/new_matches.csv"
```

Options:
- `--model-path`: Path to the saved model (required)
- `--data-path`: Path to the data for prediction (required)
- `--output-path`: Path to save predictions (default: auto-generated)
- `--threshold`: Prediction threshold (default: 0.5)
- `--confidence-threshold`: Confidence threshold for filtering (default: None)
- `--apply-feature-engineering`: Apply feature engineering to input data
- `--metadata-path`: Path to model metadata (default: auto-detected)

### Running Experiments

```bash
python models/pycaret/run_experiments.py --config-path "models/pycaret/configs/experiments.yaml"
```

Options:
- `--config-path`: Path to experiment configuration file (default: "configs/experiments.yaml")
- `--output-dir`: Directory to save experiment results (default: "experiment_results")
- `--parallel`: Run experiments in parallel
- `--max-workers`: Maximum number of parallel workers (default: 4)

## Experiment Configuration

The experiment configuration file (`configs/experiments.yaml`) defines a grid of parameters to explore:

```yaml
# Base configuration (common to all experiments)
base_config:
  experiment_name: "soccer_prediction_grid"
  min_recall: 0.25
  # ...

# Grid configuration (parameters to vary)
grid_config:
  target_precision:
    - 0.35
    - 0.40
    - 0.45
    - 0.50
  # ...
```

## Requirements

- Python 3.8+
- PyCaret 3.3.2+
- scikit-learn
- pandas
- numpy
- mlflow
- PyYAML

## Development

To contribute to this module:

1. Ensure all functions have proper docstrings
2. Add appropriate error handling
3. Follow the existing architecture
4. Add tests for new functionality
5. Update this README with any new components or usage instructions 