# MLflow Guide

[← Back to Documentation Home](../README.md)

## Table of Contents
- [Overview](#overview)
- [Setup & Configuration](#setup--configuration)
- [Experiment Tracking](#experiment-tracking)
- [Model Management](#model-management)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Additional Resources](#additional-resources)
- [Precision-Recall Optimization Tracking](#precision-recall-optimization-tracking)

## Overview
This guide outlines how MLflow is used in the Soccer Prediction Project for experiment tracking, model management, and deployment. Our setup is optimized for CPU-only environments and focuses on reproducibility.

### Technical Decisions
- **Local Tracking**: File-based MLflow tracking for simplicity
- **CPU Optimization**: Configurations for non-GPU environments
- **Versioning**: Systematic model versioning with metadata
- **Artifact Management**: Organized storage of model artifacts

### Target Audience
- **ML Engineers**: Focus on experiment tracking and model management
- **Data Scientists**: Review experiment organization and logging
- **Contributors**: Understand basic MLflow operations

## Setup & Configuration

### 1. Environment Setup
```bash
# Initialize MLflow directory
mkdir -p mlruns
export MLFLOW_TRACKING_URI=file:///$(pwd)/mlruns
```

### 2. Configuration File
```yaml
# mlflow_config.yaml
tracking_server:
  uri: file:///$(pwd)/mlruns
  experiment_name: soccer_prediction
artifacts:
  location: ./artifacts
  retention: 30  # days
logging:
  level: INFO
```

## Experiment Tracking

### 1. Basic Experiment
```python
import mlflow

def train_model():
    mlflow.set_experiment("soccer_prediction")
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("learning_rate", 0.01)
        mlflow.log_param("max_depth", 5)
        
        # Train model
        model = train_xgboost_model(params)
        
        # Log metrics
        mlflow.log_metrics({
            "precision": 0.82,
            "recall": 0.76
        })
        
        # Log model
        mlflow.xgboost.log_model(model, "model")
```

### 2. Advanced Tracking
```python
def log_feature_importance(model, feature_names):
    """Log feature importance plot."""
    fig = plot_feature_importance(model, feature_names)
    mlflow.log_figure(fig, "feature_importance.png")
    
    # Log feature importance scores
    importance_dict = {
        name: score for name, score 
        in zip(feature_names, model.feature_importances_)
    }
    mlflow.log_params(importance_dict)
```

## Model Management

### 1. Model Registration
```python
def register_model(run_id: str, model_name: str):
    """Register model with versioning."""
    client = mlflow.tracking.MlflowClient()
    
    # Register model
    result = mlflow.register_model(
        f"runs:/{run_id}/model",
        model_name
    )
    
    # Add description and version tags
    client.update_registered_model(
        name=model_name,
        description="XGBoost model for draw prediction"
    )
    
    return result.version
```

### 2. Model Serving
```python
def load_production_model():
    """Load the current production model."""
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/draw_predictor/Production"
    )
    return model
```

## Best Practices

### 1. Experiment Organization
- Use consistent naming conventions
- Group related parameters
- Tag experiments by purpose
- Document experiment goals

### 2. Artifact Management
- Version data sources
- Log environment details
- Store evaluation plots
- Track feature importance

### 3. Model Versioning
- Use semantic versioning
- Document model changes
- Track dependencies
- Log validation results

## Troubleshooting

### Common Issues

1. **Tracking URI Issues**
```python
# Solution: Check tracking URI
import os
print(f"MLFLOW_TRACKING_URI: {os.getenv('MLFLOW_TRACKING_URI')}")
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
```

2. **Artifact Storage**
```python
# Solution: Verify artifact path
def verify_artifact_path():
    """Verify MLflow artifact storage."""
    artifact_path = mlflow.get_artifact_uri()
    if not os.path.exists(artifact_path):
        os.makedirs(artifact_path)
    return artifact_path
```

3. **Model Loading**
```python
# Solution: Safe model loading
def safe_load_model(model_uri: str):
    """Safely load MLflow model with error handling."""
    try:
        return mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None
```

## Common Pitfalls & FAQ

### 1. Environment Setup
Q: Why isn't MLflow tracking my experiments?
A: Common causes:
- MLFLOW_TRACKING_URI not set correctly
- Missing write permissions in mlruns directory
- Running from wrong working directory

Solution:
```bash
# Check environment and permissions
echo $MLFLOW_TRACKING_URI
ls -la mlruns/
pwd
```

### 2. Artifact Storage
Q: Why are my artifacts not being saved?
A: Common causes:
- Relative paths in artifact URI
- Disk space issues
- Network connectivity (for remote storage)

Solution:
```python
# Use absolute paths for artifacts
import os
mlflow.set_tracking_uri(f"file://{os.path.abspath('./mlruns')}")
```

### 3. Model Loading
Q: Why do I get "Model not found" errors?
A: Common causes:
- Model not registered in MLflow
- Wrong model name or version
- Missing model dependencies

Solution:
```python
# List available models
client = mlflow.tracking.MlflowClient()
for rm in client.list_registered_models():
    print(f"Model: {rm.name}")
    for mv in client.get_latest_versions(rm.name):
        print(f"  Version: {mv.version}")
```

## OS-Specific Instructions

### Windows 11 Setup
```bash
# Set MLflow tracking URI with Windows path
set MLFLOW_TRACKING_URI=file:///%cd%/mlruns

# Create directory with proper permissions
mkdir mlruns
icacls mlruns /grant Users:(OI)(CI)F

# Install required packages
pip install mlflow==2.8.0 # Latest stable version
```

### Path Configuration
```python
# Windows-specific path handling
def get_mlflow_path():
    """Get correct MLflow path for Windows."""
    base_path = os.path.abspath('./mlruns')
    # Convert Windows path to MLflow format
    return f"file:///{base_path.replace('\\', '/')}"
```

## Code Quality Integration

### Prospector Configuration
Our project uses Prospector for code quality checks. MLflow-related settings are in `.prospector.yaml`:

```yaml
# .prospector.yaml MLflow-specific settings
pylint:
  disable:
    - logging-fstring-interpolation  # For MLflow logging
    - broad-except  # For MLflow error handling
```

### Logging Standards
Following project standards from `utils/logger.py`:

```python
from utils.logger import ExperimentLogger
logger = ExperimentLogger()

def log_mlflow_event(event_type: str, details: Dict):
    """Log MLflow events using standard logger."""
    logger.info(f"MLflow {event_type}", extra={
        "component": "mlflow",
        "event": event_type,
        **details
    })
```

## Additional Resources

- [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html)
- [XGBoost with MLflow](https://www.mlflow.org/docs/latest/python_api/mlflow.xgboost.html)
- [Model Registry Guide](https://www.mlflow.org/docs/latest/model-registry.html)

## Related Documentation

### Core Documentation
- [Model Training Architecture](../architecture/model_training.md) - Training pipeline and model development
- [Prediction Service](../architecture/prediction.md) - Model serving and predictions
- [Data Pipeline](../architecture/data_pipeline.md) - Data processing and feature engineering

### Supporting Guides
- [Environment Setup](environment.md) - Development environment configuration
- [Troubleshooting Guide](troubleshooting.md) - Common issues and solutions

### Project Management
- [Changelog](../CHANGELOG.md) - Version history and updates

## Precision-Recall Optimization Tracking

### Feature Selection Metrics
```python
# Feature selection metrics structure
{
    'n_initial_features': int,
    'n_final_features': int,
    'feature_reduction_percent': float,
    'standard_precision': float,
    'standard_recall': float,
    'precision_focused_precision': float,
    'precision_focused_recall': float
}
```

### Threshold Optimization Metrics
```python
# Threshold optimization metrics
{
    'best_threshold': float,
    'best_precision': float,
    'best_recall': float,
    'best_f1': float,
    'pruned_trials_count': int
}
```

### Artifacts
- `feature_importance.csv`: Feature importance analysis
- `standard_features.txt`: Initially selected features
- `precision_focused_features.txt`: Final feature set
- `parameter_importance.html`: Parameter impact visualization

### Best Practices
1. Monitor recall threshold compliance
2. Track pruned trials ratio
3. Analyze precision-recall trade-offs
4. Review parameter importance regularly

---
[🔝 Back to Top](#mlflow-guide) 