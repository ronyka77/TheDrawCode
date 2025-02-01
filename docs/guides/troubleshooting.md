# Troubleshooting Guide

[← Back to Documentation Home](../README.md)

## Table of Contents
- [Overview](#overview)
- [Common Issues](#common-issues)
  - [Environment Setup](#1-environment-setup)
  - [Data Pipeline](#2-data-pipeline)
  - [Model Training](#3-model-training)
  - [MLflow](#4-mlflow)
  - [Prediction Service](#5-prediction-service)
- [Debugging Tools](#debugging-tools)
- [Additional Resources](#additional-resources)

## Overview
This guide provides solutions for common issues encountered in the Soccer Prediction Project. It covers environment setup, data pipeline, model training, MLflow integration, and prediction service problems.

### How to Use This Guide
1. Identify the category of your issue
2. Check the common solutions
3. Use the debugging tools if needed
4. Follow the escalation process if unresolved

## Common Issues

### 1. Environment Setup

#### Conda Environment Issues
```bash
# Error: Package conflicts
# Solution: Clean environment creation
conda create -n soccer_pred python=3.8 --no-default-packages
conda activate soccer_pred
conda install --file requirements.txt
```

#### Package Installation Problems
```bash
# Error: XGBoost installation fails
# Solution: Install CPU-only version
pip install xgboost --no-deps
pip install -r requirements.txt
```

#### Path Configuration
```python
# Error: Module not found
# Solution: Add project root to PYTHONPATH
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

### 2. Data Pipeline

#### API Connection Issues
```python
# Error: API rate limiting
def fetch_with_retry(endpoint: str, max_retries: int = 3):
    """Fetch data with retry logic."""
    for attempt in range(max_retries):
        try:
            response = requests.get(endpoint)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
```

#### Data Quality Problems
```python
# Error: Missing or invalid data
def validate_dataset(df: pd.DataFrame) -> bool:
    """Validate dataset quality."""
    checks = {
        "missing_values": df.isnull().sum().sum() == 0,
        "sample_size": len(df) >= 1000,
        "feature_completeness": all(col in df.columns for col in required_columns)
    }
    return all(checks.values())
```

#### Feature Engineering Errors
```python
# Error: Feature calculation errors
def safe_feature_calculation(df: pd.DataFrame) -> pd.DataFrame:
    """Safely calculate features with error handling."""
    try:
        df['goal_diff'] = df['home_goals'] - df['away_goals']
        df['form'] = df['points'].rolling(window=5, min_periods=1).mean()
        return df
    except Exception as e:
        logger.error(f"Feature calculation error: {e}")
        return df
```

### 3. Model Training

#### Memory Issues
```python
# Error: Memory overflow
def batch_training(X: np.ndarray, y: np.ndarray, batch_size: int = 1000):
    """Train model in batches."""
    model = XGBClassifier(tree_method='hist')  # CPU-optimized
    for i in range(0, len(X), batch_size):
        batch_X = X[i:i+batch_size]
        batch_y = y[i:i+batch_size]
        model.fit(batch_X, batch_y, xgb_model=model if i > 0 else None)
    return model
```

#### Convergence Problems
```python
# Error: Model not converging
def optimize_learning_rate(X: np.ndarray, y: np.ndarray):
    """Find optimal learning rate."""
    learning_rates = [0.01, 0.05, 0.1]
    best_score = float('-inf')
    best_lr = None
    
    for lr in learning_rates:
        model = XGBClassifier(learning_rate=lr)
        scores = cross_val_score(model, X, y, cv=5)
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_lr = lr
    
    return best_lr
```

#### Validation Errors
```python
# Error: Poor model performance
def diagnose_model_performance(model, X_val, y_val):
    """Diagnose model performance issues."""
    pred = model.predict(X_val)
    metrics = {
        'precision': precision_score(y_val, pred),
        'recall': recall_score(y_val, pred),
        'f1': f1_score(y_val, pred)
    }
    
    if metrics['precision'] < 0.3:
        logger.warning("Low precision - check feature importance")
    if metrics['recall'] < 0.2:
        logger.warning("Low recall - check class balance")
        
    return metrics
```

### 4. MLflow

#### Tracking Issues
```python
# Error: MLflow tracking URI not found
def verify_mlflow_setup():
    """Verify MLflow configuration."""
    try:
        tracking_uri = mlflow.get_tracking_uri()
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("soccer_prediction")
        return True
    except Exception as e:
        logger.error(f"MLflow setup error: {e}")
        return False
```

#### Artifact Storage Problems
```python
# Error: Artifact storage issues
def cleanup_artifacts(experiment_id: str, min_age_days: int = 30):
    """Clean up old artifacts."""
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs([experiment_id])
    
    for run in runs:
        run_age = (datetime.now() - run.info.start_time).days
        if run_age > min_age_days:
            client.delete_run(run.info.run_id)
```

### 5. Prediction Service

#### Model Loading Issues
```python
# Error: Model loading fails
def load_fallback_model():
    """Load fallback model if primary fails."""
    try:
        return mlflow.pyfunc.load_model("models:/draw_predictor/Production")
    except Exception:
        logger.warning("Loading fallback model")
        return mlflow.pyfunc.load_model("models:/draw_predictor/Staging")
```

#### Prediction Errors
```python
# Error: Prediction pipeline failures
def safe_predict(model, X: pd.DataFrame) -> np.ndarray:
    """Make predictions with error handling."""
    try:
        predictions = model.predict(X)
        if len(predictions) != len(X):
            raise ValueError("Prediction length mismatch")
        return predictions
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return np.zeros(len(X))  # Safe fallback
```

## Debugging Tools

### 1. Logging Configuration
```python
def setup_debug_logging():
    """Configure detailed logging for debugging."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('debug.log'),
            logging.StreamHandler()
        ]
    )
```

### 2. Model Inspection
```python
def inspect_model_state(model):
    """Inspect model state and parameters."""
    return {
        'params': model.get_params(),
        'feature_importance': model.feature_importances_,
        'n_features': model.n_features_in_,
        'classes': model.classes_
    }
```

### 3. Data Validation
```python
def validate_prediction_input(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate prediction input data."""
    checks = [
        (len(df) > 0, "Empty dataframe"),
        (all(col in df.columns for col in required_columns), "Missing columns"),
        (df.isnull().sum().sum() == 0, "Contains null values")
    ]
    
    passed = all(check[0] for check in checks)
    message = "; ".join(msg for passed, msg in checks if not passed)
    return passed, message
```

## Additional Resources

- [Environment Setup Guide](environment.md)
- [MLflow Guide](mlflow.md)
- [Data Pipeline Documentation](../architecture/data_pipeline.md)

---
[🔝 Back to Top](#troubleshooting-guide)

## Related Documentation
- [Model Training](../architecture/model_training.md)
- [Prediction Service](../architecture/prediction.md)
- [Data Pipeline](../architecture/data_pipeline.md) 