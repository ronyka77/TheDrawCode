# Ensemble Model Documentation

## Overview

This document provides a detailed overview of the ensemble model used in the Soccer Prediction Project. The ensemble model leverages multiple state-of-the-art base models combined through a stacking approach with dynamic weighting to achieve high-precision predictions (target precision ≥50%). It is optimized for CPU-only environments and integrates robust experiment tracking with MLflow.

## Architecture and Components

The ensemble model is primarily implemented in the following files:

- `models/ensemble/ensemble_model_0324.py`: Contains the core `EnsembleModel` class, which defines the ensemble architecture, base model configuration, and meta-feature generation.
- `models/ensemble/run_ensemble.py`: Orchestrates the end-to-end process of data loading, model training, evaluation, and MLflow integration for model tracking and registration.
- `models/ensemble/weights.py`: Implements dynamic weighting algorithms with precision-focused calculations.
- `models/ensemble/thresholds.py`: Provides optimized threshold tuning functions to balance precision and recall.

## Base Models

The ensemble model integrates the following primary base models:

- **XGBoost:** Utilizes `XGBClassifier` with CPU-only settings:
  ```python
  # Key environment variables for CPU-only operation
  os.environ["OMP_NUM_THREADS"] = "4"
  os.environ["MKL_NUM_THREADS"] = "4"
  os.environ["OPENBLAS_NUM_THREADS"] = "4"
  ```

- **TabNet:** Integrated using `TabNetClassifier` from the `pytorch_tabnet.tab_model` package. Key configuration parameters include:
    - learning_rate: 0.02196
    - n_d: 11
    - n_a: 16
    - n_steps: 9
    - gamma: 1.8
    - lambda_sparse: 2.48893e-05
    - momentum: 0.95
    - mask_type: 'entmax'

- **LightGBM:** Configured with `LGBMClassifier` using a binary objective and tuned hyperparameters for robust performance.

- **Extra Base Model:** Currently configured to use Random Forest as the default extra model, with options to use CatBoost, SVM, or MLP instead:
  ```python
  # Extra model loading
  self.rf_run_id = 'cbfda1f197654fd2bdcb610a73cf8fad'
  ```

## MLflow Integration

The ensemble model uses MLflow to load pre-trained base models and register the final model:

```python
# Load models from MLflow with specific run IDs
self.xgb_run_id = '30402608b8dc4c899d675e5b56c48c01'
self.lgb_run_id = '8312e6c4f0184ed9afb56f87c10f45a0'
self.tabnet_run_id = '46e86bfb663e4548a1a91360f9827de7'
self.rf_run_id = 'cbfda1f197654fd2bdcb610a73cf8fad'
```

The `load_models_from_mlflow` method loads models and extracts their feature signatures, ensuring consistent feature selection.

## Dynamic Weighting and Precision Focus

A key enhancement to the ensemble model is the implementation of precision-focused dynamic weighting:

```python
# Compute dynamic weights based on validation performance
self.dynamic_weights, self.thresholds = compute_precision_focused_weights_optimized(
    p_xgb, p_tabnet, p_lgb, p_extra, y_val, self.target_precision, self.min_recalls, self.logger
)
```

This approach:
1. Calculates optimal thresholds for each base model
2. Assigns weights proportional to each model's precision
3. Ensures minimum weights for each model (5%)
4. Normalizes the weights to sum to 1.0

## Threshold Optimization

The model implements vectorized threshold optimization to efficiently balance precision and recall:

```python
tune_threshold_for_precision_optimized(
    meta_val_probs, y_val, 
    target_precision=self.target_precision,
    required_recall=self.required_recall,
    logger=self.logger
)
```

This approach:
1. Scans potential thresholds using vectorized operations
2. Filters thresholds that meet minimum recall requirements
3. Selects the threshold that maximizes precision
4. Falls back to best F1 score if minimum recall cannot be achieved

## Meta-Learning Process

The ensemble model uses meta-features derived from base model predictions:

1. Base model predictions are combined with dynamic weights
2. A meta-learner (default: LightGBM) is trained on these meta-features
3. The meta-learner is hypertuned for optimal performance
4. A global threshold is applied to the meta-learner's predictions

## Model Registration and Deployment

The ensemble model is wrapped in a scikit-learn compatible wrapper and registered in MLflow with a timestamp-based name:

```python
# Register model with timestamp-based name
model_name = f"ensemble_{datetime.now().strftime('%Y%m%d_%H%M')}"

# Log model with signature
mlflow.sklearn.log_model(
    sk_model=model_wrapper,
    artifact_path="ensemble_model",
    signature=signature,
    registered_model_name=model_name,
    pip_requirements=["scikit-learn==1.6.1"]
)
```

## Additional Features

- **Reproducibility:** Comprehensive seed setting and deterministic operations:
  ```python
  SEED = 19
  os.environ["PYTHONHASHSEED"] = str(SEED)
  random.seed(SEED)
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  torch.use_deterministic_algorithms(True)
  ```

- **Precision Filtering:** Post-prediction filtering to boost precision:
  ```python
  def precision_filter(self, X, probabilities):
      high_conf = probabilities > self.optimal_threshold
      X_high_conf = X[high_conf]
      if 'home_form' in X_high_conf.columns and 'away_form' in X_high_conf.columns:
          form_diff = abs(X_high_conf['home_form'] - X_high_conf['away_form'])
          likely_not_draw = form_diff > 0.5
          high_conf[high_conf] = ~likely_not_draw
      return high_conf
  ```

- **Error Analysis:** Built-in methods for prediction explanation and error analysis.

## Future Enhancements

- Exploration of additional meta-learner architectures
- Integration of more sophisticated feature selection techniques
- Extension of dynamic weighting to consider model variance and complementarity
- Implementation of uncertainty quantification in predictions

This documentation serves as a comprehensive guide to the ensemble model's architecture, training process, evaluation methods, and deployment strategy in the Soccer Prediction Project. :-] 