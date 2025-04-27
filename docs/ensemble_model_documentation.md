# Ensemble Model Documentation

## Overview

This document provides a detailed overview of the ensemble model used in the Soccer Prediction Project. The ensemble model leverages multiple state-of-the-art base models combined through a stacking approach with dynamic weighting to achieve high-precision predictions (target precision ≥50%). It is optimized for CPU-only environments and integrates robust experiment tracking with MLflow.

## Architecture and Components

The ensemble model is primarily implemented in the following files:

- `src/models/ensemble/ensemble_model_0414.py`: Contains the core `EnsembleModel` class, which defines the ensemble architecture, base model configuration, and meta-feature generation. (Note: filename might differ, e.g., `ensemble_model.py`)
- `src/models/ensemble/run_ensemble.py`: Orchestrates the end-to-end process of data loading, model training, evaluation, and MLflow integration for model tracking and registration.
- `src/models/ensemble/weights_0414.py`: Implements dynamic weighting algorithms with precision-focused calculations.
- `src/models/ensemble/thresholds.py`: Provides optimized threshold tuning functions to balance precision and recall.

## Base Models

The ensemble model integrates the following primary base models:

| Model           | Library/Type         | Description/Role                                      |
|-----------------|---------------------|-------------------------------------------------------|
| XGBoost         | XGBClassifier        | Gradient boosting, strong tabular performance         |
| LightGBM        | LGBMClassifier       | Fast, efficient gradient boosting                     |
| TabNet          | TabNetClassifier     | Deep learning for tabular data                        |
| RandomForest    | Extra Trees/Sklearn  | Bagging-based ensemble, robust to overfitting         |
| MLP             | Sklearn MLP          | Shallow neural network for tabular data               |
| PyTorch         | Custom/PyTorch       | Deep neural network, hypertuned                       |
| SVM             | Sklearn SVM          | Kernel-based, good for complex boundaries             |
| Specialized FNN | Custom/PyTorch       | Domain-optimized FNN for soccer prediction            |

All models are loaded from MLflow using specific run IDs and their associated feature signatures and scalers.

## MLflow Integration

The ensemble model uses MLflow to load pre-trained base models and register the final model:

```python
# Example: Load models from MLflow with specific run IDs
self.xgb_run_id = 'f731a0b52acb4803869eab6039b7d621'
self.lgb_run_id = 'be439e143bd04b768309ca1f4e03199d'
self.tabnet_run_id = '8f17f9bc76384ac1be363ab16764f899'
self.extra_run_id = '2f6dbe0a4b844febae0ab2a601c656cd'
self.mlp_run_id = 'a99c793397414cb98cf2bc1ac7a5246d'
self.pytorch_run_id = '639bafb274bb459dadc6fe9eb46c5d35'
self.svm_run_id = '3c4ad60c660a42139edc79bd26fece65'
self.fnn_run_id = 'e77bcbaf413f47039e33acfeb21f105e'
```

The `load_models_from_mlflow` method loads models, scalers, and extracts their feature signatures, ensuring consistent feature selection and preprocessing.

## Dynamic Weighting and Precision Focus

A key enhancement to the ensemble model is the implementation of precision-focused dynamic weighting:

```python
# Compute dynamic weights based on validation performance
self.dynamic_weights, self.thresholds = compute_precision_focused_weights_optimized(
    p_xgb, p_tabnet, p_lgb, p_extra, p_mlp, p_pytorch, p_svm, p_fnn, y_val, self.target_precision, self.min_recalls, self.logger
)
```

This approach:
1. Calculates optimal thresholds for each base model
2. Assigns weights proportional to each model's precision
3. Ensures minimum weights for each model
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
2. A meta-learner (default: XGBoost/LightGBM) is trained on these meta-features
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

- **Specialized FNN Integration:** The ensemble now includes a domain-optimized feed-forward neural network (FNN) as a base model, leveraging soccer-specific feature processing, feature interaction layers, and calibrated probability outputs. See `README_specialized_fnn.md` for details.

## Future Enhancements

- Exploration of additional meta-learner architectures
- Integration of more sophisticated feature selection techniques
- Extension of dynamic weighting to consider model variance and complementarity
- Implementation of uncertainty quantification in predictions

This documentation serves as a comprehensive guide to the ensemble model's architecture, training process, evaluation methods, and deployment strategy in the Soccer Prediction Project. :-] 