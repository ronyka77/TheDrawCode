# Ensemble Model Documentation

## Overview

The `ensemble_model.py` module implements an ensemble learning system tailored for the Soccer Prediction Project. It leverages multiple base models to boost precision and recall through ensemble strategies. The module supports:

- Loading pre-trained models from MLflow using run IDs.
- Building base classifiers with a fallback to training new models if pre-trained versions are unavailable.
- Optional calibration of model predictions using `CalibratedClassifierCV`.
- Dynamic threshold selection for converting prediction probabilities into binary outcomes.
- Dynamic adjustment of ensemble voting weights based on validation performance.
- An enhanced stacking strategy with an XGBoost-based meta-learner.
- Detailed logging using the `ExperimentLogger` for training insights and debugging.

## Key Features

- **Multiple Base Models:** Utilizes an ensemble comprising XGBoost, CatBoost, LightGBM, RandomForest, KNN, and SVM.
- **Fallback Mechanism:** Attempts to load pre-trained models from MLflow; if unavailable, new models are trained using parameters defined in the `ModelTrainingFeatures` class.
- **Model Calibration:** Offers calibration (using methods such as isotonic or sigmoid) to adjust overconfident predictions.
- **Threshold Optimization:** Implements dynamic threshold selection by maximizing the F1 score on a validation set.
- **Dynamic Ensemble Voting:** Adjusts voting weights based on each model's precision on validation data, ensuring that better-performing models contribute more strongly to predictions.
- **Enhanced Meta-Learner:** Replaces a simple logistic regression meta-learner with an XGBoost-based model to capture nonlinear relationships among base model predictions.
- **Safety Threshold Checks:** Enforces minimum precision and recall safety thresholds and logs warnings (or raises errors) when not met.
- **Debugging Insights:** Prints key debug information (e.g., the number of features used by XGB and SVM models) to facilitate troubleshooting.

## Module Structure

### 1. Pre-trained Model Loading

- **Function:** `load_pretrained_model(run_id: str, model_type: str = "xgboost")`
  
  Loads a model from MLflow using the provided run ID and model type. Supported model types include 'xgboost', 'catboost', 'lightgbm', 'random_forest', 'knn', and 'svm'.

### 2. Building Base Models

- **Function:** `build_base_models(selected_features, calibrate: bool = False, calibration_method: str = "isotonic")`

  Iterates over model types and attempts to load pre-trained models. If loading fails, new models are created using training parameters from `ModelTrainingFeatures`. Notably, the CatBoost training now uses a verbose level of 100 to provide detailed logs.

### 3. ModelTrainingFeatures Class

This class manages training parameters and feature sets for the ensemble models:

- **Attributes:**
  - `feature_sets`: Default features loaded via `import_selected_features_ensemble` from a JSON file.
  - `training_params`: Hyperparameters for each base model (XGBoost, CatBoost, LightGBM, etc.).

- **Key Methods:**
  - `get_features(model_type: str)`: Retrieves the feature set for the specified model.
  - `get_training_params(model_type: str)`: Retrieves the training parameters for the specified model.
  - Methods to update and save features and training parameters.

### 4. EnsembleModel Class

The `EnsembleModel` class manages the ensemble workflow:

- **Constructor Parameters:**
  - `logger`: Logging instance (`ExperimentLogger`) for tracking training events.
  - `selected_features`: Feature set used for model training and inference.
  - `voting_method`: Ensemble voting scheme, defaulting to "soft".
  - `weights`: Initial voting weights for each base model.
  - `calibrate`: Flag to enable calibration.
  - `calibration_method`: Method for calibration, such as "isotonic" or "sigmoid".

- **Methods:**

  - `train(X_train, y_train, X_val, y_val)`:
    - Ensures all training and validation data include the complete set of selected features.
    - Trains each base model; implements subsampling for SVM and KNN when training data is large.
    - Uses dynamic threshold optimization by selecting the threshold that maximizes the F1 score.
    - Dynamically adjusts ensemble voting weights based on each model's precision on the validation set.
    - Trains an XGBoost-based meta-learner on meta-features (the base model probabilities) for stacking.
    - Enforces safety checks on precision and recall thresholds, logging warnings or raising errors where necessary.
    - Prints feature counts for the XGB and SVM models (expected to be 99 and 50, respectively) to assist with debugging.

  - `predict(X)`:
    - Converts predicted probabilities to binary predictions using the dynamically determined optimal threshold.
    - Ensures a minimum ratio of positive predictions for balanced output.

  - `predict_proba(X)`:
    - Computes a weighted average of prediction probabilities from all base models, using dynamically updated weights.
    - Aligns input features to match each model's expectation.

  - `_meta_predict(X_subset)`:
    - Generates meta-features from the base models and uses the trained XGBoost meta-learner to produce final predictions.

  - `_validate_features(X)`:
    - Checks that the input data contains all required features and that the dimensions match the expected counts.

### 5. Usage Example

```python
if __name__ == "__main__":
    # Load feature configuration and training data using utility functions
    selected_features = import_selected_features_ensemble('all')
    X_train, y_train, X_test, y_test = import_training_data_ensemble()
    X_val, y_val = create_ensemble_evaluation_set()

    # Align data to selected features
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    X_val = X_val[selected_features]

    # Initialize the ensemble model with custom weights and calibration
    ensemble_model = EnsembleModel(
        logger=logger,
        selected_features=selected_features,
        weights={'xgb': 1.5, 'cat': 1.8, 'lgbm': 1.7, 'rf': 1.2, 'knn': 0.9, 'svm': 1.1},
        calibrate=True,
        calibration_method="sigmoid"
    )

    # Train and evaluate within an MLflow run
    with mlflow.start_run(run_name="ensemble_training") as run:
        ensemble_model.train(X_train, y_train, X_test, y_test)
        predictions = ensemble_model.predict(pd.DataFrame(X_val))
        print("Classification Report:")
        print(classification_report(y_val, predictions))
```

## MLflow Integration

- **Parameter Logging:** Critical training parameters, including the optimal threshold and dynamic voting weights, are logged.
- **Metric Logging:** Both precision and recall metrics are tracked, with safety thresholds enforced.
- **Model Tracking:** The system tracks model artifacts and the details of each training run for reproducibility.

## Logging & Error Handling

- Uses `ExperimentLogger` to record structured logs (JSON format, ISO 8601 timestamps) and to issue warnings when safety thresholds are not met.
- Raises errors if the aggregated precision falls below the set threshold.

## Calibration & Weighting

- Optionally applies calibration to each base model using `CalibratedClassifierCV`.
- Dynamically adjusts voting weights based on model-specific precision on validation data to improve overall ensemble performance.

## Conclusion

The `ensemble_model.py` module offers a robust ensemble learning framework that integrates dynamic threshold optimization, adaptive vote weighting, and a powerful XGBoost-based meta-learner. These enhancements, combined with detailed logging and MLflow integration, aim to improve prediction precision and recall for soccer predictions, while ensuring transparency and reproducibility throughout the training process. 