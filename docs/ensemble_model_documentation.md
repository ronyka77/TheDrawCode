# Ensemble Model Documentation

## Overview

This document provides a detailed overview of the ensemble model used in the Soccer Prediction Project. The ensemble model leverages multiple state-of-the-art base models combined through a stacking approach to achieve high-precision predictions (target precision ≥50%). It is optimized for CPU-only environments and integrates robust experiment tracking with MLflow.

## Architecture and Components

The ensemble model is primarily implemented in the following files:

- `models/ensemble/ensemble_model_0321.py`: Contains the core `EnsembleModel` class, which defines the ensemble architecture, base model configuration, and meta-feature generation.
- `models/ensemble/run_ensemble.py`: Orchestrates the end-to-end process of data loading, model training, evaluation, and MLflow integration for model tracking and registration.
- `models/ensemble/training.py`: Provides utility functions for training base models, initializing and hypertuning the meta-learner, and performing hyperparameter optimization using Optuna.

## Base Models

The ensemble model integrates the following primary base models:

- **XGBoost:** Utilizes `XGBClassifier` with CPU-only settings, configured with parameters optimized for precision.
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

Extra base model options include additional models such as **CatBoost**, **RandomForest**, **SVM**, and **MLP**. Particularly, CatBoost has been moved from the primary base models to the extra model options.

## Data Preparation and Feature Selection

- Features are selected using utility functions (e.g., `import_selected_features_ensemble`) to ensure that each model is trained on its optimal subset of features.
- Data is preprocessed and split into training, validation, and test sets, with care taken to avoid data leakage.

## Training Pipeline

The ensemble training pipeline involves:

1. **Data Preparation:** Preprocessing and feature selection to prepare data for training and evaluation.
2. **Base Model Training:** Independent training of primary base models (XGBoost, TabNet, LightGBM) on selected feature subsets.
3. **Probability Calibration (Optional):** Calibration of model probabilities to refine predictions.
4. **Dynamic Weighting and Meta-Feature Creation:** Combining base model predictions through stacking to create meta-features.
5. **Meta-Learner Training and Tuning:** Training a meta-learner on the generated meta-features with hyperparameter tuning via Optuna.
6. **Threshold Optimization:** Fine-tuning the decision threshold to balance precision and recall, ensuring the target precision is met.
7. **Evaluation and Deployment:** Final evaluation on validation data and deployment of the ensemble model through the prediction service.

## Running and Deployment

- **Execution**: The `run_ensemble.py` script manages the end-to-end training and evaluation process. It sets up MLflow tracking, loads the data, initiates model training, conducts error analysis, and registers the final model with a timestamp-based registry name.

- **MLflow Integration**: Detailed parameter logging, metric tracking, and artifact registration in MLflow ensure reproducibility and facilitate deployment.

## Hyperparameter Tuning and Training Utilities

The `training.py` module contains functions that:

- Initialize and train the base models and the meta-learner with early stopping and regularization techniques.
- Implement hyperparameter optimization for the meta-learner using Optuna, ensuring that the ensemble meets the desired performance criteria.
- Log detailed training metrics and model parameters for transparency and reproducibility.

## Additional Features

- **Model Explanation and Diagnostics:** Integrated functions for SHAP-based explanation and error analysis.
- **Precision Filtering:** Post-prediction filters are applied to boost prediction precision by eliminating low-confidence outputs.
- **Reproducibility:** Fixed seeds and controlled environment variables ensure consistent results across runs.

## Future Enhancements

- Incorporation of additional base models and advanced neural architectures.
- Extension of GPU support for faster training and inference in future releases.
- Further development of data validation and anomaly detection mechanisms in the preprocessing pipeline.

This documentation serves as a comprehensive guide to the ensemble model's architecture, training process, evaluation methods, and deployment strategy in the Soccer Prediction Project. ಠ_ಠ 