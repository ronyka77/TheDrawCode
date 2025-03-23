# Soccer Prediction Project Technical Documentation

## Overview
The Soccer Prediction Project is designed to predict soccer match draws and goal patterns using an ensemble of machine learning models. The system is optimized for CPU-only environments, ensuring high-precision model predictions with robust experiment tracking via MLflow.

## Development Environment
- **Operating System:** Windows 11
- **Python Version:** 3.9+
- **Hardware:** CPU-only (configured with device='cpu' for all training tasks)

## Project Structure
The project is organized into clearly defined modules:

- **/models:**
  - Contains implementations of base machine learning models including those in `/models/StackedEnsemble` and `/models/ensemble`.
  - Models such as LightGBM, XGBoost, and other ensemble techniques are implemented here.

- **/utils:**
  - Provides utility functions for logging (`logger.py`), MLflow integration (`mlflow_utils.py`), and feature engineering (`advanced_goal_features.py`).
  - Supports data ingestion, preprocessing, and error monitoring.

- **/predictors:**
  - Contains the prediction service (`predict_ensemble.py`) which deploys the final ensemble model for real-time predictions.

## Key Technologies and Dependencies

- **Machine Learning Libraries:** LightGBM, XGBoost, scikit-learn, and Optuna for hyperparameter tuning.
- **Experiment Tracking:** MLflow for logging parameters, model metrics, and model registration.
- **Data Processing:** Pandas and NumPy for data manipulation.
- **Utilities:** Joblib for model serialization; custom logging via `ExperimentLogger` in `/utils/logger.py`.

## Configuration and Environment Management

- **Environment Variables:**
  - `PYTHONHASHSEED=19` (for reproducibility)
  - `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=4` (for parallel computations)
  - `TF_ENABLE_ONEDNN_OPTS=0` (to disable oneDNN optimizations and ensure numerical consistency)

- **Virtual Environment:** Use a Python virtual environment; install dependencies using `pip install -r requirements.txt`.

## MLflow Integration

- Experiments are tracked with MLflow, with models registered using timestamped registry names.
- Parameters, metrics, and artifacts (including feature importance and analysis reports) are logged for reproducibility.
- Launch the MLflow UI with: `mlflow ui --port 5000`.

## Testing and Reproducibility

- **Reproducible Results:** Fixed seeds and deterministic operations are enforced to maintain consistency across runs.
- **Testing:** Unit tests and integration tests are set up for critical components. Execute tests via `python -m pytest python_tests/` to ensure that all functionality performs as expected.

## Future Enhancements and Optimizations

- **Model Extensions:** Integration of additional base models and exploration of deeper neural network architectures.
- **GPU Support:** While currently optimized for CPU, future updates may incorporate GPU-based training.
- **Data Validation:** Continued improvements to data ingestion and anomaly detection mechanisms.

## Ensemble Model Implementation Details

The ensemble model is implemented in `models/ensemble/ensemble_model_0321.py` and integrates multiple base models as follows:

- **XGBoost:** Utilizes `XGBClassifier` with CPU-only settings (e.g., `tree_method='hist'`, `device='cpu'`, `nthread=4`) and parameters tuned for optimal precision.
- **TabNet:** Integrated using `TabNetClassifier` from the `pytorch_tabnet.tab_model` package. It is configured with the following key parameters:
    - learning_rate: 0.02196
    - n_d: 11
    - n_a: 16
    - n_steps: 9
    - gamma: 1.8
    - lambda_sparse: 2.48893e-05
    - momentum: 0.95
    - mask_type: 'entmax'
- **LightGBM:** Configured with `LGBMClassifier` using a binary objective with hyperparameters set for robust performance.

Extra base model options have been updated to include **CatBoost** (along with RandomForest, SVM, and MLP), which is now removed from the primary base model lineup.

These base models are trained on selected feature subsets independently. Their probability outputs on validation data are then used to create meta-features through a stacking approach. The ensemble combines these predictions by:

- **Dynamic Weighting:** Calculating model-specific weights based on validation precision, emphasizing the models with stronger performance.

- **Probability Calibration:** Optionally calibrating the outputs (using methods such as sigmoid calibration) to refine each model's probability estimates.

- **Threshold Tuning:** Determining an optimal decision threshold (via functions like `tune_threshold_for_precision`) to achieve a target precision (typically ≥50%), balancing precision and recall effectively.

This configuration is designed to drive improved precision, which is critical for reliable predictions in betting applications.