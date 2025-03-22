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
  - Provides utility functions for logging (`logger.py`), MLflow integration (`mlflow_utils.py`), dynamic sampling (`dynamic_sampler.py`), and feature engineering (`feature_selection.py`, `advanced_goal_features.py`).
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

By leveraging these technical strategies, the Soccer Prediction Project delivers robust, high-quality predictions crucial for effective soccer analytics in betting environments. ಠ_ಠ
