# Soccer Prediction Project Technical Documentation

## Overview
The Soccer Prediction Project is designed to predict soccer match draws and goal patterns using an ensemble of machine learning models. The system is optimized for CPU-only environments, ensuring high-precision model predictions with robust experiment tracking via MLflow.

## Development Environment
- **Operating System:** Windows 11
- **Python Version:** 3.9+
- **Hardware:** CPU-only (explicitly configured with tree_method='hist' for XGBoost and device='cpu' for all training tasks)
- **Environment Variables:**
  - `PYTHONHASHSEED=19` (for reproducibility)
  - `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=4` (for parallel computations)
  - `TF_ENABLE_ONEDNN_OPTS=0` (to disable oneDNN optimizations and ensure numerical consistency)

## Project Structure
The project is organized into clearly defined modules:

- **/models:**
  - Contains implementations of base machine learning models including those in `/models/StackedEnsemble` and `/models/ensemble`.
  - Current implementation in `ensemble_model_0324.py` integrates XGBoost, TabNet, and LightGBM as primary models with RandomForest as the default extra model.
  - Includes specialized modules for:
    - Dynamic weighting (`weights.py`)
    - Threshold optimization (`thresholds.py`)
    - MLflow integration and model loading

- **/utils:**
  - Provides utility functions for logging (`logger.py`), MLflow integration (`mlflow_utils.py`), and feature engineering (`advanced_goal_features.py`).
  - Supports data ingestion, preprocessing, and error monitoring.

- **/predictors:**
  - Contains the prediction service (`predict_ensemble.py`) which deploys the final ensemble model for real-time predictions.

## Key Technologies and Dependencies

- **Machine Learning Libraries:** 
  - XGBoost: Used with CPU-only settings (tree_method='hist', device='cpu')
  - LightGBM: Configured with binary objective and optimal hyperparameters
  - TabNet: Implemented via pytorch_tabnet.tab_model.TabNetClassifier 
  - scikit-learn: For metrics, preprocessing, and model compatibility

- **Experiment Tracking:** 
  - MLflow for tracking experiments, logging parameters, metrics, and model registration
  - Models are registered with timestamp-based naming (ensemble_YYYYMMDD_HHMM)

- **Data Processing:** 
  - Pandas and NumPy for data manipulation and vectorized operations
  - Feature selection and validation for each base model

- **Utilities:** 
  - Joblib for model serialization
  - Custom logging via `ExperimentLogger` in `/utils/logger.py`
  - Precision-focused weighting and threshold optimization

## Configuration and Environment Management

- **Virtual Environment:** Use a Python virtual environment; install dependencies using `pip install -r requirements.txt`.
- **Reproducibility:** Fixed seeds and controlled environment variables ensure consistency:
  ```python
  SEED = 19
  os.environ["PYTHONHASHSEED"] = str(SEED)
  random.seed(SEED)
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  torch.use_deterministic_algorithms(True)
  ```

## MLflow Integration

- Experiments are tracked with MLflow, with models registered using timestamp-based registry names.
- Parameters, metrics, and artifacts (including feature importance and analysis reports) are logged for reproducibility.
- Base models are loaded from MLflow with specific run IDs:
  ```python
  self.xgb_run_id = '30402608b8dc4c899d675e5b56c48c01'
  self.lgb_run_id = '8312e6c4f0184ed9afb56f87c10f45a0'
  self.tabnet_run_id = '46e86bfb663e4548a1a91360f9827de7'
  self.rf_run_id = 'cbfda1f197654fd2bdcb610a73cf8fad'
  ```

## Precision-Focused Weighting and Threshold Optimization

- **Dynamic Weighting:** The system calculates weights for each base model based on validation performance, focusing on precision.
- **Threshold Optimization:** Thresholds are tuned to achieve a target precision (≥50%) while maintaining minimum recall requirements (≥25%).
- **Vectorized Operations:** Performance-optimized functions use vectorized operations for efficiency.

## Testing and Reproducibility

- **Reproducible Results:** Fixed seeds and deterministic operations are enforced to maintain consistency across runs.
- **Testing:** Unit tests and integration tests are set up for critical components. Execute tests via `python -m pytest python_tests/` to ensure that all functionality performs as expected.

## Future Enhancements and Optimizations

- **Model Extensions:** Integration of additional base models and exploration of deeper neural network architectures.
- **GPU Support:** While currently optimized for CPU, future updates may incorporate GPU-based training.
- **Data Validation:** Continued improvements to data ingestion and anomaly detection mechanisms in the preprocessing pipeline.

## Model Execution

To run the ensemble model:

```python
python -m models.ensemble.run_ensemble --extra_model random_forest --meta_learner_type lgb --target_precision 0.5 --required_recall 0.25
```

This documentation serves as a comprehensive guide to the technical implementation of the Soccer Prediction Project.