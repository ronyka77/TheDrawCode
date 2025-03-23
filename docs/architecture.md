# Soccer Prediction Project Architecture

This document outlines the system architecture for the Soccer Prediction Project. The project aims to accurately predict soccer match draws and goal patterns using an ensemble of machine learning models, optimized for high precision. The system is designed for CPU-only environments and leverages MLflow for experiment tracking and hyperparameter optimization.

## Architecture Diagram

```mermaid
graph TD;
    A[Data Ingestion & Preprocessing] --> B[Feature Engineering];
    B --> C[Base Model Training];
    C --> D[Hyperparameter Tuning & Optimization];
    D --> E[Ensemble Learning];
    E --> F[Threshold Optimization];
    F --> G[Prediction Service];
    subgraph Utilities
      H[MLflow Tracking & Logging]
      I[Dynamic Sampling & Feature Selection]
    end
    B --> I;
    D --> H;
    G --> H;
```

## System Components

- **Data Ingestion & Preprocessing:**  
  Data is loaded, cleaned, and validated using utilities in the `/utils` folder. This stage prepares the dataset for training by ensuring quality and proper formatting.

- **Feature Engineering:**  
  Custom feature engineering is implemented in  `/utils/advanced_goal_features.py` to extract soccer-specific insights.

- **Base Model Training:**  
  Base models, including **XGBoost**, **TabNet**, and **LightGBM**, are implemented in the `/models/StackedEnsemble` and `/models/ensemble` directories. The ensemble pipeline trains these as core models. **TabNet** is integrated via `pytorch_tabnet.tab_model.TabNetClassifier` with parameters tuned for high precision (learning_rate=0.02196, n_d=11, n_a=16, n_steps=9, gamma=1.8, lambda_sparse=2.48893e-05, momentum=0.95, mask_type='entmax').
  
- **Extra Model Options:**  
  Additional models such as **CatBoost**, **RandomForest**, **SVM**, and **MLP** are available as extra model options. CatBoost has been moved from the primary base models to the extra options.

- **Hyperparameter Tuning & Optimization:**  
  Hyperparameter tuning is performed using Optuna with persistent storage (e.g., SQLite via `optuna_lightgbm.db`).

- **Ensemble Learning:**  
  The ensemble model integrates multiple base models by combining their predictions through a stacking approach. The meta-learner dynamically weights and calibrates each model's output, and threshold tuning refines the final prediction decision boundary.

- **Threshold Optimization:**  
  Post-training threshold tuning is performed on the ensemble's consolidated output to fine-tune the decision boundary, ensuring reliable predictions tailored for betting applications.

- **Prediction Service:**  
  The final ensemble model is deployed via the prediction service found in `/predictors/predict_ensemble.py`, which serves prediction requests in real-time.

- **Utilities:**  
  MLflow is used for experiment tracking, model registration, and logging, with additional utilities for dynamic feature selection and error monitoring.

## Data Flow & Process Overview
1. The system ingests raw soccer match data.
2. Features are engineered and selected based on domain knowledge.
3. Multiple base models are trained and tuned.
4. An ensemble method combines the predictions from these models.
5. Optimization of decision thresholds is carried out to meet target precision and recall.
6. Predictions are served via a dedicated prediction service.
7. Detailed logs and metrics are tracked using MLflow for reproducibility and analysis.

## Future Enhancements
- Integration of additional base models and deeper neural network architectures.
- Extended support for GPU-based training in future releases.
- Enhanced data validation and anomaly detection in the preprocessing stage.

By following this architecture, the Soccer Prediction Project ensures high-precision predictions with robust error handling and experiment tracking, making it a valuable tool for soccer analytics and betting applications.
