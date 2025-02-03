"""
EnsembleModel Module

This module implements an ensemble model that boosts precision and recall 
using two ensemble strategies:
  1. Soft Voting with weighted probability averaging.
  2. Stacking with a meta-learner (default = Logistic Regression).

Optional calibration is provided to adjust overconfident predictions.
The ensemble uses three base models:
  - XGBoost (XGBClassifier)
  - CatBoost (CatBoostClassifier)
  - LightGBM (LGBMClassifier)
  
For imbalanced datasets, focal loss parameters can be passed to LightGBM.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from datetime import datetime
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
import os
import sys

# Add project root to Python path
try:
    project_root = Path(__file__).parent.parent
    if not project_root.exists():
        # Handle network path by using raw string
        project_root = Path(r"\\".join(str(project_root).split("\\")))
    sys.path.append(str(project_root))
    print(f"Project root ensemble_model: {project_root}")
except Exception as e:
    print(f"Error setting project root path: {e}")
    # Fallback to current directory if path resolution fails
    sys.path.append(os.getcwd().parent)
    print(f"Current directory ensemble_model: {os.getcwd().parent}")
    
os.environ['GIT_PYTHON_GIT_EXECUTABLE'] = "C:/Program Files/Git/bin/git.exe"

from utils.create_evaluation_set import import_training_data_draws_api, create_evaluation_sets_draws_api, setup_mlflow_tracking
from utils.logger import ExperimentLogger

logger = ExperimentLogger(
    experiment_name="soccer_prediction",
    log_dir="logs/soccer_prediction"
)
mlruns_dir = setup_mlflow_tracking('stacked_ensemble_model')

def build_base_models(calibrate: bool = False, calibration_method: str = "isotonic"):
    """
    Build and return the three base models (XGBoost, CatBoost, LightGBM).
    Optionally wraps each model in CalibratedClassifierCV.
    """
    # Create XGBoost model with CPU-optimized parameters
    xgb_model = XGBClassifier(tree_method='hist', use_label_encoder=False, eval_metric='logloss')
    
    # Create CatBoost model with silent mode
    cat_model = CatBoostClassifier(verbose=0)
    
    # Create LightGBM model.
    # For instance, if using focal loss, parameters like 'alpha' and 'gamma' could be set.
    lgbm_model = LGBMClassifier()
    
    if calibrate:
        logger.info("Calibrating base models with %s method", calibration_method)
        xgb_model = CalibratedClassifierCV(xgb_model, method=calibration_method, cv=3)
        cat_model = CalibratedClassifierCV(cat_model, method=calibration_method, cv=3)
        lgbm_model = CalibratedClassifierCV(lgbm_model, method=calibration_method, cv=3)
    
    return xgb_model, cat_model, lgbm_model

class EnsembleModel:
    """
    Ensemble Model using VotingClassifier for draw predictions.
    
    This model trains an ensemble classifier and logs training details and metrics
    using the standardized ExperimentLogger and MLflow.
    """
    def __init__(self, voting_method="soft", base_models=None, weights=None, calibrate=True, calibration_method="isotonic"):
        """
        Initialize the EnsembleModel.

        Args:
            base_models (list): List of tuples (name, model).
                Defaults to LogisticRegression and RandomForestClassifier.
            weights (list): Optional weights for each base model.
            voting_method (str): Voting method: "soft" or "hard".
            calibrate (bool): If True, each base model will be calibrated using CalibratedClassifierCV.
            calibration_method (str): Calibration method: "isotonic" or "sigmoid" (default "isotonic").
        """
        
        
        # Ensure self.weights is set with a default value if not provided
        self.weights = weights if weights is not None else [1.0] * len(self.base_models)
        
        self.logger = ExperimentLogger(
            experiment_name="soccer_prediction",
            log_dir="logs/soccer_prediction"
        )
        self.calibrate = calibrate
        self.calibration_method = calibration_method   
        self.voting_method = voting_method
        if base_models is None:
            # Use the base models from the build_base_models function
            xgb_model, cat_model, lgbm_model = build_base_models(calibrate=self.calibrate, calibration_method=self.calibration_method)
            base_models = [
                ("xgb", xgb_model),
                ("cat", cat_model),
                ("lgbm", lgbm_model)
            ]
        self.base_models = base_models
        self.model = VotingClassifier(
            estimators=self.base_models,
            voting=self.voting_method,
            weights=self.weights,
            n_jobs=-1
        )
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
        """
        Train the ensemble model, log parameters, metrics, and register model with MLflow.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation labels.
        
        Returns:
            self: Trained model instance.
        """
        # Start logging for training run
        
        # Log parameters for reproducibility
        params = {
            "model": "VotingClassifier",
            "voting": self.voting_method,
            "weights": self.weights,
            "base_models": [name for name, _ in self.base_models]
        }
        self.logger.log_params(params)
       
        # Log parameters to MLflow
        mlflow.log_params(params)
        y_train = y_train.astype(int)
        
        # Train the ensemble model
        self.model.fit(X_train, y_train)
        
        # Evaluate on validation set
        predictions = self.model.predict(X_val)
        accuracy = accuracy_score(y_val, predictions)
        precision = precision_score(y_val, predictions, zero_division=0)
        recall = recall_score(y_val, predictions, zero_division=0)
        f1 = f1_score(y_val, predictions, zero_division=0)
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        
        # Log metrics via ExperimentLogger and MLflow
        self.logger.log_metrics(metrics)
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Prepare model signature for MLflow logging using an input example
        input_example = X_train.head(1)
        signature = infer_signature(input_example, self.model.predict(input_example))
        
        # Register and log the model with MLflow
        registered_model_name = f"ensemble_{datetime.now().strftime('%Y%m%d_%H%M')}"
        mlflow.sklearn.log_model(
            model=self.model,
            artifact_path="ensemble_model",
            registered_model_name=registered_model_name,
            signature=signature
        )
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels for the given input data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input feature data.
            
        Returns:
        --------
        np.ndarray:
            Predicted class labels.
        """
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities for the given input data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input feature data.
            
        Returns:
        --------
        np.ndarray:
            Predicted class probability estimates.
        """
        return self.model.predict_proba(X)
    
if __name__ == "__main__":
    
    
    # Generate synthetic binary classification data
    X_train, y_train, X_test, y_test = import_training_data_draws_api()
    X_val, y_val = create_evaluation_sets_draws_api()
    
    # Initialize ensemble model using soft voting strategy with custom weights
    ensemble_model = EnsembleModel(
        voting_method="soft",
        weights=[1.2, 1.0, 1.3],
        calibrate=True,  # Enable probability calibration
        calibration_method="isotonic"
    )
     # Set MLflow experiment and start run
    with mlflow.start_run(run_name="ensemble_training") as run:
        # Fit the ensemble model
        ensemble_model.train(X_train, y_train, X_test, y_test)
        
        # Make predictions and evaluate
        predictions = ensemble_model.predict(pd.DataFrame(X_val))
        proba = ensemble_model.predict_proba(pd.DataFrame(X_val))
        
        print("Classification Report:")
        print(classification_report(y_val, predictions)) 
        mlflow.end_run()