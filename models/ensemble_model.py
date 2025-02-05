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
from sklearn.metrics import classification_report, precision_score, recall_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from datetime import datetime
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.catboost
import mlflow.lightgbm
from mlflow.models.signature import infer_signature
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
import os
import sys
from imblearn.over_sampling import ADASYN
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import make_scorer


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

# Local imports
from utils.logger import ExperimentLogger
experiment_name = "stacked_ensemble_model"
logger = ExperimentLogger(experiment_name=experiment_name, log_dir='./logs/stacked_ensemble_model')


from utils.create_evaluation_set import create_ensemble_evaluation_set, import_selected_features_ensemble, import_training_data_ensemble, setup_mlflow_tracking
mlruns_dir = setup_mlflow_tracking(experiment_name)

def load_pretrained_model(run_id: str, model_type: str = "xgboost"):
    """Load the pretrained model from MLflow using run ID and model type.

    Supported model types: 'xgboost', 'catboost', and 'lightgbm'.
    """
    try:
        
        if model_type.lower() == "xgboost":
            model_uri = f"runs:/{run_id}/xgboost_api_model"
            return mlflow.xgboost.load_model(model_uri)
        elif model_type.lower() == "catboost":
            model_uri = f"runs:/{run_id}/catboost_model"
            return mlflow.catboost.load_model(model_uri)
        elif model_type.lower() == "lightgbm":
            model_uri = f"runs:/{run_id}/lightgbm_model"
            return mlflow.lightgbm.load_model(model_uri)
        else:
            logger.error(f"Unsupported model type: {model_type}")
            raise ValueError(f"Unsupported model type: {model_type}")
    except Exception as e:
        logger.error(f"Failed to load {model_type} model from MLflow run {run_id}: {str(e)}")
        raise

def build_base_models(calibrate: bool = False, calibration_method: str = "isotonic"):
    """
    Build ensemble base models using the pretrained XGBoost and new CatBoost and LightGBM models.
    """
    pretrained_xgb_run_id = "7a9b90e58dc74ca2a2b596fc0cdd1e04"
    pretrained_cat_run_id = "91ffd6888c39443eb76eba02a06f55bf"
    pretrained_lgbm_run_id = "506e82faa72a4c63912becda17b6c5f2"
    # Load pretrained XGBoost model
    xgb_model = load_pretrained_model(pretrained_xgb_run_id, model_type="xgboost")
    
    # Create new CatBoost model (with tuned hyperparameters if available)
    cat_model = load_pretrained_model(pretrained_cat_run_id, model_type="catboost")
    
    # Create new LightGBM model (with tuned hyperparameters if available)
    lgbm_model = load_pretrained_model(pretrained_lgbm_run_id, model_type="lightgbm")
    
    # Optional calibration for all models
    if calibrate:
        # Use the same calibration for consistency; note pretrained_xgb_model might already be calibrated
        xgb_model = CalibratedClassifierCV(xgb_model, method=calibration_method, cv=3)
        cat_model = CalibratedClassifierCV(cat_model, method=calibration_method, cv=3)
        lgbm_model = CalibratedClassifierCV(lgbm_model, method=calibration_method, cv=3)
    
    return xgb_model, cat_model, lgbm_model

class EnsembleModel:
    """
    Modified EnsembleModel that respects individual model feature requirements
    """
    def __init__(self, logger, voting_method="soft", weights=None, calibrate=False, calibration_method="isotonic"):
        """
        Initialize the EnsembleModel.
        """
        self.calibrate = calibrate
        self.calibration_method = calibration_method   
        self.voting_method = voting_method

        # Build base models
        xgb_model, cat_model, lgbm_model = build_base_models(calibrate=self.calibrate, calibration_method=self.calibration_method)
        self.base_models = [
            ("xgb", XGBClassifier(scale_pos_weight=3.5)),  # Higher class weight
            ("cat", CatBoostClassifier(auto_class_weights='Balanced')),
            ("lgbm", LGBMClassifier(class_weight={0:1, 1:3}))
        ]
        
        # Setup custom weights if provided; default to equal weighting
        self.weights = weights if weights is not None else [1.5, 1.8, 1.7]
        self.logger = logger

        # Store feature sets for each model
        self.feature_sets = {
            'xgb': xgb_features,
            'cat': cat_features,
            'lgbm': lgbm_features
        }

        # Add to EnsembleModel __init__:
        def recall_precision_balance(y_true, y_pred):
            recall = recall_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            return 0.7 * recall + 0.3 * precision  # Custom weighting

        custom_scorer = make_scorer(recall_precision_balance)

    def train(self, X_train, y_train, X_val, y_val):
        """Modified training that respects individual model features"""
        # Validate feature consistency
        self._validate_features(X_train)

        # Train each model on its specific features
        for name, model in self.base_models:
            model_features = self.feature_sets[name]
            X_train_subset = X_train[model_features]
            X_test_subset = X_test[model_features]
            
            # Check if model needs training (only for new models)
            if not hasattr(model, 'fit_'):
                model.fit(X_train_subset, y_train, eval_set=[(X_test_subset, y_test)])
                
            # Log model-specific features
            mlflow.log_params({
                f"{name}_features": str(model_features),
                f"{name}_feature_count": len(model_features)
            })

        # Add to train method:
        proba = self.predict_proba(X_val)
        precisions, recalls, thresholds = precision_recall_curve(y_val, proba)
        optimal_idx = np.argmax(precisions >= 0.5)  # First threshold reaching 50% precision
        optimal_threshold = thresholds[optimal_idx]
        mlflow.log_param("optimal_threshold", optimal_threshold)

        # Add to train method:
        # Convert probabilities to binary predictions using optimal threshold
        binary_preds = (proba >= optimal_threshold).astype(int)
        class1_precision = precision_score(y_val, binary_preds, labels=[1], zero_division=0)
        mlflow.log_metric("class1_precision", class1_precision)
        if class1_precision < 0.4:
            logger.warning("Class 1 precision below safety threshold")

        # Add to train method:
        class1_recall = recall_score(y_val, binary_preds, labels=[1], zero_division=0)
        mlflow.log_metric("class1_recall", class1_recall)
        if class1_recall < 0.15:
            logger.warning("Recall below minimum threshold")

        if precision_score(y_val, binary_preds) < 0.4:
            raise ValueError("Precision safety check failed")

    def predict(self, X):
        proba = self.predict_proba(X)
        
        # Primary threshold (precision focus)
        high_conf = proba > 0.65
        # Secondary threshold (recall focus)
        med_conf = (proba > 0.53) & (proba <= 0.65)
        
        predictions = np.zeros_like(proba)
        predictions[high_conf] = 1
        predictions[med_conf] = self._meta_predict(X[med_conf])  # Additional meta-model check
        
        return predictions
    
    def predict_proba(self, X):
        """Aggregate predictions using model-specific features"""
        predictions = []
        for (name, model), weight in zip(self.base_models, self.weights):
            model_features = self.feature_sets[name]
            X_subset = X[model_features]
            preds = model.predict_proba(X_subset)[:, 1]  # Get positive class probabilities
            predictions.append(preds * weight)
        return np.mean(predictions, axis=0)
    
    def _meta_predict(self, X_subset):
        # Use simple logistic regression check
        return (X_subset['draw_probability_score'] > 0.6).astype(int)

    def _validate_features(self, X):
        """Ensure all required features are present"""
        all_required = set()
        for features in self.feature_sets.values():
            all_required.update(features)
            
        missing = set(all_required) - set(X.columns)
        if missing:
            raise ValueError(f"Missing features in input data: {missing}")

if __name__ == "__main__":
    # Load data using your existing utility functions
    selected_features = import_selected_features_ensemble('all')
    xgb_features = selected_features
    cat_features = selected_features
    lgbm_features = selected_features

    # all_features = list(
    #     set(xgb_features + cat_features + lgbm_features)
    # )
    X_train, y_train, X_test, y_test = import_training_data_ensemble()
    X_val, y_val = create_ensemble_evaluation_set()
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    X_val = X_val[selected_features]
    
    # Initialize ensemble model 
    ensemble_model = EnsembleModel(
        logger=logger,
        weights=[1.5, 1.8, 1.7],  # Boost LightGBM (better recall potential)
        calibrate=True,
        calibration_method="sigmoid"
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