# Standard library imports
from typing import Any, Dict, Optional, Tuple, Union, List
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, average_precision_score
)
from sklearn.model_selection import train_test_split
import json
import pickle
import glob
import os
import sys
from pathlib import Path

# Add project root to Python path
try:
    project_root = Path(__file__).parent.parent
    if not project_root.exists():
        # Handle network path by using raw string
        project_root = Path(r"\\".join(str(project_root).split("\\")))
    sys.path.append(str(project_root))
    print(f"Project root: {project_root}")
except Exception as e:
    print(f"Error setting project root path: {e}")
    # Fallback to current directory if path resolution fails
    sys.path.append(os.getcwd().parent)
    print(f"Current directory: {os.getcwd().parent}")
    
os.environ['GIT_PYTHON_GIT_EXECUTABLE'] = "C:/Program Files/Git/bin/git.exe"

import mlflow
import mlflow.catboost
from datetime import datetime

# Local imports
from utils.logger import ExperimentLogger
from utils.create_evaluation_set import create_evaluation_sets_draws, import_training_data_draws_new, get_selected_columns_draws, setup_mlflow_tracking

class CatBoostModel(BaseEstimator, ClassifierMixin):
    """CatBoost model implementation with global training."""
    
    def __init__(
        self,
        logger: Optional[ExperimentLogger] = None,
        categorical_features: Optional[List[str]] = None
    ) -> None:
        """Initialize CatBoost model."""
        self.logger = logger or ExperimentLogger()
        self.model_type = 'catboost_binary_model'
        
        # Global training parameters
        self.global_params = {
            'learning_rate': 0.0030616221990601507,
            'depth': 7,
            'min_data_in_leaf': 108,
            'l2_leaf_reg': 3.422353711694752,
            'random_strength': 28.37562521442237,
            'bagging_temperature': 0.7154344866799414,
            'scale_pos_weight': 3.050390907170957,
            'n_estimators': 11599,
            'leaf_estimation_iterations': 8,
            'bootstrap_type': 'Bayesian',
            'grow_policy': 'SymmetricTree',
            'eval_metric': 'AUC',
            'early_stopping_rounds': 100,
            'verbose': False,
            'random_seed': 42,
            'task_type': 'CPU'
        }
        # Initialize other attributes
        self.model = None
        self.feature_importance = {}
        self.selected_features = get_selected_columns_draws()
        self.categorical_features = categorical_features or []
        self.scaler = None
        self.threshold = 0.5  # Default threshold for predictions

    def _validate_data(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame], Optional[pd.Series]]:
        """Validate and format input data."""
        # Convert to pandas
        X_train_df = pd.DataFrame(X_train) if isinstance(X_train, np.ndarray) else X_train
        y_train_s = pd.Series(y_train) if isinstance(y_train, np.ndarray) else y_train
        
        # Validate validation data if provided
        X_val_df = None
        y_val_s = None
        if X_val is not None and y_val is not None:
            X_val_df = pd.DataFrame(X_val) if isinstance(X_val, np.ndarray) else X_val
            y_val_s = pd.Series(y_val) if isinstance(y_val, np.ndarray) else y_val
            
            # Validate shapes and columns match
            if X_train_df.shape[1] != X_val_df.shape[1]:
                raise ValueError("Training and validation features must have same number of columns")
                
        return X_train_df, y_train_s, X_val_df, y_val_s

    def _optimize_threshold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray) -> float:
        """Find optimal prediction threshold using validation data."""
        best_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'threshold': 0.5}
        best_score = 0
        
        # Focus on higher thresholds for better precision
        for threshold in np.arange(0.3, 0.95, 0.01):
            y_pred = (y_prob >= threshold).astype(int)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            
            # Modified scoring to prioritize precision
            if precision >= 0.35:  # Higher minimum precision requirement
                score = precision * min(recall, 0.30)  # Lower recall cap
                if score > best_score:
                    best_score = score
                    best_metrics.update({
                        'precision': precision,
                        'recall': recall,
                        'f1': f1_score(y_true, y_pred, zero_division=0),
                        'threshold': threshold
                    })
        
        self.logger.info(f"Best threshold: {best_metrics['threshold']}")
        return best_metrics['threshold']

    def train(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None) -> None:
        """Train the model globally."""
        print("Starting training")
        
        X_train_selected = X_train[self.selected_features]
        X_val_selected = X_val[self.selected_features] if X_val is not None else None
        
        # Calculate draw rate for scale_pos_weight
        draw_rate = y_train.mean()
        safe_draw_rate = max(draw_rate, 0.001)
        self.global_params['scale_pos_weight'] = (1.0/safe_draw_rate) * (0.26/safe_draw_rate)
        
        # Initialize CatBoostClassifier
        self.model = CatBoostClassifier(**self.global_params)
        
        # Fit the model
        if X_val_selected is not None:
            self.model.fit(
                X_train_selected, 
                y_train, 
                eval_set=[(X_train_selected, y_train)],
                verbose=False
            )
        else:
            self.model.fit(X_train_selected, y_train)
        
        # Optimize threshold if validation data is provided
        if X_val_selected is not None:
            val_probs = self.model.predict_proba(X_val_selected)[:, 1]
            self.threshold = self._optimize_threshold(y_val, val_probs)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict probabilities using the global model."""
        X_df = pd.DataFrame(X)
        X_selected = X_df[self.selected_features]
        return self.model.predict_proba(X_selected)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict using the global threshold."""
        probas = self.predict_proba(X)[:, 1]
        return (probas >= self.threshold).astype(int)

    def predict_draw_probability(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Get draw probabilities."""
        X_df = pd.DataFrame(X)
        X_df = X_df[self.selected_features]
        return self.predict_proba(X_df)[:, 1]

    def analyze_predictions(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """Analyze model predictions."""
        probas = self.predict_proba(X)[:, 1]
        predictions = (probas >= self.threshold).astype(int)
        
        # Overall metrics
        analysis = {
            'metrics': {
                'precision': precision_score(y, predictions, zero_division=0),
                'recall': recall_score(y, predictions, zero_division=0),
                'f1': f1_score(y, predictions, zero_division=0)
            },
            'probability_stats': {
                'mean': float(np.mean(probas)),
                'std': float(np.std(probas)),
                'min': float(np.min(probas)),
                'max': float(np.max(probas))
            },
            'class_distribution': {
                'predictions': pd.Series(predictions).value_counts().to_dict(),
                'actual': pd.Series(y).value_counts().to_dict()
            },
            'draw_rate': float(y.mean()),
            'predicted_rate': float(predictions.mean()),
            'n_samples': len(y),
            'n_draws': int(y.sum()),
            'n_predicted': int(predictions.sum()),
            'n_correct': int(np.logical_and(y, predictions).sum())
        }
        
        return analysis

    def save(self, path: str) -> None:
        """Save the model and its metadata to disk."""
        try:
            # Save the model using CatBoost's API
            model_path = f"{path}_global.cbm"
            self.model.save_model(model_path)
            
            # Save metadata
            metadata = {
                'global_params': self.global_params,
                'threshold': self.threshold,
                'selected_features': self.selected_features
            }
            
            with open(f"{path}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error saving model: {str(e)}")
            raise

    def load(self, path: str) -> None:
        """Load the model and its metadata from disk."""
        try:
            # Load the model using CatBoost's API
            model_path = f"{path}_global.cbm"
            self.model = CatBoostClassifier()
            self.model.load_model(model_path)
            
            # Load metadata
            try:
                with open(f"{path}_metadata.json", 'r') as f:
                    metadata = json.load(f)
                    self.global_params = metadata.get('global_params', {})
                    self.threshold = metadata.get('threshold', 0.5)
                    self.selected_features = metadata.get('selected_features', [])
            except FileNotFoundError:
                if self.logger:
                    self.logger.warning("Model metadata not found")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading model: {str(e)}")
            raise

def train_with_mlflow():
    """Train CatBoost model with MLflow tracking."""
    logger = ExperimentLogger()
    experiment_name = "catboost_draw_prediction"
    artifact_path = setup_mlflow_tracking(experiment_name)
    # Set artifact location
    os.environ["MLFLOW_ARTIFACT_ROOT"] = artifact_path
    
    X_train, y_train, X_test, y_test = import_training_data_draws_new()
    X_val, y_val = create_evaluation_sets_draws()
    
    with mlflow.start_run(run_name=f"catboost_binary_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log dataset info
        mlflow.log_params({
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "n_features_original": X_train.shape[1],
            "draw_ratio_train": (y_train == 1).mean(),
            "draw_ratio_val": (y_val == 1).mean()
        })
        
        # Create and train model
        cat_model = CatBoostModel(logger=logger)
        
        # Log model parameters
        mlflow.log_params(cat_model.global_params)
        mlflow.log_param("selected_features", cat_model.selected_features)
        
        # Train the model
        cat_model.train(X_train, y_train, X_val, y_val)
        
        # Create input example
        input_example = X_train.iloc[:1].copy()
        
        # Create model signature
        signature = mlflow.models.infer_signature(
            input_example,
            cat_model.predict(input_example)
        )
        
        # Log the model
        mlflow.catboost.log_model(
            cat_model.model,
            "model_global",
            signature=signature,
            input_example=input_example
        )
        
        # Analyze and log metrics
        train_analysis = cat_model.analyze_predictions(X_train, y_train)
        val_analysis = cat_model.analyze_predictions(X_val, y_val)
        test_analysis = cat_model.analyze_predictions(X_test, y_test)
        
        logger.info(f"Train analysis: {train_analysis}")
        logger.info(f"Val analysis: {val_analysis}")
        logger.info(f"Test analysis: {test_analysis}")
        
        metrics_to_log = {
            "train_precision": train_analysis['metrics']['precision'],
            "train_recall": train_analysis['metrics']['recall'],
            "val_precision": val_analysis['metrics']['precision'],
            "val_recall": val_analysis['metrics']['recall'],
            "test_precision": test_analysis['metrics']['precision'],
            "test_recall": test_analysis['metrics']['recall']
        }
        
        mlflow.log_metrics(metrics_to_log)
        
        # Log analyses
        mlflow.log_dict(train_analysis, "train_analysis.json")
        mlflow.log_dict(val_analysis, "val_analysis.json")
        mlflow.log_dict(test_analysis, "test_analysis.json")
        

        logger.info(f"Training completed. MLflow run ID: {mlflow.active_run().info.run_id}")
        return cat_model

if __name__ == "__main__":
    model = train_with_mlflow() 