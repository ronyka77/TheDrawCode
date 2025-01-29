"""
XGBoost model implementation for binary classification with global training capabilities.
Provides methods for model training, prediction, and analysis with configurable parameters.
"""

# Standard library imports
import glob
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, List

# Third-party imports
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    log_loss, average_precision_score
)
from sklearn.model_selection import train_test_split
import mlflow

global project_root
# Add project root to Python path
try:
    project_root = Path(__file__).parent.parent
    if not project_root.exists():
        # Handle network path by using raw string
        project_root = Path(r"\\".join(str(project_root).split("\\")))
    sys.path.append(str(project_root))
    
    print(f"Project root xgboost_model: {project_root}")
except Exception as e:
    print(f"Error setting project root path: {e}")
    # Fallback to current directory if path resolution fails
    sys.path.append(os.getcwd().parent)
    print(f"Current directory xgboost_model: {os.getcwd().parent}")
    
os.environ['GIT_PYTHON_GIT_EXECUTABLE'] = "C:/Program Files/Git/bin/git.exe"

# Local imports
from utils.logger import ExperimentLogger
from utils.create_evaluation_set import get_selected_api_columns_draws, create_evaluation_sets_draws_api, import_training_data_draws_api, setup_mlflow_tracking

experiment_name = "xgboost_api_model"
mlruns_dir = setup_mlflow_tracking(experiment_name)

class XGBoostModel(BaseEstimator, ClassifierMixin):
    """XGBoost model implementation with global training."""
    
    def __init__(
        self,
        logger: Optional[ExperimentLogger] = None,
        categorical_features: Optional[List[str]] = None) -> None:
        """Initialize XGBoost model."""
        self.logger = logger or ExperimentLogger()
        self.categorical_features = categorical_features or []
        
        # Global training parameters
        # Start of Selection
        self.global_params = {
            'colsample_bytree': 0.6326244049159165,
            'gamma': 2.277462973881582,
            'learning_rate': 0.012239960389414113,
            'min_child_weight': 250,
            'n_estimators': 19605,
            'reg_alpha': 0.07956634443021414,
            'reg_lambda': 0.3392475206143923,
            'scale_pos_weight': 1.4113957295480144,
            'subsample': 0.5399522463762816,
            'objective': 'binary:logistic',
            'tree_method': 'hist',
            'eval_metric': ['error', 'auc', 'aucpr'],
            'verbosity': 0,
            'early_stopping_rounds': 500,
            'random_state': 42
        }
        
        # Initialize other attributes
        self.model = None
        self.feature_importance = {}
        self.selected_features = get_selected_api_columns_draws()
        self.threshold = 0.55  # Default threshold for predictions

    def _validate_data(
        self,
        x_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        x_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame], Optional[pd.Series]]:
        """Validate and format input data."""
        # Convert to pandas
        x_train_df = pd.DataFrame(x_train) if isinstance(x_train, np.ndarray) else x_train.copy()
        y_train_s = pd.Series(y_train) if isinstance(y_train, np.ndarray) else y_train.copy()
        
        # Convert data types for training data
        for col in x_train_df.columns:
            try:
                if x_train_df[col].dtype == 'object':
                    x_train_df[col] = pd.to_numeric(x_train_df[col], errors='coerce')
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Could not convert column {col}: {str(e)}")
                # Drop columns that can't be converted
                x_train_df = x_train_df.drop(columns=[col])
        
        # Handle validation data if provided
        x_val_df = None
        y_val_s = None
        if x_val is not None and y_val is not None:
            x_val_df = pd.DataFrame(x_val) if isinstance(x_val, np.ndarray) else x_val.copy()
            y_val_s = pd.Series(y_val) if isinstance(y_val, np.ndarray) else y_val.copy()
            
            # Convert data types for validation data
            for col in x_val_df.columns:
                try:
                    if x_val_df[col].dtype == 'object':
                        x_val_df[col] = pd.to_numeric(x_val_df[col], errors='coerce')
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Could not convert column {col}: {str(e)}")
                    # Drop columns that can't be converted
                    x_val_df = x_val_df.drop(columns=[col])
            
            # Validate shapes and columns match
            if x_train_df.shape[1] != x_val_df.shape[1]:
                raise ValueError("Training and validation features must have same number of columns")
        
        # Fill any remaining NaN values
        x_train_df = x_train_df.fillna(0)
        if x_val_df is not None:
            x_val_df = x_val_df.fillna(0)
        
        return x_train_df, y_train_s, x_val_df, y_val_s

    def _log_completion_metrics(
        self,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series]) -> None:
        """Log completion metrics."""
        if X_val is not None and y_val is not None:
            analysis = self.analyze_predictions(X_val, y_val)
            
            # Log overall metrics
            if self.logger:
                self.logger.info(f"best_iteration: {self.model.best_iteration}")
                self.logger.info(f"best_score: {self.model.best_score}")
                self.logger.info(f"n_features: {len(self.feature_importance)}")
                self.logger.info(f"val_accuracy: {analysis['metrics']['accuracy']}")
                self.logger.info(f"val_precision: {analysis['metrics']['precision']}")
                self.logger.info(f"val_recall: {analysis['metrics']['recall']}")
                self.logger.info(f"val_f1: {analysis['metrics']['f1']}")

    def _optimize_threshold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray) -> float:
        """Find optimal prediction threshold using validation data."""
        best_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'threshold': 0.55}
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
        features_train: Union[pd.DataFrame, np.ndarray],
        target_train: Union[pd.Series, np.ndarray],
        features_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        target_test: Optional[Union[pd.Series, np.ndarray]] = None,
        features_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        target_val: Optional[Union[pd.Series, np.ndarray]] = None) -> None:
        """Train the model globally."""
        
        features_train_selected = features_train[self.selected_features]
        features_val_selected = features_val[self.selected_features] if features_val is not None else None
        features_test_selected = features_test[self.selected_features] if features_test is not None else None
        
        # Initialize XGBClassifier with aligned params from hypertuning
        self.model = xgb.XGBClassifier(**self.global_params)
        
        # Fit the model
        if features_test_selected is not None:
            self.model.fit(
                features_train_selected, 
                target_train, 
                eval_set=[(features_test_selected, target_test)],
                verbose=False
            )
        else:
            self.model.fit(features_train_selected, target_train)
        
        # Optimize threshold if validation data is provided
        if features_val_selected is not None:
            val_probs = self.model.predict_proba(features_val_selected)[:, 1]
            self.threshold = self._optimize_threshold(target_val, val_probs)

    def predict_proba(self, features: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict probabilities using the global model."""
        features_df = pd.DataFrame(features)
        
        # Apply feature selection
        features_selected = features_df[self.selected_features]
        
        # Use predict_proba directly since we're using XGBClassifier
        probas = self.model.predict_proba(features_selected)
        
        return probas

    def predict(self, features: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict using the global threshold."""
        probas = self.predict_proba(features)[:, 1]
        return (probas >= self.threshold).astype(int)

    def analyze_predictions(
        self,
        features: Union[pd.DataFrame, np.ndarray],
        target: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """Analyze model predictions."""
        probas = self.predict_proba(features)[:, 1]
        predictions = (probas >= self.threshold).astype(int)
        
        # Overall metrics
        analysis = {
            'metrics': {
                'precision': precision_score(target, predictions, zero_division=0),
                'recall': recall_score(target, predictions, zero_division=0),
                'f1': f1_score(target, predictions, zero_division=0)
            },
            'probability_stats': {
                'mean': float(np.mean(probas)),
                'std': float(np.std(probas)),
                'min': float(np.min(probas)),
                'max': float(np.max(probas))
            },
            'class_distribution': {
                'predictions': pd.Series(predictions).value_counts().to_dict(),
                'actual': pd.Series(target).value_counts().to_dict()
            },
            'draw_rate': float(target.mean()),
            'predicted_rate': float(predictions.mean()),
            'n_samples': len(target),
            'n_draws': int(target.sum()),
            'n_predicted': int(predictions.sum()),
            'n_correct': int(np.logical_and(target, predictions).sum())
        }
        
        return analysis

    def save(self, path: str) -> None:
        """Save the model and its metadata to disk."""
        try:
            # Save the model using sklearn's API
            model_path = f"{path}_global.model"
            self.model.save_model(model_path)
            
            # Save metadata
            metadata = {
                'global_params': self.global_params,
                'threshold': self.threshold,
                'selected_features': self.selected_features
            }
            
            with open(f"{path}_metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error saving model: {str(e)}")
            raise

    def load(self, path: str) -> None:
        """Load the model and its metadata from disk."""
        try:
            # Load the model using sklearn's API
            model_path = f"{path}_global.json"
            self.model = xgb.XGBClassifier()
            self.model.load_model(model_path)
            
            # Load metadata
            try:
                with open(f"{path}_metadata.json", 'r', encoding='utf-8') as f:
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

def log_feature_importance(feature_importance):
    global project_root
    """Log feature importance values."""
    os.makedirs(f"{project_root}/feature_importance", exist_ok=True)
    
    # Sort features by importance
    sorted_features = sorted(
        feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Save top 20 features to a text file
    with open(f"{project_root}/feature_importance/top_60_features.txt", 'w', encoding='utf-8') as f:
        for feature, importance in sorted_features[:60]:
            f.write(f"{feature}: {importance:.6f}\n")
    
    # Save all feature importance values as JSON
    with open(f"{project_root}/feature_importance/feature_importance.json", 'w', encoding='utf-8') as f:
        json.dump(feature_importance, f, indent=2)

def convert_int_columns(df):
    """Convert integer columns to float64 for MLflow compatibility."""
    for col in df.select_dtypes(include=['int']).columns:
        df[col] = df[col].astype('float64')
    return df

def train_with_mlflow():
    """Train XGBoost model with MLflow tracking."""
    logger = ExperimentLogger(experiment_name="xgboost_api_model", log_dir='./logs/xgboost_model')
    
    features_train, target_train, features_test, target_test = import_training_data_draws_api()
    features_val, target_val = create_evaluation_sets_draws_api()
    
    with mlflow.start_run(run_name=f"xgboost_api_model"):
        # Log dataset info
        mlflow.log_params({
            "train_samples": len(features_train),
            "val_samples": len(features_val),
            "test_samples": len(features_test),
            "n_features_original": features_train.shape[1]
        })
        
        # Create and train model
        xgb_model = XGBoostModel(logger=logger)
        features_train = features_train[xgb_model.selected_features]
        features_val = features_val[xgb_model.selected_features]
        features_test = features_test[xgb_model.selected_features]
        # Log model parameters
        mlflow.log_params(xgb_model.global_params)
        mlflow.log_param("selected_features", xgb_model.selected_features)
        
        print("Starting training...")
        # Train the model
        xgb_model.train(features_train, target_train, features_test, target_test, features_val, target_val)
        
        # Create input example using only the selected features
        input_example = features_train.iloc[:1].copy()
        input_example = convert_int_columns(input_example)
    
        # Create model signature
        signature = mlflow.models.infer_signature(
            input_example,
            xgb_model.predict(input_example)
        )
        
        # Log the global model
        mlflow.xgboost.log_model(
            xgb_model.model,
            "model_global",
            signature=signature,
            input_example=input_example
        )
        
        # Analyze and log metrics
        train_analysis = xgb_model.analyze_predictions(features_train, target_train)
        val_analysis = xgb_model.analyze_predictions(features_val, target_val)
        test_analysis = xgb_model.analyze_predictions(features_test, target_test)
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
        return xgb_model

if __name__ == "__main__":
    model = train_with_mlflow()
