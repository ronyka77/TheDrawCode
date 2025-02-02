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
import math

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
from utils.create_evaluation_set import (
    get_selected_api_columns_draws,
    create_evaluation_sets_draws_api,
    import_training_data_draws_api,
    setup_mlflow_tracking
)

experiment_name = "xgboost_api_model"

# Configure XGBoost for CPU-only training
xgb.set_config(verbosity=2)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU

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
        
        # Updated global parameters based on hypertuning insights
        self.global_params = {
            'colsample_bytree': 0.6067781275312805,
            'gamma': 16.545660404260257,
            'learning_rate': 0.015,  # Increased from 0.003
            'min_child_weight': 203,
            'n_estimators': 8500,  # Reduced from 21177
            'reg_alpha': 1.6561176081957816,
            'reg_lambda': 0.18858356401987753,
            'scale_pos_weight': 8.2,  # Adjusted from 4.9
            'subsample': 0.7955139034577977,
            'max_depth': 9,  # Reduced from 11
            'objective': 'binary:logistic',
            'tree_method': 'hist',
            'eval_metric': ['error', 'auc', 'aucpr'],
            'early_stopping_rounds': 500,  # Reduced from 1000
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

    def validate_completion_metrics(
        self,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series]) -> Dict[str, float]:
        """Log and return completion metrics.
        
        Args:
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Dictionary of valid metrics
        """
        try:
            # Get metrics from prediction analysis
            metrics = self.analyze_predictions(X_val, y_val)['metrics']
            
            # Filter out invalid metrics (None or NaN)
            valid_metrics = {
                k: v for k, v in metrics.items() 
                if v is not None and not (isinstance(v, float) and math.isnan(v))
            }
            
            # Log each valid metric
            if self.logger:
                for metric_name, metric_value in valid_metrics.items():
                    self.logger.info(f"{metric_name}: {metric_value}")
            
            return valid_metrics

        except Exception as e:
            self.logger.error(f"Error logging metrics: {e}")
            raise
        
    def _optimize_threshold(self, y_true, y_prob):
        """Revised threshold optimization with recall constraint"""
        best_score = -np.inf
        best_metrics = {'threshold': 0.55}
        
        # Wider threshold range with finer increments
        for threshold in np.linspace(0.2, 0.7, 200):
            y_pred = (y_prob >= threshold).astype(int)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            
            # Enforce minimum recall constraint
            if recall < 0.40:
                continue  # Skip thresholds that don't meet recall requirement
            
            # Weighted score favoring precision while maintaining recall
            precision_ratio = precision 
            recall_ratio = recall 
            score = (0.7 * precision_ratio) + (0.3 * recall_ratio)
            
            if score > best_score:
                best_score = score
                best_metrics.update({
                    'precision': precision,
                    'recall': recall,
                    'threshold': threshold
                })
        
        self.logger.info(f"Optimized threshold: {best_metrics['threshold']:.3f}")
        return best_metrics

    def train(
        self,
        features_train: Union[pd.DataFrame, np.ndarray],
        target_train: Union[pd.Series, np.ndarray],
        features_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        target_val: Optional[Union[pd.Series, np.ndarray]] = None) -> None:
        """Train the XGBoost model."""
        try:
            # Unified data handling with hypertuning
            X_train, y_train, X_val, y_val = self._validate_data(
                features_train, target_train, features_val, target_val
            )
            
            # Align with hypertuning's XGBClassifier approach
            model = xgb.XGBClassifier(**self.global_params)
            early_stopping = xgb.callback.EarlyStopping(
                rounds=self.global_params['early_stopping_rounds'],
                metric_name='aucpr',
                save_best=True
            )
            
            # Mirror hypertuning's fit procedure
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=100
            )
            
            # Store model and feature importance
            self.model = model
            self.feature_importance = model.get_booster().get_score(importance_type='gain')
            
            # Unified threshold optimization
            val_probs = model.predict_proba(X_val)[:, 1]
            best_metrics = self._optimize_threshold(y_val.values, val_probs)
            self.threshold = best_metrics['threshold']
            
        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")
            raise

    def predict_proba(self, features: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Get probability predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Validate and prepare features
        features_df = pd.DataFrame(features) if isinstance(features, np.ndarray) else features.copy()
        features_df = features_df[self.selected_features]
        
        # Get predictions using XGBClassifier
        return self.model.predict_proba(features_df)

    def predict(self, features: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Get binary predictions using the optimal threshold."""
        probabilities = self.predict_proba(features)
        return (probabilities[:, 1] >= self.threshold).astype(int)

    def analyze_predictions(
        self,
        features: Union[pd.DataFrame, np.ndarray],
        target: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """Analyze model predictions."""
        # Get predictions
        y_prob = self.predict_proba(features)
        y_pred = (y_prob[:, 1] >= self.threshold).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': np.mean(y_pred == target),
            'precision': precision_score(target, y_pred, zero_division=0),
            'recall': recall_score(target, y_pred, zero_division=0),
            'f1': f1_score(target, y_pred, zero_division=0),
            'log_loss': log_loss(target, y_prob[:, 1]),
            'average_precision': average_precision_score(target, y_prob[:, 1])
        }
        
        # Calculate confusion matrix
        true_positives = np.sum((target == 1) & (y_pred == 1))
        false_positives = np.sum((target == 0) & (y_pred == 1))
        true_negatives = np.sum((target == 0) & (y_pred == 0))
        false_negatives = np.sum((target == 1) & (y_pred == 0))
        
        confusion = {
            'true_positives': int(true_positives),
            'false_positives': int(false_positives),
            'true_negatives': int(true_negatives),
            'false_negatives': int(false_negatives)
        }
        
        # Log analysis results
        if self.logger:
            self.logger.info("Prediction Analysis Results:")
            for metric_name, value in metrics.items():
                self.logger.info(f"{metric_name}: {value:.4f}")
            self.logger.info(f"Confusion Matrix: {confusion}")
        
        return {
            'metrics': metrics,
            'confusion_matrix': confusion,
            'threshold': self.threshold
        }

    def save_model(self, model_path: str) -> None:
        """Save model with proper format handling"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save in JSON format for better compatibility
        self.model.save_model(model_path)  # Removed format parameter
        
        # Save metadata separately
        metadata_path = model_path.replace('.json', '_metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'feature_importance': self.feature_importance,
                'threshold': self.threshold,
                'global_params': self.global_params,
                'selected_features': self.selected_features
            }, f)
        
        if self.logger:
            self.logger.info(f"Model saved to {model_path}")

    def load_model(self, model_path: str) -> None:
        """Load a saved model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_importance = model_data['feature_importance']
        self.threshold = model_data['threshold']
        self.global_params = model_data['global_params']
        self.selected_features = model_data['selected_features']
        
        if self.logger:
            self.logger.info(f"Model loaded from {model_path}")

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

def train_global_model(experiment_name: str = "xgboost_api_model") -> None:
    """Train the global XGBoost model."""
    try:
        # Initialize logger
        logger = ExperimentLogger(experiment_name=experiment_name)
        logger.info("Starting global model training...")
        
        # Load and prepare data
        logger.info("Loading training data...")
        X_train, y_train, X_test, y_test = import_training_data_draws_api()
        
        # Create evaluation set
        logger.info("Creating evaluation set...")
        X_eval, y_eval = create_evaluation_sets_draws_api()
        
        # Initialize model
        logger.info("Initializing model...")
        xgb_model = XGBoostModel(logger=logger)
        
        # Align data handling with hypertuning's full dataset approach
        full_X = pd.concat([X_train, X_test])
        full_y = pd.concat([y_train, y_test])
        
        # Start MLflow run with experiment tracking
        with mlflow.start_run(experiment_id=mlflow.get_experiment_by_name(experiment_name).experiment_id):
            logger.info(f"Started MLflow run for experiment: {experiment_name}")
            
            # Log data shapes and statistics
            logger.info(f"Full training data shape: {full_X.shape}")
            logger.info(f"Evaluation data shape: {X_eval.shape}")
            logger.info(f"Training draw rate: {full_y.mean():.2%}")
            logger.info(f"Evaluation draw rate: {y_eval.mean():.2%}")
            
            # Log initial parameters
            mlflow.log_params(xgb_model.global_params)
            mlflow.log_param("training_samples", full_X.shape[0])
            mlflow.log_param("evaluation_samples", X_eval.shape[0])
        
            # Mirror hypertuning's two-phase training
            xgb_model.train(full_X, full_y, X_eval, y_eval)
            
            # Get and log metrics
            test_metrics = xgb_model.validate_completion_metrics(X_test, y_test)
            eval_metrics = xgb_model.validate_completion_metrics(X_eval, y_eval)
            
            # Save model
            model_path = os.path.join(project_root, "models", "saved", f"xgboost_api_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            xgb_model.save_model(model_path)
            
    
            # Log model with signature
            signature = mlflow.models.infer_signature(
                X_train,
                xgb_model.predict(X_train)
            )
            mlflow.xgboost.log_model(
                xgb_model.model,
                artifact_path="xgboost_api_model",
                registered_model_name=f"xgboost_api_{datetime.now().strftime('%Y%m%d_%H%M')}",
                signature=signature
            )
        
            # Log parameters and metrics
            mlflow.log_metrics(test_metrics)
            mlflow.log_metrics(eval_metrics)
        
            # Log feature importance as artifact
            feature_importance_path = os.path.join(project_root, "feature_importance")
            mlflow.log_artifacts(feature_importance_path)
            
            # Log model path
            mlflow.log_artifact(model_path)
            
            logger.info("Global model training completed successfully")
            mlflow.end_run()
        
        
        
    except Exception as e:
        logger.error(f"Error in global model training: {str(e)}")
        raise

if __name__ == "__main__":
    train_global_model()
