"""
LightGBM model implementation for binary classification with global training capabilities.
Provides methods for model training, prediction, and analysis with configurable parameters.
"""

# Standard library imports
import glob
import json
import os
import pickle
import sys
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, List
import warnings

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    log_loss, average_precision_score
)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN
import mlflow
import mlflow.lightgbm
from mlflow.models.signature import infer_signature
from mlflow.exceptions import MlflowException
import lightgbm as lgb

# Add project root to Python path
try:
    project_root = Path(__file__).parent.parent
    if not project_root.exists():
        # Handle network path by using raw string
        project_root = Path(r"\\".join(str(project_root).split("\\")))
    sys.path.append(str(project_root))
    print(f"Project root lightgbm_model: {project_root}")
except Exception as e:
    print(f"Error setting project root path: {e}")
    sys.path.append(os.getcwd().parent)
    print(f"Current directory lightgbm_model: {os.getcwd().parent}")

# Environment settings: enforce CPU-only training
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['GIT_PYTHON_GIT_EXECUTABLE'] = "C:/Program Files/Git/bin/git.exe"

# Local imports
from utils.logger import ExperimentLogger
from utils.create_evaluation_set import (
    import_selected_features_ensemble,
    create_ensemble_evaluation_set,
    import_training_data_ensemble,
    setup_mlflow_tracking
)

# Configure warnings
warnings.filterwarnings('ignore', category=UserWarning)

experiment_name = "lightgbm_ensemble_model"

class LightGBMModel(BaseEstimator, ClassifierMixin):
    """LightGBM model implementation with global training."""
    
    def __init__(
        self,
        logger: Optional[ExperimentLogger] = None,
        categorical_features: Optional[List[str]] = None) -> None:
        """Initialize LightGBM model."""
        self.logger = logger or ExperimentLogger(
            experiment_name="lightgbm_api_model", 
            log_dir="logs/lightgbm_api_model"
        )
        self.categorical_features = categorical_features or []
        
        # Global LightGBM parameters
        self.global_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 21,
            'learning_rate': 0.010613454122159499,
            'feature_fraction': 0.9184993834178807,
            'bagging_fraction': 0.7510496512913906,
            'bagging_freq': 2,
            'verbose': -1,
            'seed': 42,
            'max_depth': 8,
            'is_unbalance': True,
            'early_stopping_rounds': 390,
            'reg_lambda': 12.892697330760878,
            'n_estimators': 1523,
            'colsample_bytree': 0.8954024976110304,
            'min_child_samples': 15
        }
        
        self.model = None
        self.threshold = 0.50
        self.selected_features = import_selected_features_ensemble('lgbm')
        self.feature_importance = {}

    def _validate_data(
        self,
        x_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        x_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_test: Optional[Union[pd.Series, np.ndarray]] = None,
        x_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame], Optional[pd.Series]]:
        """Validate and format input data."""

        x_train_df = pd.DataFrame(x_train) if isinstance(x_train, np.ndarray) else x_train.copy()
        y_train_series = pd.Series(y_train) if isinstance(y_train, np.ndarray) else y_train.copy()
        x_test_df = pd.DataFrame(x_test) if isinstance(x_test, np.ndarray) else x_test.copy()
        y_test_series = pd.Series(y_test) if isinstance(y_test, np.ndarray) else y_test.copy()
        x_val_df = pd.DataFrame(x_val) if isinstance(x_val, np.ndarray) else x_val.copy()
        y_val_series = pd.Series(y_val) if isinstance(y_val, np.ndarray) else y_val.copy()
        
        # Process all datasets consistently
        datasets = [
            (x_train_df, 'training'),
            (x_test_df, 'test'),
            (x_val_df, 'validation')
        ]
        
        # Convert non-numeric columns and handle missing values for each dataset
        for df, dataset_name in datasets:
            if df is not None:
                # Convert object columns to numeric where possible
                for col in df.columns:
                    try:
                        if df[col].dtype == 'object':
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    except Exception as e:
                        if self.logger:
                            self.logger.warning(f"Could not convert column {col} in {dataset_name} dataset: {str(e)}")
                        df.drop(columns=[col], inplace=True)
                
                # Fill missing values
                df.fillna(0, inplace=True)
        
        # Validate feature consistency across datasets
        if x_train_df is not None and x_val_df is not None:
            if x_train_df.shape[1] != x_val_df.shape[1]:
                raise ValueError("Training and validation features must have the same number of columns")
        return x_train_df, y_train_series, x_test_df, y_test_series, x_val_df, y_val_series

    def validate_completion_metrics(
        self,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series]) -> Dict[str, float]:
        """Log and return completion metrics."""
        try:
            metrics = self.analyze_predictions(X_val, y_val)['metrics']
            valid_metrics = {
                k: v for k, v in metrics.items() 
                if v is not None and not (isinstance(v, float) and math.isnan(v))
            }
            return valid_metrics
        except Exception as e:
            self.logger.error(f"Error logging metrics: {e}")
            raise

    def _optimize_threshold(self, y_true, y_prob):
        """Find optimal prediction threshold prioritizing precision while maintaining recall."""
        try:
            best_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'threshold': 0.5}
            best_score = 0
            
            for threshold in np.arange(0.5, 0.65, 0.01):
                preds = (y_prob >= threshold).ast(int)
                recall = recall_score(y_true, preds, zero_division=0)
                
                if recall >= 0.20:
                    true_positives = ((preds == 1) & (y_true == 1)).sum()
                    false_positives = ((preds == 1) & (y_true == 0)).sum()
                    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                    f1 = f1_score(y_true, preds, zero_division=0)
                    
                    score = precision * min(1.0, (recall - 0.20) / 0.20)
                    
                    if score > best_score:
                        best_score = score
                        best_metrics.update({
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'threshold': threshold
                        })
                        
                        self.logger.info(
                            f"New best threshold {threshold:.3f}: "
                            f"Precision={precision:.4f}, Recall={recall:.4f}"
                        )
            
            if best_metrics['recall'] < 0.20:
                self.logger.warning(
                    f"Could not find threshold meeting recall requirement. "
                    f"Best recall: {best_metrics['recall']:.4f}"
                )
            
            return best_metrics
            
        except Exception as e:
            self.logger.error(f"Error in threshold optimization: {str(e)}")
            raise

    def train(
        self,
        features_train: Union[pd.DataFrame, np.ndarray],
        target_train: Union[pd.Series, np.ndarray],
        features_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        target_test: Optional[Union[pd.Series, np.ndarray]] = None,
        features_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        target_val: Optional[Union[pd.Series, np.ndarray]] = None,
        ) -> None:
        """Train the LightGBM model with ADASYN oversampling."""
        try:
            X_train, y_train, X_test, y_test, X_val, y_val = self._validate_data(
                features_train, target_train, features_test, target_test, features_val, target_val
            )
            class_weights = {0: 1, 1: len(np.where(y_train == 0)[0])/len(np.where(y_train == 1)[0])}
            

            self.model = lgb.LGBMClassifier(**self.global_params,
                class_weight=class_weights )
            if X_test is not None and y_test is not None:   

                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_test, y_test)]
                )
            else:
                self.model.fit(X_train, y_train)
            
            
            # Optimize prediction threshold based on validation F1-score if validation data is provided
            if X_val is not None and y_val is not None:
                val_pred_proba = self.model.predict_proba(X_val)[:, 1]
                best_threshold = 0.5
                best_f1 = 0
                for threshold in np.arange(0.5, 0.61, 0.01):
                    preds = (val_pred_proba >= threshold).astype(int)
                    current_f1 = f1_score(y_val, preds, zero_division=0)
                    if current_f1 > best_f1:
                        best_f1 = current_f1
                        best_threshold = threshold
                self.threshold = best_threshold
                self.logger.info(f"Optimal threshold set to {self.threshold}")
            
        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")
            raise

    def predict(self, features: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Get binary predictions using the optimal threshold."""
        if self.model is None:

            raise ValueError("Model not trained. Call train() first.")
        
        # Validate and prepare features
        features_df = pd.DataFrame(features) if isinstance(features, np.ndarray) else features.copy()
        features_df = features_df[self.selected_features]
        
        # Get predictions using CatBoostClassifier
        probabilities = self.predict_proba(features_df)
        return (probabilities[:, 1] >= self.threshold).astype(int)
    
    def predict_proba(self, features: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Get probability predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Validate and prepare features
        features_df = pd.DataFrame(features) if isinstance(features, np.ndarray) else features.copy()
        features_df = features_df[self.selected_features]
        
        # Get predictions using CatBoostClassifier
        return self.model.predict_proba(features_df)

    def analyze_predictions(
        self,
        features: Union[pd.DataFrame, np.ndarray],
        target: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """Analyze model predictions."""
        # Get predictions
        y_prob = self.predict_proba(features)
        y_pred = (y_prob[:, 1] >= self.threshold).astype(int)
        
        # Calculate confusion matrix
        true_positives = ((target == 1) & (y_pred == 1)).sum()
        false_positives = ((target == 0) & (y_pred == 1)).sum()
        true_negatives = ((target == 0) & (y_pred == 0)).sum()
        false_negatives = ((target == 1) & (y_pred == 0)).sum()
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = f1_score(target, y_pred, zero_division=0)
        
        metrics = {
            'accuracy': np.mean(y_pred == target),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'log_loss': log_loss(target, y_prob[:, 1]),
            'average_precision': average_precision_score(target, y_prob[:, 1]),
            'draw_rate': float(target.mean()),
            'predicted_rate': float(y_pred.mean()),
            'n_samples': len(target),
            'n_draws': int(target.sum()),
            'n_predicted': int(y_pred.sum()),
            'n_correct': int(np.logical_and(target, y_pred).sum()),
            'n_incorrect': int(np.logical_not(np.logical_and(target, y_pred)).sum())
        }
        
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

if __name__ == "__main__":
    # Set up MLflow tracking and start run
    mlruns_dir = setup_mlflow_tracking('lightgbm_model')

    # Load data using utility functions
    selected_features = import_selected_features_ensemble('lgbm')
    features_train, target_train, features_test, target_test = import_training_data_ensemble()
    features_val, target_val = create_ensemble_evaluation_set()
    features_train = features_train[selected_features]
    features_test = features_test[selected_features]
    features_val = features_val[selected_features]
    print(features_train.shape)
    print(features_test.shape)
    print(features_val.shape)


    with mlflow.start_run(run_name="lightgbm_training") as run:
        # Create and train the LightGBM model
        lgb_model = LightGBMModel()
        lgb_model.train(features_train, target_train, features_test, target_test, features_val, target_val)
        

        # Make predictions and calculate metrics
        predictions = lgb_model.predict(features_val)
        f1 = f1_score(target_val, predictions, zero_division=0)
        precision = precision_score(target_val, predictions, zero_division=0)
        recall = recall_score(target_val, predictions, zero_division=0)
        

        # Log metrics to MLflow and logger
        metrics = {
            "f1_score": f1,
            "precision": precision,
            "recall": recall
        }
        mlflow.log_metrics(metrics)
        lgb_model.logger.info(f"Validation Metrics: {metrics}")
        
        # Log model parameters
        mlflow.log_params(lgb_model.global_params)
        
        # Log model with signature and input example
        input_example = features_val.head(1)

        signature = infer_signature(input_example, lgb_model.predict(input_example))
        registered_model_name = f"lightgbm_{datetime.now().strftime('%Y%m%d_%H%M')}"

        mlflow.lightgbm.log_model(
            lgb_model.model,
            artifact_path="lightgbm_model",
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=input_example
        )
        
        print("LightGBM training and evaluation complete.")
        print("Run ID:", run.info.run_id)
        mlflow.end_run()
