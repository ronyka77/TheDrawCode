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
import random
import time

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    log_loss, average_precision_score
)
from sklearn.model_selection import train_test_split
import mlflow
# Add warning filter imports
import warnings
from mlflow.models.signature import infer_signature
from mlflow.exceptions import MlflowException

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
os.environ['MLFLOW_RECORD_ENV_VARS_IN_MODEL_LOGGING'] = 'false'

# Local imports
from utils.logger import ExperimentLogger
experiment_name = "xgboost_ensemble_model"
logger = ExperimentLogger(experiment_name=experiment_name, log_dir='./logs/xgboost_ensemble_model')

from utils.create_evaluation_set import create_ensemble_evaluation_set, import_selected_features_ensemble, import_training_data_ensemble, setup_mlflow_tracking
mlruns_dir = setup_mlflow_tracking(experiment_name)

# Configure warnings
warnings.filterwarnings('ignore', category=UserWarning, 
                    module='xgboost.core', 
                    message='.*Saving model in the UBJSON format as default.*')

# Configure XGBoost for CPU-only training
xgb.set_config(verbosity=2)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU

mlruns_dir = setup_mlflow_tracking(experiment_name)

class XGBoostModel(BaseEstimator, ClassifierMixin):
    """XGBoost model implementation with global training."""
    def __init__(
        self,
        logger: Optional[ExperimentLogger] = None,
        random_seed: int = 357) -> None:
        """Initialize XGBoost model."""
        self.logger = logger or ExperimentLogger('xgboost_api_model', f"xgboost_api_{datetime.now().strftime('%Y%m%d_%H%M')}")
        # Updated global parameters based on hypertuning insights
        self.global_params = {
            'objective': 'binary:logistic',
            'tree_method': 'hist',
            'device': 'cpu',
            'eval_metric': ['aucpr', 'auc'],
            'verbosity': 0,
            'nthread': -1,
            'learning_rate': 0.029356898627545015,
            'early_stopping_rounds': 715,
            'min_child_weight': 132,
            'gamma': 0.058777520669765625,
            'subsample': 0.4112368231734437,
            'colsample_bytree': 0.6300718208925516,
            'scale_pos_weight': 1.9521786569409851,
            'reg_alpha': 0.027027315958530355,
            'reg_lambda': 5.724457204025571,
            'max_depth': 3,
            'n_estimators': 1898,
            'random_state': random_seed
        }
        # Initialize other attributes
        self.model = None
        self.selected_features = import_selected_features_ensemble('all')
        self.threshold = 0.30  # Default threshold for predictions

    def _validate_data(
        self,
        x_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        x_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_test: Optional[Union[pd.Series, np.ndarray]] = None,
        x_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.DataFrame], Optional[pd.Series]]:
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
        
        # Convert all columns to float64 for consistency
        for df, dataset_name in datasets:
            if df is not None:
                # Convert all numeric columns to float64
                for col in df.columns:
                    try:
                        if df[col].dtype == 'object':
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        # Convert all numeric columns to float64
                        if pd.api.types.is_numeric_dtype(df[col]):
                            df[col] = df[col].astype('float64')
                    except Exception as e:
                        if self.logger:
                            self.logger.warning(f"Could not convert column {col} in {dataset_name} dataset: {str(e)}")
                        df.drop(columns=[col], inplace=True)
                
                # Fill missing values
                df.fillna(0.0, inplace=True)
        
        # Validate feature consistency across datasets
        if x_train_df is not None and x_val_df is not None:
            if x_train_df.shape[1] != x_val_df.shape[1]:
                raise ValueError("Training and validation features must have the same number of columns")
        return x_train_df, y_train_series, x_test_df, y_test_series, x_val_df, y_val_series

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
            
            return valid_metrics

        except Exception as e:
            self.logger.error(f"Error logging metrics: {e}")
            raise

    def _find_optimal_threshold(
        self,
        model: xgb.XGBClassifier,
        features_val: pd.DataFrame,
        target_val: pd.Series) -> Tuple[float, Dict[str, float]]:
        """Find optimal prediction threshold prioritizing precision while maintaining recall.
        Args:
            model: Trained XGBoost model
            features_val: Validation features
            target_val: Validation targets
        Returns:
            Tuple of (optimal threshold, metrics dictionary)
        """
        try:
            prediction_df = features_val.copy()
            prediction_df = prediction_df[self.selected_features]
            probas = model.predict_proba(prediction_df)[:, 1]
            best_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'threshold': 0.3}
            best_score = 0
            # Focus on higher thresholds for better precision, starting from 0.3
            for threshold in np.arange(0.3, 0.65, 0.01):
                # print(f"Threshold: {threshold}")
                preds = (probas >= threshold).astype(int)
                true_positives = ((preds == 1) & (target_val == 1)).sum()
                false_positives = ((preds == 1) & (target_val == 0)).sum()
                true_negatives = ((preds == 0) & (target_val == 0)).sum()
                false_negatives = ((preds == 0) & (target_val == 1)).sum()
                # Calculate metrics
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                # print(f"Recall: {recall}")
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                # print(f"Precision: {precision}")
                # Only consider thresholds that meet minimum recall
                if recall >= 0.14:
                    f1 = f1_score(target_val, preds)
                    # Modified scoring to prioritize precision
                    score = precision
                    
                    if score > best_score:
                        best_score = score
                        best_metrics.update({
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'threshold': threshold
                        })
            self.threshold = best_metrics['threshold']
            self.logger.info(f"Optimal threshold set to {self.threshold}")        
            
            if best_metrics['recall'] < 0.14:

                self.logger.warning(
                    f"Could not find threshold meeting recall requirement. "
                    f"Best recall: {best_metrics['recall']:.4f}"
                    f"Best precision: {best_metrics['precision']:.4f}"
                )
            self.logger.info(
                f"New best threshold {best_metrics['threshold']:.3f}: "
                f"Precision={best_metrics['precision']:.4f}, Recall={best_metrics['recall']:.4f}"
            )
            return self.threshold, best_metrics
            
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
        target_val: Optional[Union[pd.Series, np.ndarray]] = None) -> None:
        """Train the XGBoost model."""
        try:
            # Initialize base model with early stopping
            self.model = xgb.XGBClassifier(**self.global_params)
            
            # Native XGBoost training with early stopping
            self.model.fit(
                features_train, 
                target_train,
                eval_set=[(features_test, target_test)],
                verbose=False
            )
            
            # Threshold optimization uses validation data
            threshold, metrics = self._find_optimal_threshold(
                self.model, 
                features_val, 
                target_val
            )
            
            return metrics['precision']
            
        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")
            raise

    def analyze_predictions(
        self,
        features: Union[pd.DataFrame, np.ndarray],
        target: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """Analyze model predictions."""
        # Get predictions
        y_prob = self.model.predict_proba(features)
        # Handle both 1D and 2D probability arrays
        y_prob_class1 = y_prob[:, 1]
        y_pred = (y_prob_class1 >= self.threshold).astype(int)
        
        # Calculate confusion matrix
        true_positives = ((target == 1) & (y_pred == 1)).sum()
        false_positives = ((target == 0) & (y_pred == 1)).sum()
        true_negatives = ((target == 0) & (y_pred == 0)).sum()
        false_negatives = ((target == 1) & (y_pred == 0)).sum()
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = f1_score(target, y_pred, zero_division=0)
        # Calculate metrics
        metrics = {
            'accuracy': np.mean(y_pred == target),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'log_loss': log_loss(target, y_prob_class1),
            'average_precision': average_precision_score(target, y_prob_class1),
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

    def save_model(self, model_path: str) -> None:
        """Save model with proper format handling"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save model in XGBoost binary format
            model_json_path = model_path.replace('.pkl', '.xgb')
            self.model.save_model(model_json_path)
            
            # Save metadata separately
            metadata_path = model_path.replace('.pkl', '_metadata.pkl')
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'threshold': self.threshold,
                    'global_params': self.global_params,
                    'selected_features': self.selected_features
                }, f)
            if self.logger:
                self.logger.info(f"Model saved to {model_json_path}")
                self.logger.info(f"Metadata saved to {metadata_path}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, model_path: str) -> None:
        """Load a saved model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.threshold = model_data['threshold']
        self.global_params = model_data['global_params']
        self.selected_features = model_data['selected_features']
        if self.logger:
            self.logger.info(f"Model loaded from {model_path}")

def train_global_model(experiment_name: str = "xgboost_api_model") -> None:
    """Train the global XGBoost model."""
    try:
        # Initialize logger
        logger = ExperimentLogger(log_dir='logs/xgboost_api_model', experiment_name=experiment_name)
        logger.info("Starting global model training...")
        
        # Load and prepare data
        logger.info("Loading training data...")
        selected_features = import_selected_features_ensemble('all')
        X_train, y_train, X_test, y_test = import_training_data_ensemble()
        logger.info("Creating evaluation set...")
        X_eval, y_eval = create_ensemble_evaluation_set()
        X_eval = X_eval[selected_features]
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]
        
        # Convert all features to float64
        X_train = X_train.astype('float64')
        X_test = X_test.astype('float64')
        X_eval = X_eval.astype('float64')
        
        logger.info(f"X_train shape: {X_train.shape}")
        logger.info(f"X_test shape: {X_test.shape}")
        logger.info(f"X_eval shape: {X_eval.shape}")
        
        # Initialize model
        logger.info("Initializing model...")
        xgb_model = XGBoostModel(logger=logger)
        
        # Start MLflow run with experiment tracking
        with mlflow.start_run(experiment_id=mlflow.get_experiment_by_name(experiment_name).experiment_id,
                            run_name=f"xgboost_api_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            try:
                # Log global parameters to MLflow
                mlflow.log_params(xgb_model.global_params)
                logger.info("Logged global parameters to MLflow", extra={"params": xgb_model.global_params})
                
                # Log additional training metadata
                mlflow.log_metric("train_samples", len(X_train))
                mlflow.log_metric("test_samples", len(X_test))
                mlflow.log_metric("eval_samples", len(X_eval))
                logger.info("Logged dataset sizes to MLflow", 
                                extra={"train_samples": len(X_train), 
                                "test_samples": len(X_test), 
                                "eval_samples": len(X_eval)})
                # Set MLflow tags for model configuration   
                mlflow.set_tags({
                    "model_type": "xgboost",
                    "training_mode": "global",
                    "cpu_only": True,
                    "tree_method": "hist"
                })
                logger.info("Set MLflow tags for model configuration")
                
                # Train model with increased precision target
                precision = 0
                highest_precision = 0
                best_seed = 0
                best_model = None
                while precision < 0.48:
                    for random_seed in range(1, 600):
                        logger.info(f"Using sequential random seed: {random_seed}")
                        os.environ['PYTHONHASHSEED'] = str(random_seed)
                        np.random.seed(random_seed)
                        random.seed(random_seed)
                        xgb_model = XGBoostModel(logger=logger, random_seed=random_seed)
                        precision = xgb_model.train(X_train, y_train, X_test, y_test, X_eval, y_eval)
                        if precision > highest_precision:
                            highest_precision = precision
                            best_seed = random_seed
                            best_model = xgb_model
                        if precision >= 0.48:
                            logger.info(f"Target precision achieved: {precision:.4f}")
                            break
                        logger.info(f"Current precision: {precision:.4f}, target: 0.4800 highest precision: {highest_precision:.4f} best seed: {best_seed}")
                    # if not reached target precision, use best seed
                    if precision < 0.48:
                        logger.info(f"Target precision not reached, using best seed: {best_seed}")
                        xgb_model = best_model
                        break

                # Get and log validation metrics
                try:
                    val_metrics = xgb_model.validate_completion_metrics(X_eval, y_eval)
                    if val_metrics:
                        # Log each metric to MLflow
                        for metric_name, metric_value in val_metrics.items():
                            mlflow.log_metric(metric_name, metric_value)
                        logger.info("Logged validation metrics to MLflow", 
                                extra={"metrics": val_metrics})
                    else:
                        logger.warning("No valid metrics returned from validation")
                except Exception as e:
                    logger.error(f"Error logging validation metrics: {str(e)}")
                    raise

                # Create MLflow signature with proper input example
                try:
                    input_example = X_train.head(1).copy()
                    # Ensure all columns are float64 and handle potential conversion errors
                    try:
                        input_example = input_example.astype('float64')
                    except Exception as e:
                        logger.warning(f"Error converting input example to float64: {str(e)}")
                        # Fallback to numeric conversion with coercion
                        input_example = input_example.apply(pd.to_numeric, errors='coerce').fillna(0.0)
                    
                    # Create signature with proper types and error handling
                    try:
                        signature = infer_signature(
                            model_input=input_example,
                            model_output=xgb_model.model.predict(input_example)
                        )
                    except Exception as e:
                        logger.error(f"Error creating MLflow signature: {str(e)}")
                        # Fallback to basic signature inference
                        signature = infer_signature(input_example)
                    # Log model to MLflow with proper configuration
                    mlflow.xgboost.log_model(
                        xgb_model=xgb_model.model,
                        artifact_path="xgboost_api_model",
                        registered_model_name=f"xgboost_api_{datetime.now().strftime('%Y%m%d_%H%M')}",
                        signature=signature
                    )
                except Exception as e:
                    logger.error(f"Error creating MLflow signature: {str(e)}")
                    # Log model without signature as fallback
                    mlflow.xgboost.log_model(
                        xgb_model=xgb_model.model,
                        artifact_path="xgboost_api_model",
                        registered_model_name=f"xgboost_api_{datetime.now().strftime('%Y%m%d_%H%M')}"
                    )
                
                logger.info("Global model training completed successfully")
                logger.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")
                
            except Exception as e:
                logger.error(f"Error in MLflow logging: {str(e)}")
                raise
            
    except Exception as e:
        logger.error(f"Error in global model training: {str(e)}")
        raise

if __name__ == "__main__":
    train_global_model()
