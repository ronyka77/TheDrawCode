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
from imblearn.over_sampling import ADASYN
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

# Local imports
from utils.logger import ExperimentLogger
from utils.create_evaluation_set import (
    import_selected_features_ensemble,
    create_ensemble_evaluation_set,
    import_training_data_ensemble,
    setup_mlflow_tracking
)


# Configure warnings
warnings.filterwarnings('ignore', category=UserWarning, 
                       module='xgboost.core', 
                       message='.*Saving model in the UBJSON format as default.*')

experiment_name = "xgboost_model_ensemble"

# Configure XGBoost for CPU-only training
xgb.set_config(verbosity=2)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU

mlruns_dir = setup_mlflow_tracking(experiment_name)

USE_DMATRIX = os.getenv('USE_DMATRIX', 'false').lower() == 'true'

# 1. Create custom callback class
class MLflowCallback(xgb.callback.TrainingCallback):
    def __init__(self, logger: ExperimentLogger, eval_sets: dict):
        super().__init__()
        self.logger = logger
        self.eval_sets = eval_sets
        self.iteration = 0

    def after_iteration(self, model, epoch, evals_log):
        """Log metrics after each boosting round"""
        # Log feature importance
        importance = model.get_score(importance_type='gain')
        self.logger.log_metrics({
            f'feature_importance/{feat}': float(val)
            for feat, val in importance.items()
        }, step=self.iteration)

        # Log evaluation metrics
        for eval_name, metrics in evals_log.items():
            for metric_name, metric_values in metrics.items():
                if len(metric_values) > 0:
                    self.logger.log_metrics({
                        f"{eval_name}_{metric_name}": float(metric_values[-1])
                    }, step=self.iteration)
        
        self.iteration += 1
        return False  # Continue training

class XGBoostModel(BaseEstimator, ClassifierMixin):
    """XGBoost model implementation with global training."""
    
    def __init__(
        self,
        logger: Optional[ExperimentLogger] = None,
        categorical_features: Optional[List[str]] = None) -> None:
        """Initialize XGBoost model."""
        self.logger = logger or ExperimentLogger('xgboost_api_model', f"xgboost_api_{datetime.now().strftime('%Y%m%d_%H%M')}")
        self.categorical_features = categorical_features or []
        
        # Updated global parameters based on hypertuning insights
        self.global_params = {
            'learning_rate': 0.00760022315772835,
            'early_stopping_rounds': 468,
            'min_child_weight': 135,
            'gamma': 2.6424389418822702,
            'subsample': 0.5507474166255953,
            'colsample_bytree': 0.8605669463072969,
            'scale_pos_weight': 2.3728539564011863,
            'reg_alpha': 0.00024338752831206268,
            'reg_lambda': 0.012210974240024406,
            'max_depth': 8,
            'n_estimators': 25840,
            # 'objective': 'binary:logistic',
            'tree_method': 'hist',
            'device': 'cpu',
            'eval_metric': ['error', 'auc', 'aucpr'],
            'verbosity': 0,
            'nthread': -1,
            'random_state': 42
        }
        self.adasyn_params = {
            'random_state': 42,
            'n_neighbors': 10,
            'sampling_strategy': 0.35 / (1 - 0.35)
        }
        # Initialize other attributes
        self.model = None
        self.feature_importance = {}
        self.selected_features = import_selected_features_ensemble('xgb')
        self.threshold = 0.50  # Default threshold for predictions
        self.focal_alpha = 2.0  # Added for focal loss
        self.focal_gamma = 2.0  # Added for focal loss
        # Remove conflicting parameters
        self.global_params.pop('objective', None)
        self.global_params.pop('booster', None)
    # 1. Define focal loss as static method with proper registration
    @staticmethod
    def _focal_loss(
        self,
        preds: np.ndarray, 
        dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
        """XGBoost-compatible focal loss implementation"""
        y_true = dtrain.get_label()
        p = 1.0 / (1.0 + np.exp(-preds))  # sigmoid
        
        alpha = 2.0  # Class weighting factor
        gamma = 2.0  # Focusing parameter
        
        grad = (p - y_true) * alpha * (1 - p)**gamma
        hess = p * (1 - p) * alpha * (1 - p)**gamma
        
        return grad.astype(np.float32), hess.astype(np.float32)

    
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
        
    def _optimize_threshold(self, y_true, y_prob):
        """Find optimal prediction threshold prioritizing precision while maintaining recall.
        
        Args:
            y_true: True target values
            y_prob: Predicted probabilities
            
        Returns:
            Dictionary of optimal threshold and metrics
        """
        try:
            best_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'threshold': 0.5}
            best_score = 0
            
            # Focus on higher thresholds for better precision, starting from 0.5
            for threshold in np.arange(0.5, 0.65, 0.01):
                preds = (y_prob >= threshold).astype(int)
                
                # Calculate metrics
                recall = recall_score(y_true, preds, zero_division=0)
                
                # Only consider thresholds that meet minimum recall
                if recall >= 0.20:
                    true_positives = ((preds == 1) & (y_true == 1)).sum()
                    false_positives = ((preds == 1) & (y_true == 0)).sum()
                    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                    f1 = f1_score(y_true, preds, zero_division=0)
                    
                    # Modified scoring to prioritize precision
                    score = precision * min(1.0, (recall - 0.20) / 0.20)
                    
                    if score > best_score:
                        best_score = score
                        best_metrics.update({
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'threshold': threshold
                        })
                        
                        # Log improvement
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

    # 1. Add DMatrix conversion methods
    def _create_dmatrix(
        self, 
        features: pd.DataFrame, 
        target: Optional[pd.Series] = None,
        weights: Optional[np.ndarray] = None) -> xgb.DMatrix:
        """Convert pandas data to optimized DMatrix"""
        return xgb.DMatrix(
            data=features[self.selected_features],
            label=target.values if target is not None else None,
            weight=weights,
            feature_names=self.selected_features,
            enable_categorical=False
        )

    # 2. Modify training method
    def train(
        self,
        features_train: Union[pd.DataFrame, np.ndarray],
        target_train: Union[pd.Series, np.ndarray],
        features_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        target_test: Optional[Union[pd.Series, np.ndarray]] = None,
        features_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        target_val: Optional[Union[pd.Series, np.ndarray]] = None) -> None:
        """Train the XGBoost model with ADASYN oversampling for the training set."""
        try:
            
            X_train, y_train, X_test, y_test, X_val, y_val = self._validate_data(
                features_train, target_train, features_test, target_test, features_val, target_val
            )
            X_train_original = X_train.copy()
            y_train_original = y_train.copy()
            # Create DMatrices
            dtrain = self._create_dmatrix(
                features_train,
                target_train,
                weights=np.where(target_train == 1, 5.0, 1.0)
            )
            dtest = self._create_dmatrix(features_test, target_test)
            
            # Initialize callback
            mlflow_callback = MLflowCallback(
                logger=self.logger,
                eval_sets={'test': dtest}
            )
            xgb.register_parallel_serializable('custom:focalloss', XGBoostModel._focal_loss)
            # Train with native API
            self.model = xgb.train(
                params={
                    **self.global_params,
                    'objective': 'custom:focalloss',  # Use registered name
                    'disable_default_eval_metric': 1
                },
                dtrain=dtrain,
                num_boost_round=self.global_params['n_estimators'],
                early_stopping_rounds=self.global_params['early_stopping_rounds'],
                evals=[(dtest, 'test')],
                callbacks=[mlflow_callback]  # Use the callback class instance
            )

            if X_test is not None and y_test is not None:
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_test, y_test)],
                    verbose=100
                )

            else:
                self.model.fit(X_train, y_train)
                
            # Optimize prediction threshold based on validation F1-score if validation data is provided
            if X_val is not None and y_val is not None:
                val_pred_proba = self.model.predict_proba(X_val)[:, 1]
                best_threshold = 0.5
                best_f1 = 0
                for threshold in np.arange(0.5, 0.65, 0.01):
                    preds = (val_pred_proba >= threshold).astype(int)
                    current_f1 = f1_score(y_val, preds, zero_division=0)
                    if current_f1 > best_f1:
                        best_f1 = current_f1
                        best_threshold = threshold
                self.threshold = best_threshold
                self.logger.info(f"Optimal threshold set to {self.threshold}")
            
            # Test point 1: DMatrix creation
            assert dtrain.num_row() == len(features_train), "Row count mismatch"

            # Test point 2: Gradient shape
            grad, hess = self._focal_loss(np.zeros(10), dtrain)
            assert grad.shape == (10,), "Invalid gradient shape"

            # Test point 3: Probability range
            proba = self.predict_proba(features_train)
            assert (proba >= 0).all() and (proba <= 1).all(), "Invalid probabilities"
            
        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")
            raise

   # 5. Modify prediction methods
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Get class probabilities using DMatrix"""
        dmatrix = self._create_dmatrix(features)
        raw_preds = self.model.predict(dmatrix, output_margin=True)
        proba_1 = 1.0 / (1.0 + np.exp(-raw_preds))  # sigmoid
        return np.vstack([1 - proba_1, proba_1]).T

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
                    'feature_importance': self.feature_importance,
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
        self.feature_importance = model_data['feature_importance']
        self.threshold = model_data['threshold']
        self.global_params = model_data['global_params']
        self.selected_features = model_data['selected_features']
        
        if self.logger:
            self.logger.info(f"Model loaded from {model_path}")

    # Add pickle support for distributed training
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove unpicklable callback
        state.pop('mlflow_callback', None)  
        return state


def convert_int_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert integer columns to float64 for MLflow compatibility."""
    try:
        # Convert all integer columns to float64
        for col in df.select_dtypes(include=['int']).columns:
            df[col] = df[col].astype('float64')
        return df
    except Exception as e:
        raise ValueError(f"Error converting integer columns: {str(e)}")

def train_global_model(experiment_name: str = "xgboost_api_model") -> None:
    """Train the global XGBoost model."""
    try:
        # Initialize logger
        logger = ExperimentLogger(log_dir='logs/xgboost_api_model', experiment_name=experiment_name)
        logger.info("Starting global model training...")
        
        # Load and prepare data
        logger.info("Loading training data...")
        selected_features = import_selected_features_ensemble('xgb')
        X_train, y_train, X_test, y_test = import_training_data_ensemble()
        logger.info("Creating evaluation set...")
        X_eval, y_eval = create_ensemble_evaluation_set()
        X_eval = X_eval[selected_features]
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]
        
        logger.info(f"X_train shape: {X_train.shape}")
        logger.info(f"X_test shape: {X_test.shape}")
        logger.info(f"X_eval shape: {X_eval.shape}")
        logger.info(f"y_train shape: {y_train.shape}")
        logger.info(f"y_test shape: {y_test.shape}")
        logger.info(f"y_eval shape: {y_eval.shape}")


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
                
                # Train model
                xgb_model.train(X_train, y_train, X_test, y_test, X_eval, y_eval)
                
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
                
                # Create MLflow signature with float64 types
                input_example = X_train.head(1)
                try:
                    signature = infer_signature(X_train, xgb_model.predict(X_train))
                except MlflowException as e:
                    logger.warning(f"Could not infer MLflow signature: {str(e)}")
                    signature = None
                
                # Save model with explicit format
                model_path = os.path.join(
                    project_root, 
                    "models", 
                    "saved", 
                    f"xgboost_api_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
                )
                xgb_model.save_model(model_path)
                
                # Log model to MLflow with signature
                if signature:
                    mlflow.xgboost.log_model(
                        xgb_model.model,
                        artifact_path="xgboost_api_model",
                        input_example=input_example,
                        registered_model_name=f"xgboost_api_{datetime.now().strftime('%Y%m%d_%H%M')}",
                        signature=signature
                    )
                else:
                    logger.warning("MLflow signature could not be inferred")
                
                logger.info("Global model training completed successfully")
                logger.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")
                
            except Exception as e:
                logger.error(f"Error in MLflow logging: {str(e)}")
                raise
            
    except Exception as e:
        logger.error(f"Error in global model training: {str(e)}")
        raise

class WeightedFocalLoss:
    def __init__(self, alpha=0.25, gamma=2):
        self.alpha = alpha
        self.gamma = gamma
        
    def __call__(self, y_true, y_pred):
        p = 1.0 / (1.0 + np.exp(-y_pred))
        ce = -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
        fl = self.alpha * (1 - p) ** self.gamma * ce
        return np.mean(fl)

if __name__ == "__main__":
    train_global_model()
