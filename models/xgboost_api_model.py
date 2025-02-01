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
            'objective': 'binary:logistic',
            'tree_method': 'hist',
            'eval_metric': ['error', 'auc', 'aucpr'],
            'verbosity': 0,
            'early_stopping_rounds': 1000,
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
        """Log completion metrics to both logger and MLflow."""
        if X_val is not None and y_val is not None:
            analysis = self.analyze_predictions(X_val, y_val)
            
            # Prepare metrics for logging
            metrics = {
                'best_iteration': self.model.best_iteration,
                'best_score': self.model.best_score,
                'n_features': len(self.feature_importance),
                'val_accuracy': analysis['metrics']['accuracy'],
                'val_precision': analysis['metrics']['precision'],
                'val_recall': analysis['metrics']['recall'],
                'val_f1': analysis['metrics']['f1']
            }
            
            # Log to MLflow
            mlflow.log_metrics(metrics)
            
            # Log feature importance plot
            self._log_feature_importance()
            
            # Log model parameters
            mlflow.log_params(self.global_params)
            
            # Log model
            mlflow.xgboost.log_model(
                xgb_model=self.model,
                artifact_path="model",
                registered_model_name=f"xgboost_api_{datetime.now().strftime('%Y%m%d_%H%M')}",
                conda_env={
                    'name': 'soccerprediction_env',
                    'channels': ['conda-forge'],
                    'dependencies': [
                        'python=3.9',
    
                        'xgboost',
                        'scikit-learn',
                        'pandas',
                        'numpy'
                    ]

                }
            )
            
            # Log to logger
            if self.logger:
                for metric_name, metric_value in metrics.items():
                    self.logger.info(f"{metric_name}: {metric_value}")

    def _log_feature_importance(self) -> None:
        """Log feature importance plot to MLflow."""
        import matplotlib.pyplot as plt
        
        # Create feature importance plot
        plt.figure(figsize=(10, 6))
        sorted_idx = np.argsort(list(self.feature_importance.values()))
        pos = np.arange(sorted_idx.shape[0]) + .5
        plt.barh(pos, np.array(list(self.feature_importance.values()))[sorted_idx])
        plt.yticks(pos, np.array(list(self.feature_importance.keys()))[sorted_idx])
        plt.xlabel('Feature Importance Score')
        plt.title('Feature Importance')
        
        # Save plot
        plot_path = "feature_importance.png"
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        # Log to MLflow
        mlflow.log_artifact(plot_path)
        os.remove(plot_path)  # Clean up

    def _optimize_threshold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray) -> float:
        """Find optimal prediction threshold using validation data."""
        best_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'threshold': 0.55}
        best_score = 0
        
        # Focus on higher thresholds for better precision
        for threshold in np.arange(0.4, 0.95, 0.01):
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
        features_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        target_val: Optional[Union[pd.Series, np.ndarray]] = None) -> None:
        """Train the XGBoost model."""
        try:
            # Start MLflow run
            with mlflow.start_run(run_name=f"xgboost_train_{datetime.now().strftime('%Y%m%d_%H%M')}"):
                # Log training start
                self.logger.info("Starting model training...")
                mlflow.log_param("training_start", datetime.now().isoformat())
                
                # Validate and prepare data
                X_train, y_train, X_val, y_val = self._validate_data(
                    features_train, target_train, features_val, target_val
                )
                
                # Create DMatrix objects for training
                dtrain = xgb.DMatrix(X_train, label=y_train)
                if X_val is not None and y_val is not None:
                    dval = xgb.DMatrix(X_val, label=y_val)
                    eval_set = [(dtrain, 'train'), (dval, 'val')]
                else:
                    eval_set = [(dtrain, 'train')]
                
                # Configure CPU-specific parameters
                train_params = self.global_params.copy()
                train_params.update({
                    'tree_method': 'hist',  # Histogram-based algorithm for CPU
                    'max_bin': 256,  # Optimal for CPU training
                    'grow_policy': 'lossguide',  # More efficient tree growth
                    'max_leaves': 64,  # Control tree complexity
                    'nthread': -1  # Use all CPU cores
                })
                
                # Log parameters
                mlflow.log_params(train_params)
                
                # Train model
                self.model = xgb.train(
                    train_params,
                    dtrain,
                    evals=eval_set,
                    verbose_eval=100  # Print progress every 100 rounds
                )
                
                # Calculate feature importance
                importance_scores = self.model.get_score(importance_type='gain')
                self.feature_importance = {
                    feature: importance_scores.get(feature, 0)
                    for feature in X_train.columns
                }
                
                # Optimize threshold if validation data is available
                if X_val is not None and y_val is not None:
                    val_probs = self.model.predict(dval)
                    self.threshold = self._optimize_threshold(y_val.values, val_probs)
                    mlflow.log_param('optimal_threshold', self.threshold)
                
                # Log completion metrics and artifacts
                self._log_completion_metrics(X_val, y_val)
                
                # Log training completion
                mlflow.log_param("training_end", datetime.now().isoformat())
                self.logger.info("Model training completed successfully")
                
        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}")
            raise

    def predict_proba(
        self,
        features: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Get probability predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Validate and prepare features
        features_df = pd.DataFrame(features) if isinstance(features, np.ndarray) else features.copy()
        features_df = features_df[self.selected_features]
        
        # Convert to DMatrix
        dtest = xgb.DMatrix(features_df)
        
        # Get raw predictions
        return self.model.predict(dtest)

    def predict(
        self,
        features: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Get binary predictions using the optimal threshold."""
        probabilities = self.predict_proba(features)
        return (probabilities >= self.threshold).astype(int)

    def analyze_predictions(
        self,
        features: Union[pd.DataFrame, np.ndarray],
        target: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """Analyze model predictions."""
        # Get predictions
        y_prob = self.predict_proba(features)
        y_pred = (y_prob >= self.threshold).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': np.mean(y_pred == target),
            'precision': precision_score(target, y_pred, zero_division=0),
            'recall': recall_score(target, y_pred, zero_division=0),
            'f1': f1_score(target, y_pred, zero_division=0),
            'log_loss': log_loss(target, y_prob),
            'average_precision': average_precision_score(target, y_prob)
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
        """Save the model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'feature_importance': self.feature_importance,
            'threshold': self.threshold,
            'global_params': self.global_params,
            'selected_features': self.selected_features
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
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
        # Initialize MLflow
        mlruns_dir = setup_mlflow_tracking(experiment_name)
        
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
        
        # Train model
        logger.info("Starting model training...")
        xgb_model.train(X_train, y_train, X_test, y_test)
        
        # Evaluate on test set
        logger.info("Evaluating model on test set...")
        test_analysis = xgb_model.analyze_predictions(X_test, y_test)
        
        # Evaluate on evaluation set
        logger.info("Evaluating model on evaluation set...")
        eval_analysis = xgb_model.analyze_predictions(X_eval, y_eval)
        
        # Save model
        model_path = os.path.join(project_root, "models", "saved", f"xgboost_api_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        xgb_model.save_model(model_path)
        
        # Create input example for model signature
        input_example = X_train.head(1)
        
        # Create model signature
        signature = mlflow.models.infer_signature(
            X_train,
            xgb_model.predict(X_train)
        )
        
        # Log the model to MLflow
        mlflow.xgboost.log_model(
            xgb_model=xgb_model.model,
            artifact_path="model",
            registered_model_name=f"xgboost_api_{datetime.now().strftime('%Y%m%d_%H%M')}",
            conda_env={
                'name': 'soccerprediction_env',
                'channels': ['conda-forge'],
                'dependencies': [
                    'python=3.9',
                    'xgboost',
                    'scikit-learn',
                    'pandas',
                    'numpy'
                ]
            },
            signature=signature,
            input_example=input_example
        )
        
        logger.info("Global model training completed successfully")
        
    except Exception as e:
        logger.error(f"Error in global model training: {str(e)}")
        raise

if __name__ == "__main__":
    train_global_model()
