"""
XGBoost training module for the Model Context Protocol Server.

This module implements:
- CPU-only XGBoost model training
- MLflow integration for tracking
- Hyperparameter tuning support
- Error handling and logging
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime

# Third-party imports
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import precision_score, recall_score, f1_score
import mlflow
from mlflow.models.signature import infer_signature

# Add project root to path
try:
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
except Exception as e:
    print(f"Error setting project root path: {e}")
    sys.path.append(os.getcwd())

# Local imports
from utils.logger import ExperimentLogger
from data.data_ingestion import DataIngestion
from utils.mlflow_utils import MLFlowManager

# Configure XGBoost for CPU-only training
xgb.set_config(verbosity=2)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU

# Initialize logger
logger = ExperimentLogger(
    experiment_name="xgboost_training",
    log_dir="logs/xgboost_training"
)

class XGBoostTrainer(BaseEstimator, ClassifierMixin):
    """XGBoost model trainer with MLflow integration."""
    
    def __init__(
        self,
        run_id: str,
        hyperparameters: Optional[Dict[str, Any]] = None
    ):
        """Initialize trainer with MLflow run ID and optional hyperparameters."""
        self.run_id = run_id
        
        # Default hyperparameters (can be overridden)
        self.hyperparameters = {
            'learning_rate': 0.02190378119690766,
            'early_stopping_rounds': 500,
            'min_child_weight': 165,
            'gamma': 0.06268817081995443,
            'subsample': 0.39841425636563177,
            'colsample_bytree': 0.9311323559904169,
            'scale_pos_weight': 2.003902849861491,
            'reg_alpha': 0.0004175183986636507,
            'reg_lambda': 1.9321415795995613,
            'max_depth': 5,
            'n_estimators': 556,
            'objective': 'binary:logistic',
            'tree_method': 'hist',  # CPU-only
            'device': 'cpu',
            'eval_metric': ['error', 'auc', 'aucpr'],
            'verbosity': 0,
            'nthread': -1
        }
        
        # Update with provided hyperparameters
        if hyperparameters:
            self.hyperparameters.update(hyperparameters)
            
        # Initialize components
        self.model = None
        self.feature_importance = {}
        self.threshold = 0.50  # Default threshold
        self.data_ingestion = DataIngestion('xgb')
        
    def _optimize_threshold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, float]:
        """Find optimal prediction threshold prioritizing precision.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            
        Returns:
            Dictionary with optimal threshold and metrics
        """
        try:
            best_metrics = {
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'threshold': self.threshold
            }
            best_score = 0
            
            # Focus on higher thresholds for better precision
            for threshold in np.arange(0.5, 0.65, 0.01):
                preds = (y_prob >= threshold).astype(int)
                recall = recall_score(y_true, preds)
                
                # Only consider thresholds meeting minimum recall
                if recall >= 0.20:
                    precision = precision_score(y_true, preds)
                    f1 = f1_score(y_true, preds)
                    
                    # Score prioritizes precision while maintaining recall
                    score = precision * min(1.0, (recall - 0.20) / 0.20)
                    
                    if score > best_score:
                        best_score = score
                        best_metrics.update({
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'threshold': threshold
                        })
                        
                        logger.info(
                            f"New best threshold {threshold:.3f}: "
                            f"Precision={precision:.4f}, Recall={recall:.4f}"
                        )
                        
                        # Log to MLflow
                        mlflow.log_metrics({
                            'best_threshold': threshold,
                            'best_precision': precision,
                            'best_recall': recall,
                            'best_f1': f1
                        })
            
            if best_metrics['recall'] < 0.20:
                logger.warning(
                    f"Could not find threshold meeting recall requirement. "
                    f"Best recall: {best_metrics['recall']:.4f}"
                )
                
            return best_metrics
            
        except Exception as e:
            logger.error(f"Error in threshold optimization: {str(e)}")
            raise
            
    def train(self) -> None:
        """Train the XGBoost model with MLflow tracking."""
        try:
            # Load and validate data
            X_train, y_train, X_test, y_test = self.data_ingestion.load_training_data()
            X_eval, y_eval = self.data_ingestion.load_evaluation_data()
            
            # Convert to MLflow-compatible format
            X_train = self.data_ingestion.convert_to_mlflow_format(X_train)
            X_test = self.data_ingestion.convert_to_mlflow_format(X_test)
            X_eval = self.data_ingestion.convert_to_mlflow_format(X_eval)
            
            # Initialize and train model
            self.model = xgb.XGBClassifier(**self.hyperparameters)
            
            # Train with early stopping
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=100
            )
            
            # Get feature importance
            self.feature_importance = self.model.get_booster().get_score(
                importance_type='gain'
            )
            
            # Optimize threshold on validation set
            val_probs = self.model.predict_proba(X_eval)[:, 1]
            best_metrics = self._optimize_threshold(y_eval.values, val_probs)
            self.threshold = best_metrics['threshold']
            
            # Log model and metrics to MLflow
            with mlflow.start_run(run_id=self.run_id):
                # Log hyperparameters
                mlflow.log_params(self.hyperparameters)
                
                # Log metrics
                mlflow.log_metrics({
                    'final_threshold': self.threshold,
                    'n_features': X_train.shape[1],
                    'n_training_samples': len(X_train),
                    'n_validation_samples': len(X_eval)
                })
                
                # Log feature importance
                mlflow.log_dict(
                    self.feature_importance,
                    "feature_importance.json"
                )
                
                # Create model signature
                try:
                    signature = infer_signature(X_train, self.model.predict(X_train))
                except Exception as e:
                    logger.warning(f"Could not infer MLflow signature: {str(e)}")
                    signature = None
                
                # Log model
                if signature:
                    mlflow.xgboost.log_model(
                        self.model,
                        artifact_path="model",
                        signature=signature
                    )
                else:
                    mlflow.xgboost.log_model(self.model, "model")
                    
                logger.info(
                    "Training completed successfully",
                    extra={
                        "run_id": self.run_id,
                        "threshold": self.threshold,
                        "metrics": best_metrics
                    }
                )
                
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
            
    def predict_proba(
        self,
        features: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Get probability predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        # Convert to DataFrame if needed
        features_df = pd.DataFrame(features) if isinstance(
            features, np.ndarray
        ) else features.copy()
        
        return self.model.predict_proba(features_df)
        
    def predict(
        self,
        features: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Get binary predictions using optimal threshold."""
        probabilities = self.predict_proba(features)
        return (probabilities[:, 1] >= self.threshold).astype(int)

def train_model(
    run_id: str,
    hyperparameters: Optional[Dict[str, Any]] = None
) -> None:
    """Train XGBoost model with given configuration.
    
    Args:
        run_id: MLflow run ID
        hyperparameters: Optional hyperparameters to override defaults
    """
    try:
        trainer = XGBoostTrainer(run_id, hyperparameters)
        trainer.train()
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise 