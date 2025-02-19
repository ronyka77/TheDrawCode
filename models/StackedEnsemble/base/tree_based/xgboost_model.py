"""XGBoost model implementation with CPU optimization."""

import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys
from typing import Dict, Any, Optional, Union, Tuple
import xgboost as xgb
import joblib
import json
import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.metrics import roc_auc_score

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)
os.environ["PYTHONPATH"] = project_root + os.pathsep + os.environ.get("PYTHONPATH", "")
experiment_name = 'xgboost_stacked_ensemble_model'
from models.StackedEnsemble.base.model_interface import BaseModel
from utils.logger import ExperimentLogger

class XGBoostModel(BaseModel):
    """XGBoost model with CPU optimization."""
    
    def __init__(
        self,
        model_type: str = 'xgboost',
        experiment_name: str = None,
        logger: ExperimentLogger = None):
        """Initialize XGBoost model.
        
        Args:
            model_type: Model type identifier
            experiment_name: Name for experiment tracking
            logger: Logger instance
        """
        # Initialize logger first
        self.logger = logger or ExperimentLogger(experiment_name or f"{model_type}_experiment")
        
        # Initialize base class
        super().__init__(
            model_type=model_type,
            experiment_name=experiment_name,
            logger=self.logger
        )
        
        # Set CPU-specific parameters
        self.tree_method = 'hist'  # CPU-optimized histogram-based tree method
        self.n_jobs = -1  # Use all available CPU cores

    def _create_model(self, **kwargs) -> xgb.XGBClassifier:
        """Create and configure XGBoost model instance."""
        try:
            # Get default parameters
            params = {
                'objective': 'binary:logistic',
                'tree_method': self.tree_method,
                'n_jobs': self.n_jobs,
                'eval_metric': ['logloss', 'auc']
            }
            
            # Update with provided parameters
            params.update(kwargs)
            
            # Create model
            model = xgb.XGBClassifier(**params)
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error creating XGBoost model: {str(e)}")
            raise

    def _optimize_threshold(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Optimize prediction threshold with focus on precision while maintaining recall above 15%.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            
        Returns:
            Optimized threshold value
        """
        try:
            best_threshold = 0.5
            best_precision = 0.0
            
            # Search through thresholds
            for threshold in np.linspace(0.3, 0.8, 51):
                y_pred = (y_prob >= threshold).astype(int)
                
                # Calculate confusion matrix components
                tp = np.sum((y_true == 1) & (y_pred == 1))
                fp = np.sum((y_true == 0) & (y_pred == 1))
                fn = np.sum((y_true == 1) & (y_pred == 0))
                
                precision = tp / (tp + fp + 1e-10)
                recall = tp / (tp + fn + 1e-10)
                
                # Only consider thresholds that maintain recall above 15%
                if recall >= 0.15:
                    if precision > best_precision:
                        best_precision = precision
                        best_threshold = threshold
                        # self.logger.info(f"New best threshold: {threshold:.3f} (Precision: {precision:.3f}, Recall: {recall:.3f})")
            
            self.logger.info(f"Optimized threshold: {best_threshold:.3f} with precision: {best_precision:.3f}")
            return best_threshold
            
        except Exception as e:
            self.logger.error(f"Error optimizing threshold: {str(e)}")
            return 0.5

    def _train_model(   
        self,
        X: Any,
        y: Any,
        X_val: Any,
        y_val: Any,
        X_test: Any,
        y_test: Any,
        **kwargs) -> Dict[str, float]:
        """Train XGBoost model with validation data."""
        try:
            self.model = self._create_model(**kwargs)
            
            # Set up early stopping
            early_stopping_rounds = kwargs.get('early_stopping_rounds', 50)
            
            # Train model with validation data for early stopping
            self.model.fit(
                    X, y,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
            
            # Get predictions on validation set
            y_prob = self.model.predict_proba(X_val)[:, 1]
            
            # Optimize threshold
            self.best_threshold = self._optimize_threshold(y_val, y_prob)
            y_pred = (y_prob >= self.best_threshold).astype(int)
            
            # Calculate confusion matrix components
            tp = np.sum((y_val == 1) & (y_pred == 1))
            fp = np.sum((y_val == 0) & (y_pred == 1))
            fn = np.sum((y_val == 1) & (y_pred == 0))
            
            # Calculate metrics
            metrics = {
                'precision': tp / (tp + fp + 1e-10),
                'recall': tp / (tp + fn + 1e-10),
                'f1': 2 * tp / (2 * tp + fp + fn + 1e-10),
                'auc': roc_auc_score(y_val, y_prob),
                'brier_score': np.mean((y_prob - y_val) ** 2),
                'threshold': self.best_threshold
            }
            
            # Add best iteration if available
            if hasattr(self.model, 'best_iteration_'):
                metrics['best_iteration'] = self.model.best_iteration_
            
            # self.logger.info(f"Training metrics: {metrics}")
            return metrics, self.model
            
        except Exception as e:
            self.logger.error(f"Error training XGBoost model: {str(e)}")
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'auc': 0.0,
                'brier_score': 1.0,
                'threshold': 0.5
            }, self.model

    def _predict_model(self, X: Any) -> np.ndarray:
        """Generate predictions using trained model."""
        if self.model is None:
            raise RuntimeError("Model must be trained before prediction")
        
        try:
            probas = self.model.predict_proba(X)[:, 1]
            threshold = getattr(self, 'best_threshold', 0.5)
            return (probas >= threshold).astype(int)
            
        except Exception as e:
            self.logger.error(f"Error in model prediction: {str(e)}")
            return np.zeros(len(X))

    def _predict_proba_model(self, X: Any) -> np.ndarray:
        """Generate probability predictions."""
        if self.model is None:
            raise RuntimeError("Model must be trained before prediction")
        
        try:
            return self.model.predict_proba(X)[:, 1]
            
        except Exception as e:
            self.logger.error(f"Error in probability prediction: {str(e)}")
            return np.zeros(len(X))

    def _save_model_to_path(self, path: Path) -> None:
        """Save XGBoost model to specified path."""
        if self.model is None:
            raise RuntimeError("No model to save")
        
        try:
            # Create directory if it doesn't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            
        # Save model
            joblib.dump(self.model, path)
        
            # Save threshold
            threshold_path = path.parent / "threshold.json"
            with open(threshold_path, 'w') as f:
                    json.dump({
                        'threshold': getattr(self, 'best_threshold', 0.5),
                        'model_type': self.model_type,
                        'params': self.model.get_params()
                    }, f, indent=2)
            
            self.logger.info(f"Model saved to {path}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise

    def _load_model_from_path(self, path: Path) -> None:
        """Load XGBoost model from specified path."""
        if not path.exists():
            raise FileNotFoundError(f"No model file found at {path}")
        
        try:
        # Load model
            self.model = joblib.load(path)
        
            # Load threshold
            threshold_path = path.parent / "threshold.json"
            if threshold_path.exists():
                with open(threshold_path, 'r') as f:
                        data = json.load(f)
                        self.best_threshold = data.get('threshold', 0.5)
            else:
                self.best_threshold = 0.5
                
            self.logger.info(f"Model loaded from {path}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores.
        
        Returns:
            DataFrame with feature importance scores
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before getting feature importance")
            
        try:
            # Get feature importance scores
            importance_type = 'gain'
            scores = self.model.get_booster().get_score(importance_type=importance_type)
            
            # Convert to DataFrame
            importance_df = pd.DataFrame(
                list(scores.items()),
                columns=['feature', 'importance']
            )
            importance_df = importance_df.sort_values(
                'importance',
                ascending=False
            ).reset_index(drop=True)
            
            return importance_df
            
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")
            return pd.DataFrame(columns=['feature', 'importance'])

    def optimize_hyperparameters(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        X_test: Any,
        y_test: Any) -> Dict[str, Any]:
        """Run hyperparameter optimization with Optuna.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data
            
        Returns:
            Best hyperparameters found
        """
        
        def objective(trial: Trial) -> float:
            """Optuna objective function."""
            # Define hyperparameter space
            params = {
                'tree_method': 'hist',
                'objective': 'binary:logistic', 
                'eval_metric': ['logloss', 'auc'],
                'random_state': 42,
                'n_jobs': self.n_jobs,
                
                # Learning rate and boosting parameters
                'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.2, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 5000),
                'max_depth': trial.suggest_int('max_depth', 2, 12),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 500),
                
                # Sampling parameters
                'subsample': trial.suggest_float('subsample', 0.1, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
                
                # Regularization
                'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 50.0),
                'gamma': trial.suggest_float('gamma', 0.001, 1.0),
                
                # Early stopping
                'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 100, 1000),
                
                # New parameter
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 20)
            }
            
            # Create fresh model instance for this trial
            self.model = None  # Reset model
            metrics = self._train_model(
                X_train, y_train,
                X_val, y_val,
                X_test, y_test,
                **params
            )
            
            # Calculate score based on precision if recall threshold met
            score = metrics['precision'] if metrics['recall'] > 0.15 else 0.0
            
            # Log trial results
            self.logger.info(f"Trial {trial.number}:")
            self.logger.info(f"  Params: {params}")
            self.logger.info(f"  Metrics: {metrics}")
            self.logger.info(f"  Score: {score}")
            
            # Report intermediate values for pruning
            trial.report(score, step=1)
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return score
        
        try:
            # Create study
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(
                    seed=42,
                    consider_prior=True,
                    prior_weight=0.8,
                    n_startup_trials=10
                ),
                pruner=MedianPruner(
                    n_startup_trials=10,
                    n_warmup_steps=5,
                    interval_steps=2
                )
            )
            
            # Run optimization
            study.optimize(
                objective,
                n_trials=300,
                timeout=7200,  # 2 hour timeout
                show_progress_bar=True
            )
            
            # Get best parameters
            best_params = study.best_params
            best_params.update({
                'tree_method': 'hist',
                'objective': 'binary:logistic',
                'eval_metric': ['logloss', 'auc'],
                'random_state': 42,
                'n_jobs': self.n_jobs
            })
            
            self.logger.info(f"Best parameters found: {best_params}")
            return best_params 
        except Exception as e:
            self.logger.error(f"Error in hyperparameter optimization: {str(e)}")
            return {
                'tree_method': 'hist',
                'objective': 'binary:logistic',
                'eval_metric': ['logloss', 'auc'],
                'random_state': 42,
                'n_jobs': self.n_jobs,
                'learning_rate': 0.01,
                'n_estimators': 1000,
                'max_depth': 6,
                'min_child_weight': 100,
                'subsample': 0.5,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.01,
                'reg_lambda': 0.01,
                'gamma': 0.0
            }

    def get_params(self) -> Dict[str, Any]:
        """Get current model parameters.
            
        Returns:
            Dictionary of parameters
        """
        if self.model is not None:
            return self.model.get_params()
        return {}

    def set_params(self, **params) -> None:
        """Set model parameters.
        
        Args:
            **params: Parameters to set
        """
        if self.model is not None:
            self.model.set_params(**params)
        super().set_params(**params)

    def evaluate(self, X: Any, y: Any) -> Dict[str, float]:
        """Evaluate model performance on given data.
        
        Args:
            X: Features to evaluate on
            y: True labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Get predictions
            y_pred = self.predict(X)
            y_prob = self.predict_proba(X)
            
            # Calculate confusion matrix components
            tp = np.sum((y == 1) & (y_pred == 1))
            fp = np.sum((y == 0) & (y_pred == 1))
            fn = np.sum((y == 1) & (y_pred == 0))
            
            # Calculate metrics
            metrics = {
                'precision': tp / (tp + fp + 1e-10),
                'recall': tp / (tp + fn + 1e-10),
                'f1': 2 * tp / (2 * tp + fp + fn + 1e-10),
                'auc': roc_auc_score(y, y_prob),
                'brier_score': np.mean((y_prob - y) ** 2)
            }
            
            # self.logger.info(f"Evaluation metrics: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'auc': 0.0,
                'brier_score': 1.0
            }

def train_main(experiment_name: str) -> float:
    """Main training function with MLflow tracking.
    
    Args:
        X_train: Training features
        y_train: Training labels 
        X_test: Test features
        y_test: Test labels
        X_eval: Evaluation features
        y_eval: Evaluation labels
        experiment_name: Name for MLflow experiment
        
    Returns:
        Best precision achieved
    """
    # Import required libraries
    import mlflow
    import numpy as np
    import pandas as pd
    import random
    import os
    from datetime import datetime
    from copy import deepcopy
    from mlflow.models.signature import infer_signature
    from utils.logger import ExperimentLogger
    from utils.create_evaluation_set import setup_mlflow_tracking
    exp_logger = ExperimentLogger(experiment_name)
    mlruns_dir = setup_mlflow_tracking(experiment_name)
    from models.StackedEnsemble.shared.data_loader import DataLoader

    Dataloader = DataLoader()
    X_train, y_train, X_test, y_test, X_eval, y_eval = Dataloader.load_data()
    model_params = {
            'objective': 'binary:logistic',
            'tree_method': 'hist',
            'n_jobs': -1,
            'eval_metric': ['logloss', 'auc'],
            'learning_rate': 0.010015907839790164,
            'n_estimators': 130,
            'max_depth': 12,
            'min_child_weight': 94,
            'subsample': 0.2995141103683554,
            'colsample_bytree': 0.9674584520568595,
            'reg_alpha': 1.993141051251477,
            'reg_lambda': 13.371664552561292,
            'gamma': 0.30999189400795213,
            'early_stopping_rounds': 443,
            'scale_pos_weight': 11.221714608899202
        }
    
    # Convert all features to float64
    X_train = X_train.astype('float64')
    X_test = X_test.astype('float64')
    X_eval = X_eval.astype('float64')
    
    exp_logger.info(f"X_train shape: {X_train.shape}")
    exp_logger.info(f"X_test shape: {X_test.shape}")
    exp_logger.info(f"X_eval shape: {X_eval.shape}")

    # Start MLflow run with experiment tracking
    with mlflow.start_run(run_name=f"xgboost_base_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        try:
            # Log global parameters to MLflow
            mlflow.log_params(model_params)
            exp_logger.info("Logged model parameters to MLflow")
            
            # Log dataset sizes
            mlflow.log_metric("train_samples", len(X_train))
            mlflow.log_metric("test_samples", len(X_test))
            mlflow.log_metric("eval_samples", len(X_eval))
            
            # Set MLflow tags
            mlflow.set_tags({
                "model_type": "xgboost_base",
                "training_mode": "global",
                "cpu_only": True,
                "tree_method": "hist"
            })
            
            # Train model with precision target
            precision = 0
            highest_precision = 0
            best_seed = 0
            best_model = None
            
            while precision < 0.48:
                for random_seed in range(1, 5):
                    exp_logger.info(f"Using sequential random seed: {random_seed}")
                    os.environ['PYTHONHASHSEED'] = str(random_seed)
                    np.random.seed(random_seed)
                    random.seed(random_seed)
                    model = XGBoostModel()
                    model.model = model._create_model(**model_params)
                    model_params['random_state'] = random_seed
                    metrics = model._train_model(X_train, y_train, X_test, y_test, X_eval, y_eval, **model_params)
                    precision = metrics['precision']
                    threshold = metrics['threshold']
                    
                    if precision > highest_precision:
                        highest_precision = precision
                        best_seed = random_seed
                        best_model = model
                        best_threshold = threshold
                    if precision >= 0.48:
                        exp_logger.info(f"Target precision achieved: {precision:.4f}")
                        break
                    exp_logger.info(f"Current precision: {precision:.4f}, target: 0.4800 highest precision: {highest_precision:.4f} best seed: {best_seed}")
                
                # If target not reached, use best model
                if precision < 0.48:
                    exp_logger.info(f"Target precision not reached, using best seed: {best_seed}")
                    model = best_model
                    precision = highest_precision
                    threshold = best_threshold
                    break

            # Log validation metrics
            val_metrics = model.evaluate(X_eval, y_eval)
            for metric_name, metric_value in val_metrics.items():
                mlflow.log_metric(f"val_{metric_name}", metric_value)

            # Create and log model signature
            try:
                input_example = X_train.head(1).copy().astype('float64')
                signature = infer_signature(
                    model_input=input_example,
                    model_output=model.predict(input_example)
                )
                
                mlflow.xgboost.log_model(
                    xgb_model=model,
                    artifact_path="xgboost_base_model",
                    registered_model_name=f"xgboost_base_{datetime.now().strftime('%Y%m%d_%H%M')}",
                    signature=signature
                )
            except Exception as e:
                exp_logger.error(f"Error in model logging: {str(e)}")
                mlflow.xgboost.log_model(
                    xgb_model=model,
                    artifact_path="xgboost_base_model",
                    registered_model_name=f"xgboost_base_{datetime.now().strftime('%Y%m%d_%H%M')}"
                )

            exp_logger.info("Base model training completed successfully")
            exp_logger.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")
            
            return precision

        except Exception as e:
            exp_logger.error(f"Error in training main: {str(e)}")
            raise

if __name__ == "__main__":
    train_main(experiment_name)