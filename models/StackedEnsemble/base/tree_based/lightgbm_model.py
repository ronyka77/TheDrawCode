"""LightGBM model implementation with CPU optimization."""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import lightgbm as lgb
import joblib
import json
import os
import sys
import time
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.metrics import roc_auc_score
import yaml
os.environ["ARROW_S3_DISABLE"] = "1"

from utils.logger import ExperimentLogger
from models.StackedEnsemble.base.model_interface import BaseModel
from models.StackedEnsemble.utils.metrics import calculate_metrics

class LightGBMModel(BaseModel):
    """LightGBM model with CPU optimization."""

    def __init__(
        self,
        model_type: str = 'lightgbm',
        experiment_name: str = None,
        logger: ExperimentLogger = None):
        """Initialize LightGBM model."""
        self.logger = logger or ExperimentLogger(experiment_name or f"{model_type}_experiment")
        super().__init__(
            model_type=model_type,
            experiment_name=experiment_name,
            logger=self.logger
        )
        
        # Set CPU-specific parameters
        self.n_jobs = -1  # Use all available CPU cores
        # Load hyperparameter space
        try:
            config_path = Path(__file__).parent.parent.parent / "config" / "hyperparameter_spaces" / "lightgbm_space.yaml"
            with open(config_path, 'r') as f:
                self.hyperparameter_space = yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Error loading hyperparameter space: {str(e)}")
            self.hyperparameter_space = None

    def _create_model(self, **kwargs) -> lgb.LGBMClassifier:
        """Create and configure LightGBM model instance."""
        try:
            # Get default parameters from config
            params = {
                'objective': 'binary',
                'metric': ['binary_logloss', 'auc'],
                'n_jobs': self.n_jobs,
                'random_state': 42,
                'verbose': -1
            }
            
            # Update with model config parameters
            if self.model_config:
                params.update(self.model_config.get('params', {}))
            
            # Update with provided parameters
            params.update(kwargs)
            
            # Create model
            model = lgb.LGBMClassifier(**params)
            
            self.logger.info(f"Created LightGBM model with parameters: {params}")
            return model
            
        except Exception as e:
            self.logger.error(f"Error creating LightGBM model: {str(e)}")
            raise

    def _optimize_threshold(self, X_val: Any, y_val: Any) -> float:
        """Optimize decision threshold based on precision-recall trade-off."""
        probas = self.predict_proba(X_val)
        thresholds = np.linspace(0.25, 0.8, 56)
        best_threshold = 0.3
        best_score = 0.0
        
        for threshold in thresholds:
            preds = (probas >= threshold).astype(int)
            try:
                metrics = calculate_metrics(y_val, preds, probas)
            except Exception as e:
                self.logger.error(f"Error calculating metrics: {str(e)}")
                continue
            
            if metrics['recall'] >= 0.15 and metrics['recall'] < 0.9 and metrics['precision'] > 0.30:
                score = metrics['precision']
                if score > best_score:
                    self.logger.info(f"Threshold: {threshold:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
                    best_score = score
                    best_threshold = threshold
        
        self.best_threshold = best_threshold
        self.logger.info(f"Optimized threshold: {best_threshold:.4f}")
        return best_threshold

    def _train_model(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        X_test: Any = None,
        y_test: Any = None,
        **kwargs) -> Dict[str, float]:
        """Train LightGBM model with early stopping."""
        try:
            # Extract early stopping rounds if present
            early_stopping_rounds = kwargs.pop('early_stopping_rounds', 100)
            
            # Create model with remaining parameters
            self.model = self._create_model(**kwargs)
            
            # Create eval set for early stopping
            eval_set = [(X_val, y_val)]
            
            # Fit model with early stopping
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                    lgb.log_evaluation(period=100)
                ]
            )
            
            # Get validation metrics
            metrics = self.evaluate(X_val, y_val)
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error training LightGBM model: {str(e)}")
            raise

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
        """Save LightGBM model to specified path."""
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
        """Load LightGBM model from specified path."""
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
            feature_importance = self.model.feature_importances_
            feature_names = self.model.feature_name_
            
            # Convert to DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            })
            importance_df = importance_df.sort_values(
                'importance',
                ascending=False
            ).reset_index(drop=True)
            
            return importance_df
            
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")
            return pd.DataFrame(columns=['feature', 'importance'])

    def get_params(self) -> Dict[str, Any]:
        """Get current model parameters.
        
        Returns:
            Dictionary of parameters
        """
        if self.model is not None:
            return self.model.get_params()
        return super().get_params()

    def set_params(self, **params) -> None:
        """Set model parameters.
        
        Args:
            **params: Parameters to set
        """
        if self.model is not None:
            self.model.set_params(**params)
        super().set_params(**params)

    def _convert_to_model_format(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None) -> Tuple[Any, Optional[Any]]:
        """Convert data to LightGBM format.
        
        Args:
            X: Feature matrix
            y: Optional target vector
            
        Returns:
            Tuple of (X, y) in LightGBM format
        """
        if X is None:
            raise ValueError("The feature dataset X must not be None.")
        
        # LightGBM can handle pandas DataFrames directly
        # Just ensure y is numeric if provided
        if y is not None and isinstance(y, pd.Series):
            y = y.astype(int)
        
        return X, y

    def optimize_hyperparameters(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        X_test: Any,
        y_test: Any) -> Dict[str, Any]:
        """Run hyperparameter optimization with Optuna."""
        
        self.logger.info("Starting hyperparameter optimization")
        
        if not self.hyperparameter_space:
            raise ValueError("Hyperparameter space configuration is missing")
        
        opt_config = self.hyperparameter_space.get('lightgbm', {})
        
        try:
            study = optuna.create_study(
                study_name=opt_config.get('study_name', 'lightgbm_optimization'),
                direction=opt_config.get('direction', 'maximize'),
                sampler=TPESampler(seed=opt_config.get('sampler', {}).get('seed', 42)),
                pruner=MedianPruner(
                    n_startup_trials=opt_config.get('pruner', {}).get('n_startup_trials', 5),
                    n_warmup_steps=opt_config.get('pruner', {}).get('n_warmup_steps', 2),
                    interval_steps=opt_config.get('pruner', {}).get('interval_steps', 1)
                )
            )
            
            # Run optimization
            study.optimize(
                lambda trial: self._objective(trial, X_train, y_train, X_val, y_val),
                n_trials=opt_config.get('n_trials', 100),
                timeout=opt_config.get('timeout', 7200),
                show_progress_bar=True
            )
            
            # Get best parameters and add fixed parameters
            best_params = study.best_params
            best_params.update({
                'objective': 'binary',
                'metric': ['binary_logloss', 'auc'],
                'verbose': -1,
                'n_jobs': self.n_jobs,
                'random_state': 42,
                'device': 'cpu'
            })
            
            self.logger.info(f"Best trial value: {study.best_value}")
            self.logger.info(f"Best parameters found: {best_params}")
            return best_params
                
        except Exception as e:
            self.logger.error(f"Error in hyperparameter optimization: {str(e)}")
            raise

    def fit(
        self,
        X: Any,
        y: Any,
        X_val: Any,
        y_val: Any,
        X_test: Optional[Any] = None,
        y_test: Optional[Any] = None,
        **kwargs) -> Dict[str, float]:
        """Train the model with early stopping if test data is provided.
        
        Args:
            X: Training features
            y: Training labels
            X_val: Validation features
            y_val: Validation labels
            X_test: Test features for stopping
            y_test: Test labels for stopping
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary of training metrics
            
        Raises:
            ValueError: If input data is invalid or empty
            RuntimeError: If model training fails
        """
        try:
            self.logger.info("Starting model training")
            
            # Validate input data
            if X is None or y is None or X_val is None or y_val is None:
                raise ValueError("Training and validation data must be provided")
            if len(X) == 0 or len(y) == 0 or len(X_val) == 0 or len(y_val) == 0:
                raise ValueError("Input data cannot be empty")
                
            # Create fresh model instance
            try:
                self.model = self._create_model(**kwargs)
            except Exception as e:
                raise RuntimeError(f"Failed to create model: {str(e)}")
            
            # Train model with error handling
            try:
                train_metrics = self._train_model(X, y, X_val, y_val, X_test, y_test)
            except Exception as e:
                raise RuntimeError(f"Model training failed: {str(e)}")
            
            return train_metrics
            
        except Exception as e:
            self.logger.error(f"Error in fit(): {str(e)}")
            raise  # Re-raise the exception after logging

    def predict(self, X: Any) -> np.ndarray:
        """Make predictions using the trained model.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of predictions
        """
        return self._predict_model(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Get prediction probabilities.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of prediction probabilities
        """
        return self._predict_proba_model(X)

    def evaluate(
        self,
        X_val: Any,
        y_val: Any,
        optimize_threshold: bool = True) -> Dict[str, float]:
        """Evaluate model performance.
        
        Args:
            X_val: Features to evaluate on
            y_val: True labels
            optimize_threshold: Whether to optimize decision threshold
            
        Returns:
            Dictionary of evaluation metrics
        """
        if optimize_threshold:
            threshold = self._optimize_threshold(X_val, y_val)
        else:
            threshold = 0.3
        
        metrics = self._calculate_metrics(X_val, y_val, threshold)
        return metrics

    def _calculate_metrics(self, X: Any, y: Any, threshold: float = 0.3) -> Dict[str, float]:
        """Calculate performance metrics.
        
        Args:
            X: Features
            y: True labels
            threshold: Decision threshold
            
        Returns:
            Dictionary of metrics
        """
        if threshold is None:
            threshold = 0.3
            
        probas = self.predict_proba(X)
        preds = (probas >= threshold).astype(int)
        try:
            metrics = calculate_metrics(y, preds, probas)
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            return None
        return metrics

    def _objective(self, trial: optuna.Trial, X_train: Any, y_train: Any, X_val: Any, y_val: Any) -> float:
        """Optuna objective function for hyperparameter optimization."""
        try:
            # Define hyperparameter space
            params = {
                'objective': 'binary',
                'metric': ['binary_logloss', 'auc'],
                'verbose': -1,
                'n_jobs': self.n_jobs,
                'random_state': 42,
                'device': 'cpu'
            }
            
            # Add hyperparameters from config
            for param_name, param_config in self.hyperparameter_space['hyperparameters'].items():
                if param_config['distribution'] == 'log_uniform':
                    params[param_name] = trial.suggest_float(
                        param_name, 
                        float(param_config['min']), 
                        float(param_config['max']), 
                        log=True
                    )
                elif param_config['distribution'] == 'uniform':
                    params[param_name] = trial.suggest_float(
                        param_name,
                        float(param_config['min']),
                        float(param_config['max'])
                    )
                elif param_config['distribution'] == 'int_uniform':
                    params[param_name] = trial.suggest_int(
                        param_name,
                        int(param_config['min']),
                        int(param_config['max'])
                    )

            # Train model and get metrics
            metrics = self._train_model(
                X_train, y_train,
                X_val, y_val,
                **params
            )
            
            # Calculate score based on precision if recall threshold met
            recall = metrics.get('recall', 0.0)
            precision = metrics.get('precision', 0.0)
            
            # Report intermediate values for pruning
            trial.report(precision, step=1)
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            # Optimize for precision while maintaining minimum recall
            score = precision if recall >= 0.15 else 0.0
            
            # Log trial results
            self.logger.info(f"Trial {trial.number}:")
            self.logger.info(f"  Params: {params}")
            self.logger.info(f"  Metrics: {metrics}")
            self.logger.info(f"  Score: {score}")
            
            return score
            
        except Exception as e:
            self.logger.error(f"Trial failed: {str(e)}")
            return 0.0 