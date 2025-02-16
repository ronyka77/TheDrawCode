"""LightGBM model implementation with CPU optimization and hyperparameter tuning."""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import lightgbm as lgb
import joblib
import json
import os
import ray.tune as tune
import mlflow
import sys
import ray
import time
os.environ["ARROW_S3_DISABLE"] = "1"
os.environ["RAY_DEDUP_LOGS"] = "0"

from utils.logger import ExperimentLogger
from models.StackedEnsemble.base.model_interface import BaseModel
from models.StackedEnsemble.utils.metrics import calculate_metrics
from models.StackedEnsemble.shared.validation import NestedCVValidator
from models.StackedEnsemble.shared.mlflow_utils import MLFlowManager

# Global validator actor name
VALIDATOR_ACTOR_NAME = "global_validator"

@ray.remote
class ValidatorActor:
    """Ray actor for validation that ensures single instance."""
    
    def __init__(self, logger=None, model_type='lightgbm'):
        """Initialize validator actor.
        
        Args:
            logger: Logger instance
            model_type: Model type
        """
        # Create a new logger instance for the actor
        self.logger = logger or ExperimentLogger('lightgbm_hypertuning')
        self.validator = NestedCVValidator(logger=self.logger, model_type=model_type)
        self.logger.info("Created new validator instance")
        
    def optimize_hyperparameters(self, model, X, y, X_val, y_val, X_test, y_test, param_space, search_strategy):
        """Run hyperparameter optimization."""
        try:
            # Ensure logger is available
            if not hasattr(self, 'logger') or self.logger is None:
                self.logger = ExperimentLogger('lightgbm_hypertuning')
                self.validator.logger = self.logger
            
            self.logger.info("Starting hyperparameter optimization in validator actor")
            result = self.validator.optimize_hyperparameters(
                model, X, y, X_val, y_val, X_test, y_test, param_space, search_strategy
            )
            self.logger.info("Completed hyperparameter optimization in validator actor")
            return result
        except Exception as e:
            if hasattr(self, 'logger') and self.logger is not None:
                self.logger.error(f"Error in optimize_hyperparameters: {str(e)}")
            return self._get_default_params(param_space)
    
    def _get_default_params(self, param_space):
        """Get default parameters if optimization fails."""
        try:
            # Ensure logger is available
            if not hasattr(self, 'logger') or self.logger is None:
                self.logger = ExperimentLogger('lightgbm_hypertuning')
                self.validator.logger = self.logger
            
            self.logger.info("Getting default parameters")
            params = self.validator._get_default_params(param_space)
            self.logger.info(f"Retrieved default parameters: {params}")
            return params
        except Exception as e:
            if hasattr(self, 'logger') and self.logger is not None:
                self.logger.error(f"Error getting default parameters: {str(e)}")
            return {
                'device': 'cpu',
                'num_threads': -1,
                'objective': 'binary',
                'metric': ['binary_logloss', 'auc'],
                'verbose': -1,
                'deterministic': True,
                'random_state': 19,
                'learning_rate': 0.01,
                'num_leaves': 31,
                'max_depth': 6,
                'min_data_in_leaf': 20,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'early_stopping_rounds': 100
            }
    
    def get_info(self):
        """Get validator info."""
        return "Validator is set up"

class LightGBMModel(BaseModel):
    """LightGBM model implementation with CPU optimization."""
    
    _validator_actor = None  # Class-level validator actor reference
    
    @classmethod
    def get_validator_actor(cls, logger=None):
        """Get or create the validator actor."""
        if cls._validator_actor is None:
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    # Try to get existing actor
                    cls._validator_actor = ray.get_actor(VALIDATOR_ACTOR_NAME)
                    logger.info("Retrieved existing validator actor")
                    break
                except ValueError:
                    try:
                        # Create new actor with proper options
                        cls._validator_actor = ValidatorActor.options(
                            name=VALIDATOR_ACTOR_NAME,
                            lifetime="detached",  # Keep actor alive across failures
                            max_restarts=-1,  # Unlimited restarts
                            max_task_retries=3  # Retry failed tasks
                        ).remote(logger)
                        logger.info("Created new validator actor")
                        break
                    except Exception as e:
                        retry_count += 1
                        logger.warning(f"Attempt {retry_count} to create validator actor failed: {str(e)}")
                        if retry_count == max_retries:
                            raise RuntimeError("Failed to create validator actor after maximum retries")
                        time.sleep(1)  # Wait before retrying
        
        return cls._validator_actor

    def __init__(self, experiment_name: str = "lightgbm_experiment", model_type: str = "lightgbm", logger: ExperimentLogger = ExperimentLogger('lightgbm_experiment')):
        """Initialize LightGBM model.
        
        Args:
            experiment_name: Name for MLflow experiment tracking
            model_type: Type of model (e.g., 'lightgbm')
            logger: Logger instance
        """
        # Get project root path
        project_root = Path(__file__).parent.parent.parent.parent.parent
        
        # Set up configuration paths
        self.config_path = os.path.join(
            project_root,
            "models",
            "StackedEnsemble",
            "config",
            "model_configs",
            "lightgbm_config.yaml"
        )
        
        # Initialize base class
        super().__init__(model_type=model_type, experiment_name=experiment_name)
        self.logger = logger
        self.mlflow = MLFlowManager(experiment_name)
        self.model = None
        self.best_threshold = 0.3
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(
                num_cpus=os.cpu_count(),
                ignore_reinit_error=True,
                include_dashboard=False,
                log_to_driver=True
            )
            self.logger.info("Ray initialized for hyperparameter tuning")
        
        # Get or create validator actor
        self.validator = self.get_validator_actor(self.logger)
        
        # Load model configuration
        self.model_config = self.config_loader.load_model_config(model_type)
        self.hyperparameter_space = self.config_loader.load_hyperparameter_space(model_type)
        self.best_params = {}
        self.best_score = 0
        
        self.logger.info(f"Initialized {model_type} model with experiment name: {experiment_name}")
    
    def _create_model(self, **kwargs) -> lgb.LGBMClassifier:
        """Create and configure LightGBM model instance.
        
        Args:
            **kwargs: Model parameters to override defaults
            
        Returns:
            Configured LightGBM classifier
            
        Raises:
            ValueError: If invalid parameters are provided
            RuntimeError: If model creation fails
        """
        try:
            # Start with CPU-optimized defaults
            params = {
                'device': 'cpu',
                'num_threads': -1,
                'objective': 'binary',
                'metric': ['binary_logloss', 'auc'],
                'verbose': -1,
                'deterministic': True,
                'random_state': 19
            }
            
            # Validate model config exists and has params
            if self.model_config:
                if not isinstance(self.model_config, dict):
                    raise ValueError("Model config must be a dictionary")
                config_params = self.model_config.get('params', {})
                if not isinstance(config_params, dict):
                    raise ValueError("Model config params must be a dictionary")
                params.update(config_params)
            
            # Validate and update with provided parameters
            if kwargs:
                if not isinstance(kwargs, dict):
                    raise ValueError("Keyword arguments must be a dictionary")
                for key, value in kwargs.items():
                    if value is None:
                        raise ValueError(f"Parameter {key} cannot be None")
                params.update(kwargs)
            
            # Create and return model
            try:
                return lgb.LGBMClassifier(**params)
            except Exception as e:
                raise RuntimeError(f"Failed to create LightGBM model: {str(e)}")
                
        except Exception as e:
            self.logger.error(f"Error in _create_model: {str(e)}")
            raise

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

    def _train_model(
        self,
        X: Any,
        y: Any,
        X_val: Any,
        y_val: Any,
        X_test: Any,
        y_test: Any) -> Dict[str, float]:
        """Train LightGBM model with early stopping.
        
        Args:
            X: Training features
            y: Training labels
            X_val: Validation features
            y_val: Validation labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of training metrics
        """
        # Convert data to LightGBM format
        X_train, y_train = self._convert_to_model_format(X, y)
        
        if X_val is not None and y_val is not None:
            X_val, y_val = self._convert_to_model_format(X_val, y_val)
        
        if X_test is not None and y_test is not None:
            X_test, y_test = self._convert_to_model_format(X_test, y_test)
        
        # Set up validation if provided
        eval_set = [(X_test, y_test)] if X_test is not None else None
        
        # Train model with early stopping
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set
        )
        
        # Evaluate on validation set
        metrics = self.evaluate(X_val, y_val)
        return metrics

    def _predict_model(self, X: Any) -> np.ndarray:
        """Generate predictions using trained model.
        
        Args:
            X: Features to predict
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before prediction")
        
        X_lgb, _ = self._convert_to_model_format(X)
        probas = self.model.predict_proba(X_lgb)[:, 1]
        return (probas >= self.best_threshold).astype(int)

    def _predict_proba_model(self, X: Any) -> np.ndarray:
        """Generate probability predictions.
        
        Args:
            X: Features to predict
            
        Returns:
            Array of probability predictions
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before prediction")
        
        X_lgb, _ = self._convert_to_model_format(X)
        return self.model.predict_proba(X_lgb)[:, 1]

    def optimize_hyperparameters(
        self,
        X: Any,
        y: Any,
        X_val: Any,
        y_val: Any,
        X_test: Any,
        y_test: Any,
        **kwargs) -> Dict[str, Any]:
        """Optimize hyperparameters using nested cross-validation.
        
        Args:
            X: Training features
            y: Training labels
            X_val: Validation features
            y_val: Validation labels
            X_test: Test features
            y_test: Test labels
            **kwargs: Additional parameters for optimization
            
        Returns:
            Dictionary of best hyperparameters
        """
        self.logger.info("Starting hyperparameter optimization")
        
        # Prepare hyperparameter space
        param_space = self._prepare_parameter_space()
        self.logger.info("Parameter space prepared for optimization")
        
        try:
            # Get optimization results from the Ray actor
            self.logger.info("Starting hyperparameter optimization with Ray actor")
            
            # Log data shapes for debugging
            self.logger.info(
                f"Data shapes for optimization:"
                f"\n - Training: {X.shape}"
                f"\n - Validation: {X_val.shape}"
                f"\n - Test: {X_test.shape}"
            )
            
            # Start optimization with timeout
            self.logger.info("Submitting optimization task to Ray actor")
            optimization_future = self.validator.optimize_hyperparameters.remote(
                self, X, y, X_val, y_val, X_test, y_test,
                param_space, self.hyperparameter_space.get('search_strategy', {})
            )
            
            # Wait for results with timeout and logging
            try:
                self.logger.info("Waiting for optimization results...")
                best_params = ray.get(optimization_future, timeout=3600)  # 1 hour timeout
                self.logger.info("Received optimization results from Ray actor")
            except ray.exceptions.GetTimeoutError:
                self.logger.error("Optimization timed out after 1 hour")
                return self._get_default_params(param_space)
            except Exception as e:
                self.logger.error(f"Error getting optimization results: {str(e)}")
                return self._get_default_params(param_space)
            
            # Log successful completion
            self.logger.info("Hyperparameter optimization completed successfully")
            self.logger.info(f"Best parameters found: {best_params}")
            return best_params
                
        except Exception as e:
            self.logger.error(f"Error in hyperparameter optimization: {str(e)}")
            # Get default parameters
            try:
                self.logger.info("Attempting to get default parameters")
                default_params = ray.get(self.validator._get_default_params.remote(param_space))
                self.logger.info(f"Using default parameters: {default_params}")
                return default_params
            except Exception as inner_e:
                self.logger.error(f"Error getting default parameters: {str(inner_e)}")
                return {
                    'device': 'cpu',
                    'num_threads': -1,
                    'objective': 'binary',
                    'metric': ['binary_logloss', 'auc'],
                    'verbose': -1,
                    'deterministic': True,
                    'random_state': 19,
                    'learning_rate': 0.01,
                    'num_leaves': 31,
                    'max_depth': 6,
                    'min_data_in_leaf': 20,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'early_stopping_rounds': 100
                }

    def _prepare_parameter_space(self) -> Dict[str, Any]:
        """Prepare hyperparameter space for optimization."""
        self.logger.info("Preparing parameter space for optimization")
        param_space = {}
        param_ranges = {}  # Store human-readable ranges for logging
        
        for param, config in self.hyperparameter_space['hyperparameters'].items():
            try:
                # Handle numeric values
                if isinstance(config, (int, float)):
                    param_space[param] = config
                    param_ranges[param] = f"fixed({config})"
                    self.logger.debug(f"Added fixed parameter {param}: {config}")
                # Handle string values
                elif isinstance(config, str):
                    param_space[param] = config
                    param_ranges[param] = f"fixed({config})"
                    self.logger.debug(f"Added fixed parameter {param}: {config}")
                # Handle distribution dictionaries
                elif isinstance(config, dict) and 'distribution' in config:
                    if config['distribution'] == 'log_uniform':
                        min_val = max(config['min'], 1e-8)
                        max_val = max(config['max'], min_val + 1e-8)
                        param_space[param] = tune.uniform(min_val, max_val)
                        param_ranges[param] = f"uniform({min_val:.2e}, {max_val:.2e})"
                        self.logger.debug(f"Added log_uniform parameter {param}")
                    elif config['distribution'] == 'uniform':
                        param_space[param] = tune.uniform(config['min'], config['max'])
                        param_ranges[param] = f"uniform({config['min']:.3f}, {config['max']:.3f})"
                        self.logger.debug(f"Added uniform parameter {param}")
                    elif config['distribution'] == 'int_uniform':
                        min_val = max(1, int(config['min']))
                        max_val = max(min_val + 1, int(config['max']))
                        param_space[param] = tune.randint(min_val, max_val)
                        param_ranges[param] = f"int_uniform({min_val}, {max_val})"
                        self.logger.debug(f"Added int_uniform parameter {param}")
            except Exception as e:
                self.logger.error(f"Error processing parameter {param}: {str(e)}")
                param_space[param] = config
                param_ranges[param] = f"default({config})"

        # Add CPU-specific parameters
        cpu_params = {
            'device': 'cpu',
            'num_threads': -1,
            'objective': 'binary',
            'metric': ['binary_logloss', 'auc'],
            'verbose': -1,
            'deterministic': True
        }
        param_space.update(cpu_params)
        self.logger.info("Added CPU-specific parameters")
        
        # Log the final parameter space
        self.logger.info("Final parameter space for optimization:")
        for param, range_str in param_ranges.items():
            self.logger.info(f"{param}: {range_str}")
            
        return param_space

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
            
            # Log metrics
            try:
                self.mlflow.log_metrics(train_metrics)
            except Exception as e:
                self.logger.error(f"Failed to log metrics: {str(e)}")
            
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
        self.mlflow.log_metrics({f'val_{k}': v for k, v in metrics.items()})
        return metrics
    
    def _optimize_threshold(self, X_val: Any, y_val: Any) -> float:
        """Optimize decision threshold based on precision-recall trade-off.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Optimized threshold value
        """
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
    
    def _save_model_to_path(self, path: Path) -> None:
        """Save LightGBM model to specified path.
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise RuntimeError("No model to save")
        
        # Save model in both formats for flexibility
        # Save scikit-learn model format
        joblib.dump(self.model, path)
        
        # Also save native format for faster loading
        native_path = path.with_suffix('.txt')
        self.model.booster_.save_model(
            str(native_path),
            num_iteration=self.model.best_iteration_ or 0
        )
        
        # Save threshold
        threshold_path = path.parent / "threshold.json"
        with open(threshold_path, 'w') as f:
            json.dump({'threshold': self.best_threshold}, f)
            
        self.logger.info(f"Model saved to {path} (sklearn format) and {native_path} (native format)")
    
    def _load_model_from_path(self, path: Path) -> None:
        """Load LightGBM model from specified path.
        
        Args:
            path: Path to load the model from
        """
        if not path.exists():
            raise FileNotFoundError(f"No model file found at {path}")
            
        # Try loading native format first (it's faster)
        native_path = path.with_suffix('.txt')
        if native_path.exists():
            self.model = lgb.Booster(model_file=str(native_path))
            self.logger.info(f"Loaded native model from {native_path}")
        else:
            # Fall back to scikit-learn format
            self.model = joblib.load(path)
            self.logger.info(f"Loaded scikit-learn model from {path}")
        
        # Load threshold if available
        threshold_path = path.parent / "threshold.json"
        if threshold_path.exists():
            with open(threshold_path, 'r') as f:
                self.best_threshold = json.load(f)['threshold']
        else:
            self.best_threshold = 0.3  # Default threshold
            
        self.logger.info(f"Model loaded with threshold {self.best_threshold}") 