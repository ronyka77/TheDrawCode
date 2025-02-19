"""CatBoost model implementation with CPU optimization and hyperparameter tuning."""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import catboost as cb
from catboost import Pool
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
from models.StackedEnsemble.shared.config_loader import ConfigurationLoader

# Global validator actor name
VALIDATOR_ACTOR_NAME = "global_validator"

@ray.remote
class ValidatorActor:
    """Ray actor for validation that ensures single instance."""
    
    def __init__(self, logger=None, model_type='catboost'):
        """Initialize validator actor.
        
        Args:
            logger: Logger instance
            model_type: Model type
        """
        # Create a new logger instance for the actor
        self.logger = logger or ExperimentLogger('catboost_hypertuning')
        self.validator = NestedCVValidator(logger=self.logger, model_type=model_type)
        self.logger.info("Created new validator instance")
        
    def optimize_hyperparameters(self, model, X, y, X_val, y_val, X_test, y_test, param_space, search_strategy):
        """Run hyperparameter optimization."""
        try:
            # Ensure logger is available
            if not hasattr(self, 'logger') or self.logger is None:
                self.logger = ExperimentLogger('catboost_hypertuning')
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
                self.logger = ExperimentLogger('catboost_hypertuning')
                self.validator.logger = self.logger
            
            self.logger.info("Getting default parameters")
            params = self.validator._get_default_params(param_space)
            self.logger.info(f"Retrieved default parameters: {params}")
            return params
        except Exception as e:
            if hasattr(self, 'logger') and self.logger is not None:
                self.logger.error(f"Error getting default parameters: {str(e)}")
            return {
                'learning_rate': 0.01,
                'iterations': 500,
                'depth': 6,
                'task_type': 'CPU',
                'thread_count': -1,
                'bootstrap_type': 'Bernoulli',
                'grow_policy': 'SymmetricTree'
            }
    
    def get_info(self):
        """Get validator info."""
        return "Validator is set up"

class CatBoostModel(BaseModel):
    """CatBoost model implementation with CPU optimization."""
    
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

    def __init__(self, experiment_name: str = 'catboost_experiment', model_type: str = "catboost", logger: ExperimentLogger = None):
        """Initialize CatBoost model.
        
        Args:
            experiment_name: Name for MLflow experiment tracking
            model_type: Type of model (e.g., 'catboost')
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
            "catboost_config.yaml"
        )
        
        # Initialize base class
        super().__init__(model_type=model_type, experiment_name=experiment_name)
        self.logger = logger or ExperimentLogger('catboost_hypertuning')
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
        
        # Get or create validator actor
        self.validator = self.get_validator_actor(self.logger)
            
        # Load model configuration
        self.config_loader = ConfigurationLoader(logger=self.logger, experiment_name=experiment_name)
        self.model_config = self.config_loader.load_model_config(model_type)
        self.hyperparameter_space = self.config_loader.load_hyperparameter_space(model_type)
        self.best_params = {}
        self.best_score = 0
        
        self.logger.info(f"Initialized {model_type} model with experiment name: {experiment_name}")

    def _create_model(self, **kwargs) -> cb.CatBoostClassifier:
        """Create and configure CatBoost model instance.
        
        Args:
            **kwargs: Model parameters to override defaults
            
        Returns:
            Configured CatBoost classifier
        """
        params = {
            'task_type': 'CPU',
            'thread_count': -1,
            'bootstrap_type': 'Bernoulli',
            'grow_policy': 'SymmetricTree',
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'random_seed': 19
        }
        
        if self.model_config:
            params.update(self.model_config.get('params', {}))
        params.update(kwargs)
        
        # Ensure task_type is in the proper uppercase format
        if 'task_type' in params and isinstance(params['task_type'], str):
            params['task_type'] = params['task_type'].upper()
        
        return cb.CatBoostClassifier(**params)

    def _convert_to_model_format(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None) -> Tuple[Any, Optional[Any]]:
        """Convert data to CatBoost format.
        
        Args:
            X: Feature matrix
            y: Optional target vector
            
        Returns:
            Tuple of (X, y) in CatBoost format
        """
        if X is None:
            raise ValueError("The feature dataset X must not be None.")
        
        # CatBoost can handle pandas DataFrames directly
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
        y_test: Any,
        **kwargs) -> Dict[str, float]:
        """Train CatBoost model with early stopping.
        
        Args:
            X: Training features
            y: Training labels
            X_val: Validation features
            y_val: Validation labels
            X_test: Test features
            y_test: Test labels
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary of training metrics
        """
        # Convert data to CatBoost format
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
            X_val,
            y_val,
            X_test,
            y_test,
            eval_set=eval_set,
            silent=True
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
        
        X_cb, _ = self._convert_to_model_format(X)
        probas = self.model.predict_proba(X_cb)[:, 1]
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
        
        X_cb, _ = self._convert_to_model_format(X)
        return self.model.predict_proba(X_cb)[:, 1]

    def optimize_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        **kwargs) -> Dict[str, Any]:
        """Optimize hyperparameters using nested cross-validation.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            X_test: Testing features
            y_test: Testing labels
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
                f"\n - Training: {X_train.shape}"
                f"\n - Validation: {X_val.shape}"
                f"\n - Test: {X_test.shape}"
            )
            
            # Start optimization with timeout
            self.logger.info("Submitting optimization task to Ray actor")
            optimization_future = self.validator.optimize_hyperparameters.remote(
                self, X_train, y_train, X_val, y_val, X_test, y_test,
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
                    'task_type': 'CPU',
                    'thread_count': -1,
                    'bootstrap_type': 'Bernoulli',
                    'grow_policy': 'SymmetricTree',
                }

    def _prepare_parameter_space(self) -> Dict[str, Any]:
        """Prepare hyperparameter space for optimization."""
        self.logger.info("Preparing parameter space for optimization")
        param_space = {}
        param_ranges = {}
        
        for param, config in self.hyperparameter_space['hyperparameters'].items():
            try:
                if isinstance(config, (int, float, str)):
                    param_space[param] = config
                    param_ranges[param] = f"fixed({config})"
                    self.logger.debug(f"Added fixed parameter {param}: {config}")
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
            'task_type': 'CPU',
            'thread_count': -1,
            'bootstrap_type': 'Bernoulli',
            'grow_policy': 'SymmetricTree',
        }
        param_space.update(cpu_params)
        self.logger.info("Added CPU-specific parameters")
        
        # Log the final parameter space
        self.logger.info("Final parameter space for optimization:")
        for param, range_str in param_ranges.items():
            self.logger.info(f"{param}: {range_str}")
            
        return param_space

    def _save_model_to_path(self, path: Path) -> None:
        """Save CatBoost model to specified path.
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise RuntimeError("No model to save")
        
        # Save model in both formats for flexibility
        # Save scikit-learn model format
        joblib.dump(self.model, path)
        
        # Also save native format for faster loading
        native_path = path.with_suffix('.cbm')
        self.model.save_model(str(native_path))
        
        # Save threshold
        threshold_path = path.parent / "threshold.json"
        with open(threshold_path, 'w') as f:
            json.dump({'threshold': self.best_threshold}, f)
            
        self.logger.info(f"Model saved to {path} (sklearn format) and {native_path} (native format)")

    def _load_model_from_path(self, path: Path) -> None:
        """Load CatBoost model from specified path.
        
        Args:
            path: Path to load the model from
        """
        if not path.exists():
            raise FileNotFoundError(f"No model file found at {path}")
            
        # Try loading native format first (it's faster)
        native_path = path.with_suffix('.cbm')
        if native_path.exists():
            self.model = cb.CatBoostClassifier()
            self.model.load_model(str(native_path))
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

    def fit(
        self,
        X: Any,
        y: Any,
        X_val: Any,
        y_val: Any,
        X_test: Optional[Any] = None,
        y_test: Optional[Any] = None,
        **kwargs) -> Dict[str, float]:
        """Train the CatBoost model with early stopping."""
        self.logger.info("Starting model training")
        
        try:
            # Remove incompatible parameters
            if 'scale_pos_weight' in kwargs:
                del kwargs['scale_pos_weight']
            
            # Ensure proper parameter types
            int_params = [
                'iterations', 'early_stopping_rounds', 'depth', 
                'min_data_in_leaf', 'thread_count', 'random_seed'
            ]
            for param in int_params:
                if param in kwargs:
                    kwargs[param] = int(round(float(kwargs[param])))
            
            # Ensure proper string parameters
            str_params = ['task_type', 'bootstrap_type', 'grow_policy', 'loss_function']
            for param in str_params:
                if param in kwargs and not isinstance(kwargs[param], str):
                    kwargs[param] = str(kwargs[param])
            
            # Initialize model with parameters
            self.model = self._create_model(**kwargs)
            
            # Prepare data
            feature_names = list(X.columns) if hasattr(X, 'columns') else None
            train_pool = Pool(data=X, label=y, feature_names=feature_names)
            
            # Prepare validation data
            if X_val is not None and y_val is not None:
                valid_pool = Pool(data=X_val, label=y_val, feature_names=feature_names)
            else:
                self.logger.info("No validation set provided")
                valid_pool = None
            
            # Prepare test data
            if X_test is not None and y_test is not None:
                test_pool = Pool(data=X_test, label=y_test, feature_names=feature_names)
            else:
                self.logger.info("No test set provided")
                test_pool = None
            
            # Train model
            self.model.fit(
                train_pool,
                eval_set=test_pool if test_pool is not None else valid_pool,
                use_best_model=True,
                silent=True
            )
            
            # Evaluate model
            metrics = self.evaluate(X_val, y_val) if X_val is not None else {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'auc': 0.0,
                'brier_score': 1.0
            }
            
            self.logger.info("Model training completed successfully")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}")
            # Return default metrics
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'auc': 0.0,
                'brier_score': 1.0
            }

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

    def evaluate(self, X_val: Any, y_val: Any, optimize_threshold: bool = True) -> Dict[str, float]:
        """Evaluate model performance."""
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
            
            # Apply precision-recall constraints
            if metrics['recall'] >= 0.15 and metrics['recall'] < 0.9 and metrics['precision'] > 0.30:
                score = metrics['precision']
                if score > best_score:
                    self.logger.info(
                        f"Threshold: {threshold:.4f}, "
                        f"Precision: {metrics['precision']:.4f}, "
                        f"Recall: {metrics['recall']:.4f}"
                    )
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
            threshold: Decision threshold (default: 0.3)
            
        Returns:
            Dictionary of metrics
            
        Raises:
            ValueError: If threshold is None
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

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores.
        
        Returns:
            DataFrame with feature importance scores
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before getting feature importance")
            
        try:
            # Get feature importance scores
            feature_importance = self.model.get_feature_importance()
            feature_names = self.model.feature_names_
            
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