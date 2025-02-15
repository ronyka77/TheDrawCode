"""Nested cross-validation framework for hyperparameter optimization."""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sklearn.model_selection import StratifiedKFold
from ray.tune.search.bayesopt import BayesOptSearch
from ray import tune
import ray
from pathlib import Path
import random
import os
import sys
os.environ["ARROW_S3_DISABLE"] = "1"
# Disable Ray log deduplication to ensure all logs are captured
os.environ["RAY_DEDUP_LOGS"] = "0"

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)
os.environ["PYTHONPATH"] = project_root + os.pathsep + os.environ.get("PYTHONPATH", "")

from utils.logger import ExperimentLogger

class NestedCVValidator:
    """Nested cross-validation for hyperparameter optimization."""
    
    def __init__(
        self,
        outer_splits: int = 3,
        inner_splits: int = 2,
        logger: ExperimentLogger = ExperimentLogger("nested_cv_validator")):
        """Initialize nested CV validator.
        
        Args:
            outer_splits: Number of outer CV folds
            inner_splits: Number of inner CV folds
        """
        self.outer_splits = outer_splits
        self.inner_splits = inner_splits
        self.logger = logger
        self.best_score = 0
        self.best_params = {}
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None

    def optimize_hyperparameters(
        self,
        model: Any,
        X: Any,
        y: Any,
        X_val: Any,
        y_val: Any,
        X_test: Any,
        y_test: Any,
        param_space: Dict[str, Any],
        search_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Run nested cross-validation for hyperparameter optimization.
        
        Args:
            model: Model instance to optimize
            X: Training features
            y: Training labels
            param_space: Hyperparameter search space
            search_strategy: Search strategy configuration
            
        Returns:
            Dictionary of best hyperparameters
        """
        self.logger.info("Starting nested cross-validation")
        self.X_train = X
        self.y_train = y
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.model = model
        # Initialize outer CV
        outer_cv = StratifiedKFold(
            n_splits=self.outer_splits,
            shuffle=True,
            random_state=19
        )
        
        # Store results from each fold
        fold_results = []
            
        # Inner CV for hyperparameter tuning
        best_params = self._optimize_inner_cv(
            X,
            y,
            X_val,
            y_val,
            X_test,
            y_test,
            model,
            param_space
        )
        if "max_depth" in best_params:
            best_params["max_depth"] = int(round(best_params["max_depth"]))
        
        # Repeat for any other parameters that need to be integers:
        for key in ['seed', 'n_estimators', 'random_seed','early_stopping_rounds','min_child_weight']:
            if key in best_params:
                best_params[key] = int(round(best_params[key]))
        # Train model with best parameters
        self.logger.info(f"Training model with best parameters: {str(best_params)}")
        metrics = model.fit(X, y, X_val, y_val, X_test, y_test, **best_params)
        
        # Evaluate on validation set
        # metrics = model.evaluate(X_val, y_val)
        
        # Store results
        fold_results.append({
            'params': best_params,
            'metrics': metrics
        })
        
        self.logger.info(
            f"Hyperparameters,"
            f"Precision: {metrics['precision']:.4f}, "
            f"Recall: {metrics['recall']:.4f}"
        )
        
        # Select best parameters across folds
        # best_params = self._select_best_params(fold_results)
        self.logger.info(f"Best parameters selected: {str(best_params)}")
        
        return best_params, metrics
    # Define objective function for hyperparameter tuning
    def objective(self, config):
        # Convert continuous candidates to integers where needed.
        # For example, if max_depth is sampled using tune.uniform,
        # we convert it to an integer here.
        if "max_depth" in config:
            config["max_depth"] = int(round(config["max_depth"]))
        
        # Repeat for any other parameters that need to be integers:
        for key in ['seed', 'n_estimators', 'random_seed','early_stopping_rounds','min_child_weight']:
            if key in config:
                config[key] = int(round(config[key]))
        
        # Proceed with your cross-validation and model evaluation logic.
        self.logger.info(f"Cross-validating model with config: {config}")
        precision = self.cross_validate_model(
            config, self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test, self.model
        )
        if not isinstance(precision, (int, float)) or np.isnan(precision):
            precision = float('-inf')
        
        tune.report({"precision": precision, "training_iteration": 1})

    def _optimize_inner_cv(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        X_test: Any,
        y_test: Any,
        model: Any,
        param_space: Dict[str, Any],
        num_trials: int = 40) -> Dict[str, Any]:
        """Run inner cross-validation for hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training labels
            model: Model instance to optimize
            param_space: Hyperparameter search space
            num_trials: Number of trials to run
            
        Returns:
            Dictionary of best hyperparameters
        """
        # Set up storage path for Ray Tune with shorter base path
        storage_path = Path.cwd() / "ray_results"
        storage_path.mkdir(parents=True, exist_ok=True)
        
        # Custom trial name creator for shorter paths
        def trial_dirname_creator(trial: "Trial") -> str:
            """Create shorter trial directory names."""
            return f"trial_{trial.trial_id[-8:]}"  # Use last 8 chars of trial ID
            
        # Initialize Ray with proper resource limits and error handling
        if not ray.is_initialized():
            ray.init(
                num_cpus=os.cpu_count(),  # Use all available CPUs
                _temp_dir=str(storage_path / "tmp"),  # Shorter temp directory
                ignore_reinit_error=True,
                include_dashboard=False,  # Disable dashboard for reduced overhead
                log_to_driver=True,  # Enable logging to driver for better debugging
            )
        # Convert param_space to concrete values for Ray Tune
        concrete_param_space = {}
        for k, v in param_space.items():
            try:
                # Handle numeric values first
                if isinstance(v, (int, float)):
                    concrete_param_space[k] = v
                # Handle string values
                elif isinstance(v, str):
                    concrete_param_space[k] = v
                # Handle Ray Tune search objects with explicit conversion
                elif isinstance(v, tune.search.sample.Integer):
                    # Convert Integer sample to concrete value
                    concrete_param_space[k] = tune.randint(v.lower, v.upper + 1)
                elif isinstance(v, tune.search.sample.Float):
                    # Convert Float sample to concrete value
                    concrete_param_space[k] = tune.uniform(v.lower, v.upper)
                # Handle dictionaries with distribution configs
                elif isinstance(v, dict) and 'distribution' in v:
                    if v['distribution'] == 'log_uniform':
                        min_val = max(v['min'], 1e-8)
                        max_val = max(v['max'], min_val + 1e-8)
                        concrete_param_space[k] = tune.loguniform(min_val, max_val)
                    elif v['distribution'] == 'uniform':
                        concrete_param_space[k] = tune.uniform(v['min'], v['max'])
                    elif v['distribution'] == 'int_uniform':
                        min_val = int(v['min'])
                        max_val = int(v['max']) + 1  # ray.randint upper-bound is exclusive
                        concrete_param_space[k] = tune.randint(min_val, max_val)
                else:
                    self.logger.warning(f"Unhandled parameter type for {k}: {type(v)}")
                    concrete_param_space[k] = v
            except Exception as e:
                self.logger.error(f"Error processing parameter {k}: {str(e)}")
                concrete_param_space[k] = v  # Fallback to original value
        
        # Run hyperparameter optimization with improved error handling
        try:
            
            # Configure BayesOpt search space with proper type handling
            bayesopt_space = {}
            for k, v in param_space.items():
                try:
                    # Handle numeric ranges with explicit type conversion
                    if isinstance(v, dict) and 'min' in v and 'max' in v:
                        min_val = float(v['min']) if not isinstance(v['min'], str) else 0
                        max_val = float(v['max']) if not isinstance(v['max'], str) else 1
                        bayesopt_space[k] = tune.uniform(min_val, max_val)
                    # Handle CPU-specific parameters (enforce CPU-only training)
                    elif isinstance(v, str) and v.lower() == 'cpu':
                        bayesopt_space[k] = 'cpu'
                    # Skip string parameters that aren't CPU-related
                    elif isinstance(v, str):
                        continue
                    # Handle Integer parameters with explicit float conversion for BayesOpt
                    elif isinstance(v, tune.search.sample.Integer):
                        bayesopt_space[k] = tune.uniform(float(v.lower), float(v.upper))
                    # Handle other parameter types with fallback
                    else:
                        bayesopt_space[k] = v
                except Exception as e:
                    self.logger.warning(f"Error processing parameter {k}: {str(e)}")
                    bayesopt_space[k] = v  # Fallback to original value
            
            # Configure ASHA scheduler for better CPU utilization
            metric_name = "precision"  # Optimize for maximum precision
            scheduler = tune.schedulers.ASHAScheduler(
                time_attr="training_iteration",
                metric=metric_name,
                mode="max",
                max_t=3000,  # Maximum number of training iterations
                grace_period=500,  # Minimum training iterations before pruning
                reduction_factor=2,
                brackets=1
            )
            # Initialize BayesOpt search algorithm with proper space handling
            bayesopt = BayesOptSearch(
                metric="precision",
                mode="max",
                utility_kwargs={
                    "kind": "ucb",
                    "kappa": 2.5,
                    "xi": 0.0
                }
            )
            
            # Run tuning with parameter space passed directly to tune.run
            analysis = tune.run(
                lambda config: self.objective(config),
                name="xgb_tune",  # Shorter experiment name
                trial_dirname_creator=trial_dirname_creator,  # Custom trial directory names
                scheduler=scheduler,
                search_alg=bayesopt,
                num_samples=num_trials,
                config=bayesopt_space,  # Use pre-converted concrete parameter space
                resources_per_trial={
                    "cpu": 1,  # Single CPU per trial for better resource management
                    "gpu": 0  # Explicitly set GPU to 0 as per project requirements
                },
                verbose=0, 
                raise_on_failed_trial=False,  # Don't raise on failed trials
                max_failures=3,  # Allow some failures before stopping
                storage_path=str(storage_path.absolute()),
                log_to_file=True,
                checkpoint_freq=0  # Disable checkpointing to reduce overhead
            )

            # Get best parameters with proper error handling
            try:
                best_trial = analysis.get_best_trial(metric=metric_name, mode="max")
                if best_trial is None or best_trial.last_result is None:
                    raise ValueError("No successful trials found")
                
                best_local_params = best_trial.config
                best_local_score = best_trial.last_result.get(metric_name, 0.0)
                if best_local_score > self.best_score:
                    self.best_score = best_local_score
                    self.best_params = best_local_params
                    self.logger.info(
                        f"Best parameters found with {metric_name} {self.best_score:.4f}: {self.best_params}"
                    )
                return self.best_params
                
            except Exception as e:
                self.logger.error(f"Error getting best trial(validation.py): {str(e)}")
                return self._get_default_params(param_space)

        except Exception as e:
            self.logger.error(f"Error in hyperparameter optimization(validation.py): {str(e)}")
            return self._get_default_params(param_space)
        finally:
            # Clean up Ray and S3 resources
            if ray.is_initialized():
                try:
                    # Shutdown Ray first
                    ray.shutdown()
                    
                    # Clean up S3 resources if pyarrow is available
                    if 'pyarrow' in sys.modules:
                        import pyarrow as pa
                        if hasattr(pa, 'finalize_s3'):
                            pa.finalize_s3()
                except Exception as e:
                    self.logger.error(f"Error during resource cleanup: {str(e)}")

    def _get_default_params(self, param_space: Dict[str, Any]) -> Dict[str, Any]:
        """Get default parameters when optimization fails.
        
        Args:
            param_space: Parameter space configuration
            
        Returns:
            Dictionary of default parameters as concrete numeric values
        """
        self.logger.warning("Using default parameters due to optimization failure")
        default_params = {}
        
        # Process configuration from YAML (using 'min' and 'max')
        for param, config in param_space.items():
            if isinstance(config, dict) and 'distribution' in config:
                if config['distribution'] == 'log_uniform':
                    default_params[param] = np.exp(
                        (np.log(config['min']) + np.log(config['max'])) / 2
                    )
                elif config['distribution'] == 'uniform':
                    default_params[param] = (config['min'] + config['max']) / 2
                elif config['distribution'] == 'int_uniform':
                    default_params[param] = int(config['min'])
            # Process already-converted Ray Tune objects (using 'lower' and 'upper')
            elif hasattr(config, "lower") and hasattr(config, "upper"):
                lower = config.lower() if callable(config.lower) else config.lower
                upper = config.upper() if callable(config.upper) else config.upper
                
                # Ensure numeric types before arithmetic operations
                try:
                    lower = float(lower) if not isinstance(lower, (int, float)) else lower
                    upper = float(upper) if not isinstance(upper, (int, float)) else upper
                    
                    if isinstance(lower, int) and isinstance(upper, int):
                        default_params[param] = int(lower)
                    else:
                        default_params[param] = float((lower + upper) / 2)
                except (TypeError, ValueError) as e:
                    # self.logger.warning(f"Failed to convert bounds for {param}: {str(e)}. Using lower bound.")
                    default_params[param] = lower
            else:
                default_params[param] = config
                
        # Add CPU-specific defaults
        default_params.update({
            'tree_method': 'hist',
            'device': 'cpu',
            'n_jobs': 1,
            'eval_metric': ['error', 'aucpr', 'logloss'],
            'objective': 'binary:logistic'
        })
        
        self.logger.info(f"Default parameters: {default_params}")
        return default_params

    def _select_best_params(
        self,
        fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select best parameters across outer folds.
        
        Args:
            fold_results: List of results from each outer fold
            
        Returns:
            Dictionary of best parameters
        """
        # Calculate mean performance for each parameter combination
        param_performance = {}
        for result in fold_results:
            params_key = str(result['params'])
            if params_key not in param_performance:
                param_performance[params_key] = {
                    'params': result['params'],
                    'scores': [],
                    'count': 0
                }
            
            # Only consider if recall threshold is met
            if result['metrics']['recall'] >= 0.15 and result['metrics']['recall'] < 0.9 and result['metrics']['precision'] > 0.32:
                param_performance[params_key]['scores'].append(
                    result['metrics']['precision']
                )
                param_performance[params_key]['count'] += 1
        
        # Select parameters that perform well consistently
        best_params = None
        best_score = float('-inf')
        for perf in param_performance.values():
            if perf['count'] > 0:
                mean_score = np.mean(perf['scores'])
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = perf['params']
        if best_params is None:
            self.logger.warning("No parameter set met recall threshold consistently")
            # Return parameters from best performing fold
            best_fold = max(fold_results, key=lambda x: x['metrics']['precision'])
            best_params = best_fold['params']
        return best_params 

    def cross_validate_model(
        self,
        config: Dict[str, Any],
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        X_test: Any,
        y_test: Any,
        model: Any) -> float:
        """Perform cross-validation evaluation of model with given configuration.
        
        Args:
            config: Dictionary of hyperparameters to evaluate
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            X_test: Test features
            y_test: Test labels
            model: Model instance to evaluate
            
        Returns:
            float: Mean precision score across CV folds
        """
        # self.logger.info(f"Cross-validating model with config: {str(config)}")

        # Inner CV
        inner_cv = StratifiedKFold(
            n_splits=self.inner_splits,
            shuffle=True,
            random_state=19
        )
        cv_scores = []
        # Train and evaluate
        for fold_idx, (train_idx, val_idx) in enumerate(inner_cv.split(X_train, y_train)):
            try:
                self.logger.info(f"Fold {fold_idx + 1}")
                X_train_inner = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
                X_test_inner = X_train.iloc[val_idx] if hasattr(X_train, 'iloc') else X_train[val_idx]
                y_train_inner = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
                y_test_inner = y_train.iloc[val_idx] if hasattr(y_train, 'iloc') else y_train[val_idx]
                
                # Train with early stopping using validation set
                metrics = model.fit(
                    X_train_inner,
                    y_train_inner,
                    X_val,
                    y_val,
                    X_test_inner,
                    y_test_inner,
                    **config
                )
                
                # Get precision score, defaulting to 0.0 if undefined
                precision = metrics.get('precision', 0.0)
                recall = metrics.get('recall', 0.0)
                
                # Only consider precision if recall threshold is met (20%)
                if recall >= 0.15 and recall < 0.9 and precision > 0.30:
                    cv_scores.append(precision)
                    self.logger.info(f"Fold {fold_idx + 1}: Precision {precision:.4f}, Recall {recall:.4f}")
                else:
                    cv_scores.append(0.0)  # Penalize models that don't meet recall threshold
                    self.logger.info(f"Fold {fold_idx + 1}: Recall {recall:.4f} Precision {precision:.4f} outside threshold range, assigning score 0.0")
            
            except Exception as e:
                self.logger.error(f"Error in fold {fold_idx + 1}: {str(e)}")
                cv_scores.append(0.0)  # Penalize failed evaluations
        # Return the mean score
        best_score = np.max(cv_scores) if len(cv_scores) > 0 else 0.0
        self.logger.info(f"Cross-validation complete - Best precision: {best_score:.4f}")
        return best_score 