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
# Configure environment variables
os.environ["GIT_PYTHON_GIT_EXECUTABLE"] = "git"

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)
os.environ["PYTHONPATH"] = project_root + os.pathsep + os.environ.get("PYTHONPATH", "")

from utils.logger import ExperimentLogger
from models.StackedEnsemble.shared.config_loader import ConfigurationLoader
from tqdm import tqdm

class NestedCVValidator:
    """Nested cross-validation for hyperparameter optimization."""
    
    _instance = None  # Singleton instance
    
    def __new__(cls, *args, **kwargs):
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        outer_splits: int = 3,
        inner_splits: int = 2,
        model_type: str = None,
        logger: ExperimentLogger = None):
        """Initialize nested CV validator.
        
        Args:
            outer_splits: Number of outer CV folds
            inner_splits: Number of inner CV folds
            model_type: Model type
            logger: Logger instance
        """
        # Only initialize if not already initialized
        if not hasattr(self, 'initialized'):
            self.outer_splits = outer_splits
            self.inner_splits = inner_splits
            self.logger = logger or ExperimentLogger(f"{model_type}_hypertuning")
            self.best_score = 0
            self.best_params = {}
            self.X_train = None
            self.y_train = None
            self.X_val = None
            self.y_val = None
            self.X_test = None
            self.y_test = None
            self.actual_folds = 1
            self.initialized = True
            self.model_type = model_type
            self.config_loader = ConfigurationLoader(self.model_type)
            
            if ray.is_initialized():
                self.logger.info("NestedCVValidator initialized in Ray context")
            else:
                self.logger.info("NestedCVValidator initialized in standalone context")

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
        self.best_precision = 0
        
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
        
        # Repeat for any other parameters that need to be integers:
        for key in ['seed', 'n_estimators', 'iterations', 'random_seed', 'early_stopping_rounds', 'min_child_weight', 'num_leaves',
                'min_data_in_leaf', 'bagging_freq', 'max_depth', 'num_train_epochs', 'per_device_train_batch_size', 
                'gradient_accumulation_steps', 'max_seq_length', 'num_labels', 'num_workers', 'data_seed']:
            if key in best_params:
                best_params[key] = int(round(best_params[key]))
        # Train model with best parameters
        self.logger.info(f"Training model with best parameters: {str(best_params)}")
        metrics = model.fit(X, y, X_val, y_val, X_test, y_test, **best_params)
        
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

    def objective(self, config, X_train, y_train, X_val, y_val, X_test, y_test, model, logger):
        """Ray Tune objective function."""
        try:
            # Debug log at start of objective function
            if hasattr(self, 'logger') and self.logger is not None:
                self.logger.debug("Starting objective function evaluation")
                for handler in self.logger.handlers:
                    handler.flush()
            # Initialize logger if not already set up
            if not hasattr(self, 'logger') or self.logger is None:
                self.logger = logger
                from utils.logger import ExperimentLogger
                if not self.logger.handlers:
                    self.logger = ExperimentLogger(
                        experiment_name=f"{self.model_type}_hypertuning",
                        log_dir=f"logs/{self.model_type}_hypertuning"
                    )
                self.logger.propagate = True
                self.logger.debug("Logger initialized in objective function")
                for handler in self.logger.handlers:
                    handler.flush()

            # Convert parameters and log configuration
            config = self._convert_params_to_correct_types(config)
            self.logger.info(f"Evaluating configuration: {config}")
            for handler in self.logger.handlers:
                handler.flush()
            
            # Perform cross-validation
            precision = self.cross_validate_model(
                config, 
                X_train, 
                y_train, 
                X_val, 
                y_val,
                X_test, 
                y_test, 
                model
            )
            
            # Handle invalid precision values
            if not isinstance(precision, (int, float)) or np.isnan(precision):
                self.logger.warning("Invalid precision value, setting to -inf")
                precision = float('-inf')
            else:
                self.logger.info(f"Trial completed with precision: {precision:.4f}")
            
            # Ensure logs are flushed after precision check
            for handler in self.logger.handlers:
                handler.flush()
            
            # Log additional metrics with immediate flush
            self.logger.debug("Logging trial metrics")
            self.logger.info(
                "Trial metrics logged",
                error_code=None,
                extra={
                    'precision': precision,
                    'config': config,
                    'training_iteration': 1
                }
            )
            for handler in self.logger.handlers:
                handler.flush()
            
            # Report results to Ray Tune
            tune.report({"precision": precision, "training_iteration": 1})
            
        except Exception as e:
            if hasattr(self, 'logger') and self.logger is not None:
                self.logger.error(f"Error in objective function: {str(e)}")
                for handler in self.logger.handlers:
                    handler.flush()
            tune.report({"precision": float('-inf'), "training_iteration": 1})

    def _convert_params_to_correct_types(self, config):
        """Convert parameters to their correct types."""
        integer_params = [
            'seed', 'n_estimators', 'iterations', 'random_seed', 
            'early_stopping_rounds', 'min_child_weight', 'num_leaves', 
            'min_data_in_leaf', 'bagging_freq', 'max_depth'
        ]
        
        for key in integer_params:
            if key in config:
                config[key] = int(round(config[key]))
        
        return config

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
        num_trials: int = 10) -> Dict[str, Any]:
        """Run inner cross-validation for hyperparameter tuning."""
        # Ensure logger is available
        if not hasattr(self, 'logger') or self.logger is None:
            self.logger = ExperimentLogger(f"{model.model_type}_hypertuning")
            
        self.logger.info("Starting inner CV optimization")
        
        # Set up storage path for Ray Tune
        storage_path = Path.cwd() / "ray_results"
        storage_path.mkdir(parents=True, exist_ok=True)
        
        # Custom trial name creator for shorter paths
        def trial_dirname_creator(trial: "Trial") -> str:
            """Create shorter trial directory names."""
            return f"trial_{trial.trial_id[-8:]}"  # Use last 8 chars of trial ID
            
        # Initialize Ray with proper resource limits and error handling
        if not ray.is_initialized():
            ray.init(
                num_cpus=os.cpu_count(),
                _temp_dir=str(storage_path / "tmp"),
                ignore_reinit_error=True,
                include_dashboard=False,
                log_to_driver=True,
                logging_level="WARNING"  # Reduce Ray's logging
            )
            self.logger.info("Ray initialized for hyperparameter tuning")
            
        # Configure BayesOpt search space with proper type handling
        bayesopt_space = {}
        for k, v in param_space.items():
            try:
                # Handle numeric ranges with explicit type conversion
                if isinstance(v, dict) and 'min' in v and 'max' in v:
                    min_val = float(v['min']) if not isinstance(v['min'], str) else 0
                    max_val = float(v['max']) if not isinstance(v['max'], str) else 1
                    bayesopt_space[k] = tune.uniform(min_val, max_val)
                    self.logger.info(f"Parameter {k}: uniform({min_val}, {max_val})")
                # Handle CPU-specific parameters (enforce CPU-only training)
                elif isinstance(v, str) and v.lower() == 'cpu':
                    bayesopt_space[k] = 'cpu'
                    self.logger.info(f"Parameter {k}: fixed(cpu)")
                # Skip string parameters that aren't CPU-related
                elif isinstance(v, str):
                    continue
                # Handle Integer parameters with explicit float conversion for BayesOpt
                elif isinstance(v, tune.search.sample.Integer):
                    bayesopt_space[k] = tune.uniform(float(v.lower), float(v.upper))
                    self.logger.info(f"Parameter {k}: int_uniform({v.lower}, {v.upper})")
                # Handle other parameter types with fallback
                else:
                    bayesopt_space[k] = v
                    self.logger.info(f"Parameter {k}: {v}")
            except Exception as e:
                self.logger.warning(f"Error processing parameter {k}: {str(e)}")
                bayesopt_space[k] = v  # Fallback to original value
        
        self.logger.info("Configured parameter space for optimization")
        
        # Set up search algorithm
        search_alg = BayesOptSearch(
            metric="precision",
            mode="max",
            random_search_steps=20,
            utility_kwargs={
                "kind": "ucb",
                "kappa": 2.5,
                "xi": 0.0
            }
        )
        self.logger.info("Initialized BayesOpt search algorithm")
        
        # Set up scheduler
        scheduler = tune.schedulers.ASHAScheduler(
            metric="precision",
            mode="max",
            max_t=1000,
            grace_period=100,
            reduction_factor=2
        )
        self.logger.info("Initialized ASHA scheduler")
        
        # Configure tuning
        tune_config = tune.TuneConfig(
            num_samples=num_trials,
            search_alg=search_alg,
            scheduler=scheduler,
            trial_dirname_creator=trial_dirname_creator
        )
        
        # Set up run configuration
        run_config = tune.RunConfig(
            name=f"{self.model_type}_tuning",
            storage_path=str(storage_path),
            log_to_file=True,
            checkpoint_config=tune.CheckpointConfig(
                num_to_keep=2,
                checkpoint_score_attribute="precision"
            ),
            verbose=2 
        )
        
        try:
            self.logger.info("Starting hyperparameter search with Ray Tune")
            # Run hyperparameter search
            tuner = tune.Tuner(
                tune.with_parameters(
                    self.objective,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    X_test=X_test,
                    y_test=y_test,
                    model=model,
                    logger=self.logger
                ),
                param_space=bayesopt_space,
                tune_config=tune_config,
                run_config=run_config
            )
            
            self.logger.info("Running hyperparameter optimization...")
            results = tuner.fit()
            self.logger.info("Hyperparameter optimization completed")
            
            # Get best trial and parameters (choose the best precision from any iteration)
            try:
                best_result = results.get_best_result(metric="precision", mode="max", scope="all")
                if best_result is None:
                    self.logger.warning("No successful trials completed")
                    return self._get_default_params(param_space)
                
                best_params = best_result.config
                best_precision = best_result.metrics.get("precision", 0.0)
                self.logger.info(f"Best trial parameters: {best_params} with precision: {best_precision:.4f}")
            except AttributeError as e:
                self.logger.error(f"Error getting best result: {str(e)}")
                return self._get_default_params(param_space)
            
            # Convert integer parameters
            for key in ['iterations', 'min_data_in_leaf', 'early_stopping_rounds', 'num_train_epochs', 'per_device_train_batch_size', 
                    'gradient_accumulation_steps', 'max_seq_length', 'num_labels', 'num_workers', 'seed', 'data_seed']:
                if key in best_params:
                    best_params[key] = int(round(best_params[key]))
            
            self.logger.info(f"Best trial precision: {best_precision:.4f}")
            self.logger.info(f"Best parameters: {best_params}")
            
            return best_params
            
        except Exception as e:
            self.logger.error(f"Error in hyperparameter optimization: {str(e)}")
            return self._get_default_params(param_space)
            
        finally:
            # Clean up Ray resources
            if ray.is_initialized():
                ray.shutdown()
                self.logger.info("Ray resources cleaned up")

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
            try:
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
                    try:
                        # Get lower/upper bounds, handling both callable and non-callable cases
                        lower = config.lower() if callable(config.lower) else config.lower
                        upper = config.upper() if callable(config.upper) else config.upper
                        
                        # Handle string values by using lower bound
                        if isinstance(lower, str) or isinstance(upper, str):
                            default_params[param] = lower
                            continue
                            
                        # Convert to float if not already numeric
                        lower = float(lower) if not isinstance(lower, (int, float)) else lower
                        upper = float(upper) if not isinstance(upper, (int, float)) else upper
                        
                        if isinstance(lower, int) and isinstance(upper, int):
                            default_params[param] = int(lower)
                        else:
                            default_params[param] = float((lower + upper) / 2)
                    except (TypeError, ValueError, AttributeError):
                        # If any conversion fails, use lower bound
                        default_params[param] = lower
                else:
                    default_params[param] = config
                    
            except Exception as e:
                self.logger.error(f"Error processing parameter {param}: {str(e)}")
                default_params[param] = config
                
        # Get model-specific defaults from config
        model_config = self.config_loader.load_model_config(self.model_type)
        if model_config:
            # Update with CPU-specific configs from YAML
            default_params.update(model_config.get('cpu_config', {}))
            # Update with core model parameters from YAML
            default_params.update(model_config.get('params', {}))
        
        # Add model-type specific defaults based on configs
        if self.model_type == 'xgboost':
            default_params.update({
                'tree_method': 'hist',  # CPU-optimized histogram-based tree method
                'device': 'cpu',
                'n_jobs': -1,  # Use all available CPU cores
                'tree_learner': 'serial',
                'force_row_wise': True,
                'eval_metric': ['logloss', 'auc'],  # From xgboost_config.yaml
                'objective': 'binary:logistic',
                'random_state': 19,  # From xgboost_config.yaml
                'early_stopping_rounds': 300  # From xgboost_config.yaml
            })
        elif self.model_type == 'lightgbm':
            default_params.update({
                'device': 'cpu',
                'num_threads': -1,
                'verbosity': -1,
                'objective': 'binary',
                'metric': ['binary_logloss', 'auc'],
                'boosting_type': 'gbdt',
                'random_state': 19,
                'early_stopping_rounds': 100,
                'first_metric_only': True
            })
        elif self.model_type == 'catboost':
            default_params.update({
                'task_type': 'CPU',
                'thread_count': -1,
                'bootstrap_type': 'Bernoulli',
                'grow_policy': 'SymmetricTree',
                'loss_function': 'Logloss',
                'eval_metric': 'AUC',
                'random_seed': 19,
                'early_stopping_rounds': 100,
                'boosting_type': 'Plain',
                'use_best_model': True,
                'verbose': 0
            })
        elif self.model_type == 'bert':
            default_params.update({
                'device': 'cpu',
                'fp16': False,
                'fp16_opt_level': 'O1', 
                'max_grad_norm': 1.0,
                'num_workers': 4,
                'model_type': 'bert',
                'model_name': 'bert-base-uncased',
                'num_labels': 2,
                'problem_type': 'single_label_classification',
                'hidden_dropout_prob': 0.1,
                'attention_probs_dropout_prob': 0.1,
                'max_seq_length': 256,
                'per_device_train_batch_size': 16,
                'gradient_accumulation_steps': 4,
                'num_train_epochs': 3,
                'warmup_ratio': 0.1,
                'weight_decay': 0.01,
                'learning_rate': 2e-5,
                'optim': 'adamw_torch',
                'lr_scheduler_type': 'linear',
                'early_stopping_patience': 3,
                'early_stopping_threshold': 0.01,
                'eval_strategy': 'steps',
                'save_strategy': 'steps',
                'metric_for_best_model': 'precision',
                'greater_is_better': True,
                'load_best_model_at_end': True,
                'dataloader_num_workers': 0,
                'gradient_checkpointing': True,
                'group_by_length': True,
                'bf16': False,
                'disable_tqdm': True,
                'report_to': 'none',
                'remove_unused_columns': False,
                'seed': 42,
                'data_seed': 42,
                'ignore_mismatched_sizes': True,
                'label_names': ['labels']
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
        """Perform cross-validation evaluation of model with given configuration."""
        # Ensure logger is available
        if not hasattr(self, 'logger') or self.logger is None:
            self.logger = ExperimentLogger(f"{model.model_type}_hypertuning")
        
        # Inner CV: Wrap the cross-validation split with tqdm to display progress
        inner_cv = StratifiedKFold(
            n_splits=self.inner_splits,
            shuffle=True,
            random_state=19
        )
        cv_scores = []
        
        # Special handling for BERT models
        is_bert = model.model_type == 'bert'
        if is_bert:
            try:
                self.logger.info("BERT model detected - using single fold evaluation")
                
                # Set model initialization parameters
                config['model_init_params'] = {
                    'pretrained_model_name_or_path': config.get('model_name', 'bert-base-uncased'),
                    'num_labels': config.get('num_labels', 2),
                    'problem_type': 'single_label_classification',
                    'ignore_mismatched_sizes': True
                }
                
                # Log model configuration
                self.logger.info(f"BERT model configuration: {config}")
                # Ensure integer parameters are properly converted
                int_params = ['max_seq_length', 'per_device_train_batch_size', 'gradient_accumulation_steps']
                for param in int_params:
                    if param in config:
                        config[param] = int(round(config[param]))
                        self.logger.debug(f"Converted {param} to int: {config[param]}")
                # Log the final configuration
                self.logger.info("Final configuration for BERT model:")
                for key, value in config.items():
                    self.logger.info(f"{key}: {value}")
                metrics = model.fit(
                    X_train, y_train,
                    X_val, y_val,
                    X_test, y_test,
                    **config
                )
                
                # Get precision score, defaulting to 0.0 if undefined
                precision = metrics.get('val_eval_precision', 0.0)
                recall = metrics.get('val_eval_recall', 0.0)
                
                # Only consider precision if recall meets threshold requirements
                if recall >= 0.15 and recall < 0.9 and precision > 0.30:
                    cv_scores.append(precision)
                    self.logger.info(
                        f"BERT evaluation completed - "
                        f"Precision: {precision:.4f}, "
                        f"Recall: {recall:.4f}"
                    )
                else:
                    cv_scores.append(0.0)
                    self.logger.info(
                        f"BERT metrics outside thresholds - "
                        f"Precision: {precision:.4f}, "
                        f"Recall: {recall:.4f}"
                    )
            except Exception as e:
                self.logger.error(f"Error in BERT evaluation: {str(e)}")
                cv_scores.append(0.0)
        else:
            # Use tqdm to monitor progress of cross-validation folds for non-BERT models
            for fold_idx, (train_idx, val_idx) in enumerate(
                    tqdm(inner_cv.split(X_train, y_train), total=self.inner_splits, desc="CV Folds", ascii=True)
                ):
                try:
                    self.actual_folds += 1
                    self.logger.info(f"Starting fold {self.actual_folds}")
                    
                    # Prepare data for this fold
                    X_train_inner = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
                    X_test_inner = X_train.iloc[val_idx] if hasattr(X_train, 'iloc') else X_train[val_idx]
                    y_train_inner = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
                    y_test_inner = y_train.iloc[val_idx] if hasattr(y_train, 'iloc') else y_train[val_idx]
                    
                    self.logger.info(f"Training model for fold {self.actual_folds}")
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
                    
                    # Only consider precision if recall meets threshold requirements
                    if recall >= 0.15 and recall < 0.9 and precision > 0.30:
                        cv_scores.append(precision)
                        self.logger.info(
                            f"Fold {self.actual_folds} completed - "
                            f"Precision: {precision:.4f}, "
                            f"Recall: {recall:.4f}"
                        )
                    else:
                        cv_scores.append(0.0)
                        self.logger.info(
                            f"Fold {self.actual_folds} metrics outside thresholds - "
                            f"Precision: {precision:.4f}, "
                            f"Recall: {recall:.4f}"
                        )
                except Exception as e:
                    self.logger.error(f"Error in fold {self.actual_folds}: {str(e)}")
                    cv_scores.append(0.0)
        
        best_score = np.max(cv_scores) if cv_scores else 0.0
        if best_score > self.best_precision:
            self.best_precision = best_score
            self.logger.info(
                f"New best precision found: {best_score:.4f} "
                f"(previous best: {self.best_precision:.4f})"
            )
        
        self.logger.info(
            f"Cross-validation complete - "
            f"Best precision: {best_score:.4f}, "
            f"Overall best precision: {self.best_precision:.4f}"
        )
        return best_score 