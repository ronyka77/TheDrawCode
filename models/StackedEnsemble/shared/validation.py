"""Validation utilities for model training and hyperparameter optimization."""

from typing import Dict, Any, Optional
import numpy as np
from pathlib import Path
import os
import sys
import optuna
from optuna.trial import Trial
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import yaml
import json
import time

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)
os.environ["PYTHONPATH"] = project_root + os.pathsep + os.environ.get("PYTHONPATH", "")

from utils.logger import ExperimentLogger
from models.StackedEnsemble.shared.config_loader import ConfigurationLoader

class OptunaValidator:
    """Optuna-based hyperparameter optimization with validation."""
    
    def __init__(
        self,
        model_type: str = None,
        logger: ExperimentLogger = None):
        """Initialize validator.
        
        Args:
            model_type: Type of model being validated
            logger: Logger instance
        """
        self.logger = logger or ExperimentLogger(f"{model_type}_hypertuning")
        self.model_type = model_type
        self.config_loader = ConfigurationLoader(self.model_type)
            
        # Load optimization configuration
        self.hyperparameter_space = self.config_loader.load_hyperparameter_space(model_type)
        
        # Set up study parameters
        self.study_name = self.optimization_config.get('study_name', f'{model_type}_study')
        self.n_trials = self.optimization_config.get('n_trials', 10)
        self.timeout = self.optimization_config.get('timeout', 72000)  # 20 hours default
        self.direction = self.optimization_config.get('direction', 'maximize')
        self.metric = self.optimization_config.get('metric', 'precision')
        
        # Set up pruning
        pruning_config = self.optimization_config.get('pruning', {})
        self.enable_pruning = pruning_config.get('enable', True)
        self.n_warmup_steps = pruning_config.get('n_warmup_steps', 100)
        self.pruning_interval = pruning_config.get('interval', 100)
        
        # Set up early stopping
        early_stopping_config = self.optimization_config.get('early_stopping', {})
        self.enable_early_stopping = early_stopping_config.get('enable', True)
        self.patience = early_stopping_config.get('patience', 3)
        self.min_delta = early_stopping_config.get('min_delta', 0.01)
        
        # self.logger.info(f"Initialized OptunaValidator for {model_type}")

    def _create_study(self) -> optuna.Study:
        """Create and configure Optuna study."""
        # Set up pruner
        pruner = MedianPruner(
            n_warmup_steps=self.n_warmup_steps,
            n_startup_trials=5,
            interval_steps=self.pruning_interval
        ) if self.enable_pruning else optuna.pruners.NopPruner()
        
        # Set up sampler
        sampler = TPESampler(
            n_startup_trials=5,
            seed=42
        )
        
        # Create study
        study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            sampler=sampler,
            pruner=pruner
        )
        
        return study

    def _suggest_params(self, trial: Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for the trial.
        
        Args:
            trial: Optuna trial instance
            
        Returns:
            Dictionary of suggested parameters
        """
        params = {}
        
        try:
            for param_name, param_config in self.hyperparameter_space['hyperparameters'].items():
                # Skip fixed parameters
                if not isinstance(param_config, dict):
                    params[param_name] = param_config
                    continue
                
                # Get parameter type and distribution
                param_type = param_config.get('type', 'float')
                distribution = param_config.get('distribution', 'uniform')
                
                if param_type == 'int':
                    if distribution == 'int_uniform':
                        params[param_name] = trial.suggest_int(
                            param_name,
                            param_config['min'],
                            param_config['max']
                        )
                elif param_type == 'float':
                    if distribution == 'log_uniform':
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_config['min'],
                            param_config['max'],
                            log=True
                        )
                    elif distribution == 'uniform':
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_config['min'],
                            param_config['max']
                        )
            
            return params
            
        except Exception as e:
            self.logger.error(f"Error suggesting parameters: {str(e)}")
            return self._get_default_params()

    def _get_default_params(self) -> Dict[str, Any]:
        """Get default parameters if suggestion fails."""
        try:
            # Get default values from hyperparameter space
            defaults = {}
            for param_name, param_config in self.hyperparameter_space['hyperparameters'].items():
                if isinstance(param_config, dict):
                    if 'min' in param_config and 'max' in param_config:
                        defaults[param_name] = (param_config['min'] + param_config['max']) / 2
                else:
                    defaults[param_name] = param_config
            
            # Add required fixed parameters
            defaults.update({
                'device': 'cpu',
                'fp16': False,
                'max_grad_norm': 1.0,
                'num_workers': 2,
                'num_labels': 2,
                'problem_type': 'single_label_classification',
                'model_name': 'bert-base-uncased',
                'evaluation_strategy': 'steps',
                'save_strategy': 'steps',
                'metric_for_best_model': 'precision',
                'greater_is_better': True,
                'load_best_model_at_end': True,
                'save_total_limit': 2,
                'remove_unused_columns': True,
                'dataloader_drop_last': False,
                'gradient_checkpointing': False,
                'local_rank': -1,
                'seed': 42,
                'data_seed': 42,
                'ddp_find_unused_parameters': False,
                'use_cpu': True,
                'disable_tqdm': False,
                'report_to': 'none',
                'logging_strategy': 'steps',
                'logging_first_step': True,
                'logging_nan_inf_filter': True
            })
            
            return defaults
            
        except Exception as e:
            self.logger.error(f"Error getting default parameters: {str(e)}")
            return {
                'learning_rate': 2e-5,
                'per_device_train_batch_size': 4,
                'gradient_accumulation_steps': 4,
                'num_train_epochs': 3,
                'warmup_ratio': 0.1,
                'weight_decay': 0.01,
                'hidden_dropout_prob': 0.1,
                'attention_probs_dropout_prob': 0.1,
                'max_seq_length': 256,
                'device': 'cpu',
                'fp16': False
            }

    def optimize_hyperparameters(
        self,
        model: Any,
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        X_test: Any,
        y_test: Any) -> Dict[str, Any]:
        """Run hyperparameter optimization with validation.
        
        Args:
            model: Model instance
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data
            
        Returns:
            Best hyperparameters found
        """
        # self.logger.info("Starting hyperparameter optimization")
        
        def objective(trial: Trial) -> float:
            """Objective function for optimization."""
            # Get hyperparameters for this trial
            params = self._suggest_params(trial)
            
            try:
                # Train and evaluate model
                metrics = model.fit(
                    X_train, y_train,
                    X_val, y_val,
                    X_test, y_test,
                    **params
                )
                
                # Get optimization metric
                score = metrics.get(self.metric, 0.0)
                
                # Report values for pruning
                trial.report(score, step=1)
                
                # Handle pruning
                if trial.should_prune():
                    self.logger.info(f"Trial {trial.number} pruned with {self.metric}: {score:.4f}")
                    raise optuna.TrialPruned()
                
                self.logger.info(f"Trial {trial.number} finished with {self.metric}: {score:.4f}")
                return score
                
            except Exception as e:
                self.logger.error(f"Error in trial {trial.number}: {str(e)}")
                raise optuna.TrialPruned()
        
        try:
            # Create and run study
            study = self._create_study()
            study.optimize(
                objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                show_progress_bar=True
            )
            
            # Get best parameters
            best_params = study.best_params
            
            # Add fixed parameters
            for param_name, param_config in self.hyperparameter_space['hyperparameters'].items():
                if not isinstance(param_config, dict):
                    best_params[param_name] = param_config
            
            self.logger.info(f"Best trial achieved {self.metric}: {study.best_value:.4f}")
            self.logger.info(f"Best parameters: {best_params}")
            
            # Save study results
            self._save_study_results(study)
            
            return best_params
            
        except Exception as e:
            self.logger.error(f"Error in hyperparameter optimization: {str(e)}")
            return self._get_default_params()

    def _save_study_results(self, study: optuna.Study) -> None:
        """Save study results to file.
        
        Args:
            study: Completed Optuna study
        """
        try:
            # Create results directory
            results_dir = Path(project_root) / "results" / "hypertuning"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare results
            results = {
                'study_name': study.study_name,
                'best_trial': {
                    'number': study.best_trial.number,
                    'value': study.best_value,
                    'params': study.best_params
                },
                'n_trials': len(study.trials),
                'optimization_history': [
                    {
                        'trial': t.number,
                        'value': t.value,
                        'params': t.params,
                        'state': t.state.name
                    }
                    for t in study.trials
                ]
            }
            
            # Save results
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            results_file = results_dir / f"{self.model_type}_optuna_results_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
                
            self.logger.info(f"Study results saved to {results_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving study results: {str(e)}") 