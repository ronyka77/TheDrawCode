"""
XGBoost Population-Based Training (PBT) pipeline implementation.
Focuses on precision optimization through evolutionary hyperparameter tuning.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score
import ray
from ray import tune, train
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.schedulers.pb2 import PB2
from ray.train import SyncConfig, Checkpoint
import mlflow
from scipy import stats
import random

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.logger import ExperimentLogger
from utils.create_evaluation_set import (
    create_ensemble_evaluation_set,
    import_selected_features_ensemble,
    import_training_data_ensemble,
    setup_mlflow_tracking
)

experiment_name = "xgboost_pbt_pipeline"
logger = ExperimentLogger(experiment_name=experiment_name, log_dir='./logs/xgboost_pbt_pipeline')
mlruns_dir = setup_mlflow_tracking(experiment_name)

def _trial_dirname_creator(trial: "Trial") -> str:
    """Create shorter trial directory names."""
    return f"trial_{trial.trial_id}"

class XGBoostPBTTrainer:
    """XGBoost trainer with Population-Based Training for precision optimization."""
    
    MIN_SAMPLES: int = 1000
    DEFAULT_THRESHOLD: float = 0.50
    TARGET_PRECISION: float = 0.5
    TARGET_RECALL: float = 0.15
    POPULATION_SIZE: int = 8
    PERTURBATION_INTERVAL: int = 4

    def __init__(
        self,
        logger: Optional[ExperimentLogger] = None,
        max_runtime_secs: int = 7200,  # Increased for population evolution
        seed: int = 42,
        model_dir: Optional[str] = None,
        population_size: int = POPULATION_SIZE):
        
        self.logger = logger or ExperimentLogger(
            experiment_name=experiment_name,
            log_dir='./logs/xgboost_pbt_pipeline'
        )
        self.max_runtime_secs = max_runtime_secs
        self.seed = seed
        self.population_size = population_size
        self.model_dir = model_dir or os.path.join(project_root, "models", "xgboost_pbt_pipeline")
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.models = []
        self.threshold = self.DEFAULT_THRESHOLD
        self.best_precision = 0
        self.best_recall = 0
        
        self._setup_ray()
        
    def _setup_ray(self) -> None:
        """Initialize Ray with appropriate resources."""
        if not ray.is_initialized():
            ray.init(
                num_cpus=os.cpu_count(),
                _temp_dir=os.path.join(self.model_dir, "ray_temp"),
                include_dashboard=True
            )

    def _get_pbt_scheduler(self) -> PopulationBasedTraining:
        """Configure the PBT scheduler with precision-focused perturbation."""
        return PopulationBasedTraining(
            time_attr="training_iteration",
            metric="precision",
            mode="max",
            perturbation_interval=self.PERTURBATION_INTERVAL,
            hyperparam_mutations={
                "learning_rate": tune.loguniform(0.0001, 0.1),
                "max_depth": [3, 4, 5, 6],
                "min_child_weight": [1, 2, 3, 4, 5],
                "subsample": tune.uniform(0.5, 1.0),
                "colsample_bytree": tune.uniform(0.5, 1.0),
                "scale_pos_weight": tune.uniform(1.0, 5.0),
                "gamma": tune.uniform(0, 10),
                "lambda": tune.uniform(0.1, 10),
                "alpha": tune.uniform(0, 10)
            },
            custom_explore_fn=self._custom_explore
        )
    
    def _custom_explore(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Custom exploration strategy focusing on precision improvement."""
        new_config = config.copy()
        
        # Adaptive learning rate adjustment
        if config["precision"] > 0.4:  # If precision is good, fine-tune
            new_config["learning_rate"] *= 0.8
        else:  # If precision is poor, try larger steps
            new_config["learning_rate"] *= 1.2
        
        # Adjust class weight based on current precision/recall trade-off
        if config["recall"] > self.TARGET_RECALL * 1.2:  # If recall is too high
            new_config["scale_pos_weight"] *= 0.9  # Reduce positive class weight
        elif config["precision"] < self.TARGET_PRECISION:  # If precision needs improvement
            new_config["scale_pos_weight"] *= 1.1  # Increase positive class weight
        
        # Regularization adaptation
        if config["precision"] < self.TARGET_PRECISION:
            # Increase regularization to reduce overfitting
            new_config["lambda"] *= 1.2
            new_config["alpha"] *= 1.2
        else:
            # Reduce regularization to allow more complex patterns
            new_config["lambda"] *= 0.9
            new_config["alpha"] *= 0.9
        
        return new_config

    def _setup_search_space(self) -> Dict[str, Any]:
        """Define initial hyperparameter search space for the population."""
        return {
            # Core parameters
            "learning_rate": tune.loguniform(0.001, 0.01),
            "n_estimators": tune.randint(3000, 8000),
            "max_depth": tune.randint(3, 6),
            
            # Sampling parameters
            "subsample": tune.uniform(0.7, 0.9),
            "colsample_bytree": tune.uniform(0.7, 0.9),
            "colsample_bylevel": tune.uniform(0.7, 0.9),
            
            # Regularization
            "lambda": tune.loguniform(0.1, 10.0),
            "alpha": tune.loguniform(0.1, 10.0),
            "gamma": tune.uniform(0, 10),
            
            # Class imbalance
            "scale_pos_weight": tune.uniform(2.0, 4.0),
            "min_child_weight": tune.randint(4, 8),
            
            # Fixed parameters
            "tree_method": "hist",
            "objective": "binary:logistic",
            "eval_metric": ["aucpr", "map"],
            "grow_policy": "lossguide",
            "max_bin": 256
        }

    def _train_population(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series) -> None:
        """Train a population of models using PBT."""
        
        def training_function(config: Dict[str, Any], checkpoint_dir: Optional[str] = None):
            """Training function for a single model in the population."""
            # Set seeds for reproducibility
            np.random.seed(config.get("seed", 19))
            random.seed(config.get("seed", 19))
            
            model = xgb.XGBClassifier(**config)
            start_iter = 0
            
            # Load checkpoint if exists
            if checkpoint_dir:
                model_path = os.path.join(checkpoint_dir, "model.json")
                if os.path.exists(model_path):
                    model.load_model(model_path)
                    # Get iteration from checkpoint
                    checkpoint_name = os.path.basename(checkpoint_dir)
                    if checkpoint_name.startswith("checkpoint_"):
                        start_iter = int(checkpoint_name.split("_")[1])
            
            for iteration in range(start_iter, 100):
                # Train for a few boosting rounds
                model.fit(
                    X_train, y_train,
                    xgb_model=model if iteration > 0 else None,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
                
                # Evaluate current performance
                y_pred = model.predict_proba(X_val)[:, 1]
                precision = precision_score(y_val, y_pred > self.threshold)
                recall = recall_score(y_val, y_pred > self.threshold)
                
                # Save checkpoint using Ray Train
                trial_dir = train.get_context().get_trial_dir()
                checkpoint_dir = os.path.join(trial_dir, "checkpoints", f"checkpoint_{iteration}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                model_path = os.path.join(checkpoint_dir, "model.json")
                model.save_model(model_path)
                
                # Create checkpoint object
                checkpoint = Checkpoint.from_directory(checkpoint_dir)
                
                # Report metrics with checkpoint
                train.report(
                    {
                        "training_iteration": iteration,
                        "precision": precision,
                        "recall": recall,
                        **{k: v for k, v in config.items() if isinstance(v, (int, float))}
                    },
                    checkpoint=checkpoint
                )
                
                # Early stopping if precision target met with sufficient recall
                if precision >= self.TARGET_PRECISION and recall >= self.TARGET_RECALL:
                    break
        
        # Configure PBT
        pbt = self._get_pbt_scheduler()
        
        # Setup storage path - using shorter base path
        storage_path = os.path.abspath(os.path.join(project_root, "ray_results"))
        os.makedirs(storage_path, exist_ok=True)
        
        # Run population training with updated parameters
        tuner = tune.Tuner(
            training_function,
            param_space=self._setup_search_space(),
            tune_config=tune.TuneConfig(
                scheduler=pbt,
                num_samples=self.population_size,
                time_budget_s=self.max_runtime_secs,
                trial_dirname_creator=_trial_dirname_creator,
                trial_name_creator=lambda trial: f"trial_{trial.trial_id}",
            ),
            run_config=train.RunConfig(
                storage_path=storage_path,
                name="pbt_tune",
                sync_config=SyncConfig(
                    sync_artifacts_on_checkpoint=True,
                    sync_period=1
                ),
                log_to_file=True
            )
        )
        
        # Run the tuner
        results = tuner.fit()
        
        # Get best trial and checkpoint
        best_result = results.get_best_result(
            metric="precision",
            mode="max",
            scope="last"
        )
        
        if best_result:
            best_checkpoint = best_result.checkpoint
            
            if best_checkpoint:
                # Load best model
                best_model = xgb.XGBClassifier()
                with best_checkpoint.as_directory() as checkpoint_dir:
                    model_path = os.path.join(checkpoint_dir, "model.json")
                    if os.path.exists(model_path):
                        best_model.load_model(model_path)
                        self.models = [best_model]
                        
                        # Update best metrics
                        self.best_precision = best_result.metrics["precision"]
                        self.best_recall = best_result.metrics["recall"]
                        
                        # Log results
                        self.logger.info(f"\nBest model achieved:")
                        self.logger.info(f"Precision: {self.best_precision:.4f}")
                        self.logger.info(f"Recall: {self.best_recall:.4f}")
                        
                        # Log evolution path
                        for trial_result in results.results.values():
                            if "precision" in trial_result.metrics and "recall" in trial_result.metrics:
                                mlflow.log_metrics({
                                    f"trial_{trial_result.trial_id}_precision": trial_result.metrics["precision"],
                                    f"trial_{trial_result.trial_id}_recall": trial_result.metrics["recall"]
                                })
                    else:
                        self.logger.warning("Best model checkpoint not found")
            else:
                self.logger.warning("No checkpoint found in best result")
        else:
            self.logger.warning("No successful trials found")

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series) -> Dict[str, Any]:
        """Main training method using population-based training."""
        try:
            with mlflow.start_run(run_name=f"xgboost_pbt_{datetime.now().strftime('%Y%m%d_%H%M')}"):
                # Log training parameters
                mlflow.log_params({
                    "population_size": self.population_size,
                    "max_runtime_secs": self.max_runtime_secs,
                    "target_precision": self.TARGET_PRECISION,
                    "target_recall": self.TARGET_RECALL,
                    "perturbation_interval": self.PERTURBATION_INTERVAL
                })
                
                # Train population
                self._train_population(X_train, y_train, X_val, y_val)
                
                # Save best model
                if self.models:
                    best_model_path = os.path.join(self.model_dir, "best_model.json")
                    self.models[0].save_model(best_model_path)
                    mlflow.log_artifact(best_model_path)
                
                return {
                    "models": self.models,
                    "best_precision": self.best_precision,
                    "best_recall": self.best_recall,
                    "threshold": self.threshold
                }
                
        except Exception as e:
            self.logger.error(f"Error in PBT training: {str(e)}")
            raise
        finally:
            ray.shutdown()

def train_xgboost_pbt_pipeline():
    """Main function to train XGBoost models using PBT."""
    try:
        # Initialize trainer
        trainer = XGBoostPBTTrainer(
            logger=logger,
            max_runtime_secs=7200,
            seed=42
        )
        
        # Load and prepare data
        selected_features = import_selected_features_ensemble('all')
        X_train, y_train, X_test, y_test = import_training_data_ensemble()
        X_val, y_val = create_ensemble_evaluation_set()
        
        X_train = X_train[selected_features]
        X_val = X_val[selected_features]
        
        # Train models
        results = trainer.train(X_train, y_train, X_val, y_val)
        
        # Print results
        print("\nFinal Model Performance:")
        print("-" * 80)
        print(f"Best Precision: {results['best_precision']:.4f}")
        print(f"Best Recall: {results['best_recall']:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in PBT pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    results = train_xgboost_pbt_pipeline() 