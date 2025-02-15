"""
XGBoost CPU-only pipeline implementation for the Soccer Prediction Project.
Designed to work alongside the H2O stacked ensemble while providing Windows compatibility.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
import optuna
from optuna.samplers import TPESampler
import mlflow
from scipy import stats

# Add project root to Python path
try:
    project_root = Path(__file__).parent.parent.parent
    if not project_root.exists():
        project_root = Path(r"\\".join(str(project_root).split("\\")))
    sys.path.append(str(project_root))
except Exception as e:
    print(f"Error setting project root path: {e}")
    sys.path.append(os.getcwd().parent.parent)

# Local imports
from utils.logger import ExperimentLogger
experiment_name = "xgboost_cpu_pipeline"
logger = ExperimentLogger(experiment_name=experiment_name, log_dir='./logs/xgboost_cpu_pipeline')

from utils.create_evaluation_set import (
    create_ensemble_evaluation_set,
    import_selected_features_ensemble,
    import_training_data_ensemble,
    setup_mlflow_tracking
)

mlruns_dir = setup_mlflow_tracking(experiment_name)

class XGBoostModel:
    """XGBoost model wrapper with memory optimization."""
    
    def __init__(
        self,
        params: Dict[str, Any],
        logger: Any,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50):
        self.params = params
        self.train_params = {k: v for k, v in params.items() if k not in ['n_estimators', 'early_stopping_rounds']}
        self.logger = logger
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None
        self._best_iteration = None
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, eval_set: List[Tuple[pd.DataFrame, pd.Series]]) -> None:
        """Train the model using pandas DataFrames directly."""
        try:
            # Create classifier
            self.model = xgb.XGBClassifier(
                **self.train_params,
                n_estimators=self.num_boost_round,
                early_stopping_rounds=self.early_stopping_rounds
            )
            
            # Train model
            self.model.fit(
                X_train, 
                y_train,
                eval_set=eval_set,
                verbose=False
            )
            
            # Store best iteration
            self._best_iteration = self.model.best_iteration
            
            # Log training results
            self.logger.info(f"Best iteration: {self._best_iteration}")
            evals_result = self.model.evals_result()
            if evals_result:
                for metric, values in evals_result['validation_0'].items():
                    best_value = min(values) if 'error' in metric else max(values)
                    self.logger.info(f"Best {metric}: {best_value:.4f}")
            
            # Force garbage collection
            import gc
            gc.collect()
            
        except Exception as e:
            self.logger.error(f"Error fitting XGBoost model: {str(e)}")
            raise
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using pandas DataFrame directly."""
        try:
            # Process in chunks for memory efficiency
            chunk_size = 10000
            predictions = []
            
            for i in range(0, len(X), chunk_size):
                chunk = X.iloc[i:i + chunk_size]
                chunk_preds = self.model.predict(chunk)
                predictions.append(chunk_preds)
            
            return np.concatenate(predictions)
            
        except Exception as e:
            self.logger.error(f"Error predicting with XGBoost model: {str(e)}")
            raise
            
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities using pandas DataFrame directly."""
        try:
            # Process in chunks for memory efficiency
            chunk_size = 10000
            probabilities = []
            
            for i in range(0, len(X), chunk_size):
                chunk = X.iloc[i:i + chunk_size]
                chunk_probs = self.model.predict_proba(chunk)
                probabilities.append(chunk_probs)
            
            return np.concatenate(probabilities)
            
        except Exception as e:
            self.logger.error(f"Error getting probabilities from XGBoost model: {str(e)}")
            raise
            
    def save(self, path: str) -> None:
        """Save the model using MLflow."""
        try:
            # Log model with MLflow
            # Create signature for model input/output validation
            input_example = pd.DataFrame(columns=self.model.feature_names)
            signature = mlflow.models.infer_signature(
                input_example,
                self.model.predict(input_example)
            )

            # Log model with MLflow, ensuring all parameters are serializable
            mlflow.xgboost.log_model(
                xgb_model=self.model,
                artifact_path=path,
                registered_model_name=f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M')}",
                signature=signature,
                metadata={
                    'best_iteration': int(self._best_iteration),  # Ensure serializable
                    'params': {k: str(v) for k, v in self.params.items()}  # Convert all params to strings
                }
            )
        except Exception as e:
            self.logger.error(f"Error saving XGBoost model: {str(e)}")
            raise
            
    def load(self, path: str) -> None:
        """Load the model from MLflow."""
        try:
            # Load model from MLflow
            self.model = mlflow.xgboost.load_model(path)
            
            # Get metadata from MLflow
            metadata = mlflow.get_run(path).data.params
            self._best_iteration = int(metadata.get('best_iteration', 0))
            self.params.update({k: v for k, v in metadata.items() 
                                if k not in ['best_iteration']})
                
        except Exception as e:
            self.logger.error(f"Error loading XGBoost model: {str(e)}")
            raise
                
        except Exception as e:
            self.logger.error(f"Error loading XGBoost model: {str(e)}")
            raise

class XGBoostCPUTrainer:
    """XGBoost trainer with CPU-only configuration and precision-focused tuning."""
    
    MIN_SAMPLES: int = 1000
    DEFAULT_THRESHOLD: float = 0.50
    TARGET_PRECISION: float = 0.5
    TARGET_RECALL: float = 0.15
    N_SPLITS: int = 5  # Number of cross-validation folds
    N_REPEATS: int = 3  # Number of CV repetitions for nested CV

    def __init__(
        self,
        logger: Optional[ExperimentLogger] = None,
        max_runtime_secs: int = 3600,
        seed: int = 19,
        max_models: int = 30,
        model_dir: Optional[str] = None,
        n_trials: int = 100,
        cv_strategy: str = 'stratified'):  # Added cv_strategy parameter
        self.logger = logger or ExperimentLogger(
            experiment_name="xgboost_cpu_pipeline",
            log_dir='./logs/xgboost_cpu_pipeline'
        )
        self.max_runtime_secs = max_runtime_secs
        self.seed = seed
        self.max_models = max_models
        self.model_dir = model_dir or os.path.join(project_root, "models", "xgboost_cpu_pipeline")
        os.makedirs(self.model_dir, exist_ok=True)
        self.n_trials = n_trials
        self.cv_strategy = cv_strategy
        self._setup_cv_splitter()
        
        self.models: List[XGBoostModel] = []
        self.threshold = self.DEFAULT_THRESHOLD
        self.best_precision = 0
        self.best_recall = 0
        self._check_logger()
        self._monitor_memory_usage()

    def _check_logger(self) -> None:
        """Verify logger is properly initialized and functional."""
        if not self.logger:
            raise ValueError("Logger not initialized")
        try:
            self.logger.info("Logger check: Initialization successful")
        except Exception as e:
            raise ValueError(f"Logger test failed: {str(e)}")

    def _monitor_memory_usage(self) -> None:
        """Monitor memory usage during training."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_usage_gb = memory_info.rss / (1024 * 1024 * 1024)
            
            self.logger.info(f"Current memory usage: {memory_usage_gb:.2f} GB")
            
            if memory_usage_gb > 12:
                self.logger.warning(f"High memory usage detected: {memory_usage_gb:.2f} GB")
                
        except ImportError:
            self.logger.warning("psutil not installed, memory monitoring disabled")
        except Exception as e:
            self.logger.error(f"Error monitoring memory: {str(e)}")

    def _setup_cv_splitter(self) -> None:
        """Initialize the appropriate cross-validation splitter based on strategy."""
        if self.cv_strategy == 'stratified':
            self.cv_splitter = StratifiedKFold(
                n_splits=self.N_SPLITS,
                shuffle=True,
                random_state=self.seed
            )
        elif self.cv_strategy == 'timeseries':
            self.cv_splitter = TimeSeriesSplit(
                n_splits=self.N_SPLITS,
                test_size=int(len(self.X_train) / self.N_SPLITS)
            )
        else:
            raise ValueError(f"Unsupported CV strategy: {self.cv_strategy}")

    def _calculate_cv_statistics(self, scores: List[float]) -> Dict[str, float]:
        """Calculate detailed statistics for cross-validation scores."""
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        ci = stats.t.interval(
            confidence=0.95,
            df=len(scores)-1,
            loc=mean_score,
            scale=stats.sem(scores)
        )
        return {
            'mean': mean_score,
            'std': std_score,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'n_samples': len(scores)
        }

    def _evaluate_fold(
        self,
        model: XGBoostModel,
        X_fold_val: pd.DataFrame,
        y_fold_val: pd.Series) -> Dict[str, float]:
        """Evaluate model performance on a single fold."""
        val_probs = model.predict_proba(X_fold_val)[:, 1]
        threshold, metrics = self._find_optimal_threshold(val_probs, y_fold_val)
        
        return {
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'threshold': threshold,
            'f1': 2 * (metrics['precision'] * metrics['recall']) / 
                    (metrics['precision'] + metrics['recall'] + 1e-10)
        }

    def _setup_hyperparameters(self) -> Dict[str, Any]:
        """Define hyperparameter search space using Optuna with enhanced CV."""
        def objective(trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> float:
            """Optuna objective function with enhanced cross-validation."""
            # Define hyperparameter space
            params = {
                "tree_method": "hist",
                "objective": "binary:logistic",
                "eval_metric": ["aucpr", "map", "ndcg"],
                "verbosity": 0,
                "seed": self.seed,
                
                # Learning rate and boosting parameters
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.01, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 3000, 8000),
                "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 300, 500),
                
                # Tree-specific parameters
                "max_depth": trial.suggest_int("max_depth", 3, 6),
                "min_child_weight": trial.suggest_int("min_child_weight", 4, 8),
                "max_bin": trial.suggest_int("max_bin", 200, 300),
                
                # Sampling parameters
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.9),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.7, 0.9),
                "colsample_bynode": trial.suggest_float("colsample_bynode", 0.7, 0.9),
                "subsample": trial.suggest_float("subsample", 0.7, 0.9),
                
                # Regularization
                "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 2.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
                "gamma": trial.suggest_float("gamma", 0.1, 15.0, log=True),
                
                # Class weight and growth control
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 2.0, 3.0),
                "max_delta_step": trial.suggest_float("max_delta_step", 3, 10),
                "max_leaves": trial.suggest_int("max_leaves", 8, 32),
                "min_split_loss": trial.suggest_float("min_split_loss", 0, 10),
                "grow_policy": "lossguide"
            }
            
            # Initialize metrics storage
            cv_metrics = {
                'precision': [],
                'recall': [],
                'f1': [],
                'threshold': []
            }
            
            # Perform cross-validation
            for fold_idx, (train_idx, val_idx) in enumerate(self.cv_splitter.split(X_train, y_train)):
                X_fold_train = X_train.iloc[train_idx]
                y_fold_train = y_train.iloc[train_idx]
                
                # Create and train model
                model = XGBoostModel(
                    params=params,
                    logger=self.logger,
                    num_boost_round=params['n_estimators'],
                    early_stopping_rounds=params['early_stopping_rounds']
                )
                
                model.fit(
                    X_fold_train,
                    y_fold_train,
                    eval_set=[(X_test, y_test)]
                )
                
                # Evaluate fold
                fold_metrics = self._evaluate_fold(model, X_val, y_val)
                
                # Store metrics
                for metric, value in fold_metrics.items():
                    cv_metrics[metric].append(value)
                
                # Calculate intermediate value for pruning
                fold_score = 0.7 * fold_metrics['precision'] + 0.3 * fold_metrics['recall']
                
                # Report intermediate value for pruning
                trial.report(fold_score, fold_idx)
                
                if trial.should_prune():
                    self.logger.info(f"Trial {trial.number} pruned at fold {fold_idx}")
                    raise optuna.TrialPruned()
                
                # Clean up
                del model
                import gc
                gc.collect()
            
            # Calculate detailed CV statistics
            cv_stats = {
                metric: self._calculate_cv_statistics(scores)
                for metric, scores in cv_metrics.items()
            }
            
            # Log detailed CV results
            self.logger.info(f"\nTrial {trial.number} CV Results:")
            for metric, stats in cv_stats.items():
                self.logger.info(
                    f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                    f"(95% CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}])"
                )
            
            # Log to MLflow
            for metric, stats in cv_stats.items():
                for stat_name, value in stats.items():
                    mlflow.log_metric(
                        f"trial_{trial.number}_{metric}_{stat_name}",
                        value
                    )
            
            # Calculate final score with confidence
            if cv_stats['recall']['ci_lower'] < self.TARGET_RECALL:
                return float('-inf')
            
            # Weighted score favoring precision with confidence
            score = (
                cv_stats['precision']['mean'] - 
                0.1 * cv_stats['precision']['std']  # Penalty for high variance in precision
            )
            
            return score
        
        # Create Optuna study with TPE sampler and MedianPruner
        sampler = TPESampler(seed=self.seed)
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=7,     # Increased from 5
            n_warmup_steps=3,       # Increased from 2
            interval_steps=1
        )
        
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner
        )
        
        # Run optimization
        study.optimize(
            lambda trial: objective(trial, self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val),
            n_trials=self.n_trials,
            timeout=self.max_runtime_secs,
            show_progress_bar=True  # Show progress bar for better monitoring
        )
        
        # Log pruning statistics
        n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        
        self.logger.info(f"\nPruning Statistics:")
        self.logger.info(f"Total Trials: {len(study.trials)}")
        self.logger.info(f"Completed Trials: {n_complete}")
        self.logger.info(f"Pruned Trials: {n_pruned}")
        self.logger.info(f"Pruning Rate: {n_pruned/len(study.trials)*100:.2f}%")
        
        # Get best parameters
        best_params = study.best_params
        best_params.update({
            "tree_method": "hist",
            "objective": "binary:logistic",
            "eval_metric": ["aucpr", "map", "ndcg"],
            "seed": self.seed
        })
        
        # Log best parameters
        self.logger.info("\nBest trial:")
        self.logger.info(f"Value: {study.best_value:.4f}")
        self.logger.info("Params:")
        for key, value in best_params.items():
            self.logger.info(f"    {key}: {value}")
        
        # Log pruning statistics to MLflow
        mlflow.log_metrics({
            "pruned_trials": n_pruned,
            "completed_trials": n_complete,
            "pruning_rate": n_pruned/len(study.trials)
        })
        
        return best_params

    def _train_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series) -> None:
        """Train multiple XGBoost models with memory-efficient batch processing."""
        try:
            # Store training data for hyperparameter optimization
            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test
            self.X_val = X_val
            self.y_val = y_val
            
            # Get optimized hyperparameters
            best_params = self._setup_hyperparameters()
            
            # Train final model with best parameters
            self.logger.info("Training final model with best parameters...")
            model = XGBoostModel(
                params=best_params,
                logger=self.logger,
                num_boost_round=best_params['n_estimators'],
                early_stopping_rounds=best_params['early_stopping_rounds']
            )
            
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)]
            )
                
                # Evaluate model
            val_preds = model.predict_proba(X_val)[:, 1]
                optimal_threshold, metrics = self._find_optimal_threshold(val_preds, y_val)
                
            if metrics['recall'] >= self.TARGET_RECALL:
                self.models.append(model)
                self.threshold = optimal_threshold
                    self.best_precision = metrics['precision']
                    self.best_recall = metrics['recall']
                
                    self.logger.info(
                f"Final model metrics - "
                        f"Precision: {metrics['precision']:.4f}, "
                f"Recall: {metrics['recall']:.4f}, "
                f"Threshold: {optimal_threshold:.4f}"
                )
            else:
                self.logger.warning(
                    f"Final model did not meet recall threshold. "
                    f"Got {metrics['recall']:.4f}, needed {self.TARGET_RECALL}"
                )
            
            # Force garbage collection
            import gc
            gc.collect()
            
        except Exception as e:
            self.logger.error(f"Error training models: {str(e)}")
            raise

    def _evaluate_predictions(
        self,
        probabilities: np.ndarray,
        actuals: pd.Series,
        threshold: Optional[float] = None) -> Dict[str, float]:
        """Evaluate binary classification metrics."""
        try:
            threshold = threshold or self.threshold
            predictions = (probabilities >= threshold).astype(int)
            
            metrics = {
                'precision': precision_score(actuals, predictions, pos_label=1, zero_division=0),
                'recall': recall_score(actuals, predictions, pos_label=1, zero_division=0)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating predictions: {str(e)}")
            raise

    def _find_optimal_threshold(
        self,
        probabilities: np.ndarray,
        actuals: pd.Series) -> Tuple[float, Dict[str, float]]:
        """Find optimal prediction threshold prioritizing precision while ensuring recall."""
        try:
            self.logger.info("Finding optimal threshold...")
            best_metrics = {
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'threshold': self.DEFAULT_THRESHOLD
            }
            best_score = 0
            
            for threshold in np.arange(0.2, 0.80, 0.01):
                metrics = self._evaluate_predictions(probabilities, actuals, threshold)
                
                if metrics['recall'] >= self.TARGET_RECALL:
                    score = metrics['precision']
                    if score > best_score:
                        best_score = score
                        best_metrics = {**metrics, 'threshold': threshold}
            
            self.logger.info(
                f"Optimal threshold {best_metrics['threshold']:.3f}: "
                f"Precision={best_metrics['precision']:.4f}, "
                f"Recall={best_metrics['recall']:.4f}"
            )
            return best_metrics['threshold'], best_metrics
            
        except Exception as e:
            self.logger.error(f"Error in threshold optimization: {str(e)}")
            raise

    def _save_models(self) -> None:
        """Save trained models to disk."""
        try:
            for i, model in enumerate(self.models):
                model_path = os.path.join(self.model_dir, f"xgboost_{i}.json")
                model.save(model_path)
                self.logger.info(f"Saved model to: {model_path}")
                
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")
            raise

    def _log_training_results(self, test_metrics: Dict[str, float]) -> None:
        """Log training results to MLflow."""
        try:
            mlflow.log_params({
                "max_runtime_secs": self.max_runtime_secs,
                "seed": self.seed,
                "n_models": len(self.models),
                "optimal_threshold": self.threshold
            })
            
            mlflow.log_metrics({
                f"test_{k}": v for k, v in test_metrics.items()
            })
            
            mlflow.log_artifacts(self.model_dir, "models")
            
        except Exception as e:
            self.logger.error(f"Error logging results: {str(e)}")
            raise

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series) -> Dict[str, Any]:
        """Train XGBoost model with validation-based threshold optimization."""
        try:
            # Train model
            self._train_models(X_train, y_train, X_test, y_test, X_val, y_val)
            
            if not self.models:
                raise ValueError("No models met the recall threshold")
            
            # Evaluate on test set
            model = self.models[0]  # We now only have one best model
            test_predictions = model.predict_proba(X_val)[:, 1]
            test_metrics = self._evaluate_predictions(test_predictions, y_val, self.threshold)
            
            # Log final metrics
            mlflow.log_metrics({
                "test_precision": test_metrics['precision'],
                "test_recall": test_metrics['recall'],
                "final_threshold": self.threshold,
                "best_precision": self.best_precision,
                "best_recall": self.best_recall
            })
            
            # Save model
            self._save_models()
            
            # Log feature importance plot if available
            if hasattr(model.model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': model.model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 6))
                plt.bar(range(len(feature_importance)), feature_importance['importance'])
                plt.xticks(range(len(feature_importance)), feature_importance['feature'], rotation=45, ha='right')
                plt.title('Feature Importance')
                plt.tight_layout()
                
                # Save plot
                plot_path = os.path.join(self.model_dir, 'feature_importance.png')
                plt.savefig(plot_path)
                plt.close()
                
                # Log to MLflow
                mlflow.log_artifact(plot_path, "feature_importance")
            
            return {
                'models': self.models,
                'threshold': self.threshold,
                'test_metrics': test_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}")
            raise

def train_xgboost_pipeline():
    """Main function to train the XGBoost models."""
    try:
        trainer = XGBoostCPUTrainer(
            logger=logger,
            max_runtime_secs=3600,
            seed=42
        )
        
        selected_features = import_selected_features_ensemble('all')
        X_train, y_train, X_test, y_test = import_training_data_ensemble()
        X_val, y_val = create_ensemble_evaluation_set()
        
        X_train = X_train[selected_features]
        X_val = X_val[selected_features]
        X_test = X_test[selected_features]
        
        with mlflow.start_run(
            run_name=f"xgboost_cpu_{datetime.now().strftime('%Y%m%d_%H%M')}"
        ):
            # Log dataset statistics
            mlflow.log_params({
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "test_samples": len(X_test),
                "draw_ratio_train": (y_train == 1).mean(),
                "draw_ratio_val": (y_val == 1).mean(),
                "draw_ratio_test": (y_test == 1).mean(),
                "n_trials": trainer.n_trials,
                "n_splits": trainer.N_SPLITS,
                "target_precision": trainer.TARGET_PRECISION,
                "target_recall": trainer.TARGET_RECALL,
                "max_runtime_secs": trainer.max_runtime_secs,
                "seed": trainer.seed
            })
            
            results = trainer.train(
                X_train, y_train,
                X_val, y_val,
                X_test, y_test
            )
            
            # Log artifacts
            mlflow.log_artifacts(trainer.model_dir, "models")
            
            print("\nModel Performance Summary:")
            print("-" * 80)
            print("Test Set Metrics:")
            for k, v in results['test_metrics'].items():
                print(f"{k:20}: {v:.4f}")
            
            return results
            
    except Exception as e:
        logger.error(f"Error during XGBoost training: {str(e)}")
        raise

if __name__ == "__main__":
    results = train_xgboost_pipeline()
    print("\nFinal Model Performance:")
    print("-" * 80)
    print("Test Set:")
    print(f"Precision: {results['test_metrics']['precision']:.4f}")
    print(f"Recall: {results['test_metrics']['recall']:.4f}")