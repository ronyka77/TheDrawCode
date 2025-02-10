"""
CatBoost model hypertuning implementation for binary classification with global training capabilities.
Provides methods for hyperparameter tuning, prediction, and analysis with configurable parameters.
"""

# Standard library imports
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

# Third-party imports
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
import mlflow
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score
from optuna.trial import FrozenTrial


# Szűrd a specifikus figyelmeztetést
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.base")
warnings.filterwarnings("ignore", category=UserWarning, module="catboost")
warnings.filterwarnings("ignore", category=UserWarning, module="optuna.samplers._tpe.sampler")

# Add project root to Python path
try:
    project_root = Path(__file__).parent.parent.parent
    if not project_root.exists():
        # Handle network path by using raw string
        project_root = Path(r"\\".join(str(project_root).split("\\")))
    sys.path.append(str(project_root))
    print(f"Project root catboost_model: {project_root}")
except Exception as e:
    print(f"Error setting project root path: {e}")
    sys.path.append(os.getcwd().parent.parent)
    print(f"Current directory catboost_model: {os.getcwd().parent.parent}")

# Local imports
from utils.logger import ExperimentLogger
experiment_name = "catboost_ensemble_hypertuning"
logger = ExperimentLogger(experiment_name=experiment_name, log_dir='./logs/catboost_ensemble_hypertuning')


from utils.create_evaluation_set import create_ensemble_evaluation_set, import_selected_features_ensemble, import_training_data_ensemble, setup_mlflow_tracking
mlruns_dir = setup_mlflow_tracking(experiment_name)

# Core Configuration for CPU-only training with CatBoost


HYPERPARAM_SPEC = {}
HYPERPARAM_SPEC['sampler'] = optuna.samplers.TPESampler(
    consider_prior=True,
    prior_weight=1.5,
    n_startup_trials=50,
    n_ei_candidates=100,
    seed=42,
    constant_liar=True,
    multivariate=True,
    warn_independent_sampling=False
)

class GlobalHypertuner:
    """
    Global hyperparameter tuner for CatBoost model optimizing for validation metrics.
    
    Attributes:
        MIN_SAMPLES: Minimum number of samples required for training.
        DEFAULT_THRESHOLD: Default prediction threshold.
        TARGET_PRECISION: Target precision threshold.
        TARGET_RECALL: Target recall threshold.
        PRECISION_WEIGHT: Weight for precision in optimization.
        RECALL_CAP: Maximum recall to consider.
    """
    MIN_SAMPLES: int = 1000
    DEFAULT_THRESHOLD: float = 0.50
    TARGET_PRECISION: float = 0.5
    TARGET_RECALL: float = 0.6
    PRECISION_WEIGHT: float = 0.7
    RECALL_CAP: float = 0.30

    def __init__(self, 
                logger: Optional[ExperimentLogger] = None,
                target_precision: float = TARGET_PRECISION,
                target_recall: float = TARGET_RECALL,
                precision_weight: float = PRECISION_WEIGHT,
                hyperparam_spec: Dict = HYPERPARAM_SPEC):
        self.logger = logger or ExperimentLogger()
        self.target_precision = target_precision
        self.target_recall = target_recall
        self.precision_weight = precision_weight
        self.hyperparam_spec = hyperparam_spec
        self.best_params = {}

    def _find_optimal_threshold(
        self,
        model: CatBoostClassifier,
        features_val: pd.DataFrame,
        target_val: pd.Series) -> Tuple[float, Dict[str, float]]:
        """
        Find optimal prediction threshold prioritizing precision while maintaining recall.
        
        Args:
            model: Trained CatBoost model.
            features_val: Validation features.
            target_val: Validation targets.
        
        Returns:
            Tuple of (optimal threshold, metrics dictionary).
        """
        try:
            probas = model.predict_proba(features_val)[:, 1]
            best_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'threshold': self.DEFAULT_THRESHOLD}
            best_score = 0.0
            
            # Focus on higher thresholds for better precision, starting from 0.5 up to 0.65
            for threshold in np.arange(0.5, 0.65, 0.01):
                preds = (probas >= threshold).astype(int)
                recall = recall_score(target_val, preds)
                if recall >= self.target_recall:
                    true_positives = ((preds == 1) & (target_val == 1)).sum()
                    false_positives = ((preds == 1) & (target_val == 0)).sum()
                    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                    f1 = f1_score(target_val, preds)
                    score = precision * min(1.0, (recall - self.target_recall) / self.target_recall)
                    if score > best_score:
                        best_score = score
                        best_metrics.update({
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'threshold': threshold
                        })
                        # self.logger.info(
                        #     f"New best threshold {threshold:.3f}: Precision={precision:.4f}, Recall={recall:.4f}"
                        # )
                        mlflow.log_metrics({
                            'best_threshold': threshold,
                            'best_precision': precision,
                            'best_recall': recall,
                            'best_f1': f1
                        })
            if best_metrics['recall'] < self.target_recall:
                self.logger.warning(
                    f"Could not find threshold meeting recall requirement. Best recall: {best_metrics['recall']:.4f}"
                )
            return best_metrics['threshold'], best_metrics
        except Exception as e:
            self.logger.error(f"Error in threshold optimization: {str(e)}")
            raise

    def objective(
        self, 
        trial: optuna.Trial, 
        features_train: pd.DataFrame, 
        target_train: pd.Series,
        features_val: pd.DataFrame,
        target_val: pd.Series,
        features_test: pd.DataFrame,
        target_test: pd.Series) -> float:
        """
        Optuna objective function that optimizes CatBoost precision while hypertuning CatBoost parameters.
        """
        try:
            # --- Always start from the base dataset ---
            original_X_train = features_train.copy()
            original_y_train = target_train.copy()
            # --- Tune CatBoost parameters using trial suggestions ---
            cat_params = {
                'loss_function': 'Logloss',
                'eval_metric': 'AUC',
                'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 200, 1000),  # Sikeres trialok alapján (290-998)
                'learning_rate': trial.suggest_float('learning_rate', 0.002, 0.05, log=True),  # Legjobb eredmények 0.002-0.04 között
                'depth': trial.suggest_int('depth', 6, 9),  # Trial 7: depth=9, Trial 29: depth=6
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 15, log=True),  # Sikeres értékek 0.1-14.9 között
                'iterations': trial.suggest_int('iterations', 2000, 5000),  # Legtöbb sikeres trial >2000 iteráció
                'border_count': trial.suggest_int('border_count', 128, 224, step=32),  # Optimalizált tartomány a figyelmeztetések elkerülésére
                'subsample': trial.suggest_float('subsample', 0.7, 0.95),  # Sikeres trialok 0.71-0.93 között
                'random_strength': trial.suggest_float('random_strength', 0.1, 7, log=True),  # Legjobb eredmények 0.17-6.2 között
                'auto_class_weights': trial.suggest_categorical('auto_class_weights', ['Balanced']),  # Minden sikeres trial 'Balanced'-t használt
                'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree']),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 80),  # Sikeres trialok 23-80 között
                # 'feature_weights': [trial.suggest_float('feature_weights', 0.5, 1.3) for _ in range(features_train.shape[1])],  # Bővített tartomány a trialok alapján
                'verbose': False
                # 'random_seed': 42
            }
            # --- Create and Train the CatBoost model on oversampled data ---
            model = CatBoostClassifier(**cat_params)
            model.fit(
                features_train, target_train,
                eval_set=(features_test, target_test),
                verbose=False
            )
            
            # --- Find optimal threshold and evaluate model performance ---
            threshold, metrics = self._find_optimal_threshold(model, features_val, target_val)
            
            # Early prune if recall requirement is not met
            if metrics['recall'] < self.target_recall:
                self.logger.info(f"Trial {trial.number} pruned: Recall {metrics['recall']:.4f} < {self.target_recall}")
                raise optuna.exceptions.TrialPruned()
            
            trial.set_user_attr('threshold', metrics['threshold'])
            trial.set_user_attr('precision', metrics['precision'])
            trial.set_user_attr('recall', metrics['recall'])
            trial.set_user_attr('f1', metrics['f1'])
            
            self.logger.info(
                f"Trial {trial.number} results: \n"
                f"Precision={metrics['precision']:.4f}, \n"
                f"Recall={metrics['recall']:.4f}, \n"
                f"Threshold={metrics['threshold']:.4f}"
            )
            
            if trial.should_prune() and trial.user_attrs['precision'] < 0.25:
                raise optuna.exceptions.TrialPruned()
            
            return metrics['precision']
        except Exception as e:
            self.logger.error(f"Error in trial {trial.number}: {str(e)}")
            raise

    def _log_metrics_callback(self, study: optuna.Study, trial: FrozenTrial) -> None:
        """Callback to log metrics for each trial."""
        self.logger.info(f"Trial {trial.number} completed with value: {trial.value}")
        self.logger.info(f"Trial params: {trial.params}")

    def _pruned_trials_callback(self, study: optuna.Study, trial: FrozenTrial) -> None:
        """Callback to track pruned trials."""
        if trial.state == optuna.trial.TrialState.PRUNED:
            self.logger.info(f"Trial {trial.number} was pruned.")

    def tune_model(
        self,
        features_train: pd.DataFrame,
        target_train: pd.Series,
        features_val: pd.DataFrame,
        target_val: pd.Series,
        features_test: pd.DataFrame,
        target_test: pd.Series,
        n_trials: int = 400
    ) -> Dict[str, Any]:
        """Tune global model with enhanced precision-recall tracking."""
        try:
            num_features = features_train.shape[1]
            HYPERPARAM_SPEC['border_count'] = (
                max(32, num_features//2), 
                min(254, num_features*2)
            )
            study = optuna.create_study(
                direction="maximize",
                sampler=self.hyperparam_spec['sampler']
            )
            # Run optimization
            study.optimize(
                lambda trial: self.objective(
                    trial,
                    features_train,
                    target_train,
                    features_val,
                    target_val,
                    features_test,
                    target_test
                ),
                n_trials=n_trials,
                callbacks=[self._log_metrics_callback, self._pruned_trials_callback]
            )
            
            best_trial = study.best_trial
            mlflow.log_metrics({
                "best_precision": best_trial.user_attrs["precision"],
                "best_recall": best_trial.user_attrs["recall"],
                "best_threshold": best_trial.user_attrs["threshold"],
                "n_pruned_trials": len(study.get_trials(states=[optuna.trial.TrialState.PRUNED])),
                "n_completed_trials": len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))
            })
            self.best_params = study.best_params
            return study.best_params
        except Exception as e:
            self.logger.error(f"Error during hyperparameter tuning: {str(e)}")
            raise

    def evaluate_model(
        self,
        features_train: pd.DataFrame,
        target_train: pd.Series,
        features_eval: pd.DataFrame,
        target_eval: pd.Series,
        features_other: pd.DataFrame,
        target_other: pd.Series
    ) -> Dict[str, float]:
        """Evaluate the model performance on a given evaluation set."""
        try:
            model = CatBoostClassifier(
                loss_function='Logloss',
                eval_metric='AUC',
                learning_rate=self.best_params.get('learning_rate', 0.01),
                depth=self.best_params.get('depth', 6),
                l2_leaf_reg=self.best_params.get('l2_leaf_reg', 1),
                iterations=self.best_params.get('iterations', 5000),
                border_count=self.best_params.get('border_count', 64),
                od_type='Iter',
                od_wait=self.best_params.get('od_wait', 50),
                verbose=False,
                random_seed=42
            )
            model.fit(features_train, target_train, verbose=False)
            threshold, metrics = self._find_optimal_threshold(model, features_eval, target_eval)
            return metrics
        except Exception as e:
            self.logger.error(f"Error in model evaluation: {str(e)}")
            raise


def tune_global_model():
    """Main function to tune the global CatBoost model with precision-focused features."""
    try:
        selected_columns = import_selected_features_ensemble('cat')
        # Initialize hypertuner with target metrics
        hypertuner = GlobalHypertuner(
            logger=logger,
            target_precision=0.50,
            target_recall=0.20,  # Updated to match new target
            precision_weight=0.8
        )
        
        # Load data with selected features
        X_train, y_train, X_test, y_test = import_training_data_ensemble()
        X_val, y_val = create_ensemble_evaluation_set()
        X_train = X_train[selected_columns]
        X_val = X_val[selected_columns]
        X_test = X_test[selected_columns]
        
        with mlflow.start_run(run_name=f"ensemble_catboost_tuning_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            mlflow.log_params({
                "target_precision": 0.50,
                "target_recall": 0.20,
                "precision_weight": 0.8,
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "test_samples": len(X_test),
                "draw_ratio_train": (y_train == 1).mean(),
                "draw_ratio_val": (y_val == 1).mean(),
                "draw_ratio_test": (y_test == 1).mean()
            })
            
            best_params = hypertuner.tune_model(
                X_train, y_train, X_val, y_val, X_test, y_test, n_trials=600
            )
            mlflow.log_params(best_params)
            
            val_metrics = hypertuner.evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test)
            test_metrics = hypertuner.evaluate_model(X_train, y_train, X_test, y_test, X_val, y_val)
            
            mlflow.log_metrics({
                f"val_{k}": v for k, v in val_metrics.items() 
                if isinstance(v, (int, float)) and k != 'best_params'
            })
            mlflow.log_metrics({
                f"test_{k}": v for k, v in test_metrics.items() 
                if isinstance(v, (int, float)) and k != 'best_params'
            })
            
            logger.info("\nModel Performance Summary:")
            logger.info("-" * 80)
            logger.info("Validation Set Metrics:")
            for k, v in val_metrics.items():
                if isinstance(v, (int, float)) and k != 'best_params':
                    logger.info(f"{k:20}: {v:.4f}")
            logger.info("\nTest Set Metrics:")
            for k, v in test_metrics.items():
                if isinstance(v, (int, float)) and k != 'best_params':
                    logger.info(f"{k:20}: {v:.4f}")
            logger.info("-" * 80)
            logger.info("Global model tuning completed successfully")
            
            return {
                'best_params': best_params,
                'validation_metrics': val_metrics,
                'test_metrics': test_metrics
            }
    except Exception as e:
        logger.error(f"Error during global model tuning: {str(e)}")
        raise


if __name__ == "__main__":
    results = tune_global_model()
    print("\nFinal Model Performance:")
    print("-" * 80)
    print("Validation Set:")
    print(f"Precision: {results['validation_metrics']['precision']:.4f}")
    print(f"Recall: {results['validation_metrics']['recall']:.4f}")
    print(f"F1 Score: {results['validation_metrics']['f1']:.4f}")
    print("\nTest Set:")
    print(f"Precision: {results['test_metrics']['precision']:.4f}")
    print(f"Recall: {results['test_metrics']['recall']:.4f}")
    print(f"F1 Score: {results['test_metrics']['f1']:.4f}") 