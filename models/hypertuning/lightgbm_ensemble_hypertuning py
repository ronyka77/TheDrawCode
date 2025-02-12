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
import lightgbm as lgb
import mlflow
import warnings
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.over_sampling import ADASYN
from optuna.trial import FrozenTrial
from sklearn.calibration import CalibratedClassifierCV

# Szűrd a specifikus figyelmeztetést
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.base")
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
warnings.filterwarnings("ignore", category=UserWarning, module="optuna.samplers._tpe.sampler")
# A specifikus figyelmeztetés szűrése a CalibratedClassifierCV cv='prefit' paraméteréhez
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.calibration", message="The `cv='prefit'` option is deprecated.*")


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
experiment_name = "ensemble_lightgbm_hypertuning"
logger = ExperimentLogger(experiment_name=experiment_name, log_dir='./logs/lightgbm_ensemble_hypertuning')

from utils.create_evaluation_set import create_ensemble_evaluation_set, import_selected_features_ensemble, import_training_data_ensemble, setup_mlflow_tracking
mlruns_dir = setup_mlflow_tracking(experiment_name)

# Core Configuration for CPU-only training with LightGBM
HYPERPARAM_SPEC = {}
HYPERPARAM_SPEC['sampler'] = optuna.samplers.TPESampler(
    consider_prior=True,
    prior_weight=0.8,  # Csökkentve a korábbi sikeres tartományok hangsúlyozásához
    n_startup_trials=100,  # Növelt véletlen keresés a kezdeti fázisban
    multivariate=True,  # Multivariációs mintavételezés engedélyezése
    group=True,  # Paraméterek közötti kapcsolatok modellezése
    constant_liar=True
)

class GlobalHypertuner:
    """
    Global hyperparameter tuner for LightGBM model optimizing for validation metrics.
    
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
        self.logger = logger or ExperimentLogger(experiment_name="ensemble_lightgbm_hypertuning", log_dir='./logs/lightgbm_hypertuning')
        self.target_precision = target_precision
        self.target_recall = target_recall
        self.precision_weight = precision_weight
        self.hyperparam_spec = hyperparam_spec
        self.best_params = {}
        self.random_seeds = [13,19,20,27,38,54,69,74,75,76,84,130,138,229,230,257,319,275]

    def _find_optimal_threshold(
        self,
        model: lgb.LGBMClassifier,
        features_val: pd.DataFrame,
        target_val: pd.Series) -> Tuple[float, Dict[str, float]]:
        """Find optimal prediction threshold prioritizing precision while maintaining recall.
        Args:
            model: Trained LightGBM model
            features_val: Validation features
            target_val: Validation targets
        Returns:
            Tuple of (optimal threshold, metrics dictionary)
        """
        try:
            probas = model.predict_proba(features_val)[:, 1]
            avg_preds = np.mean(probas)
            # self.logger.info(f"Average predicted probability: {avg_preds:.4f}")
            # mlflow.log_metric("avg_predicted_probability", float(avg_preds))
            best_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'threshold': 0.4}
            best_score = 0
            # Focus on higher thresholds for better precision, starting from 0.2
            for threshold in np.arange(0.35, 0.65, 0.01):
                # self.logger.info(f"Threshold: {threshold}")
                preds = (probas >= threshold).astype(int)
                true_positives = ((preds == 1) & (target_val == 1)).sum()
                false_positives = ((preds == 1) & (target_val == 0)).sum()
                true_negatives = ((preds == 0) & (target_val == 0)).sum()
                false_negatives = ((preds == 0) & (target_val == 1)).sum()
                # Calculate metrics
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                # Only consider thresholds that meet minimum recall
                if recall >= 0.15:
                    # self.logger.info(f"Recall: {recall}")
                    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                    f1 = f1_score(target_val, preds)
                    # Modified scoring to prioritize precision
                    score = precision
                    if score > best_score:
                        best_score = score
                        best_metrics.update({
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'threshold': threshold,
                            'correct': true_positives,
                            'draws_predicted': false_positives + true_positives
                        })
            if best_metrics['recall'] < 0.15:
                self.logger.warning(
                    f"Could not find threshold meeting recall requirement. "
                    f"Best recall: {best_metrics['recall']:.4f}"
                    f"Best precision: {best_metrics['precision']:.4f}"
                )
            self.logger.info(
                f"New best threshold {best_metrics['threshold']:.3f}: "
                f"Precision={best_metrics['precision']:.4f}, Recall={best_metrics['recall']:.4f}"
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
        Optuna objective function that optimizes CatBoost precision while hypertuning ADASYN parameters.
        """
        try:
            # --- Tune CatBoost parameters using trial suggestions ---
            lgbm_params = {
                'objective': 'binary',
                'metric': ['binary_logloss', 'auc', 'aucpr'],
                'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'goss', 'dart']),
                'verbose': -1,
                'device_type': 'cpu',
                'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.01),
                'max_depth': trial.suggest_int('max_depth', 5, 10),
                'num_leaves': trial.suggest_int('num_leaves', 50, 100),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.3, 0.5),
                'n_estimators': trial.suggest_int('n_estimators', 2000, 10000),
                'subsample': trial.suggest_float('subsample', 0.75, 0.95),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 0.95),
                'min_child_samples': trial.suggest_int('min_child_samples', 20, 40),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.8, 0.95),
                'random_state': trial.suggest_categorical('random_state', self.random_seeds),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 2.0, 2.3),
                'min_child_weight': trial.suggest_int('min_child_weight', 20, 40),
                'feature_fraction_bynode': trial.suggest_float('feature_fraction_bynode', 0.99, 1.0)
            }
            
            # Handle boosting type specific parameters
            boosting_type = lgbm_params['boosting_type']
            if boosting_type == 'goss':
                # GOSS cannot use bagging, so remove related parameters
                lgbm_params.pop('subsample', None)
                lgbm_params.pop('bagging_freq', None)
                lgbm_params['top_rate'] = trial.suggest_float('top_rate', 0.1, 0.3)
                lgbm_params['other_rate'] = trial.suggest_float('other_rate', 0.1, 0.3)
            else:
                # Add bagging parameters for non-GOSS types
                lgbm_params['bagging_freq'] = trial.suggest_int('bagging_freq', 1, 3)
                if boosting_type == 'dart':
                    # DART specific parameters
                    lgbm_params['drop_rate'] = trial.suggest_float('drop_rate', 0.1, 0.3)
                    lgbm_params['max_drop'] = trial.suggest_int('max_drop', 10, 50)
            # --- Create and Train the LightGBM model on oversampled data ---
            model = lgb.LGBMClassifier(**lgbm_params)
            model.fit(
                features_train, target_train,
                eval_set=(features_test, target_test)
            )
            # Továbbfejlesztett verzió
            calibrated_model = CalibratedClassifierCV(
                estimator=FrozenEstimator(model),
                method=trial.suggest_categorical('calibration_method', ['sigmoid', 'isotonic']),
                ensemble='auto',  # Automatikus ensemble legjobb kalibrátorokkal
                n_jobs=-1
            ).fit(
                features_train,  # Külön validation set kalibráláshoz
                target_train, eval_set=(features_test, target_test)
            )
            # --- Find optimal threshold and evaluate model performance ---
            threshold, metrics = self._find_optimal_threshold(calibrated_model, features_val, target_val)
            # Early prune if recall requirement is not met
            if metrics['recall'] < self.target_recall:
                self.logger.info(f"Trial {trial.number} pruned: Recall {metrics['recall']:.4f} < {self.target_recall}")
                raise optuna.exceptions.TrialPruned()
            
            trial.set_user_attr('threshold', threshold)
            trial.set_user_attr('precision', metrics['precision'])
            trial.set_user_attr('recall', metrics['recall'])
            trial.set_user_attr('f1', metrics['f1'])
            
            self.logger.info(
                f"Trial {trial.number} results: \n"
                f"Precision={metrics['precision']:.4f}, \n"
                f"Recall={metrics['recall']:.4f}, \n"
                f"Threshold={threshold:.4f}"
            )
            
            return metrics['precision']
        except Exception as e:
            self.logger.error(f"Error in trial {trial.number}: {str(e)}")
            raise

    def _log_metrics_callback(self, study: optuna.Study, trial: FrozenTrial) -> None:
        """Callback to log metrics for each trial."""
        self.logger.info(
            f"Trial {trial.number}: precision={trial.user_attrs.get('precision')}, "
            f"recall={trial.user_attrs.get('recall')}, "
            f"threshold={trial.user_attrs.get('threshold')}"
        )
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
        n_trials: int = 1000) -> Dict[str, Any]:
        """Tune global model with enhanced precision-recall tracking."""
        try:
            # Then focus on LightGBM with more trials
            study = optuna.create_study(
                direction="maximize",
                sampler=self.hyperparam_spec['sampler']
            )
            
            # Run optimization with enhanced logging
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
            
            # Log final metrics and parameters
            best_trial = study.best_trial
            with mlflow.start_run(nested=True):
                mlflow.log_params({
                    "n_trials": n_trials,
                    "precision_weight": self.precision_weight,
                    "final_threshold": best_trial.user_attrs["threshold"]
                })
                mlflow.log_metrics({
                    "best_precision": best_trial.user_attrs["precision"],
                    "best_recall": best_trial.user_attrs["recall"],
                    "pruned_trials": len(study.get_trials(states=[optuna.trial.TrialState.PRUNED])),
                    "completed_trials": len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))
                })
                
            self.best_params = study.best_params
            self.logger.info(f"Optimization completed. Best params: {study.best_params}")
            return study.best_params
            
        except Exception as e:
            self.logger.error(f"Hyperparameter tuning failed: {str(e)}", exc_info=True)
            self.logger.log_to_mlflow({
                "error": str(e),
                "stage": "hyperparameter_tuning",
                "severity": "critical"
            })
            raise

    def evaluate_model(
        self,
        features_train: pd.DataFrame,
        target_train: pd.Series,
        features_test: pd.DataFrame,
        target_test: pd.Series,
        features_eval: pd.DataFrame,
        target_eval: pd.Series) -> Dict[str, float]:
        """Evaluate the model performance on a given evaluation set."""
        try:
            if not self.best_params:
                raise ValueError("No tuned parameters found. Run tune_global_model first.")
            params = self.best_params.copy()
            self.logger.info(f"Best params: {params}")

            model = lgb.LGBMClassifier(**params)
            model.fit(features_train, target_train, eval_set=(features_test, target_test))
            threshold, metrics = self._find_optimal_threshold(model, features_eval, target_eval)
            return metrics

        except Exception as e:
            self.logger.error(f"Error in model evaluation: {str(e)}")
            raise


def tune_global_model():
    """Main function to tune the global CatBoost model with precision-focused features."""
    selected_columns = import_selected_features_ensemble('all')
    try:
        # Initialize hypertuner with target metrics
        hypertuner = GlobalHypertuner(
            logger=logger,
            target_precision=0.50,
            target_recall=0.15,  # Updated to match new target
            precision_weight=0.8
        )
        
        # Load data with selected features
        X_train, y_train, X_test, y_test = import_training_data_ensemble()
        X_val, y_val = create_ensemble_evaluation_set()
        X_train = X_train[selected_columns]
        X_val = X_val[selected_columns]
        X_test = X_test[selected_columns]
        
        with mlflow.start_run(run_name=f"ensemble_lightgbm_tuning_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            mlflow.log_params({
                "target_precision": 0.50,
                "target_recall": 0.15,
                "precision_weight": 0.8,
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "test_samples": len(X_test),
                "draw_ratio_train": (y_train == 1).mean(),
                "draw_ratio_val": (y_val == 1).mean(),
                "draw_ratio_test": (y_test == 1).mean()
            })
            
            best_params = hypertuner.tune_model(
                X_train, y_train, X_val, y_val, X_test, y_test, n_trials=3000
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