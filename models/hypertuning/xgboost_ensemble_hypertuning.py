"""
XGBoost model implementation for binary classification with global training capabilities.
Provides methods for model training, prediction, and analysis with configurable parameters.
"""

# Standard library imports
import os
import sys
from pathlib import Path
from tabnanny import verbose
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime

# Third-party imports
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import mlflow
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.over_sampling import ADASYN
from optuna.trial import FrozenTrial

# Szűrd a specifikus figyelmeztetést
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.base")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", category=UserWarning, module="optuna.samplers._tpe.sampler")

# Add project root to Python path
try:
    project_root = Path(__file__).parent.parent.parent
    if not project_root.exists():
        # Handle network path by using raw string
        project_root = Path(r"\\".join(str(project_root).split("\\")))
    sys.path.append(str(project_root))
    print(f"Project root xgboost_ensemble_hypertuning: {project_root}")
except Exception as e:
    print(f"Error setting project root path: {e}")
    # Fallback to current directory if path resolution fails
    sys.path.append(os.getcwd().parent.parent)
    print(f"Current directory xgboost_ensemble_hypertuning: {os.getcwd().parent.parent}")
os.environ['GIT_PYTHON_GIT_EXECUTABLE'] = "C:/Program Files/Git/bin/git.exe"
# Figyelmeztetések kikapcsolása
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow figyelmeztetések letiltása
warnings.filterwarnings('ignore', category=UserWarning, module='optuna.distributions')
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost.core')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, module='mlflow')

# Local imports
from utils.logger import ExperimentLogger
experiment_name = "xgboost_ensemble_hypertuning"
logger = ExperimentLogger(experiment_name=experiment_name, log_dir='./logs/xgboost_ensemble_hypertuning')


from utils.create_evaluation_set import create_ensemble_evaluation_set, import_selected_features_ensemble, import_training_data_ensemble, setup_mlflow_tracking
mlruns_dir = setup_mlflow_tracking(experiment_name)

HYPERPARAM_SPEC = {}
HYPERPARAM_SPEC['sampler'] = optuna.samplers.TPESampler(
    consider_prior=True,
    prior_weight=0.7,  # Csökkentve a korábbi sikeres tartományok hangsúlyozásához
    n_startup_trials=100,  # Növelt véletlen keresés a kezdeti fázisban
    multivariate=True,  # Multivariációs mintavételezés engedélyezése
    group=True,  # Paraméterek közötti kapcsolatok modellezése
    constant_liar=True
)
# Új custom célfüggvény súlyozással
HYPERPARAM_SPEC['direction'] = "maximize"


class GlobalHypertuner:
    """Global hyperparameter tuner for XGBoost model optimizing for validation metrics.
    
    Attributes:
        MIN_SAMPLES: Minimum number of samples required for training
        DEFAULT_THRESHOLD: Default prediction threshold
        PRECISION_THRESHOLD: Minimum required precision
        RECALL_CAP: Maximum recall to consider
        THRESHOLD_RANGE: Range for threshold optimization
        THRESHOLD_STEP: Step size for threshold optimization
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
                hyperparam_spec: Dict = HYPERPARAM_SPEC,
                random_seeds: List[int] = [13,19,20,27,38,54,69,74,75,76,84,130,138,229,230,257,319,275]):
        """Initialize the hypertuner with specified configuration.
        Args:
            logger: Optional logger instance

            target_precision: Target precision threshold
            target_recall: Target recall threshold
            precision_weight: Weight for precision in optimization
            hyperparam_spec: Hyperparameter specification dictionary
        """
        self.logger = logger or ExperimentLogger(experiment_name="xgboost_ensemble_hypertuning", log_dir='./logs/xgboost_ensemble_hypertuning')
        self.best_params: Dict[str, Any] = {}
        self.target_precision = target_precision
        self.target_recall = target_recall
        self.precision_weight = precision_weight
        self.best_metrics = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'threshold': 0.5}
        self.hyperparam_spec = hyperparam_spec
        self.random_seeds = random_seeds
        self._check_logger()

    def _check_logger(self) -> None:
        """Verify logger is properly initialized and functional.
        Raises:
            ValueError: If logger is not properly configured
        """
        if not self.logger:
            raise ValueError("Logger not initialized")
            
        if not hasattr(self.logger, 'info'):
            raise ValueError("Logger missing required 'info' method")
            
        if not hasattr(self.logger, 'error'):
            raise ValueError("Logger missing required 'error' method")
            
        # Test logger functionality and print configuration
        try:
            self.logger.info(f"Logger configuration - Experiment: {self.logger.experiment_name}, Log directory: {self.logger.log_dir}")
            self.logger.info("Logger check: Initialization successful")
        except Exception as e:
            raise ValueError(f"Logger test failed: {str(e)}")

    def _find_optimal_threshold(
        self,
        model: xgb.XGBClassifier,
        features_val: pd.DataFrame,
        target_val: pd.Series) -> Tuple[float, Dict[str, float]]:
        """Find optimal prediction threshold prioritizing precision while maintaining recall.
        Args:
            model: Trained XGBoost model
            features_val: Validation features
            target_val: Validation targets
        Returns:
            Tuple of (optimal threshold, metrics dictionary)
        """
        try:
            probas = model.predict_proba(features_val)[:, 1]
            best_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'threshold': 0.5}
            best_score = 0
            # Focus on higher thresholds for better precision, starting from 0.5
            for threshold in np.arange(0.5, 0.65, 0.01):
                preds = (probas >= threshold).astype(int)
                true_positives = ((preds == 1) & (target_val == 1)).sum()
                false_positives = ((preds == 1) & (target_val == 0)).sum()
                true_negatives = ((preds == 0) & (target_val == 0)).sum()
                false_negatives = ((preds == 0) & (target_val == 1)).sum()
                # Calculate metrics
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                # Only consider thresholds that meet minimum recall
                if recall >= 0.15:
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

    def objective(self, 
                trial: optuna.Trial, 
                features_train: pd.DataFrame, 
                target_train: pd.Series,
                features_val: pd.DataFrame,
                target_val: pd.Series,
                features_test: pd.DataFrame,
                target_test: pd.Series) -> float:
        """Optuna objective function that applies ADASYN to the base dataset in every trial."""
        try:
            self.logger.info(f"Starting trial {trial.number}...")
            param = {
                'objective': 'binary:logistic',
                'tree_method': 'hist',
                'device': 'cpu',
                'eval_metric': trial.suggest_categorical('eval_metric', [
                    ['error','auc', 'aucpr'],
                    ['aucpr', 'logloss'],
                    ['error', 'aucpr'], 
                    ['aucpr', 'auc'],
                    ['aucpr', 'error', 'auc']
                ]),
                'verbose': 0,
                'nthread': -1,
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.03),  # Best results in 0.02-0.06 range
                'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 100, 400),
                'min_child_weight': trial.suggest_int('min_child_weight', 150, 250),  # Wider range for stability
                'gamma': trial.suggest_float('gamma', 0.02, 0.12, step=0.01),  # Regularization sweet spot
                'subsample': trial.suggest_float('subsample', 0.3, 0.8, step=0.03),  # Lower values helped precision
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 0.95, step=0.01),  # Higher values improved recall
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 2.0, 2.5, step=0.02),  # Class imbalance adjustment
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0001, 0.01, step=0.0005),  # Lower bounds from best trials
                'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 4.0, step=0.1),  # Wider range for regularization
                'max_depth': trial.suggest_int('max_depth', 3, 6),  # Optimal for soccer prediction

                'n_estimators': trial.suggest_int('n_estimators', 1000, 3000),  # Let early stopping handle this
                'random_state': trial.suggest_categorical('random_state', self.random_seeds)
            }
            # --- Create and Train the XGBoost model on oversampled data ---
            model = xgb.XGBClassifier(**param)
            model.fit(features_train, target_train, eval_set=[(features_test, target_test)], verbose=False)
            # --- Find optimal threshold and evaluate model performance ---
            threshold, metrics = self._find_optimal_threshold(model, features_val, target_val)
            # Early prune if recall is below required minimum
            if metrics['recall'] < 0.15:
                self.logger.info(
                    f"Trial {trial.number} pruned: Recall {metrics['recall']:.4f} < 0.15"
                )
                raise optuna.exceptions.TrialPruned()
            # Log trial attributes for later inspection
            trial.set_user_attr('threshold', metrics['threshold'])
            trial.set_user_attr('precision', metrics['precision'])
            trial.set_user_attr('recall', metrics['recall'])
            trial.set_user_attr('f1', metrics['f1'])
            self.logger.info(
                f"Trial {trial.number} results: \n"
                f"Precision: {metrics['precision']:.4f}\n"
                f"Recall: {metrics['recall']:.4f}\n" 
                f"Threshold: {threshold:.4f}\n"
                f"F1: {metrics['f1']:.4f}\n"
                f"Draws Predicted: {metrics['draws_predicted']}\n"
                f"Correct: {metrics['correct']}\n"
            )
            return metrics['precision']
            
        except Exception as e:
            self.logger.error(f"Error in trial {trial.number}: {str(e)}")
            raise

    def _log_metrics_callback(self, study: optuna.Study, trial: FrozenTrial) -> None:
        """Log trial metrics using the ExperimentLogger."""
        self.logger.info(f"Trial {trial.number} completed with value: {trial.value}")
        self.logger.info(f"Trial params: {trial.params}")

    def tune_model(self,
                    features_train: pd.DataFrame,
                    target_train: pd.Series,
                    features_val: pd.DataFrame,
                    target_val: pd.Series,
                    features_test: pd.DataFrame,
                    target_test: pd.Series,
                    n_trials: int = 1000) -> Dict[str, Any]:
        """Tune global model with enhanced precision-recall tracking."""
        try:
            self.logger.info("Starting hyperparameter tuning...")
            study = optuna.create_study(
                sampler=self.hyperparam_spec['sampler'],
                direction=self.hyperparam_spec['direction']
            )
            self.logger.info(f"Created Optuna study with sampler: {self.hyperparam_spec['sampler']}")
            
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
                callbacks=[
                    self._log_metrics_callback,
                    self._pruned_trials_callback
                ]
            )
            # Log best trial results
            best_trial = study.best_trial
            # Log numeric metrics separately from params
            mlflow.log_metrics({
                "best_precision": float(best_trial.user_attrs["precision"]),
                "best_recall": float(best_trial.user_attrs["recall"]),
                "best_threshold": float(best_trial.user_attrs["threshold"]),
                "n_pruned_trials": float(len(study.get_trials(states=[optuna.trial.TrialState.PRUNED]))),
                "n_completed_trials": float(len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE])))
            })
            # Log params as a separate artifact
            mlflow.log_params(study.best_params)
            self.best_params = study.best_params
            self.logger.info(f"Hyperparameter tuning completed. Best trial: {study.best_trial.number}")
            return study.best_params
                
        except Exception as e:
            self.logger.error(f"Error during hyperparameter tuning: {str(e)}")
            raise

    def _pruned_trials_callback(self, study: optuna.Study, trial: FrozenTrial):
        """Callback to track pruned trials."""
        if trial.state == optuna.trial.TrialState.PRUNED:
            mlflow.log_metrics(
                {
                    "pruned_trials_count": len(study.get_trials(states=[optuna.trial.TrialState.PRUNED])),
                    f"trial_{trial.number}/pruned_recall": trial.user_attrs.get("recall", 0)
                },
                step=trial.number
            )

    def evaluate_model(self,
                    features_train: pd.DataFrame,
                    target_train: pd.Series,
                    features_val: pd.DataFrame,
                    target_val: pd.Series,
                    features_test: pd.DataFrame,
                    target_test: pd.Series) -> Dict[str, float]:
        """Evaluate model with tuned parameters and create a full model with all data.        
        Args:
            features_train: Training features
            target_train: Training labels
            features_test: Test features for evaluation
            target_test: Test labels for evaluation      
        Returns:
            Dict containing evaluation metrics and model parameters
        Raises:
            ValueError: If no tuned parameters are found
            Exception: For any other errors during evaluation
        """
        try:
            if not self.best_params:
                raise ValueError("No tuned parameters found. Run tune_global_model first.")
            params = self.best_params.copy()
            self.logger.info(f"Best params: {params}")
            # First fit - Training data only
            model = xgb.XGBClassifier(**params)
            model.fit(
                features_train,
                target_train,
                eval_set=[(features_test, target_test)],
                verbose=False
            )
            # Store model
            self.model = model  # Original model
            # --- Find optimal threshold and evaluate model performance ---
            threshold, metrics = self._find_optimal_threshold(model, features_val, target_val)
            self.logger.info("Model evaluation completed successfully with metrics:")
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.logger.info(f"{k}: {v:.4f}")
                else:
                    self.logger.info(f"{k}: {v}")
            return metrics
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {str(e)}")
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'draw_rate': 0.0,
                'predicted_rate': 0.0,
                'n_samples': len(target_val),
                'n_draws': 0,
                'n_predicted': 0,
                'n_correct': 0,
                'best_params': {}
            }

def tune_global_model():
    """Main function to tune the global model with precision-focused features."""
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
        selected_columns = import_selected_features_ensemble('all')
        X_train = X_train[selected_columns]
        X_val = X_val[selected_columns]
        X_test = X_test[selected_columns]
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"ensemble_xgboost_tuning_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            # Log data statistics
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
            
            # Tune model
            best_params = hypertuner.tune_model(
                X_train, y_train, X_val, y_val, X_test, y_test, n_trials=1000
            )
            
            # Log best parameters
            mlflow.log_params(best_params)
            
            # Evaluate on validation set
            val_metrics = hypertuner.evaluate_model(
                X_train, y_train, X_val, y_val, X_test, y_test
            )
            
            # Evaluate on test set
            test_metrics = hypertuner.evaluate_model(
                X_train, y_train, X_test, y_test, X_val, y_val
            )
            
            # Log validation metrics
            mlflow.log_metrics({
                f"val_{k}": v for k, v in val_metrics.items() 
                if isinstance(v, (int, float)) and k != 'best_params'
            })
            
            # Log test metrics
            mlflow.log_metrics({
                f"test_{k}": v for k, v in test_metrics.items() 
                if isinstance(v, (int, float)) and k != 'best_params'
            })
            
            # Log comprehensive performance summary
            print("\nModel Performance Summary:")
            print("-" * 80)
            print("Validation Set Metrics:")
            for k, v in val_metrics.items():
                if isinstance(v, (int, float)) and k != 'best_params':
                    print(f"{k:20}: {v:.4f}")
            
            print("\nTest Set Metrics:")
            for k, v in test_metrics.items():
                if isinstance(v, (int, float)) and k != 'best_params':
                    print(f"{k:20}: {v:.4f}")
            
            print("-" * 80)
            print("Global model tuning completed successfully")
            
            return {
                'best_params': best_params,
                'validation_metrics': val_metrics,
                'test_metrics': test_metrics
            }
            
    except Exception as e:
        print(f"Error during global model tuning: {str(e)}")
        raise

if __name__ == "__main__":
    results = tune_global_model()
    # Print final results
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