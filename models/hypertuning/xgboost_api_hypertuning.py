"""
XGBoost model implementation for binary classification with global training capabilities.
Provides methods for model training, prediction, and analysis with configurable parameters.
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
import xgboost as xgb
import mlflow
from sklearn.metrics import precision_score, recall_score, f1_score
from optuna.trial import FrozenTrial

# Add project root to Python path
# Add project root to Python path
try:
    project_root = Path(__file__).parent.parent.parent
    if not project_root.exists():
        # Handle network path by using raw string
        project_root = Path(r"\\".join(str(project_root).split("\\")))
    sys.path.append(str(project_root))
    print(f"Project root xgboost_model: {project_root}")
except Exception as e:
    print(f"Error setting project root path: {e}")
    # Fallback to current directory if path resolution fails
    sys.path.append(os.getcwd().parent.parent)
    print(f"Current directory xgboost_model: {os.getcwd().parent.parent}")
os.environ['GIT_PYTHON_GIT_EXECUTABLE'] = "C:/Program Files/Git/bin/git.exe"

# Local imports
from utils.logger import ExperimentLogger
from utils.create_evaluation_set import create_evaluation_sets_draws_api, get_selected_api_columns_draws, import_training_data_draws_api, setup_mlflow_tracking

# Core Configuration for CPU-Only Training
HYPERPARAM_SPEC = {
    'tree_method': 'hist',
    'device': 'cpu',
    'sampler': optuna.samplers.TPESampler(
        consider_prior=True,
        prior_weight=1.0,
        n_startup_trials=10,
        n_ei_candidates=24,
        seed=42
    ),
    'ranges': {
        'learning_rate': (0.0001, 0.1, 'log'),
        'min_child_weight': (10, 500),
        'gamma': (1.0, 30.0),
        'subsample': (0.2, 1.0),
        'colsample_bytree': (0.2, 1.0),
        'scale_pos_weight': (0.1, 20.0),
        'reg_alpha': (0.01, 20.0, 'log'),
        'reg_lambda': (0.01, 30.0, 'log'),
        'n_estimators': (3000, 30000),
        'max_depth': (3, 12)  # Added max_depth parameter
    }
}

selected_columns = get_selected_api_columns_draws()
experiment_name = "global_xgboost_hypertuning"
mlruns_dir = setup_mlflow_tracking(experiment_name)

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
    DEFAULT_THRESHOLD: float = 0.53
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
        """Initialize the hypertuner with specified configuration.
        
        Args:
            logger: Optional logger instance
            target_precision: Target precision threshold
            target_recall: Target recall threshold
            precision_weight: Weight for precision in optimization
            hyperparam_spec: Hyperparameter specification dictionary
        """
        self.logger = logger or ExperimentLogger()
        self.best_params: Dict[str, Any] = {}
        self.target_precision = target_precision
        self.target_recall = target_recall
        self.precision_weight = precision_weight
        self.best_metrics = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'threshold': 0.5}
        self.hyperparam_spec = hyperparam_spec
    
    def _find_optimal_threshold(
        self,
        model: xgb.XGBClassifier,
        features_val: pd.DataFrame,
        target_val: pd.Series
    ) -> Tuple[float, Dict[str, float]]:
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
                
                # Calculate metrics
                recall = recall_score(target_val, preds)
                
                # Only consider thresholds that meet minimum recall
                if recall >= 0.20:
                    true_positives = ((preds == 1) & (target_val == 1)).sum()
                    false_positives = ((preds == 1) & (target_val == 0)).sum()
                    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                    f1 = f1_score(target_val, preds)
                    
                    # Modified scoring to prioritize precision
                    score = precision * min(1.0, (recall - 0.20) / 0.20)
                    
                    if score > best_score:
                        best_score = score
                        best_metrics.update({
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'threshold': threshold
                        })
                        
                        # Log improvement
                        self.logger.info(
                            f"New best threshold {threshold:.3f}: "
                            f"Precision={precision:.4f}, Recall={recall:.4f}"
                        )
                        
                        # Log to MLflow
                        mlflow.log_metrics({
                            'best_threshold': threshold,
                            'best_precision': precision,
                            'best_recall': recall,
                            'best_f1': f1
                        })
            
            if best_metrics['recall'] < 0.20:
                self.logger.warning(
                    f"Could not find threshold meeting recall requirement. "
                    f"Best recall: {best_metrics['recall']:.4f}"
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
        """Optuna objective function optimizing for precision while maintaining minimum recall."""
        try:
            # Initialize parameters with CPU-specific settings
            param = {
                'objective': 'binary:logistic',
                'tree_method': self.hyperparam_spec['tree_method'],
                'device': self.hyperparam_spec['device'],
                'early_stopping_rounds': 500,
                'eval_metric': ['error', 'auc', 'aucpr'],
                'verbosity': 0,
                'nthread': -1
            }
            
            # Expand hyperparameter ranges for better precision optimization
            param.update({
                'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.01, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 500),
                'gamma': trial.suggest_float('gamma', 1e-2, 50, log=True),
                'subsample': trial.suggest_float('subsample', 0.3, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 2.0, 4.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'n_estimators': trial.suggest_int('n_estimators', 1000, 30000)
            })

            # Create and train model
            model = xgb.XGBClassifier(**param)
            model.fit(
                features_train, 
                target_train,
                eval_set=[(features_test, target_test)],
                verbose=False
            )
            
            # Find optimal threshold and get metrics
            threshold, metrics = self._find_optimal_threshold(
                model, features_val, target_val
            )
            
            # Early pruning if recall requirement not met
            if metrics['recall'] < 0.20:
                self.logger.info(
                    f"Trial {trial.number} pruned: "
                    f"Recall {metrics['recall']:.4f} < 0.20"
                )
                raise optuna.exceptions.TrialPruned()
            
            # Log metrics
            self.logger.info(f"\nTrial {trial.number} Results:")
            self.logger.info(f"Parameters: {param}")
            self.logger.info(f"Precision: {metrics['precision']:.4f}")
            self.logger.info(f"Recall: {metrics['recall']:.4f}")
            self.logger.info(f"Threshold: {metrics['threshold']:.4f}")
            
            # Store trial attributes
            trial.set_user_attr('threshold', metrics['threshold'])
            trial.set_user_attr('precision', metrics['precision'])
            trial.set_user_attr('recall', metrics['recall'])
            trial.set_user_attr('f1', metrics['f1'])
            
            # Return precision as optimization target
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
                         n_trials: int = 100) -> Dict[str, Any]:
        """Tune global model with enhanced precision-recall tracking."""
        try:
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
                callbacks=[
                    self._log_metrics_callback,  # New callback for detailed metric logging
                    self._pruned_trials_callback  # New callback to track pruned trials
                ]
            )
            
            # Log best trial results
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
            params.update({
                'objective': 'binary:logistic',
                'early_stopping_rounds': 500,
                'eval_metric': ['error', 'auc', 'aucpr'],
                'tree_method': 'hist'
            })
            
            # First fit - Training data only
            model = xgb.XGBClassifier(**params)
            model.fit(
                features_train,
                target_train,
                eval_set=[(features_test, target_test)],
                verbose=False
            )
            
            # # Second fit - Including test data
            # full_features = pd.concat([features_train, features_test])
            # full_target = pd.concat([target_train, target_test])
            
            # model_full = xgb.XGBClassifier(**params)
            # model_full.fit(
            #     full_features,
            #     full_target,
            #     eval_set=[(features_val, target_val)],
            #     verbose=False
            # )
            
            # Store both models
            self.model = model  # Original model
            # self.model_full = model_full  # Model trained on all data
            
            # Get predictions from original model for metrics
            y_pred = model.predict(features_val)
            true_positives = ((y_pred == 1) & (target_val == 1)).sum()
            false_positives = ((y_pred == 1) & (target_val == 0)).sum()
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = recall_score(target_val, y_pred, zero_division=0)
            f1 = f1_score(target_val, y_pred, zero_division=0)
            
            metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'draw_rate': float(target_val.mean()),
                'predicted_rate': float(y_pred.mean()),
                'n_samples': len(target_val),
                'n_draws': int(target_val.sum()),
                'n_predicted': int(y_pred.sum()),
                'n_correct': int(np.logical_and(target_val, y_pred).sum()),
                'best_params': params
            }
            
            self.logger.info("Model evaluation completed successfully")
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
    logger = ExperimentLogger(experiment_name="global_xgboost_hypertuning", log_dir='./logs/xgboost_hypertuning')
    
    try:
        # Initialize hypertuner with target metrics
        hypertuner = GlobalHypertuner(
            logger=logger,
            target_precision=0.50,
            target_recall=0.20,  # Updated to match new target
            precision_weight=0.8
        )
        
        # Load data with selected features
        X_train, y_train, X_test, y_test = import_training_data_draws_api()
        X_val, y_val = create_evaluation_sets_draws_api()
        
  
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"api_xgboost_tuning_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            # Log data statistics
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
            
            # Tune model
            best_params = hypertuner.tune_model(
                X_train, y_train, X_val, y_val, X_test, y_test, n_trials=300
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