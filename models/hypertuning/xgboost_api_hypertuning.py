"""
XGBoost model implementation for binary classification with global training capabilities.
Provides methods for model training, prediction, and analysis with configurable parameters.
"""

# Standard library imports
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Third-party imports
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import mlflow
from sklearn.metrics import precision_score, recall_score, f1_score

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
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
        features_val: pd.DataFrame,  # Rename from X_val for consistency
        target_val: pd.Series  # Rename from y_val for consistency
        ) -> Tuple[float, Dict[str, float]]:
        """Find optimal prediction threshold using validation data."""
        probas = model.predict_proba(features_val)[:, 1]
        best_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'threshold': 0.5}
        best_score = 0
        
        # Focus on higher thresholds for better precision
        for threshold in np.arange(0.3, 0.65, 0.01):
            preds = (probas >= threshold).astype(int)
            precision = precision_score(target_val, preds, zero_division=0)
            recall = recall_score(target_val, preds, zero_division=0)
            
            # Modified scoring to prioritize precision
            if precision >= 0.35:  # Higher minimum precision
                score = precision * min(recall, 0.30)  # Lower recall cap
                if score > best_score:
                    best_score = score
                    best_metrics.update({
                        'precision': precision,
                        'recall': recall,
                        'f1': f1_score(target_val, preds, zero_division=0),
                        'threshold': threshold
                    })
                    # print(f"New best score: {best_score:.4f}")
                    
        # print(f"Best metrics: {best_metrics}")
        return best_metrics['threshold'], best_metrics
    
    def objective(self, 
                 trial: optuna.Trial, 
                 features_train: pd.DataFrame, 
                 target_train: pd.Series,
                 features_val: pd.DataFrame,
                 target_val: pd.Series,
                 features_test: pd.DataFrame,
                 target_test: pd.Series) -> float:
        """Optuna objective function optimizing for validation metrics."""
        try:
            draw_rate = target_train.mean()
            safe_draw_rate = max(draw_rate, 0.001)
            
            # Initialize parameters with CPU-specific settings
            param = {
                'objective': 'binary:logistic',
                'tree_method': self.hyperparam_spec['tree_method'],
                'device': self.hyperparam_spec['device'],
                'early_stopping_rounds': 500,
                'eval_metric': ['error', 'auc', 'aucpr'],
                'verbosity': 0,  # Reduce verbosity
                'nthread': -1    # Use all CPU cores
            }
            
            # Add hyperparameters based on ranges specification
            for param_name, param_range in self.hyperparam_spec['ranges'].items():
                if len(param_range) == 3 and param_range[2] == 'log':
                    param[param_name] = trial.suggest_float(
                        param_name, 
                        param_range[0], 
                        param_range[1], 
                        log=True
                    )
                elif isinstance(param_range[0], int):
                    param[param_name] = trial.suggest_int(
                        param_name,
                        param_range[0],
                        param_range[1]
                    )
                else:
                    param[param_name] = trial.suggest_float(
                        param_name,
                        param_range[0],
                        param_range[1]
                    )

            # Create and train model
            model = xgb.XGBClassifier(**param)
            
            # Add early stopping callback
            early_stopping_callback = xgb.callback.EarlyStopping(
                rounds=param['early_stopping_rounds'],
                metric_name='aucpr',
                save_best=True
            )
            
            # Fit model with callbacks
            model.fit(
                features_train, 
                target_train,
                eval_set=[(features_val, target_val)],
                verbose=False
            )
            
            # Find optimal threshold
            threshold, metrics = self._find_optimal_threshold(
                model, features_val, target_val
            )
            
            # Calculate weighted score based on precision and recall targets
            precision_score = metrics['precision'] / self.target_precision
            recall_score = metrics['recall'] / self.target_recall
            
            weighted_score = (
                precision_score * self.precision_weight + 
                recall_score * (1 - self.precision_weight)
            )
            
            # Apply penalties for not meeting minimum requirements
            if metrics['precision'] < 0.35:
                weighted_score *= 0.5
            if metrics['recall'] < 0.20:
                weighted_score *= 0.5
                
            # Log metrics
            self.logger.info(f"\nTrial {trial.number} Results:")
            self.logger.info(f"Parameters: {param}")
            self.logger.info(f"Precision: {metrics['precision']:.4f}")
            self.logger.info(f"Recall: {metrics['recall']:.4f}")
            self.logger.info(f"F1: {metrics['f1']:.4f}")
            self.logger.info(f"Threshold: {threshold:.4f}")
            self.logger.info(f"Weighted Score: {weighted_score:.4f}")
            
            # Store trial attributes
            trial.set_user_attr('threshold', threshold)
            trial.set_user_attr('precision', metrics['precision'])
            trial.set_user_attr('recall', metrics['recall'])
            trial.set_user_attr('f1', metrics['f1'])
            
            return weighted_score
            
        except Exception as e:
            self.logger.error(f"Error in trial {trial.number}: {str(e)}")
            raise

    def _log_mlflow_metrics(self, metrics: Dict[str, float], prefix: str) -> None:
        """Log metrics to MLflow with proper prefix."""
        mlflow.log_metrics({
            f"{prefix}_{k}": v for k, v in metrics.items() 
            if isinstance(v, (int, float)) and k != 'best_params'
        })
    
    def tune_global_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        n_trials: int = 100) -> Dict[str, Any]:
        """Tune hyperparameters optimizing for validation metrics."""
        if len(X_train) < self.MIN_SAMPLES:
            raise ValueError(f"Insufficient samples: {len(X_train)} < {self.MIN_SAMPLES}")
        
        self.logger.info("Starting global hyperparameter tuning")
        self.logger.info(f"Target metrics - Precision: {self.target_precision:.2f}, "
                        f"Recall: {self.target_recall:.2f}")
        
        # Create study with specified sampler from hyperparam_spec
        study = optuna.create_study(
            direction='maximize',
            sampler=self.hyperparam_spec['sampler'],
        )
        
        def callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
            if study.best_trial == trial:
                self._log_mlflow_metrics({
                    'best_val_precision': trial.user_attrs['precision'],
                    'best_val_recall': trial.user_attrs['recall'],
                    'best_val_f1': trial.user_attrs['f1'],
                    'best_threshold': trial.user_attrs['threshold']
                }, "val")
        
        try:
           
            study.optimize(
                lambda trial: self.objective(
                    trial, X_train, y_train, 
                    X_val, y_val, 
                    X_test, y_test
                ),
                n_trials=n_trials,
                callbacks=[callback],
                catch=(Exception,)
            )
            
            best_trial = study.best_trial
            best_params = best_trial.params.copy()
            self.best_params = best_params
            
            self.logger.info(f"Best parameters: {best_params}")
            self.logger.info(f"Best validation metrics:")
            self.logger.info(f"Precision: {best_trial.user_attrs['precision']:.4f}")
            self.logger.info(f"Recall: {best_trial.user_attrs['recall']:.4f}")
            self.logger.info(f"F1: {best_trial.user_attrs['f1']:.4f}")
            self.logger.info(f"Threshold: {best_trial.user_attrs['threshold']:.4f}")
            
            return best_params
                
        except optuna.exceptions.TrialPruned as e:
            self.logger.warning(f"Trial was pruned: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error during optimization: {str(e)}")
            raise

    def evaluate_tuned_model(self,
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
            
            # Second fit - Including test data
            full_features = pd.concat([features_train, features_test])
            full_target = pd.concat([target_train, target_test])
            
            model_full = xgb.XGBClassifier(**params)
            model_full.fit(
                full_features,
                full_target,
                eval_set=[(features_val, target_val)],
                verbose=False
            )
            
            # Store both models
            self.model = model  # Original model
            self.model_full = model_full  # Model trained on all data
            
            # Get predictions from original model for metrics
            preds = model.predict_proba(features_val)[:, 1]
            y_pred = (preds >= 0.53).astype(int)
            
            metrics = {
                'precision': precision_score(target_val, y_pred, zero_division=0),
                'recall': recall_score(target_val, y_pred, zero_division=0),
                'f1': f1_score(target_val, y_pred, zero_division=0),
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
    """Main function to tune the global model."""
    logger = ExperimentLogger(experiment_name="global_xgboost_hypertuning", log_dir='./logs/xgboost_hypertuning')
    
    try:
        # Initialize hypertuner with target metrics
        hypertuner = GlobalHypertuner(
            logger=logger,
            target_precision=0.50,  # Target precision threshold
            target_recall=0.60,     # Target recall threshold
            precision_weight=0.8     # Weight for precision vs recall balance
        )
        
        X_train, y_train, X_test, y_test = import_training_data_draws_api()
        X_val, y_val = create_evaluation_sets_draws_api()
        print(X_train.shape)
        print(X_val.shape)
        print(X_test.shape)
        
        with mlflow.start_run(run_name=f"api_xgboost_tuning"):
            # Log target metrics and data statistics
            params_to_log = {
                "target_precision": 0.50,
                "target_recall": 0.60,
                "precision_weight": 0.6,
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "test_samples": len(X_test),
                "draw_ratio_train": (y_train == 1).mean(),
                "draw_ratio_val": (y_val == 1).mean(),
                "draw_ratio_test": (y_test == 1).mean()
            }
            
            mlflow.log_params(params_to_log)
            
            for param, value in params_to_log.items():
                print(f"{param}: {value}")
            
            # Tune model using training and validation sets
            best_params = hypertuner.tune_global_model(
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                n_trials=300
            )
            
            mlflow.log_params(best_params)
            
            # Evaluate on validation set
            val_metrics = hypertuner.evaluate_tuned_model(
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test
            )
            
            # Evaluate on test set
            test_metrics = hypertuner.evaluate_tuned_model(
                X_train,
                y_train,
                X_test,
                y_test,
                X_val,
                y_val
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
            
            # Print comprehensive performance summary
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