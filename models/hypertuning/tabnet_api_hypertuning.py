# Standard library imports
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Third-party imports
import numpy as np
import optuna
import pandas as pd
import mlflow
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import torch

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Local imports
from utils.logger import ExperimentLogger
from utils.create_evaluation_set import create_evaluation_sets_draws_api, get_selected_api_columns_draws, import_training_data_draws_api, setup_mlflow_tracking

selected_columns = get_selected_api_columns_draws()
experiment_name = "global_tabnet_hypertuning"
mlruns_dir = setup_mlflow_tracking(experiment_name)

class GlobalHypertuner:
    """Global hyperparameter tuner for TabNet model optimizing for validation metrics.

    Attributes:
        MIN_SAMPLES: Minimum number of samples required for training
        DEFAULT_THRESHOLD: Default prediction threshold
        PRECISION_THRESHOLD: Minimum required precision
        RECALL_CAP: Maximum recall to consider
    """

    MIN_SAMPLES: int = 1000
    DEFAULT_THRESHOLD: float = 0.55
    TARGET_PRECISION: float = 0.5
    TARGET_RECALL: float = 0.6
    PRECISION_WEIGHT: float = 0.7
    RECALL_CAP: float = 0.30

    def __init__(self, 
                 logger: Optional[ExperimentLogger] = None,
                 target_precision: float = TARGET_PRECISION,
                 target_recall: float = TARGET_RECALL,
                 precision_weight: float = PRECISION_WEIGHT):
        self.logger = logger or ExperimentLogger()
        self.best_params: Dict[str, Any] = {}
        self.target_precision = target_precision
        self.target_recall = target_recall
        self.precision_weight = precision_weight
        self.best_metrics = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'threshold': 0.5}

    def _find_optimal_threshold(
        self,
        model: TabNetClassifier,
        features_val: pd.DataFrame,
        target_val: pd.Series) -> Tuple[float, Dict[str, float]]:
        """Find optimal prediction threshold using validation data."""
        probas = model.predict_proba(features_val)[:, 1]
        best_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'threshold': 0.5}
        best_score = 0

        for threshold in np.arange(0.3, 0.65, 0.01):
            preds = (probas >= threshold).astype(int)
            precision = precision_score(target_val, preds, zero_division=0)
            recall = recall_score(target_val, preds, zero_division=0)

            if precision >= 0.30:  # Higher minimum precision
                score = precision * min(recall, 0.20)  # Lower recall cap
                if score > best_score:
                    best_score = score
                    best_metrics.update({
                        'precision': precision,
                        'recall': recall,
                        'f1': f1_score(target_val, preds, zero_division=0),
                        'threshold': threshold
                    })

        return best_metrics['threshold'], best_metrics

    def objective(
        self,
        trial: optuna.Trial,
        features_train: pd.DataFrame,
        target_train: pd.Series,
        features_val: pd.DataFrame,
        target_val: pd.Series,
        features_test: pd.DataFrame,
        target_test: pd.Series) -> float:
        """Optuna objective function optimizing for validation metrics."""
        try:
            # Check for NaN values and handle them
            if features_train.isnull().any().any() or features_val.isnull().any().any():
                self.logger.warning("NaN values found in features. Filling with column means.")
                features_train = features_train.fillna(features_train.mean())
                features_val = features_val.fillna(features_val.mean())

            # Check for empty data
            if len(features_train) == 0 or len(features_val) == 0:
                self.logger.error("Empty dataset provided.")
                return 0

            # Define hyperparameters to tune
            params = {
                'n_d': trial.suggest_int('n_d', 4, 16),
                'n_a': trial.suggest_int('n_a', 4, 16),
                'n_steps': trial.suggest_int('n_steps', 1, 5),
                'gamma': trial.suggest_float('gamma', 1.0, 2.0),
                'cat_idxs': [],
                'cat_dims': [],
                'cat_emb_dim': 1,
                'optimizer_fn': torch.optim.Adam,
                'optimizer_params': dict(lr=trial.suggest_float('lr', 1e-4, 1e-2, log=True)),
                'scheduler_params': {"step_size": 10, "gamma": 0.9},
                'scheduler_fn': torch.optim.lr_scheduler.StepLR,
                'mask_type': "sparsemax",
                'verbose': 0,
            }

            # Initialize TabNet model
            model = TabNetClassifier(**params)

            # Train the model
            self.logger.info("Training TabNet model...")
            model.fit(
                X_train=np.array(features_train[selected_columns]),
                y_train=np.array(target_train),
                eval_set=[(np.array(features_val[selected_columns]), np.array(target_val))],
                eval_metric=["auc", "logloss"],
                max_epochs=100,
                patience=10,
                batch_size=1024,
                virtual_batch_size=128,
            )
            self.logger.info("Model training completed successfully.")

            # Get predictions on validation set
            preds = model.predict(features_val[selected_columns])
            if np.sum(preds) == 0:  # No positive predictions
                self.logger.warning("No positive predictions made. Precision and recall will be 0.")
                precision = 0
                recall = 0
            else:
                precision = precision_score(target_val, preds, zero_division=0)
                recall = recall_score(target_val, preds, zero_division=0)
            f1 = f1_score(target_val, preds, zero_division=0)

            # Calculate weighted score with fallback for zero precision/recall
            precision_score_value = precision / self.target_precision if precision > 0 else 0
            recall_score_value = recall / self.target_recall if recall > 0 else 0
            weighted_score = (precision_score_value * self.precision_weight +
                             recall_score_value * (1 - self.precision_weight))
            
            self.logger.info(f"Weighted score: {weighted_score:.4f} Precision: {precision:.4f} Recall: {recall:.4f} F1: {f1:.4f}")
            # Ensure the weighted score is non-negative
            weighted_score = max(weighted_score, 0)

            # Update best metrics if this trial is better
            if (precision >= self.best_metrics['precision'] and
                recall >= self.best_metrics['recall']):
                self.best_metrics = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'threshold': 0.5  # Default threshold
                }
                # Set trial attributes for callback
                trial.set_user_attr('precision', precision)
                trial.set_user_attr('recall', recall)
                trial.set_user_attr('f1', f1)
                trial.set_user_attr('threshold', 0.5)
            return weighted_score

        except Exception as e:
            self.logger.error(f"Error in objective function: {str(e)}")
            return 0

    def tune_global_model(
        self,
        features_train: pd.DataFrame,
        target_train: pd.Series,
        features_val: pd.DataFrame,
        target_val: pd.Series,
        features_test: pd.DataFrame,
        target_test: pd.Series,
        n_trials: int = 100) -> Dict[str, Any]:
        """Tune hyperparameters optimizing for validation metrics."""
        if len(features_train) < self.MIN_SAMPLES:
            raise ValueError(f"Insufficient samples: {len(features_train)} < {self.MIN_SAMPLES}")

        self.logger.info("Starting global hyperparameter tuning")
        study = optuna.create_study(direction='maximize')

        def callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
            # Only log metrics if the trial was successful
            if study.best_trial == trial and 'precision' in trial.user_attrs:
                mlflow.log_metrics({
                    'best_val_precision': trial.user_attrs['precision'],
                    'best_val_recall': trial.user_attrs['recall'],
                    'best_val_f1': trial.user_attrs['f1'],
                    'best_threshold': trial.user_attrs['threshold']
                })

        study.optimize(
            lambda trial: self.objective(
                trial, features_train, target_train,
                features_val, target_val,
                features_test, target_test
            ),
            n_trials=n_trials,
            callbacks=[callback],
            catch=(Exception,)
        )

        best_trial = study.best_trial
        best_params = best_trial.params.copy()
        self.best_params = best_params

        self.logger.info(f"Best parameters: {best_params}")
        return best_params
    
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
            features_val: Validation features
            target_val: Validation labels
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
                'device': 'cpu',  # Force CPU usage
                'verbose': 0
            })
            
            # First fit - Training data only
            model = TabNetClassifier(**params)
            model.fit(
                features_train.values,
                target_train.values,
                eval_set=[(features_val.values, target_val.values)],
                eval_metric=['auc', 'accuracy']
            )
            
            # Second fit - Including test data
            full_features = pd.concat([features_train, features_test])
            full_target = pd.concat([target_train, target_test])
            model_full = TabNetClassifier(**params)
            model_full.fit(
                full_features.values,
                full_target.values,
                eval_set=[(features_val.values, target_val.values)],
                eval_metric=['auc', 'accuracy']
            )
            
            # Store both models
            self.model = model  # Original model
            self.model_full = model_full  # Model trained on all data
            
            # Get predictions from original model for metrics
            preds = model.predict_proba(features_val.values)[:, 1]
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
    logger = ExperimentLogger(experiment_name="global_tabnet_hypertuning", log_dir='./logs/tabnet_hypertuning')

    try:
        # Initialize hypertuner with target metrics
        hypertuner = GlobalHypertuner(
            logger=logger,
            target_precision=0.50,
            target_recall=0.60,
            precision_weight=0.7
        )

        # Load data
        features_train, target_train, features_test, target_test = import_training_data_draws_api()
        features_val, target_val = create_evaluation_sets_draws_api()

        with mlflow.start_run(run_name=f"api_tabnet_tuning"):
            hypertuner.tune_global_model(
                features_train,
                target_train,
                features_val,
                target_val,
                features_test,
                target_test,
                n_trials=100
            )
            # Log data statistics
            params_to_log = {
                "train_samples": len(features_train),
                "val_samples": len(features_val),
                "test_samples": len(features_test),
                "draw_ratio_train": (target_train == 1).mean(),
                "draw_ratio_val": (target_val == 1).mean(),
                "draw_ratio_test": (target_test == 1).mean()
            }
            mlflow.log_params(params_to_log)

            # Evaluate on validation set
            val_metrics = hypertuner.evaluate_tuned_model(
                features_train,
                target_train,
                features_val,
                target_val,
                features_test,
                target_test
            )

            # Evaluate on test set
            test_metrics = hypertuner.evaluate_tuned_model(
                features_train,
                target_train,
                features_test,
                target_test,
                features_val,
                target_val
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
            logger.info("Tuning complete.")
            return 
        
    except Exception as e:
        logger.error(f"Error during tuning: {str(e)}")
        raise


if __name__ == "__main__":
    tune_global_model()
