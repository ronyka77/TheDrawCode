import optuna
from catboost import CatBoostClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import mlflow
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from utils.logger import ExperimentLogger
from utils.create_evaluation_set import create_evaluation_sets, get_selected_columns, import_training_data_draws, setup_mlflow_tracking

selected_columns = get_selected_columns()

class GlobalCatBoostHypertuner:
    """Global hyperparameter tuner for CatBoost model optimizing for validation metrics."""
    
    def __init__(self, 
                 logger: Optional[ExperimentLogger] = None,
                 target_precision: float = 0.50,  # Target precision threshold
                 target_recall: float = 0.60,     # Target recall threshold
                 precision_weight: float = 0.8     # Weight for precision vs recall
                ):
        self.logger = logger or ExperimentLogger()
        self.best_params: Dict[str, Any] = {}
        self.MIN_SAMPLES = 1000
        self.target_precision = target_precision
        self.target_recall = target_recall
        self.precision_weight = precision_weight
        self.best_metrics = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'threshold': 0.5}
    
    def _find_optimal_threshold(self, 
                              model: CatBoostClassifier, 
                              X_val: pd.DataFrame, 
                              y_val: pd.Series) -> Tuple[float, Dict[str, float]]:
        """Find optimal prediction threshold using validation data."""
        probas = model.predict_proba(X_val)[:, 1]
        best_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'threshold': 0.5}
        best_score = 0
        
        # Focus on higher thresholds for better precision
        for threshold in np.arange(0.3, 0.7, 0.01):
            preds = (probas >= threshold).astype(int)
            precision = precision_score(y_val, preds, zero_division=0)
            recall = recall_score(y_val, preds, zero_division=0)
            
            # Modified scoring to prioritize precision
            if precision >= 0.35:  # Higher minimum precision
                score = precision * min(recall, 0.30)  # Lower recall cap
                if score > best_score:
                    best_score = score
                    best_metrics.update({
                        'precision': precision,
                        'recall': recall,
                        'f1': f1_score(y_val, preds, zero_division=0),
                        'threshold': threshold
                    })
        
        return best_metrics['threshold'], best_metrics
    
    def objective(self, 
                 trial: optuna.Trial, 
                 X_train: pd.DataFrame, 
                 y_train: pd.Series,
                 X_val: pd.DataFrame,
                 y_val: pd.Series) -> float:
        """Optuna objective function optimizing for validation metrics."""
        try:
            draw_rate = y_train.mean()
            safe_draw_rate = max(draw_rate, 0.001)
            
            param = {
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.01, log=True),
                'depth': 7,
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 40, 150),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 35.0, log=True),
                'random_strength': trial.suggest_float('random_strength', 10.0, 40.0),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.3, 1.0),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 2.0, 15.0) * (0.26/safe_draw_rate),
                'n_estimators': trial.suggest_int('n_estimators', 8000, 20000),
                'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 5, 30),
                'bootstrap_type': 'Bayesian',
                'grow_policy': 'SymmetricTree',
                'eval_metric': 'AUC',
                'early_stopping_rounds': 100,
                'verbose': False,
                'task_type': 'CPU'  # Change to 'GPU' if available
            }
            model = CatBoostClassifier(**param)
            model.fit(
                X_train, 
                y_train,
                eval_set=[(X_train, y_train)],
                verbose=False
            )
            
            threshold = 0.5
            probas = model.predict_proba(X_val)[:, 1]
            y_pred = (probas >= threshold).astype(int)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            
            metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'threshold': threshold
            }
            
            # Debug logging
            print(f"\nTrial {trial.number} metrics:")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1: {metrics['f1']:.4f}")
            print(f"Threshold: {threshold:.4f}")
            
            # Calculate normalized scores
            precision_score_value = metrics['precision'] / self.target_precision
            recall_score_value = metrics['recall'] / self.target_recall
            
            # Prioritize precision more heavily
            precision_weight = 0.8
            recall_weight = 0.2
            weighted_score = (precision_score_value * precision_weight + 
                            recall_score_value * recall_weight)
            
            # Add precision threshold penalty
            if metrics['precision'] < 0.37:
                weighted_score *= 0.6
            if metrics['recall'] < 0.30:
                weighted_score *= 0.6
            if metrics['recall'] < 0.10:
                weighted_score *= 0.2
                
            print(f"Final weighted score: {weighted_score:.4f}")
            
            # Update best metrics if this trial is better
            if (metrics['precision'] >= self.best_metrics['precision'] and 
                metrics['recall'] >= self.best_metrics['recall']):
                self.best_metrics = metrics
            
            # Set trial attributes for callback
            trial.set_user_attr('threshold', threshold)
            trial.set_user_attr('precision', metrics['precision'])
            trial.set_user_attr('recall', metrics['recall'])
            trial.set_user_attr('f1', metrics['f1'])
            
            return weighted_score
            
        except Exception as e:
            self.logger.error(f"Error in objective function: {str(e)}")
            print(f"Error in trial {trial.number}: {str(e)}")
            return float('-inf')

    def tune_global_model(self, 
                         X_train: pd.DataFrame,
                         y_train: pd.Series,
                         X_val: pd.DataFrame,
                         y_val: pd.Series,
                         n_trials: int = 100) -> Dict[str, Any]:
        """Tune hyperparameters optimizing for validation metrics."""
        if len(X_train) < self.MIN_SAMPLES:
            raise ValueError(f"Insufficient samples: {len(X_train)} < {self.MIN_SAMPLES}")
        
        self.logger.info("Starting global hyperparameter tuning")
        self.logger.info(f"Target metrics - Precision: {self.target_precision:.2f}, "
                        f"Recall: {self.target_recall:.2f}")
        
        study = optuna.create_study(direction='maximize')
        
        def callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
            if study.best_trial == trial:
                mlflow.log_metrics({
                    'best_val_precision': trial.user_attrs['precision'],
                    'best_val_recall': trial.user_attrs['recall'],
                    'best_val_f1': trial.user_attrs['f1'],
                    'best_threshold': trial.user_attrs['threshold']
                })
        
        try:
            study.optimize(
                lambda trial: self.objective(trial, X_train, y_train, X_val, y_val),
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
            
        except Exception as e:
            self.logger.error(f"Error during optimization: {str(e)}")
            raise

def tune_global_model():
    """Main function to tune the global model."""
    logger = ExperimentLogger()
    experiment_name = "global_catboost_hypertuning"
    artifact_path = setup_mlflow_tracking(experiment_name)
    
    # Set artifact location
    os.environ["MLFLOW_ARTIFACT_ROOT"] = artifact_path
    
    mlflow.set_experiment(experiment_name)
       
    try:
        X_train, y_train, X_test, y_test = import_training_data_draws()
        X_val, y_val = create_evaluation_sets()
        
        # Initialize hypertuner with target metrics
        hypertuner = GlobalCatBoostHypertuner(
            logger=logger,
            target_precision=0.50,
            target_recall=0.60,
            precision_weight=0.8
        )
        
        with mlflow.start_run(run_name=f"global_catboost_tuning"):
            # Log target metrics and data statistics
            params_to_log = {
                "target_precision": 0.50,
                "target_recall": 0.60,
                "precision_weight": 0.8,
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "test_samples": len(X_test),
                "draw_ratio_train": (y_train == 1).mean(),
                "draw_ratio_val": (y_val == 1).mean(),
                "draw_ratio_test": (y_test == 1).mean()
            }
            
            mlflow.log_params(params_to_log)
            
            # Tune model using training and validation sets
            best_params = hypertuner.tune_global_model(
                X_train,
                y_train,
                X_val,
                y_val,
                n_trials=100
            )
            
            mlflow.log_params(best_params)
            
    except Exception as e:
        logger.error(f"Error in tune_global_model: {str(e)}")
        raise


if __name__ == "__main__":
    tune_global_model() 