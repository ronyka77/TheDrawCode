# Standard library imports
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import tempfile

# Third-party imports
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import mlflow
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import KFold

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

# Local imports
from utils.logger import ExperimentLogger
from utils.create_evaluation_set import create_evaluation_sets_draws, import_training_data_draws_new, setup_mlflow_tracking, get_selected_columns_draws
from models.xgboost_ensemble_model import TwoStageEnsemble, VotingEnsemble


def setup_xgboost_temp_directory(logger: ExperimentLogger, project_root: Path) -> str:
    """Set up and verify XGBoost temporary directory.
    
    Args:
        logger: Logger instance for logging messages
        project_root: Project root path
        
    Returns:
        str: Path to the verified temporary directory
    """
    # Set up temp directory for XGBoost using project_root
    temp_dir = os.path.join(project_root, "temp", "xgboost")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Ensure temp_dir exists and is writable
    try:
        test_file = os.path.join(temp_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
    except Exception as e:
        logger.error(f"Temp directory {temp_dir} is not writable: {e}")
        # Fallback to system temp directory
        temp_dir = os.path.join(tempfile.gettempdir(), "xgboost_temp")
        os.makedirs(temp_dir, exist_ok=True)
        logger.info(f"Using fallback temp directory: {temp_dir}")
    
    # Set environment variables with verified temp directory
    os.environ.update({
        'XGBOOST_CACHE_DIR': temp_dir,
        'TMPDIR': temp_dir,
        'TEMP': temp_dir,
        'TMP': temp_dir
    })
    
    # Log the actual paths being used
    logger.info(f"Using temp directory: {temp_dir}")
    # logger.info(f"XGBOOST_CACHE_DIR: {os.environ.get('XGBOOST_CACHE_DIR')}")
    
    return temp_dir


class EnsembleHypertuner:
    """Hyperparameter tuner for ensemble XGBoost models."""
    
    def __init__(self, logger: Optional[ExperimentLogger] = None, temp_dir: str = None):
        self.logger = logger or ExperimentLogger()
        
        # Set up XGBoost temp directory using the new function
        self.temp_dir = temp_dir or setup_xgboost_temp_directory(self.logger, project_root)
        
        self.best_params = {
            'two_stage': {'stage1': {}, 'stage2': {}},
            'voting': {'base': {}, 'models': []}
        }
        self.best_metrics = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    def tune_ensemble(self,
                     X_train: pd.DataFrame,
                     y_train: pd.Series,
                     X_val: pd.DataFrame,
                     y_val: pd.Series,
                     n_trials: int = 200) -> Dict[str, Any]:
        """Tune hyperparameters for both ensemble models."""
        
        study = optuna.create_study(direction='maximize')
        
        def objective(trial: optuna.Trial, X_train, y_train, X_val, y_val):
            try:
                # Stage 1 Parameters
                stage1_params = {
                    'objective': 'binary:logistic',
                    'tree_method': 'hist',
                    'eta': trial.suggest_float('s1_eta', 0.001, 0.015),
                    'min_child_weight': trial.suggest_int('s1_min_child_weight', 40, 80),
                    'gamma': trial.suggest_float('s1_gamma', 2.0, 8.0),
                    'subsample': trial.suggest_float('s1_subsample', 0.7, 0.95),
                    'colsample_bytree': trial.suggest_float('s1_colsample_bytree', 0.7, 0.9),
                    'scale_pos_weight': trial.suggest_float('s1_scale_pos_weight', 2.0, 3.5),
                    'max_depth': trial.suggest_int('s1_max_depth', 4, 5),
                    'reg_alpha': trial.suggest_float('s1_alpha', 0.1, 3.0),
                    'reg_lambda': trial.suggest_float('s1_lambda', 0.3, 3.0),
                    'n_estimators': trial.suggest_int('s1_n_estimators', 11000, 20000),
                }
                
                # Stage 2 Parameters
                stage2_params = {
                    'objective': 'binary:logistic',
                    'tree_method': 'hist',
                    'eta': trial.suggest_float('s2_eta', 0.001, 0.015),
                    'min_child_weight': trial.suggest_int('s2_min_child_weight', 40, 80),
                    'gamma': trial.suggest_float('s2_gamma', 2.0, 8.0),
                    'subsample': trial.suggest_float('s2_subsample', 0.7, 0.95),
                    'colsample_bytree': trial.suggest_float('s2_colsample_bytree', 0.7, 0.9),
                    'scale_pos_weight': trial.suggest_float('s2_scale_pos_weight', 2.0, 3.5),
                    'max_depth': trial.suggest_int('s2_max_depth', 4, 5),
                    'reg_alpha': trial.suggest_float('s2_alpha', 0.1, 3.0),
                    'reg_lambda': trial.suggest_float('s2_lambda', 0.3, 3.0),
                    'n_estimators': trial.suggest_int('s2_n_estimators', 11000, 20000),
                }
                
                # Voting Ensemble Parameters
                voting_base = {
                    'objective': 'binary:logistic',
                    'tree_method': 'hist',
                    'eta': trial.suggest_float('v_eta', 0.001, 0.015),
                    'min_child_weight': trial.suggest_int('v_min_child_weight', 40, 80),
                    'gamma': trial.suggest_float('v_gamma', 2.0, 8.0),
                    'subsample': trial.suggest_float('v_subsample', 0.7, 0.95),
                    'colsample_bytree': trial.suggest_float('v_colsample_bytree', 0.7, 0.9),
                    'scale_pos_weight': trial.suggest_float('v_scale_pos_weight', 2.0, 3.5),
                    'max_depth': trial.suggest_int('v_max_depth', 4, 5),
                    'reg_alpha': trial.suggest_float('v_alpha', 0.1, 3.0),
                    'reg_lambda': trial.suggest_float('v_lambda', 0.3, 3.0),
                    'n_estimators': trial.suggest_int('v_n_estimators', 11000, 20000),
                }
                
                # Modified threshold ranges
                threshold1 = trial.suggest_float('threshold1', 0.15, 0.25)  # Lower range for first stage
                threshold2 = trial.suggest_float('threshold2', 0.40, 0.55)  # Lower range for second stage
                voting_thresholds = [
                    trial.suggest_float(f'v_threshold_{i}', 0.40, 0.55)  # Lower range for voting
                    for i in range(5)
                ]
                
                # Initialize and train models
                two_stage = TwoStageEnsemble(logger=self.logger)
                two_stage.stage1_params.update(stage1_params)
                two_stage.stage2_params.update(stage2_params)
                two_stage.threshold1 = threshold1
                two_stage.threshold2 = threshold2

                voting = VotingEnsemble(n_models=5, logger=self.logger)
                voting.base_params.update(voting_base)
                voting.thresholds = voting_thresholds

                # Train models with early stopping to reduce training time
                two_stage.stage1_params['early_stopping_rounds'] = 500
                two_stage.stage2_params['early_stopping_rounds'] = 500
                voting.base_params['early_stopping_rounds'] = 500
                
                # Train with reduced verbosity
                two_stage.fit(X_train, y_train, X_val, y_val)
                voting.fit(X_train, y_train, X_val, y_val)

                # Get probabilities
                two_stage_probs = two_stage.predict_proba(X_val)[:, 1]
                voting_probs = np.mean([model.predict_proba(X_val)[:, 1] for model in voting.models], axis=0)

                # Modified prediction combination with lower thresholds
                combined_preds = ((two_stage_probs >= threshold2) & 
                                (voting_probs >= np.mean(voting_thresholds))).astype(int)

                # Calculate metrics
                precision = precision_score(y_val, combined_preds, zero_division=0)
                recall = recall_score(y_val, combined_preds, zero_division=0)
                f1 = f1_score(y_val, combined_preds, zero_division=0)

                # Store metrics in trial
                print(f"precision: {precision:.4f}")
                print(f"recall: {recall:.4f}")
                print(f"f1: {f1:.4f}")
                trial.set_user_attr('precision', precision)
                trial.set_user_attr('recall', recall)
                trial.set_user_attr('f1', f1)

                # Modified scoring function with more balanced approach
                if precision == 0 or recall == 0:
                    score = 0.0
                else:
                    # Base weights
                    precision_weight = 0.7
                    recall_weight = 0.3
                    
                    # Calculate weighted score
                    weighted_score = (precision * precision_weight + recall * recall_weight)
                    
                    # Apply f1 multiplier to encourage balance
                    score = weighted_score * f1
                    
                    # Apply minimum thresholds but with lower requirements
                    if precision < 0.25 or recall < 0.1:
                        score *= 0.5
                self.logger.info(f"Trial parameters: {trial.params}")
                self.logger.info(f"Score: {score:.4f}")
                self.logger.info(f"Precision: {precision:.4f}")
                self.logger.info(f"Recall: {recall:.4f}")
                return score
            except Exception as e:
                self.logger.error(f"Trial failed with error: {str(e)}")
                self.logger.error(f"Trial parameters: {trial.params}")
                return float('-inf')
        
        try:
            study.optimize(
                lambda trial: objective(trial, X_train, y_train, X_val, y_val),
                n_trials=n_trials,
                catch=(Exception,)
            )
            
            if study.best_trial.value == float('-inf'):
                raise ValueError("No successful trials completed")
            
            # Log best parameters and metrics without starting a new run
            mlflow.log_metrics({
                'best_val_precision': study.best_trial.user_attrs['precision'],
                'best_val_recall': study.best_trial.user_attrs['recall'],
                'best_val_f1': study.best_trial.user_attrs['f1']
            })
            
            # Create and log best parameters
            best_params = {
                'two_stage': {
                    'stage1': {k[3:]: v for k, v in study.best_trial.params.items() if k.startswith('s1_')},
                    'stage2': {k[3:]: v for k, v in study.best_trial.params.items() if k.startswith('s2_')},
                    'thresholds': {
                        'threshold1': study.best_trial.params['threshold1'],
                        'threshold2': study.best_trial.params['threshold2']
                    }
                },
                'voting': {
                    'base': {k[2:]: v for k, v in study.best_trial.params.items() 
                            if k.startswith('v_') and not k.startswith('v_threshold')},
                    'thresholds': [study.best_trial.params[f'v_threshold_{i}'] for i in range(5)]
                }
            }
            
            mlflow.log_params({
                f"best_{k}": str(v) for k, v in best_params.items()
            })
            
            return best_params
                
        except Exception as e:
            self.logger.error(f"Error during optimization: {str(e)}")
            raise

def tune_ensemble_model():
    """Main function to tune the ensemble model."""
    logger = ExperimentLogger(experiment_name="xgboost_ensemble_hypertuning")
    experiment_name = "xgboost_ensemble_hypertuning"
    artifact_dir = os.path.join(project_root, "mlflow_artifacts", experiment_name)
    temp_dir = setup_xgboost_temp_directory(logger, project_root)
       
    # Setup MLflow tracking with explicit artifact location
    setup_mlflow_tracking(experiment_name)
    mlflow.set_experiment(experiment_name)
    
    # Set artifact location explicitly
    os.environ['MLFLOW_ARTIFACT_ROOT'] = artifact_dir

    try:
        # Load and prepare data
        X_train, y_train, X_test, y_test = import_training_data_draws_new()
        X_val, y_val = create_evaluation_sets_draws()
        
        print(f"X_train: {X_train.shape}")
        print(f"X_val: {X_val.shape}")
        print(f"X_test: {X_test.shape}")
        print(f"y_train: {y_train.shape}")
        print(f"y_val: {y_val.shape}")
        print(f"y_test: {y_test.shape}")
        
        with mlflow.start_run(run_name="ensemble_hypertuning"):
            hypertuner = EnsembleHypertuner(logger=logger, temp_dir=temp_dir)
            # Log data statistics
            mlflow.log_params({
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "test_samples": len(X_test),
                "train_draws": y_train.mean(),
                "val_draws": y_val.mean(),
                "test_draws": y_test.mean()
            })
            
            # Inspect data types before tuning
            # print(f"X_train dtypes: {X_train.dtypes.to_dict()}")
            # print(f"X_val dtypes: {X_val.dtypes.to_dict()}")
            # print(f"X_test dtypes: {X_test.dtypes.to_dict()}")
            
            # Tune models
            best_params = hypertuner.tune_ensemble(
                X_train, y_train,
                X_val, y_val,
                n_trials=200
            )
            
            # Train final models with best parameters
            two_stage = TwoStageEnsemble(logger=logger, temp_dir=temp_dir)
            two_stage.stage1_params.update(best_params['two_stage']['stage1'])
            two_stage.stage2_params.update(best_params['two_stage']['stage2'])
            two_stage.threshold1 = best_params['two_stage']['thresholds']['threshold1']
            two_stage.threshold2 = best_params['two_stage']['thresholds']['threshold2']

            voting = VotingEnsemble(n_models=5, logger=logger, temp_dir=temp_dir)
            voting.base_params.update(best_params['voting']['base'])
            voting.thresholds = best_params['voting']['thresholds']

            # Train and evaluate
            two_stage.fit(X_train, y_train, X_val, y_val)
            voting.fit(X_train, y_train, X_val, y_val)

            # Final evaluation
            def evaluate(X, y, prefix):
                # Get probabilities for class 1 only
                two_stage_probs = two_stage.predict_proba(X)[:, 1]  # Extract second column for positive class
                voting_probs = np.mean([
                    model.predict_proba(X)[:, 1]  # Extract second column for each model
                    for model in voting.models
                ], axis=0)
                
                combined_preds = ((two_stage_probs >= two_stage.threshold2) & 
                                (voting_probs >= np.mean(voting.thresholds))).astype(int)
                
                metrics = {
                    'precision': precision_score(y, combined_preds, zero_division=0),
                    'recall': recall_score(y, combined_preds, zero_division=0),
                    'f1': f1_score(y, combined_preds, zero_division=0)
                }
                
                mlflow.log_metrics({
                    f"{prefix}_{k}": v for k, v in metrics.items()
                })
                
                return metrics

            val_metrics = evaluate(X_val, y_val, "val")
            test_metrics = evaluate(X_test, y_test, "test")

            # Log best parameters
            mlflow.log_params({"best_" + k: str(v) for k, v in best_params.items()})
            print(f"best_params: {best_params}")
            return {
                'best_params': best_params,
                'validation_metrics': val_metrics,
                'test_metrics': test_metrics
            }

    except Exception as e:
        logger.error(f"Error during ensemble model tuning: {str(e)}")
        raise

if __name__ == "__main__":
    results = tune_ensemble_model()
    
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