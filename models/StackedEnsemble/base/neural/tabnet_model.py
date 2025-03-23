# Imports
import os
import sys
import json
import pickle
import random
import logging
import warnings
import numpy as np
import pandas as pd
import mlflow
import optuna
import time
import gc
from datetime import datetime
from pathlib import Path
import torch
import torch.optim as optim
from pytorch_tabnet.tab_model import TabNetClassifier

# Set project root similar to xgboost_model.py
try:
    project_root = Path(__file__).parent.parent.parent.parent.parent
    if not project_root.exists():
        project_root = Path(r"\\".join(str(project_root).split("\\")))
    sys.path.append(str(project_root))
    print(f"Project root: {project_root}")
except Exception as e:
    print(f"Error setting project root path: {e}")
    sys.path.append(os.getcwd())
    print(f"Current directory: {os.getcwd()}")

# Logger and shared utilities
from utils.logger import ExperimentLogger
experiment_name = "tabnet_soccer_prediction"
logger = ExperimentLogger(experiment_name=experiment_name)

# Import shared utility functions
from models.StackedEnsemble.shared.hypertuner_utils import predict, predict_proba, evaluate, optimize_threshold, calculate_feature_importance
from models.StackedEnsemble.shared.data_loader import DataLoader
from utils.create_evaluation_set import setup_mlflow_tracking, import_selected_features_ensemble

# Filter specific TabNet weight-related warnings
warnings.filterwarnings("ignore", message=".*imbalanced.*|.*weight.*|.*class_weight.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*sample_weight.*", category=UserWarning)

# Global settings
min_recall = 0.20
# You can adjust n_trials if needed
n_trials = 20000

# Base parameters specific for TabNet
base_params = {
    'optimizer_fn': optim.Adam,
    'mask_type': 'sparsemax',
    'eval_metric': ['auc', 'logloss'],
    'verbose': 0,
    'seed': 19,
    'device_name': 'cpu'
}

# Set fixed seed and hash seed for determinism
SEED = 19
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Restrict parallel threads across various libraries
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
# PyTorch specific reproducibility settings
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)  # Force deterministic algorithms
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Configure PyTorch threads
torch.set_num_threads(4)
torch.set_num_interop_threads(4)

def load_hyperparameter_space():
    """
    Define hyperparameter space for TabNet tuning.
    """
    hyperparameter_space = {
        'learning_rate': {
            'type': 'float',
            'low': 1e-3,
            'high': 1e-1,
            'log': True
        },
        'n_d': {
            'type': 'int',
            'low': 4,
            'high': 30
        },
        'n_a': {
            'type': 'int',
            'low': 4,
            'high': 20
        },
        'n_steps': {
            'type': 'int',
            'low': 3,
            'high': 10
        },
        'gamma': {
            'type': 'float',
            'low': 0.8,
            'high': 2.5,
            'step': 0.05
        },
        'lambda_sparse': {
            'type': 'float',
            'low': 1e-6,
            'high': 1e-3,
            'log': True
        },
        'momentum': {
            'type': 'float',
            'low': 0.8,
            'high': 0.99,
            'step': 0.005
        },
        'patience': {
            'type': 'int',
            'low': 3,
            'high': 20
        },
        'max_epochs': {
            'type': 'int',
            'low': 40,
            'high': 100,
            'step': 2
        }
    }
    return hyperparameter_space

class TabNetWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
        
    def predict(self, context, model_input):
        """
        Returns class predictions.
        """
        if hasattr(model_input, 'values'):
            data = model_input.values
        else:
            data = model_input
        return self.model.predict(data)
    
    def predict_proba(self, context, model_input):
        """
        Returns probability estimates for each class.
        """
        if hasattr(model_input, 'values'):
            data = model_input.values
        else:
            data = model_input
        return self.model.predict_proba(data)

def create_model(model_params):
    """
    Create and configure TabNet model instance based on provided parameters.
    """
    try:
        params = base_params.copy()
        params.update(model_params)
        # Update optimizer parameters with the tuned learning rate
        if 'learning_rate' in params:
            params['optimizer_params'] = {'lr': params['learning_rate']}
            # Optionally remove learning_rate from params to avoid passing it to TabNetClassifier
            del params['learning_rate']
        # Remove eval_metric from parameters since it's added during model fitting
        if 'eval_metric' in params:
            params.pop('eval_metric')
        if 'patience' in params:
            params.pop('patience')
        if 'max_epochs' in params:
            params.pop('max_epochs')
        model = TabNetClassifier(**params)
        return model
    except Exception as e:
        logger.error(f"Error creating TabNet model: {str(e)}")
        raise

def train_model(X_train, y_train, X_test, y_test, X_eval, y_eval, model_params):
    """
    Train a TabNet model with early stopping.
    Converts input data to numpy arrays if they are pandas DataFrames.
    Returns the trained model and evaluation metrics after threshold optimization.
    """
    try:
        model = create_model(model_params)
        # Convert to numpy arrays if needed
        if hasattr(X_train, 'values'):
            X_train = X_train.values
            y_train = y_train.values if hasattr(y_train, 'values') else y_train
            X_test = X_test.values if hasattr(X_test, 'values') else X_test
            y_test = y_test.values if hasattr(y_test, 'values') else y_test
            X_eval = X_eval.values if hasattr(X_eval, 'values') else X_eval
            y_eval = y_eval.values if hasattr(y_eval, 'values') else y_eval
        # Combine training and testing data similar to xgboost_model.py
        X_combined = np.concatenate([X_train, X_test], axis=0)
        y_combined = np.concatenate([y_train, y_test], axis=0)
        # Train the model with early stopping on the evaluation set
        model.fit(
            X_combined, y_combined,
            eval_set=[(X_eval, y_eval)],
            eval_metric=model_params.get('eval_metric', ['auc', 'logloss', 'accuracy']),
            max_epochs=model_params.get('max_epochs', 50),
            patience=model_params.get('patience', 10)
        )
        # Optimize threshold using shared utility
        best_threshold, metrics = optimize_threshold(model, X_eval, y_eval, min_recall=min_recall)
        return model, metrics
    except Exception as e:
        logger.error(f"Error training TabNet model: {str(e)}")
        raise

def optimize_hyperparameters(X_train, y_train, X_test, y_test, X_eval, y_eval, hyperparameter_space):
    logger.info("Starting hyperparameter optimization for TabNet")
    if not hyperparameter_space:
        hyperparameter_space = load_hyperparameter_space()
    best_score = -float('inf')
    best_params = {}
    global_top_trials = []
    top_trials = []

    def objective(trial):
        try:
            params = base_params.copy()
            # Iterate over hyperparameter space and suggest values
            for param_name, param_config in hyperparameter_space.items():
                if param_config['type'] == 'float':
                    if 'step' in param_config:
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_config['low'],
                            param_config['high'],
                            step=param_config['step'],
                            log=param_config.get('log', False)
                        )
                    else:
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_config['low'],
                            param_config['high'],
                            log=param_config.get('log', False)
                        )
                elif param_config['type'] == 'int':
                    if 'step' in param_config:
                        params[param_name] = trial.suggest_int(
                            param_name,
                            param_config['low'],
                            param_config['high'],
                            step=param_config['step']
                        )
                    else:
                        params[param_name] = trial.suggest_int(
                            param_name,
                            param_config['low'],
                            param_config['high']
                        )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(param_name, param_config['choices'])
            
            # Train model and get metrics
            model, metrics = train_model(X_train, y_train, X_test, y_test, X_eval, y_eval, params)
            recall = metrics.get('recall', 0.0)
            precision = metrics.get('precision', 0.0)
            score = precision if recall >= min_recall else 0.0
            logger.info(f"  Score: {score}")
            for metric_name, metric_value in metrics.items():
                trial.set_user_attr(metric_name, metric_value)
            return score
        except Exception as e:
            logger.error(f"Trial failed: {str(e)}")
            return 0.0

    def callback(study, trial):
        nonlocal best_score, best_params, top_trials
        logger.info(f"Current best score in this batch: {best_score:.4f}")
        if trial.value > best_score:
            best_score = trial.value
            best_params = trial.params
            logger.info(f"New best score found in trial {trial.number}: {best_score:.4f}")
        current_run = (trial.value, trial.params, trial.number)
        top_trials.append(current_run)
        top_trials.sort(key=lambda x: x[0], reverse=True)
        top_trials[:] = top_trials[:10]
        if trial.number % 9 == 0:
            table_rows = [f"| {i+1} | {rec[2]} | {rec[0]:.4f} | {rec[1]} |" for i, rec in enumerate(top_trials)]
            logger.info("Top trials in current batch:")
            for row in table_rows:
                logger.info(row)
        if trial.number % 100 == 0 and global_top_trials:
            table_rows = [f"| {i+1} | {rec[2]} | {rec[0]:.4f} | {rec[1]} |" for i, rec in enumerate(global_top_trials[:10])]
            logger.info("Global top trials:")
            for row in table_rows:
                logger.info(row)
        return best_score

    storage_url = "sqlite:///optuna_tabnet.db"
    study_name = "tabnet_optimization"
    total_trials = n_trials
    batch_size = 1000
    num_batches = total_trials // batch_size
    if total_trials % batch_size != 0:
        num_batches += 1

    for batch in range(num_batches):
        random_seed = int(time.time())
        new_sampler = optuna.samplers.RandomSampler(seed=random_seed)
        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',
            storage=storage_url,
            load_if_exists=True,
            sampler=new_sampler
        )
        logger.info(f"Starting batch {batch+1}/{num_batches} with new sampler (seed={random_seed})")
        study.optimize(objective, n_trials=batch_size, show_progress_bar=True, callbacks=[callback])
        for trial_record in top_trials:
            global_top_trials.append(trial_record)
        global_top_trials.sort(key=lambda x: x[0], reverse=True)
        global_top_trials = global_top_trials[:10]
    if global_top_trials:
        best_score, best_params, best_trial_number = global_top_trials[0]
    else:
        best_params = {}

    best_params.update(base_params)
    logger.info(f"Best trial value across batches: {best_score:.4f}")
    logger.info(f"Best parameters found: {best_params}")
    return best_params

def hypertune_tabnet(experiment_name: str):
    """
    Main hypertuning function for TabNet with MLflow tracking.
    Returns best_params and metrics.
    """
    try:
        with mlflow.start_run(run_name=f"tabnet_base_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            mlflow.set_tags({
                "model_type": "tabnet",
                "training_mode": "global",
                "cpu_only": True
            })
            hyperparameter_space = load_hyperparameter_space()
            logger.info("Starting hyperparameter optimization for TabNet")
            best_params = optimize_hyperparameters(X_train, y_train, X_test, y_test, X_eval, y_eval, hyperparameter_space=hyperparameter_space)
            logger.info("Training final TabNet model with best parameters")
            model, metrics = train_model(X_train, y_train, X_test, y_test, X_eval, y_eval, best_params)
            mlflow.log_metrics({
                "precision": metrics.get('precision', 0.0),
                "recall": metrics.get('recall', 0.0),
                "f1": metrics.get('f1', 0.0),
                "auc": metrics.get('auc', 0.0),
                "threshold": metrics.get('threshold', 0.5)
            })
            logger.info("Logging best parameters to MLflow")
            for param_name, param_value in best_params.items():
                mlflow.log_param(param_name, param_value)
            # Create input example from X_eval (convert to DataFrame if needed)
            if not isinstance(X_eval, pd.DataFrame):
                input_example = pd.DataFrame(X_eval[:5])
            else:
                input_example = X_eval.iloc[:5].copy()
            signature = mlflow.models.infer_signature(
                input_example,
                model.predict(input_example)
            )
            # Log model using a custom pyfunc wrapper
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=TabNetWrapper(model),
                signature=signature
            )
            return best_params, metrics
    except Exception as e:
        logger.error(f"Error in TabNet hypertuning: {str(e)}")
        return None, None

def main():
    """
    Main execution function for TabNet hypertuning.
    """
    try:
        logger.info("Starting TabNet model hypertuning")
        # Setup MLflow tracking directory
        mlruns_dir = setup_mlflow_tracking(experiment_name)
        global X_train, y_train, X_test, y_test, X_eval, y_eval
        # Load data using shared DataLoader
        dataloader = DataLoader()
        X_train, y_train, X_test, y_test, X_eval, y_eval = dataloader.load_data()
        # Select features for TabNet if needed
        features = import_selected_features_ensemble(model_type='tabnet')
        X_train = X_train[features]
        X_test = X_test[features]
        X_eval = X_eval[features]

        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Testing data shape: {X_test.shape}")
        logger.info(f"Evaluation data shape: {X_eval.shape}")
        logger.info(f"Positive class ratio - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}, Eval: {y_eval.mean():.3f}")
        logger.info(f"Current base parameters: {base_params}")
        best_params, metrics = hypertune_tabnet(experiment_name)

        logger.info(f"Hypertuning completed with parameters: {best_params}")
        logger.info(f"Evaluation metrics: {metrics}")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")


if __name__ == "__main__":
    main() 