"""
CatBoost-based model for predicting soccer match draws.

This module implements a CatBoost-based model for predicting soccer match draws,
including model creation, training, hyperparameter optimization, and MLflow integration.
"""

import os
import sys
import json
import random
import time
import numpy as np
import pandas as pd
import catboost as cb
from catboost import Pool
import optuna
import mlflow
import time
from pathlib import Path
from typing import Dict, Union, Any, List, Tuple
from datetime import datetime

# Add project root to Python path
try:
    project_root = Path(__file__).parent.parent.parent.parent.parent
    if not project_root.exists():
        # Handle network path by using raw string
        project_root = Path(r"\\".join(str(project_root).split("\\")))
    sys.path.append(str(project_root))
    print(f"Project root: {project_root}")
except Exception as e:
    print(f"Error setting project root path: {e}")
    # Fallback to current directory if path resolution fails
    sys.path.append(os.getcwd())
    print(f"Current directory: {os.getcwd()}")

# Import logger
from utils.logger import ExperimentLogger
experiment_name = "catboost_soccer_prediction"
logger = ExperimentLogger(experiment_name)

from utils.create_evaluation_set import setup_mlflow_tracking, import_selected_features_ensemble
mlrunds_dir = setup_mlflow_tracking(experiment_name)

# Import shared utility functions
from models.StackedEnsemble.shared.hypertuner_utils import (
    predict, predict_proba, evaluate, optimize_threshold, calculate_feature_importance,
    DEFAULT_MIN_RECALL
)
from models.StackedEnsemble.shared.data_loader import DataLoader

# Global settings
min_recall = 0.20  # Minimum acceptable recall
n_trials = 1000  # Number of hyperparameter optimization trials as in notebook

# Base parameters as in the notebook
base_params = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',                        # Primary metric
    'custom_metric': ['Precision', 'Recall'],
    'thread_count': 4,
    'random_seed': 19,
    'task_type': 'CPU'
}

pip_requirements = [
    f'catboost=={cb.__version__}',
    f'mlflow=={mlflow.__version__}'
]

# Set fixed seed and hash seed for determinism
SEED = 19
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Restrict parallel threads across various libraries
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

def load_hyperparameter_space():
    """
    Define the hyperparameter space for CatBoost model tuning.
    
    Returns:
        dict: Hyperparameter space configuration
    """
    try:
        # Define hyperparameter space directly as in notebook
        hyperparameter_space = {
            'learning_rate': {
                'type': 'float',
                'low': 0.005,
                'high': 0.1,
                'log': False,
                'step': 0.005
            },
            'depth': {
                'type': 'int',
                'low': 6,
                'high': 12,
                'step': 1
            },
            'min_data_in_leaf': {
                'type': 'int', 
                'low': 10,
                'high': 200,
                'step': 5
            },
            'subsample': {
                'type': 'float',
                'low': 0.55,
                'high': 0.95,
                'step': 0.02
            },
            'colsample_bylevel': {
                'type': 'float',
                'low': 0.3,
                'high': 0.7,
                'step': 0.02
            },
            'reg_lambda': {
                'type': 'float',
                'low': 0.5,
                'high': 10.0,
                'log': True
            },
            'leaf_estimation_iterations': {
                'type': 'int',
                'low': 2,
                'high': 20,
                'step': 2
            },
            'bagging_temperature': {
                'type': 'float',
                'low': 1.0,
                'high': 10.0,
                'step': 0.1
            },
            'scale_pos_weight': {
                'type': 'float',
                'low': 1.5,
                'high': 5.0,
                'step': 0.05
            },
            'early_stopping_rounds': {
                'type': 'int',
                'low': 200,
                'high': 700,
                'step': 10
            }
        }
        return hyperparameter_space
    except Exception as e:
        logger.error(f"Error creating hyperparameter space: {str(e)}")
        return None

def create_model(model_params):
    """
    Create and configure CatBoost model instance.
    
    Args:
        model_params (dict): Model parameters
        
    Returns:
        cb.CatBoostClassifier: Configured CatBoost model
    """
    if model_params is None:
        model_params = base_params
    
    model_params.update(base_params)
    
    return cb.CatBoostClassifier(**model_params)

def train_model(X_train, y_train, X_test, y_test, X_eval, y_eval, model_params):
    """
    Train a CatBoost model with early stopping and threshold optimization.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Validation features
        y_test: Validation labels
        X_eval: Evaluation features
        y_eval: Evaluation labels
        model_params: Model parameters
        
    Returns:
        tuple: (trained_model, metrics)
    """
    try:
        X_combined = pd.concat([X_train, X_test])
        y_combined = pd.concat([y_train, y_test])
        # Extract early stopping rounds if present
        early_stopping_rounds = model_params.pop('early_stopping_rounds', 100)
        
        # Create model with remaining parameters
        model = create_model(model_params)
        
        # Create eval set for early stopping
        eval_set = Pool(X_eval, y_eval)
        
        # Fit model with early stopping
        model.fit(
            X_combined, y_combined,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,    
            verbose=False
        )
        
        # Get validation predictions and optimize threshold
        best_threshold, metrics = optimize_threshold(model, X_eval, y_eval, min_recall=min_recall)
        
        return model, metrics
        
    except Exception as e:
        logger.error(f"Error training CatBoost model: {str(e)}")
        raise

def optimize_hyperparameters(X_train, y_train, X_test, y_test, X_eval, y_eval, hyperparameter_space):
    """
    Run hyperparameter optimization with Optuna.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Testing features
        y_test: Testing labels
        X_eval: Evaluation features
        y_eval: Evaluation labels
        hyperparameter_space: Hyperparameter space configuration
        
    Returns:
        dict: Best parameters
    """
    logger.info("Starting hyperparameter optimization")
    
    if not hyperparameter_space:
        hyperparameter_space = load_hyperparameter_space()
    
    best_score = 0.0
    
    def objective(trial):
        try:
            params = base_params.copy()
            
            # Add hyperparameters from config with step size if provided
            for param_name, param_config in hyperparameter_space.items():
                if param_config['type'] == 'float':
                    if 'step' in param_config:
                        # Use step if it is provided
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
            # Train model and get metrics
            model, metrics = train_model(
                X_train, y_train,
                X_test, y_test,
                X_eval, y_eval,
                params
            )
            
            recall = metrics.get('recall', 0.0)
            precision = metrics.get('precision', 0.0)
            # Optimize for precision while maintaining minimum recall
            score = precision if recall >= min_recall else 0.0
            
            logger.info(f"Trial {trial.number}:")
            logger.info(f"  Params: {params}")
            logger.info(f"  Score: {score}")
            
            for metric_name, metric_value in metrics.items():
                trial.set_user_attr(metric_name, metric_value)
            return score
            
        except Exception as e:
            logger.error(f"Trial failed: {str(e)}")
            return 0.0
    
    try:
        # Generate a seed based on the current time
        random_seed = int(time.time())
        random_sampler = optuna.samplers.RandomSampler(seed=random_seed)
        study = optuna.create_study(
            study_name='catboost_optimization',
            direction='maximize',
            sampler=random_sampler
        )
        # --- New: Initialize and add a callback function ---
        best_score = -float('inf')  # Initialize with worst possible score
        best_params = {}
        top_trials = []  # To record top trials

        def callback(study, trial):
            nonlocal best_score, best_params, top_trials
            logger.info(f"Current best score: {best_score:.4f}")
            if trial.value > best_score:
                best_score = trial.value
                best_params = trial.params
                logger.info(f"New best score found in trial {trial.number}: {best_score:.4f}")
            # Record and log top trials every few iterations
            current_run = (trial.value, trial.params, trial.number)
            top_trials.append(current_run)
            top_trials.sort(key=lambda x: x[0], reverse=True)
            top_trials = top_trials[:10]
            if trial.number % 9 == 0:
                logger.info("Top 10 trials so far:")
                for rank, t in enumerate(top_trials, 1):
                    logger.info(f"Rank {rank}, Trial {t[2]}, Score: {t[0]:.4f}, Params: {t[1]}")
            return best_score

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=1000000,
            show_progress_bar=True,
            callbacks=[callback]
        )
        best_params.update(base_params)
        logger.info(f"Best trial value: {best_score}")
        logger.info(f"Best parameters found: {best_params}")
        return best_params
            
    except Exception as e:
        logger.error(f"Error in hyperparameter optimization: {str(e)}")
        raise

def hypertune_catboost(experiment_name: str):
    """
    Main training function with MLflow tracking.
    
    Args:
        experiment_name (str): Experiment name for MLflow tracking
        
    Returns:
        tuple: (best_params, best_metrics)
    """
    try:
        # Start MLflow run
        with mlflow.start_run(run_name=f"catboost_base_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            # Set tags
            mlflow.set_tags({
                "model_type": "catboost_base",
                "training_mode": "global",
                "cpu_only": True
            })
            
            # Load hyperparameter space
            hyperparameter_space = load_hyperparameter_space()
            
            # Run hyperparameter optimization
            logger.info("Starting hyperparameter optimization")
            best_params = optimize_hyperparameters(
                X_train, y_train,
                X_test, y_test,
                X_eval, y_eval,
                hyperparameter_space=hyperparameter_space
            )
            
            # Train final model with best parameters
            logger.info("Training final model with best parameters")
            model, metrics = train_model(
                X_train, y_train,
                X_test, y_test,
                X_eval, y_eval,
                best_params
            )
            
            # Log final metrics
            mlflow.log_metrics({
                "precision": metrics.get('precision', 0.0),
                "recall": metrics.get('recall', 0.0),
                "f1": metrics.get('f1', 0.0),
                "auc": metrics.get('auc', 0.0),
                "threshold": metrics.get('threshold', 0.5)
            })
            
            # Log model
            # Create input example with a sample from evaluation data
            # Handle integer columns by converting them to float64 to properly manage missing values
            input_example = X_eval.iloc[:5].copy() if hasattr(X_eval, 'iloc') else X_eval[:5].copy()
            
            # Identify and convert integer columns to float64 to prevent schema enforcement errors
            if hasattr(input_example, 'dtypes'):
                for col in input_example.columns:
                    if input_example[col].dtype.kind == 'i':
                        logger.info(f"Converting integer column '{col}' to float64 to handle potential missing values")
                        input_example[col] = input_example[col].astype('float64')

            # Infer signature with proper handling for integer columns with potential missing values
            signature = mlflow.models.infer_signature(
                input_example,
                model.predict(input_example)
            )
            
            # Log warning about integer columns in signature
            logger.info("Model signature created - check logs for any warnings about integer columns")

            mlflow.catboost.log_model(
                model,
                "model",
                pip_requirements=pip_requirements,  # Explicitly set requirements
                registered_model_name=f"catboost_{datetime.now().strftime('%Y%m%d_%H%M')}",
                signature=signature
            )
            
            return best_params, metrics
            
    except Exception as e:
        logger.error(f"Error in hyperparameter tuning: {str(e)}")
        return None, None

def log_to_mlflow(model, metrics, params, experiment_name):
    """
    Log trained model, metrics, and parameters to MLflow.
    
    Args:
        model: Trained CatBoost model
        metrics: Model evaluation metrics
        params: Model parameters
        experiment_name: Experiment name
        
    Returns:
        str: Run ID
    """
    try:
        # Set up MLflow tracking
        mlflow.set_experiment(experiment_name)
        
        # Start a new run
        with mlflow.start_run(run_name=f"catboost_{datetime.now().strftime('%Y%m%d_%H%M')}") as run:
            # Log parameters
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model
            model_info = mlflow.catboost.log_model(
                model,
                "model",
                registered_model_name=f"catboost_{datetime.now().strftime('%Y%m%d_%H%M')}"
            )
            
            # Log feature importance using the shared utility
            importance_df = calculate_feature_importance(model)
            
            # Save to CSV and log as artifact
            if not importance_df.empty:
                importance_path = "feature_importance.csv"
                importance_df.to_csv(importance_path, index=False)
                mlflow.log_artifact(importance_path)
                os.remove(importance_path)
            
            logger.info(f"Model logged to MLflow: {model_info.model_uri}")
            logger.info(f"Run ID: {run.info.run_id}")
            return run.info.run_id
            
    except Exception as e:
        logger.error(f"Error logging to MLflow: {str(e)}")
        return None

def train_with_precision_target(X_train, y_train, X_test, y_test, X_eval, y_eval):
    """
    Train CatBoost model with focus on precision target.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Testing features
        y_test: Testing labels
        X_eval: Evaluation features
        y_eval: Evaluation labels
        
    Returns:
        tuple: (best_model, best_metrics)
    """
    try:
        # Run hyperparameter tuning
        logger.info("Running hyperparameter tuning")
        best_params, _ = hypertune_catboost(experiment_name)
        
        if not best_params:
            logger.warning("Hyperparameter tuning failed. Using default parameters.")
            best_params = base_params.copy()
            best_params.update({
                'learning_rate': 0.05,
                'depth': 6,
                'min_data_in_leaf': 20,
                'subsample': 0.8,
                'colsample_bylevel': 0.8,
                'reg_lambda': 1.0
            })
        
        # Train final model with best parameters
        logger.info("Training final model with best parameters")
        model, metrics = train_model(
            X_train, y_train,
            X_test, y_test,
            X_eval, y_eval,
            best_params
        )
        
        # Log to MLflow
        log_to_mlflow(model, metrics, best_params, experiment_name)
        
        return model, metrics
        
    except Exception as e:
        logger.error(f"Error in precision-focused training: {str(e)}")
        return None, None

def main():
    """
    Main execution function.
    """
    try:
        logger.info("Starting CatBoost model training")
        
        # Import data at runtime to avoid global scope issues
        from models.StackedEnsemble.shared.data_loader import DataLoader
        
        global X_train, y_train, X_test, y_test, X_eval, y_eval
        
        # Load data
        dataloader = DataLoader()
        X_train, y_train, X_test, y_test, X_eval, y_eval = dataloader.load_data()
        features = import_selected_features_ensemble(model_type='cat')
        X_train = X_train[features]
        X_test = X_test[features]
        X_eval = X_eval[features]
        # Log data shapes
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Testing data shape: {X_test.shape}")
        logger.info(f"Evaluation data shape: {X_eval.shape}")
        logger.info(f"Positive class ratio - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}, Eval: {y_eval.mean():.3f}")
        
        # Hyperparameter optimization
        logger.info("Starting hyperparameter optimization")
        best_params = hypertune_catboost(experiment_name)
        logger.info(f"Best parameters: {best_params}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main() 