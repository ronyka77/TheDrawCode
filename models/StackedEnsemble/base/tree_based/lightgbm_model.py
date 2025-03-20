"""
LightGBM Model for Soccer Draw Prediction

This module implements a LightGBM-based model for predicting soccer match draws.
It includes functionality for model creation, training, hyperparameter optimization,
threshold tuning, and MLflow integration for experiment tracking.

The implementation focuses on high precision while maintaining a minimum recall threshold.
"""

import os
import sys
import json
import pickle
import random
import logging
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
import joblib
import optuna
from optuna.integration import LightGBMPruningCallback
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, precision_recall_curve
)
from datetime import datetime
import time
import gc
from pathlib import Path
from typing import Any, Dict, Tuple

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

from utils.logger import ExperimentLogger
experiment_name = "lightgbm_soccer_prediction"
logger = ExperimentLogger(experiment_name)

from utils.dynamic_sampler import DynamicTPESampler
from utils.create_evaluation_set import setup_mlflow_tracking, import_selected_features_ensemble
mlrunds_dir = setup_mlflow_tracking(experiment_name)

# Import shared utility functions
from models.StackedEnsemble.shared.hypertuner_utils import (
    predict, predict_proba, evaluate, optimize_threshold, calculate_feature_importance
)
from models.StackedEnsemble.shared.data_loader import DataLoader

# Global settings
min_recall = 0.20  # Minimum acceptable recall
n_trials = 20000  # Number of hyperparameter optimization trials as in notebook

# Base parameters as in the notebook
base_params = {
    'objective': 'binary',
    'metric': ['binary_logloss', 'auc'],
    'verbose': -1,
    'n_jobs': 4,
    'random_state': 19,
    'device': 'cpu'
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

def load_hyperparameter_space():
    """
    Define a tightened hyperparameter space for LightGBM tuning based on the
    top 10 best trials from the last hypertuning cycle. The updated ranges aim
    to increase precision by focusing on the sweet-spot regions observed.
    
    Returns:
        dict: Hyperparameter space configuration with narrowed ranges and steps.
    """
    hyperparameter_space = {
        'learning_rate': {
            'type': 'float',
            'low': 0.05,
            'high': 0.14,
            'log': False,
            'step': 0.005
        },
        'num_leaves': {
            'type': 'int',
            'low': 50,
            'high': 150,
            'log': False,
            'step': 5
        },
        'max_depth': {
            'type': 'int',
            'low': 4,
            'high': 10,
            'log': False,
            'step': 1
        },
        'min_child_samples': {
            'type': 'int',
            'low': 150,
            'high': 320,
            'log': False,
            'step': 10
        },
        'feature_fraction': {
            'type': 'float',
            'low': 0.55,
            'high': 0.70,
            'log': False,
            'step': 0.01
        },
        'bagging_fraction': {
            'type': 'float',
            'low': 0.55,
            'high': 0.65,
            'log': False,
            'step': 0.005
        },
        'bagging_freq': {
            'type': 'int',
            'low': 7,
            'high': 12,
            'log': False,
            'step': 1
        },
        'reg_alpha': {
            'type': 'float',
            'low': 0.5,
            'high': 11.0,
            'log': False,
            'step': 0.1
        },
        'reg_lambda': {
            'type': 'float',
            'low': 1.0,
            'high': 11.0,
            'log': False,
            'step': 0.1
        },
        'min_split_gain': {
            'type': 'float',
            'low': 0.10,
            'high': 0.20,
            'log': False,
            'step': 0.01
        },
        'early_stopping_rounds': {
            'type': 'int',
            'low': 300,
            'high': 700,
            'log': False,
            'step': 10
        },
        'path_smooth': {
            'type': 'float',
            'low': 0.005,
            'high': 0.60,
            'log': False,
            'step': 0.005
        },
        'cat_smooth': {
            'type': 'float',
            'low': 10.0,
            'high': 30.0,
            'log': False,
            'step': 0.1
        },
        'max_bin': {
            'type': 'int',
            'low': 200,
            'high': 700,
            'log': False,
            'step': 10
        }
    }
    return hyperparameter_space

def create_model(model_params):
    """
    Create and configure LightGBM model instance.
    Matches the notebook implementation.
    
    Args:
        model_params (dict): Model parameters
        
    Returns:
        lgb.LGBMClassifier: Configured LightGBM model
    """
    try:
        params = base_params.copy()
        
        # Update with provided parameters
        params.update(model_params)
        
        # Create model
        model = lgb.LGBMClassifier(**params)
        
        return model
        
    except Exception as e:
        logger.error(f"Error creating LightGBM model: {str(e)}")
        raise

def train_model(X_train, y_train, X_test, y_test, X_eval, y_eval, model_params):
    """
    Train a LightGBM model with early stopping and threshold optimization.
    Updated to match notebook implementation.
    
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
        # Combine training and validation data while preserving indexes
        X_combined = pd.concat([X_train, X_test], axis=0)
        y_combined = pd.concat([y_train, y_test], axis=0)
        
        # Reset indexes to ensure proper alignment
        X_combined.reset_index(drop=True, inplace=True)
        y_combined.reset_index(drop=True, inplace=True)
        # Extract early stopping rounds if present
        early_stopping_rounds = model_params.pop('early_stopping_rounds', 100)
        
        # Create model with remaining parameters
        model = create_model(model_params)
        
        # Create eval set for early stopping
        eval_set = [(X_eval, y_eval)]
        
        # Fit model with early stopping
        model.fit(
            X_combined, y_combined,
            eval_set=eval_set,
            callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds)]
        )
        
        # Get validation predictions
        best_threshold, metrics = optimize_threshold(
            model, X_eval, y_eval, min_recall=min_recall
        )
        
        return model, metrics
        
    except Exception as e:
        logger.error(f"Error training LightGBM model: {str(e)}")
        raise

def optimize_hyperparameters(X_train, y_train, X_test, y_test, X_eval, y_eval, hyperparameter_space):
    """
    Run hyperparameter optimization with Optuna.
    Added to match notebook implementation.
    
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
            
            # Add hyperparameters from config with step size if provided.
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
        # Use dynamic sampler to expand the search space after 200 trials
        sampler = DynamicTPESampler(
            dynamic_threshold=200,
            dynamic_search_space={
                "learning_rate": lambda orig: optuna.distributions.FloatDistribution(low=0.01, high=0.11, step=0.005),
                "num_leaves": lambda orig: optuna.distributions.IntDistribution(low=50, high=200, step=2),
                "max_depth": lambda orig: optuna.distributions.IntDistribution(low=5, high=10, step=1),
                "min_child_samples": lambda orig: optuna.distributions.IntDistribution(low=250, high=500, step=5),
                "feature_fraction": lambda orig: optuna.distributions.FloatDistribution(low=0.6, high=0.8, step=0.01),
                "bagging_fraction": lambda orig: optuna.distributions.FloatDistribution(low=0.4, high=0.8, step=0.01),
                "bagging_freq": lambda orig: optuna.distributions.IntDistribution(low=6, high=10, step=1),
                "reg_alpha": lambda orig: optuna.distributions.FloatDistribution(low=0.1, high=10.5, step=0.1),
                "reg_lambda": lambda orig: optuna.distributions.FloatDistribution(low=0.1, high=9.0, step=0.05),
                "min_split_gain": lambda orig: optuna.distributions.FloatDistribution(low=0.1, high=0.55, step=0.01),
                "early_stopping_rounds": lambda orig: optuna.distributions.IntDistribution(low=400, high=800, step=10),
                "path_smooth": lambda orig: optuna.distributions.FloatDistribution(low=0.005, high=0.8, step=0.005),
                "cat_smooth": lambda orig: optuna.distributions.FloatDistribution(low=1.0, high=30.0, step=0.5),
                "max_bin": lambda orig: optuna.distributions.IntDistribution(low=200, high=700, step=5)
            },
            n_startup_trials=200,
            prior_weight=0.2,
            warn_independent_sampling=False
        )
        random_sampler = optuna.samplers.RandomSampler(
            seed=19
        )
        study = optuna.create_study(
            study_name='lightgbm_optimization',
            direction='maximize',
            sampler=random_sampler
        )
        
        # Optimize
        best_score = -float('inf')  # Initialize with worst possible score
        best_params = {}
        top_trials = []

        def callback(study, trial):
            nonlocal best_score
            nonlocal best_params
            nonlocal top_trials
            logger.info(f"Current best score: {best_score:.4f}")
            if trial.value > best_score:
                best_score = trial.value
                best_params = trial.params
                logger.info(f"New best score found in trial {trial.number}: {best_score:.4f}")
            # Create a record for the current trial
            current_run = (trial.value, trial.params, trial.number)
            
            # Append the trial's record to the list
            top_trials.append(current_run)
            
            # Sort the list in descending order by score (trial.value)
            top_trials.sort(key=lambda x: x[0], reverse=True)
            
            # Keep only the top 10 best trials
            top_trials = top_trials[:10]
            # Log top trials as a table every 10 trials
            if trial.number % 9 == 0:
                table_header = "| Rank | Trial # | Score | Parameters |"
                table_separator = "|------|---------|-------|------------|"
                table_rows = [f"| {i+1} | {trial[2]} | {trial[0]:.4f} | {trial[1]} |" for i, trial in enumerate(top_trials)]
                
                logger.info("Top 10 trials:")
                logger.info(table_header)
                logger.info(table_separator)
                for row in table_rows:
                    logger.info(row)
            return best_score
        
        study.optimize(
            objective, 
            n_trials=n_trials, 
            timeout=900000,
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

def hypertune_lightgbm(experiment_name: str):
    """
    Main training function with MLflow tracking.
    Updated name from hypertune_mlp to hypertune_lightgbm to match notebook.
    
    Args:
        experiment_name (str): Experiment name for MLflow tracking
        
    Returns:
        tuple: (best_params, best_metrics)
    """
    try:
        # Start MLflow run
        with mlflow.start_run(run_name=f"lightgbm_base_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            # Set tags
            mlflow.set_tags({
                "model_type": "lightgbm_base",
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
            # Log best parameters to MLflow
            logger.info("Logging best parameters to MLflow")
            for param_name, param_value in best_params.items():
                mlflow.log_param(param_name, param_value)
            
            # Log final metrics
            mlflow.log_metrics({
                "precision": metrics.get('precision', 0.0),
                "recall": metrics.get('recall', 0.0),
                "f1": metrics.get('f1', 0.0),
                "auc": metrics.get('auc', 0.0),
                "threshold": metrics.get('threshold', 0.5)
            })
            
            # Create input example with a sample from evaluation data
            # Handle integer columns by converting them to float64 to properly manage missing values
            input_example = X_eval.iloc[:5].copy() if hasattr(X_eval, 'iloc') else X_eval[:5].copy()
            
            # Identify and convert integer columns to float64 to prevent schema enforcement errors
            if hasattr(input_example, 'dtypes'):
                for col in input_example.columns:
                    if X_eval[col].dtype.kind == 'i':
                        logger.info(f"Converting integer column '{col}' to float64 to handle potential missing values")
                        X_eval[col] = X_eval[col].astype('float64')
            
            # Infer signature with proper handling for integer columns with potential missing values
            signature = mlflow.models.infer_signature(
                input_example,
                model.predict(input_example)
            )
            mlflow.lightgbm.log_model(
                model,
                "model",
                registered_model_name=f"lightgbm_{datetime.now().strftime('%Y%m%d_%H%M')}",
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
        model: Trained LightGBM model
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
        with mlflow.start_run(run_name=f"lightgbm_{datetime.now().strftime('%Y%m%d_%H%M')}") as run:
            # Log parameters
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model
            model_info = mlflow.lightgbm.log_model(
                model,
                "model",
                registered_model_name=f"lightgbm_{datetime.now().strftime('%Y%m%d_%H%M')}"
            )
            
            # Log feature importance using the shared utility
            importance_df = calculate_feature_importance(
                model, 
                feature_names=X_train.columns if hasattr(X_train, 'columns') else None
            )
            
            if not importance_df.empty:
                # Save to CSV and log as artifact
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
    Train LightGBM model with focus on precision target.
    
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
        best_params, _ = hypertune_lightgbm(experiment_name)
        
        if not best_params:
            logger.warning("Hyperparameter tuning failed. Using default parameters.")
            best_params = base_params.copy()
            best_params.update({
                'learning_rate': 0.05,
                'num_leaves': 31,
                'max_depth': 6,
                'min_child_samples': 20,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 0,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'min_split_gain': 0.01
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
        logger.info("Starting LightGBM model training")
        
        # Import data at runtime to avoid global scope issues
        from models.StackedEnsemble.shared.data_loader import DataLoader
        
        global X_train, y_train, X_test, y_test, X_eval, y_eval
        
        # Load data
        dataloader = DataLoader()
        X_train, y_train, X_test, y_test, X_eval, y_eval = dataloader.load_data()
        features = import_selected_features_ensemble(model_type='lgbm')
        X_train = X_train[features]
        X_test = X_test[features]
        X_eval = X_eval[features]
        # Log data shapes
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Testing data shape: {X_test.shape}")
        logger.info(f"Evaluation data shape: {X_eval.shape}")
        logger.info(f"Positive class ratio - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}, Eval: {y_eval.mean():.3f}")
        
        # Hyperparameter optimization - run 3 times and select best
        logger.info("Starting hyperparameter optimization with 3 runs")
        best_overall_params = None
        best_overall_metrics = None
        
        # for run in range(1, 4):
        logger.info(f"Starting hyperparameter optimization run")
        current_params, current_metrics = hypertune_lightgbm(experiment_name)
        
        logger.info(f"Run completed with parameters: {current_params}")
        logger.info(f"Run metrics: {current_metrics}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main() 