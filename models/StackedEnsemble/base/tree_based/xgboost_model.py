"""
XGBoost Model for Soccer Draw Prediction

This module implements a XGBoost-based model for predicting soccer match draws.
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
import xgboost as xgb
import mlflow
import joblib
import optuna
from optuna.integration import XGBoostPruningCallback
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, precision_recall_curve
)
from datetime import datetime
import time
import gc
from pathlib import Path
from typing import Any, Dict, Tuple
import sklearn


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
experiment_name = "xgboost_soccer_prediction"
logger = ExperimentLogger(experiment_name)

from utils.dynamic_sampler import DynamicTPESampler
from utils.create_evaluation_set import setup_mlflow_tracking
mlrunds_dir = setup_mlflow_tracking(experiment_name)

# Import shared utility functions
from models.StackedEnsemble.shared.hypertuner_utils import (
    predict, predict_proba, evaluate, optimize_threshold, calculate_feature_importance
)
from models.StackedEnsemble.shared.data_loader import DataLoader

# Global settings
min_recall = 0.20  # Minimum acceptable recall
n_trials = 2000  # Number of hyperparameter optimization trials as in notebook
# Get current versions
xgb_version = xgb.__version__
sklearn_version = sklearn.__version__

# Create explicit pip requirements list
pip_requirements = [
    f"xgboost=={xgb_version}",
    f"scikit-learn=={sklearn_version}",
    f"mlflow=={mlflow.__version__}"
]

# Base parameters as in the notebook
base_params = {
    'objective': 'binary:logistic',
    'verbosity': 0,
    'nthread': -1,
    'seed': 19,
    'device': 'cpu',
    'tree_method': 'hist'
}

def load_hyperparameter_space():
    """
    Define a tightened hyperparameter space for XGBoost tuning based on the
    top 10 best trials from the last hypertuning cycle. The updated ranges
    focus on regions where higher precision was observed.
    
    Returns:
        dict: Hyperparameter space configuration with narrowed ranges and steps.
    """
    hyperparameter_space = {
        'learning_rate': {
            'type': 'float',
            'low': 0.005,               # narrowed based on top trials (0.06-0.09)
            'high': 0.05,
            'log': False,
            'step': 0.005
        },
        'max_depth': {
            'type': 'int',
            'low': 5,                  # narrowed based on top trials (6-7)
            'high': 10,
            'step': 1
        },
        'min_child_weight': {
            'type': 'int',
            'low': 200,                # narrowed based on top trials (~450)
            'high': 500,
            'step': 5
        },
        'colsample_bytree': {
            'type': 'float',
            'low': 0.65,                # narrowed based on top trials (0.62-0.65)
            'high': 0.90,
            'log': False,
            'step': 0.01
        },
        'subsample': {
            'type': 'float',
            'low': 0.6,                # narrowed based on top trials (0.85-0.91)
            'high': 0.95,
            'log': False,
            'step': 0.01
        },
        'gamma': {
            'type': 'float',
            'low': 0.02,                # narrowed based on top trials (0.52-1.59)
            'high': 1.0,
            'log': False,
            'step': 0.02
        },
        'lambda': {
            'type': 'float',
            'low': 1.0,                # narrowed based on top trials (7.51-8.01)
            'high': 8.0,
            'log':  False,
            'step': 0.01
        },
        'alpha': {
            'type': 'float',
            'low': 10.0,               # narrowed based on top trials (61.14-61.92)
            'high': 70.0,
            'log': False,
            'step': 0.1
        },
        'early_stopping_rounds': {
            'type': 'int',
            'low': 400,                # kept lower bound as per project rules, widened upper bound
            'high': 1200,
            'step': 20
        },
        'scale_pos_weight': {
            'type': 'float',
            'low': 2.0,                # narrowed based on top trials (1.8-2.28)
            'high': 4.5,
            'log': False,
            'step': 0.05
        }
    }
    return hyperparameter_space

def create_model(model_params):
    """
    Create and configure XGBoost model instance.
    Matches the notebook implementation.
    
    Args:
        model_params (dict): Model parameters
        
    Returns:
        xgb.XGBClassifier: Configured XGBoost model
    """
    try:
        params = base_params.copy()
        
        # Update with provided parameters
        params.update(model_params)
        
        # Create model
        model = xgb.XGBClassifier(**params)
        
        return model
        
    except Exception as e:
        logger.error(f"Error creating XGBoost model: {str(e)}")
        raise

def train_model(X_train, y_train, X_test, y_test, X_eval, y_eval, model_params):
    """
    Train a XGBoost model with early stopping and threshold optimization.
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
        
        # Create model with remaining parameters
        model = create_model(model_params)
        
        # Create eval set for early stopping
        eval_set = [(X_test, y_test)]
        # Fit model with early stopping
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        # Get validation predictions
        best_threshold, metrics = optimize_threshold(
            model, X_eval, y_eval, min_recall=min_recall
        )
        
        return model, metrics
        
    except Exception as e:
        logger.error(f"Error training XGBoost model: {str(e)}")
        raise

def save_model(model, path, threshold=0.5):
    """
    Save XGBoost model to specified path.
    Updated to match notebook implementation using joblib.
    
    Args:
        model: Trained XGBoost model
        path: Path to save model
        threshold: Optimal decision threshold
    """
    if model is None:
        raise RuntimeError("No model to save")
        
    try:
        # Create directory if it doesn't exist
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(model, path)
        
        # Save threshold
        threshold_path = path.parent / "threshold.json"
        with open(threshold_path, 'w') as f:
            json.dump({
                'threshold': threshold,
                'model_type': 'xgboost',
                'params': model.get_params()
            }, f, indent=2)
            
        logger.info(f"Model saved to {path}")
        
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

def load_model(path):
    """
    Load XGBoost model from specified path.
    Updated to match notebook implementation using joblib.
    
    Args:
        path: Path to load model from
        
    Returns:
        tuple: (model, threshold)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No model file found at {path}")
        
    try:
        # Load model
        model = joblib.load(path)
        
        # Load threshold
        threshold_path = path.parent / "threshold.json"
        if threshold_path.exists():
            with open(threshold_path, 'r') as f:
                data = json.load(f)
                threshold = data.get('threshold', 0.5)
        else:
            threshold = 0.5
            
        logger.info(f"Model loaded from {path}")
        return model, threshold
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
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
            
            for metric_name, metric_value in metrics.items():
                trial.set_user_attr(metric_name, metric_value)
            return score
            
        except Exception as e:
            logger.error(f"Trial failed: {str(e)}")
            return 0.0
    
    try:
        # Use dynamic sampler to expand the search space after 200 trials
        # sampler = DynamicTPESampler(
        #     dynamic_threshold=200,
        #     dynamic_search_space={
        #         "learning_rate": lambda orig: optuna.distributions.FloatDistribution(low=0.005, high=0.05, step=0.005),
        #         "max_depth": lambda orig: optuna.distributions.IntDistribution(low=5, high=10, step=1),
        #         "min_child_weight": lambda orig: optuna.distributions.IntDistribution(low=200, high=500, step=5),
        #         "colsample_bytree": lambda orig: optuna.distributions.FloatDistribution(low=0.60, high=0.90, step=0.01),
        #         "subsample": lambda orig: optuna.distributions.FloatDistribution(low=0.55, high=0.95, step=0.01),
        #         "gamma": lambda orig: optuna.distributions.FloatDistribution(low=0.02, high=1.5, step=0.02),
        #         "lambda": lambda orig: optuna.distributions.FloatDistribution(low=1.0, high=8.0, step=0.01),
        #         "alpha": lambda orig: optuna.distributions.FloatDistribution(low=10.0, high=70.0, step=0.1),
        #         "early_stopping_rounds": lambda orig: optuna.distributions.IntDistribution(low=400, high=1200, step=20),
        #         "scale_pos_weight": lambda orig: optuna.distributions.FloatDistribution(low=2.0, high=5.0, step=0.05)
        #     },
        #     n_startup_trials=200,
        #     prior_weight=0.2,
        #     warn_independent_sampling=False
        # )
        cmaes_sampler = optuna.samplers.CmaEsSampler(
            x0={'learning_rate': 0.025, 'max_depth': 7, 'min_child_weight': 350,
                'colsample_bytree': 0.75, 'subsample': 0.75, 'gamma': 0.5,
                'lambda': 4.0, 'alpha': 40.0, 'early_stopping_rounds': 800,
                'scale_pos_weight': 3.5},
            sigma0=0.1,
            seed=42,
            n_startup_trials=2000
        )
<<<<<<< HEAD
        random_sampler = optuna.samplers.RandomSampler(
            seed=19
        )
        study = optuna.create_study(
            study_name='xgboost_optimization',
            direction='maximize',
            sampler=random_sampler
=======
        study = optuna.create_study(
            study_name='xgboost_optimization',
            direction='maximize',
            sampler=cmaes_sampler
>>>>>>> 3798e304ba09a95ae05e21747b9f93b4e52eb5fd
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
            timeout=10000,  # 10000 seconds as in notebook
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

def hypertune_xgboost(experiment_name: str):
    """
    Main training function with MLflow tracking.
    Updated name from hypertune_mlp to hypertune_xgboost to match notebook.
    
    Args:
        experiment_name (str): Experiment name for MLflow tracking
        
    Returns:
        tuple: (best_params, best_metrics)
    """
    try:
        # Start MLflow run
        with mlflow.start_run(run_name=f"xgboost_base_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            # Set tags
            mlflow.set_tags({
                "model_type": "xgboost_base",
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
            # Log best parameters to MLflow
            logger.info("Logging best parameters to MLflow")
            for param_name, param_value in best_params.items():
                mlflow.log_param(param_name, param_value)
                
            # Infer signature with proper handling for integer columns with potential missing values
            signature = mlflow.models.infer_signature(
                input_example,
                model.predict(input_example)
            )
            
            # Log warning about integer columns in signature
            logger.info("Model signature created - check logs for any warnings about integer columns")
            # When saving model, explicitly specify requirements
            mlflow.xgboost.log_model(
                model,
                "model",
                pip_requirements=pip_requirements,  # Explicitly set requirements
                registered_model_name=f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M')}",
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
        model: Trained XGBoost model
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
        with mlflow.start_run(run_name=f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M')}") as run:
            # Log parameters
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model
            model_info = mlflow.xgboost.log_model(
                model,
                "model",
                registered_model_name=f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M')}"
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
    Train XGBoost model with focus on precision target.
    
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
        best_params, _ = hypertune_xgboost(experiment_name)
        
        if not best_params:
            logger.warning("Hyperparameter tuning failed. Using default parameters.")
            best_params = base_params.copy()
            best_params.update({
                'learning_rate': 0.05,
                'max_depth': 6,
                'min_child_weight': 20,
                'colsample_bytree': 0.8,
                'subsample': 0.8,
                'alpha': 0.1,
                'lambda': 0.1,
                'gamma': 0.01
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
        
        # Save model locally
        model_path = f"models/StackedEnsemble/base/tree_based/xgboost_model_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
        save_model(model, model_path, metrics.get('threshold', 0.5))
        
        return model, metrics
        
    except Exception as e:
        logger.error(f"Error in precision-focused training: {str(e)}")
        return None, None

def main():
    """
    Main execution function.
    """
    try:
        logger.info("Starting XGBoost model training")
        
        # Import data at runtime to avoid global scope issues
        from models.StackedEnsemble.shared.data_loader import DataLoader
        
        global X_train, y_train, X_test, y_test, X_eval, y_eval
        
        # Load data
        dataloader = DataLoader()
        X_train, y_train, X_test, y_test, X_eval, y_eval = dataloader.load_data()
        
        # Log data shapes
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Testing data shape: {X_test.shape}")
        logger.info(f"Evaluation data shape: {X_eval.shape}")
        logger.info(f"Positive class ratio - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}, Eval: {y_eval.mean():.3f}")
        
        # Update base parameters to use aucpr and custom precision metric
        base_params['eval_metric'] =  ['aucpr', 'error', 'logloss']
        
        logger.info(f"Current base parameters: {base_params}")
        
        current_params, current_metrics = hypertune_xgboost(experiment_name)
        
        logger.info(f"Run completed with parameters: {current_params}")
        logger.info(f"Run metrics: {current_metrics}")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main() 