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
from sklearn.ensemble import RandomForestClassifier


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
experiment_name = "random_forest_soccer_prediction"
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
n_trials = 100  # Number of hyperparameter optimization trials as in notebook
# Get current versions
sklearn_version = sklearn.__version__

# Create explicit pip requirements list
pip_requirements = [
    f"scikit-learn=={sklearn_version}",
    f"mlflow=={mlflow.__version__}"
]

# Update base parameters for RandomForest
base_params = {
    'random_state': 19,
    'n_jobs': 4,
    'verbose': 0
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
    Define hyperparameter space for RandomForest tuning.
    """
    hyperparameter_space = {
        'n_estimators': {
            'type': 'int',
            'low': 100,
            'high': 1000,
            'step': 10
        },
        'max_depth': {
            'type': 'int',
            'low': 4,
            'high': 15,
            'step': 1
        },
        'min_samples_split': {
            'type': 'int',
            'low': 2,
            'high': 40,
            'step': 2
        },
        'min_samples_leaf': {
            'type': 'int',
            'low': 2,
            'high': 40,
            'step': 2
        },
        'max_features': {
            'type': 'float',
            'low': 0.1,
            'high': 1.0,
            'step': 0.02
        },
        'class_weight': {
            'type': 'float',
            'low': 1.8,
            'high': 5.0,
            'step': 0.1
        }
        # 'ccp_alpha': {
        #     'type': 'float',
        #     'low': 0.002,
        #     'high': 0.2,
        #     'step': 0.002
        # }
    }
    return hyperparameter_space

def create_model(model_params):
    """
    Create and configure RandomForest model instance.
    
    Args:
        model_params (dict): Model parameters
        
    Returns:
        RandomForestClassifier: Configured RandomForest model
    """
    try:
        params = base_params.copy()
        params.update(model_params)
        
        # Convert class_weight parameter to dictionary format
        if 'class_weight' in params:
            class_weight_value = params.pop('class_weight')
            params['class_weight'] = {0: 1.0, 1: class_weight_value}
        
        model = RandomForestClassifier(**params)
        return model
        
    except Exception as e:
        logger.error(f"Error creating RandomForest model: {str(e)}")
        raise

def train_model(X_train, y_train, X_test, y_test, X_eval, y_eval, model_params):
    """
    Train a RandomForest model and optimize threshold.
    
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
        model = create_model(model_params)
        
        # Combine training and validation data
        X_combined = pd.concat([X_train, X_test], axis=0)
        y_combined = pd.concat([y_train, y_test], axis=0)
        
        # Reset indexes
        X_combined.reset_index(drop=True, inplace=True)
        y_combined.reset_index(drop=True, inplace=True)
        # Fit model
        model.fit(X_combined, y_combined)
        
        # Get validation predictions and optimize threshold
        best_threshold, metrics = optimize_threshold(
            model, X_eval, y_eval, min_recall=min_recall
        )
        
        return model, metrics
        
    except Exception as e:
        logger.error(f"Error training RandomForest model: {str(e)}")
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
        random_sampler = optuna.samplers.RandomSampler(
            seed=19
        )
        study = optuna.create_study(
            study_name='xgboost_optimization',
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

def hypertune_random_forest(experiment_name: str):
    """
    Main training function with MLflow tracking.
    """
    try:
        with mlflow.start_run(run_name=f"rf_base_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            mlflow.set_tags({
                "model_type": "random_forest_base",
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
            # When saving model, use sklearn instead of xgboost
            mlflow.sklearn.log_model(
                model,
                "model",
                pip_requirements=pip_requirements,
                registered_model_name=f"rf_{datetime.now().strftime('%Y%m%d_%H%M')}",
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
        model: Trained RandomForest model
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
        with mlflow.start_run(run_name=f"rf_final_{datetime.now().strftime('%Y%m%d_%H%M')}") as run:
            # Log parameters
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model
            model_info = mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=f"rf_{datetime.now().strftime('%Y%m%d_%H%M')}"
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
    Train RandomForest model with focus on precision target.
    
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
        best_params, _ = hypertune_random_forest(experiment_name)
        
        if not best_params:
            logger.warning("Hyperparameter tuning failed. Using default parameters.")
            best_params = base_params.copy()
            best_params.update({
                'n_estimators': 500,
                'max_depth': 20,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 0.5,
                'class_weight': 2.0
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
        logger.info("Starting RandomForest model training")
        
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
        
        current_params, current_metrics = hypertune_random_forest(experiment_name)
        
        logger.info(f"Run completed with parameters: {current_params}")
        logger.info(f"Run metrics: {current_metrics}")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main() 