"""
RUSBoost Model for Soccer Draw Prediction

This module implements a RUSBoost-based model for predicting soccer match draws.
It includes functionality for model creation, training, hyperparameter optimization,
threshold tuning, and MLflow integration for experiment tracking.

The implementation focuses on achieving high precision while maintaining a minimum recall threshold,
especially useful when dealing with imbalanced classes such as soccer draws.
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
from datetime import datetime
import time
import gc
from pathlib import Path
from typing import Any, Dict, Tuple
import optuna
import mlflow
# Import RUSBoostClassifier and DecisionTreeClassifier
from imblearn.ensemble import RUSBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Add project root to Python path
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

from utils.logger import ExperimentLogger
experiment_name = "rusboost_soccer_prediction"
logger = ExperimentLogger(experiment_name)
from utils.create_evaluation_set import setup_mlflow_tracking
mlrunds_dir = setup_mlflow_tracking(experiment_name)

# Import shared utility functions
from models.StackedEnsemble.shared.hypertuner_utils import (
    predict, predict_proba, evaluate, optimize_threshold, calculate_feature_importance
)
from models.StackedEnsemble.shared.data_loader import DataLoader    

# Global settings
min_recall = 0.20  # Minimum acceptable recall
n_trials = 2000   # Number of hyperparameter optimization trials

# Base parameters for RUSBoost
base_params = {
    'random_state': 19,
    # 'eval_metric': ['aucpr', 'error', 'logloss'],
}


def load_hyperparameter_space():
    """
    Define a tightened hyperparameter space for RUSBoost tuning based on the
    top trials. The ranges are chosen to focus on regions that may lead to
    higher precision while ensuring decent recall.
    
    Returns:
        dict: Hyperparameter space configuration with narrowed ranges and steps.
    """
    hyperparameter_space = {
        'learning_rate': {
            'type': 'float',
            'low': 0.002,
            'high': 0.02,
            'log': False,
            'step': 0.002
        },
        'n_estimators': {
            'type': 'int',
            'low': 100,
            'high': 600,
            'step': 10
        },
        # Base estimator (DecisionTreeClassifier) parameters
        'max_depth': {
            'type': 'int',
            'low': 3,
            'high': 10,
            'step': 1
        },
        'min_samples_split': {
            'type': 'int',
            'low': 2,
            'high': 20,
            'step': 1
        },
        'min_samples_leaf': {
            'type': 'int',
            'low': 2,
            'high': 15,
            'step': 1
        },
        # Optional parameter for undersampling strategy if supported
        'sampling_strategy': {
            'type': 'float',
            'low': 0.30,
            'high': 0.6,
            'log': False,
            'step': 0.02
        },
        'criterion': {
            'type': 'categorical',
            'choices': ['gini', 'entropy']
        },
        'max_features': {
            'type': 'categorical',
            'choices': [None, 'sqrt', 'log2']
        }
    }
    return hyperparameter_space

def create_model(model_params):
    """
    Create and configure a RUSBoost model instance using the provided parameters.
    Args:
        model_params (dict): Model parameters including both RUSBoost and base estimator params.
        
    Returns:
        RUSBoostClassifier: Configured RUSBoost model.
    """
    try:
        # Copy base parameters and update with hyperparameter tuning results
        params = base_params.copy()
        params.update(model_params)
        
        # Extract base estimator parameters
        base_estimator_params = {}
        for key in ['max_depth', 'min_samples_split', 'min_samples_leaf', 'criterion', 'max_features']:
            if key in params:
                base_estimator_params[key] = params.pop(key)
        
        # Create base estimator with extracted parameters
        base_estimator = DecisionTreeClassifier(**base_estimator_params, random_state=base_params['random_state'])
        logger.info(f"Base estimator parameters: {base_estimator_params}")
        # Create RUSBoostClassifier with the remaining parameters
        # Note: RUSBoostClassifier doesn't accept base_estimator as a parameter
        model = RUSBoostClassifier(estimator=base_estimator, algorithm='SAMME', **params)
        logger.info(f"RUSBoostClassifier parameters: {params}")
        
        return model
    except Exception as e:
        logger.error(f"Error creating RUSBoost model: {str(e)}")
        raise

def train_model(X_train, y_train, X_test, y_test, X_eval, y_eval, model_params):
    """
    Train a RUSBoost model with early stopping and threshold optimization.
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Validation features
        y_test: Validation labels
        X_eval: Evaluation features
        y_eval: Evaluation labels
        model_params: Model parameters for RUSBoost and base estimator.
        
    Returns:
        tuple: (trained_model, metrics)
    """
    try:
        # Create model using the provided parameters
        model = create_model(model_params)
        
        # For early stopping simulation, we use a validation set and perform threshold optimization after training.
        # Note: RUSBoost (via imblearn) does not natively support early stopping like XGBoost.
        
        model.fit(X_train, y_train)
        
        # Optimize threshold based on evaluation set to maximize precision while ensuring min_recall
        best_threshold, metrics = optimize_threshold(model, X_eval, y_eval, min_recall=min_recall)
        
        return model, metrics
    except Exception as e:
        logger.error(f"Error training RUSBoost model: {str(e)}")
        raise

def optimize_hyperparameters(X_train, y_train, X_test, y_test, X_eval, y_eval, hyperparameter_space):
    """
    Run hyperparameter optimization with Optuna for RUSBoost model.
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Validation features
        y_test: Validation labels
        X_eval: Evaluation features
        y_eval: Evaluation labels
        hyperparameter_space: Hyperparameter space configuration
        
    Returns:
        dict: Best parameters found
    """
    logger.info("Starting hyperparameter optimization for RUSBoost")
    if not hyperparameter_space:
        hyperparameter_space = load_hyperparameter_space()
    best_score = 0.0

    def objective(trial):
        try:
            params = base_params.copy()
            
            # Add hyperparameters from config
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

            for metric_name, metric_value in metrics.items():
                trial.set_user_attr(metric_name, metric_value)
            return score
        except Exception as e:
            logger.error(f"Trial failed: {str(e)}")
            return 0.0
    try:
        study = optuna.create_study(
            study_name='rusboost_optimization',
            direction='maximize',
            sampler=optuna.samplers.TPESampler(n_startup_trials=300, prior_weight=0.3)
        )

        best_score = -float('inf')
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
            current_run = (trial.value, trial.params, trial.number)
            top_trials.append(current_run)
            top_trials.sort(key=lambda x: x[0], reverse=True)
            top_trials = top_trials[:10]
            if trial.number % 9 == 0:
                table_header = "| Rank | Trial # | Score | Parameters |"
                table_separator = "|------|---------|-------|------------|"
                table_rows = [f"| {i+1} | {t[2]} | {t[0]:.4f} | {t[1]} |" for i, t in enumerate(top_trials)]
                logger.info("Top 10 trials:")
                logger.info(table_header)
                logger.info(table_separator)
                for row in table_rows:
                    logger.info(row)
            return best_score
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=4,
            timeout=10000,
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


def hypertune_rusboost(experiment_name: str):
    """
    Main training function with MLflow tracking for RUSBoost.
    Args:
        experiment_name (str): Experiment name for MLflow tracking
        
    Returns:
        tuple: (best_params, best_metrics)
    """
    try:
        with mlflow.start_run(run_name=f"rusboost_base_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            mlflow.set_tags({
                "model_type": "rusboost_base",
                "training_mode": "global",
                "cpu_only": True
            })
            hyperparameter_space = load_hyperparameter_space()
            logger.info("Starting hyperparameter optimization")
            best_params = optimize_hyperparameters(
                X_train, y_train,
                X_test, y_test,
                X_eval, y_eval,
                hyperparameter_space=hyperparameter_space
            )
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
            
            # Log metrics
            mlflow.log_metrics({
                "precision": metrics.get('precision', 0.0),
                "recall": metrics.get('recall', 0.0),
                "f1": metrics.get('f1', 0.0),
                "aucpr": metrics.get('aucpr', 0.0),
                "threshold": metrics.get('threshold', 0.5)
            })
            
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
            
            # When saving model, explicitly specify requirements
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=f"rusboost_{datetime.now().strftime('%Y%m%d_%H%M')}",
                signature=signature
            )
            return best_params, metrics
    except Exception as e:
        logger.error(f"Error in hyperparameter tuning: {str(e)}")
        return None, None


def log_to_mlflow(model, metrics, params, experiment_name):
    """
    Log trained model, metrics, and parameters to MLflow for RUSBoost.

    Args:
        model: Trained RUSBoost model
        metrics: Model evaluation metrics
        params: Model parameters
        experiment_name: Experiment name
        
    Returns:
        str: Run ID
    """
    try:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=f"rusboost_{datetime.now().strftime('%Y%m%d_%H%M')}") as run:
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            model_info = mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=f"rusboost_{datetime.now().strftime('%Y%m%d_%H%M')}",
                signature=mlflow.models.infer_signature(
                    X_eval.iloc[:5] if hasattr(X_eval, 'iloc') else X_eval[:5],
                    model.predict(X_eval.iloc[:5] if hasattr(X_eval, 'iloc') else X_eval[:5])
                )
            )
            logger.info(f"Model logged to MLflow: {model_info.model_uri}")
            logger.info(f"Run ID: {run.info.run_id}")
            return run.info.run_id
    except Exception as e:
        logger.error(f"Error logging to MLflow: {str(e)}")
        return None


def train_with_precision_target(X_train, y_train, X_test, y_test, X_eval, y_eval):
    """
    Train RUSBoost model with a focus on precision target.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Testing features
        y_test: Testing labels
        X_eval: Evaluation features
        y_eval: Evaluation labels
        
    Returns:
        tuple: (trained_model, metrics)
    """
    try:
        logger.info("Running hyperparameter tuning")
        best_params, _ = hypertune_rusboost(experiment_name)
        if not best_params:
            logger.warning("Hyperparameter tuning failed. Using default parameters.")
            best_params = base_params.copy()
            best_params.update({
                'learning_rate': 0.1,
                'n_estimators': 100,
                'max_depth': 5,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'sampling_strategy': 1.0
            })
        logger.info("Training final model with best parameters")
        model, metrics = train_model(
            X_train, y_train,
            X_test, y_test,
            X_eval, y_eval,
            best_params
        )
        log_to_mlflow(model, metrics, best_params, experiment_name)
        model_path = f"models/StackedEnsemble/base/tree_based/rusboost_model_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Model saved locally at {model_path}")
        return model, metrics
    except Exception as e:
        logger.error(f"Error in precision-focused training: {str(e)}")
        return None, None


def main():
    """
    Main execution function for training RUSBoost model.
    """
    try:
        logger.info("Starting RUSBoost model training")
        global X_train, y_train, X_test, y_test, X_eval, y_eval
        dataloader = DataLoader()
        X_train, y_train, X_test, y_test, X_eval, y_eval = dataloader.load_data()
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Testing data shape: {X_test.shape}")
        logger.info(f"Evaluation data shape: {X_eval.shape}")
        logger.info(f"Positive class ratio - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}, Eval: {y_eval.mean():.3f}")
        train_with_precision_target(X_train, y_train, X_test, y_test, X_eval, y_eval)
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")


if __name__ == "__main__":
    main()
