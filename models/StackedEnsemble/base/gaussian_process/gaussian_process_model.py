"""
Random Fourier Features (RFF) Pipeline Model for Soccer Draw Prediction

This module implements a pipeline that uses Random Fourier Features (RBFSampler)
to approximate the RBF kernel, followed by LogisticRegression for predicting
soccer match draws. It includes functionality for model creation, training,
hyperparameter optimization, threshold tuning, and MLflow integration for experiment tracking.

The implementation focuses on high precision while maintaining a minimum recall threshold.
"""

import os
import sys
import json
import random
import logging
import warnings
import time
import gc
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import joblib
import optuna
import mlflow
import mlflow.sklearn

from sklearn.pipeline import make_pipeline
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

import sklearn  # to get the version

# Set project root similar to other modules
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
experiment_name = "gaussian_process_soccer_prediction"
logger = ExperimentLogger(experiment_name=experiment_name)

from utils.create_evaluation_set import setup_mlflow_tracking
mlruns_dir = setup_mlflow_tracking(experiment_name)

from models.StackedEnsemble.shared.data_loader import DataLoader

# Global settings
min_recall = 0.20  # Minimum acceptable recall value
n_trials = 200  # Number of trials for hyperparameter optimization

# Get current package versions
sklearn_version = sklearn.__version__
mlflow_version = mlflow.__version__

# Create explicit pip requirements list
pip_requirements = [
    f"scikit-learn=={sklearn_version}",
    f"mlflow=={mlflow_version}"
]

# Base parameters for the RFF pipeline.
base_params = {
    'random_state': 19,
    'n_jobs': -1,
    'verbose': False
}

def load_hyperparameter_space() -> Dict:
    """
    Define an extended hyperparameter space for the RFF-based pipeline tuning.
    
    Returns:
        dict: Updated hyperparameter space configuration.
    """
    hyperparameter_space = {
        'length_scale': {
            'type': 'float',
            'low': 0.01,
            'high': 20.0,
            'log': True
        },
        'n_components': {
            'type': 'int',
            'low': 100,
            'high': 1000,
            'step': 100
        },
        'C': {
            'type': 'float',
            'low': 0.01,
            'high': 100.0,
            'log': True
        },
        'penalty': {
            'type': 'categorical',
            'choices': ['l2', 'elasticnet', 'l1']
        },
        'solver': {
            'type': 'categorical',
            'choices': ['lbfgs', 'liblinear', 'saga']
        },
        'tol': {
            'type': 'float',
            'low': 1e-5,
            'high': 1e-2,
            'log': True
        },
        'max_iter': {
            'type': 'int',
            'low': 500,
            'high': 2000,
            'step': 100
        },
        'l1_ratio': {
            'type': 'float',
            'low': 0.0,
            'high': 1.0,
            'step': 0.1
        },
        'l2_ratio': {
            'type': 'float',
            'low': 0.1,
            'high': 1.0,
            'step': 0.1
        }
    }
    return hyperparameter_space

def create_model(model_params: Dict):
    """
    Create and configure a pipeline model using RBFSampler for kernel approximation
    and LogisticRegression for classification.
    
    Args:
        model_params (dict): Model parameters including 'length_scale', 'n_components', and 'C'.
    
    Returns:
        Pipeline: Configured pipeline model.
    """
    try:
        # Retrieve RFF and classifier hyperparameters
        length_scale = model_params.get('length_scale', 1.0)
        n_components = model_params.get('n_components', 500)
        C = model_params.get('C', 1.0)
        penalty = model_params.get('penalty', 'l2')
        solver = model_params.get('solver', 'lbfgs')
        tol = model_params.get('tol', 1e-4)
        max_iter = model_params.get('max_iter', 1000)
        l1_ratio = model_params.get('l1_ratio', 0.0)
        l2_ratio = model_params.get('l2_ratio', 0.1)
        # Compute gamma for RBFSampler: gamma = 1/(2*length_scale^2)
        gamma = 1.0 / (2 * (length_scale ** 2))
        # Initialize RBFSampler
        rbf_sampler = RBFSampler(gamma=gamma, n_components=n_components, random_state=19)
        # Initialize LogisticRegression classifier
        lr_params = {
            'random_state': 19,
            'max_iter': max_iter,
            'C': C,
            'penalty': penalty,
            'solver': solver,
            'tol': tol,
            'n_jobs': base_params.get('n_jobs', -1)
        }
        # Only add l1_ratio for elasticnet penalty
        if penalty == 'elasticnet':
            lr_params['l1_ratio'] = l1_ratio
        clf = LogisticRegression(**lr_params)
        # Create a pipeline: RFF transform followed by classifier
        model = make_pipeline(rbf_sampler, clf)
        return model
    except Exception as e:
        logger.error(f"Error creating RFF-based model: {str(e)}")
        raise

def optimize_threshold(model, 
                        X_val: np.ndarray, 
                        y_val: np.ndarray, 
                        min_threshold: float = 0.1, 
                        max_threshold: float = 0.9, 
                        step: float = 0.01) -> Tuple[float, Dict]:
    """
    Optimize decision threshold on evaluation data to maximize precision while ensuring
    recall >= min_recall.
    
    Args:
        model: Trained model with predict_proba method.
        X_val: Evaluation features.
        y_val: Evaluation labels.
        min_threshold: Minimum threshold to test.
        max_threshold: Maximum threshold to test.
        step: Threshold increment.
    
    Returns:
        tuple: (best_threshold, metrics dictionary)
    """
    try:
        y_prob = model.predict_proba(X_val)[:, 1]
        thresholds = np.arange(min_threshold, max_threshold + step, step)
        best_thresh = 0.5
        best_precision = 0.0
        best_metrics = {}
        
        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            prec = precision_score(y_val, y_pred, zero_division=0)
            rec = recall_score(y_val, y_pred, zero_division=0)
            if rec >= min_recall and prec > best_precision:
                best_precision = prec
                best_thresh = thresh
                best_metrics = {
                    'precision': prec,
                    'recall': rec,
                    'f1': f1_score(y_val, y_pred, zero_division=0),
                    'auc': roc_auc_score(y_val, y_prob)
                }
        logger.info(f"Optimized threshold: {best_thresh:.3f} with precision: {best_precision:.3f}")
        return best_thresh, best_metrics
    except Exception as e:
        logger.error(f"Error optimizing threshold: {str(e)}")
        return 0.5, {}

def train_model(X_train: np.ndarray, 
                y_train: np.ndarray, 
                X_test: np.ndarray, 
                y_test: np.ndarray, 
                X_eval: np.ndarray, 
                y_eval: np.ndarray, 
                model_params: Dict) -> Tuple[Any, Dict]:
    """
    Train the RFF-based pipeline model and optimize the decision threshold on evaluation data.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: (Unused) Test features.
        y_test: (Unused) Test labels.
        X_eval: Evaluation features.
        y_eval: Evaluation labels.
        model_params: Hyperparameters for the model.
        
    Returns:
        tuple: (trained model, metrics dictionary)
    """
    try:
        model = create_model(model_params)
        model.fit(X_train, y_train)
        best_thresh, metrics = optimize_threshold(model, X_eval, y_eval)
        metrics['threshold'] = best_thresh
        return model, metrics
    except Exception as e:
        logger.error(f"Error training RFF-based model: {str(e)}")
        raise

def optimize_hyperparameters(X_train: pd.DataFrame, y_train: pd.Series,
                                X_test: pd.DataFrame, y_test: pd.Series,
                                X_eval: pd.DataFrame, y_eval: pd.Series,
                                hyperparameter_space: Dict) -> Dict:
    """
    Optimize hyperparameters for the RFF-based pipeline model using Optuna.
    
    Args:
        X_train, y_train: Training data.
        X_test, y_test: Testing data.
        X_eval, y_eval: Evaluation data.
        hyperparameter_space: Configuration for the hyperparameter space.
        
    Returns:
        dict: Best hyperparameters.
    """
    logger.info("Starting hyperparameter optimization for RFF-based model")
    
    if not hyperparameter_space:
        hyperparameter_space = load_hyperparameter_space()
    
    def objective(trial):
        try:
            params = {}
            for param_name, param_config in hyperparameter_space.items():
                if param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        step=param_config.get('step', 1)
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config['choices']
                    )
            # Handle elasticnet penalty case
            if 'penalty' in params and params['penalty'] == 'elasticnet':
                if 'l1_ratio' not in params:
                    params['l1_ratio'] = trial.suggest_float(
                        'l1_ratio',
                        0.0,
                        1.0,
                        step=0.1
                    )
                # Ensure solver is set to 'saga' for elasticnet penalty
                if 'solver' in params and params['solver'] != 'saga':
                    # logger.warning("Forcing solver to 'saga' for elasticnet penalty")
                    params['solver'] = 'saga'

            params.update(base_params)
            model, metrics = train_model(X_train, y_train, X_test, y_test, X_eval, y_eval, params)
            prec = metrics.get('precision', 0.0)
            rec = metrics.get('recall', 0.0)
            score = prec if rec >= min_recall else 0.0
            for metric_name, metric_value in metrics.items():
                trial.set_user_attr(metric_name, metric_value)
            return score
        except Exception as e:
            logger.error(f"Trial failed: {str(e)}")
            return 0.0
    
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.RandomSampler(seed=19)
    )
    study.optimize(objective, n_trials=n_trials, timeout=10000, show_progress_bar=True)
    
    best_params = study.best_trial.params
    logger.info(f"Best hyperparameters: {best_params}")
    best_params.update(base_params)
    return best_params

def hypertune_gaussian(experiment_name: str) -> Tuple[Dict, Dict]:
    """
    Run hyperparameter tuning and final training for the RFF-based pipeline model
    with MLflow tracking.
    
    Args:
        experiment_name: Experiment name for MLflow.
        
    Returns:
        tuple: (best hyperparameters, final metrics)
    """
    try:
        with mlflow.start_run(run_name=f"gp_base_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            mlflow.set_tags({
                "model_type": "rff_based",
                "training_mode": "global",
                "cpu_only": True
            })
            dataloader = DataLoader()
            X_train, y_train, X_test, y_test, X_eval, y_eval = dataloader.load_data()
            # Log basic data info
            mlflow.log_params({
                "train_size": len(X_train),
                "test_size": len(X_test),
                "eval_size": len(X_eval),
                "positive_rate_train": float(y_train.mean())
            })
            hyperparameter_space = load_hyperparameter_space()
            best_params = optimize_hyperparameters(X_train, y_train, X_test, y_test, X_eval, y_eval, hyperparameter_space)
            logger.info("Training final RFF-based model with best hyperparameters")
            model, metrics = train_model(X_train, y_train, X_test, y_test, X_eval, y_eval, best_params)
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            for param_name, param_value in best_params.items():
                mlflow.log_param(param_name, param_value)
            # Create an input example for model signature inference
            input_example = X_eval.iloc[:5].copy() if hasattr(X_eval, 'iloc') else X_eval[:5].copy()
            if hasattr(input_example, 'dtypes'):
                for col in input_example.columns:
                    if input_example[col].dtype.kind == 'i':
                        logger.info(f"Converting integer column '{col}' to float64 to handle potential missing values")
                        input_example[col] = input_example[col].astype('float64')
            signature = mlflow.models.infer_signature(input_example, model.predict(input_example))
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                pip_requirements=pip_requirements,
                registered_model_name=f"rff_{datetime.now().strftime('%Y%m%d_%H%M')}",
                signature=signature
            )
            mlflow.end_run()
            return best_params, metrics
    except Exception as e:
        logger.error(f"Error during hypertuning: {str(e)}")
        return {}, {}

def log_to_mlflow(model, metrics: Dict, params: Dict, experiment_name: str) -> str:
    """
    Log the final RFF-based model, its metrics, and parameters to MLflow.
    
    Args:
        model: Trained pipeline model.
        metrics: Evaluation metrics.
        params: Hyperparameters.
        experiment_name: MLflow experiment name.
        
    Returns:
        str: Run ID.
    """
    try:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=f"rff_final_{datetime.now().strftime('%Y%m%d_%H%M')}") as run:
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            sample_input = np.zeros((1, X_train.shape[1]))
            signature = mlflow.models.infer_signature(sample_input, model.predict(sample_input))
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                pip_requirements=pip_requirements,
                registered_model_name=f"rff_{datetime.now().strftime('%Y%m%d_%H%M')}",
                signature=signature
            )
            mlflow.end_run()
            run_id = run.info.run_id
            logger.info(f"MLflow run ID: {run_id}")
            return run_id
    except Exception as e:
        logger.error(f"Error logging to MLflow: {str(e)}")
        return ""

def main():
    """
    Main execution function for RFF-based pipeline hypertuning.
    """
    try:
        logger.info("Starting RFF-based model hypertuning")
        global X_train, y_train, X_test, y_test, X_eval, y_eval
        dataloader = DataLoader()
        X_train, y_train, X_test, y_test, X_eval, y_eval = dataloader.load_data()
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Testing data shape: {X_test.shape}")
        logger.info(f"Evaluation data shape: {X_eval.shape}")
        best_params, metrics = hypertune_gaussian(experiment_name)
        logger.info(f"Hypertuning completed. Best parameters: {best_params}")
        logger.info(f"Evaluation metrics: {metrics}")
        model, _ = train_model(X_train, y_train, X_test, y_test, X_eval, y_eval, best_params)
        logger.info("RFF-based model training completed successfully.")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main() 