import os
import sys
import json
import pickle
import random
import logging
import warnings
import numpy as np
import pandas as pd
import joblib
import optuna
import mlflow
from datetime import datetime
import time
import gc
from pathlib import Path
import sklearn

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

experiment_name = "bayesian_soccer_prediction"
logger = ExperimentLogger(experiment_name)

from utils.create_evaluation_set import setup_mlflow_tracking
mlruns_dir = setup_mlflow_tracking(experiment_name)

# Import shared utility functions
from models.StackedEnsemble.shared.hypertuner_utils import (
    predict, predict_proba, evaluate, calculate_feature_importance
)
from models.StackedEnsemble.shared.data_loader import DataLoader
from models.ensemble.thresholds import tune_threshold_for_precision

# Import BayesianMetaLearner
from models.ensemble.bayesian_meta_learner import BayesianMetaLearner, optimize_threshold

# Global settings
min_recall = 0.20  # Minimum acceptable recall
n_trials = 2000   # Number of hyperparameter optimization trials

# Get current versions
sklearn_version = sklearn.__version__

pip_requirements = [
    f"scikit-learn=={sklearn_version}",
    f"mlflow=={mlflow.__version__}"
]

# Base parameters for Bayesian classifier
base_params = {
    'random_state': 19
}


def load_hyperparameter_space():
    """
    Define a tightened hyperparameter space for Bayesian classifier tuning.
    Returns:
        dict: Hyperparameter space configuration with narrowed ranges.
    """
    hyperparameter_space = {
        'alpha': {
            'type': 'float',
            'low': 1e-6,
            'high': 1e-3,
            'log': True
        },
        'n_iter': {
            'type': 'int',
            'low': 1000,
            'high': 5000,
            'step': 500
        },
        'tol': {
            'type': 'float',
            'low': 1e-6,
            'high': 1e-4,
            'log': True
        },
        'fit_intercept': {
            'type': 'categorical',
            'choices': [True, False]
        },
        'normalize': {
            'type': 'categorical',
            'choices': [True, False]
        },
        'class_weight': {
            'type': 'categorical',
            'choices': [None, 'balanced']
        },
        'learning_rate': {
            'type': 'categorical',
            'choices': ['adaptive', 'constant', 'optimal']
        },
        'eta0': {
            'type': 'float',
            'low': 0.01,
            'high': 0.1,
            'log': True
        }
    }
    return hyperparameter_space

def create_model(model_params):
    """
    Create and configure a Bayesian classifier instance using the provided parameters.
    Args:
        model_params (dict): Model parameters for the Bayesian classifier.
    Returns:
        BayesianMetaLearner: Configured Bayesian model.
    """
    try:
        params = base_params.copy()
        params.update(model_params)
        logger.info(f"Bayesian model parameters: {params}")
        model = BayesianMetaLearner(**params)
        return model
    except Exception as e:
        logger.error(f"Error creating Bayesian model: {str(e)}")
        raise

def predict_proba(model, X):
    """
    Make probabilistic predictions with the Bayesian model
    
    Args:
        model: Trained Bayesian model instance
        X: Features to predict on
    
    Returns:
        Predicted probabilities as a 2D array where each row contains [1-p, p]
    """
    if model.model is None or model.trace is None:
        raise ValueError("Model has not been trained yet")
    
    with model.model:
        weights_posterior = model.trace.posterior.weights.values
        intercept_posterior = model.trace.posterior.intercept.values
        
        # Average across chains and samples
        weights_mean = weights_posterior.mean(axis=(0, 1))
        intercept_mean = intercept_posterior.mean(axis=(0, 1))
        
        # Linear combination
        linear_pred = intercept_mean + np.dot(X, weights_mean)
        
        # Apply sigmoid to get probabilities
        y_pred_proba = 1.0 / (1.0 + np.exp(-linear_pred))
        
    # Return probabilities in the format expected by scikit-learn
    return np.vstack((1 - y_pred_proba, y_pred_proba)).T

def train_model(X_train, y_train, X_test, y_test, X_eval, y_eval, model_params):
    """
    Train a Bayesian model with threshold optimization.
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Validation features
        y_test: Validation labels
        X_eval: Evaluation features
        y_eval: Evaluation labels
        model_params: Model parameters for Bayesian classifier.
    Returns:
        tuple: (trained_model, metrics)
    """
    try:
        model = create_model(model_params)
        # For Bayesian model, assume a custom train method
        model.train(X_train, y_train, X_test, y_test)
        # Optimize threshold based on evaluation set
        best_threshold, metrics = optimize_threshold(model, X_eval, y_eval, min_recall=min_recall)
        return model, metrics
    except Exception as e:
        logger.error(f"Error training Bayesian model: {str(e)}")
        raise

# def optimize_threshold(model, X_val, y_val, target_precision=0.5, min_recall=0.25):
#     """
#     Optimize the prediction threshold for the Bayesian model.
    
#     Args:
#         model: Trained Bayesian model
#         X_val: Validation features
#         y_val: Validation targets
#         target_precision: Target precision to achieve (default: 0.5)
#         min_recall: Minimum required recall (default: 0.25)
        
#     Returns:
#         Tuple of (optimal_threshold, metrics_at_threshold)
#     """
    
    
#     logger.info("Optimizing threshold for Bayesian model...")
    
#     # Get predictions on validation set
#     y_proba = predict_proba(model, X_val)[:, 1]
    
#     # Find optimal threshold using the utility function
#     optimal_threshold, metrics = tune_threshold_for_precision(
#         y_proba, y_val, 
#         target_precision=target_precision, 
#         required_recall=min_recall,
#         min_threshold=0.1,
#         max_threshold=0.9,
#         step=0.01,
#         logger=logger
#     )
    
#     # Store the optimal threshold
#     model.optimal_threshold = optimal_threshold
    
#     # Log results
#     logger.info(f"Optimal threshold: {optimal_threshold:.4f}")
#     logger.info(f"Precision at threshold: {metrics['precision']:.4f}")
#     logger.info(f"Recall at threshold: {metrics['recall']:.4f}")
    
#     # Log to MLflow
#     mlflow.log_param("bayesian_model_threshold", optimal_threshold)
#     mlflow.log_metrics({
#         "bayesian_precision_at_threshold": metrics['precision'],
#         "bayesian_recall_at_threshold": metrics['recall'],
#         "bayesian_f1_at_threshold": metrics['f1']
#     })
    
#     return optimal_threshold, metrics

def save_model(model, path, threshold=0.5):
    """
    Save Bayesian model to specified path using joblib.
    Args:
        model: Trained Bayesian model
        path: Path to save model
        threshold: Optimal decision threshold
    """
    if model is None:
        raise RuntimeError("No model to save")
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        threshold_path = path.parent / "threshold.json"
        with open(threshold_path, 'w') as f:
            json.dump({
                'threshold': threshold,
                'model_type': 'bayesian',
                'params': model.get_params() if hasattr(model, 'get_params') else {}
            }, f, indent=2)
        logger.info(f"Model saved to {path}")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

def load_model(path):
    """
    Load Bayesian model from specified path.
    Args:
        path: Path to load model from
    Returns:
        tuple: (model, threshold)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No model file found at {path}")
    try:
        model = joblib.load(path)
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
    Run hyperparameter optimization with Optuna for Bayesian classifier.
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
    logger.info("Starting hyperparameter optimization for Bayesian model")
    if not hyperparameter_space:
        hyperparameter_space = load_hyperparameter_space()
    best_score = 0.0

    def objective(trial):
        try:
            params = base_params.copy()
            for param_name, param_config in hyperparameter_space.items():
                if param_config['type'] == 'float':
                    if 'step' in param_config and param_config['step'] is not None:
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
                    if 'step' in param_config and param_config['step'] is not None:
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
            study_name='bayesian_optimization',
            direction='maximize',
            sampler=optuna.samplers.TPESampler(n_startup_trials=50, prior_weight=0.2, seed=int(time.time()))
        )
        best_score = -float('inf')
        best_params = {}
        top_trials = []

        def callback(study, trial):
            nonlocal best_score, best_params, top_trials
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

def hypertune_bayesian(experiment_name: str):
    """
    Main training function with MLflow tracking for Bayesian model.
    Args:
        experiment_name (str): Experiment name for MLflow tracking
        
    Returns:
        tuple: (best_params, best_metrics)
    """
    try:
        with mlflow.start_run(run_name=f"bayesian_model_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            mlflow.set_tags({
                "model_type": "bayesian",
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
            logger.info("Logging best parameters to MLflow")
            for param_name, param_value in best_params.items():
                mlflow.log_param(param_name, param_value)
            mlflow.log_metrics({
                "precision": metrics.get('precision', 0.0),
                "recall": metrics.get('recall', 0.0),
                "f1": metrics.get('f1', 0.0),
                "auc": metrics.get('auc', 0.0),
                "threshold": metrics.get('threshold', 0.5)
            })
            input_example = X_eval.iloc[:5].copy() if hasattr(X_eval, 'iloc') else X_eval[:5].copy()
            if hasattr(input_example, 'dtypes'):
                for col in input_example.columns:
                    if input_example[col].dtype.kind == 'i':
                        logger.info(f"Converting integer column '{col}' to float64")
                        input_example[col] = input_example[col].astype('float64')
            signature = mlflow.models.infer_signature(
                input_example,
                model.predict(input_example)
            )
            logger.info("Model signature created")
            mlflow.sklearn.log_model(
                model,
                "model",
                pip_requirements=pip_requirements,
                registered_model_name=f"bayesian_{datetime.now().strftime('%Y%m%d_%H%M')}",
                signature=signature
            )
            return best_params, metrics
    except Exception as e:
        logger.error(f"Error in hyperparameter tuning: {str(e)}")
        return None, None

def log_to_mlflow(model, metrics, params, experiment_name):
    """
    Log trained model, metrics, and parameters to MLflow for Bayesian model.
    Args:
        model: Trained Bayesian model
        metrics: Model evaluation metrics
        params: Model parameters
        experiment_name: Experiment name
        
    Returns:
        str: Run ID
    """
    try:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=f"bayesian_{datetime.now().strftime('%Y%m%d_%H%M')}") as run:
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            model_info = mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=f"bayesian_{datetime.now().strftime('%Y%m%d_%H%M')}"
            )
            logger.info(f"Model logged to MLflow: {model_info.model_uri}")
            logger.info(f"Run ID: {run.info.run_id}")
            return run.info.run_id
    except Exception as e:
        logger.error(f"Error logging to MLflow: {str(e)}")
        return None

def train_with_precision_target(X_train, y_train, X_test, y_test, X_eval, y_eval):
    """
    Train Bayesian model with focus on precision target.
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
        best_params, _ = hypertune_bayesian(experiment_name)
        if not best_params:
            logger.warning("Hyperparameter tuning failed. Using default parameters.")
            best_params = base_params.copy()
        logger.info("Training final Bayesian model with best parameters")
        model, metrics = train_model(
            X_train, y_train,
            X_test, y_test,
            X_eval, y_eval,
            best_params
        )
        log_to_mlflow(model, metrics, best_params, experiment_name)
        model_path = f"models/StackedEnsemble/base/tree_based/bayesian_model_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Model saved locally at {model_path}")
        return model, metrics
    except Exception as e:
        logger.error(f"Error in precision-focused training: {str(e)}")
        return None, None

def main():
    """
    Main execution function for training Bayesian model.
    """
    try:
        logger.info("Starting Bayesian model training")
        global X_train, y_train, X_test, y_test, X_eval, y_eval
        dataloader = DataLoader()
        X_train, y_train, X_test, y_test, X_eval, y_eval = dataloader.load_data()
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Testing data shape: {X_test.shape}")
        logger.info(f"Evaluation data shape: {X_eval.shape}")
        logger.info(f"Positive class ratios - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}, Eval: {y_eval.mean():.3f}")
        train_with_precision_target(X_train, y_train, X_test, y_test, X_eval, y_eval)
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")


if __name__ == "__main__":
    main() 