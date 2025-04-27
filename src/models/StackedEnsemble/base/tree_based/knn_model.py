"""
KNN Model for Soccer Draw Prediction

This module implements a KNN-based model for predicting soccer match draws.
It includes functionality for model creation, training, hyperparameter optimization,
threshold tuning, and MLflow integration for experiment tracking.

The implementation focuses on high precision while maintaining a minimum recall threshold.
"""

import os
import pickle
import random
import time
from datetime import datetime

import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from src.utils.logger import ExperimentLogger

experiment_name = "knn_soccer_prediction"
logger = ExperimentLogger(experiment_name)

# Import shared utility functions
from src.models.ensemble.data_utils import prepare_data
from src.models.StackedEnsemble.shared.data_loader import DataLoader
from src.models.StackedEnsemble.shared.hypertuner_utils import optimize_threshold
from src.utils.create_evaluation_set import (
    import_selected_features_ensemble,
    setup_mlflow_tracking,
)

# Setup MLflow tracking
tmp_dir = setup_mlflow_tracking(experiment_name)

# Global settings
min_recall = 0.25  # Minimum acceptable recall
n_trials = 10000  # Number of hyperparameter optimization trials (KNN is fast)

# Set fixed seed and hash seed for determinism
SEED = 19
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Restrict parallel threads
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"


def load_hyperparameter_space():
    """
    Define the hyperparameter space for KNN tuning.
    Returns:
        dict: Hyperparameter space configuration.
    """
    return {
        "n_neighbors": {"type": "int", "low": 300, "high": 2000, "step": 10},
        "weights": {"type": "categorical", "choices": ["uniform", "distance"]},
        "p": {"type": "int", "low": 1, "high": 10, "step": 1},  # 1=Manhattan, 2=Euclidean
        "leaf_size": {"type": "int", "low": 16, "high": 150, "step": 2},
        "algorithm": {"type": "categorical", "choices": ["auto", "ball_tree", "kd_tree", "brute"]},
        "metric": {"type": "categorical", "choices": [
            "minkowski", "euclidean", "manhattan", "chebyshev", "canberra", "cosine", "hamming"
        ]},
    }


def create_model(model_params):
    """
    Create and configure KNN model instance.
    Args:
        model_params (dict): Model parameters
    Returns:
        KNeighborsClassifier: Configured KNN model
    """
    params = model_params.copy()
    return KNeighborsClassifier(**params)


def train_model(X_train, y_train, X_test, y_test, X_eval, y_eval, model_params, scaler):
    """
    Train a KNN model and optimize threshold.
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
        # KNN is sensitive to feature scaling
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_eval_scaled = scaler.transform(X_eval)

        model = create_model(model_params)
        model.fit(X_train_scaled, y_train)

        # Optimize threshold on evaluation set
        best_threshold, metrics = optimize_threshold(
            model, X_eval_scaled, y_eval, min_recall=min_recall
        )
        metrics["threshold"] = best_threshold
        metrics["scaler"] = scaler  # For logging
        return (model, metrics)
    except Exception as e:
        logger.error(f"Error training KNN model: {str(e)}")
        raise


def optimize_hyperparameters(
    X_train, y_train, X_test, y_test, X_eval, y_eval, hyperparameter_space, scaler
):
    logger.info("Starting hyperparameter optimization")
    if not hyperparameter_space:
        hyperparameter_space = load_hyperparameter_space()

    best_score = -float("inf")
    best_params = {}
    top_trials = []

    def objective(trial):
        try:
            params = {}
            for param_name, param_config in hyperparameter_space.items():
                if param_config["type"] == "float":
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                        step=param_config.get("step"),
                        log=param_config.get("log", False),
                    )
                elif param_config["type"] == "int":
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                        step=param_config.get("step", 1),
                    )
                elif param_config["type"] == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config["choices"]
                    )

            model, metrics = train_model(
                X_train, y_train, X_test, y_test, X_eval, y_eval, params, scaler
            )
            recall = metrics.get("recall", 0.0)
            precision = metrics.get("precision", 0.0)
            threshold = metrics.get("threshold", 0.5)
            score = precision if recall >= min_recall else 0.0

            logger.info(f"Trial {trial.number}: Score: {score:.4f}, Threshold: {threshold}, Params: {params}")
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
        # Create a record for the current trial
        current_run = (trial.value, trial.params, trial.number)
        top_trials.append(current_run)
        # Sort and keep only top 10 for this batch
        top_trials.sort(key=lambda x: x[0], reverse=True)
        top_trials[:] = top_trials[:10]
        table_header = "| Rank | Trial # | Score | Parameters |"
        table_separator = "|------|---------|-------|------------|"
        if trial.number % 9 == 0:
            table_rows = [
                f"| {i + 1} | {rec[2]} | {rec[0]:.4f} | {rec[1]} |"
                for i, rec in enumerate(top_trials)
            ]
            logger.info("Top trials in current batch:")
            logger.info(table_header)
            logger.info(table_separator)
            for row in table_rows:
                logger.info(row)
        return best_score

    sampler = optuna.samplers.RandomSampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, callbacks=[callback])

    best_params = study.best_params
    logger.info(f"Best trial value: {study.best_value:.4f}")
    logger.info(f"Best parameters found: {best_params}")
    logger.info("Top 10 trials across all batches:")
    table_header = "| Rank | Trial # | Score | Parameters |"
    table_separator = "|------|---------|-------|------------|"
    logger.info(table_header)
    logger.info(table_separator)
    for i, trial_record in enumerate(top_trials):
        score_val, params_val, trial_num = trial_record
        logger.info(f"| {i + 1} | {trial_num} | {score_val:.4f} | {params_val} |")
    return best_params


def log_to_mlflow(model, scaler, metrics, params, experiment_name, X_eval):
    """
    Log trained model, scaler, metrics, and parameters to MLflow.
    Args:
        model: Trained KNN model
        scaler: Fitted scaler
        metrics: Model evaluation metrics
        params: Model parameters
        experiment_name: Experiment name
        X_eval: Evaluation features for signature
    Returns:
        str: Run ID
    """
    try:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=f"knn_{datetime.now().strftime('%Y%m%d_%H%M')}") as run:
            mlflow.log_params(params)
            for metric_name, metric_value in metrics.items():
                if metric_name != "scaler":
                    mlflow.log_metric(metric_name, metric_value)
            # Log scaler as artifact
            scaler_path = "src/models/scalers/scaler_knn.pkl"
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)
            mlflow.log_artifact(scaler_path, artifact_path="scaler")
            # Log model
            mlflow.sklearn.log_model(model, "model")
            # Log input example and signature
            input_example = X_eval[:5]
            signature = mlflow.models.infer_signature(input_example, model.predict(input_example))
            mlflow.sklearn.log_model(model, "model_with_signature", signature=signature)
            run_id = run.info.run_id
            logger.info(f"Model logged to MLflow: {run_id}")
            mlflow.end_run()
            return run_id
    except Exception as e:
        logger.error(f"Error logging to MLflow: {str(e)}")
        return None


def main():
    """
    Main execution function for KNN hypertuning and training.
    """
    try:
        logger.info("Starting KNN model training")
        global X_train, y_train, X_test, y_test, X_eval, y_eval, scaler
        # Load data
        dataloader = DataLoader()
        X_train, y_train, X_test, y_test, X_eval, y_eval = dataloader.load_data()
        features = import_selected_features_ensemble(model_type="lgbm")
        X_train = prepare_data(X_train, features)
        X_test = prepare_data(X_test, features)
        X_eval = prepare_data(X_eval, features)

        # Try to load existing scaler
        scaler_path = "src/models/scalers/scaler_knn.pkl"
        try:
            logger.info("Attempting to load existing scaler...")
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            logger.info("Successfully loaded existing scaler")
        except (FileNotFoundError, pickle.UnpicklingError) as e:
            logger.info(f"Could not load scaler ({str(e)}). Creating new StandardScaler...")
            scaler = StandardScaler()
            scaler.fit(X_train)
            # Save the fitted scaler
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)
            logger.info("New scaler fitted and saved")

        # Transform data using scaler
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test) 
        X_eval = scaler.transform(X_eval)
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Testing data shape: {X_test.shape}")
        logger.info(f"Evaluation data shape: {X_eval.shape}")
        logger.info(
            f"Positive class ratio - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}, Eval: {y_eval.mean():.3f}"
        )
        logger.info("Starting hyperparameter optimization run")
        hyperparameter_space = load_hyperparameter_space()
        current_params = optimize_hyperparameters(
            X_train, y_train, X_test, y_test, X_eval, y_eval, hyperparameter_space, scaler
        )
        logger.info(f"Run completed with parameters: {current_params}")
        # Train model with best parameters
        model, metrics = train_model(
            X_train, y_train, X_test, y_test, X_eval, y_eval, current_params, scaler
        )
        log_to_mlflow(model, scaler, metrics, current_params, experiment_name, X_eval)
        logger.info(f"Best model: {model}")
        logger.info(f"Best metrics: {metrics}")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")


if __name__ == "__main__":
    main() 