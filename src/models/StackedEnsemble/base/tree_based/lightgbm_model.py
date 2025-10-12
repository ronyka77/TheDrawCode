"""
LightGBM Model for Soccer Draw Prediction

This module implements a LightGBM-based model for predicting soccer match draws.
It includes functionality for model creation, training, hyperparameter optimization,
threshold tuning, and MLflow integration for experiment tracking.

The implementation focuses on high precision while maintaining a minimum recall threshold.
"""

import os
import random
import time
from datetime import datetime

import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import pandas as pd
from mlflow.models import infer_signature

from src.utils.logger import ExperimentLogger

experiment_name = "lightgbm_soccer_prediction"
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
mlrunds_dir = setup_mlflow_tracking(experiment_name)

# Global settings
min_recall = 0.20  # Minimum acceptable recall
n_trials = 100000  # Number of hyperparameter optimization trials as in notebook

# Base parameters as in the notebook
base_params = {
    "objective": "binary",
    "metric": ["aucpr", "binary_logloss"],
    "verbose": -1,
    "n_jobs": 8,
    "random_state": 19,
    "device": "cpu",
}

# Set fixed seed and hash seed for determinism
SEED = 19
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Restrict parallel threads across various libraries
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"


def load_hyperparameter_space():
    """
    Define a tightened hyperparameter space for LightGBM tuning based on the
    top 10 best trials from the last hypertuning cycle. The updated ranges aim
    to increase precision by focusing on the sweet-spot regions observed.
    Returns:
        dict: Hyperparameter space configuration with narrowed ranges and steps.
    """
    hyperparameter_space = {
        "learning_rate": {
            "type": "float",
            "low": 0.045,
            "high": 0.18,
            "log": False,
            "step": 0.0025,
        },
        "num_leaves": {"type": "int", "low": 55, "high": 200, "log": False, "step": 5},
        "max_depth": {"type": "int", "low": 5, "high": 12, "log": False, "step": 1},
        "min_child_samples": {"type": "int", "low": 200, "high": 600, "log": False, "step": 10},
        "feature_fraction": {
            "type": "float",
            "low": 0.58,
            "high": 0.75,
            "log": False,
            "step": 0.01,
        },
        "bagging_fraction": {
            "type": "float",
            "low": 0.56,
            "high": 0.75,
            "log": False,
            "step": 0.005,
        },
        "bagging_freq": {"type": "int", "low": 10, "high": 15, "log": False, "step": 1},
        "reg_alpha": {"type": "float", "low": 8.0, "high": 20.0, "log": False, "step": 0.1},
        "reg_lambda": {"type": "float", "low": 8.0, "high": 20.0, "log": False, "step": 0.1},
        "min_split_gain": {"type": "float", "low": 0.12, "high": 0.30, "log": False, "step": 0.005},
        "early_stopping_rounds": {
            "type": "int",
            "low": 600,
            "high": 1200,
            "log": False,
            "step": 10,
        },
        "path_smooth": {"type": "float", "low": 0.10, "high": 0.60, "log": False, "step": 0.005},
        "cat_smooth": {"type": "float", "low": 20.0, "high": 35.0, "log": False, "step": 0.1},
        "max_bin": {"type": "int", "low": 200, "high": 700, "log": False, "step": 10},
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


def train_model(x_train, y_train, x_test, y_test, x_eval, y_eval, model_params):
    """
    Train a LightGBM model with early stopping and threshold optimization.
    Updated to match notebook implementation.
    Args:
        x_train: Training features
        y_train: Training labels
        x_test: Validation features
        y_test: Validation labels
        x_eval: Evaluation features
        y_eval: Evaluation labels
        model_params: Model parameters
    Returns:
        tuple: (trained_model, metrics)
    """
    try:
        # Combine training and validation data while preserving indexes
        x_combined = pd.concat([x_train, x_test], axis=0)
        y_combined = pd.concat([y_train, y_test], axis=0)

        # Reset indexes to ensure proper alignment
        x_combined.reset_index(drop=True, inplace=True)
        y_combined.reset_index(drop=True, inplace=True)
        # Extract early stopping rounds if present
        early_stopping_rounds = model_params.pop("early_stopping_rounds", 100)

        # Create model with remaining parameters
        model = create_model(model_params)

        # Create eval set for early stopping
        eval_set = [(x_eval, y_eval)]

        # Fit model with early stopping
        model.fit(
            x_combined,
            y_combined,
            eval_set=eval_set,
            callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds)],
        )

        # Get validation predictions
        _, metrics = optimize_threshold(model, x_eval, y_eval, min_recall=min_recall)

        return model, metrics

    except Exception as e:
        logger.error(f"Error training LightGBM model: {str(e)}")
        raise


def suggest_hyperparameters(trial, hyperparameter_space):
    """
    Suggest hyperparameters for a trial based on the hyperparameter space configuration.
    """
    params = base_params.copy()
    # Add hyperparameters from config with step size if provided
    for param_name, param_config in hyperparameter_space.items():
        if param_config["type"] == "float":
            if "step" in param_config:
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    step=param_config["step"],
                    log=param_config.get("log", False),
                )
            else:
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=param_config.get("log", False),
                )
        elif param_config["type"] == "int":
            if "step" in param_config:
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    step=param_config["step"],
                )
            else:
                params[param_name] = trial.suggest_int(
                    param_name, param_config["low"], param_config["high"]
                )
    return params


def create_objective_function(x_train, y_train, x_test, y_test, x_eval, y_eval, hyperparameter_space):
    """
    Create the objective function for Optuna optimization.
    """
    def objective(trial):
        try:
            params = suggest_hyperparameters(trial, hyperparameter_space)

            # Train model and get metrics
            _, metrics = train_model(x_train, y_train, x_test, y_test, x_eval, y_eval, params)

            recall = metrics.get("recall", 0.0)
            precision = metrics.get("precision", 0.0)
            threshold = metrics.get("threshold", 0.5)
            # Optimize for precision while maintaining minimum recall
            score = precision if recall >= min_recall else 0.0

            logger.info(f"Trial {trial.number}:")
            logger.info(f"  Score: {score}")
            logger.info(f"  Threshold: {threshold}")
            logger.info(f"  Params: {params}")

            for metric_name, metric_value in metrics.items():
                trial.set_user_attr(metric_name, metric_value)

            return score

        except Exception as e:
            logger.error(f"Trial failed: {str(e)}")
            return 0.0

    return objective


def optimize_hyperparameters(
    x_train, y_train, x_test, y_test, x_eval, y_eval, hyperparameter_space
):
    """
    Optimize hyperparameters using Optuna with batching strategy.
    """
    logger.info("Starting hyperparameter optimization")

    if not hyperparameter_space:
        hyperparameter_space = load_hyperparameter_space()

    # Get the objective function
    objective = create_objective_function(x_train, y_train, x_test, y_test, x_eval, y_eval, hyperparameter_space)

    # Run optimization with batching
    best_params = run_batched_optimization(objective)

    return best_params


def create_callback_function(best_score, best_params, top_trials, global_top_trials):
    """
    Create the callback function for Optuna optimization.
    """
    def callback(study, trial):
        nonlocal best_score, best_params, top_trials, global_top_trials
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
        # Log global top trials every 100 trials
        if trial.number % 100 == 0 and global_top_trials:
            logger.info("Global top 10 trials:")
            table_rows = [
                f"| {i + 1} | {rec[2]} | {rec[0]:.4f} | {rec[1]} |"
                for i, rec in enumerate(global_top_trials[:10])
            ]
            logger.info(table_header)
            logger.info(table_separator)
            for row in table_rows:
                logger.info(row)

    return callback


def run_batched_optimization(objective):
    """
    Run Optuna optimization with batching strategy.
    """
    best_score = -float("inf")
    best_params = {}
    global_top_trials = []
    top_trials = []

    # Create callback function
    callback = create_callback_function(best_score, best_params, top_trials, global_top_trials)

    # Set persistent storage path using SQLite
    storage_url = "sqlite:///optuna_lightgbm.db"
    study_name = "lightgbm_optimization"
    total_trials = n_trials
    batch_size = 1000
    num_batches = total_trials // batch_size
    if total_trials % batch_size != 0:
        num_batches += 1

    # Loop over batches, resetting the sampler each time
    for batch in range(num_batches):
        # Create a new sampler with a dynamic seed
        random_seed = int(time.time())
        new_sampler = optuna.samplers.RandomSampler(seed=random_seed)

        # (Re-)create the study with persistent storage; previous trials are loaded automatically.
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            storage=storage_url,
            load_if_exists=True,
            sampler=new_sampler,
        )

        logger.info(
            f"Starting batch {batch + 1}/{num_batches} with new sampler (seed={random_seed})"
        )
        study.optimize(objective, n_trials=batch_size, show_progress_bar=True, callbacks=[callback])

        # Merge current batch's top trials with global_top_trials
        for trial_record in top_trials:
            global_top_trials.append(trial_record)
        # Keep only the best 10 across all batches
        global_top_trials.sort(key=lambda x: x[0], reverse=True)
        global_top_trials = global_top_trials[:10]

    # After all batches, update best_params (assume the best trial is the first in global_top_trials)
    if global_top_trials:
        best_score, best_params, _ = global_top_trials[0]
    else:
        best_params = {}

    best_params.update(base_params)

    logger.info(f"Best trial value across batches: {best_score:.4f}")
    logger.info(f"Best parameters found: {best_params}")
    logger.info("Top 10 trials across all batches:")
    for i, trial_record in enumerate(global_top_trials):
        score_val, params_val, trial_num = trial_record
        logger.info(f"| {i + 1} | {trial_num} | {score_val:.4f} | {params_val} |")

    return best_params


def hypertune_lightgbm():
    """
    Main training function with MLflow tracking.
    Updated name from hypertune_mlp to hypertune_lightgbm to match notebook.
    Args:
    Returns:
        tuple: (best_params, best_metrics)
    """
    try:
        # Load data
        dataloader = DataLoader()
        x_train_raw, y_train, x_test_raw, y_test, x_eval_raw, y_eval = dataloader.load_data()
        features = import_selected_features_ensemble(model_type="lgbm")

        # Ensure features is a list of strings
        if not isinstance(features, list):
            raise ValueError("Expected features to be a list for lgbm model type")

        x_train = prepare_data(x_train_raw, features)
        x_test = prepare_data(x_test_raw, features)
        x_eval = prepare_data(x_eval_raw, features)

        # Load hyperparameter space
        hyperparameter_space = load_hyperparameter_space()

        # Run hyperparameter optimization
        logger.info("Starting hyperparameter optimization")
        best_params = optimize_hyperparameters(
            x_train,
            y_train,
            x_test,
            y_test,
            x_eval,
            y_eval,
            hyperparameter_space=hyperparameter_space,
        )

        # Train final model with best parameters
        logger.info("Training final model with best parameters")
        _, metrics = train_model(x_train, y_train, x_test, y_test, x_eval, y_eval, best_params)

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
                registered_model_name=f"lightgbm_{datetime.now().strftime('%Y%m%d_%H%M')}",
            )
            # Create input example for model signature
            input_example = X_train.head(5)
            # Handle integer columns by converting them to float64 to properly manage missing values
            input_example = X_eval.iloc[:5].copy() if hasattr(X_eval, "iloc") else X_eval[:5].copy()

            # Identify and convert integer columns to float64 to prevent schema enforcement errors
            if hasattr(input_example, "dtypes"):
                for col in input_example.columns:
                    if X_eval[col].dtype.kind == "i":
                        logger.info(
                            f"Converting integer column '{col}' to float64 to handle potential missing values"
                        )
                        X_eval[col] = X_eval[col].astype("float64")

            # Infer signature with proper handling for integer columns with potential missing values
            signature = infer_signature(input_example, model.predict(input_example))

            # Update model registration with signature
            model_info = mlflow.lightgbm.log_model(
                model,
                "model",
                registered_model_name=f"lightgbm_{datetime.now().strftime('%Y%m%d_%H%M')}",
                signature=signature,
            )
            run_id = run.info.run_id
            logger.info(f"Model logged to MLflow: {model_info.model_uri}")
            logger.info(f"Run ID: {run_id}")
            mlflow.end_run()
            return run_id

    except Exception as e:
        logger.error(f"Error logging to MLflow: {str(e)}")
        return None


def train_with_precision_target(x_train, y_train, x_test, y_test, x_eval, y_eval):
    """
    Train LightGBM model with focus on precision target.
    Args:
        x_train: Training features
        y_train: Training labels
        x_test: Testing features
        y_test: Testing labels
        x_eval: Evaluation features
        y_eval: Evaluation labels
    Returns:
        tuple: (best_model, best_metrics)
    """
    try:
        logger.info("Training model with precision target")
        params = base_params.copy()
        params.update(
            {
                "learning_rate": 0.14,
                "num_leaves": 85,
                "max_depth": 6,
                "min_child_samples": 270,
                "feature_fraction": 0.6100000000000001,
                "bagging_fraction": 0.5750000000000001,
                "bagging_freq": 14,
                "reg_alpha": 16.200000000000003,
                "reg_lambda": 15.5,
                "min_split_gain": 0.14,
                "early_stopping_rounds": 670,
                "path_smooth": 0.34500000000000003,
                "cat_smooth": 23.400000000000002,
                "max_bin": 250,
            }
        )

        # Train final model with best parameters
        logger.info("Training final model with best parameters")
        model, metrics = train_model(x_train, y_train, x_test, y_test, x_eval, y_eval, params)

        # Log to MLflow
        log_to_mlflow(model, metrics, params, experiment_name)

        return model, metrics

    except Exception as e:
        logger.error(f"Error in precision-focused training: {str(e)}")
        return None, None


def select_best_feature_combination(
    x,
    y,
    x_test,
    y_test,
    x_eval,
    y_eval,
    num_features=95,
    num_trials=1000,
    min_recall=0.2,
    random_state=19,
):
    """
    Try multiple random combinations of features, train a model for each,
    and select the best set based on precision (if recall >= min_recall).
    Args:
        x (pd.DataFrame): Training features (all 165 columns)
        y (pd.Series): Training labels
        x_test, y_test, x_eval, y_eval: Validation/eval sets (same columns as x)
        num_features (int): Number of features to select in each trial
        num_trials (int): Number of random combinations to try
        min_recall (float): Minimum recall threshold for score
        random_state (int): Random seed
    Returns:
        best_features (list): List of best feature names
        best_mask (np.ndarray): Boolean mask for best features
        best_score (float): Best score achieved
    """

    rng = np.random.default_rng(random_state)
    all_features = list(x.columns)
    best_score = -1.0
    best_features = None
    best_mask = None
    model_params = base_params.copy()
    model_params.update(
        {
            "learning_rate": 0.1625,
            "num_leaves": 75,
            "max_depth": 10,
            "min_child_samples": 540,
            "feature_fraction": 0.61,
            "bagging_fraction": 0.645,
            "bagging_freq": 15,
            "reg_alpha": 19.4,
            "reg_lambda": 11.8,
            "min_split_gain": 0.17,
            "early_stopping_rounds": 670,
            "path_smooth": 0.51,
            "cat_smooth": 33.8,
            "max_bin": 670,
            "device": "cpu",
            "n_jobs": 8,
            "objective": "binary",
            "metric": ["aucpr", "binary_logloss"],
            "random_state": 19,
            "verbose": -1,
        }
    )
    logger.info(
        f"Trying {num_trials} random combinations of {num_features} features out of {len(all_features)}..."
    )

    for trial in range(num_trials):
        # Randomly select features
        selected = rng.choice(all_features, size=num_features, replace=False)
        selected = list(selected)
        # Subset data
        x_train_sel = x[selected]
        x_test_sel = x_test[selected]
        x_eval_sel = x_eval[selected]
        # Train model and get metrics
        try:
            _, metrics = train_model(
                x_train_sel, y, x_test_sel, y_test, x_eval_sel, y_eval, model_params
            )
            recall = metrics.get("recall", 0.0)
            precision = metrics.get("precision", 0.0)
            score = precision if recall >= min_recall else 0.0

            if score > best_score:
                best_score = score
                best_features = selected
                # Create boolean mask for best features
                best_mask = np.array([f in best_features for f in all_features])
            logger.info(
                f"Trial {trial + 1}/{num_trials}: Score={score:.4f} (Precision={precision:.4f}, Recall={recall:.4f} best_score={best_score:.4f})"
            )
        except Exception as e:
            logger.error(f"Trial {trial + 1} failed: {e}")

    logger.info(f"Best score: {best_score:.4f} with {num_features} features: {best_features}")
    return best_features, best_mask, best_score


def main():
    """
    Main execution function.
    """
    try:
        logger.info("Starting LightGBM model training")
        global X_train, y_train, X_test, y_test, X_eval, y_eval

        # Load data
        dataloader = DataLoader()
        X_train, y_train, X_test, y_test, X_eval, y_eval = dataloader.load_data()
        features = import_selected_features_ensemble(model_type="lgbm")

        # Ensure features is a list of strings
        if not isinstance(features, list):
            raise ValueError("Expected features to be a list for lgbm model type")

        x_train = prepare_data(X_train, features)
        x_test = prepare_data(X_test, features)
        x_eval = prepare_data(X_eval, features)


        # Log data shapes
        logger.info(f"Training data shape: {x_train.shape}")
        logger.info(f"Testing data shape: {x_test.shape}")
        logger.info(f"Evaluation data shape: {x_eval.shape}")
        logger.info(
            f"Positive class ratio - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}, Eval: {y_eval.mean():.3f}"
        )

        logger.info("Starting hyperparameter optimization run")
        current_params, _ = hypertune_lightgbm()
        logger.info(f"Run completed with parameters: {current_params}")


    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")


if __name__ == "__main__":
    main()
