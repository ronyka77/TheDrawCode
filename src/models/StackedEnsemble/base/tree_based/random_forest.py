"""
XGBoost Model for Soccer Draw Prediction

This module implements a XGBoost-based model for predicting soccer match draws.
It includes functionality for model creation, training, hyperparameter optimization,
threshold tuning, and MLflow integration for experiment tracking.

The implementation focuses on high precision while maintaining a minimum recall threshold.
"""

import os
import random
import time
from datetime import datetime

import mlflow
import numpy as np
import optuna
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier

from src.models.ensemble.data_utils import prepare_data
from src.utils.logger import ExperimentLogger

experiment_name = "random_forest_soccer_prediction"
logger = ExperimentLogger(experiment_name)

from src.models.StackedEnsemble.shared.data_loader import DataLoader
from src.models.StackedEnsemble.shared.hypertuner_utils import optimize_threshold
from src.utils.create_evaluation_set import import_selected_features_ensemble, setup_mlflow_tracking

mlrunds_dir = setup_mlflow_tracking(experiment_name)

# Global settings
min_recall = 0.40  # Minimum acceptable recall
n_trials = 10000  # Number of hyperparameter optimization trials as in notebook
# Get current versions
sklearn_version = sklearn.__version__

# Create explicit pip requirements list
pip_requirements = [f"scikit-learn=={sklearn_version}", f"mlflow=={mlflow.__version__}"]

# Update base parameters for RandomForest
base_params = {"random_state": 19, "n_jobs": 6, "verbose": 0, "criterion": "entropy"}
# Set fixed seed and hash seed for determinism
SEED = 19
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Restrict parallel threads across various libraries
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"


def load_hyperparameter_space_for_hpo():
    """
    Refined hyperparameter space based on top-performing trials.
    """
    hyperparameter_space = {
        "n_estimators": {
            "type": "int",
            "low": 800,    # Focus on range of top performers
            "high": 1300,  # Cover the successful range
            "step": 10,    # Larger step to save computation
        },
        "max_depth": {
            "type": "categorical",  # Change to categorical to focus on two successful regions
            "choices": [6, 7, 8, 9, 18, 19, 20, 21],  # Target both shallow and deep trees
        },
        "min_samples_split": {
            "type": "int",
            "low": 30,  # Allow splitting slightly easier
            "high": 80,  # Allow slightly more constrained splitting too
            "step": 2,  # Can increase step slightly if range is wider
        },
        "min_samples_leaf": {
            "type": "int",
            "low": 16,  # Allow smaller leaf nodes
            "high": 70,  # Allow slightly larger leaf nodes too
            "step": 2,  # Can increase step slightly
        },
        "max_features": {
            "type": "categorical",  # Change to categorical to focus on two successful regions
            "choices": [0.22, 0.24, 0.26, 0.52, 0.70, 0.74, 0.84, 0.88, 0.98, 1.0],  # Target both low and high values
        },
        "class_weight": {
            "type": "float",
            "low": 1.8,   # Slightly under lowest successful value
            "high": 3.6,  # Maximum from successful trials
            "step": 0.05, # Keep fine-grained control
        },
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
        if "class_weight" in params and not isinstance(params["class_weight"], dict):
            class_weight_value = params.pop("class_weight")
            params["class_weight"] = {0: 1.0, 1: class_weight_value}
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
        best_threshold, metrics = optimize_threshold(model, X_eval, y_eval, min_recall=min_recall)

        return model, metrics

    except Exception as e:
        logger.error(f"Error training RandomForest model: {str(e)}")
        raise


def optimize_hyperparameters(
    X_train, y_train, X_test, y_test, X_eval, y_eval, hyperparameter_space
):
    logger.info("Starting hyperparameter optimization")

    if not hyperparameter_space:
        hyperparameter_space = load_hyperparameter_space_for_hpo()

    best_score = -float("inf")
    best_params = {}
    # Global list to store best trials across the entire hypertuning process
    global_top_trials = []
    top_trials = []

    def objective(trial):
        try:
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

            # Train model and get metrics
            model, metrics = train_model(X_train, y_train, X_test, y_test, X_eval, y_eval, params)

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
            # Log to MLflow
            if score > 0.33:
                log_to_mlflow(model, metrics, params, experiment_name)
            return score

        except Exception as e:
            logger.error(f"Trial failed: {str(e)}")
            return 0.0

    # Callback function defined outside the loop so that its modifications affect the outer scope.
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
        return best_score

    # Set persistent storage path using SQLite
    storage_url = "sqlite:///optuna_random_forest.db"
    study_name = "random_forest_optimization"
    # Total trials to conduct
    total_trials = n_trials  # Example; you can set n_trials accordingly.
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
        study.optimize(objective, n_trials=batch_size, show_progress_bar=True, callbacks=[callback], n_jobs=8)

        # Merge current batch's top trials with global_top_trials
        for trial_record in top_trials:
            global_top_trials.append(trial_record)
        # Keep only the best 10 across all batches
        global_top_trials.sort(key=lambda x: x[0], reverse=True)
        global_top_trials = global_top_trials[:10]

    # After all batches, update best_params (assume the best trial is the first in global_top_trials)
    if global_top_trials:
        best_score, best_params, best_trial_number = global_top_trials[0]
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


def hypertune_random_forest(experiment_name: str):
    """
    Main training function with MLflow tracking.
    """
    try:
        # Load hyperparameter space
        hyperparameter_space = load_hyperparameter_space_for_hpo()

        # Run hyperparameter optimization
        logger.info("Starting hyperparameter optimization")
        best_params = optimize_hyperparameters(
            X_train,
            y_train,
            X_test,
            y_test,
            X_eval,
            y_eval,
            hyperparameter_space=hyperparameter_space,
        )

        # Train final model with best parameters
        logger.info("Training final model with best parameters")
        model, metrics = train_model(
            X_train, y_train, X_test, y_test, X_eval, y_eval, best_params
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
            signature = mlflow.models.infer_signature(input_example, model.predict(input_example))

            # Log model
            model_info = mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=f"rf_{datetime.now().strftime('%Y%m%d_%H%M')}",
                signature=signature,
            )

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
        logger.warning("Training model with precision target.")
        params = base_params.copy()
        params.update(
            {
                "bootstrap": True,
                "class_weight": 3.55,
                "criterion": "entropy", 
                "max_depth": 21,
                "max_features": 0.58,
                "min_samples_leaf": 54,
                "min_samples_split": 36,
                "n_estimators": 960,
                "n_jobs": 6,
                "random_state": 19,
                "verbose": 0,
            }
        )
        # Train final model with best parameters
        logger.info("Training final model with best parameters")
        model, metrics = train_model(X_train, y_train, X_test, y_test, X_eval, y_eval, params)
        # Log to MLflow
        # log_to_mlflow(model, metrics, params, experiment_name)
        top_features = select_top_features_rf(model, X_train)
        logger.info(f"Top features: {top_features}")
        return model, metrics
    except Exception as e:
        logger.error(f"Error in precision-focused training: {str(e)}")
        return None, None


def select_top_features_rf(model: RandomForestClassifier, X_features: pd.DataFrame, n_features: int = 40) -> list[str]:
    """
    Selects the top N features based on Random Forest feature importances.

    Args:
        model: Trained RandomForestClassifier model.
        X_features: DataFrame containing the features used for training (to get names).
        n_features: The number of top features to select.

    Returns:
        A list of the names of the top N features.
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("The provided model has not been trained yet or does not support feature importances.")

    importances = model.feature_importances_
    feature_names = X_features.columns

    if len(importances) != len(feature_names):
        raise ValueError("Mismatch between the number of feature importances and feature names.")

    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    top_features = feature_importance_df['Feature'].head(n_features).tolist()
    logger.info(f"Selected top {n_features} features based on RF importance.")
    logger.info(f"Top features: {top_features}") # Log the selected features for visibility

    return top_features


def main():
    """
    Main execution function.
    """
    try:
        logger.info("Starting RandomForest model training")
        # Import data at runtime to avoid global scope issues
        global X_train, y_train, X_test, y_test, X_eval, y_eval

        # Load data
        dataloader = DataLoader()
        X_train, y_train, X_test, y_test, X_eval, y_eval = dataloader.load_data()
        features = import_selected_features_ensemble(model_type="rf")
        X_train = prepare_data(X_train, features)
        X_test = prepare_data(X_test, features)
        X_eval = prepare_data(X_eval, features)
        # Log data shapes
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Testing data shape: {X_test.shape}")
        logger.info(f"Evaluation data shape: {X_eval.shape}")
        logger.info(
            f"Positive class ratio - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}, Eval: {y_eval.mean():.3f}"
        )

        current_params, current_metrics = hypertune_random_forest(experiment_name)
        logger.info(f"Run completed with parameters: {current_params}")
        logger.info(f"Run metrics: {current_metrics}")

        # Train model with precision target
        best_model, best_metrics = train_with_precision_target(
            X_train, y_train, X_test, y_test, X_eval, y_eval
        )
        logger.info(f"Best model: {best_model}")
        logger.info(f"Best metrics: {best_metrics}")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")


if __name__ == "__main__":
    main()
