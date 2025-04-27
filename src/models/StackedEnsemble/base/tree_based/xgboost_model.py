"""
XGBoost Model for Soccer Draw Prediction

This module implements a XGBoost-based model for predicting soccer match draws.
It includes functionality for model creation, training, hyperparameter optimization,
threshold tuning, and MLflow integration for experiment tracking.

The implementation focuses on high precision while maintaining a minimum recall threshold.
"""

import gc
import os
import random
import time
from datetime import datetime

import mlflow
import numpy as np
import optuna
import sklearn
import xgboost as xgb
from optuna.integration import XGBoostPruningCallback

# Import own modules
from src.utils.logger import ExperimentLogger

# Define experiment name
experiment_name = "xgboost_soccer_prediction"
logger = ExperimentLogger(experiment_name)

# Import data at runtime to avoid global scope issues
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
min_recall = 0.30  # Minimum acceptable recall
n_trials = 100000  # Number of hyperparameter optimization trials as in notebook
# Get current versions
xgb_version = xgb.__version__
sklearn_version = sklearn.__version__
mlflow_version = mlflow.__version__
optuna_version = optuna.__version__
# Create explicit pip requirements list
pip_requirements = [
    f"xgboost=={xgb_version}",
    f"scikit-learn=={sklearn_version}",
    f"mlflow=={mlflow_version}",
    f"optuna=={optuna_version}",
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

# Base parameters as in the notebook
base_params = {
    "objective": "binary:logistic",
    "verbosity": 0,
    "eval_metric": ["aucpr", "error", "logloss"],
    "nthread": 8,
    "seed": 19,
    "device": "cuda",
    "tree_method": "hist",
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
        "early_stopping_rounds": {
            "type": "int",
            "low": 400,   # Slightly below min
            "high": 2000, # Slightly above max
            "step": 10,
        },
        "learning_rate": {
            "type": "float",
            "low": 0.038,   # Slightly below min
            "high": 0.17,   # Slightly above max
            "step": 0.001,
        },
        "max_depth": {
            "type": "int",
            "low": 5,      # At min
            "high": 16,    # Slightly above max
            "step": 1,
        },
        "min_child_weight": {
            "type": "int",
            "low": 320,    # Slightly below min
            "high": 700,   # Slightly above max
            "step": 10,
        },
        "colsample_bytree": {
            "type": "float",
            "low": 0.61,   # Slightly below min
            "high": 0.97,  # Slightly above max
            "step": 0.005,
        },
        "subsample": {
            "type": "float",
            "low": 0.60,   # Slightly below min
            "high": 0.94,  # Slightly above max
            "step": 0.005,
        },
        "gamma": {
            "type": "float",
            "low": 0.20,   # Slightly below min
            "high": 5.1,   # Slightly above max
            "step": 0.01,
        },
        "lambda": {
            "type": "float",
            "low": 2.8,    # Slightly below min
            "high": 17.0,  # Slightly above max
            "step": 0.01,
        },
        "alpha": {
            "type": "float",
            "low": 24.0,   # Slightly below min
            "high": 58.0,  # Slightly above max
            "step": 0.1,
        },
        "scale_pos_weight": {
            "type": "float",
            "low": 1.5,    # Slightly below min
            "high": 3.2,   # Slightly above max
            "step": 0.01,
        },
    }
    return hyperparameter_space


def create_model(model_params):
    """
    Create and configure XGBoost model instance.
    Matches the notebook implementation.
    Now includes early_stopping_rounds in constructor.

    Args:
        model_params (dict): Model parameters (including early_stopping_rounds)
    Returns:
        xgb.XGBClassifier: Configured XGBoost model instance
    """
    # Update with provided parameters
    model_params.update(base_params)
    # Create model
    # Pass all params, including early_stopping_rounds, to the constructor
    model = xgb.XGBClassifier(**model_params)
    return model


def train_model(X_train, y_train, X_test, y_test, X_eval, y_eval, model_params):
    """
    Train a XGBoost model with early stopping and threshold optimization.
    Uses DataFrame/Array for fitting (required by XGBClassifier wrapper)
    Requires X_eval, y_eval for threshold optimization and eval_set.
    Early stopping is handled by the model constructor.
    Args:
        X_train (pd.DataFrame): Training features
        y_train: Training labels (needed by create_model if scale_pos_weight calculation required, though currently static)
        X_test: Validation features (currently unused)
        y_test: Validation labels (currently unused)
        X_eval (pd.DataFrame): Evaluation features for eval_set and threshold optimization
        y_eval (pd.Series): Evaluation labels for threshold optimization
        model_params (dict): Model parameters
    Returns:
        tuple: (trained_model, metrics)
    """
    try:
        # Create model - Pass the full model_params including early_stopping_rounds
        model = create_model(model_params)

        # Create eval set for early stopping using DMatrix
        # Use DataFrame/Array for eval_set as required by fit when using wrapper
        eval_set = [(X_test, y_test)]

        # Fit model with early stopping
        model.fit(
            X=X_train,
            y=y_train,
            eval_set=eval_set,
            verbose=False,
            # early_stopping_rounds is now part of the model's parameters
        )

        # Get validation predictions using the DataFrame for optimize_threshold
        best_threshold, metrics = optimize_threshold(model, X_eval, y_eval, min_recall=min_recall)

        return model, metrics

    except Exception as e:
        logger.error(f"Error training XGBoost model: {str(e)}")
        raise


def optimize_hyperparameters(
    X_train, y_train, X_test, y_test, X_eval, y_eval, hyperparameter_space
):
    """
    Optimize hyperparameters using Optuna, passing DataFrames.
    """
    logger.info("Starting hyperparameter optimization")
    if not hyperparameter_space:
        hyperparameter_space = load_hyperparameter_space()

    best_score = -float("inf")
    best_params_from_hpo = {}
    # Global list to store best trials across the entire hypertuning process
    global_top_trials = []
    top_trials = []

    # Pass necessary data to the objective function
    def objective_func(trial):
        return objective(
            trial, X_train, y_train, X_test, y_test, X_eval, y_eval, hyperparameter_space
        )

    # Callback function defined outside the loop so that its modifications affect the outer scope.
    def callback(study, trial):
        nonlocal best_score, best_params_from_hpo, top_trials
        logger.info(f"Current best score in this batch: {best_score:.4f}")
        if trial.value > best_score:
            best_score = trial.value
            best_params_from_hpo = trial.params
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
    storage_url = "sqlite:///optuna_xgboost.db"
    study_name = "xgboost_optimization"
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
        # Pass the lambda function wrapping objective
        study.optimize(
            objective_func, n_trials=batch_size, show_progress_bar=True, callbacks=[callback]
        )

        # Merge current batch's top trials with global_top_trials
        for trial_record in top_trials:
            global_top_trials.append(trial_record)
        # Keep only the best 10 across all batches
        global_top_trials.sort(key=lambda x: x[0], reverse=True)
        global_top_trials = global_top_trials[:10]

    # After all batches, update best_params (assume the best trial is the first in global_top_trials)
    if global_top_trials:
        best_score, best_params_from_hpo, best_trial_number = global_top_trials[0]
    else:
        best_params_from_hpo = {}  # Initialize if no trials were successful

    # Combine base parameters with the best HPO parameters.
    # HPO results (including early_stopping_rounds) take precedence.
    final_best_params = base_params.copy()
    final_best_params.update(best_params_from_hpo)

    logger.info(f"Best trial value across batches: {best_score:.4f}")
    logger.info(f"Best HPO parameters found: {best_params_from_hpo}")
    logger.info(f"Combined best parameters for model: {final_best_params}")
    logger.info("Top 10 trials across all batches:")
    table_header = "| Rank | Trial # | Score | Parameters |"
    table_separator = "|------|---------|-------|------------|"
    logger.info(table_header)
    logger.info(table_separator)
    for i, trial_record in enumerate(global_top_trials):
        score_val, params_val, trial_num = trial_record
        logger.info(f"| {i + 1} | {trial_num} | {score_val:.4f} | {params_val} |")

    # Return the fully combined parameters
    return final_best_params


# Objective function now needs to accept the data explicitly
def objective(trial, X_train, y_train, X_test, y_test, X_eval, y_eval, hyperparameter_space):
    try:
        params = base_params.copy()
        # Extract early_stopping_rounds separately
        early_stopping_rounds_config = hyperparameter_space.get("early_stopping_rounds")
        if early_stopping_rounds_config:
            params["early_stopping_rounds"] = trial.suggest_int(
                "early_stopping_rounds",
                early_stopping_rounds_config["low"],
                early_stopping_rounds_config["high"],
                step=early_stopping_rounds_config.get("step", 1),
            )
        # Add other hyperparameters from config
        for param_name, param_config in hyperparameter_space.items():
            if param_name == "early_stopping_rounds":  # Already handled
                continue
            if param_config["type"] == "float":
                # ... (suggest float logic remains the same)
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
                # ... (suggest int logic remains the same)
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
        # Pruning Callback - Monitor AUC PR on eval set (default name 'validation_0')
        XGBoostPruningCallback(trial, "validation_0-aucpr")
        # Train model and get metrics using DataFrames
        model, metrics = train_model(
            X_train, y_train, X_test, y_test, X_eval, y_eval, params
        )

        recall = metrics.get("recall", 0.0)
        precision = metrics.get("precision", 0.0)
        threshold = metrics.get("threshold", 0.5)
        # Optimize for precision while maintaining minimum recall
        score = precision if recall >= min_recall else 0.0

        # Pruning: report the score back to Optuna
        trial.report(score, step=model.best_iteration if hasattr(model, "best_iteration") else 0)
        # if trial.should_prune():
        #     raise optuna.TrialPruned()
        logger.info(f"Trial {trial.number}:")
        logger.info(f"  Score: {score:.4f} (Precision: {precision:.4f}, Recall: {recall:.4f})")
        logger.info(f"  Threshold: {threshold:.4f}")
        logger.info(f"  Params: {trial.params}")  # Log trial params directly

        for metric_name, metric_value in metrics.items():
            trial.set_user_attr(metric_name, metric_value)
        
        if score > 0.37:
            log_to_mlflow(model, metrics, params, experiment_name, X_eval)
        return score

    except optuna.TrialPruned:
        logger.info(f"Trial {trial.number} pruned.")
        raise  # Re-raise the exception
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {str(e)}")
        return 0.0  # Return low score for failed trials


def hypertune_xgboost(X_train, y_train, X_test, y_test, X_eval, y_eval, experiment_name: str):
    """
    Main training function with MLflow tracking using DataFrames.
    Args:
        X_train (pd.DataFrame): Training features
        y_train: Training labels
        X_test: Validation features
        y_test: Validation labels
        X_eval (pd.DataFrame): Evaluation features for signature/logging
        y_eval (pd.Series): Evaluation labels
        experiment_name (str): Experiment name for MLflow tracking
    Returns:
        tuple: (best_params, best_metrics)
    """
    try:

        # Load hyperparameter space
        hyperparameter_space = load_hyperparameter_space()

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
        # best_params already contains the combined HPO and base params, including early_stopping_rounds
        final_train_params = best_params

        model, metrics = train_model(
            X_train,
            y_train,
            X_test,
            y_test,
            X_eval,  # Pass X_eval for threshold optimization within train_model
            y_eval,
            final_train_params,  # Pass the full best parameters including early stopping
        )

        return best_params, metrics

    except Exception as e:
        logger.error(f"Error in hyperparameter tuning: {str(e)}")
        return None, None


def log_to_mlflow(model, metrics, params, experiment_name, X_eval):
    """
    Log trained model, metrics, and parameters to MLflow.
    Requires X_eval DataFrame for signature generation.
    Args:
        model: Trained XGBoost model
        metrics: Model evaluation metrics
        params: Model parameters
        experiment_name: Experiment name
        X_eval (pd.DataFrame): Evaluation features for signature generation
    Returns:
        str: Run ID
    """
    try:
        # Set up MLflow tracking
        mlflow.set_experiment(experiment_name)

        # Start a new run
        with mlflow.start_run(
            run_name=f"xgboost_final_{datetime.now().strftime('%Y%m%d_%H%M')}"
        ) as run:
            # Log parameters (excluding base params if desired)
            params_to_log = {
                k: v
                for k, v in params.items()
                if k not in ["device", "objective", "verbosity", "seed", "nthread", "tree_method"]
            }
            mlflow.log_params(params_to_log)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Create input example using the DataFrame X_eval
            input_example = X_eval.iloc[:5].copy()

            # Identify and convert integer columns to float64
            if hasattr(input_example, "dtypes"):
                for col in input_example.columns:
                    if input_example[col].dtype.kind == "i":
                        logger.info(f"Converting integer column '{col}' to float64 for signature")
                        input_example[col] = input_example[col].astype("float64")

            # Infer signature
            signature = mlflow.models.infer_signature(input_example, model.predict(input_example))

            # Update model registration with signature
            model_info = mlflow.xgboost.log_model(
                model,
                "model",
                pip_requirements=pip_requirements,  # Add pip_requirements here as well
                registered_model_name=f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M')}",
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
    Train XGBoost model with focus on precision target using DataFrames.
    Args:
        X_train (pd.DataFrame): Training features
        y_train: Training labels
        X_test: Validation features
        y_test: Validation labels
        X_eval (pd.DataFrame): Evaluation features for logging/thresholding
        y_eval (pd.Series): Evaluation labels
    Returns:
        tuple: (best_model, best_metrics)
    """
    try:
        logger.warning(
            "Training model with precision target - Ensure parameters are updated from HPO."
        )
        params = base_params.copy()
        params.update(
            {
                "early_stopping_rounds": 700,
                "learning_rate": 0.05,
                "max_depth": 8,
                "min_child_weight": 430,
                "colsample_bytree": 0.84,
                "subsample": 0.71,
                "gamma": 2.14,
                "lambda": 9.700000000000001,
                "alpha": 25.200000000000003,
                "scale_pos_weight": 2.36,
            }
        )
        # Train final model with specific parameters
        model, metrics = train_model(
            X_train, y_train, X_test, y_test, X_eval, y_eval, params
        )

        # Log to MLflow using the DataFrame X_eval for signature
        # log_to_mlflow(model, metrics, params, experiment_name, X_eval)
        top_features = select_top_features_xgb(model, X_eval)
        logger.info(f"Top features: {top_features}")
        return model, metrics
    except Exception as e:
        logger.error(f"Error in precision-focused training: {str(e)}")
        return None, None

def select_best_feature_combination(
    X, y, X_test, y_test, X_eval, y_eval,
    num_features=70, num_trials=500, min_recall=0.3, random_state=19
):
    """
    Try multiple random combinations of features, train a model for each,
    and select the best set based on precision (if recall >= min_recall).
    Args:
        X (pd.DataFrame): Training features (all 165 columns)
        y (pd.Series): Training labels
        logger: Logger instance
        X_test, y_test, X_eval, y_eval: Validation/eval sets (same columns as X)
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
    all_features = list(X_eval.columns)
    best_score = -1.0
    best_features = None
    best_mask = None
    model_params = base_params.copy()
    model_params.update(
        {
            "n_estimators": 100,
            "max_depth": 14,
            "learning_rate": 0.125,
            "random_state": random_state,
            "n_jobs": 4,
            "tree_method": "hist",
            "verbosity": 0,
            "eval_metric": ['aucpr', 'error', 'logloss'],
            "colsample_bytree": 0.81,
            "subsample": 0.68,
            "gamma": 3.41,
            "reg_lambda": 3.46,
            "reg_alpha": 48.6,
            "min_child_weight": 400,
            "scale_pos_weight": 2.04,
            "objective": "binary:logistic",
            "nthread": 8,
            "seed": 19,
            "device": "cuda",
            "early_stopping_rounds": 700
        }
    )
    logger.info(f"Trying {num_trials} random combinations of {num_features} features out of {len(all_features)}...")

    for trial in range(num_trials):
        # Randomly select features
        selected = rng.choice(all_features, size=num_features, replace=False)
        selected = list(selected)
        # Subset data
        X_train_sel = X[selected]
        X_test_sel = X_test[selected]
        X_eval_sel = X_eval[selected]
        # Train model and get metrics
        try:
            model, metrics = train_model(
                X_train_sel, y, X_test_sel, y_test, X_eval_sel, y_eval, model_params
            )
            recall = metrics.get("recall", 0.0)
            precision = metrics.get("precision", 0.0)
            score = precision if recall >= min_recall else 0.0
            
            if score > best_score:
                best_score = score
                best_features = selected
                # Create boolean mask for best features
                best_mask = np.array([f in best_features for f in all_features])
            logger.info(f"Trial {trial+1}/{num_trials}: Score={score:.4f} (Precision={precision:.4f}, Recall={recall:.4f} best_score={best_score:.4f})")
        except Exception as e:
            logger.error(f"Trial {trial+1} failed: {e}")

    logger.info(f"Best score: {best_score:.4f} with features: {best_features}")
    return best_features, best_mask, best_score

def select_top_features_xgb(model, X_features, n_features: int = 30) -> list[str]:
    """
    Selects the top N features based on XGBoost feature importances.

    Args:
        model: Trained xgb.XGBClassifier model.
        X_features: DataFrame containing the features used for training (to get names).
        n_features: The number of top features to select.

    Returns:
        A list of the names of the top N features.
    """
    import pandas as pd
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
        logger.info("Starting XGBoost model training")
        
        # Load data
        dataloader = DataLoader()
        X_train, y_train, X_test, y_test, X_eval, y_eval = dataloader.load_data()

        features = import_selected_features_ensemble(model_type="xgb")
        logger.info(f"Features: {len(features)}")
        X_train = prepare_data(X_train, features)
        X_test = prepare_data(X_test, features)
        X_eval = prepare_data(X_eval, features)
        # best_features, best_mask, best_score = select_best_feature_combination(X_train, y_train, X_test, y_test, X_eval, y_eval)
        # Log data shapes
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Testing data shape: {X_test.shape}")
        logger.info(f"Evaluation data shape: {X_eval.shape}")
        logger.info(
            f"Positive class ratio - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}, Eval: {y_eval.mean():.3f}"
        )

        # Run Hyperparameter Optimization
        hypertune_xgboost(
            X_train, y_train, X_test, y_test, X_eval, y_eval, # Pass DataFrames
            experiment_name
        )

        # logger.info("Proceeding to train final model with precision target settings.")
        # best_model, best_metrics = train_with_precision_target(
        #     X_train,
        #     y_train,
        #     X_test,
        #     y_test,
        #     X_eval,
        #     y_eval,  # Pass DataFrames
        # )
        # logger.error("Hyperparameter optimization failed. Skipping precision target training.")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")  # Add traceback
    finally:
        # Clean up DMatrix objects if needed (usually not necessary)
        # del dtrain, dtest, deval
        gc.collect()  # Force garbage collection
        logger.info("Main execution finished.")


if __name__ == "__main__":
    main()
