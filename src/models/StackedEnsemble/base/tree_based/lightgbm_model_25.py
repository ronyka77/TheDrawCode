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
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

from src.utils.logger import ExperimentLogger

experiment_name = "lightgbm_soccer_prediction_25"
logger = ExperimentLogger(experiment_name)

# Import shared utility functions
from src.models.ensemble.data_utils import prepare_data
from src.models.StackedEnsemble.shared.data_loader_new import DataLoader
from src.models.StackedEnsemble.shared.hypertuner_utils import optimize_threshold
from src.utils.create_evaluation_set import (
    import_selected_features_ensemble_new,
    setup_mlflow_tracking,
)

# Setup MLflow tracking
mlrunds_dir = setup_mlflow_tracking(experiment_name)

# Global settings
min_recall = 0.25  # Minimum acceptable recall
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
os.environ["OMP_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["OPENBLAS_NUM_THREADS"] = "12"


def load_hyperparameter_space():
    """
    Define a tightened hyperparameter space for LightGBM tuning based on the
    top 10 best trials from the last hypertuning cycle. The updated ranges aim
    to increase precision by focusing on the sweet-spot regions observed.
    Returns:
        dict: Hyperparameter space configuration with narrowed ranges and steps.
    """
    hyperparameter_space = {
        "learning_rate": {"type": "float", "low": 0.060, "high": 0.20, "log": False, "step": 0.001},
        "num_leaves": {"type": "int", "low": 55, "high": 200, "log": False, "step": 5},
        "max_depth": {"type": "int", "low": 5, "high": 12, "log": False, "step": 1},
        "min_child_samples": {"type": "int", "low": 200, "high": 600, "log": False, "step": 10},
        "feature_fraction": {"type": "float", "low": 0.58, "high": 0.75, "log": False, "step": 0.01},
        "bagging_fraction": {"type": "float", "low": 0.56, "high": 0.75, "log": False, "step": 0.005},
        "bagging_freq": {"type": "int", "low": 10, "high": 25, "log": False, "step": 1},
        "reg_alpha": {"type": "float", "low": 8.0, "high": 20.0, "log": False, "step": 0.1},
        "reg_lambda": {"type": "float", "low": 6.0, "high": 20.0, "log": False, "step": 0.1},
        "min_split_gain": {"type": "float", "low": 0.12, "high": 0.30, "log": False, "step": 0.005},
        "early_stopping_rounds": {"type": "int", "low": 500, "high": 1200, "log": False, "step": 10},
        "path_smooth": {"type": "float", "low": 0.10, "high": 0.60, "log": False, "step": 0.005},
        "cat_smooth": {"type": "float", "low": 20.0, "high": 40.0, "log": False, "step": 0.1},
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
        early_stopping_rounds = model_params.pop("early_stopping_rounds", 100)

        # Create model with remaining parameters
        model = create_model(model_params)

        # Create eval set for early stopping
        eval_set = [(X_eval, y_eval)]

        # Fit model with early stopping
        model.fit(
            X_combined,
            y_combined,
            eval_set=eval_set,
            callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds)],
        )

        # Get validation predictions
        best_threshold, metrics = optimize_threshold(model, X_eval, y_eval, min_recall=min_recall)

        return model, metrics

    except Exception as e:
        logger.error(f"Error training LightGBM model: {str(e)}")
        raise


def optimize_hyperparameters(
    X_train, y_train, X_test, y_test, X_eval, y_eval, hyperparameter_space
):
    logger.info("Starting hyperparameter optimization")

    if not hyperparameter_space:
        hyperparameter_space = load_hyperparameter_space()

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

            if score > 0.37 and score > best_score:
                log_to_mlflow(model, metrics, params, experiment_name)
            return score

        except Exception as e:
            logger.error(f"Trial failed: {str(e)}")
            return 0.0

    # Callback function defined outside the loop so that its modifications affect the outer scope.
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
        return best_score

    # Set persistent storage path using SQLite
    storage_url = "sqlite:///optuna_lightgbm.db"
    study_name = "lightgbm_optimization"
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
        study.optimize(objective, n_trials=batch_size, show_progress_bar=True, callbacks=[callback])

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
            signature = mlflow.models.infer_signature(input_example, model.predict(input_example))

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
        logger.info("Training model with precision target")
        params = base_params.copy()
        params.update(
            {
                "learning_rate": 0.10250000000000001,
                "num_leaves": 120,
                "max_depth": 5,
                "min_child_samples": 440,
                "feature_fraction": 0.7,
                "bagging_fraction": 0.625,
                "bagging_freq": 15,
                "reg_alpha": 12.0,
                "reg_lambda": 9.5,
                "min_split_gain": 0.13,
                "path_smooth": 0.155,
                "cat_smooth": 32.5,
                "max_bin": 660,
                "device": "cpu",
                "metric": ["aucpr", "binary_logloss"],
                "n_jobs": 8,
                "objective": "binary",
                "random_state": 19,
                "verbose": -1,
            }
        )

        # Train final model with best parameters
        logger.info("Training final model with best parameters")
        model, metrics = train_model(X_train, y_train, X_test, y_test, X_eval, y_eval, params)
        compute_permutation_importance(model, X_eval, y_eval, metrics['threshold'])
        # Log to MLflow
        # log_to_mlflow(model, metrics, params, experiment_name)

        return model, metrics

    except Exception as e:
        logger.error(f"Error in precision-focused training: {str(e)}")
        return None, None

def compute_permutation_importance(
    model,
    X_val: pd.DataFrame, 
    y_val: np.ndarray,
    threshold: float = 0.3,
    n_repeats: int = 50,
    number_of_features: int = 100,
) -> pd.DataFrame:
    """
    Compute permutation feature importance for a given metric and threshold.
    Args:
        model: Trained model with predict_proba(X) method.
        X_val: Validation features (DataFrame).
        y_val: Validation labels (array-like).
        metric: Metric function (e.g., sklearn.metrics.precision_score).
        threshold: Threshold for positive class prediction.
        n_repeats: Number of shuffles per feature.
        random_state: Seed for reproducibility.
    Returns:
        DataFrame with columns: ['feature', 'importance'] (mean drop in metric), sorted descending.
    """
    feature_names = X_val.columns.tolist()
    y_val_np = y_val.values
    # Compute baseline metric
    probs = model.predict_proba(X_val)[:, 1]
    preds = (probs >= threshold).astype(int)
    # Fix: metric is being passed as a float value instead of a function
    # We'll calculate precision directly since that's what was passed in
    baseline = np.sum((y_val_np == 1) & (preds == 1)) / (np.sum(preds == 1))
    logger.info(f"Baseline metric: {baseline:.4f}")
    importances = []
    for feat in feature_names:
        drops = []
        for i in range(n_repeats):
            logger.info(f"Shuffling feature: {feat} - Repeat: {i+1}")
            X_shuffled = X_val.copy()
            X_shuffled[feat] = np.random.permutation(X_shuffled[feat].values)
            probs_shuffled = model.predict_proba(X_shuffled)[:, 1]
            preds_shuffled = (probs_shuffled >= threshold).astype(int)
            # Calculate precision directly instead of using metric parameter
            precision = np.sum((y_val_np == 1) & (preds_shuffled == 1)) / (np.sum(preds_shuffled == 1))
            drop = baseline - precision
            drops.append(drop)
        mean_drop = np.mean(drops)
        importances.append((feat, mean_drop))
        logger.debug(f"Feature: {feat}, Mean drop: {mean_drop:.4f}")
    # Sort by importance descending
    importances.sort(key=lambda x: x[1], reverse=True)
    df_importance = pd.DataFrame(importances, columns=["feature", "importance"])
    logger.info("Top features by permutation importance:")
    logger.info(df_importance.head(number_of_features).to_string(index=False))
    return df_importance

def select_features_rfecv(X, y, logger, min_features=150, step=1, scoring='roc_auc', random_state=19):
    """
    Perform RFECV-based feature selection using LightGBM.
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series or np.ndarray): Target vector
        logger: Logger instance
        min_features (int): Minimum number of features to select
        step (int): Number of features to remove at each iteration
        scoring (str): Scoring metric for cross-validation
        random_state (int): Random seed for reproducibility
    Returns:
        tuple: (List[str], pd.DataFrame)
    """
    logger.info(f"Starting RFECV feature selection with min_features={min_features}, step={step}, scoring={scoring}")
    params = base_params.copy()
    params.update(
        {
            "learning_rate": 0.10250000000000001,
            "num_leaves": 120,
            "max_depth": 5,
            "min_child_samples": 440,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.625,
            "bagging_freq": 15,
            "reg_alpha": 12.0,
            "reg_lambda": 9.5,
            "min_split_gain": 0.13,
            "path_smooth": 0.155,
            "cat_smooth": 32.5,
            "max_bin": 660,
            "device": "cpu",
            "metric": ["aucpr", "binary_logloss"],
            "n_jobs": 8,
            "objective": "binary",
            "random_state": 19,
            "verbose": -1,
        }
    )
    estimator = lgb.LGBMClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    selector = RFECV(
        estimator=estimator,
        step=step,
        cv=cv,
        scoring=scoring,
        min_features_to_select=min_features,
        n_jobs=-1,
        verbose=2
    )
    selector.fit(X, y)
    selected_features = X.columns[selector.support_].tolist()
    importances = selector.estimator_.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': selected_features,
        'importance': importances
    }).sort_values('importance', ascending=False)
    logger.info(f"RFECV selected {len(selected_features)} features:")
    for feat, imp in zip(feature_importance_df['feature'], feature_importance_df['importance']):
        logger.info(f"  - {feat}: {imp}")
    return selected_features, feature_importance_df

def hypertune_with_feature_importance(X_train, y_train, X_test, y_test, X_eval, y_eval, n_trials=50):
    """
    Perform hyperparameter optimization with Optuna while tracking feature importances.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels 
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
        n_trials (int): Number of optimization trials
        
    Returns:
        tuple: (best_params, feature_importance_df)
    """
    logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
    
    # Store feature importances across trials
    feature_importances = []
    hyperparameter_space = load_hyperparameter_space()
    def objective(trial):
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

        
        # Train model
        model, metrics = train_model(X_train, y_train, X_test, y_test, X_eval, y_eval, params)
        
        # Store feature importances for this trial
        importance_dict = dict(zip(X_train.columns, model.feature_importances_))
        feature_importances.append(importance_dict)
        
        return metrics['precision']
    
    # Create and run study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    # Calculate average feature importance across all trials
    avg_importances = {}
    for feature in X_train.columns:
        importance_values = [trial_imp[feature] for trial_imp in feature_importances]
        avg_importances[feature] = np.mean(importance_values)
    
    # Create DataFrame and sort by importance
    importance_df = pd.DataFrame({
        'feature': list(avg_importances.keys()),
        'importance': list(avg_importances.values())
    })
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # Get top 100 features
    top_100_features = importance_df.head(100)
    
    logger.info("Top 100 features by average importance across trials:")
    for idx, row in top_100_features.iterrows():
        logger.info(f"{row['feature']}: {row['importance']:.4f}")
    
    return study.best_params, importance_df


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
        features = import_selected_features_ensemble_new(model_type="all")
        
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

        # --- Feature Selection with RFECV ---
        # selected_features, feature_importance_df = select_features_rfecv(X_eval, y_eval, logger, min_features=150, step=1, scoring='roc_auc', random_state=SEED)
        # print(feature_importance_df)

        # --- Hyperparameter Optimization with Feature Importance ---
        # best_params, importance_df = hypertune_with_feature_importance(
        #     X_train, y_train, X_test, y_test, X_eval, y_eval, n_trials=100
        # )

        logger.info("Starting hyperparameter optimization run")
        current_params, current_metrics = hypertune_lightgbm(experiment_name)
        logger.info(f"Run completed with parameters: {current_params}")

        # Train model with precision target
        # best_model, best_metrics = train_with_precision_target(
        #     X_train, y_train, X_test, y_test, X_eval, y_eval
        # )

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")


if __name__ == "__main__":
    main()
