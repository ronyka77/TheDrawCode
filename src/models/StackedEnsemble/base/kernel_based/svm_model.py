"""
SVM Model for Soccer Draw Prediction (using scikit-learn SVC)

This module implements a Support Vector Machine (SVM) based model for predicting
soccer match draws. It includes functionality for model creation, training,
hyperparameter optimization (using Optuna), threshold tuning, MLflow integration,
and crucially, data scaling which is essential for SVM performance.

The implementation focuses on high precision while maintaining a minimum recall threshold.
"""

import gc
import os
import pickle  # For saving the scaler
import random
import sys
import time
from datetime import datetime

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler  # Or RobustScaler
from sklearn.svm import SVC

from src.models.StackedEnsemble.shared.data_loader import DataLoader
from src.models.StackedEnsemble.shared.hypertuner_utils import optimize_threshold
from src.utils.create_evaluation_set import (
    import_selected_features_ensemble,
    setup_mlflow_tracking,
)

# Import project utilities
from src.utils.logger import ExperimentLogger

# --- Global Settings & Initialization ---
experiment_name = "svm_soccer_prediction"
logger = ExperimentLogger(experiment_name)
mlrunds_dir = setup_mlflow_tracking(experiment_name)

min_recall = 0.30  # Minimum acceptable recall (Adjust as needed)
n_trials = 10000  # Number of Optuna trials (Adjust as needed)

# Get current versions
sklearn_version = sklearn.__version__

# Create explicit pip requirements list
pip_requirements = [
    f"scikit-learn=={sklearn_version}",
    f"mlflow=={mlflow.__version__}",
    f"optuna=={optuna.__version__}",
    "pandas",
    "numpy",
]

# Set fixed seed and hash seed for determinism
SEED = 19
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Restrict parallel threads (Less critical for SVM but good practice)
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

# Base parameters for SVC
base_params = {
    "probability": True,  # MUST be True for predict_proba
    "class_weight": "balanced",  # Good for imbalanced classes
    "random_state": SEED,
    'kernel': 'rbf', # Can be fixed here or tuned
    "verbose": False,  # Set to True for more SVC logs
}


# --- Data Preprocessing ---
def preprocess_data(X_train, X_test, X_eval=None):
    try:
        with open('src/models/scalers/scaler_svm.pkl', 'rb') as f:
            scaler = pickle.load(f)
        logger.info("Loaded existing SVM scaler")
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_eval_scaled = scaler.transform(X_eval)
    except Exception as e:
        logger.error(f"Error loading SVM scaler: {str(e)}")
        scaler = StandardScaler()
        logger.info("Created new SVM scaler")
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_eval_scaled = scaler.transform(X_eval)
        with open('src/models/scalers/scaler_svm.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
    return X_train_scaled, X_test_scaled, X_eval_scaled, scaler


# --- Hyperparameter Space ---
def load_hyperparameter_space_svm():
    """
    Defines the hyperparameter search space for SVC tuning using Optuna.
    Focuses on 'rbf' kernel initially.
    Returns:
        dict: Hyperparameter space configuration.
    """
    hyperparameter_space = {
        "C": {
            "type": "float",
            "low": 1e-2,  # Adjusted range for C
            "high": 1e2,
            "log": True,
        },
        "gamma": {
            "type": "float",
            "low": 1e-4,  # Adjusted range for gamma
            "high": 1e0,
            "log": True,
        },
        "cache_size": {"type": "int", "low": 1000, "high": 10000, "step": 100},
        # 'kernel': {'type': 'categorical', 'choices': ['rbf', 'poly', 'sigmoid']},
        'degree': {'type': 'int', 'low': 2, 'high': 5}, # Only if kernel='poly'
        'coef0': {'type': 'float', 'low': 0.0, 'high': 1.0}, # Only if kernel='poly' or 'sigmoid'
        "tol": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
    }
    return hyperparameter_space


# --- Model Training ---
def train_model_svm(X_train_scaled, y_train, X_eval_scaled, y_eval, model_params):
    """
    Trains an SVC model and evaluates using optimize_threshold.
    Args:
        X_train_scaled: Scaled training features.
        y_train: Training labels.
        X_eval_scaled: Scaled evaluation features.
        y_eval: Evaluation labels.
        model_params (dict): Model parameters including C, gamma, kernel etc.
    Returns:
        tuple: (trained_model, metrics)
    """
    try:
        # Combine base and suggested params
        full_params = base_params.copy()
        full_params.update(model_params)

        logger.info(f"Training SVC with parameters: {full_params}")
        model = SVC(**full_params)

        # Fit model
        model.fit(X_train_scaled, y_train)
        logger.info("SVC fitting complete.")

        # Evaluate using the shared threshold optimization logic
        best_threshold, metrics = optimize_threshold(
            model, X_eval_scaled, y_eval, min_recall=min_recall
        )
        logger.info(f"Threshold optimization complete. Metrics: {metrics}")

        return model, metrics

    except Exception as e:
        logger.error(f"Error training SVC model: {str(e)}")
        raise


# --- Optuna Objective ---
def objective(trial, X_train, y_train, X_test, y_test, X_eval, y_eval, hyperparameter_space):
    """
    Objective function for Optuna hyperparameter optimization.
    Uses SCALED data.
    """
    try:
        params = {}
        # Suggest hyperparameters
        for param_name, param_config in hyperparameter_space.items():
            if param_config["type"] == "float":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=param_config.get("log", False),
                )
            elif param_config["type"] == "int":
                params[param_name] = trial.suggest_int(
                    param_name, param_config["low"], param_config["high"]
                )
            elif param_config["type"] == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name, param_config["choices"]
                )
            # --- Preprocessing --- (Scale the data)
        X_train_scaled, X_test_scaled, X_eval_scaled, scaler = preprocess_data(
            X_train, X_test, X_eval
        )
        # Train model and get metrics
        # Pass only the suggested params, train_model_svm combines with base_params
        model, metrics = train_model_svm(
            X_train_scaled, y_train, X_eval_scaled, y_eval, params
        )

        recall = metrics.get("recall", 0.0)
        precision = metrics.get("precision", 0.0)
        threshold = metrics.get("threshold", 0.5)

        # Optimize for precision while maintaining minimum recall
        score = precision if recall >= min_recall else 0.0

        # Log metrics to trial attributes
        for metric_name, metric_value in metrics.items():
            trial.set_user_attr(metric_name, metric_value)

        logger.info(
            f"Trial {trial.number}: Score={score:.4f} (Precision={precision:.4f}, Recall={recall:.4f}, Thresh={threshold:.3f}) Params={trial.params}"
        )
        # Log to MLflow
        if score > 0.35:
            input_example = X_eval[:5]
            log_to_mlflow_svm(model, metrics, params, scaler, input_example)
        return score

    except Exception as e:
        logger.error(f"Optuna trial {trial.number} failed: {e}")
        return 0.0  # Return low score for failed trials


# --- Optuna Optimization Runner ---
def optimize_hyperparameters_svm(
    X_train, y_train, X_test, y_test, X_eval, y_eval, hyperparameter_space
):
    """Runs Optuna optimization using the objective function and SCALED data."""
    logger.info(f"Starting Optuna optimization for SVC with {n_trials} trials.")

    best_score = -float("inf")
    best_params_from_hpo = {}
    global_top_trials = []
    top_trials = []  # Track top trials within a batch

    # Wrapper for objective function to pass scaled data
    objective_func = lambda trial: objective(
        trial, X_train, y_train, X_test, y_test, X_eval, y_eval, hyperparameter_space
    )

    # --- Callback Logic --- (Similar to xgboost/lightgbm)
    def callback(study, trial):
        nonlocal best_score, best_params_from_hpo, top_trials, global_top_trials
        if trial.value is not None and trial.state == optuna.trial.TrialState.COMPLETE:
            # Update overall best score if current trial is better
            if trial.value > best_score:
                best_score = trial.value
                best_params_from_hpo = trial.params
                logger.info(
                    f"New best score found in trial {trial.number}: {best_score:.4f} with params {best_params_from_hpo}"
                )

            # Update top trials list for the current batch/overall
            current_run_metrics = trial.user_attrs
            current_run_score = trial.value
            current_run_params = trial.params
            current_run_number = trial.number
            trial_record = (
                current_run_score,
                current_run_params,
                current_run_number,
                current_run_metrics.get("precision", 0.0),
                current_run_metrics.get("recall", 0.0),
            )
            # Add to overall list
            global_top_trials.append(trial_record)
            global_top_trials.sort(
                key=lambda x: x[0] if x[0] is not None else float("-inf"), reverse=True
            )
            global_top_trials[:] = global_top_trials[:10]  # Keep only overall top 10

            # Log top trials periodically
            if trial.number > 0 and trial.number % 50 == 0:
                log_top_trials_svm(global_top_trials, "Current Overall Top 10 Trials")
        elif trial.state != optuna.trial.TrialState.COMPLETE:
            logger.warning(
                f"Trial {trial.number} did not complete successfully. State: {trial.state}"
            )

    # --- Study Execution --- (Similar batching logic)
    storage_url = "sqlite:///optuna_svm.db"
    study_name = "svm_optimization"
    total_trials = n_trials
    batch_size = 500  # Adjust batch size based on expected trial duration
    num_batches = max(1, total_trials // batch_size)
    if total_trials % batch_size != 0 and total_trials > batch_size:
        num_batches += 1

    for batch in range(num_batches):
        random_seed = int(time.time()) + batch
        # Consider TPESampler if random search is too slow or gets stuck
        sampler = optuna.samplers.RandomSampler(seed=random_seed)
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            storage=storage_url,
            load_if_exists=True,
            sampler=sampler,
        )
        logger.info(
            f"Starting Optuna batch {batch+1}/{num_batches} (Sampler: {type(sampler).__name__})"
        )

        study.optimize(
            objective_func, 
            n_trials=batch_size, 
            show_progress_bar=True, 
            callbacks=[callback],
            n_jobs=8  # Use all available CPU cores for parallel trials
        )

        # Update overall best score from the study instance after batch
        if study.best_trial and study.best_value > best_score:
            best_score = study.best_value
            best_params_from_hpo = study.best_params
            logger.info(f"Batch {batch+1} completed. New overall best score: {best_score:.4f}")

    logger.info("Optuna optimization finished for SVM.")
    if best_params_from_hpo:
        logger.info(f"Best HPO parameters found: {best_params_from_hpo}")
        log_top_trials_svm(global_top_trials, "Final Top 10 Trials (SVM)")
    else:
        logger.warning("Optuna did not find any successful trials for SVM.")

    return best_params_from_hpo


def log_top_trials_svm(trials_list, title="Top SVM Trials"):
    """Helper function to log top trials in a formatted table."""
    if not trials_list:
        logger.info(f"{title}: No successful trials to log.")
        return

    table_header = "| Rank | Trial # | Score  | Precision | Recall | Parameters |"
    table_separator = "|------|---------|--------|-----------|--------|------------|"

    logger.info(f"{title}:")
    logger.info(table_header)
    logger.info(table_separator)
    for i, rec in enumerate(trials_list):
        score, params, number, precision, recall = rec
        score_str = f"{score:.4f}" if score is not None else "N/A"
        prec_str = f"{precision:.4f}" if precision is not None else "N/A"
        rec_str = f"{recall:.4f}" if recall is not None else "N/A"
        logger.info(f"| {i+1} | {number} | {score_str} | {prec_str} | {rec_str} | {params} |")


# --- MLflow Logging ---
def log_to_mlflow_svm(model, metrics, params, fitted_scaler, input_example):
    """Logs parameters, metrics, scaler, and model to MLflow."""
    logger.info("Logging results to MLflow for SVM...")
    with mlflow.start_run(
        run_name=f"svm_fixed_params_{datetime.now().strftime('%Y%m%d_%H%M')}"
    ):
        run_id = mlflow.active_run().info.run_id

        # Log combined final parameters used for the model
        final_params = base_params.copy()
        final_params.update(params)
        mlflow.log_params(final_params)

        # Log evaluation metrics
        mlflow.log_metrics(metrics)

        # Log the scaler
        scaler_path = "src/models/scalers/scaler_svm.pkl"
        try:
            with open(scaler_path, "wb") as f:
                pickle.dump(fitted_scaler, f)
            mlflow.log_artifact(scaler_path)
            logger.info(f"Scaler artifact logged as {scaler_path}")
            os.remove(scaler_path) # Clean up local scaler file
        except Exception as e:
            logger.error(f"Failed to save or log scaler artifact: {e}")

        # Log the model using mlflow.sklearn
        logger.info("Logging model using mlflow.sklearn...")
        signature = None
        if input_example is not None:
            try:
                # Ensure dtypes are float for numeric cols
                num_cols = input_example.select_dtypes(include=np.number).columns
                input_example[num_cols] = input_example[num_cols].astype('float64')
                logger.info("Created input_example from DataFrame for signature.")

                model_prediction = model.predict(input_example)
                signature = mlflow.models.infer_signature(input_example, model_prediction)
                logger.info("Model signature inferred successfully.")

            except Exception as sig_err:
                logger.error(f"Error inferring model signature: {sig_err}")
                logger.error(f"Input example details:\n{input_example.head()}\n{input_example.dtypes}")

        # Define registered model name
        model_name_suffix = datetime.now().strftime("%Y%m%d_%H%M")
        registered_model_name = f"model_svm_{model_name_suffix}"

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model_svm",
            registered_model_name=registered_model_name,
            signature=signature,
            pip_requirements=pip_requirements,
        )

        logger.info(f"MLflow Run ID: {run_id}")
        logger.info(f"Scikit-learn SVM model logged successfully as {registered_model_name}.")
        mlflow.end_run()

# --- Main Hypertuning Orchestration ---
def hypertune_svm(X_train, y_train, X_test, y_test, X_eval, y_eval, experiment_name: str):
    """
    Orchestrates the SVC tuning (Optuna), evaluation, and logging process.
    Handles data scaling internally.
    """
    try:
        logger.info("--- Starting hypertuning process for SVM ---")
        # --- Hyperparameter Optimization (Optuna) ---
        hyperparameter_space = load_hyperparameter_space_svm()
        best_hpo_params = optimize_hyperparameters_svm(
            X_train,
            y_train,
            X_test,
            y_test,
            X_eval,
            y_eval,
            hyperparameter_space,
        )

        if not best_hpo_params:
            logger.error("Optuna optimization failed for SVM. Aborting.")
            return None, None
        return best_hpo_params, final_metrics

    except Exception as e:
        logger.error(f"Error in hypertune_svm: {str(e)}")
        return None, None

# --- (Optional) Precision Target Training ---
def train_with_precision_target_svm(
    X_train, y_train, X_test, y_test, X_eval, y_eval, experiment_name: str
):
    """
    Train SVM model with fixed parameters, focusing on precision target.
    Handles data scaling internally.
    """
    try:
        logger.info("--- Training SVM model with fixed precision-target parameters ---")

        # --- Preprocessing --- (Scale the data)
        X_train_scaled, X_test_scaled, X_eval_scaled, fitted_scaler = preprocess_data(
            X_train, X_test, X_eval
        )

        # Define fixed parameters (Update these based on prior tuning or best guess)
        fixed_params = {
            'C': 10.0,
            'gamma': 0.01,
            'tol': 0.001,
            # Add other relevant fixed params like kernel if not 'rbf'
        }
        model_params = base_params.copy()
        model_params.update(fixed_params)
        logger.info(f"Using fixed SVM parameters: {model_params}")

        # Combine scaled training and test sets
        logger.info("Combining scaled training and test sets...")
        if isinstance(X_train_scaled, pd.DataFrame) and isinstance(X_test_scaled, pd.DataFrame):
            X_train_combined_scaled = pd.concat([X_train_scaled, X_test_scaled], ignore_index=True)
        else: # Assume numpy arrays
            X_train_combined_scaled = np.vstack((X_train_scaled, X_test_scaled))
        y_train_combined = pd.concat([y_train, y_test], ignore_index=True) if isinstance(y_train, pd.Series) else np.concatenate((y_train, y_test))
        logger.info(f"Combined scaled training set shape: {X_train_combined_scaled.shape}")

        # --- Create and Train Model ---
        with mlflow.start_run(
            run_name=f"svm_fixed_params_{datetime.now().strftime('%Y%m%d_%H%M')}"
        ):
            mlflow.set_tags(
                {
                    "model_type": "svm",
                    "training_mode": "fixed_params",
                    "scaling_method": type(fitted_scaler).__name__,
                }
            )

            model = SVC(**model_params)
            logger.info("Fitting final model on combined scaled train/test data...")
            model.fit(X_train_combined_scaled, y_train_combined)

            # --- Evaluate (on separate evaluation set) ---
            logger.info("Evaluating final model on the scaled evaluation set...")
            best_threshold, metrics = optimize_threshold(
                model, X_eval_scaled, y_eval, min_recall=min_recall
            )
            metrics["threshold"] = best_threshold
            logger.info(f"Final evaluation metrics (SVM Fixed Params): {metrics}")

            # --- Log to MLflow ---
            input_example_data = X_eval_scaled[:5]
            log_to_mlflow_svm(
                model, metrics, fixed_params, fitted_scaler, input_example_data
            )

            return model, metrics

    except Exception as e:
        logger.error(f"Error in precision-focused SVM training: {str(e)}")
        if mlflow.active_run():
            mlflow.end_run("FAILED")
        return None, None


# --- Main Execution Block ---
if __name__ == "__main__":
    try:
        logger.info("--- Starting SVM Model Training/Tuning --- ")

        # Load data
        dataloader = DataLoader()
        X_train, y_train, X_test, y_test, X_eval, y_eval = dataloader.load_data()

        # Select features
        try:
            # Attempt to load features specific to SVM if defined
            features = import_selected_features_ensemble(model_type="mlp")
            logger.info(f"Using 'mlp' specific feature set with {len(features)} features.")
        except (KeyError, FileNotFoundError):
            logger.warning("SVM specific features not found. Falling back to 'all' features.")
            features = import_selected_features_ensemble(model_type="all")
            if not features:
                logger.error("Failed to load any features. Exiting.")
                sys.exit(1) # Or handle differently

        X_train = X_train[features]
        X_test = X_test[features]
        X_eval = X_eval[features]

        # Ensure data types are appropriate (DataLoader might handle this)
        # Convert if necessary, e.g., all to float64 for scaler
        X_train = X_train.astype("float64")
        X_test = X_test.astype("float64")
        X_eval = X_eval.astype("float64")

        # Log data shapes
        logger.info(f"Training data shape after selection: {X_train.shape}")
        logger.info(f"Testing data shape after selection: {X_test.shape}")
        logger.info(f"Evaluation data shape after selection: {X_eval.shape}")
        logger.info(
            f"Positive class ratio - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}, Eval: {y_eval.mean():.3f}"
        )

        # --- Choose Mode: hypertune or fixed params ---
        # mode = "hypertune"
        mode = "hypertune" # Or "hypertune"

        best_model_params = None
        final_metrics = None

        if mode == "hypertune":
            logger.info("--- Running Hyperparameter Tuning for SVM ---")
            best_model_params, final_metrics = hypertune_svm(
                X_train, y_train, X_test, y_test, X_eval, y_eval, experiment_name
            )
        elif mode == "fixed_params":
            logger.info("--- Running Fixed Parameter Training for SVM ---")
            # Note: train_with_precision_target_svm returns model, metrics
            # We don't get the 'best_params' back in this mode directly
            final_model, final_metrics = train_with_precision_target_svm(
                X_train, y_train, X_test, y_test, X_eval, y_eval, experiment_name
            )
            if final_model:
                best_model_params = final_model.get_params() # Get params from trained model
        else:
            logger.error(f"Invalid mode selected: {mode}")

        # --- Log Final Results Summary ---
        if best_model_params and final_metrics:
            logger.info(f"SVM process (mode: {mode}) completed successfully.")
            logger.info(f"Final Run Metrics: {final_metrics}")
            logger.info(f"Parameters Used/Found: {best_model_params}")
        else:
            logger.error(f"SVM process (mode: {mode}) failed or did not produce results.")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
    finally:
        gc.collect()  # Force garbage collection
        logger.info("--- SVM Main execution finished. ---")