"""
MLP Model (scikit-learn) for Soccer Draw Prediction

This module implements a simpler MLP-based model using scikit-learn's MLPClassifier
for predicting soccer match draws. It includes Optuna for hyperparameter
tuning (optimizing precision subject to min recall) and MLflow integration,
following a structure similar to other base models.
"""

import os
import sys
import pickle
import random
import time
import gc
import joblib
import logging
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn # Use sklearn logging
import optuna # Import Optuna
from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import RandomizedSearchCV # Remove RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import loguniform, randint, uniform # For parameter distributions

# Set project root (similar to xgboost_model.py)
try:
    project_root = Path(__file__).parent.parent.parent.parent.parent
    if not project_root.exists():
        project_root = Path(r"\\".join(str(project_root).split("\\")))
    sys.path.append(str(project_root))
    print(f"Project root mlp_sklearn_model: {project_root}")
except Exception as e:
    print(f"Error setting project root path: {e}")
    sys.path.append(os.getcwd().parent)
    print(f"Current directory mlp_sklearn_model: {os.getcwd().parent}")

# Configure Git executable path if available
git_executable = os.environ.get("GIT_PYTHON_GIT_EXECUTABLE")
if git_executable and os.path.exists(git_executable):
    import git
    git.refresh(git_executable)

from utils.logger import ExperimentLogger
experiment_name = "mlp_sklearn_optuna_soccer_prediction" # Updated experiment name
logger = ExperimentLogger(experiment_name=experiment_name)

# Set random seeds for reproducibility
random_seed = 19
random.seed(random_seed)
np.random.seed(random_seed)
os.environ['PYTHONHASHSEED'] = str(random_seed)

# Import shared utility for threshold optimization
from models.StackedEnsemble.shared.hypertuner_utils import optimize_threshold
from utils.create_evaluation_set import setup_mlflow_tracking, import_selected_features_ensemble
mlflow_tracking = setup_mlflow_tracking(experiment_name)
from models.StackedEnsemble.shared.data_loader import DataLoader # Custom data loader

# Global settings
n_trials_optuna = 100 # Number of Optuna trials
min_recall = 0.30  # Minimum acceptable recall for optimization objective
pip_requirements = [
    "scikit-learn",
    f"mlflow=={mlflow.__version__}",
    f"optuna=={optuna.__version__}" # Add Optuna
]
scaler = None  # Global scaler object - Will be fitted in hypertune function
X_train, y_train, X_test, y_test, X_eval, y_eval = (None,) * 6 # Global data variables
# --- Base Parameters ---

base_params = {
    # 'hidden_layer_sizes': (128,), # REMOVED - Will be set dynamically in objective
    'activation': 'relu',             # ReLU activation function
    'solver': 'adam',    
    'early_stopping': True,           # Enable early stopping
    'random_state': random_seed,      # For reproducibility
    'verbose': False,                # Minimize output
}

# --- Parameter Space Function (Used by Optuna) ---
def load_hyperparameter_space_sklearn():
    """Defines the hyperparameter search space configuration for Optuna."""
    # This dictionary now defines the *structure* for Optuna's suggest methods
    # Actual suggestion happens within the objective function
    hyperparameter_space = {
        'alpha': {
            'type': 'float',
            'low': 1e-7,
            'high': 1.0,
            'log': True
        },
        'learning_rate_init': {
            'type': 'float', 
            'low': 1e-6,
            'high': 1e-1,
            'log': True
        },
        'batch_size': {
            'type': 'int',
            'low': 256,
            'high': 4096,
            'log': False,
            'step': 64
        },
        'max_iter': {
            'type': 'int',
            'low': 50,
            'high': 200,
            'log': False,
            'step': 5
        },
        'n_iter_no_change': {
            'type': 'int',
            'low': 3,
            'high': 30
        },
        'beta_1': {
            'type': 'float',
            'low': 0.75,
            'high': 0.95,
            'log': False,
            'step': 0.01
        },
        'beta_2': {
            'type': 'float',
            'low': 0.98,
            'high': 0.9999,
            'log': False,
            'step': 0.0001
        }
    }
    return hyperparameter_space

# --- Architecture Rotation Definition ---
# Define the fixed sequence of architectures to cycle through
ARCHITECTURE_ROTATION = [
    (128, 64),  
    (128, 64, 32)  
]
logger.info(f"Defined architecture rotation: {ARCHITECTURE_ROTATION}")
# -------------------------------------

def preprocess_data(X_train_local, X_test_local, X_eval_local):
    """
    Preprocess data using StandardScaler. Fits scaler on train, transforms all.
    Uses local copies of data passed in.
    Returns: scaled X_train, X_test, X_eval, and the fitted scaler.
    """
    local_scaler = StandardScaler()
    X_train_scaled = local_scaler.fit_transform(X_train_local)
    X_test_scaled = local_scaler.transform(X_test_local)
    X_eval_scaled = local_scaler.transform(X_eval_local)
    logger.info("StandardScaler fitted and data transformed.")
    return X_train_scaled, X_test_scaled, X_eval_scaled, local_scaler

# --- Model Creation ---
def create_model_sklearn(model_params):
    """Creates an MLPClassifier instance with given parameters."""
    # base_params no longer contains hidden_layer_sizes
    # Ensure model_params passed in *always* contains hidden_layer_sizes
    params = base_params.copy() # Get fixed params like activation, solver etc.
    params.update(model_params) # Add tuned params AND the dynamically set hidden_layer_sizes
    try:
        model = MLPClassifier(**params)
        return model
    except Exception as e:
        logger.error(f"Error creating MLPClassifier model: {str(e)}")
        raise

# --- Model Training ---
def train_model_sklearn(X_train_scaled, y_train_local, X_eval_scaled, y_eval_local, model_params):
    """Trains an MLPClassifier model and evaluates using optimize_threshold."""
    try:
        # Create the model using the dedicated function
        model = create_model_sklearn(model_params)
        # Train the model
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
            model.fit(X_train_scaled, y_train_local)
        # Evaluate using the shared threshold optimization logic
        best_threshold, metrics = optimize_threshold(
            model, X_eval_scaled, y_eval_local, min_recall=min_recall
        )
        return model, metrics # Return fitted model and metrics
    except Exception as e:
        logger.error(f"Error training MLPClassifier model: {str(e)}")
        # Reraise to be caught by Optuna trial exception handler
        raise

# --- Helper Function for Logging Top Trials ---
def log_top_trials(trials_list, title="Top Trials"):
    """Helper function to log top trials in a formatted table."""
    table_header = "| Rank | Trial # | Score  | Parameters |"
    table_separator = "|------|---------|--------|------------|"
    
    logger.info(f"{title}:")
    table_rows = [f"| {i+1} | {rec[2]} | {rec[0]:.4f} | {rec[1]} |" for i, rec in enumerate(trials_list[:10])]
    logger.info(table_header)
    logger.info(table_separator)
    for row in table_rows:
        logger.info(row)

def optimize_hyperparameters_optuna_sklearn(X_train_scaled, y_train_local, X_eval_scaled, y_eval_local):
    """Runs Optuna optimization using the structured train_model function.""" # Docstring updated
    logger.info(f"Starting Optuna optimization with {n_trials_optuna} trials.")
    hyperparameter_space_config = load_hyperparameter_space_sklearn()
    # Variables to track best score and params within this optimization run
    best_score = -float('inf')
    best_params_in_run = {}
    top_trials = [] # List to store top trial info (score, params, number, precision, recall)

    # Define Objective function for Optuna
    def objective(trial):
        params = {}
        # --- Determine Architecture Dynamically ---
        trial_num = trial.number
        num_archs = len(ARCHITECTURE_ROTATION)
        arch_index = (trial_num // 10) % num_archs # Change architecture every 10 trials
        current_hidden_layers = ARCHITECTURE_ROTATION[arch_index]
        params['hidden_layer_sizes'] = current_hidden_layers # Set forced architecture
        # logger.debug(f"Trial {trial_num}: Using architecture {current_hidden_layers} (Index {arch_index})") # Optional debug log
        # ---------------------------------------

        # --- Suggest Other Hyperparameters ---
        for name, config in hyperparameter_space_config.items():
            # Ensure we DON'T suggest hidden_layer_sizes if it accidentally exists in config
            if name == 'hidden_layer_sizes': continue
            if config['type'] == 'int':
                step = config.get('step', 1)
                params[name] = trial.suggest_int(name, config['low'], config['high'], step=step)
            elif config['type'] == 'float':
                log = config.get('log', False)
                step = config.get('step') # Can be None
                if step:
                    params[name] = trial.suggest_float(name, config['low'], config['high'], step=step, log=log)
                else:
                    params[name] = trial.suggest_float(name, config['low'], config['high'], log=log)

        try:
            # Call the structured train and evaluate function
            # The 'params' dict now includes the dynamically set 'hidden_layer_sizes'
            model, metrics = train_model_sklearn(
                X_train_scaled, y_train_local,
                X_eval_scaled, y_eval_local,
                params # Pass suggested params
            )
            recall = metrics.get('recall', 0.0)
            precision = metrics.get('precision', 0.0)
            threshold = metrics.get('threshold', 0.5)
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
            logger.error(f"Optuna trial {trial.number} failed: {e}") # Keep log cleaner
            return 0.0 # Return low score for failed trials

    # --- Callback Function (similar structure to lightgbm) ---
    def callback(study, trial):
        nonlocal best_score, best_params_in_run, top_trials
        # logger.info(f"Current best score in run: {best_score:.4f}") # Optional verbosity
        if trial.value is not None:
            nonlocal best_score, best_params, top_trials, global_top_trials
            logger.info(f"Current best score in this batch: {best_score:.4f}")
            if trial.value > best_score:
                best_score = trial.value
                best_params = trial.params
                logger.info(f"New best score found in trial {trial.number}: {best_score:.4f}")
            # Create a record for the current trial
            current_run = (trial.value, trial.params, trial.number)
            # Ensure current_run has all required values before appending
            if all(x is not None for x in current_run):
                top_trials.append(current_run)
                # Sort and keep only top 10 for this batch
                top_trials.sort(key=lambda x: x[0] if x[0] is not None else float('-inf'), reverse=True)
                top_trials[:] = top_trials[:10]

                # Log top trials periodically
                if trial.number > 0 and trial.number % 10 == 0:
                    logger.info(f"Logging top 10 trials for trial {trial.number}")
                    if top_trials:  # Only log if we have trials to log
                        log_top_trials(top_trials, "Current Top 10 Trials")
            else:
                logger.warning(f"Skipping trial {trial.number} due to incomplete metrics")
            # -----------------------------
    # ------------------------------------------------------
    storage_url = "sqlite:///optuna_mlp.db"
    study_name = "mlp_optimization"
    total_trials = n_trials_optuna
    batch_size = 1000
    num_batches = total_trials // batch_size
    if total_trials % batch_size != 0:
        num_batches += 1

    global_top_trials = []
    best_score = float('-inf')
    best_params = {}

    for batch in range(num_batches):
        random_seed = int(time.time())
        new_sampler = optuna.samplers.RandomSampler(seed=random_seed)
        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',
            storage=storage_url,
            load_if_exists=True,
            sampler=new_sampler
        )
        logger.info(f"Starting batch {batch+1}/{num_batches} with new sampler (seed={random_seed})")
        study.optimize(objective, n_trials=batch_size, show_progress_bar=True, callbacks=[callback])
        for trial_record in top_trials:
            global_top_trials.append(trial_record)
        global_top_trials.sort(key=lambda x: x[0], reverse=True)
        global_top_trials = global_top_trials[:10]

    if global_top_trials:
        best_score, best_params, best_trial_number, _, _ = global_top_trials[0]
    else:
        best_params = {}

    logger.info(f"Best trial value across batches: {best_score:.4f}")
    logger.info(f"Best parameters found: {best_params}")

    # Log final top 10 trials
    log_top_trials(top_trials, "Final Top 10 Trials")

    # Return parameters tracked by the callback, falling back to study object if needed
    return best_params_in_run if best_params_in_run else study.best_params

def evaluate_model_sklearn(model, X_eval_scaled, y_eval_local):
    """Evaluates the trained scikit-learn model on the evaluation set (standard 0.5 threshold)."""
    logger.info("Evaluating model on the evaluation set (standard 0.5 threshold)...")
    y_pred_eval = model.predict(X_eval_scaled)
    y_proba_eval = model.predict_proba(X_eval_scaled)[:, 1] # Probabilities for class 1

    eval_precision = precision_score(y_eval_local, y_pred_eval, zero_division=0)
    eval_recall = recall_score(y_eval_local, y_pred_eval, zero_division=0)
    eval_f1 = f1_score(y_eval_local, y_pred_eval, zero_division=0)
    eval_auc = roc_auc_score(y_eval_local, y_proba_eval)

    metrics = {
        "std_eval_precision": eval_precision,
        "std_eval_recall": eval_recall,
        "std_eval_f1": eval_f1,
        "std_eval_auc": eval_auc
    }
    logger.info(f"Standard evaluation metrics (0.5 thresh): {metrics}")
    return metrics

def log_to_mlflow_sklearn(model, metrics, params, scaler_obj, input_example_df):
    """Logs parameters, metrics, scaler, and model to MLflow."""
    logger.info("Logging results to MLflow...")

    # Log best parameters found by optuna
    mlflow.log_params({f"best_optuna_{k}": v for k, v in params.items()})
    # Log evaluation metrics (including custom threshold metrics)
    mlflow.log_metrics(metrics)
    # Log the scaler
    scaler_path = "scaler_sklearn.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler_obj, f)
    mlflow.log_artifact(scaler_path)
    logger.info("Scaler artifact logged.")

    # Log the model using mlflow.sklearn
    logger.info("Logging model using mlflow.sklearn...")
    # Output example is typically the prediction probabilities
    # Create input example for model signature
    input_example = X_train.head(5)
    
    # Handle integer columns by converting them to float64 to properly manage missing values
    input_example = X_eval.iloc[:5].copy() if hasattr(X_eval, 'iloc') else X_eval[:5].copy()
    
    # Identify and convert integer columns to float64 to prevent schema enforcement errors
    if hasattr(input_example, 'dtypes'):
        for col in input_example.columns:
            if X_eval[col].dtype.kind == 'i':
                logger.info(f"Converting integer column '{col}' to float64 to handle potential missing values")
                X_eval[col] = X_eval[col].astype('float64')
    
    # Infer signature with proper handling for integer columns with potential missing values
    signature = mlflow.models.infer_signature(
        input_example,
        model.predict(input_example)
    )
    
    # Update model registration with signature
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model_sklearn",
        registered_model_name=f"mlp_sklearn_optuna_{datetime.now().strftime('%Y%m%d_%H%M')}",
        signature=signature,
        pip_requirements=pip_requirements
    )
    run_id = mlflow.active_run().info.run_id
    logger.info(f"Run ID: {run_id}")
    logger.info("Scikit-learn MLP model logged successfully.")

def hypertune_mlp_sklearn(experiment_name: str):
    """
    Orchestrates the MLPClassifier tuning (Optuna), evaluation, and logging process.
    """
    global X_train, y_train, X_test, y_test, X_eval, y_eval # Access global data

    if X_train is None: # Check if data is loaded
        logger.error("Data (X_train) is not loaded. Aborting.")
        return None, None
    try:
        # --- Preprocessing ---
        X_train_scaled, X_test_scaled, X_eval_scaled, fitted_scaler = preprocess_data(
            X_train.copy(), X_test.copy(), X_eval.copy()
        )
        X_train_scaled = X_train_scaled.astype('float64')
        X_test_scaled = X_test_scaled.astype('float64')
        X_eval_scaled = X_eval_scaled.astype('float64')
        # Combine X_train and X_test for full training set
        logger.info("Combining training and test sets for final model training...")
        X_train_scaled = np.vstack((X_train_scaled, X_test_scaled))
        y_train = np.concatenate((y_train, y_test))
        logger.info(f"Combined training set shape: {X_train_scaled.shape}")
        with mlflow.start_run(run_name=f"mlp_sklearn_optuna_hypertune_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            mlflow.set_tags({
                "model_type": "mlp_sklearn",
                "tuning_method": "Optuna",
                "optuna_trials": n_trials_optuna,
                "optimization_metric": "precision_at_min_recall_0.30"
            })
            # --- Hyperparameter Optimization (Optuna) ---
            best_params = optimize_hyperparameters_optuna_sklearn(
                X_train_scaled, y_train, X_eval_scaled, y_eval
            )
            if not best_params:
                logger.error("Optuna optimization failed to find suitable parameters. Aborting.")
                if mlflow.active_run(): mlflow.end_run("FAILED")
                return None, None
            # --- Train Final Model ---
            logger.info("Training final model with best Optuna parameters...")
            final_model = create_model_sklearn(best_params)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
                final_model.fit(X_train_scaled, y_train)
            logger.info("Final model training complete.")

            # --- Final Evaluation ---
            # 1. Standard evaluation (0.5 threshold)
            std_metrics = evaluate_model_sklearn(final_model, X_eval_scaled, y_eval)
            # 2. Evaluation at optimized threshold
            final_threshold, opt_metrics = optimize_threshold(
                final_model, X_eval_scaled, y_eval, min_recall=min_recall
            )
            # Combine metrics
            final_metrics = std_metrics.copy()
            final_metrics["final_threshold"] = final_threshold
            final_metrics["precision_at_threshold"] = opt_metrics["precision"]
            final_metrics["recall_at_threshold"] = opt_metrics["recall"]
            final_metrics["f1_at_threshold"] = opt_metrics["f1"]
            # AUC should be the same, but log it from opt_metrics for consistency
            final_metrics["auc_at_threshold"] = opt_metrics["auc"]
            logger.info(f"Final evaluation metrics combined: {final_metrics}")

            # --- MLflow Logging ---
            # Create input example from SCALED eval data for signature
            try:
                columns = X_eval.columns # Use original columns if available
            except AttributeError:
                columns = [f"feature_{i}" for i in range(X_eval_scaled.shape[1])]
            input_example = pd.DataFrame(X_eval_scaled[:5], columns=columns)
            log_to_mlflow_sklearn(final_model, final_metrics, best_params, fitted_scaler, input_example)
            return final_model, final_metrics

    except Exception as e:
        logger.error(f"Error in hypertune_mlp_sklearn: {str(e)}")
        # Ensure MLflow run ends if started
        if mlflow.active_run():
            mlflow.end_run("FAILED")
        return None, None

def train_with_precision_target(X_train, y_train, X_test, y_test, X_eval, y_eval):
    """
    Train MLP model with focus on precision target.
    
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
        params = {
            'hidden_layer_sizes': (128, 64, 32),
            'alpha': 0.04259050184156992,
            'batch_size': 1536,
            'learning_rate_init': 0.04342726319130476,
            'max_iter': 70,
            'n_iter_no_change': 27,
            'beta_1': 0.9,
            'beta_2': 0.983
        }
        
        # Train final model with best parameters
        logger.info("Training final model with best parameters")
        
        # Scale the data
        X_train_scaled, X_test_scaled, X_eval_scaled, fitted_scaler = preprocess_data(
            X_train.copy(), X_test.copy(), X_eval.copy()
        )
        X_train_scaled = X_train_scaled.astype('float64')
        X_test_scaled = X_test_scaled.astype('float64')
        X_eval_scaled = X_eval_scaled.astype('float64')
        # Combine X_train and X_test for full training set
        logger.info("Combining training and test sets for final model training...")
        X_train_scaled = np.vstack((X_train_scaled, X_test_scaled))
        y_train = np.concatenate((y_train, y_test))
        logger.info(f"Combined training set shape: {X_train_scaled.shape}")
        # Create model
        model = create_model_sklearn(params)
        
        # Train model
        model, metrics = train_model_sklearn(X_train_scaled, y_train, X_eval_scaled, y_eval, params)
        
        # Log to MLflow
        log_to_mlflow_sklearn(model, metrics, params, fitted_scaler, X_eval_scaled[:1])
        
        return model, metrics
            
    except Exception as e:
        logger.error(f"Error in precision-focused training: {str(e)}")
        return None, None

def main():
    """
    Main execution function for scikit-learn MLP tuning (Optuna) and final training.
    """
    global X_train, y_train, X_test, y_test, X_eval, y_eval # Declare usage of globals
    try:
        logger.info("Starting scikit-learn MLP model training & Optuna tuning process")
        # Setup MLflow tracking directory
        mlruns_dir = setup_mlflow_tracking(experiment_name)
        # Load data using shared DataLoader
        logger.info("Loading data...")
        data_loader_instance = DataLoader() # Instantiate custom loader
        # Assign to global variables
        X_train, y_train, X_test, y_test, X_eval, y_eval = data_loader_instance.load_data()
        # --- Feature Selection (Keep this part) ---
        try:
            # Attempt to load features, fallback logic remains
            features = import_selected_features_ensemble(model_type='mlp') # Consider 'mlp' if specific features exist
            logger.info(f"Using feature set with {len(features)} features.")
        except (KeyError, FileNotFoundError):
            logger.warning("Shared features ('mlp') not found. Falling back to 'all' features.")
            features = import_selected_features_ensemble(model_type='all')

        X_train = X_train[features]
        X_test = X_test[features]
        X_eval = X_eval[features]
        # -----------------------
        # Convert all columns to float64 AFTER feature selection
        X_train = X_train.astype('float64')
        X_test = X_test.astype('float64')
        X_eval = X_eval.astype('float64')
        # Log data shapes and info
        logger.info(f"Data shapes after feature selection - Train: {X_train.shape}, Test: {X_test.shape}, Eval: {X_eval.shape}")
        logger.info(f"Positive class ratio - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}, Eval: {y_eval.mean():.3f}")
        # --- Run Tuning and Final Training/Logging --- Call the main orchestrator
        best_model, final_metrics = hypertune_mlp_sklearn(experiment_name)
        # ----------------------------------------------
        if best_model and final_metrics:
            logger.info(f"Optuna Tuning and final model training completed successfully.")
            logger.info(f"Final Model Eval Metrics: {final_metrics}")
        else:
            logger.error("MLP (sklearn) training process failed.")
        # --- Train with Precision Target ---
        best_model, final_metrics = train_with_precision_target(X_train, y_train, X_test, y_test, X_eval, y_eval)
        if best_model and final_metrics:
            logger.info(f"Precision-focused training completed successfully.")
            logger.info(f"Final Model Eval Metrics: {final_metrics}")
        else:
            logger.error("Precision-focused training process failed.")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
    finally:
        gc.collect()
        logger.info("MLP (sklearn) Main execution finished.")



if __name__ == "__main__":
    main() 