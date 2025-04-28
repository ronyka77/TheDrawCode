import os
import random
import traceback
import warnings
from datetime import datetime

import mlflow
import mlflow.pyfunc
import numpy as np
import optuna
import pandas as pd
import torch
import torch.optim as optim
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.base import BaseEstimator
from sklearn.metrics import precision_score, recall_score

# from sklearn.preprocessing import QuantileTransformer
from sklearn.utils.multiclass import type_of_target
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau

# Logger and shared utilities
from src.utils.logger import ExperimentLogger

experiment_name = "tabnet_soccer_prediction"
logger = ExperimentLogger(experiment_name=experiment_name)

# Import shared utility functions
from src.models.StackedEnsemble.shared.data_loader import DataLoader
from src.models.StackedEnsemble.shared.hypertuner_utils import optimize_threshold
from src.utils.create_evaluation_set import import_selected_features_ensemble, setup_mlflow_tracking

# Filter specific TabNet weight-related warnings
warnings.filterwarnings(
    "ignore", message=".*imbalanced.*|.*weight.*|.*class_weight.*", category=UserWarning
)
warnings.filterwarnings("ignore", message=".*sample_weight.*", category=UserWarning)

# Global settings
min_recall = 0.30
# You can adjust n_trials if needed
n_trials = 20000

# Then modify your base_params to include the custom metrics
base_params = {
    "optimizer_fn": optim.Adam,  # Use Adam as default optimizer
    "mask_type": "sparsemax",
    "eval_metric": ["auc", "logloss"],  # Default metrics, custom one passed in fit
    "verbose": 0,
    "seed": 19,
    "device_name": "cuda",
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
os.environ["NUMEXPR_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"

# PyTorch specific reproducibility settings and optimizations
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True  # Auto-optimizes for hardware if input sizes don't change
    torch.backends.cudnn.deterministic = False  # Better performance, less deterministic
    # Enable TF32 for better performance on Ampere GPUs (RTX 30xx and newer)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Check if PyTorch version supports torch.compile
    if hasattr(torch, 'compile'):
        logger.info("torch.compile is available - will use it for performance optimization")
        USE_TORCH_COMPILE = True
    else:
        logger.info("torch.compile not available in this PyTorch version")
        USE_TORCH_COMPILE = False
else:
    USE_TORCH_COMPILE = False

# Create gradient scaler for mixed precision training
scaler = GradScaler() if torch.cuda.is_available() else None

# Verify CUDA availability
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
    logger.info(f"CUDA is available! Found {torch.cuda.device_count()} GPU(s).")
    logger.info(f"Using GPU: {gpu_name} with {gpu_memory:.2f} GB memory")
    logger.info(f"CUDA Version: {torch.version.cuda}")
    logger.info(f"PyTorch CUDA capabilities: TF32={torch.backends.cuda.matmul.allow_tf32}, cuDNN benchmark={torch.backends.cudnn.benchmark}")
else:
    logger.warning("CUDA is NOT available. TabNet will run on CPU.")
    # Force base_params to CPU if CUDA isn't found, to avoid potential errors
    base_params["device_name"] = "cpu"

def load_hyperparameter_space():
    """
    Define hyperparameter space for TabNet tuning.
    """
    hyperparameter_space = {
        "learning_rate": {
            "type": "float",
            "low": 0.0003,
            "high": 0.02,
            "log": True,
        },
        "eps": {"type": "float", "low": 1e-7, "high": 1e-4, "log": True},
        "n_d": {"type": "int", "low": 32, "high": 96},
        "n_a": {"type": "int", "low": 32, "high": 96},
        "n_steps": {"type": "int", "low": 3, "high": 7},
        "gamma": {"type": "float", "low": 0.5, "high": 3.0, "step": 0.05},
        "lambda_sparse": {"type": "float", "low": 1e-7, "high": 1e-2, "log": True},
        "momentum": {"type": "float", "low": 0.7, "high": 0.99, "step": 0.005},
        "patience": {"type": "int", "low": 15, "high": 45},
        "max_epochs": {"type": "int", "low": 90, "high": 180, "step": 5},
        "batch_size": {"type": "int", "low": 128, "high": 4096, "step": 64},
        "virtual_batch_size": {"type": "int", "low": 128, "high": 4096, "step": 64},
        "n_independent": {"type": "int", "low": 2, "high": 4},
        "n_shared": {"type": "int", "low": 2, "high": 4},
        "weight_decay": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
        "scheduler_type": {
            "type": "categorical",
            "choices": ["cosine", "plateau", "onecycle", "none"],
        },
        "scheduler_patience": {"type": "int", "low": 3, "high": 10},
        "scheduler_factor": {"type": "float", "low": 0.1, "high": 0.5},
        "scheduler_min_lr": {"type": "float", "low": 1e-6, "high": 1e-4, "log": True},
        "scheduler_t_max": {"type": "int", "low": 5, "high": 20},
        "scheduler_pct_start": {"type": "float", "low": 0.1, "high": 0.5, "step": 0.05},
        "scheduler_div_factor": {"type": "float", "low": 10.0, "high": 40.0, "step": 0.5},
        "scheduler_final_div_factor": {"type": "float", "low": 1000.0, "high": 10000.0, "step": 100.0},
        "fit_weights": {
            "type": "categorical",
            "choices": [0, 1]
        },
    }
    return hyperparameter_space

# Create a custom metric that heavily weights precision
class PrecisionFocusedMetric(Metric):
    def __init__(self, beta=0.5):
        self._name = "precision_focused"
        self._maximize = True
        self.beta = beta

    def __call__(self, y_true, y_score):
        """F-beta score with beta < 1 to favor precision over recall"""

        # Ensure y_true is a 1D array
        # Check type of target
        y_true_type = type_of_target(y_true)
        if y_true_type == "multilabel-indicator":
            # Assuming binary classification represented as one-hot
            # Convert back to 1D: take the argmax along the class axis (axis=1)
            y_true_flat = np.argmax(y_true, axis=1)
        elif y_true_type == "binary":
            y_true_flat = y_true.astype(int) # Ensure integer type
        else:
            # Handle unexpected types or raise an error
            logger.warning(f"Unexpected y_true type '{y_true_type}' in PrecisionFocusedMetric. Attempting to flatten.")
            try:
                y_true_flat = y_true.astype(int).ravel() # General attempt to flatten
            except Exception as e:
                logger.error(f"Could not convert y_true to 1D array: {e}")
                return 0.0 # Return 0 score if conversion fails

        # Ensure y_score handling is robust
        # Check if y_score has 2 columns (expected for binary probabilities)
        if y_score.ndim == 2 and y_score.shape[1] == 2:
            pred = (y_score[:, 1] > 0.5).astype(int) # Use probability of positive class
        elif y_score.ndim == 1: # If y_score is already 1D predictions/scores
            pred = (y_score > 0.5).astype(int) # Threshold directly
        else:
            logger.error(f"Unexpected y_score shape {y_score.shape} in PrecisionFocusedMetric.")
            return 0.0 # Return 0 score if y_score format is wrong

        # Calculate precision and recall safely
        try:
            # Check target types again just before sklearn call for debugging
            # logger.debug(f"y_true_flat type: {type_of_target(y_true_flat)}, pred type: {type_of_target(pred)}")
            precision = precision_score(y_true_flat, pred, zero_division=0)
            recall = recall_score(y_true_flat, pred, zero_division=0)
        except ValueError as e:
            logger.error(f"Error calculating scores in PrecisionFocusedMetric: {e}")
            logger.error(f"y_true_flat sample: {y_true_flat[:5]}, shape: {y_true_flat.shape}, type: {type_of_target(y_true_flat)}")
            logger.error(f"pred sample: {pred[:5]}, shape: {pred.shape}, type: {type_of_target(pred)}")
            return 0.0 # Return 0 score if scikit-learn metric fails

        # If recall below threshold, return 0
        if recall < min_recall:
            return 0.0

        # F-beta with beta < 1 favors precision
        f_beta = (
            (1 + self.beta**2) * (precision * recall) / (self.beta**2 * precision + recall + 1e-8)
        )
        return f_beta

class TabNetSklearnWrapper(BaseEstimator):
    """
    A scikit-learn compatible wrapper for TabNet that makes it compatible with MLflow's sklearn flavor.
    """

    def __init__(self, model=None, **kwargs):
        self.model = model
        self.kwargs = kwargs

    def fit(self, X, y):
        """
        Fit method for scikit-learn compatibility.
        """
        if self.model is None:
            self.model = TabNetClassifier(**self.kwargs)
            self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict method for scikit-learn compatibility.
        """
        if hasattr(X, "values"):
            data = X.values
        else:
            data = X
        return self.model.predict(data)

    def predict_proba(self, X):
        """
        Predict probability method for scikit-learn compatibility.
        """
        if hasattr(X, "values"):
            data = X.values
        else:
            data = X
        return self.model.predict_proba(data)

class TabNetWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, model_input):
        """
        Returns class predictions.
        """
        if hasattr(model_input, "values"):
            data = model_input.values
        else:
            data = model_input
        return self.model.predict(data)

    def predict_proba(self, model_input):
        """
        Returns probability estimates for each class.
        """
        if hasattr(model_input, "values"):
            data = model_input.values
        else:
            data = model_input
        return self.model.predict_proba(data)

def create_model(model_params):
    """
    Create and configure TabNet model instance based on provided parameters.
    Uses internal TabNet loss.
    """
    try:
        # Start with base parameters and update with model_params
        params = base_params.copy()
        # Define valid constructor args and config keys
        valid_constructor_args = {
            "n_d", "n_a", "n_steps", "gamma", "lambda_sparse", "optimizer_fn",
            "optimizer_params", "scheduler_fn", "scheduler_params", "mask_type",
            "n_independent", "n_shared", "epsilon", "momentum", "device_name",
            "seed", "verbose", "cat_idxs", "cat_dims", "cat_emb_dim"
        }
        config_keys = {"learning_rate", "weight_decay", "scheduler_type",
                        "scheduler_t_max", "scheduler_min_lr", "scheduler_patience",
                        "scheduler_factor", "scheduler_div_factor"}

        # Extract constructor and config params from input model_params
        constructor_params = {k: v for k, v in model_params.items() if k in valid_constructor_args}
        config_params = {k: v for k, v in model_params.items() if k in config_keys}

        # Always use Adam optimizer for GPU optimization
        params["optimizer_fn"] = optim.Adam
        
        # Update base params with constructor params
        params.update(constructor_params)

        # Configure optimizer with GPU-optimized settings
        lr = config_params.get("learning_rate", 0.01)
        weight_decay = config_params.get("weight_decay", 1e-5)
        if "optimizer_params" not in params: 
            params["optimizer_params"] = {}
        params["optimizer_params"]["lr"] = lr
        params["optimizer_params"]["weight_decay"] = weight_decay
        # Add optimizer settings that can improve GPU performance
        params["optimizer_params"]["eps"] = config_params.get("eps", 1e-7)  # Improves numerical stability
        params["optimizer_params"]["amsgrad"] = True  # Can improve convergence on GPU
        
        # Configure scheduler
        scheduler_type = config_params.get("scheduler_type", "none")
        scheduler_params_config = {}
        if scheduler_type == "cosine":
            scheduler_fn = CosineAnnealingLR
            scheduler_params_config = {
                "T_max": config_params.get("scheduler_t_max", 10),
                "eta_min": config_params.get("scheduler_min_lr", 1e-5),
            }
            params["scheduler_fn"] = scheduler_fn
            params["scheduler_params"] = scheduler_params_config
        elif scheduler_type == "plateau":
            scheduler_fn = ReduceLROnPlateau
            scheduler_params_config = {
                "patience": config_params.get("scheduler_patience", 5),
                "factor": config_params.get("scheduler_factor", 0.1),
                "min_lr": config_params.get("scheduler_min_lr", 1e-6),
                "mode": "max",
            }
            params["scheduler_fn"] = scheduler_fn
            params["scheduler_params"] = scheduler_params_config
        elif scheduler_type == "onecycle":
            scheduler_fn = OneCycleLR
            scheduler_params_config = {
                "div_factor": config_params.get("scheduler_div_factor", 25.0),
                "final_div_factor": config_params.get("scheduler_final_div_factor", 10000.0),
                "pct_start": config_params.get("scheduler_pct_start", 0.3),
            }
            params["scheduler_fn"] = scheduler_fn
            params["scheduler_params"] = scheduler_params_config
        else:
            params.pop("scheduler_fn", None)
            params.pop("scheduler_params", None)

        # Ensure only valid args are passed to constructor
        final_params = {k:v for k,v in params.items() if k in valid_constructor_args or k in ["optimizer_params", "scheduler_fn", "scheduler_params"]}
        if "verbose" in final_params:
            final_params["verbose"] = int(final_params["verbose"])

        # Instantiate model (without loss_fn argument)
        model = TabNetClassifier(**final_params)
        
        # Apply torch.compile if available and using GPU (for PyTorch 2.0+)
        if USE_TORCH_COMPILE and torch.cuda.is_available():
            try:
                # We need to compile specific parts of the model
                # TabNetClassifier is complex, so we compile only the network component
                if hasattr(model, 'network') and hasattr(torch, 'compile'):
                    logger.info("Applying torch.compile to TabNet network for GPU acceleration")
                    # Apply compilation with 'reduce-overhead' mode which is good for GPU performance
                    model.network = torch.compile(model.network, mode="reduce-overhead")
                    logger.info("Successfully applied torch.compile to TabNet network")
            except Exception as e:
                logger.warning(f"Could not apply torch.compile: {str(e)}")
                
        return model
    except Exception as e:
        logger.error(f"Error creating TabNet model: {str(e)}")
        logger.error(f"Parameters passed to TabNetClassifier attempt: {final_params}")
        logger.error(traceback.format_exc())
        raise

def train_model(X_train, y_train, X_test, y_test, X_eval, y_eval, model_params):
    """
    Train a TabNet model with early stopping.
    Uses internal TabNet loss, controls imbalance via fit(weights=...).
    Returns the trained model and evaluation metrics after threshold optimization.
    """
    try:
        # Create the model
        model = create_model(model_params)

        # Get fit parameters from model_params
        batch_size_to_use = int(model_params.get("batch_size", 1024))
        max_epochs = int(model_params.get("max_epochs", 50))
        patience_to_use = int(model_params.get("patience", 10))
        virtual_batch_size_to_use = int(model_params.get("virtual_batch_size", 128))
        fit_weights_value = int(model_params.get("fit_weights", 1)) # Default to 1 (unbalanced) if not found

        # Update OneCycleLR scheduler params if needed
        if model_params.get("scheduler_type") == "onecycle":
            # Combine X_train and X_test for step calculation as they are used together in fit
            total_samples = (len(X_train) if hasattr(X_train, "__len__") else X_train.shape[0]) + \
                            (len(X_test) if hasattr(X_test, "__len__") else X_test.shape[0])
            steps_per_epoch = total_samples // batch_size_to_use + (1 if total_samples % batch_size_to_use != 0 else 0)
            total_steps = steps_per_epoch * max_epochs
            if hasattr(model, 'scheduler_params') and isinstance(model.scheduler_params, dict):
                lr = model_params.get("learning_rate", 0.01)
                model.scheduler_params["max_lr"] = lr
                model.scheduler_params["total_steps"] = total_steps
                logger.info(f"Updated OneCycleLR params: max_lr={lr}, total_steps={total_steps}")
            else:
                logger.warning("OneCycleLR selected but model.scheduler_params not found/dict.")

        # Convert data to numpy arrays
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, (pd.Series, pd.DataFrame)):
            y_train = y_train.values
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values
        if isinstance(y_test, (pd.Series, pd.DataFrame)):
            y_test = y_test.values
        if isinstance(X_eval, pd.DataFrame):
            X_eval = X_eval.values
        if isinstance(y_eval, (pd.Series, pd.DataFrame)):
            y_eval = y_eval.values

        # Reshape y
        if y_train.ndim == 2 and y_train.shape[1] == 1:
            y_train = y_train.ravel()
        if y_test.ndim == 2 and y_test.shape[1] == 1:
            y_test = y_test.ravel()
        if y_eval.ndim == 2 and y_eval.shape[1] == 1:
            y_eval = y_eval.ravel()

        # Combine training and testing data
        X_combined = np.concatenate([X_train, X_test], axis=0)
        y_combined = np.concatenate([y_train, y_test], axis=0)

        # Define fit parameters dictionary, now using fit_weights_value
        fit_params = {
            "eval_set": [(X_eval, y_eval)],
            "eval_metric": [PrecisionFocusedMetric],
            "max_epochs": max_epochs,
            "patience": patience_to_use,
            "batch_size": batch_size_to_use,
            "virtual_batch_size": virtual_batch_size_to_use,
            "weights": fit_weights_value, # Pass the sampled weight value here
            "drop_last": False,
        }

        # Fit the model
        logger.info(f"Starting model.fit with epochs={max_epochs}, patience={patience_to_use}, batch_size={batch_size_to_use}, weights={fit_weights_value}")
        model.fit(
            X_combined,
            y_combined,
            **fit_params
        )
        
        # Log peak GPU memory usage during training
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
            logger.info(f"Peak GPU memory during training: {peak_mem:.2f} MB")

        # Optimize threshold using shared utility
        best_threshold, metrics = optimize_threshold(model, X_eval, y_eval, min_recall=min_recall)

        # Add fit_weights used to metrics dict for logging
        metrics["fit_weights_used"] = fit_weights_value
        # Add GPU memory usage to metrics if available
        if torch.cuda.is_available():
            metrics["peak_gpu_memory_mb"] = peak_mem

        return model, metrics
    except Exception as e:
        logger.error(f"Error training TabNet model: {str(e)}")
        logger.error(traceback.format_exc())
        logger.error(f"Parameters during failed training: {model_params}")
        raise

def optimize_hyperparameters(
    X_train, y_train, X_test, y_test, X_eval, y_eval, hyperparameter_space
):
    logger.info("Starting hyperparameter optimization for TabNet (tuning fit_weights)")
    if not hyperparameter_space:
        hyperparameter_space = load_hyperparameter_space()

    best_score = -float("inf")
    best_params = {}
    global_top_trials = []
    top_trials = []

    def objective(trial):
        nonlocal best_score, best_params
        current_params = {}
        try:
            # --- Sample fit_weights first ---
            current_params["fit_weights"] = trial.suggest_categorical("fit_weights", [0, 1])

            # --- Sample other parameters ---
            # Sample scheduler type needed for conditional params
            scheduler_type = trial.suggest_categorical("scheduler_type", hyperparameter_space["scheduler_type"]["choices"])
            current_params["scheduler_type"] = scheduler_type

            # Iterate over the rest of the hyperparameter space
            for param_name, param_config in hyperparameter_space.items():
                # Skip params already handled or handled conditionally
                if param_name in ["fit_weights", "scheduler_type"]:
                    continue

                # Conditional suggestion for scheduler params
                is_relevant_scheduler_param = False
                if scheduler_type == "plateau" and param_name in ["scheduler_patience", "scheduler_factor", "scheduler_min_lr"]:
                    is_relevant_scheduler_param = True
                elif scheduler_type == "cosine" and param_name in ["scheduler_t_max", "scheduler_min_lr"]:
                    is_relevant_scheduler_param = True
                elif scheduler_type == "onecycle" and param_name == "scheduler_div_factor":
                    is_relevant_scheduler_param = True
                elif param_name not in ["scheduler_patience", "scheduler_factor", "scheduler_min_lr", "scheduler_t_max", "scheduler_div_factor"]:
                    # Not a scheduler-specific param, suggest normally
                    is_relevant_scheduler_param = True

                if is_relevant_scheduler_param:
                    # Suggest parameter
                    if param_config["type"] == "float":
                        if "step" in param_config:
                            current_params[param_name] = trial.suggest_float(param_name, param_config["low"], param_config["high"], step=param_config["step"], log=param_config.get("log", False))
                        else:
                            current_params[param_name] = trial.suggest_float(param_name, param_config["low"], param_config["high"], log=param_config.get("log", False))
                    elif param_config["type"] == "int":
                        if "step" in param_config:
                            current_params[param_name] = trial.suggest_int(param_name, param_config["low"], param_config["high"], step=param_config["step"])
                        else:
                            current_params[param_name] = trial.suggest_int(param_name, param_config["low"], param_config["high"])
                    elif param_config["type"] == "categorical":
                        # Only suggest if not scheduler_type (already handled)
                        if param_name != "scheduler_type":
                            choices = param_config.get("choices", [])
                            if isinstance(choices, list) and choices:
                                current_params[param_name] = trial.suggest_categorical(param_name, choices)
                            else:
                                logger.warning(f"Skipping categorical param '{param_name}' due to invalid/empty choices.")

            # Train model and get metrics
            model, metrics = train_model(X_train, y_train, X_test, y_test, X_eval, y_eval, current_params)
            # Store model reference for callback but don't try to serialize it
            setattr(trial, 'model', model)  # noqa: B010
            trial.set_user_attr('metrics', metrics)
            trial.set_user_attr('params', current_params)
            # Scoring logic
            recall = metrics.get("recall", 0.0)
            precision = metrics.get("precision", 0.0)
            score = precision if recall >= min_recall else 0.0

            # Log trial results
            logger.info(f"  Trial {trial.number}: fit_weights={current_params['fit_weights']}, Score={score:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
            for metric_name, metric_value in metrics.items():
                # Serialize for Optuna
                if isinstance(metric_value, (int, float, str, bool)) or metric_value is None:
                    trial.set_user_attr(metric_name, metric_value)
                elif isinstance(metric_value, np.generic):
                    trial.set_user_attr(metric_name, metric_value.item())
                else:
                    trial.set_user_attr(metric_name, str(metric_value))

            # Update best score and params FOR THIS RUN
            if score > best_score:
                best_score = score
                best_params = current_params.copy()
                logger.info(f"  >>> New best score in this run: {best_score:.4f} (Trial {trial.number})")

            if score >= 0.36:
                logger.info(f"Trial {trial.number} completed with score {score:.4f}")
                X_eval_orig_df = X_eval.copy()
                log_to_mlflow(model, metrics, current_params, experiment_name, X_eval_orig_df)
            return score
        except optuna.TrialPruned:
            logger.info(f"Trial {trial.number} pruned.")
            raise # Re-raise to signal Optuna
        except Exception as e:
            logger.error(f"Trial {trial.number} failed.")
            logger.error(f"Failed trial parameters: {current_params}")
            logger.error(f"Error: {str(e)}")
            logger.error(traceback.format_exc())
            return 0.0 # Return low score for failed trials

    def callback(study, trial, experiment_name, X_eval):
        nonlocal best_score, best_params, top_trials
        logger.info(f"Current best score in this batch: {best_score:.4f}")
        if trial.value > best_score:
            best_score = trial.value
            best_params = trial.params
            logger.info(f"New best score found in trial {trial.number}: {best_score:.4f}")
        current_run = (trial.value, trial.params, trial.number)
        top_trials.append(current_run)
        top_trials.sort(key=lambda x: x[0], reverse=True)
        top_trials[:] = top_trials[:10]
        if trial.number % 9 == 0:
            table_rows = [
                f"| {i + 1} | {rec[2]} | {rec[0]:.4f} | {rec[1]} |"
                for i, rec in enumerate(top_trials)
            ]
            logger.info("Top trials in current batch:")
            for row in table_rows:
                logger.info(row)
        if trial.number % 100 == 0 and global_top_trials:
            table_rows = [
                f"| {i + 1} | {rec[2]} | {rec[0]:.4f} | {rec[1]} |"
                for i, rec in enumerate(global_top_trials[:10])
            ]
            logger.info("Global top trials:")
            for row in table_rows:
                logger.info(row)
        return best_score

    # --- Optuna Study Execution ---
    storage_url = "sqlite:///optuna_tabnet.db"
    study_name = "tabnet_optimization"
    total_trials = n_trials
    batch_size = 1000
    num_batches = total_trials // batch_size
    if total_trials % batch_size != 0:
        num_batches += 1

    logger.info(f"Starting Optuna study '{study_name}' with {total_trials} trials.")
    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5, interval_steps=1)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage_url,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )
    for _ in range(num_batches):
        try:
            study.optimize(objective, n_trials=batch_size, callbacks=[lambda study, trial: callback(study, trial, experiment_name, X_eval)], n_jobs=3)
        except KeyboardInterrupt:
            logger.warning("Optimization interrupted by user.")
            break

    logger.info(f"Best parameters selected: {best_params}")
    return best_params

def hypertune_tabnet(experiment_name: str, X_train, y_train, X_test, y_test, X_eval, y_eval):
    """
    Main hypertuning function for TabNet with MLflow tracking.
    Returns best_params and metrics from the final trained model.
    """
    try:
        hyperparameter_space = load_hyperparameter_space()
        logger.info("Starting hyperparameter optimization for TabNet (tuning fit_weights)")

        # === Run Optimization ===
        best_params_found = optimize_hyperparameters(
            X_train, y_train, X_test, y_test, X_eval, y_eval, # Pass actual data
            hyperparameter_space=hyperparameter_space
        )

        if not best_params_found:
            logger.error("Hyperparameter optimization failed to find best parameters.")
            return None, None

        logger.info(f"Hyperparameter optimization completed. Best parameters found: {best_params_found}")

        # === Train Final Model with Best Params ===
        logger.info("Training final TabNet model with best parameters found...")
        # Pass data correctly
        final_model, final_metrics = train_model(
            X_train, y_train,
            X_test, y_test,
            X_eval, y_eval,
            best_params_found
        )

        logger.info("Final model trained successfully.")
        logger.info(f"Final Metrics: {final_metrics}")

        global X_eval_orig_df # Need original DataFrame for signature
        log_run_id = log_to_mlflow(final_model, final_metrics, best_params_found, experiment_name, X_eval_orig_df)
        logger.info(f"Final model and metrics logged to MLflow run_id: {log_run_id}")

        # Return the best parameters and the metrics from the model trained with those params
        return best_params_found, final_metrics

    except Exception as e:
        logger.error(f"Error in TabNet hypertuning process: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None

def log_to_mlflow(model, metrics, params, experiment_name, X_eval_df_for_sig):
    """Logs model, metrics, params to MLflow."""
    try:
        # Set up MLflow tracking
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=f"tabnet_final_train_{datetime.now().strftime('%Y%m%d_%H%M')}", nested=True) as run:
            mlflow.log_params(params)
            mlflow.set_tags({"final_model_training": True})
            mlflow.log_metrics(metrics)
            active_run_id = mlflow.active_run().info.run_id
            logger.info(f"Logging final model artifacts to MLflow run_id: {active_run_id}")

            # --- Signature ---
            input_example = None
            signature = None
            if isinstance(X_eval_df_for_sig, pd.DataFrame):
                input_example = X_eval_df_for_sig.iloc[:5].copy()
                # Ensure dtypes are float for numeric cols
                num_cols = input_example.select_dtypes(include=np.number).columns
                input_example[num_cols] = input_example[num_cols].astype('float64')
                logger.info("Created input_example from DataFrame for signature.")

                # Wrap model for prediction
                sklearn_wrapper = TabNetSklearnWrapper(model=model)
                try:
                    logger.info("Inferring model signature...")
                    # Ensure model is fitted
                    if not hasattr(sklearn_wrapper.model, 'network'):
                        raise ValueError("Model inside wrapper doesn't seem fitted (no network attribute).")
                    prediction_output = sklearn_wrapper.predict_proba(input_example)
                    signature = mlflow.models.infer_signature(input_example, prediction_output)
                    logger.info("Signature inferred successfully.")
                except Exception as sig_err:
                    logger.error(f"Failed to infer signature: {sig_err}. Logging model without signature.")
                    logger.error(traceback.format_exc())
                    signature = None
            else:
                logger.warning(f"Cannot create input example for MLflow signature from X_eval of type {type(X_eval_df_for_sig)}.)")

            # --- Log Model ---
            # Re-wrap model just before logging to be safe
            sklearn_wrapper_for_log = TabNetSklearnWrapper(model=model)
            model_reg_name = f"tabnet_final_{datetime.now().strftime('%Y%m%d_%H%M')}"
            try:
                logger.info(f"Logging model with mlflow.sklearn.log_model (signature={'present' if signature else 'absent'})...")
                model_info = mlflow.sklearn.log_model(
                    sk_model=sklearn_wrapper_for_log,
                    artifact_path="model_sklearn",
                    signature=signature,
                    registered_model_name=model_reg_name,
                    input_example=input_example if signature else None
                )
                run_id = run.info.run_id
                logger.info(f"Final model logged to MLflow (sklearn flavor): {model_info.model_uri}")
                logger.info(f"Registered as: {model_reg_name}")
                logger.info(f"Run ID: {run_id}")
                mlflow.end_run()
                return run_id

            except Exception as log_model_err:
                logger.error(f"mlflow.sklearn.log_model failed: {log_model_err}")
                logger.error(traceback.format_exc())
            return active_run_id # Return run_id even if model logging had issues
    except Exception as e:
        logger.error(f"Error in log_to_mlflow: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def select_top_features_tabnet(model: TabNetClassifier, X_features: pd.DataFrame, n_features: int = 60) -> list[str]:
    """
    Selects the top N features based on TabNet feature importances.

    Args:
        model: Trained TabNetClassifier model.
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
    logger.info(f"Selected top {n_features} features based on TabNet importance.")
    logger.info(f"Top features: {top_features}") # Log the selected features for visibility

    return top_features

def train_with_precision_target(X_train, y_train, X_test, y_test, X_eval, y_eval):
    """
    Train TabNet model with focus on precision target.
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
        params = base_params.copy()  # Inherits 'device_name': 'cuda'
        # Specific parameters for this training run with advanced scheduling
        params.update(
            {
                "batch_size": 128,
                "eps": 3.364173308706195e-06,
                "fit_weights": 1,
                "gamma": 1.6,
                "lambda_sparse": 0.009889443656051256,
                "learning_rate": 0.0038334747996737117,
                "max_epochs": 135,
                "momentum": 0.7949999999999999,
                "n_a": 45,
                "n_d": 81,
                "n_independent": 3,
                "n_shared": 3,
                "n_steps": 7,
                "patience": 19,
                "scheduler_factor": 0.21834593629990895,
                "scheduler_final_div_factor": 7100.0,
                "scheduler_min_lr": 7.839143819235056e-05,
                "scheduler_patience": 8,
                "scheduler_pct_start": 0.1,
                "scheduler_type": "plateau",
                "virtual_batch_size": 1856,
                "weight_decay": 2.8797977154497002e-05,
            }
        )
        # Train final model with best parameters
        logger.info("Training final model with best parameters")
        model, metrics = train_model(X_train, y_train, X_test, y_test, X_eval, y_eval, params)
        # Log to MLflow
        # log_to_mlflow(model, metrics, params, experiment_name, X_eval)
        # Select top features
        top_features = compute_permutation_importance(model, X_eval, y_eval, metrics['threshold'])
        return model, metrics, top_features
    except Exception as e:
        logger.error(f"Error during MLflow artifact logging: {str(e)}")
        logger.error(traceback.format_exc())
        return mlflow.active_run().info.run_id if mlflow.active_run() else None

def compute_permutation_importance(
    model,
    X_val: pd.DataFrame, 
    y_val: np.ndarray,
    threshold: float = 0.3,
    n_repeats: int = 3,
    number_of_features: int = 100,
) -> pd.DataFrame:
    """
    Compute permutation feature importance for TabNet model.
    Args:
        model: Trained TabNet model with predict_proba(X) method
        X_val: Validation features (DataFrame)
        y_val: Validation labels (array-like)
        threshold: Threshold for positive class prediction
        n_repeats: Number of shuffles per feature
        number_of_features: Number of top features to display in logs
    Returns:
        DataFrame with columns: ['feature', 'importance'] sorted by importance descending
    """
    try:
        feature_names = X_val.columns.tolist()
        if isinstance(X_val, pd.DataFrame):
            X_val = X_val.values
        y_val_np = y_val.values if hasattr(y_val, 'values') else y_val
        
        # Convert to numpy and ensure correct shape
        if y_val_np.ndim == 2 and y_val_np.shape[1] == 1:
            y_val_np = y_val_np.ravel()
            
        # Compute baseline metric
        probs = model.predict_proba(X_val)[:, 1]
        preds = (probs >= threshold).astype(int)
        
        # Calculate baseline precision
        baseline = np.sum((y_val_np == 1) & (preds == 1)) / (np.sum(preds == 1))
        logger.info(f"Baseline precision: {baseline:.4f}")
        
        importances = []
        for feat_idx, feat in enumerate(feature_names):
            drops = []
            for i in range(n_repeats):
                logger.info(f"Shuffling feature: {feat} - Repeat: {i+1}")
                X_shuffled = X_val.copy()
                # Use column index since X_val is numpy array
                X_shuffled[:, feat_idx] = np.random.permutation(X_val[:, feat_idx])
                
                # Get predictions with shuffled feature
                probs_shuffled = model.predict_proba(X_shuffled)[:, 1]
                preds_shuffled = (probs_shuffled >= threshold).astype(int)
                
                # Calculate precision with shuffled feature
                precision = np.sum((y_val_np == 1) & (preds_shuffled == 1)) / (np.sum(preds_shuffled == 1) + 1e-7)
                drop = baseline - precision
                drops.append(drop)
                
            mean_drop = np.mean(drops)
            importances.append((feat, mean_drop))
            logger.debug(f"Feature: {feat}, Mean importance drop: {mean_drop:.4f}")
            
        # Sort by importance descending
        importances.sort(key=lambda x: x[1], reverse=True)
        df_importance = pd.DataFrame(importances, columns=["feature", "importance"])
        
        # Log top features
        logger.info("Top features by permutation importance:")
        logger.info(df_importance.head(number_of_features).to_string(index=False))
        
        return df_importance
        
    except Exception as e:
        logger.error(f"Error computing permutation importance: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def main():
    """
    Main execution function for TabNet hypertuning including loss function.
    """
    try:
        logger.info("Starting TabNet model hypertuning (tuning fit_weights)")
        setup_mlflow_tracking(experiment_name)
        global X_eval_orig_df # Used in log_to_mlflow

        # Load data
        dataloader = DataLoader()
        X_train_orig, y_train_orig, X_test_orig, y_test_orig, X_eval_orig_df, y_eval_orig = dataloader.load_data()
        X_eval_orig_df = X_eval_orig_df.copy() # Store original for signature

        # Select features
        features = import_selected_features_ensemble(model_type="all")
        if not features:
            logger.warning("No features selected. Using all numeric features.")
            features = import_selected_features_ensemble("all")

        X_train = X_train_orig[features]
        X_test = X_test_orig[features]
        X_eval_df = X_eval_orig_df[features]

        # Assign labels (ensure 1D numpy)
        y_train = y_train_orig.values.ravel() if hasattr(y_train_orig, 'values') else np.array(y_train_orig).ravel()
        y_test = y_test_orig.values.ravel() if hasattr(y_test_orig, 'values') else np.array(y_test_orig).ravel()
        y_eval = y_eval_orig.values.ravel() if hasattr(y_eval_orig, 'values') else np.array(y_eval_orig).ravel()

        # Convert features to float64
        X_train = X_train.astype("float64")
        X_test = X_test.astype("float64")
        X_eval = X_eval_df.astype("float64")

        logger.info(f"Data shapes: Train={X_train.shape}, Test={X_test.shape}, Eval={X_eval_df.shape}")
        logger.info(f"Positive ratios: Train={y_train.mean():.3f}, Test={y_test.mean():.3f}, Eval={y_eval.mean():.3f}")

        # === Run Hypertuning ===
        # best_params_final, best_metrics_final = hypertune_tabnet(
        #     experiment_name,
        #     X_train, y_train,
        #     X_test, y_test,
        #     X_eval, y_eval 
        # )

        train_with_precision_target(X_train, y_train, X_test, y_test, X_eval, y_eval)

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")


if __name__ == "__main__":
    main()

