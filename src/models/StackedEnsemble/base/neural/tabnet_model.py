# Imports
import os
import random
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

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
from sklearn.preprocessing import QuantileTransformer

# Add imports for schedulers
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau

# Set project root similar to xgboost_model.py
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

# Logger and shared utilities
from utils.logger import ExperimentLogger

experiment_name = "tabnet_soccer_prediction"
logger = ExperimentLogger(experiment_name=experiment_name)

# Import shared utility functions
from models.StackedEnsemble.shared.data_loader import DataLoader
from models.StackedEnsemble.shared.hypertuner_utils import optimize_threshold
from utils.create_evaluation_set import import_selected_features_ensemble, setup_mlflow_tracking

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
    "optimizer_fn": optim.Adam,
    "mask_type": "sparsemax",
    "eval_metric": ["logloss", "auc"],  # Remove function reference here
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
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
# PyTorch specific reproducibility settings
torch.manual_seed(SEED)

# Verify CUDA availability
if torch.cuda.is_available():
    logger.info(f"CUDA is available! Found {torch.cuda.device_count()} GPU(s).")
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
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
            "low": 1e-4,  # Lower bound decreased
            "high": 5e-1,  # Upper bound increased
            "log": True,
        },
        "n_d": {
            "type": "int",
            "low": 8,  # Increased lower bound
            "high": 64,  # Increased upper bound for more complex features
        },
        "n_a": {
            "type": "int",
            "low": 8,  # Increased lower bound
            "high": 64,  # Increased upper bound for attention
        },
        "n_steps": {
            "type": "int",
            "low": 3,  # Increased lower bound
            "high": 15,  # Increased upper bound for deeper networks
        },
        "gamma": {
            "type": "float",
            "low": 0.5,  # Decreased lower bound
            "high": 3.0,  # Increased upper bound
            "step": 0.05,
        },
        "lambda_sparse": {
            "type": "float",
            "low": 1e-7,  # Lower bound decreased
            "high": 1e-2,  # Upper bound increased
            "log": True,
        },
        "momentum": {
            "type": "float",
            "low": 0.7,  # Decreased lower bound
            "high": 0.99,
            "step": 0.005,
        },
        "patience": {
            "type": "int",
            "low": 5,  # Increased lower bound
            "high": 30,  # Increased upper bound
        },
        "max_epochs": {
            "type": "int",
            "low": 60,  # Increased lower bound
            "high": 200,  # Increased upper bound
            "step": 5,  # Increased step size
        },
        "batch_size": {  # Added batch size tuning
            "type": "int",
            "low": 1024,
            "high": 16384,
        },
        "virtual_batch_size": {  # Added virtual batch size tuning
            "type": "int",
            "low": 128,
            "high": 4096,
        },
        "n_independent": {"type": "int", "low": 1, "high": 5},
        "n_shared": {"type": "int", "low": 1, "high": 5},
        # New regularization parameter
        "weight_decay": {"type": "float", "low": 1e-6, "high": 1e-3, "log": True},
        # Scheduler type parameter
        "scheduler_type": {
            "type": "categorical",
            "choices": ["cosine", "plateau", "onecycle", "none"],
        },
        # Scheduler specific parameters
        "scheduler_patience": {"type": "int", "low": 3, "high": 10},
        "scheduler_factor": {"type": "float", "low": 0.1, "high": 0.5},
        "scheduler_min_lr": {"type": "float", "low": 1e-6, "high": 1e-4, "log": True},
        "scheduler_t_max": {"type": "int", "low": 5, "high": 20},
        "scheduler_div_factor": {"type": "float", "low": 10.0, "high": 30.0},
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
        pred = (y_score > 0.5).astype(int)
        precision = precision_score(y_true, pred)
        recall = recall_score(y_true, pred)

        # If recall below threshold, return 0
        if recall < min_recall:
            return 0

        # F-beta with beta < 1 favors precision
        return (
            (1 + self.beta**2) * (precision * recall) / (self.beta**2 * precision + recall + 1e-8)
        )


class FocalLoss(Metric):
    def __init__(self, gamma=2.0):
        self._name = "focal_loss"
        self._maximize = False
        self.gamma = gamma

    def __call__(self, y_true, y_score):
        """Compute focal loss for binary classification."""
        pt = np.where(y_true == 1, y_score, 1 - y_score)
        return -np.mean((1 - pt) ** self.gamma * np.log(pt + 1e-7))


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
    """
    try:
        params = base_params.copy()
        params.update(model_params)

        # Configure optimizer with weight decay for L2 regularization
        weight_decay = params.pop("weight_decay", 1e-5)
        lr = params.pop("learning_rate", 0.01)

        # Update optimizer parameters with learning rate and weight decay
        params["optimizer_params"] = {
            "lr": lr,
            "weight_decay": weight_decay,  # Add L2 regularization
        }

        # Configure scheduler if specified
        scheduler_type = params.pop("scheduler_type", "none")
        scheduler_params = {}

        if scheduler_type == "cosine":
            scheduler_fn = CosineAnnealingLR
            scheduler_params = {
                "T_max": params.pop("scheduler_t_max", 10),
                "eta_min": params.pop("scheduler_min_lr", 1e-5),
            }
            params["scheduler_fn"] = scheduler_fn
            params["scheduler_params"] = scheduler_params
        elif scheduler_type == "plateau":
            scheduler_fn = ReduceLROnPlateau
            scheduler_params = {
                "patience": params.pop("scheduler_patience", 5),
                "factor": params.pop("scheduler_factor", 0.1),
                "min_lr": params.pop("scheduler_min_lr", 1e-6),
                "mode": "max",  # For metrics like AUC where higher is better
            }
            params["scheduler_fn"] = scheduler_fn
            params["scheduler_params"] = scheduler_params
        elif scheduler_type == "onecycle":
            scheduler_fn = OneCycleLR
            scheduler_params = {
                "max_lr": lr,
                "div_factor": params.pop("scheduler_div_factor", 25.0),
                "final_div_factor": 10000.0,
                "pct_start": 0.3,
            }
            # Will set total_steps in train_model based on epochs and batch size
            params["scheduler_fn"] = scheduler_fn
            params["scheduler_params"] = scheduler_params

        # Remove other scheduler params that might have been sampled but not used
        for param in [
            "scheduler_patience",
            "scheduler_factor",
            "scheduler_min_lr",
            "scheduler_t_max",
            "scheduler_div_factor",
            "eval_metric",
            "patience",
            "max_epochs",
            "batch_size",
            "virtual_batch_size",
        ]:
            if param in params:
                params.pop(param)

        model = TabNetClassifier(**params)
        return model
    except Exception as e:
        logger.error(f"Error creating TabNet model: {str(e)}")
        raise


def train_model(X_train, y_train, X_test, y_test, X_eval, y_eval, model_params):
    """
    Train a TabNet model with early stopping.
    Converts input data to numpy arrays if they are pandas DataFrames.
    Returns the trained model and evaluation metrics after threshold optimization.
    """
    try:
        # Ensure batch_size and virtual_batch_size are retrieved for fit method
        batch_size_to_use = model_params.get("batch_size", 1024)
        max_epochs = model_params.get("max_epochs", 50)

        # Configure OneCycleLR scheduler if specified
        if model_params.get("scheduler_type") == "onecycle":
            # Calculate total steps for OneCycleLR
            # Assuming X_train is already concatenated with X_test for training
            total_samples = len(X_train) if hasattr(X_train, "__len__") else X_train.shape[0]
            steps_per_epoch = total_samples // batch_size_to_use + (
                1 if total_samples % batch_size_to_use != 0 else 0
            )
            total_steps = steps_per_epoch * max_epochs

            if "scheduler_params" in model_params:
                model_params["scheduler_params"]["total_steps"] = total_steps

        model = create_model(model_params)

        # Convert to numpy arrays if needed
        if hasattr(X_train, "values"):
            X_train = X_train.values
            y_train = y_train.values if hasattr(y_train, "values") else y_train
            X_test = X_test.values if hasattr(X_test, "values") else X_test
            y_test = y_test.values if hasattr(y_test, "values") else y_test
            X_eval = X_eval.values if hasattr(X_eval, "values") else X_eval
            y_eval = y_eval.values if hasattr(y_eval, "values") else y_eval

        # Combine training and testing data similar to xgboost_model.py
        X_combined = np.concatenate([X_train, X_test], axis=0)
        y_combined = np.concatenate([y_train, y_test], axis=0)

        # Use the class (not an instance) in the eval_metric list
        # TabNet will instantiate it internally
        model.fit(
            X_combined,
            y_combined,
            eval_set=[(X_eval, y_eval)],
            eval_metric=model_params.get("eval_metric", "auc"),
            max_epochs=max_epochs,
            patience=model_params.get("patience", 10),
            batch_size=batch_size_to_use,
            virtual_batch_size=model_params.get("virtual_batch_size", 1024),
            weights=1,
            drop_last=False,
        )

        # Log learning rate evolution if model has a scheduler
        if hasattr(model, "scheduler") and model.scheduler is not None:
            logger.info("Learning rate evolution:")
            for i, lr in enumerate(model.scheduler_history):
                if i % 10 == 0 or i == len(model.scheduler_history) - 1:
                    logger.info(f"Epoch {i}: LR = {lr:.8f}")

        # Optimize threshold using shared utility
        best_threshold, metrics = optimize_threshold(model, X_eval, y_eval, min_recall=min_recall)
        return model, metrics
    except Exception as e:
        logger.error(f"Error training TabNet model: {str(e)}")
        raise


def optimize_hyperparameters(
    X_train, y_train, X_test, y_test, X_eval, y_eval, hyperparameter_space
):
    logger.info("Starting hyperparameter optimization for TabNet")
    if not hyperparameter_space:
        hyperparameter_space = load_hyperparameter_space()
    best_score = -float("inf")
    best_params = {}
    global_top_trials = []
    top_trials = []

    def objective(trial):
        try:
            params = base_params.copy()
            # Iterate over hyperparameter space and suggest values
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
                elif param_config["type"] == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config["choices"]
                    )

            # Train model and get metrics
            model, metrics = train_model(X_train, y_train, X_test, y_test, X_eval, y_eval, params)
            recall = metrics.get("recall", 0.0)
            precision = metrics.get("precision", 0.0)
            score = precision if recall >= min_recall else 0.0
            logger.info(f"  Score: {score}")
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

    storage_url = "sqlite:///optuna_tabnet.db"
    study_name = "tabnet_optimization"
    total_trials = n_trials
    batch_size = 1000
    num_batches = total_trials // batch_size
    if total_trials % batch_size != 0:
        num_batches += 1

    for batch in range(num_batches):
        batch_random_seed = int(time.time()) + batch
        new_sampler = optuna.samplers.RandomSampler(seed=batch_random_seed)
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            storage=storage_url,
            load_if_exists=True,
            sampler=new_sampler,
        )
        logger.info(
            f"Starting batch {batch + 1}/{num_batches} with new sampler (seed={batch_random_seed})"
        )
        study.optimize(objective, n_trials=batch_size, show_progress_bar=True, callbacks=[callback])
        for trial_record in top_trials:
            global_top_trials.append(trial_record)
        global_top_trials.sort(key=lambda x: x[0], reverse=True)
        global_top_trials = global_top_trials[:10]
    if global_top_trials:
        best_score, best_params, best_trial_number = global_top_trials[0]
    else:
        best_params = {}

    best_params.update(base_params)
    logger.info(f"Best trial value across batches: {best_score:.4f}")
    logger.info(f"Best parameters found: {best_params}")
    return best_params


def hypertune_tabnet(experiment_name: str):
    """
    Main hypertuning function for TabNet with MLflow tracking.
    Returns best_params and metrics.
    """
    try:
        with mlflow.start_run(run_name=f"tabnet_base_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            mlflow.set_tags(
                {"model_type": "tabnet", "training_mode": "global", "gpu_enabled": True}
            )
            hyperparameter_space = load_hyperparameter_space()
            logger.info("Starting hyperparameter optimization for TabNet")
            best_params = optimize_hyperparameters(
                X_train,
                y_train,
                X_test,
                y_test,
                X_eval,
                y_eval,
                hyperparameter_space=hyperparameter_space,
            )
            logger.info("Training final TabNet model with best parameters")
            model, metrics = train_model(
                X_train_transformed,
                y_train,
                X_test_transformed,
                y_test,
                X_eval_transformed,
                y_eval,
                best_params,
            )
            mlflow.log_metrics(
                {
                    "precision": metrics.get("precision", 0.0),
                    "recall": metrics.get("recall", 0.0),
                    "f1": metrics.get("f1", 0.0),
                    "auc": metrics.get("auc", 0.0),
                    "threshold": metrics.get("threshold", 0.5),
                }
            )
            logger.info("Logging best parameters to MLflow")
            for param_name, param_value in best_params.items():
                mlflow.log_param(param_name, param_value)
            # Create input example from X_eval (convert to DataFrame if needed)
            if not isinstance(X_eval, pd.DataFrame):
                input_example = pd.DataFrame(X_eval[:5])
            else:
                input_example = X_eval.iloc[:5].copy()
            signature = mlflow.models.infer_signature(input_example, model.predict(input_example))
            # Log model using a custom pyfunc wrapper
            mlflow.pyfunc.log_model(
                artifact_path="model", python_model=TabNetWrapper(model), signature=signature
            )
            return best_params, metrics
    except Exception as e:
        logger.error(f"Error in TabNet hypertuning: {str(e)}")
        return None, None


def log_to_mlflow(model, metrics, params, experiment_name):
    """
    Log trained model, metrics, and parameters to MLflow using sklearn flavor.

    Args:
        model: Trained TabNet model
        metrics: Model evaluation metrics
        params: Model parameters
        experiment_name: Experiment name

    Returns:
        str: Run ID
    """
    try:
        # Set up MLflow tracking
        mlflow.set_experiment(experiment_name)
        logger.info(f"Logging model to MLflow: {experiment_name}")

        # Start a new run
        with mlflow.start_run(run_name=f"tabnet_{datetime.now().strftime('%Y%m%d_%H%M')}") as run:
            # Log parameters
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
            logger.info(f"Logged parameters: {params}")

            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            logger.info(f"Logged metrics: {metrics}")

            # Create input example for signature
            input_example = X_eval.iloc[:5] if hasattr(X_eval, "iloc") else pd.DataFrame(X_eval[:5])
            # Wrap the TabNet model in a scikit-learn compatible wrapper
            sklearn_wrapper = TabNetSklearnWrapper(model=model)
            # Create a signature for the model
            signature = mlflow.models.infer_signature(
                input_example, sklearn_wrapper.predict(input_example)
            )

            # Log the model using sklearn flavor
            model_info = mlflow.sklearn.log_model(
                sk_model=sklearn_wrapper,
                artifact_path="model",
                signature=signature,
                registered_model_name=f"tabnet_{datetime.now().strftime('%Y%m%d_%H%M')}",
            )

            # # For backward compatibility, also save in PyFunc format
            # tabnet_wrapper = TabNetWrapper(model)
            # mlflow.pyfunc.log_model(
            #     artifact_path="model_pyfunc",
            #     python_model=tabnet_wrapper,
            #     signature=signature
            # )

            logger.info(f"Model logged to MLflow: {model_info.model_uri}")
            logger.info(f"Run ID: {run.info.run_id}")
            return run.info.run_id

    except Exception as e:
        logger.error(f"Error logging to MLflow: {str(e)}")
        return None


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
                "learning_rate": 0.20164682030656028,
                "n_d": 22,
                "n_a": 9,
                "n_steps": 14,
                "gamma": 0.75,
                "lambda_sparse": 6.29086521217929e-06,
                "momentum": 0.74,
                "patience": 29,
                "max_epochs": 185,
                "batch_size": 2337,
                "virtual_batch_size": 2334,
                "verbose": 0,
                "n_independent": 4,
                "n_shared": 3,
                "weight_decay": 1.663263385027431e-05,
                "scheduler_type": "none",
                "scheduler_patience": 4,
                "scheduler_factor": 0.12562431278390354,
                "scheduler_min_lr": 5.734463997966396e-06,
                "scheduler_t_max": 14,
                "scheduler_div_factor": 19.03707394042698,
            }
        )
        # Train final model with best parameters
        logger.info("Training final model with best parameters")
        model, metrics = train_model(X_train, y_train, X_test, y_test, X_eval, y_eval, params)
        # Log to MLflow
        log_to_mlflow(model, metrics, params, experiment_name)
        return model, metrics
    except Exception as e:
        logger.error(f"Error in precision-focused training: {str(e)}")
        return None, None


def main():
    """
    Main execution function for TabNet hypertuning.
    """
    try:
        logger.info("Starting TabNet model hypertuning")
        # Setup MLflow tracking directory
        setup_mlflow_tracking(experiment_name)
        global \
            X_train, \
            y_train, \
            X_test, \
            y_test, \
            X_eval, \
            y_eval, \
            X_train_transformed, \
            X_test_transformed, \
            X_eval_transformed
        # Load data using shared DataLoader
        dataloader = DataLoader()
        X_train, y_train, X_test, y_test, X_eval, y_eval = dataloader.load_data()
        # Select features for TabNet if needed
        features = import_selected_features_ensemble(model_type="tabnet")
        X_train = X_train[features]
        X_test = X_test[features]
        X_eval = X_eval[features]
        # Convert all columns to float64 to ensure consistent data types
        X_train = X_train.astype("float64")
        X_test = X_test.astype("float64")
        X_eval = X_eval.astype("float64")
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Testing data shape: {X_test.shape}")
        logger.info(f"Evaluation data shape: {X_eval.shape}")
        logger.info(
            f"Positive class ratio - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}, Eval: {y_eval.mean():.3f}"
        )
        logger.info(f"Current base parameters: {base_params}")

        # Before training
        preprocessor = QuantileTransformer(output_distribution="normal")
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        X_eval_transformed = preprocessor.transform(X_eval)

        best_params, metrics = hypertune_tabnet(experiment_name)
        logger.info(f"Hypertuning completed with parameters: {best_params}")
        logger.info(f"Evaluation metrics: {metrics}")

        # Train model with precision target
        best_model, best_metrics = train_with_precision_target(
            X_train_transformed, y_train, X_test_transformed, y_test, X_eval_transformed, y_eval
        )
        logger.info(f"Best model: {best_model}")
        logger.info(f"Best metrics: {best_metrics}")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")


if __name__ == "__main__":
    main()
