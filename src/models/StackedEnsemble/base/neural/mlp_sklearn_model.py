"""
MLP Model (scikit-learn) for Soccer Draw Prediction

This module implements a simpler MLP-based model using scikit-learn's MLPClassifier
for predicting soccer match draws. It includes Optuna for hyperparameter
tuning (optimizing precision subject to min recall) and MLflow integration,
following a structure similar to other base models.
"""

import gc
import os
import pickle
import random
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.sklearn  # Use sklearn logging
import numpy as np
import optuna  # Import Optuna
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network._stochastic_optimizers import AdamOptimizer, SGDOptimizer
from sklearn.preprocessing import KBinsDiscretizer, RobustScaler, StandardScaler

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

experiment_name = "mlp_model"  # Updated experiment name
logger = ExperimentLogger(experiment_name=experiment_name)

# Set random seeds for reproducibility
random_seed = 19
random.seed(random_seed)
np.random.seed(random_seed)
os.environ["PYTHONHASHSEED"] = str(random_seed)

# Import shared utility for threshold optimization
from models.StackedEnsemble.shared.hypertuner_utils import optimize_threshold
from utils.create_evaluation_set import import_selected_features_ensemble, setup_mlflow_tracking

mlflow_tracking = setup_mlflow_tracking(experiment_name)
from models.StackedEnsemble.shared.data_loader import DataLoader  # Custom data loader

# Global settings
n_trials_optuna = 10000  # Number of Optuna trials
min_recall = 0.30  # Minimum acceptable recall for optimization objective
pip_requirements = [
    "scikit-learn",
    f"mlflow=={mlflow.__version__}",
    f"optuna=={optuna.__version__}",  # Add Optuna
]
scaler = None  # Global scaler object - Will be fitted in hypertune function
X_train, y_train, X_test, y_test, X_eval, y_eval = (None,) * 6  # Global data variables
# --- Base Parameters ---

base_params = {
    # 'hidden_layer_sizes': (128,), # REMOVED - Will be set dynamically in objective
    "activation": "relu",  # ReLU activation function
    "early_stopping": True,  # Enable early stopping
    "random_state": random_seed,  # For reproducibility
    "verbose": False,
}


# --- Parameter Space Function (Used by Optuna) ---
def load_hyperparameter_space_sklearn(solver="adam"):
    """Defines the hyperparameter search space configuration for Optuna."""
    hyperparameter_space = {}
    if solver == "adam":
        # --- ADAM Hyperparameter Space ---
        logger.info("Loading hyperparameter space for ADAM solver")
        hyperparameter_space = {
            "alpha": {"type": "float", "low": 1e-9, "high": 10.0, "log": True},
            "learning_rate_init": {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "batch_size": {"type": "int", "low": 256, "high": 4096, "log": False, "step": 64},
            "max_iter": {"type": "int", "low": 50, "high": 200, "log": False, "step": 5},
            "n_iter_no_change": {"type": "int", "low": 3, "high": 30},
            "beta_1": {"type": "float", "low": 0.75, "high": 0.95, "log": False, "step": 0.01},
            "beta_2": {"type": "float", "low": 0.98, "high": 0.9999, "log": False, "step": 0.0001},
            # --- Custom parameters for Adam/SGD based MLPDropoutClassifier ---
            # 'dropout': { 'type': 'float', 'low': 0.0, 'high': 0.5, 'log': False },
            # 'lr_decay_rate': { 'type': 'float', 'low': 0.8, 'high': 0.99, 'log': False },
            # 'lr_decay_steps': { 'type': 'int', 'low': 10, 'high': 100, 'log': False }
            # Note: Dropout and LR schedule are part of the custom class, maybe tune alpha directly instead
        }
    elif solver == "lbfgs":
        # --- LBFGS Hyperparameter Space ---
        logger.info("Loading hyperparameter space for LBFGS solver")
        hyperparameter_space = {
            "alpha": {  # L2 regularization strength
                "type": "float",
                "low": 1e-7,  # LBFGS might benefit from smaller alpha
                "high": 1.0,
                "log": True,
            },
            "max_iter": {  # Max iterations for LBFGS
                "type": "int",
                "low": 500,  # LBFGS might need more iterations
                "high": 2000,  # Increased upper bound
                "log": False,
                "step": 50,
            },
            "tol": {  # Tolerance for convergence
                "type": "float",
                "low": 1e-6,
                "high": 1e-3,
                "log": True,
            },
            # Note: LBFGS does not use batch_size, learning_rate_init, betas, momentum, early_stopping params
        }
    else:
        raise ValueError(f"Unsupported solver for hyperparameter space: {solver}")

    return hyperparameter_space


# --- Architecture Rotation Definition ---
# Define the fixed sequence of architectures to cycle through
ARCHITECTURE_ROTATION = [
    (128, 64),
    (128, 64, 32),
    (256, 128, 64),  # Wider network
    (128, 64, 32, 16),  # Deeper network
    (256, 128, 64, 32),  # Both wider and deeper
]
logger.info(f"Defined architecture rotation: {ARCHITECTURE_ROTATION}")
# -------------------------------------


def preprocess_data(X_train_local, X_test_local, X_eval_local, use_robust_scaler=True):
    """
    Preprocess data using RobustScaler (default) or StandardScaler.
    Fits scaler on train, transforms all.
    Uses local copies of data passed in.
    Returns: scaled X_train, X_test, X_eval, and the fitted scaler.
    """
    if use_robust_scaler:
        # Use RobustScaler for better handling of outliers
        local_scaler = RobustScaler()
        logger.info("Using RobustScaler for preprocessing")
    else:
        # Fall back to StandardScaler if requested
        local_scaler = StandardScaler()
        logger.info("Using StandardScaler for preprocessing")

    X_train_scaled = local_scaler.fit_transform(X_train_local)
    X_test_scaled = local_scaler.transform(X_test_local)
    X_eval_scaled = local_scaler.transform(X_eval_local)
    logger.info("Data scaling complete")
    return X_train_scaled, X_test_scaled, X_eval_scaled, local_scaler


# Add this function to your code
def safe_sparse_dot(a, b, dense_output=False):
    """
    Dot product that handles both dense and sparse matrices.

    Parameters
    ----------
    a, b : array or sparse matrix
    dense_output : bool, default=False
        When False, a sparse matrix is returned if at least one input is sparse.
        When True, always return a dense array.

    Returns
    -------
    dot_product : array or sparse matrix
        Sparse if a or b is sparse and dense_output=False.
    """
    if a is None or b is None:
        raise ValueError("Input matrices cannot be None")

    import numpy as np
    from scipy import sparse

    if sparse.issparse(a) or sparse.issparse(b):
        if dense_output:
            # Convert sparse to dense
            a_dense = a.toarray() if sparse.issparse(a) else a
            b_dense = b.toarray() if sparse.issparse(b) else b
            return np.dot(a_dense, b_dense)
        else:
            # Use appropriate sparse dot products
            if sparse.issparse(a) and sparse.issparse(b):
                return a @ b
            elif sparse.issparse(a):
                return a @ b
            else:
                return a @ b
    else:
        # Both dense, use numpy's dot
        return np.dot(a, b)


# Custom MLP class with dropout and learning rate scheduling
class MLPDropoutClassifier(MLPClassifier):
    """
    Custom MLPClassifier with dropout support and learning rate scheduling.

    Additional Parameters:
    ----------
    dropout : float in range (0, 1), default=None
        Dropout rate for each layer (fraction of neurons to drop during training)
    lr_schedule : dict, default=None
        Learning rate schedule with format {'decay_rate': float, 'decay_steps': int}
    """

    def __init__(
        self,
        hidden_layer_sizes=(100,),
        activation="relu",
        *,
        solver="adam",
        alpha=0.0001,
        batch_size="auto",
        learning_rate="constant",
        learning_rate_init=0.001,
        power_t=0.5,
        max_iter=200,
        shuffle=True,
        random_state=None,
        tol=1e-4,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        n_iter_no_change=10,
        max_fun=15000,
        dropout=None,
        lr_schedule=None,
    ):
        # Store custom parameters
        self.dropout = dropout
        self.lr_schedule = lr_schedule
        self.current_lr = learning_rate_init
        self.initial_lr = learning_rate_init
        self.current_step = 0

        # Initialize parent class
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            power_t=power_t,
            max_iter=max_iter,
            shuffle=shuffle,
            random_state=random_state,
            tol=tol,
            verbose=verbose,
            warm_start=warm_start,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            n_iter_no_change=n_iter_no_change,
            max_fun=max_fun,
        )

    def _fit_stochastic(
        self, X, y, activations, deltas, coef_grads, intercept_grads, layer_units, incremental
    ):
        """Implementation of stochastic gradient descent with learning rate scheduling."""
        params = self.coefs_ + self.intercepts_

        # Create optimizer if not already created or if not incremental
        if not incremental or not hasattr(self, "_optimizer"):
            if self.solver == "sgd":
                self._optimizer = SGDOptimizer(
                    params,
                    self.learning_rate_init,
                    self.learning_rate,
                    self.momentum,
                    self.nesterovs_momentum,
                    self.power_t,
                )
            elif self.solver == "adam":
                self._optimizer = AdamOptimizer(
                    params,
                    self.learning_rate_init,
                    self.beta_1,
                    self.beta_2,
                    self.epsilon,
                )

        # Call parent implementation
        result = super()._fit_stochastic(
            X, y, activations, deltas, coef_grads, intercept_grads, layer_units, incremental
        )

        # Apply learning rate scheduling if defined
        if self.lr_schedule is not None:
            decay_rate = self.lr_schedule.get("decay_rate", 0.9)
            decay_steps = self.lr_schedule.get("decay_steps", 100)

            # Update step counter
            self.current_step += 1

            # Calculate new learning rate based on exponential decay
            if self.current_step % decay_steps == 0:
                self.current_lr = self.initial_lr * (
                    decay_rate ** (self.current_step / decay_steps)
                )

                # Update optimizer's learning rate
                if hasattr(self._optimizer, "learning_rate"):
                    self._optimizer.learning_rate = self.current_lr
                    if self.verbose:
                        print(
                            f"Step {self.current_step}: Updated learning rate to {self.current_lr}"
                        )

        return result

    def fit(self, X, y):
        """
        Fit the model using the parent implementation.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values

        Returns
        -------
        self : returns a trained MLP model.
        """
        # If dropout is enabled, apply a simple regularization approach
        if self.dropout is not None and self.dropout > 0:
            # Increase alpha (L2 regularization) to compensate for dropout
            adjusted_alpha = self.alpha / (1.0 - self.dropout)
            logger.info(
                f"Applying dropout-like regularization with adjusted alpha: {adjusted_alpha}"
            )

            # Temporarily modify alpha
            original_alpha = self.alpha
            self.alpha = adjusted_alpha

            # Train with increased regularization
            result = super().fit(X, y)

            # Restore original alpha
            self.alpha = original_alpha
            return result
        else:
            # No dropout, just use standard training
            return super().fit(X, y)

    def predict(self, X):
        """Predict using the neural network model."""
        return super().predict(X)

    def predict_proba(self, X):
        """Probability estimates."""
        return super().predict_proba(X)


# Update the create_model_sklearn function to handle LBFGS and use standard MLPClassifier
def create_model_sklearn(model_params):
    """Creates an MLPClassifier instance with given parameters."""
    # Determine solver, default to adam if not specified
    solver = model_params.get("solver", "adam")
    # Start with base parameters appropriate for the solver
    params = base_params.copy()  # Keep base_params as adam defaults for now
    params["solver"] = solver  # Ensure solver is set
    # Filter model_params to only include those relevant to the chosen solver
    relevant_params = {}
    valid_params_for_solver = (
        MLPClassifier().get_params().keys()
    )  # Get valid params for standard MLPClassifier

    for key, value in model_params.items():
        if key in valid_params_for_solver:
            # Filter out params not used by LBFGS if that's the solver
            if solver == "lbfgs":
                lbfgs_ignored_params = [
                    "learning_rate",
                    "learning_rate_init",
                    "power_t",
                    "shuffle",
                    "momentum",
                    "nesterovs_momentum",
                    "early_stopping",
                    "validation_fraction",
                    "beta_1",
                    "beta_2",
                    "epsilon",
                    "n_iter_no_change",
                    "batch_size",
                ]
                if key not in lbfgs_ignored_params:
                    relevant_params[key] = value
            else:  # Keep param if solver is adam (or others, assuming MLPClassifier handles them)
                relevant_params[key] = value
        elif key in ["dropout", "lr_schedule"] and solver != "lbfgs":
            # Keep custom params only if NOT using LBFGS (and potentially using MLPDropoutClassifier)
            relevant_params[key] = value

    # Update base params with relevant suggested/fixed params
    params.update(relevant_params)
    # Remove parameters not applicable to the final chosen solver
    final_params = {}
    if solver == "lbfgs":
        lbfgs_specific_params = [
            "hidden_layer_sizes",
            "activation",
            "solver",
            "alpha",
            "max_iter",
            "random_state",
            "tol",
            "verbose",
            "warm_start",
            "max_fun",
        ]
        for k, v in params.items():
            if k in lbfgs_specific_params:
                final_params[k] = v
    else:  # Assume Adam or other MLPClassifier compatible solver
        final_params = {k: v for k, v in params.items() if k in valid_params_for_solver}

    logger.info(
        f"Creating MLPClassifier model with solver '{solver}' and final params: {final_params}"
    )
    try:
        # Always use standard MLPClassifier - custom class logic was for stochastic optimizers
        model = MLPClassifier(**final_params)
        return model
    except Exception as e:
        logger.error(f"Error creating MLPClassifier model: {str(e)}")
        logger.error(f"Parameters attempted: {final_params}")
        raise


# --- Model Training ---
def train_model_sklearn(X_train_scaled, y_train_local, X_eval_scaled, y_eval_local, model_params):
    """Trains an MLPClassifier model and evaluates using optimize_threshold."""
    try:
        # Create the model - create_model_sklearn now handles solver specifics
        model = create_model_sklearn(model_params)
        solver = model.solver  # Get the actual solver used
        # Train model
        logger.info(f"Fitting MLPClassifier with solver: {solver}")
        # No need for ConvergenceWarning filter with LBFGS typically
        if solver == "lbfgs":
            model.fit(X_train_scaled, y_train_local)
        else:  # Keep filter for adam/sgd
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
                model.fit(X_train_scaled, y_train_local)

        # Evaluate using the shared threshold optimization logic
        best_threshold, metrics = optimize_threshold(
            model, X_eval_scaled, y_eval_local, min_recall=min_recall
        )
        return model, metrics  # Return fitted model and metrics
    except Exception as e:
        logger.error(f"Error training MLPClassifier model: {str(e)}")
        raise


# --- Helper Function for Logging Top Trials ---
def log_top_trials(trials_list, title="Top Trials"):
    """Helper function to log top trials in a formatted table."""
    table_header = "| Rank | Trial # | Score  | Parameters |"
    table_separator = "|------|---------|--------|------------|"

    logger.info(f"{title}:")
    table_rows = [
        f"| {i + 1} | {rec[2]} | {rec[0]:.4f} | {rec[1]} |"
        for i, rec in enumerate(trials_list[:10])
    ]
    logger.info(table_header)
    logger.info(table_separator)
    for row in table_rows:
        logger.info(row)


def apply_feature_binning(
    X_train_scaled, X_test_scaled, X_eval_scaled, n_bins=30, strategy="quantile"
):
    """
    Apply feature binning to discover nonlinear patterns in the data.

    Args:
        X_train_scaled: Scaled training data
        X_test_scaled: Scaled test data
        X_eval_scaled: Scaled evaluation data
        n_bins: Number of bins to use for discretization
        strategy: Strategy for binning ('uniform', 'quantile', or 'kmeans')

    Returns:
        Tuple of binned data arrays and the fitted discretizer
    """
    logger.info(f"Applying feature binning with {n_bins} bins using {strategy} strategy")

    # Initialize KBinsDiscretizer with specified parameters
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode="onehot-dense", strategy=strategy)

    # Convert to numpy arrays if they're not already
    if not isinstance(X_train_scaled, np.ndarray):
        X_train_scaled = np.array(X_train_scaled)
    if not isinstance(X_test_scaled, np.ndarray):
        X_test_scaled = np.array(X_test_scaled)
    if not isinstance(X_eval_scaled, np.ndarray):
        X_eval_scaled = np.array(X_eval_scaled)

    # Fit and transform the training data
    X_train_binned = discretizer.fit_transform(X_train_scaled)

    # Transform the test and evaluation data
    X_test_binned = discretizer.transform(X_test_scaled)
    X_eval_binned = discretizer.transform(X_eval_scaled)

    logger.info(
        f"Feature binning applied. New shapes - Train: {X_train_binned.shape}, Test: {X_test_binned.shape}, Eval: {X_eval_binned.shape}"
    )

    return X_train_binned, X_test_binned, X_eval_binned, discretizer


def optimize_hyperparameters_optuna_sklearn(
    X_train_scaled, y_train_local, X_eval_scaled, y_eval_local, solver_to_tune="lbfgs"
):
    """Runs Optuna optimization using the structured train_model function."""
    logger.info(
        f"Starting Optuna optimization for '{solver_to_tune}' solver with {n_trials_optuna} trials."
    )
    # Load hyperparameter space specific to the chosen solver
    hyperparameter_space_config = load_hyperparameter_space_sklearn(solver=solver_to_tune)
    # Variables to track best score and params
    best_score = -float("inf")
    best_params_in_run = {}
    top_trials = []

    # Define Objective function for Optuna
    def objective(trial):
        params = {"solver": solver_to_tune}  # Explicitly set the solver for this trial
        # --- Determine Architecture Dynamically (Keep this logic) ---
        trial_num = trial.number
        num_archs = len(ARCHITECTURE_ROTATION)
        # Cycle architecture less frequently for LBFGS as it might take longer? Adjust as needed.
        arch_change_freq = 10
        arch_index = (trial_num // arch_change_freq) % num_archs
        current_hidden_layers = ARCHITECTURE_ROTATION[arch_index]
        params["hidden_layer_sizes"] = current_hidden_layers
        # --- Suggest Other Hyperparameters (Based on loaded space) ---
        for name, config in hyperparameter_space_config.items():
            # Skip parameters handled elsewhere or manually set
            if name in ["hidden_layer_sizes"]:
                continue
            # Suggest parameter based on its type defined in the space config
            if config["type"] == "int":
                step = config.get("step", 1)
                params[name] = trial.suggest_int(
                    name, config["low"], config["high"], step=step, log=config.get("log", False)
                )
            elif config["type"] == "float":
                log = config.get("log", False)
                step = config.get("step")
                if step:
                    params[name] = trial.suggest_float(
                        name, config["low"], config["high"], step=step, log=log
                    )
                else:
                    params[name] = trial.suggest_float(name, config["low"], config["high"], log=log)

        try:
            # train_model_sklearn internally calls create_model_sklearn which now handles solver specifics
            model, metrics = train_model_sklearn(
                X_train_scaled,
                y_train_local,
                X_eval_scaled,
                y_eval_local,
                params,  # Pass suggested params (including solver)
            )
            recall = metrics.get("recall", 0.0)
            precision = metrics.get("precision", 0.0)
            threshold = metrics.get("threshold", 0.5)
            # Optimize for precision while maintaining minimum recall
            score = precision if recall >= min_recall else 0.0
            logger.info(
                f"Trial {trial.number} ({solver_to_tune}): Score={score:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, Thresh={threshold:.3f}, Arch={params.get('hidden_layer_sizes')}"
            )
            # logger.debug(f"Trial {trial.number} Params: {params}") # Debug logging if needed
            # Log metrics to trial attributes
            for metric_name, metric_value in metrics.items():
                trial.set_user_attr(metric_name, metric_value)
            return score
        except Exception as e:
            logger.error(f"Optuna trial {trial.number} ({solver_to_tune}) failed: {e}")
            logger.debug(f"Failed Params: {params}")  # Log params that caused failure
            return 0.0  # Return low score for failed trials

    # --- Optuna Study Setup & Execution (Callback remains largely the same) ---
    # Callback function (modified slightly for clarity)
    def callback(study, trial):
        nonlocal best_score, best_params_in_run, top_trials
        if trial.value is not None and trial.state == optuna.trial.TrialState.COMPLETE:
            # Use study.best_value which tracks the best across the whole study run (safer)
            if study.best_value is not None and study.best_value > best_score:
                best_score = study.best_value
                best_params_in_run = study.best_params  # Update with study's best
                logger.info(
                    f"New best score found via study: {best_score:.4f} in trial {study.best_trial.number}"
                )
                logger.info(f"Best params so far: {best_params_in_run}")
            # --- Top trials tracking (remains the same logic) ---
            current_run_metrics = trial.user_attrs  # Get metrics logged earlier
            current_run_score = trial.value
            current_run_params = trial.params
            current_run_number = trial.number
            # Create a record for the current trial
            # Store score, params, number, precision, recall
            trial_record = (
                current_run_score,
                current_run_params,
                current_run_number,
                current_run_metrics.get("precision", 0.0),
                current_run_metrics.get("recall", 0.0),
            )
            # Add to top trials list
            top_trials.append(trial_record)
            top_trials.sort(key=lambda x: x[0] if x[0] is not None else float("-inf"), reverse=True)
            top_trials[:] = top_trials[:10]  # Keep only top 10
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

    global_top_trials = []  # Track top trials across all batches
    # Reset best score/params for this run
    best_score = float("-inf")
    best_params_in_run = {}  # Use this to return the best params found specifically in this call

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

    logger.info(f"Optuna optimization finished for '{solver_to_tune}'.")
    if best_params_in_run:
        logger.info(f"Best score found across all batches: {best_score:.4f}")
        logger.info(f"Best parameters found: {best_params_in_run}")
        log_top_trials(
            [(t[0], t[1], t[2]) for t in global_top_trials],
            f"Final Top 10 Trials ({solver_to_tune})",
        )  # Use consistent format for logging
    else:
        best_params = {}

    logger.info(f"Best trial value across batches: {best_score:.4f}")
    logger.info(f"Best parameters found: {best_params}")
    # Log final top 10 trials
    log_top_trials(top_trials, "Final Top 10 Trials")
    # Return parameters tracked by the callback, falling back to study object if needed
    return best_params_in_run if best_params_in_run else study.best_params


def log_to_mlflow_sklearn(model, metrics, params, fitted_scaler, discretizer, input_example):
    """Logs parameters, metrics, scaler, and model to MLflow."""
    logger.info("Logging results to MLflow...")
    # Log best parameters found by optuna
    mlflow.log_params({f"best_optuna_{k}": v for k, v in params.items()})
    # Log evaluation metrics (including custom threshold metrics)
    mlflow.log_metrics(metrics)
    # Log the scaler
    scaler_path = "scaler_robust.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(fitted_scaler, f)
    mlflow.log_artifact(scaler_path)
    logger.info("Scaler artifact logged.")

    discretizer_path = "discretizer.pkl"
    with open(discretizer_path, "wb") as f:
        pickle.dump(discretizer, f)
    mlflow.log_artifact(discretizer_path)
    logger.info("Discretizer artifact logged.")

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
    signature = mlflow.models.infer_signature(input_example, model.predict(input_example))
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model_sklearn_precision",
        registered_model_name=f"mlp_sklearn_precision_{datetime.now().strftime('%Y%m%d_%H%M')}",
        signature=signature,
        pip_requirements=pip_requirements,
    )

    run_id = mlflow.active_run().info.run_id
    logger.info(f"Run ID: {run_id}")
    logger.info("Scikit-learn MLP model logged successfully.")


def hypertune_mlp_sklearn(experiment_name: str):
    """
    Orchestrates the MLPClassifier tuning (Optuna), evaluation, and logging process.
    Allows specifying the solver to tune.
    """
    global X_train, y_train, X_test, y_test, X_eval, y_eval  # Access global data
    solver_to_tune = "lbfgs"  # <<< Set to 'lbfgs' or 'adam'

    if X_train is None:  # Check if data is loaded
        logger.error("Data (X_train) is not loaded. Aborting.")
        return None, None
    try:
        logger.info(f"--- Starting hypertuning process for MLP with '{solver_to_tune}' solver ---")
        # --- Preprocessing (Remains the same) ---
        X_train_scaled, X_test_scaled, X_eval_scaled, fitted_scaler = preprocess_data(
            X_train.copy(), X_test.copy(), X_eval.copy(), use_robust_scaler=True
        )
        # Feature binning logic (remains the same, ensure discretizer is handled)
        use_binning = False
        discretizer = None  # Initialize discretizer
        if use_binning:
            X_train_scaled, X_test_scaled, X_eval_scaled, discretizer = apply_feature_binning(
                X_train_scaled, X_test_scaled, X_eval_scaled, n_bins=10, strategy="quantile"
            )
            logger.info("Applied feature binning.")
        else:
            logger.info("Feature binning not applied.")

        X_train_scaled = X_train_scaled.astype("float64")
        X_test_scaled = X_test_scaled.astype("float64")
        X_eval_scaled = X_eval_scaled.astype("float64")
        # Combine X_train and X_test for full training set (using scaled data)
        logger.info("Combining scaled training and test sets for final model training...")
        X_train_combined_scaled = np.vstack((X_train_scaled, X_test_scaled))
        y_train_combined = np.concatenate((y_train, y_test))
        logger.info(f"Combined scaled training set shape: {X_train_combined_scaled.shape}")

        run_name_prefix = f"mlp_{solver_to_tune}_optuna_hypertune"
        with mlflow.start_run(
            run_name=f"{run_name_prefix}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        ):
            mlflow.set_tags(
                {
                    "model_type": "mlp_sklearn",
                    "solver": solver_to_tune,
                    "tuning_method": "Optuna",
                    "optuna_trials": n_trials_optuna,
                    "optimization_metric": f"precision_at_min_recall_{min_recall}",
                }
            )

            # --- Hyperparameter Optimization (Optuna) ---
            # Pass the chosen solver to the optimization function
            best_params = optimize_hyperparameters_optuna_sklearn(
                X_train_combined_scaled,
                y_train_combined,  # Use combined data for tuning evaluation? Or just train? Let's use eval set for objective
                X_eval_scaled,
                y_eval,  # Pass eval data to objective function
                solver_to_tune=solver_to_tune,
            )
            if not best_params:
                logger.error(f"Optuna optimization failed for '{solver_to_tune}'. Aborting.")
                if mlflow.active_run():
                    mlflow.end_run("FAILED")
                return None, None

            # Add solver to best_params if not already present (Optuna might not include fixed params)
            best_params["solver"] = solver_to_tune

            # --- Train Final Model ---
            logger.info(
                f"Training final model with best Optuna parameters for '{solver_to_tune}'..."
            )
            # Create model using best params (create_model_sklearn handles solver specifics)
            final_model = create_model_sklearn(best_params)

            # Fit final model on the combined scaled training data
            logger.info("Fitting final model on combined scaled train/test data...")
            if solver_to_tune == "lbfgs":
                final_model.fit(X_train_combined_scaled, y_train_combined)
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
                    # Need to pass validation data if early stopping is True
                    if best_params.get("early_stopping", False):
                        logger.info(
                            "Using original test set as validation for early stopping during final Adam training."
                        )
                        final_model.fit(
                            X_train_scaled, y_train
                        )  # Fit only on initial train, stopping based on test
                    else:
                        logger.info("Fitting on combined data without early stopping")
                        final_model.fit(X_train_combined_scaled, y_train_combined)

            # --- Final Evaluation (on separate evaluation set) ---
            logger.info("Evaluating final model on the evaluation set...")
            final_threshold, final_metrics = optimize_threshold(
                final_model, X_eval_scaled, y_eval, min_recall=min_recall
            )
            final_metrics["threshold"] = final_threshold  # Ensure threshold is included
            logger.info(f"Final evaluation metrics ({solver_to_tune}): {final_metrics}")

            # --- MLflow Logging ---
            # Create input example from SCALED eval data
            try:
                columns = X_eval.columns  # Use original columns if available
            except AttributeError:
                columns = [f"feature_{i}" for i in range(X_eval_scaled.shape[1])]
            # Ensure input example is DataFrame for signature inference
            input_example_df = pd.DataFrame(X_eval_scaled[:5], columns=columns)
            # Pass relevant params, scaler, discretizer, input_example
            log_to_mlflow_sklearn(
                final_model,
                final_metrics,
                best_params,
                fitted_scaler,
                discretizer,
                input_example_df,
            )
            return final_model, final_metrics
    except Exception as e:
        logger.error(f"Error in hypertune_mlp_sklearn ({solver_to_tune}): {str(e)}")
        if mlflow.active_run():
            mlflow.end_run("FAILED")
        return None, None


def train_with_precision_target(
    X_train, y_train, X_test, y_test, X_eval, y_eval, solver="lbfgs"
):  # Added solver argument
    """
    Train MLP model with focus on precision target, using a specified solver.
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Testing features
        y_test: Testing labels
        X_eval: Evaluation features
        y_eval: Evaluation labels
        solver: Solver to use ('lbfgs' or 'adam')
    Returns:
        tuple: (best_model, best_metrics)
    """
    try:
        logger.info(f"--- Training model with precision-focused target using '{solver}' solver ---")
        # Define fixed parameters based on the chosen solver
        if solver == "lbfgs":
            params = {
                "solver": "lbfgs",
                "hidden_layer_sizes": (128, 64, 32),  # Keep architecture or adjust if needed
                "activation": "relu",
                "alpha": 1e-5,  # Example: Adjust based on potential LBFGS tuning results
                "max_iter": 500,  # Example: Use a reasonable max_iter for LBFGS
                "tol": 1e-5,  # Example: Tighter tolerance
                "random_state": random_seed,
                "verbose": False,
                # Add other relevant LBFGS params if needed, e.g., max_fun
            }
            logger.info(f"Using fixed LBFGS parameters: {params}")
        elif solver == "adam":
            # Use the previous Adam parameters or define new fixed ones
            params = {
                "solver": "adam",
                "hidden_layer_sizes": (128, 64, 32),
                "activation": "relu",
                "alpha": 0.07587979834232715,  # Previous value
                "batch_size": 2112,  # Previous value
                "learning_rate_init": 0.04423572731623534,  # Previous value
                "max_iter": 140,  # Previous value - maybe increase for final?
                "n_iter_no_change": 17,  # Previous value
                "beta_1": 0.95,  # Previous value
                "beta_2": 0.9851,  # Previous value
                "early_stopping": True,  # Use early stopping with Adam
                "validation_fraction": 0.1,  # Default or specify
                "random_state": random_seed,
                "verbose": False,
                # Note: Dropout/LR schedule from custom class not included here, using standard MLPClassifier
            }
            logger.info(f"Using fixed Adam parameters: {params}")
        else:
            raise ValueError(f"Unsupported solver for precision training: {solver}")

        # --- Data Scaling (remains the same) ---
        X_train_scaled, X_test_scaled, X_eval_scaled, fitted_scaler = preprocess_data(
            X_train.copy(), X_test.copy(), X_eval.copy(), use_robust_scaler=True
        )
        # Feature binning logic (remains the same, ensure discretizer is handled)
        use_binning = False
        discretizer = None  # Initialize discretizer
        if use_binning:
            X_train_scaled, X_test_scaled, X_eval_scaled, discretizer = apply_feature_binning(
                X_train_scaled, X_test_scaled, X_eval_scaled, n_bins=10, strategy="quantile"
            )
            logger.info("Applied feature binning.")
        else:
            logger.info("Feature binning not applied.")

        X_train_scaled = X_train_scaled.astype("float64")
        X_test_scaled = X_test_scaled.astype("float64")
        X_eval_scaled = X_eval_scaled.astype("float64")

        # --- Combine Data (remains the same) ---
        logger.info("Combining scaled training and test sets for final model training...")
        X_train_combined_scaled = np.vstack((X_train_scaled, X_test_scaled))
        y_train_combined = np.concatenate((y_train, y_test))
        logger.info(f"Combined scaled training set shape: {X_train_combined_scaled.shape}")

        # --- Create and Train Model ---
        # create_model_sklearn handles solver specifics and uses standard MLPClassifier
        model = create_model_sklearn(params)
        logger.info(f"Fitting model with solver: {solver}")
        if solver == "lbfgs":
            model.fit(X_train_combined_scaled, y_train_combined)
        else:  # Adam
            # Use early stopping if enabled in params
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
                # Need to pass validation data if early stopping is True
                if params.get("early_stopping", False):
                    # We need a validation set separate from the eval set for early stopping during final training
                    # Option 1: Split combined train again (introduces slight data difference from tuning)
                    # Option 2: Use the original X_test_scaled/y_test as validation (simpler)
                    logger.info(
                        "Using original test set as validation for early stopping during final Adam training."
                    )
                    model.fit(
                        X_train_scaled, y_train
                    )  # Fit only on initial train, stopping based on test
                    # Or fit on combined and rely on internal split? Let's stick to internal split for simplicity now.
                    # logger.info("Fitting on combined data with early stopping (internal validation split)")
                    # model.fit(X_train_combined_scaled, y_train_combined) # This uses internal split based on validation_fraction
                else:
                    logger.info("Fitting on combined data without early stopping")
                    model.fit(X_train_combined_scaled, y_train_combined)

        # --- Evaluate (on separate evaluation set) ---
        logger.info("Evaluating final model on the evaluation set...")
        best_threshold, metrics = optimize_threshold(
            model, X_eval_scaled, y_eval, min_recall=min_recall
        )
        metrics["threshold"] = best_threshold  # Add threshold to metrics
        logger.info(f"Final evaluation metrics ({solver}): {metrics}")

        # --- Log to MLflow ---
        # Create input example from SCALED eval data
        input_example_df = X_eval_scaled[:5]  # Remove column names to avoid warning
        # Pass relevant params, scaler, discretizer, input_example
        log_to_mlflow_sklearn(model, metrics, params, fitted_scaler, discretizer, input_example_df)
        return model, metrics
    except Exception as e:
        logger.error(f"Error in precision-focused training ({solver}): {str(e)}")
        if mlflow.active_run():
            mlflow.end_run("FAILED")
        return None, None


def main():
    """
    Main execution function for scikit-learn MLP tuning and final training.
    Can be configured to run tuning or precision training for a specific solver.
    """
    global X_train, y_train, X_test, y_test, X_eval, y_eval
    # --- Configuration ---
    # Choose mode: 'tune' or 'precision_train'
    mode = "tune"
    # Choose solver: 'lbfgs' or 'adam'
    solver = "lbfgs"
    # --------------------
    try:
        logger.info("--- Starting MLP (sklearn) Main Execution ---")
        logger.info(f"Mode: {mode}, Solver: {solver}")
        # Setup MLflow tracking
        setup_mlflow_tracking(experiment_name)  # Ensure experiment_name is defined
        # Load data
        logger.info("Loading data...")
        data_loader_instance = DataLoader()
        X_train, y_train, X_test, y_test, X_eval, y_eval = data_loader_instance.load_data()
        # Feature Selection
        try:
            features = import_selected_features_ensemble(model_type="mlp")
            logger.info(f"Using 'mlp' feature set with {len(features)} features.")
        except (KeyError, FileNotFoundError):
            logger.warning("Shared features ('mlp') not found. Falling back to 'all' features.")
            features = import_selected_features_ensemble(model_type="all")

        X_train = X_train[features]
        X_test = X_test[features]
        X_eval = X_eval[features]
        # Convert columns to float64 AFTER selection
        X_train = X_train.astype("float64")
        X_test = X_test.astype("float64")
        X_eval = X_eval.astype("float64")
        logger.info(
            f"Data shapes after selection - Train: {X_train.shape}, Test: {X_test.shape}, Eval: {X_eval.shape}"
        )
        logger.info(
            f"Positive class ratio - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}, Eval: {y_eval.mean():.3f}"
        )

        # --- Execute chosen mode ---
        best_model = None
        final_metrics = None
        if mode == "tune":
            logger.info(f"--- Running Hyperparameter Tuning for '{solver}' ---")
            best_model, final_metrics = hypertune_mlp_sklearn(
                experiment_name
            )  # It will use the 'solver_to_tune' set inside
        elif mode == "precision_train":
            logger.info(f"--- Running Precision-Focused Training for '{solver}' ---")
            best_model, final_metrics = train_with_precision_target(
                X_train, y_train, X_test, y_test, X_eval, y_eval, solver=solver
            )
        else:
            logger.error(f"Invalid mode selected: {mode}")

        # --- Log Results ---
        if best_model and final_metrics:
            logger.info(f"'{mode}' mode for '{solver}' completed successfully.")
            logger.info(f"Final Model Eval Metrics: {final_metrics}")
        else:
            logger.error(f"MLP (sklearn) '{mode}' process for '{solver}' failed.")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
    finally:
        gc.collect()
        logger.info("--- MLP (sklearn) Main execution finished. ---")


if __name__ == "__main__":
    main()
