"""
MLP Model for Soccer Draw Prediction

This module implements an MLP-based model for predicting soccer match draws.
It includes functionality for model creation, training, hyperparameter optimization,
threshold tuning, and MLflow integration for experiment tracking.

The implementation focuses on high precision while maintaining a minimum recall threshold.
"""

import os
import sys
import json
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
# import tensorflow as tf  # Remove TensorFlow
# from tensorflow import keras # Remove Keras
# from tensorflow.keras import layers, regularizers, callbacks # Remove Keras

# --- PyTorch Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset # Alias DataLoader
# -----------------------

import optuna
import mlflow
# import mlflow.keras # Remove Keras logging
import mlflow.pytorch # Add PyTorch logging
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
# from sklearn.neural_network import MLPClassifier # Remove sklearn MLP
from sklearn.exceptions import ConvergenceWarning # Keep for potential scaler warnings?
from sklearn.base import BaseEstimator # Keep for potential wrapper needs

# Set project root (similar to xgboost_model.py)
try:
    project_root = Path(__file__).parent.parent.parent.parent.parent
    if not project_root.exists():
        project_root = Path(r"\\\\".join(str(project_root).split("\\\\"))) # Use r-string or double backslashes
    sys.path.append(str(project_root))
    print(f"Project root mlp model: {project_root}")
except Exception as e:
    print(f"Error setting project root path: {e}")
    # Fallback to current directory if path resolution fails
    sys.path.append(os.getcwd().parent) # Check this path, might need adjustment
    print(f"Current directory mlp model: {os.getcwd().parent}") # Check this path

from utils.logger import ExperimentLogger
experiment_name = "mlp_pytorch_soccer_prediction" # Updated name
logger = ExperimentLogger(experiment_name=experiment_name)

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # Remove TensorFlow specific env var
# Set random seeds for reproducibility
random_seed = 19
random.seed(random_seed)
np.random.seed(random_seed)
# tf.random.set_seed(random_seed) # Remove TensorFlow seed
torch.manual_seed(random_seed) # Add PyTorch seed
os.environ['PYTHONHASHSEED'] = str(random_seed)

# --- CUDA Setup ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(random_seed) # Seed all GPUs
    # torch.backends.cudnn.deterministic = True # Can impact performance
    # torch.backends.cudnn.benchmark = False    # Can impact performance
    logger.info(f"CUDA is available! Using device: {device}")
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    logger.warning("CUDA not available, using CPU.")
# ------------------

# Configure Git executable path if available
git_executable = os.environ.get("GIT_PYTHON_GIT_EXECUTABLE")
if git_executable and os.path.exists(git_executable):
    import git
    git.refresh(git_executable)

from utils.create_evaluation_set import setup_mlflow_tracking, import_selected_features_ensemble
mlflow_tracking = setup_mlflow_tracking(experiment_name)

from models.StackedEnsemble.shared.data_loader import DataLoader

# Global settings
min_recall = 0.30            # Minimum acceptable recall
n_trials = 100               # Adjust trials for PyTorch training time
pip_requirements = [
    f"torch=={torch.__version__}", # Use torch version
    "scikit-learn",
    f"mlflow=={mlflow.__version__}"
]
scaler = None  # Global scaler object

# --- Base Training Parameters (PyTorch) ---
# Parameters NOT tuned by Optuna, used as defaults or fixed settings
base_train_params = {
    'optimizer': 'adam',       # Fixed optimizer type
    'criterion': 'bce_logits' # Fixed loss type (Binary Cross Entropy with Logits)
    # Add other fixed params if needed, e.g., a default dropout if not tuned
}
logger.info(f"PyTorch MLP Base Training Params set: {base_train_params}")
# ----------------------------------------

# --- PyTorch MLP Definition ---
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=1, dropout_rate=0.2):
        super(SimpleMLP, self).__init__()
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim)) # Added Batch Norm
            layers.append(nn.Dropout(dropout_rate)) # Added Dropout
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        # No sigmoid here, handled by BCEWithLogitsLoss during training
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
# ---------------------------

def load_hyperparameter_space():
    """
    Define an improved hyperparameter space for PyTorch MLP tuning,
    using separate parameters for layer structure.
    """
    hyperparameter_space = {
        # Define Layer Structure Numerically
        'n_layers': {
            'type': 'int',
            'low': 1,
            'high': 3
        },
        'units_l1': {
            'type': 'int',
            'low': 32,
            'high': 256,
            'log': True # Use log scale for wider exploration
        },
        'units_l2': { # Only used if n_layers >= 2
            'type': 'int',
            'low': 16,
            'high': 128,
            'log': True
        },
        'units_l3': { # Only used if n_layers == 3
            'type': 'int',
            'low': 8,
            'high': 64,
            'log': True
        },
        # --- Other Parameters ---
        'learning_rate': {
            'type': 'float',
            'low': 1e-5,
            'high': 1e-2,
            'log': True
        },
        'dropout_rate': {
            'type': 'float',
            'low': 0.1,
            'high': 0.6,
            'step': 0.05
        },
        'weight_decay': {
            'type': 'float',
            'low': 1e-6,
            'high': 1e-3,
            'log': True
        },
        'batch_size': {
            'type': 'int',
            'low': 256,
            'high': 2048,
            'step': 256
        },
        'epochs': {
            'type': 'int',
            'low': 20,
            'high': 150
        },
        'patience': {
            'type': 'int',
            'low': 5,
            'high': 25
        },
        'verbose': {
            'type': 'int',
            'low': 0,
            'high': 0
        }
    }
    return hyperparameter_space

def preprocess_data(X_train, X_test, X_eval=None):
    """
    Preprocess data using StandardScaler.
    
    Returns:
        tuple: scaled X_train, X_test, (and X_eval if provided), and the scaler.
    """
    global scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    if X_eval is not None:
        X_eval_scaled = scaler.transform(X_eval)
        return X_train_scaled, X_test_scaled, X_eval_scaled, scaler
    return X_train_scaled, X_test_scaled, scaler

def optimize_threshold(model, dataloader, min_threshold=0.2, max_threshold=0.9, step=0.01, min_recall=0.30):
    """
    Optimize the decision threshold for a PyTorch model using a DataLoader.
    Ensures model is in eval mode and handles device placement.
    Returns: tuple: (best_threshold, metrics_dict)
    """
    # NOTE: The 'dataloader' argument here is expected to be a TorchDataLoader instance
    # The caller function is responsible for creating it correctly.
    model.eval() # Set model to evaluation mode
    all_probs = []
    all_labels = []
    with torch.no_grad(): # Disable gradient calculations
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs) # Apply sigmoid to get probabilities
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs).flatten()
    all_labels = np.concatenate(all_labels).flatten()
    best_thresh = 0.5
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0
    best_auc = roc_auc_score(all_labels, all_probs) # Calculate AUC once
    thresholds = np.arange(min_threshold, max_threshold + step, step)
    for thresh in thresholds:
        y_pred = (all_probs >= thresh).astype(int)
        precision = precision_score(all_labels, y_pred, zero_division=0)
        recall = recall_score(all_labels, y_pred, zero_division=0)

        if recall >= min_recall and precision > best_precision:
            best_precision = precision
            best_recall = recall
            best_thresh = thresh
            best_f1 = f1_score(all_labels, y_pred, zero_division=0)
    logger.info(f"Optimized threshold: {best_thresh:.4f} -> Precision: {best_precision:.4f}, Recall: {best_recall:.4f}, F1: {best_f1:.4f}, AUC: {best_auc:.4f}")
    metrics = {
        'threshold': best_thresh,
        'precision': best_precision,
        'recall': best_recall,
        'f1': best_f1,
        'auc': best_auc
    }
    return best_thresh, metrics

def train_model_pytorch(model, params, X_train_scaled, y_train, X_eval_scaled, y_eval):
    """
    Trains the provided PyTorch MLP model instance.
    Handles the training loop, validation, and early stopping.

    Args:
        model (torch.nn.Module): The instantiated PyTorch model to train.
        params (dict): Dictionary containing parameters for training
                       (lr, batch_size, epochs, patience, weight_decay).
        X_train_scaled (np.ndarray): Scaled training features.
        y_train (np.ndarray): Training labels.
        X_eval_scaled (np.ndarray): Scaled evaluation features.
        y_eval (np.ndarray): Evaluation labels.

    Returns:
        tuple: (trained_model, training_history_dict)
    """
    # --- Hyperparameter Extraction --- Extract from the full params dict
    lr = params.get('learning_rate', 0.001)
    batch_size = params.get('batch_size', 1024)
    epochs = params.get('epochs', 100)
    patience = params.get('patience', 10)
    weight_decay = params.get('weight_decay', 0)
    # ----------------------------------------------------

    # --- DataLoaders --- (Remains the same conceptually, but use alias)
    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True) # Use alias
    eval_dataset = TensorDataset(torch.tensor(X_eval_scaled, dtype=torch.float32), torch.tensor(y_eval, dtype=torch.float32))
    eval_loader = TorchDataLoader(eval_dataset, batch_size=batch_size * 2) # Use alias
    # --------------------

    # --- Loss, Optimizer --- (Model is already instantiated and moved to device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # ----------------------------

    # --- Training Loop --- (Remains the same)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}

    logger.info(f"Starting PyTorch MLP training for {epochs} epochs with patience {patience}...")
    for epoch in range(epochs):
        model.train() # Set model to training mode
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1) # Ensure labels are [batch_size, 1]
            optimizer.zero_grad()       # Clear previous gradients
            outputs = model(inputs)     # Forward pass
            loss = criterion(outputs, labels) # Calculate loss
            loss.backward()             # Backpropagate
            optimizer.step()            # Update weights
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # --- Validation --- Calculates loss and AUC on the validation set after each epoch.
        model.eval() # Set model to evaluation mode
        val_loss = 0.0
        all_val_probs = []
        all_val_labels = []
        with torch.no_grad():
            for inputs, labels in eval_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                probs = torch.sigmoid(outputs) # Get probabilities for AUC
                all_val_probs.append(probs.cpu().numpy())
                all_val_labels.append(labels.cpu().numpy())

        avg_val_loss = val_loss / len(eval_loader)
        all_val_probs = np.concatenate(all_val_probs).flatten()
        all_val_labels = np.concatenate(all_val_labels).flatten()
        try:
            val_auc = roc_auc_score(all_val_labels, all_val_probs)
        except ValueError as e:
            logger.warning(f"Epoch {epoch+1}: Could not calculate AUC (perhaps only one class present in batch?): {e}")
            val_auc = 0.0 # Or handle as appropriate
        history['val_loss'].append(avg_val_loss)
        history['val_auc'].append(val_auc)
        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val AUC: {val_auc:.4f}")
        # ------------------

        # --- Early Stopping Check --- Stops training if the validation loss doesn't improve for a set number of epochs.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Save the best model state potentially
            # torch.save(model.state_dict(), 'best_mlp_model.pth')
            logger.info(f"Validation loss improved to {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            logger.info(f"Validation loss did not improve for {epochs_no_improve} epoch(s). Patience: {patience}")
        if epochs_no_improve >= patience:
            logger.info(f"Early stopping triggered after epoch {epoch+1}")
            break
        # -------------------------

    # Optional: Load best model state if saved
    # model.load_state_dict(torch.load('best_mlp_model.pth'))
    logger.info("PyTorch MLP training finished.")
    return model, history

def create_model_pytorch(input_dim, params):
    """
    Instantiates the SimpleMLP model and moves it to the correct device.

    Args:
        input_dim (int): Number of input features.
        params (dict): Dictionary containing model initialization parameters
                       (e.g., hidden_dims, dropout_rate, n_layers, units_l*).
    Returns:
        torch.nn.Module: The instantiated SimpleMLP model.
    """
    try:
        # Extract params specifically needed for SimpleMLP init from the full params dict
        dropout_rate = params.get('dropout_rate', 0.2)

        # --- Reconstruct Hidden Dims ---
        # Use get() with defaults in case HPO failed or params are incomplete
        n_layers = params.get('n_layers', 2) # Default to 2 layers if missing
        units_l1 = params.get('units_l1', 64) # Default unit sizes
        hidden_dims_list = [units_l1]
        if n_layers >= 2:
            units_l2 = params.get('units_l2', 32)
            hidden_dims_list.append(units_l2)
        if n_layers == 3:
            units_l3 = params.get('units_l3', 16)
            hidden_dims_list.append(units_l3)
        hidden_dims = tuple(hidden_dims_list)
        # --- End Reconstruction ---


        model = SimpleMLP(input_dim=input_dim,
                          hidden_dims=hidden_dims, # Use reconstructed hidden_dims
                          dropout_rate=dropout_rate).to(device)
        logger.debug(f"Created SimpleMLP model with: hidden_dims={hidden_dims}, dropout={dropout_rate}")
        return model
    except Exception as e:
        logger.error(f"Error creating PyTorch MLP model: {str(e)}", exc_info=True)
        raise

def optimize_hyperparameters_pytorch(scaler, input_dim, X_train, y_train, X_eval, y_eval, hyperparameter_space):
    """
    Run Optuna hyperparameter optimization for PyTorch MLP.
    Handles scaling within the objective function.
    """
    logger.info("Starting hyperparameter optimization for PyTorch MLP")
    if not hyperparameter_space:
        hyperparameter_space = load_hyperparameter_space()
    best_score = -float('inf')
    best_params_hpo = {} # Store only HPO params here
    global_top_trials = []
    top_trials = [] # Track top trials per batch
    # --- Objective Function --- Defines how each Optuna trial is executed and evaluated
    def objective(trial):
        try:
            # --- Parameter Suggestion & Parsing ---
            suggested_params = {}
            for param_name, param_config in hyperparameter_space.items():
                if param_config['type'] == 'float':
                    suggested_params[param_name] = trial.suggest_float(
                        param_name, param_config['low'], param_config['high'],
                        log=param_config.get('log', False),
                        step=param_config.get('step', None)
                    )
                elif param_config['type'] == 'int':
                    suggested_params[param_name] = trial.suggest_int(
                        param_name, param_config['low'], param_config['high'],
                        log=param_config.get('log', False), # Add log scale support for int
                        step=param_config.get('step', 1)
                    )
                elif param_config['type'] == 'categorical':
                    suggested_params[param_name] = trial.suggest_categorical(param_name, param_config['choices'])
            # -----------------------------------------

            # --- Construct Hidden Dims Tuple --- Based on suggested n_layers
            n_layers = suggested_params['n_layers']
            units_l1 = suggested_params['units_l1']
            units_l2 = suggested_params['units_l2'] # Get unconditionally suggested value
            units_l3 = suggested_params['units_l3'] # Get unconditionally suggested value
            hidden_dims_list = [units_l1]

            if n_layers >= 2:
                hidden_dims_list.append(units_l2)
            if n_layers == 3:
                hidden_dims_list.append(units_l3)

            hidden_dims_tuple = tuple(hidden_dims_list)
            # -----------------------------------

            # --- No explicit separation needed now ---
            # model_init_params = { ... }
            # train_loop_params = { ... }
            # logger.debug(f"Trial {trial.number}: Model Init Params: {model_init_params}") # Removed
            # logger.debug(f"Trial {trial.number}: Train Loop Params: {train_loop_params}") # Removed
            logger.debug(f"Trial {trial.number}: Suggested Params: {suggested_params}")
            # ----------------------------------------------------

            # --- Data Scaling --- (Remains the same)
            X_train_scaled = scaler.transform(X_train)
            X_eval_scaled = scaler.transform(X_eval)
            # ------------------

            # --- Instantiate Model --- Use the dedicated function, pass full dict
            model = create_model_pytorch(input_dim=input_dim, params=suggested_params)
            # ------------------------

            # --- Train Model --- Pass the instantiated model and full dict
            model, history = train_model_pytorch(
                model=model,
                params=suggested_params, # Pass full dict
                X_train_scaled=X_train_scaled,
                y_train=y_train,
                X_eval_scaled=X_eval_scaled,
                y_eval=y_eval
            )
            # -----------------

            # --- Evaluate & Score --- Evaluates the trained model and calculates the optimization score
            eval_dataset = TensorDataset(torch.tensor(X_eval_scaled, dtype=torch.float32), torch.tensor(y_eval, dtype=torch.float32))
            eval_loader = TorchDataLoader(eval_dataset, batch_size=suggested_params['batch_size'] * 2) # Use alias and larger batch size
            # Optimize threshold on the validation set predictions
            best_threshold, metrics = optimize_threshold(model, eval_loader, min_recall=min_recall)
            recall = metrics.get('recall', 0.0)
            precision = metrics.get('precision', 0.0)
            score = precision if recall >= min_recall else 0.0 # Optimization target score
            logger.info(f"Trial {trial.number}: Score: {score:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Threshold: {best_threshold:.4f}, Val AUC: {metrics.get('auc', 0.0):.4f}")
            # -----------------------

            # --- Optuna Reporting --- Logs metrics and handles pruning
            # Log all metrics from optimize_threshold and training history
            for metric_name, metric_value in metrics.items():
                trial.set_user_attr(f"eval_{metric_name}", metric_value)
            trial.set_user_attr("final_train_loss", history['train_loss'][-1])
            trial.set_user_attr("final_val_loss", history['val_loss'][-1])
            trial.set_user_attr("stopped_epoch", len(history['train_loss'])) # Log epoch where training stopped

            # Report the primary score to Optuna for pruning/optimization direction
            # Report based on validation loss for pruning (lower is better, but study direction is maximize score)
            intermediate_value = history['val_loss'][-1] # Use last validation loss for pruning check
            trial.report(intermediate_value, step=len(history['train_loss']))
            if trial.should_prune():
                logger.info(f"Trial {trial.number} pruned based on intermediate value: {intermediate_value:.4f}")
                raise optuna.TrialPruned()
            # ----------------------
            return score
        except optuna.TrialPruned:
            # logger.info(f"Trial {trial.number} pruned.") # Already logged above
            raise # Re-raise to signal Optuna
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {str(e)}")
            return -1.0 # Return *very* low score for failed trials

    # --- Callback Function (Similar logic, adapted variable names) ---
    def callback(study, trial):
        nonlocal best_score, best_params_hpo, top_trials
        logger.info(f"Current best score (HPO objective): {best_score:.4f}")
        if trial.value is not None and trial.value > best_score:
            best_score = trial.value
            best_params_hpo = trial.params
            logger.info(f"New best score found in trial {trial.number}: {best_score:.4f} with HPO params: {best_params_hpo}")
        # Record, sort, log top trials (keep this logic)
        # ... (Callback logic for top_trials remains largely the same) ...
        current_run = (trial.value, trial.params, trial.number, trial.user_attrs.get('eval_precision', 0.0), trial.user_attrs.get('eval_recall', 0.0))
        top_trials.append(current_run)
        top_trials.sort(key=lambda x: (x[0] is not None, x[0]), reverse=True) # Handle None scores
        top_trials[:] = top_trials[:10]
        if trial.number % 9 == 0:
            table_header = "| Rank | Trial # | Score  | Precision | Recall | Parameters |"
            table_separator = "|------|---------|--------|-----------|--------|------------|"
            table_rows = [
                f"| {i+1:<4} | {rec[2]:<7} | {rec[0]:<6.4f} | {rec[3]:<9.4f} | {rec[4]:<6.4f} | {rec[1]} |" if rec[0] is not None
                else f"| {i+1:<4} | {rec[2]:<7} | None   | {rec[3]:<9.4f} | {rec[4]:<6.4f} | {rec[1]} |"
                for i, rec in enumerate(top_trials)
            ]
            logger.info("Top trials in current batch:")
            logger.info(table_header)
            logger.info(table_separator)
            for row in table_rows:
                logger.info(row)
        # Log global top trials periodically (keep this logic)
        # ... (Global top trials logging remains the same, potentially add prec/recall columns) ...
        if trial.number % 50 == 0 and global_top_trials:
            logger.info("Global top 10 trials:")
            # ... (Log global top trials table) ...
        return best_score
    # -------------------------------------------------------------

    # --- Optuna Study Execution --- Runs the HPO process using Optuna
    storage_url = f"sqlite:///optuna_{experiment_name}.db"
    study_name = f"{experiment_name}_optimization"
    total_trials = n_trials
    # Adjust batch_size for Optuna study based on expected trial duration
    optuna_batch_size = 20 # Fewer trials per batch commit due to longer training time
    num_batches = total_trials // optuna_batch_size
    if total_trials % optuna_batch_size != 0:
        num_batches += 1

    for batch in range(num_batches):
        random_seed = int(time.time()) + batch
        # sampler = optuna.samplers.TPESampler(seed=random_seed, n_startup_trials=5) # Consider TPE after some random starts
        sampler = optuna.samplers.RandomSampler(seed=random_seed)
        # Pruner monitors intermediate validation loss (lower is better)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5, interval_steps=1)

        study = optuna.create_study(
            study_name=study_name,
            direction='maximize', # Maximizing precision score (where recall >= min_recall)
            storage=storage_url,
            load_if_exists=True,
            sampler=sampler,
            pruner=pruner
        )

        logger.info(f"Starting Optuna batch {batch+1}/{num_batches} with sampler (seed={random_seed})")
        study.optimize(
            objective,
            n_trials=min(optuna_batch_size, total_trials - batch * optuna_batch_size),
            show_progress_bar=True,
            callbacks=[callback]
        )

        # Merge top trials logic (keep this)
        # ... (Merging global_top_trials logic remains the same) ...
        for trial_record in top_trials:
            global_top_trials.append(trial_record)
        global_top_trials.sort(key=lambda x: (x[0] is not None, x[0]), reverse=True)
        global_top_trials = global_top_trials[:10]
    # ----------------------------------------------------------

    # --- Final Parameter Handling --- Selects the best parameters found during HPO
    best_hpo_params = {}
    if global_top_trials and global_top_trials[0][0] is not None and global_top_trials[0][0] > -1.0:
        best_score, best_hpo_params, best_trial_number, _, _ = global_top_trials[0]
        logger.info(f"Best trial value across batches: {best_score:.4f}")
        logger.info(f"Best HPO parameters found (Trial #{best_trial_number}): {best_hpo_params}")
        # Log top 10 trials (keep this)
        # ... (Log final top 10 table) ...
    else:
        logger.warning("No successful trials found meeting criteria or best score was <= 0. Cannot determine best HPO parameters.")
        best_hpo_params = {} # Indicate failure to find good params

    return best_hpo_params # Return only the HPO params
    # ---------------------------------------------------

def hypertune_mlp_pytorch(experiment_name: str):
    """
    Main hypertuning function for PyTorch MLP with MLflow tracking.
    Handles data scaling before HPO and trains the final model with best HPO params.
    Returns best_hpo_params and metrics from the final trained model.
    """
    global X_train, y_train, X_eval, y_eval # Access global data loaded in main
    global scaler # Access global scaler

    try:
        # --- Fit Scaler --- Ensures data is scaled before HPO
        logger.info("Fitting StandardScaler on training data for HPO...")
        scaler = StandardScaler()
        scaler.fit(X_train)
        logger.info("StandardScaler fitted.")
        # ------------------

        with mlflow.start_run(run_name=f"mlp_pytorch_base_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            mlflow.set_tags({
                "model_type": "mlp_pytorch",
                "training_mode": "global",
                "gpu_enabled": torch.cuda.is_available() # Dynamically set tag
            })

            # --- Run HPO --- Calls the PyTorch HPO function
            input_dim = X_train.shape[1] # Get input dimension after feature selection
            hyperparameter_space = load_hyperparameter_space()
            logger.info("Starting hyperparameter optimization for PyTorch MLP")

            # Pass the fitted scaler and unscaled data
            best_hpo_params = optimize_hyperparameters_pytorch(
                scaler,
                input_dim,
                X_train, y_train,
                X_eval, y_eval,
                hyperparameter_space=hyperparameter_space
            )
            # ----------------
            if not best_hpo_params:
                logger.error("Hyperparameter optimization failed to find suitable parameters. Aborting.")
                return None, None

            # Log HPO parameters found
            logger.info("Logging best HPO parameters found to MLflow...")
            mlflow.log_params({f"hpo_{k}": v for k, v in best_hpo_params.items()})

            # --- Train Final Model ---
            logger.info("Training final PyTorch MLP model with best HPO parameters...")
            # --- No need to reconstruct params separately ---
            # try:
            #     ... reconstruction logic ...
            # except KeyError as e:
            #     ... error handling ...
            # final_model_init_params = { ... }
            # final_train_loop_params = { ... }
            # -----------------------------------------

            # --- Instantiate Final Model --- Use the dedicated function, pass full best_hpo_params
            final_model = create_model_pytorch(input_dim=input_dim, params=best_hpo_params) # Pass full dict
            # --------------------------------

            # Scale data
            X_train_final_scaled = scaler.transform(X_train)
            X_eval_final_scaled = scaler.transform(X_eval)

            # Train the final model - Pass the INSTANTIATED final_model and full best_hpo_params
            final_model, final_history = train_model_pytorch(
                model=final_model,
                params=best_hpo_params, # Pass full dict
                X_train_scaled=X_train_final_scaled,
                y_train=y_train,
                X_eval_scaled=X_eval_final_scaled,
                y_eval=y_eval
            )

            # --- Final Evaluation & Thresholding --- Evaluates the final model and determines the optimal threshold.
            logger.info("Evaluating final model and optimizing threshold...")
            final_eval_dataset = TensorDataset(torch.tensor(X_eval_final_scaled, dtype=torch.float32), torch.tensor(y_eval, dtype=torch.float32))
            # Use get for batch_size with a default if HPO failed or param missing
            final_batch_size = best_hpo_params.get('batch_size', 1024) * 2
            final_eval_loader = TorchDataLoader(final_eval_dataset, batch_size=final_batch_size) # Use alias
            # Pass the correctly created TorchDataLoader instance
            final_threshold, final_metrics = optimize_threshold(final_model, final_eval_loader, min_recall=min_recall)
            logger.info(f"Final model evaluation complete. Metrics: {final_metrics}")
            # -------------------------------------
            # --- MLflow Logging --- Logs parameters, metrics, scaler, and the trained PyTorch model.
            # Log final metrics
            mlflow.log_metrics({
                "final_precision": final_metrics.get('precision', 0.0),
                "final_recall": final_metrics.get('recall', 0.0),
                "final_f1": final_metrics.get('f1', 0.0),
                "final_auc": final_metrics.get('auc', 0.0),
                "final_threshold": final_threshold
                # Add relevant training history metrics if desired
                # "final_train_loss": final_history['train_loss'][-1],
                # "final_val_loss": final_history['val_loss'][-1],
            })
            # Log final HPO parameters used for the model (already logged before, maybe redundant?)
            # logger.info("Logging final HPO parameters to MLflow...")
            # mlflow.log_params(best_hpo_params)
            # Log the scaler
            scaler_path = "scaler.pkl"
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)
            mlflow.log_artifact(scaler_path)
            logger.info("Scaler artifact logged.")
            # Log the model using mlflow.pytorch
            logger.info("Logging model using mlflow.pytorch...")
            # Create input example from SCALED eval data
            # Need original column names if available for signature
            try:
                columns = X_eval.columns
            except AttributeError:
                columns = [f"feature_{i}" for i in range(X_eval_final_scaled.shape[1])]
            input_example = pd.DataFrame(X_eval_final_scaled[:5], columns=columns)
            # Get model output for the example (raw logits)
            final_model.eval()
            with torch.no_grad():
                example_tensor = torch.tensor(input_example.values, dtype=torch.float32).to(device)
                output_example_tensor = final_model(example_tensor)
            # We typically log probabilities or classes for signature, apply sigmoid here
            output_example_probs = torch.sigmoid(output_example_tensor).cpu().numpy()
            output_example = pd.DataFrame(output_example_probs, columns=['probability'])
            signature = mlflow.models.infer_signature(input_example, output_example)

            mlflow.pytorch.log_model(
                pytorch_model=final_model,
                artifact_path="model", # Standard artifact path name
                signature=signature,
                pip_requirements=pip_requirements,
                registered_model_name=f"mlp_pytorch_{datetime.now().strftime('%Y%m%d_%H%M')}"
            )
            logger.info("PyTorch MLP model logged successfully.")
            # ----------------------

            # Return the HPO parameters (as they define the final model) and the final metrics
            return best_hpo_params, final_metrics

    except Exception as e:
        logger.error(f"Error in PyTorch MLP hypertuning: {str(e)}")
        return None, None

def main():
    """
    Main execution function for PyTorch MLP hypertuning and final training.
    """
    global X_train, y_train, X_test, y_test, X_eval, y_eval # Declare usage of globals
    global scaler # Ensure main can access the scaler fitted in hypertune

    try:
        logger.info("Starting PyTorch MLP model training & hypertuning process")
        # Setup MLflow tracking directory
        mlruns_dir = setup_mlflow_tracking(experiment_name)

        # Load data using shared DataLoader
        logger.info("Loading data...")
        data_loader_instance = DataLoader() # Instantiate custom loader
        X_train, y_train, X_test, y_test, X_eval, y_eval = data_loader_instance.load_data() # Use instance

        # --- Feature Selection ---
        try:
            features = import_selected_features_ensemble(model_type='all') # Try mlp specific first
            logger.info(f"Using MLP-specific feature set with {len(features)} features.")
        except (KeyError, FileNotFoundError):
            logger.warning("MLP-specific features not found. Falling back to 'xgb' features.") # Example fallback
            features = import_selected_features_ensemble(model_type='xgb')

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

        # --- Run Hypertuning and Final Training/Logging --- Calls the main PyTorch HPO and training function
        best_hpo_params, final_metrics = hypertune_mlp_pytorch(experiment_name)
        # ----------------------------------------------------

        if best_hpo_params and final_metrics:
            logger.info(f"Hypertuning and final model training completed successfully.")
            logger.info(f"Best HPO Params Used: {best_hpo_params}")
            logger.info(f"Final Model Eval Metrics: {final_metrics}")
        else:
            logger.error("MLP training process failed during hypertuning or final training.")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
    finally:
        gc.collect()
        logger.info("MLP Main execution finished.")

if __name__ == "__main__":
    main() 