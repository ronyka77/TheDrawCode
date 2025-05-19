"""
PyTorch Custom Neural Network Hypertuner

This module implements training, hyperparameter optimization (Optuna),
and MLflow logging for a custom PyTorch neural network model for soccer draw prediction,
focusing on precision under a minimum recall constraint.
"""

import gc
import json
import os
import pickle
import random
from datetime import datetime

import mlflow
import numpy as np
import optuna
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import TensorDataset

from src.models.ensemble.data_utils import prepare_data

# Import custom utilities
from src.models.StackedEnsemble.shared.data_loader_new import DataLoader
from src.models.StackedEnsemble.shared.hypertuner_utils import optimize_threshold
from src.utils.create_evaluation_set import (
    import_selected_features_ensemble_new,
    setup_mlflow_tracking,
)
from src.utils.logger import ExperimentLogger

# Set fixed seed and hash seed for determinism
SEED = 19
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Restrict parallel threads across various libraries
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"
os.environ["VECLIB_MAXIMUM_THREADS"] = "16"

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
        USE_TORCH_COMPILE = True
    else:
        USE_TORCH_COMPILE = False
else:
    USE_TORCH_COMPILE = False

# Define a placeholder if the actual model file doesn't exist yet
class PytorchModel(nn.Module):
    def __init__(self, input_dim, **kwargs):
        super().__init__()
        hidden_units = kwargs.get("hidden_units", 64)
        num_layers = kwargs.get("num_layers", 2)
        dropout_rate = kwargs.get("dropout_rate", 0.5)
        # Activation mapping needs to be implemented here in actual model
        activation_name = kwargs.get("activation_fn", "ReLU")
        if activation_name == "LeakyReLU":
            act_fn = nn.LeakyReLU()
        elif activation_name == "SiLU":
            act_fn = nn.SiLU()
        else: # Default to ReLU
            act_fn = nn.ReLU()

        layers = [nn.Linear(input_dim, hidden_units), act_fn, nn.Dropout(dropout_rate)]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_units, hidden_units), act_fn, nn.Dropout(dropout_rate)])
        layers.append(nn.Linear(hidden_units, 1)) # Output layer for binary classification
        self.network = nn.Sequential(*layers)
        # Add attributes to store scaler and device for predict_proba
        self.scaler_ = None 
        self.device_ = None

    def forward(self, x):
        return self.network(x)

    def predict_proba(self, X, batch_size=1024):
        """
        Predict probabilities, mimicking scikit-learn interface.
        Requires scaler_ and device_ attributes to be set.
        Returns probabilities for both classes (0 and 1) in shape (N, 2).
        """
        if self.scaler_ is None or self.device_ is None:
            raise ValueError("Scaler and Device must be set on the model before calling predict_proba.")
        
        self.network.eval() # Set model to evaluation mode
        all_probs_class1 = []
        
        # Ensure X is DataFrame or Array that scaler expects
        X_scaled = self.scaler_.transform(X)
        
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        dataset = TensorDataset(X_tensor)
        dataloader = TorchDataLoader(
            dataset, 
            batch_size=32, 
            num_workers=1,
            # pin_memory=True,  # Enables faster CPU to GPU transfers
            # persistent_workers=True  # Keeps workers alive between epochs
        )

        with torch.no_grad():
            for batch_X_tuple in dataloader:
                batch_X = batch_X_tuple[0].to(self.device_)
                outputs = self.network(batch_X)
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_probs_class1.append(probs)
                
        # Concatenate probabilities for class 1
        probs_class1 = np.concatenate(all_probs_class1) # Shape (N, 1)
        # Calculate probabilities for class 0
        probs_class0 = 1.0 - probs_class1            # Shape (N, 1)
        
        # Stack them horizontally to get shape (N, 2)
        return np.hstack((probs_class0, probs_class1))

# Global settings
MIN_RECALL = 0.20  # Minimum acceptable recall
N_TRIALS = 10000  # Number of hyperparameter optimization trials (adjust as needed)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define pip requirements
pip_requirements = [
    f"torch=={torch.__version__}",
    f"scikit-learn=={sklearn.__version__}",
    f"mlflow=={mlflow.__version__}",
    f"optuna=={optuna.__version__}",
    f"numpy=={np.__version__}",
]

base_params = {
    "random_state": SEED
}

global logger, experiment_name

# --- Phase 2: Hyperparameter Space and Model Handling ---
def load_hyperparameter_space():
    """
    Define the hyperparameter space for the PyTorch Neural Network tuning.

    Returns:
        dict: Hyperparameter space configuration for Optuna.
    """
    hyperparameter_space = {
        "learning_rate": {
            "type": "float",
            "low": 1e-6,
            "high": 1e-2,
            "log": True,
        },
        "optimizer": {
            "type": "categorical",
            "choices": ["Adam", "AdamW", "SGD"]
        },
        "weight_decay": {
            "type": "float",
            "low": 1e-7,
            "high": 1e-3,
            "log": True,
        },
        "batch_size": {
            "type": "categorical", 
            "choices": [64, 128, 256, 512, 1024, 2048]
        },
        "num_epochs": {
            "type": "int",
            "low": 50, # Min epochs
            "high": 300, # Max epochs (can be higher, depends on early stopping)
            "step": 10,
        },
        "num_layers": {
            "type": "int", 
            "low": 1,
            "high": 6 # Example range for custom NN
        },
        "hidden_units": {
            "type": "int",
            "low": 32,
            "high": 256,
            "step": 32,
        },
        "activation_fn": {
            "type": "categorical",
            "choices": ["ReLU", "LeakyReLU", "SiLU"]
        },
        "dropout_rate": {
            "type": "float",
            "low": 0.1,
            "high": 0.6,
            "step": 0.02,
        },
        "early_stopping_patience": {
            "type": "int",
            "low": 10,
            "high": 50,
            "step": 2,
        },
    }
    logger.info("Hyperparameter space loaded.")
    return hyperparameter_space


def create_pytorch_model(model_params, input_dim, device, scaler):
    """
    Create and configure the PyTorch model instance.
    Also attaches scaler and device needed for the predict_proba method.
    Args:
        model_params (dict): Hyperparameters suggested by Optuna.
        input_dim (int): Number of input features.
        device (torch.device): The device (CPU or CUDA) to run the model on.
        scaler (StandardScaler): The fitted scaler instance.
    Returns:
        PytorchModel: Configured and device-placed PyTorch model instance.
    """
    # Extract relevant parameters for the model architecture
    # Ensure keys match the expected arguments in PytorchModel.__init__
    # Example: Adjust based on PytorchModel's actual signature
    architecture_params = {
        "num_layers": model_params.get("num_layers"),
        "hidden_units": model_params.get("hidden_units"),
        "activation_fn": model_params.get("activation_fn"), 
        "dropout_rate": model_params.get("dropout_rate")
    }
    # Remove None values if PytorchModel handles defaults
    architecture_params = {k: v for k, v in architecture_params.items() if v is not None}
    
    model = PytorchModel(input_dim=input_dim, **architecture_params)
    model.to(device)
    
    # Attach scaler and device to the model instance
    model.scaler_ = scaler
    model.device_ = device
    
    # logger.info(f"Created PyTorch model with params: {architecture_params}")
    logger.info(f"Model placed on device: {device}, scaler attached.")
    return model


# --- Phase 3: Core Training Logic ---
def train_pytorch_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    X_val,
    y_val,
    model_params,
    device,
    scaler, # Pass the fitted scaler
    trial=None, # Optional: Pass the Optuna trial for pruning
):
    """
    Trains the PyTorch model, handles validation, early stopping, and threshold tuning.
    Args:
        model (torch.nn.Module): The PyTorch model instance.
        X_train, y_train: Training data and labels (numpy/pandas).
        X_val, y_val: Validation data and labels (numpy/pandas).
        X_eval, y_eval: Evaluation data for final threshold tuning (numpy/pandas).
        model_params (dict): Hyperparameters including batch_size, epochs, optimizer, etc.
        device (torch.device): CPU or CUDA device.
        scaler (StandardScaler): The fitted scaler instance.
        trial (optuna.Trial, optional): Optuna trial for pruning. Defaults to None.
    Returns:
        tuple: (trained_model, metrics_dict)
                metrics_dict contains precision, recall, f1, auc, threshold.
    """
    logger.info("Starting PyTorch model training...")
    try:
        # --- Data Preparation ---
        batch_size = model_params["batch_size"]
        X_combined = pd.concat([X_train, X_test], axis=0)
        y_combined = pd.concat([y_train, y_test], axis=0)

        X_combined_scaled = scaler.transform(X_combined)
        X_val_scaled = scaler.transform(X_val)
        # No need to scale X_eval here, will be handled by pytorch_predict_proba
        X_train_tensor = torch.tensor(X_combined_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_combined.values, dtype=torch.float32).unsqueeze(1) # Ensure (N, 1)
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1) # Ensure (N, 1)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        logger.info(f"Data loaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        # --- Initialization ---
        criterion = nn.BCEWithLogitsLoss() # Numerically stable
        optimizer_name = model_params.get("optimizer", "AdamW")
        lr = model_params["learning_rate"]
        weight_decay = model_params.get("weight_decay", 0)
        
        if optimizer_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9) # Add momentum for SGD
        else: # Default to AdamW
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        logger.info(f"Using optimizer: {optimizer_name}")

        # Early stopping variables
        patience = model_params.get("early_stopping_patience", 20)
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state_dict = None
        epochs = model_params["num_epochs"]
        
        # --- Training Loop ---
        for epoch in range(epochs):
            # Training Phase
            model.train()
            epoch_train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
            avg_train_loss = epoch_train_loss / len(train_loader)

            # Validation Phase
            model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    epoch_val_loss += loss.item()
            avg_val_loss = epoch_val_loss / len(val_loader)
            
            logger.debug(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # Early Stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state_dict = model.state_dict() # Save best model state
                logger.debug(f"Epoch {epoch+1}: New best validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                logger.debug(f"Epoch {epoch+1}: No improvement in validation loss. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}.")
                    break
        
        # --- Post-Training ---
        if best_model_state_dict:
            model.load_state_dict(best_model_state_dict)
            logger.info(f"Loaded best model state from epoch with validation loss: {best_val_loss:.4f}")
        else:
            logger.warning("No best model state found (early stopping might not have been triggered or training was too short).")

        # Threshold Optimization using validation data (X_val, y_val)
        logger.info("Optimizing threshold on validation data...")
        y_val_np = y_val.values # optimize_threshold expects numpy array
        
        # Call optimize_threshold with the model, X, and y
        # Assumes optimize_threshold will internally call model.predict_proba(X_val)
        try:
            best_threshold, metrics = optimize_threshold( 
                model,      # Pass the model object
                X_val,      # Pass validation features
                y_val_np,   # Pass validation true labels
                min_recall=MIN_RECALL,
                # Removed pre-calculated probs and other args
            )
        except TypeError as te:
            # This error might still occur if optimize_threshold itself has issues
            logger.error(f"TypeError calling optimize_threshold: {te}")
            logger.error("Ensure optimize_threshold in hypertuner_utils.py expects (model, X, y, min_recall).")
            return model, {} # Return empty metrics on failure
        
        logger.info(f"Threshold optimization complete. Best Threshold: {best_threshold:.4f}")
        logger.info(f"Metrics at threshold: {metrics}")
        return model, metrics

    except optuna.TrialPruned:
        raise # Re-raise prune exception for Optuna to handle
    except Exception as e:
        logger.error(f"Error during PyTorch model training: {str(e)}")
        # Return None or raise to indicate failure
        return None, {} # Return empty metrics on failure


# --- Phase 4: Optuna Integration ---
def objective(
    trial,
    X_train,
    y_train,
    X_test,
    y_test,
    X_val,
    y_val,
    hyperparameter_space,
    input_dim,
    device,
    scaler,
    best_score_overall,
):
    """
    Optuna objective function.
    Suggests hyperparameters, creates and trains the model, returns the score.
    """
    try:
        # Suggest hyperparameters
        params = {}
        for param_name, config in hyperparameter_space.items():
            if config["type"] == "float":
                params[param_name] = trial.suggest_float(
                    param_name, config["low"], config["high"], 
                    log=config.get("log", False), step=config.get("step")
                )
            elif config["type"] == "int":
                params[param_name] = trial.suggest_int(
                    param_name, config["low"], config["high"], 
                    step=config.get("step", 1)
                )
            elif config["type"] == "categorical":
                choices = config["choices"]
                if not isinstance(choices, (list, tuple)):
                    logger.error(f"Invalid choices for {param_name}: {choices}")
                    if param_name == "batch_size":
                        choices = [128] # Default batch size
                    else:
                        choices = ["Adam"] # Default optimizer
                params[param_name] = trial.suggest_categorical(param_name, choices)
        
        logger.info(f"Trial {trial.number}: Suggested params: {params}")

        # Create model
        model = create_pytorch_model(params, input_dim, device, scaler)

        # Train model (passing the trial for pruning)
        # train_pytorch_model handles potential TrialPruned exception
        model, metrics = train_pytorch_model(
            model,
            X_train, y_train, 
            X_test, y_test, 
            X_val, y_val,
            params, 
            device, 
            scaler, 
            trial=trial # Pass trial for pruning
        )
        
        # Check if training failed
        if not metrics:
            logger.warning(f"Trial {trial.number}: Training failed or returned no metrics.")
            return 0.0 # Return low score for failed trials

        # Calculate score (e.g., precision constrained by recall)
        recall = metrics.get("recall", 0.0)
        precision = metrics.get("precision", 0.0)
        score = precision if recall >= MIN_RECALL else 0.0

        # Log metrics as user attributes for the trial
        for metric_name, metric_value in metrics.items():
            trial.set_user_attr(metric_name, metric_value)
        trial.set_user_attr("score", score) # Log the final score too

        logger.info(f"Trial {trial.number}: Score: {score:.4f} (Precision: {precision:.4f}, Recall: {recall:.4f})")
        
        if score > 0.37 :
            log_to_mlflow_pytorch(
                model,
                metrics,
                params,
                experiment_name,
                X_val,
                scaler,
                pip_requirements,
                run_name_prefix=f"pytorch_trial_{trial.number}",
            )
        return score

    except optuna.TrialPruned as e:
        logger.info(f"Trial {trial.number} pruned during training.")
        raise e # Re-raise for Optuna
    except Exception as e:
        logger.error(f"Trial {trial.number} failed unexpectedly: {str(e)}")
        return 0.0  # Return low score for other failed trials


def optimize_hyperparameters(
    X_train, y_train, 
    X_test, y_test, 
    X_val, y_val, 
    hyperparameter_space,
    input_dim,
    device,
    scaler,
    n_trials=N_TRIALS # Use global constant
):
    """
    Orchestrates the hyperparameter optimization process using Optuna.
    Args:
        X_train, y_train: Training data.
        X_test, y_test: Test data.
        X_val, y_val: Validation data.
        hyperparameter_space (dict): The search space definition.
        input_dim (int): Number of input features.
        device (torch.device): Device for training.
        scaler (StandardScaler): Fitted scaler.
        n_trials (int): Number of Optuna trials to run.
    Returns:
        dict: The best hyperparameters found by Optuna.
    """
    logger.info(f"Starting hyperparameter optimization with {n_trials} trials...")
    # Add a callback to log progress (optional, can be simpler than XGBoost version)
    best_score_overall = -float("inf")
    top_trials_overall = []

    # Define the objective function with fixed arguments using lambda
    objective_func = lambda trial: objective(
        trial, 
        X_train, y_train, 
        X_test, y_test, 
        X_val, y_val, 
        hyperparameter_space, 
        input_dim, 
        device, 
        scaler,
        best_score_overall
    )

    # Set up Optuna study
    # Consider adding persistent storage like the XGBoost version if needed
    storage_url = "sqlite:///optuna_pytorch.db"
    sampler = optuna.samplers.RandomSampler(seed=SEED) # Example sampler
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=experiment_name,
        storage=storage_url,
        load_if_exists=True 
    )

    def log_callback(study, trial):
        nonlocal best_score_overall, top_trials_overall
        current_score = trial.value if trial.value is not None else -float("inf")
        # Update overall best score
        if current_score > best_score_overall:
            best_score_overall = current_score

        # Keep track of top N trials
        top_trials_overall.append((trial.number, current_score, trial.params))
        top_trials_overall.sort(key=lambda x: x[1], reverse=True)
        top_trials_overall[:] = top_trials_overall[:10] # Keep top 10
        logger.info(f"Best Score: {best_score_overall}")
        # Log top trials periodically (e.g., every 10 trials)
        if trial.number % 10 == 0:
            logger.info(f"--- Top Trials (after Trial {trial.number}) ---")
            header = "| Rank | Trial # | Score  | Params |"
            sep =    "|------|---------|--------|--------|"
            logger.info(header)
            logger.info(sep)
            for i, (t_num, t_score, t_params) in enumerate(top_trials_overall):
                params_str = json.dumps(t_params, sort_keys=True, default=lambda x: f"{x:.4g}" if isinstance(x, float) else x) # Compact params
                logger.info(f"| {i+1:<4} | {t_num:<7} | {t_score:.4f} | {params_str} |")
            logger.info(f"Current Best Score Overall: {best_score_overall:.4f}")

    # Run the optimization
    try:
        study.optimize(objective_func, n_trials=n_trials, callbacks=[log_callback], n_jobs=4)
    except KeyboardInterrupt:
        logger.warning("Optimization stopped manually via KeyboardInterrupt.")
    
    # --- Post-Optimization --- 
    logger.info("Hyperparameter optimization finished.")
    if not study.trials:
        logger.warning("No trials completed successfully.")
        return {}

    return study.best_trial.params


# --- Phase 5: MLflow Logging and Main Workflow ---
def log_to_mlflow_pytorch(
    model,
    metrics,
    params,
    experiment_name, 
    X_eval, # For input example
    scaler, # Log the scaler as well
    pip_requirements,
    run_name_prefix="pytorch_final",
):
    """
    Log trained model, scaler, metrics, and parameters to MLflow.
    Args:
        model (torch.nn.Module): The trained PyTorch model.
        metrics (dict): Model evaluation metrics (precision, recall, threshold, etc.).
        params (dict): Hyperparameters used for the final model.
        experiment_name (str): Target MLflow experiment.
        X_eval (pd.DataFrame or np.ndarray): Evaluation data to create input example.
        scaler (StandardScaler): The fitted scaler instance to log.
        pip_requirements (list): List of pip requirements strings.
        run_name_prefix (str): Prefix for the MLflow run name.
    Returns:
        str: The MLflow Run ID of the logged run, or None on failure.
    """
    try:
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(
            run_name=f"{run_name_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ) as run:
            run_id = run.info.run_id
            logger.info(f"Starting MLflow run: {run_id}")
            
            # Log parameters (consider filtering if needed)
            mlflow.log_params(params)
            logger.info(f"Logged parameters: {params}")
            
            # Log metrics
            mlflow.log_metrics(metrics)
            logger.info(f"Logged metrics: {metrics}")

            # Log the scaler
            scaler_path = "src/models/scalers/scaler_pytorch.pkl"
            mlflow.log_artifact(scaler_path, artifact_path="scaler")
            logger.info("Logged scaler artifact.")

            # Log the PyTorch model
            # --- Signature Inference --- 
            input_example_df = X_eval.iloc[:5].copy() # Get DataFrame slice first
            # Scale this specific example for prediction
            input_example_np_scaled = scaler.transform(input_example_df).astype(np.float32)
            input_example_tensor = torch.tensor(input_example_np_scaled, dtype=torch.float32)
            
            # Infer signature (optional but recommended)
            try:
                # Ensure model is on CPU for signature inference
                original_device = next(model.parameters()).device # Store original device
                model.to('cpu') 
                # Get predictions for the example
                with torch.no_grad():
                    # Use the tensor created from the scaled DataFrame example
                    predictions = model(input_example_tensor.cpu()) 
                # Move model back to original device
                model.to(original_device) 

                # Infer signature using the DataFrame input and numpy output
                signature = mlflow.models.infer_signature(
                    input_example_df, # Pass DataFrame with names
                    predictions.cpu().numpy()
                    # Removed input_names argument
                )
                logger.info("Successfully inferred model signature.")
            except Exception as sig_e:
                logger.warning(f"Could not infer signature: {sig_e}. Proceeding without signature.")
                signature = None

            # --- Model Logging ---
            # Still use numpy for input_example in log_model if required by mlflow
            input_example_for_log = input_example_np_scaled 

            # Log model using mlflow.pytorch with signature only
            model_info = mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                signature=signature,
                pip_requirements=pip_requirements,
                input_example=input_example_for_log, # Use numpy array here for input_example param
                registered_model_name=f"pytorch_{datetime.now().strftime('%Y%m%d_%H%M')}" # Optional registration
            )

            # Validate serving input
            try:
                serving_input = mlflow.models.convert_input_example_to_serving_input(input_example_df)
                mlflow.models.validate_serving_input(model_info.model_uri, serving_input)
                logger.info("Successfully validated model serving input using DataFrame example.")
            except Exception as val_e:
                logger.warning(f"Failed to validate serving input: {val_e}")

            logger.info(f"PyTorch model logged to MLflow artifact path 'model' in run {run_id}")
            mlflow.end_run()
            return run_id

    except Exception as e:
        logger.error(f"Error logging to MLflow: {str(e)}")
        return None


def hypertune_pytorch(
    X_train, y_train, 
    X_test, y_test, 
    X_val, y_val,
    experiment_name: str, 
    input_dim: int,
    device: torch.device,
    scaler: StandardScaler
):
    """
    Main orchestration function for hyperparameter tuning and final model training.
    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        X_eval, y_eval: Evaluation data.
        experiment_name (str): MLflow experiment name.
        input_dim (int): Number of input features.
        device (torch.device): Device for training.
        scaler (StandardScaler): Fitted scaler.
    Returns:
        tuple: (best_params, final_metrics) or (None, None) on failure.
    """
    try:
        # Load hyperparameter space
        hyperparameter_space = load_hyperparameter_space()

        # Run hyperparameter optimization
        logger.info("Initiating Optuna hyperparameter search...")
        best_hpo_params = optimize_hyperparameters(
            X_train, y_train, 
            X_test, y_test, 
            X_val, y_val, 
            hyperparameter_space,
            input_dim,
            device,
            scaler
        )

        if not best_hpo_params:
            logger.error("Hyperparameter optimization failed to find best parameters.")
            mlflow.log_param("status", "HPO Failed")
            return None, None

        logger.info(f"Best HPO parameters identified: {best_hpo_params}")
        mlflow.log_params({f"best_{k}": v for k,v in best_hpo_params.items()}) # Log best HPO params to HPO run

        # --- Train Final Model with Best Parameters ---
        logger.info("Training final model using best hyperparameters...")
        final_model = create_pytorch_model(best_hpo_params, input_dim, device, scaler)
        
        # Train the final model (without passing Optuna trial)
        final_model, final_metrics = train_pytorch_model(
            final_model,
            X_train, y_train,
            X_test, y_test,
            X_val, y_val,
            best_hpo_params,
            device,
            scaler,
            trial=None # Not an Optuna trial run
        )
        
        # --- Log Final Model Separately ---
        logger.info("Logging the final trained model and artifacts to MLflow...")
        log_to_mlflow_pytorch(
            model=final_model,
            metrics=final_metrics,
            params=best_hpo_params,
            experiment_name=experiment_name, 
            X_eval=X_val,
            scaler=scaler,
            pip_requirements=pip_requirements,
            run_name_prefix=experiment_name + "_model"
        )

        return best_hpo_params, final_metrics

    except Exception as e:
        logger.error(f"Error during hyperparameter tuning orchestration: {str(e)}")
        # Optionally log failure to current MLflow run if active
        if mlflow.active_run():
            mlflow.log_param("status", "Orchestration Error")
            mlflow.set_tag("error", str(e))
        return None, None


def train_with_precision_target_pytorch(
    X_train, y_train, 
    X_test, y_test, 
    X_val, y_val, 
    experiment_name: str, 
    input_dim: int,
    device: torch.device,
    scaler: StandardScaler,
):
    """
    Trains a single model using fixed, hardcoded hyperparameters.
    Mirrors the structure of the XGBoost version but uses PyTorch logic.
    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        X_eval, y_eval: Evaluation data.
        experiment_name (str): MLflow experiment name.
        input_dim (int): Number of input features.
        device (torch.device): Device for training.
        scaler (StandardScaler): Fitted scaler.
    Returns:
        tuple: (trained_model, metrics) or (None, None) on failure.
    """
    try:
        logger.warning(
            "Training model with hardcoded parameters - Update these values with actual best params."
        )
        
        # --- Define hardcoded best parameters here ---
        fixed_params = {
            "learning_rate": 0.00022325621053802486,
            "optimizer": "AdamW",
            "weight_decay": 3.4404448540336636e-06,
            "batch_size": 256,
            "num_epochs": 270,
            "num_layers": 5,
            "hidden_units": 32,
            "activation_fn": "SiLU",
            "dropout_rate": 0.22,
            "early_stopping_patience": 24,
            "team_feature_pct": 0.5,
            "use_interactions": False,
            "use_residual": True
        }
        logger.info(f"Using hardcoded parameters: {fixed_params}")

        # Create model with fixed parameters
        model = create_pytorch_model(fixed_params, input_dim, device, scaler)

        # Train the model
        model, metrics = train_pytorch_model(
            model,
            X_train, y_train,
            X_test, y_test,
            X_val, y_val,
            fixed_params, # Use the hardcoded dict
            device,
            scaler,
            trial=None # Not an Optuna trial
        )

        if not metrics:
            logger.error("Training with fixed parameters failed.")
            return None, None

        # Log the trained model to MLflow
        # log_to_mlflow_pytorch(
        #     model=model,
        #     metrics=metrics,
        #     params=fixed_params, # Log the hardcoded params used
        #     experiment_name=experiment_name,
        #     X_eval=X_val,
        #     scaler=scaler,
        #     pip_requirements=pip_requirements,
        #     run_name_prefix="pytorch_fixed_params"
        # )
        compute_permutation_importance(
            model=model,
            X_val=X_val,
            y_val=y_val,
            metric=metrics["precision"],
            threshold=metrics["threshold"],
            n_repeats=3,
            random_state=19
        )
        logger.info("Training with fixed parameters completed successfully.")
        return model, metrics

    except Exception as e:
        logger.error(f"Error in fixed parameter training: {str(e)}")
        return None, None


def compute_permutation_importance(
    model,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    metric,
    threshold: float,
    n_repeats: int = 3,
    random_state: int = None,
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
    logger.info(df_importance.head(70).to_string(index=False))
    return df_importance

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
    feature_importances_trials = []

    hyperparameter_space = load_hyperparameter_space()
    def objective(trial):
        params = {}
        params.update(base_params)

        # Iterate over the rest of the hyperparameter space
        for param_name, config in hyperparameter_space.items():
            if config["type"] == "float":
                params[param_name] = trial.suggest_float(
                    param_name, config["low"], config["high"], 
                    log=config.get("log", False), step=config.get("step")
                )
            elif config["type"] == "int":
                params[param_name] = trial.suggest_int(
                    param_name, config["low"], config["high"], 
                    step=config.get("step", 1)
                )
            elif config["type"] == "categorical":
                choices = config["choices"]
                if not isinstance(choices, (list, tuple)):
                    logger.error(f"Invalid choices for {param_name}: {choices}")
                    if param_name == "batch_size":
                        choices = [128] # Default batch size
                    else:
                        choices = ["Adam"] # Default optimizer
                params[param_name] = trial.suggest_categorical(param_name, choices)

        # Train model and get metrics
        # Create model
        model = create_pytorch_model(params, input_dim, device, scaler)
        model, metrics = train_pytorch_model(model, X_train, y_train, X_test, y_test, X_eval, y_eval, params, device, scaler)
        
        # Compute permutation importance for this trial
        importances = []
        feature_names = X_eval.columns.tolist()
        threshold = metrics["threshold"]
        y_val_np = y_eval.values if hasattr(y_eval, 'values') else y_eval
        probs = model.predict_proba(X_eval)[:, 1]
        preds = (probs >= threshold).astype(int)
        baseline = np.sum((y_val_np == 1) & (preds == 1)) / (np.sum(preds == 1))
        for feat in feature_names:
            X_shuffled = X_eval.copy()
            X_shuffled[feat] = np.random.permutation(X_shuffled[feat].values)
            probs_shuffled = model.predict_proba(X_shuffled)[:, 1]
            preds_shuffled = (probs_shuffled >= threshold).astype(int)
            precision = np.sum((y_val_np == 1) & (preds_shuffled == 1)) / (np.sum(preds_shuffled == 1))
            drop = baseline - precision
            importances.append(drop)
        feature_importances_trials.append(importances)
        return metrics["precision"]
    
    # Create and run study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, n_jobs=4)
    
    # Aggregate importances
    importances_array = np.array(feature_importances_trials)  # shape: (n_trials, n_features)
    mean_importances = np.mean(importances_array, axis=0)
    importance_df = pd.DataFrame({
        'feature': X_eval.columns,
        'mean_importance': mean_importances
    }).sort_values('mean_importance', ascending=False)
    logger.info('Top features by average permutation importance across trials:')
    for idx, row in importance_df.head(100).iterrows():
        logger.info(f'  {row.feature}: {row.mean_importance:.6f}')
    # You can return this DataFrame or the top N features as a list:
    top_features = importance_df.head(100)['feature'].tolist()
    
    return study.best_params, importance_df


def main():
    """
    Main execution function: loads data, selects features, fits scaler,
    runs hyperparameter tuning, and handles final logging.
    """
    global logger, experiment_name, input_dim, device, scaler
    experiment_name = "pytorch_optimization_20"
    logger = ExperimentLogger(experiment_name)

    # Setup MLflow tracking
    setup_mlflow_tracking(experiment_name)
    try:
        logger.info("Starting PyTorch Model HPO script...")
        
        # Load data using the shared DataLoader
        dataloader = DataLoader() 
        X_train, y_train, X_test, y_test, X_val, y_val = dataloader.load_data()
        
        if X_train is None:
            logger.error("Data loading failed. Exiting.")
            return

        # Select features (using a placeholder type for PyTorch)
        features = import_selected_features_ensemble_new(model_type="all") 
        if not features:
            logger.warning("No features selected for pytorch_model. Using all columns.")
            features = X_train.columns.tolist()
            # Ensure target column is not in features if it exists initially
            if 'target' in features: 
                features.remove('target') 

        X_train = prepare_data(X_train, features)
        X_test = prepare_data(X_test, features)
        X_val = prepare_data(X_val, features)
        input_dim = len(features)
        logger.info(f"Selected {input_dim} features.")

        # Log data shapes and target means
        logger.info(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")
        logger.info(f"Target mean - Train: {y_train.mean():.3f}, Val: {y_val.mean():.3f}, Test: {y_test.mean():.3f}")

        # Fit the scaler ONLY on training data
        scaler_path = "src/models/scalers/scaler_pytorch.pkl"
        if os.path.exists(scaler_path):
            logger.info(f"Loading existing scaler from {scaler_path}")
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        else:
            logger.info("Creating new RobustScaler")
            scaler = RobustScaler()
            scaler.fit(X_train)
            logger.info("RobustScaler fitted on training data.")
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)

        
        # --- Hyperparameter Optimization with Feature Importance ---
        best_params, importance_df = hypertune_with_feature_importance(
            X_train, y_train, X_test, y_test, X_val, y_val, n_trials=50
        )

        # Run Hyperparameter Optimization and Final Model Training
        # best_params, final_metrics = hypertune_pytorch(
        #     X_train, y_train, 
        #     X_test, y_test, 
        #     X_val, y_val,  
        #     experiment_name, 
        #     input_dim,
        #     device,
        #     scaler
        # )

        # train_with_precision_target_pytorch(
        #                 X_train, y_train, X_test, y_test, X_val, y_val,
        #                 experiment_name, input_dim, device, scaler
        #             )
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
    finally:
        gc.collect() # Force garbage collection
        logger.info("--- PyTorch HPO script finished. ---")


if __name__ == "__main__":
    main()

# --- End of File --- 