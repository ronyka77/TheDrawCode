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
from sklearn.metrics import precision_score

# Add sklearn imports for staged feature selection
from sklearn.model_selection import StratifiedKFold
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
    torch.backends.cudnn.benchmark = True 
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
        Optimized predict probabilities method with performance instrumentation.
        Requires scaler_ and device_ attributes to be set.
        Returns probabilities for both classes (0 and 1) in shape (N, 2).
        """
        import time
        start_time = time.time()
        
        if self.scaler_ is None or self.device_ is None:
            raise ValueError("Scaler and Device must be set on the model before calling predict_proba.")
        
        self.network.eval()
        
        # Timing: Data scaling
        scale_start = time.time()
        X_scaled = self.scaler_.transform(X)
        scale_time = time.time() - scale_start
        
        # Timing: Tensor conversion
        tensor_start = time.time()
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.device_)
        tensor_time = time.time() - tensor_start
        
        # Timing: Model inference
        inference_start = time.time()
        n_samples = X_tensor.shape[0]
        all_probs = []
        
        with torch.no_grad():
            # Process in proper batches
            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                batch_X = X_tensor[i:batch_end]
                
                # Forward pass
                outputs = self.network(batch_X)
                # Apply sigmoid and keep on GPU until all batches processed
                probs = torch.sigmoid(outputs)
                all_probs.append(probs)
        
        # Concatenate all results on GPU, then move to CPU once
        all_probs_tensor = torch.cat(all_probs, dim=0)
        probs_class1 = all_probs_tensor.cpu().numpy()  # Shape (N, 1)
        inference_time = time.time() - inference_start
        
        # Timing: Result formatting
        format_start = time.time()
        probs_class0 = 1.0 - probs_class1  # Shape (N, 1)
        result = np.hstack((probs_class0, probs_class1))
        format_time = time.time() - format_start
        
        total_time = time.time() - start_time
        
        # Performance logging (only log if slow)
        if total_time > 1.0:  # Log if prediction takes more than 1 second
            print(f"PERFORMANCE: predict_proba took {total_time:.3f}s for {n_samples} samples")
            print(f"  - Scaling: {scale_time:.3f}s ({scale_time/total_time*100:.1f}%)")
            print(f"  - Tensor conversion: {tensor_time:.3f}s ({tensor_time/total_time*100:.1f}%)")
            print(f"  - Inference: {inference_time:.3f}s ({inference_time/total_time*100:.1f}%)")
            print(f"  - Formatting: {format_time:.3f}s ({format_time/total_time*100:.1f}%)")
            print(f"  - Effective batch size: {batch_size}, Batches: {(n_samples + batch_size - 1) // batch_size}")
        
        return result

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

def preprocess_data(X_train, X_test, X_eval=None):
    try:
        with open('src/models/scalers/scaler_pytorch.pkl', 'rb') as f:
            scaler = pickle.load(f)
        logger.info("Loaded existing PyTorch scaler")
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_eval_scaled = scaler.transform(X_eval)
    except Exception as e:
        logger.error(f"Error loading PyTorch scaler: {str(e)}")
        scaler = RobustScaler()
        logger.info("Created new PyTorch scaler")
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_eval_scaled = scaler.transform(X_eval)
        with open('src/models/scalers/scaler_pytorch.pkl', 'wb') as f:
            pickle.dump(scaler, f)
    return X_train_scaled, X_test_scaled, X_eval_scaled, scaler

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
# Define the objective function with fixed arguments using lambda
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
        
        if score > 0.31 and score > best_score_overall:
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

    # Pass necessary data to the objective function
    def objective_func(trial):
        nonlocal best_score_overall
        return objective(
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
        study.optimize(objective_func, n_trials=n_trials, callbacks=[log_callback], n_jobs=6)
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
        # --- Define hardcoded best parameters here ---
        fixed_params = {
            "learning_rate": 0.00020272553087508855,
            "optimizer": "Adam", 
            "weight_decay": 0.0005416238713396992,
            "batch_size": 512,
            "num_epochs": 300,
            "num_layers": 2,
            "hidden_units": 96,
            "activation_fn": "LeakyReLU",
            "dropout_rate": 0.56,
            "early_stopping_patience": 20
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
            threshold=metrics["threshold"]
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
    n_repeats: int = 20,
    random_state: int = 19,
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
    try:
        feature_names = X_val.columns.tolist()
        y_val_np = y_val.values if hasattr(y_val, 'values') else y_val
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
                feat_idx = feature_names.index(feat) + 1
                logger.info(f"Shuffling feature: {feat} ({feat_idx}) - Repeat: {i+1}")
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
        logger.info(df_importance.head(100).to_string(index=False))
        return df_importance
    except Exception as e:
        logger.error(f"Error in compute_permutation_importance: {str(e)}")
        return None

def pytorch_staged_selection(X, y, X_eval, y_eval, target_features=80, device=None, scaler=None):
    """
    Multi-stage PyTorch neural network feature selection with different objectives.
    Args:
        X (pd.DataFrame): Training features
        y (pd.Series): Training labels
        X_eval (pd.DataFrame): Evaluation features  
        y_eval (pd.Series): Evaluation labels
        target_features (int): Number of final features to select
        device (torch.device): Device for training (defaults to global device)
        scaler (sklearn scaler): Fitted scaler (will create new if None)
        
    Returns:
        tuple: (selected_features_list, final_importance_scores)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Starting PyTorch staged selection with {X.shape[1]} initial features")
    
    # Stage 1: Quick filter with simple architecture and high learning rate
    logger.info("Stage 1: Quick filter with simple architecture")
    
    # Prepare scaler for Stage 1
    if scaler is None:
        stage1_scaler = RobustScaler()
        stage1_scaler.fit(X)  # Just fit the scaler, don't transform
    else:
        stage1_scaler = scaler
    
    # Stage 1 model: Simple and fast
    stage1_params = {
        "num_layers": 2,
        "hidden_units": 64,
        "activation_fn": "ReLU",
        "dropout_rate": 0.3,
        "learning_rate": 0.01,
        "batch_size": 512,
        "num_epochs": 100, 
        "early_stopping_patience": 15,
        "optimizer": "Adam",
        "weight_decay": 1e-4
    }
    
    input_dim = X.shape[1]
    stage1_model = create_pytorch_model(stage1_params, input_dim, device, stage1_scaler)
    
    # Train Stage 1 model
    stage1_model, stage1_metrics = train_pytorch_model(
        stage1_model, X, y, X, y, X_eval, y_eval,  # Use same data for train/test in stage 1
        stage1_params, device, stage1_scaler, trial=None
    )
    
    if not stage1_metrics:
        logger.error("Stage 1 training failed")
        return X.columns.tolist()[:target_features], np.ones(target_features)
    
    # Compute permutation importance for Stage 1
    logger.info("Computing Stage 1 permutation importance...")
    stage1_importance = compute_permutation_importance(
        model=stage1_model,
        X_val=X_eval,
        y_val=y_eval.values if hasattr(y_eval, 'values') else y_eval,
        metric=stage1_metrics["precision"],
        threshold=stage1_metrics["threshold"],
        n_repeats=5  # Fewer repeats for speed in stage 1
    )
    logger.info("Stage 1 importance: permutation complete")
    
    # Select top 200 features from Stage 1
    stage1_features = stage1_importance.head(200)['feature'].tolist()
    logger.info(f"Stage 1: Selected {len(stage1_features)} features")
    
    # Stage 2: Refined selection with cross-validation
    logger.info("Stage 2: Refined selection with cross-validation")
    X_stage1 = X[stage1_features]
    X_eval_stage1 = X_eval[stage1_features]
    
    # Stage 2 model: More complex and thorough
    stage2_params = {
        "num_layers": 3,
        "hidden_units": 128,
        "activation_fn": "LeakyReLU",
        "dropout_rate": 0.4,
        "learning_rate": 0.001,  # Lower learning rate for refined training
        "batch_size": 256,
        "num_epochs": 200,
        "early_stopping_patience": 25,
        "optimizer": "AdamW",
        "weight_decay": 1e-3
    }
    
    # Cross-validation feature importance
    cv_importances = []
    cv_scores = []
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_stage1, y)):
        logger.info(f"Processing fold {fold + 1}/5")
        
        X_train_cv = X_stage1.iloc[train_idx]
        X_val_cv = X_stage1.iloc[val_idx]
        y_train_cv = y.iloc[train_idx]
        y_val_cv = y.iloc[val_idx]
        
        # Create fold-specific scaler
        fold_scaler = RobustScaler()
        fold_scaler.fit(X_train_cv)  # Just fit the scaler, don't transform
        
        # Create and train model for this fold
        input_dim_stage2 = len(stage1_features)
        fold_model = create_pytorch_model(stage2_params, input_dim_stage2, device, fold_scaler)
        
        fold_model, fold_metrics = train_pytorch_model(
            fold_model, X_train_cv, y_train_cv, X_val_cv, y_val_cv, 
            X_eval_stage1, y_eval, stage2_params, device, fold_scaler, trial=None
        )
        
        if fold_metrics:
            # Compute permutation importance for this fold
            fold_importance = compute_permutation_importance(
                model=fold_model,
                X_val=X_eval_stage1,
                y_val=y_eval.values if hasattr(y_eval, 'values') else y_eval,
                metric=fold_metrics["precision"],
                threshold=fold_metrics["threshold"],
                n_repeats=10
            )
            
            # Store importance scores in the same order as stage1_features
            importance_dict = dict(zip(fold_importance['feature'], fold_importance['importance']))
            fold_importance_scores = [importance_dict.get(feat, 0.0) for feat in stage1_features]
            cv_importances.append(fold_importance_scores)
            
            # Calculate validation score
            y_eval_pred_proba = fold_model.predict_proba(X_eval_stage1)[:, 1]
            y_eval_pred = (y_eval_pred_proba >= fold_metrics["threshold"]).astype(int)
            y_eval_np = y_eval.values if hasattr(y_eval, 'values') else y_eval
            
            # Calculate precision for this fold
            if np.sum(y_eval_pred) > 0:
                fold_precision = precision_score(y_eval_np, y_eval_pred)
                cv_scores.append(fold_precision)
            else:
                cv_scores.append(0.0)
        else:
            logger.warning(f"Fold {fold + 1} training failed, using zero importance")
            cv_importances.append([0.0] * len(stage1_features))
            cv_scores.append(0.0)
        
        # Clean up GPU memory
        del fold_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Average importance across folds
    if cv_importances:
        avg_importance = np.mean(cv_importances, axis=0)
        
        # Select top features based on average importance
        feature_importance_pairs = list(zip(stage1_features, avg_importance))
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        stage2_features = [feat for feat, _ in feature_importance_pairs[:target_features]]
        final_importance_scores = [imp for _, imp in feature_importance_pairs[:target_features]]
        CV_Precision_Score = np.mean(cv_scores)
        logger.info(f"Stage 2: Selected {len(stage2_features)} features")

        return stage2_features, final_importance_scores, CV_Precision_Score
    else:
        logger.error("No valid cross-validation results obtained")
        return stage1_features[:target_features], np.ones(target_features)

def run_staged_feature_selection_workflow(
    X_train, y_train, X_test, y_test, X_val, y_val, 
    experiment_name: str, target_features: int = 80
):
    """
    Run the complete staged feature selection workflow and save results.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data  
        X_val, y_val: Validation data
        experiment_name (str): Experiment name for file naming
        target_features (int): Number of features to select
        
    Returns:
        tuple: (selected_features, importance_scores, selection_metrics)
    """
    logger.info(f"Starting staged feature selection workflow for {target_features} features")
    
    # Combine train and test data for feature selection
    X_combined = pd.concat([X_train, X_test], axis=0)
    y_combined = pd.concat([y_train, y_test], axis=0)
    
    # Run staged feature selection
    selected_features, importance_scores, CV_Precision_Score = pytorch_staged_selection(
        X=X_combined,
        y=y_combined, 
        X_eval=X_val,
        y_eval=y_val,
        target_features=target_features,
        device=device,
        scaler=None  # Let the function create its own scaler
    )
    logger.info(f"Selected features: {selected_features}")
    logger.info(f"CV Precision Score: {CV_Precision_Score}")
    # Save as JSON for programmatic use
    feature_dict = {
        "selected_features": selected_features,
        "importance_scores": importance_scores,
        "metadata": {
            "target_features": target_features,
            "original_feature_count": X_combined.shape[1],
            "selection_timestamp": datetime.now().isoformat(),
            "selection_method": "pytorch_staged_selection",
            "feature_reduction_ratio": len(selected_features) / X_combined.shape[1],
            "mean_importance_score": float(np.mean(importance_scores)),
            "std_importance_score": float(np.std(importance_scores)),
            "CV_Precision_Score": CV_Precision_Score
        }
    }
    
    feature_json_path = f"selected_features_pytorch_{target_features}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(feature_json_path, 'w') as f:
        json.dump(feature_dict, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    logger.info("Feature selection results saved to:")
    logger.info(f"  - JSON file: {feature_json_path}")
    logger.info(f"Selected {len(selected_features)} features out of {X_combined.shape[1]} original features")
    logger.info(f"Feature reduction ratio: {len(selected_features) / X_combined.shape[1]:.3f}")
    
    return selected_features, importance_scores, CV_Precision_Score


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
        # Select features using the existing method
        model_type = "pytorch"
        features = import_selected_features_ensemble_new(model_type=model_type) 
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
        logger.info(f"Final feature set: {input_dim} features.")
        X_train_scaled, X_test_scaled, X_val_scaled, scaler = preprocess_data(X_train, X_test, X_val)

        # Configuration: Set to True to run staged feature selection
        RUN_TYPE = "tuning"  # Change this to enable/disable staged feature selection
        
        if RUN_TYPE == "staged_selection":
            TARGET_FEATURES = 100  # Number of features to select
            logger.info("=== RUNNING STAGED FEATURE SELECTION ===")
            
            # Run staged feature selection workflow
            selected_features, importance_scores, CV_Precision_Score = run_staged_feature_selection_workflow(
                X_train, y_train, X_test, y_test, X_val, y_val,
                experiment_name, target_features=TARGET_FEATURES
            )
            
            # Use selected features for the rest of the pipeline
            features = selected_features
            logger.info(f"Using {len(features)} features from staged selection, features: {features}")
            logger.info(f"CV Precision Score: {CV_Precision_Score:.4f}")
        elif RUN_TYPE == "tuning":
            # Run Hyperparameter Optimization and Final Model Training
            best_params, final_metrics = hypertune_pytorch(
                X_train, y_train, 
                X_test, y_test, 
                X_val, y_val,  
                experiment_name, 
                input_dim,
                device,
                scaler
            )
        elif RUN_TYPE == "training":
            train_with_precision_target_pytorch(
                            X_train, y_train, X_test, y_test, X_val, y_val,
                            experiment_name, input_dim, device, scaler
                        )
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
    finally:
        gc.collect() # Force garbage collection
        logger.info("--- PyTorch HPO script finished. ---")


if __name__ == "__main__":
    main()

# --- End of File --- 