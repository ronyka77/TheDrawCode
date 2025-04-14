"""
RNN Model for Soccer Draw Prediction

This module implements a Recurrent Neural Network-based model for predicting soccer match draws.
It processes team sequence data to capture temporal patterns in performance,
with functionality for model creation, training, hyperparameter optimization,
threshold tuning, and MLflow integration for experiment tracking.

The implementation focuses on high precision while maintaining a minimum recall threshold.
"""

import os
import pickle
import random
import time
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Import own modules
from src.utils.logger import ExperimentLogger

# Define experiment name
experiment_name = "rnn_soccer_prediction"
logger = ExperimentLogger(experiment_name)

# Import data utilities at runtime to avoid global scope issues
from src.models.StackedEnsemble.shared.data_loader import DataLoader as MatchDataLoader
from src.models.StackedEnsemble.shared.hypertuner_utils import optimize_threshold
from src.utils.create_evaluation_set import (
    import_selected_features_ensemble,
    setup_mlflow_tracking,
)

# Setup MLflow tracking
mlrunds_dir = setup_mlflow_tracking(experiment_name)

# Global settings
min_recall = 0.30  # Minimum acceptable recall
n_trials = 100     # Number of hyperparameter optimization trials

# Define pip requirements
pip_requirements = [
    f"torch=={torch.__version__}",
    f"mlflow=={mlflow.__version__}",
    "scikit-learn",
    "numpy",
    "pandas"
]

# Set fixed seed and hash seed for determinism
SEED = 19
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    
# Restrict parallel threads
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"

# Device configuration - use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Base parameters
base_params = {
    "batch_size": 32,
    "sequence_length": 5,
    "hidden_size": 64,
    "num_layers": 1,
    "dropout": 0.2,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "epochs": 100,
    "patience": 20,
    "early_stopping_criterion": "loss"  # Options: 'loss' or 'auc'
}


def create_sequences(df, feature_cols, team_id_col, date_col, sequence_length=5):
    """
    Transform tabular match data into sequences for each team.
    
    Args:
        df (pd.DataFrame): DataFrame with match features sorted by date
        feature_cols (list): Features to include in the sequences
        team_id_col (str): Column name for team identifier
        date_col (str): Column name for match date
        sequence_length (int): Number of previous matches to include in sequence
        
    Returns:
        dict: Dictionary mapping team IDs to their sequence data
    """
    logger.info(f"Creating team sequences with length {sequence_length}...")
    
    team_sequences = {}
    teams = df[team_id_col].unique()
    teams_with_less_matches = []
    for team in teams:
        # Get matches for this team
        team_matches = df[df[team_id_col] == team].sort_values(date_col)
        
        if len(team_matches) < sequence_length:
            # logger.warning(f"Team {team} has fewer than {sequence_length} matches: {len(team_matches)}")
            teams_with_less_matches.append(team)
            continue
            
        # Store sequences
        sequences = []
        for i in range(len(team_matches) - sequence_length + 1):
            seq = team_matches.iloc[i:i+sequence_length][feature_cols].values
            sequences.append(seq)
            
        if sequences:
            team_sequences[team] = sequences
            
    logger.info(f"Created sequences for {len(team_sequences)} teams")
    logger.info(f"Teams with less than {sequence_length} matches: {len(teams_with_less_matches)}")
    return team_sequences


def prepare_match_sequences(matches_df, home_team_col, away_team_col, team_sequences, target_col):
    """
    Prepare sequence data for each match by combining home and away team sequences.
    
    Args:
        matches_df (pd.DataFrame): DataFrame with match information
        home_team_col (str): Column name for home team ID
        away_team_col (str): Column name for away team ID
        team_sequences (dict): Dictionary of team sequences from create_sequences()
        target_col (str): Column name for target variable
        
    Returns:
        tuple: (X_sequences, y_targets) where:
            - X_sequences is numpy array of shape [num_matches, 2, sequence_length, features]
            - y_targets is numpy array of match outcomes
    """
    logger.info("Preparing match sequence data...")
    
    X_sequences = []
    y_targets = []
    
    for idx, match in matches_df.iterrows():
        home_team = match[home_team_col]
        away_team = match[away_team_col]
        
        # Skip if we don't have sequence data for either team
        if home_team not in team_sequences or away_team not in team_sequences:
            continue
            
        # Take the most recent sequence for each team
        home_seq = team_sequences[home_team][-1]  # Last available sequence
        away_seq = team_sequences[away_team][-1]  # Last available sequence
        
        # Stack home and away sequences
        match_sequence = np.stack([home_seq, away_seq])
        X_sequences.append(match_sequence)
        y_targets.append(match[target_col])
    
    logger.info(f"Created sequence data for {len(X_sequences)} matches")
    return np.array(X_sequences), np.array(y_targets)


def create_sequence_data(X, y, match_id_col, home_team_col, away_team_col, date_col, feature_cols, sequence_length=5):
    """
    Process data into sequences for RNN model.
    
    Args:
        X (pd.DataFrame): Feature dataframe
        y (pd.Series): Target variable
        match_id_col (str): Column name for match ID
        home_team_col (str): Column name for home team ID
        away_team_col (str): Column name for away team ID  
        date_col (str): Column name for match date
        feature_cols (list): Features to include in the sequences
        sequence_length (int): Length of sequences to create
        
    Returns:
        tuple: (X_sequences, y_sequences) for model training
    """
    # First create a combined dataframe with features and target
    combined_df = X.copy()
    combined_df[y.name] = y
    
    # Create home team dataframe
    home_df = combined_df.copy()
    home_df['team_id'] = home_df[home_team_col]
    
    # Create away team dataframe  
    away_df = combined_df.copy()
    away_df['team_id'] = away_df[away_team_col]
    
    # Combine both perspectives
    team_df = pd.concat([home_df, away_df], axis=0)
    team_df = team_df.sort_values(date_col)
    
    # Create sequences for each team
    team_sequences = create_sequences(
        team_df, 
        feature_cols=feature_cols,
        team_id_col='team_id',
        date_col=date_col,
        sequence_length=sequence_length
    )
    
    # Prepare match sequences
    X_sequences, y_sequences = prepare_match_sequences(
        combined_df,
        home_team_col=home_team_col,
        away_team_col=away_team_col,
        team_sequences=team_sequences,
        target_col=y.name
    )
    
    return X_sequences, y_sequences


class VanillaRNN(nn.Module):
    """
    Vanilla RNN model for soccer match prediction.
    
    Processes sequences of home and away team data separately, then combines
    them for final prediction.
    """
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2):
        super(VanillaRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN layers for processing team data
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Team feature processors
        self.team_processor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Combined team features
        self.combiner = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, 2 (home/away), sequence_length, features]
            
        Returns:
            Tensor with prediction probabilities
        """
        batch_size = x.size(0)
        
        # Process each team separately
        team_vectors = []
        for i in range(2):  # home=0, away=1
            team_input = x[:, i, :, :]  # [batch, seq_len, features]
            
            # Initialize hidden state
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            
            # Forward propagate RNN
            out, _ = self.rnn(team_input, h0)
            
            # Take the output of the last time step
            team_vec = self.team_processor(out[:, -1, :])
            team_vectors.append(team_vec)
        
        # Combine team representations
        combined = torch.cat([team_vectors[0], team_vectors[1]], dim=1)
        output = self.combiner(combined)
        
        return output


class RNNWrapper:
    """
    Wraps PyTorch RNN model to provide scikit-learn compatible interface.
    """
    def __init__(self, model, device, scaler=None):
        self.model = model
        self.device = device
        self.scaler = scaler
        self.classes_ = np.array([0, 1])
    
    def predict_proba(self, X):
        """Return probability estimates for samples in X.
        
        Args:
            X: Input features of shape [n_samples, 2, sequence_length, n_features]
            
        Returns:
            Array of shape [n_samples, 2] with probabilities for each class
        """
        self.model.eval()
        with torch.no_grad():
            # Convert input to tensor if it's not already
            if not isinstance(X, torch.Tensor):
                X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            else:
                X_tensor = X.to(self.device)
                
            # Get model outputs
            outputs = self.model(X_tensor).cpu().numpy().flatten()
            
            # Return probabilities for both classes [P(0), P(1)]
            return np.column_stack((1 - outputs, outputs))
    
    def predict(self, X, threshold=0.5):
        """Return class predictions.
        
        Args:
            X: Input features
            threshold: Classification threshold
            
        Returns:
            Array of predicted classes
        """
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


def load_hyperparameter_space():
    """
    Define hyperparameter space for RNN tuning.
    
    Returns:
        dict: Hyperparameter space configuration.
    """
    hyperparameter_space = {
        'hidden_size': {
            'type': 'int',
            'low': 32,
            'high': 256,
            'step': 32
        },
        'num_layers': {
            'type': 'int',
            'low': 1,
            'high': 5
        },
        'dropout': {
            'type': 'float',
            'low': 0.05,
            'high': 0.5,
            'step': 0.05
        },
        'learning_rate': {
            'type': 'float',
            'low': 1e-6,
            'high': 1e-2,
            'log': True
        },
        'batch_size': {
            'type': 'int',
            'low': 16,
            'high': 256,
            'step': 16
        },
        'epochs': {
            'type': 'int',
            'low': 50,
            'high': 300,
            'step': 10
        },
        'sequence_length': {
            'type': 'int',
            'low': 3,
            'high': 10
        },
        'patience': {
            'type': 'int',
            'low': 4,
            'high': 50,
            'step': 2
        },
        'weight_decay': {
            'type': 'float',
            'low': 1e-6,
            'high': 1e-3,
            'log': True
        },
        'early_stopping_criterion': {
            'type': 'categorical',
            'choices': ['loss', 'auc']
        }
    }
    return hyperparameter_space


def train_model(X_train, y_train, X_val, y_val, X_eval, y_eval, params):
    """
    Train RNN model with early stopping.
    
    Args:
        X_train: Training features [n_samples, 2, sequence_length, n_features]
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        X_eval: Evaluation features
        y_eval: Evaluation labels
        params: Model parameters
        
    Returns:
        tuple: (model, metrics, best_threshold)
    """
    logger.info("Training RNN model with params: " + str(params))
    
    # Get input dimension from data
    input_size = X_train.shape[-1]
    logger.info(f"Input size: {input_size}")
    
    # Create model
    model = VanillaRNN(
        input_size=input_size,
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        dropout=params['dropout']
    ).to(device)
    
    # Prepare datasets
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=params['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=params['batch_size']
    )
    
    # Binary Cross Entropy Loss
    criterion = nn.BCELoss()
    
    # Adam optimizer with weight decay
    optimizer = optim.Adam(
        model.parameters(), 
        lr=params['learning_rate'],
        weight_decay=params['weight_decay']
    )
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    best_val_auc = 0.0
    patience_counter = 0
    best_model_state = None
    
    # Store metrics for tracking
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc': []
    }
    
    for epoch in range(params['epochs']):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                # Store predictions and targets for AUC calculation
                val_preds.extend(outputs.cpu().numpy().flatten())
                val_targets.extend(labels.cpu().numpy().flatten())
                
        val_loss = val_loss / len(val_loader.dataset)
        history['val_loss'].append(val_loss)
        
        # Calculate validation AUC
        val_auc = roc_auc_score(val_targets, val_preds)
        history['val_auc'].append(val_auc)
        
        # Early stopping based on criterion
        if params.get('early_stopping_criterion', 'loss') == 'auc':
            # For AUC, we want to maximize it
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_val_loss = val_loss  # Still track loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                # logger.info(f"Epoch {epoch}: New best validation AUC: {best_val_auc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= params['patience']:
                    logger.info(f"Early stopping at epoch {epoch}. Best AUC: {best_val_auc:.4f}")
                    break
        else:
            # Original loss-based criterion
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_auc = val_auc  # Still track AUC
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                # logger.info(f"Epoch {epoch}: New best validation loss: {best_val_loss:.4f}, AUC: {best_val_auc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= params['patience']:
                    logger.info(f"Early stopping at epoch {epoch}. Best loss: {best_val_loss:.4f}")
                    break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Create wrapper
    wrapper = RNNWrapper(model, device)
    
    # Optimize threshold
    logger.info("Optimizing classification threshold...")
    best_threshold, metrics = optimize_threshold(wrapper, X_eval, y_eval, min_recall=min_recall)
    
    # Add AUC to metrics
    metrics['best_val_auc'] = best_val_auc
    
    # Log final metrics
    logger.info(f"Best threshold: {best_threshold}")
    logger.info(f"Precision: {metrics.get('precision', 0.0)}")
    logger.info(f"Recall: {metrics.get('recall', 0.0)}")
    logger.info(f"F1: {metrics.get('f1', 0.0)}")
    logger.info(f"Best validation AUC: {best_val_auc:.4f}")
    
    # Return model, metrics, threshold and history
    return model, metrics, best_threshold


def optimize_hyperparameters(X_train, y_train, X_val, y_val, X_eval, y_eval, hyperparameter_space=None):
    """
    Run hyperparameter optimization using Optuna.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        X_eval: Evaluation features for threshold optimization
        y_eval: Evaluation labels for threshold optimization
        hyperparameter_space: Optional custom hyperparameter space
        
    Returns:
        dict: Best hyperparameters
    """
    logger.info("Starting hyperparameter optimization")
    
    if not hyperparameter_space:
        hyperparameter_space = load_hyperparameter_space()
    
    best_score = -float("inf")
    best_params = {}
    top_trials = []
    
    def objective(trial):
        nonlocal best_score, best_params
        
        # Sample hyperparameters
        params = {}
        for param_name, param_config in hyperparameter_space.items():
            if param_config['type'] == 'float':
                if 'step' in param_config and not param_config.get('log', False):
                    params[param_name] = trial.suggest_float(
                        param_name, 
                        param_config['low'], 
                        param_config['high'],
                        step=param_config['step']
                    )
                else:
                    params[param_name] = trial.suggest_float(
                        param_name, 
                        param_config['low'], 
                        param_config['high'],
                        log=param_config.get('log', False)
                    )
            elif param_config['type'] == 'int':
                if 'step' in param_config and param_config['step'] > 1:
                    params[param_name] = trial.suggest_int(
                        param_name, 
                        param_config['low'], 
                        param_config['high'],
                        step=param_config['step']
                    )
                else:
                    params[param_name] = trial.suggest_int(
                        param_name, 
                        param_config['low'], 
                        param_config['high']
                    )
        
        try:
            # Train model with these parameters
            model, metrics, threshold = train_model(
                X_train, y_train, X_val, y_val, X_eval, y_eval, params
            )
            
            precision = metrics.get('precision', 0.0)
            recall = metrics.get('recall', 0.0)
            
            # Score based on precision, requiring minimum recall
            score = precision if recall >= min_recall else 0.0
            
            # Log trial results
            logger.info(f"Trial {trial.number}:")
            logger.info(f"  Score: {score:.4f} (Precision: {precision:.4f}, Recall: {recall:.4f})")
            logger.info(f"  Threshold: {threshold:.4f}")
            logger.info(f"  Params: {params}")
            
            # Log metrics to trial
            for metric_name, metric_value in metrics.items():
                trial.set_user_attr(metric_name, metric_value)
                
            # Update best score and params
            if score > best_score:
                best_score = score
                best_params = params.copy()
            if score > 0.35:
                log_to_mlflow(model, metrics, params, experiment_name, X_eval)
            return score
            
        except Exception as e:
            logger.error(f"Error in trial {trial.number}: {str(e)}")
            return 0.0
    
    # Callback function to track top trials
    def callback(study, trial):
        nonlocal best_score, best_params, top_trials
        
        if trial.value > best_score:
            best_score = trial.value
            best_params = trial.params
            logger.info(f"New best score found in trial {trial.number}: {best_score:.4f}")
            
        # Track top trials for logging
        current_run = (trial.value, trial.params, trial.number)
        top_trials.append(current_run)
        top_trials.sort(key=lambda x: x[0], reverse=True)
        top_trials = top_trials[:10]  # Keep only top 10
        
        # Log top trials periodically
        if trial.number % 5 == 0:
            table_header = "| Rank | Trial # | Score | Parameters |"
            table_separator = "|------|---------|-------|------------|"
            table_rows = [
                f"| {i + 1} | {rec[2]} | {rec[0]:.4f} | {rec[1]} |"
                for i, rec in enumerate(top_trials)
            ]
            logger.info("Top trials in current batch:")
            logger.info(table_header)
            logger.info(table_separator)
            for row in table_rows:
                logger.info(row)
    
    # Create Optuna study
    storage_url = "sqlite:///optuna_rnn.db"
    study_name = "rnn_optimization"
    
    # Create sampler with seed
    sampler = optuna.samplers.RandomSampler(seed=SEED)
    
    try:
        # Create or load existing study
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            storage=storage_url,
            load_if_exists=True,
            sampler=sampler
        )
        
        # Run optimization
        study.optimize(objective, n_trials=n_trials, callbacks=[callback])
        
        # Extract best parameters
        best_params = study.best_trial.params
        logger.info(f"Best trial value: {study.best_trial.value:.4f}")
        logger.info(f"Best hyperparameters: {best_params}")
        
        # Combine with base parameters
        final_best_params = base_params.copy()
        final_best_params.update(best_params)
        
        return final_best_params
        
    except Exception as e:
        logger.error(f"Error in hyperparameter optimization: {str(e)}")
        return base_params  # Fall back to base params on error


def log_to_mlflow(model, metrics, params, experiment_name, X_eval):
    """
    Log trained RNN model, metrics, and parameters to MLflow.
    
    Args:
        model: Trained PyTorch model
        metrics: Evaluation metrics dictionary
        params: Model parameters dictionary
        experiment_name: MLflow experiment name
        X_eval: Evaluation data for model signature
    
    Returns:
        str: Run ID
    """
    try:
        # Set up MLflow tracking
        mlflow.set_experiment(experiment_name)
        
        # Start a new run
        with mlflow.start_run(run_name=f"rnn_final_{datetime.now().strftime('%Y%m%d_%H%M')}") as run:
            # Log parameters
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Create a wrapper for the model
            wrapper = RNNWrapper(model, device)
            
            # Create input example using a small sample of X_eval
            input_example = X_eval[:5] if len(X_eval) >= 5 else X_eval
            
            # Infer signature
            signature = mlflow.models.infer_signature(
                input_example, 
                wrapper.predict_proba(input_example)
            )
            
            # Save model's state dictionary
            model_state = model.state_dict()
            state_dict_path = "model_state.pt"
            torch.save(model_state, state_dict_path)
            mlflow.log_artifact(state_dict_path, "state_dict")
            os.remove(state_dict_path)  # Clean up
            
            # Log PyTorch model
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                pip_requirements=pip_requirements,
                registered_model_name=f"rnn_{datetime.now().strftime('%Y%m%d_%H%M')}",
                signature=signature
            )
            
            # Log threshold
            mlflow.log_metric("threshold", metrics.get("threshold", 0.5))
            
            logger.info(f"Model logged to MLflow: {run.info.run_id}")
            return run.info.run_id
            
    except Exception as e:
        logger.error(f"Error logging to MLflow: {str(e)}")
        return None


def train_with_precision_target(X_train, y_train, X_test, y_test, X_eval, y_eval):
    """
    Train RNN model with focus on precision target.
    
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
        
        # Parameters tuned for precision
        params = base_params.copy()
        params.update({
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.3,
            "learning_rate": 0.0005,
            "batch_size": 64,
            "epochs": 150,
            "patience": 30,
            "weight_decay": 0.0001,
            "sequence_length": 5
        })
        
        # Train final model with precision-focused parameters
        model, metrics, threshold = train_model(
            X_train, y_train, X_test, y_test, X_eval, y_eval, params
        )
        
        # Log to MLflow
        log_to_mlflow(model, metrics, params, experiment_name, X_eval)
        
        return model, metrics, params
        
    except Exception as e:
        logger.error(f"Error in precision-focused training: {str(e)}")
        return None, None, None


def main():
    """
    Main execution function for RNN model training.
    """
    try:
        logger.info("Starting RNN model training")
        
        # Ensure directories exist
        os.makedirs("src/models/scalers", exist_ok=True)
        
        # Load data
        data_loader = MatchDataLoader()
        X_train_raw, y_train, X_test_raw, y_test, X_eval_raw, y_eval = data_loader.load_data()
        
        # Load selected features
        features = import_selected_features_ensemble(model_type="all")
        logger.info(f"Using {len(features)} features: {features[:5]}...")
        
        # Filter to selected features
        X_train = X_train_raw[features]
        X_test = X_test_raw[features]
        X_eval = X_eval_raw[features]
        
        # Create sequence data
        # These are placeholder column names - replace with your actual column names
        match_id_col = "fixture_id"
        home_team_col = "home_encoded"
        away_team_col = "away_encoded"
        date_col = "date_encoded"
        
        # Default sequence length from base params
        sequence_length = base_params["sequence_length"]
        
        logger.info(f"Creating sequence data with length {sequence_length}...")
        X_train_seq, y_train_seq = create_sequence_data(
            X_train, y_train, 
            match_id_col=match_id_col,
            home_team_col=home_team_col, 
            away_team_col=away_team_col,
            date_col=date_col,
            feature_cols=features,
            sequence_length=sequence_length
        )
        
        X_test_seq, y_test_seq = create_sequence_data(
            X_test, y_test,
            match_id_col=match_id_col,
            home_team_col=home_team_col, 
            away_team_col=away_team_col,
            date_col=date_col,
            feature_cols=features,
            sequence_length=sequence_length
        )
        
        X_eval_seq, y_eval_seq = create_sequence_data(
            X_eval, y_eval,
            match_id_col=match_id_col,
            home_team_col=home_team_col, 
            away_team_col=away_team_col,
            date_col=date_col,
            feature_cols=features,
            sequence_length=sequence_length
        )
        
        logger.info(f"Sequence data shapes - Train: {X_train_seq.shape}, Test: {X_test_seq.shape}, Eval: {X_eval_seq.shape}")
        
        # Run hyperparameter optimization
        logger.info("Starting hyperparameter optimization")
        best_params = optimize_hyperparameters(
            X_train_seq, y_train_seq,
            X_test_seq, y_test_seq,
            X_eval_seq, y_eval_seq
        )
        
        logger.info(f"Best parameters: {best_params}")
        
        # Optionally train with precision target focus
        model, metrics, final_params = train_with_precision_target(
            X_train_seq, y_train_seq,
            X_test_seq, y_test_seq,
            X_eval_seq, y_eval_seq
        )
        
        if model is not None:
            logger.info("Training completed successfully")
            logger.info(f"Final metrics: {metrics}")
        else:
            logger.error("Training failed")
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")


if __name__ == "__main__":
    main() 