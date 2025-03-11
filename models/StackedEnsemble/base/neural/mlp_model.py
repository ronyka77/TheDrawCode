"""
<<<<<<< HEAD
MLP Model for Soccer Draw Prediction

This module implements an MLP-based model for predicting soccer match draws.
It includes functionality for model creation, training, hyperparameter optimization,
threshold tuning, and MLflow integration for experiment tracking.

The implementation focuses on high precision while maintaining a minimum recall threshold.
"""

import os
import sys
import json
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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks

import optuna
import mlflow
import mlflow.keras
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Set project root (similar to xgboost_model.py)
try:
    project_root = Path(__file__).parent.parent.parent.parent.parent
    if not project_root.exists():
=======
MLP (Multi-Layer Perceptron) Model Implementation with CPU Optimization

This module implements an MLP-based model for soccer match draw prediction with CPU optimization.
The implementation includes:
- Model creation and configuration
- Training with early stopping
- Threshold optimization
- Hyperparameter tuning
- Model evaluation
- MLflow integration for experiment tracking
"""

import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
import joblib
import json
import os
import sys
import time
import optuna
import mlflow
import mlflow.keras
import random
from typing import Any, Dict, Tuple, List, Union
from datetime import datetime
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner, HyperbandPruner
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import yaml

# Add project root to Python path
try:
    project_root = Path(__file__).parent.parent.parent.parent.parent
    if not project_root.exists():
        # Handle network path by using raw string
>>>>>>> 3798e304ba09a95ae05e21747b9f93b4e52eb5fd
        project_root = Path(r"\\".join(str(project_root).split("\\")))
    sys.path.append(str(project_root))
    print(f"Project root mlp model: {project_root}")
except Exception as e:
    print(f"Error setting project root path: {e}")
    # Fallback to current directory if path resolution fails
    sys.path.append(os.getcwd().parent)
    print(f"Current directory mlp model: {os.getcwd().parent}")
# Set TensorFlow environment variables for CPU optimization
os.environ["ARROW_S3_DISABLE"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Enable oneDNN optimizations
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # Reduce TensorFlow logging verbosity
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"  # Disable GPU memory growth
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU-only mode
<<<<<<< HEAD
# Set random seeds for reproducibility
random_seed = 19
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
os.environ['PYTHONHASHSEED'] = str(random_seed)
=======
>>>>>>> 3798e304ba09a95ae05e21747b9f93b4e52eb5fd

# Configure TensorFlow for deterministic operations
tf.config.experimental.enable_op_determinism()  # Enable deterministic operations


# Configure Git executable path if available
git_executable = os.environ.get("GIT_PYTHON_GIT_EXECUTABLE")
if git_executable and os.path.exists(git_executable):
    import git
    git.refresh(git_executable)

from utils.logger import ExperimentLogger
experiment_name = "mlp_soccer_prediction"
<<<<<<< HEAD
logger = ExperimentLogger(experiment_name=experiment_name)

from utils.create_evaluation_set import setup_mlflow_tracking
mlflow_tracking = setup_mlflow_tracking(experiment_name)

from models.StackedEnsemble.shared.data_loader import DataLoader

# Global settings
min_recall = 0.20            # Minimum acceptable recall
n_trials = 100               # Fewer trials for MLP due to longer training times
pip_requirements = [
    f"tensorflow=={tf.__version__}",
    "scikit-learn", 
    f"mlflow=={mlflow.__version__}"
]
scaler = None  # Global scaler object
=======
logger = ExperimentLogger(experiment_name)

from utils.create_evaluation_set import setup_mlflow_tracking
from models.StackedEnsemble.utils.metrics import calculate_metrics
from models.StackedEnsemble.shared.data_loader import DataLoader

# Global variables
min_recall = 0.20
n_trials = 100  # Fewer trials since MLPs take longer to train
scaler = None  # Global scaler to be initialized during preprocessing
>>>>>>> 3798e304ba09a95ae05e21747b9f93b4e52eb5fd

# Define base configurations
base_params = {
    'epochs': 100,
    'batch_size': 64,
    'verbose': 0,
    'metrics': ['accuracy', 'AUC']
}

def load_hyperparameter_space():
<<<<<<< HEAD
    """
    Define hyperparameter space for MLP tuning.
    
    Returns:
        dict: Hyperparameter space configuration.
    """
    hyperparameter_space = {
        'learning_rate': {
            'type': 'float',
            'low': 1e-4,
            'high': 1e-2,
            'log': True
        },
        'hidden_layers': {
            'type': 'int',
            'low': 1,
            'high': 5
        },
        'neurons_per_layer': {
            'type': 'int',
            'low': 32,
            'high': 256
        },
        'dropout_rate': {
            'type': 'float',
            'low': 0.001,
            'high': 0.5
        },
        'activation': {
            'type': 'categorical',
            'choices': ['relu', 'elu', 'tanh']
        },
        'l1_regularization': {
            'type': 'float',
            'low': 0.0001,
            'high': 1e-2,
            'log': True
        },
        'l2_regularization': {
            'type': 'float',
            'low': 0.0001,
            'high': 1e-2,
            'log': True
        },
        'batch_size': {
            'type': 'int',
            'low': 16,
            'high': 128
        },
        'epochs': {
            'type': 'int',
            'low': 50,
            'high': 200
        },
        'patience': {
            'type': 'int',
            'low': 10,
            'high': 50
        },
        'class_weight_multiplier': {
            'type': 'float',
            'low': 1.0,
            'high': 5.0
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

def create_model(model_params):
    """
    Create and compile a Keras MLP model based on provided hyperparameters.
    
    Args:
        model_params (dict): Hyperparameters for model configuration.
        
    Returns:
        keras.Model: Compiled MLP model.
    """
    try:
        input_dim = model_params.pop('input_dim')
=======
    """Define hyperparameter space for MLP optimization."""
    try:
        hyperparameter_space = {
            'learning_rate': {
                'type': 'float',
                'low': 1e-4,
                'high': 1e-2,
                'log': True
            },
            'hidden_layers': {
                'type': 'int',
                'low': 1,
                'high': 5
            },
            'neurons_per_layer': {
                'type': 'int',
                'low': 32,
                'high': 512
            },
            'dropout_rate': {
                'type': 'float',
                'low': 0.0,
                'high': 0.5
            },
            'activation': {
                'type': 'categorical',
                'choices': ['relu', 'elu', 'selu', 'tanh']
            },
            'l1_regularization': {
                'type': 'float',
                'low': 1e-6,
                'high': 1e-2,
                'log': True
            },
            'l2_regularization': {
                'type': 'float',
                'low': 1e-6,
                'high': 1e-2,
                'log': True
            },
            'batch_size': {
                'type': 'int',
                'low': 16,
                'high': 256
            },
            'epochs': {
                'type': 'int',
                'low': 50,
                'high': 300
            },
            'patience': {
                'type': 'int',
                'low': 10,
                'high': 50
            },
            'class_weight_multiplier': {
                'type': 'float',
                'low': 1.0,
                'high': 10.0
            }
        }
        
        return hyperparameter_space
    except Exception as e:
        logger.error(f"Error creating hyperparameter space: {str(e)}")
        return None


def preprocess_data(X_train, X_test, X_eval=None):
    """Preprocess data for neural network training - scaling is essential for MLPs."""
    global scaler
    
    try:
        # Create a scaler using training data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Transform test and eval data
        X_test_scaled = scaler.transform(X_test)
        
        if X_eval is not None:
            X_eval_scaled = scaler.transform(X_eval)
            return X_train_scaled, X_test_scaled, X_eval_scaled, scaler
        
        return X_train_scaled, X_test_scaled, scaler
        
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        raise

# Load data
dataloader = DataLoader()
X_train, y_train, X_test, y_test, X_eval, y_eval = dataloader.load_data()
logger.info(f"Data loaded successfully with {X_train.shape[1]} features")

# Preprocess data for neural network
X_train_scaled, X_test_scaled, X_eval_scaled, _ = preprocess_data(X_train, X_test, X_eval)
logger.info("Data preprocessed and scaled")


def create_model(model_params):
    """Create and configure MLP model instance."""
    try:
        # Extract parameters
        input_dim = model_params.pop('input_dim', 102)  # Feature count
>>>>>>> 3798e304ba09a95ae05e21747b9f93b4e52eb5fd
        hidden_layers = model_params.pop('hidden_layers', 2)
        neurons_per_layer = model_params.pop('neurons_per_layer', 128)
        dropout_rate = model_params.pop('dropout_rate', 0.2)
        activation = model_params.pop('activation', 'relu')
        l1_reg = model_params.pop('l1_regularization', 0.0)
        l2_reg = model_params.pop('l2_regularization', 0.0)
        learning_rate = model_params.pop('learning_rate', 0.001)
<<<<<<< HEAD
        
        model = keras.Sequential()
        model.add(layers.InputLayer(shape=(input_dim,)))
        
        # Add hidden layers
        for _ in range(hidden_layers):
            model.add(layers.Dense(
                neurons_per_layer,
=======
        # Set random seed for TensorFlow to ensure reproducibility
        # This is required when determinism is enabled
        seed = model_params.pop('seed', 42)  # Default seed value if not provided
        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        # Force thread settings for CPU determinism
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        # Create model
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.InputLayer(shape=(input_dim,)))
        
        # Hidden layers with decreasing size
        for i in range(hidden_layers):
            neurons = int(neurons_per_layer / (2**i))  # Decrease by factor of 2 each layer
            model.add(layers.Dense(
                neurons,
>>>>>>> 3798e304ba09a95ae05e21747b9f93b4e52eb5fd
                activation=activation,
                kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg)
            ))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate))
        
        # Output layer for binary classification
        model.add(layers.Dense(1, activation='sigmoid'))
<<<<<<< HEAD
=======
        
        # Compile model
>>>>>>> 3798e304ba09a95ae05e21747b9f93b4e52eb5fd
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
<<<<<<< HEAD
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        return model
=======
            metrics=['accuracy', keras.metrics.AUC()]
        )
        
        return model
        
>>>>>>> 3798e304ba09a95ae05e21747b9f93b4e52eb5fd
    except Exception as e:
        logger.error(f"Error creating MLP model: {str(e)}")
        raise

<<<<<<< HEAD
def optimize_threshold(model, X_val, y_val, min_threshold=0.2, max_threshold=0.9, step=0.01):
    """
    Optimize the decision threshold to maximize precision while ensuring recall
    meets the minimum requirement.
    
    Returns:
        tuple: (best_threshold, best_precision, best_recall)
    """
    try:
        y_prob = model.predict(X_val, verbose=0).flatten()
        thresholds = np.arange(min_threshold, max_threshold + step, step)
        best_thresh = 0.5
        best_precision = 0.0
        best_recall = 0.0
        
        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            tp = np.sum((y_val == 1) & (y_pred == 1))
            fp = np.sum((y_val == 0) & (y_pred == 1))
            fn = np.sum((y_val == 1) & (y_pred == 0))
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            if recall >= min_recall and precision > best_precision:
                best_precision = precision
                best_thresh = thresh
                best_recall = recall
        logger.info(f"Optimized threshold: {best_thresh:.3f} with precision: {best_precision:.3f} and recall: {best_recall:.3f}")
        return best_thresh, best_precision, best_recall
    except Exception as e:
        logger.error(f"Error optimizing threshold: {str(e)}")
        return 0.5, 0.0, 0.0

def train_model(X_train, y_train, X_test, y_test, X_eval, y_eval, model_params):
    """
    Train the MLP model with early stopping and optimize the decision threshold.
    
    Returns:
        tuple: (trained model, metrics dictionary)
    """
    try:
        # Set the input dimension based on training data
        model_params['input_dim'] = X_train.shape[1]
        
        model = create_model(model_params.copy())
        
        # Compute class weights for imbalanced data
        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)
        class_weight = {0: 1.0, 1: (neg_count / pos_count)}
        class_weight[1] *= model_params.get('class_weight_multiplier', 1.0)
        
        # Early stopping callback based on validation AUC
        early_stop = callbacks.EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=model_params.get('patience', 20),
=======
def predict(model, X, threshold=0.5):
    """Generate predictions using trained model."""
    if model is None:
        raise RuntimeError("Model must be trained before prediction")
        
    try:
        X_scaled = scaler.transform(X)  # Using the global scaler
        probas = model.predict(X_scaled, verbose=0).flatten()
        return (probas >= threshold).astype(int)
        
    except Exception as e:
        logger.error(f"Error in model prediction: {str(e)}")
        return np.zeros(len(X))

def predict_proba(model, X):
    """Generate probability predictions."""
    if model is None:
        raise RuntimeError("Model must be trained before prediction")
        
    try:
        X_scaled = scaler.transform(X)  # Using the global scaler
        return model.predict(X_scaled, verbose=0).flatten()
        
    except Exception as e:
        logger.error(f"Error in probability prediction: {str(e)}")
        return np.zeros(len(X))

def evaluate(model, X, y, best_threshold):
    """Evaluate model performance on given data."""
    if model is None:
        raise RuntimeError("Model must be trained before evaluation")
    
    try:
        # Scale data
        X_scaled = scaler.transform(X)
        
        # Get probability predictions
        y_prob = model.predict(X_scaled, verbose=0).flatten()
        
        # Get binary predictions using best threshold
        y_pred = (y_prob >= best_threshold).astype(int)
        
        # Calculate metrics
        tp = np.sum((y == 1) & (y_pred == 1))
        fp = np.sum((y == 0) & (y_pred == 1))
        fn = np.sum((y == 1) & (y_pred == 0))
        
        metrics = {
            'precision': tp / (tp + fp + 1e-10),
            'recall': tp / (tp + fn + 1e-10),
            'f1': 2 * tp / (2 * tp + fp + fn + 1e-10),
            'auc': roc_auc_score(y, y_prob),
            'brier_score': np.mean((y_prob - y) ** 2),
            'threshold': best_threshold
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error in model evaluation: {str(e)}")
        return {
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 
            'auc': 0.0, 'brier_score': 1.0, 'threshold': best_threshold
        }


def optimize_threshold(model, y_true, y_prob):
    """Optimize prediction threshold with focus on precision while maintaining recall above minimum threshold."""
    try:
        best_threshold = 0.5
        best_precision = 0.0
        best_recall = 0.0
        
        # Search through thresholds
        for threshold in np.linspace(0.2, 0.9, 71):
            y_pred = (y_prob >= threshold).astype(int)
            
            # Calculate confusion matrix components
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            
            # Only consider thresholds that maintain recall above minimum threshold
            if recall >= min_recall:
                if precision > best_precision:
                    best_precision = precision
                    best_threshold = threshold
                    best_recall = recall
        
        logger.info(f"Optimized threshold: {best_threshold:.3f} with precision: {best_precision:.3f} and recall: {best_recall:.3f}")
        
        # Return all metrics for this threshold
        X_eval_path = Path(project_root) / "data" / "prediction" / "api_prediction_eval.xlsx"
        if not X_eval_path.exists():
            logger.error(f"Evaluation data not found at {X_eval_path}")
            return {'threshold': best_threshold, 'precision': best_precision, 'recall': 0.0}
        
        return best_threshold, best_precision, best_recall
        
    except Exception as e:
        logger.error(f"Error optimizing threshold: {str(e)}")
        return {'threshold': 0.5, 'precision': 0.0, 'recall': 0.0}


def train_model(model_params):
    """Train MLP model with early stopping."""
    try:
        # Extract training parameters
        epochs = model_params.pop('epochs', 100)
        batch_size = model_params.pop('batch_size', 64)
        patience = model_params.pop('patience', 20)
        class_weight_multiplier = model_params.pop('class_weight_multiplier', 3.0)
        input_dim = X_train.shape[1]  # Get actual feature count
        model_params['input_dim'] = input_dim
        
        # Calculate class weights for imbalanced data
        neg_class_count = np.sum(y_train == 0)
        pos_class_count = np.sum(y_train == 1)
        weight_for_0 = 1.0
        weight_for_1 = (neg_class_count / pos_class_count) * class_weight_multiplier
        class_weight = {0: weight_for_0, 1: weight_for_1}
        
        # Create model
        model = create_model(model_params.copy())
        
        # Early stopping callback
        early_stopping = callbacks.EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=patience,
>>>>>>> 3798e304ba09a95ae05e21747b9f93b4e52eb5fd
            restore_best_weights=True,
            verbose=0
        )
        
<<<<<<< HEAD
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=model_params.get('epochs', 100),
            batch_size=model_params.get('batch_size', 64),
            class_weight=class_weight,
            callbacks=[early_stop],
            verbose=0
        )
        
        best_thresh, best_precision, best_recall = optimize_threshold(model, X_eval, y_eval)
        y_prob = model.predict(X_eval, verbose=0).flatten()
        f1 = f1_score(y_eval, (y_prob >= best_thresh).astype(int))
        auc = roc_auc_score(y_eval, y_prob)
        metrics = {
            'threshold': best_thresh,
            'precision': best_precision,
            'recall': best_recall,
            'f1': f1,
            'auc': auc
        }
        return model, metrics
=======
        # Train model
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Get validation predictions
        y_prob = model.predict(X_eval, verbose=0).flatten()
        
        # Optimize threshold using the optimize_threshold function from this file
        best_threshold, best_precision, best_recall = optimize_threshold(
            model,
            y_eval, 
            y_prob
        )
        
        logger.info(f"Best threshold: {best_threshold:.3f} with precision: {best_precision:.3f}")
        
        # Get final metrics with best threshold
        metrics = evaluate(model, X_eval, y_eval, best_threshold)
        metrics['threshold'] = best_threshold
        
        return model, metrics
        
>>>>>>> 3798e304ba09a95ae05e21747b9f93b4e52eb5fd
    except Exception as e:
        logger.error(f"Error training MLP model: {str(e)}")
        raise

<<<<<<< HEAD
def optimize_hyperparameters(X_train, y_train, X_test, y_test, X_eval, y_eval, hyperparameter_space):
    """
    Run hyperparameter optimization using Optuna.
    
    Returns:
        dict: Best hyperparameters.
    """
    if not hyperparameter_space:
        hyperparameter_space = load_hyperparameter_space()
    
    def objective(trial):
        try:
            params = {}
            for param_name, param_config in hyperparameter_space.items():
                if param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config['low'],
                        param_config['high']
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config['choices']
                    )
            params.update(base_params)
            model, metrics = train_model(X_train, y_train, X_test, y_test, X_eval, y_eval, params.copy())
            precision = metrics.get('precision', 0.0)
            recall = metrics.get('recall', 0.0)
            score = precision if recall >= min_recall else 0.0
            for metric_name, metric_value in metrics.items():
                trial.set_user_attr(metric_name, metric_value)
            return score
        except Exception as e:
            logger.error(f"Trial failed: {str(e)}")
            return 0.0

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(n_startup_trials=20, prior_weight=0.4, seed=42)
    )
    study.optimize(objective, n_trials=n_trials, timeout=10000, show_progress_bar=True)
    best_params = study.best_trial.params
    logger.info(f"Best hyperparameters: {best_params}")
    best_params.update(base_params)
    return best_params

def hypertune_mlp(experiment_name):
    """
    Run hyperparameter tuning and final training for the MLP model with MLflow tracking.
    
    Returns:
        tuple: (best hyperparameters, final metrics)
    """
    try:
        with mlflow.start_run(run_name=f"mlp_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            mlflow.set_tags({
                "model_type": "mlp",
                "training_mode": "global",
                "cpu_only": True
            })
            X_train_scaled, X_test_scaled, X_eval_scaled, _ = preprocess_data(X_train, X_test, X_eval)
            hyperparameter_space = load_hyperparameter_space()
            best_params = optimize_hyperparameters(X_train_scaled, y_train, X_test_scaled, y_test, X_eval_scaled, y_eval, hyperparameter_space)
            logger.info("Training final MLP model with best hyperparameters")
            model, metrics = train_model(X_train_scaled, y_train, X_test_scaled, y_test, X_eval_scaled, y_eval, best_params.copy())
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            for param_name, param_value in best_params.items():
                mlflow.log_param(param_name, param_value)
            input_example = X_eval_scaled[:5].copy() if hasattr(X_eval_scaled, 'iloc') else X_eval_scaled[:5].copy()
            
            # Identify and convert integer columns to float64 to prevent schema enforcement errors
            if hasattr(input_example, 'dtypes'):
                for col in input_example.columns:
                    if input_example[col].dtype.kind == 'i':
                        logger.info(f"Converting integer column '{col}' to float64 to handle potential missing values")
                        input_example[col] = input_example[col].astype('float64')
            
            signature = mlflow.models.infer_signature(
                input_example,
                model.predict(input_example)
            )
            mlflow.keras.log_model(
                model,
                artifact_path="model",
                pip_requirements=pip_requirements,
                registered_model_name=f"mlp_{datetime.now().strftime('%Y%m%d_%H%M')}",
                signature=signature
            )
            mlflow.end_run()
            return best_params, metrics
    except Exception as e:
        logger.error(f"Error during hypertuning: {str(e)}")
        return None, None

def log_to_mlflow(model, metrics, params, experiment_name):
    """
    Log the final MLP model, its metrics, and parameters to MLflow.
    
    Returns:
        str: Run ID.
    """
    try:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=f"mlp_final_{datetime.now().strftime('%Y%m%d_%H%M')}") as run:
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            # Create a dummy input example based on feature dimension
            sample_input = np.zeros((1, model.shape[1]))
            signature = mlflow.models.infer_signature(sample_input, model.predict(sample_input))
            mlflow.keras.log_model(
                model,
                artifact_path="model",
                pip_requirements=pip_requirements,
                registered_model_name=f"mlp_{datetime.now().strftime('%Y%m%d_%H%M')}",
                signature=signature
            )
            mlflow.end_run()
            run_id = run.info.run_id
            logger.info(f"MLflow run ID: {run_id}")
            return run_id
    except Exception as e:
        logger.error(f"Error logging to MLflow: {str(e)}")
        return None

def train_with_precision_target(X_train, y_train, X_test, y_test, X_eval, y_eval):
    """
    Train the MLP model over multiple seeds to achieve a target precision.
    
    Returns:
        tuple: (best model, best precision, best recall, best hyperparameters)
    """
    best_precision = 0.0
    best_params = {}
    best_seed = None
    best_model = None
    target_precision = 0.45
    for seed in range(1, 21):
        logger.info(f"Trying seed: {seed}")
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)
        try:
            dataloader = DataLoader()
            X_train, y_train, X_test, y_test, X_eval, y_eval = dataloader.load_data()
            X_train_scaled, X_test_scaled, X_eval_scaled, _ = preprocess_data(X_train, X_test, X_eval)
            hyperparameter_space = load_hyperparameter_space()
            params = optimize_hyperparameters(X_train_scaled, y_train, X_test_scaled, y_test, X_eval_scaled, y_eval, hyperparameter_space)
            model, metrics = train_model(X_train_scaled, y_train, X_test_scaled, y_test, X_eval_scaled, y_eval, params.copy())
            precision = metrics.get('precision', 0.0)
            if precision > best_precision:
                best_precision = precision
                best_params = params.copy()
                best_seed = seed
                best_model = model
                logger.info(f"New best precision: {precision:.4f} with seed {seed}")
            if precision >= target_precision:
                logger.info(f"Target precision reached with seed {seed}")
                return best_model, precision, metrics.get('recall', 0.0), best_params
        except Exception as e:
            logger.error(f"Error with seed {seed}: {str(e)}")
            continue
        tf.keras.backend.clear_session()
    logger.info(f"Best precision achieved: {best_precision:.4f} with seed {best_seed}")
    return best_model, best_precision, metrics.get('recall', 0.0), best_params

def main():
    """
    Main execution function for MLP model training.
    """
    try:
        logger.info("Starting MLP model training")
        global X_train, y_train, X_test, y_test, X_eval, y_eval
        dataloader = DataLoader()
        X_train, y_train, X_test, y_test, X_eval, y_eval = dataloader.load_data()
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Testing data shape: {X_test.shape}")
        logger.info(f"Evaluation data shape: {X_eval.shape}")
        logger.info(f"Positive class ratio (Train): {np.mean(y_train):.3f}")
        hyperparameter_space = load_hyperparameter_space()
        X_train_scaled, X_test_scaled, X_eval_scaled, _ = preprocess_data(X_train, X_test, X_eval)
        best_params, metrics = hypertune_mlp(experiment_name)
        logger.info(f"Hypertuning completed with hyperparameters: {best_params}")
        logger.info(f"Hypertuning metrics: {metrics}")
        # Optional seed-based fine-tuning for improved precision
        model, precision, recall, final_params = train_with_precision_target(X_train, y_train, X_test, y_test, X_eval, y_eval)
        if model is not None:
            log_to_mlflow(model, metrics, final_params, experiment_name)
            model_path = f"models/StackedEnsemble/base/neural/mlp_model_{datetime.now().strftime('%Y%m%d_%H%M')}.h5"
            save_model(model, model_path, metrics.get('threshold', 0.5))
            logger.info(f"Final model trained with precision: {precision:.4f}, recall: {recall:.4f}")
        else:
            logger.error("MLP model training failed.")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
=======

def save_model(model, path, threshold=None):
    """Save MLP model and threshold to specified path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Save keras model
        model_path = str(path)
        model.save(model_path)
        
        # Save scaler for preprocessing
        scaler_path = path.parent / "scaler.joblib"
        joblib.dump(scaler, scaler_path)
        
        # Save threshold and metadata
        if threshold:
            metadata_path = path.parent / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump({
                    'threshold': threshold,
                    'model_type': 'mlp',
                    'input_shape': model.shape[1:],
                    'output_shape': model.output_shape[1:],
                    'layers_info': [
                        {
                            'name': layer.name,
                            'type': layer.__class__.__name__,
                            'units': getattr(layer, 'units', None)
                        } for layer in model.layers
                    ]
                }, f, indent=2)
                
        logger.info(f"Model saved to {path}")
        
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise


def load_model(path):
    """Load MLP model from specified path."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No model file found at {path}")
        
    try:
        # Load Keras model
        model = keras.models.load_model(str(path))
        
        # Load scaler
        scaler_path = path.parent / "scaler.joblib"
        if scaler_path.exists():
            global scaler
            scaler = joblib.load(scaler_path)
        
        # Load threshold
        metadata_path = path.parent / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                data = json.load(f)
                threshold = data.get('threshold', 0.5)
        else:
            threshold = 0.5
            
        logger.info(f"Model loaded from {path}")
        return model, threshold
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


class OptunaPruningCallback(tf.keras.callbacks.Callback):
    """Custom callback for Optuna pruning integration with Keras."""
    def __init__(self, trial, monitor="val_auc"):
        super().__init__()
        self.trial = trial
        self.monitor = monitor
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_score = logs.get(self.monitor)
        if current_score is None:
            return
        
        # Report value for pruning
        self.trial.report(current_score, step=epoch)
        
        # Handle pruning based on the intermediate value
        if self.trial.should_prune():
            raise optuna.TrialPruned()


def objective(trial):
    """Optuna objective function for hyperparameter optimization."""
    try:
        params = {}
        
        # Add hyperparameters from config
        hyperparameter_space = load_hyperparameter_space()
        for param_name, param_config in hyperparameter_space.items():
            if param_config['type'] == 'float':
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config['low'], 
                    param_config['high'],
                    log=param_config.get('log', False)
                )
            elif param_config['type'] == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config['low'],
                    param_config['high']
                )
            elif param_config['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config['choices']
                )
        # Create a pruning callback for this trial
        pruning_callback = OptunaPruningCallback(trial)
        
        # Extract parameters for callbacks to avoid passing them to model creation
        epochs = params.pop('epochs', 100)
        batch_size = params.pop('batch_size', 64)
        patience = params.pop('patience', 20)
        
        # Put back parameters needed for model training
        params['epochs'] = epochs
        params['batch_size'] = batch_size
        params['patience'] = patience
        
        # Train model with pruning callback
        model, metrics = train_model(
            params.copy()
        )
        
        recall = metrics.get('recall', 0.0)
        precision = metrics.get('precision', 0.0)
        
        # Optimize for precision while maintaining minimum recall
        score = precision if recall >= min_recall else 0.0
        
        logger.info(f"Trial {trial.number}:")
        logger.info(f"  Params: {params}")
        logger.info(f"  Score: {score}")
        
        # Save trial attributes
        for metric_name, metric_value in metrics.items():
            trial.set_user_attr(metric_name, metric_value)
            
        # Clean up Keras/TF memory
        tf.keras.backend.clear_session()
        
        return score
    except optuna.TrialPruned:
        # Handle pruned trials gracefully
        logger.info(f"Trial {trial.number} pruned")
        tf.keras.backend.clear_session()
        return 0.0
    except Exception as e:
        logger.error(f"Error in trial {trial.number}: {str(e)}")
        tf.keras.backend.clear_session()
        raise


def hypertune_mlp(experiment_name):
    """Run hyperparameter optimization with MLflow tracking."""
    try:
        # Create study
        study = optuna.create_study(
            study_name=f"mlp_optimization_{datetime.now().strftime('%Y%m%d_%H%M')}",
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=20,
                prior_weight=0.4,
                seed=int(time.time())  # Dynamic seed
            )
            # pruner=optuna.pruners.HyperbandPruner(
            #     min_resource=3,         # Min epochs before pruning
            #     max_resource=50,        # Max epochs to consider
            #     reduction_factor=3      # Controls aggressiveness
            # )
        )
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"mlp_optimization_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            # Set tags
            mlflow.set_tags({
                "model_type": "mlp_base",
                "optimization": "optuna",
                "cpu_only": True
            })
            
            # Optimize
            best_score = -float('inf')
            best_params = {}
            
            def callback(study, trial):
                nonlocal best_score
                nonlocal best_params
                logger.info(f"Current best score: {best_score:.4f}")
                if trial.value and trial.value > best_score:
                    best_score = trial.value
                    best_params = trial.params
                    logger.info(f"New best score found in trial {trial.number}: {best_score:.4f}")
                return best_score
            
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=18000,  # 5 hours timeout
                callbacks=[callback],
                show_progress_bar=True,
                gc_after_trial=True
            )
            
            # Log best trial info
            logger.info(f"Best trial value: {best_score}")
            logger.info(f"Best Parameters: {best_params}")
            
            # Add base parameters
            best_params.update(base_params)
            
            # Train final model with best parameters
            logger.info("Training final model with best parameters")
            final_model, final_metrics = train_model(
                X_train_scaled, y_train,
                X_test_scaled, y_test,
                X_eval_scaled, y_eval,
                best_params.copy()
            )
            
            # Log model to MLflow
            mlflow.keras.log_model(final_model, "model")
            mlflow.log_metrics(final_metrics)
            # Log scaler to MLflow
            scaler_path = "scaler.joblib"
            joblib.dump(scaler, scaler_path)
            mlflow.log_artifact(scaler_path, "preprocessing")
            
            # Log additional metadata about the scaler
            scaler_metadata = {
                "feature_count": scaler.n_features_in_,
                "scale_mean": scaler.mean_.mean(),
                "scale_var": scaler.var_.mean()
            }
            mlflow.log_params({"scaler_type": "StandardScaler"})
            mlflow.log_metrics(scaler_metadata)
            
            # Clean up temporary file
            if os.path.exists(scaler_path):
                os.remove(scaler_path)
            
            logger.info(f"Training completed with precision: {final_metrics['precision']:.4f}")
            return final_metrics['precision'], best_params
            
    except Exception as e:
        logger.error(f"Error in hyperparameter optimization: {str(e)}")
        raise

def log_to_mlflow(model, precision, recall, params, experiment_name):
    """Log model, metrics and parameters to MLflow."""
    from utils.create_evaluation_set import setup_mlflow_tracking
    
    mlruns_dir = setup_mlflow_tracking(experiment_name)
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"mlp_base_{datetime.now().strftime('%Y%m%d_%H%M')}") as run:
        
        # Log parameters
        mlflow.log_params(params)
        
        # Log metrics
        mlflow.log_metrics({
            "precision": precision,
            "recall": recall
        })
        
        # Register model with timestamp
        model_name = f"mlp_base_{datetime.now().strftime('%Y%m%d_%H%M')}"
        
        # Create proper input example for model signature
        input_example = X_train_scaled[0:1]
        
        # Get prediction for signature
        pred = model.predict(input_example, verbose=0)
        
        signature = mlflow.models.infer_signature(
            model_input=input_example,
            model_output=pred
        )
        
        # Log Keras model
        mlflow.keras.log_model(
            keras_model=model,
            artifact_path="model",
            registered_model_name=model_name,
            signature=signature
        )
        
        # Log run ID
        run_id = run.info.run_id
        logger.info(f"Run ID: {run_id}")
        return run_id

def train_with_precision_target(X_train, y_train, X_test, y_test, X_eval, y_eval, logger):
    """Train MLP model with target precision threshold, trying different seeds."""
    
    precision = 0
    best_precision = 0
    best_recall = 0
    best_params = None
    best_seed = 0
    best_model = None
    
    # Base parameters from previous optimization - to be updated after hypertuning
    params = {
        'hidden_layers': 3,
        'neurons_per_layer': 128,
        'dropout_rate': 0.3,
        'activation': 'relu',
        'learning_rate': 0.0005,
        'l1_regularization': 0.0001,
        'l2_regularization': 0.001,
        'batch_size': 64,
        'epochs': 100,
        'patience': 20,
        'class_weight_multiplier': 3.0,
    }
    
    target_precision = 0.45  # Target precision threshold
    
    while best_precision < target_precision:
        for random_seed in range(1, 101):  # Try up to 100 different seeds
            logger.info(f"Using random seed: {random_seed}")
            
            # Set all random seeds
            os.environ['PYTHONHASHSEED'] = str(random_seed)
            np.random.seed(random_seed)
            random.seed(random_seed)
            tf.random.set_seed(random_seed)
            
            try:
                # Create and train model
                tf.keras.backend.clear_session()  # Clear TF session
                
                params_copy = params.copy()
                params_copy['random_seed'] = random_seed
                
                model, metrics = train_model(
                    X_train, y_train,
                    X_test, y_test,
                    X_eval, y_eval,
                    params_copy
                )
                
                precision = metrics['precision']
                recall = metrics['recall']

                # Update best model if precision improved
                if precision > best_precision:
                    best_precision = precision
                    best_recall = recall
                    best_params = params_copy.copy()
                    best_seed = random_seed
                    
                    # Save the best model
                    best_model = model
                    logger.info(f"New best precision: {precision:.4f} with seed {best_seed}")
                
                # Check if target precision reached
                if precision >= target_precision:
                    logger.info(f"Target precision achieved: {precision:.4f}")
                    return best_model, precision, recall, best_params
                
                logger.info(
                    f"Current precision: {precision:.4f}, "
                    f"target: {target_precision:.4f}, highest: {best_precision:.4f}, "
                    f"best seed: {best_seed}"
                )
                
            except Exception as e:
                logger.error(f"Error training with seed {random_seed}: {str(e)}")
                continue
            
            # Clear model to free memory
            tf.keras.backend.clear_session()
            model = None
        
        # If target not reached after all seeds, return best model
        if precision < target_precision:
            logger.info(f"Target precision not reached, using best seed: {best_seed}")
            return best_model, best_precision, best_recall, best_params
            
    return best_model, best_precision, best_recall, best_params

def main():
    """Main execution function."""
    # First, set up MLflow tracking
    mlruns_dir = setup_mlflow_tracking(experiment_name)
    logger.info(f"MLflow tracking set up at {mlruns_dir}")
    
    # Run multiple hypertuning sessions
    best_precision = 0
    best_params = {}

    # First, do hypertuning
    for i in range(3):  # Fewer runs for MLP as they take longer
        logger.info(f"Starting hypertuning run {i+1}/3")
        precision, params = hypertune_mlp(experiment_name)
        
        logger.info(f"Run {i+1} completed with precision: {precision:.4f}")
        
        # Track the best run
        if precision > best_precision:
            best_precision = precision
            best_params = params
    
    logger.info(f"Best precision from hypertuning: {best_precision:.4f}")
    logger.info(f"Best parameters: {best_params}")
    
    # Optional: Fine-tune with different seeds for further improvement
    logger.info("Starting seed-based fine-tuning for higher precision")
    model, precision, recall, final_params = train_with_precision_target(
        X_train_scaled, y_train,
        X_test_scaled, y_test,
        X_eval_scaled, y_eval,
        logger
    )
    
    # Log final model to MLflow
    if model is not None:
        log_to_mlflow(model, precision, recall, final_params, experiment_name)
        
    logger.info(f"Final model precision: {precision:.4f}, recall: {recall:.4f}")

>>>>>>> 3798e304ba09a95ae05e21747b9f93b4e52eb5fd

if __name__ == "__main__":
    main() 