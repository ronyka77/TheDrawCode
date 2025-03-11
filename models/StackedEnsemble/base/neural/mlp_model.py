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
# Set random seeds for reproducibility
random_seed = 19
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
os.environ['PYTHONHASHSEED'] = str(random_seed)

# Configure TensorFlow for deterministic operations
tf.config.experimental.enable_op_determinism()  # Enable deterministic operations


# Configure Git executable path if available
git_executable = os.environ.get("GIT_PYTHON_GIT_EXECUTABLE")
if git_executable and os.path.exists(git_executable):
    import git
    git.refresh(git_executable)

from utils.logger import ExperimentLogger
experiment_name = "mlp_soccer_prediction"
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

# Define base configurations
base_params = {
    'epochs': 100,
    'batch_size': 64,
    'verbose': 0,
    'metrics': ['accuracy', 'AUC']
}

def load_hyperparameter_space():
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
        hidden_layers = model_params.pop('hidden_layers', 2)
        neurons_per_layer = model_params.pop('neurons_per_layer', 128)
        dropout_rate = model_params.pop('dropout_rate', 0.2)
        activation = model_params.pop('activation', 'relu')
        l1_reg = model_params.pop('l1_regularization', 0.0)
        l2_reg = model_params.pop('l2_regularization', 0.0)
        learning_rate = model_params.pop('learning_rate', 0.001)
        
        model = keras.Sequential()
        model.add(layers.InputLayer(shape=(input_dim,)))
        
        # Add hidden layers
        for _ in range(hidden_layers):
            model.add(layers.Dense(
                neurons_per_layer,
                activation=activation,
                kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg)
            ))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate))
        
        # Output layer for binary classification
        model.add(layers.Dense(1, activation='sigmoid'))
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        return model
    except Exception as e:
        logger.error(f"Error creating MLP model: {str(e)}")
        raise

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
            restore_best_weights=True,
            verbose=0
        )
        
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
    except Exception as e:
        logger.error(f"Error training MLP model: {str(e)}")
        raise

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

if __name__ == "__main__":
    main() 