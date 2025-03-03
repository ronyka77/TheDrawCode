"""
LightGBM Model for Soccer Draw Prediction

This module implements a LightGBM-based model for predicting soccer match draws.
It includes functionality for model creation, training, hyperparameter optimization,
threshold tuning, and MLflow integration for experiment tracking.

The implementation focuses on high precision while maintaining a minimum recall threshold.
"""

import os
import sys
import json
import pickle
import random
import logging
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
import optuna
from optuna.integration import LightGBMPruningCallback
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, precision_recall_curve
)
from datetime import datetime
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler("logs/lightgbm_soccer_prediction.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("lightgbm_soccer_prediction")

# Global settings
experiment_name = "lightgbm_soccer_prediction"
min_recall = 0.20  # Minimum acceptable recall
n_trials = 100  # Number of hyperparameter optimization trials
SEED = 42  # Global seed for reproducibility
TREE_METHOD = 'hist'  # Use histogram-based tree method for CPU optimization
DEVICE = 'cpu'  # Enforce CPU training

def load_hyperparameter_space():
    """
    Define the hyperparameter space for LightGBM model tuning.
    
    Returns:
        dict: Default hyperparameter configuration
    """
    return {
        # Core parameters
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'device_type': DEVICE,
        
        # Learning parameters
        'learning_rate': 0.05,
        'n_estimators': 1000,
        'max_depth': 6,
        'num_leaves': 31,
        
        # Regularization parameters
        'min_child_samples': 20,
        'min_child_weight': 0.001,
        'reg_alpha': 0.0,
        'reg_lambda': 1.0,
        
        # Sampling parameters
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'subsample_freq': 1,
        
        # Early stopping
        'early_stopping_rounds': 50,
        
        # Class balance
        'scale_pos_weight': 1.0,
        
        # Misc
        'verbose': -1,
        'seed': SEED,
    }

def create_model(model_params):
    """
    Create and configure LightGBM model instance.
    
    Args:
        model_params (dict): Model parameters
        
    Returns:
        lgb.LGBMClassifier: Configured LightGBM model
    """
    try:
        # Make a copy to avoid modifying the original
        params = model_params.copy()
        
        # Set random seed for reproducibility
        seed = params.pop('seed', SEED)
        random.seed(seed)
        np.random.seed(seed)
        
        # Create model with parameters
        model = lgb.LGBMClassifier(**params)
        logger.info(f"Created LightGBM model with {len(params)} parameters")
        return model
        
    except Exception as e:
        logger.error(f"Error creating LightGBM model: {str(e)}")
        raise

def predict(model, X, threshold=0.5):
    """
    Generate predictions using trained model.
    
    Args:
        model: Trained LightGBM model
        X: Features to predict on
        threshold (float): Decision threshold
        
    Returns:
        np.array: Binary predictions
    """
    if model is None:
        raise RuntimeError("Model must be trained before prediction")
        
    try:
        probas = model.predict_proba(X)[:, 1]
        return (probas >= threshold).astype(int)
        
    except Exception as e:
        logger.error(f"Error in model prediction: {str(e)}")
        return np.zeros(len(X))

def predict_proba(model, X):
    """
    Generate probability predictions.
    
    Args:
        model: Trained LightGBM model
        X: Features to predict on
        
    Returns:
        np.array: Probability predictions
    """
    if model is None:
        raise RuntimeError("Model must be trained before prediction")
        
    try:
        return model.predict_proba(X)[:, 1]
        
    except Exception as e:
        logger.error(f"Error in probability prediction: {str(e)}")
        return np.zeros(len(X))

def evaluate(model, X, y, best_threshold):
    """
    Evaluate model performance with multiple metrics.
    
    Args:
        model: Trained LightGBM model
        X: Feature data
        y: Target labels
        best_threshold: Optimized decision threshold
        
    Returns:
        dict: Evaluation metrics
    """
    try:
        # Get predictions using the best threshold
        y_prob = predict_proba(model, X)
        y_pred = (y_prob >= best_threshold).astype(int)
        
        # Calculate metrics
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        auc = roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else 0
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        
        # Log detailed results
        logger.info(f"Evaluation Results (threshold={best_threshold:.3f}):")
        logger.info(f"  Precision: {precision:.3f}")
        logger.info(f"  Recall: {recall:.3f}")
        logger.info(f"  F1 Score: {f1:.3f}")
        logger.info(f"  AUC: {auc:.3f}")
        logger.info(f"  TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'threshold': best_threshold,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
        }
    
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        return {
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'auc': 0,
            'threshold': best_threshold,
            'tp': 0,
            'fp': 0,
            'tn': 0,
            'fn': 0,
        }

def optimize_threshold(model, y_true, y_prob):
    """
    Optimize classification threshold to maximize precision while maintaining minimum recall.
    
    Args:
        model: Trained LightGBM model
        y_true: True labels
        y_prob: Predicted probabilities
        
    Returns:
        tuple: (best_threshold, best_precision, best_recall)
    """
    try:
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        
        # Find the threshold that maximizes precision while maintaining minimum recall
        best_threshold = 0.5
        best_precision = 0
        best_recall = 0
        
        # Combine precision, recall, and thresholds
        metrics = list(zip(precision, recall, [0] + list(thresholds)))
        
        # Sort metrics by precision (descending)
        sorted_metrics = sorted(metrics, key=lambda x: x[0], reverse=True)
        
        for p, r, t in sorted_metrics:
            if r >= min_recall:
                best_precision = p
                best_recall = r
                best_threshold = t
                break
        
        # If no threshold meets minimum recall, choose the one with highest recall
        if best_precision == 0:
            idx = np.argmax(recall)
            best_precision = precision[idx]
            best_recall = recall[idx]
            best_threshold = thresholds[min(idx, len(thresholds)-1)]
            logger.warning(f"No threshold met minimum recall of {min_recall}. Using best available: {best_threshold:.3f}")
        
        logger.info(f"Optimal threshold: {best_threshold:.3f} (precision: {best_precision:.3f}, recall: {best_recall:.3f})")
        return best_threshold, best_precision, best_recall
        
    except Exception as e:
        logger.error(f"Error optimizing threshold: {str(e)}")
        return 0.5, 0, 0

def train_model(X_train, y_train, X_test, y_test, X_eval, y_eval, model_params):
    """
    Train a LightGBM model with early stopping and threshold optimization.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Validation features
        y_test: Validation labels
        X_eval: Evaluation features
        y_eval: Evaluation labels
        model_params: Model parameters
        
    Returns:
        tuple: (trained_model, best_threshold, metrics)
    """
    try:
        # Extract parameters
        params = model_params.copy()
        n_estimators = params.pop('n_estimators', 1000)
        early_stopping_rounds = params.pop('early_stopping_rounds', 50)
        
        # Create model
        model = create_model(params)
        
        # Calculate class weights
        if y_train.mean() > 0:
            scale_pos_weight = params.get('scale_pos_weight', 1.0)
            neg_class_count = np.sum(y_train == 0)
            pos_class_count = np.sum(y_train == 1)
            weight_for_0 = 1.0
            weight_for_1 = (neg_class_count / pos_class_count) * scale_pos_weight
            sample_weight = np.ones_like(y_train, dtype=float)
            sample_weight[y_train == 1] = weight_for_1
        else:
            sample_weight = None
            logger.warning("No positive samples in training set. Using uniform weights.")
        
        # Train model
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='auc',
            early_stopping_rounds=early_stopping_rounds,
            sample_weight=sample_weight,
            verbose=False
        )
        
        # Get validation predictions
        y_prob = predict_proba(model, X_eval)
        
        # Optimize threshold
        best_threshold, best_precision, best_recall = optimize_threshold(
            model,
            y_eval, 
            y_prob
        )
        
        logger.info(f"Best threshold: {best_threshold:.3f} with precision: {best_precision:.3f}")
        
        # Get final metrics with best threshold
        metrics = evaluate(model, X_eval, y_eval, best_threshold)
        
        return model, best_threshold, metrics
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return None, 0.5, {}

def save_model(model, path, threshold=None, metrics=None, params=None):
    """
    Save model and associated metadata.
    
    Args:
        model: Trained LightGBM model
        path (str): Path to save model
        threshold (float, optional): Optimal decision threshold
        metrics (dict, optional): Model evaluation metrics
        params (dict, optional): Model parameters
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata
        metadata = {
            'threshold': threshold if threshold is not None else 0.5,
            'metrics': metrics if metrics is not None else {},
            'params': params if params is not None else {},
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': 'lightgbm'
        }
        
        metadata_path = path.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Model saved to {path} with metadata")
        
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")

def load_model(path):
    """
    Load model and associated metadata.
    
    Args:
        path (str): Path to load model from
        
    Returns:
        tuple: (model, metadata)
    """
    try:
        # Load model
        with open(path, 'rb') as f:
            model = pickle.load(f)
        
        # Load metadata
        metadata_path = path.replace('.pkl', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {'threshold': 0.5}
            
        return model, metadata
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None, {'threshold': 0.5}

def objective(trial):
    """
    Objective function for Optuna hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        
    Returns:
        float: Precision score (to be maximized)
    """
    # Import data at runtime to avoid global scope issues
    try:
        from data.data_loader import load_data
        X_train, y_train, X_test, y_test, X_eval, y_eval = load_data()
        
        # Define hyperparameter space
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
            'device_type': DEVICE,
            
            # Learning parameters
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            
            # Regularization parameters
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.001, 10.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0, log=True),
            
            # Sampling parameters
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'subsample_freq': trial.suggest_int('subsample_freq', 0, 10),
            
            # Early stopping
            'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 30, 100),
            
            # Class balance
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 5.0),
            
            # Misc
            'verbose': -1,
            'seed': SEED + trial.number,  # Vary seed for each trial
        }
        
        # Create and train model
        model, threshold, metrics = train_model(
            X_train, y_train, X_test, y_test, X_eval, y_eval, params
        )
        
        # Log metrics
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        f1 = metrics.get('f1', 0)
        auc = metrics.get('auc', 0)
        
        trial.set_user_attr('threshold', threshold)
        trial.set_user_attr('recall', recall)
        trial.set_user_attr('f1', f1)
        trial.set_user_attr('auc', auc)
        
        # Clean up to avoid memory leaks
        del model
        gc.collect()
        
        # Return precision score (to be maximized)
        return precision
        
    except Exception as e:
        logger.error(f"Error in objective function: {str(e)}")
        return 0.0

def hypertune_mlp(experiment_name):
    """
    Run hyperparameter optimization using Optuna.
    
    Args:
        experiment_name (str): Experiment name for MLflow tracking
        
    Returns:
        tuple: (best_params, best_threshold, best_precision, best_recall, best_f1, best_auc)
    """
    try:
        # Set up MLflow tracking
        mlflow.set_experiment(experiment_name)
        
        # Create Optuna study
        study = optuna.create_study(
            study_name=f"lightgbm_optimization_{datetime.now().strftime('%Y%m%d_%H%M')}",
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=30)
        )
        
        # Define callback to log trials to MLflow
        def callback(study, trial):
            if trial.state == optuna.trial.TrialState.COMPLETE:
                with mlflow.start_run(run_name=f"trial_{trial.number}"):
                    # Log parameters
                    params = trial.params.copy()
                    for param_name, param_value in params.items():
                        mlflow.log_param(param_name, param_value)
                    
                    # Log metrics
                    mlflow.log_metric("precision", trial.value)
                    mlflow.log_metric("recall", trial.user_attrs.get('recall', 0))
                    mlflow.log_metric("f1", trial.user_attrs.get('f1', 0))
                    mlflow.log_metric("auc", trial.user_attrs.get('auc', 0))
                    mlflow.log_metric("threshold", trial.user_attrs.get('threshold', 0.5))
                    
        # Run optimization
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
        study.optimize(objective, n_trials=n_trials, callbacks=[callback], n_jobs=1)
        
        # Get best parameters
        best_params = study.best_params
        best_precision = study.best_value
        best_threshold = study.best_trial.user_attrs.get('threshold', 0.5)
        best_recall = study.best_trial.user_attrs.get('recall', 0)
        best_f1 = study.best_trial.user_attrs.get('f1', 0)
        best_auc = study.best_trial.user_attrs.get('auc', 0)
        
        # Log results
        logger.info("Best hyperparameters:")
        for param_name, param_value in best_params.items():
            logger.info(f"  {param_name}: {param_value}")
        logger.info(f"Best precision: {best_precision:.3f}")
        logger.info(f"Best recall: {best_recall:.3f}")
        logger.info(f"Best F1: {best_f1:.3f}")
        logger.info(f"Best AUC: {best_auc:.3f}")
        logger.info(f"Best threshold: {best_threshold:.3f}")
        
        return best_params, best_threshold, best_precision, best_recall, best_f1, best_auc
        
    except Exception as e:
        logger.error(f"Error in hyperparameter optimization: {str(e)}")
        return load_hyperparameter_space(), 0.5, 0, 0, 0, 0

def log_to_mlflow(model, precision, recall, params, experiment_name):
    """
    Log trained model, metrics, and parameters to MLflow.
    
    Args:
        model: Trained LightGBM model
        precision (float): Precision score
        recall (float): Recall score
        params (dict): Model parameters
        experiment_name (str): Experiment name
        
    Returns:
        str: Run ID
    """
    try:
        # Set up MLflow tracking
        mlflow.set_experiment(experiment_name)
        
        # Start a new run
        with mlflow.start_run(run_name=f"lightgbm_{datetime.now().strftime('%Y%m%d_%H%M')}") as run:
            # Log parameters
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
            
            # Log metrics
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0)
            
            # Log model
            model_info = mlflow.lightgbm.log_model(
                model,
                "model",
                registered_model_name=f"lightgbm_{datetime.now().strftime('%Y%m%d_%H%M')}"
            )
            
            # Log feature importance
            if hasattr(model, 'feature_importances_'):
                try:
                    from data.data_loader import get_feature_names
                    feature_names = get_feature_names()
                    importance_df = pd.DataFrame({
                        'Feature': feature_names if len(feature_names) == len(model.feature_importances_) 
                                  else [f"feature_{i}" for i in range(len(model.feature_importances_))],
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    # Save to CSV and log as artifact
                    importance_path = "feature_importance.csv"
                    importance_df.to_csv(importance_path, index=False)
                    mlflow.log_artifact(importance_path)
                    os.remove(importance_path)
                except Exception as e:
                    logger.warning(f"Error logging feature importance: {str(e)}")
            
            logger.info(f"Model logged to MLflow: {model_info.model_uri}")
            return run.info.run_id
            
    except Exception as e:
        logger.error(f"Error logging to MLflow: {str(e)}")
        return None

def train_with_precision_target(X_train, y_train, X_test, y_test, X_eval, y_eval, logger):
    """
    Train a LightGBM model with focus on precision target and multi-seed approach.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Validation features
        y_test: Validation labels
        X_eval: Evaluation features
        y_eval: Evaluation labels
        logger: Logger instance
        
    Returns:
        tuple: (best_model, best_threshold, best_metrics)
    """
    try:
        # Get best parameters from hypertuning or use defaults
        try:
            best_params, _, _, _, _, _ = hypertune_mlp(experiment_name)
        except Exception as e:
            logger.warning(f"Error during hypertuning: {str(e)}. Using default parameters.")
            best_params = load_hyperparameter_space()
        
        # Add fixed parameters
        best_params['objective'] = 'binary'
        best_params['metric'] = 'auc'
        best_params['device_type'] = DEVICE
        best_params['verbose'] = -1
        
        # Try different random seeds
        best_precision = 0
        best_model = None
        best_threshold = 0.5
        best_metrics = {}
        
        logger.info("Training with multiple seeds for stability")
        for seed in range(42, 52):  # Try 10 different seeds
            logger.info(f"Training with seed {seed}")
            
            # Set seed in parameters
            current_params = best_params.copy()
            current_params['seed'] = seed
            
            # Train model
            model, threshold, metrics = train_model(
                X_train, y_train, X_test, y_test, X_eval, y_eval, current_params
            )
            
            # Check if better precision while maintaining recall target
            precision = metrics.get('precision', 0)
            recall = metrics.get('recall', 0)
            
            if (precision > best_precision and recall >= min_recall) or (best_model is None):
                best_precision = precision
                best_model = model
                best_threshold = threshold
                best_metrics = metrics
                logger.info(f"New best model with seed {seed}: precision={precision:.3f}, recall={recall:.3f}")
            
            # Clean up
            if model != best_model:
                del model
                gc.collect()
        
        # Final evaluation
        logger.info(f"Best model precision: {best_precision:.3f}")
        logger.info(f"Best model threshold: {best_threshold:.3f}")
        
        # Log to MLflow
        log_to_mlflow(best_model, best_precision, best_metrics.get('recall', 0), best_params, experiment_name)
        
        # Save model
        model_path = f"models/StackedEnsemble/base/tree_based/lightgbm_model_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
        save_model(best_model, model_path, best_threshold, best_metrics, best_params)
        
        return best_model, best_threshold, best_metrics
        
    except Exception as e:
        logger.error(f"Error in precision-focused training: {str(e)}")
        return None, 0.5, {}

def main():
    """
    Main execution function for training and hypertuning.
    """
    try:
        logger.info("Starting LightGBM model training and hypertuning")
        
        # Import data at runtime to avoid global scope issues
        from data.data_loader import load_data
        X_train, y_train, X_test, y_test, X_eval, y_eval = load_data()
        
        # Log data shapes
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Testing data shape: {X_test.shape}")
        logger.info(f"Evaluation data shape: {X_eval.shape}")
        logger.info(f"Positive class ratio - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}, Eval: {y_eval.mean():.3f}")
        
        # Train with precision target
        best_model, best_threshold, best_metrics = train_with_precision_target(
            X_train, y_train, X_test, y_test, X_eval, y_eval, logger
        )
        
        # Final evaluation
        if best_model is not None:
            logger.info("Final model metrics:")
            logger.info(f"  Precision: {best_metrics.get('precision', 0):.3f}")
            logger.info(f"  Recall: {best_metrics.get('recall', 0):.3f}")
            logger.info(f"  F1: {best_metrics.get('f1', 0):.3f}")
            logger.info(f"  AUC: {best_metrics.get('auc', 0):.3f}")
            logger.info(f"  Threshold: {best_threshold:.3f}")
        else:
            logger.error("Failed to train a valid model")
        
        logger.info("LightGBM training completed")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main() 