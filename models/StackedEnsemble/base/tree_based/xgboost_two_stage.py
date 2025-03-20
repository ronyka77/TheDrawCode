"""
Two-Stage XGBoost Model for Soccer Draw Prediction

This module implements a two-stage XGBoost model approach for soccer match prediction,
targeting 50% precision and 20% recall. The first stage is recall-optimized to capture
potential draws, while the second stage refines predictions to improve precision.

The implementation follows the Soccer Prediction Project v2.1 Development Guidelines,
ensuring CPU-only training and proper logging through MLflow.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import mlflow
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score, precision_recall_curve,
    auc, confusion_matrix, classification_report
)
from datetime import datetime
import time
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union, Optional

# Add project root to Python path
try:
    project_root = Path(__file__).parent.parent.parent.parent.parent
    if not project_root.exists():
        # Handle network path by using raw string
        project_root = Path(r"\\".join(str(project_root).split("\\")))
    sys.path.append(str(project_root))
    print(f"Project root: {project_root}")
except Exception as e:
    print(f"Error setting project root path: {e}")
    # Fallback to current directory if path resolution fails
    sys.path.append(os.getcwd())
    print(f"Current directory: {os.getcwd()}")

# Import utilities
from utils.logger import ExperimentLogger
experiment_name = "two_stage_xgboost"
logger = ExperimentLogger(
    experiment_name=experiment_name,
    log_dir="logs/two_stage_model"
)

from utils.create_evaluation_set import setup_mlflow_tracking
mlruns_dir = setup_mlflow_tracking(experiment_name)

# Import shared utility functions
from models.StackedEnsemble.shared.data_loader import DataLoader
from models.StackedEnsemble.shared.hypertuner_utils import (
    predict, predict_proba, evaluate, calculate_feature_importance
)

# Global settings
SEED = 42
DEVICE = 'cpu'  # Enforce CPU-only training
min_recall = 0.20  # Minimum acceptable recall
target_precision = 0.50  # Target precision
target_stage1_recall = 0.40  # Target recall for stage 1

# Default parameter configurations for both stages
first_stage_params = {
    'objective': 'binary:logistic',
    'tree_method': 'hist',  # CPU-optimized
    'learning_rate': 0.1,
    'max_depth': 7,
    'min_child_weight': 2,
    'scale_pos_weight': 3,  # Adjusted for recall focus
    'subsample': 0.85,
    'colsample_bytree': 0.8,
    'eval_metric': ['auc', 'logloss', 'error']
}

second_stage_params = {
    'objective': 'binary:logistic',
    'tree_method': 'hist',  # CPU-optimized
    'learning_rate': 0.03,
    'max_depth': 5,
    'min_child_weight': 4,  # Higher to prevent overfitting
    'scale_pos_weight': 1,  # Balanced for precision
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'eval_metric': ['auc', 'logloss', 'error']
}

def prepare_data():
    """
    Load and prepare data for the two-stage model.
    
    Returns:
        tuple: (train_data, val_data, test_data) with features and target
    """
    try:
        
        # Create data loader and load data splits
        data_loader = DataLoader(experiment_name=experiment_name)
        X_train, y_train, X_test, y_test, X_val, y_val = data_loader.load_data()
        
        # Store in appropriate variables for the two-stage model
        train_data = X_train.copy()
        train_data['is_draw'] = y_train
        val_data = X_val.copy()
        val_data['is_draw'] = y_val
        test_data = X_test.copy()
        test_data['is_draw'] = y_test
        
        # Log data preparation completion
        logger.info({
            "train_samples": len(train_data),
            "validation_samples": len(val_data),
            "test_samples": len(test_data),
            "positive_train_ratio": train_data['is_draw'].mean()
        })
        
        return train_data, val_data, test_data
        
    except Exception as e:
        logger.error(f"Error in data preparation: {str(e)}")
        raise

def load_hyperparameter_space():
    """
    Define the hyperparameter space for XGBoost model tuning.
    Matches exactly the notebook implementation.
    
    Returns:
        dict: Hyperparameter space configuration
    """
    try:
        # Define hyperparameter space directly as in notebook
        hyperparameter_space = {
            'learning_rate': {
                'type': 'float',
                'low': 0.005,                  # Lower bound to allow more careful learning
                'high': 0.1,                   # Slightly wider range than current
                'log': True
            },
            'max_leaves': {
                'type': 'int',
                'low': 24,                     # Increase min leaves for better precision
                'high': 60                     # Expand upper bound for more complex trees
            },
            'max_depth': {
                'type': 'int',
                'low': 5,                      # Increase lower limit for more specificity
                'high': 8                      # Keep reasonable upper bound to avoid overfitting
            },
            'min_child_weight': {
                'type': 'int',
                'low': 100,                    # Increase to reduce noise from outliers
                'high': 300                    # Higher values help prevent overfitting
            },
            'colsample_bytree': {
                'type': 'float',
                'low': 0.6,                    # Increase lower bound for more stable feature selection
                'high': 0.85                   # Focusing on more consistent features
            },
            'subsample': {
                'type': 'float',
                'low': 0.5,                    # Keep current lower bound
                'high': 0.8                    # Decrease upper bound for more stable models
            },
            'alpha': {
                'type': 'float',
                'low': 5.0,                    # Significantly increase L1 regularization
                'high': 15.0,                  # Higher regularization generally improves precision
                'log': True
            },
            'lambda': {
                'type': 'float',
                'low': 5.0,                    # Increase lower bound for L2 regularization
                'high': 15.0,                  # Higher L2 regularization promotes precision
                'log': True
            },
            'gamma': {
                'type': 'float',
                'low': 0.1,                    # Increase to prevent weak splits
                'high': 0.3,                   # Higher values focus on more significant patterns
                'log': False
            },
            'early_stopping_rounds': {
                'type': 'int',
                'low': 300,                    # Increase patience for better convergence
                'high': 800                    # Higher values prevent premature stopping
            },
            'max_bin': {
                'type': 'int',
                'low': 400,                    # Increase for finer binning granularity
                'high': 1000                    # Higher bin count helps identify patterns more precisely
            }
        }
        return hyperparameter_space
    except Exception as e:
        logger.error(f"Error creating hyperparameter space: {str(e)}")
        return None

def create_model(model_params):
    """
    Create and configure XGBoost model instance.
    Matches the notebook implementation.
    
    Args:
        model_params (dict): Model parameters
        
    Returns:
        xgb.XGBClassifier: Configured XGBoost model
    """
    try:
        params = model_params.copy()
        
        # Create model
        model = xgb.XGBClassifier(**params)
        
        return model
        
    except Exception as e:
        logger.error(f"Error creating XGBoost model: {str(e)}")
        raise

def train_model(X_train, y_train, X_test, y_test, X_eval, y_eval, model_params):
    """
    Train a XGBoost model with early stopping and threshold optimization.
    Updated to match notebook implementation.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Validation features
        y_test: Validation labels
        X_eval: Evaluation features
        y_eval: Evaluation labels
        model_params: Model parameters
        
    Returns:
        tuple: (trained_model, metrics)
    """
    try:
        
        # Create model with remaining parameters
        model = create_model(model_params)
        
        # Create eval set for early stopping
        eval_set = [(X_test, y_test)]
        
        # Fit model with early stopping
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        # Get validation predictions
        best_threshold, metrics = optimize_threshold(
            model, X_eval, y_eval, min_recall=min_recall
        )
        
        return model, metrics
        
    except Exception as e:
        logger.error(f"Error training XGBoost model: {str(e)}")
        raise

def save_model(model, path, threshold=0.5):
    """
    Save XGBoost model to specified path.
    Updated to match notebook implementation using joblib.
    
    Args:
        model: Trained XGBoost model
        path: Path to save model
        threshold: Optimal decision threshold
    """
    if model is None:
        raise RuntimeError("No model to save")
        
    try:
        # Create directory if it doesn't exist
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(model, path)
        
        # Save threshold
        threshold_path = path.parent / "threshold.json"
        with open(threshold_path, 'w') as f:
            json.dump({
                'threshold': threshold,
                'model_type': 'xgboost',
                'params': model.get_params()
            }, f, indent=2)
            
        logger.info(f"Model saved to {path}")
        
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

def load_model(path):
    """
    Load XGBoost model from specified path.
    Updated to match notebook implementation using joblib.
    
    Args:
        path: Path to load model from
        
    Returns:
        tuple: (model, threshold)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No model file found at {path}")
        
    try:
        # Load model
        model = joblib.load(path)
        
        # Load threshold
        threshold_path = path.parent / "threshold.json"
        if threshold_path.exists():
            with open(threshold_path, 'r') as f:
                data = json.load(f)
                threshold = data.get('threshold', 0.5)
        else:
            threshold = 0.5
            
        logger.info(f"Model loaded from {path}")
        return model, threshold
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def optimize_hyperparameters(X_train, y_train, X_test, y_test, X_eval, y_eval, hyperparameter_space):
    """
    Run hyperparameter optimization with Optuna.
    Added to match notebook implementation.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Testing features
        y_test: Testing labels
        X_eval: Evaluation features
        y_eval: Evaluation labels
        hyperparameter_space: Hyperparameter space configuration
        
    Returns:
        dict: Best parameters
    """
    logger.info("Starting hyperparameter optimization")
    
    if not hyperparameter_space:
        hyperparameter_space = load_hyperparameter_space()
    
    best_score = 0.0
    
    def objective(trial):
        try:
            params = first_stage_params.copy()
            
            # Add hyperparameters from config
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
            
            # Train model and get metrics
            model, metrics = train_model(
                X_train, y_train,
                X_test, y_test,
                X_eval, y_eval,
                params
            )
            
            recall = metrics.get('recall', 0.0)
            precision = metrics.get('precision', 0.0)
            
            # Optimize for precision while maintaining minimum recall
            score = precision if recall >= min_recall else 0.0
            
            # logger.info(f"Trial {trial.number}:")
            # logger.info(f"  Params: {params}")
            # logger.info(f"  Score: {score}")
            
            for metric_name, metric_value in metrics.items():
                trial.set_user_attr(metric_name, metric_value)
            return score
            
        except Exception as e:
            logger.error(f"Trial failed: {str(e)}")
            return 0.0
    
    try:
        study = optuna.create_study(
            study_name='xgboost_optimization',
            direction='maximize',
            sampler=optuna.samplers.TPESampler(   # Different seed for better randomization
                n_startup_trials=50,     # Reduced from 50 - more efficient
                prior_weight=0.2
            )
        )
        
        # Optimize
        best_score = -float('inf')  # Initialize with worst possible score
        best_params = {}
        
        def callback(study, trial):
            nonlocal best_score
            nonlocal best_params
            logger.info(f"Current best score: {best_score:.4f}")
            if trial.value > best_score:
                best_score = trial.value
                best_params = trial.params
                logger.info(f"New best score found in trial {trial.number}: {best_score:.4f}")
            return best_score
        
        study.optimize(
            objective, 
            n_trials=100, 
            timeout=10000,  # 10000 seconds as in notebook
            show_progress_bar=True,
            callbacks=[callback]
        )
        
        best_params.update(first_stage_params)
        
        logger.info(f"Best trial value: {best_score}")
        logger.info(f"Best parameters found: {best_params}")
        return best_params
            
    except Exception as e:
        logger.error(f"Error in hyperparameter optimization: {str(e)}")
        raise

def hypertune_xgboost(experiment_name: str):
    """
    Main training function with MLflow tracking.
    Updated name from hypertune_mlp to hypertune_xgboost to match notebook.
    
    Args:
        experiment_name (str): Experiment name for MLflow tracking
        
    Returns:
        tuple: (best_params, best_metrics)
    """
    try:
            
        # Load hyperparameter space
        hyperparameter_space = load_hyperparameter_space()
        
        # Run hyperparameter optimization
        logger.info("Starting hyperparameter optimization")
        best_params = optimize_hyperparameters(
            X_train, y_train,
            X_test, y_test,
            X_eval, y_eval,
            hyperparameter_space=hyperparameter_space
        )
        
        # Train final model with best parameters
        logger.info("Training final model with best parameters")
        model, metrics = train_model(
            X_train, y_train,
            X_test, y_test,
            X_eval, y_eval,
            best_params
        )
        
        # Log final metrics
        mlflow.log_metrics({
            "precision": metrics.get('precision', 0.0),
            "recall": metrics.get('recall', 0.0),
            "f1": metrics.get('f1', 0.0),
            "auc": metrics.get('auc', 0.0),
            "threshold": metrics.get('threshold', 0.5)
        })
        
        # Log model
        # Create input example with a sample from evaluation data
        # Handle integer columns by converting them to float64 to properly manage missing values
        input_example = X_eval.iloc[:5].copy() if hasattr(X_eval, 'iloc') else X_eval[:5].copy()
        
        # Identify and convert integer columns to float64 to prevent schema enforcement errors
        if hasattr(input_example, 'dtypes'):
            for col in input_example.columns:
                if input_example[col].dtype.kind == 'i':
                    logger.info(f"Converting integer column '{col}' to float64 to handle potential missing values")
                    input_example[col] = input_example[col].astype('float64')
        
        # Infer signature with proper handling for integer columns with potential missing values
        signature = mlflow.models.infer_signature(
            input_example,
            model.predict(input_example)
        )
        
        # Log warning about integer columns in signature
        logger.info("Model signature created - check logs for any warnings about integer columns")
        mlflow.xgboost.log_model(
            model,
            "model",
            registered_model_name=f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M')}",
            signature=signature
        )
        
        # Save model locally
        model_path = Path(f"models/StackedEnsemble/base/tree_based/xgboost_model_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl")
        save_model(model, model_path, metrics.get('threshold', 0.5))
        
        return best_params, metrics
            
    except Exception as e:
        logger.error(f"Error in hyperparameter tuning: {str(e)}")
        return None, None

def log_to_mlflow(model, metrics, params, experiment_name):
    """
    Log trained model, metrics, and parameters to MLflow.
    
    Args:
        model: Trained XGBoost model
        metrics: Model evaluation metrics
        params: Model parameters
        experiment_name: Experiment name
        
    Returns:
        str: Run ID
    """
    try: 
    
        # Log parameters
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log model
        model_info = mlflow.xgboost.log_model(
            model,
            "model",
            registered_model_name=f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M')}"
        )
        
        # Log feature importance using the shared utility
        importance_df = calculate_feature_importance(
            model, 
            feature_names=X_train.columns if hasattr(X_train, 'columns') else None
        )
        
        if not importance_df.empty:
            # Save to CSV and log as artifact
            importance_path = "feature_importance.csv"
            importance_df.to_csv(importance_path, index=False)
            mlflow.log_artifact(importance_path)
            os.remove(importance_path)
        
        logger.info(f"Model logged to MLflow: {model_info.model_uri}")
        logger.info(f"Run ID: {run.info.run_id}")
        return run.info.run_id
            
    except Exception as e:
        logger.error(f"Error logging to MLflow: {str(e)}")
        return None

def train_with_precision_target(X_train, y_train, X_test, y_test, X_eval, y_eval):
    """
    Train XGBoost model with focus on precision target.
    
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
        # Run hyperparameter tuning
        logger.info("Running hyperparameter tuning")
        best_params, _ = hypertune_xgboost(experiment_name)
        
        if not best_params:
            logger.warning("Hyperparameter tuning failed. Using default parameters.")
            best_params = first_stage_params.copy()
            best_params.update({
                'learning_rate': 0.05,
                'max_depth': 6,
                'min_child_weight': 20,
                'colsample_bytree': 0.8,
                'subsample': 0.8,
                'alpha': 0.1,
                'lambda': 0.1,
                'gamma': 0.01
            })
        
        # Train final model with best parameters
        logger.info("Training final model with best parameters")
        model, metrics = train_model(
            X_train, y_train,
            X_test, y_test,
            X_eval, y_eval,
            best_params
        )
        
        # Log to MLflow
        log_to_mlflow(model, metrics, best_params, experiment_name)
        
        # Save model locally
        model_path = f"models/StackedEnsemble/base/tree_based/xgboost_model_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
        save_model(model, model_path, metrics.get('threshold', 0.5))
        
        return model, metrics
        
    except Exception as e:
        logger.error(f"Error in precision-focused training: {str(e)}")
        return None, None

def train_stage1_model(train_data, val_data, params=None):
    """
    Train the first stage model optimized for recall.
    
    Args:
        train_data: DataFrame with training data
        val_data: DataFrame with validation data
        params: Optional parameters to override defaults
        
    Returns:
        tuple: (model, threshold, val_predictions)
    """
    try:
        
        # Use provided params or default
        model_params = params.copy() if params else first_stage_params.copy()
        
        # Create DMatrix objects
        dtrain = xgb.DMatrix(train_data, label=train_data['is_draw'])
        dval = xgb.DMatrix(val_data, label=val_data['is_draw'])
            
        # Train with early stopping
        evals_result = {}
        model_stage1 = xgb.train(
            model_params,
            dtrain,
            num_boost_round=5000,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=500,
            evals_result=evals_result,
            verbose_eval=100
        )
        
        # Log metrics
        for metric in ['auc', 'logloss', 'error']:
            if metric in evals_result['val']:
                mlflow.log_metric(f"val_{metric}", evals_result['val'][metric][-1])
        
        # Generate predictions
        val_preds = model_stage1.predict(dval)
        
        # Calculate recall-focused metrics
        from sklearn.metrics import precision_recall_curve, auc
        precision, recall, thresholds = precision_recall_curve(val_data['is_draw'], val_preds)
        
        # Find threshold that gives target recall
        target_recall = target_stage1_recall  # Aiming for 40%+ recall
        
        # Handle edge case where highest recall is lower than target
        if np.max(recall) < target_recall:
            logger.warning(f"Unable to achieve target recall of {target_recall}. Max recall: {np.max(recall)}")
            # Select highest recall point
            recall_threshold_idx = np.argmax(recall)
        else:
            # Find point with recall just above target
            recall_threshold_idx = np.where(recall >= target_recall)[0][-1]
            
        selected_threshold = thresholds[recall_threshold_idx]
        selected_precision = precision[recall_threshold_idx]
        
        # Log model with signature
        signature = mlflow.models.infer_signature(
            val_data,
            model_stage1.predict(dval)
        )
        
        mlflow.xgboost.log_model(
            xgb_model=model_stage1,
            artifact_path="model_stage1",
            registered_model_name=f"xgboost_stage1_{datetime.now().strftime('%Y%m%d_%H%M')}",
            signature=signature
        )
        
        # Log feature importance
        feature_importance = model_stage1.get_score(importance_type='gain')
        mlflow.log_dict(feature_importance, "artifacts/feature_importance_stage1.json")
        
        # Create and log feature importance plot
        try:
            importance_df = pd.DataFrame.from_dict(feature_importance, orient='index', columns=['importance'])
            importance_df = importance_df.sort_values('importance', ascending=False)
        
        except Exception as e:
            logger.warning(f"Error creating feature importance plot: {str(e)}")
        
        # End logging
        # Log stage 1 metrics to both MLflow and logger
        metrics = {
            "stage1_recall": recall[recall_threshold_idx],
            "stage1_precision": selected_precision,
            "stage1_threshold": selected_threshold
        }

        logger.info(f"Stage 1 model metrics: {metrics}")
        
        return model_stage1, selected_threshold, val_preds
        
    except Exception as e:
        logger.error(f"Error in stage 1 training: {str(e)}")
        raise

def prepare_stage2_data(val_data, val_preds, threshold):
    """
    Prepare data for stage 2 training by filtering stage 1 predictions.
    
    Args:
        val_data: DataFrame with validation data
        val_preds: Stage 1 predictions
        threshold: Stage 1 threshold
        
    Returns:
        DataFrame: Filtered data for stage 2 training
    """
    # Add stage 1 predictions and derived features
    val_data_with_preds = val_data.copy()
    val_data_with_preds['stage1_prediction'] = val_preds
    val_data_with_preds['stage1_prediction_confidence'] = abs(val_preds - 0.5) * 2
    val_data_with_preds['stage1_positive'] = (val_preds >= threshold).astype(int)
    
    # Filter data based on stage 1 predictions
    val_filtered = val_data_with_preds[val_data_with_preds['stage1_positive'] == 1].copy()
    
    # Log filtering stats
    logger.info(f"Stage 1 filtering: {len(val_filtered)}/{len(val_data)} samples kept ({len(val_filtered)/len(val_data):.2%})")
    logger.info(f"Positive rate in filtered data: {val_filtered['is_draw'].mean():.2%}")
    
    return val_filtered

def train_stage2_model(val_filtered, params=None):
    """
    Train the second stage model optimized for precision.
    
    Args:
        val_filtered: DataFrame with filtered data from stage 1
        params: Optional parameters to override defaults
        
    Returns:
        tuple: (model, threshold)
    """
    try:
        
        # Use provided params or default
        model_params = params.copy() if params else second_stage_params.copy()
        
        # Define stage 2 features including stage 1 prediction features
        stage2_features = val_filtered.columns.tolist() + ['stage1_prediction', 'stage1_prediction_confidence']
        
        # Split filtered data for validation
        stage2_train, stage2_val = train_test_split(
            val_filtered, test_size=0.3, stratify=val_filtered['is_draw'], random_state=SEED
        )
        
        # Create DMatrix objects for second stage
        dtrain_s2 = xgb.DMatrix(stage2_train[stage2_features], label=stage2_train['is_draw'])
        dval_s2 = xgb.DMatrix(stage2_val[stage2_features], label=stage2_val['is_draw'])
            
        # Train with early stopping
        evals_result = {}
        model_stage2 = xgb.train(
            model_params,
            dtrain_s2,
            num_boost_round=3000,
            evals=[(dtrain_s2, 'train'), (dval_s2, 'val')],
            early_stopping_rounds=300,
            evals_result=evals_result,
            verbose_eval=100
        )
        
        # Log metrics
        for metric in ['auc', 'logloss', 'error']:
            if metric in evals_result['val']:
                mlflow.log_metric(f"val_{metric}", evals_result['val'][metric][-1])
        
        # Generate predictions
        stage2_preds = model_stage2.predict(dval_s2)
        
        # Calculate precision-focused metrics
        precision, recall, thresholds = precision_recall_curve(stage2_val['is_draw'], stage2_preds)
        
        # Find threshold that gives target precision
        target_precision_s2 = target_precision  # Aiming for 50% precision
        
        # Handle edge case where highest precision is lower than target
        if np.max(precision) < target_precision_s2:
            logger.warning(f"Unable to achieve target precision of {target_precision_s2}. Max precision: {np.max(precision)}")
            # Select highest precision point
            precision_threshold_idx = np.argmax(precision)
        else:
            # Find point with precision just above target
            precision_threshold_idx = np.where(precision >= target_precision_s2)[0][0]
        
        selected_threshold_s2 = thresholds[precision_threshold_idx]
        selected_recall_s2 = recall[precision_threshold_idx]
        
        # Log model with signature
        signature = mlflow.models.infer_signature(
            stage2_val[stage2_features],
            model_stage2.predict(dval_s2)
        )
        
        mlflow.xgboost.log_model(
            xgb_model=model_stage2,
            artifact_path="model_stage2",
            registered_model_name=f"xgboost_stage2_{datetime.now().strftime('%Y%m%d_%H%M')}",
            signature=signature
        )
        
        # Log feature importance
        feature_importance = model_stage2.get_score(importance_type='gain')
        mlflow.log_dict(feature_importance, "artifacts/feature_importance_stage2.json")
        
        # Create and log feature importance plot
        try:
            importance_df = pd.DataFrame.from_dict(feature_importance, orient='index', columns=['importance'])
            importance_df = importance_df.sort_values('importance', ascending=False)

        except Exception as e:
            logger.warning(f"Error creating feature importance plot: {str(e)}")
        
        # Calculate combined metrics if val_data is available
        if 'val_data' in globals() and val_data is not None:
            # Calculate stage1 recall from the original val_data that contains all samples
            stage1_recall = len(val_filtered[val_filtered['is_draw'] == 1]) / len(val_data[val_data['is_draw'] == 1])
            combined_recall = stage1_recall * selected_recall_s2
        else:
            # If val_data is not available, we can't calculate combined recall
            logger.warning("Original validation data not available, can't calculate combined recall")
            stage1_recall = None
            combined_recall = None
        
        # Also log to experiment logger for structured JSON logging
        logger.info(f"Stage 2 model metrics: recall={selected_recall_s2:.4f}, "
                    f"precision={precision[precision_threshold_idx]:.4f}, "
                    f"threshold={selected_threshold_s2:.4f}")
        logger.info(f"Combined model metrics: recall={combined_recall if combined_recall is not None else 0:.4f}, "
                    f"precision={precision[precision_threshold_idx]:.4f}")
        
        return model_stage2, selected_threshold_s2, stage2_features
        
    except Exception as e:
        logger.error(f"Error in stage 2 training: {str(e)}")
        raise

def predict_two_stage(data, model1, model2, threshold1, threshold2, features1):
    """
    Apply the full two-stage pipeline to data.
    
    Args:
        data: DataFrame with features
        model1: First stage model
        model2: Second stage model
        threshold1: First stage threshold
        threshold2: Second stage threshold
        features1: Features for first stage
        
    Returns:
        tuple: (stage1_positives, final_positives)
    """
    try:
        # First stage prediction
        dtest1 = xgb.DMatrix(data[features1])
        stage1_preds = model1.predict(dtest1)
        data_with_preds = data.copy()
        data_with_preds['stage1_prediction'] = stage1_preds
        data_with_preds['stage1_prediction_confidence'] = abs(stage1_preds - 0.5) * 2
        data_with_preds['stage1_positive'] = (stage1_preds >= threshold1).astype(int)
        
        # Filter based on first stage
        stage1_positives = data_with_preds[data_with_preds['stage1_positive'] == 1].copy()
        
        if len(stage1_positives) == 0:
            logger.info("No samples passed the first stage filter")
            return data_with_preds, pd.DataFrame()
        
        # Second stage prediction
        dtest2 = xgb.DMatrix(stage1_positives)
        stage2_preds = model2.predict(dtest2)
        stage1_positives['stage2_prediction'] = stage2_preds
        stage1_positives['stage2_positive'] = (stage2_preds >= threshold2).astype(int)
        
        # Final predictions
        final_positives = stage1_positives[stage1_positives['stage2_positive'] == 1].copy()
        
        # Update original dataframe with second stage predictions
        data_with_preds['stage2_prediction'] = np.nan
        data_with_preds['stage2_positive'] = 0
        data_with_preds.loc[stage1_positives.index, 'stage2_prediction'] = stage2_preds
        data_with_preds.loc[final_positives.index, 'stage2_positive'] = 1
        
        # Add final prediction column
        data_with_preds['final_prediction'] = data_with_preds['stage1_positive'] & data_with_preds['stage2_positive']
        
        return data_with_preds, final_positives
        
    except Exception as e:
        logger.error(f"Error in two-stage prediction: {str(e)}")
        raise

def evaluate_two_stage_model(test_data, model1, model2, threshold1, threshold2):
    """
    Evaluate the two-stage model on test data.
    
    Args:
        test_data: DataFrame with test data
        model1: First stage model
        model2: Second stage model
        threshold1: First stage threshold
        threshold2: Second stage threshold
        
    Returns:
        dict: Evaluation metrics
    """
    try:
        
        # Apply the two-stage pipeline
        test_with_preds, final_positives = predict_two_stage(
            test_data, model1, model2, threshold1, threshold2
        )
        
        # Calculate metrics
        stage1_positives = test_with_preds[test_with_preds['stage1_positive'] == 1]
        
        # First stage metrics
        stage1_precision = stage1_positives['is_draw'].mean() if len(stage1_positives) > 0 else 0
        stage1_recall = len(stage1_positives[stage1_positives['is_draw'] == 1]) / len(test_data[test_data['is_draw'] == 1]) if len(test_data[test_data['is_draw'] == 1]) > 0 else 0
        
        # Final metrics
        final_precision = final_positives['is_draw'].mean() if len(final_positives) > 0 else 0
        final_recall = len(final_positives[final_positives['is_draw'] == 1]) / len(test_data[test_data['is_draw'] == 1]) if len(test_data[test_data['is_draw'] == 1]) > 0 else 0
        
        # F1 score
        final_f1 = 2 * (final_precision * final_recall) / (final_precision + final_recall) if (final_precision + final_recall) > 0 else 0
        
        # Log metrics
        metrics = {
            "stage1_precision": stage1_precision,
            "stage1_recall": stage1_recall,
            "final_precision": final_precision,
            "final_recall": final_recall,
            "final_f1": final_f1,
            "stage1_positives": len(stage1_positives),
            "final_positives": len(final_positives),
            "stage1_filtering_rate": len(stage1_positives) / len(test_data),
            "stage2_filtering_rate": len(final_positives) / len(stage1_positives) if len(stage1_positives) > 0 else 0
        }
        
        mlflow.log_metrics(metrics)
        
        # Create confusion matrix
        y_true = test_data['is_draw']
        y_pred = test_with_preds['final_prediction']
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Log confusion matrix
        logger.info(f"Confusion Matrix:\n{cm}")
        
        # Create and log classification report
        from sklearn.metrics import classification_report
        report = classification_report(y_true, y_pred)
        logger.info(f"Classification Report:\n{report}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error in model evaluation: {str(e)}")
        raise


def objective_stage1(trial):
    """
    Optuna objective function for Stage 1 model optimization.
    Focused on maximizing recall while maintaining reasonable precision.
    
    Parameters:
    -----------
    trial : optuna.Trial
        Optuna trial object
        
    Returns:
    --------
    float
        Composite score favoring recall
    """
    try:
        # Define hyperparameters to tune
        params = {
            'objective': 'binary:logistic',
            'tree_method': 'hist',  # CPU-optimized as per guidelines
            'eval_metric': 'logloss',
            
            # Core hyperparameters
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
            'gamma': trial.suggest_float('gamma', 0.1, 1.0),
            
            # Regularization parameters
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
            
            # Subsampling parameters
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            
            # Recall-focused parameters
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 5.0)
        }
        
        # Load data if not already available
        global train_data, val_data, test_data
        if 'train_data' not in globals() or train_data is None:
            train_data, val_data, test_data = prepare_data()
        
        # Create DMatrix objects
        dtrain = xgb.DMatrix(train_data, label=train_data['is_draw'])
        dval = xgb.DMatrix(val_data, label=val_data['is_draw'])
        
        # Train with early stopping
        evals_result = {}
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=3000,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=500,
            evals_result=evals_result,
            verbose_eval=False
        )
        
        # Generate predictions
        val_preds = model.predict(dval)
        
        # Calculate recall-focused metrics
        precision, recall, thresholds = precision_recall_curve(val_data['is_draw'], val_preds)
        
        # Find the point with recall >= 0.4 that maximizes precision
        valid_indices = np.where(recall >= target_stage1_recall)[0]
        if len(valid_indices) > 0:
            best_idx = valid_indices[np.argmax(precision[valid_indices])]
            best_threshold = thresholds[best_idx]
            best_precision = precision[best_idx]
            best_recall = recall[best_idx]
        else:
            # If no point reaches 0.4 recall, find maximum recall
            best_idx = np.argmax(recall)
            best_threshold = thresholds[best_idx]
            best_precision = precision[best_idx]
            best_recall = recall[best_idx]
        
        # Calculate PR-AUC
        pr_auc = auc(recall, precision)
        
        # Create composite score that favors recall
        # Weights: 60% recall, 30% precision, 10% PR-AUC
        composite_score = (0.6 * best_recall) + (0.3 * best_precision) + (0.1 * pr_auc)
        
        # Also log to experiment logger for structured JSON logging
        logger.info(f"Stage 1 optimization metrics: threshold={best_threshold:.4f}, "
                    f"precision={best_precision:.4f}, recall={best_recall:.4f}, "
                    f"pr_auc={pr_auc:.4f}, composite_score={composite_score:.4f}")
        
        # Return composite score
        return composite_score
    
    except Exception as e:
        logger.error(f"Error in Stage 1 trial {trial.number}: {str(e)}")
        return 0.0


def objective_stage2(trial, stage1_model, stage1_threshold):
    """
    Optuna objective function for Stage 2 model optimization.
    Focused on maximizing precision while maintaining target recall.
    
    Parameters:
    -----------
    trial : optuna.Trial
        Optuna trial object
    stage1_model : xgb.Booster
        Trained Stage 1 model
    stage1_threshold : float
        Threshold for Stage 1 model
        
    Returns:
    --------
    float
        Composite score favoring precision
    """
    
    try:
        # Load data if not already available
        global train_data, val_data, test_data
        if 'train_data' not in globals() or train_data is None:
            train_data, val_data, test_data = prepare_data()
        
        # Generate Stage 1 predictions
        dval_s1 = xgb.DMatrix(val_data)
        val_preds_s1 = stage1_model.predict(dval_s1)
        
        # Filter validation set based on Stage 1 predictions
        val_data_s1 = val_data.copy()
        val_data_s1['stage1_prediction'] = val_preds_s1
        val_data_s1['stage1_prediction_confidence'] = abs(val_preds_s1 - 0.5) * 2
        val_filtered = val_data_s1[val_preds_s1 >= stage1_threshold].copy()
        
        # If filtered set is too small, abort trial
        if len(val_filtered) < 50 or val_filtered['is_draw'].sum() < 10:
            logger.warning(f"Insufficient samples after filtering: {len(val_filtered)} total, {val_filtered['is_draw'].sum()} positive")
            return 0.0
        
        # Define hyperparameters for Stage 2
        params = {
            'objective': 'binary:logistic',
            'tree_method': 'hist',  # CPU-optimized
            'eval_metric': 'logloss',
            
            # Core hyperparameters
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 2, 6),  # Shallower to prevent overfitting
            'min_child_weight': trial.suggest_int('min_child_weight', 2, 8),  # Higher to prevent overfitting
            'gamma': trial.suggest_float('gamma', 0.0, 2.0),
            
            # Regularization parameters (stronger)
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 20.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 20.0, log=True),
            
            # Subsampling parameters
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            
            # Precision-focused parameters
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.8, 3.0)
        }
        
        # Define stage 2 features
        stage2_features = val_filtered.columns.tolist() + ['stage1_prediction', 'stage1_prediction_confidence']
        
        # Split filtered data for training/validation
        stage2_train, stage2_val = train_test_split(
            val_filtered, test_size=0.3, stratify=val_filtered['is_draw'], random_state=SEED
        )
        
        # Create DMatrix objects for Stage 2
        dtrain_s2 = xgb.DMatrix(stage2_train[stage2_features], label=stage2_train['is_draw'])
        dval_s2 = xgb.DMatrix(stage2_val[stage2_features], label=stage2_val['is_draw'])
        
        # Train with early stopping
        evals_result = {}
        model = xgb.train(
            params,
            dtrain_s2,
            num_boost_round=2000,
            evals=[(dtrain_s2, 'train'), (dval_s2, 'val')],
            early_stopping_rounds=300,
            evals_result=evals_result,
            verbose_eval=False
        )
        
        # Generate predictions
        stage2_preds = model.predict(dval_s2)
        
        # Calculate precision-focused metrics
        precision, recall, thresholds = precision_recall_curve(stage2_val['is_draw'], stage2_preds)
        
        # Find the point with precision >= 0.5 that maximizes recall
        valid_indices = np.where(precision >= target_precision)[0]
        if len(valid_indices) > 0:
            best_idx = valid_indices[np.argmax(recall[valid_indices])]
            best_threshold = thresholds[best_idx]
            best_precision = precision[best_idx]
            best_recall = recall[best_idx]
        else:
            # If no point reaches 0.5 precision, find maximum precision
            best_idx = np.argmax(precision)
            best_threshold = thresholds[best_idx]
            best_precision = precision[best_idx]
            best_recall = recall[best_idx]
        
        # Calculate PR-AUC
        pr_auc = auc(recall, precision)
        
        # Calculate effective overall recall
        stage1_recall = len(val_filtered[val_filtered['is_draw'] == 1]) / len(val_data[val_data['is_draw'] == 1])
        combined_recall = stage1_recall * best_recall
        
        # Create composite score that favors precision
        # Weights: 60% precision, 20% stage2 recall, 10% combined recall, 10% PR-AUC
        composite_score = (0.6 * best_precision) + (0.2 * best_recall) + (0.1 * combined_recall) + (0.1 * pr_auc)
        
        # Check if we meet minimum combined recall of 20%
        if combined_recall < min_recall:
            # Apply penalty for not meeting minimum recall
            composite_score *= (combined_recall / min_recall)
        
        # Log metrics
        logger.info(f"Stage 2 optimization metrics: threshold={best_threshold:.4f}, "
                    f"precision={best_precision:.4f}, recall={best_recall:.4f}, "
                    f"combined_recall={combined_recall:.4f}, pr_auc={pr_auc:.4f}, "
                    f"composite_score={composite_score:.4f}")
        
        # Return composite score
        return composite_score
    
    except Exception as e:
        logger.error(f"Error in Stage 2 trial {trial.number}: {str(e)}")
        return 0.0


def run_hyperparameter_optimization():
    """
    Run the full two-stage hyperparameter optimization process.
    
    Returns:
        dict: Optimization results
    """
    try:
        
        # Load data
        global train_data, val_data, test_data
        train_data, val_data, test_data = prepare_data()
        
        # 1. Optimize Stage 1 model for recall
        logger.info("Starting Stage 1 (Recall-Focused) Optimization")
        
        # Create Optuna study with TPESampler for Stage 1
        stage1_sampler = TPESampler(
            consider_prior=True,
            prior_weight=1.0,
            seed=SEED,
            n_startup_trials=10
        )
        
        stage1_study = optuna.create_study(
            direction="maximize",
            sampler=stage1_sampler,
            study_name="stage1_optimization"
        )
        
        # Run optimization
        stage1_study.optimize(objective_stage1, n_trials=50, timeout=10800)  # 3 hours timeout
        
        # Log best Stage 1 parameters
        logger.info(f"Best Stage 1 Parameters: {stage1_study.best_params}")
        logger.info(f"Best Stage 1 Composite Score: {stage1_study.best_value}")
        
        # Train best Stage 1 model
        stage1_best_params = stage1_study.best_params.copy()
        stage1_best_params.update({
            'objective': 'binary:logistic',
            'tree_method': 'hist',
            'eval_metric': ['auc', 'logloss', 'error']
        })
        
        model_stage1, threshold_stage1, val_preds = train_stage1_model(train_data, val_data, stage1_best_params)
        
        # 2. Optimize Stage 2 model for precision
        logger.info("Starting Stage 2 (Precision-Focused) Optimization")
        
        # Create filtered dataset for Stage 2
        val_filtered = prepare_stage2_data(val_data, val_preds, threshold_stage1)
        
        # Create Optuna study for Stage 2
        stage2_sampler = TPESampler(
            consider_prior=True,
            prior_weight=1.0,
            seed=SEED + 1,
            n_startup_trials=10
        )
        
        stage2_study = optuna.create_study(
            direction="maximize",
            sampler=stage2_sampler,
            study_name="stage2_optimization"
        )
        
        # Create partial function for stage2 objective with fixed stage1 model
        from functools import partial
        stage2_objective = partial(objective_stage2, stage1_model=model_stage1, stage1_threshold=threshold_stage1)
        
        # Run optimization
        stage2_study.optimize(stage2_objective, n_trials=50, timeout=10800)  # 3 hours timeout
        
        # Log best Stage 2 parameters
        logger.info(f"Best Stage 2 Parameters: {stage2_study.best_params}")
        logger.info(f"Best Stage 2 Composite Score: {stage2_study.best_value}")
        
        # Train best Stage 2 model
        stage2_best_params = stage2_study.best_params.copy()
        stage2_best_params.update({
            'objective': 'binary:logistic',
            'tree_method': 'hist',
            'eval_metric': ['auc', 'logloss', 'error']
        })
        
        model_stage2, threshold_stage2, stage2_features = train_stage2_model(val_filtered, stage2_best_params)
        
        # 3. Evaluate the full pipeline
        metrics = evaluate_two_stage_model(
            test_data, 
            model_stage1, 
            model_stage2, 
            threshold_stage1, 
            threshold_stage2
        )
        

        # Log parameters for both models
        mlflow.log_params({
            "stage1_" + k: v for k, v in stage1_best_params.items()
        })
        mlflow.log_params({
            "stage2_" + k: v for k, v in stage2_best_params.items()
        })
        mlflow.log_params({
            "stage1_threshold": threshold_stage1,
            "stage2_threshold": threshold_stage2
        })
        
        # Log metrics
        mlflow.log_metrics({
            "final_precision": metrics["final_precision"],
            "final_recall": metrics["final_recall"],
            "final_f1": metrics["final_f1"]
        })
        
        # Register the combined model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        model_info = {
            "stage1_model": model_stage1,
            "stage2_model": model_stage2,
            "stage1_threshold": threshold_stage1,
            "stage2_threshold": threshold_stage2
        }
        
        # Save combined model info to file
        with open(f"two_stage_model_{timestamp}.pkl", "wb") as f:
            pickle.dump(model_info, f)
        
        # Log model file as artifact
        mlflow.log_artifact(f"two_stage_model_{timestamp}.pkl", "model")
        
        # Register model
        mlflow.register_model(
            f"runs:/{run.info.run_id}/model",
            f"two_stage_xgboost_{timestamp}"
        )
        
        # Clean up
        if os.path.exists(f"two_stage_model_{timestamp}.pkl"):
            os.remove(f"two_stage_model_{timestamp}.pkl")
        
        # 5. Summary of optimization results
        logger.info("Hyperparameter Optimization Complete!")
        logger.info(f"Target Metrics: Precision {target_precision:.1%}, Recall {min_recall:.1%}")
        logger.info(f"Final Performance: Precision {metrics['final_precision']:.2%}, Recall {metrics['final_recall']:.2%}, F1: {metrics['final_f1']:.2%}")
        
        return {
            'stage1_params': stage1_best_params,
            'stage2_params': stage2_best_params,
            'stage1_threshold': threshold_stage1,
            'stage2_threshold': threshold_stage2,
            'final_metrics': metrics
        }
        
    except Exception as e:
        logger.error(f"Error in hyperparameter optimization: {str(e)}")
        raise

def main():
    """
    Main execution function for the two-stage XGBoost model.
    """
    try:
        logger.info("Starting Two-Stage XGBoost model training")
        
        # Load data if not already available
        global train_data, val_data, test_data
        if 'train_data' not in globals() or train_data is None:
            train_data, val_data, test_data = prepare_data()
        
        # Log data shapes
        logger.info(f"Training data shape: {train_data.shape}")
        logger.info(f"Validation data shape: {val_data.shape}")
        logger.info(f"Testing data shape: {test_data.shape}")
        logger.info(f"Positive class ratio - Train: {train_data['is_draw'].mean():.3f}, "
                    f"Val: {val_data['is_draw'].mean():.3f}, Test: {test_data['is_draw'].mean():.3f}")
        
        # Create nested runs for the complete pipeline
        with mlflow.start_run(run_name="complete_pipeline") as parent_run:
            # Log overall parameters
            mlflow.log_params({
                "target_precision": target_precision,
                "target_recall": min_recall,
                "stage1_focus": "recall",
                "stage2_focus": "precision",
                "device": DEVICE
            })
            
            # Track lineage
            mlflow.set_tag("model_lineage", "two_stage_xgboost_v1")
            
            # Choose the training approach based on whether to do hyperparameter optimization
            do_hyperopt = True  # Set to False to use default parameters
            
            if do_hyperopt:
                logger.info("Running with hyperparameter optimization")
                results = run_hyperparameter_optimization()
                final_metrics = results['final_metrics']
            else:
                logger.info("Running with default parameters")
                
                # 1. Train Stage 1 model
                model_stage1, threshold_stage1, val_preds = train_stage1_model(train_data, val_data)
                
                # 2. Prepare data for Stage 2
                val_filtered = prepare_stage2_data(val_data, val_preds, threshold_stage1)
                
                # 3. Train Stage 2 model
                model_stage2, threshold_stage2, stage2_features = train_stage2_model(val_filtered)
                
                # 4. Evaluate the full pipeline
                final_metrics = evaluate_two_stage_model(
                    test_data, 
                    model_stage1, 
                    model_stage2, 
                    threshold_stage1, 
                    threshold_stage2
                )
            
            # Log final metrics to parent run
            mlflow.log_metrics({
                "final_precision": final_metrics["final_precision"],
                "final_recall": final_metrics["final_recall"],
                "final_f1": final_metrics["final_f1"],
                "stage1_filtering_rate": final_metrics["stage1_filtering_rate"],
                "stage2_filtering_rate": final_metrics["stage2_filtering_rate"]
            })
            
            # Summary message
            logger.info("Two-Stage XGBoost Training Complete!")
            logger.info(f"Target Metrics: Precision {target_precision:.1%}, Recall {min_recall:.1%}")
            logger.info(f"Final Performance: Precision {final_metrics['final_precision']:.2%}, "
                        f"Recall {final_metrics['final_recall']:.2%}, F1: {final_metrics['final_f1']:.2%}")
            
            if final_metrics['final_precision'] >= target_precision and final_metrics['final_recall'] >= min_recall:
                logger.info("SUCCESS: Target metrics achieved!")
            else:
                logger.warning("Target metrics not fully achieved. Consider further optimization.")
            mlflow.end_run()
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


def load_two_stage_model(model_path):
    """
    Load a trained two-stage model from file.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        dict: Model information
    """
    try:
        with open(model_path, 'rb') as f:
            model_info = pickle.load(f)
        
        logger.info(f"Two-stage model loaded from {model_path}")
        return model_info
        
    except Exception as e:
        logger.error(f"Error loading two-stage model: {str(e)}")
        raise


def two_stage_predict(match_data, model_info=None, model_path=None):
    """
    Apply the two-stage model to new match data.
    
    Parameters:
    -----------
    match_data : DataFrame
        Features for the matches to predict
    model_info : dict, optional
        Model information dictionary (if already loaded)
    model_path : str, optional
        Path to load model from (if not already loaded)
        
    Returns:
    --------
    DataFrame with original data and prediction columns
    """
    try:
        
        # Load model if needed
        if model_info is None:
            if model_path is not None:
                model_info = load_two_stage_model(model_path)
            else:
                # Load from MLflow registry
                stage1_model = mlflow.xgboost.load_model("models:/xgboost_stage1_latest/Production")
                stage2_model = mlflow.xgboost.load_model("models:/xgboost_stage2_latest/Production")
                # TODO: Load thresholds and features from model metadata
                model_info = {
                    "stage1_model": stage1_model,
                    "stage2_model": stage2_model,
                    "stage1_threshold": 0.3,  # Default threshold if not available
                    "stage2_threshold": 0.6,  # Default threshold if not available
                }
        
        # Extract model components
        model1 = model_info["stage1_model"]
        model2 = model_info["stage2_model"]
        threshold1 = model_info["stage1_threshold"]
        threshold2 = model_info["stage2_threshold"]

        # Make predictions
        result_df, final_positives = predict_two_stage(
            match_data, model1, model2, threshold1, threshold2
        )
        
        # Log prediction stats
        mlflow.log_metrics({
            "prediction_count": len(match_data),
            "stage1_positive_count": len(result_df[result_df['stage1_positive'] == 1]),
            "final_positive_count": len(final_positives)
        })
        
        return result_df
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise


if __name__ == "__main__":
    main() 