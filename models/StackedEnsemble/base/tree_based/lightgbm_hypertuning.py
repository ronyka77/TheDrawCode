"""Script to run hyperparameter tuning for LightGBM model."""

import sys
from pathlib import Path
from typing import Dict, Any
import traceback
import time
import json
import os
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import pandas as pd
import gc
from tqdm import tqdm
from datetime import datetime

os.environ["ARROW_S3_DISABLE"] = "1"

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
os.environ["PYTHONPATH"] = str(project_root) + os.pathsep + os.environ.get("PYTHONPATH", "")

try:
    from utils.logger import ExperimentLogger
    exp_logger = ExperimentLogger(experiment_name="lightgbm_hypertuning")
    from models.StackedEnsemble.base.tree_based.lightgbm_model import LightGBMModel
    from models.StackedEnsemble.shared.data_loader import DataLoader
    
except Exception as e:
    raise ImportError(f"Error importing modules: {str(e)}")

def convert_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_types(item) for item in obj]
    return obj

def preprocess_data(X, y, name="dataset"):
    """Preprocess and validate input data."""
    if X is None or y is None:
        raise ValueError(f"{name} X or y is None")
    if len(X) == 0 or len(y) == 0:
        raise ValueError(f"{name} X or y is empty")
    
    if not isinstance(X, pd.DataFrame):
        raise ValueError(f"{name} X must be a pandas DataFrame")
    if not isinstance(y, (pd.Series, np.ndarray)):
        raise ValueError(f"{name} y must be a pandas Series or numpy array")
    
    # Log NaN statistics before filling
    null_counts = X.isnull().sum()
    columns_with_nulls = null_counts[null_counts > 0]
    if not columns_with_nulls.empty:
        exp_logger.info(f"NaN statistics for {name} features:")
        exp_logger.info(columns_with_nulls)
    
    # Fill NaN values with appropriate strategies
    X_processed = X.copy()
    
    # Fill numeric columns with median
    numeric_cols = X_processed.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if X_processed[col].isnull().any():
            median_val = X_processed[col].median()
            X_processed[col].fillna(median_val, inplace=True)
            exp_logger.info(f"Filled NaN in {col} with median: {median_val}")
    
    # Verify no NaN values remain
    if X_processed.isnull().any().any():
        raise ValueError(f"Failed to handle all NaN values in {name}")
        
    return X_processed, y

def cleanup_memory():
    """Force garbage collection."""
    gc.collect()

def save_checkpoint(results, checkpoint_dir="checkpoints"):
    """Save intermediate results."""
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_file = checkpoint_path / f"lightgbm_checkpoint_{timestamp}.json"
    
    with open(checkpoint_file, 'w') as f:
        json.dump(convert_types(results), f, indent=2)
    
    return checkpoint_file

def run_lightgbm_hypertuning() -> Dict[str, Any]:
    """Run hyperparameter tuning for LightGBM model."""
    try:
        start_time = time.time()
        exp_logger.info("Starting LightGBM hyperparameter tuning with soccer prediction data")
        
        # Load and preprocess data
        data_loader = DataLoader()
        X_train, y_train, X_test, y_test, X_val, y_val = data_loader.load_data()
        
        # Log data info
        exp_logger.info("Checking data types:")
        exp_logger.info(f"X_train type: {type(X_train)}")
        exp_logger.info(f"y_train type: {type(y_train)}")
        exp_logger.info(f"X_train dtypes:\n{X_train.dtypes}")
        exp_logger.info(f"y_train dtype: {y_train.dtype}")
        
        # Preprocess data
        exp_logger.info("Preprocessing training data...")
        X_train, y_train = preprocess_data(X_train, y_train, "training")
        X_test, y_test = preprocess_data(X_test, y_test, "test")
        X_val, y_val = preprocess_data(X_val, y_val, "validation")
        
        exp_logger.info("Data preprocessing completed successfully")
        
        # Initialize model for optimization
        model = LightGBMModel(
            model_type='lightgbm',
            experiment_name='lightgbm_hypertuning',
            logger=exp_logger
        )
        
        # Run hyperparameter optimization
        optimization_start = time.time()
        best_params = model.optimize_hyperparameters(
            X_train, y_train,
            X_val, y_val,
            X_test, y_test
        )
        optimization_time = time.time() - optimization_start
        
        exp_logger.info(f"Best parameters found: {best_params}")
        exp_logger.info(f"Hyperparameter optimization completed in {optimization_time:.2f} seconds")
        
        # Train final model with best parameters
        exp_logger.info("Training final model with best parameters:", best_params)
        
        # Create new model instance with best parameters
        final_model = LightGBMModel(
            model_type='lightgbm',
            experiment_name='lightgbm_hypertuning',
            logger=exp_logger
        )
        
        # Train with best parameters
        training_start = time.time()
        train_metrics = final_model.fit(
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            **best_params  # Use best parameters from optimization
        )
        training_time = time.time() - training_start
        
        # Evaluate on validation set
        exp_logger.info("Evaluating final model on validation set")
        val_metrics = final_model.evaluate(X_val, y_val)
        
        # Prepare results
        results = {
            'timing': {
                'optimization_time': optimization_time,
                'training_time': training_time,
                'total_time': time.time() - start_time
            },
            'data_stats': {
                'train_shape': X_train.shape,
                'test_shape': X_test.shape,
                'val_shape': X_val.shape,
                'draw_rates': {
                    'train': y_train.mean(),
                    'test': y_test.mean(),
                    'val': y_val.mean()
                }
            },
            'best_params': best_params,
            'training_metrics': train_metrics,
            'validation_metrics': val_metrics
        }
        
        return results
        
    except Exception as e:
        exp_logger.error(f"\nError: {str(e)}")
        error_results = {
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        error_path = Path('results/hypertuning') / f'lightgbm_hypertuning_error_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        error_path.parent.mkdir(exist_ok=True, parents=True)
        with open(error_path, 'w') as f:
            json.dump(error_results, f, indent=4)
        exp_logger.info(f"Error results saved to {error_path}")
        raise

def objective(trial, X_train, y_train, X_val, y_val):
    """Objective function for LightGBM hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        X_train: Training features
        y_train: Training labels 
        X_val: Validation features
        y_val: Validation labels
        
    Returns:
        float: Validation metric score
    """
    # Define hyperparameter space
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        
        # Learning parameters
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        
        # Tree structure parameters
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 200),
        
        # Sampling parameters
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        
        # Regularization parameters
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.001, 1.0, log=True),
        
        # Force CPU usage
        'device': 'cpu',
        'num_threads': -1
    }
    
    # Create and train model
    model = LightGBMModel(
        model_type='lightgbm',
        experiment_name='lightgbm_hypertuning',
        logger=exp_logger
    )
    
    # Use early stopping
    metrics = model.fit(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        **params
    )
    
    precision = metrics['precision']
    threshold = metrics['threshold']
    recall = metrics['recall']
    # Only consider trials with acceptable recall
    min_recall = 0.15
    if recall >= min_recall:
        # Calculate objective value focusing on precision
        objective_value = precision
    else:
        objective_value = 0.0  # Penalize low recall
    
    # Log detailed metrics
    log_metrics = {
        'precision': precision,
        'recall': recall,
        'objective_value': objective_value,
        'threshold': threshold
    }
    
    for metric_name, metric_value in log_metrics.items():
        trial.set_user_attr(metric_name, metric_value)
    
    # Early stopping if precision is too low
    if precision < 0.2:
        trial.report(objective_value, step=1)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    model = None
    return objective_value


if __name__ == '__main__':
    try:
        exp_logger.info("Starting hyperparameter tuning script...")
        results = run_lightgbm_hypertuning()
        
        exp_logger.info("\nHyperparameter Tuning Results:")
        exp_logger.info("=" * 50)
        
        exp_logger.info("\nTiming Information:")
        exp_logger.info("-" * 30)
        for timing_key, timing_value in results['timing'].items():
            exp_logger.info(f"{timing_key}: {timing_value:.2f} seconds")
        
        exp_logger.info("\nData Statistics:")
        exp_logger.info("-" * 30)
        exp_logger.info("Dataset Shapes:")
        for key, shape in results['data_stats'].items():
            if 'shape' in key:
                exp_logger.info(f"{key}: {shape}")
        exp_logger.info("\nDraw Rates:")
        for key, rate in results['data_stats']['draw_rates'].items():
            exp_logger.info(f"{key}: {rate:.2%}")
        
        exp_logger.info("\nBest Parameters:")
        exp_logger.info("-" * 30)
        for param, value in results['best_params'].items():
            exp_logger.info(f"{param}: {value}")
        
        exp_logger.info("\nTraining Metrics:")
        exp_logger.info("-" * 30)
        for metric, value in results['training_metrics'].items():
            exp_logger.info(f"{metric}: {value:.4f}")
        
        exp_logger.info("\nValidation Metrics:")
        exp_logger.info("-" * 30)
        for metric, value in results['validation_metrics'].items():
            exp_logger.info(f"{metric}: {value:.4f}")
            
    except Exception as e:
        exp_logger.error(f"\nError: {str(e)}")
        traceback.print_exc()
        sys.exit(1) 