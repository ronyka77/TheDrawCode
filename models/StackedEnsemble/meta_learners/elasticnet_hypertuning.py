"""ElasticNet hyperparameter tuning implementation."""

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
import mlflow
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
os.environ["PYTHONPATH"] = str(project_root) + os.pathsep + os.environ.get("PYTHONPATH", "")

try:
    from utils.logger import ExperimentLogger
    exp_logger = ExperimentLogger(experiment_name="elasticnet_hypertuning")
    from models.StackedEnsemble.meta_learners.elasticnet_model import ElasticNetModel
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

def create_param_space(trial):
    """Create hyperparameter search space for ElasticNet with optimized ranges."""
    # Regularization parameters - adjusted for better sparsity and stability
    base_alpha = trial.suggest_float('base_alpha', 1e-3, 5e-2, log=True)  # Narrowed range for more stable convergence
    alpha_grid_size = trial.suggest_int('alpha_grid_size', 150, 250)  # Increased minimum for better granularity
    
    # Feature selection threshold - tightened for more aggressive feature selection
    feature_selection_threshold = trial.suggest_float(
        'feature_selection_threshold', 
        5e-3,  # Increased minimum to be more selective
        5e-2,  # Reduced maximum to prevent too much sparsity
        log=True
    )
    # Limited polynomial features to reduce complexity
    poly_degree = trial.suggest_int('poly_degree', 1, 2)  # Keep polynomial features limited
    
    # Create focused alpha range for better exploration
    alphas = np.logspace(
        np.log10(base_alpha / 5),  # Narrower range for more focused search
        np.log10(base_alpha * 5),
        num=alpha_grid_size
    )
    
    # Core model parameters with optimized ranges
    params = {
        'alpha': base_alpha,
        'l1_ratio': trial.suggest_float(
            'l1_ratio', 
            0.8,   # Increased minimum for stronger L1 regularization
            0.99,  # Avoid exactly 1.0 to maintain some L2 regularization
            log=False  # Linear scale for better control
        ),
        'max_iter': trial.suggest_int(
            'max_iter',
            100000,  # Increased minimum for better convergence
            300000,  # Extended maximum for complex cases
            log=True  # Log scale for better exploration
        ),
        'tol': trial.suggest_float(
            'tol',
            1e-6,  # Balanced tolerance
            1e-4,  # Slightly relaxed maximum
            log=True
        ),
        'eps': trial.suggest_float(
            'eps',
            1e-6,  # Matched with tolerance scale
            1e-4,
            log=True
        ),
        
        # Feature selection and preprocessing
        'feature_selection_threshold': feature_selection_threshold,
        'poly_degree': poly_degree,
        'alphas': alphas,
        
        # Fixed model configuration for stability
        'selection': 'cyclic',  # Cyclic is more stable than random
        'positive': trial.suggest_categorical('positive', [False, True]),  # Prioritize unconstrained coefficients
        
        # Fixed parameters for reproducibility and performance
        'random_state': 42,
        'fit_intercept': True,
        'n_jobs': -1,
        'precompute': True  # Enable precomputation for faster convergence
    }
    
    trial.set_user_attr('model_params', {
        'alpha': params['alpha'],
        'alphas': params['alphas'],
        'fit_intercept': params['fit_intercept'],
        'precompute': params['precompute'],
        'l1_ratio': params['l1_ratio'],
        'max_iter': params['max_iter'],
        'tol': params['tol'],
        'eps': params['eps'],
        'selection': params['selection'],
        'positive': params['positive'],
        'feature_selection_threshold': params['feature_selection_threshold'],
        'poly_degree': params['poly_degree']
    })
    return params

def run_elasticnet_hypertuning():
    """Run hyperparameter tuning for ElasticNet model."""
    start_time = time.time()
    exp_logger.info("Starting ElasticNet hypertuning script...")
    
    # Load and prepare data
    data_load_start = time.time()
    exp_logger.info("Loading soccer prediction data")
    data_loader = DataLoader(experiment_name="elasticnet_hypertuning")
    X_train, y_train, X_test, y_test, X_val, y_val = data_loader.load_data()
    
    # Convert targets to numpy arrays if they aren't already
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_val = np.array(y_val)
    # Ensure data is in DataFrame format with proper column names
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
    if not isinstance(X_val, pd.DataFrame):
        X_val = pd.DataFrame(X_val, columns=X_train.columns)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=X_train.columns)
        
    # Scale features to ensure consistent scaling
    scaler = RobustScaler() 
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    data_load_time = time.time() - data_load_start
    
    exp_logger.info(
        f"Data loaded successfully:"
        f"\n - Training set: {X_train.shape}"
        f"\n - Test set: {X_test.shape}"
        f"\n - Validation set: {X_val.shape}"
        f"\n - Draw rate (train): {(y_train == 1).mean():.2%}"
        f"\n - Draw rate (test): {(y_test == 1).mean():.2%}"
        f"\n - Draw rate (val): {(y_val == 1).mean():.2%}"
        f"\n - Features: {list(X_train.columns)[:5]}..."  # Log first few feature names
    )
    
    # Create study with improved configuration
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(
            seed=42,
            n_startup_trials=10,  # Increased for better exploration
            multivariate=True  # Enable multivariate TPE
        ),
        pruner=MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=1
        )
    )
    
    # Run optimization with improved callback
    optimization_start = time.time()
    study.optimize(
        lambda trial: objective(trial, X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test),
        n_trials=200,
        timeout=3600,
        callbacks=[
            lambda study, trial: log_trial_status(study, trial, exp_logger)
        ]
    )
    optimization_time = time.time() - optimization_start
    
    # Train final model with best parameters
    exp_logger.info(f"Training final model with best parameters: {study.best_params}")
    final_training_start = time.time()
    
    model = ElasticNetModel()
    best_metrics = model.fit(X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, **study.best_params)
    final_training_time = time.time() - final_training_start
    
    # Prepare and log detailed results
    results = prepare_results(
        study=study,
        best_metrics=best_metrics,
        timings={
            'data_loading': data_load_time,
            'optimization': optimization_time,
            'final_training': final_training_time,
            'total': time.time() - start_time
        }
    )
    
    log_results(results, exp_logger)
    save_results(results)
    
    return results

def objective(trial, X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test):
    """Optuna objective function with focus on precision and convergence monitoring."""
    try:
        # Get hyperparameters for this trial
        params = create_param_space(trial)
        
        # Create and fit model
        model = ElasticNetModel()
        
        # Train model and get metrics
        try:
            metrics = model.fit(
                X_train_scaled, y_train,
                X_val_scaled, y_val,
                X_test_scaled, y_test,
                **params
            )
            
            # Check for convergence issues
            if metrics.get('converged', True) is False:
                exp_logger.warning(
                    f"Convergence failure in trial {trial.number}. "
                    f"Current parameters: alpha={params['alpha']}, l1_ratio={params['l1_ratio']}"
                )
                return 0.0
            
            # Calculate objective value with emphasis on precision
            precision = metrics['precision']
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
                'alpha_best': metrics.get('alpha_best', 0.0),
                'l1_ratio_best': metrics.get('l1_ratio_best', 0.0),
                'base_alpha': params['alpha'],
                'n_features': metrics.get('n_selected_features', 0)
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
            
        except Exception as e:
            exp_logger.error(f"Error in model fitting: {str(e)}")
            return 0.0
        
    except Exception as e:
        exp_logger.error(f"Error in trial {trial.number}: {str(e)}")
        return 0.0

def log_trial_status(study, trial, logger):
    """Log detailed trial information."""
    if trial.state == optuna.trial.TrialState.COMPLETE:
        logger.info(
            f"\nTrial {trial.number} completed:"
            f"\n - Value: {trial.value:.4f}"
            f"\n - Precision: {trial.user_attrs.get('precision', 0.0):.4f}"
            f"\n - Recall: {trial.user_attrs.get('recall', 0.0):.4f}"
            f"\n - Features: {trial.user_attrs.get('n_features', 0)}"
            f"\n - Parameters: {trial.params}"
        )

def prepare_results(study, best_metrics, timings):
    """Prepare detailed results dictionary."""
    return {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'timing': timings,
        'best_metrics': best_metrics,
        'n_trials': len(study.trials),
        'study_statistics': {
            'best_trial': study.best_trial.number,
            'n_finished': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'n_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        }
    }

def log_results(results, logger):
    """Log detailed results."""
    logger.info("\nHyperparameter Tuning Results:")
    logger.info("=" * 45)
    
    logger.info("\nTiming Information:")
    logger.info("-" * 30)
    for timing_key, timing_value in results['timing'].items():
        logger.info(f"{timing_key}: {timing_value:.2f} seconds")
    
    logger.info("\nBest Parameters:")
    logger.info("-" * 30)
    for param, value in results['best_params'].items():
        logger.info(f"{param}: {value}")
    
    logger.info("\nBest Model Metrics:")
    logger.info("-" * 30)
    for metric, value in results['best_metrics'].items():
        logger.info(f"{metric}: {value}")
    
    logger.info("\nStudy Statistics:")
    logger.info("-" * 30)
    stats = results['study_statistics']
    logger.info(f"Total trials: {results['n_trials']}")
    logger.info(f"Completed trials: {stats['n_finished']}")
    logger.info(f"Pruned trials: {stats['n_pruned']}")
    logger.info(f"Best trial: #{stats['best_trial']}")

def set_params(model, params):
    """Set model parameters.
    
    Args:
        model: ElasticNetModel instance
        params: Dict of parameters to set
    """
    # Ensure proper parameter types
    params['max_iter'] = int(params['max_iter'])
    params['n_alphas'] = int(params['n_alphas']) 
    params['tol'] = float(params['tol'])
    params['l1_ratio'] = float(params['l1_ratio'])
    params['feature_selection_threshold'] = float(params['feature_selection_threshold'])
    
    # Create new model with parameters
    model.model = model._create_model(**params)
    return model

def get_best_params(study):
    """Get best parameters from completed study."""
    return {
        'l1_ratio': study.best_params['l1_ratio'],
        'max_iter': int(study.best_params['max_iter']),
        'tol': float(study.best_params['tol']), 
        'n_alphas': int(study.best_params['n_alphas']),
        'feature_selection_threshold': float(study.best_params['feature_selection_threshold'])
    }

def save_results(results):
    """Save hypertuning results to a JSON file.
    
    Args:
        results: Dictionary containing hypertuning results
    """
    try:
        # Convert numpy types to native Python types
        results = convert_types(results)
        
        # Create results directory if it doesn't exist
        results_dir = Path(project_root) / "results" / "hypertuning"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"elasticnet_hypertuning_{timestamp}.json"
        filepath = results_dir / filename
        
        # Save results
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)
            
        exp_logger.info(f"Results saved to: {filepath}")
        
    except Exception as e:
        exp_logger.error(f"Error saving results: {str(e)}")
        raise

if __name__ == '__main__':
    try:
        exp_logger.info("Starting ElasticNet hypertuning script...")
        results = run_elasticnet_hypertuning()
        
        exp_logger.info("\nHyperparameter Tuning Results:")
        exp_logger.info("=" * 50)
        
        exp_logger.info("\nTiming Information:")
        exp_logger.info("-" * 30)
        for timing_key, timing_value in results['timing'].items():
            exp_logger.info(f"{timing_key}: {timing_value:.2f} seconds")
        
        exp_logger.info("\nBest Parameters:")
        exp_logger.info("-" * 30)
        for param, value in results['best_params'].items():
            exp_logger.info(f"{param}: {value}")
        
        exp_logger.info("\nTraining Metrics:")
        exp_logger.info("-" * 30)
        for metric, value in results['best_metrics'].items():
            exp_logger.info(f"{metric}: {value:.4f}")
        
    except Exception as e:
        exp_logger.error(f"\nError: {str(e)}")
        traceback.print_exc()
        sys.exit(1) 