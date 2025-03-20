"""
SVM Model Tuning

This script tunes a Support Vector Machine (SVM) model for the soccer prediction project using Optuna.
It performs hyperparameter optimization to find the best configuration
that maximizes precision while maintaining a minimum recall threshold.

The tuned model can be used as an additional base learner in the ensemble model.
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import mlflow
import mlflow.sklearn
from datetime import datetime
import os
import sys
from pathlib import Path
import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Add project root to Python path
try:
    project_root = Path(__file__).parent.parent.parent
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

# Local imports
from utils.logger import ExperimentLogger
from utils.create_evaluation_set import create_ensemble_evaluation_set, import_selected_features_ensemble, import_training_data_ensemble, setup_mlflow_tracking

# Set up logger and MLflow tracking
experiment_name = "svm_tuning"
logger = ExperimentLogger(experiment_name=experiment_name, log_dir='./logs/svm_tuning')
mlruns_dir = setup_mlflow_tracking(experiment_name)

def optimize_threshold(y_true, y_prob, min_recall=0.40, grid_start=0.3, grid_stop=0.8, grid_step=0.01):
    """
    Optimize prediction threshold to maximize precision while ensuring minimum recall.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        min_recall: Minimum recall threshold
        grid_start: Start of threshold grid
        grid_stop: End of threshold grid
        grid_step: Step size for threshold grid
        
    Returns:
        Tuple of (best_threshold, precision, recall)
    """
    best_threshold = 0.5
    best_precision = 0.0
    best_recall = 0.0
    
    # Search through thresholds
    for threshold in np.arange(grid_start, grid_stop, grid_step):
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calculate confusion matrix components
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        # Calculate precision and recall
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        
        # Only consider thresholds that maintain recall above minimum
        if recall >= min_recall:
            if precision > best_precision:
                best_precision = precision
                best_recall = recall
                best_threshold = threshold
                    
    logger.info(f"Optimized threshold: {best_threshold:.3f} with precision: {best_precision:.3f} and recall: {best_recall:.3f}")
    return best_threshold, best_precision, best_recall

def objective(trial, X_train, y_train, X_val, y_val, min_recall=0.40):
    """
    Optuna objective function for SVM hyperparameter optimization.
    
    Args:
        trial: Optuna trial
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        min_recall: Minimum recall threshold
        
    Returns:
        Score (precision if recall >= min_recall, otherwise 0)
    """
    # Handle missing values first
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    
    # Scale features (SVM is sensitive to feature scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_val_scaled = scaler.transform(X_val_imputed)
    
    # Define hyperparameter space for SVM
    params = {
        'C': trial.suggest_float('C', 1e-3, 1e3, log=True),
        'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly']),
        'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']) if trial.suggest_categorical('gamma_type', ['auto', 'value']) == 'auto' else trial.suggest_float('gamma_value', 1e-4, 1e0, log=True),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
        'tol': trial.suggest_float('tol', 1e-5, 1e-3, log=True),
        'shrinking': trial.suggest_categorical('shrinking', [True, False]),
        'cache_size': 10000,  # Use 10GB of cache for kernel calculations
    }
    
    # Add poly-specific parameters if kernel is poly
    if params['kernel'] == 'poly':
        params['degree'] = trial.suggest_int('degree', 2, 4)
        params['coef0'] = trial.suggest_float('coef0', 0.0, 1.0)
    
    # Remove gamma_type from params as it's just for suggestion logic
    if 'gamma_type' in params:
        del params['gamma_type']
    if 'gamma_value' in params:
        params['gamma'] = params.pop('gamma_value')
    
    # Create and train the SVM model with probability=True for threshold optimization
    model = SVC(probability=True, random_state=42, **params)
    try:
        logger.info(f"Training SVM with params: {params}")
        model.fit(X_train_scaled, y_train)
    except Exception as e:
        logger.error(f"Error training SVM model: {str(e)}")
        return 0.0
    
    # Get predictions on validation set
    y_prob = model.predict_proba(X_val_scaled)[:, 1]
    
    # Optimize threshold
    threshold, precision, recall = optimize_threshold(y_val, y_prob, min_recall=min_recall)
    
    # Log trial results
    logger.info(f"Trial {trial.number}:")
    logger.info(f"  Params: {params}")
    logger.info(f"  Threshold: {threshold:.3f}")
    logger.info(f"  Precision: {precision:.3f}")
    logger.info(f"  Recall: {recall:.3f}")
    
    # Calculate score (precision if recall >= min_recall, otherwise 0)
    score = precision if recall >= min_recall else 0.0
    
    return score

def optimize_hyperparameters(X_train, y_train, X_val, y_val, min_recall=0.40, n_trials=100):
    """
    Run hyperparameter optimization with Optuna.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        min_recall: Minimum recall threshold
        n_trials: Number of trials to run
        
    Returns:
        Tuple of (best_params, best_score)
    """
    logger.info("Starting hyperparameter optimization with Optuna...")
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(
            seed=42,
            consider_prior=True,
            prior_weight=0.8,
            n_startup_trials=10
        ),
        pruner=MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=5,
            interval_steps=2
        )
    )
    
    # Set Optuna to use all available CPU cores
    n_jobs = -1  # Use all available cores
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, min_recall),
        n_trials=n_trials,
        timeout=7200,  # 2 hour timeout
        show_progress_bar=True,
        n_jobs=n_jobs  # Parallelize trials across all cores
    )
    
    # Get best parameters
    best_params = study.best_params
    best_score = study.best_value
    
    # Handle gamma_value if it exists
    if 'gamma_value' in best_params:
        best_params['gamma'] = best_params.pop('gamma_value')
    if 'gamma_type' in best_params:
        del best_params['gamma_type']
    
    logger.info(f"Best parameters found: {best_params}")
    logger.info(f"Best score: {best_score}")
    
    return best_params, best_score

def train_svm(X_train, y_train, X_val, y_val, params, min_recall=0.40):
    """
    Train SVM model with the given parameters.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        params: Model parameters
        min_recall: Minimum recall threshold
        
    Returns:
        Tuple of (model, imputer, scaler, threshold, metrics)
    """
    logger.info("Training SVM model with best parameters...")
    
    # Handle missing values first
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_val_scaled = scaler.transform(X_val_imputed)
    
    # Ensure cache_size is set for maximum performance
    if 'cache_size' not in params:
        params['cache_size'] = 2000  # Use 2GB of cache for kernel calculations
    
    # Create and train model
    model = SVC(probability=True, random_state=42, **params)
    logger.info(f"Training on {len(X_train)} samples with cache_size={params.get('cache_size', 200)}MB")
    model.fit(X_train_scaled, y_train)
    
    # Get predictions on validation set
    y_prob = model.predict_proba(X_val_scaled)[:, 1]
    
    # Optimize threshold
    threshold, precision, recall = optimize_threshold(y_val, y_prob, min_recall=min_recall)
    
    # Calculate metrics
    y_pred = (y_prob >= threshold).astype(int)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    auc = roc_auc_score(y_val, y_prob)
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'threshold': threshold
    }
    
    logger.info(f"Model trained successfully.")
    logger.info(f"Validation metrics: {metrics}")
    
    return model, imputer, scaler, threshold, metrics

def main():
    """Main function to run the SVM tuning process."""
    logger.info("Loading data...")
    
    # Load data
    selected_features = import_selected_features_ensemble('all')
    X_train, y_train, X_test, y_test = import_training_data_ensemble()
    X_val, y_val = create_ensemble_evaluation_set()
    
    min_recall = 0.30
    n_trials = 100
    
    # Use only selected features
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    X_val = X_val[selected_features]
    
    # Get available system resources
    import psutil
    
    # Get CPU count and available memory
    cpu_count = os.cpu_count()
    available_memory = psutil.virtual_memory().available / (1024 * 1024)  # Available memory in MB
    
    # Calculate reasonable cache size (50% of available memory, max 8GB)
    cache_size = min(int(available_memory * 0.5), 10000)
    
    logger.info(f"System resources: {cpu_count} CPUs, {available_memory:.0f}MB available memory")
    logger.info(f"Setting SVM cache_size to {cache_size}MB")
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"svm_tuning_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        # Log resource configuration
        mlflow.log_params({
            "cpu_count": cpu_count,
            "available_memory_mb": int(available_memory),
            "cache_size_mb": cache_size
        })
        
        # Log dataset info
        mlflow.log_params({
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "val_samples": len(X_val),
            "n_features": len(selected_features),
            "min_recall_threshold": min_recall
        })
        
        # Set MLflow tags
        mlflow.set_tags({
            "model_type": "svm",
            "training_mode": "optuna",
            "cpu_only": True
        })
        
        # Optimize hyperparameters
        logger.info("Starting SVM hyperparameter optimization...")
        best_params, best_score = optimize_hyperparameters(
            X_train, y_train, X_val, y_val, min_recall=min_recall, n_trials=n_trials
        )
        logger.info("Hyperparameter optimization completed.")
        
        # Train model with best parameters
        best_model, imputer, scaler, best_threshold, metrics = train_svm(
            X_train, y_train, X_val, y_val, best_params, min_recall=min_recall
        )
        
        # Log best parameters to MLflow
        mlflow.log_params(best_params)
        mlflow.log_metric("best_optuna_score", best_score)
        mlflow.log_metric("threshold", best_threshold)
        
        # Log validation metrics to MLflow
        mlflow.log_metrics({
            "val_precision": metrics['precision'],
            "val_recall": metrics['recall'],
            "val_f1": metrics['f1'],
            "val_auc": metrics['auc']
        })
        
        # Create model signature and log model
        try:
            input_example = X_train.head(1).copy()
            # Apply the same preprocessing for the example
            input_example_imputed = imputer.transform(input_example)
            input_example_scaled = scaler.transform(input_example_imputed)
            
            signature = mlflow.models.infer_signature(
                model_input=input_example_scaled,
                model_output=best_model.predict(input_example_scaled)
            )
            
            # Log model to MLflow
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="svm_model",
                registered_model_name=f"svm_{datetime.now().strftime('%Y%m%d_%H%M')}",
                signature=signature
            )
            
            # Log preprocessing components separately
            mlflow.sklearn.log_model(
                sk_model=imputer,
                artifact_path="imputer"
            )
            
            mlflow.sklearn.log_model(
                sk_model=scaler,
                artifact_path="scaler"
            )
        except Exception as e:
            logger.error(f"Error creating model signature: {str(e)}")
            # Log model without signature
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="svm_model",
                registered_model_name=f"svm_{datetime.now().strftime('%Y%m%d_%H%M')}"
            )
            
            # Log preprocessing components separately
            mlflow.sklearn.log_model(
                sk_model=imputer,
                artifact_path="imputer"
            )
            
            mlflow.sklearn.log_model(
                sk_model=scaler,
                artifact_path="scaler"
            )
        
        # Print results
        print("\nBest SVM Parameters:")
        for param, value in best_params.items():
            print(f"{param}: {value}")
        
        print(f"\nValidation Metrics:")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"AUC: {metrics['auc']:.4f}")
        print(f"Threshold: {best_threshold:.4f}")
        
        # Provide code snippet for ensemble integration
        param_str = ", ".join([f"{k}={repr(v)}" for k, v in best_params.items()])
        print(f"""
self.model_extra = SVC(
    {param_str},
    probability=True,
    random_state=42
)
        """)

if __name__ == "__main__":
    main() 