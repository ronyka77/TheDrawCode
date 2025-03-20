"""
Random Forest Model Tuning

This script tunes a Random Forest model for the soccer prediction project using Optuna.
It performs hyperparameter optimization to find the best configuration
that maximizes precision while maintaining a minimum recall threshold.

The tuned model can be used as an additional base learner in the ensemble model.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer, roc_auc_score
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
# Suppress specific sklearn warning about class weights
import warnings
warnings.filterwarnings('ignore', 
    message='class_weight presets "balanced" or "balanced_subsample" are not recommended for warm_start.*',
    category=UserWarning,
    module='sklearn.ensemble._forest')

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
# Set up logger and MLflow tracking
experiment_name = "random_forest_tuning"
logger = ExperimentLogger(experiment_name=experiment_name, log_dir='./logs/random_forest_tuning')
from utils.create_evaluation_set import create_ensemble_evaluation_set, import_selected_features_ensemble, import_training_data_ensemble, setup_mlflow_tracking

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
    Optuna objective function for Random Forest hyperparameter optimization.
    
    Args:
        trial: Optuna trial
        X_train: Combined training features (original train + test)
        y_train: Combined training labels (original train + test)
        X_val: Validation features
        y_val: Validation labels
        min_recall: Minimum recall threshold
        
    Returns:
        Score (precision if recall >= min_recall, otherwise 0)
    """
    # Define parameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),  # Allow higher values
        'max_depth': trial.suggest_int('max_depth', 14, 25),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 19),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 6),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', None, 'log2']),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced_subsample']),
        'criterion': trial.suggest_categorical('criterion', ['entropy', 'gini']),
        'random_state': 42,
        'n_jobs': 6,
        'warm_start': True  # Enable warm start
    }
    
    model = RandomForestClassifier(**params)
    
    # Parameters for early stopping
    max_estimators = params['n_estimators']
    step_size = max(10, max_estimators // 20)  # 5% of max_estimators or at least 10
    patience = 5
    
    best_score = 0
    best_n_estimators = step_size
    no_improvement_count = 0
    
    # Start with a small number of trees
    model.n_estimators = step_size
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_val)[:, 1]
    threshold, precision, recall = optimize_threshold(y_val, y_prob, min_recall=min_recall)
    score = precision if recall >= min_recall else 0
    best_score = score
    
    # logger.info(f"Trial {trial.number}: Trees: {step_size}, Score: {score:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    
    # Incrementally add more trees
    for n_estimators in range(2*step_size, max_estimators+1, step_size):
        model.n_estimators = n_estimators
        model.fit(X_train, y_train)
        
        y_prob = model.predict_proba(X_val)[:, 1]
        threshold, precision, recall = optimize_threshold(y_val, y_prob, min_recall=min_recall)
        score = precision if recall >= min_recall else 0
        
        logger.info(f"Trial {trial.number}: Trees: {n_estimators}, Score: {score:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        
        if score > best_score:
            best_score = score
            best_n_estimators = n_estimators
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            
        if no_improvement_count >= patience:
            logger.info(f"Trial {trial.number}: Early stopping at {n_estimators} trees. Best was {best_n_estimators} trees.")
            break
    
    # Log the optimal number of trees found
    trial.set_user_attr('best_n_estimators', best_n_estimators)
    
    return best_score

def optimize_hyperparameters(X_train, y_train, X_val, y_val, min_recall=0.40, n_trials=100):
    """
    Run hyperparameter optimization with Optuna.
    
    Args:
        X_train: Combined training features (original train + test)
        y_train: Combined training labels (original train + test)
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
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, min_recall),
        n_trials=n_trials,
        timeout=7200,  # 2 hour timeout
        show_progress_bar=True
    )
    
    # Get best parameters
    best_params = study.best_params
    best_params.update({
        'random_state': 42,
        'n_jobs': 6
    })
    
    best_score = study.best_value
    
    logger.info(f"Best parameters found: {best_params}")
    logger.info(f"Best score: {best_score}")
    
    return best_params, best_score

def train_random_forest(X_train, y_train, X_val, y_val, params, min_recall=0.40):
    """
    Train Random Forest model with the given parameters.
    
    Args:
        X_train: Combined training features (original train + test)
        y_train: Combined training labels (original train + test)
        X_val: Validation features
        y_val: Validation labels
        params: Model parameters
        min_recall: Minimum recall threshold
        
    Returns:
        Tuple of (model, threshold, metrics)
    """
    logger.info("Training Random Forest model with best parameters...")
    
    # Create and train model on combined data
    model = RandomForestClassifier(**params)
    logger.info(f"Training on {len(X_train)} samples with {params['n_estimators']} estimators")
    model.fit(X_train, y_train)
    
    # Get predictions on validation set
    y_prob = model.predict_proba(X_val)[:, 1]
    
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
    
    return model, threshold, metrics

def main():
    """Main function to run the Random Forest tuning process."""
    logger.info("Loading data...")
    
    # Load data
    selected_features = import_selected_features_ensemble('all')
    X_train_orig, y_train_orig, X_test_orig, y_test_orig = import_training_data_ensemble()
    X_val, y_val = create_ensemble_evaluation_set()
    
    min_recall = 0.30
    n_trials = 100
    # Use only selected features
    X_train_orig = X_train_orig[selected_features]
    X_test_orig = X_test_orig[selected_features]
    X_val = X_val[selected_features]
    
    # Combine training and test data
    X_train = pd.concat([X_train_orig, X_test_orig])
    y_train = pd.concat([y_train_orig, y_test_orig])
    
    logger.info(f"Combined training data: {len(X_train)} samples ({len(X_train_orig)} from train, {len(X_test_orig)} from test)")
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"random_forest_tuning_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        # Log dataset info
        mlflow.log_params({
            "train_samples_original": len(X_train_orig),
            "test_samples_original": len(X_test_orig),
            "combined_train_samples": len(X_train),
            "val_samples": len(X_val),
            "n_features": len(selected_features),
            "min_recall_threshold": 0.40
        })
        
        # Set MLflow tags
        mlflow.set_tags({
            "model_type": "random_forest",
            "training_mode": "optuna",
            "cpu_only": True
        })
        
        # Optimize hyperparameters
        logger.info("Starting Random Forest hyperparameter optimization...")
        best_params, best_score = optimize_hyperparameters(
            X_train, y_train, X_val, y_val, min_recall=min_recall, n_trials=n_trials
        )
        logger.info("Hyperparameter optimization completed.")
        
        # Train model with best parameters
        best_model, best_threshold, metrics = train_random_forest(
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
        
        # Create model signature
        try:
            input_example = X_train.head(1).copy()
            signature = mlflow.models.infer_signature(
                model_input=input_example,
                model_output=best_model.predict(input_example)
            )
            
            # Log model to MLflow
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="random_forest_model",
                registered_model_name=f"random_forest_{datetime.now().strftime('%Y%m%d_%H%M')}",
                signature=signature
            )
        except Exception as e:
            logger.error(f"Error creating model signature: {str(e)}")
            # Log model without signature
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="random_forest_model",
                registered_model_name=f"random_forest_{datetime.now().strftime('%Y%m%d_%H%M')}"
            )
        
        # Print results
        print("\nBest Random Forest Parameters:")
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
            self.model_extra = RandomForestClassifier(
                {param_str},
                random_state=42,
                n_jobs=-1
            )
        """)

if __name__ == "__main__":
    main() 