"""
MLP Model Tuning

This script tunes a Multi-Layer Perceptron (MLP) model for the soccer prediction project using Optuna.
It performs hyperparameter optimization to find the best configuration
that maximizes precision while maintaining a minimum recall threshold.

The tuned model can be used as an additional base learner in the ensemble model.
"""

import numpy as np
import pandas as pd
import random
from sklearn.neural_network import MLPClassifier
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
import warnings
import ast
from sklearn.exceptions import ConvergenceWarning

# Suppress specific warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning, message=".*categorical distribution.*")

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
experiment_name = "mlp_tuning"
logger = ExperimentLogger(experiment_name=experiment_name, log_dir='./logs/mlp_tuning')
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

def parse_hidden_layers(hidden_layers_str):
    """
    Parse hidden layers string representation back to tuple.
    
    Args:
        hidden_layers_str: String representation of hidden layers
        
    Returns:
        Tuple of integers representing hidden layer sizes
    """
    try:
        # Convert string representation to actual tuple
        return ast.literal_eval(hidden_layers_str)
    except (ValueError, SyntaxError) as e:
        logger.error(f"Error parsing hidden layers: {str(e)}")
        # Default fallback
        return (100, 50)

def objective(trial, X_train_processed, y_train, X_val_processed, y_val, scaler, imputer, min_recall=0.40):
    """
    Optuna objective function for MLP hyperparameter optimization.
    
    Args:
        trial: Optuna trial
        X_train_processed: Combined training features (original train + test)
        y_train: Combined training labels (original train + test)
        X_val_processed: Validation features
        y_val: Validation labels
        scaler: StandardScaler for feature scaling
        imputer: SimpleImputer for handling missing values
        min_recall: Minimum recall threshold
        
    Returns:
        Score (precision if recall >= min_recall, otherwise 0)
    """
    # Define hidden layer options as strings to avoid Optuna warnings
    hidden_layer_options = [
        "(50,)", "(100,)", "(200,)", "(50, 25)", 
        "(100, 50)", "(200, 100)", "(100, 50, 25)", "(200, 100, 50)"
    ]
    
    # Define parameters
    hidden_layers_str = trial.suggest_categorical('hidden_layer_sizes_str', hidden_layer_options)
    hidden_layers = parse_hidden_layers(hidden_layers_str)
    
    params = {
        'hidden_layer_sizes': hidden_layers,
        'activation': trial.suggest_categorical('activation', ['logistic']),
        'solver': trial.suggest_categorical('solver', ['adam']),
        'alpha': trial.suggest_float('alpha', 0.001, 0.01, log=True),
        'learning_rate': trial.suggest_categorical('learning_rate', ['adaptive']),
        'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-3, log=True),
        'max_iter': trial.suggest_int('max_iter', 300, 500),
        'early_stopping': True,
        'validation_fraction': trial.suggest_float('validation_fraction', 0.15, 0.20),
        'beta_1': trial.suggest_float('beta_1', 0.80, 0.90),
        'beta_2': trial.suggest_float('beta_2', 0.99, 1.0),
        'epsilon': trial.suggest_float('epsilon', 1e-8, 3e-8, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
        'tol': trial.suggest_float('tol', 1e-5, 1e-3, log=True),
        'random_state': 42
    }
    
    # Create and train model
    model = MLPClassifier(**params)
    
    try:
        model.fit(X_train_processed, y_train)
        
        # Get predictions on validation set
        y_prob = model.predict_proba(X_val_processed)[:, 1]
        
        # Optimize threshold
        threshold, precision, recall = optimize_threshold(y_val, y_prob, min_recall=min_recall)
        
        # Calculate score (precision if recall >= min_recall, otherwise 0)
        score = precision if recall >= min_recall else 0
        
        logger.info(f"Trial {trial.number}: Score: {score:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        
        # Store additional information in trial user attributes
        trial.set_user_attr('threshold', threshold)
        trial.set_user_attr('precision', precision)
        trial.set_user_attr('recall', recall)
        trial.set_user_attr('hidden_layer_sizes', str(hidden_layers))
        
        return score
    
    except Exception as e:
        logger.error(f"Error in trial {trial.number}: {str(e)}")
        return 0

def process_data(X, scaler, imputer):
    """
    Process data by imputing missing values and scaling.
    
    Args:
        X: Input features
        scaler: StandardScaler for feature scaling
        imputer: SimpleImputer for handling missing values
        
    Returns:
        Processed features
    """
    # Check for NaN values
    if X.isna().any().any():
        logger.info(f"Found {X.isna().sum().sum()} NaN values in data. Replacing with zeros...")
        # Replace all NaN values with 0
        X = X.fillna(0)
        
        # Log that NaNs were replaced
        logger.info("All NaN values have been replaced with zeros")
        X_imputed = imputer.transform(X)
    else:
        X_imputed = X.values
    
    # Scale features
    X_scaled = scaler.transform(X_imputed)
    
    return X_scaled

def optimize_hyperparameters(X_train, y_train, X_val, y_val, scaler, imputer, min_recall=0.30, n_trials=100):
    """
    Run hyperparameter optimization with Optuna.
    
    Args:
        X_train: Combined training features (original train + test)
        y_train: Combined training labels (original train + test)
        X_val: Validation features
        y_val: Validation labels
        scaler: StandardScaler for feature scaling
        imputer: SimpleImputer for handling missing values
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
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, scaler, imputer, min_recall),
        n_trials=n_trials,
        timeout=7200,  # 2 hour timeout
        show_progress_bar=True
    )
    
    # Get best parameters
    best_params = study.best_params
    
    # Convert hidden_layer_sizes_str back to tuple
    if 'hidden_layer_sizes_str' in best_params:
        best_params['hidden_layer_sizes'] = parse_hidden_layers(best_params['hidden_layer_sizes_str'])
        del best_params['hidden_layer_sizes_str']
    
    best_params.update({
        'early_stopping': True,
        'random_state': 42
    })
    
    best_score = study.best_value
    
    logger.info(f"Best parameters found: {best_params}")
    logger.info(f"Best score: {best_score}")
    
    return best_params, best_score

def train_mlp(X_train_processed, y_train, X_val_processed, y_val, params, scaler, imputer, min_recall=0.30):
    """
    Train MLP model with the given parameters.
    
    Args:
        X_train: Combined training features (original train + test)
        y_train: Combined training labels (original train + test)
        X_val: Validation features
        y_val: Validation labels
        params: Model parameters
        scaler: StandardScaler for feature scaling
        imputer: SimpleImputer for handling missing values
        min_recall: Minimum recall threshold
        
    Returns:
        Tuple of (model, threshold, metrics)
    """
    logger.info("Training MLP model with best parameters...")
    
    # Create and train model on combined data
    model = MLPClassifier(**params)
    logger.info(f"Training on {len(X_train)} samples with hidden layers {params['hidden_layer_sizes']}")
    model.fit(X_train_processed, y_train)
    
    # Get predictions on validation set
    y_prob = model.predict_proba(X_val_processed)[:, 1]
    
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

def evaluate_random_states(X_train_processed, y_train, X_val_processed, y_val, scaler, imputer, min_recall=0.30, n_states=300):
    """
    Train and evaluate MLP models with different random states to find the most stable configuration.
    
    Args:
        X_train_processed: Processed training features
        y_train: Training labels
        X_val_processed: Processed validation features
        y_val: Validation labels
        scaler: StandardScaler for feature scaling
        imputer: SimpleImputer for handling missing values
        min_recall: Minimum recall threshold
        n_states: Number of random states to evaluate (default: 300)
        
    Returns:
        Tuple of (best_model, best_threshold, best_metrics, best_random_state)
    """
    logger.info(f"Evaluating {n_states} random states for MLP model...")
    
    best_precision = 0
    best_model = None
    best_threshold = 0.5
    best_metrics = {}
    best_random_state = 1
    
    parameters = {
        'hidden_layer_sizes': (50),
        'activation': 'logistic',
        'solver': 'adam',
        'alpha': 0.007712811947156352,
        'learning_rate': 'adaptive',
        'learning_rate_init': 0.00029662989987000704,
        'max_iter': 324,
        'early_stopping': True,
        'validation_fraction': 0.18566223936114976,
        'beta_1': 0.8760785048616898,
        'beta_2': 0.995612771975695,
        'epsilon': 2.33262447559419e-08,
        'batch_size': 64,
        'tol': 1.6435497475111308e-05
    }
    # Store all results for analysis
    results = []
    
    # Evaluate each random state
    
    for random_state in range(1, n_states + 1):
        logger.info(f"Evaluating random state {random_state} of {n_states}")
        # Update parameters with current random state
        current_params = parameters.copy()
        current_params['random_state'] = random_state
        # Set Python's random seed to match the model's random state
        random.seed(random_state)
        
        try:
            # Create and train model
            model = MLPClassifier(**current_params)
            model.fit(X_train_processed, y_train)
            
            # Get predictions on validation set
            y_prob = model.predict_proba(X_val_processed)[:, 1]
            
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
                'threshold': threshold,
                'random_state': random_state
            }
            
            # Store results
            results.append(metrics)
            
            # Update best model if this one is better
            if precision > best_precision:
                best_precision = precision
                best_model = model
                best_threshold = threshold
                best_metrics = metrics
                best_random_state = random_state
                logger.info(f"New best model found with random_state={random_state}, Precision={precision:.4f}, Recall={recall:.4f}")
            
            # Log progress every 10 states
            if random_state % 10 == 0:
                logger.info(f"Evaluated {random_state}/{n_states} random states. Current best Precision: {best_precision:.4f}")
                
        except Exception as e:
            logger.error(f"Error training model with random_state={random_state}: {str(e)}")
    
    # Create a DataFrame with all results for analysis
    results_df = pd.DataFrame(results)
    
    # Log summary statistics
    logger.info(f"Random state evaluation complete. Best random_state: {best_random_state}")
    logger.info(f"Best metrics: Precision={best_metrics['precision']:.4f}, Recall={best_metrics['recall']:.4f}, F1={best_metrics['f1']:.4f}, AUC={best_metrics['auc']:.4f}")
    
    # Calculate stability metrics
    stability_metrics = {
        'f1_mean': results_df['f1'].mean(),
        'f1_std': results_df['f1'].std(),
        'precision_mean': results_df['precision'].mean(),
        'precision_std': results_df['precision'].std(),
        'recall_mean': results_df['recall'].mean(),
        'recall_std': results_df['recall'].std(),
        'auc_mean': results_df['auc'].mean(),
        'auc_std': results_df['auc'].std(),
    }
    
    logger.info(f"Stability metrics: {stability_metrics}")
    
    return best_model, best_threshold, best_metrics, best_random_state, results_df, stability_metrics

def train_and_log_mlp_model(X_train_orig, X_test_orig, X_train, y_train, X_val, y_val, X_train_processed, X_val_processed, selected_features, scaler, imputer, min_recall, n_trials):
    """
    Train and log MLP model with MLflow tracking.
    
    Args:
        X_train_orig: Original training features
        X_test_orig: Original test features
        X_train: Combined training features
        y_train: Combined training labels
        X_val: Validation features
        y_val: Validation labels
        X_train_processed: Processed training features
        X_val_processed: Processed validation features
        selected_features: List of selected feature names
        scaler: StandardScaler for feature scaling
        imputer: SimpleImputer for handling missing values
        min_recall: Minimum recall threshold
        n_trials: Number of trials for hyperparameter optimization
        
    Returns:
        Tuple of (best_model, best_threshold, metrics)
    """
    # Start MLflow run
    with mlflow.start_run(run_name=f"mlp_tuning_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        # Log dataset info
        mlflow.log_params({
            "train_samples_original": len(X_train_orig),
            "test_samples_original": len(X_test_orig),
            "combined_train_samples": len(X_train),
            "val_samples": len(X_val),
            "n_features": len(selected_features),
            "min_recall_threshold": min_recall,
        })
        
        # Set MLflow tags
        mlflow.set_tags({
            "model_type": "mlp",
            "training_mode": "optuna",
            "cpu_only": True,
            "imputation_strategy": "mean"
        })
        
        # Optimize hyperparameters
        logger.info("Starting MLP hyperparameter optimization...")
        best_params, best_score = optimize_hyperparameters(
            X_train_processed, y_train, X_val_processed, y_val, scaler, imputer, min_recall=min_recall, n_trials=n_trials
        )
        logger.info("Hyperparameter optimization completed.")
        
        # Train model with best parameters
        best_model, best_threshold, metrics = train_mlp(
            X_train_processed, y_train, X_val_processed, y_val, best_params, scaler, imputer, min_recall=min_recall
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
            processed_example = process_data(input_example, scaler, imputer)
            signature = mlflow.models.infer_signature(
                model_input=processed_example,
                model_output=best_model.predict(processed_example)
            )
            
            # Log model to MLflow
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="mlp_model",
                registered_model_name=f"mlp_{datetime.now().strftime('%Y%m%d_%H%M')}",
                signature=signature
            )
            
            # Log preprocessing components
            mlflow.sklearn.log_model(
                sk_model=scaler,
                artifact_path="scaler"
            )
            
            mlflow.sklearn.log_model(
                sk_model=imputer,
                artifact_path="imputer"
            )
        except Exception as e:
            logger.error(f"Error creating model signature: {str(e)}")
            # Log model without signature
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="mlp_model",
                registered_model_name=f"mlp_{datetime.now().strftime('%Y%m%d_%H%M')}"
            )
            
            # Log preprocessing components
            mlflow.sklearn.log_model(
                sk_model=scaler,
                artifact_path="scaler"
            )
            
            mlflow.sklearn.log_model(
                sk_model=imputer,
                artifact_path="imputer"
            )
        
        # Print results
        print("\nBest MLP Parameters:")
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
            # For ensemble integration, you'll need the model, scaler, and imputer:
            from sklearn.neural_network import MLPClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.impute import SimpleImputer

            # Initialize the preprocessing components
            self.mlp_imputer = SimpleImputer(strategy='mean')
            self.mlp_imputer.fit(X_train)  # Fit on your training data

            self.mlp_scaler = StandardScaler()
            self.mlp_scaler.fit(self.mlp_imputer.transform(X_train))  # Fit on imputed data

            # Initialize the model
            self.model_extra = MLPClassifier(
                {param_str}
            )

            # When using in ensemble:
            # X_processed = self.mlp_scaler.transform(self.mlp_imputer.transform(X))
            # predictions = self.model_extra.predict_proba(X_processed)[:, 1]
        """)
        
        return best_model, best_threshold, metrics
    

def main():
    """Main function to run the MLP tuning process."""
    logger.info("Loading data...")
    
    # Load data
    selected_features = import_selected_features_ensemble('all')
    X_train_orig, y_train_orig, X_test_orig, y_test_orig = import_training_data_ensemble()
    X_val, y_val = create_ensemble_evaluation_set()
    
    min_recall = 0.30
    n_trials = 300
    # Use only selected features
    X_train_orig = X_train_orig[selected_features]
    X_test_orig = X_test_orig[selected_features]
    X_val = X_val[selected_features]
    
    # Combine training and test data
    X_train = pd.concat([X_train_orig, X_test_orig])
    y_train = pd.concat([y_train_orig, y_test_orig])

        # Create and fit imputer for handling missing values
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(X_train)
    
    # Create and fit scaler
    scaler = StandardScaler()
    scaler.fit(imputer.transform(X_train))

    # Process data
    X_train_processed = process_data(X_train, scaler, imputer)
    X_val_processed = process_data(X_val, scaler, imputer)
    logger.info(f"Combined training data: {len(X_train_processed)} samples ({len(X_train_orig)} from train, {len(X_test_orig)} from test)")
    

    # Call the function in main
    train_and_log_mlp_model(
        X_train_orig, X_test_orig, X_train, y_train, X_val, y_val, 
        X_train_processed, X_val_processed, selected_features, 
        scaler, imputer, min_recall, n_trials
    )

    # evaluate_random_states(X_train_processed, y_train, X_val_processed, y_val, scaler, imputer, min_recall=0.30, n_states=300)

if __name__ == "__main__":
    main() 