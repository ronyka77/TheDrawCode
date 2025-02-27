"""
Ensemble Model Runner

Main script for running the ensemble model training and evaluation.
"""

import os
import sys
import numpy as np
import pandas as pd
import mlflow
from pathlib import Path
import argparse
from datetime import datetime

# Add project root to Python path
try:
    project_root = Path(__file__).parent.parent.parent
    if not project_root.exists():
        # Handle network path by using raw string
        project_root = Path(r"\\".join(str(project_root).split("\\")))
    sys.path.append(str(project_root))
    print(f"Project root run_ensemble: {project_root}")
except Exception as e:
    print(f"Error setting project root path: {e}")
    # Fallback to current directory if path resolution fails
    sys.path.append(os.getcwd().parent)
    print(f"Current directory run_ensemble: {os.getcwd().parent}")

# Set environment variables for Git
os.environ['GIT_PYTHON_GIT_EXECUTABLE'] = "C:/Program Files/Git/bin/git.exe"

# Local imports
from utils.logger import ExperimentLogger
experiment_name = "ensemble_model_run_ensemble"
logger = ExperimentLogger(experiment_name=experiment_name,
                            log_dir="./logs/ensemble_model_run_ensemble")
from utils.create_evaluation_set import setup_mlflow_tracking, import_selected_features_ensemble

from models.ensemble.ensemble_model import EnsembleModel

def run_ensemble(extra_base_model_type: str = 'mlp',
                meta_learner_type: str = 'mlp',
                calibrate: bool = True,
                dynamic_weighting: bool = True,
                target_precision: float = 0.50,
                required_recall: float = 0.25,
                experiment_name: str = "ensemble_model_run_ensemble",
                time_based_split: bool = True,
                logger: ExperimentLogger = logger):
    """
    Main function to run the ensemble model training and evaluation.
    
    Args:
        extra_base_model_type: Type of fourth base model ('random_forest', 'svm', or 'mlp')
        meta_learner_type: Type of meta-learner ('xgb', 'logistic', or 'mlp')
        calibrate: Whether to calibrate base model probabilities
        dynamic_weighting: Whether to use dynamic weighting for base model probabilities
        target_precision: Target precision for threshold tuning
        required_recall: Minimum required recall for threshold tuning
        experiment_name: Name of the MLflow experiment
        time_based_split: Whether to use time-based data splits
    """
    
    # Set up MLflow tracking
    mlruns_dir = setup_mlflow_tracking(experiment_name)

    try:
        # Start MLflow run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"ensemble_run_{timestamp}"
        
        with mlflow.start_run(run_name=run_name):
            # Log run parameters
            mlflow.log_params({
                'extra_base_model_type': extra_base_model_type,
                'meta_learner_type': meta_learner_type,
                'calibrate': calibrate,
                'dynamic_weighting': dynamic_weighting,
                'target_precision': target_precision,
                'required_recall': required_recall,
                'time_based_split': time_based_split
            })
            
            logger.info("Starting ensemble model execution...")
            
            # Import data
            if time_based_split:
                # Time-based data loading
                try:
                    logger.info("Loading data with time-based splits...")
                    from models.StackedEnsemble.shared.data_loader import DataLoader
                    # X_train, y_train = historical data (oldest)
                    # X_test, y_test = intermediate data (for tuning)
                    # X_val, y_val = most recent data (for final evaluation)
                    X_train, y_train, X_test, y_test, X_val, y_val = DataLoader().load_data()
                except Exception as e:
                    logger.error(f"Error loading time-based data: {str(e)}")
                    logger.info("Falling back to standard data loading...")
                    time_based_split = False
            
            if not time_based_split:
                # Standard data loading
                logger.info("Loading data with random splits...")
                # Import selected features and data
                from utils.create_evaluation_set import load_training_data
                df = load_training_data()
                
                # Split features and target
                X = df.drop('target', axis=1)
                y = df['target']
                
                # Split into train/test/val
                from sklearn.model_selection import train_test_split
                X_train, X_temp, y_train, y_temp = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=y
                )
                X_test, X_val, y_test, y_val = train_test_split(
                    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
                )
            
            # Log dataset sizes
            logger.info(f"Dataset sizes - Training: {X_train.shape}, Test: {X_test.shape}, Validation: {X_val.shape}")
            mlflow.log_params({
                'train_size': len(X_train),
                'test_size': len(X_test),
                'val_size': len(X_val),
                'positive_rate_train': y_train.mean(),
                'positive_rate_test': y_test.mean(),
                'positive_rate_val': y_val.mean()
            })
            
            # Feature selection
            logger.info("Selecting features...")
            selected_features = import_selected_features_ensemble('all')
            
            # Filter features for all datasets
            X_train_filtered = X_train[selected_features]
            X_test_filtered = X_test[selected_features]
            X_val_filtered = X_val[selected_features]
            
            # Create ensemble with configuration
            ensemble_model = EnsembleModel(
                logger=logger,
                extra_base_model_type=extra_base_model_type,
                meta_learner_type=meta_learner_type,
                calibrate=calibrate,
                dynamic_weighting=dynamic_weighting,
                target_precision=target_precision,
                required_recall=required_recall
            )
            
            # Train the model
            logger.info("Training ensemble model...")
            training_results = ensemble_model.train(
                X_train=X_train_filtered, 
                y_train=y_train, 
                X_test=X_test_filtered, 
                y_test=y_test,
                X_val=X_val_filtered, 
                y_val=y_val,
                split_validation=False  # Don't split again, we already have splits
            )
            
            # Analyze prediction errors on validation set (most recent data)
            logger.info("Analyzing prediction errors on validation set...")
            error_analysis = ensemble_model.analyze_prediction_errors(X_val_filtered, y_val)
            
            # Explain model predictions on validation set
            logger.info("Explaining model predictions on validation set...")
            explanation = ensemble_model.explain_predictions(X_val_filtered)
            
            # Final metrics on validation set
            logger.info("Final metrics on validation set:")
            for metric, value in training_results.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {metric}: {value:.4f}")
            
            logger.info("Ensemble model execution completed successfully.")
            return ensemble_model
            
    except Exception as e:
        logger.error(f"Error in ensemble model execution: {str(e)}")
        raise

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the ensemble model training and evaluation.")
    parser.add_argument("--extra-model", type=str, default="mlp", choices=["random_forest", "svm", "mlp"],
                        help="Type of fourth base model")
    parser.add_argument("--meta-learner", type=str, default="xgb", choices=["xgb", "logistic", "mlp"],
                        help="Type of meta-learner")
    parser.add_argument("--calibrate", action="store_true", default=True,
                        help="Whether to calibrate base model probabilities")
    parser.add_argument("--dynamic-weighting", action="store_true", default=True,
                        help="Whether to use dynamic weighting for base model probabilities")
    parser.add_argument("--target-precision", type=float, default=0.50,
                        help="Target precision for threshold tuning")
    parser.add_argument("--required-recall", type=float, default=0.25,
                        help="Minimum required recall for threshold tuning")
    parser.add_argument("--experiment-name", type=str, default="ensemble_model_improved",
                        help="Name of the MLflow experiment")
    parser.add_argument("--time-based-split", action="store_true", default=True,
                        help="Whether to use time-based data splits")
    
    args = parser.parse_args()
    
    # Run the ensemble model
    run_ensemble(
        extra_base_model_type=args.extra_model,
        meta_learner_type=args.meta_learner,
        calibrate=args.calibrate,
        dynamic_weighting=args.dynamic_weighting,
        target_precision=args.target_precision,
        required_recall=args.required_recall,
        experiment_name=args.experiment_name,
        time_based_split=args.time_based_split
    )