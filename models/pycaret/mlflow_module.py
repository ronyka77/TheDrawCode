"""
MLflow integration module for PyCaret soccer prediction.

This module handles MLflow experiment tracking and model registration.
"""

import os
import sys
from pathlib import Path
import mlflow
from datetime import datetime

# Add project root to Python path if needed
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from utils.logger import ExperimentLogger

# Setup logger
logger = ExperimentLogger(experiment_name="pycaret_mlflow_module")

def setup_mlflow_for_pycaret(experiment_name="pycaret_soccer_prediction"):
    """
    Configure MLflow for PyCaret experiments.
    
    Args:
        experiment_name (str): Name of the MLflow experiment
        
    Returns:
        str: Experiment ID
    """
    # Use the existing MLflow tracking URI if available
    try:
        from utils.create_evaluation_set import setup_mlflow_tracking
        mlflow_dir = setup_mlflow_tracking(experiment_name)
        logger.info(f"Using existing MLflow tracking directory: {mlflow_dir}")
    except (ImportError, AttributeError):
        # Fallback to local mlruns directory
        mlflow_dir = os.path.join(project_root, "mlruns")
        os.makedirs(mlflow_dir, exist_ok=True)
        mlflow.set_tracking_uri(f"file:{mlflow_dir}")
        logger.info(f"Created new MLflow tracking directory: {mlflow_dir}")
    
    # Set experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            artifact_location=f"file:{os.path.join(mlflow_dir, experiment_name)}"
        )
        logger.info(f"Created new MLflow experiment '{experiment_name}' with ID: {experiment_id}")
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing MLflow experiment '{experiment_name}' with ID: {experiment_id}")
    
    return experiment_id

def log_pycaret_model(model, model_name, metrics, feature_importance=None):
    """
    Log a PyCaret model to MLflow.
    
    Args:
        model: Trained PyCaret model
        model_name (str): Name of the model
        metrics (dict): Dictionary of metrics to log
        feature_importance (pd.DataFrame, optional): Feature importance DataFrame
        
    Returns:
        str: Run ID
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    registered_model_name = f"{model_name}_{timestamp}"
    
    # Log metrics to the active run instead of starting a new one
    # Log metrics
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)
    
    # Log model
    try:
        # Try to use the PyCaret model's native MLflow logging
        from pycaret.classification import save_model
        model_path = f"models/{registered_model_name}"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        save_model(model, model_path)
        mlflow.log_artifact(model_path)
        logger.info(f"Logged model artifacts to {model_path}")
    except Exception as e:
        logger.error(f"Error logging model: {str(e)}")
        # Fallback to generic logging
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=registered_model_name
        )
    
    # Get the current run ID
    current_run = mlflow.active_run()
    if current_run:
        run_id = current_run.info.run_id
        logger.info(f"Model {registered_model_name} logged to MLflow with run ID: {run_id}")
        return run_id
    else:
        logger.warning("No active MLflow run found when logging model")
        return None

def log_threshold_optimization_results(threshold, metrics, model_name):
    """
    Log threshold optimization results to MLflow.
    
    Args:
        threshold (float): Optimized threshold
        metrics (dict): Dictionary of metrics at the optimized threshold
        model_name (str): Name of the model
        
    Returns:
        None
    """
    # Use the active run instead of starting a new one
    # Log threshold as a parameter
    mlflow.log_param("optimal_threshold", threshold)
    
    # Log metrics at the optimized threshold
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(f"threshold_optimized_{metric_name}", metric_value)
    
    logger.info(f"Logged threshold optimization results for {model_name}")

def log_pycaret_experiment_summary(experiment_name, phase, models_info):
    """
    Log a summary of the PyCaret experiment.
    
    Args:
        experiment_name (str): Name of the experiment
        phase (int): Phase number
        models_info (list): List of dictionaries with model information
        
    Returns:
        None
    """
    # Use the active run instead of starting a new one
    mlflow.log_param("experiment_name", experiment_name)
    mlflow.log_param("phase", phase)
    
    # Log summary metrics
    best_precision = max([model["metrics"].get("precision", 0) for model in models_info])
    best_recall = max([model["metrics"].get("recall", 0) for model in models_info])
    best_f1 = max([model["metrics"].get("f1", 0) for model in models_info])
    
    mlflow.log_metric("best_precision", best_precision)
    mlflow.log_metric("best_recall", best_recall)
    mlflow.log_metric("best_f1", best_f1)
    
    # Create and log summary table
    import pandas as pd
    
    summary_data = []
    for model in models_info:
        summary_data.append({
            "model_name": model["name"],
            "precision": model["metrics"].get("precision", 0),
            "recall": model["metrics"].get("recall", 0),
            "f1": model["metrics"].get("f1", 0),
            "auc": model["metrics"].get("auc", 0),
            "threshold": model["metrics"].get("threshold", 0.5)
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = f"experiment_summary_phase{phase}.csv"
    summary_df.to_csv(summary_path, index=False)
    mlflow.log_artifact(summary_path)
    
    logger.info(f"Logged experiment summary for {experiment_name} phase {phase}")

def save_model_and_predictions(model, model_name, predictions=None):
    """
    Save a trained model and its predictions.
    
    Args:
        model: Trained model
        model_name (str): Name to save the model as
        predictions (pd.DataFrame, optional): Predictions dataframe
        
    Returns:
        str: Path to saved model
    """
    try:
        from pycaret.classification import save_model
    except ImportError:
        logger.error("PyCaret not installed. Please install with 'pip install pycaret[full]'")
        return None
    
    logger.info(f"Saving model: {model_name}")
    
    try:
        # Create models directory if it doesn't exist
        import os
        models_dir = 'models/saved'
        os.makedirs(models_dir, exist_ok=True)
        
        # Save model - PyCaret 3.3.2 compatible
        model_path = os.path.join(models_dir, model_name)
        save_model(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save predictions if provided
        if predictions is not None:
            try:
                preds_path = os.path.join(models_dir, f"{model_name}_predictions.csv")
                predictions.to_csv(preds_path, index=False)
                logger.info(f"Predictions saved to {preds_path}")
            except Exception as e:
                logger.error(f"Error saving predictions: {str(e)}")
        
        # Save model metadata
        try:
            import json
            from datetime import datetime
            
            metadata = {
                'model_name': model_name,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'pycaret_version': '3.3.2'
            }
            
            # Try to get model type
            try:
                from pycaret.classification import get_config
                model_type = str(type(model).__name__)
                metadata['model_type'] = model_type
            except:
                pass
            
            # Save metadata
            meta_path = os.path.join(models_dir, f"{model_name}_metadata.json")
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            logger.info(f"Model metadata saved to {meta_path}")
        except Exception as e:
            logger.warning(f"Could not save model metadata: {str(e)}")
        
        return model_path
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        return None 