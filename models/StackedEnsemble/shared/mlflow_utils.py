"""MLflow utilities for experiment tracking and model management."""

from typing import Dict, Optional, Any
import os
from pathlib import Path
import mlflow
from utils.logger import ExperimentLogger


class MLFlowManager:
    """Manages MLflow experiment tracking and model logging."""
    
    def __init__(self, base_experiment_name: str):
        """Initialize MLflow manager.
        
        Args:
            base_experiment_name: Base name for the experiment
        """
        self.logger = ExperimentLogger(experiment_name=base_experiment_name)
        self.base_experiment_name = base_experiment_name
        self.current_run = None
        
    def setup_model_experiment(self, model_type: str) -> str:
        """Setup MLflow experiment for specific model type.
        
        Args:
            model_type: Type of model (e.g., 'xgboost', 'lightgbm')
            
        Returns:
            Path to mlruns directory
        """
        from utils.create_evaluation_set import setup_mlflow_tracking
        experiment_name = f"{self.base_experiment_name}_{model_type}"
        mlruns_dir = setup_mlflow_tracking(experiment_name)
        self.logger.info(f"Set up MLflow experiment: {experiment_name}")
        return mlruns_dir
        
    def start_run(self, run_name: Optional[str] = None):
        """Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run
        """
        self.current_run = mlflow.start_run(run_name=run_name)
        self.logger.info(f"Started MLflow run: {self.current_run.info.run_id}")
        
    def end_run(self):
        """End the current MLflow run."""
        if self.current_run:
            mlflow.end_run()
            self.logger.info(f"Ended MLflow run: {self.current_run.info.run_id}")
            self.current_run = None
            
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow.
        
        Args:
            params: Dictionary of parameters to log
        """
        mlflow.log_params(params)
        self.logger.info(f"Logged parameters: {params}")
        
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLflow.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number for the metrics
        """
        mlflow.log_metrics(metrics, step=step)
        # self.logger.info(f"Logged metrics at step {step}: {metrics}")
        
    def log_split_metrics(self, metrics: Dict[str, float], split: str):
        """Log metrics with split identifier.
        
        Args:
            metrics: Dictionary of metrics to log
            split: Split identifier ('train', 'test', or 'validation')
        """
        split_metrics = {f"{split}_{k}": v for k, v in metrics.items()}
        self.log_metrics(split_metrics)
        self.logger.info(f"Logged {split} split metrics: {metrics}")
        
    def log_model(self, model: Any, artifact_path: str, registered_model_name: Optional[str] = None):
        """Log model to MLflow.
        
        Args:
            model: Model to log
            artifact_path: Path to store the model artifact
            registered_model_name: Optional name to register the model under
        """
        mlflow.sklearn.log_model(
            model,
            artifact_path,
            registered_model_name=registered_model_name
        )
        self.logger.info(
            f"Logged model to {artifact_path}"
            + (f" and registered as {registered_model_name}" if registered_model_name else "")
        )
        
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log an artifact to MLflow.
        
        Args:
            local_path: Path to the artifact file
            artifact_path: Optional path for artifact in MLflow
        """
        mlflow.log_artifact(local_path, artifact_path)
        self.logger.info(
            f"Logged artifact from {local_path} "
            f"to {artifact_path or 'root'}"
        ) 