"""
MLflow integration utilities for the Model Context Protocol Server.

This module provides:
- MLflow experiment and run management
- Model logging and artifact tracking
- Integration with ExperimentLogger for structured logging
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json

# MLflow imports
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models.signature import ModelSignature

# Add project root to path
try:
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
except Exception as e:
    print(f"Error setting project root path: {e}")
    sys.path.append(os.getcwd())

# Local imports
from utils.logger import ExperimentLogger

# Initialize logger
logger = ExperimentLogger(
    experiment_name="mlflow_integration",
    log_dir="logs/mlflow_integration"
)

class MLflowIntegration:
    """MLflow integration for experiment tracking and model management."""
    
    def __init__(self):
        """Initialize MLflow integration."""
        self.client = MlflowClient()
        self.mlruns_dir = project_root / "mlruns"
        self.mlruns_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up MLflow tracking
        mlflow.set_tracking_uri(f"file://{self.mlruns_dir}")
        
    def setup_experiment(
        self,
        experiment_name: str,
        artifact_location: Optional[str] = None
    ) -> str:
        """Set up MLflow experiment.
        
        Args:
            experiment_name: Name of the experiment
            artifact_location: Optional custom artifact location
            
        Returns:
            Experiment ID
        """
        try:
            # Get or create experiment
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                if artifact_location:
                    experiment_id = mlflow.create_experiment(
                        experiment_name,
                        artifact_location=artifact_location
                    )
                else:
                    experiment_id = mlflow.create_experiment(experiment_name)
                    
                mlflow.set_experiment(experiment_name)
                logger.info(
                    f"Created new experiment: {experiment_name}",
                    extra={"experiment_id": experiment_id}
                )
            else:
                experiment_id = experiment.experiment_id
                mlflow.set_experiment(experiment_name)
                logger.info(
                    f"Using existing experiment: {experiment_name}",
                    extra={"experiment_id": experiment_id}
                )
                
            return experiment_id
            
        except Exception as e:
            logger.error(f"Error setting up experiment: {str(e)}")
            raise
            
    def start_run(
        self,
        run_name: Optional[str] = None,
        experiment_id: Optional[str] = None,
        nested: bool = False,
        tags: Optional[Dict[str, str]] = None
    ) -> mlflow.ActiveRun:
        """Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run
            experiment_id: Optional experiment ID
            nested: Whether this is a nested run
            tags: Optional tags for the run
            
        Returns:
            MLflow ActiveRun context
        """
        try:
            # Start run with specified parameters
            run = mlflow.start_run(
                run_name=run_name,
                experiment_id=experiment_id,
                nested=nested,
                tags=tags
            )
            
            logger.info(
                f"Started MLflow run: {run_name or 'unnamed'}",
                extra={
                    "run_id": run.info.run_id,
                    "experiment_id": run.info.experiment_id
                }
            )
            
            return run
            
        except Exception as e:
            logger.error(f"Error starting run: {str(e)}")
            raise
            
    def log_params(
        self,
        params: Dict[str, Any],
        run_id: Optional[str] = None
    ) -> None:
        """Log parameters to MLflow.
        
        Args:
            params: Dictionary of parameters to log
            run_id: Optional run ID (uses active run if not specified)
        """
        try:
            if run_id:
                with mlflow.start_run(run_id=run_id):
                    mlflow.log_params(params)
            else:
                mlflow.log_params(params)
                
            logger.info(
                "Logged parameters to MLflow",
                extra={"params": params}
            )
            
        except Exception as e:
            logger.error(f"Error logging parameters: {str(e)}")
            raise
            
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        run_id: Optional[str] = None
    ) -> None:
        """Log metrics to MLflow.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
            run_id: Optional run ID (uses active run if not specified)
        """
        try:
            if run_id:
                with mlflow.start_run(run_id=run_id):
                    mlflow.log_metrics(metrics, step=step)
            else:
                mlflow.log_metrics(metrics, step=step)
                
            logger.info(
                "Logged metrics to MLflow",
                extra={"metrics": metrics, "step": step}
            )
            
        except Exception as e:
            logger.error(f"Error logging metrics: {str(e)}")
            raise
            
    def log_model(
        self,
        model: Any,
        artifact_path: str,
        flavor: str = "sklearn",
        signature: Optional[ModelSignature] = None,
        run_id: Optional[str] = None,
        **kwargs
    ) -> None:
        """Log a model to MLflow.
        
        Args:
            model: Model object to log
            artifact_path: Path for model artifact
            flavor: MLflow model flavor to use
            signature: Optional model signature
            run_id: Optional run ID (uses active run if not specified)
            **kwargs: Additional arguments for model logging
        """
        try:
            # Select appropriate logging function based on flavor
            if flavor == "sklearn":
                log_func = mlflow.sklearn.log_model
            elif flavor == "xgboost":
                log_func = mlflow.xgboost.log_model
            elif flavor == "catboost":
                log_func = mlflow.catboost.log_model
            else:
                raise ValueError(f"Unsupported model flavor: {flavor}")
                
            # Log model with specified parameters
            if run_id:
                with mlflow.start_run(run_id=run_id):
                    if signature:
                        log_func(
                            model,
                            artifact_path,
                            signature=signature,
                            **kwargs
                        )
                    else:
                        log_func(model, artifact_path, **kwargs)
            else:
                if signature:
                    log_func(
                        model,
                        artifact_path,
                        signature=signature,
                        **kwargs
                    )
                else:
                    log_func(model, artifact_path, **kwargs)
                    
            logger.info(
                f"Logged {flavor} model to MLflow",
                extra={
                    "artifact_path": artifact_path,
                    "has_signature": signature is not None
                }
            )
            
        except Exception as e:
            logger.error(f"Error logging model: {str(e)}")
            raise
            
    def log_artifact(
        self,
        local_path: Union[str, Path],
        artifact_path: Optional[str] = None,
        run_id: Optional[str] = None
    ) -> None:
        """Log an artifact to MLflow.
        
        Args:
            local_path: Path to artifact file
            artifact_path: Optional path within artifact directory
            run_id: Optional run ID (uses active run if not specified)
        """
        try:
            if run_id:
                with mlflow.start_run(run_id=run_id):
                    mlflow.log_artifact(local_path, artifact_path)
            else:
                mlflow.log_artifact(local_path, artifact_path)
                
            logger.info(
                "Logged artifact to MLflow",
                extra={
                    "local_path": str(local_path),
                    "artifact_path": artifact_path
                }
            )
            
        except Exception as e:
            logger.error(f"Error logging artifact: {str(e)}")
            raise
            
    def get_run_context(self, run_id: str) -> Dict[str, Any]:
        """Get context information for a run.
        
        Args:
            run_id: MLflow run ID
            
        Returns:
            Dictionary with run context information
        """
        try:
            run = self.client.get_run(run_id)
            
            context = {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "artifact_uri": run.info.artifact_uri,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags
            }
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting run context: {str(e)}")
            raise
            
    def load_model(
        self,
        run_id: str,
        model_path: str = "model"
    ) -> Any:
        """Load a model from MLflow.
        
        Args:
            run_id: MLflow run ID
            model_path: Path to model within run artifacts
            
        Returns:
            Loaded model object
        """
        try:
            model_uri = f"runs:/{run_id}/{model_path}"
            model = mlflow.pyfunc.load_model(model_uri)
            
            logger.info(
                "Loaded model from MLflow",
                extra={"run_id": run_id, "model_path": model_path}
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

def setup_mlflow(base_path: Optional[Union[str, Path]] = None) -> MLflowIntegration:
    """Set up MLflow integration.
    
    Args:
        base_path: Optional base path for MLflow files
        
    Returns:
        Configured MLflowIntegration instance
    """
    try:
        # Use provided base path or default to project root
        if base_path:
            mlruns_dir = Path(base_path) / "mlruns"
        else:
            mlruns_dir = project_root / "mlruns"
            
        # Ensure directory exists
        mlruns_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up MLflow tracking
        mlflow.set_tracking_uri(f"file://{mlruns_dir}")
        
        logger.info(
            "MLflow setup complete",
            extra={"tracking_uri": mlflow.get_tracking_uri()}
        )
        
        return MLflowIntegration()
        
    except Exception as e:
        logger.error(f"Error setting up MLflow: {str(e)}")
        raise 