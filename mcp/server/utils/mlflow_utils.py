"""
MLflow integration utilities for MCP server.
Provides experiment tracking and model management functionality.
"""

import os
from typing import Any, Dict, Optional, List
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime
from fastmcp import Context

from .logger import ExperimentLogger

class MCPMLflow:
    """
    MLflow integration for MCP server with experiment tracking and model management.
    Enforces CPU-only operations and structured logging.
    """
    
    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: str = "mcp_server",
        artifact_location: Optional[str] = None
    ):
        """
        Initialize MLflow tracking.
        
        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Name of the experiment
            artifact_location: Location for storing artifacts
        """
        self.logger = ExperimentLogger()
        
        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Initialize client
        self.client = MlflowClient()
        
        # Set up experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                # Create new experiment
                experiment_id = mlflow.create_experiment(
                    name=experiment_name,
                    artifact_location=artifact_location
                )
                self.experiment = mlflow.get_experiment(experiment_id)
            else:
                self.experiment = experiment
            
            self.experiment_id = self.experiment.experiment_id
            self.logger.info(
                f"MLflow experiment initialized: {experiment_name}",
                extra={"experiment_id": self.experiment_id}
            )
        except Exception as e:
            self.logger.error(
                f"Failed to initialize MLflow experiment: {str(e)}",
                extra={"experiment_name": experiment_name}
            )
            raise

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> mlflow.ActiveRun:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for the run
            tags: Run tags
            
        Returns:
            MLflow ActiveRun context
        """
        if not run_name:
            run_name = f"mcp_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        # Add default tags
        if tags is None:
            tags = {}
        tags.update({
            "mcp_version": "1.0.0",
            "device": "cpu",  # Enforce CPU-only
            "timestamp": datetime.now().isoformat()
        })
        
        try:
            run = mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=run_name,
                tags=tags
            )
            self.logger.info(
                f"Started MLflow run: {run_name}",
                extra={"run_id": run.info.run_id}
            )
            return run
        except Exception as e:
            self.logger.error(
                f"Failed to start MLflow run: {str(e)}",
                extra={"run_name": run_name}
            )
            raise

    def log_context_update(
        self,
        context: Context,
        step: Optional[int] = None
    ) -> None:
        """
        Log context updates to MLflow.
        
        Args:
            context: Context object containing data to log
            step: Optional step number
        """
        try:
            context_data = context.data or {}
            
            # Log metrics
            if "metrics" in context_data:
                mlflow.log_metrics(context_data["metrics"], step=step)
            
            # Log parameters
            if "parameters" in context_data:
                mlflow.log_params(context_data["parameters"])
            
            # Log artifacts
            if "artifacts" in context_data:
                for name, artifact in context_data["artifacts"].items():
                    mlflow.log_artifact(artifact, name)
            
            self.logger.info(
                "Logged context update to MLflow",
                extra={"step": step, "request_id": context.request_id}
            )
        except Exception as e:
            self.logger.error(
                f"Failed to log context update: {str(e)}",
                extra={"step": step, "request_id": context.request_id}
            )
            raise

    def get_run_context(self, run_id: str) -> Dict[str, Any]:
        """
        Get context data for a specific run.
        
        Args:
            run_id: MLflow run ID
            
        Returns:
            Dictionary containing run context data
        """
        try:
            run = self.client.get_run(run_id)
            context = {
                "run_id": run_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "metrics": run.data.metrics,
                "parameters": run.data.params,
                "tags": run.data.tags
            }
            
            self.logger.info(
                f"Retrieved context for run: {run_id}",
                extra={"status": run.info.status}
            )
            return context
        except Exception as e:
            self.logger.error(
                f"Failed to get run context: {str(e)}",
                extra={"run_id": run_id}
            )
            raise

    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        List all experiments.
        
        Returns:
            List of experiment information dictionaries
        """
        try:
            experiments = self.client.search_experiments()
            experiment_list = []
            
            for exp in experiments:
                experiment_list.append({
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "artifact_location": exp.artifact_location,
                    "lifecycle_stage": exp.lifecycle_stage
                })
            
            self.logger.info(
                f"Listed {len(experiment_list)} experiments",
                extra={"count": len(experiment_list)}
            )
            return experiment_list
        except Exception as e:
            self.logger.error(f"Failed to list experiments: {str(e)}")
            raise

# Create default MLflow instance
default_mlflow = MCPMLflow() 