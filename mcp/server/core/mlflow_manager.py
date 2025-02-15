"""
MLflow manager for MCP server.
Handles MLflow experiment tracking and model management.
"""

import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import yaml

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

from ..utils.logger import default_logger as logger

class MLflowManager:
    """
    Manages MLflow experiment tracking and model management.
    Provides methods for tracking experiments, runs, and models.
    """
    
    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: str = "mcp_server"
    ):
        """
        Initialize MLflow manager.
        
        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Default experiment name
        """
        self.start_time = datetime.now()
        
        # Set up default tracking URI in the project directory
        default_mlruns = os.path.join(os.getcwd(), "mlruns")
        os.makedirs(default_mlruns, exist_ok=True)
        
        # Clean up any malformed experiments before MLflow initialization
        self._cleanup_malformed_experiments(default_mlruns)
        
        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            mlflow.set_tracking_uri(f"file://{default_mlruns}")
        
        # Initialize client
        self.client = MlflowClient()
        
        try:
            # Set up experiment
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                # Create new experiment
                experiment_id = mlflow.create_experiment(name=experiment_name)
                self.experiment = mlflow.get_experiment(experiment_id)
                
                # Ensure meta.yaml exists
                self._create_meta_yaml(experiment_id, experiment_name)
            else:
                self.experiment = experiment
                # Verify meta.yaml exists for existing experiment
                self._create_meta_yaml(experiment.experiment_id, experiment_name)
            
            self.experiment_id = self.experiment.experiment_id
            logger.info(
                f"MLflow experiment initialized: {experiment_name}",
                extra_fields={"experiment_id": self.experiment_id}
            )
            
        except Exception as e:
            logger.error(
                f"Failed to initialize MLflow experiment: {str(e)}",
                extra_fields={"experiment_name": experiment_name}
            )
            self.experiment = None
            self.experiment_id = None

    def _cleanup_malformed_experiments(self, mlruns_dir: str) -> None:
        """Clean up malformed experiments by creating missing meta.yaml files."""
        try:
            # Handle special directories first
            special_dirs = ['models', '.trash']
            for special_dir in special_dirs:
                exp_dir = os.path.join(mlruns_dir, special_dir)
                if os.path.isdir(exp_dir):
                    meta_path = os.path.join(exp_dir, "meta.yaml")
                    if not os.path.exists(meta_path):
                        meta_data = {
                            "artifact_location": os.path.join(exp_dir, "artifacts"),
                            "experiment_id": special_dir,
                            "lifecycle_stage": "active",
                            "name": special_dir,
                            "creation_time": int(datetime.now().timestamp() * 1000)
                        }
                        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
                        with open(meta_path, 'w') as f:
                            yaml.safe_dump(meta_data, f)
                        logger.info(f"Created missing meta.yaml for experiment: {special_dir}")
            
            # Handle other experiment directories
            for item in os.listdir(mlruns_dir):
                if item not in special_dirs and os.path.isdir(os.path.join(mlruns_dir, item)):
                    exp_dir = os.path.join(mlruns_dir, item)
                    meta_path = os.path.join(exp_dir, "meta.yaml")
                    if not os.path.exists(meta_path):
                        meta_data = {
                            "artifact_location": os.path.join(exp_dir, "artifacts"),
                            "experiment_id": item,
                            "lifecycle_stage": "active",
                            "name": f"experiment_{item}",
                            "creation_time": int(datetime.now().timestamp() * 1000)
                        }
                        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
                        with open(meta_path, 'w') as f:
                            yaml.safe_dump(meta_data, f)
                        logger.info(f"Created missing meta.yaml for experiment: {item}")
                        
        except Exception as e:
            logger.warning(f"Error cleaning up experiments: {str(e)}")

    def _create_meta_yaml(self, experiment_id: str, experiment_name: str) -> None:
        """Create meta.yaml file for an experiment if it doesn't exist."""
        try:
            exp_dir = os.path.join(
                mlflow.get_tracking_uri().replace("file://", ""),
                str(experiment_id)
            )
            meta_path = os.path.join(exp_dir, "meta.yaml")
            
            if not os.path.exists(meta_path):
                meta_data = {
                    "artifact_location": os.path.join(exp_dir, "artifacts"),
                    "experiment_id": experiment_id,
                    "lifecycle_stage": "active",
                    "name": experiment_name,
                    "creation_time": int(datetime.now().timestamp() * 1000)
                }
                
                os.makedirs(os.path.dirname(meta_path), exist_ok=True)
                with open(meta_path, 'w') as f:
                    yaml.safe_dump(meta_data, f)
                logger.info(f"Created meta.yaml for experiment: {experiment_name}")
        except Exception as e:
            logger.warning(f"Error creating meta.yaml: {str(e)}")

    async def list_experiments(self) -> List[Dict[str, Any]]:
        """
        List all MLflow experiments.
        
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
                    "lifecycle_stage": exp.lifecycle_stage,
                    "creation_time": exp.creation_time,
                    "last_update_time": exp.last_update_time
                })
            
            logger.info(
                f"Listed {len(experiment_list)} experiments",
                extra_fields={"count": len(experiment_list)}
            )
            return experiment_list
            
        except Exception as e:
            logger.error(f"Failed to list experiments: {str(e)}")
            return []

    async def get_run_info(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information for a specific run.
        
        Args:
            run_id: MLflow run ID
            
        Returns:
            Dictionary containing run information
        """
        try:
            run = self.client.get_run(run_id)
            info = {
                "run_id": run_id,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "artifact_uri": run.info.artifact_uri,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags
            }
            
            logger.info(
                f"Retrieved run info: {run_id}",
                extra_fields={"status": run.info.status}
            )
            return info
            
        except Exception as e:
            logger.error(f"Failed to get run info: {str(e)}")
            return None

    def is_healthy(self) -> bool:
        """
        Check if MLflow manager is healthy.
        
        Returns:
            Boolean indicating health status
        """
        return (
            self.client is not None and
            self.experiment_id is not None
        )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get MLflow manager statistics.
        
        Returns:
            Dictionary containing statistics
        """
        return {
            "is_healthy": self.is_healthy(),
            "experiment_id": self.experiment_id,
            "uptime": (datetime.now() - self.start_time).total_seconds()
        } 