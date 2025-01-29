import xgboost as xgb
import numpy as np
from sklearn.datasets import make_classification
import os
import pandas as pd
import shutil
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
import sys
import time
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient

# Add project root to Python path
try:
    project_root = Path(__file__).parent.parent
    if not project_root.exists():
        # Handle network path by using raw string
        project_root = Path(r"\\".join(str(project_root).split("\\")))
    sys.path.append(str(project_root))
    print(f"Project root mlflow_utils: {project_root}")
except Exception as e:
    print(f"Error setting project root path: {e}")
    # Fallback to current directory if path resolution fails
    sys.path.append(os.getcwd().parent)
    print(f"Current directory mlflow_utils: {os.getcwd().parent}")

from utils.logger import ExperimentLogger

class MLFlowConfig:
    """MLflow configuration management"""
    
    def __init__(self):
        project_root = Path(__file__).resolve().parent.parent
        self.shared_path = "B:/mlflow/mlruns"
        self.local_path = project_root / "mlruns"
        self.local_path_uri = str(self.local_path).replace('\\', '/')
        self.shared_path_uri = str(self.shared_path).replace('\\', '/')
        
        # Ensure paths are absolute and normalized
        self.local_path = self.local_path.resolve()

        # Convert to proper MLflow URI format
        local_uri = f"file:///{self.local_path_uri}"
        shared_path_uri = f"file:///{self.shared_path_uri}"
        
        # print(f"local_uri: {local_uri}")
        # print(f"shared_path_uri: {shared_path_uri}")
        print(f"mlflow local_path_uri: {self.local_path_uri}")
        

        self.config = {
            "tracking_uri": local_uri,
            "shared_location": shared_path_uri,
            "local_path_uri": self.local_path_uri
        }
        
        # Initialize logger
        self.logger = ExperimentLogger('mlflow_utils', log_dir='./logs/mlflow_utils')
        
        # Ensure local directory exists
        self.local_path.mkdir(parents=True, exist_ok=True)
     

class MLFlowManager:
    """MLflow experiment and model management"""
    
    def __init__(self):
        self.config = MLFlowConfig()
        self.logger = self.config.logger
        # print(f"config: {self.config.config}")
        

        # Set tracking URI on initialization
        mlflow.set_tracking_uri(self.config.config["tracking_uri"])
    
    def sync_with_shared(self) -> None:
        """Sync local MLflow data with shared storage based on run IDs"""
        try:
            local_path = Path(self.config.local_path)
            shared_path = Path(self.config.shared_path)
            if not shared_path.exists():
                self.logger.warning("Shared MLflow directory not accessible")
                return

            # Iterate through experiments in the shared directory
            for experiment_directory in shared_path.iterdir():
                if not experiment_directory.is_dir() or experiment_directory.name in ['.trash', 'metadata']:
                    continue

                # Iterate through run directories within each experiment
                for run_directory in experiment_directory.iterdir():
                    if not run_directory.is_dir():
                        continue

                    run_id = run_directory.name
                    target_directory = local_path / experiment_directory.name / run_id

                    # Check if the run exists locally, if so, remove it before syncing
                    if target_directory.exists():
                        shutil.rmtree(target_directory)

                    # Copy the run directory from shared to local
                    shutil.copytree(run_directory, target_directory)
                    self.logger.info(f"Synced run: {run_id} from experiment: {experiment_directory.name}")

            # After copying, update paths in meta.yaml
            for experiment_dir in local_path.iterdir():
                if experiment_dir.is_dir() and experiment_dir.name != '.trash':
                    meta_path = experiment_dir / 'meta.yaml'
                    local_path = self.config.config["tracking_uri"]
                    print(f"meta_path: {meta_path}")
                    self.normalize_meta_yaml_paths(meta_path, local_path)
                        
                    # Update run meta.yaml files
                    for run_dir in experiment_dir.iterdir():
                        if run_dir.is_dir() and run_dir.name != '.trash':
                            run_meta = run_dir / 'meta.yaml'
                            print(f"run_meta: {run_meta}")
                            self.normalize_meta_yaml_paths(run_meta, local_path)

        except Exception as e:
            self.logger.error(f"Error syncing with shared storage: {e}")

    def backup_to_shared(self) -> None:
        """Backup local MLflow data to shared storage based on run IDs"""
        try:
            shared_path = Path(self.config.shared_path)
            if not shared_path.exists():
                self.logger.warning("Shared MLflow directory not accessible")
                return

            # Iterate through experiments in the local directory
            for experiment_directory in self.config.local_path.iterdir():
                if not experiment_directory.is_dir() or experiment_directory.name in ['.trash', 'metadata']:
                    continue

                # Iterate through run directories within each experiment
                for run_directory in experiment_directory.iterdir():
                    if not run_directory.is_dir():
                        continue

                    run_id = run_directory.name
                    target_directory = shared_path / experiment_directory.name / run_id

                    # Check if the run exists in the shared location, if so, remove it before backing up
                    if target_directory.exists():
                        continue

                    # Copy the run directory from local to shared
                    shutil.copytree(run_directory, target_directory)
                    self.logger.info(f"Backed up run: {run_id} from experiment: {experiment_directory.name}")

        except Exception as e:
            self.logger.error(f"Error backing up to shared storage: {e}")

    def setup_experiment(self, experiment_name: Optional[str] = None) -> str:
        """Setup MLflow experiment"""
        name = experiment_name
        
        try:
            experiment = mlflow.get_experiment_by_name(name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    name
                )
                mlflow.set_experiment(name)
                self.logger.info(f"Created new experiment: {name} experiment_id: {experiment_id}")
            else:
                experiment_id = experiment.experiment_id
                mlflow.set_experiment(name)
                self.logger.info(f"Using existing experiment: {name} experiment_id: {experiment_id}")

            return experiment_id
            
        except Exception as e:
            self.logger.error(f"Error setting up experiment: {e}")
            raise

    def load_latest_model(self, experiment_name: str):
        """Load the most recent model from the specified experiment"""
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                raise ValueError(f"Experiment {experiment_name} not found")
                
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"]
            )
            if runs.empty:
                raise ValueError(f"No runs found for experiment {experiment_name}")
                
            latest_run_id = runs.iloc[0].run_id
            
            # Get the actual model path including experiment folders
            client = MlflowClient()
            run = client.get_run(latest_run_id)
            
            model_uri = f"runs:/{latest_run_id}/model_global"
            
            return mlflow.xgboost.load_model(model_uri), latest_run_id
            
        except Exception as e:
            self.logger.error(f"Error loading latest model: {e}")
            raise

    def normalize_meta_yaml_paths(self, meta_path: Path, new_base_path: str) -> None:
        """Normalize paths in meta.yaml to match the current machine"""
        try:
            if not meta_path.exists():
                print(f"Meta.yaml not found at {meta_path}, creating a new one.")
                path_parts = str(meta_path).split('\\')
                if len(path_parts) >= 3 and path_parts[-3].isdigit() and path_parts[-2].isdigit():
                    experiment_id = path_parts[-3]
                    run_id = path_parts[-2]
                    self.create_missing_meta_yaml(meta_path.parent, experiment_id, run_id)
                    print("Meta.yaml has experiment and run id")
                elif len(path_parts) >= 2 and path_parts[-2].isdigit():
                    experiment_id = path_parts[-2]
                    self.create_missing_meta_yaml(meta_path.parent, experiment_id)
                    print("Meta.yaml has only experiment id")
                
                return
            
            with open(meta_path, 'r') as f:
                meta_data = yaml.safe_load(f)
            
            # Check if artifact_uri exists
            if 'artifact_uri' not in meta_data:
                old_uri = meta_data['artifact_location']
                # Extract the old URI and convert to Path object
                if old_uri.startswith('file:///'):
                    old_path = Path(old_uri[8:])  # Remove 'file:///' prefix
                else:
                    old_path = Path(old_uri)
            
                # Get the experiment and run ID from the path
                path_parts = old_path.parts
                experiment_id = path_parts[-1]
 
                
                # Construct new path using the base path and relative components
                new_uri = f"{new_base_path}/{experiment_id}"
                print(f"new_uri: {new_uri}")
                # Update with normalized path
                meta_data['artifact_location'] = new_uri.replace('\\', '/')
                
            else:
                old_uri = meta_data['artifact_uri']
                # Extract the old URI and convert to Path object
                if old_uri.startswith('file:///'):
                    old_path = Path(old_uri[8:])  # Remove 'file:///' prefix
                else:
                    old_path = Path(old_uri)
            
                # Get the experiment and run ID from the path
                path_parts = old_path.parts
                experiment_id = path_parts[-3]
                run_id = path_parts[-2]
                
                # Construct new path using the base path and relative components
                new_uri = f"{new_base_path}/{experiment_id}/{run_id}/artifacts"
                print(f"new_uri: {new_uri}")
                # Update with normalized path
                meta_data['artifact_uri'] = new_uri.replace('\\', '/')
            
            with open(meta_path, 'w') as f:
                yaml.safe_dump(meta_data, f)
            
        except Exception as e:
            self.logger.error(f"Error normalizing meta.yaml paths: {str(e)}")
            return

    def create_missing_meta_yaml(self, run_path: Path, experiment_id: str, run_id: Optional[str] = None) -> None:
        """Create meta.yaml file if it doesn't exist"""
        run_path = str(run_path).replace('\\', '/')
        meta_path = run_path / 'meta.yaml'
        try:
            if run_id is None:
                meta_data = {
                    'artifact_location': f"file:///{run_path}",
                    'creation_time': int(time.time() * 1000),
                    'experiment_id': experiment_id,
                    'last_update_time': int(time.time() * 1000),
                    'lifecycle_stage': 'active',
                    'name': f"experiment_{experiment_id}"
                }
            else:
             
                # Create basic meta.yaml structure
                meta_data = {
                    'artifact_uri': f"file:///{run_path}/artifacts",
                    'end_time': int(time.time() * 1000),
                    'entry_point_name': '',
                    'experiment_id': experiment_id,
                    'lifecycle_stage': 'active',
                    'run_id': run_id,
                    'run_name': f"run_{run_id}",
                    'run_uuid': run_id,
                    'source_name': '',
                    'source_type': 4,
                    'source_version': '',
                    'start_time': int(time.time() * 1000),
                    'status': 3,
                    'tags': [],
                    'user_id': os.getlogin()
                }
                
            # Write the file
            with open(meta_path, 'w') as f:
                yaml.safe_dump(meta_data, f)
                
            self.logger.info(f"Created missing meta.yaml at {meta_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating meta.yaml: {str(e)}")
            raise

def create_experiment_run(experiment_name: str, experiment_id: str):
    """Decorator for creating MLflow runs"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            manager = MLFlowManager()
            
            # Sync from shared before starting
            manager.sync_with_shared()
            
            experiment_id = manager.setup_experiment(experiment_name)
            with mlflow.start_run(experiment_id=experiment_id) as run:
                mlflow.set_tracking_uri(manager.config.config["tracking_uri"])             
                result = func(*args, **kwargs)        
            # Backup to shared after completion
            manager.backup_to_shared()
            return result
            
        return wrapper
    return decorator

def train_model(X, y):

    # Create and train a dummy model
    dummy_model = xgb.XGBClassifier(
        n_estimators=10,
        max_depth=3,
        random_state=42
    )
    dummy_model.fit(X, y)

    return dummy_model



# Usage example:
if __name__ == "__main__":
  
    manager = MLFlowManager()
    client = MlflowClient()
   
    manager.sync_with_shared()
    manager.backup_to_shared()
        
   