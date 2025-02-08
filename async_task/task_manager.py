"""
Asynchronous task manager for the Model Context Protocol Server.

This module implements:
- Background task processing for model training
- Task state management and monitoring
- Integration with MLflow and ExperimentLogger
- Error handling and recovery procedures
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Any, Callable
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue
import json

# Add project root to Python path
try:
    project_root = Path(__file__).parent.parent
    if not project_root.exists():
        # Handle network path by using raw string
        project_root = Path(r"\\".join(str(project_root).split("\\")))
    sys.path.append(str(project_root))
    print(f"Project root ensemble_model: {project_root}")
except Exception as e:
    print(f"Error setting project root path: {e}")
    # Fallback to current directory if path resolution fails
    sys.path.append(os.getcwd().parent)
    print(f"Current directory ensemble_model: {os.getcwd().parent}")
    
os.environ['GIT_PYTHON_GIT_EXECUTABLE'] = "C:/Program Files/Git/bin/git.exe"

# Local imports
from utils.logger import ExperimentLogger
from utils.mlflow_utils import MLFlowManager
from models.xgboost_training import train_model as train_xgboost
from models.catboost_training import train_model as train_catboost

# Initialize logger
logger = ExperimentLogger(
    experiment_name="task_manager",
    log_dir="logs/task_manager"
)

class TaskState:
    """Task state container."""
    
    def __init__(
        self,
        task_id: str,
        task_type: str,
        status: str = "PENDING"
    ):
        """Initialize task state.
        
        Args:
            task_id: Unique task identifier
            task_type: Type of task (e.g., 'training')
            status: Initial task status
        """
        self.task_id = task_id
        self.task_type = task_type
        self.status = status
        self.start_time = None
        self.end_time = None
        self.error = None
        self.progress = 0
        self.result = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "error": str(self.error) if self.error else None,
            "progress": self.progress,
            "result": self.result
        }

class TaskManager:
    """
    A simple in-memory task manager for local model training.
    
    - run_training: Launches a training function in a background thread.
      It passes a stop flag which the training function should periodically check.
    
    - stop_training: Sets the stop flag for a running training task.
    """
    def __init__(self):
        # Mapping of run_id -> (Thread, stop_flag)
        self.running_tasks = {}
    
    def run_training(self, train_func, *args, **kwargs):
        """
        Starts the given training function asynchronously.
        
        Parameters:
            train_func (callable): The training function to be executed.
                                   It must accept a 'stop_flag' as its first parameter.
            *args, **kwargs: Arguments to pass to the training function.
        
        Returns:
            run_id (str): A unique identifier for the training run.
        """
        run_id = str(uuid.uuid4())
        stop_flag = threading.Event()
        # Include the stop_flag in the training function's kwargs
        thread = threading.Thread(
            target=self._execute_training,
            args=(run_id, train_func, stop_flag) + args,
            kwargs=kwargs,
            daemon=True
        )
        thread.start()
        self.running_tasks[run_id] = (thread, stop_flag)
        logger.info(f"Started training task with run_id: {run_id}")
        return run_id
    
    def _execute_training(self, run_id, train_func, stop_flag, *args, **kwargs):
        try:
            # The training function should periodically check stop_flag.is_set()
            train_func(stop_flag, *args, **kwargs)
            logger.info(f"Training task {run_id} completed successfully.")
        except Exception as e:
            # Use minimal error reporting here, as per your requirements.
            logger.error(f"Training task {run_id} encountered an error: {e}")
        finally:
            # Clean up by removing the task once it has finished.
            self.running_tasks.pop(run_id, None)
    
    def stop_training(self, run_id):
        """
        Signals a training task to stop by setting its stop_flag.
        
        Note: The training function must be coded to periodically check the flag.
        
        Parameters:
            run_id (str): The unique identifier of the training task.
        
        Returns:
            bool: True if the stop signal was sent, False if the task was not found.
        """
        task_tuple = self.running_tasks.get(run_id)
        if task_tuple:
            _, stop_flag = task_tuple
            stop_flag.set()
            logger.info(f"Stop signal sent for training task {run_id}")
            return True
        else:
            logger.warning(f"No running task found with run_id: {run_id}")
            return False

    def list_running_tasks(self):
        """Returns a list of run_ids for currently running training tasks."""
        return list(self.running_tasks.keys())
    
def create_task_manager(max_workers: int = 4) -> TaskManager:
    """Create and initialize task manager.
    
    Args:
        max_workers: Maximum number of concurrent tasks
        
    Returns:
        Initialized TaskManager instance
    """
    try:
        return TaskManager(max_workers=max_workers)
        
    except Exception as e:
        logger.error(f"Error creating task manager: {str(e)}")
        raise 