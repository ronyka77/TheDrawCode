"""
Training context manager for the Model Context Protocol Server.

This module implements:
- Training context state management
- Metadata tracking and persistence
- MLflow integration for context history
- Thread-safe state updates
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
import json
import threading
from dataclasses import dataclass, asdict
import pickle

# Add project root to path
try:
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
except Exception as e:
    print(f"Error setting project root path: {e}")
    sys.path.append(os.getcwd())

# Local imports
from utils.logger import ExperimentLogger
from utils.mlflow_utils import MLFlowManager

# Initialize logger
logger = ExperimentLogger(
    experiment_name="context_manager",
    log_dir="logs/context_manager"
)

@dataclass
class TrainingMetadata:
    """Training metadata container."""
    model_type: str
    feature_count: int
    sample_count: int
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]
    start_time: datetime
    last_updated: datetime
    status: str
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        data = asdict(self)
        # Convert datetime objects to ISO format
        data['start_time'] = self.start_time.isoformat()
        data['last_updated'] = self.last_updated.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingMetadata':
        """Create metadata from dictionary."""
        # Convert ISO format strings back to datetime
        data['start_time'] = datetime.fromisoformat(data['start_time'])
        data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        return cls(**data)

class TrainingContext:
    """Training context manager with thread-safe state handling."""
    
    def __init__(self, state_dir: Optional[str] = None):
        """Initialize context manager.
        
        Args:
            state_dir: Optional directory for state persistence
        """
        self.state_dir = Path(state_dir) if state_dir else project_root / "state"
        self.state_file = self.state_dir / "training_context.pkl"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # Thread-safe state management
        self._lock = threading.RLock()
        self._contexts: Dict[str, TrainingMetadata] = {}
        self._active_runs: Set[str] = set()
        
        # Load existing state if available
        self._load_state()
        
        logger.info(
            "Context manager initialized",
            extra={"state_dir": str(self.state_dir)}
        )
        
    def _load_state(self) -> None:
        """Load persisted state from disk."""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'rb') as f:
                    state_data = pickle.load(f)
                    self._contexts = state_data.get('contexts', {})
                    self._active_runs = state_data.get('active_runs', set())
                    
                logger.info(
                    "Loaded existing state",
                    extra={
                        "context_count": len(self._contexts),
                        "active_runs": len(self._active_runs)
                    }
                )
                
        except Exception as e:
            logger.error(f"Error loading state: {str(e)}")
            # Initialize empty state on error
            self._contexts = {}
            self._active_runs = set()
            
    def _save_state(self) -> None:
        """Persist current state to disk."""
        try:
            state_data = {
                'contexts': self._contexts,
                'active_runs': self._active_runs
            }
            
            with open(self.state_file, 'wb') as f:
                pickle.dump(state_data, f)
                
            logger.info("State saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving state: {str(e)}")
            
    def create_context(
        self,
        run_id: str,
        model_type: str,
        hyperparameters: Dict[str, Any],
        feature_count: int,
        sample_count: int,
        status: str = "INITIALIZING"
    ) -> None:
        """Create new training context.
        
        Args:
            run_id: MLflow run ID
            model_type: Type of model being trained
            hyperparameters: Training hyperparameters
            feature_count: Number of features
            sample_count: Number of training samples
            status: Initial status
        """
        try:
            with self._lock:
                now = datetime.now()
                metadata = TrainingMetadata(
                    model_type=model_type,
                    feature_count=feature_count,
                    sample_count=sample_count,
                    hyperparameters=hyperparameters,
                    metrics={},
                    start_time=now,
                    last_updated=now,
                    status=status
                )
                
                self._contexts[run_id] = metadata
                self._active_runs.add(run_id)
                
                # Persist updated state
                self._save_state()
                
                logger.info(
                    f"Created context for run {run_id}",
                    extra={
                        "run_id": run_id,
                        "model_type": model_type,
                        "status": status
                    }
                )
                
        except Exception as e:
            logger.error(f"Error creating context: {str(e)}")
            raise
            
    def update_metrics(
        self,
        run_id: str,
        metrics: Dict[str, float]
    ) -> None:
        """Update metrics for a training run.
        
        Args:
            run_id: MLflow run ID
            metrics: New metrics to update
        """
        try:
            with self._lock:
                if run_id not in self._contexts:
                    raise ValueError(f"No context found for run_id: {run_id}")
                    
                context = self._contexts[run_id]
                context.metrics.update(metrics)
                context.last_updated = datetime.now()
                
                # Persist updated state
                self._save_state()
                
                logger.info(
                    f"Updated metrics for run {run_id}",
                    extra={
                        "run_id": run_id,
                        "metrics": metrics
                    }
                )
                
        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")
            raise
            
    def update_status(
        self,
        run_id: str,
        status: str,
        error: Optional[str] = None
    ) -> None:
        """Update status of a training run.
        
        Args:
            run_id: MLflow run ID
            status: New status
            error: Optional error message
        """
        try:
            with self._lock:
                if run_id not in self._contexts:
                    raise ValueError(f"No context found for run_id: {run_id}")
                    
                context = self._contexts[run_id]
                context.status = status
                context.error = error
                context.last_updated = datetime.now()
                
                # Remove from active runs if terminal status
                if status in ["COMPLETED", "FAILED", "STOPPED"]:
                    self._active_runs.discard(run_id)
                    
                # Persist updated state
                self._save_state()
                
                logger.info(
                    f"Updated status for run {run_id}",
                    extra={
                        "run_id": run_id,
                        "status": status,
                        "error": error
                    }
                )
                
        except Exception as e:
            logger.error(f"Error updating status: {str(e)}")
            raise
            
    def get_context(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get context for a training run.
        
        Args:
            run_id: MLflow run ID
            
        Returns:
            Dictionary with context data or None if not found
        """
        try:
            with self._lock:
                context = self._contexts.get(run_id)
                return context.to_dict() if context else None
                
        except Exception as e:
            logger.error(f"Error getting context: {str(e)}")
            raise
            
    def get_active_runs(self) -> List[Dict[str, Any]]:
        """Get all active training runs.
        
        Returns:
            List of context data for active runs
        """
        try:
            with self._lock:
                return [
                    self._contexts[run_id].to_dict()
                    for run_id in self._active_runs
                ]
                
        except Exception as e:
            logger.error(f"Error getting active runs: {str(e)}")
            raise
            
    def cleanup_context(self, run_id: str) -> None:
        """Clean up context for a training run.
        
        Args:
            run_id: MLflow run ID
        """
        try:
            with self._lock:
                if run_id in self._contexts:
                    del self._contexts[run_id]
                    self._active_runs.discard(run_id)
                    
                    # Persist updated state
                    self._save_state()
                    
                    logger.info(
                        f"Cleaned up context for run {run_id}",
                        extra={"run_id": run_id}
                    )
                    
        except Exception as e:
            logger.error(f"Error cleaning up context: {str(e)}")
            raise

def create_context_manager(state_dir: Optional[str] = None) -> TrainingContext:
    """Create and initialize context manager.
    
    Args:
        state_dir: Optional directory for state persistence
        
    Returns:
        Initialized TrainingContext instance
    """
    try:
        return TrainingContext(state_dir)
        
    except Exception as e:
        logger.error(f"Error creating context manager: {str(e)}")
        raise 