"""Base model interface for all models in the StackedEnsemble framework."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys
import json
import time
from datetime import datetime

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)
os.environ["PYTHONPATH"] = project_root + os.pathsep + os.environ.get("PYTHONPATH", "")

from utils.logger import ExperimentLogger
from models.StackedEnsemble.shared.config_loader import ConfigurationLoader
# from models.StackedEnsemble.shared.validation import OptunaValidator

class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(
        self,
        model_type: str,
        experiment_name: str = None,
        logger: ExperimentLogger = None):
        """Initialize base model.
        
        Args:
            model_type: Type of model (e.g., 'bert', 'xgboost')
            experiment_name: Name for experiment tracking
            logger: Logger instance
        """
        self.model_type = model_type
        self.experiment_name = experiment_name or f"{model_type}_experiment"
        self.logger = logger or ExperimentLogger(self.experiment_name)
        
        # Initialize configuration loader
        self.config_loader = ConfigurationLoader(model_type)
        
        # Load configurations
        self.model_config = self.config_loader.load_model_config(model_type)
        self.hyperparameter_space = self.config_loader.load_hyperparameter_space(model_type)
        
        # # Initialize validator
        # self.validator = OptunaValidator(
        #     model_type=model_type,
        #     logger=self.logger
        # )
        
        # Initialize state
        self.best_params = {}
        self.best_score = 0.0
        self.model = None
        self.is_fitted = False
        
        self.logger.info(f"Initialized {model_type} model")

    @abstractmethod
    def _create_model(self, **kwargs) -> Any:
        """Create and configure model instance."""
        pass

    @abstractmethod
    def _train_model(
        self,
        X: Any,
        y: Any,
        X_val: Any,
        y_val: Any,
        X_test: Any,
        y_test: Any,
        **kwargs) -> Dict[str, float]:
        """Train model with validation data."""
        pass

    @abstractmethod
    def _predict_model(self, X: Any) -> np.ndarray:
        """Generate predictions using trained model."""
        pass

    @abstractmethod
    def _predict_proba_model(self, X: Any) -> np.ndarray:
        """Generate probability predictions."""
        pass

    @abstractmethod
    def _save_model_to_path(self, path: Path) -> None:
        """Save model to specified path."""
        pass

    @abstractmethod
    def _load_model_from_path(self, path: Path) -> None:
        """Load model from specified path."""
        pass

    def optimize_hyperparameters(
        self,
        X: Any,
        y: Any,
        X_val: Any,
        y_val: Any,
        X_test: Any,
        y_test: Any) -> Dict[str, Any]:
        """Run hyperparameter optimization.
        
        Args:
            X, y: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data
            
        Returns:
            Best hyperparameters found
        """
        try:
            self.logger.info(f"Starting hyperparameter optimization for {self.model_type}")
            
            # Run optimization
            best_params = {0: 0}
            
            # Store best parameters
            self.best_params = best_params
            
            # Save optimization results
            self._save_optimization_results(best_params)
            
            return best_params
            
        except Exception as e:
            self.logger.error(f"Error in hyperparameter optimization: {str(e)}")
            return {}

    def _save_optimization_results(self, best_params: Dict[str, Any]) -> None:
        """Save optimization results to file.
        
        Args:
            best_params: Best parameters found
        """
        try:
            # Create results directory
            results_dir = Path(project_root) / "results" / "hypertuning" / self.model_type
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare results
            results = {
                'model_type': self.model_type,
                'experiment_name': self.experiment_name,
                'timestamp': datetime.now().isoformat(),
                'best_parameters': best_params
            }
            
            # Save results
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            results_file = results_dir / f"optimization_results_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
                
            self.logger.info(f"Optimization results saved to {results_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving optimization results: {str(e)}")

    def fit(
        self,
        X: Any,
        y: Any,
        X_val: Any,
        y_val: Any,
        X_test: Optional[Any] = None,
        y_test: Optional[Any] = None,
        **kwargs) -> Dict[str, float]:
        """Train model with validation data.
        
        Args:
            X, y: Training data
            X_val, y_val: Validation data
            X_test, y_test: Optional test data
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary of training metrics
        """
        try:
            # Initialize model if not already done
            if self.model is None:
                self.model = self._create_model(**kwargs)
            
            # Train model
            metrics = self._train_model(X, y, X_val, y_val, X_test, y_test, **kwargs)
            
            # Update state
            self.is_fitted = True
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in model training: {str(e)}")
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'auc': 0.0,
                'brier_score': 1.0
            }

    def predict(self, X: Any) -> np.ndarray:
        """Make predictions using trained model.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before prediction")
        return self._predict_model(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Get prediction probabilities.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of prediction probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before prediction")
        return self._predict_proba_model(X)

    def save(self, path: Union[str, Path]) -> None:
        """Save model to file.
        
        Args:
            path: Path to save model
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before saving")
            
        path = Path(path)
        self._save_model_to_path(path)
        self.logger.info(f"Model saved to {path}")

    def load(self, path: Union[str, Path]) -> None:
        """Load model from file.
        
        Args:
            path: Path to load model from
        """
        path = Path(path)
        self._load_model_from_path(path)
        self.is_fitted = True
        self.logger.info(f"Model loaded from {path}")

    def get_params(self) -> Dict[str, Any]:
        """Get current model parameters.
        
        Returns:
            Dictionary of parameters
        """
        return self.best_params.copy() if self.best_params else {}

    def set_params(self, **params) -> None:
        """Set model parameters.
        
        Args:
            **params: Parameters to set
        """
        self.best_params.update(params)
        if self.model is not None:
            self.model = self._create_model(**self.best_params) 