"""Base interface for all models in the ensemble."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import os

from models.StackedEnsemble.shared.mlflow_utils import MLFlowManager
from models.StackedEnsemble.shared.config_loader import ConfigurationLoader
from models.StackedEnsemble.shared.validation import NestedCVValidator
from models.StackedEnsemble.utils.metrics import calculate_metrics
from utils.logger import ExperimentLogger

class BaseModel(ABC):
    """Abstract base class for all models in the ensemble."""
    
    def __init__(
        self,
        model_type: str,
        experiment_name: str = "ensemble",
        config_path: Optional[str] = None
    ):
        """Initialize the base model.
        
        Args:
            model_type: Type of model (e.g., 'xgboost', 'lightgbm')
            experiment_name: Name for the experiment
            config_path: Optional path to model configuration
        """
        self.model_type = model_type
        self.experiment_name = f"{experiment_name}_{model_type}"
        
        # Get project root path
        self.project_root = Path(__file__).parent.parent.parent.parent
        
        # Set up logging and tracking
        self.logger = ExperimentLogger(experiment_name=self.experiment_name)
        self.mlflow_manager = MLFlowManager(self.experiment_name)
        self.mlruns_dir = self.mlflow_manager.setup_model_experiment(model_type)
        
        # Load configurations
        self.config_loader = ConfigurationLoader(self.experiment_name)
        
        # Set up configuration paths
        if config_path is None:
            config_path = os.path.join(
                self.project_root,
                "models",
                "StackedEnsemble",
                "config",
                "model_configs",
                f"{model_type}_config.yaml"
            )
        
        self.model_config = self.config_loader.load_model_config(model_type)
        self.hyperparameter_space = self.config_loader.load_hyperparameter_space(model_type)
        
        # Initialize model state
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.best_threshold = 0.5  # Default threshold
        
        self.logger.info(
            f"Initialized {model_type} model with experiment name: {self.experiment_name}"
        )
    
    @abstractmethod
    def _create_model(self, **kwargs) -> Any:
        """Create and return the actual model instance.
        
        Args:
            **kwargs: Model parameters
            
        Returns:
            Model instance
        """
        pass
    
    @abstractmethod
    def _convert_to_model_format(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None) -> Tuple[Any, Optional[Any]]:
        """Convert data to model-specific format.
        
        Args:
            X: Feature matrix
            y: Optional target vector
            
        Returns:
            Tuple of (X, y) in model-specific format
        """
        pass
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        **kwargs) -> Dict[str, float]:
        """Fit the model using training data with early stopping on test data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features for early stopping
            y_test: Test labels for early stopping
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary of training metrics
        """
        self.logger.info("Starting model training")
        self.feature_names = list(X_train.columns)
        
        # Convert data to model format
        X_train_model, y_train_model = self._convert_to_model_format(X_train, y_train)
        X_test_model, y_test_model = self._convert_to_model_format(X_test, y_test)
        
        # Start MLflow run
        self.mlflow_manager.start_run()
        try:
            # Create and train model
            self.model = self._create_model(**kwargs)
            metrics = self._train_model(
                X_train_model, y_train_model,
                validation_data=(X_test_model, y_test_model),
                **kwargs
            )
            
            # Log metrics and parameters
            self.mlflow_manager.log_split_metrics(metrics, "train")
            if "val_" in str(metrics):
                val_metrics = {k[4:]: v for k, v in metrics.items() if k.startswith("val_")}
                self.mlflow_manager.log_split_metrics(val_metrics, "test")
            
            self.mlflow_manager.log_params(kwargs)
            
            # Save model
            self._save_model()
            
            self.is_fitted = True
            self.logger.info("Model training completed successfully")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}")
            raise
        finally:
            self.mlflow_manager.end_run()
    
    def optimize_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series) -> Dict[str, Any]:
        """Optimize hyperparameters using nested cross-validation.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary of best parameters
        """
        self.logger.info("Starting hyperparameter optimization")
        
        # Initialize nested CV
        validator = NestedCVValidator(
            outer_splits=self.model_config['model']['validation']['cv_folds'],
            inner_splits=3,
            experiment_name=f"{self.experiment_name}_cv"
        )
        
        # Define objective function
        def objective_fn(X_val: pd.DataFrame, y_val: pd.Series, params: Dict[str, Any]) -> float:
            """Objective function for hyperparameter optimization."""
            # Create and train model with given parameters
            model = self._create_model(**params)
            X_val_model, y_val_model = self._convert_to_model_format(X_val, y_val)
            model.fit(X_val_model, y_val_model)
            
            # Calculate metrics
            y_pred = model.predict_proba(X_val_model)[:, 1]
            metrics = calculate_metrics(y_val, (y_pred >= 0.5).astype(int), y_pred)
            
            # Return objective value (e.g., precision)
            return metrics['precision']
        
        # Start MLflow run for tuning
        self.mlflow_manager.start_run(run_name="hyperparameter_tuning")
        try:
            # Perform nested CV
            best_params, best_score = validator.perform_nested_cv(
                X_train, y_train,
                self.hyperparameter_space['hyperparameters'],
                objective_fn,
                self.mlflow_manager
            )
            
            # Log results
            self.mlflow_manager.log_metrics({
                'best_cv_score': best_score
            })
            self.mlflow_manager.log_params(best_params)
            
            self.logger.info(f"Found best parameters: {best_params}")
            return best_params
            
        finally:
            self.mlflow_manager.end_run()
    
    def evaluate(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        optimize_threshold: bool = True) -> Dict[str, float]:
        """Evaluate model on validation data.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            optimize_threshold: Whether to optimize decision threshold
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before evaluation")
            
        self.logger.info("Evaluating model on validation data")
        
        # Convert data
        X_val_model, y_val_model = self._convert_to_model_format(X_val, y_val)
        
        # Get predictions
        y_prob = self._predict_proba_model(X_val_model)
        
        # Optimize threshold if requested
        if optimize_threshold:
            self.best_threshold = self._optimize_threshold(y_val, y_prob)
            self.logger.info(f"Optimized decision threshold: {self.best_threshold:.4f}")
        
        # Calculate metrics
        y_pred = (y_prob >= self.best_threshold).astype(int)
        metrics = calculate_metrics(y_val, y_pred, y_prob)
        
        # Log validation metrics
        self.mlflow_manager.start_run()
        try:
            self.mlflow_manager.log_split_metrics(metrics, "validation")
            if optimize_threshold:
                self.mlflow_manager.log_params({
                    'decision_threshold': self.best_threshold
                })
        finally:
            self.mlflow_manager.end_run()
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions using optimized threshold.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of binary predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
            
        X_model, _ = self._convert_to_model_format(X)
        y_prob = self._predict_proba_model(X_model)
        return (y_prob >= self.best_threshold).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate probability predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of probability predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
            
        X_model, _ = self._convert_to_model_format(X)
        return self._predict_proba_model(X_model)
    
    @abstractmethod
    def _train_model(
        self,
        X: Any,
        y: Any,
        validation_data: Optional[Tuple[Any, Any]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """Train model and return metrics.
        
        Args:
            X: Training features in model format
            y: Training labels in model format
            validation_data: Optional tuple of (X_val, y_val) in model format
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def _predict_model(self, X: Any) -> np.ndarray:
        """Generate model predictions.
        
        Args:
            X: Features in model format
            
        Returns:
            Array of predictions
        """
        pass
    
    @abstractmethod
    def _predict_proba_model(self, X: Any) -> np.ndarray:
        """Generate model probability predictions.
        
        Args:
            X: Features in model format
            
        Returns:
            Array of probability predictions
        """
        pass
    
    def _optimize_threshold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        min_recall: float = 0.20
    ) -> float:
        """Optimize decision threshold with minimum recall constraint.
        
        Args:
            y_true: True labels
            y_prob: Probability predictions
            min_recall: Minimum required recall
            
        Returns:
            Optimal threshold value
        """
        thresholds = np.linspace(0, 1, 100)
        best_threshold = 0.5
        best_precision = 0.0
        
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            metrics = calculate_metrics(y_true, y_pred, y_prob)
            
            # Check if recall constraint is satisfied
            if metrics['recall'] >= min_recall:
                if metrics['precision'] > best_precision:
                    best_precision = metrics['precision']
                    best_threshold = threshold
        
        return best_threshold
    
    @abstractmethod
    def _save_model_to_path(self, path: Path) -> None:
        """Save model to specified path.
        
        Args:
            path: Path to save the model
        """
        pass
    
    @abstractmethod
    def _load_model_from_path(self, path: Path) -> None:
        """Load model from specified path.
        
        Args:
            path: Path to load the model from
        """
        pass
    
    def _save_model(self) -> None:
        """Save the trained model."""
        if self.model is None:
            return
            
        save_path = Path(self.model_config['model']['logging']['log_dir'])
        save_path.mkdir(parents=True, exist_ok=True)
        
        model_path = save_path / f"{self.model_type}_model.pkl"
        self._save_model_to_path(model_path)
        
        # Log model to MLflow
        self.mlflow_manager.log_model(
            self.model,
            "model",
            registered_model_name=f"{self.model_type}_{self.model_config['model']['version']}"
        )
        
        self.logger.info(f"Saved model to {model_path} and logged to MLflow") 