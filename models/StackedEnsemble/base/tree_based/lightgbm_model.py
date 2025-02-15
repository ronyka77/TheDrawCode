"""LightGBM model implementation for the ensemble."""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import lightgbm as lgb
import joblib

from base.model_interface import BaseModel
from utils.metrics import calculate_metrics

class LightGBMModel(BaseModel):
    """LightGBM model optimized for CPU execution."""
    
    def __init__(
        self,
        experiment_name: str = "ensemble",
        config_path: Optional[str] = None
    ):
        """Initialize LightGBM model.
        
        Args:
            experiment_name: Name for the experiment
            config_path: Optional path to model configuration
        """
        super().__init__(
            model_type="lightgbm",
            experiment_name=experiment_name,
            config_path=config_path
        )
        
        # Set CPU-specific configurations
        self.cpu_config = self.model_config['model']['cpu_config']
        self.logger.info(
            "Initialized LightGBM with CPU configuration: "
            f"{self.cpu_config}"
        )
    
    def _create_model(self, **kwargs) -> lgb.LGBMClassifier:
        """Create and configure LightGBM model instance.
        
        Args:
            **kwargs: Model parameters to override defaults
            
        Returns:
            Configured LightGBM classifier
        """
        # Start with CPU-optimized defaults
        params = {
            'device': self.cpu_config['device'],
            'num_threads': self.cpu_config['num_threads'],
            'tree_learner': self.cpu_config['tree_learner'],
            'force_row_wise': self.cpu_config['force_row_wise'],
            'objective': 'binary',
            'metric': ['binary_logloss', 'auc'],
            'verbose': -1,  # Suppress default output
            'deterministic': True  # For reproducibility
        }
        
        # Update with any provided parameters
        params.update(kwargs)
        
        self.logger.info(f"Creating LightGBM model with parameters: {params}")
        return lgb.LGBMClassifier(**params)
    
    def _convert_to_model_format(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> Tuple[Any, Optional[Any]]:
        """Convert data to LightGBM format.
        
        Args:
            X: Feature matrix
            y: Optional target vector
            
        Returns:
            Tuple of (X, y) in LightGBM format
        """
        # LightGBM can handle pandas DataFrames directly
        # Just ensure y is numeric if provided
        if y is not None:
            y = y.astype(int)
        return X, y
    
    def _train_model(
        self,
        X: Any,
        y: Any,
        validation_data: Optional[Tuple[Any, Any]] = None,
        **kwargs) -> Dict[str, float]:
        """Train LightGBM model and return metrics.
        
        Args:
            X: Training features
            y: Training labels
            validation_data: Optional tuple of (X_val, y_val)
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary of training metrics
        """
        fit_params = {
            'early_stopping_rounds': self.model_config['model']['training']['early_stopping_rounds'],
            'verbose': self.model_config['model']['training']['verbose_eval']
        }
        
        # Set up validation if provided
        eval_set = None
        if validation_data is not None:
            X_val, y_val = validation_data
            eval_set = [(X_val, y_val)]
            fit_params['eval_set'] = eval_set
            fit_params['eval_names'] = ['validation']
            fit_params['eval_metric'] = ['binary_logloss', 'auc']
        
        # Create categorical feature list if any
        categorical_features = self._get_categorical_features(X)
        if categorical_features:
            fit_params['categorical_feature'] = categorical_features
        
        # Train model
        self.model.fit(
            X, y,
            **fit_params
        )
        
        # Calculate metrics
        train_metrics = calculate_metrics(
            y_true=y,
            y_pred=self.model.predict(X),
            y_prob=self.model.predict_proba(X)[:, 1]
        )
        
        # Add validation metrics if available
        if eval_set is not None:
            val_metrics = calculate_metrics(
                y_true=y_val,
                y_pred=self.model.predict(X_val),
                y_prob=self.model.predict_proba(X_val)[:, 1]
            )
            train_metrics.update({
                f'val_{k}': v for k, v in val_metrics.items()
            })
        
        # Log feature importance (both split and gain)
        self._log_feature_importance()
        
        return train_metrics
    
    def _predict_model(self, X: Any) -> np.ndarray:
        """Generate LightGBM model predictions.
        
        Args:
            X: Features in model format
            
        Returns:
            Array of binary predictions
        """
        return self.model.predict(X)
    
    def _predict_proba_model(self, X: Any) -> np.ndarray:
        """Generate LightGBM model probability predictions.
        
        Args:
            X: Features in model format
            
        Returns:
            Array of probability predictions
        """
        return self.model.predict_proba(X)[:, 1]
    
    def _save_model_to_path(self, path: Path) -> None:
        """Save LightGBM model to specified path.
        
        Args:
            path: Path to save the model
        """
        joblib.dump(self.model, path)
        
        # Also save native format for faster loading
        native_path = path.with_suffix('.txt')
        self.model.booster_.save_model(
            native_path,
            num_iteration=self.model.best_iteration_ or 0
        )
        self.logger.info(f"Saved native model format to {native_path}")
    
    def _load_model_from_path(self, path: Path) -> None:
        """Load LightGBM model from specified path.
        
        Args:
            path: Path to load the model from
        """
        # Try loading native format first
        native_path = path.with_suffix('.txt')
        if native_path.exists():
            self.model = lgb.Booster(model_file=str(native_path))
            self.logger.info(f"Loaded native model from {native_path}")
        else:
            self.model = joblib.load(path)
            self.logger.info(f"Loaded joblib model from {path}")
        
        self.is_fitted = True
    
    def _get_categorical_features(self, X: pd.DataFrame) -> list:
        """Identify categorical features in the dataset.
        
        Args:
            X: Feature matrix
            
        Returns:
            List of categorical feature names
        """
        categorical_features = []
        for col in X.columns:
            if X[col].dtype == 'category' or X[col].dtype == 'object':
                categorical_features.append(col)
        
        if categorical_features:
            self.logger.info(
                f"Identified categorical features: {categorical_features}"
            )
        
        return categorical_features
    
    def _log_feature_importance(self) -> None:
        """Log both split and gain feature importance."""
        # Get feature names
        feature_names = self.feature_names
        
        # Log split importance
        split_importance = self.model.feature_importances_
        split_importance_dict = {
            name: score for name, score in zip(
                feature_names,
                split_importance
            )
        }
        self.mlflow_manager.log_params({
            f'split_importance_{k}': v 
            for k, v in split_importance_dict.items()
        })
        
        # Log gain importance
        gain_importance = self.model.booster_.feature_importance(
            importance_type='gain'
        )
        gain_importance_dict = {
            name: score for name, score in zip(
                feature_names,
                gain_importance
            )
        }
        self.mlflow_manager.log_params({
            f'gain_importance_{k}': v 
            for k, v in gain_importance_dict.items()
        }) 