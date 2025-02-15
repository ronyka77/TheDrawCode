"""CatBoost model implementation for the ensemble."""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from pathlib import Path
import catboost as cb
import joblib

from base.model_interface import BaseModel
from utils.metrics import calculate_metrics

class CatBoostModel(BaseModel):
    """CatBoost model optimized for CPU execution."""
    
    def __init__(
        self,
        experiment_name: str = "ensemble",
        config_path: Optional[str] = None
    ):
        """Initialize CatBoost model.
        
        Args:
            experiment_name: Name for the experiment
            config_path: Optional path to model configuration
        """
        super().__init__(
            model_type="catboost",
            experiment_name=experiment_name,
            config_path=config_path
        )
        
        # Set CPU-specific configurations
        self.cpu_config = self.model_config['model']['cpu_config']
        self.logger.info(
            "Initialized CatBoost with CPU configuration: "
            f"{self.cpu_config}"
        )
        
        # Initialize feature info
        self.categorical_features_idx = None
        self.text_features_idx = None
    
    def _create_model(self, **kwargs) -> cb.CatBoostClassifier:
        """Create and configure CatBoost model instance.
        
        Args:
            **kwargs: Model parameters to override defaults
            
        Returns:
            Configured CatBoost classifier
        """
        # Start with CPU-optimized defaults
        params = {
            'task_type': self.cpu_config['task_type'],
            'thread_count': self.cpu_config['thread_count'],
            'bootstrap_type': self.cpu_config.get('bootstrap_type', 'Bernoulli'),
            'subsample': self.cpu_config.get('subsample', 0.8),
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'random_seed': 42,
            'verbose': False,
            'allow_writing_files': False,  # Disable automatic metric logging
            'boosting_type': 'Plain',  # Use plain boosting for better precision
            'feature_border_type': 'UniformAndQuantiles',  # Better feature discretization
            'leaf_estimation_method': 'Newton',  # Use Newton method for better precision
            'grow_policy': 'SymmetricTree'  # Build symmetric trees
        }
        
        # Update with any provided parameters
        params.update(kwargs)
        
        self.logger.info(f"Creating CatBoost model with parameters: {params}")
        return cb.CatBoostClassifier(**params)
    
    def _convert_to_model_format(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> Tuple[Any, Optional[Any]]:
        """Convert data to CatBoost format.
        
        Args:
            X: Feature matrix
            y: Optional target vector
            
        Returns:
            Tuple of (X, y) in CatBoost format
        """
        # Identify categorical and text features if not done yet
        if self.categorical_features_idx is None:
            self._identify_special_features(X)
        
        # Convert to CatBoost's Pool format
        pool = cb.Pool(
            data=X,
            label=y if y is not None else None,
            cat_features=self.categorical_features_idx,
            text_features=self.text_features_idx,
            feature_names=list(X.columns)
        )
        
        return pool, None  # Second element is None as y is included in pool
    
    def _train_model(
        self,
        X: cb.Pool,
        y: None,  # y is included in the Pool
        validation_data: Optional[Tuple[cb.Pool, None]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """Train CatBoost model and return metrics.
        
        Args:
            X: Training data Pool
            y: None (included in Pool)
            validation_data: Optional tuple of (val_pool, None)
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary of training metrics
        """
        fit_params = {
            'early_stopping_rounds': self.model_config['model']['training']['early_stopping_rounds'],
            'verbose': self.model_config['model']['training']['verbose_eval']
        }
        
        # Set up validation if provided
        if validation_data is not None:
            val_pool, _ = validation_data
            fit_params['eval_set'] = val_pool
        
        # Train model
        self.model.fit(
            X,
            eval_set=fit_params.get('eval_set'),
            early_stopping_rounds=fit_params['early_stopping_rounds'],
            verbose=fit_params['verbose']
        )
        
        # Calculate training metrics
        train_metrics = calculate_metrics(
            y_true=X.get_label(),
            y_pred=self.model.predict(X),
            y_prob=self.model.predict_proba(X)[:, 1]
        )
        
        # Add validation metrics if available
        if validation_data is not None:
            val_pool, _ = validation_data
            val_metrics = calculate_metrics(
                y_true=val_pool.get_label(),
                y_pred=self.model.predict(val_pool),
                y_prob=self.model.predict_proba(val_pool)[:, 1]
            )
            train_metrics.update({
                f'val_{k}': v for k, v in val_metrics.items()
            })
        
        # Log feature importance
        self._log_feature_importance(X)
        
        return train_metrics
    
    def _predict_model(self, X: Any) -> np.ndarray:
        """Generate CatBoost model predictions.
        
        Args:
            X: Features in model format (Pool)
            
        Returns:
            Array of binary predictions
        """
        return self.model.predict(X)
    
    def _predict_proba_model(self, X: Any) -> np.ndarray:
        """Generate CatBoost model probability predictions.
        
        Args:
            X: Features in model format (Pool)
            
        Returns:
            Array of probability predictions
        """
        return self.model.predict_proba(X)[:, 1]
    
    def _save_model_to_path(self, path: Path) -> None:
        """Save CatBoost model to specified path.
        
        Args:
            path: Path to save the model
        """
        # Save model in CatBoost format
        model_path = path.with_suffix('.cbm')
        self.model.save_model(
            model_path,
            format='cbm',  # Binary format
            export_parameters=True,
            pool=None
        )
        
        # Save feature information
        info_path = path.with_suffix('.info.json')
        feature_info = {
            'feature_names': self.feature_names,
            'categorical_features_idx': self.categorical_features_idx,
            'text_features_idx': self.text_features_idx,
            'best_threshold': self.best_threshold,
            'best_iteration': self.model.tree_count_
        }
        joblib.dump(feature_info, info_path)
        
        self.logger.info(
            f"Saved model to {model_path} "
            f"and feature info to {info_path}"
        )
    
    def _load_model_from_path(self, path: Path) -> None:
        """Load CatBoost model from specified path.
        
        Args:
            path: Path to load the model from
        """
        # Load model
        model_path = path.with_suffix('.cbm')
        self.model = cb.CatBoostClassifier()
        self.model.load_model(model_path)
        
        # Load feature information
        info_path = path.with_suffix('.info.json')
        if info_path.exists():
            feature_info = joblib.load(info_path)
            self.feature_names = feature_info['feature_names']
            self.categorical_features_idx = feature_info['categorical_features_idx']
            self.text_features_idx = feature_info['text_features_idx']
            self.best_threshold = feature_info['best_threshold']
            self.logger.info(f"Loaded model info from {info_path}")
        
        self.is_fitted = True
    
    def _identify_special_features(self, X: pd.DataFrame) -> None:
        """Identify categorical and text features in the dataset.
        
        Args:
            X: Feature matrix
        """
        categorical_features = []
        text_features = []
        
        for idx, (col, dtype) in enumerate(X.dtypes.items()):
            # Identify categorical features
            if dtype == 'category' or dtype == 'object':
                if X[col].nunique() / len(X) < 0.1:  # Less than 10% unique values
                    categorical_features.append(idx)
                else:
                    text_features.append(idx)
        
        self.categorical_features_idx = categorical_features
        self.text_features_idx = text_features
        
        if categorical_features:
            self.logger.info(
                f"Identified {len(categorical_features)} categorical features"
            )
        if text_features:
            self.logger.info(
                f"Identified {len(text_features)} text features"
            )
    
    def _log_feature_importance(self, train_pool: cb.Pool) -> None:
        """Log various types of feature importance.
        
        Args:
            train_pool: Training data pool for feature importance calculation
        """
        feature_names = self.feature_names
        
        # Get different types of importance
        importance_types = [
            'PredictionValuesChange',
            'LossFunctionChange',
            'ShapValues'
        ]
        
        for imp_type in importance_types:
            try:
                # Calculate importance
                if imp_type == 'ShapValues':
                    importance = np.abs(
                        self.model.get_feature_importance(
                            type=imp_type,
                            data=train_pool,
                            verbose=False
                        )
                    ).mean(axis=0)
                else:
                    importance = self.model.get_feature_importance(
                        type=imp_type,
                        data=train_pool,
                        verbose=False
                    )
                
                # Create importance dictionary
                importance_dict = {
                    name: float(score) for name, score in zip(
                        feature_names,
                        importance
                    )
                }
                
                # Log to MLflow
                self.mlflow_manager.log_params({
                    f'{imp_type.lower()}_importance_{k}': v 
                    for k, v in importance_dict.items()
                })
                
            except Exception as e:
                self.logger.error(
                    f"Error calculating {imp_type} importance: {str(e)}"
                ) 