"""
Model interface layer for standardizing interactions between H2O and XGBoost models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import pandas as pd
import h2o
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin

class BaseModelWrapper(ABC, BaseEstimator, ClassifierMixin):
    """Abstract base class for model wrappers."""
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save the model."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load the model."""
        pass

class H2OModelWrapper(BaseModelWrapper):
    """Wrapper for H2O models."""
    
    def __init__(self, model: Any, logger: Any):
        self.model = model
        self.logger = logger
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        try:
            train_frame = h2o.H2OFrame(pd.concat([X, y.rename('target')], axis=1))
            train_frame['target'] = train_frame['target'].asfactor()
            self.model.train(
                x=list(X.columns),
                y='target',
                training_frame=train_frame
            )
        except Exception as e:
            self.logger.error(f"Error fitting H2O model: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        try:
            test_frame = h2o.H2OFrame(X)
            preds = self.model.predict(test_frame)
            return h2o.as_list(preds)['predict'].values
        except Exception as e:
            self.logger.error(f"Error predicting with H2O model: {str(e)}")
            raise
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        try:
            test_frame = h2o.H2OFrame(X)
            preds = self.model.predict(test_frame)
            probs = h2o.as_list(preds)
            return np.column_stack([probs['p0'].values, probs['p1'].values])
        except Exception as e:
            self.logger.error(f"Error getting probabilities from H2O model: {str(e)}")
            raise
    
    def save(self, path: str) -> None:
        try:
            h2o.save_model(self.model, path=path, force=True)
        except Exception as e:
            self.logger.error(f"Error saving H2O model: {str(e)}")
            raise
    
    def load(self, path: str) -> None:
        try:
            self.model = h2o.load_model(path)
        except Exception as e:
            self.logger.error(f"Error loading H2O model: {str(e)}")
            raise

class XGBoostModelWrapper(BaseModelWrapper):
    """Wrapper for XGBoost models with memory optimization."""
    
    def __init__(
        self,
        params: Dict[str, Any],
        logger: Any,
        eval_set: Optional[List[Tuple[pd.DataFrame, pd.Series]]] = None,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50
    ):
        self.params = params
        self.logger = logger
        self.model = None
        self.eval_set = eval_set
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self._best_iteration = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model with memory optimization."""
        try:
            # Convert data to DMatrix with memory efficiency
            dtrain = xgb.DMatrix(data=X, label=y)
            
            # Convert evaluation set if provided
            evals = []
            if self.eval_set:
                for i, (X_val, y_val) in enumerate(self.eval_set):
                    deval = xgb.DMatrix(data=X_val, label=y_val)
                    evals.append((deval, f'eval_{i}'))
            
            # Train model with early stopping
            evals_result = {}
            self.model = xgb.train(
                params=self.params,
                dtrain=dtrain,
                num_boost_round=self.num_boost_round,
                early_stopping_rounds=self.early_stopping_rounds,
                evals=evals,
                evals_result=evals_result,
                verbose_eval=False
            )
            
            # Store best iteration
            self._best_iteration = self.model.best_iteration
            
            # Log training results
            if evals_result:
                self.logger.info(f"Best iteration: {self._best_iteration}")
                for metric, values in evals_result['eval_0'].items():
                    self.logger.info(f"Best {metric}: {min(values) if 'error' in metric else max(values)}")
            
            # Clear memory
            del dtrain
            for eval_set in evals:
                del eval_set[0]
            import gc
            gc.collect()
            
        except Exception as e:
            self.logger.error(f"Error fitting XGBoost model: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with memory optimization."""
        try:
            # Convert to DMatrix in chunks if data is large
            chunk_size = 10000
            predictions = []
            
            for i in range(0, len(X), chunk_size):
                chunk = X.iloc[i:i + chunk_size]
                dtest = xgb.DMatrix(chunk)
                chunk_preds = (self.model.predict(dtest, ntree_limit=self._best_iteration) >= 0.5).astype(int)
                predictions.append(chunk_preds)
                del dtest
            
            return np.concatenate(predictions)
            
        except Exception as e:
            self.logger.error(f"Error predicting with XGBoost model: {str(e)}")
            raise
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities with memory optimization."""
        try:
            # Convert to DMatrix in chunks if data is large
            chunk_size = 10000
            probabilities = []
            
            for i in range(0, len(X), chunk_size):
                chunk = X.iloc[i:i + chunk_size]
                dtest = xgb.DMatrix(chunk)
                chunk_probs = self.model.predict(dtest, ntree_limit=self._best_iteration)
                probabilities.append(chunk_probs)
                del dtest
            
            probs = np.concatenate(probabilities)
            return np.column_stack([1 - probs, probs])
            
        except Exception as e:
            self.logger.error(f"Error getting probabilities from XGBoost model: {str(e)}")
            raise
    
    def save(self, path: str) -> None:
        """Save the model with additional metadata."""
        try:
            # Save model
            self.model.save_model(path)
            
            # Save metadata
            metadata = {
                'best_iteration': self._best_iteration,
                'params': self.params
            }
            import json
            with open(f"{path}.meta", 'w') as f:
                json.dump(metadata, f)
                
        except Exception as e:
            self.logger.error(f"Error saving XGBoost model: {str(e)}")
            raise
    
    def load(self, path: str) -> None:
        """Load the model with metadata."""
        try:
            # Load model
            self.model = xgb.Booster()
            self.model.load_model(path)
            
            # Load metadata if available
            try:
                with open(f"{path}.meta", 'r') as f:
                    metadata = json.load(f)
                self._best_iteration = metadata.get('best_iteration')
                self.params.update(metadata.get('params', {}))
            except FileNotFoundError:
                self.logger.warning("Model metadata not found, using defaults")
                
        except Exception as e:
            self.logger.error(f"Error loading XGBoost model: {str(e)}")
            raise

class ModelFactory:
    """Factory class for creating model wrappers."""
    
    @staticmethod
    def create_model(model_type: str, config: Dict[str, Any], logger: Any) -> BaseModelWrapper:
        """Create a model wrapper instance with memory optimization.
        
        Args:
            model_type: Type of model ('xgboost')
            config: Model configuration
            logger: Logger instance
            
        Returns:
            Model wrapper instance
        """
        if model_type == 'xgboost':
            # Ensure early stopping is properly configured
            if 'eval_set' not in config:
                raise ValueError("XGBoost models require eval_set for early stopping")
            
            # Create wrapper with memory-efficient configuration
            wrapper = XGBoostModelWrapper(
                params=config['params'],
                logger=logger,
                eval_set=config['eval_set'],
                num_boost_round=config.get('num_boost_round', 1000),
                early_stopping_rounds=config.get('early_stopping_rounds', 50)
            )
            return wrapper
        else:
            raise ValueError(f"Unsupported model type: {model_type}") 