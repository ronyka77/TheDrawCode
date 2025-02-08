"""
Data ingestion and validation module for the Model Context Protocol Server.

This module handles:
- Loading and validating training/evaluation datasets
- Ensuring data quality and minimum sample counts
- Converting datasets into MLflow-compatible formats
- Providing feature extraction and transformation utilities
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
try:
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
except Exception as e:
    print(f"Error setting project root path: {e}")
    sys.path.append(os.getcwd())

# Local imports
from utils.logger import ExperimentLogger
from utils.create_evaluation_set import (
    import_training_data_ensemble,
    create_ensemble_evaluation_set,
    import_selected_features_ensemble,
    DataProcessingError
)

# Initialize logger
logger = ExperimentLogger(
    experiment_name="data_ingestion",
    log_dir="logs/data_ingestion"
)

class DataValidator:
    """Data validation and quality assurance."""
    
    MIN_SAMPLES = 1000
    REQUIRED_COLUMNS = None  # Will be loaded dynamically
    
    def __init__(self, model_type: str):
        """Initialize validator for specific model type."""
        self.model_type = model_type
        self.REQUIRED_COLUMNS = import_selected_features_ensemble(model_type)
        
    def validate_dataset(
        self,
        data: pd.DataFrame,
        is_training: bool = True
    ) -> Tuple[bool, Optional[str]]:
        """Validate a dataset meets all requirements.
        
        Args:
            data: Dataset to validate
            is_training: Whether this is training data
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check minimum samples for training data
            if is_training and len(data) < self.MIN_SAMPLES:
                return False, f"Dataset has {len(data)} samples, minimum required is {self.MIN_SAMPLES}"
            
            # Validate required columns
            missing_cols = set(self.REQUIRED_COLUMNS) - set(data.columns)
            if missing_cols:
                return False, f"Missing required columns: {missing_cols}"
            
            # Check for null values
            null_cols = data[self.REQUIRED_COLUMNS].columns[
                data[self.REQUIRED_COLUMNS].isnull().any()
            ].tolist()
            if null_cols:
                return False, f"Null values found in columns: {null_cols}"
            
            # Validate numeric types
            non_numeric = data[self.REQUIRED_COLUMNS].select_dtypes(
                exclude=['int64', 'float64']
            ).columns.tolist()
            if non_numeric:
                return False, f"Non-numeric columns found: {non_numeric}"
            
            return True, None
            
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return False, str(e)

class DataIngestion:
    """Data ingestion and preprocessing for model training."""
    
    def __init__(self, model_type: str):
        """Initialize ingestion for specific model type."""
        self.model_type = model_type
        self.validator = DataValidator(model_type)
        
    def load_training_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Load and validate training data.
        
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
            
        Raises:
            DataProcessingError: If data validation fails
        """
        try:
            logger.info("Loading training data...")
            X_train, y_train, X_test, y_test = import_training_data_ensemble()
            
            # Select required features
            X_train = X_train[self.validator.REQUIRED_COLUMNS]
            X_test = X_test[self.validator.REQUIRED_COLUMNS]
            
            # Validate datasets
            is_valid, error = self.validator.validate_dataset(X_train)
            if not is_valid:
                raise DataProcessingError(f"Training data validation failed: {error}")
                
            is_valid, error = self.validator.validate_dataset(X_test, is_training=False)
            if not is_valid:
                raise DataProcessingError(f"Test data validation failed: {error}")
            
            logger.info(
                "Data loading complete",
                extra={
                    "train_shape": X_train.shape,
                    "test_shape": X_test.shape,
                    "train_draws": int(y_train.sum()),
                    "test_draws": int(y_test.sum())
                }
            )
            
            return X_train, y_train, X_test, y_test
            
        except Exception as e:
            logger.error(f"Error loading training data: {str(e)}")
            raise
            
    def load_evaluation_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and validate evaluation data.
        
        Returns:
            Tuple of (X_eval, y_eval)
            
        Raises:
            DataProcessingError: If data validation fails
        """
        try:
            logger.info("Loading evaluation data...")
            X_eval, y_eval = create_ensemble_evaluation_set()
            
            # Select required features
            X_eval = X_eval[self.validator.REQUIRED_COLUMNS]
            
            # Validate dataset
            is_valid, error = self.validator.validate_dataset(X_eval, is_training=False)
            if not is_valid:
                raise DataProcessingError(f"Evaluation data validation failed: {error}")
            
            logger.info(
                "Evaluation data loading complete",
                extra={
                    "eval_shape": X_eval.shape,
                    "eval_draws": int(y_eval.sum())
                }
            )
            
            return X_eval, y_eval
            
        except Exception as e:
            logger.error(f"Error loading evaluation data: {str(e)}")
            raise
            
    def convert_to_mlflow_format(
        self,
        data: Union[pd.DataFrame, np.ndarray]
    ) -> pd.DataFrame:
        """Convert data to MLflow-compatible format.
        
        Args:
            data: Input data to convert
            
        Returns:
            DataFrame with MLflow-compatible types
        """
        try:
            # Convert to DataFrame if numpy array
            df = pd.DataFrame(data) if isinstance(data, np.ndarray) else data.copy()
            
            # Convert integer columns to float64 for MLflow compatibility
            for col in df.select_dtypes(include=['int']).columns:
                df[col] = df[col].astype('float64')
                
            return df
            
        except Exception as e:
            logger.error(f"Error converting data format: {str(e)}")
            raise

def get_feature_metadata() -> Dict[str, Dict[str, str]]:
    """Get metadata about features used in the models.
    
    Returns:
        Dictionary mapping feature names to their metadata
    """
    try:
        # Load feature metadata
        metadata = {}
        for model_type in ['xgb', 'cat', 'all']:
            features = import_selected_features_ensemble(model_type)
            metadata[model_type] = {
                feature: {
                    "type": "numeric",
                    "description": "Feature selected for prediction"
                }
                for feature in features
            }
        return metadata
        
    except Exception as e:
        logger.error(f"Error loading feature metadata: {str(e)}")
        raise 