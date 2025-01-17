"""Data transformation utilities."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
from sklearn.preprocessing import StandardScaler

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger import ExperimentLogger
from pipeline.config_loader import load_config, get_environment


class DataTransformer:
    """Handles data transformations for model training."""
    
    def __init__(
        self,
        logger: Optional[ExperimentLogger] = None,
        feature_groups: Optional[Dict[str, List[str]]] = None
    ) -> None:
        """Initialize data transformer.
        
        Args:
            logger: Optional logger instance
            feature_groups: Optional feature groups from config
        """
        try:
            # Initialize logger first
            self.logger = logger or ExperimentLogger()
            
            # Load configurations with environment
            environment = get_environment()
            self.logger.info(
                "Loading transformer configurations",
                extra={
                    'environment': environment
                }
            )
            
            # Load configurations
            self.base_config = load_config('base', environment)
            self.processing_config = load_config('processing', environment)
            self.feature_config = load_config('feature', environment)
            
            # Use provided feature groups or load from config
            self.feature_groups = feature_groups or self.feature_config.get('feature_groups', {})
            
            # Validate feature groups
            if not self.feature_groups:
                raise ValueError("No feature groups found in config")
            
            # Initialize transformation components
            self.scalers = {}
            
            # Get all expected columns
            self.expected_columns = self._get_all_columns()
            
            self.logger.info(
                "DataTransformer initialized successfully",
                extra={
                    'preprocessing_config': self.processing_config.get('preprocessing', {}),
                    'feature_config': self.feature_config.get('preprocessing', {}),
                    'expected_columns': len(self.expected_columns)
                }
            )
            
        except Exception as e:
            self.logger.error(
                f"Error initializing DataTransformer: {str(e)}",
                extra={
                    'error_type': type(e).__name__,
                    'error_details': str(e)
                }
            )
            raise
            
    def _get_all_columns(self) -> Set[str]:
        """Get all expected columns from feature groups.
        
        Returns:
            Set of column names
        """
        columns = set()
        for group in self.feature_groups.values():
            columns.update(group)
        return columns
            
    def _validate_columns(self, data: pd.DataFrame) -> None:
        """Validate that all required columns are present.
        
        Args:
            data: Input DataFrame
            
        Raises:
            ValueError: If required columns are missing
        """
        missing_columns = self.expected_columns - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform data according to configuration.
        
        Args:
            data: Input DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        try:
            self.logger.info(
                "Starting data transformation",
                extra={
                    'input_shape': data.shape
                }
            )
            
            # Validate columns
            self._validate_columns(data)
            
            # Scale numerical features
            data = self._scale_features(data, fit=True)
            
            # Apply feature transformations
            data = self._transform_features(data)
                
            self.logger.info(
                "Data transformation completed",
                extra={
                    'output_shape': data.shape,
                    'transformation_steps': [
                        'feature_scaling',
                        'feature_transformation'
                    ]
                }
            )
            
            return data
            
        except Exception as e:
            self.logger.error(
                f"Error during transformation: {str(e)}",
                extra={
                    'error_type': type(e).__name__,
                    'error_details': str(e)
                }
            )
            raise
            
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted transformers.
        
        Args:
            data: Input DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        try:
            self.logger.info(
                "Starting data transformation",
                extra={
                    'input_shape': data.shape
                }
            )
            
            # Validate columns
            self._validate_columns(data)
            
            # Scale numerical features
            data = self._scale_features(data, fit=False)
            
            # Apply feature transformations
            data = self._transform_features(data)
            
            self.logger.info(
                "Data transformation completed",
                extra={
                    'output_shape': data.shape,
                    'transformation_steps': [
                        'feature_scaling',
                        'feature_transformation'
                    ]
                }
            )
            
            return data
            
        except Exception as e:
            self.logger.error(
                f"Error during transformation: {str(e)}",
                extra={
                    'error_type': type(e).__name__,
                    'error_details': str(e)
                }
            )
            raise
            
    def _scale_features(self, data: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Scale numerical features.
        
        Args:
            data: Input DataFrame
            fit: Whether to fit the scaler
            
        Returns:
            DataFrame with scaled features
        """
        try:
            scaling_config = self.processing_config.get('preprocessing', {}).get('scaling', {})
            method = scaling_config.get('method', 'standard')
            
            # Get numerical columns from feature groups
            numerical_groups = ['raw_numerical', 'engineered_numerical', 'sequence_numerical', 'h2h_numerical']
            numerical_columns = []
            
            for group in numerical_groups:
                if group in self.feature_groups:
                    group_columns = self.feature_groups[group]
                    numerical_columns.extend([col for col in group_columns if col in data.columns])
            
            if not numerical_columns:
                self.logger.warning("No numerical columns found for scaling")
                return data
            
            self.logger.info(
                "Scaling numerical features",
                extra={
                    'num_columns': len(numerical_columns),
                    'columns': numerical_columns,
                    'method': method
                }
            )
            
            if method == 'standard':
                if fit:
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(data[numerical_columns])
                    data[numerical_columns] = scaled_data
                    self.scalers['numerical'] = scaler
                else:
                    if 'numerical' not in self.scalers:
                        raise ValueError("Scaler not fitted. Call fit_transform first.")
                    scaled_data = self.scalers['numerical'].transform(data[numerical_columns])
                    data[numerical_columns] = scaled_data
            
            return data
            
        except Exception as e:
            self.logger.error(
                f"Error scaling features: {str(e)}",
                extra={
                    'error_type': type(e).__name__,
                    'error_details': str(e)
                }
            )
            raise
            
    def _transform_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply feature transformations.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with transformed features
        """
        try:
            feature_config = self.feature_config.get('preprocessing', {})
            
            # Get categorical columns
            categorical_group = self.feature_groups.get('categorical', [])
            categorical_columns = [col for col in categorical_group if col in data.columns]
            
            if categorical_columns:
                self.logger.info(
                    "Processing categorical features",
                    extra={
                        'num_columns': len(categorical_columns),
                        'columns': categorical_columns
                    }
                )
            
            return data
            
        except Exception as e:
            self.logger.error(
                f"Error transforming features: {str(e)}",
                extra={
                    'error_type': type(e).__name__,
                    'error_details': str(e)
                }
            )
            raise