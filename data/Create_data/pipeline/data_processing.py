"""Data processing utilities."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger import ExperimentLogger
from pipeline.config_loader import load_config, get_environment


class DataProcessor:
    """Handles data processing for model training."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        feature_groups: Optional[Dict[str, List[str]]] = None,
        logger: Optional[ExperimentLogger] = None
    ) -> None:
        """Initialize data processor.
        
        Args:
            config: Processing configuration dictionary
            feature_groups: Optional feature groups configuration
            logger: Optional logger instance
        """
        try:
            # Initialize logger first
            self.logger = logger or ExperimentLogger()
            
            # Store configuration
            self.config = config
            self.feature_groups = feature_groups or {}
            
            # Load configurations with environment
            environment = get_environment()
            self.logger.info(
                "Loading processor configurations",
                extra={
                    'environment': environment
                }
            )
            
            # Load configurations
            self.base_config = load_config('base', environment)
            self.feature_config = load_config('feature', environment)
            
            # Initialize processing components
            self.scalers = {}
            self.encoders = {}
            
            self.logger.info(
                "DataProcessor initialized successfully",
                extra={
                    'preprocessing_config': self.config.get('preprocessing', {}),
                    'feature_config': self.feature_config.get('preprocessing', {}),
                    'feature_groups': list(self.feature_groups.keys())
                }
            )
            
        except Exception as e:
            self.logger.error(
                f"Error initializing DataProcessor: {str(e)}",
                extra={
                    'error_type': type(e).__name__,
                    'error_details': str(e)
                }
            )
            raise
            
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process data according to configuration.
        
        Args:
            data: Input DataFrame to process
            
        Returns:
            Processed DataFrame
        """
        try:
            self.logger.info(
                "Starting data processing",
                extra={
                    'input_shape': data.shape
                }
            )
            
            # Get processing settings
            processing_config = self.config.get('processing', {})
            
            # Handle missing values
            data = self._handle_missing_values(data)
            
            # Convert all columns to float32
            data = self._convert_to_float32(data)
            
            # Scale numerical features
            data = self._scale_features(data)
            
            # Ensure all columns are numeric
            data = self._ensure_numeric(data)
            
            self.logger.info(
                "Data processing completed",
                extra={
                    'output_shape': data.shape,
                    'processing_steps': [
                        'missing_value_handling',
                        'categorical_encoding',
                        'feature_scaling',
                        'feature_transformation',
                        'numeric_conversion',
                        'float32_conversion'
                    ]
                }
            )
            
            return data
            
        except Exception as e:
            self.logger.error(
                f"Error during data processing: {str(e)}",
                extra={
                    'error_type': type(e).__name__,
                    'error_details': str(e)
                }
            )
            raise
            
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with handled missing values
        """
        try:
            missing_config = self.config.get('missing_handling', {})
            numerical_strategy = missing_config.get('numerical_strategy', 'mean')
            categorical_strategy = missing_config.get('categorical_strategy', 'mode')
            
            # Handle numerical columns
            numerical_columns = self.config.get('numerical_columns', [])
            numerical_columns = [col for col in numerical_columns if col in data.columns]
            
            if numerical_columns:
                for col in numerical_columns:
                    if data[col].isnull().any():
                        if numerical_strategy == 'mean':
                            data[col].fillna(data[col].mean(), inplace=True)
                        elif numerical_strategy == 'median':
                            data[col].fillna(data[col].median(), inplace=True)
                        elif numerical_strategy == 'zero':
                            data[col].fillna(0, inplace=True)
            
            # Handle categorical columns
            categorical_columns = self.config.get('categorical_columns', [])
            categorical_columns = [col for col in categorical_columns if col in data.columns]
            
            if categorical_columns:
                for col in categorical_columns:
                    if data[col].isnull().any():
                        if categorical_strategy == 'mode':
                            data[col].fillna(data[col].mode()[0], inplace=True)
                        elif categorical_strategy == 'unknown':
                            data[col].fillna('UNKNOWN', inplace=True)
            
            return data
            
        except Exception as e:
            self.logger.error(
                f"Error handling missing values: {str(e)}",
                extra={
                    'error_type': type(e).__name__,
                    'error_details': str(e)
                }
            )
            raise
            
    def _scale_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with scaled features
        """
        try:
            scaling_config = self.config.get('preprocessing', {}).get('scaling', {})
            method = scaling_config.get('method', 'standard')
            
            # Get numerical columns from config
            numerical_columns = self.config.get('numerical_columns', [])
            numerical_columns = [col for col in numerical_columns if col in data.columns]
            
            if not numerical_columns:
                self.logger.warning("No numerical columns found for scaling")
                return data
            
            if method == 'standard':
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(data[numerical_columns])
                data[numerical_columns] = scaled_data
                self.scalers['numerical'] = scaler
            
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
            
            # Apply transformations based on config
            # Add custom transformations here
            
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
            
    def _ensure_numeric(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure all columns are numeric.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with all numeric columns
        """
        try:
            # Get target column
            target_col = self.config.get('target_column')
            
            # Convert each column to numeric
            for col in data.columns:
                if col != target_col:  # Skip target column
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        self.logger.warning(f"Converting non-numeric column {col} to numeric")
                        try:
                            # Try to convert directly to float
                            data[col] = pd.to_numeric(data[col], errors='coerce')
                        except:
                            # If that fails, encode as categorical
                            self.logger.warning(f"Encoding non-numeric column {col} as categorical")
                            encoder = LabelEncoder()
                            data[col] = encoder.fit_transform(data[col].astype(str))
                            self.encoders[col] = encoder
                        
                        # Fill any NaN values with 0
                        data[col] = data[col].fillna(0)
            
            return data
            
        except Exception as e:
            self.logger.error(
                f"Error ensuring numeric columns: {str(e)}",
                extra={
                    'error_type': type(e).__name__,
                    'error_details': str(e)
                }
            )
            raise
            
    def _convert_to_float32(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert all columns to float32.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with all columns as float32
        """
        try:
            # Get target column
            target_col = self.config.get('target_column')
            
            # Convert each column to float32
            for col in data.columns:
                if col != target_col:  # Skip target column
                    try:
                        data[col] = data[col].astype(np.float32)
                    except:
                        self.logger.warning(f"Failed to convert column {col} to float32, dropping")
                        data.drop(columns=[col], inplace=True)
            
            return data
            
        except Exception as e:
            self.logger.error(
                f"Error converting to float32: {str(e)}",
                extra={
                    'error_type': type(e).__name__,
                    'error_details': str(e)
                }
            )
            raise