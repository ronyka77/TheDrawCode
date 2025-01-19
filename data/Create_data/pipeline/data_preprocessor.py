"""Data preprocessing utilities."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger import ExperimentLogger
from pipeline.config_loader import load_config, get_environment


class DataPreprocessor:
    """Handles data preprocessing for soccer match prediction."""
    
    def __init__(
        self,
        feature_groups: Optional[Dict[str, List[str]]] = None,
        logger: Optional[ExperimentLogger] = None
    ) -> None:
        """Initialize data preprocessor.
        
        Args:
            feature_groups: Optional feature groups from config
            logger: Optional logger instance
        """
        try:
            # Initialize logger first
            self.logger = logger or ExperimentLogger()
            
            # Load configurations with environment
            environment = get_environment()
            self.logger.info(
                "Loading preprocessor configurations",
                extra={
                    'environment': environment
                }
            )
            
            # Load configurations
            self.base_config = load_config('base', environment)
            self.processing_config = load_config('processing', environment)
            self.feature_config = load_config('feature', environment)
            
            # Store feature groups
            self.feature_groups = feature_groups or self.feature_config.get('feature_groups', {})
            
            self.logger.info(
                "DataPreprocessor initialized successfully",
                extra={
                    'preprocessing_config': self.processing_config.get('preprocessing', {}),
                    'feature_config': self.feature_config.get('preprocessing', {}),
                    'feature_groups': list(self.feature_groups.keys())
                }
            )
            
        except Exception as e:
            self.logger.error(
                f"Error initializing DataPreprocessor: {str(e)}",
                extra={
                    'error_type': type(e).__name__,
                    'error_details': str(e)
                }
            )
            raise
            
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data according to configuration.
        
        Args:
            data: Input DataFrame to preprocess
            
        Returns:
            Preprocessed DataFrame
        """
        try:
            self.logger.info(
                "Starting data preprocessing",
                extra={
                    'input_shape': data.shape,
                    'initial_nan_count': data.isna().sum().sum()
                }
            )
            
            # Handle infinite values first
            data = self._handle_infinite_values(data)
            
            # Validate input data
            self._validate_input_data(data)
            
            # Get preprocessing settings
            preprocessing_config = self.processing_config.get('preprocessing', {})
            
            # Handle missing values
            data = self._handle_missing_values(data)
            
            # Validate after missing value handling
            self._validate_missing_values(data)
            
            # Scale numerical features
            data = self._scale_features(data)
            
            # Apply feature transformations
            data = self._transform_features(data)
            
            # Final validation
            self._validate_output_data(data)
            
            self.logger.info(
                "Data preprocessing completed",
                extra={
                    'output_shape': data.shape,
                    'final_nan_count': data.isna().sum().sum(),
                    'preprocessing_steps': [
                        'missing_value_handling',
                        'feature_scaling',
                        'feature_encoding',
                        'feature_transformation'
                    ]
                }
            )
            
            return data
            
        except Exception as e:
            self.logger.error(
                f"Error during preprocessing: {str(e)}",
                extra={
                    'error_type': type(e).__name__,
                    'error_details': str(e)
                }
            )
            raise
            
    def _handle_infinite_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle infinite values in the data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with handled infinite values
        """
        try:
            # Get numerical columns
            numerical_columns = [col for col in data.select_dtypes(include=np.number).columns]
            
            # Handle infinite values in mean columns
            mean_columns = [col for col in numerical_columns if 'mean' in col]
            for col in mean_columns:
                # Get mask for infinite values
                inf_mask = np.isinf(data[col])
                if inf_mask.any():
                    # Calculate mean excluding infinite values
                    valid_mean = data[col][~inf_mask].mean()
                    
                    # Replace infinite values with valid mean
                    data.loc[inf_mask, col] = valid_mean
                    
                    self.logger.info(
                        f"Replaced {inf_mask.sum()} infinite values in {col} with mean {valid_mean:.2f}"
                    )
            
            # Handle infinite values in other numerical columns
            other_num_cols = [col for col in numerical_columns if col not in mean_columns]
            for col in other_num_cols:
                # Get mask for infinite values
                inf_mask = np.isinf(data[col])
                if inf_mask.any():
                    # Calculate median excluding infinite values
                    valid_median = data[col][~inf_mask].median()
                    
                    # Replace infinite values with valid median
                    data.loc[inf_mask, col] = valid_median
                    
                    self.logger.info(
                        f"Replaced {inf_mask.sum()} infinite values in {col} with median {valid_median:.2f}"
                    )
            
            return data
            
        except Exception as e:
            self.logger.error(
                f"Error handling infinite values: {str(e)}",
                extra={
                    'error_type': type(e).__name__,
                    'error_details': str(e)
                }
            )
            raise
            
    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """Validate input data before preprocessing.
        
        Args:
            data: Input DataFrame
            
        Raises:
            ValueError: If validation fails
        """
        # Check for required columns
        required_columns = (
            self.feature_config.get('numerical', []) +
            self.feature_config.get('categorical', [])
        )
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Check for infinite values
        inf_mask = np.isinf(data.select_dtypes(include=np.number))
        if inf_mask.any().any():
            inf_cols = data.columns[inf_mask.any()].tolist()
            raise ValueError(f"Infinite values found in columns: {inf_cols}")
            
        # Log NaN statistics
        nan_stats = data.isna().sum()
        if nan_stats.any():
            self.logger.warning(
                "NaN values found in input data",
                extra={
                    'nan_counts': nan_stats[nan_stats > 0].to_dict()
                }
            )
            
    def _validate_missing_values(self, data: pd.DataFrame) -> None:
        """Validate data after missing value handling.
        
        Args:
            data: DataFrame after missing value handling
            
        Raises:
            ValueError: If validation fails
        """
        # Check if any NaN values remain
        nan_mask = data.isna()
        if nan_mask.any().any():
            nan_cols = data.columns[nan_mask.any()].tolist()
            raise ValueError(f"NaN values still present after handling in columns: {nan_cols}")
            
    def _validate_output_data(self, data: pd.DataFrame) -> None:
        """Validate output data after all preprocessing steps.
        
        Args:
            data: Preprocessed DataFrame
            
        Raises:
            ValueError: If validation fails
        """
        # Check for NaN values
        if data.isna().any().any():
            raise ValueError("NaN values found in preprocessed data")
            
        # Check for infinite values
        if np.isinf(data.select_dtypes(include=np.number)).any().any():
            raise ValueError("Infinite values found in preprocessed data")
            
        # Check value ranges for scaled numerical features
        numerical_columns = self.feature_config.get('numerical', [])
        for col in numerical_columns:
            if col in data.columns:
                col_data = data[col]
                if col_data.min() < -10 or col_data.max() > 10:
                    self.logger.warning(
                        f"Unusual values in scaled column {col}",
                        extra={
                            'min_value': col_data.min(),
                            'max_value': col_data.max()
                        }
                    )
            
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with handled missing values
        """
        try:
            missing_config = self.processing_config.get('missing_handling', {})
            numerical_strategy = missing_config.get('numerical_strategy', 'mean')
            categorical_strategy = missing_config.get('categorical_strategy', 'mode')
            
            # Handle numerical columns
            numerical_columns = self.feature_config.get('numerical', [])
            numerical_columns = [col for col in numerical_columns if col in data.columns]
            
            if numerical_columns:
                for col in numerical_columns:
                    null_mask = data[col].isnull()
                    if null_mask.any():
                        if numerical_strategy == 'mean':
                            # Calculate mean excluding infinite values
                            valid_data = data[col][~np.isinf(data[col])]
                            fill_value = valid_data.mean()
                        elif numerical_strategy == 'median':
                            # Calculate median excluding infinite values
                            valid_data = data[col][~np.isinf(data[col])]
                            fill_value = valid_data.median()
                        elif numerical_strategy == 'zero':
                            fill_value = 0
                        else:
                            raise ValueError(f"Unknown numerical strategy: {numerical_strategy}")
                            
                        data.loc[null_mask, col] = fill_value
                        self.logger.info(
                            f"Filled {null_mask.sum()} missing values in {col}",
                            extra={
                                'strategy': numerical_strategy,
                                'fill_value': fill_value
                            }
                        )
            
            # Handle categorical columns
            categorical_columns = self.feature_config.get('categorical', [])
            categorical_columns = [col for col in categorical_columns if col in data.columns]
            
            if categorical_columns:
                for col in categorical_columns:
                    null_mask = data[col].isnull()
                    if null_mask.any():
                        if categorical_strategy == 'mode':
                            fill_value = data[col].mode()[0]
                        elif categorical_strategy == 'unknown':
                            fill_value = 'UNKNOWN'
                        else:
                            raise ValueError(f"Unknown categorical strategy: {categorical_strategy}")
                            
                        data.loc[null_mask, col] = fill_value
                        self.logger.info(
                            f"Filled {null_mask.sum()} missing values in {col}",
                            extra={
                                'strategy': categorical_strategy,
                                'fill_value': fill_value
                            }
                        )
            
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
            scaling_config = self.processing_config.get('preprocessing', {}).get('scaling', {})
            method = scaling_config.get('method', 'standard')
            
            # Get numerical columns from feature config
            numerical_columns = self.feature_config.get('numerical', [])
            numerical_columns = [col for col in numerical_columns if col in data.columns]
            
            if not numerical_columns:
                self.logger.warning("No numerical columns found for scaling")
                return data
            
            if method == 'standard':
                # Handle infinite values before scaling
                for col in numerical_columns:
                    inf_mask = np.isinf(data[col])
                    if inf_mask.any():
                        # Calculate mean excluding infinite values
                        valid_mean = data[col][~inf_mask].mean()
                        
                        self.logger.warning(
                            f"Replacing infinite values in {col}",
                            extra={'count': inf_mask.sum()}
                        )
                        data.loc[inf_mask, col] = valid_mean
                
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(data[numerical_columns])
                
                # Replace any remaining NaN or infinite values after scaling
                scaled_data = np.nan_to_num(scaled_data, nan=0.0, posinf=10.0, neginf=-10.0)
                
                data[numerical_columns] = scaled_data
                self.scalers['numerical'] = scaler
                
                # Log scaling statistics
                for i, col in enumerate(numerical_columns):
                    self.logger.info(
                        f"Scaled column {col}",
                        extra={
                            'mean': scaler.mean_[i],
                            'scale': scaler.scale_[i]
                        }
                    )
            
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
     
        except Exception as e:
            self.logger.error(
                f"Error encoding features: {str(e)}",
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
            
            # Validate transformed data
            self._validate_output_data(data)
            
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