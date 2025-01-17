"""Data splitting utilities."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from sklearn.model_selection import train_test_split

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger import ExperimentLogger
from pipeline.config_loader import load_config, get_environment


class DataSplitter:
    """Handles data splitting for model training."""
    
    def __init__(
        self,
        logger: Optional[ExperimentLogger] = None
    ) -> None:
        """Initialize data splitter.
        
        Args:
            logger: Optional logger instance
        """
        try:
            # Initialize logger first
            self.logger = logger or ExperimentLogger()
            
            # Load configurations with environment
            environment = get_environment()
            self.logger.info(
                "Loading splitter configurations",
                extra={
                    'environment': environment
                }
            )
            
            # Load configurations
            self.base_config = load_config('base', environment)
            self.processing_config = load_config('processing', environment)
            
            # Get split settings
            self.split_config = self.processing_config.get('validation', {})
            self.test_size = self.split_config.get('test_split', 0.2)
            self.val_size = self.split_config.get('validation_split', 0.2)
            self.random_state = self.processing_config.get('random_seed', 42)
            
            self.logger.info(
                "DataSplitter initialized successfully",
                extra={
                    'split_config': self.split_config,
                    'test_size': self.test_size,
                    'val_size': self.val_size
                }
            )
            
        except Exception as e:
            self.logger.error(
                f"Error initializing DataSplitter: {str(e)}",
                extra={
                    'error_type': type(e).__name__,
                    'error_details': str(e)
                }
            )
            raise
            
    def split_data(
        self,
        data: pd.DataFrame,
        target_col: str,
        stratify: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """Split data into train, validation, and test sets.
        
        Args:
            data: Input DataFrame
            target_col: Name of target column
            stratify: Whether to use stratified splitting
            
        Returns:
            Dictionary containing train/val/test splits
        """
        try:
            self.logger.info(
                "Starting data splitting",
                extra={
                    'data_shape': data.shape,
                    'target_col': target_col,
                    'stratify': stratify
                }
            )
            
            # Get features and target
            X = data.drop(columns=[target_col])
            y = data[target_col]
            
            # Get stratification if needed
            strat = y if stratify else None
            
            # First split: train+val vs test
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=strat
            )
            
            # Second split: train vs val
            # Adjust val_size to account for test split
            adjusted_val_size = self.val_size / (1 - self.test_size)
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val,
                test_size=adjusted_val_size,
                random_state=self.random_state,
                stratify=y_train_val if stratify else None
            )
            
            # Prepare output dictionary
            split_data = {
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test
            }
            
            # Log split sizes
            self.logger.info(
                "Data splitting completed",
                extra={
                    'train_shape': X_train.shape,
                    'val_shape': X_val.shape,
                    'test_shape': X_test.shape,
                    'train_target_dist': y_train.value_counts(normalize=True).to_dict(),
                    'val_target_dist': y_val.value_counts(normalize=True).to_dict(),
                    'test_target_dist': y_test.value_counts(normalize=True).to_dict()
                }
            )
            
            return split_data
            
        except Exception as e:
            self.logger.error(
                f"Error during data splitting: {str(e)}",
                extra={
                    'error_type': type(e).__name__,
                    'error_details': str(e)
                }
            )
            raise