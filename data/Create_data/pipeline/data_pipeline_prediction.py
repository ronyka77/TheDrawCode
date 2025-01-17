"""Data import pipeline orchestration utilities."""

import time
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
import numpy as np
from pathlib import Path
import structlog
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
import os
import sys
from sklearn.model_selection import train_test_split

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Change relative imports to absolute imports
from logger import ExperimentLogger
from pipeline.config_loader import load_config, get_environment
from pipeline.data_loader import DataLoader
from pipeline.data_processing import DataProcessor
from pipeline.data_preprocessor import DataPreprocessor
from pipeline.data_splitter import DataSplitter
from pipeline.data_transformer import DataTransformer
from pipeline.data_manager import ProcessedDataManager
from pipeline.features.feature_engineering_base import FeatureEngineer_base

class DataImportError(Exception):
    """Custom exception for data import errors."""
    pass


class PipelineConfig(BaseModel):
    """Schema for data import pipeline configuration."""
    batch_size: int = Field(..., description="Batch size for data processing")
    num_workers: int = Field(..., description="Number of worker processes")
    cache_data: bool = Field(True, description="Whether to cache processed data")
    data_paths: Dict[str, Path] = Field(..., description="Data file paths")
    
    @field_validator('batch_size', 'num_workers')
    @classmethod
    def validate_positive(cls, v: int, field: Field) -> int:
        """Validate numeric values are positive."""
        if v <= 0:
            raise ValueError(f"{field.name} must be positive")
        return v
    
    class Config:
        arbitrary_types_allowed = True


class ValidationConfig(BaseModel):
    """Schema for data validation configuration."""
    perform_checks: bool = Field(True, description="Whether to perform validation checks")
    error_threshold: int = Field(..., description="Maximum allowed validation errors")
    validation_rules: Dict[str, Any] = Field(..., description="Data validation rules")


class ProcessingConfig(BaseModel):
    """Schema for data processing configuration."""
    processing: Dict[str, Any] = Field(..., description="Processing settings")
    data_validation: Dict[str, Any] = Field(..., description="Data validation settings")
    excluded_columns: List[str] = Field(..., description="Columns to exclude")
    column_thresholds: Dict[str, Any] = Field(..., description="Column threshold settings")
    encoding: Dict[str, Any] = Field(..., description="Encoding settings")
    preprocessing: Dict[str, Any] = Field(..., description="Preprocessing settings")
    missing_handling: Dict[str, Any] = Field(..., description="Missing value handling settings")
    validation: Dict[str, Any] = Field(..., description="Validation settings")
    
    class Config:
        extra = "allow"


class DataPipeline:
    """Orchestrates the complete data loading and processing pipeline."""
    
    def __init__(
        self,
        logger: Optional[ExperimentLogger] = None,
        config_path: Optional[str] = None
    ) -> None:
        """Initialize data importer.
        
        Args:
            logger: Optional logger instance
            config_path: Optional path to config file
        """
        self.logger = logger or ExperimentLogger()
        
        # Load and validate configurations
        base_config = load_config('base')
        processing_config = load_config('processing')
        feature_config = load_config('feature')
        
        # Get feature groups
        self.feature_groups = feature_config.get('feature_groups', {})
        if not self.feature_groups:
            raise ValueError("No feature groups found in feature config")
        
        # Initialize pipeline configuration
        self.pipeline_config = PipelineConfig(
            batch_size=processing_config['processing'].get('batch_size', 1000),
            num_workers=processing_config['processing'].get('num_workers', 4),
            cache_data=processing_config['processing'].get('cache_data', True),
            data_paths={
                k: Path(v) for k, v in base_config.get('data_paths', {}).items()
            }
        )
        
        # Initialize validation configuration
        self.validation_config = ValidationConfig(
            perform_checks=processing_config['validation'].get('perform_checks', True),
            error_threshold=processing_config['validation'].get('error_threshold', 100),
            validation_rules=processing_config['validation'].get('rules', {})
        )
        
        # Initialize processing configuration
        self.processing_config = ProcessingConfig(**processing_config)
        
        # Initialize components
        self.data_loader = DataLoader(
            data_path=self.pipeline_config.data_paths['raw_data_path'],
            target_column=processing_config['processing']['target_column'],
            logger=self.logger,
            is_prediction=True #IMPORTANT: This is for prediction data
        )
        
        self.data_processor = DataProcessor(
            config=processing_config,
            feature_groups=self.feature_groups,
            logger=self.logger
        )
        
        self.data_preprocessor = DataPreprocessor(
            feature_groups=self.feature_groups,
            logger=self.logger
        )
        
        self.data_splitter = DataSplitter(
            logger=self.logger
        )
        
        self.data_transformer = DataTransformer(
            feature_groups=self.feature_groups,
            logger=self.logger
        )
        
        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer_base(
            logger=self.logger,
            target_variable=self.processing_config.processing['target_column'],
            is_prediction=True
        )
        
        self.logger.info(
            "Initialized data importer",
            extra={
                'pipeline_config': self.pipeline_config.dict(),
                'validation_config': self.validation_config.dict(),
                'processing_config': self.processing_config.dict(),
                'feature_groups': list(self.feature_groups.keys())
            }
        )
        
    def load_and_process_data(
        self,
        force_reload: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Load and process data through the complete pipeline.
        
        Args:
            force_reload: Whether to force reload data from source
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        try:
            # Step 1: Load raw data
            self.logger.info("Loading raw data...")
            raw_data = self.data_loader.preprocess_data()
            
            # Step 2: Initial data processing
            self.logger.info("Processing data...")
            processed_data = self.data_processor.process_data(raw_data)
            
            # Step 3: Feature preprocessing
            self.logger.info("Preprocessing features...")
            preprocessed_data = self.data_preprocessor.preprocess(processed_data)
            
            # Add feature engineering step after preprocessing
            self.logger.info("Engineering features...")
            preprocessed_data = self.feature_engineer.engineer_all_features(preprocessed_data)
            
            # Save preprocessed data to Excel
            raw_data_path = "./data/prediction"
            preprocessed_data.to_excel(f"{raw_data_path}/preprocessed_data_prediction.xlsx", index=False)
            self.logger.info(f"Preprocessed data saved to {raw_data_path}/preprocessed_data_prediction.xlsx")
            # Drop 'match_outcome' column if it exists
            if 'match_outcome' in preprocessed_data.columns:
                preprocessed_data.drop(columns=['match_outcome'], inplace=True)
                self.logger.info("'match_outcome' column dropped from preprocessed data")
            
            self.logger.info("Saving preprocessed data to parquet...")
            preprocessed_data.to_parquet(f"{raw_data_path}/preprocessed_data_prediction.parquet", index=False)
            self.logger.info(f"Preprocessed data saved to {raw_data_path}/preprocessed_data_prediction.parquet")
            
            # Step 4: Split data
            self.logger.info("Splitting data...")
            split_data = self.data_splitter.split_data(
                preprocessed_data,
                target_col=self.processing_config.processing['target_column']
            )
            
            # Add feature engineering specific to splits
            X_train = split_data['X_train']
            X_val = split_data['X_val']
            X_test = split_data['X_test']
            
            # Transform features after engineering
            self.logger.info("Transforming features...")
            X_train = self.data_transformer.fit_transform(X_train)
            X_val = self.data_transformer.transform(X_val)
            X_test = self.data_transformer.transform(X_test)
            
            # Get target variables
            y_train = split_data['y_train']
            y_val = split_data['y_val']
            y_test = split_data['y_test']
            
            self.logger.info(
                "Data loading and processing completed",
                extra={
                    'train_shape': X_train.shape,
                    'val_shape': X_val.shape,
                    'test_shape': X_test.shape
                }
            )
            # Save processed data using ProcessedDataManager
            self.logger.info("Saving processed data...")
            processed_data_manager = ProcessedDataManager(
                base_path=self.pipeline_config.data_paths['processed_data_path'],
                data_path=self.pipeline_config.data_paths['raw_data_path']
            )
            
            data_to_save = {
                'X_train': pd.DataFrame(X_train),
                'X_val': pd.DataFrame(X_val),
                'X_test': pd.DataFrame(X_test),
                'y_train': pd.Series(y_train),
                'y_val': pd.Series(y_val),
                'y_test': pd.Series(y_test)
            }
            
            metadata = {
                'train_shape': X_train.shape,
                'val_shape': X_val.shape,
                'test_shape': X_test.shape,
                'feature_groups': list(self.feature_groups.keys()),
                'target_distribution': y_train.value_counts(normalize=True).to_dict(),
                'feature_count': X_train.shape[1],
                'preprocessing_steps': [
                    'train_val_test_split',
                    'numerical_scaling',
                    'categorical_encoding',
                    'feature_engineering',
                    'draw_specific_features'
                ],
                'timestamp': datetime.now().isoformat()
            }
            
            version_id = processed_data_manager.save_processed_data(data_to_save, metadata)
            self.logger.info(f"Processed data saved with version ID: {version_id}")
            
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except Exception as e:
            self.logger.error(f"Error in data loading and processing: {str(e)}")
            raise DataImportError(f"Failed to load and process data: {str(e)}")
        
    def check_and_import_latest_parquet(
        self,
        data_paths: Dict[str, Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Check the latest exported parquet version and import if newer than raw data.
        
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        try:
            if data_paths is None:
                base_config = load_config('base')
                data_paths = {
                    k: Path(v) for k, v in base_config.get('data_paths', {}).items()
                }
                
            processed_data_manager = ProcessedDataManager(
                base_path=data_paths['processed_data_path'],
                data_path=data_paths['raw_data_path']
            )
            
            self.logger.info("Processing new data...")
            return self.load_and_process_data()
            
        except Exception as e:
            self.logger.error(f"Error in checking and importing latest parquet: {str(e)}")
            raise DataImportError(f"Failed to check and import latest parquet: {str(e)}")
        
    def load_raw_data(self, data_paths: Dict[str, Path]) -> pd.DataFrame:
        """Load raw data before any splitting."""
        parquet_path = Path(data_paths['raw_data_path']) / 'preprocessed_data.parquet'
        print(f"parquet_path: {parquet_path}")
        if not parquet_path.exists():
            self.logger.warning(f"Raw data file not found: {parquet_path}")
            self.load_and_process_data()
            return pd.read_parquet(parquet_path)
        else:
            return pd.read_parquet(parquet_path)

    def split_data(
        self,
        data: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train/val/test sets."""
        
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        # First split: train+val and test
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Second split: train and val
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_size/(1-test_size), random_state=random_state
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
if __name__ == "__main__":
    # Initialize logger
    logger = ExperimentLogger()

    base_config = load_config('base')
    data_paths={
                k: Path(v) for k, v in base_config.get('data_paths', {}).items()
            }
    # Initialize DataPipeline
    data_pipeline = DataPipeline(logger=logger)

    # Run check_and_import_latest_parquet
    data_pipeline.check_and_import_latest_parquet(data_paths=data_paths)