from typing import Dict, Tuple, Optional, List, Any
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_manager import ProcessedDataManager
from logger import ExperimentLogger
from pipeline.config_loader import load_config


class DataLoader:
    """Handles data loading and preprocessing for soccer match prediction."""
    
    def __init__(
        self,
        data_path: Path,
        target_column: str,
        logger: Optional[ExperimentLogger] = None,
        is_prediction: bool = False
    ) -> None:
        """Initialize data loader.
        
        Args:
            data_path: Path to data directory
            target_column: Name of target column
            logger: Optional logger instance
            is_prediction: Whether the data is for prediction
        """
        # Initialize logger first
        self.logger = logger or ExperimentLogger()
        
        try:
            # Load configurations
            self.logger.info("Loading configurations...")
            self.base_config = load_config('base')
            self.data_config = load_config('processing')
            self.feature_config = load_config('feature')
            self.is_prediction = is_prediction
            
            # Validate required configuration
            self._validate_config()
            
            # Set paths
            data_paths = self.base_config.get('data_paths', {})
            self.raw_data_path = Path(data_path)
            self.processed_data_path = Path(data_paths.get('processed_data_path'))
            self.cache_path = Path(data_paths.get('cache_path'))
            
            # Create directories
            for path in [self.processed_data_path, self.cache_path]:
                path.mkdir(parents=True, exist_ok=True)
            
            # Set configuration values
            self.target_column = self.data_config['processing']['target_column']
            self.data_validation = self.data_config['data_validation']
            self.encoding = self.data_config['encoding']
            self.column_thresholds = self.data_config['column_thresholds']
            
            # Get feature groups from feature config
            self.feature_groups = self.feature_config['feature_groups']
            
            # Initialize components
            self.data_manager = ProcessedDataManager(
                base_path=str(self.processed_data_path),
                data_path=str(self.raw_data_path)
            )
            
            # Initialize containers
            self.scalers: Dict[str, StandardScaler] = {}
            self.label_encoders: Dict[str, LabelEncoder] = {}
            self.raw_data: Optional[pd.DataFrame] = None
            self.processed_data: Optional[Dict[str, pd.DataFrame]] = None
            
            self.logger.info("DataLoader initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing DataLoader: {str(e)}")
            raise
            
    def _validate_config(self) -> None:
        """Validate required configuration values."""
        # Validate data paths
        required_paths = ['processed_data_path', 'cache_path']
        for path in required_paths:
            if path not in self.base_config.get('data_paths', {}):
                raise ValueError(f"Missing required path '{path}' in base config")
        
        # Validate data validation config
        required_validation = ['max_goals', 'min_matches']
        for param in required_validation:
            if param not in self.data_config.get('data_validation', {}):
                raise ValueError(f"Missing required parameter '{param}' in data validation config")
        
        # Validate processing config
        if 'processing' not in self.data_config:
            raise ValueError("Missing 'processing' section in data config")
        if 'target_column' not in self.data_config['processing']:
            raise ValueError("Missing 'target_column' in processing config")
        
        # Validate feature config
        if 'feature_groups' not in self.feature_config:
            raise ValueError("Missing 'feature_groups' section in feature config")
        
    def load_or_convert_data(self) -> pd.DataFrame:
        """Load data from source file or cached parquet file.
        
        Returns:
            Loaded DataFrame from either or source file
        """
        # Get file paths and formats from config
        source_path = None
        source_format = None
        
        if self.is_prediction:
            # Use absolute path for prediction data
            prediction_path = Path(self.base_config.get('data_paths', {}).get(
                'raw_data_path', 
                'data/prediction_raw'
            ))
            prediction_path.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
            excel_files = list(prediction_path.glob('*.xlsx'))
            if excel_files:
                source_path = excel_files[0]
                source_format = 'xlsx'
        else:
            excel_files = list(self.raw_data_path.glob('*.xlsx'))
            if excel_files:
                source_path = excel_files[0]
                source_format = 'xlsx'
        
        if not source_path:
            raise ValueError(f"No supported data file found in {'prediction_raw' if self.is_prediction else 'raw'} data directory, prediction_path: {prediction_path}")
        
        cache_format = self.data_config.get('cache_format', 'parquet')
        cache_path = self.cache_path / f"raw_data.{cache_format}"
        
        self.logger.info(f"Checking for cached data at {cache_path}")
        
        # If cache exists and is newer than source, load it
        if cache_path.exists() and cache_path.stat().st_mtime > source_path.stat().st_mtime:
            self.logger.info(f"Loading from {cache_format} cache...")
            if cache_format == 'parquet':
                return pd.read_parquet(cache_path)
            else:
                raise ValueError(f"Unsupported cache format: {cache_format}")
        
        # Otherwise load from source and cache
        self.logger.info("Loading from source and creating cache...")
        
        try:
            # Load based on format
            if source_format == 'xlsx':
                df = pd.read_excel(
                    source_path,
                    engine='openpyxl',
                    sheet_name=self.data_config.get('excel_sheet', 'Sheet1'),
                    dtype_backend='pyarrow',
                    engine_kwargs={'read_only': True}
                )
            elif source_format == 'parquet':
                df = pd.read_parquet(source_path)
            else:
                raise ValueError(f"Unsupported source format: {source_format}")
            
            # Cache the data
            df.to_parquet(
                cache_path,
                engine='pyarrow',
                compression=self.data_config.get('parquet_compression', 'snappy')
            )
            self.logger.info(f"Successfully cached data to {cache_path}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
            
    def load_raw_data(self) -> pd.DataFrame:
        """Load raw data from source or cache.
        
        Returns:
            Raw DataFrame
        """
        self.logger.info("Loading raw data...")
        
        self.raw_data = self.load_or_convert_data()
        self.logger.info(f"Loaded {len(self.raw_data)} rows of raw data")
        
        return self.raw_data
        
    def preprocess_data(self, df: pd.DataFrame = None) -> Dict[str, pd.DataFrame]:
        """Preprocess data and split into train/val/test sets.
        
        Returns:
            Dictionary containing train/val/test splits
        """
        
        # Load raw data if not already loaded
        if df is None:
            df = self.load_raw_data()
        else:
            self.raw_data = df
        # Get feature groups from config
        feature_groups = self.feature_config.get('feature_groups', {})
        
        # Encode 'Datum' column for 'date_encoded'
        if 'Datum' in self.raw_data.columns:
            earliest_date = pd.to_datetime(df['Datum']).min()
            df['date_encoded'] = (pd.to_datetime(df['Datum']) - earliest_date).dt.days
            df['date_encoded'] = df['date_encoded'].astype(int)
        
        # Map 'match_outcome' column as 1:0, 2:1, 3:2
        if 'match_outcome' in df.columns:
            df['match_outcome'] = df['match_outcome'].map({1: 0, 2: 1, 3: 2})
            
        # Add is_draw column based on match outcome
        if 'match_outcome' in df.columns:
            df['is_draw'] = df['match_outcome'].apply(lambda x: 1 if x == 1 else 0)   
        # Drop excluded columns
        excluded_columns = self.data_config.get('excluded_columns', [])
        print(f"Excluded columns: {excluded_columns}")
        df = df.drop(columns=excluded_columns, errors='ignore')
        
        self.logger.info(f"Columns after exclusion: {df.columns}")
        print(df.head())
        df = self.convert_pyarrow_dtypes(df)

        df.dropna(inplace=True)
          
        # Cache the data
        cache_path = self.cache_path / f"raw_data.parquet"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(
            cache_path,
            engine='pyarrow',
            compression=self.data_config.get('parquet_compression', 'snappy')
        )
        
        return df
    
    def convert_pyarrow_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert PyArrow dtypes to standard pandas dtypes.
        
        Args:
            df: Input DataFrame with PyArrow dtypes
            
        Returns:
            DataFrame with standard pandas dtypes
        """
        df_converted = df.copy()
        
        # Get dtype mappings from config
        dtype_mappings = self.data_config.get('dtype_mappings', {
            'int64[pyarrow]': 'int64',
            'double[pyarrow]': 'float64',
            'string[pyarrow]': 'object',
            'timestamp[us][pyarrow]': 'datetime64[ns]'
        })
        
        for source_type, target_type in dtype_mappings.items():
            cols = df.select_dtypes(include=[source_type]).columns
            for col in cols:
                try:
                    df_converted[col] = df[col].astype(target_type)
                except Exception as e:
                    self.logger.warning(f"Could not convert column {col} to {target_type}: {str(e)}")
        
        return df_converted
    
    def get_feature_stats(self) -> Dict[str, Dict[str, float]]:
        """Get basic statistics for numerical features.
        
        Returns:
            Dictionary of feature statistics
        """
        if self.processed_data is None:
            raise ValueError("Data has not been processed yet")
        
        X = self.processed_data['X_train']
        numerical_features = [
            col for col in X.columns 
            if any(col.startswith(prefix) for prefix in self.feature_prefixes['numerical'])
        ]
        
        stats = {}
        for feature in numerical_features:
            stats[feature] = {
                'mean': float(X[feature].mean()),
                'std': float(X[feature].std()),
                'min': float(X[feature].min()),
                'max': float(X[feature].max()),
                'missing': float(X[feature].isnull().sum() / len(X))
            }
        
        return stats
    
    def get_target_distribution(self) -> pd.Series:
        """Get target variable distribution.
        
        Returns:
            Series with target distribution
        """
        if self.processed_data is None:
            raise ValueError("Data has not been processed yet")
        
        return self.processed_data['y_train'].value_counts(normalize=True)
    
   