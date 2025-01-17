"""Data management utilities."""

import os
import json
import shutil
from typing import Dict, Optional, Any, Union
import pandas as pd
from pathlib import Path
from datetime import datetime


class ProcessedDataManager:
    """Manages processed data storage and versioning."""
    
    def __init__(
        self,
        base_path: str,
        data_path: str
    ) -> None:
        """Initialize data manager.
        
        Args:
            base_path: Base path for storing processed data
            data_path: Path to raw data
        """
        self.base_path = Path(base_path)
        self.data_path = Path(data_path)
        
        # Create directories
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Set paths
        self.versions_path = self.base_path / 'versions'
        self.metadata_path = self.base_path / 'metadata'
        
        # Create subdirectories
        self.versions_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path.mkdir(parents=True, exist_ok=True)
        
    def is_processed_data_newer(self) -> bool:
        """Check if processed data is newer than raw data.
        
        Returns:
            True if processed data exists and is newer than raw data
        """
        # Get latest version if any
        versions = list(self.versions_path.glob('*.parquet'))
        if not versions:
            return False
            
        # Get latest version timestamp
        latest_version = max(versions, key=lambda p: p.stat().st_mtime)
        latest_version_time = latest_version.stat().st_mtime
        
        # Get raw data timestamp
        raw_data_files = list(self.data_path.glob('*'))
        if not raw_data_files:
            return False
            
        raw_data_time = max(f.stat().st_mtime for f in raw_data_files)
        raw_data_datetime = datetime.fromtimestamp(raw_data_time)
        latest_version_datetime = datetime.fromtimestamp(latest_version_time)
        print(f"raw_data_time: {raw_data_datetime}")
        print(f"latest_version_time: {latest_version_datetime}")
        
        return latest_version_time > raw_data_time
        
    def save_processed_data(
        self,
        data: Dict[str, Union[pd.DataFrame, pd.Series]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save processed data and metadata.
        
        Args:
            data: Dictionary of DataFrames or Series to save
            metadata: Optional metadata dictionary
            
        Returns:
            Version ID of saved data
        """
        # Generate version ID
        version_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Verify all required keys are present
        required_keys = ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise ValueError(f"Missing required data keys for saving: {missing_keys}")
        
        # Save each DataFrame or Series
        for name, df in data.items():
            file_path = self.versions_path / f"{name}_{version_id}.parquet"
            if isinstance(df, pd.Series):
                df = df.to_frame(name=name)
            df.to_parquet(file_path)
        
        # Save metadata if provided
        if metadata:
            metadata_file = self.metadata_path / f"metadata_{version_id}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        return version_id
        
    def load_processed_data(self) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
        """Load latest processed data.
        
        Returns:
            Dictionary of loaded DataFrames and Series
        """
        # Get latest version
        versions = list(self.versions_path.glob('*.parquet'))
        if not versions:
            raise ValueError("No processed data found")
        
        # Get version ID from latest file
        latest_file = max(versions, key=lambda p: p.stat().st_mtime)
        version_id = latest_file.stem.split('_')[-1]
        
        # Load all files for this version
        data = {}
        available_files = list(self.versions_path.glob(f'*_{version_id}.parquet'))
        print(f"Found files for version {version_id}, loading files...")
        
        for file in available_files:
            # Extract just the base name (X_train, y_test, etc.) without the date and version
            name = file.stem.split('_')[0]  # Get the first part (X or y)
            second_part = file.stem.split('_')[1]  # Get the second part (train, test, val)
            if second_part in ['train', 'test', 'val']:
                name = f"{name}_{second_part}"
            
            df = pd.read_parquet(file)
            # Convert single column DataFrames to Series for y values
            if name.startswith('y_') and df.shape[1] == 1:
                data[name] = df.iloc[:, 0]
            else:
                data[name] = df
        
        # Verify all required keys are present
        required_keys = ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            print(f"Available keys: {list(data.keys())}")
            raise ValueError(f"Missing required data: {missing_keys}")
        
        return data
        
    def get_metadata(self, version_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific version.
        
        Args:
            version_id: Version ID to get metadata for
            
        Returns:
            Metadata dictionary if found, None otherwise
        """
        metadata_file = self.metadata_path / f"metadata_{version_id}.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        return None
        
    def cleanup_old_versions(self, keep_latest: int = 5) -> None:
        """Clean up old versions of processed data.
        
        Args:
            keep_latest: Number of latest versions to keep
        """
        # Get all versions sorted by modification time
        versions = sorted(
            self.versions_path.glob('*.parquet'),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        # Keep only the latest N versions
        versions_to_keep = set()
        for file in versions[:keep_latest]:
            version_id = file.stem.split('_')[-1]
            versions_to_keep.add(version_id)
        
        # Remove old versions
        for file in self.versions_path.glob('*.parquet'):
            version_id = file.stem.split('_')[-1]
            if version_id not in versions_to_keep:
                file.unlink()
                
                # Remove corresponding metadata
                metadata_file = self.metadata_path / f"metadata_{version_id}.json"
                if metadata_file.exists():
                    metadata_file.unlink()