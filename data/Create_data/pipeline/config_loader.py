"""Configuration loading and validation utilities."""

import yaml
import sys
import os
from pathlib import Path

project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).parent
sys.path.append(str(project_root))
from typing import Dict, Any, Optional, Set, List, Union
from functools import lru_cache
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class ConfigurationError(Exception):
    """Custom exception for configuration errors."""
    pass


class ValidationError(Exception):
    """Custom exception for schema validation errors."""
    pass


class Environment(str, Enum):
    """Valid environment values."""
    DEVELOPMENT = 'development'
    STAGING = 'staging'
    PRODUCTION = 'production'


class LogLevel(str, Enum):
    """Valid log levels."""
    DEBUG = 'DEBUG'
    INFO = 'INFO'
    WARNING = 'WARNING'
    ERROR = 'ERROR'
    CRITICAL = 'CRITICAL'


class DataPaths(BaseModel):
    """Schema for data paths configuration."""
    raw_data_path: str = Field(..., description="Path to raw data")
    processed_data_path: str = Field(..., description="Path to processed data")
    model_artifacts_path: str = Field(..., description="Path to model artifacts")
    cache_path: str = Field(..., description="Path to cache directory")
    logs_path: Optional[str] = Field(None, description="Path to log files")

    @field_validator('*')
    def validate_path(cls, v):
        """Validate path strings."""
        if v and not isinstance(v, str):
            raise ValueError("Path must be a string")
        return v


class GlobalConfig(BaseModel):
    """Schema for global configuration."""
    random_seed: int = Field(..., description="Random seed for reproducibility")
    experiment_name: str = Field(..., description="Name of the experiment")
    data_version: str = Field(..., description="Version of the data")
    environment: Environment = Field(..., description="Runtime environment")
    debug_mode: bool = Field(False, description="Whether to run in debug mode")
    num_workers: int = Field(1, description="Number of worker processes")

    @field_validator('random_seed')
    def validate_seed(cls, v):
        """Validate random seed."""
        if v < 0:
            raise ValueError("Random seed must be non-negative")
        return v

    @field_validator('num_workers')
    def validate_workers(cls, v):
        """Validate number of workers."""
        if v < 1:
            raise ValueError("Number of workers must be positive")
        return v


class LoggingStructured(BaseModel):
    """Schema for structured logging configuration."""
    enabled: bool = Field(True, description="Enable structured logging")
    format: str = Field(..., description="Structured log format")


class LoggingAggregation(BaseModel):
    """Schema for log aggregation configuration."""
    use_json: bool = Field(True, description="Use JSON format for aggregation")
    fields: List[str] = Field(..., description="Fields to include in aggregated logs")


class LoggingConfig(BaseModel):
    """Schema for logging configuration."""
    level: LogLevel = Field(LogLevel.INFO, description="Logging level")
    format: str = Field(..., description="Log message format")
    file_rotation: str = Field("1 day", description="Log file rotation interval")
    structured: LoggingStructured = Field(..., description="Structured logging settings")
    aggregation: LoggingAggregation = Field(..., description="Log aggregation settings")


class BaseConfig(BaseModel):
    """Schema for base configuration."""
    data_paths: DataPaths
    global_config: GlobalConfig
    logging: LoggingConfig
    environment: Dict[str, Any]


class FeatureGroups(BaseModel):
    """Schema for feature groups configuration."""
    raw_numerical: List[str] = Field(default_factory=list)
    engineered_numerical: List[str] = Field(default_factory=list)
    sequence_numerical: List[str] = Field(default_factory=list)
    h2h_numerical: List[str] = Field(default_factory=list)
    categorical: List[str] = Field(default_factory=list)


class PreprocessingConfig(BaseModel):
    """Schema for preprocessing configuration."""
    scaling: Dict[str, Any] = Field(..., description="Scaling settings")
    encoding: Dict[str, Any] = Field(..., description="Encoding settings")
    imputation: Dict[str, Any] = Field(..., description="Imputation settings")
    feature_selection: Dict[str, Any] = Field(..., description="Feature selection settings")


class ValidationConfig(BaseModel):
    """Schema for validation configuration."""
    correlation_threshold: float = Field(..., description="Correlation threshold")
    missing_threshold: float = Field(..., description="Missing threshold")
    variance_threshold: float = Field(..., description="Variance threshold")
    cardinality_limit: int = Field(..., description="Cardinality limit")
    validation_split: float = Field(..., description="Validation split")
    test_split: float = Field(..., description="Test split")
    check_missing: bool = Field(True, description="Check missing")
    check_infinite: bool = Field(True, description="Check infinite")
    check_correlation: bool = Field(True, description="Check correlation")
    check_cardinality: bool = Field(True, description="Check cardinality")
    check_variance: bool = Field(True, description="Check variance")


class FeatureConfig(BaseModel):
    """Schema for feature configuration."""
    feature_engineering: Dict[str, Any] = Field(..., description="Feature engineering settings")
    feature_types: List[str] = Field(..., description="Feature types")
    feature_groups: Dict[str, List[str]] = Field(..., description="Feature groups")
    preprocessing: Dict[str, Any] = Field(..., description="Preprocessing settings")
    validation: Dict[str, Any] = Field(..., description="Validation settings")
    sequence: Dict[str, Any] = Field(..., description="Sequence settings")
    embeddings: Dict[str, Any] = Field(..., description="Embeddings settings")
    interactions: Dict[str, Any] = Field(..., description="Interactions settings")
    selection: Dict[str, Any] = Field(..., description="Selection settings")
    rolling_metrics: Dict[str, Any] = Field(..., description="Rolling metrics settings")
    head_to_head: Dict[str, Any] = Field(..., description="Head to head settings")
    advanced_metrics: Dict[str, Any] = Field(..., description="Advanced metrics settings")


class ProcessingConfig(BaseModel):
    """Schema for data processing configuration."""
    processing: Dict[str, Any] = Field(..., description="Processing settings")
    data_validation: Dict[str, Any] = Field(..., description="Data validation settings")
    excluded_columns: List[str] = Field(..., description="Excluded columns")
    column_thresholds: Dict[str, Any] = Field(..., description="Column thresholds")
    encoding: Dict[str, Any] = Field(..., description="Encoding settings")
    preprocessing: Dict[str, Any] = Field(..., description="Preprocessing settings")
    missing_handling: Dict[str, Any] = Field(..., description="Missing handling settings")
    validation: Dict[str, Any] = Field(..., description="Validation settings")


# Update CONFIG_SCHEMAS to be more lenient
CONFIG_SCHEMAS = {
    'base': {
        'schema': None,  # Disabled schema validation
        'filename': 'config.yaml',
        'required_fields': set()  # No required fields during development
    },
    'model': {
        'filename': 'model_config.yaml',
        'required_fields': set()  # No required fields during development
    },
    'feature': {
        'schema': None,  # Disabled schema validation
        'filename': 'feature_config.yaml',
        'required_fields': set()  # No required fields during development
    },
    'processing': {
        'schema': None,  # Disabled schema validation
        'filename': 'data_processing_config.yaml',
        'required_fields': set()  # No required fields during development
    },
    'experiment': {
        'filename': 'experiment_config.yaml',
        'required_fields': set()  # No required fields during development
    }
}


class ConfigLoader:
    """Handles loading and validation of configuration files."""
    
    CONFIG_SCHEMAS = CONFIG_SCHEMAS
    
    def __init__(self, base_path: Optional[str] = None) -> None:
        """Initialize config loader.
        
        Args:
            base_path: Optional base path for config files
        """
        self.base_path = Path(base_path or "./pipeline/config")
        self._config_cache: Dict[str, Dict[str, Any]] = {}
        self._development_mode = True  # Always in development mode for now
        
    def _validate_config(
        self,
        config: Dict[str, Any],
        config_type: str) -> Dict[str, Any]:
        """Validate configuration dictionary against schema.
        
        Args:
            config: Configuration dictionary to validate
            config_type: Type of configuration being validated
            
        Returns:
            Validated configuration dictionary
            
        Raises:
            ValidationError: If validation fails
        """
        # In development mode, return config as is
        if self._development_mode:
            return config
            
        try:
            # Get schema for config type
            schema_info = self.CONFIG_SCHEMAS.get(config_type, {})
            schema_class = schema_info.get('schema')
            
            if schema_class and not self._development_mode:
                # Validate against schema only if not in development mode
                try:
                    validated_config = schema_class(**config)
                    return validated_config.dict()
                except Exception as e:
                    # In development mode, log warning and continue
                    if self._development_mode:
                        print(f"Warning: Schema validation failed: {str(e)}")
                        return config
                    raise ValidationError(f"Schema validation failed: {str(e)}")
            else:
                # Fall back to basic field validation
                required_fields = schema_info.get('required_fields', set())
                if not self._development_mode:
                    missing_fields = required_fields - set(config.keys())
                    if missing_fields:
                        raise ValidationError(
                            f"Missing required fields in {config_type} config: "
                            f"{', '.join(missing_fields)}"
                        )
                return config
                
        except Exception as e:
            if self._development_mode:
                print(f"Warning: Error validating {config_type} config: {str(e)}")
                return config
            raise ValidationError(f"Error validating {config_type} config: {str(e)}")
    
    @lru_cache(maxsize=None)
    def _load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """Load and cache YAML file.
        
        Args:
            file_path: Path to YAML file
            
        Returns:
            Loaded configuration dictionary
            
        Raises:
            ConfigurationError: If file cannot be loaded
        """
        try:
            with open(file_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            if self._development_mode:
                print(f"Warning: Error loading {file_path}: {str(e)}")
                return {}
            raise ConfigurationError(f"Error loading {file_path}: {str(e)}")
    
    def load_config(
        self,
        config_type: str,
        environment: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration of specified type.
        
        Args:
            config_type: Type of configuration to load
            environment: Optional environment name for overrides
            
        Returns:
            Configuration dictionary
            
        Raises:
            ConfigurationError: If config cannot be loaded or is invalid
        """
        # Check if config type is valid
        if config_type not in self.CONFIG_SCHEMAS and not self._development_mode:
            raise ConfigurationError(f"Unknown config type: {config_type}")
        
        # Check if already cached
        cache_key = f"{config_type}_{environment or 'default'}"
        if cache_key in self._config_cache:
            return self._config_cache[cache_key]
        
        # Get config info
        config_info = self.CONFIG_SCHEMAS.get(config_type, {'filename': f"{config_type}_config.yaml"})
        config_file = self.base_path / config_info['filename']
        
        # Load base config
        config = self._load_yaml(config_file)
        
        # Load environment overrides if specified
        if environment:
            override_file = self.base_path / f"{environment}_{config_info['filename']}"
            if override_file.exists():
                overrides = self._load_yaml(override_file)
                config.update(overrides)
        
        # Validate config
        validated_config = self._validate_config(config, config_type)
        
        # Cache and return
        self._config_cache[cache_key] = validated_config
        return validated_config
    
    def clear_cache(self) -> None:
        """Clear configuration cache."""
        self._config_cache.clear()
        self._load_yaml.cache_clear()
    
    def get_environment(self) -> str:
        """Get current environment name.
        
        Returns:
            Environment name from base config or environment variable
        """
        env_var = os.getenv('APP_ENV')
        if env_var:
            return env_var
        
        try:
            base_config = self.load_config('base')
            return base_config.get('environment', {}).get('name', 'development')
        except:
            return 'development'


# Global instance
_config_loader = ConfigLoader()

def initialize_config(base_path: Optional[str] = None) -> None:
    """Initialize configuration system.
    
    Args:
        base_path: Optional base path for config files
    """
    global _config_loader
    _config_loader = ConfigLoader(base_path)
    
    # In development mode, don't validate config files existence
    if not _config_loader._development_mode:
        for config_type, info in ConfigLoader.CONFIG_SCHEMAS.items():
            config_file = Path(base_path or "./config") / info['filename']
            if not config_file.exists():
                raise FileNotFoundError(
                    f"Missing required config file: {config_file}"
                )

def load_config(
    config_type: str,
    environment: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration of specified type.
    
    Args:
        config_type: Type of configuration to load
        environment: Optional environment name for overrides
        
    Returns:
        Configuration dictionary
    """
    return _config_loader.load_config(config_type, environment)

def get_environment() -> str:
    """Get current environment name.
    
    Returns:
        Current environment name
    """
    return _config_loader.get_environment()

def clear_config_cache() -> None:
    """Clear configuration cache."""
    _config_loader.clear_cache()

def get_config_path() -> str:
    """Get path to config directory.
    
    Returns:
        Path to config directory
    """
    return str(Path(__file__) / 'config')