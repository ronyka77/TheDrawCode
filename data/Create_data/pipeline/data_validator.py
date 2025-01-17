"""Data validation utilities for ensuring data quality and consistency."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import structlog
from pydantic import BaseModel, Field, validator
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logger import ExperimentLogger
from pipeline.config_loader import load_config, get_environment


class ValidationConfig(BaseModel):
    """Schema for validation configuration."""
    max_missing_ratio: float = Field(..., description="Maximum allowed ratio of missing values")
    min_unique_values: int = Field(..., description="Minimum required unique values for categorical columns")
    outlier_threshold: float = Field(3.0, description="Z-score threshold for outlier detection")
    correlation_threshold: float = Field(0.95, description="Threshold for high correlation warning")
    
    @validator('max_missing_ratio')
    def validate_missing_ratio(cls, v):
        """Validate missing ratio is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("max_missing_ratio must be between 0 and 1")
        return v


class ColumnTypeConfig(BaseModel):
    """Schema for column type configuration."""
    numerical: List[str] = Field(..., description="List of numerical columns")
    categorical: List[str] = Field(..., description="List of categorical columns")
    datetime: List[str] = Field(..., description="List of datetime columns")
    target: str = Field(..., description="Target column name")


class DataValidator:
    """Data validator class for ensuring data quality and consistency."""
    
    def __init__(
        self,
        logger: Optional[ExperimentLogger] = None,
        config_type: str = 'processing'
    ) -> None:
        """Initialize data validator.
        
        Args:
            logger: Optional logger instance
            config_type: Type of config to load
            
        Raises:
            ValueError: If configuration validation fails
        """
        try:
            # Initialize logger first
            self.logger = logger or ExperimentLogger()
            
            # Load configurations with environment
            environment = get_environment()
            self.logger.info(
                "Loading validator configurations",
                environment=environment,
                config_type=config_type
            )
            
            self.base_config = load_config('base', environment)
            self.data_config = load_config(config_type, environment)
            
            # Validate configurations using Pydantic
            self.validation_config = ValidationConfig(**self.data_config['validation'])
            self.column_config = ColumnTypeConfig(**self.data_config['column_types'])
            
            # Initialize validation report with structured format
            self.validation_report: Dict[str, Any] = {
                'summary': {
                    'total_checks': 0,
                    'passed_checks': 0,
                    'failed_checks': 0,
                    'warnings': 0
                },
                'checks': [],
                'warnings': [],
                'metadata': {
                    'config_type': config_type,
                    'environment': environment,
                    'timestamp': pd.Timestamp.now().isoformat()
                }
            }
            
            self.logger.info(
                "DataValidator initialized successfully",
                validation_config=self.validation_config.dict(),
                column_types=self.column_config.dict()
            )
            
        except Exception as e:
            error_msg = f"Error initializing DataValidator: {str(e)}"
            if hasattr(self, 'logger'):
                self.logger.error(
                    error_msg,
                    error_type=type(e).__name__,
                    config_type=config_type
                )
            raise ValueError(error_msg) from e
    
    def validate_dataset(
        self,
        data: pd.DataFrame,
        is_training: bool = True
    ) -> Dict[str, Any]:
        """Validate dataset against defined rules and thresholds.
        
        Args:
            data: DataFrame to validate
            is_training: Whether this is training data
            
        Returns:
            Validation report dictionary
            
        Raises:
            ValueError: If validation fails critically
        """
        try:
            self.logger.info(
                "Starting dataset validation",
                dataset_shape=data.shape,
                is_training=is_training
            )
            
            # Reset validation report
            self._reset_validation_report()
            
            # Perform validation checks
            self._validate_column_presence(data)
            self._validate_data_types(data)
            self._validate_missing_values(data)
            self._validate_unique_values(data)
            
            if is_training:
                self._validate_target_distribution(data)
                self._validate_correlations(data)
                self._detect_outliers(data)
            
            # Update summary
            self.validation_report['summary']['total_checks'] = len(self.validation_report['checks'])
            self.validation_report['summary']['passed_checks'] = sum(
                1 for check in self.validation_report['checks'] if check['status'] == 'passed'
            )
            self.validation_report['summary']['failed_checks'] = sum(
                1 for check in self.validation_report['checks'] if check['status'] == 'failed'
            )
            self.validation_report['summary']['warnings'] = len(self.validation_report['warnings'])
            
            # Log validation results
            self._log_validation_results()
            
            return self.validation_report
            
        except Exception as e:
            error_msg = f"Error during dataset validation: {str(e)}"
            self.logger.error(
                error_msg,
                error_type=type(e).__name__,
                dataset_shape=data.shape if isinstance(data, pd.DataFrame) else None
            )
            raise ValueError(error_msg) from e
    
    def _reset_validation_report(self) -> None:
        """Reset validation report to initial state."""
        self.validation_report['checks'] = []
        self.validation_report['warnings'] = []
        self.validation_report['summary'] = {
            'total_checks': 0,
            'passed_checks': 0,
            'failed_checks': 0,
            'warnings': 0
        }
    
    def _add_validation_check(
        self,
        check_name: str,
        status: str,
        details: Dict[str, Any]
    ) -> None:
        """Add validation check to report.
        
        Args:
            check_name: Name of the validation check
            status: Status of the check ('passed' or 'failed')
            details: Additional check details
        """
        self.validation_report['checks'].append({
            'name': check_name,
            'status': status,
            'timestamp': pd.Timestamp.now().isoformat(),
            'details': details
        })
    
    def _add_warning(
        self,
        warning_type: str,
        message: str,
        details: Dict[str, Any]
    ) -> None:
        """Add warning to report.
        
        Args:
            warning_type: Type of warning
            message: Warning message
            details: Additional warning details
        """
        self.validation_report['warnings'].append({
            'type': warning_type,
            'message': message,
            'timestamp': pd.Timestamp.now().isoformat(),
            'details': details
        })
    
    def _log_validation_results(self) -> None:
        """Log validation results using structured logging."""
        summary = self.validation_report['summary']
        self.logger.info(
            "Validation completed",
            total_checks=summary['total_checks'],
            passed_checks=summary['passed_checks'],
            failed_checks=summary['failed_checks'],
            warnings=summary['warnings']
        )
        
        # Log failed checks
        failed_checks = [
            check for check in self.validation_report['checks']
            if check['status'] == 'failed'
        ]
        if failed_checks:
            self.logger.warning(
                "Failed validation checks",
                failed_checks=failed_checks
            )
        
        # Log warnings
        if self.validation_report['warnings']:
            self.logger.warning(
                "Validation warnings",
                warnings=self.validation_report['warnings']
            )