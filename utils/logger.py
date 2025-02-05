"""Logging utilities for experiment tracking and monitoring."""

import os
import sys
from pathlib import Path
project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).parent
sys.path.append(str(project_root))

import logging
import datetime
import json
from typing import Optional, Dict, Any, Union
import mlflow
from pathlib import Path
from logging.handlers import RotatingFileHandler
import structlog
from pythonjsonlogger import jsonlogger

from data.Create_data.pipeline.config_loader import load_config, get_environment

class ReadableFormatter(logging.Formatter):
    def format(self, record):
        # Format the basic message
        record.extra_fields = ''
        if hasattr(record, 'extra'):
            # Format extra fields in a readable way
            extra_str = []
            for key, value in record.extra.items():
                if isinstance(value, dict):
                    # Format nested dictionaries more compactly
                    value = json.dumps(value, default=str)
                elif isinstance(value, (list, tuple)):
                    value = str(value)
                extra_str.append(f"{key}={value}")
            if extra_str:
                record.extra_fields = f" | {' | '.join(extra_str)}"
        
        return super().format(record)
    
class StructuredLogger:
    """Handles structured logging configuration."""
    
    @staticmethod
    def get_logger(name: str, log_format: Optional[str] = None) -> structlog.BoundLogger:
        """Get structured logger instance.
        
        Args:
            name: Logger name
            log_format: Optional log format string
            
        Returns:
            Structured logger instance
        """
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.stdlib.render_to_log_kwargs,
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        return structlog.get_logger(name)

class ExperimentLogger:
    _instance = None
    structured_logger = None
    
    def __new__(cls, experiment_name: str = 'default_experiment', log_dir: str = 'logs'):
        # If an instance already exists, return it regardless of new arguments.
        if cls._instance is not None:

            return cls._instance
        instance = super(ExperimentLogger, cls).__new__(cls)
        cls._instance = instance
        return instance

    def __init__(self, experiment_name: str = "default_experiment", log_dir: str = "logs"):
        # Do not reinitialize if already initialized.
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir) / self.experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.experiment_name)
        self._configure_logging()
        self._initialized = True
        self.logger.info(f"Initialized ExperimentLogger for {self.experiment_name} at {self.log_dir}")

    def _configure_logging(self) -> None:
        """Configure logging for the logger."""
        # Get logging settings from config
        log_level = 'INFO'
        
        # Get structured logging config
        structured_config =  {}
        use_structured =  False
        
        # Define a more readable format
        log_format = (
            '%(asctime)s | %(levelname)-8s | %(name)s | '
            '%(message)s'
            '%(extra_fields)s'
        )
        
        # Get file rotation settings
        max_bytes = 10 * 1024 * 1024
        backup_count = 5
        

        # Set up file logging with rotation
        timestamp = datetime.datetime.now().strftime(
            '%Y%m%d_%H%M%S'
        )
        self.log_file = self.log_dir / f'{self.experiment_name}_{timestamp}.log'
        

        # Create handlers
        file_handler = RotatingFileHandler(
            self.log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        console_handler = logging.StreamHandler()
        
        # Configure formatter
        formatter = ReadableFormatter(log_format)
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Remove any existing handlers
        root_logger.handlers = []
        
        # Add our handlers
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Set up structured logging if enabled
        if use_structured:
            self.structured_logger = StructuredLogger.get_logger(
                self.experiment_name,
                log_format
            )
            self.logger = self.structured_logger
        else:
            self.logger = logging.getLogger(self.experiment_name)
        self.logger.propagate = True
        self.logger.info("Verified that logger propagate is True")
    
    def _log(
        self,
        level: str,
        msg: str,
        extra: Optional[Dict[str, Any]] = None) -> None:
        """Internal logging method with structuring.
        
        Args:
            level: Log level
            msg: Log message
            extra: Optional extra fields for structured logging
        """

        
        if self.structured_logger:
            log_method = getattr(self.structured_logger, level.lower())
            log_method(msg, **(extra or {}))
        else:
            log_method = getattr(self.logger, level.lower())
            if extra:
                log_method(msg, extra=extra)
            else:
                log_method(msg)
    
    def info(
        self,
        msg: str,
        error_code: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None) -> None:
        """Log an info message.
        
        Args:
            msg: Message to log
            extra: Optional extra fields for structured logging
        """
        self._log('INFO', msg, {**(extra or {}), 'error_code': error_code})
    
    def warning(
        self,
        msg: str,
        error_code: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None) -> None:
        """Log a warning message.
        
        Args:
            msg: Message to log
            error_code: Optional error code
            extra: Optional extra fields for structured logging
        """
        self._log('WARNING', msg, {**(extra or {}), 'error_code': error_code})
    
    def error(
        self,
        msg: str,
        error_code: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None) -> None:
        """Log an error message.
        

        Args:
            msg: Message to log
            error_code: Optional error code
            extra: Optional extra fields for structured logging
        """
        self._log('ERROR', msg, {**(extra or {}), 'error_code': error_code})
    
    def debug(
        self,
        msg: str,
        extra: Optional[Dict[str, Any]] = None) -> None:
        """Log a debug message.
        

        Args:
            msg: Message to log
            extra: Optional extra fields for structured logging
        """
        self._log('DEBUG', msg, extra)
    
    # def log_metrics(
    #     self,
    #     metrics: Dict[str, float],
    #     step: Optional[int] = None,
    #     timestamp: Optional[int] = None,
    #     extra: Optional[Dict[str, Any]] = None) -> None:
    #     """Log metrics with optional structuring.
        

    #     Args:
    #         metrics: Dictionary of metrics to log
    #         step: Optional step number
    #         timestamp: Optional timestamp
    #         extra: Optional extra fields for structured logging
    #     """
    #     mlflow.log_metrics(metrics, step=step)
        
    #     # Add metrics to structured logging if enabled
    #     if self.structured_logger:
    #         log_data = {
    #             'metrics': metrics,
    #             'step': step,
    #             'timestamp': timestamp,
    #             **(extra or {})
    #         }
    #         self.debug("Logged metrics", extra=log_data)
    #     else:
    #         self.debug(f"Logged metrics at step {step or 'None'}: {metrics}")
        
    # def start_run(
    #     self,
    #     run_name: Optional[str] = None,
    #     nested: bool = False,
    #     tags: Optional[Dict[str, str]] = None) -> None:
    #     """Start a new MLflow run.
        

    #     Args:
    #         run_name: Optional name for the run
    #         nested: Whether this is a nested run
    #         tags: Optional tags for the run
    #     """
    #     mlflow.start_run(run_name=run_name, nested=nested, tags=tags)
    #     self.logger.info(f"Started MLflow run: {run_name or 'unnamed'}")
        
    # def end_run(self, status: str = 'FINISHED') -> None:
    #     """End the current MLflow run.
        
    #     Args:
    #         status: Run status ('FINISHED', 'FAILED', etc.)
    #     """
    #     mlflow.end_run(status=status)
    #     self.logger.info(f"Ended MLflow run with status: {status}")
        
    # def log_params(self, params: Dict[str, Any]) -> None:
    #     """Log parameters to MLflow.
        
    #     Args:
    #         params: Dictionary of parameters to log
    #     """
    #     mlflow.log_params(params)
    #     self.logger.debug(f"Logged parameters: {params}")
        
    # def log_artifact(
    #     self,
    #     local_path: str,
    #     artifact_path: Optional[str] = None) -> None:
    #     """Log an artifact to MLflow.
        

    #     Args:
    #         local_path: Path to the artifact file
    #         artifact_path: Optional path for artifact in MLflow
    #     """
    #     mlflow.log_artifact(local_path, artifact_path)
    #     self.logger.info(
    #         f"Logged artifact from {local_path} "
    #         f"to {artifact_path or 'root'}"
    #     )
        
    # def log_model(
    #     self,
    #     model: Any,
    #     artifact_path: str,
    #     conda_env: Optional[Dict[str, Any]] = None) -> None:
    #     """Log a model to MLflow.
        

    #     Args:
    #         model: Model object to log
    #         artifact_path: Path for the model artifact
    #         conda_env: Optional conda environment
    #     """
    #     mlflow.sklearn.log_model(
    #         model,
    #         artifact_path,
    #         conda_env=conda_env
    #     )
    #     self.logger.info(f"Logged model to {artifact_path}")
 