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
    def get_logger(
        name: str,
        log_format: Optional[str] = None
    ) -> structlog.BoundLogger:
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
    """Logger class for experiment tracking and monitoring."""
    
    def __init__(
        self,
        experiment_name: Optional[str] = None,
        log_dir: Optional[str] = None
    ) -> None:
        """Initialize logger.
        
        Args:
            experiment_name: Optional experiment name
            log_dir: Optional log directory
        """
        try:
            # Load base config for logging settings
            base_config = load_config('base')
            self.config = base_config.get('logging', {})
            
            # Set log directory
            data_paths = base_config.get('data_paths', {})
            self.log_dir = Path(log_dir or data_paths.get('logs_path', 'logs'))
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            # Set experiment name from config or parameter
            global_config = base_config.get('global_config', {})
            self.experiment_name = experiment_name or global_config.get('experiment_name', 'default_experiment')
            
            self.structured_logger = None
            
            # Configure logging
            self._configure_logging()
            
        except Exception as e:
            print(f"Error initializing logger: {str(e)}")
            raise
    
    def _configure_logging(self) -> None:
        """Configure logging for the logger."""
        # Get logging settings from config
        log_level = getattr(logging, self.config.get('level', 'INFO'))
        
        # Get structured logging config
        structured_config = self.config.get('structured', {})
        use_structured = structured_config.get('enabled', False)
        
        # Define a more readable format
        log_format = (
            '%(asctime)s | %(levelname)-8s | %(name)s | '
            '%(message)s'
            '%(extra_fields)s'
        )
        
        # Get file rotation settings
        max_bytes = self.config.get('max_bytes', 10 * 1024 * 1024)  # 10MB default
        backup_count = self.config.get('backup_count', 5)
        
        # Set up file logging with rotation
        timestamp = datetime.datetime.now().strftime(
            self.config.get('timestamp_format', '%Y%m%d_%H%M%S')
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
        
        # Initialize MLflow if enabled
        if self.config.get('use_mlflow', True):
            tracking_config = self.config.get('tracking', {})
            mlflow_uri = tracking_config.get('uri')
            if mlflow_uri:
                mlflow.set_tracking_uri(mlflow_uri)
                mlflow.set_experiment(self.experiment_name)
                self.logger.info(f"Initialized MLflow tracking at {mlflow_uri}")
    
    def _log(
        self,
        level: str,
        msg: str,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
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
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log an info message.
        
        Args:
            msg: Message to log
            extra: Optional extra fields for structured logging
        """
        self._log('INFO', msg, extra)
    
    def warning(
        self,
        msg: str,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a warning message.
        
        Args:
            msg: Message to log
            extra: Optional extra fields for structured logging
        """
        self._log('WARNING', msg, extra)
    
    def error(
        self,
        msg: str,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log an error message.
        
        Args:
            msg: Message to log
            extra: Optional extra fields for structured logging
        """
        self._log('ERROR', msg, extra)
    
    def debug(
        self,
        msg: str,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a debug message.
        
        Args:
            msg: Message to log
            extra: Optional extra fields for structured logging
        """
        self._log('DEBUG', msg, extra)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        timestamp: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log metrics with optional structuring.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
            timestamp: Optional timestamp
            extra: Optional extra fields for structured logging
        """
        mlflow.log_metrics(metrics, step=step)
        
        # Add metrics to structured logging if enabled
        if self.structured_logger:
            log_data = {
                'metrics': metrics,
                'step': step,
                'timestamp': timestamp,
                **(extra or {})
            }
            self.debug("Logged metrics", extra=log_data)
        else:
            self.debug(f"Logged metrics at step {step or 'None'}: {metrics}")
        
    def start_run(
        self,
        run_name: Optional[str] = None,
        nested: bool = False,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run
            nested: Whether this is a nested run
            tags: Optional tags for the run
        """
        mlflow.start_run(run_name=run_name, nested=nested, tags=tags)
        self.logger.info(f"Started MLflow run: {run_name or 'unnamed'}")
        
    def end_run(self, status: str = 'FINISHED') -> None:
        """End the current MLflow run.
        
        Args:
            status: Run status ('FINISHED', 'FAILED', etc.)
        """
        mlflow.end_run(status=status)
        self.logger.info(f"Ended MLflow run with status: {status}")
        
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow.
        
        Args:
            params: Dictionary of parameters to log
        """
        mlflow.log_params(params)
        self.logger.debug(f"Logged parameters: {params}")
        
    def log_artifact(
        self,
        local_path: str,
        artifact_path: Optional[str] = None
    ) -> None:
        """Log an artifact to MLflow.
        
        Args:
            local_path: Path to the artifact file
            artifact_path: Optional path for artifact in MLflow
        """
        mlflow.log_artifact(local_path, artifact_path)
        self.logger.info(
            f"Logged artifact from {local_path} "
            f"to {artifact_path or 'root'}"
        )
        
    def log_model(
        self,
        model: Any,
        artifact_path: str,
        conda_env: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a model to MLflow.
        
        Args:
            model: Model object to log
            artifact_path: Path for the model artifact
            conda_env: Optional conda environment
        """
        mlflow.sklearn.log_model(
            model,
            artifact_path,
            conda_env=conda_env
        )
        self.logger.info(f"Logged model to {artifact_path}")
 