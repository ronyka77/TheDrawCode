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
            # Even for an already initialized instance (e.g., in a Ray worker),
            # ensure a StreamHandler exists.
            self._ensure_stream_handler()
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
        
        # Define a more readable format
        log_format = (
            '%(asctime)s | %(levelname)-8s | %(name)s | '
            '%(message)s%(extra_fields)s'
        )
        
        # File rotation settings
        max_bytes = 10 * 1024 * 1024  # 10MB
        backup_count = 5

        # Create log file path with timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.log_dir / f'{self.experiment_name}_{timestamp}.log'

        # Create handlers
        file_handler = RotatingFileHandler(
            self.log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        # console_handler = logging.StreamHandler(sys.stdout)
        
        # Configure formatter
        formatter = ReadableFormatter(log_format)
        file_handler.setFormatter(formatter)
        # console_handler.setFormatter(formatter)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        # Remove any existing handlers
        root_logger.handlers = []
        # Add our handlers
        root_logger.addHandler(file_handler)
        # root_logger.addHandler(console_handler)
        
        # Set up structured logging if enabled (optional)
        use_structured = False
        if use_structured:
            self.structured_logger = StructuredLogger.get_logger(
                self.experiment_name,
                log_format
            )
            self.logger = self.structured_logger
        else:
            self.logger = logging.getLogger(self.experiment_name)
        
        # Ensure our logger propagates messages up to the root logger.
        self.logger.propagate = True
        self.logger.info("Verified that logger propagate is True")
        
        # In case the logger loses its handlers (e.g. in a Ray worker), add a stream handler.
        self._ensure_stream_handler()

    @property
    def handlers(self):
        """Property to access logger's handlers."""
        return self.logger.handlers
    
    def _ensure_stream_handler(self) -> None:
        """Ensure that the logger has at least one StreamHandler attached."""
        if not self.logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            log_format = (
                '%(asctime)s | %(levelname)-8s | %(name)s | '
                '%(message)s%(extra_fields)s'
            )
            formatter = ReadableFormatter(log_format)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        self.logger.propagate = True

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
        """Log an info message."""
        self._log('INFO', msg, {**(extra or {}), 'error_code': error_code})
    
    def warning(
        self,
        msg: str,
        error_code: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None) -> None:
        """Log a warning message."""
        self._log('WARNING', msg, {**(extra or {}), 'error_code': error_code})
    
    def error(
        self,
        msg: str,
        error_code: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None) -> None:
        """Log an error message."""
        self._log('ERROR', msg, {**(extra or {}), 'error_code': error_code})
    
    def debug(
        self,
        msg: str,
        extra: Optional[Dict[str, Any]] = None) -> None:
        """Log a debug message."""
        self._log('DEBUG', msg, extra)
