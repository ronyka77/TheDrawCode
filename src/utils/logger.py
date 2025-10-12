"""Logging utilities for experiment tracking and monitoring with Unicode support."""

import datetime
import json
import logging
import os
import sys
import unicodedata
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

import structlog

project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).parent
sys.path.append(str(project_root))


def sanitize_unicode_for_console(text: str) -> str:
    """
    Sanitize Unicode text for console output, replacing problematic characters.
    
    Args:
        text: Input text that may contain Unicode characters
        
    Returns:
        Sanitized text safe for console output
    """
    # Dictionary of common emoji replacements for console-safe alternatives
    emoji_replacements = {
        '📊': '[CHART]',
        '🎯': '[TARGET]', 
        '🚀': '[ROCKET]',
        '✓': '[OK]',
        '✗': '[FAIL]',
        '⚠': '[WARN]',
        '🎉': '[SUCCESS]',
        '📈': '[GRAPH]',
        '🔍': '[SEARCH]',
        '⭐': '[STAR]',
        '🔧': '[TOOL]',
        '📦': '[PACKAGE]',
        '⚡': '[FAST]',
        '🎲': '[DICE]',
        '🧠': '[BRAIN]',
        '🏆': '[TROPHY]',
        '🔥': '[FIRE]',
        '💡': '[IDEA]',
        '📝': '[NOTE]',
        '🎪': '[CIRCUS]',
        '🎨': '[ART]',
        '🌟': '[SPARKLE]',
        '🎵': '[MUSIC]',
        '🎭': '[THEATER]',
        '🎬': '[MOVIE]',
        '🎮': '[GAME]',
        '🎸': '[GUITAR]',
        '🎤': '[MIC]',
        '🎧': '[HEADPHONE]',
        '🎺': '[TRUMPET]',
        '🎻': '[VIOLIN]',
        '🥁': '[DRUM]',
        '🎹': '[PIANO]',
    }
    
    # Replace known emojis first
    sanitized = text
    for emoji, replacement in emoji_replacements.items():
        sanitized = sanitized.replace(emoji, replacement)
    
    # Handle any remaining problematic Unicode characters
    try:
        # Try to encode with the system's default encoding
        sanitized.encode(sys.stdout.encoding or 'utf-8', errors='strict')
        return sanitized
    except (UnicodeEncodeError, LookupError):
        # If that fails, replace problematic characters
        try:
            # Try UTF-8 first
            sanitized.encode('utf-8', errors='strict')
            return sanitized
        except UnicodeEncodeError:
            # Last resort: replace all non-ASCII characters
            return ''.join(char if ord(char) < 128 else f'[U+{ord(char):04X}]' for char in sanitized)


class UnicodeAwareFormatter(logging.Formatter):
    """
    Custom formatter that handles Unicode characters safely.
    """
    def format(self, record):
        # Sanitize the message for console output
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            record.msg = sanitize_unicode_for_console(record.msg)
        
        # Format the basic message
        record.extra_fields = ""
        if hasattr(record, "extra"):
            # Format extra fields in a readable way
            extra_str = []
            for key, value in record.extra.items():
                if isinstance(value, dict):
                    # Format nested dictionaries more compactly
                    value = json.dumps(value, default=str)
                elif isinstance(value, (list, tuple)):
                    value = str(value)
                # Sanitize extra field values too
                key = sanitize_unicode_for_console(str(key))
                value = sanitize_unicode_for_console(str(value))
                extra_str.append(f"{key}={value}")
            if extra_str:
                record.extra_fields = f" | {' | '.join(extra_str)}"

        return super().format(record)


class UnicodeAwareStreamHandler(logging.StreamHandler):
    """
    Stream handler that properly handles Unicode encoding issues.
    """
    def __init__(self, stream=None):
        super().__init__(stream)
        
        if stream is None:
            stream = sys.stdout
            
        if hasattr(stream, 'reconfigure'):
            try:
                stream.reconfigure(encoding='utf-8', errors='replace')
            except (AttributeError, OSError):
                pass
    
    def emit(self, record):
        """
        Emit a record with proper Unicode handling.
        """
        try:
            super().emit(record)
        except UnicodeEncodeError:
            # If Unicode encoding fails, sanitize and try again
            original_msg = record.getMessage()
            record.msg = sanitize_unicode_for_console(original_msg)
            record.args = None  # Clear args to prevent re-formatting
            try:
                super().emit(record)
            except Exception:
                # Last resort: print a simple error message
                try:
                    self.stream.write(f"[UNICODE ERROR] Log message could not be displayed\n")
                    self.stream.flush()
                except Exception:
                    pass


class ReadableFormatter(UnicodeAwareFormatter):
    """Legacy formatter name for backward compatibility."""
    pass


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

    def __new__(cls, experiment_name: str = "default_experiment", log_dir: str = "logs"):
        # If an instance already exists, return it regardless of new arguments.
        if cls._instance is not None:
            return cls._instance
        instance = super().__new__(cls)
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
        self.logger.info(
            f"Initialized ExperimentLogger for {self.experiment_name} at {self.log_dir}"
        )

    def _configure_logging(self) -> None:
        """Configure logging for the logger with Unicode support."""
        # Get logging settings from config
        log_level = "INFO"

        # Define a more readable format
        log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s%(extra_fields)s"

        # File rotation settings
        max_bytes = 10 * 1024 * 1024  # 10MB
        backup_count = 5

        # Create log file path with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{self.experiment_name}_{timestamp}.log"

        # Create handlers with Unicode support
        file_handler = RotatingFileHandler(
            self.log_file, 
            maxBytes=max_bytes, 
            backupCount=backup_count,
            encoding='utf-8'  # Ensure UTF-8 encoding for file
        )
        
        # Use Unicode-aware stream handler for console
        console_handler = UnicodeAwareStreamHandler(sys.stdout)

        # Configure formatters with Unicode support
        file_formatter = UnicodeAwareFormatter(log_format)
        console_formatter = UnicodeAwareFormatter(log_format)
        
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)

        # Configure the instance logger directly (not root logger)
        self.logger.setLevel(log_level)
        # Clear any existing handlers to prevent duplication
        self.logger.handlers.clear()
        # Add handlers to the instance logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # CRITICAL: Disable propagation to prevent double logging
        self.logger.propagate = False

        # Set up structured logging if enabled (optional)
        use_structured = False
        if use_structured:
            self.structured_logger = StructuredLogger.get_logger(self.experiment_name, log_format)
            self.logger = self.structured_logger
        else:
            self.logger = logging.getLogger(self.experiment_name)

        self.logger.info("Unicode-aware logger configured successfully")

        # Ensure stream handler exists after configuration
        self._ensure_stream_handler()

    @property
    def handlers(self):
        """Property to access logger's handlers."""
        return self.logger.handlers

    def _ensure_stream_handler(self) -> None:
        """Ensure that the logger has at least one Unicode-aware StreamHandler attached."""
        # Check if we already have a StreamHandler to avoid duplication
        has_stream_handler = any(
            isinstance(handler, (logging.StreamHandler, UnicodeAwareStreamHandler)) 
            for handler in self.logger.handlers
        )
        
        if not has_stream_handler:
            console_handler = UnicodeAwareStreamHandler(sys.stdout)
            log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s%(extra_fields)s"
            formatter = UnicodeAwareFormatter(log_format)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Maintain propagation setting to prevent double logging
        self.logger.propagate = False

    def _log(self, level: str, msg: str, extra: Optional[dict[str, Any]] = None) -> None:
        """Internal logging method with Unicode-safe structuring.

        Args:
            level: Log level
            msg: Log message (will be sanitized for Unicode safety)
            extra: Optional extra fields for structured logging
        """
        # Sanitize the message for Unicode safety
        safe_msg = sanitize_unicode_for_console(msg)
        
        if self.structured_logger:
            log_method = getattr(self.structured_logger, level.lower())
            log_method(safe_msg, **(extra or {}))
        else:
            log_method = getattr(self.logger, level.lower())
            if extra:
                # Sanitize extra fields too
                safe_extra = {}
                for key, value in extra.items():
                    safe_key = sanitize_unicode_for_console(str(key))
                    safe_value = sanitize_unicode_for_console(str(value))
                    safe_extra[safe_key] = safe_value
                log_method(safe_msg, extra=safe_extra)
            else:
                log_method(safe_msg)

    def info(
        self, msg: str, error_code: Optional[str] = None, extra: Optional[dict[str, Any]] = None
    ) -> None:
        """Log an info message with Unicode support."""
        extra_dict = {**(extra or {})}
        if error_code:
            extra_dict["error_code"] = error_code
        self._log("INFO", msg, extra_dict if extra_dict else None)

    def warning(
        self, msg: str, error_code: Optional[str] = None, extra: Optional[dict[str, Any]] = None
    ) -> None:
        """Log a warning message with Unicode support."""
        extra_dict = {**(extra or {})}
        if error_code:
            extra_dict["error_code"] = error_code
        self._log("WARNING", msg, extra_dict if extra_dict else None)

    def error(
        self, msg: str, error_code: Optional[str] = None, extra: Optional[dict[str, Any]] = None
    ) -> None:
        """Log an error message with Unicode support."""
        extra_dict = {**(extra or {})}
        if error_code:
            extra_dict["error_code"] = error_code
        self._log("ERROR", msg, extra_dict if extra_dict else None)

    def debug(self, msg: str, extra: Optional[dict[str, Any]] = None) -> None:
        """Log a debug message with Unicode support."""
        self._log("DEBUG", msg, extra)
    
    def log_unicode_safe(self, level: str, msg: str, **kwargs) -> None:
        """
        Explicitly Unicode-safe logging method.
        
        Args:
            level: Log level (INFO, WARNING, ERROR, DEBUG)
            msg: Message to log (will be sanitized)
            **kwargs: Additional keyword arguments
        """
        log_method = getattr(self, level.lower(), self.info)
        log_method(msg, extra=kwargs if kwargs else None)


# --- Example Usage --- #
if __name__ == "__main__":
    # Example of using the logger
    logger = ExperimentLogger(experiment_name="test_experiment", log_to_console=True)

    logger.start_run("example_run_1")
    logger.info("This is an informational message.")
    logger.log_params({"learning_rate": 0.01, "epochs": 100})
    logger.warning("This is a warning message.")

    try:
        result = 1 / 0
    except ZeroDivisionError:
        logger.error("Division by zero occurred!", exc_info=True)
        logger.exception("Caught an exception during calculation.")

    logger.log_metrics({"accuracy": 0.95, "loss": 0.1})
    logger.log_artifact("/path/to/model.pkl")
    logger.debug("This is a debug message.")
    logger.end_run()

    print(f"Log file created at: {logger.log_file}")
