import logging
import os
from typing import Optional

def logger_if_none(logger: Optional[logging.Logger]) -> logging.Logger:
    """Create a default logger if none is provided.

    This function creates a logger that logs to both file and console output.
    The log format includes:
    - Timestamp
    - Logger name (module name)
    - Log level (INFO, ERROR, etc)
    - Message

    Args:
        logger (logging.Logger, optional): Existing logger to use. Defaults to None.

    Returns:
        logging.Logger: Either the provided logger or a new configured logger
    """
    if logger is None:
        name = 'standard_logger'
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        # Clear any existing handlers
        if logger.handlers:
            logger.handlers.clear()

        # Create log directory if it doesn't exist
        os.makedirs('log', exist_ok=True)

        try:
            # File handler with UTF-8 encoding
            log_file = f'log/{name}_model.log'
            file_handler = logging.FileHandler(
                filename=log_file,
                mode='a',  # Append mode
                encoding='utf-8'
            )
            file_handler.setLevel(logging.INFO)

            # Verify file handler is working
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write('Test log file creation\n')

            if not os.path.exists(log_file):
                raise IOError(f"Failed to create log file: {log_file}")

        except Exception as e:
            print(f"Error setting up file handler: {str(e)}")
            print("Falling back to console-only logging")
            file_handler = None

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Add formatter to handlers
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if file_handler:
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info("Logger initialized with file and console output")
        else:
            logger.info("Logger initialized with console output only")

    return logger

class LoggerSetup:
    @staticmethod
    def setup_logger(name: str, log_file: str, level: int = logging.INFO, format_string: Optional[str] = None) -> logging.Logger:
        """Configure and return a logger instance"""
        if format_string is None:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

        # Create log directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)

        # Create formatter
        formatter = logging.Formatter(format_string)
        file_handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(file_handler)

        return logger 
