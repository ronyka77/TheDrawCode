"""
Logging utilities for MCP server.
Provides structured logging with JSON formatting and rotation.
"""

import os
import logging
import logging.handlers
from datetime import datetime
from typing import Any, Dict, Optional
from pythonjsonlogger import jsonlogger

class ExperimentLogger:
    """
    Structured logger for MCP server with JSON formatting and rotation.
    """
    
    def __init__(
        self,
        experiment_name: str = "mcp_server",
        log_dir: str = "logs",
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ):
        """
        Initialize logger.
        
        Args:
            experiment_name: Name for the logger
            log_dir: Directory for log files
            max_bytes: Maximum size per log file
            backup_count: Number of backup files to keep
        """
        self.experiment_name = experiment_name
        
        # Create logger
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.INFO)
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Create rotating file handler
        log_file = os.path.join(log_dir, f"{experiment_name}.log")
        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        
        # Create JSON formatter
        formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s',
            timestamp=True
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(handler)
        
        # Add console handler for development
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        self.logger.addHandler(console)
        
        self.info("Logger initialized", extra={"experiment_name": experiment_name})
    
    def _log(
        self,
        level: int,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exc_info: Optional[Exception] = None
    ):
        """
        Internal logging method.
        
        Args:
            level: Logging level
            message: Log message
            extra: Additional fields to log
            exc_info: Exception information
        """
        extra = extra or {}
        extra.update({
            "experiment": self.experiment_name,
            "timestamp": datetime.now().isoformat()
        })
        self.logger.log(level, message, extra=extra, exc_info=exc_info)
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log debug message."""
        self._log(logging.DEBUG, message, extra)
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log info message."""
        self._log(logging.INFO, message, extra)
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log warning message."""
        self._log(logging.WARNING, message, extra)
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None, exc_info: Optional[Exception] = None):
        """Log error message."""
        self._log(logging.ERROR, message, extra, exc_info)
    
    def get_recent_logs(self, limit: int = 100) -> list:
        """
        Get recent log entries.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of recent log entries
        """
        try:
            log_file = os.path.join("logs", f"{self.experiment_name}.log")
            if not os.path.exists(log_file):
                return []
                
            with open(log_file, 'r') as f:
                # Read last N lines
                lines = f.readlines()[-limit:]
                return [line.strip() for line in lines if line.strip()]
                
        except Exception as e:
            self.error(f"Error getting recent logs: {str(e)}", exc_info=e)
            return []

# Create default logger instance
default_logger = ExperimentLogger() 