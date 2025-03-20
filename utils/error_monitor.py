"""
Error monitoring module for the Model Context Protocol Server.

This module implements:
- Error tracking and categorization
- Error pattern analysis
- Integration with ExperimentLogger and MLflow
- Error recovery suggestions
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
import json
import threading
from dataclasses import dataclass, asdict
from collections import defaultdict
import traceback
import hashlib

# Add project root to path
try:
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
except Exception as e:
    print(f"Error setting project root path: {e}")
    sys.path.append(os.getcwd())

# Local imports
from utils.logger import ExperimentLogger

# Initialize logger
logger = ExperimentLogger(
    experiment_name="error_monitor",
    log_dir="logs/error_monitor"
)

@dataclass
class ErrorEvent:
    """Error event container."""
    error_id: str
    error_type: str
    message: str
    stack_trace: str
    component: str
    severity: str
    timestamp: datetime
    context: Dict[str, Any]
    recovery_suggestion: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert error event to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ErrorEvent':
        """Create error event from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

class ErrorMonitor:
    """Error monitoring and analysis system."""
    
    # Error severity levels
    SEVERITY_LEVELS = {
        'CRITICAL': 4,  # System-wide failures
        'ERROR': 3,     # Component failures
        'WARNING': 2,   # Potential issues
        'INFO': 1       # Informational
    }
    
    def __init__(self, state_dir: Optional[str] = None):
        """Initialize error monitor.
        
        Args:
            state_dir: Optional directory for error history
        """
        self.state_dir = Path(state_dir) if state_dir else project_root / "state/errors"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # Thread-safe state management
        self._lock = threading.RLock()
        self._error_history: Dict[str, ErrorEvent] = {}
        self._error_patterns: Dict[str, int] = defaultdict(int)
        self._active_errors: Set[str] = set()
        
        # Load error history
        self._load_history()
        
        logger.info(
            "Error monitor initialized",
            extra={"state_dir": str(self.state_dir)}
        )
        
    def _load_history(self) -> None:
        """Load error history from disk."""
        try:
            history_file = self.state_dir / "error_history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self._error_history = {
                        k: ErrorEvent.from_dict(v)
                        for k, v in data.get('errors', {}).items()
                    }
                    self._error_patterns = defaultdict(
                        int,
                        data.get('patterns', {})
                    )
                    self._active_errors = set(data.get('active', []))
                    
                logger.info(
                    "Loaded error history",
                    extra={
                        "error_count": len(self._error_history),
                        "pattern_count": len(self._error_patterns)
                    }
                )
                
        except Exception as e:
            logger.error(f"Error loading history: {str(e)}")
            self._error_history = {}
            self._error_patterns = defaultdict(int)
            self._active_errors = set()
            
    def _save_history(self) -> None:
        """Save error history to disk."""
        try:
            history_file = self.state_dir / "error_history.json"
            data = {
                'errors': {
                    k: v.to_dict() for k, v in self._error_history.items()
                },
                'patterns': dict(self._error_patterns),
                'active': list(self._active_errors)
            }
            
            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info("Error history saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving history: {str(e)}")
            
    def _generate_error_id(self, error_type: str, message: str) -> str:
        """Generate unique error ID.
        
        Args:
            error_type: Type of error
            message: Error message
            
        Returns:
            Unique error ID
        """
        content = f"{error_type}:{message}"
        return hashlib.md5(content.encode()).hexdigest()
        
    def _analyze_error_pattern(
        self,
        error_type: str,
        component: str
    ) -> Optional[str]:
        """Analyze error pattern and suggest recovery.
        
        Args:
            error_type: Type of error
            component: Component where error occurred
            
        Returns:
            Optional recovery suggestion
        """
        pattern_key = f"{component}:{error_type}"
        count = self._error_patterns[pattern_key]
        
        # Suggest recovery based on error frequency
        if count > 10:
            return (
                "High frequency error detected. Consider reviewing component "
                "configuration and dependencies."
            )
        elif count > 5:
            return (
                "Recurring error pattern. Check component logs for potential "
                "systemic issues."
            )
        elif count > 2:
            return "Monitor this error pattern for potential trending issues."
            
        return None
        
    def log_error(
        self,
        message: str,
        error_type: Optional[str] = None,
        component: str = "unknown",
        severity: str = "ERROR",
        context: Optional[Dict[str, Any]] = None,
        exc_info: Optional[Exception] = None
    ) -> str:
        """Log an error event.
        
        Args:
            message: Error message
            error_type: Optional error type
            component: Component where error occurred
            severity: Error severity level
            context: Optional error context
            exc_info: Optional exception info
            
        Returns:
            Error ID
        """
        try:
            with self._lock:
                # Get error details
                if exc_info:
                    error_type = error_type or exc_info.__class__.__name__
                    stack_trace = ''.join(
                        traceback.format_exception(
                            type(exc_info),
                            exc_info,
                            exc_info.__traceback__
                        )
                    )
                else:
                    error_type = error_type or "UnknownError"
                    stack_trace = traceback.format_stack()
                    
                # Generate error ID and create event
                error_id = self._generate_error_id(error_type, message)
                
                # Update error patterns
                pattern_key = f"{component}:{error_type}"
                self._error_patterns[pattern_key] += 1
                
                # Get recovery suggestion
                recovery_suggestion = self._analyze_error_pattern(
                    error_type,
                    component
                )
                
                # Create error event
                event = ErrorEvent(
                    error_id=error_id,
                    error_type=error_type,
                    message=message,
                    stack_trace=stack_trace,
                    component=component,
                    severity=severity,
                    timestamp=datetime.now(),
                    context=context or {},
                    recovery_suggestion=recovery_suggestion
                )
                
                # Update state
                self._error_history[error_id] = event
                if severity in ['ERROR', 'CRITICAL']:
                    self._active_errors.add(error_id)
                    
                # Save updated history
                self._save_history()
                
                # Log through ExperimentLogger
                logger.error(
                    message,
                    extra={
                        "error_id": error_id,
                        "error_type": error_type,
                        "component": component,
                        "severity": severity,
                        "context": context
                    }
                )
                
                return error_id
                
        except Exception as e:
            logger.error(f"Error in error logging: {str(e)}")
            raise
            
    def resolve_error(self, error_id: str) -> None:
        """Mark an error as resolved.
        
        Args:
            error_id: Error ID to resolve
        """
        try:
            with self._lock:
                if error_id in self._active_errors:
                    self._active_errors.remove(error_id)
                    self._save_history()
                    
                    logger.info(
                        f"Resolved error {error_id}",
                        extra={"error_id": error_id}
                    )
                    
        except Exception as e:
            logger.error(f"Error resolving error: {str(e)}")
            raise
            
    def get_active_errors(
        self,
        min_severity: str = "WARNING"
    ) -> List[Dict[str, Any]]:
        """Get all active errors above severity threshold.
        
        Args:
            min_severity: Minimum severity level
            
        Returns:
            List of active error events
        """
        try:
            with self._lock:
                min_level = self.SEVERITY_LEVELS[min_severity]
                return [
                    self._error_history[error_id].to_dict()
                    for error_id in self._active_errors
                    if self.SEVERITY_LEVELS[
                        self._error_history[error_id].severity
                    ] >= min_level
                ]
                
        except Exception as e:
            logger.error(f"Error getting active errors: {str(e)}")
            raise
            
    def get_error_patterns(
        self,
        component: Optional[str] = None,
        min_count: int = 1
    ) -> List[Dict[str, Any]]:
        """Get error patterns matching criteria.
        
        Args:
            component: Optional component filter
            min_count: Minimum occurrence count
            
        Returns:
            List of error patterns
        """
        try:
            with self._lock:
                patterns = []
                for pattern, count in self._error_patterns.items():
                    if count >= min_count:
                        comp, error_type = pattern.split(':')
                        if not component or comp == component:
                            patterns.append({
                                'component': comp,
                                'error_type': error_type,
                                'count': count
                            })
                return patterns
                
        except Exception as e:
            logger.error(f"Error getting error patterns: {str(e)}")
            raise
            
    def analyze_errors(
        self,
        time_window: timedelta = timedelta(hours=24)
    ) -> Dict[str, Any]:
        """Analyze errors within time window.
        
        Args:
            time_window: Time window for analysis
            
        Returns:
            Dictionary with error analysis
        """
        try:
            with self._lock:
                now = datetime.now()
                window_start = now - time_window
                
                # Collect errors in window
                window_errors = [
                    error for error in self._error_history.values()
                    if error.timestamp >= window_start
                ]
                
                # Analyze patterns
                severity_counts = defaultdict(int)
                component_counts = defaultdict(int)
                type_counts = defaultdict(int)
                
                for error in window_errors:
                    severity_counts[error.severity] += 1
                    component_counts[error.component] += 1
                    type_counts[error.error_type] += 1
                    
                return {
                    'total_errors': len(window_errors),
                    'active_errors': len(self._active_errors),
                    'severity_distribution': dict(severity_counts),
                    'component_distribution': dict(component_counts),
                    'type_distribution': dict(type_counts),
                    'window_start': window_start.isoformat(),
                    'window_end': now.isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error analyzing errors: {str(e)}")
            raise

def create_error_monitor(state_dir: Optional[str] = None) -> ErrorMonitor:
    """Create and initialize error monitor.
    
    Args:
        state_dir: Optional directory for error history
        
    Returns:
        Initialized ErrorMonitor instance
    """
    try:
        return ErrorMonitor(state_dir)
        
    except Exception as e:
        logger.error(f"Error creating error monitor: {str(e)}")
        raise 