"""
Event models for MCP server.
Defines Pydantic models for event handling and validation.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field
from fastmcp import Context

class FileEvent(BaseModel):
    """Base model for file system events."""
    type: str = Field(..., description="Event type")
    timestamp: datetime = Field(default_factory=datetime.now)
    path: str = Field(..., description="File path")
    is_directory: bool = Field(default=False)

class FileCreatedEvent(FileEvent):
    """Event for file creation."""
    type: str = "file_created"

class FileModifiedEvent(FileEvent):
    """Event for file modification."""
    type: str = "file_modified"

class FileDeletedEvent(FileEvent):
    """Event for file deletion."""
    type: str = "file_deleted"

class FileMovedEvent(FileEvent):
    """Event for file move/rename."""
    type: str = "file_moved"
    dest_path: str = Field(..., description="Destination path")

class ContextEvent(BaseModel):
    """Event for context updates."""
    type: str = "context_updated"
    timestamp: datetime = Field(default_factory=datetime.now)
    file_path: str = Field(..., description="File path")
    changes: Dict[str, Any] = Field(default_factory=dict)
    
    def to_fastmcp_context(self, request_id: str) -> Context:
        """Convert to FastMCP Context object."""
        return Context(
            request_id=request_id,
            data={
                "type": self.type,
                "timestamp": self.timestamp.isoformat(),
                "file_path": self.file_path,
                "changes": self.changes
            }
        )

class MLflowEvent(BaseModel):
    """Event for MLflow updates."""
    type: str = "mlflow_update"
    timestamp: datetime = Field(default_factory=datetime.now)
    experiment_id: str = Field(..., description="MLflow experiment ID")
    run_id: Optional[str] = Field(None, description="MLflow run ID")
    metrics: Dict[str, float] = Field(default_factory=dict)
    params: Dict[str, str] = Field(default_factory=dict)
    tags: Dict[str, str] = Field(default_factory=dict)

class ErrorEvent(BaseModel):
    """Event for error reporting."""
    type: str = "error"
    timestamp: datetime = Field(default_factory=datetime.now)
    error_type: str = Field(..., description="Type of error")
    message: str = Field(..., description="Error message")
    details: Dict[str, Any] = Field(default_factory=dict)

class HealthEvent(BaseModel):
    """Event for health status updates."""
    type: str = "health_update"
    timestamp: datetime = Field(default_factory=datetime.now)
    component: str = Field(..., description="Component name")
    status: str = Field(..., description="Health status")
    metrics: Dict[str, Any] = Field(default_factory=dict)

class StatsEvent(BaseModel):
    """Event for statistics updates."""
    type: str = "stats_update"
    timestamp: datetime = Field(default_factory=datetime.now)
    component: str = Field(..., description="Component name")
    stats: Dict[str, Any] = Field(..., description="Statistics") 