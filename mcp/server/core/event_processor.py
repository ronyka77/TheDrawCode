"""
Event processor for handling async events in the MCP server.
Manages event queues and tool requests.
"""

import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

from ..utils.logger import default_logger as logger
from .context_manager import ContextManager
from .project_manager import ProjectManager
from .mlflow_manager import MLflowManager

class EventProcessor:
    """
    Async event processor for MCP server.
    Handles event queues and tool requests.
    """
    
    def __init__(self):
        """Initialize the event processor."""
        self.event_queue = asyncio.Queue()
        self.is_running = False
        self.start_time = None
        
        # Initialize managers
        self.context_manager = ContextManager()
        self.project_manager = ProjectManager()
        self.mlflow_manager = MLflowManager()
        
        # Statistics
        self.processed_count = 0
        self.error_count = 0
        
        logger.info("Event processor initialized")

    async def start(self):
        """Start the event processor."""
        if self.is_running:
            return
        
        self.is_running = True
        self.start_time = datetime.now()
        logger.info("Event processor started")

    async def stop(self):
        """Stop the event processor."""
        self.is_running = False
        logger.info("Event processor stopped")

    async def get_next_event(self) -> Optional[dict]:
        """Get the next event from the queue."""
        try:
            if not self.is_running:
                return None

            if self.event_queue.empty():
                # Return a ping event if queue is empty
                return {
                    "type": "ping",
                    "id": f"ping_{int(datetime.now().timestamp())}",
                    "timestamp": datetime.now().isoformat(),
                    "data": "ping"
                }

            event = await self.event_queue.get()
            self.event_queue.task_done()
            return event

        except Exception as e:
            logger.error(f"Error getting next event: {str(e)}")
            return {
                "type": "error",
                "id": f"error_{int(datetime.now().timestamp())}",
                "timestamp": datetime.now().isoformat(),
                "error": {
                    "code": -32000,
                    "message": str(e)
                }
            }

    async def handle_tool_request(self, request: dict) -> None:
        """Handle a tool request and queue the response."""
        try:
            method = request.get("method")
            params = request.get("params", {})
            request_id = request.get("id")
            
            result = None
            if method == "get_context":
                result = await self.context_manager.get_context(params["file_path"])
            elif method == "get_project_structure":
                result = await self.project_manager.get_structure()
            elif method == "get_mlflow_experiments":
                result = await self.mlflow_manager.list_experiments()
            elif method == "file_event":
                # Handle file system events
                result = {
                    "status": "processed",
                    "event": {
                        "type": params["type"],
                        "timestamp": params["timestamp"],
                        "data": params["data"]
                    }
                }
            else:
                await self.event_queue.put({
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method {method} not found"
                    }
                })
                return
            
            await self.event_queue.put({
                "id": request_id,
                "result": result
            })
            self.processed_count += 1
            
        except Exception as e:
            logger.error(f"Error handling tool request: {str(e)}")
            self.error_count += 1
            await self.event_queue.put({
                "id": request_id,
                "error": {
                    "code": -32000,
                    "message": str(e)
                }
            })

    def is_healthy(self) -> bool:
        """Check if the event processor is healthy."""
        return (
            self.is_running and
            self.error_count < 100  # Arbitrary threshold
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        return {
            "processed_count": self.processed_count,
            "error_count": self.error_count,
            "is_running": self.is_running,
            "uptime": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert EventProcessor to dictionary for JSON serialization."""
        return {
            'type': 'EventProcessor',
            'status': {
                'running': self.is_running,
                'uptime': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
                'processed_count': self.processed_count,
                'error_count': self.error_count,
                'queue_size': self.event_queue.qsize() if hasattr(self.event_queue, 'qsize') else 0
            }
        } 