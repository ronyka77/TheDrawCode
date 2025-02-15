"""
File system monitor for MCP server.
Tracks file changes and directory structure updates.
"""

import os
import asyncio
from typing import Dict, Set, Optional, List, Any
from datetime import datetime
import fnmatch
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent
import threading

from ..utils.logger import default_logger as logger
from .event_processor import EventProcessor

def run_async(coro):
    """Run an async function from a sync context."""
    loop = asyncio.new_event_loop()
    return loop.run_until_complete(coro)

class MCPFileHandler(FileSystemEventHandler):
    """
    Custom file system event handler for MCP.
    Processes file system events and forwards them to the event processor.
    """
    
    def __init__(
        self,
        event_processor: EventProcessor,
        ignore_patterns: Optional[List[str]] = None
    ):
        """
        Initialize the file handler.
        
        Args:
            event_processor: Event processor instance
            ignore_patterns: List of glob patterns to ignore
        """
        self.event_processor = event_processor
        self.ignore_patterns = ignore_patterns or [
            ".git/*",
            "__pycache__/*",
            "*.pyc",
            "*.pyo",
            "*.pyd",
            ".DS_Store"
        ]
        
        # Track modified times to prevent duplicate events
        self.last_modified = {}
        
        # Create event loop for async operations
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.thread.start()
        
        logger.info(
            "File handler initialized",
            extra_fields={"ignore_patterns": self.ignore_patterns}
        )

    def _run_event_loop(self):
        """Run the event loop in a separate thread."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _publish_event(self, event_type: str, data: Dict[str, Any]):
        """Publish event using the event loop."""
        try:
            event = {
                "id": f"file_{event_type}_{datetime.now().timestamp()}",
                "method": "file_event",
                "params": {
                    "type": event_type,
                    "timestamp": datetime.now().isoformat(),
                    "data": data
                }
            }
            
            # Use run_coroutine_threadsafe since we're in a different thread
            asyncio.run_coroutine_threadsafe(
                self.event_processor.handle_tool_request(event),
                self.loop
            )
            
        except Exception as e:
            logger.error(f"Error publishing event: {str(e)}")

    def should_ignore(self, path: str) -> bool:
        """
        Check if a path should be ignored.
        
        Args:
            path: File path to check
            
        Returns:
            Boolean indicating if path should be ignored
        """
        return any(
            fnmatch.fnmatch(path, pattern)
            for pattern in self.ignore_patterns
        )

    def on_created(self, event: FileSystemEvent):
        """Handle file/directory creation events."""
        if not self.should_ignore(event.src_path):
            self._publish_event(
                "file_created",
                {
                    "path": event.src_path,
                    "is_directory": event.is_directory
                }
            )

    def on_modified(self, event: FileSystemEvent):
        """Handle file modification events."""
        if not self.should_ignore(event.src_path):
            # Check if this is a duplicate event
            current_time = datetime.now().timestamp()
            last_time = self.last_modified.get(event.src_path, 0)
            
            # Only process if enough time has passed (debounce)
            if current_time - last_time > 0.1:  # 100ms debounce
                self.last_modified[event.src_path] = current_time
                self._publish_event(
                    "file_modified",
                    {
                        "path": event.src_path,
                        "is_directory": event.is_directory
                    }
                )

    def on_deleted(self, event: FileSystemEvent):
        """Handle file/directory deletion events."""
        if not self.should_ignore(event.src_path):
            self._publish_event(
                "file_deleted",
                {
                    "path": event.src_path,
                    "is_directory": event.is_directory
                }
            )

    def on_moved(self, event: FileSystemEvent):
        """Handle file/directory move events."""
        if not (self.should_ignore(event.src_path) or 
                self.should_ignore(event.dest_path)):
            self._publish_event(
                "file_moved",
                {
                    "src_path": event.src_path,
                    "dest_path": event.dest_path,
                    "is_directory": event.is_directory
                }
            )

    def __del__(self):
        """Clean up the event loop."""
        if hasattr(self, 'loop') and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=1.0)

class FileMonitor:
    """
    File system monitor for tracking changes in the workspace.
    Uses watchdog for file system events and forwards them to the event processor.
    """
    
    def __init__(
        self,
        event_processor: Optional[EventProcessor] = None,
        watch_patterns: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None
    ):
        """
        Initialize the file monitor.
        
        Args:
            event_processor: Event processor instance
            watch_patterns: List of glob patterns to watch
            ignore_patterns: List of glob patterns to ignore
        """
        self.event_processor = event_processor or EventProcessor()
        self.watch_patterns = watch_patterns or ["*"]
        self.ignore_patterns = ignore_patterns
        
        # Initialize observer
        self.observer = Observer()
        self.handler = MCPFileHandler(
            self.event_processor,
            self.ignore_patterns
        )
        
        self.watched_paths = set()
        self.is_running = False
        self.start_time = None
        
        logger.info(
            "File monitor initialized",
            extra_fields={
                "watch_patterns": self.watch_patterns,
                "ignore_patterns": self.ignore_patterns
            }
        )

    async def start(self, paths: Optional[List[str]] = None):
        """
        Start monitoring file system events.
        
        Args:
            paths: List of paths to monitor (defaults to current directory)
        """
        if self.is_running:
            return
        
        try:
            # Default to current directory if no paths specified
            paths = paths or [os.getcwd()]
            
            # Schedule watches
            for path in paths:
                if os.path.exists(path):
                    self.observer.schedule(
                        self.handler,
                        path,
                        recursive=True
                    )
                    self.watched_paths.add(path)
                else:
                    logger.warning(f"Path does not exist: {path}")
            
            # Start observer
            self.observer.start()
            self.is_running = True
            self.start_time = datetime.now()
            
            logger.info(
                "File monitor started",
                extra_fields={"watched_paths": list(self.watched_paths)}
            )
            
        except Exception as e:
            logger.error(f"Failed to start file monitor: {str(e)}")
            raise

    async def stop(self):
        """Stop monitoring file system events."""
        if not self.is_running:
            return
        
        try:
            self.observer.stop()
            self.observer.join()
            self.is_running = False
            self.watched_paths.clear()
            logger.info("File monitor stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop file monitor: {str(e)}")
            raise

    def add_watch(self, path: str):
        """
        Add a new path to monitor.
        
        Args:
            path: Path to monitor
        """
        if os.path.exists(path) and path not in self.watched_paths:
            try:
                self.observer.schedule(
                    self.handler,
                    path,
                    recursive=True
                )
                self.watched_paths.add(path)
                logger.info(f"Added watch for path: {path}")
                
            except Exception as e:
                logger.error(f"Failed to add watch for path: {str(e)}")
                raise

    def remove_watch(self, path: str):
        """
        Remove a monitored path.
        
        Args:
            path: Path to stop monitoring
        """
        if path in self.watched_paths:
            try:
                # Find and remove the watch
                for watch in self.observer._watches:
                    if watch.path == path:
                        self.observer.unschedule(watch)
                        break
                
                self.watched_paths.remove(path)
                logger.info(f"Removed watch for path: {path}")
                
            except Exception as e:
                logger.error(f"Failed to remove watch for path: {str(e)}")
                raise

    def is_healthy(self) -> bool:
        """
        Check if the file monitor is healthy.
        
        Returns:
            Boolean indicating health status
        """
        return (
            self.is_running and
            self.observer.is_alive() and
            len(self.watched_paths) > 0
        )

    def get_monitor_count(self) -> int:
        """
        Get number of active monitors.
        
        Returns:
            Number of paths being monitored
        """
        return len(self.watched_paths)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get monitor statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "is_running": self.is_running,
            "watched_paths": list(self.watched_paths),
            "monitor_count": self.get_monitor_count(),
            "uptime": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        } 