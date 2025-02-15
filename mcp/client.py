"""
MCP Client for IDE integration.
Provides a simple interface to communicate with the MCP server.
"""

import asyncio
import json
from typing import Dict, Any, AsyncIterator, Optional
import websockets
import aiohttp
from datetime import datetime

class MCPClient:
    """Client for interacting with the MCP server."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        websocket_url: str = "ws://localhost:8000"
    ):
        """
        Initialize the MCP client.
        
        Args:
            base_url: Base URL of the MCP server
            websocket_url: WebSocket URL for real-time events
        """
        self.base_url = base_url.rstrip('/')
        self.websocket_url = websocket_url.rstrip('/')
        self.session = None
        
    async def __aenter__(self):
        """Create aiohttp session when entering context."""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close aiohttp session when exiting context."""
        if self.session:
            await self.session.close()
            
    async def get_context(self, file_path: str) -> Dict[str, Any]:
        """
        Get context information for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary containing file context
        """
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/api/v1/context/{file_path}"
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.json()
                
    async def get_project_structure(self) -> Dict[str, Any]:
        """
        Get the current project structure.
        
        Returns:
            Dictionary containing project structure
        """
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/api/v1/structure"
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.json()
                
    async def refresh_context(self) -> Dict[str, Any]:
        """
        Force refresh of the context cache.
        
        Returns:
            Status of the refresh operation
        """
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/api/v1/refresh"
            async with session.post(url) as response:
                response.raise_for_status()
                return await response.json()
                
    async def subscribe_events(self) -> AsyncIterator[Dict[str, Any]]:
        """
        Subscribe to real-time events via WebSocket.
        
        Yields:
            Event dictionaries
        """
        url = f"{self.websocket_url}/api/v1/ws/events"
        async with websockets.connect(url) as websocket:
            while True:
                try:
                    message = await websocket.recv()
                    event = json.loads(message)
                    yield event
                except websockets.ConnectionClosed:
                    break
                    
    async def get_mlflow_experiments(self) -> Dict[str, Any]:
        """
        List all MLflow experiments.
        
        Returns:
            List of experiments
        """
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/api/v1/mlflow/experiments"
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.json()
                
    async def get_mlflow_run(self, run_id: str) -> Dict[str, Any]:
        """
        Get information for a specific MLflow run.
        
        Args:
            run_id: MLflow run ID
            
        Returns:
            Run information
        """
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/api/v1/mlflow/runs/{run_id}"
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.json()
                
    async def get_health(self) -> Dict[str, Any]:
        """
        Get health status of server components.
        
        Returns:
            Health status dictionary
        """
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/api/v1/health"
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.json()
                
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get server statistics.
        
        Returns:
            Statistics dictionary
        """
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/api/v1/stats"
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.json()

    async def get_server_info(self) -> Dict[str, Any]:
        """
        Get basic server information from the root endpoint.
        
        Returns:
            Dictionary containing server name, version, and status
        """
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/"
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.json() 