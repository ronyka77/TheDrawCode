"""MCP Server implementation using the official Model Context Protocol SDK."""

from fastmcp import FastMCP, Context
from typing import Dict, Any, Optional, AsyncGenerator, List
import os
import json
from datetime import datetime
import asyncio
import sys

from .core.context_manager import ContextManager
from .core.file_monitor import FileMonitor
from .core.event_processor import EventProcessor
from .core.project_manager import ProjectManager
from .core.mlflow_manager import MCPMLflow
from .utils.logger import ExperimentLogger

logger = ExperimentLogger()

class Local_MCP_Server():
    """Context server implementation that integrates with Cursor IDE."""
    
    def __init__(self):
        """Initialize the server components."""
        
        # Initialize event processor first
        self.event_processor = EventProcessor()
        
        # Initialize FastMCP with server info and transport
        self.mcp = FastMCP(
            name="Model Context Protocol Server",
            description="A server that provides context and semantic information for ML model development.",
            version="1.0.0",
        )
        
        # Initialize other components with event processor
        self.context_manager = ContextManager(self.event_processor)
        self.file_monitor = FileMonitor(self.event_processor)
        self.project_manager = ProjectManager()
        self.mlflow_manager = MCPMLflow()
        
        # Register MCP components
        self._register_resources()
        self._register_tools()
        self._register_prompts()

    async def start(self):
        """Initialize components on server startup."""
        try:
            await super().start()
            await self.event_processor.start()
            logger.info("Event processor started")

            await self.file_monitor.start()
            logger.info("File monitor started")

            logger.info("MLflow tracking initialized")
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise

    async def stop(self):
        """Clean up components on server shutdown."""
        try:
            await self.file_monitor.stop()
            logger.info("File monitor stopped")

            await self.event_processor.stop()
            logger.info("Event processor stopped")
            
            await super().stop()
        except Exception as e:
            logger.error(f"Failed to clean up components: {str(e)}")

    async def handle_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Handle server requests."""
        try:
            return await self.mcp.handle_request(method, params or {})
        except Exception as e:
            logger.error(f"Error handling request {method}: {str(e)}")
            raise

    def _register_resources(self):
        @self.mcp.resource("server://info")
        async def server_info() -> tuple[str, str]:
            """Provide server information as a resource."""
            try:
                info = {
                    "name": "Model Context Protocol Server",
                    "version": "1.0.0",
                    "status": "running",
                }
                return json.dumps(info), "application/json"
            except Exception as e:
                logger.error(f"Failed to get server info: {str(e)}")
                raise

        @self.mcp.resource("file://{file_path}")
        async def file_resource(file_path: str) -> tuple[bytes, str]:
            """Provide file contents as a resource."""
            try:
                content = await self.context_manager.get_file_content(file_path)
                return content.encode(), "text/plain"
            except Exception as e:
                logger.error(f"Failed to read file {file_path}: {str(e)}")
                raise

        @self.mcp.resource("context://{file_path}")
        async def context_resource(file_path: str) -> tuple[str, str]:
            """Provide semantic context for a file."""
            try:
                context = await self.context_manager.get_context(file_path)
                return json.dumps(context), "application/json"
            except Exception as e:
                logger.error(f"Failed to get context for {file_path}: {str(e)}")
                raise

        @self.mcp.resource("structure://project")
        async def structure_resource() -> tuple[str, str]:
            """Provide project structure as a resource."""
            try:
                structure = await self.project_manager.get_structure()
                return json.dumps(structure), "application/json"
            except Exception as e:
                logger.error(f"Failed to get project structure: {str(e)}")
                raise

        @self.mcp.resource("events://recent")
        async def events_resource() -> tuple[str, str]:
            """Provide recent events as a resource."""
            try:
                events = await self.event_processor.get_recent_events()
                return json.dumps(events), "application/json"
            except Exception as e:
                logger.error(f"Failed to get events: {str(e)}")
                raise

        @self.mcp.resource("tools://list")
        async def tools_resource() -> tuple[str, str]:
            """Provide available tools as a resource."""
            try:
                tools_response = {
                    "schema_version": "1.0",
                    "transport": {
                        "type": "sse",
                        "endpoint": "/sse",
                        "protocol": "json-rpc"
                    },
                    "tools": [
                        {
                            "name": "get_context",
                            "description": "Get semantic context information for a file",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "file_path": {
                                        "type": "string",
                                        "description": "Path to the file to analyze"
                                    }
                                },
                                "required": ["file_path"]
                            }
                        },
                        {
                            "name": "get_project_structure",
                            "description": "Get the current project structure tree",
                            "parameters": {
                                "type": "object",
                                "properties": {}
                            }
                        }
                    ]
                }
                return json.dumps(tools_response), "application/json"
            except Exception as e:
                logger.error(f"Failed to get tools list: {str(e)}")
                raise

    def _register_tools(self):
        @self.mcp.tool()
        async def refresh_context(ctx: Context) -> str:
            """Force refresh of the context cache."""
            try:
                ctx.info("Starting context refresh...")
                await self.context_manager.refresh()
                ctx.info("Context cache refreshed successfully")
                return "Context cache refreshed successfully"
            except Exception as e:
                ctx.error(f"Failed to refresh context: {str(e)}")
                raise

        @self.mcp.tool()
        async def track_metrics(metrics: Dict[str, float], ctx: Context) -> str:
            """Track MLflow metrics."""
            try:
                ctx.info(f"Logging {len(metrics)} metrics...")
                await self.mlflow_manager.log_metrics(metrics)
                ctx.info("Metrics logged successfully")
                return f"Logged {len(metrics)} metrics successfully"
            except Exception as e:
                ctx.error(f"Failed to log metrics: {str(e)}")
                raise

        @self.mcp.tool()
        async def analyze_files(files: List[str], ctx: Context) -> Dict[str, Any]:
            """Analyze multiple files with progress tracking."""
            try:
                results = {}
                for i, file_path in enumerate(files):
                    ctx.info(f"Analyzing {file_path}")
                    await ctx.report_progress(i, len(files))
                    
                    # Get file content
                    content = await ctx.read_resource(f"file://{file_path}")
                    
                    # Get semantic context
                    context = await self.context_manager.get_context(file_path)
                    results[file_path] = {
                        "content": content,
                        "context": context
                    }
                
                ctx.info("Analysis complete")
                return results
            except Exception as e:
                ctx.error(f"Failed to analyze files: {str(e)}")
                raise

        @self.mcp.tool()
        async def process_project(ctx: Context) -> Dict[str, Any]:
            """Process entire project structure with progress tracking."""
            try:
                ctx.info("Starting project analysis...")
                
                # Get project structure
                structure = await self.project_manager.get_structure()
                total_files = len(structure.get("files", []))
                
                processed = 0
                results = {
                    "files": {},
                    "metrics": {},
                    "dependencies": []
                }
                
                for file_path in structure.get("files", []):
                    ctx.info(f"Processing {file_path}")
                    await ctx.report_progress(processed, total_files)
                    
                    # Get file content and context
                    content = await ctx.read_resource(f"file://{file_path}")
                    context = await self.context_manager.get_context(file_path)
                    
                    results["files"][file_path] = {
                        "content": content,
                        "context": context
                    }
                    processed += 1
                
                ctx.info("Project analysis complete")
                return results
            except Exception as e:
                ctx.error(f"Failed to process project: {str(e)}")
                raise

        @self.mcp.tool()
        async def get_health(ctx: Context) -> Dict[str, Any]:
            """Get health status of server components."""
            try:
                ctx.info("Checking component health...")
                status = {
                    "server": "healthy",
                    "event_processor": self.event_processor.is_healthy(),
                    "file_monitor": self.file_monitor.is_healthy(),
                    "context_manager": self.context_manager.is_healthy(),
                    "mlflow": self.mlflow_manager.is_healthy(),
                }
                ctx.info("Health check complete")
                return status
            except Exception as e:
                ctx.error(f"Health check failed: {str(e)}")
                raise

        @self.mcp.tool()
        async def get_stats(ctx: Context) -> Dict[str, Any]:
            """Get detailed server statistics."""
            try:
                ctx.info("Gathering system statistics...")
                stats = {
                    "event_processor": {
                        **self.event_processor.get_stats(),
                        "queue_size": self.event_processor.get_queue_size(),
                    },
                    "context_manager": {
                        **self.context_manager.get_stats(),
                        "cache_size": self.context_manager.get_cache_size(),
                        "uptime": self.context_manager.get_uptime(),
                    },
                    "file_monitor": {
                        **self.file_monitor.get_stats(),
                        "active_monitors": self.file_monitor.get_monitor_count(),
                    },
                    "mlflow": {
                        "connection_status": self.mlflow_manager.is_healthy(),
                        "tracking_uri": self.mlflow_manager.get_tracking_uri(),
                    },
                    "server": {
                        "version": "1.0.0",
                        "start_time": self.context_manager.get_start_time().isoformat(),
                    }
                }
                ctx.info("Statistics gathered successfully")
                return stats
            except Exception as e:
                ctx.error(f"Failed to get stats: {str(e)}")
                raise

        @self.mcp.tool()
        async def get_logs(ctx: Context) -> Dict[str, Any]:
            """Get recent server logs."""
            try:
                logs = logger.get_recent_logs()
                return {
                    "logs": logs,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Failed to get logs: {str(e)}")
                raise

        @self.mcp.tool()
        async def list_models(ctx: Context) -> list:
            """List all registered MLflow models."""
            try:
                models = self.mlflow_manager.list_registered_models()
                logger.info(f"Listed {len(models)} registered models")
                return models
            except Exception as e:
                logger.error(f"Failed to list registered models: {str(e)}")
                raise

        @self.mcp.tool()
        async def invoke_tool(tool_name: str, params: Dict[str, Any], ctx: Context) -> Any:
            """Handle dynamic tool invocation requests."""
            try:
                # Queue the tool request
                await self.event_processor.handle_tool_request({
                    "method": tool_name,
                    "params": params,
                    "id": ctx.request_id
                })
                return "Request accepted"
            except Exception as e:
                logger.error(f"Error invoking tool {tool_name}: {str(e)}")
                raise

    def _register_prompts(self):
        @self.mcp.prompt()
        def analyze_code(file_path: str) -> str:
            """Create a prompt for code analysis."""
            return f"""Please analyze the code in {file_path} and provide:
1. A summary of its functionality
2. Potential improvements
3. Any security concerns
4. Code quality assessment"""

        @self.mcp.prompt()
        def suggest_tests(file_path: str) -> str:
            """Create a prompt for test suggestions."""
            return f"""Please suggest unit tests for {file_path} including:
1. Key test cases to cover
2. Edge cases to consider
3. Mocking requirements
4. Test structure recommendations"""

        @self.mcp.prompt()
        def suggest_documentation(file_path: str) -> str:
            """Create a prompt for documentation suggestions."""
            return f"""Please suggest documentation improvements for {file_path} including:
1. Missing docstrings
2. API documentation
3. Usage examples
4. Type hints"""

    async def _handle_events(self):
        """Handle events from the event processor."""
        try:
            async for event in self.event_processor.subscribe():
                # Format as JSON-RPC response
                response = {
                    "jsonrpc": "2.0",
                    "method": "event",
                    "params": event
                }
                
                # Send event through FastMCP's event system
                await self.mcp.broadcast_event(response)
                
        except asyncio.CancelledError:
            logger.info("Event handling cancelled")
        except Exception as e:
            logger.error(f"Error handling events: {str(e)}")

    async def run(self):
        """Run the MCP server."""
        try:
            logger.info("Starting MCP server...")
            await self.start()
            
            # Start event handling
            event_task = asyncio.create_task(self._handle_events())
            
            # Run the server
            await self.mcp.run()
            
        except Exception as e:
            logger.error(f"Failed to start MCP server: {str(e)}")
            raise
        finally:
            # Clean up
            await self.stop()

def main():
    """Console script entry point."""
    try:
        server = ContextServer()
        asyncio.run(server.run())
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    except Exception as e:
        logger.error(f"Error running server: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 