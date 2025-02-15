"""
Main entry point for the MCP server.
"""

import asyncio
import os
import sys
from typing import Optional

from .app import Local_MCP_Server

logger = ExperimentLogger()

def main():
    """Main entry point."""
    try:
        # Get configuration from environment
        host = os.getenv("MCP_HOST", "0.0.0.0")
        port = int(os.getenv("MCP_PORT", "8000"))
        env = os.getenv("MCP_ENV", "development")
        
        # Create and start server
        server = Local_MCP_Server()
        logger.info(f"Starting context server on {host}:{port}...")
        
        # Run server
        asyncio.run(server.run())
        
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()