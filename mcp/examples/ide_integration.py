"""
Example of integrating MCP client with an IDE.
This script demonstrates how to use the MCP client to get context and real-time updates.
"""

import asyncio
import os
from typing import Dict, Any
from mcp.client import MCPClient
import aiohttp

async def handle_file_context(client: MCPClient, file_path: str):
    """Handle file context updates."""
    try:
        # Get context for the current file
        context = await client.get_context(file_path)
        
        # Extract useful information
        file_info = context["file_info"]
        semantic_info = context["semantic_info"]
        related_files = context["related_files"]
        
        print(f"\nContext for {file_path}:")
        print(f"File type: {file_info['type']}")
        print(f"Last modified: {file_info['modified']}")
        print("\nSemantic information:")
        print(f"Language: {semantic_info['language']}")
        print(f"Imports: {', '.join(semantic_info['imports'])}")
        print(f"Functions: {', '.join(semantic_info['functions'])}")
        print(f"Classes: {', '.join(semantic_info['classes'])}")
        print("\nRelated files:")
        for related in related_files:
            print(f"- {related}")
            
    except Exception as e:
        print(f"Error getting context: {str(e)}")

async def handle_project_structure(client: MCPClient):
    """Handle project structure updates."""
    try:
        # Get current project structure
        structure = await client.get_project_structure()
        
        print("\nProject Structure:")
        def print_tree(node: Dict[str, Any], indent: str = ""):
            print(f"{indent}- {node['name']} ({node['type']})")
            for child in node.get("children", []):
                print_tree(child, indent + "  ")
                
        print_tree(structure["root"])
        
    except Exception as e:
        print(f"Error getting project structure: {str(e)}")

async def handle_events(client: MCPClient):
    """Handle real-time events from the server."""
    try:
        print("\nListening for events...")
        async for event in client.subscribe_events():
            event_type = event["type"]
            timestamp = event["timestamp"]
            
            if event_type == "file_changed":
                print(f"\nFile changed: {event['data']['path']}")
                # Trigger context refresh for the changed file
                await handle_file_context(client, event['data']['path'])
                
            elif event_type == "context_updated":
                print(f"\nContext updated for: {event['file_path']}")
                print(f"Changes: {event['changes']}")
                
            elif event_type == "mlflow_update":
                print(f"\nMLflow update in experiment: {event['experiment_id']}")
                if event['run_id']:
                    print(f"Run ID: {event['run_id']}")
                    print(f"Metrics: {event['metrics']}")
                    
    except Exception as e:
        print(f"Error handling events: {str(e)}")

async def main():
    """Main function demonstrating MCP client usage."""
    # Initialize client
    client = MCPClient()
    
    try:
        # Get basic server information
        server_info = await client.get_server_info()
        print("\nServer Information:")
        print(f"Name: {server_info['name']}")
        print(f"Version: {server_info['version']}")
        print(f"Status: {server_info['status']}")
        
        # Get server statistics
        stats = await client.get_stats()
        print("\nServer Statistics:")
        if "event_processor" in stats:
            print("\nEvent Processor Stats:")
            for key, value in stats["event_processor"].items():
                print(f"{key}: {value}")
        if "context_manager" in stats:
            print("\nContext Manager Stats:")
            for key, value in stats["context_manager"].items():
                print(f"{key}: {value}")
        if "file_monitor" in stats:
            print("\nFile Monitor Stats:")
            for key, value in stats["file_monitor"].items():
                print(f"{key}: {value}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    # Run the example
    asyncio.run(main()) 