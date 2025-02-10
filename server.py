#!/usr/bin/env python3
"""
Enhanced MCP Server for Cursor IDE

This server exposes MCP resources for local file context and documentation,
improving the data available to the AI. It wraps FastMCP with FastAPI and
provides:
  - MCP resources under /mcp (e.g. file read, file listing)
  - An SSE endpoint (/sse) for event updates
  - A dedicated endpoint (/ai_context) to return enhanced AI context
  - Logging and error handling for easier debugging and monitoring
"""

import os
import asyncio
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

# Import FastMCP from the mcp package
from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EnhancedMCPServer")

# Create an MCP server instance with a descriptive name
mcp = FastMCP("EnhancedLocalFileContextServer")

# MCP Resource: Read file contents from the local filesystem.
@mcp.resource("file:///{file_path}")
def read_file(file_path: str) -> str:
    """
    Reads the contents of a specified file.
    """
    logger.info(f"Reading file: {file_path}")
    try:
        with open(file_path, 'r') as f:
            data = f.read()
        return data
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return f"Error reading file: {str(e)}"

# MCP Resource: List all files in a given directory.
@mcp.resource("list:///{directory}")
def list_files(directory: str) -> str:
    """
    Lists all files in the specified directory.
    """
    logger.info(f"Listing files in directory: {directory}")
    try:
        files = os.listdir(directory)
        # Filter to return only files.
        files = [f for f in files if os.path.isfile(os.path.join(directory, f))]
        return "\n".join(files)
    except Exception as e:
        logger.error(f"Error listing files in directory {directory}: {e}")
        return f"Error listing files: {str(e)}"

# MCP Tool: Echo a message with a prefix.
@mcp.tool()
def echo(message: str) -> str:
    """
    Echoes the input message.
    """
    logger.info(f"Echoing message: {message}")
    return f"Echo: {message}"

# MCP Tool: Get Docs Folder Markdown Files.
@mcp.tool()
def get_docs_markdown() -> str:
    """
    Retrieves and concatenates all Markdown files from the docs folder.
    """
    logger.info("Retrieving Markdown files from the docs folder.")
    docs_folder = "docs"
    merged_markdown = ""
    if os.path.isdir(docs_folder):
        for fname in sorted(os.listdir(docs_folder)):
            # Process only Markdown files.
            if fname.lower().endswith(".md"):
                fpath = os.path.join(docs_folder, fname)
                try:
                    with open(fpath, 'r') as f:
                        content = f.read()
                    merged_markdown += f"\n=== {fname} ===\n{content}\n"
                except Exception as e:
                    logger.error(f"Error reading {fpath}: {e}")
                    merged_markdown += f"\n=== {fname} ===\nError reading file: {e}\n"
    else:
        logger.warning("Docs folder not found.")
        merged_markdown = "Docs folder not found."
    return merged_markdown.strip()

# Helper function: Read all files in the docs folder.
def read_docs_folder() -> str:
    docs_folder = "docs"
    docs_content = ""
    if os.path.isdir(docs_folder):
        for fname in sorted(os.listdir(docs_folder)):
            fpath = os.path.join(docs_folder, fname)
            if os.path.isfile(fpath):
                try:
                    with open(fpath, 'r') as f:
                        content = f.read()
                    docs_content += f"\n=== {fname} ===\n{content}\n"
                except Exception as e:
                    logger.error(f"Error reading {fpath}: {e}")
                    docs_content += f"\n=== {fname} ===\nError reading file: {e}\n"
    else:
        logger.warning("Docs folder not found.")
    return docs_content.strip()

# MCP Prompt: Provide a file context prompt for AI-enhanced responses.
@mcp.prompt()
def file_context_prompt(file_path: str) -> str:
    """
    Creates a prompt using the file's content along with the contents 
    of your docs folder for comprehensive AI documentation.
    """
    logger.info(f"Generating file context prompt for: {file_path}")
    content = read_file(file_path)
    docs_content = read_docs_folder()
    final_prompt = f"File content for {file_path}:\n{content}"
    if docs_content:
        final_prompt += f"\n\n--- Docs Folder Context ---\n{docs_content}"
    return final_prompt

# Create a FastAPI app and mount the MCP instance at /mcp.
app = FastAPI(
    title="Enhanced MCP Server",
    description="Provides local file context and documentation for enhanced AI responses in Cursor IDE.",
    version="1.0.0"
)
app.mount("/mcp", mcp)

# Middleware to log all incoming requests.
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

# SSE Endpoint: Real-time updates for Cursor IDE.
@app.get("/sse")
async def sse_endpoint():
    async def event_generator():
        try:
            # Simulate sending an SSE event every 2 seconds.
            while True:
                yield "data: Enhanced MCP SSE event: AI context updated\n\n"
                await asyncio.sleep(2)
        except asyncio.CancelledError:
            logger.info("SSE connection was cancelled.")
            raise
    return StreamingResponse(event_generator(), media_type="text/event-stream")

# AI Context Endpoint: Returns enhanced AI context for a file.
@app.get("/ai_context")
async def ai_context(file_path: str):
    """
    Returns enhanced context (file content combined with docs folder content) for a 
    specified file, which can be used to improve AI responses.
    """
    logger.info(f"Generating AI context for file: {file_path}")
    try:
        context = file_context_prompt(file_path)
        return JSONResponse(content={"file": file_path, "context": context})
    except Exception as e:
        logger.error(f"Failed to generate AI context for {file_path}: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate AI context")

# Root endpoint for health check and available endpoints listing.
@app.get("/")
async def root():
    return JSONResponse(content={
        "message": "Enhanced MCP Server is running",
        "endpoints": ["/mcp", "/sse", "/ai_context"]
    })

@app.get("/tools")
async def tools_endpoint():
    """
    Returns a list of registered MCP tools.
    """
    # Update the static list to reflect both tools.
    return {"tools": ["echo", "get_docs_markdown"]}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")