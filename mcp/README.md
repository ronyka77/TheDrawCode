# Model Context Protocol Server

A real-time context management server for AI IDE integration, providing file system monitoring, semantic analysis, and MLflow integration. The server supports both WebSocket and Server-Sent Events (SSE) for real-time updates, with full JSON-RPC 2.0 protocol support. Includes official Model Context Protocol SDK integration for Cursor IDE.

## Features

- Real-time file system monitoring with intelligent change detection
- Semantic context extraction and code analysis
- Project structure analysis and dependency tracking
- Documentation integration and management
- MLflow experiment tracking and model registry integration
- Dual protocol support (WebSocket and SSE) for real-time updates
- JSON-RPC 2.0 compliant API
- Structured JSON logging with rotation
- CPU-optimized operations for resource efficiency
- Context-aware code completion support
- Comprehensive health monitoring and statistics

## Server Architecture

The server provides two integration points:
1. **FastAPI Server** - Full-featured API server with WebSocket/SSE support
2. **MCP SDK Server** - Official Model Context Protocol implementation for Cursor IDE

### Core Components
- `context_manager.py` - Manages code context and semantic analysis
- `file_monitor.py` - Real-time file system monitoring and change detection
- `event_processor.py` - Asynchronous event handling and WebSocket/SSE updates
- `project_manager.py` - Project structure and dependency management
- `mlflow_manager.py` - MLflow integration and experiment tracking
- `mcp_server.py` - Official MCP SDK integration for Cursor IDE

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Starting the Server

The server can run in three modes:

```bash
# 1. Development mode - Both servers with auto-reload
python -m mcp.server

# 2. Production mode - Both servers with worker processes
MCP_ENV=production python -m mcp.server

# 3. FastAPI server only (if you don't need Cursor IDE integration)
MCP_DISABLE_SDK=true python -m mcp.server
```

By default:
- FastAPI server runs on port 8000
- MCP SDK server runs on port 8001
- Both servers share the same context and file monitoring

### Cursor IDE Integration

The MCP SDK server provides these resources for Cursor IDE:

1. **File Resource** - `file://{file_path}`
   - Provides file contents with syntax highlighting
   - Supports real-time updates

2. **Context Resource** - `context://{file_path}`
   - Provides semantic analysis
   - Code structure and dependencies
   - Real-time context updates

3. **Tools**
   - `refresh_context` - Force context refresh
   - `track_metrics` - Log MLflow metrics
   - `get_project_structure` - Get project tree

4. **Prompts**
   - `analyze_code` - Code analysis suggestions
   - `suggest_tests` - Test case recommendations

### API Endpoints

#### Core Endpoints
- `GET /api/v1/context/{file_path}` - Get semantic context for a file
- `GET /api/v1/structure` - Get project structure and dependencies
- `GET /api/v1/events` - Get recent file and context events
- `POST /api/v1/refresh` - Force context refresh
- `WS /api/v1/ws/events` - WebSocket for real-time updates
- `GET /sse` - Server-Sent Events endpoint for real-time updates

#### MLflow Integration
- `GET /api/v1/mlflow/experiments` - List MLflow experiments
- `GET /api/v1/mlflow/runs/{run_id}` - Get MLflow run details
- `POST /api/v1/mlflow/track` - Track metrics and parameters
- `GET /api/v1/mlflow/models` - List registered models

#### Management & Monitoring
- `GET /api/v1/health` - Server health check with component status
- `GET /api/v1/stats` - Detailed performance statistics
- `GET /api/v1/logs` - Access server logs

#### Tool Discovery & Invocation
- `GET /tools` - List available tools and their schemas
- `POST /tools/{tool_name}` - Invoke a specific tool

### Client Integration

```python
from mcp.client import MCPClient

# Initialize client
client = MCPClient("http://localhost:8000")

# Get semantic context for file
context = await client.get_context("path/to/file.py")

# Subscribe to real-time events (WebSocket)
async for event in client.subscribe_events():
    process_event(event)

# Track MLflow metrics
await client.track_metrics({
    "accuracy": 0.95,
    "loss": 0.05
})

# List registered models
models = await client.list_models()
```

### JSON-RPC Protocol

The server uses JSON-RPC 2.0 for all communications. Example request:

```json
{
    "jsonrpc": "2.0",
    "method": "get_context",
    "params": {
        "file_path": "path/to/file.py"
    },
    "id": 1
}
```

## Configuration

Server configuration via environment variables:

```env
# Server Settings
MCP_ENV=development                          # Environment (development/production)
MCP_HOST=0.0.0.0                            # Server host
MCP_PORT=8000                               # FastAPI server port
MCP_SDK_PORT=8001                           # MCP SDK server port
MCP_DISABLE_SDK=false                       # Disable MCP SDK server
MCP_AUTO_RELOAD=true                        # Enable auto-reload in development

# Component Settings
MCP_LOG_LEVEL=INFO                          # Logging level
MCP_MAX_CACHE_SIZE=100MB                    # Context cache size
MCP_REFRESH_INTERVAL=5                      # File monitoring interval
MCP_MLFLOW_TRACKING_URI=http://localhost:5000 # MLflow server
MCP_WEBSOCKET_PING_INTERVAL=30              # WebSocket keepalive
MCP_MAX_WORKERS=4                           # Async worker pool size

# Security Settings (Production)
MCP_ALLOWED_ORIGINS=["http://localhost:3000"] # CORS allowed origins
MCP_API_KEY=                                # API key for authentication
```

The server behavior changes based on the `MCP_ENV` setting:
- `development`: Enables auto-reload, detailed logging, and accepts all origins
- `production`: Disables auto-reload, enables worker processes, and enforces security settings

## Development

### Running Tests

```bash
# Run all tests with coverage
pytest tests/ --cov=mcp

# Run specific test categories
pytest tests/test_context_manager.py
pytest tests/test_file_monitor.py
```

### Code Style & Quality

```bash
# Format code
black mcp/

# Sort imports
isort mcp/

# Type checking
mypy mcp/

# Linting
flake8 mcp/
```

## Project Structure

```
mcp/
├── server/
│   ├── core/
│   │   ├── context_manager.py   # Semantic analysis
│   │   ├── file_monitor.py      # File system monitoring
│   │   ├── event_processor.py   # Event handling
│   │   ├── project_manager.py   # Project management
│   │   └── mlflow_manager.py    # MLflow integration
│   ├── api/
│   │   ├── routes.py           # API routing
│   │   └── endpoints/          # API implementations
│   ├── models/
│   │   ├── context.py         # Data models
│   │   └── events.py          # Event definitions
│   └── utils/
│       ├── logger.py          # Structured logging
│       └── mlflow_utils.py    # MLflow helpers
├── client.py                  # Python client library
├── tests/                     # Test suite
└── docs/                      # Documentation
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run the full test suite
5. Submit a pull request

## License

MIT License - see LICENSE file for details 