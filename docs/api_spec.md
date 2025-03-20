# Model Context Protocol Server API Specification v2.1

## Overview

This document describes the API endpoints provided by the Model Context Protocol (MCP) server. The server provides real-time context management, file monitoring, and MLflow integration through REST and SSE endpoints.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. CORS is enabled for cross-origin requests.

## Endpoints

### Tools Endpoint

#### GET /tools

Returns the list of available tools and their specifications.

**Response Format:**
```json
{
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
    // ... other tools ...
  ]
}
```

### Context Management

#### GET /api/v1/context/{file_path}

Get context information for a specific file.

**Parameters:**
- `file_path`: Path to the file (URL-encoded)

**Response:**
```json
{
  "file_info": {
    "path": "string",
    "size": "integer",
    "modified": "string (ISO 8601)",
    "created": "string (ISO 8601)",
    "type": "string"
  },
  "semantic_info": {
    "language": "string",
    "imports": ["string"],
    "functions": ["string"],
    "classes": ["string"]
  },
  "related_files": ["string"],
  "mlflow_context": {
    "experiments": [],
    "recent_runs": []
  }
}
```

#### GET /api/v1/structure

Get the current project structure.

**Response:**
```json
{
  "timestamp": "string (ISO 8601)",
  "root": {
    "name": "string",
    "type": "directory",
    "children": [
      {
        "name": "string",
        "type": "file|directory",
        "info": {
          "path": "string",
          "size": "integer",
          "modified": "string",
          "created": "string",
          "type": "string"
        }
      }
    ]
  }
}
```

#### POST /api/v1/refresh

Force refresh of the context cache.

**Response:**
```json
{
  "status": "success",
  "message": "Context cache refreshed"
}
```

### Event Streaming

#### GET /sse

Server-Sent Events endpoint for real-time updates.

**Event Types:**
- `message`: Tool response
- `ping`: Keep-alive
- `error`: Error event

**Event Format:**
```json
{
  "event": "message",
  "id": "string",
  "data": {
    "jsonrpc": "2.0",
    "result": {}
  }
}
```

### MLflow Integration

#### GET /api/v1/mlflow/experiments

List all MLflow experiments.

**Response:**
```json
[
  {
    "experiment_id": "string",
    "name": "string",
    "artifact_location": "string",
    "lifecycle_stage": "string"
  }
]
```

#### GET /api/v1/mlflow/runs/{run_id}

Get information for a specific MLflow run.

**Parameters:**
- `run_id`: MLflow run ID

**Response:**
```json
{
  "run_id": "string",
  "status": "string",
  "start_time": "integer",
  "end_time": "integer",
  "metrics": {},
  "parameters": {},
  "tags": {}
}
```

### System Management

#### GET /api/v1/health

Get health status of server components.

**Response:**
```json
{
  "server": "string",
  "file_monitor": "boolean",
  "event_processor": "boolean",
  "context_manager": "boolean",
  "mlflow": "boolean"
}
```

#### GET /api/v1/stats

Get server statistics.

**Response:**
```json
{
  "event_processor": {
    "processed_count": "integer",
    "error_count": "integer",
    "is_running": "boolean",
    "uptime": "float"
  },
  "context_manager": {
    "processed_files": "integer",
    "cache_hits": "integer",
    "cache_misses": "integer",
    "cache_size": "integer",
    "uptime": "float"
  },
  "file_monitor": {
    "is_running": "boolean",
    "watched_paths": ["string"],
    "monitor_count": "integer",
    "uptime": "float"
  }
}
```

## Error Handling

All endpoints follow a consistent error response format:

```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": "integer",
    "message": "string"
  }
}
```

Common error codes:
- `-32600`: Invalid Request
- `-32601`: Method not found
- `-32000`: Server error

## Rate Limiting

Currently, no rate limiting is implemented. However, the server includes event debouncing for file system events (100ms) and SSE retry intervals (1000ms).

## CORS Configuration

CORS is enabled with the following settings:
- `allow_origins=["*"]`
- `allow_credentials=True`
- `allow_methods=["*"]`
- `allow_headers=["*"]`
- `expose_headers=["*"]`
- `max_age=3600` 