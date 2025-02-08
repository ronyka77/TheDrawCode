# Model Context Protocol Server (MCPS) API Specification

## Overview

The Model Context Protocol Server (MCPS) provides a RESTful API for managing the lifecycle of soccer prediction models. This document details the available endpoints, their request/response formats, and usage examples.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication for local development. Production deployments should implement appropriate authentication mechanisms.

## Common Headers

All requests should include:
```http
Content-Type: application/json
Accept: application/json
```

## Error Responses

All endpoints may return the following error responses:

```json
{
    "detail": "Error message describing what went wrong"
}
```

HTTP Status codes:
- `400`: Bad Request - Invalid input
- `404`: Not Found - Resource doesn't exist
- `500`: Internal Server Error - Server-side error

## Endpoints

### Training Management

#### Start Training Run

Start a new model training run.

```http
POST /train
```

Request body:
```json
{
    "model_type": "xgboost",
    "experiment_name": "soccer_prediction_v1",
    "hyperparameters": {
        "learning_rate": 0.02,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "objective": "binary:logistic"
    }
}
```

Response:
```json
{
    "run_id": "abcd1234",
    "status": "STARTING",
    "start_time": "2024-01-31T22:12:34.567Z"
}
```

#### Get Training Status

Get the current status of a training run.

```http
GET /status/{run_id}
```

Response:
```json
{
    "status": "RUNNING",
    "metrics": {
        "precision": 0.82,
        "recall": 0.76,
        "f1": 0.79,
        "best_threshold": 0.58
    },
    "start_time": "2024-01-31T22:12:34.567Z",
    "last_updated": "2024-01-31T22:15:45.678Z"
}
```

#### Stop Training Run

Stop an active training run.

```http
POST /stop/{run_id}
```

Response:
```json
{
    "status": "Training stopped successfully"
}
```

### Model Predictions

#### Make Predictions

Make predictions using a trained model.

```http
POST /predict
```

Request body:
```json
{
    "model_uri": "runs:/abcd1234/model",
    "data": {
        "feature1": [1.0, 2.0, 3.0],
        "feature2": [0.1, 0.2, 0.3],
        "feature3": [10, 20, 30]
    }
}
```

Response:
```json
{
    "predictions": [0, 1, 0],
    "model_uri": "runs:/abcd1234/model"
}
```

### Experiment Management

#### List Experiments

Get all MLflow experiments.

```http
GET /experiments
```

Response:
```json
[
    {
        "experiment_id": "1",
        "name": "soccer_prediction_v1",
        "artifact_location": "mlruns/1",
        "lifecycle_stage": "active"
    }
]
```

#### List Runs

Get all runs for an experiment.

```http
GET /runs/{experiment_id}
```

Response:
```json
[
    {
        "run_id": "abcd1234",
        "status": "COMPLETED",
        "start_time": "2024-01-31T22:12:34.567Z",
        "end_time": "2024-01-31T22:30:12.345Z",
        "metrics": {
            "precision": 0.82,
            "recall": 0.76,
            "f1": 0.79
        }
    }
]
```

## Error Monitoring

### Get Active Errors

Get all active errors above a severity threshold.

```http
GET /errors/active?min_severity=WARNING
```

Response:
```json
[
    {
        "error_id": "ef789012",
        "error_type": "DataValidationError",
        "message": "Missing required features",
        "component": "data_ingestion",
        "severity": "ERROR",
        "timestamp": "2024-01-31T22:14:23.456Z",
        "recovery_suggestion": "Check input data completeness"
    }
]
```

### Get Error Patterns

Get error patterns for analysis.

```http
GET /errors/patterns?component=training&min_count=2
```

Response:
```json
[
    {
        "component": "training",
        "error_type": "ConvergenceWarning",
        "count": 3
    }
]
```

## Usage Examples

### Complete Training Workflow

1. Start a training run:
```bash
curl -X POST http://localhost:8000/train \
    -H "Content-Type: application/json" \
    -d '{
        "model_type": "xgboost",
        "experiment_name": "soccer_prediction_v1",
        "hyperparameters": {
            "learning_rate": 0.02,
            "max_depth": 5
        }
    }'
```

2. Monitor training status:
```bash
curl http://localhost:8000/status/abcd1234
```

3. Make predictions with trained model:
```bash
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{
        "model_uri": "runs:/abcd1234/model",
        "data": {
            "feature1": [1.0, 2.0, 3.0]
        }
    }'
```

## Implementation Notes

1. **CPU-Only Training**: All model training is configured for CPU-only operation:
   - XGBoost uses `tree_method='hist'`
   - CatBoost uses `task_type='CPU'`

2. **Logging**: All operations are logged using structured JSON format with:
   - ISO 8601 timestamps
   - Component identification
   - Error tracking
   - MLflow integration

3. **State Management**:
   - Training context is persisted to disk
   - Error history is maintained
   - MLflow tracks all experiments and runs

4. **Error Handling**:
   - Comprehensive error tracking
   - Pattern detection
   - Recovery suggestions
   - Severity-based filtering

## Future Enhancements

1. Authentication and authorization
2. Rate limiting
3. Batch prediction endpoints
4. Real-time training metrics via WebSocket
5. Model versioning and deployment
6. A/B testing support

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html)
- [Project Documentation](../docs/plan.md)
- [Error Handling Guide](../docs/error_handling.md) 