# Error Handling Documentation

## Overview
This document outlines the error handling and logging system implemented in the Soccer Prediction Project v2.1.

## Error Codes
The system uses predefined error codes through the `DataProcessingError` class:

```python
class DataProcessingError:
    FILE_NOT_FOUND = "E001"
    FILE_CORRUPTED = "E002"
    EMPTY_DATASET = "E003"
    MISSING_REQUIRED_COLUMNS = "E004"
    NUMERIC_CONVERSION_FAILED = "E005"
    INVALID_DATA_TYPE = "E006"
```

## Logging System
The project uses `ExperimentLogger` for standardized logging across all components:

```python
logger = ExperimentLogger(
    experiment_name="soccer_prediction",
    log_dir="logs/soccer_prediction"
)
```

### Log Levels
- **INFO**: General processing steps and status updates
- **WARNING**: Non-critical issues that don't halt execution
- **ERROR**: Critical issues that prevent normal operation

### Retry Mechanism
The `@retry_on_error` decorator provides automatic retry functionality:
```python
@retry_on_error(max_retries=3, delay=1.0)
def function_name():
    # Function implementation
```

## Common Error Scenarios

### File Operations
```python
try:
    data = pd.read_excel(data_path)
except FileNotFoundError:
    logger.error(
        f"Data file not found: {data_path}",
        error_code=DataProcessingError.FILE_NOT_FOUND
    )
```

### Data Validation
```python
if data.empty:
    logger.error(
        "Loaded dataset is empty",
        error_code=DataProcessingError.EMPTY_DATASET
    )
    raise ValueError("Dataset is empty")
```

### Type Conversion
```python
try:
    data[col] = data[col].astype('int64')
except Exception as e:
    logger.warning(
        f"Failed to convert {col} to integer: {str(e)}",
        error_code=DataProcessingError.NUMERIC_CONVERSION_FAILED
    )
```

## Best Practices
1. Always use error codes from `DataProcessingError`
2. Include context in error messages
3. Use appropriate log levels
4. Implement retry logic for unstable operations
5. Validate data early in the process
6. Handle both expected and unexpected errors

## Example Implementation
See `import_feature_select_draws_api()` in `utils/feature_selection.py` for a complete example of error handling implementation.

### Model Performance Errors
```python
class ModelPerformanceError:
    RECALL_BELOW_THRESHOLD = "E301"
    PRECISION_DEGRADATION = "E302"
    UNSTABLE_PREDICTIONS = "E303"

def handle_performance_error(error_code: str, metrics: Dict[str, float]):
    """Handle model performance errors."""
    if error_code == ModelPerformanceError.RECALL_BELOW_THRESHOLD:
        logger.error(
            f"Recall {metrics['recall']:.2f} below threshold 0.40",
            error_code=error_code
        )
        # Trigger model retraining or threshold adjustment
        adjust_model_threshold() 