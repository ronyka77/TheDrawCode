# Data Validation Rules Guide

## Overview
This guide outlines the data validation rules and procedures implemented in the Soccer Prediction Project v2.1. It ensures data quality and consistency across all operations.

## Data Quality Standards

### 1. Required Columns
```python
REQUIRED_COLUMNS = {
    'match_outcome': int,
    'home_team': str,
    'away_team': str,
    'date': 'datetime64[ns]',
    'league': str,
    'fixture_id': int
}

def validate_columns(df: pd.DataFrame) -> bool:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        logger.error(f"Missing columns: {missing}", error_code="E101")
        return False
    return True
```

### 2. Data Types
```python
def validate_types(df: pd.DataFrame) -> bool:
    for col, dtype in REQUIRED_COLUMNS.items():
        if df[col].dtype != dtype:
            logger.error(
                f"Invalid type for {col}: {df[col].dtype}",
                error_code="E102"
            )
            return False
    return True
```

### 3. Value Ranges
```python
VALUE_RANGES = {
    'match_outcome': (0, 2),
    'home_goals': (0, 20),
    'away_goals': (0, 20),
    'possession': (0, 100),
    'shots_on_target': (0, 50)
}

def validate_ranges(df: pd.DataFrame) -> bool:
    for col, (min_val, max_val) in VALUE_RANGES.items():
        if col in df.columns:
            if not df[col].between(min_val, max_val).all():
                logger.error(
                    f"Values out of range for {col}",
                    error_code="E103"
                )
                return False
    return True
```

## Data Validation Procedures

### 1. Input Data Validation
```python
def validate_input_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean input data.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: Validated and cleaned DataFrame
        
    Raises:
        ValueError: If validation fails
    """
    # Check required columns
    if not validate_columns(df):
        raise ValueError("Missing required columns")
        
    # Convert data types
    df = convert_types(df)
    
    # Validate ranges
    if not validate_ranges(df):
        raise ValueError("Values out of range")
        
    return df
```

### 2. Feature Validation
```python
def validate_features(df: pd.DataFrame) -> bool:
    """Validate engineered features.
    
    Args:
        df (pd.DataFrame): DataFrame with features
        
    Returns:
        bool: True if validation passes
    """
    # Check feature completeness
    missing_rate = df.isnull().mean()
    if (missing_rate > 0.1).any():
        logger.error("High missing rate in features", error_code="E201")
        return False
        
    # Check feature correlations
    corr_matrix = df.corr()
    if (corr_matrix.abs() > 0.95).any().any():
        logger.warning("High feature correlations detected", error_code="E202")
        
    return True
```

### 3. Output Validation
```python
def validate_predictions(predictions: np.ndarray) -> bool:
    """Validate model predictions.
    
    Args:
        predictions (np.ndarray): Model predictions
        
    Returns:
        bool: True if validation passes
    """
    # Check prediction ranges
    if not np.all((predictions >= 0) & (predictions <= 1)):
        logger.error("Invalid prediction values", error_code="E301")
        return False
        
    # Check prediction distribution
    if len(predictions) > 0:
        mean_pred = predictions.mean()
        if mean_pred < 0.1 or mean_pred > 0.9:
            logger.warning("Unusual prediction distribution", error_code="E302")
            
    return True
```

### 3. Model Performance Validation
```python
def validate_model_performance(predictions: np.ndarray, actuals: np.ndarray) -> bool:
    """Validate model performance meets precision-recall requirements."""
    recall = recall_score(actuals, predictions)
    if recall < 0.40:
        logger.error(
            "Model recall below threshold: {:.2f} < 0.40",
            error_code="E303"
        )
        return False
        
    precision = precision_score(actuals, predictions)
    logger.info(f"Model metrics - Precision: {precision:.2f}, Recall: {recall:.2f}")
    return True
```

## Validation Rules

### 1. Data Completeness
- Maximum missing values: 10% per column
- Required columns must be present
- No empty DataFrames allowed

### 2. Data Consistency
- Date format: YYYY-MM-DD
- Team names: Standardized format
- League names: From predefined list
- Score format: "X-Y" (e.g., "2-1")

### 3. Data Quality
- No duplicate fixture IDs
- No future dates
- Realistic score ranges
- Valid categorical values

## Validation Workflows

### 1. Training Data
```python
def validate_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate training data workflow."""
    # Basic validation
    df = validate_input_data(df)
    
    # Feature validation
    if not validate_features(df):
        raise ValueError("Feature validation failed")
        
    # Additional checks
    check_class_balance(df)
    check_feature_importance(df)
    
    return df
```

### 2. Prediction Data
```python
def validate_prediction_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate prediction data workflow."""
    # Basic validation
    df = validate_input_data(df)
    
    # Feature validation
    if not validate_features(df):
        raise ValueError("Feature validation failed")
        
    # Check required features
    check_model_features(df)
    
    return df
```

### 3. Results Data
```python
def validate_results(df: pd.DataFrame) -> pd.DataFrame:
    """Validate results data workflow."""
    # Basic validation
    df = validate_input_data(df)
    
    # Check score format
    if not validate_score_format(df):
        raise ValueError("Invalid score format")
        
    # Verify match outcomes
    verify_match_outcomes(df)
    
    return df
```

## Error Handling

### 1. Validation Errors
```python
def handle_validation_error(error: Exception, data: pd.DataFrame) -> None:
    """Handle validation errors."""
    # Log error
    logger.error(f"Validation error: {str(error)}", error_code="E401")
    
    # Save problematic data
    save_error_data(data)
    
    # Notify if critical
    if is_critical_error(error):
        send_alert("Critical validation error")
```

### 2. Recovery Procedures
```python
def recover_from_validation_error(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to recover from validation errors."""
    # Try fixing data types
    df = fix_data_types(df)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Validate again
    if not validate_input_data(df):
        raise ValueError("Recovery failed")
        
    return df
```

## Monitoring and Reporting

### 1. Validation Metrics
```python
def track_validation_metrics(df: pd.DataFrame) -> None:
    """Track data validation metrics."""
    metrics = {
        'missing_rate': df.isnull().mean().mean(),
        'invalid_types': count_invalid_types(df),
        'out_of_range': count_range_violations(df)
    }
    log_metrics(metrics)
```

### 2. Regular Checks
```python
def schedule_validation_checks():
    """Schedule regular validation checks."""
    # Daily data quality check
    schedule.daily(validate_all_data)
    
    # Weekly comprehensive check
    schedule.weekly(deep_validation)
    
    # Monthly report
    schedule.monthly(generate_validation_report)
```

## Best Practices

### 1. Data Validation
- Validate early in the pipeline
- Log all validation failures
- Keep validation rules updated
- Monitor validation metrics

### 2. Error Handling
- Implement graceful degradation
- Provide clear error messages
- Document recovery procedures
- Track error patterns

### 3. Maintenance
- Review validation rules quarterly
- Update range checks as needed
- Monitor validation performance
- Document all changes 