# System Monitoring Guide

## Overview
This guide outlines the monitoring system implemented in the Soccer Prediction Project v2.1. It covers logging, error tracking, and performance monitoring.

## Logging System

### Configuration
```python
from utils.logger import ExperimentLogger

logger = ExperimentLogger(
    experiment_name="soccer_prediction",
    log_dir="logs/soccer_prediction"
)
```

### Log Levels
1. **INFO**: General operation progress
   ```python
   logger.info("Loading training data", extra={"file": "data.xlsx"})
   ```

2. **WARNING**: Non-critical issues
   ```python
   logger.warning("Column conversion failed", error_code="E005")
   ```

3. **ERROR**: Critical issues
   ```python
   logger.error("Database connection failed", error_code="E301")
   ```

### Log File Management
- Maximum file size: 10MB
- Rotation: 5 backup files
- Format: JSON structured logging
- Timestamp: ISO 8601 format

## Error Tracking

### Error Categories
1. File Operations (E001-E099)
   - File not found
   - Permission errors
   - Corruption issues

2. Data Validation (E101-E199)
   - Missing columns
   - Invalid data types
   - Conversion failures

3. Processing Errors (E201-E299)
   - Empty datasets
   - Insufficient samples
   - Feature creation failures

4. External Services (E301-E399)
   - MongoDB connection issues
   - MLflow errors
   - API failures

### Error Monitoring Dashboard
```python
from utils.monitoring import ErrorDashboard

dashboard = ErrorDashboard()
dashboard.display_error_rates()
dashboard.show_error_distribution()
```

### Alert System
```python
from utils.alerts import AlertManager

alerts = AlertManager(
    threshold=5,
    window_minutes=15
)
alerts.monitor_error_rates()
```

## Performance Monitoring

### Metrics
1. Data Processing
   - Load times
   - Conversion rates
   - Memory usage

2. Model Performance
   - Training time
   - Prediction latency
   - Resource utilization

3. Database Operations
   - Query times
   - Connection pool status
   - Cache hit rates

### Monitoring Tools
```python
from utils.monitoring import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.track_metrics()
monitor.generate_report()
```

### Model Performance Monitoring

#### Precision-Recall Tracking
```python
def track_model_metrics(predictions: np.ndarray, actuals: np.ndarray):
    """Track model performance metrics."""
    metrics = {
        'precision': precision_score(actuals, predictions),
        'recall': recall_score(actuals, predictions),
        'f1': f1_score(actuals, predictions),
        'recall_threshold_met': recall_score(actuals, predictions) >= 0.40
    }
    
    # Log metrics
    logger.info("Model performance metrics:", extra=metrics)
    
    # Alert if recall drops below threshold
    if not metrics['recall_threshold_met']:
        alerts.send_alert(
            "Model Recall Below Threshold",
            f"Current recall: {metrics['recall']:.2f}"
        )
    
    return metrics
```

#### Visualization
```python
def plot_precision_recall_trend():
    """Plot precision-recall trends over time."""
    metrics_df = load_metrics_history()
    
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df['timestamp'], metrics_df['precision'], label='Precision')
    plt.plot(metrics_df['timestamp'], metrics_df['recall'], label='Recall')
    plt.axhline(y=0.40, color='r', linestyle='--', label='Recall Threshold')
    plt.legend()
    plt.title('Precision-Recall Trends')
    
    return plt.gcf()
```

## Recovery Procedures

### Automated Recovery
1. File Operations
   ```python
   @retry_on_error(max_retries=3, delay=1.0)
   def load_data():
       return pd.read_excel(file_path)
   ```

2. Database Operations
   ```python
   @retry_on_error(max_retries=3, delay=2.0)
   def query_mongodb():
       return collection.find(query)
   ```

### Manual Recovery
1. Data Corruption
   - Restore from backup
   - Rerun data pipeline
   - Validate outputs

2. Service Failures
   - Check service status
   - Restart if necessary
   - Verify connectivity

## Best Practices

### Logging
1. Include context in log messages
2. Use appropriate log levels
3. Add error codes for all errors
4. Include relevant metrics

### Monitoring
1. Set up alerts for critical errors
2. Monitor resource usage
3. Track performance metrics
4. Review logs regularly

### Recovery
1. Implement retry mechanisms
2. Maintain backup procedures
3. Document recovery steps
4. Test recovery processes

## Maintenance

### Daily Tasks
- Check error logs
- Monitor alert status
- Review performance metrics

### Weekly Tasks
- Analyze error patterns
- Review resource usage
- Update alert thresholds

### Monthly Tasks
- Clean up old logs
- Update documentation
- Review recovery procedures

## Troubleshooting

### Common Issues
1. File Access
   - Check permissions
   - Verify paths
   - Validate file format

2. Data Processing
   - Validate input data
   - Check memory usage
   - Review error logs

3. External Services
   - Check connectivity
   - Verify credentials
   - Monitor timeouts

### Support
For issues and support:
1. Check error logs
2. Review documentation
3. Contact system admin
4. Submit bug report 