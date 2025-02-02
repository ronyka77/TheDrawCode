# Feature Selection Guide

## Overview

The feature selection process in the Soccer Prediction Project uses XGBoost's importance metrics to identify the most relevant features for prediction. This guide explains the implementation, configuration, and usage of the feature selection system.

## Table of Contents
- [Core Components](#core-components)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)
- [Logging and Monitoring](#logging-and-monitoring)
- [Best Practices](#best-practices)
- [Windows 11 Considerations](#windows-11-considerations)
- [Troubleshooting](#troubleshooting)
- [Precision-Focused Feature Selection](#precision-focused-feature-selection)

## Core Components

### XGBoostFeatureSelector

The main feature selection class that implements:
- Multiple importance metrics (gain, weight, cover)
- Composite scoring system
- CPU-optimized training
- Detailed logging and metrics tracking

```python
from utils.feature_selector_api import XGBoostFeatureSelector

selector = XGBoostFeatureSelector(
    importance_threshold=0.001,  # Minimum importance score
    min_features=45             # Minimum features to retain
)
```

## Configuration

### CPU Optimization

The feature selector is specifically optimized for CPU-only environments:

```python
model_config = {
    'tree_method': 'hist',     # CPU-optimized algorithm
    'device': 'cpu',           # Force CPU usage
    'nthread': -1              # Use all CPU cores
}
```

### Importance Metrics

Features are scored using a weighted combination of three metrics:
- Gain (50%): Measures improvement in accuracy
- Weight (30%): Frequency of feature usage
- Cover (20%): Number of samples affected

## Usage Guide

### Basic Usage

```python
# Load your data
X_train, y_train, X_test, y_test = load_data_without_selected_columns_api(data_path)

# Initialize selector
selector = XGBoostFeatureSelector(min_features=45)

# Select features
selected_features = selector.select_features(X_train, y_train, X_test, y_test)
```

### Data Preprocessing

The system automatically handles:
- Missing value imputation
- Infinite value replacement
- Numeric type conversion
- Non-numeric column removal

## Logging and Monitoring

### Logging System

The feature selector uses the project's `ExperimentLogger` for comprehensive logging:
- Initialization parameters
- Training progress
- Feature importance scores
- Selection results

### Log Output Example

```
Feature Selection Results:
--------------------------------------------------------------------------------
Total features analyzed: 120
Features with non-zero importance: 85
Features selected: 45

Top Selected Features:
--------------------------------------------------------------------------------
1. league_home_draw_rate                         0.1009
2. home_draw_rate                                0.0173
...
```

## Best Practices

1. Data Preparation
   - Handle missing values
   - Convert data types appropriately
   - Remove constant/duplicate columns

2. Parameter Tuning
   - Adjust `n_bootstrap` based on dataset size
   - Set `correlation_threshold` based on domain knowledge
   - Define `target_features` range appropriately

3. Performance Monitoring
   - Review stability scores
   - Check correlation groups
   - Monitor elimination metrics
   - Validate final feature set

## Error Handling

The system includes comprehensive error handling:
- Data validation
- Parameter verification
- Process monitoring
- Detailed logging

## Maintenance

Regular maintenance tasks:
1. Review MLflow experiments
2. Clean up temporary files
3. Update feature selection criteria
4. Validate against new data

## Windows 11 Considerations

### System Configuration

1. **Memory Management**
   - Enable virtual memory optimization
   - Set minimum pagefile size to match RAM size
   - Keep at least 20% free disk space on system drive

2. **CPU Settings**
   - Use Windows 11 Power Mode: "Best Performance"
   - Disable CPU throttling in BIOS
   - Update Windows power plan settings:
     ```powershell
     powercfg /setactive scheme_min
     ```

3. **Process Priority**
   - Run Python with high priority for feature selection:
     ```powershell
     Start-Process python -ArgumentList "feature_selector_api.py" -WindowStyle Normal -Priority High
     ```

### Performance Optimization

1. **Disk I/O**
   - Use SSD for data storage
   - Disable Windows indexing for data directories
   - Keep data files defragmented

2. **Memory Usage**
   - Close unnecessary applications
   - Monitor memory usage with Task Manager
   - Use smaller data chunks if needed

3. **Antivirus Considerations**
   - Add project directory to exclusion list
   - Exclude Python processes from real-time scanning
   - Schedule scans during non-training periods

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce batch size
   - Use data sampling for initial selection
   - Enable histogram-based training

2. **Performance Issues**
   - Check CPU utilization
   - Verify tree_method setting
   - Monitor system resources

3. **Feature Quality**
   - Validate feature distributions
   - Check for data leakage
   - Monitor correlation between features

## Related Documentation

- [Environment Setup Guide](environment.md)
- [MLflow Integration Guide](mlflow.md)
- [Troubleshooting Guide](troubleshooting.md)

## Version History

- **2.1.0** (Current)
  - Added CPU optimization
  - Enhanced logging system
  - Improved feature importance calculation
  - Added Windows 11 specific optimizations

- **2.0.0**
  - Initial feature selection implementation
  - Basic importance metrics
  - Simple logging 

## Output and Results

The system generates several outputs:

1. Selected Features List
   - Saved to `results/feature_selection/selected_features.txt`
   - Contains final selected feature names

2. MLflow Metrics
   - Feature importance scores
   - Selection stability metrics
   - Cross-validation scores
   - Performance metrics

3. Model Configuration
   - Saved as `model_config.json`
   - Contains:
     - Selected features
     - Feature importance scores
     - Stability scores

## MLflow Integration

The system automatically logs:
- Optimization parameters
- Feature importance scores
- Selection metrics
- Performance metrics
- Model configuration

Access MLflow UI to view:
```bash
mlflow ui --backend-store-uri file:///path/to/mlruns
```

## Precision-Focused Feature Selection

### Two-Stage Process
1. Standard Feature Selection
   - Uses composite scoring (gain, weight, cover)
   - Correlation-based redundancy reduction
   - Bootstrap stability analysis

2. Precision Impact Analysis
   - Measures feature impact on precision
   - Maintains minimum recall threshold
   - Optimizes precision-recall trade-off

### Configuration
```python
# Standard selection
standard_selector = EnhancedFeatureSelector(
    correlation_threshold=0.85,
    target_features=(60, 100)
)

# Precision focus
precision_selector = PrecisionFocusedFeatureSelector(
    min_recall=0.20,
    target_precision=0.50
)
```

### Metrics and Evaluation
- Feature importance scores
- Precision impact analysis
- Correlation analysis
- Performance comparison
- Stability metrics 