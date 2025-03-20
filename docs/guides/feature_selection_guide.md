# Feature Selection System Guide

## Overview

The feature selection system provides a comprehensive approach to selecting the most relevant features for model training. It combines multiple techniques to ensure robust and stable feature selection:

1. Composite Score Optimization
2. Correlation-based Redundancy Reduction
3. Stability Selection via Bootstrapping
4. Iterative Feature Elimination

## Components

### 1. Composite Score Optimization

Optimizes feature importance weights through grid search:
- Gain importance (model performance impact)
- Weight importance (feature usage)
- Cover importance (data coverage)

```python
selector = EnhancedFeatureSelector()
best_weights = selector.optimize_composite_weights(X, y, model)
```

### 2. Correlation Analysis

Identifies and handles highly correlated features:
- Calculates correlation matrix
- Groups correlated features
- Retains most important feature from each group

```python
correlation_groups = selector.analyze_correlations(X, selected_features)
```

### 3. Stability Selection

Ensures feature selection stability through bootstrapping:
- Multiple bootstrap iterations
- Feature selection frequency tracking
- Importance score aggregation
- Visualization of stability metrics

```python
stability_scores = selector.perform_stability_selection(X, y, model, weights)
```

### 4. Iterative Elimination

Progressively removes less important features:
- Cross-validation based evaluation
- Step-wise feature removal
- Performance tracking
- Optimal feature set identification

```python
final_features = selector.perform_iterative_elimination(X, y, model)
```

## Usage

### Basic Usage

```python
from utils.feature_selection import run_feature_selection

# Run complete feature selection process
selected_features = run_feature_selection(
    data_path="data/api_training_final.xlsx",
    experiment_name="feature_selection_optimization"
)
```

### Custom Configuration

```python
from utils.feature_selection import EnhancedFeatureSelector
import xgboost as xgb

# Initialize with custom parameters
selector = EnhancedFeatureSelector(
    n_bootstrap=10,
    correlation_threshold=0.90,
    target_features=(50, 80),
    random_state=42
)

# Initialize model
model = xgb.XGBClassifier(
    tree_method='hist',
    device='cpu'
)

# Run complete selection process
selected_features = selector.select_features(X, y, model)
```

## Output and Visualization

The system generates several outputs:

1. Selected Features List
   - Saved to `results/feature_selection/selected_features.txt`

2. Visualization Plots
   - Correlation Matrix: `results/feature_selection/correlation_matrix.png`
   - Stability Selection: `stability_selection.png`
   - Feature Elimination: `elimination_curve.png`

3. MLflow Tracking
   - Experiment metrics
   - Parameter optimization
   - Feature importance scores
   - Performance plots

## MLflow Integration

The system automatically logs:
- Optimization parameters
- Feature importance scores
- Selection metrics
- Visualization artifacts
- Performance metrics

Access MLflow UI to view:
```bash
mlflow ui --backend-store-uri file:///path/to/mlruns
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
   - Monitor elimination curve
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