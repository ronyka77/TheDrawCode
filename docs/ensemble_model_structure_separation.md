# Ensemble Model Refactoring Implementation Plan v1.0

This document outlines the comprehensive plan to refactor the `ensemble_model_stacked_improved.py` file into a modular package structure with separate files for each functional component.

## Table of Contents
1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Implementation Steps](#implementation-steps)
4. [Migration Plan by Component](#migration-plan-by-component)
5. [Import Updates](#import-updates)
6. [Testing Strategy](#testing-strategy)
7. [Rollback Plan](#rollback-plan)

<a name="overview"></a>
## 1. Overview

The current implementation has over 1500 lines in a single file (`ensemble_model_stacked_improved.py`), making it difficult to maintain and extend. This refactoring will split the code into logical modules with clear separation of concerns, improving maintainability and extensibility.

**Benefits:**
- Improved code organization and readability
- Better separation of concerns
- Easier unit testing
- Simplified maintenance and future development
- Clearer code ownership

<a name="directory-structure"></a>
## 2. Directory Structure

```
models/
└── ensemble/
    ├── __init__.py                # Package initialization and version
    ├── ensemble_model.py          # Main EnsembleModel class definition
    ├── calibration.py             # Calibration utilities
    ├── data_utils.py              # Data preparation utilities
    ├── meta_features.py           # Meta-feature creation
    ├── diagnostics.py             # Analysis and explanation methods
    ├── evaluation.py              # Metrics and evaluation
    ├── training.py                # Training logic
    ├── weights.py                 # Dynamic weight calculation
    ├── thresholds.py              # Threshold optimization
    └── run_ensemble.py            # Main execution script
```

<a name="implementation-steps"></a>
## 3. Implementation Steps

### Phase 1: Setup Package Structure
1. Create the `models/ensemble` directory
2. Create empty files for each module
3. Create `__init__.py` with version info and imports

### Phase 2: Extract Core Components
1. Move EnsembleModel class definition to `ensemble_model.py`
2. Extract model initialization and basic methods
3. Update imports and references

### Phase 3: Extract Functional Components
1. Extract specialized methods to appropriate modules
2. Update method calls to use imported functions
3. Handle dependencies between modules

### Phase 4: Create Main Script
1. Move `if __name__ == "__main__"` block to `run_ensemble.py`
2. Update imports and execution flow

### Phase 5: Testing and Validation
1. Verify functionality with test suite
2. Check for regressions in performance
3. Verify MLflow logging still works correctly

<a name="migration-plan-by-component"></a>
## 4. Migration Plan by Component

### 4.1. `__init__.py`

```python
"""
Ensemble Model Package

This package implements a stacked ensemble model for soccer prediction
with dynamic weighting, calibration, and threshold optimization.
"""

__version__ = "2.1.0"

# Core class
from .ensemble_model import EnsembleModel

# Utility functions (optional exports)
from .calibration import calibrate_model, analyze_calibration
from .meta_features import create_meta_features
from .evaluation import evaluate_model
from .thresholds import tune_threshold
from .weights import compute_dynamic_weights
from .diagnostics import explain_predictions, analyze_prediction_errors

# Main execution
from .run_ensemble import run_ensemble
```

### 4.2. `ensemble_model.py`

```python
"""
EnsembleModel Class

Core implementation of the ensemble model with initialization methods
and high-level APIs.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Dict, List, Tuple, Optional, Union

# Local imports
from utils.logger import ExperimentLogger
from utils.create_evaluation_set import import_selected_features_ensemble

# Module imports
from .calibration import calibrate_models, analyze_calibration
from .data_utils import prepare_data, apply_adasyn_resampling
from .meta_features import create_meta_features
from .diagnostics import detect_data_leakage, explain_predictions, analyze_prediction_errors
from .evaluation import evaluate_model
from .training import initialize_meta_learner
from .weights import compute_dynamic_weights
from .thresholds import tune_threshold_for_precision

class EnsembleModel(BaseEstimator, ClassifierMixin):
    """
    EnsembleModel trains a soft voting ensemble (XGBoost, CatBoost, LGBM) combined with stacking
    via a meta-learner.
    """
    def __init__(self, logger=None, calibrate=False, 
                 calibration_method="sigmoid", individual_thresholding=False,
                 meta_learner_type='xgb', dynamic_weighting=True,
                 extra_base_model_type='random_forest', 
                 sampling_strategy=0.7,
                 complexity_penalty=0.01,
                 target_precision=0.60,
                 required_recall=0.40):
        # Initialize code here...
        # Keep initialization of base models and parameters
        pass
    
    def train(self, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None, 
             split_validation=True, val_size=0.2):
        # Training workflow implementation
        # This should orchestrate the training process using imported functions
        pass
    
    def predict_proba(self, X):
        # Probability prediction implementation
        # This should use imported functions for meta-feature creation
        pass
    
    def predict(self, X):
        # Binary prediction implementation
        pass
    
    # Other methods that should remain in the core class...
```

### 4.3. `calibration.py`

```python
"""
Model Calibration Utilities

Functions for calibrating model probabilities and analyzing calibration performance.
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
import mlflow
from typing import Dict, Tuple, List, Optional

from utils.logger import ExperimentLogger

def calibrate_models(self, X_train, y_train, X_test, y_test, 
                     calibration_method="sigmoid", 
                     logger=None):
    """
    Calibrate all base models' probabilities using isotonic regression or Platt scaling.
    
    Args:
        X_train: Training features for calibration
        y_train: Training labels for calibration
        X_test: Test features for calibration
        y_test: Test labels for calibration
        calibration_method: Method to use for calibration ("sigmoid" or "isotonic")
        logger: Logger instance
        
    Returns:
        Dictionary of calibrated models and calibration results
    """
    # Implementation from _calibrate_models method
    pass

def analyze_calibration(calibration_results, y_test, logger=None):
    """
    Analyze the effectiveness of calibration with adaptive binning based on data distribution.
    
    Args:
        calibration_results: Dictionary with uncalibrated and calibrated probabilities per model
        y_test: True labels for test data
        logger: Logger instance
        
    Returns:
        Dictionary with calibration analysis results
    """
    # Implementation from _analyze_calibration method
    pass

# Additional calibration utility functions...
```

### 4.4. `data_utils.py`

```python
"""
Data Preparation Utilities

Functions for preparing and transforming data for the ensemble model.
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict

from utils.logger import ExperimentLogger

def prepare_data(X, selected_features):
    """Ensure X contains the required features and fill missing values."""
    # Implementation from _prepare_data method
    pass

def apply_adasyn_resampling(X_train, y_train, sampling_strategy=0.7, logger=None):
    """
    Apply ADASYN (Adaptive Synthetic Sampling) to handle class imbalance.
    """
    # Implementation from _apply_adasyn_resampling method
    pass

def select_features_by_importance(X, y, importance_threshold=0.01, n_iterations=5, logger=None):
    """
    Select features based on importance scores from an XGBoost model.
    """
    # Implementation from select_features_by_importance method
    pass

# Additional data utility functions...
```

### 4.5. `meta_features.py`

```python
"""
Meta-Feature Creation

Functions for creating meta-features for the stacked ensemble.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

def create_meta_features(p_xgb, p_cat, p_lgb, p_extra, dynamic_weights=None):
    """
    Create meta-features for the meta-learner by combining base model predictions.
    
    Args:
        p_xgb: XGBoost predicted probabilities
        p_cat: CatBoost predicted probabilities
        p_lgb: LightGBM predicted probabilities
        p_extra: Extra model predicted probabilities
        dynamic_weights: Optional dictionary with weights for each model
        
    Returns:
        Meta-features for the meta-learner
    """
    # Implementation from _create_meta_features method
    pass

# Additional meta-feature utility functions...
```

### 4.6. `diagnostics.py`

```python
"""
Model Diagnostics Utilities

Functions for diagnosing and explaining model predictions.
"""

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import confusion_matrix
import mlflow
from typing import Dict, List, Tuple, Optional

from utils.logger import ExperimentLogger

def detect_data_leakage(X_train, X_test, X_val, logger=None):
    """
    Check for potential data leakage between datasets by detecting duplicate rows.
    """
    # Implementation from detect_data_leakage method
    pass

def explain_predictions(model, X_val, logger=None):
    """
    Generate feature importance explanations using SHAP values on validation data.
    """
    # Implementation from explain_predictions method
    pass

def analyze_prediction_errors(model, X_val, y_val, threshold=None, logger=None):
    """
    Analyze prediction errors on the validation set (most recent data).
    """
    # Implementation from analyze_prediction_errors method
    pass

# Additional diagnostic utility functions...
```

### 4.7. `evaluation.py`

```python
"""
Model Evaluation Utilities

Functions for evaluating model performance.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import mlflow
from typing import Dict, List, Tuple, Optional

from utils.logger import ExperimentLogger

def evaluate_model(model, X_val, y_val, threshold=None, logger=None):
    """
    Evaluate model performance on validation data (most recent data).
    
    Args:
        model: Trained model with predict_proba method
        X_val: Validation features
        y_val: Validation target values
        threshold: Classification threshold (default: 0.5)
        logger: Logger instance
        
    Returns:
        Dictionary with performance metrics and confidence intervals
    """
    # Implementation from evaluate method
    pass

def cross_validate(model_class, X, y, n_splits=5, logger=None, **model_params):
    """
    Perform stratified k-fold cross-validation.
    """
    # Implementation from cross_validate method
    pass

# Additional evaluation utility functions...
```

### 4.8. `training.py`

```python
"""
Model Training Utilities

Functions for training the ensemble models.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from typing import Dict, List, Tuple, Optional

from utils.logger import ExperimentLogger

def initialize_meta_learner(meta_learner_type='xgb', logger=None):
    """
    Initialize the meta learner based on the provided meta_learner_type.
    
    Args:
        meta_learner_type: Type of meta-learner ('xgb', 'logistic', 'mlp')
        logger: Logger instance
        
    Returns:
        Initialized meta-learner model
    """
    # Implementation from _initialize_meta_learner method
    pass

# Additional training utility functions...
```

### 4.9. `weights.py`

```python
"""
Dynamic Weight Calculation

Functions for calculating dynamic weights for ensemble models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score
from typing import Dict, List, Tuple, Optional

from utils.logger import ExperimentLogger

def compute_dynamic_weights(p_xgb, p_cat, p_lgb, p_extra, targets, logger=None):
    """
    Compute dynamic weights for each base model based on their precision on the validation set.
    
    Args:
        p_xgb: XGBoost predicted probabilities
        p_cat: CatBoost predicted probabilities
        p_lgb: LightGBM predicted probabilities
        p_extra: Extra model predicted probabilities
        targets: True labels
        logger: Logger instance
        
    Returns:
        Dictionary with normalized weights for each model
    """
    # Implementation from _compute_dynamic_weights method
    pass

# Additional weight utility functions...
```

### 4.10. `thresholds.py`

```python
"""
Threshold Optimization

Functions for optimizing classification thresholds.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score
import mlflow
from typing import Dict, List, Tuple, Optional

from utils.logger import ExperimentLogger

def tune_threshold(probs, targets, grid_start=0.0, grid_stop=1.0, grid_step=0.01,
                   target_precision=0.50):
    """
    Tune the global threshold by scanning a grid.
    """
    # Implementation from _tune_threshold method
    pass

def tune_threshold_for_precision(y_prob, y_true, target_precision=0.60, 
                                 required_recall=0.40, min_threshold=0.1,
                                 max_threshold=0.9, step=0.01, logger=None):
    """
    Tune the threshold to achieve a target precision with a minimum recall requirement.
    """
    # Implementation from _tune_threshold_for_precision method
    pass

def tune_individual_threshold(probs, targets, grid_start=0.4, grid_stop=0.7, 
                              grid_step=0.01, min_recall=0.50):
    """
    Tune threshold for a single model's probabilities.
    """
    # Implementation from _tune_individual_threshold method
    pass

# Additional threshold utility functions...
```

### 4.11. `run_ensemble.py`

```python
"""
Ensemble Model Runner

Main script for running the ensemble model training and evaluation.
"""

import os
import sys
import numpy as np
import pandas as pd
import mlflow
from pathlib import Path

from utils.logger import ExperimentLogger
from utils.create_evaluation_set import setup_mlflow_tracking, import_selected_features_ensemble

from .ensemble_model import EnsembleModel

def run_ensemble():
    """
    Main function to run the ensemble model training and evaluation.
    """
    # Implementation based on the if __name__ == "__main__" block
    pass

if __name__ == "__main__":
    run_ensemble()
```

<a name="import-updates"></a>
## 5. Import Updates

### 5.1. Update Import Strategy

Each module should:
1. Import only what it needs
2. Use relative imports for other ensemble modules
3. Use standard imports for external libraries
4. Import logger and other utilities consistently

### 5.2. Circular Import Prevention

To avoid circular imports:
1. Define interfaces clearly between modules
2. Move shared functionality to a common module
3. Use function arguments instead of direct imports where necessary
4. Consider using dependency injection for complex dependencies

<a name="testing-strategy"></a>
## 6. Testing Strategy

### 6.1. Incremental Testing

1. Start with a copy of the original file for reference
2. Migrate one module at a time
3. Run tests after each module migration
4. Verify that behavior and outputs match the original implementation

### 6.2. Test Each Module

1. Create unit tests for each module
2. Test edge cases and error handling
3. Verify all utility functions work as expected
4. Test with different parameter combinations

### 6.3. Integration Testing

1. Test the full workflow end-to-end
2. Verify that MLflow logging works as expected
3. Compare model performance metrics before and after refactoring
4. Test on various dataset sizes and distributions

<a name="rollback-plan"></a>
## 7. Rollback Plan

If issues arise during migration:

1. Keep a backup of the original implementation
2. Maintain a version history in git
3. Create a feature flag to switch between implementations
4. Document any discovered bugs in the original implementation

---

## Implementation Schedule

| Task | Description | Estimated Time |
|------|-------------|----------------|
| Setup Directory Structure | Create package folders and files | 1 hour |
| Extract EnsembleModel Core | Move core class and initialization | 3 hours |
| Extract Calibration Module | Move calibration-related methods | 2 hours |
| Extract Data Utilities | Move data preparation utilities | 2 hours |
| Extract Meta-Features | Move meta-feature creation | 1 hour |
| Extract Diagnostics | Move diagnostic methods | 2 hours |
| Extract Evaluation | Move evaluation methods | 2 hours |
| Extract Training Methods | Move training utilities | 2 hours |
| Extract Weights Module | Move weight calculation | 1 hour |
| Extract Thresholds Module | Move threshold tuning | 1 hour |
| Create Main Script | Move entry point code | 1 hour |
| Testing and Verification | Test full workflow | 4 hours |
| Documentation | Update docs and comments | 2 hours |
| **Total** | | **24 hours** |

## Backward Compatibility

To maintain backward compatibility during migration:
1. Keep the original file working during transition
2. Add deprecation warnings to the original file
3. Create an adapter that mimics the original interface but uses the new modules
4. Update the import signatures in dependent code gradually

## Conclusion

This refactoring plan provides a structured approach to breaking down the monolithic ensemble model into maintainable, reusable components. Following this plan will improve code organization, readability, and maintainability while preserving the functionality of the original implementation. 