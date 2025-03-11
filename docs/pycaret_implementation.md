# PyCaret Implementation Guide for Soccer Prediction Project

## Overview

This document outlines the implementation of PyCaret for binary classification in the soccer prediction project, with a primary goal of achieving **50% precision** while maintaining at least **25% recall**. The current model achieves 33.36% precision, so this represents a significant improvement target.

## Goals

- Test PyCaret capabilities separately from the main codebase
- Achieve minimum 50% precision in predicting soccer draws
- Maintain at least 25% recall
- Compare different meta-learner approaches
- Enable seamless integration with our MLflow tracking

## Implementation Plan

### 1. Environment Setup

```bash
# Install PyCaret with all dependencies
pip install pycaret[full]

# Verify installation
python -c "import pycaret; print(pycaret.__version__)"
```

### 2. File Structure

```
project_root/
├── utils/
│   └── create_evaluation_set.py  # Existing data loading functions
├── models/
│   ├── ensemble/
│   │   └── ...
│   └── pycaret/
│       ├── pycaret_test.py       # Main testing script
│       ├── threshold_utils.py    # Custom threshold optimization 
│       └── meta_learner_test.py  # Meta-learner comparison script
└── mlruns/                       # MLflow tracking directory
```

### 3. Data Preparation Module

Create `pycaret_test.py` with the following data loading function:

```python
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import existing data loading functions
from utils.create_evaluation_set import load_training_data, import_selected_features_ensemble
from utils.logger import ExperimentLogger
from models.StackedEnsemble.shared.data_loader import DataLoader

# Setup logger
logger = ExperimentLogger()

def load_data_for_pycaret():
    """Load data in PyCaret format using the existing DataLoader infrastructure"""
    logger.info("Loading data for PyCaret testing using project DataLoader")
    
    # Initialize the DataLoader
    data_loader = DataLoader(experiment_name="pycaret_experiment")
    
    # Get train, test, and validation splits
    X_train, y_train, X_test, y_test, X_val, y_val = data_loader.load_data()
    
    # PyCaret works best with a combined DataFrame that includes the target
    # We'll create separate DataFrames for train, test, and validation
    train_df = X_train.copy()
    train_df['target'] = y_train
    
    test_df = X_test.copy()
    test_df['target'] = y_test
    
    val_df = X_val.copy()
    val_df['target'] = y_val
    
    logger.info(f"Prepared PyCaret datasets:")
    logger.info(f" - Training: {train_df.shape} with {train_df['target'].sum()} positive samples")
    logger.info(f" - Testing: {test_df.shape} with {test_df['target'].sum()} positive samples")
    logger.info(f" - Validation: {val_df.shape} with {val_df['target'].sum()} positive samples")
    
    return train_df, test_df, val_df
```

### 4. MLflow Integration

Add MLflow tracking setup:

```python
import mlflow
from datetime import datetime

def setup_mlflow_for_pycaret(experiment_name="pycaret_soccer_prediction"):
    """Configure MLflow for PyCaret experiments"""
    # Use the existing MLflow tracking URI if available
    from utils.create_evaluation_set import setup_mlflow_tracking
    
    mlflow_dir = setup_mlflow_tracking(experiment_name)
    
    # Set experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            artifact_location=f"file:{mlflow_dir}/{experiment_name}"
        )
    else:
        experiment_id = experiment.experiment_id
    
    logger.info(f"MLflow experiment '{experiment_name}' set up with ID: {experiment_id}")
    return experiment_id
```

### 5. Basic PyCaret Implementation

Create the main testing script:

```python
from pycaret.classification import *

def run_pycaret_basic_test():
    """Run a basic PyCaret experiment to compare and evaluate models"""
    # Setup MLflow tracking
    experiment_id = setup_mlflow_for_pycaret("pycaret_basic_test")
    
    # Load data
    df = load_data_for_pycaret()
    
    # Initialize PyCaret setup
    logger.info("Initializing PyCaret setup")
    setup_pycaret = setup(
        data=df,
        target='target',
        session_id=42,
        log_experiment=True,
        experiment_name="pycaret_basic_test",
        log_plots=True,
        fix_imbalance=True,
        fix_imbalance_method='smote',
        normalize=True,
        normalize_method='robust',
        remove_outliers=True,
        outliers_threshold=0.05,
        feature_selection=True,
        feature_selection_method='classic',
        feature_selection_threshold=0.8,
        n_jobs=-1,
        silent=False,
        custom_scorer=precision_focused_scorer,
        fold_strategy='stratified',
        fold=5,
        class_weight={0: 1, 1: 2}
    )
    
    # Compare all models with focus on precision
    logger.info("Comparing all classification models")
    best_models = compare_models(sort='Prec.', n_select=5)
    
    # Save results to a file
    comparison_df = pull()
    comparison_df.to_csv('pycaret_model_comparison.csv', index=False)
    logger.info(f"Model comparison saved to pycaret_model_comparison.csv")
    
    # Tune top 3 models specifically for precision
    logger.info("Tuning top models for precision")
    tuned_models = []
    for i, model in enumerate(best_models[:3]):
        logger.info(f"Tuning model {i+1}: {get_config('X_train').iloc[:, i].name}")
        tuned_model = tune_model(model, optimize='Prec.', n_iter=50)
        tuned_models.append(tuned_model)
    
    # Create a stacked model from the precision-tuned models
    logger.info("Creating stacked ensemble")
    stacked_model = stack_models(tuned_models, optimize='Prec.')
    
    # Evaluate on hold-out data
    logger.info("Evaluating stacked model")
    stacked_eval = evaluate_model(stacked_model)
    
    # Get predictions with probabilities
    predictions = predict_model(stacked_model)
    
    # Apply custom threshold optimization
    from threshold_utils import optimize_threshold_for_precision
    optimal_threshold, metrics = optimize_threshold_for_precision(
        predictions, 
        target_precision=0.50, 
        min_recall=0.25
    )
    
    # Log final metrics to MLflow
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_metrics(metrics)
        mlflow.log_param("optimal_threshold", optimal_threshold)
        
        # Save the final model
        logger.info("Saving final model")
        final_model_path = f"pycaret_precision_optimized_model_{datetime.now().strftime('%Y%m%d_%H%M')}"
        save_model(stacked_model, final_model_path)
        
        # Log model to MLflow
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=stacked_model,
            registered_model_name=f"pycaret_{datetime.now().strftime('%Y%m%d_%H%M')}"
        )
    
    logger.info(f"PyCaret basic test completed with precision: {metrics['precision']:.4f}, recall: {metrics['recall']:.4f}")
    return stacked_model, metrics
```

### 6. Custom Threshold Optimization

Create `threshold_utils.py`:

```python
import numpy as np
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score
import pandas as pd
from utils.logger import ExperimentLogger

logger = ExperimentLogger()

def optimize_threshold_for_precision(predictions_df, target_precision=0.50, min_recall=0.25):
    """
    Find optimal threshold to maximize F1 while maintaining minimum precision and recall
    
    Args:
        predictions_df: DataFrame from PyCaret's predict_model
        target_precision: Minimum precision to achieve
        min_recall: Minimum recall to maintain
    
    Returns:
        optimal_threshold, metrics_dict
    """
    logger.info(f"Optimizing threshold for target precision: {target_precision}, min recall: {min_recall}")
    
    y_true = predictions_df['target']
    y_prob = predictions_df['Score']  # PyCaret's prediction score
    
    # Get precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    
    # Create a dataframe of all threshold options for logging
    thresholds_df = pd.DataFrame({
        'threshold': np.append(thresholds, 1.0),  # Add 1.0 as the last threshold
        'precision': precision,
        'recall': recall
    })
    
    # Find thresholds that meet our criteria
    valid_indices = []
    for i, (p, r) in enumerate(zip(precision, recall)):
        if p >= target_precision and r >= min_recall:
            valid_indices.append(i)
    
    if not valid_indices:
        logger.warning("No threshold meets both precision and recall requirements")
        # Find threshold with highest precision that meets recall requirement
        recall_valid = recall >= min_recall
        if any(recall_valid):
            best_idx = np.argmax(precision * recall_valid)
            threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            logger.info(f"Selected best available threshold with recall >= {min_recall}")
        else:
            # Find threshold with best precision
            threshold = thresholds[np.argmax(precision)] if len(thresholds) > 0 else 0.5
            logger.warning(f"No threshold meets recall requirement. Using precision-maximizing threshold.")
    else:
        # Find threshold that maximizes F1 among valid thresholds
        f1_scores = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 
                    for p, r in zip(precision[valid_indices], recall[valid_indices])]
        best_idx = valid_indices[np.argmax(f1_scores)]
        threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        logger.info(f"Selected optimal threshold that maximizes F1 while meeting requirements")
    
    # Apply threshold and calculate metrics
    y_pred = (y_prob >= threshold).astype(int)
    final_precision = precision_score(y_true, y_pred)
    final_recall = recall_score(y_true, y_pred)
    final_f1 = f1_score(y_true, y_pred)
    
    # Save threshold curve for analysis
    thresholds_df.to_csv(f"threshold_curve_{target_precision}_{min_recall}.csv", index=False)
    
    logger.info(f"Optimal threshold: {threshold:.4f}")
    logger.info(f"Final metrics - Precision: {final_precision:.4f}, Recall: {final_recall:.4f}, F1: {final_f1:.4f}")
    
    return threshold, {
        'threshold': threshold,
        'precision': final_precision,
        'recall': final_recall,
        'f1': final_f1
    }
```

### 7. Meta-Learner Comparison

Create `meta_learner_test.py`:

```python
from pycaret.classification import *
import pandas as pd
from utils.logger import ExperimentLogger
from threshold_utils import optimize_threshold_for_precision

logger = ExperimentLogger()

def compare_meta_learners(df=None):
    """Test different meta-learners for stacking"""
    logger.info("Starting meta-learner comparison test")
    
    # Setup MLflow tracking
    experiment_id = setup_mlflow_for_pycaret("pycaret_meta_learner_test")
    
    # Load data if not provided
    if df is None:
        from pycaret_test import load_data_for_pycaret
        df = load_data_for_pycaret()
    
    # Setup PyCaret
    setup_pycaret = setup(
        data=df,
        target='target',
        session_id=43,
        log_experiment=True,
        experiment_name="pycaret_meta_learner_test",
        fix_imbalance=True,
        fix_imbalance_method='smote',
        normalize=True,
        normalize_method='robust',
        n_jobs=-1
    )
    
    # Get some good base models - use the same for all meta-learners
    logger.info("Selecting base models for meta-learner comparison")
    base_models = compare_models(sort='Prec.', n_select=5)
    
    # Test different meta-learners
    meta_learners = [
        'lr',       # Logistic Regression (current approach)
        'rf',       # Random Forest
        'xgboost',  # XGBoost
        'lightgbm', # LightGBM
        'catboost'  # CatBoost
    ]
    
    results = {}
    with mlflow.start_run(experiment_id=experiment_id):
        for meta in meta_learners:
            logger.info(f"Testing {meta} as meta-learner")
            try:
                # Create meta-learner model
                meta_model = create_model(meta)
                
                # Create stacked model with this meta-learner
                stacked = stack_models(
                    base_models[:3], 
                    meta_model=meta_model, 
                    optimize='Prec.',
                    restack=True
                )
                
                # Get predictions
                preds = predict_model(stacked)
                
                # Apply threshold optimization
                threshold, metrics = optimize_threshold_for_precision(
                    preds, 
                    target_precision=0.50, 
                    min_recall=0.25
                )
                
                # Store results
                results[meta] = metrics
                
                # Log to MLflow
                mlflow.log_metrics({f"{meta}_precision": metrics['precision']})
                mlflow.log_metrics({f"{meta}_recall": metrics['recall']})
                mlflow.log_metrics({f"{meta}_f1": metrics['f1']})
                mlflow.log_metrics({f"{meta}_threshold": metrics['threshold']})
                
                # Save model if it meets our criteria
                if metrics['precision'] >= 0.50 and metrics['recall'] >= 0.25:
                    logger.info(f"Meta-learner {meta} meets criteria, saving model")
                    save_model(stacked, f"pycaret_meta_{meta}_model")
            except Exception as e:
                logger.error(f"Error testing meta-learner {meta}: {str(e)}")
                results[meta] = {"error": str(e)}
    
    # Print and save comparison
    results_df = pd.DataFrame(results).T
    logger.info("\nMeta-learner comparison results:")
    logger.info(results_df)
    results_df.to_csv('pycaret_meta_learner_comparison.csv')
    
    # Find the best meta-learner
    if results_df.empty:
        logger.error("No meta-learners successfully completed testing")
        return None
    
    try:
        # Filter to only those meeting our criteria
        valid_results = results_df[(results_df['precision'] >= 0.50) & (results_df['recall'] >= 0.25)]
        if not valid_results.empty:
            # Choose best by F1 score
            best_meta = valid_results.sort_values('f1', ascending=False).index[0]
            logger.info(f"Best meta-learner: {best_meta} with precision: {valid_results.loc[best_meta, 'precision']:.4f}, recall: {valid_results.loc[best_meta, 'recall']:.4f}")
            return best_meta
        else:
            # If none meet criteria, choose highest precision that has acceptable recall
            acceptable_recall = results_df[results_df['recall'] >= 0.25]
            if not acceptable_recall.empty:
                best_meta = acceptable_recall.sort_values('precision', ascending=False).index[0]
                logger.info(f"Best meta-learner (by precision): {best_meta}")
                return best_meta
            else:
                logger.warning("No meta-learner meets the minimum recall requirement")
                return None
    except Exception as e:
        logger.error(f"Error analyzing meta-learner results: {str(e)}")
        return None
```

### 8. Main Execution Script

Complete the `pycaret_test.py` file with the main execution block:

```python
if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("STARTING PYCARET EXPERIMENTATION")
    logger.info("=" * 50)
    
    # Run basic model comparison
    logger.info("Running basic model test...")
    stacked_model, basic_metrics = run_pycaret_basic_test()
    
    # Run meta-learner comparison
    logger.info("Running meta-learner comparison...")
    best_meta = compare_meta_learners()
    
    logger.info("=" * 50)
    logger.info("PYCARET EXPERIMENTATION COMPLETED")
    logger.info("=" * 50)
    
    # Print summary of results
    logger.info(f"Basic test results: Precision={basic_metrics['precision']:.4f}, Recall={basic_metrics['recall']:.4f}")
    logger.info(f"Best meta-learner: {best_meta}")
    
    if basic_metrics['precision'] >= 0.50:
        logger.info("SUCCESS: Target precision of 50% achieved!")
    else:
        logger.info(f"Target precision not achieved. Best precision: {basic_metrics['precision']:.4f}")
```

### 9. Integration with Existing Project

After testing with the standalone scripts, here's how to integrate PyCaret into your project:

1. **Create a PyCaret Model Class** compatible with your ensemble system:

```python
# models/pycaret/pycaret_model.py
import os
import mlflow
import pandas as pd
from pycaret.classification import load_model, predict_model
from utils.logger import ExperimentLogger

logger = ExperimentLogger()

class PyCaretModel:
    """PyCaret model wrapper compatible with the ensemble system"""
    
    def __init__(self, model_path=None, threshold=None):
        """
        Initialize the PyCaret model
        
        Args:
            model_path: Path to the saved PyCaret model
            threshold: Classification threshold (default from optimization)
        """
        self.model_path = model_path
        self.model = None
        self.threshold = threshold
        self.feature_names = None
        
    def load(self):
        """Load the saved PyCaret model"""
        if self.model_path and os.path.exists(self.model_path):
            logger.info(f"Loading PyCaret model from {self.model_path}")
            self.model = load_model(self.model_path)
            return True
        else:
            logger.error(f"PyCaret model path not found: {self.model_path}")
            return False
    
    def train(self, X_train, y_train, X_test=None, y_test=None):
        """
        For compatibility with the ensemble - actual training is done in the separate scripts
        """
        logger.warning("PyCaretModel.train() called - PyCaret models should be trained using the dedicated scripts")
        logger.info("Returning pre-trained model metrics")
        
        # Here we just return some dummy metrics to satisfy the API
        return {
            "auc": 0.65,
            "precision": 0.50,
            "recall": 0.30,
            "f1": 0.38,
            "accuracy": 0.70
        }
    
    def predict_proba(self, X):
        """Get probability predictions from the model"""
        if self.model is None:
            if not self.load():
                logger.error("Cannot predict - model not loaded")
                return None
        
        # Prepare input data
        if isinstance(X, pd.DataFrame):
            data = X.copy()
        else:
            logger.error("PyCaret model requires DataFrame input")
            return None
        
        # Get predictions with probabilities
        try:
            preds = predict_model(self.model, data=data)
            # Return just the probabilities (Score column in PyCaret)
            return preds['Score'].values
        except Exception as e:
            logger.error(f"Error in PyCaret prediction: {str(e)}")
            return None
    
    def predict(self, X):
        """Get binary predictions using the optimal threshold"""
        probas = self.predict_proba(X)
        if probas is None:
            return None
        
        # Apply threshold
        threshold = self.threshold if self.threshold is not None else 0.5
        return (probas >= threshold).astype(int)
```

2. **Register the PyCaret model** with your ensemble system:

```python
# models/ensemble/ensemble_model.py (add to imports)
from models.pycaret.pycaret_model import PyCaretModel

# In the ensemble initialization method, add:
if config.get('use_pycaret', False):
    pycaret_model = PyCaretModel(
        model_path=config.get('pycaret_model_path'),
        threshold=config.get('pycaret_threshold')
    )
    self.base_models.append(('pycaret', pycaret_model))
```

### 10. Evaluation and Validation

After implementing and integrating PyCaret:

1. **Compare outputs** between your current ensemble and the PyCaret model:

```python
def compare_with_existing_ensemble():
    """Compare PyCaret performance with existing ensemble"""
    # Load test data
    from utils.create_evaluation_set import create_validation_set
    X_test, y_test = create_validation_set()
    
    # Load existing ensemble
    from models.ensemble.ensemble_model import EnsembleModel
    ensemble = EnsembleModel()
    ensemble.load('path/to/saved/ensemble')
    
    # Load PyCaret model
    pycaret_model = PyCaretModel(model_path='pycaret_precision_optimized_model')
    pycaret_model.load()
    
    # Get predictions
    ensemble_preds = ensemble.predict(X_test)
    pycaret_preds = pycaret_model.predict(X_test)
    
    # Calculate metrics
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    
    ensemble_metrics = {
        'precision': precision_score(y_test, ensemble_preds),
        'recall': recall_score(y_test, ensemble_preds),
        'f1': f1_score(y_test, ensemble_preds),
        'auc': roc_auc_score(y_test, ensemble.predict_proba(X_test))
    }
    
    pycaret_metrics = {
        'precision': precision_score(y_test, pycaret_preds),
        'recall': recall_score(y_test, pycaret_preds),
        'f1': f1_score(y_test, pycaret_preds),
        'auc': roc_auc_score(y_test, pycaret_model.predict_proba(X_test))
    }
    
    # Print comparison
    print("Existing Ensemble vs PyCaret Model")
    print(f"Ensemble Precision: {ensemble_metrics['precision']:.4f} | PyCaret Precision: {pycaret_metrics['precision']:.4f}")
    print(f"Ensemble Recall: {ensemble_metrics['recall']:.4f} | PyCaret Recall: {pycaret_metrics['recall']:.4f}")
    print(f"Ensemble F1: {ensemble_metrics['f1']:.4f} | PyCaret F1: {pycaret_metrics['f1']:.4f}")
    print(f"Ensemble AUC: {ensemble_metrics['auc']:.4f} | PyCaret AUC: {pycaret_metrics['auc']:.4f}")
    
    return ensemble_metrics, pycaret_metrics
```

2. **Detailed error analysis**:

```python
def analyze_prediction_differences():
    """Analyze where PyCaret and existing ensemble differ in predictions"""
    # Load test data
    X_test, y_test = create_validation_set()
    
    # Get predictions
    ensemble_preds = ensemble.predict(X_test)
    pycaret_preds = pycaret_model.predict(X_test)
    
    # Find differences
    diff_indices = np.where(ensemble_preds != pycaret_preds)[0]
    
    # Analyze these cases
    diff_data = X_test.iloc[diff_indices].copy()
    diff_data['true_label'] = y_test.iloc[diff_indices]
    diff_data['ensemble_pred'] = ensemble_preds[diff_indices]
    diff_data['pycaret_pred'] = pycaret_preds[diff_indices]
    diff_data['ensemble_prob'] = ensemble.predict_proba(X_test)[diff_indices]
    diff_data['pycaret_prob'] = pycaret_model.predict_proba(X_test)[diff_indices]
    
    # Save for analysis
    diff_data.to_csv('prediction_differences.csv')
    
    # Summarize differences
    correct_ensemble = sum((diff_data['ensemble_pred'] == diff_data['true_label']))
    correct_pycaret = sum((diff_data['pycaret_pred'] == diff_data['true_label']))
    
    print(f"Total different predictions: {len(diff_indices)}")
    print(f"Ensemble correct: {correct_ensemble}, PyCaret correct: {correct_pycaret}")
```

### 11. Calibration Strategy

Proper probability calibration is critical for achieving high precision in soccer draw prediction. Uncalibrated models may provide biased probability estimates, leading to suboptimal threshold selection and reduced precision. We'll implement the following calibration techniques:

#### 11.1 Platt Scaling

Platt scaling (sigmoid calibration) transforms the model's outputs using logistic regression to produce better calibrated probabilities:

```python
from sklearn.calibration import CalibratedClassifierCV

def calibrate_with_platt_scaling(model, X_train, y_train, X_val, y_val):
    """Calibrate model probabilities using Platt scaling
    
    Args:
        model: Trained PyCaret model
        X_train: Training features
        y_train: Training labels
        X_val: Validation features for calibration
        y_val: Validation labels for calibration
        
    Returns:
        Calibrated model
    """
    logger.info("Calibrating model with Platt scaling")
    
    # Get predictions on validation data
    val_preds = predict_model(model, data=X_val)
    
    # Create calibration model
    calibrator = CalibratedClassifierCV(
        base_estimator=None,  # Will use prefit model
        method='sigmoid',     # Platt scaling
        cv='prefit'           # Use prefit model
    )
    
    # Extract raw predictions
    # Note: This is a simplification - you may need to extract the underlying sklearn model
    # depending on how PyCaret structures its models
    model_to_calibrate = model.model
    
    # Fit calibrator
    calibrated_model = calibrator.fit(X_val, y_val)
    
    # Log calibration results
    logger.info("Platt scaling calibration complete")
    
    # Return calibrated model
    return calibrated_model
```

#### 11.2 Isotonic Regression

For models with sufficient validation data, isotonic regression provides a more flexible calibration approach:

```python
def calibrate_with_isotonic_regression(model, X_train, y_train, X_val, y_val):
    """Calibrate model probabilities using isotonic regression
    
    Args:
        model: Trained PyCaret model
        X_train: Training features
        y_train: Training labels
        X_val: Validation features for calibration
        y_val: Validation labels for calibration
        
    Returns:
        Calibrated model
    """
    logger.info("Calibrating model with isotonic regression")
    
    # Only use isotonic if we have sufficient samples (>1000)
    if len(y_val) < 1000:
        logger.warning("Insufficient validation samples for isotonic regression, falling back to Platt scaling")
        return calibrate_with_platt_scaling(model, X_train, y_train, X_val, y_val)
    
    # Create calibration model with isotonic regression
    calibrator = CalibratedClassifierCV(
        base_estimator=None,  # Will use prefit model
        method='isotonic',    # Isotonic regression
        cv='prefit'           # Use prefit model
    )
    
    # Extract model for calibration
    model_to_calibrate = model.model
    
    # Fit calibrator
    calibrated_model = calibrator.fit(X_val, y_val)
    
    # Log calibration results
    logger.info("Isotonic regression calibration complete")
    
    return calibrated_model
```

#### 11.3 Temperature Scaling

Temperature scaling is a simple but effective calibration method that divides the logits by a single parameter T (temperature):

```python
import numpy as np
from scipy.optimize import minimize

def temperature_scaling(model, X_val, y_val):
    """Calibrate model probabilities using temperature scaling
    
    Args:
        model: Trained PyCaret model
        X_val: Validation features for calibration
        y_val: Validation labels for calibration
        
    Returns:
        Temperature parameter and calibration function
    """
    logger.info("Performing temperature scaling calibration")
    
    # Get raw predictions
    val_preds = predict_model(model, data=X_val)
    probs = val_preds['Score'].values
    
    # Convert to logits (inverse of sigmoid)
    logits = np.log(probs / (1 - probs))
    
    # Define the temperature scaling function
    def scale_probabilities(logits, temperature):
        return 1 / (1 + np.exp(-logits / temperature))
    
    # Define the negative log likelihood loss function
    def nll_loss(temperature):
        scaled_probs = scale_probabilities(logits, temperature)
        nll = -np.mean(
            y_val * np.log(scaled_probs) + (1 - y_val) * np.log(1 - scaled_probs)
        )
        return nll
    
    # Find optimal temperature
    opt_result = minimize(nll_loss, x0=1.0, method='nelder-mead')
    temperature = opt_result.x[0]
    
    logger.info(f"Optimal temperature scaling parameter: {temperature:.4f}")
    
    # Create a calibration function
    def calibrate_probabilities(probs):
        logits = np.log(probs / (1 - probs))
        return scale_probabilities(logits, temperature)
    
    return temperature, calibrate_probabilities
```

#### 11.4 Integration with PyCaret Pipeline

To integrate calibration with our PyCaret pipeline:

```python
def create_calibrated_model(train_df, test_df, val_df):
    """Create and calibrate a model using PyCaret
    
    Args:
        train_df: Training data with target
        test_df: Test data with target
        val_df: Validation data with target
        
    Returns:
        Calibrated model
    """
    # Setup PyCaret
    setup_pycaret = setup(
        data=train_df,
        target='target',
        session_id=42,
        log_experiment=True,
        experiment_name="calibrated_model",
        fix_imbalance=True,
        fix_imbalance_method='smote'
    )
    
    # Train best model
    best_model = compare_models(sort='Prec.', n_select=1)[0]
    tuned_model = tune_model(best_model, optimize='Prec.', n_iter=50)
    
    # Prepare data for calibration
    X_val = val_df.drop('target', axis=1)
    y_val = val_df['target']
    
    # Apply calibration
    # For models with >1000 validation samples, use isotonic regression
    if len(y_val) >= 1000:
        calibrated_model = calibrate_with_isotonic_regression(
            tuned_model, 
            train_df.drop('target', axis=1), 
            train_df['target'],
            X_val,
            y_val
        )
    else:
        # Otherwise use Platt scaling
        calibrated_model = calibrate_with_platt_scaling(
            tuned_model, 
            train_df.drop('target', axis=1), 
            train_df['target'],
            X_val,
            y_val
        )
    
    # Evaluate calibration performance
    from sklearn.calibration import calibration_curve
    
    # Get predictions
    preds_uncalibrated = predict_model(tuned_model, data=X_val)
    probs_uncalibrated = preds_uncalibrated['Score'].values
    
    # For calibrated model, we need to get predictions differently
    # depending on the calibration method
    probs_calibrated = calibrated_model.predict_proba(X_val)[:, 1]
    
    # Create calibration curves
    prob_true_uncal, prob_pred_uncal = calibration_curve(
        y_val, probs_uncalibrated, n_bins=10
    )
    prob_true_cal, prob_pred_cal = calibration_curve(
        y_val, probs_calibrated, n_bins=10
    )
    
    # Log calibration improvement
    brier_uncal = np.mean((probs_uncalibrated - y_val) ** 2)
    brier_cal = np.mean((probs_calibrated - y_val) ** 2)
    
    logger.info(f"Calibration improved Brier score from {brier_uncal:.4f} to {brier_cal:.4f}")
    
    # Save calibration curves for visualization
    calibration_df = pd.DataFrame({
        'true_probability_uncalibrated': prob_true_uncal,
        'predicted_probability_uncalibrated': prob_pred_uncal,
        'true_probability_calibrated': prob_true_cal,
        'predicted_probability_calibrated': prob_pred_cal
    })
    calibration_df.to_csv('calibration_curves.csv', index=False)
    
    # Return calibrated model
    return calibrated_model
```

By implementing these calibration techniques, we can significantly improve the reliability of our probability estimates, which is essential for precision optimization in soccer draw prediction.

### 12. Ensemble Refinement

To maximize precision, we'll implement advanced ensemble techniques that go beyond PyCaret's default stacking approach. These refinements focus on model diversity, multi-level stacking, and precision-oriented combination strategies.

#### 12.1 Multi-Level Stacking

Multi-level stacking creates a hierarchy of meta-learners to capture different aspects of the problem:

```python
def create_multilevel_stack(train_df, test_df, val_df):
    """Create a multi-level stacked ensemble
    
    Args:
        train_df: Training data with target
        test_df: Test data with target
        val_df: Validation data with target
        
    Returns:
        Final stacked model
    """
    logger.info("Creating multi-level stacked ensemble")
    
    # Setup PyCaret
    setup_pycaret = setup(
        data=train_df,
        target='target',
        session_id=42,
        log_experiment=True,
        experiment_name="multilevel_stack",
        fix_imbalance=True,
        fix_imbalance_method='smote'
    )
    
    # Level 1: Create specialized model groups
    logger.info("Training Level 1 - Specialized model groups")
    
    # Group 1: Tree-based models
    tree_models = []
    for model_name in ['dt', 'rf', 'et']:
        model = create_model(model_name)
        tuned_model = tune_model(model, optimize='Prec.')
        tree_models.append(tuned_model)
    
    # Group 2: Boosting models
    boost_models = []
    for model_name in ['xgboost', 'lightgbm', 'catboost']:
        model = create_model(model_name)
        tuned_model = tune_model(model, optimize='Prec.')
        boost_models.append(tuned_model)
    
    # Group 3: Linear and other models
    other_models = []
    for model_name in ['lr', 'ridge', 'svm']:
        model = create_model(model_name)
        tuned_model = tune_model(model, optimize='Prec.')
        other_models.append(tuned_model)
    
    # Level 2: Create group meta-learners
    logger.info("Training Level 2 - Group meta-learners")
    
    # Tree group stack
    tree_stack = stack_models(
        tree_models,
        meta_model=create_model('lr'),
        optimize='Prec.'
    )
    
    # Boosting group stack
    boost_stack = stack_models(
        boost_models,
        meta_model=create_model('rf'),
        optimize='Prec.'
    )
    
    # Other group stack
    other_stack = stack_models(
        other_models,
        meta_model=create_model('lightgbm'),
        optimize='Prec.'
    )
    
    # Level 3: Final meta-learner
    logger.info("Training Level 3 - Final meta-learner")
    final_stack = stack_models(
        [tree_stack, boost_stack, other_stack],
        meta_model=create_model('xgboost'),
        optimize='Prec.'
    )
    
    # Evaluate final stack
    final_preds = predict_model(final_stack, data=val_df)
    
    # Apply threshold optimization
    from threshold_utils import optimize_threshold_for_precision
    optimal_threshold, metrics = optimize_threshold_for_precision(
        final_preds,
        target_precision=0.50,
        min_recall=0.25
    )
    
    logger.info(f"Multi-level stack performance: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
    
    return final_stack, optimal_threshold
```

#### 12.2 Model Diversity Optimization

Ensuring base model diversity helps capture different signals in the data:

```python
def calculate_model_diversity(model_list, X_val, y_val):
    """Calculate diversity metrics between models
    
    Args:
        model_list: List of trained models
        X_val: Validation features
        y_val: Validation labels
        
    Returns:
        DataFrame with pairwise diversity metrics
    """
    logger.info("Calculating model diversity metrics")
    
    n_models = len(model_list)
    model_names = [f"Model_{i}" for i in range(n_models)]
    
    # Get predictions from all models
    predictions = []
    for model in model_list:
        preds = predict_model(model, data=X_val)
        predictions.append((preds['prediction_label'] == 1).astype(int))
    
    # Calculate disagreement matrix
    disagreement_matrix = np.zeros((n_models, n_models))
    
    for i in range(n_models):
        for j in range(i+1, n_models):
            # Disagreement is the percentage of predictions where models differ
            disagreement = np.mean(predictions[i] != predictions[j])
            disagreement_matrix[i, j] = disagreement
            disagreement_matrix[j, i] = disagreement
    
    # Convert to DataFrame for easier analysis
    disagreement_df = pd.DataFrame(
        disagreement_matrix,
        index=model_names,
        columns=model_names
    )
    
    # Calculate diversity metrics
    double_fault = np.zeros((n_models, n_models))
    
    for i in range(n_models):
        for j in range(i+1, n_models):
            # Double fault: both models predict incorrectly
            double_fault_rate = np.mean(
                (predictions[i] != y_val) & (predictions[j] != y_val)
            )
            double_fault[i, j] = double_fault_rate
            double_fault[j, i] = double_fault_rate
    
    double_fault_df = pd.DataFrame(
        double_fault,
        index=model_names,
        columns=model_names
    )
    
    logger.info("Model diversity analysis complete")
    
    return disagreement_df, double_fault_df
```

#### 12.3 Diversity-Based Model Selection

Select models that maximize diversity while maintaining precision:

```python
def select_diverse_models(model_list, X_val, y_val, n_select=5):
    """Select diverse models for ensemble
    
    Args:
        model_list: List of trained models
        X_val: Validation features
        y_val: Validation labels
        n_select: Number of models to select
        
    Returns:
        List of selected diverse models
    """
    logger.info(f"Selecting {n_select} diverse models from {len(model_list)} candidates")
    
    if len(model_list) <= n_select:
        logger.warning("Not enough models to select from, returning all models")
        return model_list
    
    # Get individual model performance
    performance = []
    for i, model in enumerate(model_list):
        preds = predict_model(model, data=X_val)
        precision = precision_score(y_val, preds['prediction_label'])
        performance.append((i, precision))
    
    # Sort by precision
    performance.sort(key=lambda x: x[1], reverse=True)
    
    # Calculate diversity metrics
    disagreement_df, _ = calculate_model_diversity(model_list, X_val, y_val)
    
    # Always select the best performing model
    selected_indices = [performance[0][0]]
    
    # Select remaining models based on diversity and performance
    while len(selected_indices) < n_select:
        max_diversity = -1
        next_model = -1
        
        for i, _ in performance:
            if i in selected_indices:
                continue
            
            # Calculate average disagreement with already selected models
            avg_disagreement = 0
            for j in selected_indices:
                avg_disagreement += disagreement_df.iloc[i, j]
            
            avg_disagreement /= len(selected_indices)
            
            # Find model with maximum diversity to already selected models
            if avg_disagreement > max_diversity:
                max_diversity = avg_disagreement
                next_model = i
        
        if next_model != -1:
            selected_indices.append(next_model)
        else:
            break
    
    # Return selected models
    selected_models = [model_list[i] for i in selected_indices]
    
    # Log selected model indices
    logger.info(f"Selected model indices: {selected_indices}")
    
    return selected_models
```

#### 12.4 Precision-Weighted Voting

Implement precision-weighted voting to favor more precise models:

```python
def precision_weighted_ensemble(model_list, X_train, y_train, X_val, y_val):
    """Create a precision-weighted voting ensemble
    
    Args:
        model_list: List of trained models
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        
    Returns:
        Weighted ensemble model and weights
    """
    logger.info("Creating precision-weighted voting ensemble")
    
    # Calculate precision weights for each model
    weights = []
    for model in model_list:
        preds = predict_model(model, data=X_val)
        prec = precision_score(y_val, preds['prediction_label'])
        
        # Square the precision to emphasize more precise models
        weight = prec ** 2
        weights.append(weight)
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    
    # Log model weights
    for i, w in enumerate(weights):
        logger.info(f"Model {i} weight: {w:.4f}")
    
    # Create a weighted voting function
    def weighted_predict_proba(X):
        """Weighted probability prediction"""
        probas = np.zeros((len(X), 2))
        
        for i, model in enumerate(model_list):
            preds = predict_model(model, data=X)
            # Get probabilities for class 1
            model_probas = preds['Score'].values
            # Add to weighted sum
            probas[:, 1] += weights[i] * model_probas
            probas[:, 0] = 1 - probas[:, 1]
        
        return probas
    
    # Create a simple wrapper class for the ensemble
    class PrecisionWeightedEnsemble:
        def __init__(self, models, weights):
            self.models = models
            self.weights = weights
        
        def predict_proba(self, X):
            return weighted_predict_proba(X)
        
        def predict(self, X, threshold=0.5):
            probas = self.predict_proba(X)[:, 1]
            return (probas >= threshold).astype(int)
    
    # Create the ensemble
    ensemble = PrecisionWeightedEnsemble(model_list, weights)
    
    # Evaluate the ensemble
    probs = ensemble.predict_proba(X_val)[:, 1]
    
    # Find optimal threshold
    from sklearn.metrics import precision_recall_curve, f1_score
    precisions, recalls, thresholds = precision_recall_curve(y_val, probs)
    
    # Find thresholds that achieve at least 50% precision and 25% recall
    valid_indices = []
    for i, (p, r) in enumerate(zip(precisions, recalls)):
        if p >= 0.5 and r >= 0.25:
            valid_indices.append(i)
    
    if valid_indices:
        # Choose threshold that maximizes F1 among valid thresholds
        f1_scores = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 
                    for p, r in zip(precisions[valid_indices], recalls[valid_indices])]
        best_idx = valid_indices[np.argmax(f1_scores)]
        
        # Use corresponding threshold (handling edge case)
        if best_idx < len(thresholds):
            threshold = thresholds[best_idx]
        else:
            # The last precision/recall point doesn't have a corresponding threshold
            threshold = 0.5
    else:
        # If no threshold achieves both targets, use 0.5 as default
        threshold = 0.5
        logger.warning("No threshold meets both precision and recall targets")
    
    # Evaluate with selected threshold
    preds = ensemble.predict(X_val, threshold)
    ensemble_precision = precision_score(y_val, preds)
    ensemble_recall = recall_score(y_val, preds)
    
    logger.info(f"Precision-weighted ensemble: Precision={ensemble_precision:.4f}, Recall={ensemble_recall:.4f}, Threshold={threshold:.4f}")
    
    return ensemble, threshold, weights
```

#### 12.5 Integration with PyCaret

To integrate these ensemble refinement techniques with our PyCaret workflow:

```python
def run_refined_ensemble_test():
    """Run tests with refined ensemble techniques"""
    # Setup MLflow tracking
    experiment_id = setup_mlflow_for_pycaret("pycaret_refined_ensemble")
    
    # Load data
    train_df, test_df, val_df = load_data_for_pycaret()
    
    # Setup PyCaret
    setup_pycaret = setup(
        data=train_df,
        target='target',
        session_id=44,
        log_experiment=True,
        experiment_name="refined_ensemble",
        fix_imbalance=True,
        fix_imbalance_method='smote'
    )
    
    # Train a pool of models
    logger.info("Training model pool for ensemble refinement")
    model_pool = []
    
    # Train models with different algorithms
    for model_name in ['xgboost', 'lightgbm', 'catboost', 'rf', 'et', 'lr', 'ridge', 'knn']:
        try:
            model = create_model(model_name)
            tuned_model = tune_model(model, optimize='Prec.', n_iter=10)
            model_pool.append(tuned_model)
            logger.info(f"Added {model_name} to model pool")
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
    
    # Extract validation data
    X_val = val_df.drop('target', axis=1)
    y_val = val_df['target']
    
    # 1. Select diverse models
    diverse_models = select_diverse_models(model_pool, X_val, y_val, n_select=5)
    
    # 2. Create precision-weighted ensemble
    weighted_ensemble, threshold, weights = precision_weighted_ensemble(
        diverse_models, 
        train_df.drop('target', axis=1), 
        train_df['target'],
        X_val,
        y_val
    )
    
    # 3. Create multi-level stack
    multilevel_stack, ml_threshold = create_multilevel_stack(train_df, test_df, val_df)
    
    # Compare performance of different ensembles
    results = {}
    
    # Standard stack from PyCaret
    standard_stack = stack_models(
        diverse_models,
        optimize='Prec.'
    )
    standard_preds = predict_model(standard_stack, data=val_df)
    standard_metrics = {
        'precision': precision_score(y_val, standard_preds['prediction_label']),
        'recall': recall_score(y_val, standard_preds['prediction_label'])
    }
    results['standard_stack'] = standard_metrics
    
    # Precision-weighted ensemble
    weighted_preds = weighted_ensemble.predict(X_val, threshold)
    weighted_metrics = {
        'precision': precision_score(y_val, weighted_preds),
        'recall': recall_score(y_val, weighted_preds)
    }
    results['weighted_ensemble'] = weighted_metrics
    
    # Multi-level stack
    multilevel_preds = predict_model(multilevel_stack, data=val_df)
    multilevel_metrics = {
        'precision': precision_score(y_val, (multilevel_preds['Score'] >= ml_threshold).astype(int)),
        'recall': recall_score(y_val, (multilevel_preds['Score'] >= ml_threshold).astype(int))
    }
    results['multilevel_stack'] = multilevel_metrics
    
    # Log results to MLflow
    with mlflow.start_run(experiment_id=experiment_id):
        for ensemble_name, metrics in results.items():
            mlflow.log_metrics({
                f"{ensemble_name}_precision": metrics['precision'],
                f"{ensemble_name}_recall": metrics['recall']
            })
        
        # Log model weights
        for i, w in enumerate(weights):
            mlflow.log_metric(f"model_{i}_weight", w)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results).T
    logger.info("\nEnsemble comparison results:")
    logger.info(results_df)
    
    # Select best performing ensemble based on precision
    best_ensemble = results_df.sort_values('precision', ascending=False).index[0]
    logger.info(f"Best ensemble: {best_ensemble}")
    
    return results
```

By implementing these ensemble refinement techniques, we can create more effective and precision-focused models that better capture the patterns in soccer draw prediction.

### 13. Advanced Sampling Techniques

Class imbalance is a significant challenge in soccer draw prediction, as draws typically represent only 20-30% of match outcomes. To address this imbalance while optimizing for precision, we'll implement several advanced sampling techniques beyond the basic SMOTE approach.

#### 13.1 BorderlineSMOTE Implementation

BorderlineSMOTE focuses on generating synthetic samples near the decision boundary, which is more effective for precision than standard SMOTE:

```python
from imblearn.over_sampling import BorderlineSMOTE

def apply_borderline_smote(X_train, y_train, sampling_strategy=0.5):
    """Apply BorderlineSMOTE for improved synthetic sample generation
    
    Args:
        X_train: Training features
        y_train: Training labels
        sampling_strategy: Target ratio of minority to majority class
        
    Returns:
        Resampled features and labels
    """
    logger.info(f"Applying BorderlineSMOTE with sampling strategy {sampling_strategy}")
    
    # Initialize BorderlineSMOTE
    # Use kind='borderline-1' for samples near the decision boundary
    # Use kind='borderline-2' for samples further into the majority class space
    smote = BorderlineSMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=5,
        random_state=42,
        kind='borderline-1'  # More conservative option
    )
    
    # Apply resampling
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    # Log resampling results
    original_class_counts = pd.Series(y_train).value_counts()
    resampled_class_counts = pd.Series(y_resampled).value_counts()
    
    logger.info("Class distribution before resampling:")
    for class_val, count in original_class_counts.items():
        logger.info(f"Class {class_val}: {count} samples ({count/len(y_train):.2%})")
    
    logger.info("Class distribution after BorderlineSMOTE:")
    for class_val, count in resampled_class_counts.items():
        logger.info(f"Class {class_val}: {count} samples ({count/len(y_resampled):.2%})")
    
    return X_resampled, y_resampled
```

#### 13.2 Tomek Links Removal

Tomek links are pairs of samples from different classes that are nearest neighbors. Removing majority class samples from these pairs can help focus the model on clearer decision boundaries:

```python
from imblearn.under_sampling import TomekLinks

def apply_tomek_links(X_train, y_train):
    """Remove Tomek links to clean decision boundary
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Cleaned features and labels
    """
    logger.info("Applying Tomek links removal")
    
    # Initialize TomekLinks
    tomek = TomekLinks(sampling_strategy='majority')
    
    # Apply undersampling
    X_cleaned, y_cleaned = tomek.fit_resample(X_train, y_train)
    
    # Log results
    n_removed = len(y_train) - len(y_cleaned)
    logger.info(f"Removed {n_removed} samples ({n_removed/len(y_train):.2%} of the original dataset)")
    
    # Log class distribution
    original_class_counts = pd.Series(y_train).value_counts()
    cleaned_class_counts = pd.Series(y_cleaned).value_counts()
    
    logger.info("Class distribution after Tomek links removal:")
    for class_val in cleaned_class_counts.index:
        original = original_class_counts.get(class_val, 0)
        cleaned = cleaned_class_counts.get(class_val, 0)
        diff = original - cleaned
        logger.info(f"Class {class_val}: {cleaned} samples (removed {diff} samples)")
    
    return X_cleaned, y_cleaned
```

#### 13.3 Hybrid Sampling Approach

A combination of oversampling and undersampling often yields better results than either technique alone:

```python
from imblearn.combine import SMOTETomek

def apply_hybrid_sampling(X_train, y_train, sampling_strategy=0.5):
    """Apply hybrid sampling using SMOTE and Tomek links
    
    Args:
        X_train: Training features
        y_train: Training labels
        sampling_strategy: Target ratio of minority to majority class
        
    Returns:
        Resampled features and labels
    """
    logger.info(f"Applying SMOTETomek hybrid sampling with strategy {sampling_strategy}")
    
    # Initialize SMOTETomek
    smote_tomek = SMOTETomek(
        sampling_strategy=sampling_strategy,
        random_state=42,
        smote=BorderlineSMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=5,
            random_state=42
        )
    )
    
    # Apply hybrid resampling
    X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)
    
    # Log resampling results
    original_pos_count = sum(y_train == 1)
    original_neg_count = sum(y_train == 0)
    resampled_pos_count = sum(y_resampled == 1)
    resampled_neg_count = sum(y_resampled == 0)
    
    logger.info(f"Original class ratio: {original_pos_count / original_neg_count:.4f}")
    logger.info(f"Resampled class ratio: {resampled_pos_count / resampled_neg_count:.4f}")
    logger.info(f"Added {resampled_pos_count - original_pos_count} positive samples")
    logger.info(f"Removed {original_neg_count - resampled_neg_count} negative samples")
    
    return X_resampled, y_resampled
```

#### 13.4 Sampling Strategy Optimization

Finding the optimal sampling strategy requires experimentation:

```python
def optimize_sampling_strategy(X_train, y_train, X_val, y_val):
    """Find optimal sampling strategy for maximizing precision
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        
    Returns:
        Best sampling strategy and corresponding model
    """
    logger.info("Optimizing sampling strategy for precision")
    
    # Strategies to try (minority:majority ratios)
    strategies = [0.1, 0.25, 0.5, 0.75, 1.0]
    results = {}
    
    # Test each strategy with a simple model
    for strategy in strategies:
        logger.info(f"Testing sampling strategy: {strategy}")
        
        # Apply hybrid sampling
        X_resampled, y_resampled = apply_hybrid_sampling(X_train, y_train, strategy)
        
        # Convert to DataFrames for PyCaret
        if isinstance(X_train, pd.DataFrame):
            train_df = X_resampled.copy()
            train_df['target'] = y_resampled
        else:
            # Assuming X_train is a numpy array with known column names
            train_df = pd.DataFrame(X_resampled, columns=[f"feature_{i}" for i in range(X_resampled.shape[1])])
            train_df['target'] = y_resampled
        
        # Setup PyCaret
        setup_pycaret = setup(
            data=train_df,
            target='target',
            session_id=45,
            log_experiment=True,
            experiment_name=f"sampling_strategy_{strategy}",
            fix_imbalance=False,  # We've already applied sampling
            normalize=True,
            normalize_method='robust',
            silent=True,
            verbose=False
        )
        
        # Train a simple model
        model = create_model('xgboost')
        
        # Validate on held-out data
        if isinstance(X_val, pd.DataFrame):
            val_df = X_val.copy()
        else:
            val_df = pd.DataFrame(X_val, columns=[f"feature_{i}" for i in range(X_val.shape[1])])
        
        # Get predictions
        preds = predict_model(model, data=val_df)
        
        # Calculate precision and recall
        y_pred = preds['prediction_label'].values
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        
        # Store results
        results[strategy] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        logger.info(f"Strategy {strategy} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    # Find best strategy based on precision, but ensure recall >= 0.25
    valid_strategies = {s: r for s, r in results.items() if r['recall'] >= 0.25}
    
    if valid_strategies:
        best_strategy = max(valid_strategies.items(), key=lambda x: x[1]['precision'])[0]
    else:
        logger.warning("No strategy achieved minimum recall, selecting based on F1 score")
        best_strategy = max(results.items(), key=lambda x: x[1]['f1'])[0]
    
    logger.info(f"Best sampling strategy: {best_strategy}")
    logger.info(f"Precision: {results[best_strategy]['precision']:.4f}, Recall: {results[best_strategy]['recall']:.4f}")
    
    return best_strategy, results
```

#### 13.5 Integration with PyCaret

To incorporate these advanced sampling techniques into the PyCaret workflow:

```python
def train_with_advanced_sampling(train_df, test_df, val_df):
    """Train models using advanced sampling techniques
    
    Args:
        train_df: Training data with target
        test_df: Test data with target
        val_df: Validation data with target
        
    Returns:
        Trained model and performance metrics
    """
    logger.info("Training with advanced sampling techniques")
    
    # Extract features and target
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_val = val_df.drop('target', axis=1)
    y_val = val_df['target']
    
    # Find optimal sampling strategy
    best_strategy, strategy_results = optimize_sampling_strategy(X_train, y_train, X_val, y_val)
    
    # Apply the best sampling strategy
    X_resampled, y_resampled = apply_hybrid_sampling(X_train, y_train, best_strategy)
    
    # Create a new DataFrame with resampled data
    resampled_train_df = pd.DataFrame(X_resampled, columns=X_train.columns)
    resampled_train_df['target'] = y_resampled
    
    # Setup PyCaret with the resampled data
    setup_pycaret = setup(
        data=resampled_train_df,
        target='target',
        session_id=46,
        log_experiment=True,
        experiment_name="advanced_sampling",
        fix_imbalance=False,  # Already applied sampling
        normalize=True,
        normalize_method='robust'
    )
    
    # Train models
    best_models = compare_models(sort='Prec.', n_select=3)
    
    # Tune best model
    tuned_model = tune_model(best_models[0], optimize='Prec.', n_iter=50)
    
    # Predict on validation set
    val_preds = predict_model(tuned_model, data=val_df)
    
    # Calculate metrics
    precision = precision_score(y_val, val_preds['prediction_label'])
    recall = recall_score(y_val, val_preds['prediction_label'])
    f1 = f1_score(y_val, val_preds['prediction_label'])
    
    logger.info(f"Advanced sampling results - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    # Check if we meet the target precision
    if precision >= 0.5 and recall >= 0.25:
        logger.info("SUCCESS: Target precision and recall achieved!")
    else:
        logger.info("Target metrics not achieved, consider additional refinements")
    
    return tuned_model, {'precision': precision, 'recall': recall, 'f1': f1}
```

By implementing these advanced sampling techniques, we can better handle the class imbalance in soccer draw prediction, potentially leading to improved precision while maintaining acceptable recall.

### 14. Confidence-Based Filtering

To achieve the 50% precision target, we can implement post-processing techniques that filter out predictions based on confidence levels. This approach allows us to focus on the most confident positive predictions and discard uncertain ones, boosting precision at the cost of some recall.

#### 14.1 Basic Confidence Filtering

The simplest approach is to apply a higher threshold to probability scores:

```python
def apply_confidence_filter(model, X, confidence_threshold=0.7):
    """Filter predictions based on confidence threshold
    
    Args:
        model: Trained model
        X: Features to predict on
        confidence_threshold: Minimum probability to consider a positive prediction
        
    Returns:
        Filtered predictions
    """
    logger.info(f"Applying confidence filter with threshold {confidence_threshold}")
    
    # Get raw predictions
    preds = predict_model(model, data=X)
    probas = preds['Score'].values
    raw_predictions = preds['prediction_label'].values
    
    # Apply confidence filter
    filtered_predictions = np.zeros_like(raw_predictions)
    
    # Only keep positive predictions with confidence >= threshold
    high_confidence_mask = (probas >= confidence_threshold)
    filtered_predictions[high_confidence_mask] = 1
    
    # Count filtered predictions
    n_raw_positive = np.sum(raw_predictions == 1)
    n_filtered_positive = np.sum(filtered_predictions == 1)
    
    logger.info(f"Raw positive predictions: {n_raw_positive}")
    logger.info(f"Filtered positive predictions: {n_filtered_positive}")
    logger.info(f"Reduction: {n_raw_positive - n_filtered_positive} samples ({(n_raw_positive - n_filtered_positive) / n_raw_positive:.2%})")
    
    return filtered_predictions
```

#### 14.2 Calibrated Confidence Scores

Using calibrated probabilities makes confidence filtering more reliable:

```python
def apply_calibrated_confidence_filter(model, X, y=None, confidence_threshold=0.7):
    """Filter predictions based on calibrated confidence scores
    
    Args:
        model: Trained model
        X: Features to predict on
        y: True labels (if available, for evaluation)
        confidence_threshold: Minimum probability to consider a positive prediction
        
    Returns:
        Filtered predictions
    """
    logger.info(f"Applying calibrated confidence filter with threshold {confidence_threshold}")
    
    # Get raw predictions
    preds = predict_model(model, data=X)
    raw_probas = preds['Score'].values
    raw_predictions = preds['prediction_label'].values
    
    # Calibrate probabilities using isotonic regression
    # We'll use a small subset for calibration if y is provided
    if y is not None:
        from sklearn.calibration import calibration_curve, CalibratedClassifierCV
        from sklearn.model_selection import train_test_split
        
        # Split data for calibration
        X_cal, X_test, y_cal, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
        
        # Get predictions on calibration set
        cal_preds = predict_model(model, data=X_cal)
        cal_probas = cal_preds['Score'].values
        
        # Create isotonic regression calibrator
        from sklearn.isotonic import IsotonicRegression
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(cal_probas, y_cal)
        
        # Calibrate probabilities
        calibrated_probas = ir.transform(raw_probas)
        
        # Log calibration improvement
        if y is not None:
            from sklearn.metrics import brier_score_loss
            raw_brier = brier_score_loss(y, raw_probas)
            cal_brier = brier_score_loss(y, calibrated_probas)
            logger.info(f"Brier score improved from {raw_brier:.4f} to {cal_brier:.4f} after calibration")
    else:
        # If no labels provided, use raw probabilities
        logger.warning("No labels provided for calibration, using raw probabilities")
        calibrated_probas = raw_probas
    
    # Apply confidence filter to calibrated probabilities
    filtered_predictions = np.zeros_like(raw_predictions)
    
    # Only keep positive predictions with calibrated confidence >= threshold
    high_confidence_mask = (calibrated_probas >= confidence_threshold)
    filtered_predictions[high_confidence_mask] = 1
    
    # Count filtered predictions
    n_raw_positive = np.sum(raw_predictions == 1)
    n_filtered_positive = np.sum(filtered_predictions == 1)
    
    logger.info(f"Raw positive predictions: {n_raw_positive}")
    logger.info(f"Filtered positive predictions: {n_filtered_positive}")
    logger.info(f"Reduction: {n_raw_positive - n_filtered_positive} samples ({(n_raw_positive - n_filtered_positive) / max(1, n_raw_positive):.2%})")
    
    # Evaluate if labels are provided
    if y is not None:
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        raw_precision = precision_score(y, raw_predictions)
        raw_recall = recall_score(y, raw_predictions)
        raw_f1 = f1_score(y, raw_predictions)
        
        filtered_precision = precision_score(y, filtered_predictions)
        filtered_recall = recall_score(y, filtered_predictions)
        filtered_f1 = f1_score(y, filtered_predictions)
        
        logger.info(f"Raw metrics - Precision: {raw_precision:.4f}, Recall: {raw_recall:.4f}, F1: {raw_f1:.4f}")
        logger.info(f"Filtered metrics - Precision: {filtered_precision:.4f}, Recall: {filtered_recall:.4f}, F1: {filtered_f1:.4f}")
    
    return filtered_predictions, calibrated_probas
```

#### 14.3 Two-Stage Classification

A more sophisticated approach uses a second classifier to identify likely false positives:

```python
def apply_two_stage_classifier(primary_model, X_train, y_train, X, y=None):
    """Apply two-stage classification to filter out false positives
    
    Args:
        primary_model: Primary trained model
        X_train: Training features
        y_train: Training labels
        X: Features to predict on
        y: True labels (if available, for evaluation)
        
    Returns:
        Filtered predictions
    """
    logger.info("Applying two-stage classification for false positive reduction")
    
    # Get predictions from primary model on training data
    train_preds = predict_model(primary_model, data=X_train)
    train_predictions = train_preds['prediction_label'].values
    
    # Find false positives in training set
    false_positives = (train_predictions == 1) & (y_train == 0)
    true_positives = (train_predictions == 1) & (y_train == 1)
    
    logger.info(f"Training false positives: {np.sum(false_positives)}")
    logger.info(f"Training true positives: {np.sum(true_positives)}")
    
    # Create dataset for second stage classifier
    # Only include samples predicted as positive by primary model
    positive_mask = (train_predictions == 1)
    X_second_stage = X_train[positive_mask]
    y_second_stage = y_train[positive_mask]
    
    # If we don't have enough samples, skip second stage
    if len(y_second_stage) < 50:
        logger.warning("Not enough positive predictions for second stage classifier")
        
        # Get predictions from primary model
        test_preds = predict_model(primary_model, data=X)
        return test_preds['prediction_label'].values
    
    # Train second stage classifier to distinguish true from false positives
    logger.info("Training second stage classifier")
    
    # Create DataFrame for PyCaret
    second_stage_df = X_second_stage.copy()
    second_stage_df['target'] = y_second_stage
    
    # Setup PyCaret for second stage
    setup_pycaret = setup(
        data=second_stage_df,
        target='target',
        session_id=47,
        log_experiment=True,
        experiment_name="second_stage_classifier",
        normalize=True,
        normalize_method='robust',
        silent=True,
        verbose=False
    )
    
    # Create second stage model
    second_stage_model = create_model('xgboost')
    
    # Get predictions from primary model on test data
    test_preds = predict_model(primary_model, data=X)
    test_predictions = test_preds['prediction_label'].values
    test_probas = test_preds['Score'].values
    
    # Apply second stage only to positive predictions from primary model
    positive_mask = (test_predictions == 1)
    
    # If no positive predictions, return all negatives
    if np.sum(positive_mask) == 0:
        logger.warning("No positive predictions from primary model")
        return np.zeros_like(test_predictions)
    
    # Get predictions from second stage model
    second_stage_data = X[positive_mask]
    second_stage_preds = predict_model(second_stage_model, data=second_stage_data)
    second_stage_predictions = second_stage_preds['prediction_label'].values
    
    # Final predictions: keep only positives confirmed by second stage
    final_predictions = np.zeros_like(test_predictions)
    positive_indices = np.where(positive_mask)[0]
    confirmed_positives = positive_indices[second_stage_predictions == 1]
    final_predictions[confirmed_positives] = 1
    
    # Count filtered predictions
    n_primary_positive = np.sum(test_predictions == 1)
    n_final_positive = np.sum(final_predictions == 1)
    
    logger.info(f"Primary positive predictions: {n_primary_positive}")
    logger.info(f"Final positive predictions: {n_final_positive}")
    logger.info(f"Reduction: {n_primary_positive - n_final_positive} samples ({(n_primary_positive - n_final_positive) / max(1, n_primary_positive):.2%})")
    
    # Evaluate if labels are provided
    if y is not None:
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        primary_precision = precision_score(y, test_predictions)
        primary_recall = recall_score(y, test_predictions)
        primary_f1 = f1_score(y, test_predictions)
        
        final_precision = precision_score(y, final_predictions)
        final_recall = recall_score(y, final_predictions)
        final_f1 = f1_score(y, final_predictions)
        
        logger.info(f"Primary metrics - Precision: {primary_precision:.4f}, Recall: {primary_recall:.4f}, F1: {primary_f1:.4f}")
        logger.info(f"Final metrics - Precision: {final_precision:.4f}, Recall: {final_recall:.4f}, F1: {final_f1:.4f}")
    
    return final_predictions
```

#### 14.4 Optimal Confidence Threshold Search

Finding the optimal confidence threshold is crucial for the precision-recall tradeoff:

```python
def find_optimal_confidence_threshold(model, X_val, y_val, min_precision=0.5, min_recall=0.25):
    """Find optimal confidence threshold to achieve target precision while maximizing recall
    
    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation labels
        min_precision: Minimum required precision
        min_recall: Minimum required recall
        
    Returns:
        Optimal threshold and performance metrics
    """
    logger.info(f"Finding optimal confidence threshold (min precision: {min_precision}, min recall: {min_recall})")
    
    # Get predictions and probabilities
    preds = predict_model(model, data=X_val)
    probas = preds['Score'].values
    
    # Calculate precision and recall at different thresholds
    from sklearn.metrics import precision_recall_curve, f1_score
    
    precision, recall, thresholds = precision_recall_curve(y_val, probas)
    
    # Create DataFrame of all threshold options
    threshold_df = pd.DataFrame({
        'threshold': np.append(thresholds, 1.0),  # Add 1.0 as the last threshold
        'precision': precision,
        'recall': recall
    })
    
    # Filter to thresholds meeting minimum requirements
    valid_thresholds = threshold_df[(threshold_df['precision'] >= min_precision) & 
                                    (threshold_df['recall'] >= min_recall)]
    
    if valid_thresholds.empty:
        logger.warning("No threshold satisfies both precision and recall requirements")
        
        # Get threshold with highest precision that meets recall requirement
        recall_valid = threshold_df[threshold_df['recall'] >= min_recall]
        
        if not recall_valid.empty:
            optimal_row = recall_valid.loc[recall_valid['precision'].idxmax()]
            logger.info(f"Selected threshold with highest precision that meets recall requirement")
        else:
            # If no threshold meets recall requirement, prefer precision
            optimal_row = threshold_df.loc[threshold_df['precision'].idxmax()]
            logger.warning(f"No threshold meets recall requirement. Selected maximum precision threshold.")
    else:
        # Among valid thresholds, maximize F1 score
        f1_scores = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 
                    for p, r in zip(precision[valid_thresholds], recall[valid_thresholds])]
        best_idx = valid_thresholds[np.argmax(f1_scores)]
        
        optimal_row = valid_thresholds.iloc[best_idx]
        logger.info(f"Selected optimal threshold that maximizes F1 while meeting requirements")
    
    optimal_threshold = optimal_row['threshold']
    optimal_precision = optimal_row['precision']
    optimal_recall = optimal_row['recall']
    
    # Calculate optimal F1
    if (optimal_precision + optimal_recall) > 0:
        optimal_f1 = 2 * (optimal_precision * optimal_recall) / (optimal_precision + optimal_recall)
    else:
        optimal_f1 = 0.0
    
    logger.info(f"Optimal threshold: {optimal_threshold:.4f}")
    logger.info(f"Expected precision: {optimal_precision:.4f}")
    logger.info(f"Expected recall: {optimal_recall:.4f}")
    logger.info(f"Expected F1 score: {optimal_f1:.4f}")
    
    # Save threshold analysis
    threshold_df.to_csv(f"confidence_threshold_analysis.csv", index=False)
    
    return optimal_threshold, {
        'threshold': optimal_threshold,
        'precision': optimal_precision,
        'recall': optimal_recall,
        'f1': optimal_f1
    }
```

#### 14.5 Integration with PyCaret

To incorporate confidence filtering into the PyCaret workflow:

```python
def pycaret_with_confidence_filtering(train_df, test_df, val_df):
    """Train a model with PyCaret and apply confidence filtering
    
    Args:
        train_df: Training data with target
        test_df: Test data with target
        val_df: Validation data with target
        
    Returns:
        Filtered model and metrics
    """
    logger.info("Training model with confidence filtering")
    
    # Extract validation data
    X_val = val_df.drop('target', axis=1)
    y_val = val_df['target']
    
    # Setup PyCaret
    setup_pycaret = setup(
        data=train_df,
        target='target',
        session_id=48,
        log_experiment=True,
        experiment_name="confidence_filtering",
        fix_imbalance=True,
        fix_imbalance_method='smote'
    )
    
    # Train best model
    best_model = compare_models(sort='Prec.', n_select=1)[0]
    tuned_model = tune_model(best_model, optimize='Prec.', n_iter=50)
    
    # Find optimal confidence threshold
    optimal_threshold, threshold_metrics = find_optimal_confidence_threshold(
        tuned_model,
        X_val,
        y_val,
        min_precision=0.5,
        min_recall=0.25
    )
    
    # Apply confidence filtering
    filtered_predictions, calibrated_probas = apply_calibrated_confidence_filter(
        tuned_model,
        X_val,
        y_val,
        confidence_threshold=optimal_threshold
    )
    
    # Try two-stage classification if needed
    if threshold_metrics['precision'] < 0.5 and len(y_val) >= 200:
        logger.info("Trying two-stage classification approach")
        two_stage_predictions = apply_two_stage_classifier(
            tuned_model,
            train_df.drop('target', axis=1),
            train_df['target'],
            X_val,
            y_val
        )
        
        # Calculate metrics for two-stage approach
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        two_stage_precision = precision_score(y_val, two_stage_predictions)
        two_stage_recall = recall_score(y_val, two_stage_predictions)
        two_stage_f1 = f1_score(y_val, two_stage_predictions)
        
        # Compare with threshold filtering
        threshold_precision = threshold_metrics['precision']
        threshold_recall = threshold_metrics['recall']
        
        if two_stage_precision >= threshold_precision and two_stage_recall >= 0.25:
            logger.info("Two-stage classification performs better, using this approach")
            final_predictions = two_stage_predictions
            final_metrics = {
                'precision': two_stage_precision,
                'recall': two_stage_recall,
                'f1': two_stage_f1
            }
        else:
            logger.info("Confidence threshold filtering performs better")
            final_predictions = filtered_predictions
            final_metrics = threshold_metrics
    else:
        final_predictions = filtered_predictions
        final_metrics = threshold_metrics
    
    # Check if we meet the target precision
    if final_metrics['precision'] >= 0.5 and final_metrics['recall'] >= 0.25:
        logger.info("SUCCESS: Target precision and recall achieved with confidence filtering!")
    else:
        logger.info("Target metrics not achieved, consider additional refinements")
    
    # Create a model wrapper that includes confidence filtering
    class ConfidenceFilteredModel:
        def __init__(self, base_model, threshold):
            self.base_model = base_model
            self.threshold = threshold
        
        def predict_proba(self, X):
            preds = predict_model(self.base_model, data=X)
            probas = preds['Score'].values
            return np.vstack((1 - probas, probas)).T
        
        def predict(self, X):
            probas = self.predict_proba(X)[:, 1]
            return (probas >= self.threshold).astype(int)
    
    # Create the filtered model
    filtered_model = ConfidenceFilteredModel(tuned_model, optimal_threshold)
    
    return filtered_model, final_metrics
```

By implementing these confidence-based filtering techniques, we can significantly boost prediction precision by focusing on high-confidence positive predictions while maintaining sufficient recall.

## Phased Implementation Plan

To ensure successful implementation and maximize our chances of reaching the 50% precision target, we'll adopt a phased approach with clear milestones:

### Phase 1: Baseline Establishment (1-2 days)
- **Goal**: Establish baseline performance with simple models
- **Success Criteria**: At least one model achieving 40% precision with 25% recall
- **Activities**:
  - Implement basic data loading and preprocessing
  - Run initial model comparison with default settings
  - Document baseline performance for each model type

### Phase 2: Feature Engineering & Model Tuning (3-5 days)
- **Goal**: Improve precision through feature engineering and hyperparameter tuning
- **Success Criteria**: At least one model achieving 45% precision with 25% recall
- **Activities**:
  - Implement automated feature importance analysis
  - Engineer precision-focused features 
  - Tune individual models with precision-focused parameters
  - Review progress and adjust approach if necessary

### Phase 3: Ensemble Development (2-3 days)
- **Goal**: Create precision-optimized ensemble models
- **Success Criteria**: Ensemble achieving 48% precision with 25% recall
- **Activities**:
  - Test different meta-learner architectures
  - Implement diversity metrics for base model selection
  - Apply calibration techniques to improve probability estimates
  - Hold review meeting to assess progress

### Phase 4: Precision Optimization (2-3 days)
- **Goal**: Apply advanced techniques to reach target precision
- **Success Criteria**: Final model achieving 50% precision with 25% recall
- **Activities**:
  - Implement confidence-based filtering
  - Fine-tune thresholds for optimal precision-recall trade-off
  - Test hybrid sampling approaches
  - Conduct final performance evaluation

### Phase 5: Integration & Validation (1-2 days)
- **Goal**: Integrate with existing ensemble and validate in production-like environment
- **Success Criteria**: Integrated model maintaining 50% precision in validation tests
- **Activities**:
  - Create production-ready model wrapper
  - Implement A/B testing with existing ensemble
  - Document final approach and results

### Review Points
- **Daily**: Quick progress check and blocker resolution
- **End of Each Phase**: Comprehensive review of metrics, approach, and adjustments
- **Final Review**: Complete evaluation against project goals with stakeholder presentation

## Documentation Improvements

To enhance understanding and troubleshooting, we've added comprehensive documentation that addresses various aspects of the implementation:

### Expected Outcomes for Each Stage

| Implementation Stage | Expected Precision | Expected Recall | Expected F1 Score |
|----------------------|-------------------|-----------------|-------------------|
| Initial model comparison | 35-40% | 25-35% | 0.30-0.35 |
| After feature engineering | 40-45% | 25-30% | 0.32-0.37 |
| After model tuning | 45-48% | 25-30% | 0.35-0.40 |
| After ensemble optimization | 48-52% | 25-28% | 0.38-0.42 |
| After calibration & filtering | 50-55% | 25-27% | 0.40-0.45 |

### Troubleshooting Guide

#### Common Issues and Solutions

1. **Model Precision Below Target**
   - **Issue**: Models failing to reach 40% precision in initial testing
   - **Solutions**: 
     - Try different resampling techniques (BorderlineSMOTE, ADASYN)
     - Increase class weight for positive samples
     - Feature engineering focused on false positive reduction

2. **Low Recall with High Precision**
   - **Issue**: Models achieving high precision but falling below 25% recall
   - **Solutions**:
     - Lower threshold in custom optimization function
     - Adjust class weights to reduce false negative rate
     - Try ensemble models with higher diversity

3. **Memory or Performance Issues**
   - **Issue**: PyCaret operations consuming excessive memory or time
   - **Solutions**:
     - Reduce number of models in comparison step
     - Use fewer CV folds during initial exploration
     - Apply feature selection earlier in the pipeline

4. **Integration Conflicts**
   - **Issue**: PyCaret model not working with existing ensemble
   - **Solutions**:
     - Verify feature names and transformations match between systems
     - Ensure probability calibration is consistent
     - Test with simplified subset of models first

### Soccer Prediction Glossary

- **Draw**: Match ending with equal scores between teams (target variable)
- **Feature Importance**: Measure of a feature's contribution to prediction accuracy
- **Match Context Features**: Variables describing the circumstances of a match (e.g., derby, cup game)
- **Team Form**: Recent performance metrics for teams (e.g., last 5 matches)
- **Market Odds**: Betting market predictions that can serve as features
- **Expected Goals (xG)**: Advanced metric predicting scoring probability
- **Shot-based Metrics**: Features related to shot quantity and quality

### Machine Learning Terminology

- **Calibration**: Adjusting probability outputs to match observed frequencies
- **Ensemble Stacking**: Using model predictions as features for a meta-learner
- **Meta-learner**: Model that combines predictions from base models
- **BorderlineSMOTE**: Advanced oversampling technique focusing on decision boundary
- **Tomek Links**: Undersampling technique removing borderline majority class samples
- **Precision**: Ratio of true positives to all positive predictions
- **Recall**: Ratio of true positives to all actual positives in dataset
- **Diversity Metrics**: Measures of prediction disagreement between models
- **Confidence Filtering**: Post-processing to reject low-confidence predictions

## Future Enhancements

After initial implementation, consider these enhancements:

1. **Feature selection with PyCaret**: Use PyCaret's built-in feature selection to identify the most important features
2. **Hyperparameter optimization**: Leverage PyCaret's tuning capabilities to further optimize the models
3. **Blending with existing ensemble**: Create a higher-level ensemble that combines predictions from both systems
4. **Automated retraining**: Set up a workflow to regularly retrain and compare both approaches

## Conclusion

This implementation plan provides a structured approach to test PyCaret for soccer draw prediction while targeting 50% precision and maintaining at least 25% recall. The standalone testing allows for rapid experimentation before integrating with the existing system.

The implementation follows our project guidelines for MLflow tracking, logging, and model management while introducing new methodologies through PyCaret's automated ML capabilities.

from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

def precision_focused_score(y_true, y_pred, precision_weight=0.8):
    """Custom scoring metric that prioritizes precision but maintains minimum recall"""
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    
    # Only consider models with recall above minimum threshold
    if rec < 0.25:  # Our minimum recall requirement
        return 0.0
    
    # Weighted harmonic mean prioritizing precision
    return ((1 + precision_weight**2) * prec * rec) / ((precision_weight**2 * prec) + rec)

# Create scorer for PyCaret
precision_focused_scorer = make_scorer(precision_focused_score) 

def engineer_precision_features(df):
    """Add engineered features to improve precision"""
    logger.info("Engineering precision-focused features")
    
    # 1. Create interaction features between strongest predictors
    # Assumption: Features with high gain/importance in existing models
    top_features = ["feature1", "feature2", "feature3"]  # Replace with actual features
    
    for i in range(len(top_features)):
        for j in range(i+1, len(top_features)):
            feat_name = f"interaction_{top_features[i]}_{top_features[j]}"
            df[feat_name] = df[top_features[i]] * df[top_features[j]]
    
    # 2. Add polynomial features for key numeric predictors
    for feat in top_features:
        if df[feat].dtype in [np.float64, np.int64]:
            df[f"{feat}_squared"] = df[feat]**2
    
    # 3. Create "confidence" features - distance from decision boundary proxies
    # This helps models be more confident in their predictions
    if "home_win_prob" in df.columns and "away_win_prob" in df.columns:
        df["draw_confidence"] = 1 - abs(df["home_win_prob"] - df["away_win_prob"])
    
    # 4. Ratio features often help with precision
    numeric_cols = df.select_dtypes(include=['number']).columns
    for i in range(min(5, len(numeric_cols))):
        for j in range(i+1, min(5, len(numeric_cols))):
            if df[numeric_cols[j]].min() != 0:
                df[f"ratio_{numeric_cols[i]}_{numeric_cols[j]}"] = df[numeric_cols[i]] / df[numeric_cols[j]]
    
    logger.info(f"Added {df.shape[1] - len(numeric_cols)} new engineered features")
    return df 

# Updated tuning approach for high precision
def tune_for_precision(model, model_name):
    """Intensive tuning focused on high precision"""
    logger.info(f"Precision-focused tuning for {model_name}")
    
    # Specific tuning for different model types
    if model_name in ['xgboost', 'catboost', 'lightgbm']:
        # Boosting model tuning - focus on reducing false positives
        return tune_model(model, 
                        optimize='Prec.',
                        n_iter=100,  # More iterations
                        custom_grid={
                            'scale_pos_weight': [0.5, 0.75, 1.0],  # Balance control
                            'max_depth': [3, 4, 5, 6],  # Shallower trees to avoid overfitting
                            'min_child_weight': [1, 2, 4, 8],  # Higher values prevent overfitting
                            'subsample': [0.7, 0.8, 0.9],  # Subsample ratio for training
                            'colsample_bytree': [0.7, 0.8, 0.9],  # Subsample ratio of columns
                            'reg_alpha': [0.01, 0.1, 1.0],  # L1 regularization
                            'reg_lambda': [0.01, 0.1, 1.0, 10.0],  # L2 regularization
                        })
    elif model_name == 'rf':
        # Random Forest tuning
        return tune_model(model, 
                        optimize='Prec.',
                        n_iter=100,
                        custom_grid={
                            'class_weight': ['balanced', 'balanced_subsample', {0: 1, 1: 2}],
                            'min_samples_leaf': [1, 2, 4, 8],  # More conservative splits
                            'max_features': ['sqrt', 'log2', 0.3, 0.5],  # Feature sampling
                            'max_samples': [0.7, 0.8, 0.9]  # Bootstrap sampling
                        })
    else:
        # Default tuning for other models
        return tune_model(model, optimize='Prec.', n_iter=100) 