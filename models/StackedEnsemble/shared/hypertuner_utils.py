"""
Hyperparameter Tuning Utilities for Tree-based Models

This module contains common functions used across different tree-based models 
(LightGBM, CatBoost, XGBoost, etc.) for:
- Prediction generation
- Model evaluation 
- Threshold optimization
- Performance metrics calculation

These utilities help standardize the training and evaluation process across 
different model types while reducing code duplication.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Tuple, Union, List, Optional, Callable

# Metrics imports
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, precision_recall_curve
)

# Add project root to Python path if not already there
try:
    project_root = Path(__file__).parent.parent.parent.parent
    if not project_root.exists():
        # Handle network path by using raw string
        project_root = Path(r"\\".join(str(project_root).split("\\")))
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
except Exception as e:
    print(f"Error setting project root path: {e}")
    # Fallback to current directory if path resolution fails
    sys.path.append(os.getcwd())

# Import logger
from utils.logger import ExperimentLogger
logger = ExperimentLogger(experiment_name="hypertuner_utils")

# Default settings
DEFAULT_MIN_RECALL = 0.20  # Minimum acceptable recall

def predict(model: Any, X: Union[pd.DataFrame, np.ndarray], threshold: float = 0.5) -> np.ndarray:
    """
    Generate binary predictions using a trained model.
    
    Works with any model that implements predict_proba with scikit-learn compatible interface.
    
    Args:
        model: Trained model object (LightGBM, CatBoost, etc.)
        X: Features to predict on
        threshold: Decision threshold for binary classification
        
    Returns:
        np.ndarray: Binary predictions (0 or 1)
    """
    if model is None:
        raise RuntimeError("Model must be trained before prediction")
        
    try:
        # Get probability predictions
        probas = predict_proba(model, X)
        
        # Apply threshold
        return (probas >= threshold).astype(int)
        
    except Exception as e:
        logger.error(f"Error in model prediction: {str(e)}")
        return np.zeros(len(X))

def predict_proba(model: Any, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """
    Generate probability predictions for the positive class.
    
    Works with any model that implements predict_proba with scikit-learn compatible interface.
    
    Args:
        model: Trained model object (LightGBM, CatBoost, etc.)
        X: Features to predict on
        
    Returns:
        np.ndarray: Probability predictions for the positive class
    """
    if model is None:
        raise RuntimeError("Model must be trained before prediction")
        
    try:
        # Get probability predictions (take positive class probabilities)
        return model.predict_proba(X)[:, 1]
        
    except Exception as e:
        logger.error(f"Error in probability prediction: {str(e)}")
        return np.zeros(len(X))

def evaluate(
    model: Any, 
    X: Union[pd.DataFrame, np.ndarray], 
    y: Union[pd.Series, np.ndarray], 
    threshold: float = 0.5) -> Dict[str, float]:
    """
    Evaluate model performance on given data.
    
    Calculates multiple classification metrics using the provided threshold.
    
    Args:
        model: Trained model object
        X: Feature data
        y: Target labels
        threshold: Decision threshold to use for binary predictions
        
    Returns:
        dict: Evaluation metrics including precision, recall, F1, AUC, etc.
    """
    if model is None:
        raise RuntimeError("Model must be trained before evaluation")
    
    try:
        # Get probability predictions
        y_prob = predict_proba(model, X)
        
        # Get binary predictions using threshold
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calculate metrics components for greater numerical stability
        tp = np.sum((y == 1) & (y_pred == 1))
        fp = np.sum((y == 0) & (y_pred == 1))
        fn = np.sum((y == 1) & (y_pred == 0))
        
        # Calculate metrics
        metrics = {
            'precision': tp / (tp + fp + 1e-10),
            'recall': tp / (tp + fn + 1e-10),
            'f1': 2 * tp / (2 * tp + fp + fn + 1e-10),
            'auc': roc_auc_score(y, y_prob),
            'brier_score': np.mean((y_prob - y) ** 2),
            'threshold': threshold
        }
        
        # Log results
        logger.info(f"Evaluation Results (threshold={threshold:.3f}):")
        logger.info(f"  Precision: {metrics['precision']:.3f}")
        logger.info(f"  Recall: {metrics['recall']:.3f}")
        logger.info(f"  F1 Score: {metrics['f1']:.3f}")
        logger.info(f"  AUC: {metrics['auc']:.3f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error in model evaluation: {str(e)}")
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'auc': 0.0,
            'brier_score': 1.0,
            'threshold': threshold
        }

def optimize_threshold(
    model: Any, 
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    min_recall: float = DEFAULT_MIN_RECALL,
    metric_to_optimize: str = 'precision',
    threshold_range: Tuple[float, float] = (0.1, 0.9),
    n_thresholds: int = 81) -> Tuple[float, Dict[str, float]]:
    """
    Optimize prediction threshold with focus on a specified metric while maintaining 
    minimum recall requirements.
    
    Args:
        model: Trained model
        X: Feature data for threshold optimization
        y: True labels for threshold optimization
        min_recall: Minimum acceptable recall (default: 0.20)
        metric_to_optimize: Metric to optimize ('precision', 'f1', etc.)
        threshold_range: Range of thresholds to search (min, max)
        n_thresholds: Number of threshold values to try
        
    Returns:
        tuple: (best_threshold, metrics_at_best_threshold)
    """
    try:
        # Get probability predictions
        y_prob = predict_proba(model, X)
        
        best_threshold = 0.5
        best_metric_value = 0.0
        
        # Search through thresholds
        for threshold in np.linspace(threshold_range[0], threshold_range[1], n_thresholds):
            y_pred = (y_prob >= threshold).astype(int)
            
            # Calculate confusion matrix components
            tp = np.sum((y == 1) & (y_pred == 1))
            fp = np.sum((y == 0) & (y_pred == 1))
            fn = np.sum((y == 1) & (y_pred == 0))
            
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            
            # Select metric to optimize
            if metric_to_optimize == 'precision':
                current_metric = precision
            elif metric_to_optimize == 'f1':
                current_metric = f1
            else:
                current_metric = precision  # Default to precision
            
            # Only consider thresholds that maintain recall above minimum
            if recall >= min_recall:
                if current_metric > best_metric_value:
                    best_metric_value = current_metric
                    best_threshold = threshold
        
        logger.info(f"Optimized threshold: {best_threshold:.3f} with {metric_to_optimize}: {best_metric_value:.3f}")
        
        # Calculate full metrics at the best threshold
        best_metrics = evaluate(model, X, y, best_threshold)
        return best_threshold, best_metrics
        
    except Exception as e:
        logger.error(f"Error optimizing threshold: {str(e)}")
        return 0.5, {'threshold': 0.5, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

def calculate_feature_importance(
    model: Any,
    feature_names: List[str] = None,
    importance_type: str = None) -> pd.DataFrame:
    """
    Extract feature importance from a tree-based model.
    
    Works with different model types by trying different attribute/method patterns.
    
    Args:
        model: Trained model (LightGBM, CatBoost, XGBoost, etc.)
        feature_names: List of feature names (if not available from model)
        importance_type: Type of importance to extract (model-specific)
        
    Returns:
        pd.DataFrame: DataFrame with Feature and Importance columns, sorted by importance
    """
    try:
        importance_values = None
        
        # Try different model types
        if hasattr(model, 'feature_importances_'):  # sklearn, LightGBM
            importance_values = model.feature_importances_
        elif hasattr(model, 'get_feature_importance'):  # CatBoost
            importance_values = model.get_feature_importance()
        elif hasattr(model, 'feature_importance'):  # XGBoost
            importance_values = model.feature_importance(importance_type=importance_type)
        
        if importance_values is None:
            logger.warning("Could not extract feature importance, model type not supported")
            return pd.DataFrame(columns=['Feature', 'Importance'])
            
        # Get feature names
        if feature_names is None:
            if hasattr(model, 'feature_name_'):
                feature_names = model.feature_name_
            elif hasattr(model, 'feature_names_'):
                feature_names = model.feature_names_
            else:
                feature_names = [f"feature_{i}" for i in range(len(importance_values))]
                
        # Create and return DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_values
        }).sort_values('Importance', ascending=False)
        
        return importance_df
        
    except Exception as e:
        logger.error(f"Error extracting feature importance: {str(e)}")
        return pd.DataFrame(columns=['Feature', 'Importance']) 