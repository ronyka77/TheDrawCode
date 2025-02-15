"""Metrics calculation utilities for model evaluation."""

from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    log_loss,
    confusion_matrix
)

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive set of evaluation metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Binary predictions
        y_prob: Probability predictions
        
    Returns:
        Dictionary of metric names and values
        
    Raises:
        ValueError: If inputs are invalid or empty
    """
    # Input validation
    if y_true is None or y_pred is None or y_prob is None:
        raise ValueError("Input arrays cannot be None")
    if len(y_true) == 0 or len(y_pred) == 0 or len(y_prob) == 0:
        raise ValueError("Input arrays cannot be empty")
    if len(y_true) != len(y_pred) or len(y_true) != len(y_prob):
        raise ValueError("Input arrays must have the same length")
    
    metrics = {}
    try:
        # Classification metrics
        metrics['precision'] = precision_score(y_true, y_pred, pos_label=1, zero_division=0.0)
        metrics['recall'] = recall_score(y_true, y_pred, pos_label=1, zero_division=0.0)
        metrics['f1'] = f1_score(y_true, y_pred, pos_label=1, zero_division=0.0)
        
        # Probability metrics
        metrics['auc_pr'] = average_precision_score(y_true, y_prob)
        metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
        metrics['log_loss'] = log_loss(y_true, y_prob)
        
        # Confusion matrix metrics
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp'] = cm.ravel()
        
        # Additional metrics
        try:
            metrics['specificity'] = metrics['tn'] / (metrics['tn'] + metrics['fp'])
        except ZeroDivisionError:
            metrics['specificity'] = 0.0
            
        metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2
        
        # Prediction distribution statistics
        metrics['mean_probability'] = np.mean(y_prob)
        metrics['std_probability'] = np.std(y_prob)
        
        # Handle None values by setting to 0
        for key, value in metrics.items():
            if value is None:
                metrics[key] = 0.0
                
    except Exception as e:
        # Initialize all metrics to 0.0 in case of error
        metric_names = ['precision', 'recall', 'f1', 'auc_pr', 'auc_roc', 'log_loss',
                        'tn', 'fp', 'fn', 'tp', 'specificity', 'balanced_accuracy',
                        'mean_probability', 'std_probability']
        metrics = {name: 0.0 for name in metric_names}
        raise RuntimeError(f"Error calculating metrics: {str(e)}")
        
    return metrics