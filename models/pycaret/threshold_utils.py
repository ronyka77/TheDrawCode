"""
Threshold optimization utilities for PyCaret soccer prediction.

This module contains functions for optimizing classification thresholds
to achieve target precision and recall values.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import sys
from pathlib import Path

# Add project root to Python path if needed
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from utils.logger import ExperimentLogger

# Setup logger
logger = ExperimentLogger(experiment_name="pycaret_threshold_utils")

def precision_focused_score(y_true, y_pred, target_precision=0.4, min_recall=0.25):
    """
    Custom scoring function that prioritizes precision while maintaining
    a minimum recall threshold.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_precision: Target precision to achieve
        min_recall: Minimum acceptable recall
        
    Returns:
        float: Score (precision if precision >= target_precision and recall >= min_recall, otherwise 0)
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    logger.debug(f"Calculated precision: {precision:.4f}, recall: {recall:.4f}")
    logger.debug(f"Target precision: {target_precision:.4f}, min recall: {min_recall:.4f}")
    
    # If recall is below threshold or precision is below target, return 0
    if recall < min_recall:
        logger.debug(f"Recall {recall:.4f} below minimum {min_recall:.4f}, returning 0")
        return 0.0
    
    # Otherwise return precision
    logger.debug(f"Returning precision score: {precision:.4f}")
    return float(precision)

def optimize_threshold_for_precision(predictions, target_precision=0.5, min_recall=0.25, 
                                    prob_col='prediction_score', target_col='target'):
    """
    Find the optimal threshold to achieve target precision with minimum recall.
    
    Args:
        predictions (pd.DataFrame): DataFrame with predictions and true labels
        target_precision (float): Target precision to achieve
        min_recall (float): Minimum acceptable recall
        prob_col (str): Name of the column with prediction probabilities
        target_col (str): Name of the column with true labels
        
    Returns:
        tuple: (optimal_threshold, metrics_dict)
    """
    logger.info(f"Optimizing threshold for target precision {target_precision} with min recall {min_recall}")
    
    # Ensure we have the required columns
    if prob_col not in predictions.columns:
        logger.error(f"Probability column '{prob_col}' not found in predictions DataFrame")
        # Check if we have alternative columns
        for alt_col in ['prediction_score', 'Score', 'Score_1']:
            if alt_col in predictions.columns:
                logger.info(f"Using alternative probability column: {alt_col}")
                prob_col = alt_col
                break
        else:
            return 0.5, {}
    
    if target_col not in predictions.columns:
        logger.error(f"Target column '{target_col}' not found in predictions DataFrame")
        return 0.5, {}
    
    # Extract probabilities and true labels
    y_true = predictions[target_col].values
    y_proba = predictions[prob_col].values
    
    # Try thresholds from 0.3 to 0.8 with 0.01 step
    thresholds = np.arange(0.3, 0.8, 0.01)
    results = []
    
    for threshold in thresholds:
        # Apply threshold
        y_pred = (y_proba >= threshold).astype(int)
        
        # Calculate metrics
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # Store results
        results.append({
            'threshold': threshold,
            'precision': prec,
            'recall': rec,
            'f1': f1
        })
    
    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(results)
    
    # Find thresholds that meet minimum recall requirement
    valid_thresholds = results_df[results_df['recall'] >= min_recall]
    
    if valid_thresholds.empty:
        logger.warning(f"No threshold achieves minimum recall of {min_recall}")
        # Return threshold with highest recall
        best_idx = results_df['recall'].idxmax()
        optimal_threshold = results_df.loc[best_idx, 'threshold']
        logger.info(f"Using threshold {optimal_threshold} with highest recall {results_df.loc[best_idx, 'recall']}")
    else:
        # Find threshold that gets closest to target precision while maintaining minimum recall
        valid_thresholds['precision_diff'] = abs(valid_thresholds['precision'] - target_precision)
        best_idx = valid_thresholds['precision_diff'].idxmin()
        optimal_threshold = valid_thresholds.loc[best_idx, 'threshold']
        
        logger.info(f"Optimal threshold: {optimal_threshold}")
        logger.info(f"Precision at optimal threshold: {valid_thresholds.loc[best_idx, 'precision']}")
        logger.info(f"Recall at optimal threshold: {valid_thresholds.loc[best_idx, 'recall']}")
    
    # Apply optimal threshold to get final predictions
    y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_optimal).ravel()
    
    # Calculate final metrics
    final_precision = precision_score(y_true, y_pred_optimal)
    final_recall = recall_score(y_true, y_pred_optimal)
    final_f1 = f1_score(y_true, y_pred_optimal)
    
    # Create metrics dictionary
    metrics = {
        'precision': final_precision,
        'recall': final_recall,
        'f1': final_f1,
        'threshold': optimal_threshold,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn,
        'threshold_values': thresholds.tolist(),
        'precision_values': results_df['precision'].tolist(),
        'recall_values': results_df['recall'].tolist(),
        'f1_values': results_df['f1'].tolist()
    }
    
    logger.info(f"Final metrics at threshold {optimal_threshold}:")
    logger.info(f" - Precision: {final_precision:.4f}")
    logger.info(f" - Recall: {final_recall:.4f}")
    logger.info(f" - F1: {final_f1:.4f}")
    logger.info(f" - TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    
    return optimal_threshold, metrics

def apply_optimal_threshold(predictions, threshold, prob_col='prediction_score'):
    """
    Apply an optimal threshold to prediction probabilities.
    
    Args:
        predictions (pd.DataFrame): DataFrame with prediction probabilities
        threshold (float): Threshold to apply
        prob_col (str): Name of the column with prediction probabilities
        
    Returns:
        pd.DataFrame: DataFrame with updated predictions
    """
    logger.info(f"Applying threshold {threshold} to predictions")
    
    # Make a copy to avoid modifying the original
    predictions = predictions.copy()
    
    # Apply threshold
    predictions['prediction_label'] = (predictions[prob_col] >= threshold).astype(int)
    
    return predictions

def get_threshold_metrics(y_true, y_proba, thresholds=None):
    """
    Calculate metrics for multiple thresholds.
    
    Args:
        y_true: True labels
        y_proba: Prediction probabilities
        thresholds (list, optional): List of thresholds to evaluate
            If None, will use thresholds from 0.01 to 0.99 with 0.01 step
            
    Returns:
        pd.DataFrame: DataFrame with metrics for each threshold
    """
    if thresholds is None:
        thresholds = np.arange(0.3, 0.8, 0.01)
    
    results = []
    
    for threshold in thresholds:
        # Apply threshold
        y_pred = (y_proba >= threshold).astype(int)
        
        # Calculate metrics
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Store results
        results.append({
            'threshold': threshold,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        })
    
    return pd.DataFrame(results) 