"""
Threshold Optimization

Functions for optimizing classification thresholds.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import mlflow
from typing import Dict, List, Tuple, Optional, Union
from utils.logger import ExperimentLogger

def tune_threshold(probs: np.ndarray, targets: pd.Series, 
                    grid_start: float = 0.0, grid_stop: float = 1.0, 
                    grid_step: float = 0.01,
                    target_precision: float = 0.50,
                    min_recall: float = 0.40,
                    logger: ExperimentLogger = None) -> Tuple[float, Dict]:
    """
    Tune the global threshold by scanning a grid.
    
    Args:
        probs: Predicted probabilities
        targets: True binary labels
        grid_start: Start of threshold grid
        grid_stop: End of threshold grid
        grid_step: Step size for threshold grid
        target_precision: Target precision to achieve
        min_recall: Minimum recall required
        logger: Logger instance
        
    Returns:
        Tuple of (optimal_threshold, metrics_at_threshold)
    """
    if logger is None:
        logger = ExperimentLogger(experiment_name="ensemble_model_thresholds",
                                log_dir="./logs/ensemble_model_thresholds")
    
    logger.info(f"Tuning threshold with target precision {target_precision:.2f}...")
    
    # Track metrics for each threshold
    thresholds = []
    metrics = []
    
    # Initialize best threshold and metrics
    best_threshold = 0.5
    best_metrics = None
    best_score = -1  # Score to maximize
    
    # Scan threshold grid and compute metrics
    for threshold in np.arange(grid_start, grid_stop, grid_step):
        preds = (probs >= threshold).astype(int)
        
        # Calculate metrics
        prec = precision_score(targets, preds, zero_division=0)
        rec = recall_score(targets, preds, zero_division=0)
        f1 = f1_score(targets, preds, zero_division=0)
        
        # Calculate weighted score that prioritizes precision
        score = prec
        
        # Store metrics for this threshold
        threshold_metrics = {
            'threshold': threshold,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'score': score
        }
        
        thresholds.append(threshold)
        metrics.append(threshold_metrics)
        
        # Update best threshold if criteria are met
        # Criteria: Score is better, and either precision meets target or it's better than before
        if (score > best_score and rec >= min_recall):
            best_threshold = threshold
            best_metrics = threshold_metrics
            best_score = score
    
    # If no threshold met the criteria, choose threshold closest to target precision
    if best_metrics is None:
        precision_diffs = [abs(m['precision'] - target_precision) for m in metrics]
        best_idx = np.argmin(precision_diffs)
        best_threshold = thresholds[best_idx]
        best_metrics = metrics[best_idx]
    
    # Log results
    logger.info(f"Selected threshold: {best_threshold:.4f}")
    logger.info(f"Metrics at selected threshold:")
    logger.info(f"  Precision: {best_metrics['precision']:.4f}")
    logger.info(f"  Recall: {best_metrics['recall']:.4f}")
    logger.info(f"  F1 Score: {best_metrics['f1']:.4f}")
    
    # Log to MLflow
    mlflow.log_metrics({
        'tuned_threshold': best_threshold,
        'tuned_precision': best_metrics['precision'],
        'tuned_recall': best_metrics['recall'],
        'tuned_f1': best_metrics['f1']
    })
    
    return best_threshold, best_metrics

def tune_individual_threshold(probs: np.ndarray, targets: pd.Series, 
                            grid_start: float = 0.3, grid_stop: float = 0.7, 
                            grid_step: float = 0.01, min_recall: float = 0.25) -> float:
    """
    Tune threshold for a single model's probabilities.
    
    Args:
        probs: Predicted probabilities from a single model
        targets: True binary labels
        grid_start: Start of threshold grid
        grid_stop: End of threshold grid
        grid_step: Step size for threshold grid
        min_recall: Minimum recall required
        
    Returns:
        Optimal threshold for the model
    """
    best_f1 = 0
    best_threshold = 0.5
    
    # Scan threshold grid
    for threshold in np.arange(grid_start, grid_stop, grid_step):
        preds = (probs >= threshold).astype(int)
        
        # Calculate metrics
        precision = precision_score(targets, preds, zero_division=0)
        recall = recall_score(targets, preds, zero_division=0)
        
        # Skip thresholds with recall below minimum
        if recall < min_recall:
            continue
        
        # Calculate F1 score
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        
        # Update best threshold
        if precision > best_f1:
            best_f1 = precision
            best_threshold = threshold
    
    return best_threshold

def tune_threshold_for_precision(y_prob: np.ndarray, y_true: pd.Series, 
                                target_precision: float = 0.50, 
                                required_recall: float = 0.25,
                                min_threshold: float = 0.1,
                                max_threshold: float = 0.9,
                                step: float = 0.01,
                                logger: ExperimentLogger = None) -> float:
    """
    Tune the threshold to achieve a target precision with a minimum recall requirement.
    
    Args:
        y_prob: Predicted probabilities
        y_true: True binary labels
        target_precision: Target precision to achieve
        required_recall: Minimum required recall
        min_threshold: Minimum threshold to consider
        max_threshold: Maximum threshold to consider
        step: Step size for threshold grid
        logger: Logger instance
        
    Returns:
        Optimal threshold for the target precision with minimum recall
    """
    if logger is None:
        logger = ExperimentLogger(experiment_name="ensemble_model_thresholds",
                                log_dir="./logs/ensemble_model_thresholds")
    
    logger.info(f"Tuning threshold for precision {target_precision:.2f} with minimum recall {required_recall:.2f}...")
    
    # Track metrics for each threshold
    thresholds = np.arange(min_threshold, max_threshold + step, step)
    precisions = []
    recalls = []
    f1_scores = []
    
    # Scan threshold grid
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        
        # Get metrics at this threshold
        try:
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
        except Exception as e:
            logger.warning(f"Error calculating metrics at threshold {threshold}: {str(e)}")
            prec, rec, f1 = 0, 0, 0
        
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)
    
    # Find eligible thresholds that meet minimum recall requirement
    eligible_indices = [i for i, r in enumerate(recalls) if r >= required_recall]
    
    if not eligible_indices:
        logger.warning(f"No thresholds meet minimum recall of {required_recall:.2f}. Relaxing constraint.")
        # Fallback: choose threshold with best F1 score
        best_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[best_idx]
        logger.info(f"Relaxed threshold selected: {optimal_threshold:.4f} (precision={precisions[best_idx]:.4f}, recall={recalls[best_idx]:.4f})")
    else:
        # Filter thresholds that meet recall requirement
        eligible_thresholds = [thresholds[i] for i in eligible_indices]
        eligible_precisions = [precisions[i] for i in eligible_indices]
        eligible_recalls = [recalls[i] for i in eligible_indices]
        eligible_f1s = [f1_scores[i] for i in eligible_indices]
        
        # Find threshold closest to target precision
        precision_diffs = [abs(p - target_precision) for p in eligible_precisions]
        best_idx = np.argmin(precision_diffs)
        optimal_threshold = eligible_thresholds[best_idx]
        optimal_precision = eligible_precisions[best_idx]
        optimal_recall = eligible_recalls[best_idx]
        optimal_f1 = eligible_f1s[best_idx]
        logger.info(f"Optimal threshold selected: {optimal_threshold:.4f}")
        logger.info(f"Metrics at optimal threshold:")
        logger.info(f"  Precision: {optimal_precision:.4f}")
        logger.info(f"  Recall: {optimal_recall:.4f}")
        logger.info(f"  F1 Score: {optimal_f1:.4f}")
        metrics = {
            'precision': optimal_precision,
            'recall': optimal_recall,
            'f1': optimal_f1
        }
    # Log to MLflow
    mlflow.log_metrics({
        'precision_tuned_threshold': optimal_threshold,
        'precision_at_threshold': precisions[thresholds.tolist().index(optimal_threshold)],
        'recall_at_threshold': recalls[thresholds.tolist().index(optimal_threshold)]
    })
    
    return optimal_threshold, metrics
