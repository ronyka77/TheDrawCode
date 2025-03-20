"""
Dynamic Weight Calculation

Functions for calculating dynamic weights for ensemble models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import Dict, List, Tuple, Optional, Union
import mlflow

from utils.logger import ExperimentLogger
from models.ensemble.thresholds import tune_threshold_for_precision

def compute_dynamic_weights(p_xgb: np.ndarray, p_cat: np.ndarray, 
                            p_lgb: np.ndarray, p_extra: np.ndarray, 
                            targets: pd.Series,
                            logger: ExperimentLogger = None) -> Dict:
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
    if logger is None:
        logger = ExperimentLogger(experiment_name="ensemble_model_weights",
                                log_dir="./logs/ensemble_model_weights")
    
    logger.info("Computing dynamic weights for base models...")
    
    # Find optimal threshold for each model to balance precision and recall
    def find_best_threshold(probs, labels):
        """Find threshold that maximizes F1 score"""
        best_f1 = 0
        best_threshold = 0.5
        best_precision = 0
        best_recall = 0
        
        # Try thresholds from 0.3 to 0.7 with small steps
        for threshold in np.arange(0.3, 0.71, 0.01):
            preds = (probs >= threshold).astype(int)
            prec = precision_score(labels, preds, zero_division=0)
            rec = recall_score(labels, preds, zero_division=0)
            f1 = f1_score(labels, preds, zero_division=0)
            
            # Prioritize higher precision if F1 scores are similar
            if f1 > best_f1 or (np.isclose(f1, best_f1, atol=0.01) and prec > best_precision):
                best_f1 = f1
                best_threshold = threshold
                best_precision = prec
                best_recall = rec
        
        return best_threshold, best_precision, best_recall, best_f1
    
    # Calculate metrics for each model with optimized thresholds
    xgb_threshold, xgb_precision, xgb_recall, xgb_f1 = find_best_threshold(p_xgb, targets)
    cat_threshold, cat_precision, cat_recall, cat_f1 = find_best_threshold(p_cat, targets)
    lgb_threshold, lgb_precision, lgb_recall, lgb_f1 = find_best_threshold(p_lgb, targets)
    extra_threshold, extra_precision, extra_recall, extra_f1 = find_best_threshold(p_extra, targets)
    
    # Log individual model metrics
    logger.info(f"XGBoost: threshold={xgb_threshold:.3f}, precision={xgb_precision:.4f}, recall={xgb_recall:.4f}, f1={xgb_f1:.4f}")
    logger.info(f"CatBoost: threshold={cat_threshold:.3f}, precision={cat_precision:.4f}, recall={cat_recall:.4f}, f1={cat_f1:.4f}")
    logger.info(f"LightGBM: threshold={lgb_threshold:.3f}, precision={lgb_precision:.4f}, recall={lgb_recall:.4f}, f1={lgb_f1:.4f}")
    logger.info(f"Extra Model: threshold={extra_threshold:.3f}, precision={extra_precision:.4f}, recall={extra_recall:.4f}, f1={extra_f1:.4f}")
    
    # Log to MLflow
    mlflow.log_metrics({
        'xgb_precision': xgb_precision,
        'xgb_recall': xgb_recall,
        'xgb_f1': xgb_f1,
        'xgb_threshold': xgb_threshold,
        'cat_precision': cat_precision,
        'cat_recall': cat_recall,
        'cat_f1': cat_f1,
        'cat_threshold': cat_threshold,
        'lgb_precision': lgb_precision,
        'lgb_recall': lgb_recall,
        'lgb_f1': lgb_f1,
        'lgb_threshold': lgb_threshold,
        'extra_precision': extra_precision,
        'extra_recall': extra_recall,
        'extra_f1': extra_f1,
        'extra_threshold': extra_threshold
    })
    
    # Calculate composite score (weighted combination of precision, recall, and F1)
    # Precision has higher weight to focus on high-precision predictions
    def composite_score(precision, recall, f1):
        return 0.6 * precision + 0.2 * recall + 0.2 * f1
    
    xgb_score = composite_score(xgb_precision, xgb_recall, xgb_f1)
    cat_score = composite_score(cat_precision, cat_recall, cat_f1)
    lgb_score = composite_score(lgb_precision, lgb_recall, lgb_f1)
    extra_score = composite_score(extra_precision, extra_recall, extra_f1)
    
    # Calculate raw weights based on composite scores
    total_score = xgb_score + cat_score + lgb_score + extra_score
    
    if total_score > 0:
        raw_weights = {
            'xgb': xgb_score / total_score,
            'cat': cat_score / total_score,
            'lgb': lgb_score / total_score,
            'extra': extra_score / total_score
        }
    else:
        # Fallback to equal weights if total score is 0
        raw_weights = {'xgb': 0.25, 'cat': 0.25, 'lgb': 0.25, 'extra': 0.25}
    
    # Apply smoothing to avoid extreme weights
    # Ensure each model gets at least 10% weight
    min_weight = 0.1
    max_weight = 0.5
    
    total_adjustment = 0
    for model in raw_weights:
        if raw_weights[model] < min_weight:
            total_adjustment += (min_weight - raw_weights[model])
        elif raw_weights[model] > max_weight:
            total_adjustment -= (raw_weights[model] - max_weight)
    
    # Adjust weights to ensure they sum to 1 and respect min/max constraints
    adjusted_weights = {}
    for model in raw_weights:
        if raw_weights[model] < min_weight:
            adjusted_weights[model] = min_weight
        elif raw_weights[model] > max_weight:
            adjusted_weights[model] = max_weight
        else:
            # Scale remaining weights proportionally
            remaining_weight = 1.0 - sum([
                min_weight if w < min_weight else max_weight if w > max_weight else 0 
                for m, w in raw_weights.items()
            ])
            remaining_raw = sum([
                w for m, w in raw_weights.items() 
                if min_weight <= w <= max_weight
            ])
            
            if remaining_raw > 0:
                adjusted_weights[model] = remaining_weight * (raw_weights[model] / remaining_raw)
            else:
                # Fallback if all weights were outside min/max bounds
                adjusted_weights[model] = 0.25
    
    # Normalize weights to ensure they sum to 1.0
    weight_sum = sum(adjusted_weights.values())
    normalized_weights = {k: v / weight_sum for k, v in adjusted_weights.items()}
    
    # Log the dynamic weights
    logger.info("Dynamic weights calculated:")
    for model, weight in normalized_weights.items():
        logger.info(f"  {model}: {weight:.4f}")
        mlflow.log_metric(f"dynamic_weight_{model}", weight)
    
    return normalized_weights

def compute_precision_focused_weights(p_xgb, p_cat, p_lgb, p_extra, y_true, target_precision, required_recall, logger=None):
    """
    Compute weights with strong focus on precision
    """
    if logger is None:
        logger = ExperimentLogger(experiment_name="ensemble_model_weights",
                                log_dir="./logs/ensemble_model_weights")
    
    # Find precision-optimal thresholds
    xgb_threshold, xgb_metrics = tune_threshold_for_precision(p_xgb, y_true, target_precision, required_recall)
    cat_threshold, cat_metrics = tune_threshold_for_precision(p_cat, y_true, target_precision, required_recall)
    lgb_threshold, lgb_metrics = tune_threshold_for_precision(p_lgb, y_true, target_precision, required_recall)
    extra_threshold, extra_metrics = tune_threshold_for_precision(p_extra, y_true, target_precision, required_recall)
    
    # Calculate weight based on precision^2 (to emphasize precision differences)
    xgb_weight = xgb_metrics['precision']**2
    cat_weight = cat_metrics['precision']**2
    lgb_weight = lgb_metrics['precision']**2
    extra_weight = extra_metrics['precision']**2
    
    # Ensure minimum contribution from each model (5%)
    total_weight = xgb_weight + cat_weight + lgb_weight + extra_weight
    xgb_weight = max(0.05, xgb_weight / total_weight)
    cat_weight = max(0.05, cat_weight / total_weight)
    lgb_weight = max(0.05, lgb_weight / total_weight)
    extra_weight = max(0.05, extra_weight / total_weight)
    
    # Renormalize
    total_weight = xgb_weight + cat_weight + lgb_weight + extra_weight
    weights = {
        'xgb': xgb_weight / total_weight,
        'cat': cat_weight / total_weight,
        'lgb': lgb_weight / total_weight,
        'extra': extra_weight / total_weight
    }
    logger.info("Precision-focused weights calculated:")
    for model, weight in weights.items():
        logger.info(f"  {model}: {weight:.4f}")
    return weights
