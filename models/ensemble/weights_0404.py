"""
Dynamic Weight Calculation - Version 0404

Functions for calculating dynamic weights for ensemble models including MLP.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import Dict, List, Tuple, Optional, Union
import mlflow

from utils.logger import ExperimentLogger
from models.ensemble.thresholds import tune_threshold_for_precision_optimized


def compute_precision_focused_weights_optimized(p_xgb, p_tabnet, p_lgb, p_extra, p_mlp, y_true, target_precision, required_recalls, logger=None):
    """
    Compute weights with strong focus on precision, including MLP model
    """
    if logger is None:
        logger = ExperimentLogger(experiment_name="ensemble_weights_0404")
        
    logger.info("Computing precision-focused weights...")
    xgb_recall = required_recalls[0]
    lgb_recall = required_recalls[1] 
    tabnet_recall = required_recalls[2]
    extra_recall = required_recalls[3]
    mlp_recall = required_recalls[4]

    # Find precision-optimal thresholds
    logger.info("Tuning thresholds for XGBoost...")
    xgb_threshold, xgb_metrics = tune_threshold_for_precision_optimized(p_xgb, y_true, target_precision, xgb_recall)
    logger.info("Tuning thresholds for TabNet...")
    tabnet_threshold, tabnet_metrics = tune_threshold_for_precision_optimized(p_tabnet, y_true, target_precision, tabnet_recall)
    logger.info("Tuning thresholds for LightGBM...")
    lgb_threshold, lgb_metrics = tune_threshold_for_precision_optimized(p_lgb, y_true, target_precision, lgb_recall)
    logger.info("Tuning thresholds for Extra Model...")
    extra_threshold, extra_metrics = tune_threshold_for_precision_optimized(p_extra, y_true, target_precision, extra_recall)
    logger.info("Tuning thresholds for MLP Model...")
    mlp_threshold, mlp_metrics = tune_threshold_for_precision_optimized(p_mlp, y_true, target_precision, mlp_recall)

    # Calculate weight based on precision^2 (to emphasize precision differences)
    xgb_weight = xgb_metrics['precision'] ** 2
    tabnet_weight = tabnet_metrics['precision'] ** 2
    lgb_weight = lgb_metrics['precision'] ** 2
    extra_weight = extra_metrics['precision'] ** 2
    mlp_weight = mlp_metrics['precision'] ** 2

    # Ensure minimum contribution from each model (5%)
    total_weight = xgb_weight + tabnet_weight + lgb_weight + extra_weight + mlp_weight
    xgb_weight = max(0.05, xgb_weight / total_weight)
    tabnet_weight = max(0.05, tabnet_weight / total_weight)
    lgb_weight = max(0.05, lgb_weight / total_weight)
    extra_weight = max(0.05, extra_weight / total_weight)
    mlp_weight = max(0.05, mlp_weight / total_weight)

    # Renormalize
    total_weight = xgb_weight + tabnet_weight + lgb_weight + extra_weight + mlp_weight
    weights = {
        'xgb': xgb_weight / total_weight,
        'tabnet': tabnet_weight / total_weight,
        'lgb': lgb_weight / total_weight,
        'extra': extra_weight / total_weight,
        'mlp': mlp_weight / total_weight
    }
    thresholds = {
        'xgb': xgb_threshold,
        'tabnet': tabnet_threshold,
        'lgb': lgb_threshold,
        'extra': extra_threshold,
        'mlp': mlp_threshold
    }

    # Log metrics if MLflow run is active
    if mlflow.active_run():
        metrics_log = {
            'xgb_precision': xgb_metrics['precision'],
            'tabnet_precision': tabnet_metrics['precision'], 
            'lgb_precision': lgb_metrics['precision'],
            'extra_precision': extra_metrics['precision'],
            'mlp_precision': mlp_metrics['precision'],
            'xgb_recall': xgb_metrics['recall'],
            'tabnet_recall': tabnet_metrics['recall'],
            'lgb_recall': lgb_metrics['recall'], 
            'extra_recall': extra_metrics['recall'],
            'mlp_recall': mlp_metrics['recall']
        }
        mlflow.log_metrics(metrics_log)
        for model, weight in weights.items():
            mlflow.log_metric(f"dynamic_weight_{model}", weight)

    logger.info("Precision-focused weights calculated:")
    for model, weight in weights.items():
        logger.info(f"  {model}: {weight:.4f}")

    return weights, thresholds

# Remove or comment out the old 4-model versions if they exist
# e.g., compute_dynamic_weights, compute_precision_focused_weights