"""
Dynamic Weight Calculation - Version 0412

Functions for calculating dynamic weights for ensemble models including MLP, PyTorch, and SVM.
"""

import mlflow

from src.models.ensemble.thresholds import tune_threshold_for_precision_optimized
from src.utils.logger import ExperimentLogger


def compute_precision_focused_weights_optimized(
    p_xgb, p_tabnet, p_lgb, p_extra, p_mlp, p_pytorch, p_svm, p_fnn,
    y_true, target_precision, required_recalls, logger=None
):
    """
    Compute weights with strong focus on precision, including MLP, PyTorch and SVM models.
    Now handles 7 base models.
    """
    if logger is None:
        logger = ExperimentLogger(experiment_name="ensemble_weights_0412")

    logger.info("Computing precision-focused weights for 8 models...")
    
    # Ensure required_recalls has 7 elements
    num_models = 8
    if len(required_recalls) != num_models:
        raise ValueError(f"Expected required_recalls list to have {num_models} elements, got {len(required_recalls)}")
        
    xgb_recall = required_recalls[0]
    lgb_recall = required_recalls[1]
    tabnet_recall = required_recalls[2]
    extra_recall = required_recalls[3]
    mlp_recall = required_recalls[4]
    pytorch_recall = required_recalls[5]
    svm_recall = required_recalls[6]
    fnn_recall = required_recalls[7]

    # Find precision-optimal thresholds
    logger.info("Tuning thresholds for XGBoost...")
    xgb_threshold, xgb_metrics = tune_threshold_for_precision_optimized(
        p_xgb, y_true, target_precision, xgb_recall
    )
    logger.info("Tuning thresholds for TabNet...")
    tabnet_threshold, tabnet_metrics = tune_threshold_for_precision_optimized(
        p_tabnet, y_true, target_precision, tabnet_recall
    )
    logger.info("Tuning thresholds for LightGBM...")
    lgb_threshold, lgb_metrics = tune_threshold_for_precision_optimized(
        p_lgb, y_true, target_precision, lgb_recall
    )
    logger.info("Tuning thresholds for Extra Model...")
    extra_threshold, extra_metrics = tune_threshold_for_precision_optimized(
        p_extra, y_true, target_precision, extra_recall
    )
    logger.info("Tuning thresholds for MLP Model...")
    mlp_threshold, mlp_metrics = tune_threshold_for_precision_optimized(
        p_mlp, y_true, target_precision, mlp_recall
    )
    logger.info("Tuning thresholds for PyTorch Model...")
    pytorch_threshold, pytorch_metrics = tune_threshold_for_precision_optimized(
        p_pytorch, y_true, target_precision, pytorch_recall
    )
    logger.info("Tuning thresholds for SVM Model...")
    svm_threshold, svm_metrics = tune_threshold_for_precision_optimized(
        p_svm, y_true, target_precision, svm_recall
    )
    logger.info("Tuning thresholds for FNN Model...")
    fnn_threshold, fnn_metrics = tune_threshold_for_precision_optimized(
        p_fnn, y_true, target_precision, fnn_recall
    )

    # Calculate weight based on precision^2 (to emphasize precision differences)
    xgb_weight = xgb_metrics["precision"] ** 2
    tabnet_weight = tabnet_metrics["precision"] ** 2
    lgb_weight = lgb_metrics["precision"] ** 2
    extra_weight = extra_metrics["precision"] ** 2
    mlp_weight = mlp_metrics["precision"] ** 2
    pytorch_weight = pytorch_metrics["precision"] ** 2
    svm_weight = svm_metrics["precision"] ** 2
    fnn_weight = fnn_metrics["precision"] ** 2

    # Ensure minimum contribution from each model (e.g., 5% -> 1/num_models? Let's keep 5% for now)
    min_contrib = 0.05 
    total_weight = xgb_weight + tabnet_weight + lgb_weight + extra_weight + mlp_weight + pytorch_weight + svm_weight + fnn_weight
    if total_weight <= 0: # Avoid division by zero if all precisions are 0
        logger.warning("All base model precisions are zero. Assigning equal weights.")
        weights = {m: 1.0/num_models for m in ["xgb", "tabnet", "lgb", "extra", "mlp", "pytorch", "svm", "fnn"]}
        thresholds = {
            "xgb": xgb_threshold, "tabnet": tabnet_threshold, "lgb": lgb_threshold,
            "extra": extra_threshold, "mlp": mlp_threshold, "pytorch": pytorch_threshold,
            "svm": svm_threshold, "fnn": fnn_threshold
        }
        return weights, thresholds

    xgb_weight = max(min_contrib, xgb_weight / total_weight)
    tabnet_weight = max(min_contrib, tabnet_weight / total_weight)
    lgb_weight = max(min_contrib, lgb_weight / total_weight)
    extra_weight = max(min_contrib, extra_weight / total_weight)
    mlp_weight = max(min_contrib, mlp_weight / total_weight)
    pytorch_weight = max(min_contrib, pytorch_weight / total_weight)
    svm_weight = max(min_contrib, svm_weight / total_weight)
    fnn_weight = max(min_contrib, fnn_weight / total_weight)

    # Renormalize
    total_weight = xgb_weight + tabnet_weight + lgb_weight + extra_weight + mlp_weight + pytorch_weight + svm_weight + fnn_weight
    weights = {
        "xgb": xgb_weight / total_weight,
        "tabnet": tabnet_weight / total_weight,
        "lgb": lgb_weight / total_weight,
        "extra": extra_weight / total_weight,
        "mlp": mlp_weight / total_weight,
        "pytorch": pytorch_weight / total_weight,
        "svm": svm_weight / total_weight,
        "fnn": fnn_weight / total_weight,
    }
    thresholds = {
        "xgb": xgb_threshold,
        "tabnet": tabnet_threshold,
        "lgb": lgb_threshold,
        "extra": extra_threshold,
        "mlp": mlp_threshold,
        "pytorch": pytorch_threshold,
        "svm": svm_threshold,
        "fnn": fnn_threshold,
    }

    # Log metrics if MLflow run is active
    if mlflow.active_run():
        metrics_log = {
            "xgb_precision": xgb_metrics["precision"],
            "tabnet_precision": tabnet_metrics["precision"],
            "lgb_precision": lgb_metrics["precision"],
            "extra_precision": extra_metrics["precision"],
            "mlp_precision": mlp_metrics["precision"],
            "pytorch_precision": pytorch_metrics["precision"],
            "svm_precision": svm_metrics["precision"],
            "xgb_recall": xgb_metrics["recall"],
            "tabnet_recall": tabnet_metrics["recall"],
            "lgb_recall": lgb_metrics["recall"],
            "extra_recall": extra_metrics["recall"],
            "mlp_recall": mlp_metrics["recall"],
            "pytorch_recall": pytorch_metrics["recall"],
            "svm_recall": svm_metrics["recall"],
            "fnn_recall": fnn_metrics["recall"],
        }
        mlflow.log_metrics(metrics_log)
        for model, weight in weights.items():
            mlflow.log_metric(f"dynamic_weight_{model}", weight)

    logger.info("Precision-focused weights calculated:")
    for model, weight in weights.items():
        logger.info(f"  {model}: {weight:.4f}")

    return weights, thresholds