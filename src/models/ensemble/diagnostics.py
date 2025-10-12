"""
Model Diagnostics Utilities

Functions for diagnosing and explaining model predictions.
"""

from typing import TYPE_CHECKING, Optional

import mlflow
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import confusion_matrix

if TYPE_CHECKING:
    from src.utils.logger import ExperimentLogger

from src.utils.logger import ExperimentLogger

# Constants
DEFAULT_LOG_DIR = "./logs/ensemble_model_diagnostics"


def _prepare_shap_data(x_val: pd.DataFrame, max_samples: int = 500) -> pd.DataFrame:
    """Prepare data for SHAP analysis by subsampling if needed."""
    if len(x_val) > max_samples:
        return x_val.sample(max_samples, random_state=42)
    return x_val


def _create_ensemble_meta_features(
    model, x_sample: pd.DataFrame, logger
) -> tuple[np.ndarray, list[str]]:
    """Create meta-features for ensemble model SHAP analysis."""
    # Get prepared data
    x_prepared = x_sample
    if hasattr(model, "selected_features"):
        if isinstance(model.selected_features, list) and len(model.selected_features) > 0:
            x_prepared = x_sample[model.selected_features]

    # Generate predictions from base models
    xgb_model = getattr(model, "model_xgb_calibrated", None) or model.model_xgb
    cat_model = getattr(model, "model_cat_calibrated", None) or model.model_cat
    lgb_model = getattr(model, "model_lgb_calibrated", None) or model.model_lgb
    extra_model = getattr(model, "model_extra_calibrated", None) or model.model_extra

    if (
        hasattr(model, "extra_base_model_type")
        and model.extra_base_model_type in ["mlp", "svm"]
        and model.extra_model_scaler is not None
    ):
        x_scaled = model.extra_model_scaler.transform(x_prepared)
        p_extra = extra_model.predict_proba(x_scaled)[:, 1]
    else:
        p_extra = extra_model.predict_proba(x_prepared)[:, 1]

    p_xgb = xgb_model.predict_proba(x_prepared)[:, 1]
    p_cat = cat_model.predict_proba(x_prepared)[:, 1]
    p_lgb = lgb_model.predict_proba(x_prepared)[:, 1]

    # Create meta-features from base model predictions
    from models.ensemble.meta_features import create_meta_features

    dynamic_weights = None
    if hasattr(model, "dynamic_weighting") and hasattr(model, "dynamic_weights"):
        if model.dynamic_weighting:
            dynamic_weights = model.dynamic_weights

    meta_features = create_meta_features(p_xgb, p_cat, p_lgb, p_extra, dynamic_weights)

    feature_names = [
        "prob_xgb", "prob_cat", "prob_lgb", "prob_extra",
        "weighted_avg", "diff_xgb_cat", "diff_xgb_lgb", "diff_cat_lgb",
        "diff_extra_xgb", "diff_extra_cat", "diff_extra_lgb",
        "max_prob", "min_prob", "range_prob",
        "rank_xgb", "rank_cat", "rank_lgb", "rank_extra",
        "vote_sum", "vote_agreement",
    ]

    logger.info(f"Created meta-features with shape: {meta_features.shape}")
    return meta_features, feature_names


def _compute_shap_values(target_model, x_for_shap: np.ndarray, logger) -> np.ndarray:
    """Compute SHAP values for the given model and data."""
    try:
        if hasattr(target_model, "tree_method"):
            # For tree-based models (XGBoost, LightGBM)
            explainer = shap.TreeExplainer(target_model)
        else:
            # For other models
            explainer = shap.KernelExplainer(
                target_model.predict_proba, shap.sample(x_for_shap, 100, random_state=42)
            )

        # Calculate SHAP values
        shap_values = explainer.shap_values(x_for_shap)

        # For binary classifiers, shap_values might be a list with one element
        if isinstance(shap_values, list):
            # Take SHAP values for positive class (class 1)
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        # Check if shap_values is a numpy array before proceeding
        if not isinstance(shap_values, np.ndarray):
            raise TypeError("SHAP values must be a numpy array")

        return shap_values

    except Exception as e:
        logger.error(f"SHAP computation failed: {str(e)}")
        raise


def _calculate_feature_importance(
    shap_values: np.ndarray, feature_names: list[str], logger
) -> dict[str, float]:
    """Calculate feature importance from SHAP values."""
    # Calculate mean absolute SHAP values for feature importance
    feature_importance = np.abs(shap_values).mean(axis=0)

    # Ensure feature_importance is 1D and matches feature_names length
    if feature_importance.ndim > 1:
        feature_importance = feature_importance.mean(axis=1)

    if len(feature_importance) != len(feature_names):
        logger.warning(
            f"Feature importance length ({len(feature_importance)}) doesn't match feature names length ({len(feature_names)})"
        )
        # Adjust feature_names if needed
        feature_names = (
            feature_names[: len(feature_importance)]
            if len(feature_names) > len(feature_importance)
            else feature_names
        )

    # Create feature importance dictionary
    importance_dict = dict(zip(feature_names, feature_importance))

    # Sort features by importance
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

    # Log top 10 features
    logger.info("Top feature importances (SHAP):")
    for feature, importance in sorted_importance[:10]:
        logger.info(f"  {feature}: {importance:.6f}")
        mlflow.log_metric(f"shap_importance_{feature}", importance)

    return dict(sorted_importance)


def _get_model_predictions(model, x_val: pd.DataFrame, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Get probability predictions and binary predictions from model."""
    # Get predictions
    y_prob = model.predict_proba(x_val)

    # Handle different return shapes from predict_proba
    if isinstance(y_prob, tuple):
        # Some models return (neg_class_prob, pos_class_prob)
        y_prob = y_prob[1]
    elif y_prob.ndim > 1 and y_prob.shape[1] > 1:
        # Some models return [neg_class_prob, pos_class_prob] for each sample
        y_prob = y_prob[:, 1]

    # Convert to binary predictions using threshold
    y_pred = (y_prob >= threshold).astype(int)

    return y_prob, y_pred


def _compute_confusion_metrics(y_val: pd.Series, y_pred: np.ndarray) -> dict:
    """Compute confusion matrix and basic classification metrics."""
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

    # Compute various metrics
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "confusion_matrix": (tn, fp, fn, tp),
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "total": total,
        }
    }


def _analyze_errors(y_prob: np.ndarray, y_pred: np.ndarray, y_val: pd.Series, threshold: float) -> dict:
    """Analyze prediction errors and create detailed error analysis."""
    # Calculate class balance
    pos_rate = np.mean(y_val)

    # Find incorrect predictions
    incorrect_mask = y_pred != y_val
    incorrect_indices = np.nonzero(incorrect_mask)[0]

    # False positives and false negatives
    fp_mask = (y_pred == 1) & (y_val == 0)
    fn_mask = (y_pred == 0) & (y_val == 1)
    fp_indices = np.nonzero(fp_mask)[0]
    fn_indices = np.nonzero(fn_mask)[0]

    # Note: fp_indices and fn_indices can be used to extract examples if needed

    # Analyze false positives (highest probability first)
    fp_probs = y_prob[fp_mask]
    fp_indices_sorted = np.argsort(-fp_probs)  # Sort in descending order
    fp_analysis = []

    for i in range(min(5, len(fp_indices_sorted))):
        idx = fp_indices[fp_indices_sorted[i]]
        prob = y_prob[idx]
        fp_analysis.append(
            {"idx": idx, "probability": prob, "threshold_difference": prob - threshold}
        )

    # Analyze false negatives (closest to threshold first)
    fn_probs = y_prob[fn_mask]
    fn_indices_sorted = np.argsort(threshold - fn_probs)  # Sort by proximity to threshold
    fn_analysis = []

    for i in range(min(5, len(fn_indices_sorted))):
        idx = fn_indices[fn_indices_sorted[i]]
        prob = y_prob[idx]
        fn_analysis.append(
            {"idx": idx, "probability": prob, "threshold_difference": threshold - prob}
        )

    return {
        "class_balance": pos_rate,
        "error_count": len(incorrect_indices),
        "error_rate": len(incorrect_indices) / len(y_val),
        "false_positives": len(fp_indices),
        "false_negatives": len(fn_indices),
        "fp_analysis": fp_analysis,
        "fn_analysis": fn_analysis,
    }


def detect_data_leakage(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    x_val: pd.DataFrame,
    logger: Optional["ExperimentLogger"] = None,
) -> dict:
    """
    Check for potential data leakage between datasets by detecting duplicate rows.

    Args:
        x_train: Training dataset
        x_test: Test dataset
        x_val: Validation dataset
        logger: Logger instance

    Returns:
        Dictionary with overlap information
    """
    if logger is None:
        logger = ExperimentLogger(
            experiment_name="ensemble_model_diagnostics",
            log_dir=DEFAULT_LOG_DIR,
        )

    logger.info("Checking for data leakage between datasets...")

    # Create unique identifier for each row (converting to tuples)
    train_tuples = set(map(tuple, x_train.values))
    test_tuples = set(map(tuple, x_test.values))
    val_tuples = set(map(tuple, x_val.values))

    # Find overlaps
    train_test_overlap = train_tuples.intersection(test_tuples)
    train_val_overlap = train_tuples.intersection(val_tuples)
    test_val_overlap = test_tuples.intersection(val_tuples)

    # Calculate overlap percentages
    train_test_pct = len(train_test_overlap) / len(train_tuples) * 100 if train_tuples else 0
    train_val_pct = len(train_val_overlap) / len(train_tuples) * 100 if train_tuples else 0
    test_val_pct = len(test_val_overlap) / len(test_tuples) * 100 if test_tuples else 0

    # Calculate maximum overlap percentage for threshold checking
    max_overlap_pct = max(train_test_pct, train_val_pct, test_val_pct)

    results = {
        "train_test_overlap": len(train_test_overlap),
        "train_val_overlap": len(train_val_overlap),
        "test_val_overlap": len(test_val_overlap),
        "train_test_overlap_pct": train_test_pct,
        "train_val_overlap_pct": train_val_pct,
        "test_val_overlap_pct": test_val_pct,
        "overlap_percentage": max_overlap_pct,  # Add this key to fix the KeyError
    }

    # Log findings
    logger.info(f"Train-Test overlap: {len(train_test_overlap)} rows ({train_test_pct:.2f}%)")
    logger.info(f"Train-Val overlap: {len(train_val_overlap)} rows ({train_val_pct:.2f}%)")
    logger.info(f"Test-Val overlap: {len(test_val_overlap)} rows ({test_val_pct:.2f}%)")

    # Log to MLflow
    mlflow.log_metrics(
        {
            "train_test_overlap_pct": train_test_pct,
            "train_val_overlap_pct": train_val_pct,
            "test_val_overlap_pct": test_val_pct,
        }
    )

    # Warning for significant overlap
    if max(train_test_pct, train_val_pct, test_val_pct) > 5:
        logger.warning("Significant data overlap detected! This may cause evaluation bias.")

    return results


def explain_predictions(model, x_val: pd.DataFrame, logger: Optional["ExperimentLogger"] = None) -> dict:
    """
    Generate feature importance explanations using SHAP values on validation data.

    Args:
        model: Trained model with predict_proba method
        x_val: Validation features
        logger: Logger instance

    Returns:
        Dictionary with explanation results
    """
    if logger is None:
        logger = ExperimentLogger(
            experiment_name="ensemble_model_diagnostics",
            log_dir=DEFAULT_LOG_DIR,
        )

    logger.info("Generating model explanations with SHAP...")

    # Prepare data for SHAP analysis
    x_sample = _prepare_shap_data(x_val)
    if len(x_sample) < len(x_val):
        logger.info(f"Using {len(x_sample)} random samples for SHAP analysis.")

    try:
        # Check if the model has the meta_learner attribute (ensemble model)
        if hasattr(model, "meta_learner") and model.meta_learner is not None:
            # For ensemble model, use meta-features for SHAP analysis
            target_model = model.meta_learner
            logger.info("Using meta-learner for SHAP explanations.")

            logger.info("Transforming features to meta-features for ensemble explanation")
            x_for_shap, feature_names = _create_ensemble_meta_features(model, x_sample, logger)
        else:
            # Otherwise use the model directly with original features
            target_model = model
            x_for_shap = x_sample.values if hasattr(x_sample, 'values') else x_sample
            feature_names = x_sample.columns.tolist()

        # Compute SHAP values
        if isinstance(x_for_shap, pd.DataFrame):
            x_for_shap_array = x_for_shap.values
        elif isinstance(x_for_shap, np.ndarray):
            x_for_shap_array = x_for_shap
        else:
            x_for_shap_array = np.asarray(x_for_shap)

        shap_values = _compute_shap_values(target_model, x_for_shap_array, logger)

        # Calculate feature importance
        feature_importance = _calculate_feature_importance(shap_values, feature_names, logger)

        return {"feature_importance": feature_importance, "shap_values": shap_values}

    except Exception as e:
        logger.error(f"SHAP explanation failed: {str(e)}")
        return {"error": str(e), "feature_importance": {}}


def analyze_prediction_errors(
    model,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    threshold: Optional[float] = None,
    logger: Optional["ExperimentLogger"] = None,
) -> dict:
    """
    Analyze prediction errors on the validation set (most recent data).

    Args:
        model: Trained model with predict_proba method
        x_val: Validation features
        y_val: Validation target values
        threshold: Classification threshold (default: model.optimal_threshold or 0.5)
        logger: Logger instance

    Returns:
        Dictionary with error analysis results
    """
    if logger is None:
        logger = ExperimentLogger(
            experiment_name="ensemble_model_diagnostics",
            log_dir=DEFAULT_LOG_DIR,
        )

    logger.info("Analyzing prediction errors...")

    # Get model threshold
    if threshold is None:
        if hasattr(model, "optimal_threshold"):
            threshold = model.optimal_threshold
        else:
            threshold = 0.5

    logger.info(f"Using classification threshold: {threshold:.4f}")

    # Ensure threshold is not None
    assert threshold is not None, "Threshold must be provided"

    # Get predictions
    y_prob, y_pred = _get_model_predictions(model, x_val, threshold)

    # Compute confusion matrix and metrics
    confusion_result = _compute_confusion_metrics(y_val, y_pred)
    tn, fp, fn, tp = confusion_result["confusion_matrix"]
    metrics = confusion_result["metrics"]

    # Analyze errors
    error_analysis = _analyze_errors(y_prob, y_pred, y_val, threshold)

    # Log results
    logger.info("Error analysis results:")
    logger.info(f"  Total samples: {metrics['total']}")
    logger.info(f"  Class balance: {error_analysis['class_balance']:.2%} positive")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    logger.info(f"  False Positive Rate: {fp / (fp + tn):.4f}")
    logger.info(f"  False Negative Rate: {fn / (fn + tp):.4f}")

    # Log to MLflow
    mlflow.log_metrics(
        {
            "error_analysis_accuracy": metrics["accuracy"],
            "error_analysis_precision": metrics["precision"],
            "error_analysis_recall": metrics["recall"],
            "error_analysis_f1": metrics["f1_score"],
            "error_analysis_fps": fp,
            "error_analysis_fns": fn,
        }
    )

    # Return compiled results
    return {
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "metrics": {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
            "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else 0,
        },
        "class_balance": error_analysis["class_balance"],
        "threshold": threshold,
        "error_count": error_analysis["error_count"],
        "error_rate": error_analysis["error_rate"],
        "false_positives": error_analysis["false_positives"],
        "false_negatives": error_analysis["false_negatives"],
        "fp_analysis": error_analysis["fp_analysis"],
        "fn_analysis": error_analysis["fn_analysis"],
    }
