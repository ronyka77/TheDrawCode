"""
Model Diagnostics Utilities

Functions for diagnosing and explaining model predictions.
"""

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import confusion_matrix
import mlflow
from typing import Dict, List, Tuple, Optional, Union

from utils.logger import ExperimentLogger

def detect_data_leakage(X_train: pd.DataFrame, X_test: pd.DataFrame, X_val: pd.DataFrame,
                        logger: ExperimentLogger = None) -> Dict:
    """
    Check for potential data leakage between datasets by detecting duplicate rows.
    
    Args:
        X_train: Training dataset
        X_test: Test dataset
        X_val: Validation dataset
        logger: Logger instance
        
    Returns:
        Dictionary with overlap information
    """
    if logger is None:
        logger = ExperimentLogger(experiment_name="ensemble_model_diagnostics",
                                log_dir="./logs/ensemble_model_diagnostics")
    
    logger.info("Checking for data leakage between datasets...")
    
    # Create unique identifier for each row (converting to tuples)
    train_tuples = set(map(tuple, X_train.values))
    test_tuples = set(map(tuple, X_test.values))
    val_tuples = set(map(tuple, X_val.values))
    
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
        'train_test_overlap': len(train_test_overlap),
        'train_val_overlap': len(train_val_overlap),
        'test_val_overlap': len(test_val_overlap),
        'train_test_overlap_pct': train_test_pct,
        'train_val_overlap_pct': train_val_pct,
        'test_val_overlap_pct': test_val_pct,
        'overlap_percentage': max_overlap_pct  # Add this key to fix the KeyError
    }
    
    # Log findings
    logger.info(f"Train-Test overlap: {len(train_test_overlap)} rows ({train_test_pct:.2f}%)")
    logger.info(f"Train-Val overlap: {len(train_val_overlap)} rows ({train_val_pct:.2f}%)")
    logger.info(f"Test-Val overlap: {len(test_val_overlap)} rows ({test_val_pct:.2f}%)")
    
    # Log to MLflow
    mlflow.log_metrics({
        'train_test_overlap_pct': train_test_pct,
        'train_val_overlap_pct': train_val_pct,
        'test_val_overlap_pct': test_val_pct
    })
    
    # Warning for significant overlap
    if max(train_test_pct, train_val_pct, test_val_pct) > 5:
        logger.warning("Significant data overlap detected! This may cause evaluation bias.")
    
    return results

def explain_predictions(model, X_val: pd.DataFrame, 
                        logger: ExperimentLogger = None) -> Dict:
    """
    Generate feature importance explanations using SHAP values on validation data.
    
    Args:
        model: Trained model with predict_proba method
        X_val: Validation features
        logger: Logger instance
        
    Returns:
        Dictionary with explanation results
    """
    if logger is None:
        logger = ExperimentLogger(experiment_name="ensemble_model_diagnostics",
                                log_dir="./logs/ensemble_model_diagnostics")
    
    logger.info("Generating model explanations with SHAP...")
    
    # Limit sample size for SHAP analysis to avoid excessive computation
    max_shap_samples = min(500, len(X_val))
    
    # Subsample data if needed
    if len(X_val) > max_shap_samples:
        X_sample = X_val.sample(max_shap_samples, random_state=42)
        logger.info(f"Using {max_shap_samples} random samples for SHAP analysis.")
    else:
        X_sample = X_val
    
    # Check if the model has the meta_learner attribute (ensemble model)
    if hasattr(model, 'meta_learner') and model.meta_learner is not None:
        # For ensemble model, use the meta-learner for SHAP analysis
        target_model = model.meta_learner
        logger.info("Using meta-learner for SHAP explanations.")
        
        # IMPORTANT FIX: Transform original features into meta-features first
        logger.info("Transforming features to meta-features for ensemble explanation")
        
        try:
            # Get prepared data
            X_prepared = X_sample
            if hasattr(model, 'selected_features'):
                # Prepare input data using selected features if available
                if isinstance(model.selected_features, list) and len(model.selected_features) > 0:
                    X_prepared = X_sample[model.selected_features]
            
            # Generate predictions from base models
            # Use calibrated models if available
            xgb_model = getattr(model, 'model_xgb_calibrated', None) or model.model_xgb
            cat_model = getattr(model, 'model_cat_calibrated', None) or model.model_cat
            lgb_model = getattr(model, 'model_lgb_calibrated', None) or model.model_lgb
            extra_model = getattr(model, 'model_extra_calibrated', None) or model.model_extra
            
            # Get predictions from base models
            if hasattr(model, 'extra_base_model_type') and model.extra_base_model_type in ['mlp', 'svm'] and model.extra_model_scaler is not None:
                X_scaled = model.extra_model_scaler.transform(X_prepared)
                p_extra = extra_model.predict_proba(X_scaled)[:, 1]
            else:
                p_extra = extra_model.predict_proba(X_prepared)[:, 1]
                
            p_xgb = xgb_model.predict_proba(X_prepared)[:, 1]
            p_cat = cat_model.predict_proba(X_prepared)[:, 1]
            p_lgb = lgb_model.predict_proba(X_prepared)[:, 1]
            
            # Create meta-features from base model predictions
            from models.ensemble.meta_features import create_meta_features
            
            dynamic_weights = None
            if hasattr(model, 'dynamic_weighting') and hasattr(model, 'dynamic_weights'):
                if model.dynamic_weighting:
                    dynamic_weights = model.dynamic_weights
                    
            meta_features = create_meta_features(
                p_xgb, p_cat, p_lgb, p_extra, 
                dynamic_weights
            )
            
            # Use meta-features for SHAP analysis instead of original features
            X_for_shap = meta_features
            feature_names = [
                'prob_xgb', 'prob_cat', 'prob_lgb', 'prob_extra',
                'weighted_avg',
                'diff_xgb_cat', 'diff_xgb_lgb', 'diff_cat_lgb',
                'diff_extra_xgb', 'diff_extra_cat', 'diff_extra_lgb',
                'max_prob', 'min_prob', 'range_prob',
                'rank_xgb', 'rank_cat', 'rank_lgb', 'rank_extra',
                'vote_sum', 'vote_agreement'
            ]
            
            logger.info(f"Created meta-features with shape: {X_for_shap.shape}")
        except Exception as e:
            logger.error(f"Failed to create meta-features: {str(e)}")
            return {
                'error': f"Meta-feature creation failed: {str(e)}",
                'feature_importance': {}
            }
    else:
        # Otherwise use the model directly with original features
        target_model = model
        X_for_shap = X_sample
        feature_names = X_sample.columns.tolist()
    
    # Initialize SHAP explainer based on model type
    try:
        if hasattr(target_model, 'tree_method'):
            # For tree-based models (XGBoost, LightGBM)
            explainer = shap.TreeExplainer(target_model)
        else:
            # For other models
            explainer = shap.KernelExplainer(
                target_model.predict_proba, 
                shap.sample(X_for_shap, 100, random_state=42)
            )
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_for_shap)
        
        # For binary classifiers, shap_values might be a list with one element
        if isinstance(shap_values, list):
            # Take SHAP values for positive class (class 1)
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        # Check if shap_values is a numpy array before proceeding
        if not isinstance(shap_values, np.ndarray):
            raise TypeError("SHAP values must be a numpy array")
            
        # Calculate mean absolute SHAP values for feature importance
        feature_importance = np.abs(shap_values).mean(axis=0)
        
        # Ensure feature_importance is 1D and matches feature_names length
        if feature_importance.ndim > 1:
            feature_importance = feature_importance.mean(axis=1)
            
        if len(feature_importance) != len(feature_names):
            logger.warning(f"Feature importance length ({len(feature_importance)}) doesn't match feature names length ({len(feature_names)})")
            # Adjust feature_names if needed
            feature_names = feature_names[:len(feature_importance)] if len(feature_names) > len(feature_importance) else feature_names
            
        # Create feature importance dictionary
        importance_dict = dict(zip(feature_names, feature_importance))
        
        # Sort features by importance
        sorted_importance = sorted(
            importance_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Log top 10 features
        logger.info("Top feature importances (SHAP):")
        for feature, importance in sorted_importance[:10]:
            logger.info(f"  {feature}: {importance:.6f}")
            mlflow.log_metric(f"shap_importance_{feature}", importance)
        
        return {
            'feature_importance': dict(sorted_importance),
            'shap_values': shap_values
        }
        
    except Exception as e:
        logger.error(f"SHAP explanation failed: {str(e)}")
        return {
            'error': str(e),
            'feature_importance': {}
        }

def analyze_prediction_errors(model, X_val: pd.DataFrame, y_val: pd.Series, 
                            threshold: Optional[float] = None,
                            logger: ExperimentLogger = None) -> Dict:
    """
    Analyze prediction errors on the validation set (most recent data).
    
    Args:
        model: Trained model with predict_proba method
        X_val: Validation features
        y_val: Validation target values
        threshold: Classification threshold (default: model.optimal_threshold or 0.5)
        logger: Logger instance
        
    Returns:
        Dictionary with error analysis results
    """
    if logger is None:
        logger = ExperimentLogger(experiment_name="ensemble_model_diagnostics",
                                log_dir="./logs/ensemble_model_diagnostics")
    
    logger.info("Analyzing prediction errors...")
    
    # Get model threshold
    if threshold is None:
        if hasattr(model, 'optimal_threshold'):
            threshold = model.optimal_threshold
        else:
            threshold = 0.5
    
    logger.info(f"Using classification threshold: {threshold:.4f}")
    
    # Get predictions
    y_prob = model.predict_proba(X_val)
    
    # Handle different return shapes from predict_proba
    if isinstance(y_prob, tuple):
        # Some models return (neg_class_prob, pos_class_prob)
        y_prob = y_prob[1]
    elif y_prob.ndim > 1 and y_prob.shape[1] > 1:
        # Some models return [neg_class_prob, pos_class_prob] for each sample
        y_prob = y_prob[:, 1]
    
    # Convert to binary predictions using threshold
    y_pred = (y_prob >= threshold).astype(int)
    
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    
    # Compute various metrics
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate class balance
    pos_rate = np.mean(y_val)
    
    # Find incorrect predictions
    incorrect_mask = y_pred != y_val
    incorrect_indices = np.where(incorrect_mask)[0]
    
    # False positives and false negatives
    fp_mask = (y_pred == 1) & (y_val == 0)
    fn_mask = (y_pred == 0) & (y_val == 1)
    fp_indices = np.where(fp_mask)[0]
    fn_indices = np.where(fn_mask)[0]
    
    # Extract false positive and false negative examples
    fp_examples = X_val.iloc[fp_indices] if len(fp_indices) > 0 else pd.DataFrame()
    fn_examples = X_val.iloc[fn_indices] if len(fn_indices) > 0 else pd.DataFrame()
    
    # Analyze false positives (highest probability first)
    fp_probs = y_prob[fp_mask]
    fp_indices_sorted = np.argsort(-fp_probs)  # Sort in descending order
    fp_analysis = []
    
    for i in range(min(5, len(fp_indices_sorted))):
        idx = fp_indices[fp_indices_sorted[i]]
        prob = y_prob[idx]
        fp_analysis.append({
            'idx': idx,
            'probability': prob,
            'threshold_difference': prob - threshold
        })
    
    # Analyze false negatives (closest to threshold first)
    fn_probs = y_prob[fn_mask]
    fn_indices_sorted = np.argsort(threshold - fn_probs)  # Sort by proximity to threshold
    fn_analysis = []
    
    for i in range(min(5, len(fn_indices_sorted))):
        idx = fn_indices[fn_indices_sorted[i]]
        prob = y_prob[idx]
        fn_analysis.append({
            'idx': idx,
            'probability': prob,
            'threshold_difference': threshold - prob
        })
    
    # Log results
    logger.info(f"Error analysis results:")
    logger.info(f"  Total samples: {total}")
    logger.info(f"  Class balance: {pos_rate:.2%} positive")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 Score: {f1_score:.4f}")
    logger.info(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    logger.info(f"  False Positive Rate: {fp/(fp+tn):.4f}")
    logger.info(f"  False Negative Rate: {fn/(fn+tp):.4f}")
    
    # Log to MLflow
    mlflow.log_metrics({
        'error_analysis_accuracy': accuracy,
        'error_analysis_precision': precision,
        'error_analysis_recall': recall,
        'error_analysis_f1': f1_score,
        'error_analysis_fps': fp,
        'error_analysis_fns': fn
    })
    
    # Return compiled results
    return {
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        },
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'false_positive_rate': fp/(fp+tn) if (fp+tn) > 0 else 0,
            'false_negative_rate': fn/(fn+tp) if (fn+tp) > 0 else 0
        },
        'class_balance': pos_rate,
        'threshold': threshold,
        'error_count': len(incorrect_indices),
        'error_rate': len(incorrect_indices) / total,
        'false_positives': len(fp_indices),
        'false_negatives': len(fn_indices),
        'fp_analysis': fp_analysis,
        'fn_analysis': fn_analysis
    }
