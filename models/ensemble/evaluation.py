"""
Model Evaluation Utilities

Functions for evaluating model performance.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.model_selection import StratifiedKFold
import mlflow
from typing import Dict, List, Tuple, Optional, Union
import scipy.stats as stats

from utils.logger import ExperimentLogger
from models.ensemble.thresholds import tune_threshold_for_precision

def evaluate_model(model, X_val: pd.DataFrame, y_val: pd.Series, 
                threshold: Optional[float] = None,
                logger: ExperimentLogger = None) -> Dict:
    """
    Evaluate model performance on validation data.
    
    Args:
        model: Trained model with predict_proba method
        X_val: Validation features
        y_val: Validation target values
        threshold: Classification threshold (default: model.optimal_threshold or 0.5)
        logger: Logger instance
        
    Returns:
        Dictionary with performance metrics and confidence intervals
    """
    if logger is None:
        logger = ExperimentLogger(experiment_name="ensemble_model_evaluation",
                                log_dir="./logs/ensemble_model_evaluation")
    
    logger.info("Evaluating model performance...")
    
    # Get model threshold
    if threshold is None:
        if hasattr(model, 'optimal_threshold'):
            threshold = model.optimal_threshold
        else:
            threshold, metrics = tune_threshold_for_precision(model, X_val, y_val, logger=logger)
    
    logger.info(f"Using classification threshold: {threshold:.4f}")
    
    # Get predictions
    y_prob = model.predict_proba(X_val)
    
    # Handle different return shapes from predict_proba
    if isinstance(y_prob, tuple):
        y_prob = y_prob[1]
    elif y_prob.ndim > 1 and y_prob.shape[1] > 1:
        y_prob = y_prob[:, 1]
    
    # Convert to binary predictions using threshold
    y_pred = (y_prob >= threshold).astype(int)
    
    # Compute metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    
    # Try to compute ROC AUC, but handle cases where there's only one class
    try:
        roc_auc = roc_auc_score(y_val, y_prob)
    except Exception as e:
        logger.warning(f"Could not compute ROC AUC: {str(e)}")
        roc_auc = np.nan
    
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    
    # Calculate class balance
    pos_rate = np.mean(y_val)
    
    # Calculate confidence intervals using bootstrap
    n_bootstrap = 1000
    bootstrap_metrics = calculate_bootstrap_metrics(y_val, y_prob, threshold, n_bootstrap)
    
    # Log base metrics
    logger.info(f"Evaluation metrics:")
    logger.info(f"  Accuracy: {accuracy:.4f} (95% CI: {bootstrap_metrics['accuracy_ci'][0]:.4f}-{bootstrap_metrics['accuracy_ci'][1]:.4f})")
    logger.info(f"  Precision: {precision:.4f} (95% CI: {bootstrap_metrics['precision_ci'][0]:.4f}-{bootstrap_metrics['precision_ci'][1]:.4f})")
    logger.info(f"  Recall: {recall:.4f} (95% CI: {bootstrap_metrics['recall_ci'][0]:.4f}-{bootstrap_metrics['recall_ci'][1]:.4f})")
    logger.info(f"  F1 Score: {f1:.4f} (95% CI: {bootstrap_metrics['f1_ci'][0]:.4f}-{bootstrap_metrics['f1_ci'][1]:.4f})")
    logger.info(f"  ROC AUC: {roc_auc:.4f} (95% CI: {bootstrap_metrics['roc_auc_ci'][0]:.4f}-{bootstrap_metrics['roc_auc_ci'][1]:.4f})")
    logger.info(f"  Class Balance: {pos_rate:.2%} positive")
    logger.info(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    # Log to MLflow
    mlflow.log_metrics({
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'threshold': threshold,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    })
    
    # Return compiled results
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        },
        'class_balance': pos_rate,
        'threshold': threshold,
        'confidence_intervals': {
            'accuracy_ci': bootstrap_metrics['accuracy_ci'],
            'precision_ci': bootstrap_metrics['precision_ci'],
            'recall_ci': bootstrap_metrics['recall_ci'],
            'f1_ci': bootstrap_metrics['f1_ci'],
            'roc_auc_ci': bootstrap_metrics['roc_auc_ci']
        }
    }

def calculate_bootstrap_metrics(y_true: np.ndarray, y_prob: np.ndarray, 
                                threshold: float, n_bootstrap: int = 1000) -> Dict:
    """
    Calculate bootstrap confidence intervals for metrics.
    
    Args:
        y_true: True target values
        y_prob: Predicted probabilities
        threshold: Classification threshold
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        Dictionary with bootstrap metrics and confidence intervals
    """
    n_samples = len(y_true)
    
    # Arrays to store bootstrap metrics
    accuracies = np.zeros(n_bootstrap)
    precisions = np.zeros(n_bootstrap)
    recalls = np.zeros(n_bootstrap)
    f1_scores = np.zeros(n_bootstrap)
    roc_aucs = np.zeros(n_bootstrap)
    
    # Bootstrap loop
    for i in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true.iloc[indices] if hasattr(y_true, 'iloc') else y_true[indices]
        y_prob_boot = y_prob[indices]
        
        # Convert to binary predictions using threshold
        y_pred_boot = (y_prob_boot >= threshold).astype(int)
        
        # Compute metrics
        accuracies[i] = accuracy_score(y_true_boot, y_pred_boot)
        
        # Handle cases with no positive predictions
        try:
            precisions[i] = precision_score(y_true_boot, y_pred_boot, zero_division=0)
        except:
            precisions[i] = 0
        
        try:
            recalls[i] = recall_score(y_true_boot, y_pred_boot, zero_division=0)
        except:
            recalls[i] = 0
        
        try:
            f1_scores[i] = f1_score(y_true_boot, y_pred_boot, zero_division=0)
        except:
            f1_scores[i] = 0
        
        # ROC AUC requires samples from both classes
        try:
            if len(np.unique(y_true_boot)) > 1:
                roc_aucs[i] = roc_auc_score(y_true_boot, y_prob_boot)
            else:
                roc_aucs[i] = np.nan
        except:
            roc_aucs[i] = np.nan
    
    # Calculate 95% confidence intervals
    alpha = 0.05
    accuracy_ci = np.nanpercentile(accuracies, [alpha/2*100, (1-alpha/2)*100])
    precision_ci = np.nanpercentile(precisions, [alpha/2*100, (1-alpha/2)*100])
    recall_ci = np.nanpercentile(recalls, [alpha/2*100, (1-alpha/2)*100])
    f1_ci = np.nanpercentile(f1_scores, [alpha/2*100, (1-alpha/2)*100])
    roc_auc_ci = np.nanpercentile(roc_aucs, [alpha/2*100, (1-alpha/2)*100])
    
    return {
        'accuracy_ci': accuracy_ci,
        'precision_ci': precision_ci,
        'recall_ci': recall_ci,
        'f1_ci': f1_ci,
        'roc_auc_ci': roc_auc_ci,
        'accuracies': accuracies,
        'precisions': precisions,
        'recalls': recalls,
        'f1_scores': f1_scores,
        'roc_aucs': roc_aucs
    }

def cross_validate(model_class, X: pd.DataFrame, y: pd.Series, 
                    n_splits: int = 3, 
                    logger: ExperimentLogger = None,
                    **model_params) -> Dict:
    """
    Perform stratified k-fold cross-validation.
    
    Args:
        model_class: Model class to instantiate
        X: Feature dataframe
        y: Target series
        n_splits: Number of cross-validation folds
        logger: Logger instance
        model_params: Parameters to pass to model constructor
        
    Returns:
        Dictionary with cross-validation results
    """
    if logger is None:
        logger = ExperimentLogger(experiment_name="ensemble_model_cv",
                                log_dir="./logs/ensemble_model_cv")
    
    logger.info(f"Starting {n_splits}-fold cross-validation...")
    
    # Initialize arrays to store metrics
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    roc_aucs = []
    thresholds = []
    
    # Initialize stratified k-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Cross-validation loop
    for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        logger.info(f"Fold {i+1}/{n_splits}")
        
        # Split data
        X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
        
        # Initialize and train model
        model = model_class(**model_params)
        model.train(X_train_fold, y_train_fold, X_val=X_test_fold, y_val=y_test_fold, split_validation=False)
        
        # Evaluate model
        eval_results = evaluate_model(model, X_test_fold, y_test_fold, logger=logger)
        
        # Store metrics
        accuracies.append(eval_results['accuracy'])
        precisions.append(eval_results['precision'])
        recalls.append(eval_results['recall'])
        f1_scores.append(eval_results['f1_score'])
        roc_aucs.append(eval_results['roc_auc'])
        thresholds.append(eval_results['threshold'])
        
        # Log fold metrics
        mlflow.log_metrics({
            f'cv_fold_{i+1}_accuracy': eval_results['accuracy'],
            f'cv_fold_{i+1}_precision': eval_results['precision'],
            f'cv_fold_{i+1}_recall': eval_results['recall'],
            f'cv_fold_{i+1}_f1': eval_results['f1_score'],
            f'cv_fold_{i+1}_roc_auc': eval_results['roc_auc'],
            f'cv_fold_{i+1}_threshold': eval_results['threshold']
        })
    
    # Calculate mean and standard deviation for each metric
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    mean_precision = np.mean(precisions)
    std_precision = np.std(precisions)
    
    mean_recall = np.mean(recalls)
    std_recall = np.std(recalls)
    
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    
    mean_roc_auc = np.mean(roc_aucs)
    std_roc_auc = np.std(roc_aucs)
    
    mean_threshold = np.mean(thresholds)
    std_threshold = np.std(thresholds)
    
    # Calculate confidence intervals
    n = len(accuracies)
    t_value = stats.t.ppf(0.975, n-1)  # 95% confidence interval
    
    accuracy_ci = (mean_accuracy - t_value * std_accuracy / np.sqrt(n),
                  mean_accuracy + t_value * std_accuracy / np.sqrt(n))
    
    precision_ci = (mean_precision - t_value * std_precision / np.sqrt(n),
                   mean_precision + t_value * std_precision / np.sqrt(n))
    
    recall_ci = (mean_recall - t_value * std_recall / np.sqrt(n),
                mean_recall + t_value * std_recall / np.sqrt(n))
    
    f1_ci = (mean_f1 - t_value * std_f1 / np.sqrt(n),
            mean_f1 + t_value * std_f1 / np.sqrt(n))
    
    roc_auc_ci = (mean_roc_auc - t_value * std_roc_auc / np.sqrt(n),
                 mean_roc_auc + t_value * std_roc_auc / np.sqrt(n))
    
    # Log overall metrics
    logger.info(f"Cross-validation results ({n_splits} folds):")
    logger.info(f"  Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f} (95% CI: {accuracy_ci[0]:.4f}-{accuracy_ci[1]:.4f})")
    logger.info(f"  Precision: {mean_precision:.4f} ± {std_precision:.4f} (95% CI: {precision_ci[0]:.4f}-{precision_ci[1]:.4f})")
    logger.info(f"  Recall: {mean_recall:.4f} ± {std_recall:.4f} (95% CI: {recall_ci[0]:.4f}-{recall_ci[1]:.4f})")
    logger.info(f"  F1 Score: {mean_f1:.4f} ± {std_f1:.4f} (95% CI: {f1_ci[0]:.4f}-{f1_ci[1]:.4f})")
    logger.info(f"  ROC AUC: {mean_roc_auc:.4f} ± {std_roc_auc:.4f} (95% CI: {roc_auc_ci[0]:.4f}-{roc_auc_ci[1]:.4f})")
    logger.info(f"  Threshold: {mean_threshold:.4f} ± {std_threshold:.4f}")
    
    # Log to MLflow
    mlflow.log_metrics({
        'cv_mean_accuracy': mean_accuracy,
        'cv_std_accuracy': std_accuracy,
        'cv_mean_precision': mean_precision,
        'cv_std_precision': std_precision,
        'cv_mean_recall': mean_recall,
        'cv_std_recall': std_recall,
        'cv_mean_f1': mean_f1,
        'cv_std_f1': std_f1,
        'cv_mean_roc_auc': mean_roc_auc,
        'cv_std_roc_auc': std_roc_auc,
        'cv_mean_threshold': mean_threshold,
        'cv_std_threshold': std_threshold
    })
    
    # Return compiled results
    return {
        'metrics': {
            'accuracy': {
                'values': accuracies,
                'mean': mean_accuracy,
                'std': std_accuracy,
                'ci': accuracy_ci
            },
            'precision': {
                'values': precisions,
                'mean': mean_precision,
                'std': std_precision,
                'ci': precision_ci
            },
            'recall': {
                'values': recalls,
                'mean': mean_recall,
                'std': std_recall,
                'ci': recall_ci
            },
            'f1_score': {
                'values': f1_scores,
                'mean': mean_f1,
                'std': std_f1,
                'ci': f1_ci
            },
            'roc_auc': {
                'values': roc_aucs,
                'mean': mean_roc_auc,
                'std': std_roc_auc,
                'ci': roc_auc_ci
            },
            'threshold': {
                'values': thresholds,
                'mean': mean_threshold,
                'std': std_threshold
            }
        },
        'n_splits': n_splits
    }
