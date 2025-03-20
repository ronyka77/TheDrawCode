"""
Model Calibration Utilities

Functions for calibrating model probabilities and analyzing calibration performance.
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
import mlflow
from typing import Dict, Tuple, List, Optional
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from utils.logger import ExperimentLogger

def calibrate_models(models: Dict, X_train: pd.DataFrame, y_train: pd.Series, 
                    X_test: pd.DataFrame, y_test: pd.Series, 
                    calibration_method: str = "sigmoid", 
                    logger: ExperimentLogger = None) -> Dict:
    """
    Calibrate all base models' probabilities using isotonic regression or Platt scaling.
    
    Args:
        models: Dictionary of models to calibrate
        X_train: Training features for calibration
        y_train: Training labels for calibration
        X_test: Test features for calibration
        y_test: Test labels for calibration
        calibration_method: Method to use for calibration ("sigmoid" or "isotonic")
        logger: Logger instance
        
    Returns:
        Dictionary of calibrated models and calibration results
    """
    if logger is None:
        logger = ExperimentLogger(experiment_name="ensemble_model_calibration",
                                    log_dir="./logs/ensemble_model_calibration")
    logger.info(f"Starting model calibration using {calibration_method} method...")
    
    calibrated_models = {}
    calibration_results = {
        'uncalibrated_probs': {},
        'calibrated_probs': {}
    }
    
    # Function to calibrate an individual model
    def calibrate_model(model, model_name, scaler=None):
        logger.info(f"Calibrating {model_name} model...")
        
        # Prepare data
        X_train_prepared = X_train.copy()
        X_test_prepared = X_test.copy()
        
        # Determine appropriate CV strategy based on sample size
        n_pos_samples = np.sum(y_test)
        n_neg_samples = len(y_test) - n_pos_samples
        min_class_samples = min(n_pos_samples, n_neg_samples)
        min_samples_per_fold = 30  # Minimum samples per class per fold
        
        # Calculate max possible folds while ensuring min_samples_per_fold
        max_folds = max(2, int(min_class_samples / min_samples_per_fold))
        n_splits = min(5, max_folds)  # Cap at 5 folds
        
        if n_splits < 2 or min_class_samples < 2 * min_samples_per_fold:
            logger.warning(f"Not enough samples for cross-validation calibration for {model_name}. Using 'prefit' mode.")
            cv_strategy = 'prefit'
            # For prefit mode, we use the existing trained model directly
            frozen_model = model
        else:
            cv_strategy = n_splits
            logger.info(f"Using {n_splits}-fold CV for {model_name} calibration")
        
        # Apply scaling if needed
        X_calibration = scaler.transform(X_test_prepared) if scaler is not None else X_test_prepared
        X_train_prepared = scaler.transform(X_train_prepared) if scaler is not None else X_train_prepared
        
        # Define frozen_model based on cv_strategy
        if cv_strategy == 'prefit':
            frozen_model = model
        else:
            frozen_model = clone(model)
            
        # Create calibrator with appropriate settings
        calibrated_model = CalibratedClassifierCV(
            estimator=frozen_model,
            method=calibration_method,
            cv=cv_strategy,
            n_jobs=-1,
            ensemble=True
        )
        
        # Fit calibration model
        calibrated_model.fit(X_train_prepared, y_train, eval_set=[(X_test_prepared, y_test)])
        
        # Get probabilities before and after calibration
        if scaler is not None:
            X_pred = scaler.transform(X_test_prepared)
            prob_uncal = model.predict_proba(X_pred)[:, 1]
            prob_cal = calibrated_model.predict_proba(X_pred)[:, 1]
        else:
            prob_uncal = model.predict_proba(X_test_prepared)[:, 1]
            prob_cal = calibrated_model.predict_proba(X_test_prepared)[:, 1]
        
        # Store results for analysis
        calibration_results['uncalibrated_probs'][model_name] = prob_uncal
        calibration_results['calibrated_probs'][model_name] = prob_cal
        
        # Calculate Brier score before and after calibration
        brier_before = brier_score_loss(y_test, prob_uncal)
        brier_after = brier_score_loss(y_test, prob_cal)
        
        logger.info(f"{model_name} Brier score - Before: {brier_before:.4f}, After: {brier_after:.4f}")
        
        # Log to MLflow
        mlflow.log_metric(f"{model_name}_brier_before", brier_before)
        mlflow.log_metric(f"{model_name}_brier_after", brier_after)
        
        return calibrated_model
    
    # Add uncalibrated models and scalers to the results
    for key, value in models.items():
        if key not in calibrated_models and key != 'extra_scaler':
            calibrated_models[key] = value
    
    return {
        'calibrated_models': calibrated_models,
        'calibration_results': calibration_results
    }

def analyze_calibration(calibration_results: Dict, y_test: pd.Series, 
                        logger: ExperimentLogger = None) -> Dict:
    """
    Analyze the effectiveness of calibration with adaptive binning based on data distribution.
    
    Args:
        calibration_results: Dictionary with uncalibrated and calibrated probabilities per model
        y_test: True labels for test data
        logger: Logger instance
        
    Returns:
        Dictionary with calibration analysis results
    """
    if logger is None:
        logger = ExperimentLogger(experiment_name="ensemble_model_calibration",
                                    log_dir="./logs/ensemble_model_calibration")
    
    logger.info("Analyzing calibration effectiveness...")
    
    uncalibrated_probs = calibration_results['uncalibrated_probs']
    calibrated_probs = calibration_results['calibrated_probs']
    
    analysis_results = {}
    
    # Create calibration plots for each model
    for model_name in uncalibrated_probs.keys():
        # Get probabilities
        probs_uncal = uncalibrated_probs[model_name]
        probs_cal = calibrated_probs[model_name]
        
        # Determine bin edges adaptively
        # Use more bins where we have more data
        hist_counts, _ = np.histogram(probs_uncal, bins=10)
        
        # Minimum samples per bin for reliable calibration estimate
        min_samples_per_bin = 20
        
        # Define bins adaptively
        bins = []
        current_count = 0
        bin_edge = 0.0
        
        for i, count in enumerate(hist_counts):
            current_count += count
            if current_count >= min_samples_per_bin or i == len(hist_counts)-1:
                current_count = 0
                bin_edge += 0.1
                bins.append(bin_edge)
        
        bins = [0.0] + bins
        if bins[-1] < 1.0:
            bins.append(1.0)
        
        # Calculate calibration metrics for both uncalibrated and calibrated probabilities
        uncal_metrics = calculate_calibration_metrics(probs_uncal, y_test, bins)
        cal_metrics = calculate_calibration_metrics(probs_cal, y_test, bins)
        
        # Save results
        analysis_results[model_name] = {
            'uncalibrated': uncal_metrics,
            'calibrated': cal_metrics,
            'bins': bins
        }
        
        # Log improvement in calibration
        cal_improvement = uncal_metrics['calibration_error'] - cal_metrics['calibration_error']
        logger.info(f"{model_name} calibration improvement: {cal_improvement:.4f}")
        mlflow.log_metric(f"{model_name}_calibration_improvement", cal_improvement)
        
    return analysis_results

def calculate_calibration_metrics(probabilities: np.ndarray, targets: np.ndarray, 
                                    bins: List[float]) -> Dict:
    """
    Calculate calibration metrics for a set of probabilities.
    
    Args:
        probabilities: Predicted probabilities
        targets: True binary labels
        bins: Bin edges for binning predictions
        
    Returns:
        Dictionary with calibration metrics
    """
    n_bins = len(bins) - 1
    
    # Initialize arrays to store bin metrics
    bin_counts = np.zeros(n_bins)
    bin_predicted_probs = np.zeros(n_bins)
    bin_true_probs = np.zeros(n_bins)
    bin_accuracies = np.zeros(n_bins)
    
    # Calculate metrics for each bin
    for i in range(n_bins):
        # Find the indices of samples in this bin
        bin_mask = np.logical_and(
            probabilities >= bins[i],
            probabilities < bins[i+1] if i < n_bins-1 else probabilities <= bins[i+1]
        )
        
        # Skip empty bins
        if np.sum(bin_mask) == 0:
            continue
        
        bin_counts[i] = np.sum(bin_mask)
        bin_predicted_probs[i] = np.mean(probabilities[bin_mask])
        bin_true_probs[i] = np.mean(targets[bin_mask])
        bin_accuracies[i] = np.mean((probabilities[bin_mask] > 0.5) == targets[bin_mask])
    
    # Calculate overall calibration error (weighted mean absolute difference)
    non_empty_bins = bin_counts > 0
    calibration_error = np.sum(
        bin_counts[non_empty_bins] * 
        np.abs(bin_predicted_probs[non_empty_bins] - bin_true_probs[non_empty_bins])
    ) / np.sum(bin_counts[non_empty_bins])
    
    # Calculate reliability diagram metrics
    reliability = {
        'bin_edges': bins,
        'bin_counts': bin_counts,
        'bin_predicted_probs': bin_predicted_probs,
        'bin_true_probs': bin_true_probs,
        'bin_accuracies': bin_accuracies,
    }
    
    return {
        'calibration_error': calibration_error,
        'reliability': reliability,
        'brier_score': brier_score_loss(targets, probabilities)
    }
