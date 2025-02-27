"""
Probability calibration module for PyCaret soccer prediction.

This module contains functions for calibrating prediction probabilities
to improve precision and reliability of predictions.
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
import sys
from pathlib import Path


# Add project root to Python path if needed
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from utils.logger import ExperimentLogger

# Setup logger
logger = ExperimentLogger(experiment_name="pycaret_calibration")

def calibrate_with_platt_scaling(model, X_val, y_val, cv=5):
    """
    Calibrate model probabilities using Platt scaling (logistic regression).
    
    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation labels
        cv (int): Number of cross-validation folds
        
    Returns:
        object: Calibrated model
    """
    logger.info(f"Calibrating model with Platt scaling (CV={cv})")
    
    # Create calibrated model
    calibrated_model = CalibratedClassifierCV(
        estimator=model,
        method='sigmoid',  # Platt scaling
        cv=cv
    )
    
    # Fit calibration model
    calibrated_model.fit(X_val, y_val)
    
    logger.info("Platt scaling calibration complete")
    
    return calibrated_model

def calibrate_with_isotonic_regression(model, X_val, y_val, cv=5):
    """
    Calibrate model probabilities using isotonic regression.
    
    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation labels
        cv (int): Number of cross-validation folds
        
    Returns:
        object: Calibrated model
    """
    logger.info(f"Calibrating model with isotonic regression (CV={cv})")
    
    # Create calibrated model
    calibrated_model = CalibratedClassifierCV(
        estimator=model,
        method='isotonic',  # Isotonic regression
        cv=cv
    )
    
    # Fit calibration model
    calibrated_model.fit(X_val, y_val)
    
    logger.info("Isotonic regression calibration complete")
    
    return calibrated_model

def calibrate_with_temperature_scaling(model, X_val, y_val):
    """
    Calibrate model probabilities using temperature scaling.
    
    Temperature scaling is a simple scaling method that divides logits by a temperature parameter.
    
    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation labels
        
    Returns:
        object: Calibrated model (as a wrapper function)
    """
    logger.info("Calibrating model with temperature scaling")
    
    # Get uncalibrated probabilities
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_val)[:, 1]
    else:
        logger.error("Model does not have predict_proba method")
        return model
    
    # Convert probabilities to logits
    eps = 1e-15
    logits = np.log(y_proba + eps) - np.log(1 - y_proba + eps)
    
    # Find optimal temperature using binary search
    def log_loss(t):
        # Apply temperature scaling
        scaled_logits = logits / t
        scaled_probs = 1 / (1 + np.exp(-scaled_logits))
        
        # Calculate log loss
        eps = 1e-15
        losses = -(y_val * np.log(scaled_probs + eps) + (1 - y_val) * np.log(1 - scaled_probs + eps))
        return np.mean(losses)
    
    # Binary search for optimal temperature
    t_min, t_max = 0.1, 10.0
    t_opt = 1.0
    best_loss = log_loss(t_opt)
    
    for _ in range(100):  # Max iterations
        t_mid = (t_min + t_max) / 2
        loss_mid = log_loss(t_mid)
        
        if loss_mid < best_loss:
            t_opt = t_mid
            best_loss = loss_mid
        
        # Try left half
        t_left = (t_min + t_mid) / 2
        loss_left = log_loss(t_left)
        
        # Try right half
        t_right = (t_mid + t_max) / 2
        loss_right = log_loss(t_right)
        
        if loss_left < loss_right:
            t_max = t_mid
        else:
            t_min = t_mid
    
    logger.info(f"Optimal temperature: {t_opt:.4f}")
    
    # Create a wrapper function for the calibrated model
    class TemperatureScaledModel:
        def __init__(self, base_model, temperature):
            self.base_model = base_model
            self.temperature = temperature
        
        def predict_proba(self, X):
            # Get base model probabilities
            base_probs = self.base_model.predict_proba(X)
            
            # Apply temperature scaling to the positive class only
            pos_probs = base_probs[:, 1]
            logits = np.log(pos_probs + eps) - np.log(1 - pos_probs + eps)
            scaled_logits = logits / self.temperature
            scaled_probs = 1 / (1 + np.exp(-scaled_logits))
            
            # Return calibrated probabilities
            return np.column_stack([1 - scaled_probs, scaled_probs])
        
        def predict(self, X):
            # Get calibrated probabilities
            probs = self.predict_proba(X)
            
            # Return class predictions
            return (probs[:, 1] >= 0.5).astype(int)
    
    # Create and return calibrated model
    calibrated_model = TemperatureScaledModel(model, t_opt)
    
    logger.info("Temperature scaling calibration complete")
    
    return calibrated_model

def evaluate_calibration(model, X_val, y_val, model_name="Model"):
    """
    Evaluate calibration of a model and return calibration metrics.
    
    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation labels
        model_name (str): Name of the model
        
    Returns:
        dict: Dictionary with calibration metrics
    """
    logger.info(f"Evaluating calibration for {model_name}")
    
    # Get predicted probabilities
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_val)[:, 1]
    else:
        logger.error("Model does not have predict_proba method")
        return {}
    
    # Calculate calibration curve
    prob_true, prob_pred = calibration_curve(y_val, y_proba, n_bins=10)
    
    # Calculate calibration error (mean absolute error between predicted and true probabilities)
    calibration_error = np.mean(np.abs(prob_true - prob_pred))
    
    # Calculate Brier score (mean squared error between predicted probabilities and actual outcomes)
    brier_score = np.mean((y_proba - y_val) ** 2)
    
    # Calculate log loss
    eps = 1e-15
    log_loss = -np.mean(y_val * np.log(y_proba + eps) + (1 - y_val) * np.log(1 - y_proba + eps))
    
    # Create metrics dictionary
    metrics = {
        'calibration_error': calibration_error,
        'brier_score': brier_score,
        'log_loss': log_loss
    }
    
    logger.info(f"Calibration metrics for {model_name}:")
    logger.info(f" - Calibration Error: {calibration_error:.4f}")
    logger.info(f" - Brier Score: {brier_score:.4f}")
    logger.info(f" - Log Loss: {log_loss:.4f}")
    
    return metrics

def integrate_calibration_with_pycaret(model, X_train, y_train, X_val, y_val, method='platt'):
    """
    Integrate calibration with PyCaret model.
    
    Args:
        model: Trained PyCaret model
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        method (str): Calibration method ('platt', 'isotonic', or 'temperature')
        
    Returns:
        object: Calibrated model
    """
    logger.info(f"Integrating {method} calibration with PyCaret model")
    
    # Extract the underlying model from PyCaret
    try:
        from pycaret.classification import get_config
        underlying_model = model
    except:
        logger.warning("Could not extract underlying model from PyCaret, using model as is")
        underlying_model = model
    
    # Apply calibration based on method
    if method.lower() == 'platt':
        calibrated_model = calibrate_with_platt_scaling(underlying_model, X_val, y_val)
    elif method.lower() == 'isotonic':
        calibrated_model = calibrate_with_isotonic_regression(underlying_model, X_val, y_val)
    elif method.lower() == 'temperature':
        calibrated_model = calibrate_with_temperature_scaling(underlying_model, X_val, y_val)
    else:
        logger.error(f"Unknown calibration method: {method}")
        return model
    
    # Evaluate calibration
    evaluate_calibration(calibrated_model, X_val, y_val, f"Calibrated ({method})")
    
    logger.info(f"Calibration integration complete")
    
    return calibrated_model 