"""
Outlier Detection Utilities

This module provides utilities for detecting and removing outliers from datasets
using various methods including Isolation Forest with StandardScaler preprocessing.
Designed for the Soccer Prediction Project with MLflow integration and reproducibility.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.utils.logger import ExperimentLogger


def remove_outliers_isolation_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scaler=None,
    contamination: float = 0.01,
    random_state: int = 19,
    logger: ExperimentLogger = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Remove outliers from training data using Isolation Forest with pre-fitted scaler.
    
    Args:
        X_train: Training features DataFrame
        y_train: Training labels Series
        scaler: Pre-fitted scaler (StandardScaler/RobustScaler) from the model
        contamination: Expected proportion of outliers (default: 0.01 = 1%)
        random_state: Random state for reproducibility (default: 19)
        logger: Logger instance for tracking
        
    Returns:
        Tuple of cleaned datasets: (X_train_clean, y_train_clean)
    """
    if logger is None:
        logger = ExperimentLogger(experiment_name="outlier_detection")
    
    logger.info("Starting outlier detection with Isolation Forest")
    logger.info(f"Training data shape before outlier removal: {X_train.shape}")
    logger.info(f"Contamination rate: {contamination} ({contamination*100:.1f}%)")
    
    # Record original shapes and class distribution
    original_train_size = len(X_train)
    original_positive_rate = y_train.mean()
    
    try:
        # Step 1: Use pre-fitted scaler or create new one
        if scaler is not None:
            logger.info("Using pre-fitted scaler for outlier detection")
            X_train_scaled = scaler.transform(X_train)
        else:
            logger.info("Creating new StandardScaler for outlier detection")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
        
        # Step 2: Fit Isolation Forest on scaled training data
        logger.info("Fitting Isolation Forest model")
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,  # Use all available cores
            bootstrap=False,  # For reproducibility
        )
        
        # Predict outliers (-1 for outliers, 1 for inliers)
        outlier_predictions = iso_forest.fit_predict(X_train_scaled)
        
        # Step 3: Identify inliers and outliers
        inlier_mask = outlier_predictions == 1
        outlier_count = np.sum(~inlier_mask)
        outlier_percentage = (outlier_count / original_train_size) * 100
        
        logger.info(f"Detected {outlier_count} outliers ({outlier_percentage:.2f}% of training data)")
        
        # Step 4: Filter training data to keep only inliers
        X_train_clean = X_train.loc[inlier_mask].copy()
        y_train_clean = y_train.loc[inlier_mask].copy()
        
        # Reset indices to ensure clean sequential indexing
        X_train_clean.reset_index(drop=True, inplace=True)
        y_train_clean.reset_index(drop=True, inplace=True)
        
        # Log results
        new_train_size = len(X_train_clean)
        new_positive_rate = y_train_clean.mean()
        
        logger.info(f"Training data shape after outlier removal: {X_train_clean.shape}")
        logger.info(f"Removed {original_train_size - new_train_size} samples")
        logger.info(f"Class distribution - Before: {original_positive_rate:.3f}, After: {new_positive_rate:.3f}")
        
        # Check if class distribution changed significantly
        distribution_change = abs(new_positive_rate - original_positive_rate)
        if distribution_change > 0.05:  # 5% threshold
            logger.warning(
                f"Significant class distribution change detected: {distribution_change:.3f}. "
                "Consider adjusting contamination rate."
            )
        
        return X_train_clean, y_train_clean
        
    except Exception as e:
        logger.error(f"Error during outlier detection: {str(e)}")
        logger.warning("Returning original datasets without outlier removal")
        return X_train, y_train


def analyze_outlier_impact(
    X_before: pd.DataFrame,
    y_before: pd.Series,
    X_after: pd.DataFrame,
    y_after: pd.Series,
    logger: ExperimentLogger = None,
) -> dict:
    """
    Analyze the impact of outlier removal on dataset characteristics.
    
    Args:
        X_before: Features before outlier removal
        y_before: Labels before outlier removal
        X_after: Features after outlier removal
        y_after: Labels after outlier removal
        logger: Logger instance
        
    Returns:
        Dictionary with analysis results
    """
    if logger is None:
        logger = ExperimentLogger(experiment_name="outlier_analysis")
    
    analysis = {}
    
    # Basic statistics
    analysis['samples_before'] = len(X_before)
    analysis['samples_after'] = len(X_after)
    analysis['samples_removed'] = analysis['samples_before'] - analysis['samples_after']
    analysis['removal_percentage'] = (analysis['samples_removed'] / analysis['samples_before']) * 100
    
    # Class distribution
    analysis['positive_rate_before'] = y_before.mean()
    analysis['positive_rate_after'] = y_after.mean()
    analysis['class_distribution_change'] = abs(analysis['positive_rate_after'] - analysis['positive_rate_before'])
    
    # Feature statistics changes
    analysis['feature_stats'] = {}
    for col in X_before.columns:
        if col in X_after.columns:
            before_stats = {
                'mean': X_before[col].mean(),
                'std': X_before[col].std(),
                'min': X_before[col].min(),
                'max': X_before[col].max()
            }
            after_stats = {
                'mean': X_after[col].mean(),
                'std': X_after[col].std(),
                'min': X_after[col].min(),
                'max': X_after[col].max()
            }
            
            analysis['feature_stats'][col] = {
                'before': before_stats,
                'after': after_stats,
                'mean_change': abs(after_stats['mean'] - before_stats['mean']),
                'std_change': abs(after_stats['std'] - before_stats['std'])
            }
    
    # Log summary
    logger.info("Outlier Removal Impact Analysis:")
    logger.info(f"  Samples removed: {analysis['samples_removed']} ({analysis['removal_percentage']:.2f}%)")
    logger.info(f"  Class distribution change: {analysis['class_distribution_change']:.4f}")
    logger.info(f"  Positive rate: {analysis['positive_rate_before']:.3f} -> {analysis['positive_rate_after']:.3f}")
    
    return analysis


def get_outlier_scores(
    X: pd.DataFrame,
    scaler=None,
    contamination: float = 0.01,
    random_state: int = 19,
    logger: ExperimentLogger = None,
) -> np.ndarray:
    """
    Get outlier scores for data samples without removing them.
    
    Args:
        X: Features DataFrame
        scaler: Pre-fitted scaler (optional, will create new one if None)
        contamination: Expected proportion of outliers
        random_state: Random state for reproducibility
        logger: Logger instance
        
    Returns:
        Array of outlier scores (lower scores indicate more outlier-like behavior)
    """
    if logger is None:
        logger = ExperimentLogger(experiment_name="outlier_scoring")
    
    logger.info(f"Computing outlier scores for {X.shape[0]} samples")
    
    try:
        # Use pre-fitted scaler or create new one
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
            bootstrap=False,
        )
        
        # Get outlier scores
        scores = iso_forest.fit(X_scaled).decision_function(X_scaled)
        
        logger.info(f"Outlier scores computed. Range: [{scores.min():.3f}, {scores.max():.3f}]")
        
        return scores
        
    except Exception as e:
        logger.error(f"Error computing outlier scores: {str(e)}")
        return np.zeros(len(X)) 