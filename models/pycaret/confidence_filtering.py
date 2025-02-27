"""
Confidence-based filtering module for PyCaret soccer prediction.

This module contains functions for filtering predictions based on confidence
to improve precision at the cost of recall.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add project root to Python path if needed
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from utils.logger import ExperimentLogger

# Setup logger
logger = ExperimentLogger(experiment_name="pycaret_confidence_filtering")

def apply_confidence_filter(predictions, confidence_threshold=0.7, prob_col='prediction_score'):
    """
    Filter predictions based on confidence threshold.
    
    Args:
        predictions (pd.DataFrame): DataFrame with predictions
        confidence_threshold (float): Confidence threshold for filtering
        prob_col (str): Name of the column with prediction probabilities
        
    Returns:
        pd.DataFrame: DataFrame with filtered predictions
    """
    logger.info(f"Applying confidence filter with threshold {confidence_threshold}")
    
    # Make a copy to avoid modifying the original
    predictions = predictions.copy()
    
    # Apply confidence filter
    predictions['high_confidence'] = (predictions[prob_col] >= confidence_threshold).astype(int)
    
    # Only predict positive for high confidence predictions
    predictions['confidence_filtered_prediction'] = predictions['high_confidence'] & (predictions[prob_col] >= 0.5).astype(int)
    
    # Count filtered predictions
    total_predictions = len(predictions)
    high_confidence_count = predictions['high_confidence'].sum()
    positive_predictions = (predictions[prob_col] >= 0.5).astype(int).sum()
    filtered_positive_predictions = predictions['confidence_filtered_prediction'].sum()
    
    logger.info(f"Confidence filtering results:")
    logger.info(f" - Total predictions: {total_predictions}")
    logger.info(f" - High confidence predictions: {high_confidence_count} ({high_confidence_count/total_predictions*100:.2f}%)")
    logger.info(f" - Original positive predictions: {positive_predictions} ({positive_predictions/total_predictions*100:.2f}%)")
    logger.info(f" - Filtered positive predictions: {filtered_positive_predictions} ({filtered_positive_predictions/total_predictions*100:.2f}%)")
    
    return predictions

def find_optimal_confidence_threshold(predictions, target_precision=0.5, min_recall=0.25, 
                                     prob_col='prediction_score', target_col='target'):
    """
    Find the optimal confidence threshold to achieve target precision with minimum recall.
    
    Args:
        predictions (pd.DataFrame): DataFrame with predictions and true labels
        target_precision (float): Target precision to achieve
        min_recall (float): Minimum acceptable recall
        prob_col (str): Name of the column with prediction probabilities
        target_col (str): Name of the column with true labels
        
    Returns:
        tuple: (optimal_threshold, metrics_dict)
    """
    logger.info(f"Finding optimal confidence threshold for target precision {target_precision} with min recall {min_recall}")
    
    # Ensure we have the required columns
    if prob_col not in predictions.columns:
        logger.error(f"Probability column '{prob_col}' not found in predictions DataFrame")
        return 0.5, {}
    
    if target_col not in predictions.columns:
        logger.error(f"Target column '{target_col}' not found in predictions DataFrame")
        return 0.5, {}
    
    # Extract probabilities and true labels
    y_true = predictions[target_col].values
    y_proba = predictions[prob_col].values
    
    # Try confidence thresholds from 0.5 to 0.99 with 0.01 step
    thresholds = np.arange(0.5, 1.0, 0.01)
    results = []
    
    for threshold in thresholds:
        # Apply confidence filter
        high_confidence = (y_proba >= threshold)
        
        # Only predict positive for high confidence predictions
        y_pred = np.zeros_like(y_true)
        y_pred[high_confidence & (y_proba >= 0.5)] = 1
        
        # Calculate metrics
        if y_pred.sum() > 0:  # Avoid division by zero
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
        else:
            prec = 0
            rec = 0
            f1 = 0
        
        # Store results
        results.append({
            'threshold': threshold,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'positive_count': y_pred.sum(),
            'positive_ratio': y_pred.sum() / len(y_pred)
        })
    
    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(results)
    
    # Find thresholds that meet minimum recall requirement
    valid_thresholds = results_df[results_df['recall'] >= min_recall]
    
    if valid_thresholds.empty:
        logger.warning(f"No confidence threshold achieves minimum recall of {min_recall}")
        # Return threshold with highest recall
        best_idx = results_df['recall'].idxmax()
        optimal_threshold = results_df.loc[best_idx, 'threshold']
        logger.info(f"Using threshold {optimal_threshold} with highest recall {results_df.loc[best_idx, 'recall']}")
    else:
        # Find threshold that gets closest to target precision while maintaining minimum recall
        valid_thresholds['precision_diff'] = abs(valid_thresholds['precision'] - target_precision)
        best_idx = valid_thresholds['precision_diff'].idxmin()
        optimal_threshold = valid_thresholds.loc[best_idx, 'threshold']
        
        logger.info(f"Optimal confidence threshold: {optimal_threshold}")
        logger.info(f"Precision at optimal threshold: {valid_thresholds.loc[best_idx, 'precision']}")
        logger.info(f"Recall at optimal threshold: {valid_thresholds.loc[best_idx, 'recall']}")
    
    # Apply optimal threshold to get final predictions
    high_confidence = (y_proba >= optimal_threshold)
    y_pred_optimal = np.zeros_like(y_true)
    y_pred_optimal[high_confidence & (y_proba >= 0.5)] = 1
    
    # Calculate final metrics
    if y_pred_optimal.sum() > 0:  # Avoid division by zero
        final_precision = precision_score(y_true, y_pred_optimal)
        final_recall = recall_score(y_true, y_pred_optimal)
        final_f1 = f1_score(y_true, y_pred_optimal)
    else:
        final_precision = 0
        final_recall = 0
        final_f1 = 0
    
    # Create metrics dictionary
    metrics = {
        'precision': final_precision,
        'recall': final_recall,
        'f1': final_f1,
        'threshold': optimal_threshold,
        'positive_count': y_pred_optimal.sum(),
        'positive_ratio': y_pred_optimal.sum() / len(y_pred_optimal),
        'threshold_values': thresholds.tolist(),
        'precision_values': results_df['precision'].tolist(),
        'recall_values': results_df['recall'].tolist(),
        'f1_values': results_df['f1'].tolist(),
        'positive_ratio_values': results_df['positive_ratio'].tolist()
    }
    
    # Plot precision-recall vs threshold
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['threshold'], results_df['precision'], label='Precision')
    plt.plot(results_df['threshold'], results_df['recall'], label='Recall')
    plt.plot(results_df['threshold'], results_df['positive_ratio'], label='Positive Ratio')
    plt.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Optimal Threshold = {optimal_threshold:.4f}')
    plt.axhline(y=min_recall, color='g', linestyle='--', label=f'Min Recall = {min_recall}')
    plt.axhline(y=target_precision, color='b', linestyle='--', label=f'Target Precision = {target_precision}')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Score')
    plt.title('Precision-Recall vs Confidence Threshold')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plot_path = 'confidence_threshold_optimization.png'
    plt.savefig(plot_path)
    plt.close()
    
    logger.info(f"Confidence threshold optimization plot saved to {plot_path}")
    logger.info(f"Final metrics at confidence threshold {optimal_threshold}:")
    logger.info(f" - Precision: {final_precision:.4f}")
    logger.info(f" - Recall: {final_recall:.4f}")
    logger.info(f" - F1: {final_f1:.4f}")
    logger.info(f" - Positive predictions: {y_pred_optimal.sum()} ({y_pred_optimal.sum()/len(y_pred_optimal)*100:.2f}%)")
    
    return optimal_threshold, metrics

def implement_two_stage_classifier(base_model, X_train, y_train, X_val, y_val):
    """
    Implement a two-stage classifier for high precision.
    
    Stage 1: Base model predicts probabilities
    Stage 2: Second model filters predictions based on features and base probabilities
    
    Args:
        base_model: Trained base model
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        
    Returns:
        object: Two-stage classifier
    """
    from sklearn.ensemble import RandomForestClassifier
    
    logger.info("Implementing two-stage classifier for high precision")
    
    # Get base model predictions on training data
    if hasattr(base_model, 'predict_proba'):
        train_probs = base_model.predict_proba(X_train)[:, 1]
    else:
        logger.error("Base model does not have predict_proba method")
        return base_model
    
    # Create new features for second stage
    X_train_stage2 = X_train.copy()
    X_train_stage2['base_prob'] = train_probs
    
    # Only use positive predictions from base model for training second stage
    positive_mask = (train_probs >= 0.5)
    X_train_stage2 = X_train_stage2[positive_mask]
    y_train_stage2 = y_train[positive_mask]
    
    logger.info(f"Training second stage classifier on {len(X_train_stage2)} positive predictions")
    logger.info(f"Positive ratio in second stage training: {y_train_stage2.mean():.4f}")
    
    # Train second stage classifier
    second_stage = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42
    )
    
    second_stage.fit(X_train_stage2, y_train_stage2)
    
    # Create two-stage classifier
    class TwoStageClassifier:
        def __init__(self, base_model, second_stage):
            self.base_model = base_model
            self.second_stage = second_stage
        
        def predict_proba(self, X):
            # Get base model probabilities
            base_probs = self.base_model.predict_proba(X)[:, 1]
            
            # Create features for second stage
            X_stage2 = X.copy()
            X_stage2['base_prob'] = base_probs
            
            # Initialize second stage probabilities
            second_probs = np.zeros((len(X), 2))
            second_probs[:, 0] = 1.0  # Default to negative class
            
            # Only apply second stage to positive predictions from base model
            positive_mask = (base_probs >= 0.5)
            
            if positive_mask.sum() > 0:
                # Get second stage probabilities for positive predictions
                positive_probs = self.second_stage.predict_proba(X_stage2[positive_mask])
                second_probs[positive_mask] = positive_probs
            
            return second_probs
        
        def predict(self, X):
            # Get probabilities
            probs = self.predict_proba(X)
            
            # Return class predictions
            return (probs[:, 1] >= 0.5).astype(int)
    
    # Create and evaluate two-stage classifier
    two_stage = TwoStageClassifier(base_model, second_stage)
    
    # Evaluate on validation data
    val_probs = two_stage.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)
    
    # Calculate metrics
    prec = precision_score(y_val, val_preds)
    rec = recall_score(y_val, val_preds)
    f1 = f1_score(y_val, val_preds)
    
    logger.info(f"Two-stage classifier validation metrics:")
    logger.info(f" - Precision: {prec:.4f}")
    logger.info(f" - Recall: {rec:.4f}")
    logger.info(f" - F1: {f1:.4f}")
    
    return two_stage

def integrate_confidence_filtering_with_pycaret(model, predictions, target_precision=0.5, min_recall=0.25):
    """
    Integrate confidence filtering with PyCaret model predictions.
    
    Args:
        model: Trained PyCaret model
        predictions (pd.DataFrame): DataFrame with predictions from model
        target_precision (float): Target precision to achieve
        min_recall (float): Minimum acceptable recall
        
    Returns:
        tuple: (filtered_predictions, optimal_threshold, metrics)
    """
    logger.info("Integrating confidence filtering with PyCaret predictions")
    
    # Find optimal confidence threshold
    optimal_threshold, metrics = find_optimal_confidence_threshold(
        predictions,
        target_precision=target_precision,
        min_recall=min_recall
    )
    
    # Apply confidence filter
    filtered_predictions = apply_confidence_filter(
        predictions,
        confidence_threshold=optimal_threshold
    )
    
    logger.info("Confidence filtering integration complete")
    
    return filtered_predictions, optimal_threshold, metrics 