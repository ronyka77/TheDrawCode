#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Soccer prediction batch prediction script.

This script loads a trained model and makes predictions on new data.
It supports applying optimized thresholds and confidence filtering.

Usage:
    python predict.py --model-path "models/saved/ensemble_20230215_1245" --data-path "data/new_matches.csv"
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import modules
from models.pycaret.threshold_utils import apply_threshold
from models.pycaret.confidence_filtering import filter_by_confidence
from utils.logger import ExperimentLogger

# Setup logging
logger = ExperimentLogger(experiment_name="soccer_prediction_prediction")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Soccer prediction batch prediction")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the saved model")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to the data for prediction")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Path to save predictions (default: auto-generated)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Prediction threshold")
    parser.add_argument("--confidence-threshold", type=float, default=None,
                        help="Confidence threshold for filtering")
    parser.add_argument("--apply-feature-engineering", action="store_true",
                        help="Apply feature engineering to input data")
    parser.add_argument("--metadata-path", type=str, default=None,
                        help="Path to model metadata (default: auto-detected)")
    return parser.parse_args()

def load_model(model_path):
    """
    Load a saved model.
    
    Args:
        model_path (str): Path to the saved model
        
    Returns:
        object: Loaded model
    """
    try:
        from pycaret.classification import load_model
        logger.info(f"Loading model from {model_path}")
        model = load_model(model_path)
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def load_metadata(model_path, metadata_path=None):
    """
    Load model metadata.
    
    Args:
        model_path (str): Path to the saved model
        metadata_path (str, optional): Path to model metadata
        
    Returns:
        dict: Model metadata
    """
    import json
    
    # If metadata path not provided, try to infer it
    if not metadata_path:
        model_dir = os.path.dirname(model_path)
        model_name = os.path.basename(model_path)
        metadata_path = os.path.join(model_dir, f"{model_name}_metadata.json")
    
    # Check if metadata file exists
    if not os.path.exists(metadata_path):
        logger.warning(f"Metadata file not found at {metadata_path}")
        return {}
    
    try:
        logger.info(f"Loading metadata from {metadata_path}")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        logger.error(f"Error loading metadata: {str(e)}")
        return {}

def load_data(data_path, apply_feature_engineering=False):
    """
    Load data for prediction.
    
    Args:
        data_path (str): Path to the data
        apply_feature_engineering (bool): Whether to apply feature engineering
        
    Returns:
        pd.DataFrame: Data for prediction
    """
    try:
        logger.info(f"Loading data from {data_path}")
        data = pd.read_csv(data_path)
        
        # Apply feature engineering if requested
        if apply_feature_engineering:
            from models.pycaret.feature_engineering import engineer_features
            logger.info("Applying feature engineering")
            data = engineer_features(data)
        
        return data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def make_predictions(model, data, threshold=0.5, confidence_threshold=None):
    """
    Make predictions using a trained model.
    
    Args:
        model: Trained model
        data (pd.DataFrame): Data for prediction
        threshold (float): Prediction threshold
        confidence_threshold (float, optional): Confidence threshold for filtering
        
    Returns:
        pd.DataFrame: Predictions dataframe
    """
    try:
        from pycaret.classification import predict_model
        
        logger.info("Making predictions")
        predictions = predict_model(model, data=data)
        
        # Apply threshold
        logger.info(f"Applying threshold: {threshold}")
        predictions = apply_threshold(predictions, threshold)
        
        # Apply confidence filtering if requested
        if confidence_threshold is not None:
            logger.info(f"Applying confidence filtering with threshold: {confidence_threshold}")
            predictions = filter_by_confidence(predictions, confidence_threshold)
        
        return predictions
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        raise

def save_predictions(predictions, output_path=None, model_path=None):
    """
    Save predictions to a file.
    
    Args:
        predictions (pd.DataFrame): Predictions dataframe
        output_path (str, optional): Path to save predictions
        model_path (str, optional): Path to the model (for auto-generating output path)
        
    Returns:
        str: Path where predictions were saved
    """
    # Generate output path if not provided
    if not output_path:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        if model_path:
            model_name = os.path.basename(model_path)
            output_path = f"predictions_{model_name}_{timestamp}.csv"
        else:
            output_path = f"predictions_{timestamp}.csv"
    
    try:
        logger.info(f"Saving predictions to {output_path}")
        predictions.to_csv(output_path, index=False)
        return output_path
    except Exception as e:
        logger.error(f"Error saving predictions: {str(e)}")
        return None

def main():
    """Main function to run batch prediction."""
    args = parse_args()
    
    try:
        # Load model
        model = load_model(args.model_path)
        
        # Load metadata
        metadata = load_metadata(args.model_path, args.metadata_path)
        
        # Get threshold from metadata if not provided
        threshold = args.threshold
        if threshold == 0.5 and 'optimized_threshold' in metadata:
            threshold = metadata['optimized_threshold']
            logger.info(f"Using threshold from metadata: {threshold}")
        
        # Load data
        data = load_data(args.data_path, args.apply_feature_engineering)
        
        # Make predictions
        predictions = make_predictions(
            model=model,
            data=data,
            threshold=threshold,
            confidence_threshold=args.confidence_threshold
        )
        
        # Save predictions
        output_path = save_predictions(
            predictions=predictions,
            output_path=args.output_path,
            model_path=args.model_path
        )
        
        # Log summary
        logger.info("Prediction completed successfully")
        logger.info(f"Total predictions: {len(predictions)}")
        
        # Count positive predictions
        pred_col = next((col for col in predictions.columns 
                         if col in ['prediction_label', 'prediction_class', 'Label', 'prediction_threshold']), None)
        if pred_col:
            positive_count = predictions[pred_col].sum()
            positive_pct = (positive_count / len(predictions)) * 100
            logger.info(f"Positive predictions: {positive_count} ({positive_pct:.2f}%)")
        
        return {
            "success": True,
            "output_path": output_path,
            "predictions": predictions
        }
        
    except Exception as e:
        logger.error(f"Error in prediction pipeline: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    result = main()
    if result and result.get("success"):
        logger.info("Prediction executed successfully")
    else:
        logger.error("Prediction execution failed")
        sys.exit(1) 