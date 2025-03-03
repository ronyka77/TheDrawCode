#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Soccer Prediction Main Script

This script orchestrates the entire workflow for soccer prediction using PyCaret.
It handles data loading, feature engineering, model training, calibration,
threshold optimization, and confidence filtering.

Usage:
    python main.py --experiment-name "soccer_prediction_v1" --target-precision 0.45

Author: AI Assistant
Date: 2023-10-15
"""

import os
import sys
import argparse
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime
import mlflow

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import local modules
try:
    from models.pycaret.model_training import (
        setup_pycaret_environment,
        compare_models_for_precision,
        tune_model_for_precision,
        create_ensemble_model,
        evaluate_model_on_holdout
    )
    from models.pycaret.mlflow_module import (
        save_model_and_predictions,
        setup_mlflow_for_pycaret,
        log_pycaret_model,
        log_threshold_optimization_results,
        log_pycaret_experiment_summary
    )
    from models.pycaret.feature_engineering import get_feature_importance
    from models.pycaret.threshold_utils import precision_focused_score
    from utils.logger import ExperimentLogger
    from models.pycaret.calibration import calibrate_with_isotonic_regression, calibrate_with_platt_scaling
    from models.pycaret.confidence_filtering import apply_confidence_filter, find_optimal_confidence_threshold
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)

# Setup logger
logger = ExperimentLogger(experiment_name="pycaret_main")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Soccer Prediction Pipeline")
    
    parser.add_argument("--target-precision", type=float, default=0.40,
                        help="Target precision to achieve (default: 0.40)")
    parser.add_argument("--min-recall", type=float, default=0.25,
                        help="Minimum recall to maintain (default: 0.25)")
    parser.add_argument("--experiment-name", type=str, default="pycaret_soccer_prediction",
                        help="MLflow experiment name (default: 'pycaret_soccer_prediction')")
    parser.add_argument("--skip-feature-engineering", action="store_true",
                        help="Skip feature engineering step")
    parser.add_argument("--skip-calibration", action="store_true",
                        help="Skip probability calibration step")
    parser.add_argument("--skip-confidence-filtering", action="store_true",
                        help="Skip confidence filtering step")
    parser.add_argument("--model-type", type=str, choices=["single", "ensemble"], default="ensemble",
                        help="Type of model to train (default: 'ensemble')")
    parser.add_argument("--include-models", type=str, default="xgboost,lightgbm,catboost",
                        help="Comma-separated list of models to include (default: 'xgboost,lightgbm,catboost')")
    parser.add_argument("--n-iter", type=int, default=50,
                        help="Number of iterations for hyperparameter tuning (default: 50)")
    parser.add_argument("--output-dir", type=str, default="models/saved",
                        help="Directory to save models and results (default: 'models/saved')")
    
    return parser.parse_args()

def load_data():
    """Load the training data."""
    try:
        from models.pycaret.data_module import load_data_for_pycaret
        train_df, test_df, val_df = load_data_for_pycaret()
        logger.info(f"Loaded data using data_module: train={train_df.shape}, test={test_df.shape}, val={val_df.shape}")
        return train_df, test_df, val_df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)

def apply_calibration(model, data):
    """Apply probability calibration to the model."""
    logger.info("Applying probability calibration")
    try:
        # Split data for calibration
        X = data.drop('target', axis=1)
        y = data['target']
        
        # Use the calibration module to apply isotonic regression calibration
        calibrated_model = calibrate_with_isotonic_regression(model, X, y, cv=5)
        
        logger.info("Calibration applied successfully")
        return calibrated_model
    except Exception as e:
        logger.error(f"Error in calibration: {e}")
        return model

def optimize_threshold(model, data, target_precision=0.40, min_recall=0.25):
    """Optimize the prediction threshold to achieve target precision."""
    logger.info(f"Optimizing threshold for target precision: {target_precision}")
    try:
        from pycaret.classification import predict_model, get_config
        import inspect
        
        # IMPORTANT: First check if we have the preprocessing pipeline and transform data if needed
        try:
            # Try to access the pipeline directly from available variables
            pipeline = None
            variables = get_config('variables')
            if 'pipeline' in variables:
                pipeline = variables['pipeline']
                logger.info("Successfully retrieved preprocessing pipeline from variables dictionary")
            else:
                logger.info("No preprocessing pipeline found in variables dictionary")
                
            if pipeline and hasattr(model, 'n_features_'):
                X_data = data.drop('target', axis=1)
                y_data = data['target'].values
                
                # Get expected feature count from model
                model_feature_count = getattr(model, 'n_features_', None)
                data_feature_count = X_data.shape[1]
                
                # If feature counts don't match, apply transformation
                if model_feature_count and model_feature_count != data_feature_count:
                    logger.info(f"Applying preprocessing to match features: model={model_feature_count}, data={data_feature_count}")
                    # Use PyCaret's pipeline to transform
                    X_transformed = pipeline.transform(X_data)
                    logger.info(f"Data transformed: {X_transformed.shape}")
        except Exception as e:
            logger.warning(f"Error accessing preprocessing pipeline: {e}")
        
        # Get predictions first
        # Check if parameters are supported in this version
        predict_model_params = inspect.signature(predict_model).parameters
        
        # Build kwargs based on available parameters
        kwargs = {
            'data': data,
            'verbose': False
        }
        
        # Use 'estimator' or 'model' parameter based on what's supported
        if 'estimator' in predict_model_params:
            kwargs['estimator'] = model
        else:
            kwargs['model'] = model
        
        # Make predictions
        predictions = predict_model(**kwargs)
        
        # Get the probability column name
        prob_col = None
        for col in ['prediction_score', 'Score_1', 'Score']:
            if col in predictions.columns:
                prob_col = col
                break
        
        if prob_col is None:
            logger.error("Could not find probability column in predictions")
            return 0.5, None, None
        
        # Use the threshold_utils module to find the optimal threshold
        from models.pycaret.threshold_utils import optimize_threshold_for_precision
        
        optimal_threshold, metrics = optimize_threshold_for_precision(
            predictions=predictions,
            target_precision=target_precision,
            min_recall=min_recall,
            prob_col=prob_col,
            target_col='target'
        )
        
        # Extract precision and recall from metrics
        achieved_precision = metrics.get('precision', None)
        achieved_recall = metrics.get('recall', None)
        
        logger.info(f"Optimal threshold: {optimal_threshold:.4f}, Precision: {achieved_precision:.4f}, Recall: {achieved_recall:.4f}")
        return optimal_threshold, achieved_precision, achieved_recall
    except Exception as e:
        logger.error(f"Error in threshold optimization: {e}")
        return 0.5, None, None

def apply_confidence_filtering(predictions, confidence_threshold=0.8):
    """Filter predictions based on confidence."""
    logger.info(f"Applying confidence filtering with threshold: {confidence_threshold}")
    try:
        # Use the confidence_filtering module to filter predictions
        filtered_predictions = apply_confidence_filter(
            predictions=predictions,
            confidence_threshold=confidence_threshold
        )
        
        # Get the count of filtered predictions
        filtered_count = len(filtered_predictions)
        original_count = len(predictions)
        
        logger.info(f"Filtered {filtered_count} high confidence predictions out of {original_count}")
        return filtered_predictions
    except Exception as e:
        logger.error(f"Error in confidence filtering: {e}")
        return predictions

def save_metadata(metadata, output_path):
    """Save metadata to a JSON file."""
    try:
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Metadata saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving metadata: {e}")

def main():
    """Main function to orchestrate the workflow."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up MLflow using the mlflow_module
    experiment_id = setup_mlflow_for_pycaret(args.experiment_name)
    
    try:
        # Use mlflow.start_run from mlflow_module or directly
        with mlflow.start_run(run_name=f"soccer_prediction_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            # Log parameters
            mlflow.log_params({
                "target_precision": args.target_precision,
                "min_recall": args.min_recall,
                "model_type": args.model_type,
                "include_models": args.include_models,
                "n_iter": args.n_iter
            })
            
            # Load data - properly unpack the tuple
            train_df, test_df, val_df = load_data()
            logger.info(f"Data loaded: train={train_df.shape}, test={test_df.shape}, val={val_df.shape}")
            
            # Create custom scorer - make sure it's in the format PyCaret expects
            # For PyCaret 3.3.2, custom metrics should be a function that takes y_true and y_pred
            custom_scorer = lambda y_true, y_pred: precision_focused_score(y_true, y_pred, 
                                                                            target_precision=args.target_precision,
                                                                            min_recall=args.min_recall)
            
            # Setup PyCaret environment with validation data
            pycaret_env = setup_pycaret_environment(
                data=train_df,
                target_col='target',
                val_data=test_df,  # Use test_df for validation during training
                fold=5,
                normalize=True,
                feature_selection=False,
                fix_imbalance=True,
                session_id=42
            )
            
            # Parse include_models
            include_models = [model.strip() for model in args.include_models.split(',')]
            
            # Compare models - ensure custom_metric is correctly passed
            best_models = compare_models_for_precision(
                include=include_models,
                n_select=3,
                sort='Precision',
                custom_metric=custom_scorer
            )
            
            if args.model_type == "single":
                # Tune the best model
                best_model = tune_model_for_precision(
                    model=best_models[0],
                    optimize='Precision',
                    n_iter=args.n_iter,
                    custom_metric=custom_scorer
                )
                final_model = best_model
            else:
                # Create ensemble model
                final_model = create_ensemble_model(
                    models=best_models,
                    method='Stacking',
                    optimize='Precision',
                    weights=[0.5, 0.3, 0.2],
                    custom_metric=custom_scorer
                )
                logger.info(f"Ensemble model created: {final_model}")
            
            # Apply calibration using test_df
            if not args.skip_calibration:
                final_model = apply_calibration(final_model, test_df)
                logger.info(f"Calibration applied: {final_model}")
            # Evaluate model on newest data (val_df)
            holdout_data = val_df  # Use val_df as final holdout
            holdout_metrics = evaluate_model_on_holdout(
                model=final_model,
                holdout_data=holdout_data,
                target_col='target'
            )
            logger.info(f"Holdout metrics: {holdout_metrics}")
            # Optimize threshold
            optimal_threshold, achieved_precision, achieved_recall = optimize_threshold(
                final_model, 
                test_df,  # Use test data instead of val_df
                target_precision=args.target_precision,
                min_recall=args.min_recall
            )
            logger.info(f"Optimal threshold: {optimal_threshold}, Achieved precision: {achieved_precision}, Achieved recall: {achieved_recall}")
            # Get feature importance
            # feature_importance = get_feature_importance(final_model)
            
            # Save model and predictions using mlflow_module
            model_path = save_model_and_predictions(
                model=final_model,
                model_name=f"soccer_prediction_{datetime.now().strftime('%Y%m%d_%H%M')}"
            )
            logger.info(f"Model saved to {model_path}")
            # Log threshold optimization results using mlflow_module
            threshold_metrics = {
                "precision": float(achieved_precision) if achieved_precision is not None else 0.0,
                "recall": float(achieved_recall) if achieved_recall is not None else 0.0
            }
            log_threshold_optimization_results(
                threshold=optimal_threshold,
                metrics=threshold_metrics,
                model_name=f"soccer_prediction_{datetime.now().strftime('%Y%m%d_%H%M')}"
            )
            logger.info(f"Threshold optimization results logged")
            # Apply confidence filtering if not skipped
            if not args.skip_confidence_filtering:
                # Get predictions first
                try:
                    from pycaret.classification import predict_model
                    import inspect
                    
                    # Check if parameters are supported in this version
                    predict_model_params = inspect.signature(predict_model).parameters
                    
                    # Build kwargs based on available parameters
                    kwargs = {
                        'data': test_df,
                        'verbose': True
                    }
                    
                    # Use 'estimator' or 'model' parameter based on what's supported
                    if 'estimator' in predict_model_params:
                        kwargs['estimator'] = final_model
                    else:
                        kwargs['model'] = final_model
                    
                    # Make predictions
                    predictions = predict_model(**kwargs)
                    
                    # Use the confidence_filtering module directly
                    filtered_predictions = apply_confidence_filter(
                        predictions=predictions, 
                        confidence_threshold=0.8
                    )
                    filtered_path = os.path.join(args.output_dir, "filtered_predictions.csv")
                    filtered_predictions.to_csv(filtered_path, index=False)
                    logger.info(f"Filtered predictions saved to {filtered_path}")
                    mlflow.log_artifact(filtered_path)
                except Exception as e:
                    logger.error(f"Error applying confidence filtering: {e}")
            
            # Save metadata
            metadata = {
                "model_type": args.model_type,
                "include_models": args.include_models,
                "target_precision": args.target_precision,
                "min_recall": args.min_recall,
                "optimal_threshold": float(optimal_threshold),
                "achieved_precision": float(achieved_precision) if achieved_precision is not None else None,
                "achieved_recall": float(achieved_recall) if achieved_recall is not None else None,
                # "feature_importance": feature_importance.to_dict() if isinstance(feature_importance, pd.DataFrame) else feature_importance,
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            metadata_path = os.path.join(args.output_dir, "model_metadata.json")
            save_metadata(metadata, metadata_path)
            mlflow.log_artifact(metadata_path)
            
            # Log metrics
            mlflow.log_metrics({
                "optimal_threshold": float(optimal_threshold),
                "achieved_precision": float(achieved_precision) if achieved_precision is not None else 0.0,
                "achieved_recall": float(achieved_recall) if achieved_recall is not None else 0.0
            })
            
            logger.info(f"Soccer prediction workflow completed successfully. Model saved to {model_path}")
    except Exception as e:
        logger.error(f"Error in main workflow: {e}")
        # Make sure to end the run even if there's an error
        if mlflow.active_run():
            mlflow.end_run()
        raise e

if __name__ == "__main__":
    main() 