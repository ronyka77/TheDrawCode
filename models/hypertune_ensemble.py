"""
Hyperparameter Tuning for Ensemble Model

This script performs grid search over hyperparameters for the ensemble model,
including meta learner parameters and dynamic weighting factors.

Author: AI Assistant
Date: 2024-02-20
"""

import numpy as np
import pandas as pd
from datetime import datetime
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple

# Add project root to Python path
try:
    project_root = Path(__file__).parent.parent
    if not project_root.exists():
        # Handle network path by using raw string
        project_root = Path(r"\\".join(str(project_root).split("\\")))
    import sys
    sys.path.append(str(project_root))
    print(f"Project root hypertune_ensemble: {project_root}")
except Exception as e:
    print(f"Error setting project root path: {e}")
    sys.path.append(os.getcwd())
    print(f"Current directory hypertune_ensemble: {os.getcwd()}")

# Local imports
from models.ensemble_model_stacked import EnsembleModel
from utils.create_evaluation_set import (
    create_ensemble_evaluation_set,
    import_selected_features_ensemble,
    import_training_data_ensemble,
    setup_mlflow_tracking
)
from utils.logger import ExperimentLogger

def save_grid_search_results(results: list, output_file: str = "grid_search_results.json"):
    """Save grid search results to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

def train_and_evaluate_candidate(
    ensemble_model: EnsembleModel,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    logger: ExperimentLogger
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Train and evaluate a candidate model configuration.
    
    Returns:
        Tuple containing:
        - Dictionary with initial dynamic weights and refined weights
        - Dictionary with evaluation metrics
    """
    # Train the ensemble model (this includes base model training and meta learner fitting)
    ensemble_model.train(X_train, y_train, X_test, y_test, X_val, y_val)
    
    # Get predictions from base models for computing initial weights
    p_xgb = ensemble_model.model_xgb.predict_proba(X_val)[:, 1]
    p_cat = ensemble_model.model_cat.predict_proba(X_val)[:, 1]
    p_lgb = ensemble_model.calibrated_model.predict_proba(X_val)[:, 1]
    
    # Compute initial dynamic weights
    initial_weights = ensemble_model._compute_dynamic_weights(y_val, p_xgb, p_cat, p_lgb)
    logger.info("Initial dynamic weights computed:", extra=initial_weights)
    
    # Search for optimal weights
    refined_weights = ensemble_model._search_optimal_weights(p_xgb, p_cat, p_lgb, y_val)
    logger.info("Refined dynamic weights computed:", extra=refined_weights)
    
    # Override the model's dynamic weights with the refined ones
    ensemble_model.dynamic_weights = refined_weights
    
    # Evaluate the model (this includes threshold tuning)
    metrics = ensemble_model.evaluate(X_val, y_val)
    
    weights_info = {
        "initial_weights": initial_weights,
        "refined_weights": refined_weights
    }
    
    return weights_info, metrics

def create_meta_learner_params(
    lr: float,
    n_est: int,
    max_depth: int,
    scale_pos_weight: float
) -> Dict[str, Any]:
    """Create meta learner parameters dictionary with CPU-only settings."""
    return {
        "learning_rate": lr,
        "n_estimators": n_est,
        "max_depth": max_depth,
        "scale_pos_weight": scale_pos_weight,
        "tree_method": "hist",
        "device": "cpu",
        "nthread": -1,
        "objective": "binary:logistic",
        "random_state": 19
    }

def main():
    # Setup MLflow tracking and logger
    experiment_name = "ensemble_model_hypertune"
    mlruns_dir = setup_mlflow_tracking(experiment_name)
    logger = ExperimentLogger(
        experiment_name=experiment_name,
        log_dir='./logs/ensemble_model_hypertune'
    )

    # Load data
    logger.info("Loading data...")
    selected_features = import_selected_features_ensemble('all')
    X_train, y_train, X_test, y_test = import_training_data_ensemble()
    X_val, y_val = create_ensemble_evaluation_set()

    # Use only selected features
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    X_val = X_val[selected_features]
    logger.info("Data loaded and prepared successfully.")

    # Define candidate grids for meta learner hyperparameters
    meta_lr_candidates = [0.01, 0.05, 0.1]
    meta_n_estimators_candidates = [200, 300, 400]
    meta_max_depth_candidates = [3, 5, 7]
    meta_scale_pos_weight_candidates = [1.0, 2.0, 3.0]

    best_precision = -np.inf
    best_candidate = None
    best_model = None
    grid_search_results = []

    # Grid search over candidate combinations
    total_combinations = (
        len(meta_lr_candidates) *
        len(meta_n_estimators_candidates) *
        len(meta_max_depth_candidates) *
        len(meta_scale_pos_weight_candidates)
    )
    current_combination = 0

    logger.info(f"Starting grid search with {total_combinations} combinations...")

    for lr in meta_lr_candidates:
        for n_est in meta_n_estimators_candidates:
            for max_depth in meta_max_depth_candidates:
                for scale_pos_weight in meta_scale_pos_weight_candidates:
                    current_combination += 1
                    logger.info(f"Testing combination {current_combination}/{total_combinations}")
                    
                    # Create meta learner parameters
                    candidate_meta_params = create_meta_learner_params(
                        lr, n_est, max_depth, scale_pos_weight
                    )
                    
                    try:
                        # Initialize model
                        ensemble_model = EnsembleModel(
                            logger=logger,
                            calibrate=True,
                            meta_learner_type="xgb",
                            dynamic_weighting=True
                        )
                        
                        # Override meta learner initialization
                        def initialize_meta_learner():
                            ensemble_model.meta_learner = XGBClassifier(**candidate_meta_params)
                        ensemble_model._initialize_meta_learner = initialize_meta_learner
                        
                        # Train and evaluate the candidate
                        weights_info, metrics = train_and_evaluate_candidate(
                            ensemble_model, X_train, y_train,
                            X_test, y_test, X_val, y_val,
                            logger
                        )
                        
                        # Record results
                        result = {
                            "meta_params": candidate_meta_params,
                            "initial_weights": weights_info["initial_weights"],
                            "refined_weights": weights_info["refined_weights"],
                            "metrics": metrics,
                            "optimal_threshold": ensemble_model.optimal_threshold
                        }
                        grid_search_results.append(result)
                        
                        # Check if the candidate meets requirements and improves precision
                        precision = metrics["precision"]
                        recall = metrics["recall"]
                        
                        if recall >= ensemble_model.required_recall and precision > best_precision:
                            best_precision = precision
                            best_candidate = result
                            best_model = ensemble_model
                        
                        logger.info(
                            f"Combination {current_combination} results:",
                            extra={
                                "meta_params": candidate_meta_params,
                                "initial_weights": weights_info["initial_weights"],
                                "refined_weights": weights_info["refined_weights"],
                                "metrics": metrics,
                                "optimal_threshold": ensemble_model.optimal_threshold
                            }
                        )
                        
                    except Exception as e:
                        logger.error(
                            f"Error in combination {current_combination}:",
                            extra={
                                "error": str(e),
                                "meta_params": candidate_meta_params
                            }
                        )
                        continue

    # Save all grid search results
    save_grid_search_results(grid_search_results)

    # Log the best candidate using MLflow
    if best_model is not None:
        logger.info("Best candidate found:", extra=best_candidate)
        
        with mlflow.start_run(run_name="hypertune_ensemble_best"):
            # Log parameters
            mlflow.log_params(best_candidate["meta_params"])
            mlflow.log_param("initial_weights", best_candidate["initial_weights"])
            mlflow.log_param("refined_weights", best_candidate["refined_weights"])
            mlflow.log_param("optimal_threshold", best_candidate["optimal_threshold"])
            
            # Log metrics
            mlflow.log_metrics(best_candidate["metrics"])
            
            # Infer signature for the final model
            signature = mlflow.models.infer_signature(
                X_val.astype('float64'),
                best_model.predict(X_val).astype('float64')
            )
            
            # Log the model
            mlflow.sklearn.log_model(
                artifact_path="ensemble_model",
                sk_model=best_model,
                signature=signature,
                registered_model_name=f"ensemble_{datetime.now().strftime('%Y%m%d_%H%M')}"
            )
            
            # Log grid search results as artifact
            mlflow.log_artifact("grid_search_results.json")
            
            # Log a summary of the hypertuning process
            summary = {
                "total_combinations_tested": current_combination,
                "best_precision": best_precision,
                "best_meta_params": best_candidate["meta_params"],
                "best_initial_weights": best_candidate["initial_weights"],
                "best_refined_weights": best_candidate["refined_weights"],
                "best_metrics": best_candidate["metrics"],
                "best_threshold": best_candidate["optimal_threshold"]
            }
            with open("hypertuning_summary.json", "w") as f:
                json.dump(summary, f, indent=4)
            mlflow.log_artifact("hypertuning_summary.json")
    else:
        logger.error("No valid candidate met the recall requirement.")

if __name__ == "__main__":
    main() 