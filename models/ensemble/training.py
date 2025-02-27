"""
Model Training Utilities

Functions for training the ensemble models.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from typing import Dict, List, Tuple, Optional, Union
import mlflow

from utils.logger import ExperimentLogger
logger = ExperimentLogger(experiment_name="ensemble_model_training",
                            log_dir="./logs/ensemble_model_training")

def initialize_meta_learner(meta_learner_type: str = 'xgb') -> object:
    """
    Initialize the meta learner based on the provided meta_learner_type.
    
    Args:
        meta_learner_type: Type of meta-learner ('xgb', 'logistic', or 'mlp')
        
    Returns:
        Initialized meta-learner model
    """
    
    logger.info(f"Initializing meta-learner of type: {meta_learner_type}")
    
    if meta_learner_type.lower() == 'xgb':
        # XGBoost meta-learner with CPU settings and reduced complexity
        meta_learner = XGBClassifier(
            tree_method='hist',  # CPU-optimized
            device='cpu',
            n_jobs=-1,
            objective='binary:logistic',
            learning_rate=0.05,
            n_estimators=200,  # Reduced from 500
            max_depth=4,       # Reduced from 6
            random_state=42,
            colsample_bytree=0.8,
            early_stopping_rounds=100,
            eval_metric=['logloss', 'auc'],
            gamma=0.1,
            min_child_weight=10,
            reg_alpha=0.5,
            reg_lambda=1.0,
            subsample=0.8
        )
        
        logger.info("XGBoost meta-learner initialized with CPU-optimized settings")
        
    elif meta_learner_type.lower() == 'logistic':
        # Logistic Regression meta-learner with L2 regularization
        meta_learner = LogisticRegression(
            penalty='l2',
            C=1.0,  # Inverse of regularization strength
            solver='lbfgs',
            max_iter=1000,
            tol=1e-4,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        logger.info("Logistic Regression meta-learner initialized with L2 regularization")
        
    elif meta_learner_type.lower() == 'logistic_cv':
        # Logistic Regression with cross-validation for C parameter
        meta_learner = LogisticRegressionCV(
            penalty='l2',
            Cs=10,
            cv=5,
            solver='lbfgs',
            max_iter=1000,
            tol=1e-4,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        logger.info("LogisticRegressionCV meta-learner initialized with auto C selection")
        
    elif meta_learner_type.lower() == 'mlp':
        # Neural Network meta-learner with reduced complexity
        meta_learner = MLPClassifier(
            hidden_layer_sizes=(20, 10),  # Reduced from (50, 20)
            activation='relu',
            solver='adam',
            alpha=0.01,  # Increased regularization
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.2,
            batch_size=32,
            random_state=42
        )
        
        logger.info("MLPClassifier meta-learner initialized with adaptive learning rate")
        
    else:
        logger.error(f"Unknown meta_learner_type: {meta_learner_type}")
        raise ValueError(f"Unknown meta_learner_type: {meta_learner_type}. "
                        f"Supported types: 'xgb', 'logistic', 'logistic_cv', 'mlp'")
    
    # Log meta-learner parameters to MLflow
    meta_learner_params = meta_learner.get_params()
    
    # Flatten nested dictionaries for MLflow
    flattened_params = {}
    for key, value in meta_learner_params.items():
        if isinstance(value, (int, float, str, bool, type(None))):
            flattened_params[f"meta_learner_{key}"] = value
    
    mlflow.log_params(flattened_params)
    
    return meta_learner

def train_base_models(models: Dict, X_train: pd.DataFrame, y_train: pd.Series, 
                    X_eval: pd.DataFrame, y_eval: pd.Series) -> Dict:
    """
    Train base models with early stopping using evaluation data.
    
    Args:
        models: Dictionary of models to train
        X_train: Training features
        y_train: Training labels
        X_eval: Evaluation features for early stopping
        y_eval: Evaluation labels for early stopping
        
    Returns:
        Dictionary of trained models
    """
    
    trained_models = {}
    
    for model_name, model in models.items():
        logger.info(f"Training base model: {model_name}")
        
        try:
            # Prepare training and evaluation data
            X_train_copy = X_train.copy()
            X_eval_copy = X_eval.copy()
            
            # Handle model fitting based on model name
            if model_name == 'extra':
                # Apply scaling if needed (for MLP or SVM models)
                if 'extra_scaler' in models:
                    scaler = models['extra_scaler']
                    X_train_scaled = scaler.transform(X_train_copy)
                    X_eval_scaled = scaler.transform(X_eval_copy)
                    
                    # Train with early stopping if model supports it
                    if hasattr(model, 'early_stopping_rounds'):
                        model.fit(
                            X_train_scaled, y_train,
                            eval_set=[(X_eval_scaled, y_eval)],
                            verbose=False
                        )
                    else:
                        model.fit(X_train_scaled, y_train)
                    
                    trained_models[model_name] = model
                    trained_models['extra_scaler'] = scaler
                else:
                    model.fit(X_train_copy, y_train)
                    trained_models[model_name] = model
            
            elif model_name == 'xgb':
                # XGBoost model with tree_method='hist' for CPU-only training
                model.fit(
                    X_train_copy, y_train,
                    eval_set=[(X_eval_copy, y_eval)],
                    verbose=False
                )
                trained_models[model_name] = model
            
            elif model_name == 'cat':
                # CatBoost model
                model.fit(
                    X_train_copy, y_train,
                    eval_set=(X_eval_copy, y_eval),
                    verbose=False
                )
                trained_models[model_name] = model
            
            elif model_name == 'lgb':
                # LightGBM model
                model.fit(
                    X_train_copy, y_train,
                    eval_set=[(X_eval_copy, y_eval)]
                )
                trained_models[model_name] = model
            
            else:
                # Generic model without specific handling
                model.fit(X_train_copy, y_train)
                trained_models[model_name] = model
            
            # # Log training metrics to MLflow
            # try:
            #     if hasattr(model, 'evals_result'):
            #         # XGBoost-style metrics
            #         evals_result = model.evals_result()
            #         for metric_name, values in evals_result.get('validation_0', {}).items():
            #             final_value = values[-1]
            #             mlflow.log_metric(f"{model_name}_{metric_name}", final_value)
            #     elif hasattr(model, 'best_score_'):
            #         # Scikit-learn style best score
            #         mlflow.log_metric(f"{model_name}_best_score", model.best_score_)
            #     # Special handling for LightGBM models
            #     elif model_name == 'lgb' and hasattr(model, 'best_score'):
            #         # LightGBM stores metrics differently
            #         best_score = model.best_score
            #         # Handle collections.defaultdict case for LightGBM
            #         if hasattr(best_score, 'items'):
            #             for metric_name, value_dict in best_score.items():
            #                 # Extract only numeric values from nested structures
            #                 if isinstance(value_dict, dict):
            #                     for eval_name, value in value_dict.items():
            #                         if isinstance(value, (int, float)):
            #                             mlflow.log_metric(f"{model_name}_{metric_name}_{eval_name}", value)
            #                 elif isinstance(value_dict, (int, float)):
            #                     mlflow.log_metric(f"{model_name}_{metric_name}", value_dict)
            #     # Special handling for CatBoost models
            #     elif model_name == 'cat' and hasattr(model, 'get_best_score'):
            #         # Extract CatBoost metrics safely
            #         try:
            #             best_score = model.get_best_score()
            #             if isinstance(best_score, dict):
            #                 for metric_name, value in best_score.items():
            #                     if isinstance(value, (int, float)):
            #                         mlflow.log_metric(f"{model_name}_{metric_name}", value)
            #         except Exception as cat_err:
            #             logger.warning(f"Error extracting CatBoost metrics: {str(cat_err)}")
            # except Exception as e:
            #     logger.warning(f"Could not log training metrics for {model_name}: {str(e)}")
            
            logger.info(f"Successfully trained base model: {model_name}")
            
        except Exception as e:
            logger.error(f"Error training base model {model_name}: {str(e)}")
            raise
    
    return trained_models

def train_meta_learner(meta_learner, meta_features: np.ndarray, meta_targets: np.ndarray,
                        eval_meta_features: Optional[np.ndarray] = None, 
                        eval_meta_targets: Optional[np.ndarray] = None) -> object:
    """
    Train the meta-learner on meta-features from base models.
    
    Args:
        meta_learner: The meta-learner model to train
        meta_features: Meta-features for training
        meta_targets: Target values for training
        eval_meta_features: Evaluation meta-features for early stopping
        eval_meta_targets: Evaluation target values for early stopping
        
    Returns:
        Trained meta-learner model
    """
    
    logger.info("Training meta-learner...")
    
    try:
        # For models that support early stopping (XGBoost)
        if hasattr(meta_learner, 'early_stopping_rounds') and eval_meta_features is not None:
            meta_learner.fit(
                meta_features, meta_targets,
                eval_set=[(eval_meta_features, eval_meta_targets)],
                verbose=False
            )
        else:
            # Standard training for other models
            meta_learner.fit(meta_features, meta_targets)
        
        # Log meta-learner performance metrics to MLflow
        try:
            if hasattr(meta_learner, 'evals_result'):
                # XGBoost-style metrics
                evals_result = meta_learner.evals_result()
                for metric_name, values in evals_result.get('validation_0', {}).items():
                    final_value = values[-1]
                    mlflow.log_metric(f"meta_learner_{metric_name}", final_value)
            elif hasattr(meta_learner, 'best_score_'):
                # Scikit-learn style best score
                mlflow.log_metric(f"meta_learner_best_score", meta_learner.best_score_)
        except Exception as e:
            logger.warning(f"Could not log meta-learner training metrics: {str(e)}")
        
        logger.info("Meta-learner training completed successfully")
        
        return meta_learner
        
    except Exception as e:
        logger.error(f"Error training meta-learner: {str(e)}")
        raise
