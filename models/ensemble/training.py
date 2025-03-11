"""
Model Training Utilities

Functions for training the ensemble models.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from typing import Dict, List, Tuple, Optional, Union
import mlflow
import time
<<<<<<< HEAD
import random
import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks
import os
=======
>>>>>>> 3798e304ba09a95ae05e21747b9f93b4e52eb5fd

from utils.logger import ExperimentLogger
logger = ExperimentLogger(experiment_name="ensemble_model_training",
                            log_dir="./logs/ensemble_model_training")

# Import the new Bayesian meta learner
from models.ensemble.bayesian_meta_learner import BayesianMetaLearner, train_with_optimal_parameters
from models.ensemble.ResNet import ResNetMetaLearner

<<<<<<< HEAD
# Set random seeds for reproducibility
random_seed = 19
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
os.environ['PYTHONHASHSEED'] = str(random_seed)

=======
>>>>>>> 3798e304ba09a95ae05e21747b9f93b4e52eb5fd
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
            eval_metric=['logloss', 'auc'],
            gamma=0.1,
            min_child_weight=10,
            reg_alpha=0.5,
            reg_lambda=1.0,
            subsample=0.8
        )
        
        logger.info("XGBoost meta-learner initialized with CPU-optimized settings")

    elif meta_learner_type.lower() == 'lgb':
        from lightgbm import LGBMClassifier
        import lightgbm as lgb
        meta_learner = LGBMClassifier(
            objective='binary',
            boosting_type='gbdt',
            n_jobs=-1,
            random_state=19,
            device='cpu',
            bagging_fraction=0.7,
            bagging_freq=7,
            cat_smooth=9.5,
            feature_fraction=0.64,
            learning_rate=0.06999999999999999,
            max_bin=445,
            max_depth=5,
            metric=['binary_logloss', 'average_precision', 'auc'],
            min_child_samples=470,
            min_split_gain=0.51,
            num_leaves=152,
            path_smooth=0.315,
            reg_alpha=9.8,
            reg_lambda=4.85,
            verbose=-1
        )
        logger.info("LightGBM meta-learner initialized with CPU-optimized settings")

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

    elif meta_learner_type.lower() == 'sgd':
        meta_learner = SGDClassifier(
            loss='log_loss',
            penalty='elasticnet',
            alpha=0.0001,
            l1_ratio=0.5,
            class_weight=None,
            max_iter=1000,
            early_stopping=True,
            random_state=42
        )
        
<<<<<<< HEAD
        logger.info("SGDClassifier meta-learner initialized with log_loss loss function")

    elif meta_learner_type.lower() == 'resnet':
        meta_learner = ResNetMetaLearner()
        logger.info("ResNetMetaLearner meta-learner initialized")

=======
    elif meta_learner_type.lower() == 'sgd':
        meta_learner = SGDClassifier(
            loss='log_loss',
            penalty='elasticnet',
            alpha=0.0001,
            l1_ratio=0.5,
            class_weight=None,
            max_iter=1000,
            early_stopping=True,
            random_state=42
        )
        
        logger.info("SGDClassifier meta-learner initialized with log_loss loss function")
        
    elif meta_learner_type.lower() == 'resnet':
        meta_learner = ResNetMetaLearner()
        logger.info("ResNetMetaLearner meta-learner initialized")
    
>>>>>>> 3798e304ba09a95ae05e21747b9f93b4e52eb5fd
    elif meta_learner_type.lower() == 'bayesian':
        meta_learner = BayesianMetaLearner()
        logger.info("BayesianMetaLearner meta-learner initialized")

    else:
        logger.error(f"Unknown meta_learner_type: {meta_learner_type}")
        raise ValueError(f"Unknown meta_learner_type: {meta_learner_type}. "
                        f"Supported types: 'xgb', 'logistic', 'logistic_cv', 'mlp', 'resnet', 'bayesian'")
    
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
            
            logger.info(f"Successfully trained base model: {model_name}")
            
        except Exception as e:
            logger.error(f"Error training base model {model_name}: {str(e)}")
            raise
    
    return trained_models

def train_meta_learner(meta_learner, meta_features: np.ndarray, meta_targets: np.ndarray,
                        eval_meta_features: Optional[np.ndarray] = None, 
                        eval_meta_targets: Optional[np.ndarray] = None,
                        meta_learner_type: str = 'xgb') -> object:
    """
    Train the meta-learner on meta-features from base models.
    
    Args:
        meta_learner: The meta-learner model to train
        meta_features: Meta-features for training
        meta_targets: Target values for training
        eval_meta_features: Evaluation meta-features for early stopping
        eval_meta_targets: Evaluation target values for early stopping
        meta_learner_type: Type of meta-learner model
        
    Returns:
        Trained meta-learner model
    """
    
    logger.info("Training meta-learner...")
    
    try:
        if meta_learner_type == 'resnet':
            # Special handling for ResNet meta-learner
            meta_learner.fit(
                meta_features, meta_targets,
                X_val=eval_meta_features, 
                y_val=eval_meta_targets,
                target_precision=target_precision,
                min_recall=min_recall
            )
            # ResNet has its own threshold tuning internally
            best_threshold = meta_learner.threshold
            
            # Get predictions using the tuned threshold
            y_proba = meta_learner.predict_proba(eval_meta_features)[:, 1]
            y_pred = (y_proba >= best_threshold).astype(int)
            
            # Calculate metrics with the model's threshold
            precision, recall = meta_learner._calculate_precision_recall(
                eval_meta_targets, y_pred
            )
            metrics = {'precision': precision, 'recall': recall}
        
        elif meta_learner_type == 'bayesian':
            # Special handling for Bayesian meta-learner
            meta_learner.train(meta_features, meta_targets, eval_meta_features, eval_meta_targets)
        
        # For models that support early stopping (XGBoost)
        elif hasattr(meta_learner, 'early_stopping_rounds') or hasattr(meta_learner, 'early_stopping'):
            if eval_meta_features is not None and eval_meta_targets is not None:
                # Ensure both evaluation features and targets are provided
<<<<<<< HEAD
                if meta_learner_type == 'lgb':
                    early_stopping_rounds = meta_learner.early_stopping_rounds
                    # LightGBM specific early stopping
                    meta_learner.fit(
                        meta_features, meta_targets,
                        eval_set=[(eval_meta_features, eval_meta_targets)],
                        callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds)]
                    )
                else:
                    # Generic early stopping
                    meta_learner.fit(
                        meta_features, meta_targets,
                        eval_set=[(eval_meta_features, eval_meta_targets)],
                        verbose=False
                    )
=======
                meta_learner.fit(
                    meta_features, meta_targets,
                    eval_set=[(eval_meta_features, eval_meta_targets)],
                    verbose=False
                )
>>>>>>> 3798e304ba09a95ae05e21747b9f93b4e52eb5fd
            else:
                # Fall back to standard training if eval data is missing
                logger.warning("Evaluation data missing for early stopping. Falling back to standard training.")
                meta_learner.fit(meta_features, meta_targets)
        else:
            # Standard training for other models
            meta_learner.fit(meta_features, meta_targets)
        
        # Log meta-learner performance metrics to MLflow
        try:
            if meta_learner_type == 'resnet':
                # ResNet has its own threshold tuning internally
                best_threshold = meta_learner.threshold
                
                # Get predictions using the tuned threshold
                y_proba = meta_learner.predict_proba(eval_meta_features)[:, 1]
                y_pred = (y_proba >= best_threshold).astype(int)
                
                # Calculate metrics with the model's threshold
                precision, recall = meta_learner._calculate_precision_recall(
                    eval_meta_targets, y_pred
                )
                metrics = {'precision': precision, 'recall': recall}
                mlflow.log_metrics(metrics)
            elif meta_learner_type == 'bayesian':
                # Log Bayesian meta-learner metrics
                if hasattr(meta_learner, 'metrics_') and meta_learner.metrics_:
                    for metric_name, value in meta_learner.metrics_.items():
                        mlflow.log_metric(f"meta_learner_{metric_name}", value)
            elif hasattr(meta_learner, 'evals_result'):
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

def hypertune_meta_learner(meta_features: np.ndarray, meta_targets: np.ndarray,
                            eval_meta_features: Optional[np.ndarray] = None, 
                            eval_meta_targets: Optional[np.ndarray] = None,
<<<<<<< HEAD
                            meta_learner_type='xgb', n_trials=2000, timeout=3600, 
=======
                            meta_learner_type='xgb', n_trials=500, timeout=3600, 
>>>>>>> 3798e304ba09a95ae05e21747b9f93b4e52eb5fd
                            target_precision=0.5, min_recall=0.25):
    """
    Hypertune meta-learner using Optuna and optimize threshold for precision/recall balance.
    
    Args:
        meta_features: Meta-features for training
        meta_targets: Target values for training
        eval_meta_features: Evaluation meta-features for early stopping
        eval_meta_targets: Evaluation target values for early stopping
        meta_learner_type: Type of meta-learner ('xgb', 'logistic', 'mlp', 'bayesian')
        n_trials: Number of Optuna trials
        timeout: Maximum time for optimization in seconds
        target_precision: Target precision for threshold optimization
        min_recall: Minimum required recall
        
    Returns:
        tuple: (best_meta_learner, best_threshold)
    """
    from utils.logger import ExperimentLogger
    import optuna
    from optuna.samplers import TPESampler
    import mlflow
    import numpy as np
    from sklearn.metrics import precision_score, recall_score, f1_score
    from models.ensemble.thresholds import tune_threshold_for_precision
<<<<<<< HEAD
    logger.info(f"Hyperparameter tuning for meta-learner type: {meta_learner_type}")
=======
    
>>>>>>> 3798e304ba09a95ae05e21747b9f93b4e52eb5fd
    def objective(trial):
        # Define hyperparameters based on meta-learner type
        if meta_learner_type == 'xgb':
            from xgboost import XGBClassifier
            base_params = {
                'tree_method': 'hist',
                'objective': 'binary:logistic',
                'n_jobs': -1,
                'eval_metric': ['aucpr', 'error', 'logloss'],
                'device': 'cpu',
                'random_state': 19
            }
            
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
                'max_depth': trial.suggest_int('max_depth', 4, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 200),
                'gamma': trial.suggest_float('gamma', 0.1, 3.0, step=0.1),
                'subsample': trial.suggest_float('subsample', 0.55, 0.9, step=0.01),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.55, 0.9, step=0.01),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 20.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1.00, 20.0, log=True),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 2.0, 6.0, step=0.1),
                'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 200, 1000, step=20)
            }
            params.update(base_params)
            meta_learner = XGBClassifier(**params)

<<<<<<< HEAD
        elif meta_learner_type == 'lgb':
            from lightgbm import LGBMClassifier
            import lightgbm as lgb
            base_params = {
                'objective': 'binary',
                'metric': ['binary_logloss', 'average_precision', 'auc'],
                'n_jobs': -1,
                'random_state': 19,
                'device': 'cpu',
                'verbose': -1
            }
            params = {
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.7, step=0.01),
                'bagging_freq': trial.suggest_int('bagging_freq', 5, 15),
                'cat_smooth': trial.suggest_float('cat_smooth', 1.0, 20.0, step=0.1),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.7, step=0.01),
                'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.1, step=0.01),
                'max_bin': trial.suggest_int('max_bin', 500, 800, step=10),
                'max_depth': trial.suggest_int('max_depth', 7, 11),
                'min_child_samples': trial.suggest_int('min_child_samples', 250, 350),
                'min_split_gain': trial.suggest_float('min_split_gain', 0.1, 0.3, step=0.01),
                'num_leaves': trial.suggest_int('num_leaves', 80, 100),
                'path_smooth': trial.suggest_float('path_smooth', 0.4, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 1.0, 6.0, step=0.1),
                'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 2.0, step=0.01),
                'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 200, 1000, step=20)
            }
            params.update(base_params)
            meta_learner = LGBMClassifier(**params)

=======
>>>>>>> 3798e304ba09a95ae05e21747b9f93b4e52eb5fd
        elif meta_learner_type == 'logistic':
            from sklearn.linear_model import LogisticRegression
            base_params = {
                'solver': 'saga',  # Compatible with all penalties
                'max_iter': 1000,
                'random_state': 19
            }
            params = {
                'C': trial.suggest_float('C', 0.001, 10.0, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
                'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
<<<<<<< HEAD
                'max_iter': trial.suggest_int('max_iter', 100, 1000, step=100)
=======
                'max_iter': 1000,
                'random_state': 19
>>>>>>> 3798e304ba09a95ae05e21747b9f93b4e52eb5fd
            }
            
            if params['penalty'] == 'elasticnet':
                params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)
                
            params['class_weight'] = trial.suggest_categorical('class_weight', [None, 'balanced'])
            
            if params['penalty'] == 'elasticnet':
                params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)
                
            meta_learner = LogisticRegression(**params)

        elif meta_learner_type == 'mlp':
            from sklearn.neural_network import MLPClassifier
            base_params = {
<<<<<<< HEAD
=======
                'max_iter': 1000,
>>>>>>> 3798e304ba09a95ae05e21747b9f93b4e52eb5fd
                'early_stopping': True,
                'random_state': 19
            }
            params = {
                'hidden_layer_sizes': (trial.suggest_int('hidden_units', 10, 100),),
                'alpha': trial.suggest_float('alpha', 0.0001, 0.1, log=True),
                'learning_rate_init': trial.suggest_float('learning_rate_init', 0.001, 0.1, log=True),
<<<<<<< HEAD
                'max_iter': trial.suggest_int('max_iter', 100, 1000, step=100),
                'batch_size': trial.suggest_int('batch_size', 16, 128),
                'epochs': trial.suggest_int('epochs', 50, 200),
                'patience': trial.suggest_int('patience', 10, 50)
=======
                'max_iter': 1000,
                'early_stopping': True,
                'random_state': 42
>>>>>>> 3798e304ba09a95ae05e21747b9f93b4e52eb5fd
            }
            
            meta_learner = MLPClassifier(**params)

        elif meta_learner_type == 'sgd':
            from sklearn.linear_model import SGDClassifier
            base_params = {
                'random_state': 19,
                'n_jobs': -1
            }
            params = {
                'loss': 'modified_huber',
                'penalty': trial.suggest_categorical('penalty', ['l2', 'elasticnet']),
                'alpha': trial.suggest_float('alpha', 1e-6, 0.5, log=True),
                'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
                'learning_rate': trial.suggest_categorical('learning_rate', ['optimal', 'constant', 'adaptive', 'invscaling']),
                'eta0': trial.suggest_float('eta0', 0.00001, 0.5, log=True),
                'max_iter': trial.suggest_int('max_iter', 400, 10000, step=200),
                'tol': trial.suggest_float('tol', 1e-6, 1e-2, log=True),
                'early_stopping': True,
                'validation_fraction': trial.suggest_float('validation_fraction', 0.05, 0.4, step=0.01),
                'n_iter_no_change': trial.suggest_int('n_iter_no_change', 5, 50),
                'average': trial.suggest_categorical('average', [True, False]),
            }
            
            if params['penalty'] == 'elasticnet':
                params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0, step=0.1)
                
            params.update(base_params)
            meta_learner = SGDClassifier(**params)

        elif meta_learner_type == 'resnet':
            base_params = {
                'tree_method': 'hist'  # Enforce CPU-only training
            }
            params = {
                'hidden_dim': trial.suggest_int('hidden_dim', 16, 128),
                'num_blocks': trial.suggest_int('num_blocks', 1, 4),
                'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
                'epochs': trial.suggest_int('epochs', 50, 300),
                'patience': trial.suggest_int('patience', 5, 30)
            }
            params.update(base_params)
            meta_learner = ResNetMetaLearner(**params)

        elif meta_learner_type == 'bayesian':
            # Bayesian meta-learner hyperparameters
            params = {
                'alpha': trial.suggest_float('alpha', 1e-6, 1e-3, log=True),  # Reduced range based on best alpha of ~2.18e-5
                'n_iter': trial.suggest_int('n_iter', 1000, 5000, step=500),  # Increased upper bound based on max_iter of 3500
                'tol': trial.suggest_float('tol', 1e-6, 1e-4, log=True),  # Narrowed range around best tol of ~2.79e-5
                'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
                'normalize': trial.suggest_categorical('normalize', [True, False]),
                'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
                'learning_rate': trial.suggest_categorical('learning_rate', ['adaptive', 'constant', 'optimal']),  # Added with 'adaptive' first
                'eta0': trial.suggest_float('eta0', 0.01, 0.1, log=True),  # Added based on best eta0 of ~0.029
            }
            meta_learner = BayesianMetaLearner(**params)
        else:
            raise ValueError(f"Unsupported meta-learner type: {meta_learner_type}")
        
        # Train meta-learner
        try:
            if meta_learner_type == 'resnet':
                # Special handling for ResNet meta-learner
                meta_learner.fit(
                    meta_features, meta_targets,
                    X_val=eval_meta_features, 
                    y_val=eval_meta_targets,
                    target_precision=target_precision,
                    min_recall=min_recall
                )
                # ResNet has its own threshold tuning internally
                best_threshold = meta_learner.threshold
                
                # Get predictions using the tuned threshold
                y_proba = meta_learner.predict_proba(eval_meta_features)[:, 1]
                y_pred = (y_proba >= best_threshold).astype(int)
                
                # Calculate metrics with the model's threshold
                precision, recall = meta_learner._calculate_precision_recall(
                    eval_meta_targets, y_pred
                )
                metrics = {'precision': precision, 'recall': recall}
                
            elif meta_learner_type == 'bayesian':
                # Special handling for Bayesian meta-learner
                meta_learner.train(meta_features, meta_targets, eval_meta_features, eval_meta_targets)
                
                # Get predictions on validation set
                y_proba = meta_learner.predict_proba(eval_meta_features)
                
                # Find optimal threshold
                best_threshold, metrics = tune_threshold_for_precision(
                    y_proba, eval_meta_targets, target_precision, min_recall
                )
                
<<<<<<< HEAD
            elif meta_learner_type == 'lgb':
                # LightGBM specific early stopping
                early_stopping_rounds = params.pop('early_stopping_rounds')
                meta_learner.fit(
                    meta_features, meta_targets,
                    eval_set=[(eval_meta_features, eval_meta_targets)],
                    callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds)] 
                )
=======
>>>>>>> 3798e304ba09a95ae05e21747b9f93b4e52eb5fd
            elif hasattr(meta_learner, 'early_stopping_rounds') or hasattr(meta_learner, 'early_stopping') and meta_learner_type != 'sgd':
                meta_learner.fit(
                    meta_features, meta_targets,
                    eval_set=[(eval_meta_features, eval_meta_targets)],
                    verbose=False
                )
            else:
                meta_learner.fit(meta_features, meta_targets)
                
            # Get predictions on validation set (for non-ResNet and non-Bayesian models)
            if meta_learner_type not in ['resnet', 'bayesian']:
                if hasattr(meta_learner, 'predict_proba'):
                    y_proba = meta_learner.predict_proba(eval_meta_features)[:, 1]
                else:
                    y_proba = meta_learner.predict(eval_meta_features)
                    
                # Find optimal threshold
                best_threshold, metrics = tune_threshold_for_precision(
                    y_proba, eval_meta_targets, target_precision, min_recall
                )
            
            # If we couldn't find a threshold meeting criteria, penalize the score
            if metrics['recall'] < min_recall:
                return -1.0
            
            return metrics['precision']
            
        except Exception as e:
            logger.error(f"Error in trial {trial.number}: {str(e)}")
            return -1.0
    
    # Create and run Optuna study
    study = optuna.create_study(
        direction="maximize",
<<<<<<< HEAD
        sampler = optuna.samplers.RandomSampler(seed=19)
    )
    logger.info(f"Running Optuna study with {n_trials} trials and {meta_learner_type} meta-learner")
=======
        sampler = optuna.samplers.TPESampler(                     # Different seed for better randomization
                    n_startup_trials=50,         # Reduced from 50 - more efficient
                    prior_weight=0.2,
                    seed=int(time.time())  # Dynamic seed
                )
    )
    
>>>>>>> 3798e304ba09a95ae05e21747b9f93b4e52eb5fd
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    
    # Get best parameters
    best_params = study.best_params
    logger.info(f"Best parameters of hypertuning: {best_params}")
    
    # Define base parameters outside objective function scope
    if meta_learner_type == 'xgb':
        base_params = {
            'tree_method': 'hist',
            'objective': 'binary:logistic',
            'n_jobs': -1,
            'eval_metric': ['aucpr', 'logloss', 'error'],
            'device': 'cpu',
            'random_state': 19
        }
<<<<<<< HEAD
    elif meta_learner_type == 'lgb':
        base_params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'n_jobs': -1,
            'random_state': 19,
            'device': 'cpu'
        }
=======
>>>>>>> 3798e304ba09a95ae05e21747b9f93b4e52eb5fd
    elif meta_learner_type == 'logistic':
        base_params = {
            'solver': 'saga',  # Compatible with all penalties
            'random_state': 19
        }
    elif meta_learner_type == 'mlp':
        base_params = {
            'max_iter': 1000,
            'early_stopping': True,
            'random_state': 19
        }
    elif meta_learner_type == 'sgd':
        base_params = {
            'random_state': 19,
            'n_jobs': -1
        }
    elif meta_learner_type == 'bayesian':
        # No base params needed for Bayesian meta-learner
        base_params = {}
    else:
        raise ValueError(f"Unsupported meta-learner type: {meta_learner_type}")
    
    # Train final meta-learner with best parameters
    if meta_learner_type == 'xgb':
        from xgboost import XGBClassifier
        best_params.update(base_params)
        best_meta_learner = XGBClassifier(**best_params)
<<<<<<< HEAD
    elif meta_learner_type == 'lgb':
        from lightgbm import LGBMClassifier
        import lightgbm as lgb
        best_params.update(base_params)
        best_meta_learner = LGBMClassifier(**best_params)
=======
>>>>>>> 3798e304ba09a95ae05e21747b9f93b4e52eb5fd
    elif meta_learner_type == 'logistic':
        from sklearn.linear_model import LogisticRegression
        best_params.update(base_params)
        best_meta_learner = LogisticRegression(**best_params)
    elif meta_learner_type == 'mlp':
        from sklearn.neural_network import MLPClassifier
        best_params.update(**base_params)
        if 'hidden_units' in best_params:
            best_params['hidden_layer_sizes'] = (best_params.pop('hidden_units'),)
        best_meta_learner = MLPClassifier(best_params)
    elif meta_learner_type == 'sgd':
        from sklearn.linear_model import SGDClassifier
        best_params.update(**base_params)
        best_meta_learner = SGDClassifier(**best_params)
    elif meta_learner_type == 'bayesian':
        logger.info("Training Bayesian meta-learner with optimal parameters")
        # Use the Bayesian meta-learner with optimal parameters
        best_meta_learner = BayesianMetaLearner(**best_params)
        # Special handling for Bayesian meta-learner
        best_meta_learner.train(meta_features, meta_targets, eval_meta_features, eval_meta_targets)
        
        # Get predictions on validation set
        y_proba = best_meta_learner.predict_proba(eval_meta_features)
        
        # Find optimal threshold
        best_threshold, metrics = tune_threshold_for_precision(
            y_proba, eval_meta_targets, target_precision, min_recall
        )
        
        # Log final results for Bayesian meta-learner
        logger.info(f"Final meta-learner: {meta_learner_type}")
        logger.info(f"Final threshold: {best_threshold:.4f}")
        logger.info(f"Final precision: {metrics['precision']:.4f}")
        logger.info(f"Final recall: {metrics['recall']:.4f}")
        logger.info("Bayesian meta-learner trained with optimal parameters")
        return best_meta_learner
    
    # This log message applies to all non-Bayesian meta-learners
    logger.info(f"Training final model with best parameters(Not Bayesian): {best_params}")
    # Train final model
<<<<<<< HEAD
    if meta_learner_type == 'lgb':
        early_stopping_rounds = best_params.pop('early_stopping_rounds')
        best_meta_learner.fit(
            meta_features, meta_targets,
            eval_set=[(eval_meta_features, eval_meta_targets)],
            callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds)]
        )
    elif hasattr(best_meta_learner, 'early_stopping_rounds') or hasattr(best_meta_learner, 'early_stopping') and meta_learner_type != 'sgd':
=======
    if hasattr(best_meta_learner, 'early_stopping_rounds') or hasattr(best_meta_learner, 'early_stopping') and meta_learner_type != 'sgd':
>>>>>>> 3798e304ba09a95ae05e21747b9f93b4e52eb5fd
        best_meta_learner.fit(
            meta_features, meta_targets,
            eval_set=[(eval_meta_features, eval_meta_targets)],
            verbose=False
        )
    else:
        best_meta_learner.fit(meta_features, meta_targets)
    
    # Get predictions and find optimal threshold
    if hasattr(best_meta_learner, 'predict_proba'):
        y_proba = best_meta_learner.predict_proba(eval_meta_features)[:, 1]
    else:
        y_proba = best_meta_learner.predict(eval_meta_features)
    
    best_threshold, metrics = tune_threshold_for_precision(
        y_proba, eval_meta_targets, target_precision, min_recall
    )
    
    # Log final results
    logger.info(f"Final meta-learner: {meta_learner_type}")
    logger.info(f"Final threshold: {best_threshold:.4f}")
    logger.info(f"Final precision: {metrics['precision']:.4f}")
    logger.info(f"Final recall: {metrics['recall']:.4f}")
    
    # Log to MLflow
    mlflow.log_params(best_params)
    mlflow.log_param("meta_learner_type", meta_learner_type)
    mlflow.log_param("best_threshold", best_threshold)
    mlflow.log_metrics({
        "final_precision": metrics['precision'],
        "final_recall": metrics['recall'],
        "final_f1": f1_score(eval_meta_targets, y_proba >= best_threshold)
    })
    
    return best_meta_learner


