# -*- coding: utf-8 -*-
"""
Model training module for PyCaret soccer prediction.

This module contains functions for training and evaluating models
using PyCaret.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import os
from pycaret.classification import setup, compare_models, tune_model, ensemble_model, predict_model, save_model


# Add project root to Python path if needed
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from utils.logger import ExperimentLogger
from models.pycaret.mlflow_module import save_model_and_predictions
from models.pycaret.feature_engineering import get_feature_importance
from models.pycaret.threshold_utils import precision_focused_score

# Setup logger
logger = ExperimentLogger(experiment_name="pycaret_model_training")

def setup_pycaret_environment(data, target_col='target', val_data=None, experiment_name='pycaret_soccer_prediction',
                                fix_imbalance=True, normalize=True, feature_selection=False,
                                custom_scorer=None, fold=5, session_id=42):
    """
    Set up the PyCaret environment for model training.
    
    Args:
        data (pd.DataFrame): DataFrame with features and target for training
        target_col (str): Name of the target column
        val_data (pd.DataFrame, optional): Validation data for early stopping
        experiment_name (str): Name of the experiment
        fix_imbalance (bool): Whether to fix class imbalance
        normalize (bool): Whether to normalize features
        feature_selection (bool): Whether to perform feature selection
        custom_scorer (callable, optional): Custom scoring function
        fold (int): Number of cross-validation folds
        session_id (int): Random seed
        
    Returns:
        object: PyCaret setup object
    """
    try:
        from pycaret.classification import setup
    except ImportError:
        logger.error("setup_pycaret_environment: PyCaret not installed. Please install with 'pip install pycaret[full]'")
        return None
    
    logger.info(f"Setting up PyCaret environment for experiment '{experiment_name}'")
    
    # Log data info
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Target distribution: {data[target_col].value_counts(normalize=True)}")
    
    # Identify categorical features (if any)
    categorical_features = []
    for col in data.columns:
        if col != target_col and data[col].dtype == 'object':
            categorical_features.append(col)
    
    # If using SMOTENC but no categorical features found, switch to regular SMOTE
    fix_imbalance_method = None
    if fix_imbalance:
        if categorical_features:
            fix_imbalance_method = 'smotenc'
            logger.info(f"Using SMOTENC with {len(categorical_features)} categorical features")
        else:
            fix_imbalance_method = 'smote'
            logger.info("No categorical features found, using regular SMOTE")
    
    # Set up PyCaret - updated for PyCaret 3.3.2
    setup_obj = setup(
        data=data,
        target=target_col,
        session_id=session_id,
        experiment_name=experiment_name,
        log_experiment=False,  # Set to False to avoid MLflow errors, we'll handle MLflow separately
        log_plots=True,
        
        # Preprocessing parameters
        preprocess=True,
        imputation_type='simple',
        numeric_imputation='mean',
        categorical_imputation='mode',
        categorical_features=categorical_features if categorical_features else None,
        
        # Sampling parameters
        fix_imbalance=fix_imbalance,
        fix_imbalance_method=fix_imbalance_method,
        
        # Scaling parameters
        normalize=normalize,
        normalize_method='robust' if normalize else None,
        
        # Feature selection
        feature_selection=feature_selection,
        feature_selection_method='classic' if feature_selection else None,
        
        # Other parameters
        remove_outliers=False,  # We handle outliers separately
        outliers_method=None,
        transformation=False,  # No automatic transformations
        transformation_method=None,
        
        # Cross-validation
        fold_strategy='stratifiedkfold',
        fold=fold,
        
        # Performance
        n_jobs=-1,
        use_gpu=False,  # CPU-only as per requirements
        
        # Visualization and output
        html=False,  # Disable HTML output
        verbose=True,
        create_date_columns=False
    )
    
    logger.info("PyCaret environment setup complete")
    
    # Store validation data in global variable or environment for use in other functions
    if val_data is not None:
        logger.info(f"External validation data provided: {val_data.shape}")
        # Store validation data in a global variable for later access
        global _validation_data
        _validation_data = {
            'data': val_data,
            'X': val_data.drop(columns=[target_col]) if target_col in val_data.columns else val_data,
            'y': val_data[target_col] if target_col in val_data.columns else None
        }
        logger.info("Validation data stored for model training")
    
    return setup_obj

def compare_models_for_precision(n_select=3, include=None, exclude=None, fold=5, sort='Prec', custom_metric=None):
    """
    Compare multiple models with a focus on precision.
    
    Args:
        n_select (int): Number of top models to return
        include (list, optional): List of models to include
        exclude (list, optional): List of models to exclude
        fold (int): Number of cross-validation folds
        sort (str): Metric to sort by
        custom_metric (callable, optional): Custom scoring function
        
    Returns:
        list: List of top models
    """
    try:
        from pycaret.classification import compare_models
    except ImportError:
        logger.error("compare_models_for_precision: PyCaret not installed. Please install with 'pip install pycaret[full]'")
        return []
    
    logger.info(f"Comparing models with focus on {sort}")
    
    # Define models to include/exclude
    if include is None:
        include = ['xgboost', 'lightgbm', 'catboost']
    
    # Compare models - updated for PyCaret 3.3.2
    try:
        # Check if custom_metric is supported in this version
        import inspect
        compare_models_params = inspect.signature(compare_models).parameters
        
        # Build kwargs based on available parameters
        kwargs = {
            'n_select': n_select,
            'include': include,
            'exclude': exclude,
            'fold': fold,
            'sort': sort,
            'verbose': True
        }
        
        # Only add custom_metric if it's supported
        if 'custom_metric' in compare_models_params and custom_metric is not None:
            kwargs['custom_metric'] = custom_metric
        elif custom_metric is not None:
            logger.warning("custom_metric parameter is not supported in this version of PyCaret. Ignoring.")
        
        # Call compare_models with appropriate parameters
        top_models = compare_models(**kwargs)
        
        # Convert to list if only one model returned
        if not isinstance(top_models, list):
            top_models = [top_models]
        
        # Get model comparison results
        try:
            from pycaret.classification import pull
            comparison_df = pull()
            
            # Save results to CSV
            comparison_path = 'model_comparison_results.csv'
            comparison_df.to_csv(comparison_path, index=True)
            logger.info(f"Model comparison results saved to {comparison_path}")
            
            # Log top models - updated for PyCaret 3.3.2 column names
            for i, model in enumerate(top_models):
                if i < len(comparison_df):
                    model_name = comparison_df.index[i]
                    try:
                        # Try to get metrics - column names might vary
                        metrics = {}
                        for metric in ['Prec', 'Precision', 'Recall', 'F1', 'AUC']:
                            if metric in comparison_df.columns:
                                metrics[metric] = comparison_df.loc[model_name, metric]
                        
                        logger.info(f"Model {i+1}: {model_name}")
                        for metric, value in metrics.items():
                            logger.info(f" - {metric}: {value:.4f}")
                    except Exception as e:
                        logger.warning(f"Could not log metrics for model {model_name}: {str(e)}")
        except Exception as e:
            logger.error(f"Error getting model comparison results: {str(e)}")
        
        return top_models
    except Exception as e:
        logger.error(f"Error during model comparison: {str(e)}")
        return []

def tune_model_for_precision(model, optimize='Prec', n_iter=50, custom_grid=None, custom_metric=None, 
                            use_validation_data=True):
    """
    Tune a model with a focus on precision.
    
    Args:
        model: Model to tune
        optimize (str): Metric to optimize
        n_iter (int): Number of iterations for hyperparameter tuning
        custom_grid (dict, optional): Custom hyperparameter grid
        custom_metric (callable, optional): Custom scoring function
        use_validation_data (bool): Whether to use stored validation data for early stopping
        
    Returns:
        object: Tuned model
    """
    try:
        from pycaret.classification import tune_model
    except ImportError:
        logger.error("tune_model_for_precision: PyCaret not installed. Please install with 'pip install pycaret[full]'")
        return model
    
    logger.info(f"Tuning model with focus on {optimize}")
    
    # Determine if we should use validation-based training
    use_custom_training = False
    if use_validation_data and '_validation_data' in globals() and globals()['_validation_data'] is not None:
        model_type = str(type(model).__name__).lower()
        if any(x in model_type for x in ['xgboost', 'lightgbm', 'catboost']):
            use_custom_training = True
            logger.info(f"Using validation-based early stopping for {model_type}")
        else:
            logger.info(f"Using PyCaret's early stopping for {model_type}")
    
    # Use custom training if appropriate
    if use_custom_training:
        val_data = globals()['_validation_data']
        
        # Dispatch to appropriate custom training function based on model type
        if 'xgboost' in model_type:
            return train_xgboost_with_validation(model, val_data, optimize, n_iter)
        elif 'lightgbm' in model_type:
            return train_lightgbm_with_validation(model, val_data, optimize, n_iter)
        elif 'catboost' in model_type:
            return train_catboost_with_validation(model, val_data, optimize, n_iter)
    
    # If custom training isn't being used, fall back to PyCaret's tune_model
    # Check if parameters are supported in this version
    import inspect
    tune_model_params = inspect.signature(tune_model).parameters
    
    # Build kwargs based on available parameters
    kwargs = {
        'estimator': model,
        'optimize': optimize,
        'n_iter': n_iter,
        'search_algorithm': 'optuna',
        'early_stopping': 'auto',
        'early_stopping_max_iters': 10,
        'verbose': True
    }
    
    # Only add custom_grid if it's supported
    if 'custom_grid' in tune_model_params and custom_grid is not None:
        kwargs['custom_grid'] = custom_grid
    
    # Only add custom_metric if it's supported
    if 'custom_metric' in tune_model_params and custom_metric is not None:
        kwargs['custom_metric'] = custom_metric
    elif custom_metric is not None:
        logger.warning("custom_metric parameter is not supported in this version of PyCaret. Ignoring.")
    
    # Call tune_model with appropriate parameters
    tuned_model = tune_model(**kwargs)
    
    # Get tuning results
    try:
        from pycaret.classification import pull
        tuning_df = pull()
        
        # Save results to CSV
        tuning_path = 'model_tuning_results.csv'
        tuning_df.to_csv(tuning_path, index=True)
        logger.info(f"Model tuning results saved to {tuning_path}")
        
        # Log best parameters
        try:
            logger.info(f"Best parameters: {tuned_model.get_params()}")
        except:
            logger.warning("Could not retrieve model parameters")
    except Exception as e:
        logger.error(f"Error getting model tuning results: {str(e)}")
    
    return tuned_model

def create_ensemble_model(models, method='Stacking', optimize='Prec', weights=None, custom_metric=None):
    """
    Create an ensemble model from multiple base models.
    
    Args:
        models (list): List of models to ensemble
        method (str): Ensemble method ('Stacking', 'Bagging', or 'Boosting')
        optimize (str): Metric to optimize
        weights (list, optional): List of weights for each model
        custom_metric (callable, optional): Custom scoring function
        
    Returns:
        object: Ensemble model
    """
    try:
        from pycaret.classification import ensemble_model
    except ImportError:
        logger.error("create_ensemble_model: PyCaret not installed. Please install with 'pip install pycaret[full]'")
        return models[0] if models else None
    
    logger.info(f"Creating {method} ensemble from {len(models)} models")
    
    # Create ensemble - updated for PyCaret 3.3.2
    try:
        # Check if parameters are supported in this version
        import inspect
        ensemble_model_params = inspect.signature(ensemble_model).parameters
        
        # Build kwargs based on available parameters
        kwargs = {
            'estimator': models,
            'method': method,
            'optimize': optimize,
            'verbose': True,
            'n_estimators': 100,
            'choose_better': True
        }
        
        # Only add weights if it's supported
        if 'weights' in ensemble_model_params and weights is not None:
            kwargs['weights'] = weights
        elif weights is not None:
            logger.warning("weights parameter is not supported in this version of PyCaret. Ignoring.")
        
        # Only add custom_metric if it's supported
        if 'custom_metric' in ensemble_model_params and custom_metric is not None:
            kwargs['custom_metric'] = custom_metric
        elif custom_metric is not None:
            logger.warning("custom_metric parameter is not supported in this version of PyCaret. Ignoring.")
        
        # Call ensemble_model with appropriate parameters
        ensemble = ensemble_model(**kwargs)
        
        # Get ensemble results
        try:
            from pycaret.classification import pull
            ensemble_df = pull()
            
            # Save results to CSV
            ensemble_path = 'ensemble_results.csv'
            ensemble_df.to_csv(ensemble_path, index=True)
            logger.info(f"Ensemble results saved to {ensemble_path}")
            
            # Log ensemble metrics - updated for PyCaret 3.3.2 column names
            logger.info(f"Ensemble metrics:")
            try:
                # Try to get metrics - column names might vary
                metrics = {}
                for metric in ['Prec', 'Precision', 'Recall', 'F1', 'AUC']:
                    if metric in ensemble_df.columns:
                        metrics[metric] = ensemble_df.iloc[0][metric]
                
                for metric, value in metrics.items():
                    logger.info(f" - {metric}: {value:.4f}")
            except Exception as e:
                logger.warning(f"Could not log ensemble metrics: {str(e)}")
        except Exception as e:
            logger.error(f"Error getting ensemble results: {str(e)}")
        
        return ensemble
    except Exception as e:
        logger.error(f"Error during ensemble creation: {str(e)}")
        return models[0] if models else None

def evaluate_model_on_holdout(model, holdout_data, target_col='target'):
    """
    Evaluate a model on a holdout dataset.
    
    Args:
        model: Trained model
        holdout_data (pd.DataFrame): Holdout dataset
        target_col (str): Target column name
        
    Returns:
        dict: Evaluation metrics
    """
    try:
        from pycaret.classification import predict_model
        import numpy as np
        import pandas as pd
    except ImportError:
        logger.error("evaluate_model_on_holdout: PyCaret not installed. Please install with 'pip install pycaret[full]'")
        return {}
    
    logger.info("Evaluating model on holdout data")
    
    try:
        # Transform data using PyCaret's pipeline if needed
        try:
            # Get the preprocessing pipeline from PyCaret's variables
            # Instead of using get_config which might not be available
            from pycaret.classification import get_config
            
            # Try to access the pipeline directly from available variables
            try:
                pipeline = get_config('pipeline')
                logger.info("Successfully retrieved preprocessing pipeline from 'pipeline' variable")
            except:
                # Check if we can access the variables dictionary
                variables = get_config('variables')
                if 'pipeline' in variables:
                    pipeline = variables['pipeline']
                    logger.info("Successfully retrieved preprocessing pipeline from variables dictionary")
                else:
                    pipeline = None
                    logger.info("No preprocessing pipeline found in variables dictionary")
            
            # Check if preprocessing pipeline exists and if shapes don't match
            if pipeline and hasattr(model, 'n_features_') and model.n_features_ != holdout_data.shape[1] - 1:
                logger.info(f"Applying preprocessing pipeline to holdout data. Original shape: {holdout_data.shape}")
                
                # Split features and target before transformation
                X_holdout = holdout_data.drop(target_col, axis=1)
                y_holdout = holdout_data[target_col]
                
                # Apply preprocessing
                X_transformed = pipeline.transform(X_holdout)
                
                # Debugging info
                logger.info(f"Transformed holdout data shape: {X_transformed.shape}")
                
                # Recreate holdout_data with transformed features
                # Note: This assumes the pipeline outputs a numpy array - adjust if it outputs DataFrame
                if isinstance(X_transformed, np.ndarray):
                    feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]
                    holdout_data_transformed = pd.DataFrame(X_transformed, columns=feature_names)
                    holdout_data_transformed[target_col] = y_holdout.reset_index(drop=True)
                    holdout_data = holdout_data_transformed
                else:
                    # If pipeline returns DataFrame or sparse matrix, handle accordingly
                    holdout_data_transformed = pd.DataFrame(X_transformed)
                    holdout_data_transformed[target_col] = y_holdout.reset_index(drop=True)
                    holdout_data = holdout_data_transformed
                
                logger.info(f"Holdout data prepared with transformed features: {holdout_data.shape}")
            else:
                logger.info("No preprocessing needed or no preprocessing pipeline found")
        except Exception as transform_error:
            logger.warning(f"Error applying preprocessing: {transform_error}. Using original data.")
        
        # Split features and target
        X_holdout = holdout_data.drop(target_col, axis=1)
        y_holdout = holdout_data[target_col]
        
        # Check if parameters are supported in this version
        import inspect
        predict_model_params = inspect.signature(predict_model).parameters
        
        # Build kwargs based on available parameters
        kwargs = {
            'data': holdout_data,
            'verbose': True
        }
        
        # Use 'estimator' or 'model' parameter based on what's supported
        if 'estimator' in predict_model_params:
            kwargs['estimator'] = model
        else:
            kwargs['model'] = model
        
        # Make predictions - updated for PyCaret 3.3.2
        holdout_preds = predict_model(**kwargs)
        
        # Extract metrics
        metrics = {}
        try:
            # Try to get metrics from PyCaret
            from pycaret.classification import pull
            metrics_df = pull()
            
            # Save metrics to CSV
            metrics_path = 'holdout_metrics.csv'
            metrics_df.to_csv(metrics_path, index=True)
            logger.info(f"Holdout metrics saved to {metrics_path}")
            
            # Extract metrics - column names might vary in PyCaret 3.3.2
            for metric in ['Prec', 'Precision', 'Recall', 'F1', 'AUC']:
                if metric in metrics_df.columns:
                    metrics[metric] = metrics_df.iloc[0][metric]
        except Exception as e:
            logger.warning(f"Could not extract metrics from PyCaret: {str(e)}")
            
            # Calculate metrics manually if PyCaret extraction fails
            try:
                from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
                
                # Get prediction column name (might be 'prediction_label' or 'prediction_class')
                pred_col = None
                for col in ['prediction_label', 'prediction_class', 'Label']:
                    if col in holdout_preds.columns:
                        pred_col = col
                        break
                
                if pred_col:
                    y_pred = holdout_preds[pred_col]
                    metrics['Precision'] = precision_score(y_holdout, y_pred)
                    metrics['Recall'] = recall_score(y_holdout, y_pred)
                    metrics['F1'] = f1_score(y_holdout, y_pred)
                    
                    # Get probability column name (might be 'prediction_score' or 'Score')
                    prob_col = None
                    for col in ['prediction_score', 'Score']:
                        if col in holdout_preds.columns:
                            prob_col = col
                            break
                    
                    if prob_col:
                        y_prob = holdout_preds[prob_col]
                        metrics['AUC'] = roc_auc_score(y_holdout, y_prob)
            except Exception as e2:
                logger.error(f"Could not calculate metrics manually: {str(e2)}")
        
        # Log metrics
        logger.info("Holdout evaluation metrics:")
        for metric, value in metrics.items():
            logger.info(f" - {metric}: {value:.4f}")
        
        # Save predictions
        preds_path = 'holdout_predictions.csv'
        holdout_preds.to_csv(preds_path, index=False)
        logger.info(f"Holdout predictions saved to {preds_path}")
        
        return metrics
    except Exception as e:
        logger.error(f"Error during holdout evaluation: {str(e)}")
        return {}

def train_xgboost_with_validation(base_model, validation_data, optimize='Prec', n_iter=50):
    """
    Train XGBoost with validation data for early stopping.
    
    Args:
        base_model: Base XGBoost model
        validation_data (dict): Dict with validation data and targets
        optimize (str): Metric to optimize
        n_iter (int): Max iterations for training
        
    Returns:
        object: Trained XGBoost model
    """
    import xgboost as xgb
    from xgboost import XGBClassifier
    from pycaret.classification import get_config
    from utils.logger import ExperimentLogger
    
    logger = ExperimentLogger()
    
    # Get training data from PyCaret
    train_X = get_config('X_train_transformed')
    train_y = get_config('y_train_transformed')
    
    # Get validation data
    val_X = validation_data['X']
    val_y = validation_data['y']
    
    # Determine which parameters to keep from base model
    try:
        base_params = base_model.get_params()
        # Filter to XGBoost relevant parameters
        xgb_params = {
            'tree_method': 'hist',          # CPU-optimized algorithm
            'device': 'cpu',                # Enforce CPU usage (per requirements)
            'n_jobs': -1,                   # Use all available cores
            'objective': 'binary:logistic', # Binary classification
            'learning_rate': 0.05,          # Conservative learning rate
            'n_estimators': 500,            # Maximum number of trees
            'max_depth': 6,                 # Reduced to avoid overfitting
            'colsample_bytree': 0.7,        # Feature subsampling
            'gamma': 0.8,                   # Minimum loss reduction for split
            'min_child_weight': 50,         # Min sum of weights needed in child
            'reg_alpha': 1.0,               # L1 regularization
            'reg_lambda': 2.0,              # L2 regularization (increased)
            'scale_pos_weight': 2.19,       # For class imbalance
            'subsample': 0.8                # Row subsampling
        }
    except:
        # Default parameters if we can't get them from base model
        xgb_params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'hist',          # CPU-optimized algorithm
            'device': 'cpu'                 # Enforce CPU usage
        }
    
    # Map PyCaret metrics to XGBoost metrics
    metric_mapping = {
        'Precision': 'aucpr',  # Precision-recall AUC
        'Prec': 'aucpr',
        'Recall': 'aucpr',
        'AUC': 'auc',
        'F1': 'aucpr',
        'Accuracy': 'error',
        'Logloss': 'logloss'
    }
    xgb_metric = metric_mapping.get(optimize, 'aucpr')
    
    # Create XGBoost classifier
    model = XGBClassifier(**xgb_params)
    
    # Train model with early stopping
    model.fit(
        train_X, 
        train_y,
        eval_set=[(val_X, val_y)],
        eval_metric=xgb_metric,
        early_stopping_rounds=300,  # As per requirements (300-500 early stopping rounds)
        verbose=100
    )
    
    logger.info(f"XGBoost training complete with {model.get_booster().best_ntree_limit} trees")
    return model

def train_lightgbm_with_validation(base_model, validation_data, optimize='Prec', n_iter=50):
    """
    Train LightGBM with validation data for early stopping.
    
    Args:
        base_model: Base LightGBM model
        validation_data (dict): Dict with validation data and targets
        optimize (str): Metric to optimize
        n_iter (int): Max iterations for training
        
    Returns:
        object: Trained LightGBM model
    """
    from lightgbm import LGBMClassifier
    from pycaret.classification import get_config

    # Get training data from PyCaret
    train_X = get_config('X_train_transformed')
    train_y = get_config('y_train_transformed')
    
    # Get validation data
    val_X = validation_data['X']
    val_y = validation_data['y']
    
    # Determine which parameters to keep from base model
    try:
        base_params = base_model.get_params()
        # Filter to LightGBM relevant parameters
        lgb_params = {
            'learning_rate': 0.05,          # Conservative learning rate
            'num_leaves': 32,               # Reduced from original 52
            'max_depth': 4,                 # Shallow trees to avoid overfitting
            'min_child_samples': 100,       # Min samples in leaf node
            'feature_fraction': 0.8,        # Feature subsampling
            'bagging_fraction': 0.8,        # Row subsampling
            'bagging_freq': 5,              # Frequency for bagging
            'reg_alpha': 2.0,               # L1 regularization (increased)
            'reg_lambda': 2.0,              # L2 regularization (increased)
            'min_split_gain': 0.1,          # Min gain for split
            'objective': 'binary',          # Binary classification
            'metric': ['binary_logloss', 'auc'], # Metrics to evaluate
            'verbose': -1,                  # Quiet mode
            'device': 'cpu',                # Enforce CPU usage
            'n_jobs': -1                    # Use all available cores
        }
    except:
        # Default parameters if we can't get them from base model
        lgb_params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'device': 'cpu'                 # Enforce CPU usage
        }
    
    # Map PyCaret metrics to LightGBM metrics
    metric_mapping = {
        'Precision': 'auc',  # LightGBM doesn't have precision-specific metric
        'Prec': 'auc',
        'Recall': 'auc',
        'AUC': 'auc',
        'F1': 'auc',
        'Accuracy': 'binary_error',
        'Logloss': 'binary_logloss'
    }
    lgb_metric = metric_mapping.get(optimize, 'auc')
    
    # Create LightGBM classifier
    model = LGBMClassifier(
        **lgb_params,
        objective='binary',
        metric=lgb_metric,
        random_state=42
    )
    
    # Train model with early stopping
    model.fit(
        train_X, 
        train_y,
        eval_set=[(val_X, val_y)],
        eval_metric=lgb_metric,
        early_stopping_rounds=300,  # As per requirements (300-500 early stopping rounds)
        verbose=100
    )
    
    logger.info(f"LightGBM training complete with {model.best_iteration_} iterations")
    return model

def train_catboost_with_validation(base_model, validation_data, optimize='Prec', n_iter=50):
    """
    Train CatBoost with validation data for early stopping.
    
    Args:
        base_model: Base CatBoost model
        validation_data (dict): Dict with validation data and targets
        optimize (str): Metric to optimize
        n_iter (int): Max iterations for training
        
    Returns:
        object: Trained CatBoost model
    """
    from catboost import CatBoostClassifier, Pool
    from pycaret.classification import get_config
    
    # Get training data from PyCaret
    train_X = get_config('X_train_transformed')
    train_y = get_config('y_train_transformed')
    
    # Get validation data
    val_X = validation_data['X']
    val_y = validation_data['y']
    
    # Determine which parameters to keep from base model
    try:
        base_params = base_model.get_params()
        # Filter to CatBoost relevant parameters
        cat_params = {
            'learning_rate': 0.05,          # Conservative learning rate
            'depth': 6,                     # Reduced from original 10
            'min_data_in_leaf': 20,         # Min samples in leaf node
            'subsample': 0.8,               # Row subsampling
            'colsample_bylevel': 0.8,       # Feature subsampling
            'reg_lambda': 5.0,              # L2 regularization (increased)
            'leaf_estimation_iterations': 2,# Leaf value calculation iterations
            'bagging_temperature': 1.0,     # Randomness in bagging
            'scale_pos_weight': 2.19,       # For class imbalance
            'loss_function': 'Logloss',     # Binary cross-entropy
            'eval_metric': 'AUC',           # Evaluation metric
            'task_type': 'CPU',             # Enforce CPU usage
            'thread_count': -1,             # Use all available cores
            'random_seed': 42               # For reproducibility
        }
    except:
        # Default parameters if we can't get them from base model
        cat_params = {
            'depth': 6,
            'learning_rate': 0.1,
            'iterations': 500
        }
    
    # Map PyCaret metrics to CatBoost metrics
    metric_mapping = {
        'Precision': 'Precision',
        'Prec': 'Precision',
        'Recall': 'Recall',
        'AUC': 'AUC',
        'F1': 'F1',
        'Accuracy': 'Accuracy',
        'Logloss': 'Logloss'
    }
    cat_metric = metric_mapping.get(optimize, 'AUC')
    
    # Configure model
    params = {
        'loss_function': 'Logloss',
        'eval_metric': cat_metric,
        'random_seed': 42,
        'verbose': False
    }
    params.update(cat_params)
    
    # Create model with parameters
    model = CatBoostClassifier(**params)
    
    # Create validation pool
    eval_set = Pool(val_X, val_y)
    
    # Train model with early stopping
    model.fit(
        train_X, train_y,
        eval_set=eval_set,
        early_stopping_rounds=50,
        verbose=False
    )
    
    # CatBoostClassifier already has scikit-learn interface but let's ensure it's properly structured
    model.n_features_in_ = train_X.shape[1]  # Ensure n_features attribute is set
    
    logger.info(f"CatBoost training complete with {model.get_best_iteration()} iterations (verified scikit-learn compatibility)")
    return model
