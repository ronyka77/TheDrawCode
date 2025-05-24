"""
XGBoost Model for Soccer Draw Prediction

This module implements a XGBoost-based model for predicting soccer match draws.
It includes functionality for model creation, training, hyperparameter optimization,
threshold tuning, and MLflow integration for experiment tracking.

The implementation focuses on high precision while maintaining a minimum recall threshold.
"""

import os
import random
import time
from datetime import datetime

import mlflow
import numpy as np
import optuna
import pandas as pd
import sklearn
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    VarianceThreshold,
    f_classif,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold

from src.models.ensemble.data_utils import prepare_data
from src.utils.logger import ExperimentLogger

experiment_name = "random_forest_soccer_prediction_30"
logger = ExperimentLogger(experiment_name)

from src.models.StackedEnsemble.shared.data_loader_new import DataLoader
from src.models.StackedEnsemble.shared.hypertuner_utils import optimize_threshold
from src.utils.create_evaluation_set import (
    import_selected_features_ensemble_new,
    setup_mlflow_tracking,
)

mlrunds_dir = setup_mlflow_tracking(experiment_name)

# Global settings
min_recall = 0.30  # Minimum acceptable recall
n_trials = 10000  # Number of hyperparameter optimization trials as in notebook
# Get current versions
sklearn_version = sklearn.__version__

# Create explicit pip requirements list
pip_requirements = [f"scikit-learn=={sklearn_version}", f"mlflow=={mlflow.__version__}"]

# Update base parameters for RandomForest
base_params = {"random_state": 19, "n_jobs": 6, "verbose": 0, "criterion": "entropy"}
# Set fixed seed and hash seed for determinism
SEED = 19
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Restrict parallel threads across various libraries
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"


def load_hyperparameter_space_for_hpo():
    """
    Refined hyperparameter space based on top-performing trials.
    """
    hyperparameter_space = {
        "n_estimators": {
            "type": "int",
            "low": 800,    # Focus on range of top performers
            "high": 1300,  # Cover the successful range
            "step": 10,    # Larger step to save computation
        },
        "max_depth": {
            "type": "categorical", 
            "choices": [6, 7, 8, 9, 18, 19, 20, 21],  
        },
        "min_samples_split": {
            "type": "int",
            "low": 30,  
            "high": 80,  
            "step": 2,  
        },
        "min_samples_leaf": {
            "type": "int",
            "low": 16,  
            "high": 70,  
            "step": 2,  
        },
        "max_features": {
            "type": "categorical",  
            "choices": [0.22, 0.24, 0.26, 0.52, 0.70, 0.74, 0.84, 0.88, 0.98, 1.0],  
        },
        "class_weight": {
            "type": "float",
            "low": 1.6,   
            "high": 3.5,  
            "step": 0.05, 
        },
    }
    return hyperparameter_space


def create_model(model_params):
    """
    Create and configure RandomForest model instance.
    Args:
        model_params (dict): Model parameters
    Returns:
        RandomForestClassifier: Configured RandomForest model
    """
    try:
        params = base_params.copy()
        params.update(model_params)
        # Convert class_weight parameter to dictionary format
        if "class_weight" in params and not isinstance(params["class_weight"], dict):
            class_weight_value = params.pop("class_weight")
            params["class_weight"] = {0: 1.0, 1: class_weight_value}
        model = RandomForestClassifier(**params)
        return model

    except Exception as e:
        logger.error(f"Error creating RandomForest model: {str(e)}")
        raise


def train_model(X_train, y_train, X_test, y_test, X_eval, y_eval, model_params):
    """
    Train a RandomForest model and optimize threshold.
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Validation features
        y_test: Validation labels
        X_eval: Evaluation features
        y_eval: Evaluation labels
        model_params: Model parameters
    Returns:
        tuple: (trained_model, metrics)
    """
    try:
        model = create_model(model_params)

        # Combine training and validation data
        X_combined = pd.concat([X_train, X_test], axis=0)
        y_combined = pd.concat([y_train, y_test], axis=0)

        # Reset indexes
        X_combined.reset_index(drop=True, inplace=True)
        y_combined.reset_index(drop=True, inplace=True)
        # Fit model
        model.fit(X_combined, y_combined)

        # Get validation predictions and optimize threshold
        best_threshold, metrics = optimize_threshold(model, X_eval, y_eval, min_recall=min_recall)

        return model, metrics

    except Exception as e:
        logger.error(f"Error training RandomForest model: {str(e)}")
        raise


def optimize_hyperparameters(
    X_train, y_train, X_test, y_test, X_eval, y_eval, hyperparameter_space
):
    logger.info("Starting hyperparameter optimization")

    if not hyperparameter_space:
        hyperparameter_space = load_hyperparameter_space_for_hpo()

    best_score = -float("inf")
    best_params = {}
    # Global list to store best trials across the entire hypertuning process
    global_top_trials = []
    top_trials = []

    def objective(trial):
        try:
            params = base_params.copy()
            # Add hyperparameters from config with step size if provided
            for param_name, param_config in hyperparameter_space.items():
                if param_config["type"] == "float":
                    if "step" in param_config:
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_config["low"],
                            param_config["high"],
                            step=param_config["step"],
                            log=param_config.get("log", False),
                        )
                    else:
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_config["low"],
                            param_config["high"],
                            log=param_config.get("log", False),
                        )
                elif param_config["type"] == "int":
                    if "step" in param_config:
                        params[param_name] = trial.suggest_int(
                            param_name,
                            param_config["low"],
                            param_config["high"],
                            step=param_config["step"],
                        )
                    else:
                        params[param_name] = trial.suggest_int(
                            param_name, param_config["low"], param_config["high"]
                        )

            # Train model and get metrics
            model, metrics = train_model(X_train, y_train, X_test, y_test, X_eval, y_eval, params)

            recall = metrics.get("recall", 0.0)
            precision = metrics.get("precision", 0.0)
            threshold = metrics.get("threshold", 0.5)
            # Optimize for precision while maintaining minimum recall
            score = precision if recall >= min_recall else 0.0

            logger.info(f"Trial {trial.number}:")
            logger.info(f"  Score: {score}")
            logger.info(f"  Threshold: {threshold}")
            logger.info(f"  Params: {params}")

            for metric_name, metric_value in metrics.items():
                trial.set_user_attr(metric_name, metric_value)
            # Log to MLflow
            if score > 0.33 and score > best_score:
                log_to_mlflow(model, metrics, params, experiment_name)
            return score

        except Exception as e:
            logger.error(f"Trial failed: {str(e)}")
            return 0.0

    # Callback function defined outside the loop so that its modifications affect the outer scope.
    def callback(study, trial):
        nonlocal best_score, best_params, top_trials
        logger.info(f"Current best score in this batch: {best_score:.4f}")
        if trial.value > best_score:
            best_score = trial.value
            best_params = trial.params
            logger.info(f"New best score found in trial {trial.number}: {best_score:.4f}")
        # Create a record for the current trial
        current_run = (trial.value, trial.params, trial.number)
        top_trials.append(current_run)
        # Sort and keep only top 10 for this batch
        top_trials.sort(key=lambda x: x[0], reverse=True)
        top_trials[:] = top_trials[:10]
        table_header = "| Rank | Trial # | Score | Parameters |"
        table_separator = "|------|---------|-------|------------|"
        if trial.number % 9 == 0:
            table_rows = [
                f"| {i + 1} | {rec[2]} | {rec[0]:.4f} | {rec[1]} |"
                for i, rec in enumerate(top_trials)
            ]
            logger.info("Top trials in current batch:")
            logger.info(table_header)
            logger.info(table_separator)
            for row in table_rows:
                logger.info(row)
        # Log global top trials every 100 trials
        if trial.number % 100 == 0 and global_top_trials:
            logger.info("Global top 10 trials:")
            table_rows = [
                f"| {i + 1} | {rec[2]} | {rec[0]:.4f} | {rec[1]} |"
                for i, rec in enumerate(global_top_trials[:10])
            ]
            logger.info(table_header)
            logger.info(table_separator)
            for row in table_rows:
                logger.info(row)
        return best_score

    # Set persistent storage path using SQLite
    storage_url = "sqlite:///optuna_random_forest.db"
    study_name = "random_forest_optimization"
    # Total trials to conduct
    total_trials = n_trials  # Example; you can set n_trials accordingly.
    batch_size = 1000
    num_batches = total_trials // batch_size
    if total_trials % batch_size != 0:
        num_batches += 1

    # Loop over batches, resetting the sampler each time
    for batch in range(num_batches):
        # Create a new sampler with a dynamic seed
        random_seed = int(time.time())
        new_sampler = optuna.samplers.RandomSampler(seed=random_seed)

        # (Re-)create the study with persistent storage; previous trials are loaded automatically.
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            storage=storage_url,
            load_if_exists=True,
            sampler=new_sampler,
        )

        logger.info(
            f"Starting batch {batch + 1}/{num_batches} with new sampler (seed={random_seed})"
        )
        study.optimize(objective, n_trials=batch_size, show_progress_bar=True, callbacks=[callback], n_jobs=8)

        # Merge current batch's top trials with global_top_trials
        for trial_record in top_trials:
            global_top_trials.append(trial_record)
        # Keep only the best 10 across all batches
        global_top_trials.sort(key=lambda x: x[0], reverse=True)
        global_top_trials = global_top_trials[:10]

    # After all batches, update best_params (assume the best trial is the first in global_top_trials)
    if global_top_trials:
        best_score, best_params, best_trial_number = global_top_trials[0]
    else:
        best_params = {}

    best_params.update(base_params)

    logger.info(f"Best trial value across batches: {best_score:.4f}")
    logger.info(f"Best parameters found: {best_params}")
    logger.info("Top 10 trials across all batches:")
    for i, trial_record in enumerate(global_top_trials):
        score_val, params_val, trial_num = trial_record
        logger.info(f"| {i + 1} | {trial_num} | {score_val:.4f} | {params_val} |")

    return best_params


def hypertune_random_forest(experiment_name: str):
    """
    Main training function with MLflow tracking.
    """
    try:
        # Load hyperparameter space
        hyperparameter_space = load_hyperparameter_space_for_hpo()

        # Run hyperparameter optimization
        logger.info("Starting hyperparameter optimization")
        best_params = optimize_hyperparameters(
            X_train,
            y_train,
            X_test,
            y_test,
            X_eval,
            y_eval,
            hyperparameter_space=hyperparameter_space,
        )

        # Train final model with best parameters
        logger.info("Training final model with best parameters")
        model, metrics = train_model(
            X_train, y_train, X_test, y_test, X_eval, y_eval, best_params
        )

        return best_params, metrics

    except Exception as e:
        logger.error(f"Error in hyperparameter tuning: {str(e)}")
        return None, None


def log_to_mlflow(model, metrics, params, experiment_name):
    """
    Log trained model, metrics, and parameters to MLflow.
    Args:
        model: Trained RandomForest model
        metrics: Model evaluation metrics
        params: Model parameters
        experiment_name: Experiment name
    Returns:
        str: Run ID
    """
    try:
        # Set up MLflow tracking
        mlflow.set_experiment(experiment_name)

        # Start a new run
        with mlflow.start_run(run_name=f"rf_final_{datetime.now().strftime('%Y%m%d_%H%M')}") as run:
            # Log parameters
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)

            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Handle integer columns by converting them to float64 to properly manage missing values
            input_example = X_eval.iloc[:5].copy() if hasattr(X_eval, "iloc") else X_eval[:5].copy()

            # Identify and convert integer columns to float64 to prevent schema enforcement errors
            if hasattr(input_example, "dtypes"):
                for col in input_example.columns:
                    if X_eval[col].dtype.kind == "i":
                        logger.info(
                            f"Converting integer column '{col}' to float64 to handle potential missing values"
                        )
                        X_eval[col] = X_eval[col].astype("float64")

            # Infer signature with proper handling for integer columns with potential missing values
            signature = mlflow.models.infer_signature(input_example, model.predict(input_example))

            # Log model
            model_info = mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=f"rf_{datetime.now().strftime('%Y%m%d_%H%M')}",
                signature=signature,
            )

            logger.info(f"Model logged to MLflow: {model_info.model_uri}")
            logger.info(f"Run ID: {run.info.run_id}")
            return run.info.run_id

    except Exception as e:
        logger.error(f"Error logging to MLflow: {str(e)}")
        return None


def train_with_precision_target(X_train, y_train, X_test, y_test, X_eval, y_eval):
    """
    Train RandomForest model with focus on precision target.
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Testing features
        y_test: Testing labels
        X_eval: Evaluation features
        y_eval: Evaluation labels
    Returns:
        tuple: (best_model, best_metrics)
    """
    try:
        logger.warning("Training model with precision target.")
        params = base_params.copy()
        params.update(
            {
                "class_weight": 2.5,
                "criterion": "entropy",
                "min_samples_leaf": 26,
                "min_samples_split": 42,
                "n_estimators": 1180,
                "n_jobs": 6,
                "random_state": 19,
                "verbose": 0,
            }
        )
        # Train final model with best parameters
        logger.info("Training final model with best parameters")
        model, metrics = train_model(X_train, y_train, X_test, y_test, X_eval, y_eval, params)
        # Log to MLflow
        # log_to_mlflow(model, metrics, params, experiment_name)
        # top_features = select_top_features_rf(model, X_train)
        # compute_permutation_importance(model, X_eval, y_eval)
        compute_sklearn_permutation_importance(model, X_eval, y_eval)
        return model, metrics
    except Exception as e:
        logger.error(f"Error in precision-focused training: {str(e)}")
        return None, None


def select_top_features_rf(model: RandomForestClassifier, X_features: pd.DataFrame, n_features: int = 40) -> list[str]:
    """
    Selects the top N features based on Random Forest feature importances.

    Args:
        model: Trained RandomForestClassifier model.
        X_features: DataFrame containing the features used for training (to get names).
        n_features: The number of top features to select.

    Returns:
        A list of the names of the top N features.
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("The provided model has not been trained yet or does not support feature importances.")

    importances = model.feature_importances_
    feature_names = X_features.columns

    if len(importances) != len(feature_names):
        raise ValueError("Mismatch between the number of feature importances and feature names.")

    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    top_features = feature_importance_df['Feature'].head(n_features).tolist()
    logger.info(f"Selected top {n_features} features based on RF importance.")
    logger.info(f"Top features: {top_features}") # Log the selected features for visibility

    return top_features

def compute_permutation_importance(
    model,
    X_val: pd.DataFrame, 
    y_val: np.ndarray,
    threshold: float = 0.3,
    n_repeats: int = 20,
    number_of_features: int = 100,
) -> pd.DataFrame:
    """
    Compute permutation feature importance for a given metric and threshold.
    Args:
        model: Trained model with predict_proba(X) method.
        X_val: Validation features (DataFrame).
        y_val: Validation labels (array-like).
        metric: Metric function (e.g., sklearn.metrics.precision_score).
        threshold: Threshold for positive class prediction.
        n_repeats: Number of shuffles per feature.
        random_state: Seed for reproducibility.
    Returns:
        DataFrame with columns: ['feature', 'importance'] (mean drop in metric), sorted descending.
    """
    feature_names = X_val.columns.tolist()
    y_val_np = y_val.values
    # Compute baseline metric
    probs = model.predict_proba(X_val)[:, 1]
    preds = (probs >= threshold).astype(int)
    # Fix: metric is being passed as a float value instead of a function
    # We'll calculate precision directly since that's what was passed in
    baseline = np.sum((y_val_np == 1) & (preds == 1)) / (np.sum(preds == 1))
    logger.info(f"Baseline metric: {baseline:.4f}")
    importances = []
    for feat in feature_names:
        drops = []
        for i in range(n_repeats):
            feat_idx = feature_names.index(feat) + 1
            logger.info(f"Shuffling feature: {feat} ({feat_idx}) - Repeat: {i+1}")
            X_shuffled = X_val.copy()
            X_shuffled[feat] = np.random.permutation(X_shuffled[feat].values)
            probs_shuffled = model.predict_proba(X_shuffled)[:, 1]
            preds_shuffled = (probs_shuffled >= threshold).astype(int)
            # Calculate precision directly instead of using metric parameter
            precision = np.sum((y_val_np == 1) & (preds_shuffled == 1)) / (np.sum(preds_shuffled == 1))
            drop = baseline - precision
            drops.append(drop)
        mean_drop = np.mean(drops)
        importances.append((feat, mean_drop))
        logger.debug(f"Feature: {feat}, Mean drop: {mean_drop:.4f}")
    # Sort by importance descending
    importances.sort(key=lambda x: x[1], reverse=True)
    df_importance = pd.DataFrame(importances, columns=["feature", "importance"])
    logger.info("Top features by permutation importance:")
    logger.info(df_importance.head(number_of_features).to_string(index=False))
    return df_importance

def compute_sklearn_permutation_importance(
    model,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    n_repeats: int = 10,
    random_state: int = 19,
    n_jobs: int = 8,
    number_of_features: int = 100,
) -> pd.DataFrame:
    """
    Compute permutation feature importance using sklearn's built-in function.
    
    Args:
        model: Trained model with predict_proba(X) method.
        X_val: Validation features (DataFrame).
        y_val: Validation labels (array-like).
        n_repeats: Number of shuffles per feature.
        random_state: Seed for reproducibility.
        n_jobs: Number of parallel jobs (-1 for all cores).
        number_of_features: Number of top features to display in logs.
    
    Returns:
        DataFrame with columns: ['feature', 'importance_mean', 'importance_std'], sorted descending.
    """
    feature_names = X_val.columns.tolist()
    
    logger.info(f"Computing permutation importance with {n_repeats} repeats...")
    
    # Permutation importance - more reliable than built-in
    perm_importance = permutation_importance(
        model, X_val, y_val, 
        n_repeats=n_repeats,
        random_state=random_state, 
        n_jobs=n_jobs
    )
    
    # Create permutation importance DataFrame
    perm_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    logger.info("Top features by sklearn permutation importance:")
    logger.info(perm_df.head(number_of_features).to_string(index=False))
    
    return perm_df


def optimal_feature_selection_pipeline(X, y, target_range=(50, 70)):
    """
    Optimized pipeline for selecting 50-70 features from 260+ features
    Returns both the transformed data and the final feature names
    """
    logger.info(f"Starting with {X.shape[1]} features")
    
    # Convert to DataFrame if it's not already, and get original feature names
    if hasattr(X, 'columns'):
        original_features = X.columns.tolist()
        X_array = X.values
    else:
        original_features = [f"feature_{i}" for i in range(X.shape[1])]
        X_array = X
    
    # Stage 1: Quick Filter Methods (260+ → ~150)
    # Remove low-variance features
    variance_selector = VarianceThreshold(threshold=0.01)
    X_var = variance_selector.fit_transform(X_array)
    # Track which features survived variance filtering
    variance_mask = variance_selector.get_support()
    features_after_variance = [original_features[i] for i, keep in enumerate(variance_mask) if keep]
    logger.info(f"After variance filtering: {X_var.shape[1]} features")
    
    # Remove highly correlated features
    corr_matrix = np.corrcoef(X_var.T)
    high_corr_pairs = np.where(np.abs(corr_matrix) > 0.95)
    features_to_remove = set()
    for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
        if i != j and i not in features_to_remove:
            features_to_remove.add(j)
    
    remaining_indices = [i for i in range(X_var.shape[1]) if i not in features_to_remove]
    X_corr = X_var[:, remaining_indices]
    # Track feature names after correlation filtering
    features_after_corr = [features_after_variance[i] for i in remaining_indices]
    logger.info(f"After correlation filtering: {X_corr.shape[1]} features")
    
    # Stage 2: Statistical Selection (150 → ~100)
    k_best = SelectKBest(score_func=f_classif, k=min(100, X_corr.shape[1]))
    X_stat = k_best.fit_transform(X_corr, y)
    # Track which features survived statistical selection
    stat_mask = k_best.get_support()
    features_after_stat = [features_after_corr[i] for i, keep in enumerate(stat_mask) if keep]
    logger.info(f"After statistical selection: {X_stat.shape[1]} features")
    logger.info(f"Features after statistical selection: {features_after_stat}")
    # Stage 3: Model-based Selection (100 → 50-70)
    rf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42, n_jobs=8)
    
    # Option A: Boruta for all-relevant features (with relaxed parameters)
    boruta = BorutaPy(
        rf, 
        n_estimators='auto', 
        verbose=1, 
        random_state=42,
        alpha=0.3,  # More lenient (default is 0.05) - allows more features
        max_iter=200,  # More iterations to find features (default is 100)
        perc=70  # Use 90th percentile instead of 100th for shadow features
    )
    boruta.fit(X_stat, y)
    X_boruta = boruta.transform(X_stat)
    # Track which features survived Boruta
    boruta_mask = boruta.support_
    features_after_boruta = [features_after_stat[i] for i, keep in enumerate(boruta_mask) if keep]
    logger.info(f"After Boruta selection: {X_boruta.shape[1]} features")
    logger.info(f"Boruta confirmed features: {sum(boruta.support_)}")
    logger.info(f"Boruta tentative features: {sum(boruta.support_weak_)}")
    
    # If Boruta is still too conservative, include tentative features
    if X_boruta.shape[1] < target_range[0]:  # If less than 50 features
        logger.info("Boruta selected too few features, including tentative features...")
        # Combine confirmed and tentative features
        combined_mask = boruta.support_ | boruta.support_weak_
        X_boruta_extended = X_stat[:, combined_mask]
        features_after_boruta_extended = [features_after_stat[i] for i, keep in enumerate(combined_mask) if keep]
        logger.info(f"After including tentative features: {X_boruta_extended.shape[1]} features")
        
        # Use the extended set for RFE
        X_boruta = X_boruta_extended
        features_after_boruta = features_after_boruta_extended
    
    # Option B: RFE for exact number
    target_features = min(target_range[1], max(target_range[0], X_boruta.shape[1]))
    rfe = RFE(rf, n_features_to_select=target_features, step=1)
    X_final = rfe.fit_transform(X_boruta, y)
    # Track which features survived RFE
    rfe_mask = rfe.get_support()
    final_feature_names = [features_after_boruta[i] for i, keep in enumerate(rfe_mask) if keep]
    
    logger.info(f"Final features selected: {X_final.shape[1]}")
    logger.info(f"Final feature names: {final_feature_names}")
    
    # Log top features for visibility
    logger.info(f"All final features: {final_feature_names}")
    
    return X_final, {
        'final_feature_names': final_feature_names,
        'variance_selector': variance_selector,
        'correlation_indices': remaining_indices,
        'statistical_selector': k_best,
        'boruta_selector': boruta,
        'final_selector': rfe
    }

def random_forest_staged_selection(X, y, X_eval, y_eval, target_features=80):
    """Multi-stage Random Forest feature selection with different objectives"""
    
    logger.info(f"Starting Random Forest staged selection with {X.shape[1]} initial features")
    # Combine training and test data for feature selection
    
    
    logger.info(f"Combined data shape: {X.shape}")
    # Stage 1: Quick filter with fewer trees
    logger.info("Stage 1: Quick filter with fewer trees")
    rf_fast = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    sample_weight = np.ones(len(y)) / len(y)
    rf_fast.fit(X, y, sample_weight=sample_weight)
    stage1_importance = rf_fast.feature_importances_
    stage1_features = X.columns[np.argsort(stage1_importance)[-200:]].tolist()

    logger.info(f"Stage 1: Selected {len(stage1_features)} features")

    # Stage 2: Refined selection with cross-validation
    logger.info("Stage 2: Refined selection with cross-validation")
    X_stage1 = X[stage1_features]
    X_eval_stage1 = X_eval[stage1_features]

    rf_refined = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )

    # Cross-validation feature importance
    cv_scores = []
    cv_importances = []
    val_weight = np.ones(len(y_eval)) / len(y_eval)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(X_stage1, y):
        X_train_cv, X_val_cv = X_stage1.iloc[train_idx], X_stage1.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        sample_weight = np.ones(len(y_train_cv)) / len(y_train_cv)
        rf_refined.fit(X_train_cv, y_train_cv)
        cv_importances.append(rf_refined.feature_importances_)

        val_score = rf_refined.score(X_eval_stage1, y_eval)
        cv_scores.append(val_score)

    # Average importance across folds
    avg_importance = np.mean(cv_importances, axis=0)
    stage2_features = [stage1_features[i] for i in np.argsort(avg_importance)[-target_features:]]

    logger.info(f"Stage 2: Selected {len(stage2_features)} features: {stage2_features}")
    logger.info(f"CV Score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    return stage2_features, avg_importance


def main():
    """
    Main execution function.
    """
    try:
        logger.info("Starting RandomForest model training")
        # Import data at runtime to avoid global scope issues
        global X_train, y_train, X_test, y_test, X_eval, y_eval

        # Load data
        dataloader = DataLoader()
        X_train, y_train, X_test, y_test, X_eval, y_eval = dataloader.load_data()
        features = import_selected_features_ensemble_new(model_type="all")
        X_train = prepare_data(X_train, features)
        X_test = prepare_data(X_test, features)
        X_eval = prepare_data(X_eval, features)
        # Log data shapes
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Testing data shape: {X_test.shape}")
        logger.info(f"Evaluation data shape: {X_eval.shape}")
        logger.info(
            f"Positive class ratio - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}, Eval: {y_eval.mean():.3f}"
        )

        # current_params, current_metrics = hypertune_random_forest(experiment_name)
        # logger.info(f"Run completed with parameters: {current_params}")
        # logger.info(f"Run metrics: {current_metrics}")

        # Train model with precision target
        # X_train_optimal, selectors_info = optimal_feature_selection_pipeline(X_train, y_train)
        # final_feature_names = selectors_info['final_feature_names']
        # logger.info(f"Feature selection completed. Selected {len(final_feature_names)} features:")
        # for i, feature_name in enumerate(final_feature_names, 1):
        #     logger.info(f"  {i:2d}. {feature_name}")
        X_combined = pd.concat([X_train, X_test], axis=0, ignore_index=True)
        y_combined = pd.concat([y_train, y_test], axis=0, ignore_index=True)
        stage2_features, avg_importance = random_forest_staged_selection(X_combined, y_combined, X_eval, y_eval, target_features=80)
        

        
        # best_model, best_metrics = train_with_precision_target(
        #     X_train, y_train, X_test, y_test, X_eval, y_eval
        # )
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")


if __name__ == "__main__":
    main()
