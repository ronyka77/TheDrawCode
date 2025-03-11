"""
Feature Selector using Composite Importance Ranking

This file implements a feature selector that leverages the feature importance scores
from XGBoost, CatBoost, and LightGBM. The composite score is computed by averaging 
the normalized importance scores from all three models. This approach is helpful 
when working with ensemble models such as the one in ensemble_model.py to maximize 
final output precision and recall.

Usage example:
    selected_features = select_features(X_train, y_train, top_k=20, verbose=True)
"""
import numpy as np
import pandas as pd
from typing import Optional, List, Dict
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import os
import sys
from pathlib import Path

# Add project root to Python path
try:
    project_root = Path(__file__).parent.parent
    if not project_root.exists():
        # Handle network path by using raw string
        project_root = Path(r"\\".join(str(project_root).split("\\")))
    sys.path.append(str(project_root))
    print(f"Project root feature_selection_ensemble: {project_root}")
except Exception as e:
    print(f"Error setting project root path: {e}")
    sys.path.append(os.getcwd().parent)
    print(f"Current directory feature_selection_ensemble: {os.getcwd().parent}")

# Local imports
from utils.logger import ExperimentLogger
experiment_name = "feature_selection_ensemble"
logger = ExperimentLogger(experiment_name=experiment_name, log_dir='./logs/feature_selection_ensemble')

from utils.create_evaluation_set import create_ensemble_evaluation_set, import_selected_features_ensemble, import_training_data_ensemble, setup_mlflow_tracking
mlruns_dir = setup_mlflow_tracking(experiment_name)

def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: Optional[int] = 120,
    verbose: bool = True) -> List[str]:
    """
    Select features based on composite importance scores from XGBoost, CatBoost, and LightGBM.
    The function fits three models on the training data, extracts their feature importances,
    normalizes these scores, and then computes a composite score (the average of the three).
    Finally, it returns the names of the top features as specified by `top_k` or those whose 
    composite score exceeds the median output.
    Args:
        X (pd.DataFrame): The input features.
        y (pd.Series): The target variable.
        top_k (Optional[int], optional): Number of top features to select. If None, features
            with composite importance above the median are selected.
        verbose (bool, optional): If True, prints the selected features.
    Returns:
        List[str]: A list of selected feature names.
    """
    # Initialize models with minimal training iterations for quick evaluation
    xgb_model = XGBClassifier(
        random_state=42,
        tree_method='hist',
        n_estimators=1373,
        verbosity=0,
        learning_rate=0.032124519390481394,
        early_stopping_rounds=171,
        min_child_weight=246,
        gamma=0.01746785914240033,
        subsample=0.3873734494018223,
        colsample_bytree=0.9797308349068996,
        scale_pos_weight=2.278590445962556,
        reg_alpha=0.00018435065186835382,
        reg_lambda=4.962042446245311,
        max_depth=4,
        device='cpu',
        eval_metric=['error', 'auc', 'aucpr'],
        nthread=-1
    )
    cat_model = CatBoostClassifier(
        random_seed=42,
        iterations=4901,
        verbose=0,
        loss_function='Logloss',
        eval_metric='AUC',
        task_type='CPU',
        auto_class_weights='Balanced',
        grow_policy='SymmetricTree',
        learning_rate=0.013288620619359776,
        depth=7,
        l2_leaf_reg=11.152891877342054,
        border_count=128,
        subsample=0.7437647532474002,
        random_strength=0.14171788543304523,
        min_data_in_leaf=47,
        early_stopping_rounds=996
    )
    lgbm_model = LGBMClassifier(
        random_state=42,
        objective='binary',
        metric='binary_logloss',
        boosting_type='gbdt',
        device_type='cpu',
        verbose=100,
        learning_rate=0.0037460291406696956,
        max_depth=6,
        reg_lambda=0.4103475806096283,
        n_estimators=3577,
        num_leaves=67,
        early_stopping_rounds=291,
        feature_fraction=0.8037822667865142,
        bagging_freq=2,
        min_child_samples=28,
        bagging_fraction=0.7718378995036574,
        feature_fraction_bynode=0.9996212749980057
    )
    models = {
        "xgb": xgb_model,
        "cat": cat_model,
        "lgbm": lgbm_model
    }
    # DataFrame to store importance scores for each feature from each model.
    importance_df = pd.DataFrame(index=X.columns)

    for name, model in models.items():
        model.fit(X, y)
        if name == "xgb":
            # same order as X.columns.
            imp = model.feature_importances_
        elif name == "cat":
            # CatBoost provide importance as an np.array corresponding to features.
            imp = model.get_feature_importance()
        elif name == "lgbm":
            imp = model.feature_importances_
        else:
            imp = np.zeros(X.shape[1])
        # Normalize the scores so that they sum to 1.
        norm_imp = imp / np.sum(imp) if np.sum(imp) > 0 else imp
        importance_df[name] = norm_imp
    # Compute composite score as the average importance across models
    importance_df["composite"] = importance_df.mean(axis=1)
    # Sort the features by the composite score in descending order
    importance_df = importance_df.sort_values(by="composite", ascending=False)
    if top_k is not None:
        selected_features = importance_df.head(top_k).index.tolist()
    else:
        # Otherwise select features with composite importance above the median value.
        median_value = importance_df["composite"].median()
        selected_features = importance_df[importance_df["composite"] > median_value].index.tolist()
    if verbose:
        print("Selected Features:")
        print(selected_features)
        print("\nComposite Importance Scores:")
        print(importance_df["composite"])
    return selected_features

def select_features_differentiated(
    X: pd.DataFrame,
    y: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    top_k_per_model: int = 50,
    fixed_features: Optional[List[str]] = None,
    verbose: bool = True) -> Dict[str, List[str]]:
    """
    Select features separately for XGBoost, CatBoost, and LightGBM, then provide the union
    of the selected features with the fixed features always included.
    
    Args:
        X (pd.DataFrame): Input feature dataframe.
        y (pd.Series): Target variable.
        top_k_per_model (int): Number of top features to select for each model.
        fixed_features (Optional[List[str]]): Features that will be included in all sets.
        verbose (bool): If True, prints the selected feature lists.
    
    Returns:
        Dict[str, List[str]]: Dictionary with keys 'xgb', 'cat', 'lgbm', and 'union'.
    """
    
    fixed_features = fixed_features or []
    
    models = {
        "xgb": XGBClassifier(
            random_state=42,
            tree_method='hist',
            n_estimators=1373,
            verbosity=0,
            learning_rate=0.032124519390481394,
            early_stopping_rounds=171,
            min_child_weight=246,
            gamma=0.01746785914240033,
            subsample=0.3873734494018223,
            colsample_bytree=0.9797308349068996,
            scale_pos_weight=2.278590445962556,
            reg_alpha=0.00018435065186835382,
            reg_lambda=4.962042446245311,
            max_depth=4,
            device='cpu',
            eval_metric=['error', 'auc', 'aucpr'],
            nthread=-1
        ),
        "cat": CatBoostClassifier(
            random_seed=42,
            iterations=4901,
            verbose=0,
            loss_function='Logloss',
            eval_metric='AUC',
            task_type='CPU',
            auto_class_weights='Balanced',
            grow_policy='SymmetricTree',
            learning_rate=0.013288620619359776,
            depth=7,
            l2_leaf_reg=11.152891877342054,
            border_count=128,
            subsample=0.7437647532474002,
            random_strength=0.14171788543304523,
            min_data_in_leaf=47,
            early_stopping_rounds=996
        ),
        "lgbm": LGBMClassifier(
            random_state=42,
            objective='binary',
            metric='binary_logloss',
            boosting_type='gbdt',
            device_type='cpu',
            verbose=-1,
            learning_rate=0.0037460291406696956,
            max_depth=6,
            reg_lambda=0.4103475806096283,
            n_estimators=3577,
            num_leaves=67,
            early_stopping_rounds=291,
            feature_fraction=0.8037822667865142,
            bagging_freq=2,
            min_child_samples=28,
            bagging_fraction=0.7718378995036574,
            feature_fraction_bynode=0.9996212749980057  
        )
    }
    
    selected = {}
    # For each model, fit on the entire dataset and get sorted features by importance.
    for name, model in models.items():
        if name == "xgb":
            model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
            imp = np.array(model.feature_importances_)
        elif name == "cat":
            model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
            imp = model.get_feature_importance()
        elif name == "lgbm":
            model.fit(X, y, eval_set=[(X_val, y_val)])
            imp = np.array(model.feature_importances_)
        else:
            imp = np.zeros(X.shape[1])
        # Create a DataFrame mapping features to their importance
        imp_df = pd.DataFrame({
            "feature": X.columns,
            "importance": imp
        })
        imp_df = imp_df.sort_values(by="importance", ascending=False)
        
        # Select the top_k features
        top_features = imp_df["feature"].head(top_k_per_model).tolist()
        selected[name] = top_features
        
        if verbose:
            print(f"\nTop features for {name}:")
            print(top_features)
    # Union of all selected features and include fixed features
    union_features = set(fixed_features)
    for feat_list in selected.values():
        union_features.update(feat_list)
    union_features = list(union_features)
    
    if verbose:
        print("\nFixed features to always include:")
        print(fixed_features)
        print("\nFinal union of selected features:")
        print(union_features)
    # Return a dictionary with details for each model and the overall union.
    return {
        "xgb": selected["xgb"],
        "cat": selected["cat"],
        "lgbm": selected["lgbm"],
        "union": union_features
    }

def sync_columns(train_df, val_df, logger):
    """Ensure both DataFrames have exactly the same columns"""
    # Find common columns
    common_cols = list(set(train_df.columns) & set(val_df.columns))
    # Log differences
    train_only = set(train_df.columns) - set(common_cols)
    val_only = set(val_df.columns) - set(common_cols)
    
    if train_only:
        logger.warning(f"Dropping training-only columns: {list(train_only)}")
    if val_only:
        logger.warning(f"Dropping validation-only columns: {list(val_only)}")
    # Return synchronized DataFrames
    return train_df[common_cols], val_df[common_cols]

if __name__ == "__main__":
    logger = ExperimentLogger(
        experiment_name="feature_selection_ensemble",
        log_dir="logs/feature_selection_ensemble"
    )
    # Load data using utility functions
    features_train, target_train, features_test, target_test = import_training_data_ensemble()
    features_val, target_val = create_ensemble_evaluation_set()
    # Drop referee and league_name columns from all datasets
    columns_to_drop = ['referee', 'league_name']
    logger.info(f"Dropping columns as per requirements: {columns_to_drop}")
    features_train = features_train.drop(columns=columns_to_drop, errors='ignore')
    features_test = features_test.drop(columns=columns_to_drop, errors='ignore')
    features_val = features_val.drop(columns=columns_to_drop, errors='ignore')

    # Validate that all columns exist in both training and validation sets
    missing_train = [col for col in features_val.columns if col not in features_train.columns]
    missing_val = [col for col in features_train.columns if col not in features_val.columns]
    features_train, features_val = sync_columns(features_train, features_val, logger)
    features_test, features_val = sync_columns(features_test, features_val, logger)
    logger.info("Starting feature selection...")
    features_train, features_val = sync_columns(features_train, features_val, logger)
    selected_features = select_features_differentiated(features_train, target_train,features_val, target_val, verbose=True)
    logger.info(f"Selected features: {selected_features}")
