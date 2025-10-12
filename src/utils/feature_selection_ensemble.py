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

import os
import pickle
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.utils.multiclass import type_of_target
from xgboost import XGBClassifier

from utils.create_evaluation_set import (
    create_ensemble_evaluation_set,
    import_selected_features_ensemble,
    import_training_data_ensemble,
)

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
    sys.path.append(str(Path(os.getcwd()).parent))
    print(f"Current directory feature_selection_ensemble: {Path(os.getcwd()).parent}")

# Local imports
from utils.logger import ExperimentLogger

experiment_name = "feature_selection_ensemble"
logger = ExperimentLogger(
    experiment_name=experiment_name, log_dir="./logs/feature_selection_ensemble"
)

# Set fixed seed and hash seed for determinism
SEED = 19
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Restrict parallel threads across various libraries
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
# PyTorch specific reproducibility settings
torch.manual_seed(SEED)

min_recall = 0.3


class PrecisionFocusedMetric(Metric):
    def __init__(self, beta=0.5):
        self._name = "precision_focused"
        self._maximize = True
        self.beta = beta

    def __call__(self, y_true, y_score):
        """F-beta score with beta < 1 to favor precision over recall"""

        # Ensure y_true is a 1D array
        # Check type of target
        y_true_type = type_of_target(y_true)
        if y_true_type == "multilabel-indicator":
            # Assuming binary classification represented as one-hot
            # Convert back to 1D: take the argmax along the class axis (axis=1)
            y_true_flat = np.argmax(y_true, axis=1)
        elif y_true_type == "binary":
            y_true_flat = y_true.astype(int)  # Ensure integer type
        else:
            # Handle unexpected types or raise an error
            logger.warning(
                f"Unexpected y_true type '{y_true_type}' in PrecisionFocusedMetric. Attempting to flatten."
            )
            try:
                y_true_flat = y_true.astype(int).ravel()  # General attempt to flatten
            except Exception as e:
                logger.error(f"Could not convert y_true to 1D array: {e}")
                return 0.0  # Return 0 score if conversion fails

        # Ensure y_score handling is robust
        # Check if y_score has 2 columns (expected for binary probabilities)
        if y_score.ndim == 2 and y_score.shape[1] == 2:
            pred = (y_score[:, 1] > 0.5).astype(int)  # Use probability of positive class
        elif y_score.ndim == 1:  # If y_score is already 1D predictions/scores
            pred = (y_score > 0.5).astype(int)  # Threshold directly
        else:
            logger.error(f"Unexpected y_score shape {y_score.shape} in PrecisionFocusedMetric.")
            return 0.0  # Return 0 score if y_score format is wrong

        # Calculate precision and recall safely
        try:
            # Check target types again just before sklearn call for debugging
            # logger.debug(f"y_true_flat type: {type_of_target(y_true_flat)}, pred type: {type_of_target(pred)}")
            precision = precision_score(y_true_flat, pred, zero_division="warn")
            recall = recall_score(y_true_flat, pred, zero_division="warn")
        except ValueError as e:
            logger.error(f"Error calculating scores in PrecisionFocusedMetric: {e}")
            logger.error(
                f"y_true_flat sample: {y_true_flat[:5]}, shape: {y_true_flat.shape}, "
                f"type: {type_of_target(y_true_flat)}"
            )
            logger.error(
                f"pred sample: {pred[:5]}, shape: {pred.shape}, type: {type_of_target(pred)}"
            )
            return 0.0  # Return 0 score if scikit-learn metric fails

        # If recall below threshold, return 0
        if recall < min_recall:
            return 0.0

        # F-beta with beta < 1 favors precision
        f_beta = (
            (1 + self.beta**2) * (precision * recall) / (self.beta**2 * precision + recall + 1e-8)
        )
        return f_beta


def select_features(
    x: pd.DataFrame, y: pd.Series, top_k: Optional[int] = 120, verbose: bool = True
) -> list[str]:
    """
    Select features based on composite importance scores from XGBoost, CatBoost, and LightGBM.
    The function fits three models on the training data, extracts their feature importances,
    normalizes these scores, and then computes a composite score (the average of the three).
    Finally, it returns the names of the top features as specified by `top_k` or those whose
    composite score exceeds the median output.
    Args:
        x (pd.DataFrame): The input features.
        y (pd.Series): The target variable.
        top_k (Optional[int], optional): Number of top features to select. If None, features
            with composite importance above the median are selected.
        verbose (bool, optional): If True, prints the selected features.
    Returns:
        List[str]: A list of selected feature names.
    """
    # Initialize models with minimal training iterations for quick evaluation
    xgb_model = XGBClassifier(
        tree_method="hist",
        device="cpu",
        nthread=4,
        objective="binary:logistic",
        eval_metric=["aucpr", "error", "logloss"],
        verbosity=0,
        learning_rate=0.055,
        max_depth=8,
        min_child_weight=390,
        subsample=0.61,
        colsample_bytree=0.78,
        reg_alpha=34.1,
        reg_lambda=6.69,
        gamma=3.86,
        early_stopping_rounds=1020,
        scale_pos_weight=3.3,
        seed=19,
    )
    cat_model = CatBoostClassifier(
        learning_rate=0.021289724174269435,
        depth=6,
        min_data_in_leaf=77,
        subsample=0.5728544513714008,
        colsample_bylevel=0.49610356838309444,
        reg_lambda=2.13261884199673,
        leaf_estimation_iterations=6,
        bagging_temperature=1.5545831677860675,
        scale_pos_weight=4.817574555909853,
        early_stopping_rounds=479,
        loss_function="Logloss",
        eval_metric="AUC",
        custom_metric=["Precision", "Recall"],
        task_type="CPU",
        thread_count=4,
        verbose=-1,
    )
    lgbm_model = LGBMClassifier(
        objective="binary",
        metric=["binary_logloss", "auc"],
        verbose=-1,
        n_jobs=4,
        random_state=19,
        device="cpu",
        learning_rate=0.14,
        num_leaves=95,
        max_depth=9,
        min_child_samples=230,
        feature_fraction=0.6000000000000001,
        bagging_fraction=0.5950000000000001,
        bagging_freq=10,
        reg_alpha=1.7000000000000002,
        reg_lambda=3.7,
        min_split_gain=0.16,
        early_stopping_rounds=660,
        path_smooth=0.405,
        cat_smooth=18.3,
        max_bin=250,
    )
    tabnet_model = TabNetClassifier(
        n_d=11,
        n_a=16,
        n_steps=9,
        gamma=1.8,
        lambda_sparse=2.4889331018199245e-05,
        momentum=0.9500000000000001,
        mask_type="entmax",
        device_name="cpu",
        optimizer_fn=torch.optim.Adam,
        optimizer_params={"lr": 0.021963953605914036},
        verbose=0,
        seed=19,
    )
    models = {"xgb": xgb_model, "cat": cat_model, "lgbm": lgbm_model, "tabnet": tabnet_model}
    # DataFrame to store importance scores for each feature from each model.
    importance_df = pd.DataFrame(index=x.columns)

    for name, model in models.items():
        model.fit(x, y)
        if name == "xgb":
            imp = model.feature_importances_
        elif name == "cat":
            imp = model.get_feature_importance()
        elif name == "lgbm":
            imp = model.feature_importances_
        elif name == "tabnet":
            imp = model.feature_importances_
        else:
            imp = np.zeros(x.shape[1])
        # Normalize the scores so that they sum to 1.
        norm_imp = imp / np.sum(imp) if np.sum(imp) > 0 else imp
        importance_df[name] = norm_imp
    # Compute composite score as the average importance across models
    importance_df["composite"] = importance_df.mean(axis=1)
    # Sort the features by the composite score in descending order
    importance_df = importance_df.sort_values(by="composite", ascending=False)
    if top_k is not None:
        selected_features = [str(col) for col in importance_df.head(top_k).index]
    else:
        # Otherwise select features with composite importance above the median value.
        median_value = importance_df["composite"].median()
        selected_features = [str(col) for col in importance_df[importance_df["composite"] > median_value].index]
    if verbose:
        print("Selected Features:")
        print(selected_features)
        print("\nComposite Importance Scores:")
        print(importance_df["composite"])
    return selected_features


def select_features_differentiated(
    x: pd.DataFrame,
    y: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    top_k_per_model: int = 65,
    fixed_features: Optional[list[str]] = None,
    verbose: bool = True,
) -> dict[str, list[str]]:
    """
    Select features separately for XGBoost, CatBoost, LightGBM and Random Forest, then provide the union
    of the selected features with the fixed features always included.
    Args:
        x (pd.DataFrame): Input feature dataframe.
        y (pd.Series): Target variable.
        top_k_per_model (int): Number of top features to select for each model.
        fixed_features (Optional[List[str]]): Features that will be included in all sets.
        verbose (bool): If True, prints the selected feature lists.
    Returns:
        Dict[str, List[str]]: Dictionary with keys 'xgb', 'cat', 'lgbm', 'rf' and 'union'.
    """

    fixed_features = fixed_features or []
    from src.models.StackedEnsemble.base.neural.mlp_model import create_model as create_model_mlp

    models = {
        "xgb": XGBClassifier(
            tree_method="hist",
            device="cuda",
            nthread=8,
            objective="binary:logistic",
            eval_metric=["aucpr", "error", "logloss"],
            verbosity=0,
            learning_rate=0.05,
            max_depth=8,
            min_child_weight=430,
            subsample=0.71,
            colsample_bytree=0.84,
            reg_alpha=25.200000000000003,
            reg_lambda=9.700000000000001,
            gamma=2.14,
            early_stopping_rounds=700,
            scale_pos_weight=2.36,
            seed=19,
        ),
        "lgbm": LGBMClassifier(
            objective="binary",
            metric=["aucpr", "binary_logloss"],
            verbose=-1,
            n_jobs=8,
            random_state=19,
            device="cpu",
            learning_rate=0.14,
            num_leaves=85,
            max_depth=6,
            min_child_samples=270,
            feature_fraction=0.6100000000000001,
            bagging_fraction=0.5750000000000001,
            bagging_freq=14,
            reg_alpha=16.200000000000003,
            reg_lambda=15.5,
            min_split_gain=0.14,
            early_stopping_rounds=670,
            path_smooth=0.34500000000000003,
            cat_smooth=23.400000000000002,
            max_bin=250,
        ),
        "mlp": create_model_mlp(
            {
                "input_dim": x.shape[1],
                "hidden_layers": 3,
                "neurons_per_layer": 62,
                "dropout_rate": 0.04,
                "activation": "tanh",
                "l1_regularization": 0.004681388569246714,
                "l2_regularization": 0.0001459323364245875,
                "learning_rate": 0.04014677776290513,
                "batch_size": 423,
                "epochs": 173,
                "patience": 23,
                "class_weight_multiplier": 2.2,
            }
        ),
    }

    selected = {}
    # For each model, fit on the entire dataset and get sorted features by importance.
    for name, model in models.items():
        if name == "xgb":
            model.fit(x, y, eval_set=[(x_val, y_val)], verbose=False)
            imp = np.array(model.feature_importances_)
        elif name == "lgbm":
            model.fit(x, y, eval_set=[(x_val, y_val)])
            imp = np.array(model.feature_importances_)
        elif name == "mlp":
            scaler = pickle.load(open("src/models/scalers/scaler_mlp.pkl", "rb"))
            x_scaled = scaler.fit_transform(x)
            scaler.transform(x_val)
            model.fit(x_scaled, y)
            # Get weight matrix of first Dense layer
            w = model.layers[0].get_weights()[0]
            # Sum abs(weights) across neurons → one score per input feature
            imp = np.abs(w).sum(axis=1)
        else:
            imp = np.zeros(x.shape[1])
        # Create a DataFrame mapping features to their importance
        imp_df = pd.DataFrame({"feature": x.columns, "importance": imp})
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
        "lgbm": selected["lgbm"],
        "mlp": selected["mlp"],
        "union": union_features,
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
        experiment_name="feature_selection_ensemble", log_dir="logs/feature_selection_ensemble"
    )
    # Load data using utility functions
    features_train, target_train, features_test, target_test = import_training_data_ensemble()
    features_val, target_val = create_ensemble_evaluation_set()
    features_all = import_selected_features_ensemble("all")
    # Drop referee and league_name columns from all datasets
    columns_to_drop = ["referee", "league_name"]
    logger.info(f"Dropping columns as per requirements: {columns_to_drop}")
    features_train = features_train[features_all]
    features_test = features_test[features_all]
    features_val = features_val[features_all]

    # Validate that all columns exist in both training and validation sets
    missing_train = [col for col in features_val.columns if col not in features_train.columns]
    missing_val = [col for col in features_train.columns if col not in features_val.columns]
    features_train, features_val = sync_columns(features_train, features_val, logger)
    features_test, features_val = sync_columns(features_test, features_val, logger)
    logger.info("Starting feature selection...")
    features_train, features_val = sync_columns(features_train, features_val, logger)
    # Merge training and test features while maintaining column consistency
    features_combined = pd.concat([features_train, features_test], axis=0)  # type: ignore
    # Ensure consistent column order and alignment
    features_combined = features_combined[features_train.columns]  # type: ignore
    target_combined = pd.concat([target_train, target_test], axis=0)  # type: ignore
    # Handle NaN values by filling with column means for numeric columns
    logger.info("Handling NaN values in features")
    numeric_cols = features_combined.select_dtypes(include=np.number).columns
    features_combined[numeric_cols] = features_combined[numeric_cols].fillna(  # type: ignore
        features_combined[numeric_cols].mean()  # type: ignore
    )
    features_val[numeric_cols] = features_val[numeric_cols].fillna(  # type: ignore
        features_combined[numeric_cols].mean()  # type: ignore
    )

    # For categorical columns, fill with mode
    categorical_cols = features_combined.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        mode_val = features_combined[col].mode()[0]  # type: ignore
        features_combined[col] = features_combined[col].fillna(mode_val)  # type: ignore
        features_val[col] = features_val[col].fillna(mode_val)  # type: ignore

    # Verify no NaN values remain
    if features_combined.isna().any().any() or features_val.isna().any().any():  # type: ignore
        logger.error("NaN values still present after imputation")
        raise ValueError("Failed to handle all NaN values")
    # Log the merge operation
    logger.info(f"Merged training and test features. Combined shape: {features_combined.shape}")
    selected_features = select_features_differentiated(
        features_combined, target_combined, features_val, target_val, verbose=True  # type: ignore
    )
    logger.info(f"Selected features: {selected_features}")
