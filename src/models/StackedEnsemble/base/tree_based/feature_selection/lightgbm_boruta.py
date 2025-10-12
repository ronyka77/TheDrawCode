import os
import random

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from BorutaShap import BorutaShap
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Import own modules
from src.utils.logger import ExperimentLogger

# Define experiment name
experiment_name = "lightgbm_soccer_prediction_25"
logger = ExperimentLogger(experiment_name)

# Import data at runtime to avoid global scope issues
from src.models.ensemble.data_utils import prepare_data
from src.models.StackedEnsemble.shared.data_loader_new import DataLoader
from src.utils.create_evaluation_set import import_selected_features_ensemble_new

# Set fixed seed and hash seed for determinism
SEED = 19
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Restrict parallel threads across various libraries
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"


def run_boruta_feature_selection(X_train, y_train, X_test, y_test, X_eval, y_eval, features):
    """
    Run Boruta feature selection using LightGBM and SHAP importance.

    Returns:
        list: Selected features from Boruta algorithm
    """
    try:
        logger.info("Starting LightGBM Boruta feature selection")

        logger.info(f"Features: {len(features)}")
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

        # Define LightGBM model
        params = {
            "objective": "binary",
            "metric": ["auc", "binary_logloss"],
            "learning_rate": 0.153,
            "num_leaves": 31,
            "max_depth": 5,
            "min_child_samples": 430,
            "feature_fraction": 0.88,
            "bagging_fraction": 0.905,
            "min_split_gain": 4.82,
            "lambda_l2": 11.72,
            "lambda_l1": 36.2,
            "scale_pos_weight": 2.36,
            "verbose": -1,
        }
        lgb_clf = lgb.LGBMClassifier(**params)

        # Run BorutaShap
        feature_selector = BorutaShap(
            model=lgb_clf,
            importance_measure="shap",  # or 'gini'
            classification=True,
            # pvalue=0.10
        )
        feature_selector.fit(
            X=X_train,
            y=y_train,
            n_trials=500,  # Number of Boruta iterations
            sample=False,  # Set to True for large datasets
            train_or_test="train",  # Use test set for SHAP values
            verbose=True,
        )

        # Get selected features
        selected_features = feature_selector.Subset().columns.tolist()
        logger.info(f"Selected features: {selected_features}")
        logger.info(f"Number of selected features: {len(selected_features)}")

        # Save results
        feature_selector.results_to_csv(filename="feature_importance")

        # Optionally, transform your data
        X_train_selected = feature_selector.transform(X_train)
        logger.info(f"Transformed training data shape: {X_train_selected.shape}")

        logger.info("Boruta feature selection completed successfully")
        return selected_features

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(f"Error type: {type(e)}")
        logger.error("Failed to run LightGBM Boruta feature selection")
        logger.error("Please check the data and model parameters")
        logger.error("Exiting the program")
        return []


def lightgbm_staged_selection(X, y, X_eval, y_eval, target_features=80):
    """Multi-stage LightGBM feature selection with different objectives"""

    logger.info(f"Starting LightGBM staged selection with {X.shape[1]} initial features")
    eval_metrics = ["auc", "binary_logloss"]
    # Stage 1: Quick filter with high learning rate
    logger.info("Stage 1: Quick filter with high learning rate")
    lgb_fast = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )

    lgb_fast.fit(X, y, eval_set=[(X_eval, y_eval)], eval_metric=eval_metrics)
    stage1_importance = lgb_fast.feature_importances_
    stage1_features = X.columns[np.argsort(stage1_importance)[-200:]].tolist()

    logger.info(f"Stage 1: Selected {len(stage1_features)} features")

    # Stage 2: Refined selection with cross-validation
    logger.info("Stage 2: Refined selection with cross-validation")
    X_stage1 = X[stage1_features]
    X_eval_stage1 = X_eval[stage1_features]

    lgb_refined = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        bagging_fraction=0.8,
        bagging_freq=5,
        feature_fraction=0.8,
        num_leaves=31,
        min_child_samples=20,
        min_split_gain=0.0,
        max_bin=255,
        cat_smooth=10.0,
        path_smooth=0.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        objective="binary",
        metric=["aucpr", "binary_logloss"],
        device="cpu",
        n_jobs=8,
        random_state=19,
        verbose=-1,
    )

    # Cross-validation feature importance
    cv_scores = []
    cv_importances = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(X_stage1, y):
        X_train_cv, X_val_cv = X_stage1.iloc[train_idx], X_stage1.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

        lgb_refined.fit(
            X_train_cv,
            y_train_cv,
            eval_set=[(X_val_cv, y_val_cv)],
            eval_metric=eval_metrics,
            callbacks=[lgb.early_stopping(stopping_rounds=200)],
        )
        cv_importances.append(lgb_refined.feature_importances_)

        val_score = lgb_refined.score(X_eval_stage1, y_eval)
        cv_scores.append(val_score)

    # Average importance across folds
    avg_importance = np.mean(cv_importances, axis=0)
    stage2_features = [stage1_features[i] for i in np.argsort(avg_importance)[-target_features:]]

    # Log average importances for the selected features
    selected_indices = np.argsort(avg_importance)[-target_features:]
    selected_importances = avg_importance[selected_indices]
    feature_importance_pairs = list(zip(stage2_features, selected_importances))

    logger.info(f"Stage 2: Selected {len(stage2_features)} features: {stage2_features}")
    logger.info(f"Feature-importance pairs: {feature_importance_pairs}")
    logger.info(f"CV Score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    return stage2_features, avg_importance


def xgboost_staged_selection(X, y, X_eval, y_eval, target_features=80):
    """Multi-stage XGBoost feature selection with different objectives"""

    logger.info(f"Starting XGBoost staged selection with {X.shape[1]} initial features")
    eval_metrics = ["aucpr", "error", "logloss"]
    # Stage 1: Quick filter with high learning rate
    logger.info("Stage 1: Quick filter with high learning rate")
    xgb_fast = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric=eval_metrics,
    )

    xgb_fast.fit(X, y, eval_set=[(X_eval, y_eval)], verbose=False)
    stage1_importance = xgb_fast.feature_importances_
    stage1_features = X.columns[np.argsort(stage1_importance)[-200:]].tolist()

    logger.info(f"Stage 1: Selected {len(stage1_features)} features")

    # Stage 2: Refined selection with cross-validation
    logger.info("Stage 2: Refined selection with cross-validation")
    X_stage1 = X[stage1_features]
    X_eval_stage1 = X_eval[stage1_features]

    xgb_refined = xgb.XGBClassifier(
        n_estimators=500,
        early_stopping_rounds=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        gamma=0.0,
        min_child_weight=1,
        scale_pos_weight=1.0,
        random_state=42,
        eval_metric=eval_metrics,
    )

    # Cross-validation feature importance
    cv_scores = []
    cv_importances = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(X_stage1, y):
        X_train_cv, X_val_cv = X_stage1.iloc[train_idx], X_stage1.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

        xgb_refined.fit(X_train_cv, y_train_cv, eval_set=[(X_val_cv, y_val_cv)], verbose=False)
        cv_importances.append(xgb_refined.feature_importances_)

        val_score = xgb_refined.score(X_eval_stage1, y_eval)
        cv_scores.append(val_score)

    # Average importance across folds
    avg_importance = np.mean(cv_importances, axis=0)
    stage2_features = [stage1_features[i] for i in np.argsort(avg_importance)[-target_features:]]

    # Log average importances for the selected features
    selected_indices = np.argsort(avg_importance)[-target_features:]
    selected_importances = avg_importance[selected_indices]
    feature_importance_pairs = list(zip(stage2_features, selected_importances))

    logger.info(f"Stage 2: Selected {len(stage2_features)} features: {stage2_features}")
    logger.info(f"Feature-importance pairs: {feature_importance_pairs}")
    logger.info(f"CV Score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    return stage2_features, avg_importance


def ensemble_gbm_selection(X, y, X_test, y_test, X_eval, y_eval, target_features=80):
    """
    Combine XGBoost and LightGBM for robust feature selection.
    """
    logger.info("Starting ensemble GBM feature selection")

    # Get features from both models
    xgb_features, _ = xgboost_staged_selection(
        X, y, X_test, y_test, X_eval, y_eval, target_features + 20
    )
    lgb_features, _ = lightgbm_staged_selection(
        X, y, X_test, y_test, X_eval, y_eval, target_features + 20
    )

    # Feature voting system
    feature_votes = {}

    # XGBoost votes (weighted by rank)
    for i, feature in enumerate(xgb_features):
        weight = (len(xgb_features) - i) / len(xgb_features)
        feature_votes[feature] = feature_votes.get(feature, 0) + weight

    # LightGBM votes (weighted by rank)
    for i, feature in enumerate(lgb_features):
        weight = (len(lgb_features) - i) / len(lgb_features)
        feature_votes[feature] = feature_votes.get(feature, 0) + weight

    # Select top voted features
    sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
    final_features = [feature for feature, _ in sorted_features[:target_features]]

    # Validation with both models
    X_selected = X[final_features]

    # XGBoost validation
    xgb_val = xgb.XGBClassifier(n_estimators=200, random_state=42, eval_metric="logloss")
    try:
        xgb_scores = cross_val_score(xgb_val, X_selected, y, cv=5, scoring="roc_auc")
    except AttributeError as e:
        if "__sklearn_tags__" in str(e):
            logger.warning(
                "XGBoost compatibility issue with scikit-learn. Using manual cross-validation."
            )

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            xgb_scores_list = []

            for train_idx, val_idx in skf.split(X_selected, y):
                X_train_cv, X_val_cv = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
                y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

                xgb_val.fit(X_train_cv, y_train_cv)
                y_pred_proba = xgb_val.predict_proba(X_val_cv)[:, 1]
                score = roc_auc_score(y_val_cv, y_pred_proba)
                xgb_scores_list.append(score)

            xgb_scores = np.array(xgb_scores_list)
        else:
            raise e

    # LightGBM validation
    lgb_val = lgb.LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
    try:
        lgb_scores = cross_val_score(lgb_val, X_selected, y, cv=5, scoring="roc_auc")
    except AttributeError as e:
        if "__sklearn_tags__" in str(e):
            logger.warning(
                "LightGBM compatibility issue with scikit-learn. Using manual cross-validation."
            )
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            lgb_scores_list = []

            for train_idx, val_idx in skf.split(X_selected, y):
                X_train_cv, X_val_cv = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
                y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

                lgb_val.fit(X_train_cv, y_train_cv)
                y_pred_proba = lgb_val.predict_proba(X_val_cv)[:, 1]
                score = roc_auc_score(y_val_cv, y_pred_proba)
                lgb_scores_list.append(score)

            lgb_scores = np.array(lgb_scores_list)
        else:
            raise e

    logger.info(f"Selected {len(final_features)} features: {final_features}")
    logger.info(f"XGBoost CV AUC: {xgb_scores.mean():.4f} ± {xgb_scores.std():.4f}")
    logger.info(f"LightGBM CV AUC: {lgb_scores.mean():.4f} ± {lgb_scores.std():.4f}")

    return final_features, {
        "xgb_scores": xgb_scores,
        "lgb_scores": lgb_scores,
        "feature_votes": feature_votes,
    }


def main():
    """
    Main execution function for Boruta feature selection.
    """
    # Load data
    dataloader = DataLoader()
    X_train, y_train, X_test, y_test, X_eval, y_eval = dataloader.load_data()
    features = import_selected_features_ensemble_new(model_type="all")
    X_train = prepare_data(X_train, features)
    X_test = prepare_data(X_test, features)
    X_eval = prepare_data(X_eval, features)

    # logger.info("Starting Boruta feature selection pipeline")
    # selected_features = run_boruta_feature_selection(X_train, y_train, X_test, y_test, X_eval, y_eval, features)
    # logger.info(f"Feature selection completed. Selected {len(selected_features)} features.")
    # logger.info("Selected features saved to feature_importance.csv")
    # baseline_xgb_mean, baseline_lgb_mean = run_baseline_comparison(X_train, y_train, X_test, y_test, X_eval, y_eval)
    # Combine training and test sets for feature selection
    X_combined = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    y_combined = pd.concat([y_train, y_test], axis=0, ignore_index=True)
    logger.info(f"Combined dataset shape: {X_combined.shape}")
    stage1_features, avg_importance = xgboost_staged_selection(
        X_combined, y_combined, X_eval, y_eval, target_features=150
    )
    stage2_features, avg_importance = lightgbm_staged_selection(
        X_combined, y_combined, X_eval, y_eval, target_features=150
    )


if __name__ == "__main__":
    main()
