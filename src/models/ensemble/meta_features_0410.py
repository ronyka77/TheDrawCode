"""
Meta-Feature Creation - Version 0404

Functions for creating meta-features for the stacked ensemble including MLP.
"""

from typing import Optional

import numpy as np
import pandas as pd


def create_meta_features_optimized(
    p_xgb: np.ndarray,
    p_tabnet: np.ndarray,
    p_lgb: np.ndarray,
    p_extra: np.ndarray,
    p_mlp: np.ndarray,
    p_pytorch: np.ndarray,
    dynamic_weights: Optional[dict] = None,
    thresholds: Optional[dict] = None,
) -> np.ndarray:
    """
    Create meta-features for the meta-learner by combining base model predictions.
    Now handles 6 base models (incl. PyTorch).

    Args:
        p_xgb: XGBoost predicted probabilities
        p_tabnet: TabNet predicted probabilities
        p_lgb: LightGBM predicted probabilities
        p_extra: Extra model predicted probabilities
        p_mlp: MLP predicted probabilities
        p_pytorch: PyTorch predicted probabilities
        dynamic_weights: Optional dictionary with weights for each model
        thresholds: Optional dictionary with thresholds for each model
    Returns:
        Meta-features for the meta-learner
    """
    # Ensure all prediction arrays are the same length
    all_preds = [p_xgb, p_tabnet, p_lgb, p_extra, p_mlp, p_pytorch]
    if not all(len(p) == len(p_xgb) for p in all_preds):
        raise ValueError("All prediction arrays must have the same length")

    # Create meta-features
    num_models = len(all_preds)
    default_weight = 1.0 / num_models

    # Basic probabilities
    meta_features = np.column_stack([p.reshape(-1, 1) for p in all_preds]) # Stack all 6

    # Add weighted average
    if dynamic_weights:
        weighted_avg = (
            dynamic_weights.get("xgb", default_weight) * p_xgb
            + dynamic_weights.get("tabnet", default_weight) * p_tabnet
            + dynamic_weights.get("lgb", default_weight) * p_lgb
            + dynamic_weights.get("extra", default_weight) * p_extra
            + dynamic_weights.get("mlp", default_weight) * p_mlp
            + dynamic_weights.get("pytorch", default_weight) * p_pytorch # Added PyTorch
        )
    else:
        # Simple average
        weighted_avg = np.mean(all_preds, axis=0) # Simpler way to average

    # Add to meta-features
    meta_features = np.column_stack([meta_features, weighted_avg.reshape(-1, 1)])

    # Add pairwise differences
    # Calculate differences for the new model against all others
    diff_xgb_pytorch = np.abs(p_xgb - p_pytorch)
    diff_tabnet_pytorch = np.abs(p_tabnet - p_pytorch)
    diff_lgb_pytorch = np.abs(p_lgb - p_pytorch)
    diff_extra_pytorch = np.abs(p_extra - p_pytorch)
    diff_mlp_pytorch = np.abs(p_mlp - p_pytorch)
    
    # Existing differences
    diff_xgb_tabnet = np.abs(p_xgb - p_tabnet)
    diff_xgb_lgb = np.abs(p_xgb - p_lgb)
    diff_xgb_extra = np.abs(p_xgb - p_extra)
    diff_xgb_mlp = np.abs(p_xgb - p_mlp)
    diff_tabnet_lgb = np.abs(p_tabnet - p_lgb)
    diff_tabnet_extra = np.abs(p_tabnet - p_extra)
    diff_tabnet_mlp = np.abs(p_tabnet - p_mlp)
    diff_lgb_extra = np.abs(p_lgb - p_extra)
    diff_lgb_mlp = np.abs(p_lgb - p_mlp)
    diff_extra_mlp = np.abs(p_extra - p_mlp)

    # Stack all differences (original 10 + new 5 = 15)
    all_diffs = [
        diff_xgb_tabnet, diff_xgb_lgb, diff_xgb_extra, diff_xgb_mlp, diff_xgb_pytorch, # xgb vs others
        diff_tabnet_lgb, diff_tabnet_extra, diff_tabnet_mlp, diff_tabnet_pytorch,       # tabnet vs others
        diff_lgb_extra, diff_lgb_mlp, diff_lgb_pytorch,                               # lgb vs others
        diff_extra_mlp, diff_extra_pytorch,                                           # extra vs others
        diff_mlp_pytorch                                                              # mlp vs pytorch
    ]
    meta_features = np.column_stack([meta_features] + [d.reshape(-1, 1) for d in all_diffs])

    # Add max and min probabilities
    max_prob = np.maximum.reduce(all_preds) # Use all_preds list
    min_prob = np.minimum.reduce(all_preds) # Use all_preds list
    range_prob = max_prob - min_prob

    meta_features = np.column_stack(
        [meta_features, max_prob.reshape(-1, 1), min_prob.reshape(-1, 1), range_prob.reshape(-1, 1)]
    )

    # Add rank features
    # Use all_preds list for simpler ranking
    probs_stacked = np.stack(all_preds, axis=1) # Shape (n_samples, num_models)
    ranks = np.argsort(np.argsort(probs_stacked, axis=1), axis=1)
    meta_features = np.column_stack([meta_features, ranks]) # Adds num_models columns

    # Add agreement features
    default_threshold = 0.5
    vote_thresholds = [
        thresholds.get("xgb", default_threshold) if thresholds else default_threshold,
        thresholds.get("tabnet", default_threshold) if thresholds else default_threshold,
        thresholds.get("lgb", default_threshold) if thresholds else default_threshold,
        thresholds.get("extra", default_threshold) if thresholds else default_threshold,
        thresholds.get("mlp", default_threshold) if thresholds else default_threshold,
        thresholds.get("pytorch", default_threshold) if thresholds else default_threshold, # Added PyTorch
    ]
    
    votes = np.column_stack([
        (pred > thresh).astype(int) for pred, thresh in zip(all_preds, vote_thresholds)
    ])

    vote_sum = np.sum(votes, axis=1)
    vote_agreement = np.where(
        (vote_sum == 0) | (vote_sum == num_models), # Check for 0 or num_models (6) votes
        1,
        0,
    )

    meta_features = np.column_stack(
        [meta_features, vote_sum.reshape(-1, 1), vote_agreement.reshape(-1, 1)]
    )

    return meta_features


def create_meta_dataframe(meta_features: np.ndarray) -> pd.DataFrame:
    """
    Convert meta-features array to a DataFrame with labeled columns.
    Updated for 6 base models.

    Args:
        meta_features: Meta-features array (expected shape: N x 33)

    Returns:
        DataFrame with labeled columns
    """
    num_features = meta_features.shape[1]
    # Define column names based on the expected structure (33 features)
    col_names = (
        ["p_xgb", "p_tabnet", "p_lgb", "p_extra", "p_mlp", "p_pytorch"] + # 6 base
        ["weighted_avg"] + # 1 avg
        [
            "diff_xgb_tabnet", "diff_xgb_lgb", "diff_xgb_extra", "diff_xgb_mlp", "diff_xgb_pytorch",
            "diff_tabnet_lgb", "diff_tabnet_extra", "diff_tabnet_mlp", "diff_tabnet_pytorch",
            "diff_lgb_extra", "diff_lgb_mlp", "diff_lgb_pytorch",
            "diff_extra_mlp", "diff_extra_pytorch",
            "diff_mlp_pytorch"
        ] + # 15 diffs
        ["max_prob", "min_prob", "range_prob"] + # 3 range
        ["rank_xgb", "rank_tabnet", "rank_lgb", "rank_extra", "rank_mlp", "rank_pytorch"] + # 6 ranks
        ["vote_sum", "vote_agreement"] # 2 votes
    )
    
    if num_features != len(col_names):
        # Fallback if the number of features doesn't match the expected 33
        # This might happen if the assert in the creation function is commented out and logic changes
        col_names = [f"meta_{i}" for i in range(num_features)]
        print(f"Number of features does not match expected 33: {num_features}")

    return pd.DataFrame(meta_features, columns=col_names)
