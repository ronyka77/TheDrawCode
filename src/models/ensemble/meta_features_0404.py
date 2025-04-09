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
    dynamic_weights: Optional[dict] = None,
    thresholds: Optional[dict] = None,
) -> np.ndarray:
    """
    Create meta-features for the meta-learner by combining base model predictions.

    Args:
        p_xgb: XGBoost predicted probabilities
        p_tabnet: TabNet predicted probabilities
        p_lgb: LightGBM predicted probabilities
        p_extra: Extra model predicted probabilities
        p_mlp: MLP predicted probabilities
        dynamic_weights: Optional dictionary with weights for each model
        thresholds: Optional dictionary with thresholds for each model
    Returns:
        Meta-features for the meta-learner
    """
    # Ensure all prediction arrays are the same length
    if not all(len(p) == len(p_xgb) for p in [p_tabnet, p_lgb, p_extra, p_mlp]):
        raise ValueError("All prediction arrays must have the same length")

    # Create meta-features
    n_samples = len(p_xgb)

    # Basic probabilities
    meta_features = np.column_stack(
        [
            p_xgb.reshape(-1, 1),
            p_tabnet.reshape(-1, 1),
            p_lgb.reshape(-1, 1),
            p_extra.reshape(-1, 1),
            p_mlp.reshape(-1, 1),
        ]
    )

    # Add weighted average
    if dynamic_weights:
        weighted_avg = (
            dynamic_weights.get("xgb", 0.2) * p_xgb
            + dynamic_weights.get("tabnet", 0.2) * p_tabnet
            + dynamic_weights.get("lgb", 0.2) * p_lgb
            + dynamic_weights.get("extra", 0.2) * p_extra
            + dynamic_weights.get("mlp", 0.2) * p_mlp
        )
    else:
        # Simple average
        weighted_avg = (p_xgb + p_tabnet + p_lgb + p_extra + p_mlp) / 5.0

    # Add to meta-features
    meta_features = np.column_stack([meta_features, weighted_avg.reshape(-1, 1)])

    # Add pairwise differences
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

    meta_features = np.column_stack(
        [
            meta_features,
            diff_xgb_tabnet.reshape(-1, 1),
            diff_xgb_lgb.reshape(-1, 1),
            diff_xgb_extra.reshape(-1, 1),
            diff_xgb_mlp.reshape(-1, 1),
            diff_tabnet_lgb.reshape(-1, 1),
            diff_tabnet_extra.reshape(-1, 1),
            diff_tabnet_mlp.reshape(-1, 1),
            diff_lgb_extra.reshape(-1, 1),
            diff_lgb_mlp.reshape(-1, 1),
            diff_extra_mlp.reshape(-1, 1),
        ]
    )

    # Add max and min probabilities
    max_prob = np.maximum.reduce([p_xgb, p_tabnet, p_lgb, p_extra, p_mlp])
    min_prob = np.minimum.reduce([p_xgb, p_tabnet, p_lgb, p_extra, p_mlp])
    range_prob = max_prob - min_prob

    meta_features = np.column_stack(
        [meta_features, max_prob.reshape(-1, 1), min_prob.reshape(-1, 1), range_prob.reshape(-1, 1)]
    )

    # Add rank features
    rank_features = []
    for i in range(n_samples):
        probs = [p_xgb[i], p_tabnet[i], p_lgb[i], p_extra[i], p_mlp[i]]
        ranks = np.argsort(np.argsort(probs))
        rank_features.append(ranks)

    rank_features = np.array(rank_features)
    meta_features = np.column_stack([meta_features, rank_features])

    # Add agreement features
    vote_threshold_xgb = thresholds["xgb"] if thresholds else 0.5
    vote_threshold_tabnet = thresholds["tabnet"] if thresholds else 0.5
    vote_threshold_lgb = thresholds["lgb"] if thresholds else 0.5
    vote_threshold_extra = thresholds["extra"] if thresholds else 0.5
    vote_threshold_mlp = thresholds["mlp"] if thresholds else 0.5
    votes = np.column_stack(
        [
            (p_xgb > vote_threshold_xgb).astype(int),
            (p_tabnet > vote_threshold_tabnet).astype(int),
            (p_lgb > vote_threshold_lgb).astype(int),
            (p_extra > vote_threshold_extra).astype(int),
            (p_mlp > vote_threshold_mlp).astype(int),
        ]
    )

    vote_sum = np.sum(votes, axis=1)
    vote_agreement = np.where(
        (vote_sum == 0) | (vote_sum == 5),  # All agree
        1,
        0,
    )

    meta_features = np.column_stack(
        [meta_features, vote_sum.reshape(-1, 1), vote_agreement.reshape(-1, 1)]
    )

    # Final feature count should be 5 (base probs) + 1 (weighted avg) + 10 (diffs) + 3 (max/min/range) + 5 (ranks) + 2 (votes) = 26
    # assert meta_features.shape[1] == 26, f"Expected 26 meta-features, got {meta_features.shape[1]}"

    return meta_features


def create_meta_dataframe(meta_features: np.ndarray) -> pd.DataFrame:
    """
    Convert meta-features array to a DataFrame with labeled columns.

    Args:
        meta_features: Meta-features array

    Returns:
        DataFrame with labeled columns
    """

    return pd.DataFrame(meta_features)
