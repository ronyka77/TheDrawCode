"""
Meta-Feature Creation

Functions for creating meta-features for the stacked ensemble.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union

def create_meta_features(p_xgb: np.ndarray, p_cat: np.ndarray, 
                        p_lgb: np.ndarray, p_extra: np.ndarray, 
                        dynamic_weights: Optional[Dict] = None) -> np.ndarray:
    """
    Create meta-features for the meta-learner by combining base model predictions.
    
    Args:
        p_xgb: XGBoost predicted probabilities
        p_cat: CatBoost predicted probabilities
        p_lgb: LightGBM predicted probabilities
        p_extra: Extra model predicted probabilities
        dynamic_weights: Optional dictionary with weights for each model
        
    Returns:
        Meta-features for the meta-learner
    """
    # Ensure all prediction arrays are the same length
    if not all(len(p) == len(p_xgb) for p in [p_cat, p_lgb, p_extra]):
        raise ValueError("All prediction arrays must have the same length")
    
    # Create meta-features
    n_samples = len(p_xgb)
    
    # Basic probabilities
    meta_features = np.column_stack([
        p_xgb.reshape(-1, 1),
        p_cat.reshape(-1, 1),
        p_lgb.reshape(-1, 1),
        p_extra.reshape(-1, 1)
    ])
    
    # Add weighted average
    if dynamic_weights:
        weighted_avg = (
            dynamic_weights.get('xgb', 0.25) * p_xgb +
            dynamic_weights.get('cat', 0.25) * p_cat +
            dynamic_weights.get('lgb', 0.25) * p_lgb +
            dynamic_weights.get('extra', 0.25) * p_extra
        )
    else:
        # Simple average
        weighted_avg = (p_xgb + p_cat + p_lgb + p_extra) / 4.0
    
    # Add to meta-features
    meta_features = np.column_stack([meta_features, weighted_avg.reshape(-1, 1)])
    
    # Add pairwise differences
    diff_xgb_cat = np.abs(p_xgb - p_cat)
    diff_xgb_lgb = np.abs(p_xgb - p_lgb)
    diff_cat_lgb = np.abs(p_cat - p_lgb)
    diff_extra_xgb = np.abs(p_extra - p_xgb)
    diff_extra_cat = np.abs(p_extra - p_cat)
    diff_extra_lgb = np.abs(p_extra - p_lgb)
    
    meta_features = np.column_stack([
        meta_features,
        diff_xgb_cat.reshape(-1, 1),
        diff_xgb_lgb.reshape(-1, 1),
        diff_cat_lgb.reshape(-1, 1),
        diff_extra_xgb.reshape(-1, 1),
        diff_extra_cat.reshape(-1, 1),
        diff_extra_lgb.reshape(-1, 1)
    ])
    
    # Add max and min probabilities
    max_prob = np.maximum.reduce([p_xgb, p_cat, p_lgb, p_extra])
    min_prob = np.minimum.reduce([p_xgb, p_cat, p_lgb, p_extra])
    range_prob = max_prob - min_prob
    
    meta_features = np.column_stack([
        meta_features,
        max_prob.reshape(-1, 1),
        min_prob.reshape(-1, 1),
        range_prob.reshape(-1, 1)
    ])
    
    # Add rank features
    rank_features = []
    for i in range(n_samples):
        probs = [p_xgb[i], p_cat[i], p_lgb[i], p_extra[i]]
        ranks = np.argsort(np.argsort(probs))
        rank_features.append(ranks)
    
    rank_features = np.array(rank_features)
    meta_features = np.column_stack([meta_features, rank_features])
    
    # Add agreement features
    vote_threshold = 0.5
    votes = np.column_stack([
        (p_xgb > vote_threshold).astype(int),
        (p_cat > vote_threshold).astype(int),
        (p_lgb > vote_threshold).astype(int),
        (p_extra > vote_threshold).astype(int)
    ])
    
    vote_sum = np.sum(votes, axis=1)
    vote_agreement = np.where(
        (vote_sum == 0) | (vote_sum == 4),  # All agree
        1,
        0
    )
    
    meta_features = np.column_stack([
        meta_features,
        vote_sum.reshape(-1, 1),
        vote_agreement.reshape(-1, 1)
    ])
    
    # Final feature count should be 4 (base probs) + 1 (weighted avg) + 6 (diffs) + 3 (max/min/range) + 4 (ranks) + 2 (votes) = 20
    assert meta_features.shape[1] == 20, f"Expected 20 meta-features, got {meta_features.shape[1]}"
    
    return meta_features

def create_meta_dataframe(meta_features: np.ndarray) -> pd.DataFrame:
    """
    Convert meta-features array to a DataFrame with labeled columns.
    
    Args:
        meta_features: Meta-features array
        
    Returns:
        DataFrame with labeled columns
    """
    column_names = [
        'prob_xgb', 'prob_cat', 'prob_lgb', 'prob_extra',
        'weighted_avg',
        'diff_xgb_cat', 'diff_xgb_lgb', 'diff_cat_lgb',
        'diff_extra_xgb', 'diff_extra_cat', 'diff_extra_lgb',
        'max_prob', 'min_prob', 'range_prob',
        'rank_xgb', 'rank_cat', 'rank_lgb', 'rank_extra',
        'vote_sum', 'vote_agreement'
    ]
    
    return pd.DataFrame(meta_features, columns=column_names)
