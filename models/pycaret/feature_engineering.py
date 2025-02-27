"""
Feature engineering module for PyCaret soccer prediction.

This module contains functions for creating and transforming features
to improve model precision.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to Python path if needed
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from utils.logger import ExperimentLogger

# Setup logger
logger = ExperimentLogger(experiment_name="pycaret_feature_engineering")

def engineer_precision_features(df, top_features=None):
    """
    Add engineered features to improve model precision.
    
    Args:
        df (pd.DataFrame): DataFrame with features
        top_features (list, optional): List of top feature names to use for interactions
            If None, will use feature importance from existing models or select based on correlation
            
    Returns:
        pd.DataFrame: DataFrame with added engineered features
    """
    logger.info("Engineering precision-focused features")
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # If top features not provided, try to determine them
    if top_features is None:
        # Method 1: Use correlation with target
        if 'target' in df.columns:
            numeric_cols = df.select_dtypes(include=['number']).columns
            numeric_cols = [col for col in numeric_cols if col != 'target']
            
            # Calculate correlation with target
            correlations = []
            for col in numeric_cols:
                corr = abs(df[col].corr(df['target']))
                if not np.isnan(corr):
                    correlations.append((col, corr))
            
            # Sort by absolute correlation
            correlations.sort(key=lambda x: x[1], reverse=True)
            
            # Take top 5 features
            top_features = [col for col, _ in correlations[:5]]
            logger.info(f"Selected top features based on correlation: {top_features}")
        else:
            # Fallback to first 5 numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            top_features = list(numeric_cols[:5])
            logger.info(f"Selected first 5 numeric features: {top_features}")
    
    # 1. Create interaction features between strongest predictors
    logger.info("Creating interaction features")
    interaction_count = 0
    for i in range(len(top_features)):
        for j in range(i+1, len(top_features)):
            feat_name = f"interaction_{top_features[i]}_{top_features[j]}"
            df[feat_name] = df[top_features[i]] * df[top_features[j]]
            interaction_count += 1
    
    logger.info(f"Created {interaction_count} interaction features")
    
    # 2. Add polynomial features for key numeric predictors
    logger.info("Creating polynomial features")
    poly_count = 0
    for feat in top_features:
        if feat in df.columns and df[feat].dtype in [np.float64, np.int64]:
            df[f"{feat}_squared"] = df[feat]**2
            poly_count += 1
    
    logger.info(f"Created {poly_count} polynomial features")
    
    # 3. Create "confidence" features - distance from decision boundary proxies
    confidence_count = 0
    if "home_win_prob" in df.columns and "away_win_prob" in df.columns:
        df["draw_confidence"] = 1 - abs(df["home_win_prob"] - df["away_win_prob"])
        confidence_count += 1
        logger.info("Created draw confidence feature")
    
    # 4. Ratio features often help with precision
    logger.info("Creating ratio features")
    ratio_count = 0
    numeric_cols = df.select_dtypes(include=['number']).columns
    numeric_cols = [col for col in numeric_cols if col != 'target']
    
    for i in range(min(5, len(numeric_cols))):
        for j in range(i+1, min(5, len(numeric_cols))):
            if df[numeric_cols[j]].min() != 0:  # Avoid division by zero
                ratio_name = f"ratio_{numeric_cols[i]}_{numeric_cols[j]}"
                df[ratio_name] = df[numeric_cols[i]] / df[numeric_cols[j]]
                ratio_count += 1
    
    logger.info(f"Created {ratio_count} ratio features")
    
    # 5. Create difference features
    logger.info("Creating difference features")
    diff_count = 0
    for i in range(min(5, len(numeric_cols))):
        for j in range(i+1, min(5, len(numeric_cols))):
            diff_name = f"diff_{numeric_cols[i]}_{numeric_cols[j]}"
            df[diff_name] = df[numeric_cols[i]] - df[numeric_cols[j]]
            diff_count += 1
    
    logger.info(f"Created {diff_count} difference features")
    
    # 6. Create binned features for top predictors
    logger.info("Creating binned features")
    bin_count = 0
    for feat in top_features:
        if feat in df.columns and df[feat].dtype in [np.float64, np.int64]:
            bin_name = f"{feat}_binned"
            df[bin_name] = pd.qcut(df[feat], 5, labels=False, duplicates='drop')
            bin_count += 1
    
    logger.info(f"Created {bin_count} binned features")
    
    # Log total new features
    total_new = interaction_count + poly_count + confidence_count + ratio_count + diff_count + bin_count
    logger.info(f"Added {total_new} new engineered features")
    
    return df

def create_domain_specific_features(df):
    """
    Create soccer-specific features that might help with draw prediction.
    
    Args:
        df (pd.DataFrame): DataFrame with features
        
    Returns:
        pd.DataFrame: DataFrame with added domain-specific features
    """
    logger.info("Creating soccer-specific features for draw prediction")
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Look for common soccer-related columns
    soccer_columns = {
        'goals': ['goals', 'home_goals', 'away_goals', 'avg_goals', 'total_goals'],
        'form': ['form', 'home_form', 'away_form', 'recent_form'],
        'odds': ['odds', 'draw_odds', 'home_odds', 'away_odds'],
        'points': ['points', 'home_points', 'away_points', 'points_diff'],
        'ranking': ['rank', 'ranking', 'home_rank', 'away_rank', 'rank_diff']
    }
    
    feature_count = 0
    
    # 1. Team strength similarity (teams of similar strength are more likely to draw)
    strength_cols = []
    for col_type in ['ranking', 'points', 'form']:
        home_col = None
        away_col = None
        
        # Find home and away columns for this type
        for col in df.columns:
            for pattern in soccer_columns[col_type]:
                if col.lower().startswith('home_') and pattern.lower() in col.lower():
                    home_col = col
                elif col.lower().startswith('away_') and pattern.lower() in col.lower():
                    away_col = col
        
        # If we found matching columns, create a similarity feature
        if home_col and away_col and home_col in df.columns and away_col in df.columns:
            similarity_name = f"similarity_{col_type}"
            df[similarity_name] = 1 - abs(df[home_col] - df[away_col]) / (df[[home_col, away_col]].max(axis=1) + 1e-10)
            strength_cols.append(similarity_name)
            feature_count += 1
    
    # 2. Create a combined team similarity index if we have multiple similarity measures
    if len(strength_cols) > 1:
        df['team_similarity_index'] = df[strength_cols].mean(axis=1)
        feature_count += 1
    
    # 3. Low scoring match indicator (low scoring matches are more likely to end in draws)
    goal_cols = []
    for col in df.columns:
        if any(pattern in col.lower() for pattern in soccer_columns['goals']):
            goal_cols.append(col)
    
    if goal_cols:
        df['low_scoring_match'] = (df[goal_cols].mean(axis=1) < df[goal_cols].mean().mean()).astype(int)
        feature_count += 1
    
    # 4. Draw odds features if available
    odds_cols = []
    for col in df.columns:
        if any(pattern in col.lower() for pattern in soccer_columns['odds']):
            odds_cols.append(col)
    
    if odds_cols:
        # Normalize odds to probabilities
        for col in odds_cols:
            prob_col = f"{col}_prob"
            df[prob_col] = 1 / df[col]
            odds_cols.append(prob_col)
            feature_count += 1
        
        # Create draw probability feature if we have draw odds
        draw_odds_col = next((col for col in odds_cols if 'draw' in col.lower()), None)
        if draw_odds_col:
            df['draw_probability'] = 1 / df[draw_odds_col]
            feature_count += 1
    
    logger.info(f"Created {feature_count} domain-specific soccer features")
    
    return df

def select_features_for_precision(df, target_col='target', n_features=None):
    """
    Select features that are most likely to help with precision.
    
    Args:
        df (pd.DataFrame): DataFrame with features and target
        target_col (str): Name of the target column
        n_features (int, optional): Number of features to select
            
    Returns:
        list: List of selected feature names
    """
    from sklearn.feature_selection import mutual_info_classif, SelectKBest
    
    logger.info("Selecting features optimized for precision")
    
    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found in DataFrame")
        return list(df.columns)
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Handle non-numeric columns
    non_numeric_cols = X.select_dtypes(exclude=['number']).columns
    if len(non_numeric_cols) > 0:
        logger.info(f"Dropping {len(non_numeric_cols)} non-numeric columns")
        X = X.drop(columns=non_numeric_cols)
    
    # Handle missing values
    X = X.fillna(0)
    
    # If n_features not specified, use half of available features
    if n_features is None:
        n_features = X.shape[1] // 2
    
    # Ensure we don't try to select more features than we have
    n_features = min(n_features, X.shape[1])
    
    # Use mutual information to select features
    selector = SelectKBest(mutual_info_classif, k=n_features)
    selector.fit(X, y)
    
    # Get selected feature names
    selected_features = X.columns[selector.get_support()].tolist()
    
    logger.info(f"Selected {len(selected_features)} features for precision")
    
    return selected_features

def get_feature_importance(model):
    """
    Get feature importance from a trained model.
    
    Args:
        model: Trained model
        
    Returns:
        pd.DataFrame: DataFrame with feature importance
    """
    try:
        from pycaret.classification import get_config, get_model_container
    except ImportError:
        logger.error("PyCaret not installed. Please install with 'pip install pycaret[full]'")
        return pd.DataFrame()
    
    logger.info("Getting feature importance")
    
    # Get feature names - updated for PyCaret 3.3.2
    try:
        # In PyCaret 3.3.2, we can get feature names from the model container
        model_container = get_model_container()
        X_train_data = model_container['X_train_transformed']
        feature_names = X_train_data.columns.tolist()
    except:
        logger.warning("Could not get feature names from PyCaret model container, trying config")
        try:
            # Fallback to get_config
            X_train_data = get_config('X_train')
            feature_names = X_train_data.columns.tolist()
        except:
            logger.error("Could not get feature names from PyCaret config")
            # Create generic feature names
            feature_names = [f'feature_{i}' for i in range(100)]  # Assume max 100 features
    
    # Get feature importance - updated for PyCaret 3.3.2
    try:
        # Try different methods to get feature importance
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models
            if len(model.coef_.shape) > 1:
                importance = np.abs(model.coef_[0])
            else:
                importance = np.abs(model.coef_)
        elif hasattr(model, 'feature_importance'):
            # LightGBM, CatBoost
            importance = model.feature_importance()
        elif hasattr(model, 'get_booster'):
            # XGBoost
            importance = model.get_booster().get_score(importance_type='gain')
            # Convert to array
            importance_array = np.zeros(len(feature_names))
            for key, value in importance.items():
                try:
                    idx = int(key.replace('f', ''))
                    if idx < len(importance_array):
                        importance_array[idx] = value
                except:
                    pass
            importance = importance_array
        # For PyCaret 3.3.2 - try to get feature importance from the model container
        elif hasattr(model, 'estimator_') and hasattr(model.estimator_, 'feature_importances_'):
            # For pipeline models in PyCaret 3.3.2
            importance = model.estimator_.feature_importances_
        elif hasattr(model, 'estimator_') and hasattr(model.estimator_, 'coef_'):
            # For pipeline models in PyCaret 3.3.2
            if len(model.estimator_.coef_.shape) > 1:
                importance = np.abs(model.estimator_.coef_[0])
            else:
                importance = np.abs(model.estimator_.coef_)
        else:
            # Try to get feature importance from the model container
            try:
                importance_df = model_container.get('feature_importance', None)
                if importance_df is not None:
                    return importance_df
            except:
                pass
                
            logger.warning("Could not get feature importance from model")
            return pd.DataFrame()
        
        # Create DataFrame
        # Ensure we don't have more features than feature names
        if len(importance) > len(feature_names):
            importance = importance[:len(feature_names)]
        elif len(importance) < len(feature_names):
            feature_names = feature_names[:len(importance)]
            
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Value': importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Value', ascending=False)
        
        # Save to CSV
        importance_path = 'feature_importance.csv'
        importance_df.to_csv(importance_path, index=False)
        logger.info(f"Feature importance saved to {importance_path}")
        
        return importance_df
    except Exception as e:
        logger.error(f"Error getting feature importance: {str(e)}")
        return pd.DataFrame() 