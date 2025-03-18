"""
Data Preparation Utilities

Functions for preparing and transforming data for the ensemble model.
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict
from xgboost import XGBClassifier
import mlflow
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from utils.logger import ExperimentLogger

def prepare_data(X: pd.DataFrame, selected_features: List[str]) -> pd.DataFrame:
    """
    Ensure X contains the required features and fill missing values.
    
    Args:
        X: Input dataframe
        selected_features: List of required feature names
        
    Returns:
        Prepared dataframe with required features
    """
    # Convert to DataFrame if it's a numpy array
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    
    # Ensure all required features are present
    for feature in selected_features:
        if feature not in X.columns:
            raise ValueError(f"Missing required feature: {feature}")
    
    # Select only the required features
    X_selected = X[selected_features].copy()
    
    # Fill missing values with appropriate strategies
    for col in X_selected.columns:
        # Use mean for numeric columns
        if np.issubdtype(X_selected[col].dtype, np.number):
            X_selected[col] = X_selected[col].fillna(X_selected[col].mean())
        else:
            X_selected[col] = X_selected[col].fillna(X_selected[col].mode()[0])
    
    return X_selected

def apply_adasyn_resampling(X_train: pd.DataFrame, y_train: pd.Series, 
                            sampling_strategy: float = 0.7,
                            logger: ExperimentLogger = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply ADASYN (Adaptive Synthetic Sampling) to handle class imbalance.
    
    Args:
        X_train: Training features
        y_train: Training target values
        sampling_strategy: Ratio of minority class to majority class after resampling
        logger: Logger instance
        
    Returns:
        Tuple of (resampled_features, resampled_targets)
    """
    if logger is None:
        logger = ExperimentLogger(experiment_name="ensemble_model_data",
                                log_dir="./logs/ensemble_model_data")
    
    # Get class balance information before resampling
    n_samples = len(y_train)
    n_pos = sum(y_train)
    n_neg = n_samples - n_pos
    minority_class = 1 if n_pos < n_neg else 0
    
    logger.info(f"Class distribution before ADASYN: {n_pos}/{n_samples} positive ({n_pos/n_samples:.2%})")
    
    # Skip ADASYN if we have very few minority samples
    min_minority_samples = 10
    min_samples_for_adasyn = 100
    
    if n_samples < min_samples_for_adasyn:
        logger.warning(f"Too few samples ({n_samples}) for ADASYN resampling. Skipping.")
        return X_train, y_train
    
    if min(n_pos, n_neg) < min_minority_samples:
        logger.warning(f"Too few minority samples ({min(n_pos, n_neg)}) for ADASYN resampling. Skipping.")
        return X_train, y_train
    
    # Apply ADASYN resampling
    logger.info(f"Applying ADASYN resampling with sampling_strategy={sampling_strategy}")
    
    try:
        # Configure ADASYN for CPU usage
        adasyn = ADASYN(
            sampling_strategy=sampling_strategy,
            random_state=42,
            n_neighbors=min(5, min(n_pos, n_neg) - 1),  # Adaptive neighbors
            n_jobs=-1
        )
        
        # Perform resampling
        X_resampled, y_resampled = adasyn.fit_resample(X_train.values, y_train.values)
        
        # Convert back to DataFrame/Series
        X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
        y_resampled = pd.Series(y_resampled)
        
        # Log resampling results
        n_samples_after = len(y_resampled)
        n_pos_after = sum(y_resampled)
        logger.info(f"Class distribution after ADASYN: {n_pos_after}/{n_samples_after} positive ({n_pos_after/n_samples_after:.2%})")
        logger.info(f"Added {n_samples_after - n_samples} synthetic samples")
        
        return X_resampled, y_resampled
        
    except Exception as e:
        logger.error(f"ADASYN resampling failed: {str(e)}. Using original data.")
        return X_train, y_train

def balance_and_clean_dataset(X_train: pd.DataFrame, y_train: pd.Series, 
                                outlier_threshold: float = 3.0,
                                sampling_strategy: float = 0.5,
                                logger: ExperimentLogger = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Balances the dataset using SMOTE and removes outliers using z-score method.
    
    Args:
        X_train: Feature dataframe
        y_train: Target series
        outlier_threshold: Z-score threshold for outlier detection (default: 3.0)
        sampling_strategy: Target ratio of minority to majority class (default: 0.5)
        logger: Logger instance
        
    Returns:
        Tuple of (cleaned_X, cleaned_y) as DataFrame and Series
    """
    if logger is None:
        logger = ExperimentLogger(experiment_name="data_cleaning",
                                    log_dir="./logs/data_cleaning")
    
    logger.info(f"Starting dataset balancing and outlier removal on {X_train.shape[0]} samples")
    
    # Step 1: Remove outliers using z-score method
    logger.info(f"Detecting outliers with z-score threshold of {outlier_threshold}")
    
    # Create a copy to avoid modifying the original data
    X_clean = X_train.copy()
    y_clean = y_train.copy()
    
    try:
        # Calculate z-scores for each feature
        z_scores = pd.DataFrame()
        for col in X_clean.columns:
            if X_clean[col].dtype in ['int64', 'float64']:
                z_scores[col] = np.abs((X_clean[col] - X_clean[col].mean()) / X_clean[col].std())
        
        # Identify outliers (samples with any feature having z-score > threshold)
        outlier_mask = (z_scores > outlier_threshold).any(axis=1)
        outlier_count = outlier_mask.sum()
        
        if outlier_count > 0:
            logger.info(f"Removing {outlier_count} outliers ({outlier_count/len(X_clean):.2%} of data)")
            
            # Remove outliers
            X_clean = X_clean.loc[~outlier_mask]
            y_clean = y_clean.loc[~outlier_mask]
            
            logger.info(f"Dataset size after outlier removal: {X_clean.shape[0]} samples")
        else:
            logger.info("No outliers detected")
    
    except Exception as e:
        logger.error(f"Outlier detection failed: {str(e)}. Skipping outlier removal.")
    
    # Step 2: Balance the dataset using SMOTE
    n_samples = len(y_clean)
    n_pos = sum(y_clean)
    n_neg = n_samples - n_pos
    
    logger.info(f"Class distribution before balancing: {n_pos}/{n_samples} positive ({n_pos/n_samples:.2%})")
    
    # Skip balancing if we have very few samples
    min_samples_for_balancing = 50
    min_minority_samples = 5
    
    if n_samples < min_samples_for_balancing:
        logger.warning(f"Too few samples ({n_samples}) for balancing. Skipping.")
        return X_clean, y_clean
    
    if min(n_pos, n_neg) < min_minority_samples:
        logger.warning(f"Too few minority samples ({min(n_pos, n_neg)}) for balancing. Skipping.")
        return X_clean, y_clean
    
    try:
        # Configure SMOTE for CPU usage
        # Using KNeighborsClassifier with n_jobs parameter to avoid FutureWarning
        from sklearn.neighbors import KNeighborsClassifier
        k_neighbors = min(5, min(n_pos, n_neg) - 1)  # Adaptive neighbors
        nn_estimator = KNeighborsClassifier(n_neighbors=k_neighbors, n_jobs=-1)
        
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=42,
            k_neighbors=k_neighbors,
            n_neighbors=nn_estimator  # Pass the estimator to avoid FutureWarning
        )
        
        # Perform resampling
        X_balanced, y_balanced = smote.fit_resample(X_clean, y_clean)
        
        # Convert back to DataFrame/Series if needed
        if not isinstance(X_balanced, pd.DataFrame):
            X_balanced = pd.DataFrame(X_balanced, columns=X_clean.columns)
        if not isinstance(y_balanced, pd.Series):
            y_balanced = pd.Series(y_balanced)
        
        # Log balancing results
        n_samples_after = len(y_balanced)
        n_pos_after = sum(y_balanced)
        logger.info(f"Class distribution after balancing: {n_pos_after}/{n_samples_after} positive ({n_pos_after/n_samples_after:.2%})")
        logger.info(f"Added {n_samples_after - n_samples} synthetic samples")
        
        return X_balanced, y_balanced
        
    except Exception as e:
        logger.error(f"Dataset balancing failed: {str(e)}. Using cleaned data without balancing.")
        return X_clean, y_clean


def select_features_by_importance(X: pd.DataFrame, y: pd.Series, 
                                importance_threshold: float = 0.01,
                                n_iterations: int = 5,
                                logger: ExperimentLogger = None) -> List[str]:
    """
    Select features based on importance scores from an XGBoost model.
    
    Args:
        X: Feature dataframe
        y: Target series
        importance_threshold: Minimum importance threshold for feature selection
        n_iterations: Number of iterations for aggregating importance scores
        logger: Logger instance
        
    Returns:
        List of selected feature names
    """
    if logger is None:
        logger = ExperimentLogger(experiment_name="ensemble_model_feature_selection",
                                log_dir="./logs/ensemble_model_feature_selection")
    
    logger.info(f"Starting feature selection with {X.shape[1]} features...")
    
    # Create dictionary to store accumulated feature importance
    feature_importance = {col: 0.0 for col in X.columns}
    
    # Perform multiple iterations with different random seeds
    for i in range(n_iterations):
        # Create and train an XGBoost model with CPU settings
        xgb_model = XGBClassifier(
            tree_method='hist',  # CPU-optimized method
            device='cpu',
            n_jobs=-1,
            objective='binary:logistic',
            learning_rate=0.05,
            n_estimators=100,  # Fewer trees for feature selection
            max_depth=4,  # Limited depth for stability
            random_state=42 + i,  # Different seed each iteration
            colsample_bytree=0.8,
            subsample=0.8
        )
        
        # Train the model
        xgb_model.fit(X, y)
        
        # Get feature importance
        importance = xgb_model.feature_importances_
        
        # Add to accumulated importance
        for j, col in enumerate(X.columns):
            feature_importance[col] += importance[j] / n_iterations
    
    # Calculate total importance
    total_importance = sum(feature_importance.values())
    
    # Normalize importance scores
    if total_importance > 0:
        for col in feature_importance:
            feature_importance[col] /= total_importance
    
    # Sort features by importance
    sorted_features = sorted(
        feature_importance.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # Select features above threshold
    selected_features = [
        col for col, importance in sorted_features 
        if importance >= importance_threshold
    ]
    
    # Ensure we have at least 1 feature
    if not selected_features:
        logger.warning("No features above threshold. Using top feature.")
        selected_features = [sorted_features[0][0]]
    
    logger.info(f"Selected {len(selected_features)}/{X.shape[1]} features.")
    logger.info(f"Top 5 features: {[f[0] for f in sorted_features[:5]]}")
    
    # Log to MLflow
    mlflow.log_param("selected_feature_count", len(selected_features))
    
    # Log individual feature importances 
    for feature, importance in sorted_features:
        if feature in selected_features:
            mlflow.log_metric(f"feature_importance_{feature}", importance)
    
    return selected_features
