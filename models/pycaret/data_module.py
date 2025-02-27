"""
Data loading and preparation module for PyCaret soccer prediction.

This module handles loading data from the existing DataLoader infrastructure
and preparing it for use with PyCaret.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import logging

# Add project root to Python path if needed
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import existing data loading functions
from models.StackedEnsemble.shared.data_loader import DataLoader
from utils.logger import ExperimentLogger

# Setup logger
logger = ExperimentLogger(experiment_name="pycaret_data_module")

def load_data_for_pycaret():
    """
    Load data in PyCaret format using the existing DataLoader infrastructure.
    
    Returns:
        tuple: (train_df, test_df, val_df) - DataFrames with target column included
    """
    logger.info("Loading data for PyCaret using project DataLoader")
    
    # Initialize the DataLoader
    data_loader = DataLoader(experiment_name="pycaret_experiment")
    
    # Get train, test, and validation splits
    X_train, y_train, X_test, y_test, X_val, y_val = data_loader.load_data()
    
    # PyCaret works best with a combined DataFrame that includes the target
    # We'll create separate DataFrames for train, test, and validation
    train_df = X_train.copy()
    train_df['target'] = y_train
    
    test_df = X_test.copy()
    test_df['target'] = y_test
    
    val_df = X_val.copy()
    val_df['target'] = y_val
    
    logger.info(f"Prepared PyCaret datasets:")
    logger.info(f" - Training: {train_df.shape} with {train_df['target'].sum()} positive samples ({train_df['target'].mean()*100:.2f}%)")
    logger.info(f" - Testing: {test_df.shape} with {test_df['target'].sum()} positive samples ({test_df['target'].mean()*100:.2f}%)")
    logger.info(f" - Validation: {val_df.shape} with {val_df['target'].sum()} positive samples ({val_df['target'].mean()*100:.2f}%)")
    
    return train_df, test_df, val_df

def get_feature_names():
    """
    Get the list of selected feature names from the DataLoader.
    
    Returns:
        list: List of feature names
    """
    data_loader = DataLoader(experiment_name="pycaret_experiment")
    return data_loader.get_feature_names()

def prepare_data_for_phase(train_df, test_df, val_df, phase_num):
    """
    Prepare data for a specific phase of the implementation.
    
    Different phases may require different data preparation steps.
    
    Args:
        train_df (pd.DataFrame): Training data
        test_df (pd.DataFrame): Testing data
        val_df (pd.DataFrame): Validation data
        phase_num (int): Phase number (1, 2, etc.)
        
    Returns:
        tuple: (train_df, test_df, val_df) - Prepared DataFrames
    """
    logger.info(f"Preparing data for phase {phase_num}")
    
    # Common preprocessing for all phases
    dfs = [train_df.copy(), test_df.copy(), val_df.copy()]
    
    # Handle missing values
    for df in dfs:
        # Replace infinities with NaN first
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Count NaNs before filling
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            logger.info(f"Replacing {nan_count} NaN values")
            
        # Fill NaNs with median for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if col != 'target' and df[col].isna().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
    
    # Phase-specific preprocessing
    if phase_num == 1:
        # Basic preprocessing for phase 1
        logger.info("Phase 1: Basic preprocessing only")
        pass
    
    elif phase_num == 2:
        # More advanced preprocessing for phase 2
        logger.info("Phase 2: Adding advanced preprocessing")
        
        # Remove outliers from training data only
        if len(dfs[0]) > 1000:  # Only if we have enough samples
            logger.info("Removing outliers from training data")
            for col in dfs[0].select_dtypes(include=['number']).columns:
                if col != 'target':
                    q1 = dfs[0][col].quantile(0.01)
                    q3 = dfs[0][col].quantile(0.99)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    dfs[0] = dfs[0][(dfs[0][col] >= lower_bound) & (dfs[0][col] <= upper_bound)]
            
            logger.info(f"After outlier removal: {len(dfs[0])} training samples")
    
    return dfs[0], dfs[1], dfs[2]

def combine_train_test(train_df, test_df):
    """
    Combine training and test data for PyCaret setup.
    
    Args:
        train_df (pd.DataFrame): Training data
        test_df (pd.DataFrame): Testing data
        
    Returns:
        pd.DataFrame: Combined DataFrame
    """
    combined_df = pd.concat([train_df, test_df], axis=0)
    logger.info(f"Combined train+test data: {combined_df.shape}")
    return combined_df 