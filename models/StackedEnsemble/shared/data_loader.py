"""Data loading utilities for the stacked ensemble."""

from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
from utils.logger import ExperimentLogger
from utils.create_evaluation_set import (
    create_ensemble_evaluation_set,
    import_selected_features_ensemble,
    import_training_data_ensemble
)

class DataLoader:
    """Handles data loading and preprocessing for the ensemble models."""
    
    def __init__(self, experiment_name: str = "data_loader"):
        """Initialize the data loader.
        
        Args:
            experiment_name: Name of the experiment for logging
        """
        self.logger = ExperimentLogger(experiment_name=experiment_name)
        self._cached_features = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Load and split data into train, test, and validation sets.
        
        The data is split according to the following strategy:
        - Training data (X_train): Used for model training and nested CV
        - Test data (X_test): Used for evaluation during training (early stopping)
        - Validation data (X_val): Held-out set for:
            - Final model evaluation
            - Threshold optimization
            - Meta-feature generation
            - Meta-learner training
        
        Returns:
            Tuple containing:
            - X_train: Training features
            - y_train: Training labels
            - X_test: Test features for early stopping
            - y_test: Test labels for early stopping
            - X_val: Validation features
            - y_val: Validation labels
        """
        self.logger.info("Loading data splits according to ensemble strategy")
        
        # Load selected features first
        if self._cached_features is None:
            self._cached_features = import_selected_features_ensemble('all')
            self.logger.info(f"Loaded {len(self._cached_features)} selected features")
        
        # Load training and test data
        X_train, y_train, X_test, y_test = import_training_data_ensemble()
        self.logger.info(
            "Loaded training/test data:"
            f"\n - Training samples: {len(X_train)}"
            f"\n - Test samples: {len(X_test)}"
        )
        
        # Load validation data (completely held-out set)
        X_val, y_val = create_ensemble_evaluation_set()
        self.logger.info(f"Loaded validation data: {len(X_val)} samples")
        
        # Apply feature selection to all splits
        X_train = X_train[self._cached_features]
        X_test = X_test[self._cached_features]
        X_val = X_val[self._cached_features]
        
        # Log final data shapes
        self.logger.info(
            "Final data split sizes:"
            f"\n - Train: {X_train.shape} (for model training and nested CV)"
            f"\n - Test: {X_test.shape} (for early stopping during training)"
            f"\n - Validation: {X_val.shape} (held-out for evaluation and meta-features)"
        )
        
        return X_train, y_train, X_test, y_test, X_val, y_val
    
    def get_feature_names(self) -> list:
        """Get the list of selected feature names.
        
        Returns:
            List of feature names
        """
        if self._cached_features is None:
            self._cached_features = import_selected_features_ensemble('all')
        return self._cached_features 