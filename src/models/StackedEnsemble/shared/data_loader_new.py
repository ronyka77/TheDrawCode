"""Data loading utilities for the stacked ensemble."""
import numpy as np
import pandas as pd

from src.utils.create_evaluation_set import (
    create_evaluation_set_new,
    import_selected_features_ensemble_new,
    import_training_data_ensemble_new,
)
from src.utils.logger import ExperimentLogger


class DataLoader:
    """Handles data loading and preprocessing for the ensemble models."""

    def __init__(self, experiment_name: str = "data_loader"):
        """Initialize the data loader.
        Args:
            experiment_name: Name of the experiment for logging
        """
        self.logger = ExperimentLogger(experiment_name=experiment_name)
        self._cached_features = None

    def load_data(
        self,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Load and split data into train, test, and validation sets.
        The data is split according to the following strategy:
        - Training data (X_train): Used for model training and nested CV
        - Test data (X_test): Used for evaluation during training (early stopping)
        - Validation data (X_val):
        """
        self.logger.info("Loading data splits according to ensemble strategy")

        # Load selected features first
        if self._cached_features is None:
            self._cached_features = import_selected_features_ensemble_new("all")
            self.logger.info(f"Loaded {len(self._cached_features)} selected features")

        X_train, y_train, X_test, y_test = import_training_data_ensemble_new()
        X_val, y_val = create_evaluation_set_new()

        # Apply feature selection to all splits and ensure consistent column order
        self.logger.info("Applying feature selection with consistent column ordering")
        X_train = X_train[self._cached_features].copy()
        X_test = X_test[self._cached_features].copy()
        X_val = X_val[self._cached_features].copy()

        # Validate column order consistency
        train_cols = list(X_train.columns)
        test_cols = list(X_test.columns)
        val_cols = list(X_val.columns)

        if not (train_cols == test_cols == val_cols):
            self.logger.warning(
                "Column order inconsistency detected - enforcing identical ordering"
            )
            # Force identical column ordering across all datasets
            X_train = X_train[self._cached_features]
            X_test = X_test[self._cached_features]
            X_val = X_val[self._cached_features]

        # Replace NaN values with 0 in all data splits
        self.logger.info("Replacing NaN values with 0 in all data splits")
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
        X_val = X_val.fillna(0)
        # Replace inf values with 0 in all data splits
        self.logger.info("Replacing inf values with 0 in all data splits")
        X_train = X_train.replace([np.inf, -np.inf], 0)
        X_test = X_test.replace([np.inf, -np.inf], 0)
        X_val = X_val.replace([np.inf, -np.inf], 0)

        # Log final data shapes
        self.logger.info(
            "Final data split sizes:"
            f"\n - Train: {X_train.shape} (for model training and nested CV)"
            f"\n - Test: {X_test.shape} (for early stopping during training)"
            f"\n - Validation: {X_val.shape} (held-out for evaluation and meta-features)"
        )
        # Log draw count in validation set
        val_draw_count = y_val.sum()
        val_total_count = len(y_val)
        val_draw_rate = val_draw_count / val_total_count if val_total_count > 0 else 0
        self.logger.info(
            f"Validation set draw statistics: {val_draw_count} draws out of {val_total_count} matches "
            f"(draw rate: {val_draw_rate:.2%})"
        )

        return X_train, y_train, X_test, y_test, X_val, y_val

    def get_feature_names(self) -> list:
        """Get the list of selected feature names.

        Returns:
            List of feature names
        """
        if self._cached_features is None:
            self._cached_features = import_selected_features_ensemble_new("all")
        return self._cached_features
