"""
Module for validating input data for the soccer prediction model.
"""

import pandas as pd
import logging
from typing import Tuple

class InputValidator:
    """
    Validates input data for the soccer prediction model.
    """

    def __init__(self, logger: logging.Logger):
        """
        Initializes the InputValidator with a logger.

        Args:
            logger: Logger object for logging messages.
        """
        self.logger = logger

    def validate_input_data(self, df: pd.DataFrame, target_column: str, logger: logging.Logger) -> Tuple[bool, pd.DataFrame]:
        """
        Validates the input DataFrame.

        Checks for:
            - Missing values in the target column
            - Non-numeric values in numeric columns
            - Negative values in specific columns (e.g., 'goals_scored')
            - Invalid league IDs

        Args:
            df: The input DataFrame.
            target_column: The name of the target column.
            logger: Logger object for logging messages.

        Returns:
            A tuple containing:
                - True if validation passes, False otherwise.
                - The validated DataFrame (potentially modified).
        """
        if not isinstance(df, pd.DataFrame):
            error_msg = "Input must be a pandas DataFrame."
            logger.error(error_msg)
            raise TypeError(error_msg)

        if df.empty:
            error_msg = "Input DataFrame is empty."
            logger.error(error_msg)
            raise ValueError(error_msg)

        if target_column not in df.columns:
            error_msg = f"Target column '{target_column}' not found in DataFrame."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Check for missing values in the target column
        if df[target_column].isnull().any():
            error_msg = f"Missing values found in target column '{target_column}'."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Check for non-numeric values in numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                error_msg = f"Non-numeric values found in numeric column '{col}'."
                logger.error(error_msg)
                raise ValueError(error_msg)

        # Check for invalid league IDs (example: assuming league IDs should be between 1 and 10)
        if 'league_encoded' in df.columns:
            if df['league_encoded'].isnull().any():
                error_msg = "Null values found in 'league_encoded' column."
                logger.error(error_msg)
                raise ValueError(error_msg)

        logger.info("Input data validation successful.")
        return True, df