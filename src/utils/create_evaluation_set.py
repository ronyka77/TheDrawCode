import json
import os
import sys
import time
from functools import wraps
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from openpyxl import Workbook
from pyexcelerate import Workbook
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split

# Add project root to Python path
try:
    project_root = Path(__file__).parent.parent.parent
    if not project_root.exists():
        # Handle network path by using raw string
        project_root = Path(r"\\".join(str(project_root).split("\\")))
    sys.path.append(str(project_root))
    print(f"Project root create_evaluation_set: {project_root}")
except Exception as e:
    print(f"Error setting project root path: {str(e)}")
    # Fallback to current directory if path resolution fails
    sys.path.append(os.getcwd().parent)
    print(f"Fallback to current directory: {os.getcwd().parent}")

from src.utils.advanced_goal_features import AdvancedGoalFeatureEngineer
from src.utils.K_factor_calculation import calculate_draw_k_factor
from src.utils.logger import ExperimentLogger
from src.utils.mlflow_utils import MLFlowManager

logger = ExperimentLogger(
    experiment_name="create_evaluation_set", log_dir="logs/create_evaluation_set"
)


# Error codes for standardized logging
class DataProcessingError:
    # File operations
    FILE_NOT_FOUND = "E001"
    FILE_PERMISSION_ERROR = "E002"
    FILE_CORRUPTED = "E003"

    # Data validation
    MISSING_REQUIRED_COLUMNS = "E101"
    INVALID_DATA_TYPE = "E102"
    NUMERIC_CONVERSION_FAILED = "E103"

    # Data processing
    EMPTY_DATASET = "E201"
    INSUFFICIENT_SAMPLES = "E202"
    FEATURE_CREATION_FAILED = "E203"

    # External services
    MONGODB_CONNECTION_ERROR = "E301"
    MLFLOW_ERROR = "E302"


# Retry decorator for file operations
def retry_on_error(max_retries: int = 3, delay: float = 1.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        time.sleep(delay * (attempt + 1))
                    continue
            if last_error:
                logger.info(f"Failed to execute function after multiple retries: {last_error}")
                raise last_error  # Re-raise the last exception after logging

        return wrapper

    return decorator


def convert_numeric_columns(
    data: pd.DataFrame,
    columns: Optional[list[str]] = None,
    drop_errors: bool = True,
    fill_value: float = 0.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Convert specified columns to numeric type with comprehensive error handling.

    This function handles various string-to-numeric conversions including:
    - Removing whitespace and quotes
    - Converting comma decimals to dots
    - Handling empty strings
    - Dealing with scientific notation
    - Managing infinite values

    Args:
        data (pd.DataFrame): Input DataFrame
        columns (Optional[List[str]]): List of columns to convert. If None, converts all columns
        drop_errors (bool): Whether to drop columns that fail conversion
        fill_value (float): Value to use for empty strings and NaN
        verbose (bool): Whether to logger.info conversion errors and warnings

    Returns:
        pd.DataFrame: DataFrame with converted numeric columns

    Example:
        >>> df = pd.DataFrame({'A': ['1.5', '2,5', '3'], 'B': ['a', 'b', 'c']})
        >>> result = convert_numeric_columns(df, columns=['A'])
        >>> logger.info(result['A'].dtype)
        float64

    Note:
        - The function creates a copy of the input DataFrame
        - Columns that fail conversion are either dropped or kept as original
        - Scientific notation (e.g., '1e-10') is properly handled
    """
    # Create a copy of the input DataFrame
    df = data.copy()

    # If no columns specified, use all columns
    if columns is None:
        columns = df.columns.tolist()

    # Track problematic columns
    failed_columns = []
    columns_to_drop = []

    # Process each column
    for col in columns:
        try:
            if col in df.columns:
                # First check if the column has any potential numeric values
                original_series = df[col].astype(str)
                has_potential_numbers = original_series.str.contains(
                    r"[0-9]|inf|-inf", case=False, regex=True
                ).any()

                if not has_potential_numbers:
                    if verbose:
                        logger.info(f"Column {col} contains no numeric values")
                    failed_columns.append(col)
                    if drop_errors:
                        columns_to_drop.append(col)
                    continue

                # Replace commas with dots (only if comma is used as decimal separator)
                df[col] = (
                    df[col]
                    .astype(str)
                    .apply(
                        lambda x: x.replace(",", ".")
                        if x.count(",") == 1 and x.count(".") == 0
                        else x
                    )
                )

                # Convert to string first to handle all cases
                series = (
                    df[col]
                    .apply(lambda x: str(x) if pd.notnull(x) else str(fill_value))
                    .str.strip()  # Remove leading/trailing whitespace
                    .str.strip("'\"")  # Remove quotes
                    .str.replace("[^0-9.eE-]", "", regex=True)  # Keep only numeric chars
                    .apply(
                        lambda x: str(fill_value) if x in ["", "e", "e-", "e+"] else x
                    )  # Handle empty and bare 'e'
                    .apply(
                        lambda x: "1" + x if x.lower().startswith(("e", "e-", "e+")) else x
                    )  # Fix sci notation
                    .apply(
                        lambda x: x.replace("-", "e-", 1)
                        if "-" in x and "e" not in x.lower()
                        else x
                    )  # Handle negatives
                )

                # Try converting to numeric
                numeric_series = pd.to_numeric(series, errors="coerce")

                # Check if conversion was successful
                if numeric_series.isna().all():
                    if verbose:
                        logger.info(f"Column {col} contains no valid numeric values")
                    failed_columns.append(col)
                    if drop_errors:
                        columns_to_drop.append(col)
                    continue

                # Apply the conversion
                df[col] = numeric_series.replace([np.inf, -np.inf], fill_value).fillna(fill_value)

                if verbose and df[col].isna().any():
                    logger.info(f"Warning: Column {col} contains NaN values after conversion")

        except Exception as e:
            if verbose:
                logger.info(f"Error converting column {col}: {str(e)}")
            failed_columns.append(col)
            if drop_errors:
                columns_to_drop.append(col)

    # Drop failed columns at the end
    if drop_errors and columns_to_drop:
        df = df.drop(columns=columns_to_drop, errors="ignore")
        if verbose:
            logger.info(f"Dropped columns: {columns_to_drop}")

    if verbose and failed_columns:
        logger.info("\nConversion summary:")
        logger.info(f"Failed columns: {failed_columns}")
        logger.info(f"Successfully converted: {len(columns) - len(failed_columns)} columns")
        logger.info(f"Failed conversions: {len(failed_columns)} columns")

    return df


def create_parquet_files(
    data: pd.DataFrame, output_path: str, partition_cols: Optional[list[str]] = None
) -> None:
    """Create Parquet files from a DataFrame.
    This function takes a Pandas DataFrame and saves it as Parquet files.
    It supports partitioning by specified columns and handles error logging.
    Args:
        data: The DataFrame to be saved.
        output_path: The base name for the Parquet files.
        partition_cols: Optional list of columns to partition by.
    Raises:
        ValueError: If the DataFrame is empty or if partitioning fails.
        Exception: For other errors during Parquet file creation.
    """
    if data.empty:
        logger.info(
            "Cannot create Parquet files from an empty DataFrame.",
            error_code=DataProcessingError.EMPTY_DATASET,
        )
        raise ValueError("DataFrame is empty")
    logger.info(f"Creating Parquet files at: {output_path}")
    try:
        if partition_cols:
            data.to_parquet(
                output_path, partition_cols=partition_cols, engine="fastparquet", index=False
            )
            logger.info(f"Successfully created partitioned Parquet files: {partition_cols}")
        else:
            data.to_parquet(output_path, engine="fastparquet", index=False)
            logger.info("Successfully created Parquet files without partitioning.")
    except Exception as e:
        logger.info(
            f"Error creating Parquet files: {str(e)}", error_code=DataProcessingError.FILE_CORRUPTED
        )
        raise


# MLFLOW SETUP
@retry_on_error(max_retries=3, delay=1.0)
def setup_mlflow_tracking(experiment_name: str) -> str:
    """Configure MLflow tracking for experiment monitoring.
    This function sets up MLflow tracking for experiment monitoring and model versioning.
    It configures the tracking URI, creates or gets the experiment, and ensures proper
    directory structure for MLflow artifacts.
    Args:
        experiment_name (str): Name of the MLflow experiment to create or get.
    Returns:
        str: Path to the mlruns directory where MLflow stores its data.
    Raises:
        ConnectionError: If MLflow tracking server is not accessible
        ValueError: If experiment name is invalid
        Exception: For other MLflow-related errors
    """
    logger.info(f"Setting up MLflow tracking for experiment: {experiment_name}")

    try:
        mlflow_manager = MLFlowManager()
        mlflow_manager.setup_experiment(experiment_name)
        logger.info(f"MLflow tracking configured successfully at: {mlflow_manager.mlruns_dir}")
        return mlflow_manager.mlruns_dir

    except ConnectionError as e:
        logger.info(
            f"MLflow connection error: {str(e)}", error_code=DataProcessingError.MLFLOW_ERROR
        )
        raise
    except ValueError as e:
        logger.info(
            f"Invalid experiment configuration: {str(e)}",
            error_code=DataProcessingError.MLFLOW_ERROR,
        )
        raise
    except Exception as e:
        logger.info(f"MLflow setup error: {str(e)}", error_code=DataProcessingError.MLFLOW_ERROR)
        raise


@retry_on_error(max_retries=3, delay=2.0)
def sync_mlflow() -> None:
    """Synchronize MLflow data with shared storage.
    This function performs a two-way sync of MLflow data:
    1. Backs up local MLflow data to shared storage
    2. Syncs any updates from shared storage back to local
    This ensures consistency across different development environments
    and provides backup of experiment tracking data.
    Raises:
        ConnectionError: If shared storage is not accessible
        PermissionError: If lacking write permissions
        Exception: For other sync-related errors
    """
    logger.info("Starting MLflow data synchronization")

    try:
        mlflow_manager = MLFlowManager()

        # Backup to shared storage
        logger.info("Backing up MLflow data to shared storage")
        mlflow_manager.backup_to_shared()

        # Sync from shared storage
        logger.info("Syncing from shared storage")
        mlflow_manager.sync_with_shared()

        logger.info("MLflow synchronization completed successfully")

    except ConnectionError as e:
        logger.info(
            f"Shared storage connection error: {str(e)}",
            error_code=DataProcessingError.MLFLOW_ERROR,
        )
        raise
    except PermissionError as e:
        logger.info(
            f"Permission denied accessing shared storage: {str(e)}",
            error_code=DataProcessingError.FILE_PERMISSION_ERROR,
        )
        raise
    except Exception as e:
        logger.info(f"MLflow sync error: {str(e)}", error_code=DataProcessingError.MLFLOW_ERROR)
        raise


def update_api_training_data_for_draws():
    """
    Update training data for draws by adding advanced goal features and saving to training_data.xlsx
    """
    try:
        # Load existing training data
        data_path = "data/api_training_final.xlsx"
        data = pd.read_excel(data_path)
        # Initialize the feature engineer
        feature_engineer = AdvancedGoalFeatureEngineer()
        # Add advanced goal features
        updated_data = feature_engineer.add_goal_features(data)
        logger.info(updated_data.shape)
        # Save updated data back to Excel
        updated_data.to_excel(data_path, index=False)
    except Exception as e:
        logger.info(f"Error updating training data for draws: {str(e)}")


def update_api_data_for_draws():
    """
    Update prediction data for draws by adding advanced goal features and saving to api_prediction_data_new.xlsx
    """
    try:
        # Load existing training data
        data_path = "data/prediction/api_prediction_data.xlsx"
        data_path_new = "data/prediction/api_prediction_data_new.xlsx"
        data = pd.read_excel(data_path)
        # Initialize the feature engineer
        feature_engineer = AdvancedGoalFeatureEngineer()
        # Add advanced goal features
        updated_data = feature_engineer.add_goal_features(data)
        logger.info(updated_data.shape)

        # Filter data for dates before 2024-11-01
        api_training_data = updated_data[updated_data["Date"] < "2025-03-01"]
        # Add is_draw column for training data
        api_training_data.loc[:, "is_draw"] = (api_training_data["match_outcome"] == 2).astype(int)
        logger.info("Added is_draw column to training data")
        # Filter data for dates after 2024-11-01 where match_outcome is not blank
        api_prediction_eval = updated_data[
            (updated_data["Date"] >= "2025-03-01") & (updated_data["match_outcome"].notna())
        ]
        # Filter data for dates after 2024-11-01 where match_outcome is blank
        api_prediction_data = updated_data[
            (updated_data["Date"] >= "2025-03-01") & (updated_data["match_outcome"].isna())
        ]

        logger.info(f"api_prediction_data.shape: {api_prediction_data.shape}")
        logger.info(f"api_prediction_eval.shape: {api_prediction_eval.shape}")
        logger.info(f"api_training_data.shape: {api_training_data.shape}")
        # Concatenate the filtered dataframes
        updated_data = pd.concat([api_prediction_eval, api_prediction_data], ignore_index=True)

        # Export df_before_2024_11_01 to data/api_training_final.xlsx and .parquet
        save_data_to_excel(api_training_data, "data/api_training_final.xlsx", "api_training_final")
        create_parquet_files(api_training_data, "data/api_training_final.parquet")
        logger.info("api_training_final.xlsx and .parquet updated")

        # Export df_after_2024_11_01_not_blank to data/prediction/api_predictions_eval.xlsx and .parquet
        save_data_to_excel(
            api_prediction_eval, "data/prediction/api_prediction_eval.xlsx", "api_prediction_eval"
        )
        create_parquet_files(api_prediction_eval, "data/prediction/api_prediction_eval.parquet")
        logger.info("api_prediction_eval.xlsx and .parquet updated")

        # Export df_after_2024_11_01_blank to data/prediction/api_predictions_data.xlsx and .parquet
        save_data_to_excel(
            api_prediction_data, "data/prediction/api_predictions_data.xlsx", "api_prediction_data"
        )
        create_parquet_files(api_prediction_data, "data/prediction/api_predictions_data.parquet")
        logger.info("api_predictions_data.xlsx and .parquet updated")
        # Save updated data back to Excel
        updated_data.to_excel(data_path_new, index=False)
    except Exception as e:
        logger.info(f"Error updating training data for draws: {str(e)}")


def update_api_prediction_data():
    """
    Update prediction data by adding advanced goal features and saving to predictions_eval.xlsx.
    """
    try:
        # Load existing prediction data
        data_path = "data/prediction/api_prediction_data.xlsx"
        data_path_new = "data/prediction/api_prediction_data_new.xlsx"
        data = pd.read_excel(data_path)
        # Load existing prediction data
        data_path_eval = "data/prediction/api_prediction_eval.xlsx"
        data_eval = pd.read_excel(data_path_eval)
        # Initialize the feature engineer
        feature_engineer = AdvancedGoalFeatureEngineer()
        # Add advanced goal features
        updated_data = feature_engineer.add_goal_features(data)
        logger.info(f"Updated prediction data shape: {updated_data.shape}")
        # Merge with api_prediction_eval but only add columns which are not
        # exists in prediction_data
        merged_data = merge_and_append(updated_data, data_eval)
        # Drop duplicates based on fixture_id column
        merged_data = merged_data.drop_duplicates(subset=["fixture_id"], keep="first")
        # Save updated data back to Excel
        merged_data.to_excel(data_path_new, index=False)
    except Exception as e:
        logger.info(f"Error updating prediction data: {str(e)}")


def merge_and_append(updated_data, data_eval):
    """
    Merge two DataFrames:
    - For same columns: Append rows from data_eval to the bottom of updated_data.
    - For new columns: Add new columns from data_eval to updated_data.
    Parameters:
        updated_data (pd.DataFrame): The primary DataFrame to be updated.
        data_eval (pd.DataFrame): The DataFrame to merge and append.
        on (str): The column to merge on (default: 'fixture_id').
    Returns:
        pd.DataFrame: The merged and updated DataFrame.
    """
    # Identify common and new columns
    common_columns = [col for col in updated_data.columns if col in data_eval.columns]
    new_columns = [col for col in updated_data.columns if col not in data_eval.columns]
    # Append rows for common columns
    merged_data = pd.concat([data_eval, updated_data[common_columns]], axis=0, ignore_index=True)
    # Add new columns
    for col in new_columns:
        merged_data[col] = updated_data[col]
    # logger.info updated shape
    logger.info(f"Updated prediction data shape after merge: {merged_data.shape}")
    return merged_data


#  FOR API
def get_selected_api_columns_draws() -> list[str]:
    """Get selected feature columns for API-based draw prediction model.
    This function returns a curated list of feature columns that have been
    selected based on feature importance analysis and domain knowledge.
    The features are ordered by their importance score from highest to lowest.
    Returns:
        List[str]: List of selected feature column names, ordered by importance.
    Example:
        >>> columns = get_selected_api_columns_draws()
        >>> logger.info(f"Number of features: {len(columns)}")
        Number of features: 57
        >>> logger.info("Top 3 features:")
        >>> for col in columns[:3]:
        ...     logger.info(f"- {col}")
        - venue_draw_rate
        - form_weighted_xg_diff
        - home_draw_rate
    Note:
        The feature selection is based on a combination of:
        - Feature importance scores from trained models
        - Domain expertise in soccer analytics
        - Performance impact on validation data
        - Feature stability and reliability
    """
    selected_columns = [
        "league_draw_rate_composite",
        "Home_possession_mean",
        "date_encoded",
        "home_yellow_cards_mean",
        "possession_balance",
        "home_saves_rollingaverage",
        "home_team_elo",
        "away_defense_index",
        "Home_saves_mean",
        "Away_fouls_mean",
        "seasonal_draw_pattern",
        "away_shot_on_target_rollingaverage",
        "defensive_stability",
        "away_yellow_cards_rollingaverage",
        "draw_xg_indicator",
        "home_fouls_rollingaverage",
        "Home_draws",
        "Home_fouls_mean",
        "elo_similarity_form_similarity",
        "home_days_since_last_draw",
        "venue_encoded",
        "away_poisson_xG",
        "home_draw_rate",
        "home_passing_efficiency",
        "home_red_cards_mean",
        "away_corners_mean",
        "Away_passes_mean",
        "home_defense_index",
        "away_tactical_adaptability",
        "away_team_elo",
        "historical_draw_tendency",
        "h2h_avg_goals",
        "mid_season_factor",
        "venue_capacity",
        "referee_goals_per_game",
        "away_yellow_cards_mean",
        "home_founded_year",
        "Away_offsides_mean",
        "away_saves_rollingaverage",
        "draw_probability_score",
        "away_defense_weakness",
        "away_days_since_last_draw",
        "Home_offsides_mean",
        "combined_draw_rate",
        "home_corners_mean",
        "Home_shot_on_target_mean",
        "xg_equilibrium",
        "away_corners_rollingaverage",
        "away_founded_year",
        "home_corners_rollingaverage",
        "away_encoded",
        "form_weighted_xg_diff",
        "Away_saves_mean",
        "draw_propensity_score",
        "venue_draw_rate",
        "home_yellow_cards_rollingaverage",
        "away_fouls_rollingaverage",
    ]
    return selected_columns


@retry_on_error(max_retries=3, delay=1.0)
def import_training_data_draws_api() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Import training data for API-based draw predictions.
    This function loads and preprocesses training data specifically for the API-based
    draw prediction model. It handles data cleaning, type conversion, and train-test
    splitting with proper stratification.
    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: A tuple containing:
            - X_train (pd.DataFrame): Training features
            - y_train (pd.Series): Training targets (1 for draw, 0 for non-draw)
            - X_test (pd.DataFrame): Testing features
            - y_test (pd.Series): Testing targets (1 for draw, 0 for non-draw)
    Raises:
        FileNotFoundError: If the data file cannot be found
        ValueError: If data validation fails
        Exception: For other processing errors
    """
    data_path = "data/api_training_final.xlsx"
    logger.info(f"Loading training data from: {data_path}")
    # Get selected columns
    selected_columns = get_selected_api_columns_draws()

    try:
        parquet_path = "data/api_training_final.parquet"
        if os.path.exists(parquet_path):
            data = pd.read_parquet(parquet_path)
            logger.info(f"Loaded data from Parquet: {parquet_path}")
        else:
            # Load data with retry mechanism
            data = pd.read_excel(data_path)
            if data.empty:
                logger.info("Loaded dataset is empty", error_code=DataProcessingError.EMPTY_DATASET)
                raise ValueError("Dataset is empty")

            logger.info(f"Successfully loaded data with shape: {data.shape}")

            # Create target variable
            data["is_draw"] = (data["match_outcome"] == 2).astype(int)
            logger.info(f"Created target variable. Draw rate: {data['is_draw'].mean():.2%}")

            missing_columns = [col for col in selected_columns if col not in data.columns]
            if missing_columns:
                logger.info(
                    f"Missing required columns: {missing_columns}",
                    error_code=DataProcessingError.MISSING_REQUIRED_COLUMNS,
                )
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Replace inf and nan values
            data = data.replace([np.inf, -np.inf], np.nan)
            logger.info("Replaced infinite values with NaN")

            # Convert numeric columns
            logger.info("Starting numeric conversion")
            data = convert_numeric_columns(
                data=data,
                columns=data.columns.tolist(),
                drop_errors=False,
                fill_value=0.0,
                verbose=True,
            )

            # Define and convert integer columns
            int_columns = [
                "h2h_draws",
                "home_h2h_wins",
                "h2h_matches",
                "Away_points_cum",
                "Home_points_cum",
                "Home_team_matches",
                "Home_draws",
                "venue_encoded",
            ]

            # Convert integer columns
            for col in int_columns:
                if col in data.columns:
                    try:
                        data[col] = data[col].astype("int64")
                    except Exception as e:
                        logger.info(
                            f"Failed to convert {col} to integer: {str(e)}",
                            error_code=DataProcessingError.NUMERIC_CONVERSION_FAILED,
                        )

            # Verify numeric conversion
            object_columns = []
            for col in selected_columns:
                if data[col].dtype == "object":
                    object_columns.append(col)
                    logger.info(
                        f"Column {col} remains as object type after conversion",
                        error_code=DataProcessingError.INVALID_DATA_TYPE,
                    )

            if object_columns:
                logger.info(
                    f"Found {len(object_columns)} non-numeric columns: {object_columns}",
                    error_code=DataProcessingError.INVALID_DATA_TYPE,
                )
                raise ValueError(f"Non-numeric columns found: {object_columns}")
            create_parquet_files(data, "data/api_training_final.parquet")

        # Split into train and test sets
        logger.info("Splitting data into train and test sets")
        train_data, test_data = train_test_split(
            data, test_size=0.2, random_state=42, stratify=data["is_draw"]
        )
        # Select features and target
        X_train = train_data[selected_columns]
        y_train = train_data["is_draw"]
        X_test = test_data[selected_columns]
        y_test = test_data["is_draw"]
        # Final validation
        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Test set shape: {X_test.shape}")
        logger.info(f"Training draw rate: {y_train.mean():.2%}")
        logger.info(f"Test draw rate: {y_test.mean():.2%}")
        return X_train, y_train, X_test, y_test

    except FileNotFoundError:
        logger.info(
            f"Data file not found: {data_path}", error_code=DataProcessingError.FILE_NOT_FOUND
        )
        raise
    except pd.errors.EmptyDataError:
        logger.info(f"Empty data file: {data_path}", error_code=DataProcessingError.EMPTY_DATASET)
        raise
    except Exception as e:
        logger.info(
            f"Error processing training data: {str(e)}",
            error_code=DataProcessingError.FILE_CORRUPTED,
        )
        raise


def import_feature_select_draws_api():
    """Import training data for draw predictions."""
    data_path = "data/api_training_final.xlsx"
    data = pd.read_excel(data_path)
    # Create target variable
    data["is_draw"] = (data["match_outcome"] == 2).astype(int)
    # Select features and target
    columns_to_drop = [
        "match_outcome",
        "home_goals",
        "away_goals",
        "total_goals",
        "score",
        "Referee",
        "draw",
        "venue_name",
        "Home",
        "Away",
        "away_win",
        "Date",
        "date",
        "referee_draw_rate",
        "referee_draws",
        "referee_match_count",
        "referee_foul_rate",
        "referee_match_count",
        "referee_encoded",
        "ref_goal_tendency",
    ]
    data = data.drop(columns=columns_to_drop, errors="ignore")
    # Convert all numeric-like columns (excluding problematic_cols that have
    # already been handled)
    data = convert_numeric_columns(
        data=data, columns=data.columns.tolist(), drop_errors=False, fill_value=0.0, verbose=True
    )
    # Define integer columns that should remain as int64
    int_columns = [
        "h2h_draws",
        "home_h2h_wins",
        "h2h_matches",
        "Away_points_cum",
        "Home_points_cum",
        "Home_team_matches",
        "Home_draws",
        "venue_encoded",
    ]
    # Convert integer columns back to int64
    for col in int_columns:
        if col in data.columns:
            data[col] = data[col].astype("int64")
    # Split into train and test sets
    train_data, test_data = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data["is_draw"]
    )
    X_train = train_data.drop(columns="is_draw", errors="ignore")
    y_train = train_data["is_draw"]
    X_test = test_data.drop(columns="is_draw", errors="ignore")
    y_test = test_data["is_draw"]
    # Add verification of dtypes
    logger.info("\nVerifying final dtypes:")
    non_numeric_cols = X_train.select_dtypes(include=["object"]).columns
    if len(non_numeric_cols) > 0:
        logger.info(f"Warning: Found object columns in X_train: {list(non_numeric_cols)}")
    logger.info("\nInteger columns dtypes:")
    for col in int_columns:
        if col in X_train.columns:
            logger.info(f"{col}: {X_train[col].dtype}")
    return X_train, y_train, X_test, y_test


@retry_on_error(max_retries=3, delay=1.0)
def create_evaluation_sets_draws_api(use_selected_columns: bool = True):
    """Load data from an Excel file and create evaluation sets for training.
    This function loads match data from the API prediction evaluation file and creates
    evaluation sets for draw prediction training. It handles data cleaning, type conversion,
    and feature selection.
    Args:
        use_selected_columns (bool): If True, restricts data to selected columns.
            If False, uses all available columns. Default is True.
    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple containing:
            - X (pd.DataFrame): Features for evaluation
            - y (pd.Series): Target variable (1 for draw, 0 for non-draw)
    Raises:
        FileNotFoundError: If the data file cannot be found
        ValueError: If data validation fails
        Exception: For other processing errors
    """
    file_path = "data/prediction/api_prediction_eval.xlsx"
    logger.info(f"Loading evaluation data from: {file_path}")

    try:
        # Load data from the Excel file
        data = pd.read_excel(file_path)
        if data.empty:
            logger.info("Loaded dataset is empty", error_code=DataProcessingError.EMPTY_DATASET)
            raise ValueError("Dataset is empty")

        logger.info(f"Successfully loaded data with shape: {data.shape}")
        # Filter data where match_outcome is not NA
        data = data.dropna(subset=["match_outcome"])
        logger.info(f"Data shape after filtering NA match outcomes: {data.shape}")
        # Replace inf and nan values
        data = data.replace([np.inf, -np.inf], np.nan)
        logger.info("Replaced infinite values with NaN")
        # Get selected columns if needed
        if use_selected_columns:
            selected_columns = get_selected_api_columns_draws()
            missing_columns = [col for col in selected_columns if col not in data.columns]
            if missing_columns:
                logger.info(
                    f"Missing required columns: {missing_columns}",
                    error_code=DataProcessingError.MISSING_REQUIRED_COLUMNS,
                )
                raise ValueError(f"Missing required columns: {missing_columns}")
        else:
            # Use all columns except the target and date columns
            selected_columns = [
                col for col in data.columns if col not in ["match_outcome", "is_draw", "Date"]
            ]
            logger.info("Using all available columns for evaluation set")
        # Process match outcome and create target variable
        try:
            data["match_outcome"] = data["match_outcome"].astype(int)
            data["is_draw"] = (data["match_outcome"] == 2).astype(int)
            logger.info(f"Created target variable. Draw rate: {data['is_draw'].mean():.2%}")
        except Exception as e:
            logger.info(
                f"Failed to process match outcome: {str(e)}",
                error_code=DataProcessingError.NUMERIC_CONVERSION_FAILED,
            )
            raise ValueError("Invalid match outcome values") from e
        # Ensure date_encoded exists
        if "date_encoded" not in data.columns:
            try:
                reference_date = pd.Timestamp("2020-08-11")
                data["date_encoded"] = (pd.to_datetime(data["Date"]) - reference_date).dt.days
                logger.info("Added date_encoded column")
            except Exception as e:
                logger.info(
                    f"Failed to create date_encoded: {str(e)}",
                    error_code=DataProcessingError.FEATURE_CREATION_FAILED,
                )
                raise ValueError("Could not create date_encoded column") from e
        # Convert integer columns
        int_columns = [
            "h2h_draws",
            "home_h2h_wins",
            "h2h_matches",
            "Away_points_cum",
            "Home_points_cum",
            "Home_team_matches",
            "Home_draws",
            "venue_encoded",
            "date_encoded",
        ]
        for col in int_columns:
            if col in data.columns:
                try:
                    data[col] = data[col].astype("int64")
                except Exception as e:
                    logger.info(
                        f"Failed to convert {col} to integer: {str(e)}",
                        error_code=DataProcessingError.NUMERIC_CONVERSION_FAILED,
                    )
        # Convert numeric columns
        logger.info("Starting numeric conversion")
        data = convert_numeric_columns(
            data=data, columns=selected_columns, drop_errors=False, fill_value=0.0, verbose=True
        )
        logger.info(f"Data shape after numeric conversion: {data.shape}")
        # Verify numeric conversion
        object_columns = []
        for col in selected_columns:
            if data[col].dtype == "object":
                object_columns.append(col)
                logger.info(
                    f"Column {col} remains as object type after conversion",
                    error_code=DataProcessingError.INVALID_DATA_TYPE,
                )

        if object_columns:
            logger.info(
                f"Found {len(object_columns)} non-numeric columns: {object_columns}",
                error_code=DataProcessingError.INVALID_DATA_TYPE,
            )
            raise ValueError(f"Non-numeric columns found: {object_columns}")
        # Create final feature set and target
        X = data[selected_columns]
        y = data["is_draw"]
        # Final validation
        logger.info(f"Final feature set shape: {X.shape}")
        logger.info(f"Draw rate in evaluation set: {y.mean():.2%}")
        return X, y
    except FileNotFoundError:
        logger.info(
            f"Data file not found: {file_path}", error_code=DataProcessingError.FILE_NOT_FOUND
        )
        raise
    except pd.errors.EmptyDataError:
        logger.info(f"Empty data file: {file_path}", error_code=DataProcessingError.EMPTY_DATASET)
        raise
    except Exception as e:
        logger.info(
            f"Error processing evaluation data: {str(e)}",
            error_code=DataProcessingError.FILE_CORRUPTED,
        )
        raise


@retry_on_error(max_retries=3, delay=1.0)
def create_prediction_set_api() -> pd.DataFrame:
    """Load and preprocess data for API-based predictions.
    This function loads match data from api_prediction_data_new.xlsx and preprocesses
    it for prediction. It handles data cleaning, type conversion, and feature selection
    based on the API model requirements.
    Returns:
        pd.DataFrame: Preprocessed features ready for prediction.
            The DataFrame contains all selected features in the correct format
            for the API prediction model.
    Raises:
        FileNotFoundError: If the data file cannot be found
        ValueError: If data validation fails
        Exception: For other processing errors
    """
    file_path = "data/prediction/api_prediction_data_new.xlsx"
    logger.info(f"Loading prediction data from: {file_path}")

    try:
        # Load data from the Excel file
        data = pd.read_excel(file_path)
        if data.empty:
            logger.info("Loaded dataset is empty", error_code=DataProcessingError.EMPTY_DATASET)
            raise ValueError("Dataset is empty")

        logger.info(f"Successfully loaded data with shape: {data.shape}")
        # Get selected columns
        selected_columns = ["fixture_id", "Home", "Away", "Date"] + get_selected_api_columns_draws()
        missing_columns = [col for col in selected_columns if col not in data.columns]
        if missing_columns:
            logger.info(
                f"Missing required columns: {missing_columns}",
                error_code=DataProcessingError.MISSING_REQUIRED_COLUMNS,
            )
            raise ValueError(f"Missing required columns: {missing_columns}")
        # Ensure date_encoded exists
        if "date_encoded" not in data.columns:
            try:
                reference_date = pd.Timestamp("2020-08-11")
                data["date_encoded"] = (pd.to_datetime(data["Date"]) - reference_date).dt.days
                logger.info("Added date_encoded column")
            except Exception as e:
                logger.info(
                    f"Failed to create date_encoded: {str(e)}",
                    error_code=DataProcessingError.FEATURE_CREATION_FAILED,
                )
                raise ValueError("Could not create date_encoded column") from e
        data_copy = data.copy()
        # Drop Date, Home, and Away columns from original data
        if "Date" in data.columns:
            data = data.drop(columns=["Date"])
        if "Home" in data.columns:
            data = data.drop(columns=["Home"])
        if "Away" in data.columns:
            data = data.drop(columns=["Away"])
        logger.info("Dropped Date, Home, and Away columns from data")
        # Convert numeric columns
        logger.info("Starting numeric conversion")
        data = convert_numeric_columns(
            data=data,
            columns=None,  # Convert all columns
            drop_errors=True,
            fill_value=0.0,
            verbose=True,
        )
        logger.info(f"Data shape after numeric conversion: {data.shape}")
        # Add Date column back from original copy
        if "Date" not in data.columns and "Date" in data_copy.columns:
            data["Date"] = data_copy["Date"]
            data["Home"] = data_copy["Home"]
            data["Away"] = data_copy["Away"]
            logger.info("Restored Date, Home, and Away columns from original data")
        # Select only the required columns
        X = data[selected_columns]
        # Final validation
        logger.info(f"Final feature set shape: {X.shape}")
        logger.info("Feature set ready for prediction")
        return X
    except FileNotFoundError:
        logger.info(
            f"Data file not found: {file_path}", error_code=DataProcessingError.FILE_NOT_FOUND
        )
        raise
    except pd.errors.EmptyDataError:
        logger.info(f"Empty data file: {file_path}", error_code=DataProcessingError.EMPTY_DATASET)
        raise
    except Exception as e:
        logger.info(
            f"Error processing prediction data: {str(e)}",
            error_code=DataProcessingError.FILE_CORRUPTED,
        )
        raise


# ENSEMBLE FUNCTIONS
def import_selected_features_ensemble(model_type: Optional[str] = None) -> Union[dict, list]:
    """Import selected features for XGBoost, CatBoost, and LightGBM models from JSON file.
    This function loads the pre-selected features for each model type from the
    selected_features_ensemble.json file. The features were selected based on
    composite importance scores from feature selection analysis.
    Args:
        model_type (Optional[str]): Specific model type to return features for.
            Options: 'xgb', 'cat', 'lgbm', 'all'. If None, returns all features.
    Returns:
        Union[dict, list]: If model_type is None, returns dictionary containing selected features
            for each model type with keys:
            - 'xgb': List of features for XGBoost
            - 'cat': List of features for CatBoost
            - 'lgbm': List of features for LightGBM
        If model_type is 'all', returns list of features that are common to all models
        If model_type is specified ('xgb', 'cat', 'lgbm'), returns list of features for that model type.
    Raises:
        FileNotFoundError: If the JSON file cannot be found
        JSONDecodeError: If the JSON file is malformed
        ValueError: If invalid model_type is provided
        Exception: For other errors during file loading
    """
    try:
        # Define path to JSON file
        json_path = project_root / "src" / "utils" / "selected_features_ensemble.json"
        # Load and parse JSON file
        with open(json_path) as f:
            features = json.load(f)
        # Validate loaded data structure
        if not all(key in features for key in ["xgb", "cat", "lgbm", "rf", "tabnet", "mlp", "pytorch", "svm"]):
            raise ValueError("JSON file missing required model keys")
        # Return specific model type if requested
        if model_type is not None:
            if model_type == "all":
                # Get intersection of features across all models
                # common_features = list(
                #     set(features["xgb"]).union(
                #         features["cat"],
                #         features["lgbm"],
                #         features["rf"],
                #         features["tabnet"],
                #         features["mlp"],
                #     )
                # )
                common_features = features["all"]
                logger.info("Returning features common to all models")
                return common_features
            elif model_type not in ["xgb", "cat", "lgbm", "rf", "tabnet", "mlp", "pytorch", "svm", "all"]:
                raise ValueError(
                    f"Invalid model_type: {model_type}. Must be one of: 'xgb', 'cat', 'lgbm', 'rf', 'tabnet', 'mlp', 'pytorch', 'svm', 'all'"
                )
            logger.info(f"Returning selected features for model type: {model_type}")
            return features[model_type]
        logger.info("Successfully loaded all selected features from JSON file")
        return features
    except FileNotFoundError as e:
        logger.info(
            f"Selected features JSON file not found: {str(e)}",
            error_code=DataProcessingError.FILE_NOT_FOUND,
        )
        raise
    except json.JSONDecodeError as e:
        logger.info(
            f"Invalid JSON format in features file: {str(e)}",
            error_code=DataProcessingError.FILE_CORRUPTED,
        )
        raise
    except Exception as e:
        logger.info(
            f"Error loading selected features: {str(e)}",
            error_code=DataProcessingError.FILE_CORRUPTED,
        )
        raise


def create_ensemble_evaluation_set() -> pd.DataFrame:
    """Create evaluation set for ensemble training with selected features and evaluation columns.
    This function creates a dataset containing all features from selected_features_ensemble.json
    along with the target variable (is_draw) and an evaluator column for model comparison.
    Returns:
        pd.DataFrame: DataFrame containing:
            - All features from selected_features_ensemble
            - is_draw: Target variable (1 for draw, 0 otherwise)
            - evaluator: Column for model evaluation tracking
    Raises:
        FileNotFoundError: If required data files are not found
        ValueError: If data validation fails
        Exception: For other processing errors
    """
    try:
        # Load training data
        data_path = os.path.join(project_root, "data", "prediction", "api_prediction_eval.parquet")
        logger.info(f"Loading training data from: {data_path}")
        data = pd.read_parquet(data_path)
        # Create target variable
        data["is_draw"] = (data["match_outcome"] == 2).astype(int)
        # Select features and target
        columns_to_drop = [
            "match_outcome",
            "home_goals",
            "away_goals",
            "total_goals",
            "score",
            "Referee",
            "draw",
            "venue_name",
            "Home",
            "Away",
            "away_win",
            "Date",
            "date",
            "referee_draw_rate",
            "referee_home_impact",
            "referee_goals_per_game",
            "referee_away_impact",
            "referee_draws",
            "referee_match_count",
            "referee_foul_rate",
            "referee_match_count",
            "referee_encoded",
            "ref_goal_tendency",
            "mid_season_factor",
        ]
        data = data.drop(columns=columns_to_drop, errors="ignore")
        # Import selected features for ensemble models
        selected_features = data.columns.tolist()  # import_selected_features_ensemble()
        all_features = [feature for feature in selected_features if feature != "is_draw"]

        # Validate all features exist in data
        missing_features = [feature for feature in all_features if feature not in data.columns]
        if missing_features:
            logger.info(
                f"Missing required features: {missing_features}",
                error_code=DataProcessingError.MISSING_REQUIRED_COLUMNS,
            )
            raise ValueError(f"Missing required features: {missing_features}")
        # Select features and add evaluator column
        evaluation_data = data[selected_features].copy()
        evaluation_data["is_draw"] = data["is_draw"]
        # Convert numeric columns
        evaluation_data = convert_numeric_columns(
            data=evaluation_data, columns=None, drop_errors=True, fill_value=0.0, verbose=True
        )
        # Split into features and target
        X_val = evaluation_data.drop(columns=["is_draw"])
        y_val = evaluation_data["is_draw"]
        # Final validation
        logger.info(f"Ensemble evaluation set created with shape: {evaluation_data.shape}")
        logger.info(f"Draw rate: {evaluation_data['is_draw'].mean():.2%}")
        logger.info(f"Train set shape: {X_val.shape}")
        logger.info(f"Test set shape: {y_val.shape}")

        return X_val, y_val
    except FileNotFoundError as e:
        logger.info(f"Data file not found: {str(e)}", error_code=DataProcessingError.FILE_NOT_FOUND)
        raise
    except ValueError as e:
        logger.info(
            f"Data validation error: {str(e)}", error_code=DataProcessingError.INVALID_DATA_TYPE
        )
        raise
    except Exception as e:
        logger.info(
            f"Error creating ensemble evaluation set: {str(e)}",
            error_code=DataProcessingError.FILE_CORRUPTED,
        )
        raise


def import_training_data_ensemble():
    """Import training data for draw predictions."""
    parquet_path = os.path.join(project_root, "data", "api_training_final.parquet")
    data_path = os.path.join(project_root, "data", "api_training_final.xlsx")

    # Check if parquet file exists and is valid
    if os.path.exists(parquet_path):
        try:
            data = pd.read_parquet(parquet_path)
            logger.info(f"Loaded training data from parquet: {parquet_path}")
            if "is_draw" not in data.columns:
                logger.info("is_draw column not found in parquet file, creating target variable")
                data["is_draw"] = (data["match_outcome"] == 2).astype(int)
        except Exception as e:
            logger.info(f"Failed to load parquet file, falling back to Excel: {str(e)}")
            data = pd.read_excel(data_path)
    else:
        data = pd.read_excel(data_path)
        logger.info(f"Loaded training data from Excel: {data_path}")
        # Create target variable
        data["is_draw"] = (data["match_outcome"] == 2).astype(int)
        # Select features and target
        columns_to_drop = [
            "match_outcome",
            "home_goals",
            "away_goals",
            "total_goals",
            "score",
            "Referee",
            "draw",
            "venue_name",
            "Home",
            "Away",
            "away_win",
            "Date",
            "date",
            "referee",
            "league_name",
            "referee_draw_rate",
            "referee_draws",
            "referee_match_count",
            "referee_foul_rate",
            "referee_match_count",
            "referee_encoded",
            "ref_goal_tendency",
            "mid_season_factor",
        ]
        data = data.drop(columns=columns_to_drop, errors="ignore")
        # Convert all numeric-like columns (excluding problematic_cols that have
        # already been handled)
        data = convert_numeric_columns(
            data=data,
            columns=data.columns.tolist(),
            drop_errors=False,
            fill_value=0.0,
            verbose=True,
        )
        # Define integer columns that should remain as int64
        int_columns = [
            "h2h_draws",
            "home_h2h_wins",
            "h2h_matches",
            "Away_points_cum",
            "Home_points_cum",
            "Home_team_matches",
            "Home_draws",
            "venue_encoded",
        ]
        # Convert integer columns back to int64
        for col in int_columns:
            if col in data.columns:
                data[col] = data[col].astype("int64")
        # Export processed data to parquet for efficient storage and retrieval
        create_parquet_files(data, "data/api_training_final.parquet")
        logger.info("Exported processed training data to parquet format")
    # Split into train and test sets
    train_data, test_data = train_test_split(
        data, test_size=0.3, random_state=42, stratify=data["is_draw"]
    )
    X_train = train_data.drop(columns="is_draw", errors="ignore")
    y_train = train_data["is_draw"]
    X_test = test_data.drop(columns="is_draw", errors="ignore")
    y_test = test_data["is_draw"]
    return X_train, y_train, X_test, y_test


def save_data_to_excel(df, output_path, type):
        # Replace NaN/None with empty string
    df = df.fillna('')
    wb = Workbook()
    wb.new_sheet("Sheet1", data=[df.columns.tolist()] + df.values.tolist())
    wb.save(output_path)
    return df


@retry_on_error(max_retries=3, delay=1.0)
def create_prediction_set_ensemble() -> pd.DataFrame:
    """Optimized data loading and preprocessing for predictions."""
    file_path = os.path.join(project_root, "data", "prediction", "new_api_prediction_data.xlsx")
    logger.info(f"Loading prediction data from: {file_path}")
    import_selected_features_ensemble_new("all")
    try:
        # Load data with optimized parameters
        data = pd.read_excel(file_path, engine="openpyxl", dtype={"fixture_id": "int64"})
        # Validate early
        if data.empty:
            logger.info("Empty dataset loaded", error_code=DataProcessingError.EMPTY_DATASET)
            raise ValueError("Dataset is empty")

        logger.info(f"Initial data shape: {data.shape}")
        data_copy = data.copy()
        # Column management
        required_columns = {"fixture_id", "Date", "Home", "Away", "league_name"}
        missing = required_columns - set(data.columns)
        if missing:
            logger.info(
                f"Missing required columns: {missing}",
                error_code=DataProcessingError.MISSING_REQUIRED_COLUMNS,
            )
            raise ValueError(f"Missing columns: {missing}")
        # Date handling
        if "date_encoded" not in data.columns:
            try:
                data["date_encoded"] = (
                    pd.to_datetime(data["Date"], errors="coerce") - pd.Timestamp("2020-08-11")
                ).dt.days
                data["date_encoded"] = data["date_encoded"].fillna(0).astype("int64")
            except Exception as e:
                logger.info(
                    f"Date encoding failed: {str(e)}",
                    error_code=DataProcessingError.FEATURE_CREATION_FAILED,
                )
                raise
        # Optimized column dropping
        cols_to_drop = {"Date", "Home", "Away", "league_name"}
        if cols_to_drop:
            data = data.drop(columns=cols_to_drop)
            logger.info(f"Dropped columns: {cols_to_drop}")
        print(f"data.columns: {data.shape}")
        # Numeric conversion with parallel processing
        data = convert_numeric_columns(data=data, drop_errors=True, fill_value=0.0, verbose=False)
        # Restore columns using vectorized merge
        if "Date" in data_copy.columns:
            restore_cols = data_copy[["fixture_id", "Date", "Home", "Away", "league_name"]]
            # Only merge columns that don't already exist in data
            cols_to_restore = [col for col in restore_cols.columns if col not in data.columns]
            print(f"cols_to_restore: {cols_to_restore}")
            if cols_to_restore:
                data = data.merge(
                    restore_cols[["fixture_id"] + cols_to_restore],
                    on="fixture_id",
                    how="left",
                    validate="one_to_one",  # Ensure no duplicate fixture_ids
                )
        logger.info(f"Final feature shape: {data.shape}")
        return data
    except Exception as e:
        logger.info(f"Critical error: {str(e)}", error_code=DataProcessingError.EMPTY_DATASET)
        raise


@retry_on_error(max_retries=3, delay=1.0)
def get_real_api_scores_from_excel() -> pd.DataFrame:
    """Get all real match scores from an Excel file.
    This function retrieves all actual match results from the API prediction evaluation
    Excel file. It handles data validation and type conversion for match outcomes.
    Returns:
        pd.DataFrame: DataFrame containing match results with columns:
    Raises:
        FileNotFoundError: If the data file cannot be found
        ValueError: If data validation fails
        Exception: For other processing errors
    """
    file_path = os.path.join(project_root, "data", "prediction", "new_api_prediction_eval.xlsx")
    logger.info(f"Loading match results from: {file_path}")
    try:
        # Load Excel file
        df = pd.read_excel(file_path)
        if df.empty:
            logger.info("Loaded dataset is empty", error_code=DataProcessingError.EMPTY_DATASET)
            raise ValueError("Dataset is empty")

        logger.info(f"Successfully loaded data with shape: {df.shape}")
        # Filter rows where match_outcome is not NA
        df = df.dropna(subset=["match_outcome"])
        logger.info(f"Data shape after filtering NA match outcomes: {df.shape}")
        # Convert fixture_id column to integer type
        df["fixture_id"] = pd.to_numeric(df["fixture_id"], errors="coerce").astype("Int64")

        # Create new column for is_draw based on match_outcome
        df["is_draw"] = (df["match_outcome"] == 2).astype(int)

        # Select and rename relevant columns
        results_df = df[
            ["fixture_id", "Home", "Away", "Date", "league_name", "match_outcome", "is_draw"]
        ].rename(
            columns={
                "Home": "home_team",
                "Away": "away_team",
                "Date": "date",
                "league_name": "league",
            }
        )

        logger.info(f"Successfully retrieved {len(results_df)} matches")
        return results_df
    except KeyError as e:
        logger.info(
            f"Missing required columns: {str(e)}",
            error_code=DataProcessingError.MISSING_REQUIRED_COLUMNS,
        )
        raise ValueError(f"Missing required columns: {str(e)}") from e
    except FileNotFoundError:
        logger.info(
            f"Data file not found: {file_path}", error_code=DataProcessingError.FILE_NOT_FOUND
        )
        raise
    except pd.errors.EmptyDataError:
        logger.info(f"Empty data file: {file_path}", error_code=DataProcessingError.EMPTY_DATASET)
        raise
    except Exception as e:
        logger.info(
            f"Error processing match results: {str(e)}",
            error_code=DataProcessingError.FILE_CORRUPTED,
        )
        raise


def update_api_data_new_for_draws():
    """
    Update prediction data for draws by adding advanced goal features, featuretools automated features,
    and saving to api_prediction_data_new.xlsx
    """
    try:
        # Load existing training data
        data_path = "data/prediction/new_api_prediction_data.xlsx"
        data_path_new = "data/prediction/new_api_prediction_data.xlsx"
        data = pd.read_excel(data_path)
        
        logger.info(f"Loaded initial data with shape: {data.shape}")
        
        # Extract year and month from Date column
        data['Date'] = pd.to_datetime(data['Date'])
        data['year'] = data['Date'].dt.year.astype(int)
        data['month'] = data['Date'].dt.month.astype(int)
        logger.info("Added year and month columns from Date")
        
        # Initialize the feature engineer
        feature_engineer = AdvancedGoalFeatureEngineer()
        # Add advanced goal features
        updated_data = feature_engineer.add_goal_features(data)
        updated_data = create_enhanced_categorical_features(updated_data)
        logger.info(f"After advanced goal features: {updated_data.shape}")

        updated_data = calculate_league_draw_k_factors(updated_data)
        logger.info(f"League draw K-factors: {updated_data['k_factor'].unique()}")

        # Add featuretools features
        updated_data = add_featuretools_features(updated_data)

        # Filter data for dates before 2025-04-15 (training data)
        api_training_data = updated_data[updated_data["Date"] < "2025-04-15"]
        # Add is_draw column for training data
        api_training_data = api_training_data.copy()
        api_training_data["is_draw"] = (api_training_data["match_outcome"] == 2).astype(int)
        logger.info("Added is_draw column to training data")
        
        # Filter data for dates after 2025-04-15 where match_outcome is not blank (evaluation data)
        api_prediction_eval = updated_data[
            (updated_data["Date"] >= "2025-04-15") & (updated_data["match_outcome"].notna())
        ]
        
        # Filter data for dates after 2025-04-15 where match_outcome is blank (prediction data)
        api_prediction_data = updated_data[
            (updated_data["Date"] >= "2025-04-15") & (updated_data["match_outcome"].isna())
        ]

        logger.info("Data split summary:")
        logger.info(f"  - Training data shape: {api_training_data.shape}")
        logger.info(f"  - Evaluation data shape: {api_prediction_eval.shape}")
        logger.info(f"  - Prediction data shape: {api_prediction_data.shape}")
        
        # Check if featuretools features are present in the splits
        ft_features_in_data = [col for col in updated_data.columns if col.startswith('ft_')]
        if ft_features_in_data:
            for dataset_name, dataset in [("Training", api_training_data), ("Evaluation", api_prediction_eval), ("Prediction", api_prediction_data)]:
                ft_features_present = [f for f in ft_features_in_data if f in dataset.columns]
                logger.info(f"  - {dataset_name} data has {len(ft_features_present)} featuretools features")
        
        # Concatenate the filtered dataframes for final output
        updated_data = pd.concat([api_prediction_eval, api_prediction_data], ignore_index=True)

        # Export training data
        save_data_to_excel(api_training_data, "data/new_api_training_final.xlsx", "api_training_final")
        create_parquet_files(api_training_data, "data/new_api_training_final.parquet")
        logger.info("api_training_final.xlsx and .parquet updated with featuretools features")

        # Export evaluation data
        save_data_to_excel(
            api_prediction_eval, "data/prediction/new_api_prediction_eval.xlsx", "api_prediction_eval"
        )
        create_parquet_files(api_prediction_eval, "data/prediction/new_api_prediction_eval.parquet")
        logger.info("api_prediction_eval.xlsx and .parquet updated with featuretools features")

        # Export prediction data
        save_data_to_excel(
            api_prediction_data, "data/prediction/new_api_predictions_data.xlsx", "api_prediction_data"
        )
        create_parquet_files(api_prediction_data, "data/prediction/new_api_predictions_data.parquet")
        logger.info("api_predictions_data.xlsx and .parquet updated with featuretools features")
        
        # Save updated data back to Excel
        updated_data.to_excel(data_path_new, index=False)
        
        logger.info("=== Data Update Complete ===")
        logger.info(f"Final data shape: {updated_data.shape}")
        
        # Feature summary
        if ft_features_in_data:
            logger.info(f"Successfully integrated {len(ft_features_in_data)} automated features from featuretools")
            logger.info("Your ensemble models can now use these enhanced features for better performance")
        else:
            logger.info("Data updated with existing feature engineering (featuretools features not added)")
            
    except Exception as e:
        logger.error(f"Error updating training data for draws: {str(e)}")
        import traceback
        traceback.print_exc()


def import_training_data_ensemble_new():
    """Import training data for draw predictions."""
    parquet_path = os.path.join(project_root, "data", "new_api_training_final.parquet")
    data_path = os.path.join(project_root, "data", "new_api_training_final.xlsx")

    # Check if parquet file exists and is valid
    if os.path.exists(parquet_path):
        try:
            data = pd.read_parquet(parquet_path)
            logger.info(f"Loaded training data from parquet: {parquet_path}")
            if "is_draw" not in data.columns:
                logger.info("is_draw column not found in parquet file, creating target variable")
                data["is_draw"] = (data["match_outcome"] == 2).astype(int)
            # Drop rows where home_failed_to_score_away is NA
            # data = data.dropna(subset=['home_failed_to_score_away'])
            # logger.info(f"Dropped rows with NA in home_failed_to_score_away, shape: {data.shape}")
        except Exception as e:
            logger.info(f"Failed to load parquet file, falling back to Excel: {str(e)}")
            data = pd.read_excel(data_path)
    else:
        data = pd.read_excel(data_path)
        logger.info(f"Loaded training data from Excel: {data_path}")
        # Create target variable
        data["is_draw"] = (data["match_outcome"] == 2).astype(int)
        # Select features and target
        columns_to_drop = [
            "match_outcome",
            "home_goals",
            "away_goals",
            "total_goals",
            "score",
            "Referee",
            "draw",
            "venue_name",
            "Home",
            "Away",
            "away_win",
            "Date",
            "date",
            "referee",
            "league_name",
            "referee_draw_rate",
            "referee_draws",
            "referee_match_count",
            "referee_foul_rate",
            "referee_encoded",
            "ref_goal_tendency",
            "mid_season_factor",
        ]
        data = data.drop(columns=columns_to_drop, errors="ignore")
        # Convert all numeric-like columns
        data = convert_numeric_columns(
            data=data,
            columns=data.columns.tolist(),
            drop_errors=False,
            fill_value=0.0,
            verbose=True,
        )
        # Define integer columns that should remain as int64
        int_columns = [
            "h2h_draws",
            "home_h2h_wins",
            "h2h_matches",
            "Away_points_cum",
            "Home_points_cum",
            "Home_team_matches",
            "Home_draws",
            "venue_encoded",
        ]
        # Convert integer columns back to int64
        for col in int_columns:
            if col in data.columns:
                data[col] = data[col].astype("int64")
        # Export processed data to parquet for efficient storage and retrieval
        create_parquet_files(data, "data/new_api_training_final.parquet")
        logger.info("Exported processed training data to parquet format")
    # Split into train and test sets
    train_data, test_data = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data["is_draw"]
    )
    X_train = train_data.drop(columns="is_draw", errors="ignore")
    y_train = train_data["is_draw"]
    X_test = test_data.drop(columns="is_draw", errors="ignore")
    y_test = test_data["is_draw"]
    return X_train, y_train, X_test, y_test


def create_evaluation_set_new() -> pd.DataFrame:
    """Create evaluation set for ensemble training with selected features and evaluation columns.
    This function creates a dataset containing all features from selected_features_ensemble.json
    along with the target variable (is_draw) and an evaluator column for model comparison.
    Returns:
        pd.DataFrame: DataFrame containing:
            - All features from selected_features_ensemble
            - is_draw: Target variable (1 for draw, 0 otherwise)
            - evaluator: Column for model evaluation tracking
    Raises:
        FileNotFoundError: If required data files are not found
        ValueError: If data validation fails
        Exception: For other processing errors
    """
    try:
        # Load training data
        data_path = os.path.join(project_root, "data", "prediction", "new_api_prediction_eval.parquet")
        logger.info(f"Loading training data from: {data_path}")
        data = pd.read_parquet(data_path)
        # Create target variable
        data["is_draw"] = (data["match_outcome"] == 2).astype(int)
        # Select features and target
        columns_to_drop = [
            "match_outcome",
            "home_goals",
            "away_goals",
            "total_goals",
            "score",
            "Referee",
            "draw",
            "venue_name",
            "Home",
            "Away",
            "away_win",
            "Date",
            "date",
            "referee_draw_rate",
            "referee_home_impact",
            "referee_goals_per_game",
            "referee_away_impact",
            "referee_draws",
            "referee_match_count",
            "referee_foul_rate",
            "referee_match_count",
            "referee_encoded",
            "ref_goal_tendency",
            "mid_season_factor",
        ]
        data = data.drop(columns=columns_to_drop, errors="ignore")
        # Import selected features for ensemble models
        selected_features = data.columns.tolist()  # import_selected_features_ensemble()
        all_features = [feature for feature in selected_features if feature != "is_draw"]

        # Validate all features exist in data
        missing_features = [feature for feature in all_features if feature not in data.columns]
        if missing_features:
            logger.info(
                f"Missing required features: {missing_features}",
                error_code=DataProcessingError.MISSING_REQUIRED_COLUMNS,
            )
            raise ValueError(f"Missing required features: {missing_features}")
        # Select features and add evaluator column
        evaluation_data = data[selected_features].copy()
        evaluation_data["is_draw"] = data["is_draw"]
        # Convert numeric columns
        evaluation_data = convert_numeric_columns(
            data=evaluation_data, columns=None, drop_errors=True, fill_value=0.0, verbose=True
        )
        # Split into features and target
        X_val = evaluation_data.drop(columns=["is_draw"])
        y_val = evaluation_data["is_draw"]
        # Final validation
        logger.info(f"Ensemble evaluation set created with shape: {evaluation_data.shape}")
        logger.info(f"Draw rate: {evaluation_data['is_draw'].mean():.2%}")
        logger.info(f"Train set shape: {X_val.shape}")
        logger.info(f"Test set shape: {y_val.shape}")

        return X_val, y_val
    except FileNotFoundError as e:
        logger.info(f"Data file not found: {str(e)}", error_code=DataProcessingError.FILE_NOT_FOUND)
        raise
    except ValueError as e:
        logger.info(
            f"Data validation error: {str(e)}", error_code=DataProcessingError.INVALID_DATA_TYPE
        )
        raise
    except Exception as e:
        logger.info(
            f"Error creating ensemble evaluation set: {str(e)}",
            error_code=DataProcessingError.FILE_CORRUPTED,
        )
        raise


def import_selected_features_ensemble_new(model_type: Optional[str] = None) -> Union[dict, list]:
    """Import selected features for XGBoost, CatBoost, and LightGBM models from JSON file.
    This function loads the pre-selected features for each model type from the
    selected_features_ensemble_new.json file. The features were selected based on
    composite importance scores from feature selection analysis.
    Args:
        model_type (Optional[str]): Specific model type to return features for.
            Options: 'xgb', 'cat', 'lgbm', 'all'. If None, returns all features.
    Returns:
        Union[dict, list]: If model_type is None, returns dictionary containing selected features
            for each model type with keys:
            - 'xgb': List of features for XGBoost
            - 'cat': List of features for CatBoost
            - 'lgbm': List of features for LightGBM
        If model_type is 'all', returns list of features that are common to all models
        If model_type is specified ('xgb', 'cat', 'lgbm'), returns list of features for that model type.
    Raises:
        FileNotFoundError: If the JSON file cannot be found
        JSONDecodeError: If the JSON file is malformed
        ValueError: If invalid model_type is provided
        Exception: For other errors during file loading
    """
    try:
        # Define path to JSON file
        json_path = project_root / "src" / "utils" / "selected_features_ensemble_new.json"
        # Load and parse JSON file
        with open(json_path) as f:
            features = json.load(f)
        # Validate loaded data structure
        if not all(key in features for key in ["xgb", "catboost", "lgbm", "rf", "tabnet", "mlp", "pytorch", "svm"]):
            raise ValueError("JSON file missing required model keys")
        # Return specific model type if requested
        if model_type is not None:
            if model_type == "all":
                common_features = features["all"]
                logger.info("Returning features common to all models")
                return common_features
            elif model_type not in ["xgb", "catboost", "lgbm", "rf", "tabnet", "mlp", "pytorch", "svm", "all"]:
                raise ValueError(
                    f"Invalid model_type: {model_type}. Must be one of: 'xgb', 'catboost', 'lgbm', 'rf', 'tabnet', 'mlp', 'pytorch', 'svm', 'all'"
                )
            logger.info(f"Returning selected features for model type: {model_type}")
            return features[model_type]
        logger.info("Successfully loaded all selected features from JSON file")
        return features
    except FileNotFoundError as e:
        logger.info(
            f"Selected features JSON file not found: {str(e)}",
            error_code=DataProcessingError.FILE_NOT_FOUND,
        )
        raise
    except json.JSONDecodeError as e:
        logger.info(
            f"Invalid JSON format in features file: {str(e)}",
            error_code=DataProcessingError.FILE_CORRUPTED,
        )
        raise
    except Exception as e:
        logger.info(
            f"Error loading selected features: {str(e)}",
            error_code=DataProcessingError.FILE_CORRUPTED,
        )
        raise


def create_enhanced_categorical_features(data):
    """
    Create enhanced categorical features that leverage team, league, and temporal interactions.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        X_eval (pd.DataFrame): Evaluation features
        
    Returns:
        tuple: (X_train_enhanced, X_test_enhanced, X_eval_enhanced)
    """
    logger.info("Creating enhanced categorical features for better model performance")
    
    # Create copies to avoid modifying original data
    data_enh = data.copy()

    # Handle NaN, inf, and -inf values in base encoded columns
    base_encoded_columns = [
        'home_encoded', 'away_encoded', 'venue_encoded', 'league_encoded', 
        'season_encoded', 'referee_encoded','away_league_position','home_league_position'
    ]
    
    for col in base_encoded_columns:
        if col in data_enh.columns:
            # Replace NaN, inf, and -inf with 0
            data_enh[col] = data_enh[col].replace([np.inf, -np.inf], np.nan).fillna(0)
            # Ensure integer type for encoded columns
            data_enh[col] = data_enh[col].astype(int)
    
    logger.info("Replaced NaN, inf, and -inf values in base encoded columns with 0")
    encoder = LabelEncoder()
    
    for df in [data_enh]:
        # 1. Team-League Interaction Features
        if 'home_encoded' in df.columns and 'league_encoded' in df.columns:
            df['home_team_league'] = df['home_encoded'].astype(str) + '_' + df['league_encoded'].astype(str)
            df['away_team_league'] = df['away_encoded'].astype(str) + '_' + df['league_encoded'].astype(str)
            df['home_team_league_encoded'] = encoder.fit_transform(df['home_team_league'])
            df['away_team_league_encoded'] = encoder.fit_transform(df['away_team_league'])
        
        # 2. Team Matchup Feature (captures historical team dynamics)
        if 'home_encoded' in df.columns and 'away_encoded' in df.columns:
            # Create consistent team pair identifier (smaller ID first)
            df['team_pair'] = df.apply(
                lambda row: f"{min(row['home_encoded'], row['away_encoded'])}_{max(row['home_encoded'], row['away_encoded'])}", 
                axis=1
            )
            df['team_pair_encoded'] = encoder.fit_transform(df['team_pair'])
        
        # 3. Season-League Context
        if 'season_encoded' in df.columns and 'league_encoded' in df.columns:
            df['season_league'] = df['season_encoded'].astype(str) + '_' + df['league_encoded'].astype(str)
            df['season_league_encoded'] = encoder.fit_transform(df['season_league'])
        
        # 4. Venue-League Context (captures venue effects within leagues)
        if 'venue_encoded' in df.columns and 'league_encoded' in df.columns:
            df['venue_league'] = df['venue_encoded'].astype(str) + '_' + df['league_encoded'].astype(str)
            df['venue_league_encoded'] = encoder.fit_transform(df['venue_league'])
        
        # 5. Temporal Features for Seasonality
        if 'year' in df.columns and 'month' in df.columns:
            df['year_month'] = df['year'].astype(str) + '_' + df['month'].astype(str)
            df['year_month_encoded'] = encoder.fit_transform(df['year_month'])
        
        # 6. Team Strength Tier (based on actual league positions)
        if 'home_league_position' in df.columns and 'away_league_position' in df.columns:
            # Create team strength tiers based on league positions
            # 1-4: tier 1 (elite), 5-8: tier 2 (strong), 9-12: tier 3 (medium), 13+: tier 4 (weak)
            def get_strength_tier(position):
                if pd.isna(position):
                    return 0  # Unknown/missing position
                elif position <= 4:
                    return 1  # Elite (top 4)
                elif position <= 8:
                    return 2  # Strong (5-8)
                elif position <= 12:
                    return 3  # Medium (9-12)
                else:
                    return 4  # Weak (13+)
            
            df['home_strength_tier'] = df['home_league_position'].apply(get_strength_tier)
            df['away_strength_tier'] = df['away_league_position'].apply(get_strength_tier)
            
            # Create strength difference feature (useful for predicting outcomes)
            df['strength_tier_difference'] = abs(df['home_strength_tier'] - df['away_strength_tier'])
            
            logger.info(f"Home strength tier distribution: {df['home_strength_tier'].value_counts().sort_index().to_dict()}")
            logger.info(f"Away strength tier distribution: {df['away_strength_tier'].value_counts().sort_index().to_dict()}")
            logger.info(f"Strength difference distribution: {df['strength_tier_difference'].value_counts().sort_index().to_dict()}")
        
        # 7. Match Competitiveness Level (average match quality)
        if 'home_strength_tier' in df.columns and 'away_strength_tier' in df.columns:
            # Create competitiveness as average of both team tiers: (home_tier + away_tier) / 2
            df['match_competitiveness'] = (df['home_strength_tier'] + df['away_strength_tier']) / 2
    
    logger.info("Enhanced categorical features created:")
    new_features = ['home_team_league_encoded', 'away_team_league_encoded', 'team_pair_encoded', 'season_league_encoded', 
                    'venue_league_encoded', 'year_month_encoded', 'home_strength_tier', 'away_strength_tier', 
                    'match_competitiveness']
    
    for feature in new_features:
        if feature in data_enh.columns:
            logger.info(f"  - {feature}: {data_enh[feature].nunique()} unique values")
    
    return data_enh

def calculate_league_draw_k_factors(data):
    """Calculate draw-specific K-factors for all leagues and add to data."""
    league_draw_k_factors = {}
    
    unique_leagues = data["league_encoded"].unique()
    for league in unique_leagues:
        league_data = data[data["league_encoded"] == league]
        draw_k_factor = calculate_draw_k_factor(league_data, logger)
        league_draw_k_factors[league] = draw_k_factor
        
        logger.info(f"League {league} - Draw K-factor: {draw_k_factor}")
    
    # Add k_factor column to data based on league_encoded
    data['k_factor'] = data['league_encoded'].map(league_draw_k_factors)
    
    return data

@retry_on_error(max_retries=3, delay=1.0)
def import_training_data_ensemble_date_stratified():
    """
    This function creates a validation strategy where 15% of matches for each date_encoded
    are selected as the validation set, ensuring temporal consistency and preventing
    data leakage while maintaining representative samples across all dates.
    """
    parquet_path = os.path.join(project_root, "data", "new_api_training_final.parquet")
    parquet_path_new = os.path.join(project_root, "data", "prediction", "new_api_prediction_eval.parquet")
    data_path = os.path.join(project_root, "data", "new_api_training_final.xlsx")
    data_path_new = os.path.join(project_root, "data", "prediction", "new_api_prediction_eval.xlsx")

    logger.info("Starting date-stratified training data import")
    
    # Check if parquet file exists and is valid
    if os.path.exists(parquet_path):
        try:
            data = pd.read_parquet(parquet_path)
            data_val = pd.read_parquet(parquet_path_new)
            logger.info(f"Loaded training data from parquet: {parquet_path}")
            if "is_draw" not in data.columns:
                logger.info("is_draw column not found in parquet file, creating target variable")
                data["is_draw"] = (data["match_outcome"] == 2).astype(int)
            if "is_draw" not in data_val.columns:
                logger.info("is_draw column not found in parquet file, creating target variable")
                data_val["is_draw"] = (data_val["match_outcome"] == 2).astype(int)
        except Exception as e:
            logger.info(f"Failed to load parquet file, falling back to Excel: {str(e)}")
            data = pd.read_excel(data_path)
    else:
        data = pd.read_excel(data_path)
        data_val = pd.read_excel(data_path_new)
        logger.info(f"Loaded training data from Excel: {data_path}")
        
        # Create target variable
        data["is_draw"] = (data["match_outcome"] == 2).astype(int)
        
        # Select features and target
        columns_to_drop = [
            "match_outcome",
            "home_goals", 
            "away_goals",
            "total_goals",
            "score",
            "Referee",
            "draw",
            "venue_name",
            "Home",
            "Away", 
            "away_win",
            "Date",
            "date",
            "referee",
            "league_name",
            "referee_draw_rate",
            "referee_draws",
            "referee_match_count",
            "referee_foul_rate",
            "referee_encoded",
            "ref_goal_tendency",
            "mid_season_factor",
        ]
        data = data.drop(columns=columns_to_drop, errors="ignore")
        
        # Convert all numeric-like columns
        data = convert_numeric_columns(
            data=data,
            columns=data.columns.tolist(),
            drop_errors=False,
            fill_value=0.0,
            verbose=True,
        )
        # Define integer columns that should remain as int64
        int_columns = [
            "h2h_draws",
            "home_h2h_wins", 
            "h2h_matches",
            "Away_points_cum",
            "Home_points_cum",
            "Home_team_matches",
            "Home_draws",
            "venue_encoded",
            "date_encoded",
        ]
        
        # Convert integer columns back to int64
        for col in int_columns:
            data[col] = data[col].astype("int64")

    common_columns = list(set(data.columns) & set(data_val.columns))
    data = data[common_columns]
    data_val = data_val[common_columns]
    
    data = pd.concat([data, data_val], ignore_index=True)
    logger.info(f"Merged training and validation data, total shape: {data.shape}")
    # Validate date_encoded column exists
    if "date_encoded" not in data.columns:
        logger.info(
            "date_encoded column not found in data",
            error_code=DataProcessingError.MISSING_REQUIRED_COLUMNS
        )
        raise ValueError("date_encoded column is required for date-stratified splitting")
    
    # Analyze date distribution
    date_counts = data['date_encoded'].value_counts().sort_index()
    logger.info(f"Found {len(date_counts)} unique dates in dataset")
    logger.info(f"Date range: {date_counts.index.min()} to {date_counts.index.max()}")
    logger.info(f"Average matches per date: {date_counts.mean():.1f}")

    # Drop last 3 dates to prevent data leakage
    sorted_dates = sorted(date_counts.index)
    dates_to_drop = sorted_dates[-7:]  # Get the last 7 dates
    
    if dates_to_drop:
        logger.info(f"Dropping last 3 dates to prevent data leakage: {dates_to_drop}")
        data = data[~data['date_encoded'].isin(dates_to_drop)]
        
        # Update date counts after dropping
        date_counts = data['date_encoded'].value_counts().sort_index()
        logger.info(f"After dropping last 3 dates: {len(date_counts)} unique dates remaining")
        logger.info(f"New date range: {date_counts.index.min()} to {date_counts.index.max()}")
        logger.info(f"Dropped {len(data) - data.shape[0] if 'original_shape' in locals() else 'unknown'} samples")
    
    # Check for dates with insufficient samples
    min_samples_per_date = 10  # Minimum to ensure at least 1 validation sample (15% of 7 = 1.05)
    insufficient_dates = date_counts[date_counts < min_samples_per_date]
    if len(insufficient_dates) > 0:
        logger.info(f"Warning: {len(insufficient_dates)} dates have fewer than {min_samples_per_date} samples")
    
    # Perform date-stratified split
    train_indices = []
    val_indices = []
    error_date_count = 0
    for date_encoded in date_counts.index:
        # Get all samples for this date
        date_mask = data['date_encoded'] == date_encoded
        date_data = data[date_mask]
        
        if len(date_data) < min_samples_per_date:
            # If too few samples, put all in training set
            train_indices.extend(date_data.index.tolist())
            continue
        
        # Stratified split within this date to maintain draw rate
        try:
            date_train_idx, date_val_idx = train_test_split(
                date_data.index,
                test_size=0.30,  # 15% for validation
                random_state=42,
                stratify=date_data['is_draw']
            )
            train_indices.extend(date_train_idx.tolist())
            val_indices.extend(date_val_idx.tolist())
            
        except ValueError as e:
            # logger.info(f"Stratification failed for date {date_encoded}")
            error_date_count += 1
            data_train_idx = date_data.index
            train_indices.extend(data_train_idx.tolist())

    logger.info(f"Error date count: {error_date_count}")
    # Create train and validation sets
    train_data = data.loc[train_indices]
    val_data = data.loc[val_indices]
    
    # Split training data further to create test set (20% of training data)
    X_train, X_test, y_train, y_test = train_test_split(
        train_data.drop(columns="is_draw", errors="ignore"),
        train_data["is_draw"],
        test_size=0.2,
        random_state=42,
        stratify=train_data["is_draw"]
    )
    
    # Update train_data to be the reduced training set
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    # Prepare features and targets
    X_train = train_data.drop(columns="is_draw", errors="ignore")
    y_train = train_data["is_draw"]
    X_test = test_data.drop(columns="is_draw", errors="ignore")
    y_test = test_data["is_draw"]
    X_val = val_data.drop(columns="is_draw", errors="ignore")
    y_val = val_data["is_draw"]
    
    # Log split statistics
    logger.info("Date-stratified split completed:")
    logger.info(f"Training set: {len(X_train)} samples ({len(X_train)/len(data)*100:.1f}%)")
    logger.info(f"Validation set: {len(X_val)} samples ({len(X_val)/len(data)*100:.1f}%)")
    logger.info(f"Training draw rate: {y_train.mean():.2%}")
    logger.info(f"Validation draw rate: {y_val.mean():.2%}")
    
    # Final validation
    if len(X_val) == 0:
        logger.info(
            "Validation set is empty after date-stratified split",
            error_code=DataProcessingError.INSUFFICIENT_SAMPLES
        )
        raise ValueError("Validation set is empty - check date distribution and minimum sample requirements")
    
    return X_train, y_train, X_test, y_test, X_val, y_val


def integrate_featuretools_features_to_selection():
    """
    Integrate featuretools-generated features into the existing feature selection JSON files.
    This function loads existing feature selections and adds the best featuretools features
    using intelligent evaluation methods.
    """
    try:
        from src.utils.featuretools_automated_features import SoccerFeaturetoolsEngineer
        
        logger.info("=== Integrating Featuretools Features to Selection ===")
        
        # Load training data to evaluate featuretools features
        logger.info("Loading training data for feature evaluation...")
        X_train, y_train, _, _, _, _ = import_training_data_ensemble_date_stratified()
        
        # Check if featuretools features are present
        ft_features = [col for col in X_train.columns if col.startswith('ft_')]
        
        if not ft_features:
            logger.warning("No featuretools features found in training data")
            logger.info("Run update_api_data_new_for_draws() first to generate featuretools features")
            return
        
        logger.info(f"Found {len(ft_features)} featuretools features in training data")
        
        # Initialize featuretools engineer for feature evaluation
        ft_engineer = SoccerFeaturetoolsEngineer(logger=logger, mlflow_tracking=False)
        
        # Evaluate featuretools features using multiple methods
        logger.info("Evaluating featuretools features using multiple methods...")
        
        # Get top features using different evaluation methods
        top_features_mi = ft_engineer.evaluate_feature_importance(
            X=X_train, 
            y=y_train, 
            new_feature_names=ft_features,
            method='mutual_info',
            top_k=50
        )
        
        top_features_corr = ft_engineer.evaluate_feature_importance(
            X=X_train, 
            y=y_train, 
            new_feature_names=ft_features,
            method='correlation',
            top_k=50
        )
        
        top_features_combined = ft_engineer.evaluate_feature_importance(
            X=X_train, 
            y=y_train, 
            new_feature_names=ft_features,
            method='combined',
            top_k=100
        )
        
        logger.info("Evaluation complete:")
        logger.info(f"  - Top features by mutual info: {len(top_features_mi)}")
        logger.info(f"  - Top features by correlation: {len(top_features_corr)}")
        logger.info(f"  - Top features by combined score: {len(top_features_combined)}")
        
        # Load existing feature selections
        json_path = project_root / "src" / "utils" / "selected_features_ensemble_new.json"
        
        if not json_path.exists():
            logger.error(f"Feature selection file not found: {json_path}")
            return
        
        with open(json_path) as f:
            existing_selections = json.load(f)
        
        logger.info("Loaded existing feature selections")
        
        # Model-specific feature selection strategy for featuretools features
        model_feature_strategies = {
            'xgb': top_features_combined[:25],        # XGBoost handles many features well
            'catboost': top_features_mi[:20],         # CatBoost + mutual info for categorical handling
            'lgbm': top_features_combined[:20],       # LightGBM similar to XGBoost
            'rf': top_features_corr[:15],             # Random Forest + correlation
            'tabnet': top_features_combined[:30],     # TabNet can handle complex interactions
            'mlp': top_features_corr[:15],            # Neural networks + correlation for linear relationships
            'pytorch': top_features_combined[:25],    # PyTorch can handle complex interactions
            'svm': top_features_corr[:10]             # SVM + correlation for linear relationships
        }
        
        # Create updated feature selections
        updated_selections = existing_selections.copy()
        
        # Add strategically selected featuretools features to each model
        for model_name, selected_ft_features in model_feature_strategies.items():
            if model_name in updated_selections:
                # Remove duplicates while preserving order
                new_model_features = []
                existing_model_features = set(updated_selections[model_name])
                
                for feature in selected_ft_features:
                    if feature not in existing_model_features:
                        new_model_features.append(feature)
                        existing_model_features.add(feature)
                
                # Add new featuretools features to existing selection
                updated_selections[model_name].extend(new_model_features)
                logger.info(f"Added {len(new_model_features)} featuretools features to {model_name}")
                
                if new_model_features:
                    logger.info(f"   {model_name} examples: {new_model_features[:3]}")
        
        # Update 'all' selection with all top featuretools features
        all_top_ft_features = list(dict.fromkeys(top_features_combined))  # Remove duplicates
        existing_all_features = set(updated_selections.get('all', []))
        new_all_features = [f for f in all_top_ft_features if f not in existing_all_features]
        
        if 'all' not in updated_selections:
            updated_selections['all'] = []
        updated_selections['all'].extend(new_all_features)
        
        # Save updated selections with metadata
        output_data = {
            'feature_selections': updated_selections,
            'metadata': {
                'last_updated': pd.Timestamp.now().isoformat(),
                'featuretools_integration': {
                    'total_ft_features_available': len(ft_features),
                    'ft_features_selected': len(new_all_features),
                    'selection_methods': ['mutual_info', 'correlation', 'combined'],
                    'model_strategies': {
                        'xgb': 'Combined score (handles many features)',
                        'catboost': 'Mutual information (categorical handling)',
                        'lgbm': 'Combined score (tree-based)',
                        'rf': 'Correlation (ensemble method)',
                        'tabnet': 'Combined score (deep tabular)',
                        'mlp': 'Correlation (linear relationships)',
                        'pytorch': 'Combined score (complex interactions)',
                        'svm': 'Correlation (linear classifier)'
                    }
                }
            }
        }
        
        # Backup original file
        backup_path = json_path.with_suffix('.backup.json')
        with open(backup_path, 'w') as f:
            json.dump(existing_selections, f, indent=2)
        logger.info(f"Backed up original selections to: {backup_path}")
        
        # Save updated selections
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Updated feature selections saved to: {json_path}")
        logger.info("Integration summary:")
        logger.info(f"  - Total featuretools features available: {len(ft_features)}")
        logger.info(f"  - Featuretools features selected for 'all': {len(new_all_features)}")
        logger.info(f"  - Selection efficiency: {len(new_all_features)}/{len(ft_features)} = {len(new_all_features)/len(ft_features)*100:.1f}%")
        
        # Create feature analysis report
        feature_analysis = {
            'temporal_features': [f for f in ft_features if 'ft_temporal_' in f],
            'relational_features': [f for f in ft_features if 'ft_relational_' in f],
            'interaction_features': [f for f in ft_features if f.startswith('ft_') and 'temporal' not in f and 'relational' not in f],
            'selected_temporal': [f for f in new_all_features if 'ft_temporal_' in f],
            'selected_relational': [f for f in new_all_features if 'ft_relational_' in f],
            'selected_interaction': [f for f in new_all_features if f.startswith('ft_') and 'temporal' not in f and 'relational' not in f]
        }
        
        logger.info("Feature type analysis:")
        for feature_type, features in feature_analysis.items():
            if 'selected_' in feature_type:
                original_type = feature_type.replace('selected_', '')
                if original_type in feature_analysis:
                    original_count = len(feature_analysis[original_type])
                    selected_count = len(features)
                    if original_count > 0:
                        logger.info(f"   {feature_type}: {selected_count}/{original_count} ({selected_count/original_count*100:.1f}%)")
        
        logger.info("=== Featuretools Integration Complete ===")
        
    except Exception as e:
        logger.error(f"Error integrating featuretools features: {str(e)}")
        import traceback
        traceback.print_exc()


def add_featuretools_features(updated_data):
    """
    Add featuretools automated features to the dataset.
    
    Args:
        updated_data (pd.DataFrame): Input data to enhance with featuretools features
        
    Returns:
        pd.DataFrame: Data enhanced with featuretools features
    """
    # Import the featuretools engineer
    from src.utils.featuretools_automated_features import SoccerFeaturetoolsEngineer
    
    
    
    # Initialize with central logger
    logger = ExperimentLogger("soccer_features")
    # FEATURETOOLS INTEGRATION - Add automated feature engineering
    logger.info("=== Starting Featuretools Automated Feature Engineering ===")
    engineer = SoccerFeaturetoolsEngineer(
        experiment_logger=logger,
        max_depth=2,
        chunk_size=5000,  # Automatically calculated if None
        memory_limit_gb=4.0
    )

    # Run optimized feature engineering
    enhanced_df, feature_defs = engineer.run_hybrid_feature_engineering(
        df=updated_data,
        include_temporal=True,
        include_relational=True,
        include_interactions=True
    )

    # Get performance statistics
    stats = engineer.get_performance_stats()
    logger.info("Feature engineering stats", extra=stats)
    return enhanced_df

if __name__ == "__main__":
    # update_api_training_data_for_draws()
    # logger.info("Training data updated successfully")
    # update_api_data_for_draws()
    # logger.info("Prediction data updated successfully")
    update_api_data_new_for_draws()
    logger.info("New prediction data updated successfully")

    X_train, y_train, X_test, y_test, X_val, y_val = import_training_data_ensemble_date_stratified()
    logger.info(f"Ensemble evaluation set created with columns: {X_train.shape} and {y_train.shape}")
    logger.info(f"Ensemble evaluation set created with columns: {X_test.shape} and {y_test.shape}")
    logger.info(f"Ensemble evaluation set created with columns: {X_val.shape} and {y_val.shape}")
