import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import mlflow
import sys
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from pymongo import MongoClient
import time
from functools import wraps
from datetime import datetime

# Add project root to Python path
try:
    project_root = Path(__file__).parent.parent
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

from utils.advanced_goal_features import AdvancedGoalFeatureEngineer
from utils.mlflow_utils import MLFlowConfig, MLFlowManager
from utils.logger import ExperimentLogger  

# Initialize logger
logger = ExperimentLogger(log_dir='logs/create_evaluation_set', experiment_name="create_evaluation_set")

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
                logger.error(f"Failed to execute function after multiple retries: {last_error}")
                raise last_error  # Re-raise the last exception after logging
        return wrapper
    return decorator


def convert_numeric_columns(
    data: pd.DataFrame,
    columns: Optional[List[str]] = None,
    drop_errors: bool = True,
    fill_value: float = 0.0,
    verbose: bool = True) -> pd.DataFrame:
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
        verbose (bool): Whether to print conversion errors and warnings

    Returns:
        pd.DataFrame: DataFrame with converted numeric columns

    Example:
        >>> df = pd.DataFrame({'A': ['1.5', '2,5', '3'], 'B': ['a', 'b', 'c']})
        >>> result = convert_numeric_columns(df, columns=['A'])
        >>> print(result['A'].dtype)
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
                    r'[0-9]|inf|-inf',
                    case=False,
                    regex=True
                ).any()
                
                if not has_potential_numbers:
                    if verbose:
                        print(f"Column {col} contains no numeric values")
                    failed_columns.append(col)
                    if drop_errors:
                        columns_to_drop.append(col)
                    continue
                
                # Replace commas with dots (only if comma is used as decimal separator)
                df[col] = df[col].astype(str).apply(
                    lambda x: x.replace(',', '.') if x.count(',') == 1 and x.count('.') == 0 else x
                )
                
                # Convert to string first to handle all cases
                series = (
                    df[col]
                    .apply(lambda x: str(x) if pd.notnull(x) else str(fill_value))
                    .str.strip()  # Remove leading/trailing whitespace
                    .str.strip("'\"")  # Remove quotes
                    .str.replace('[^0-9.eE-]', '', regex=True)  # Keep only numeric chars
                    .apply(lambda x: str(fill_value) if x in ['', 'e', 'e-', 'e+'] else x)  # Handle empty and bare 'e'
                    .apply(lambda x: '1' + x if x.lower().startswith(('e', 'e-', 'e+')) else x)  # Fix sci notation
                    .apply(lambda x: x.replace('-', 'e-', 1) if '-' in x and 'e' not in x.lower() else x)  # Handle negatives
                )
                
                # Try converting to numeric
                numeric_series = pd.to_numeric(series, errors='coerce')
                
                # Check if conversion was successful
                if numeric_series.isna().all():
                    if verbose:
                        print(f"Column {col} contains no valid numeric values")
                    failed_columns.append(col)
                    if drop_errors:
                        columns_to_drop.append(col)
                    continue
                
                # Apply the conversion
                df[col] = numeric_series.replace([np.inf, -np.inf], fill_value).fillna(fill_value)
                
                if verbose and df[col].isna().any():
                    print(f"Warning: Column {col} contains NaN values after conversion")
                    
        except Exception as e:
            if verbose:
                print(f"Error converting column {col}: {str(e)}")
            failed_columns.append(col)
            if drop_errors:
                columns_to_drop.append(col)
    
    # Drop failed columns at the end
    if drop_errors and columns_to_drop:
        df = df.drop(columns=columns_to_drop, errors='ignore')
        if verbose:
            print(f"Dropped columns: {columns_to_drop}")
    
    if verbose and failed_columns:
        print(f"\nConversion summary:")
        print(f"Failed columns: {failed_columns}")
        print(f"Successfully converted: {len(columns) - len(failed_columns)} columns")
        print(f"Failed conversions: {len(failed_columns)} columns")
    
    return df

def create_parquet_files(
    data: pd.DataFrame,
    output_path: str,
    partition_cols: Optional[List[str]] = None
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
        logger.error(
            "Cannot create Parquet files from an empty DataFrame.",
            error_code=DataProcessingError.EMPTY_DATASET
        )
        raise ValueError("DataFrame is empty")


    logger.info(f"Creating Parquet files at: {output_path}")

    try:
        if partition_cols:
            data.to_parquet(
                output_path,
                partition_cols=partition_cols,
                engine="pyarrow",
                index=False
            )
            logger.info(
                f"Successfully created partitioned Parquet files: {partition_cols}"
            )
        else:
            data.to_parquet(
                output_path,
                engine="pyarrow",
                index=False
            )
            logger.info("Successfully created Parquet files without partitioning.")
    except Exception as e:
        logger.error(
            f"Error creating Parquet files: {str(e)}",
            error_code=DataProcessingError.FILE_CORRUPTED
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
        logger.error(
            f"MLflow connection error: {str(e)}",
            error_code=DataProcessingError.MLFLOW_ERROR
        )
        raise
    except ValueError as e:
        logger.error(
            f"Invalid experiment configuration: {str(e)}",
            error_code=DataProcessingError.MLFLOW_ERROR
        )
        raise
    except Exception as e:
        logger.error(
            f"MLflow setup error: {str(e)}",
            error_code=DataProcessingError.MLFLOW_ERROR
        )
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
        logger.error(
            f"Shared storage connection error: {str(e)}",
            error_code=DataProcessingError.MLFLOW_ERROR
        )
        raise
    except PermissionError as e:
        logger.error(
            f"Permission denied accessing shared storage: {str(e)}",
            error_code=DataProcessingError.FILE_PERMISSION_ERROR
        )
        raise
    except Exception as e:
        logger.error(
            f"MLflow sync error: {str(e)}",
            error_code=DataProcessingError.MLFLOW_ERROR
        )
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
        print(updated_data.shape)

        # Save updated data back to Excel
        updated_data.to_excel(data_path, index=False)

    except Exception as e:
        print(f"Error updating training data for draws: {str(e)}")

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
        print(updated_data.shape)
        
        # Filter data for dates before 2024-11-01
        api_training_data = updated_data[updated_data['Date'] < '2024-11-01']

        # Filter data for dates after 2024-11-01 where match_outcome is not blank
        api_prediction_eval = updated_data[
            (updated_data['Date'] >= '2024-11-01') &
            (updated_data['match_outcome'].notna())
        ]

        # Filter data for dates after 2024-11-01 where match_outcome is blank
        api_prediction_data = updated_data[
            (updated_data['Date'] >= '2025-01-15') &
            (updated_data['match_outcome'].isna())
        ]
        
        print(f"api_prediction_data.shape: {api_prediction_data.shape}")
        print(f"api_prediction_eval.shape: {api_prediction_eval.shape}")
        print(f"api_training_data.shape: {api_training_data.shape}")
        # Concatenate the filtered dataframes
        updated_data = pd.concat([api_prediction_eval, api_prediction_data], ignore_index=True)
        
        # Export df_before_2024_11_01 to data/api_training_final.xlsx
        api_training_data.to_excel("data/api_training_final.xlsx", index=False, engine='xlsxwriter')
        print(f"api_training_final.xlsx updated")
        
        # Export df_after_2024_11_01_not_blank to data/prediction/api_predictions_eval.xlsx
        api_prediction_eval.to_excel("data/prediction/api_prediction_eval.xlsx", index=False, engine='xlsxwriter')
        print(f"api_prediction_eval.xlsx updated")
        
        # Export df_after_2024_11_01_blank to data/prediction/api_predictions_data.xlsx
        api_prediction_data.to_excel("data/prediction/api_predictions_data.xlsx", index=False, engine='xlsxwriter')
        print(f"api_predictions_data.xlsx updated")

        # Save updated data back to Excel
        updated_data.to_excel(data_path_new, index=False)

    except Exception as e:
        print(f"Error updating training data for draws: {str(e)}")

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
        print(f"Updated prediction data shape: {updated_data.shape}")

        # Merge with api_prediction_eval but only add columns which are not
        # exists in prediction_data
        merged_data = merge_and_append(updated_data, data_eval)
        # Drop duplicates based on fixture_id column
        merged_data = merged_data.drop_duplicates(
            subset=['fixture_id'], keep='first')
        # Save updated data back to Excel
        merged_data.to_excel(data_path_new, index=False)

    except Exception as e:
        print(f"Error updating prediction data: {str(e)}")


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
    common_columns = [
        col for col in updated_data.columns if col in data_eval.columns]
    new_columns = [
        col for col in updated_data.columns if col not in data_eval.columns]

    # Append rows for common columns
    merged_data = pd.concat(
        [data_eval, updated_data[common_columns]],
        axis=0,
        ignore_index=True
    )

    # Add new columns
    for col in new_columns:
        merged_data[col] = updated_data[col]

    # Print updated shape
    print(f"Updated prediction data shape after merge: {merged_data.shape}")
    return merged_data


#  FOR API
def get_selected_api_columns_draws() -> List[str]:
    """Get selected feature columns for API-based draw prediction model.

    This function returns a curated list of feature columns that have been
    selected based on feature importance analysis and domain knowledge.
    The features are ordered by their importance score from highest to lowest.

    Returns:
        List[str]: List of selected feature column names, ordered by importance.

    Example:
        >>> columns = get_selected_api_columns_draws()
        >>> print(f"Number of features: {len(columns)}")
        Number of features: 57
        >>> print("Top 3 features:")
        >>> for col in columns[:3]:
        ...     print(f"- {col}")
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
        "away_fouls_rollingaverage"
    ]
    return selected_columns


@retry_on_error(max_retries=3, delay=1.0)
def import_training_data_draws_api() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
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
                logger.error("Loaded dataset is empty", error_code=DataProcessingError.EMPTY_DATASET)
                raise ValueError("Dataset is empty")
           
            logger.info(f"Successfully loaded data with shape: {data.shape}")

            # Create target variable
            data['is_draw'] = (data['match_outcome'] == 2).astype(int)
            logger.info(f"Created target variable. Draw rate: {data['is_draw'].mean():.2%}")

            
            missing_columns = [col for col in selected_columns if col not in data.columns]
            if missing_columns:
                logger.error(
                    f"Missing required columns: {missing_columns}",
                    error_code=DataProcessingError.MISSING_REQUIRED_COLUMNS
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
                verbose=True
            )

            # Define and convert integer columns
            int_columns = [
                'h2h_draws', 'home_h2h_wins', 'h2h_matches', 'Away_points_cum',
                'Home_points_cum', 'Home_team_matches', 'Home_draws', 'venue_encoded'
            ]
            
            # Convert integer columns
            for col in int_columns:
                if col in data.columns:
                    try:
                        data[col] = data[col].astype('int64')
                    except Exception as e:
                        logger.warning(
                            f"Failed to convert {col} to integer: {str(e)}",
                            error_code=DataProcessingError.NUMERIC_CONVERSION_FAILED
                        )

            # Verify numeric conversion
            object_columns = []
            for col in selected_columns:
                if data[col].dtype == 'object':
                    object_columns.append(col)
                    logger.warning(
                        f"Column {col} remains as object type after conversion",
                        error_code=DataProcessingError.INVALID_DATA_TYPE
                    )
            
            if object_columns:
                logger.error(
                    f"Found {len(object_columns)} non-numeric columns: {object_columns}",
                    error_code=DataProcessingError.INVALID_DATA_TYPE
                )
                raise ValueError(f"Non-numeric columns found: {object_columns}")
            create_parquet_files(data, "data/api_training_final.parquet")
        
        # Split into train and test sets
        logger.info("Splitting data into train and test sets")
        train_data, test_data = train_test_split(
            data,
            test_size=0.2,
            random_state=42,
            stratify=data['is_draw']
        )

        # Select features and target
        X_train = train_data[selected_columns]
        y_train = train_data['is_draw']
        X_test = test_data[selected_columns]
        y_test = test_data['is_draw']

        # Final validation
        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Test set shape: {X_test.shape}")
        logger.info(f"Training draw rate: {y_train.mean():.2%}")
        logger.info(f"Test draw rate: {y_test.mean():.2%}")

        return X_train, y_train, X_test, y_test

    except FileNotFoundError as e:
        logger.error(
            f"Data file not found: {data_path}",
            error_code=DataProcessingError.FILE_NOT_FOUND
        )
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(
            f"Empty data file: {data_path}",
            error_code=DataProcessingError.EMPTY_DATASET
        )
        raise
    except Exception as e:
        logger.error(
            f"Error processing training data: {str(e)}",
            error_code=DataProcessingError.FILE_CORRUPTED
        )
        raise


def import_feature_select_draws_api():
    """Import training data for draw predictions."""
    data_path = "data/api_training_final.xlsx"
    data = pd.read_excel(data_path)

    # Create target variable
    data['is_draw'] = (data['match_outcome'] == 2).astype(int)

     # Select features and target
    columns_to_drop = [
        'match_outcome',
        'home_goals',
        'away_goals',
        'total_goals',
        'score',
        'Referee', 
        'draw', 
        'venue_name', 
        'Home', 
        'Away', 
        'away_win', 
        'Date',
        'date',
        'referee_draw_rate', 
        'referee_draws', 
        'referee_match_count',
        'referee_foul_rate',
        'referee_match_count',
        'referee_encoded',
        'ref_goal_tendency'
    ]
    data = data.drop(columns=columns_to_drop, errors='ignore')

    # Convert all numeric-like columns (excluding problematic_cols that have
    # already been handled)
    data = convert_numeric_columns(
        data=data,
        columns=data.columns.tolist(),
        drop_errors=False,
        fill_value=0.0,
        verbose=True
    )

    # Define integer columns that should remain as int64
    int_columns = [
        'h2h_draws', 'home_h2h_wins', 'h2h_matches', 'Away_points_cum',
        'Home_points_cum', 'Home_team_matches', 'Home_draws', 'venue_encoded'
    ]

    # Convert integer columns back to int64
    for col in int_columns:
        if col in data.columns:
            data[col] = data[col].astype('int64')

    # Split into train and test sets
    train_data, test_data = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=data['is_draw']
    )

   
    X_train = train_data.drop(columns='is_draw', errors='ignore')
    y_train = train_data['is_draw']
    X_test = test_data.drop(columns='is_draw', errors='ignore')
    y_test = test_data['is_draw']



    # Add verification of dtypes
    print("\nVerifying final dtypes:")
    non_numeric_cols = X_train.select_dtypes(include=['object']).columns
    if len(non_numeric_cols) > 0:
        print(
            f"Warning: Found object columns in X_train: {list(non_numeric_cols)}")

    print("\nInteger columns dtypes:")
    for col in int_columns:
        if col in X_train.columns:
            print(f"{col}: {X_train[col].dtype}")

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
            logger.error("Loaded dataset is empty", error_code=DataProcessingError.EMPTY_DATASET)
            raise ValueError("Dataset is empty")
            
        logger.info(f"Successfully loaded data with shape: {data.shape}")

        # Filter data where match_outcome is not NA
        data = data.dropna(subset=['match_outcome'])
        logger.info(f"Data shape after filtering NA match outcomes: {data.shape}")

        # Replace inf and nan values
        data = data.replace([np.inf, -np.inf], np.nan)
        logger.info("Replaced infinite values with NaN")

        # Get selected columns if needed
        if use_selected_columns:
            selected_columns = get_selected_api_columns_draws()
            missing_columns = [col for col in selected_columns if col not in data.columns]
            if missing_columns:
                logger.error(
                    f"Missing required columns: {missing_columns}",
                    error_code=DataProcessingError.MISSING_REQUIRED_COLUMNS
                )
                raise ValueError(f"Missing required columns: {missing_columns}")
        else:
            # Use all columns except the target and date columns
            selected_columns = [col for col in data.columns if col not in ['match_outcome', 'is_draw', 'Date']]
            logger.info("Using all available columns for evaluation set")

        # Process match outcome and create target variable
        try:
            data['match_outcome'] = data['match_outcome'].astype(int)
            data['is_draw'] = (data['match_outcome'] == 2).astype(int)
            logger.info(f"Created target variable. Draw rate: {data['is_draw'].mean():.2%}")
        except Exception as e:
            logger.error(
                f"Failed to process match outcome: {str(e)}",
                error_code=DataProcessingError.NUMERIC_CONVERSION_FAILED
            )
            raise ValueError("Invalid match outcome values")

        # Ensure date_encoded exists
        if 'date_encoded' not in data.columns:
            try:
                reference_date = pd.Timestamp('2020-08-11')
                data['date_encoded'] = (pd.to_datetime(data['Date']) - reference_date).dt.days
                logger.info("Added date_encoded column")
            except Exception as e:
                logger.error(
                    f"Failed to create date_encoded: {str(e)}",
                    error_code=DataProcessingError.FEATURE_CREATION_FAILED
                )
                raise ValueError("Could not create date_encoded column")

        # Convert integer columns
        int_columns = [
            'h2h_draws', 'home_h2h_wins', 'h2h_matches', 'Away_points_cum',
            'Home_points_cum', 'Home_team_matches', 'Home_draws', 'venue_encoded','date_encoded'
        ]
        for col in int_columns:
            if col in data.columns:
                try:
                    data[col] = data[col].astype('int64')
                except Exception as e:
                    logger.warning(
                        f"Failed to convert {col} to integer: {str(e)}",
                        error_code=DataProcessingError.NUMERIC_CONVERSION_FAILED
                    )

        # Convert numeric columns
        logger.info("Starting numeric conversion")
        data = convert_numeric_columns(
            data=data,
            columns=selected_columns,
            drop_errors=False,
            fill_value=0.0,
            verbose=True
        )
        logger.info(f"Data shape after numeric conversion: {data.shape}")

        # Verify numeric conversion
        object_columns = []
        for col in selected_columns:
            if data[col].dtype == 'object':
                object_columns.append(col)
                logger.warning(
                    f"Column {col} remains as object type after conversion",
                    error_code=DataProcessingError.INVALID_DATA_TYPE
                )
        
        if object_columns:
            logger.error(
                f"Found {len(object_columns)} non-numeric columns: {object_columns}",
                error_code=DataProcessingError.INVALID_DATA_TYPE
            )
            raise ValueError(f"Non-numeric columns found: {object_columns}")

        # Create final feature set and target
        X = data[selected_columns]
        y = data['is_draw']

        # Final validation
        logger.info(f"Final feature set shape: {X.shape}")
        logger.info(f"Draw rate in evaluation set: {y.mean():.2%}")

        return X, y

    except FileNotFoundError as e:
        logger.error(
            f"Data file not found: {file_path}",
            error_code=DataProcessingError.FILE_NOT_FOUND
        )
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(
            f"Empty data file: {file_path}",
            error_code=DataProcessingError.EMPTY_DATASET
        )
        raise
    except Exception as e:
        logger.error(
            f"Error processing evaluation data: {str(e)}",
            error_code=DataProcessingError.FILE_CORRUPTED
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
            logger.error("Loaded dataset is empty", error_code=DataProcessingError.EMPTY_DATASET)
            raise ValueError("Dataset is empty")
            
        logger.info(f"Successfully loaded data with shape: {data.shape}")

        # Get selected columns
        selected_columns = ['fixture_id', 'Home', 'Away', 'Date'] + get_selected_api_columns_draws()
        missing_columns = [col for col in selected_columns if col not in data.columns]
        if missing_columns:
            logger.error(
                f"Missing required columns: {missing_columns}",
                error_code=DataProcessingError.MISSING_REQUIRED_COLUMNS
            )
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Ensure date_encoded exists
        if 'date_encoded' not in data.columns:
            try:
                reference_date = pd.Timestamp('2020-08-11')
                data['date_encoded'] = (pd.to_datetime(data['Datum']) - reference_date).dt.days
                logger.info("Added date_encoded column")
            except Exception as e:
                logger.error(
                    f"Failed to create date_encoded: {str(e)}",
                    error_code=DataProcessingError.FEATURE_CREATION_FAILED
                )
                raise ValueError("Could not create date_encoded column")

        
        data_copy = data.copy()
        # Drop Date, Home, and Away columns from original data
        if 'Date' in data.columns:
            data = data.drop(columns=['Date'])
        if 'Home' in data.columns:
            data = data.drop(columns=['Home'])
        if 'Away' in data.columns:
            data = data.drop(columns=['Away'])
        logger.info("Dropped Date, Home, and Away columns from data")
        # Convert numeric columns
        logger.info("Starting numeric conversion")
        data = convert_numeric_columns(
            data=data,
            columns=None,  # Convert all columns
            drop_errors=True,
            fill_value=0.0,
            verbose=True
        )
        logger.info(f"Data shape after numeric conversion: {data.shape}")

        # Add Date column back from original copy
        if 'Date' not in data.columns and 'Date' in data_copy.columns:
            data['Date'] = data_copy['Date']
            data['Home'] = data_copy['Home']
            data['Away'] = data_copy['Away']
            logger.info("Restored Date, Home, and Away columns from original data")
        # Select only the required columns
        X = data[selected_columns]
        # Final validation
        logger.info(f"Final feature set shape: {X.shape}")
        logger.info("Feature set ready for prediction")

        return X

    except FileNotFoundError as e:
        logger.error(
            f"Data file not found: {file_path}",
            error_code=DataProcessingError.FILE_NOT_FOUND
        )
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(
            f"Empty data file: {file_path}",
            error_code=DataProcessingError.EMPTY_DATASET
        )
        raise
    except Exception as e:
        logger.error(
            f"Error processing prediction data: {str(e)}",
            error_code=DataProcessingError.FILE_CORRUPTED
        )
        raise


# OTHER FUNCTIONS
@retry_on_error(max_retries=3, delay=1.0)
def get_real_api_scores_from_excel(fixture_ids: List[str]) -> pd.DataFrame:
    """Get real match scores from an Excel file.
    
    This function retrieves actual match results from the API prediction evaluation
    Excel file for a given list of fixture IDs. It handles data validation and
    type conversion for match outcomes.
    
    Args:
        fixture_ids (List[str]): List of fixture IDs to retrieve.
        
    Returns:
        pd.DataFrame: DataFrame containing match results with columns:
            - fixture_id: Unique identifier for the match
            - home_team: Name of the home team
            - away_team: Name of the away team
            - date: Match date
            - league: League name
            - match_outcome: Match result code (2 for draw)
            - is_draw: Boolean indicating if match was a draw (1 or 0)
            
    Raises:
        FileNotFoundError: If the data file cannot be found
        ValueError: If data validation fails
        Exception: For other processing errors
    """
    file_path = Path("./data/prediction/api_prediction_eval.xlsx")
    logger.info(f"Loading match results from: {file_path}")
    
    try:
        # Load Excel file
        df = pd.read_excel(file_path)
        if df.empty:
            logger.error("Loaded dataset is empty", error_code=DataProcessingError.EMPTY_DATASET)
            raise ValueError("Dataset is empty")
            
        logger.info(f"Successfully loaded data with shape: {df.shape}")

        # Filter rows where match_outcome is not NA
        df = df.dropna(subset=['match_outcome'])
        logger.info(f"Data shape after filtering NA match outcomes: {df.shape}")

        # Convert fixture_id column and input IDs to integer type
        try:
            df['fixture_id'] = df['fixture_id'].astype(int)
            fixture_ids = [int(fixture_id) for fixture_id in fixture_ids]
        except ValueError as e:
            logger.error(
                f"Invalid fixture ID format: {str(e)}",
                error_code=DataProcessingError.INVALID_DATA_TYPE
            )
            raise ValueError("Invalid fixture ID format")

        # Filter matches by fixture_ids
        filtered_df = df[df['fixture_id'].isin(fixture_ids)]
        if filtered_df.empty:
            logger.warning(
                f"No matches found for provided fixture IDs",
                error_code=DataProcessingError.EMPTY_DATASET
            )
            return pd.DataFrame()

        try:
            # Create new column for is_draw based on match_outcome
            filtered_df['is_draw'] = (filtered_df['match_outcome'] == 2).astype(int)
            
            # Select and rename relevant columns
            results_df = filtered_df[[
                'fixture_id', 'Home', 'Away', 'Date', 'league_name', 'match_outcome', 'is_draw'
            ]].rename(columns={
                'Home': 'home_team',
                'Away': 'away_team',
                'Date': 'date',
                'league_name': 'league'
            })
            
            logger.info(f"Successfully retrieved {len(results_df)} matches")
            return results_df
            
        except KeyError as e:
            logger.error(
                f"Missing required columns: {str(e)}",
                error_code=DataProcessingError.MISSING_REQUIRED_COLUMNS
            )
            raise ValueError(f"Missing required columns: {str(e)}")
            
    except FileNotFoundError:
        logger.error(
            f"Data file not found: {file_path}",
            error_code=DataProcessingError.FILE_NOT_FOUND
        )
        raise
    except pd.errors.EmptyDataError:
        logger.error(
            f"Empty data file: {file_path}",
            error_code=DataProcessingError.EMPTY_DATASET
        )
        raise
    except Exception as e:
        logger.error(
            f"Error processing match results: {str(e)}",
            error_code=DataProcessingError.FILE_CORRUPTED
        )
        raise

def get_selected_columns_from_mlflow_run(run_id: str) -> List[str]:
    """Retrieve selected feature columns from a specific MLflow run.

    This function queries MLflow to get the list of selected features that were
    used in a particular model training run. The features are extracted from
    the model signature in the MLmodel artifact file.

    Args:
        run_id (str): The MLflow run ID to query

    Returns:
        List[str]: List of selected feature column names

    Raises:
        ValueError: If the run ID is invalid or features cannot be retrieved
        FileNotFoundError: If the MLmodel artifact is missing
        Exception: For any other errors during retrieval

    Example:
        >>> columns = get_selected_columns_from_mlflow_run("1234567890abcdef")
        >>> print(f"Retrieved {len(columns)} features from MLflow run")
    """
    try:
        # Initialize MLflow client
        manager = MLFlowManager()
        # Get artifact URI using MLFlowManager
        artifact_uri = manager.get_run_artifact_uri(run_id)
        
        print(f"Artifact URI: {artifact_uri}")
        # Construct the path to MLmodel file, checking for nested model directories
        artifact_path = Path(artifact_uri)
        
        # Find the model directory by checking for MLmodel file in subdirectories
        model_dir = None
        for item in artifact_path.iterdir():
            if item.is_dir() and (item / "MLmodel").exists():
                model_dir = item
                break
                
        # If no model directory found, use the root artifact path
        mlmodel_path = (model_dir / "MLmodel") if model_dir else (artifact_path / "MLmodel")
        
        # Read and parse the MLmodel file
        with open(mlmodel_path, "r") as f:
            mlmodel_content = f.read()
            
        # Extract the signature section
        signature_start = mlmodel_content.find("signature:")
        if signature_start == -1:
            raise ValueError("No signature found in MLmodel file")
            
        # Extract the input schema
        input_schema_start = mlmodel_content.find("inputs:", signature_start)
        if input_schema_start == -1:
            raise ValueError("No input schema found in signature")
            
        # Parse the feature names
        features = []
        for line in mlmodel_content[input_schema_start:].splitlines():
            if "name:" in line:
                feature_name = line.split("name:")[1].strip().strip('"')
                features.append(feature_name)
            elif "}" in line:  # End of input schema
                break
                
        if not features:
            raise ValueError("No features found in input schema")
            
        return features
        
    except mlflow.exceptions.MlflowException as e:
        raise ValueError(f"Invalid MLflow run ID {run_id}: {str(e)}")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"MLmodel artifact not found in run {run_id}"
        )
    except Exception as e:
        raise Exception(
            f"Error retrieving features from MLflow run {run_id}: {str(e)}"
        )


if __name__ == "__main__":

    # update_api_training_data_for_draws()
    # print("Training data updated successfully")

    update_api_data_for_draws()
    print("Prediction data updated successfully")

    # try:
    #     run_id = "bc2a97417edb42d48967315de091d12d"
    #     selected_features = get_selected_columns_from_mlflow_run(run_id)
    #     print(f"Selected features for run {run_id}:")
    #     for feature in selected_features:
    #         print(f"- {feature}")
    # except Exception as e:
    #     print(f"Error retrieving features: {str(e)}")
