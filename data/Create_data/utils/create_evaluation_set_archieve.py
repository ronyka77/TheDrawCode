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
            raise last_error
        return wrapper
    return decorator

# Initialize logger
from utils.logger import ExperimentLogger
logger = ExperimentLogger()

# Add project root to Python path
try:
    project_root = Path(__file__).parent.parent
    if not project_root.exists():
        # Handle network path by using raw string
        project_root = Path(r"\\".join(str(project_root).split("\\")))
    sys.path.append(str(project_root))
    logger.info(f"Project root create_evaluation_set: {project_root}")
except Exception as e:
    logger.error(f"Error setting project root path: {str(e)}", error_code=DataProcessingError.FILE_PERMISSION_ERROR)
    # Fallback to current directory if path resolution fails
    sys.path.append(os.getcwd().parent)
    logger.info(f"Fallback to current directory: {os.getcwd().parent}")

from utils.advanced_goal_features import AdvancedGoalFeatureEngineer
from utils.mlflow_utils import MLFlowConfig, MLFlowManager


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

# GET TRAINING DATA FOR DRAWS
def create_evaluation_sets():
    """
    Load data from an Excel file and create evaluation sets for training.

    Parameters:
    - file_path: str, path to the Excel file.
    - target_column: str, the name of the target column in the dataset.

    Returns:
    - X_eval: pd.DataFrame, features for evaluation.
    - y_eval: pd.Series, target for evaluation.
    """
    file_path = "data/prediction/predictions_eval.xlsx"
    target_column = "is_draw"
    # Load data from the Excel file
    data = pd.read_excel(file_path)
    # Filter data where 'score' is not NA
    data = data.dropna(subset=['score'])
    selected_columns = get_selected_columns()
    data['is_draw'] = data['is_draw'].astype(int)
    # Ensure 'date_encoded' column exists
    if 'date_encoded' not in data.columns:
        # Define the reference date
        reference_date = pd.Timestamp('2020-08-11')

        # Calculate 'date_encoded' as days since the reference date
        data['date_encoded'] = (
            pd.to_datetime(
                data['Datum']) -
            reference_date).dt.days
    # Separate features and target
    X = data[selected_columns]
    y = data[target_column]
    # Start of Selection
    # Replace comma with dot for ALL numeric-like columns
    for col in X.columns:
        if X[col].dtype == 'object':
            try:
                X.loc[:, col] = (
                    X[col].astype(str)
                    .str.strip()  # Remove leading/trailing whitespace
                    .str.strip("'\"")  # Remove quotes
                    .str.replace(' ', '')  # Remove any spaces
                    .str.replace(',', '.')  # Replace comma with dot
                    .astype(float)  # Convert to float
                )
            except (AttributeError, ValueError) as e:
                print(f"Could not convert column {col}: {str(e)}")
                data = data.drop(columns=[col], errors='ignore')
                continue

    return X, y


def import_training_data_draws():
    """Import training data for draw predictions."""
    data_path = "data/training_data.xlsx"
    data = pd.read_excel(data_path)

    # Create target variable
    data['is_draw'] = (data['match_outcome'] == 2).astype(int)

    # Get selected columns
    selected_columns = get_selected_columns()

    # Replace inf and nan values
    data = data.replace([np.inf, -np.inf], 0)
    data = data.fillna(0)

    # Convert all numeric-like columns
    for col in data.columns:
        if not np.issubdtype(data[col].dtype, np.number):
            try:
                # Convert string numbers with either dots or commas
                data[col] = (data[col].astype(str)
                             .str.strip()
                             .str.strip("'\"")
                             .str.replace(' ', '')
                             .str.replace(',', '.')
                             .replace('', '0')  # Replace empty strings with 0
                             .astype(float))
            except (ValueError, AttributeError) as e:
                print(f"Could not convert column {col}: {str(e)}")
                if col in selected_columns:
                    selected_columns.remove(col)
                continue

    # Verify all selected columns are numeric
    for col in selected_columns:
        if data[col].dtype == 'object':
            print(
                f"Warning: Column {col} is still object type after conversion")
            selected_columns.remove(col)

    # Split into train and test sets
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

    # Final verification of data types
    assert all(X_train.dtypes !=
               'object'), "Training data contains object columns"
    assert all(X_test.dtypes != 'object'), "Test data contains object columns"

    return X_train, y_train, X_test, y_test


# UPDATE TRAINING DATA FOR DRAWS
def update_training_data_for_draws():
    """
    Update training data for draws by adding advanced goal features and saving to training_data.xlsx
    """
    try:
        # Load existing training data
        data_path = "data/training_data.xlsx"
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


def update_prediction_data():
    """
    Update prediction data by adding advanced goal features and saving to predictions_eval.xlsx.
    """
    try:
        # Load existing prediction data
        data_path = "data/prediction/prediction_data.csv"
        data = pd.read_csv(data_path)
        eval_path = "data/prediction/predictions_eval.xlsx"
        eval_data = pd.read_excel(eval_path)

        # Initialize the feature engineer
        feature_engineer = AdvancedGoalFeatureEngineer()

        # Add advanced goal features
        updated_data = feature_engineer.add_goal_features(data)
        print(f"Updated prediction data shape: {updated_data.shape}")

        # Save updated data back to Excel
        updated_data.to_csv(data_path, index=False)

        # Merge only new columns from updated_data with eval_data
        new_columns = [
            col for col in eval_data.columns if col not in updated_data.columns and col != 'running_id']
        merged_data = pd.merge(
            updated_data,
            eval_data[['running_id'] + new_columns],
            on='running_id',
            how='left'
        )
        print(f"merged_data shape: {merged_data.shape}")
        merged_data.to_excel(eval_path, index=False)
        merged_data = merged_data.dropna()
        print(f"Updated prediction data saved to {eval_path}")

    except Exception as e:
        print(f"Error updating prediction data: {str(e)}")


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

# CREATE EVALUATION SETS


def get_selected_columns_draws():
    selected_columns = [
        # Very High Impact (>0.01)
        'league_home_draw_rate',          # 0.1009
        'home_draw_rate',                 # 0.0173
        'home_poisson_xG',                # 0.0158
        'possession_balance',             # 0.0127
        'home_corners_rollingaverage',    # 0.0125
        'form_weighted_xg_diff',          # 0.0123
        'home_goal_difference_rollingaverage',  # 0.0112
        'referee_encoded',                # 0.0110

        # High Impact (0.008-0.01)
        'Home_offsides_mean',             # 0.0098
        'position_volatility',            # 0.0095
        'league_draw_rate_composite',     # 0.0093
        'draw_xg_indicator',              # 0.0092
        'Away_fouls_mean',                # 0.0092
        'date_encoded',                   # 0.0089
        'away_encoded',                   # 0.0088

        # Medium-High Impact (0.007-0.008)
        'away_saves_rollingaverage',      # 0.0086
        'home_corners_mean',              # 0.0086
        'away_corners_rollingaverage',    # 0.0085
        'mid_season_factor',              # 0.0083
        'home_shots_on_target_accuracy_rollingaverage',  # 0.0083
        'seasonal_draw_pattern',          # 0.0082
        'home_shot_on_target_rollingaverage',  # 0.0080
        'xg_momentum_similarity',         # 0.0079
        'home_style_compatibility',       # 0.0078
        'away_possession_mean',           # 0.0077
        'home_offensive_sustainability',  # 0.0077
        'Home_passes_mean',               # 0.0075
        'Home_possession_mean',           # 0.0075

        # Medium Impact (0.006-0.007)
        'Away_offsides_mean',             # 0.0074
        'away_crowd_resistance',          # 0.0074
        'league_away_draw_rate',          # 0.0073
        'away_goal_difference_rollingaverage',  # 0.0072
        'away_interceptions_mean',        # 0.0071
        'Home_saves_mean',                # 0.0070
        'away_referee_impact',            # 0.0069
        'Away_saves_mean',                # 0.0069
        'combined_draw_rate',             # 0.0069
        'home_defensive_organization',    # 0.0069
        'attack_xg_equilibrium',          # 0.0068
        'away_team_elo',                  # 0.0068
        'home_xg_momentum',               # 0.0068
        'home_interceptions_mean',        # 0.0068
        'home_team_elo',                  # 0.0067
        'referee_foul_rate',              # 0.0067
        'xg_form_equilibrium',            # 0.0066
        'home_saves_rollingaverage',      # 0.0061
        'Home_fouls_mean'                 # 0.0061
    ]

    return selected_columns


def import_training_data_draws_new():
    """Import training data for draw predictions."""
    data_path = "data/training_data.xlsx"
    data = pd.read_excel(data_path)

    # Create target variable
    data['is_draw'] = (data['match_outcome'] == 2).astype(int)

    # Get selected columns
    selected_columns = get_selected_columns_draws()

    # Replace inf and nan values
    data = data.replace([np.inf, -np.inf], 0)
    data = data.fillna(0)

    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Define integer columns that should remain as int64
    int_columns = [
        'h2h_draws', 'home_h2h_wins', 'h2h_matches', 'Away_points_cum',
        'Home_points_cum', 'Home_team_matches', 'Home_draws', 'venue_encoded'
    ]

    # Convert integer columns back to int64
    for col in int_columns:
        if col in data.columns:
            data[col] = data[col].astype('int64')

    # Verify all selected columns are numeric
    for col in selected_columns:
        if data[col].dtype == 'object':
            print(
                f"Training data: Column {col} is still object type after conversion")
            selected_columns.remove(col)

    # Split into train and test sets
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


def create_evaluation_sets_draws() -> Tuple[pd.DataFrame, pd.Series]:
    """Load data from an Excel file and create evaluation sets for training.

    This function loads match data from the predictions_eval.xlsx file and creates
    evaluation sets for draw prediction training. It handles data preprocessing
    including numeric conversion and missing value handling.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple containing:
            - X (pd.DataFrame): Features for evaluation
            - y (pd.Series): Target variable (1 for draw, 0 for non-draw)

    Example:
        >>> X, y = create_evaluation_sets_draws()
        >>> print(f"Features shape: {X.shape}")
        Features shape: (1000, 120)
        >>> print(f"Draw rate: {y.mean():.2%}")
        Draw rate: 24.50%
    """
    file_path = "data/prediction/predictions_eval.xlsx"
    target_column = "is_draw"
    # Load data from the Excel file
    data = pd.read_excel(file_path)
    # Filter data where 'score' is not NA
    data = data.dropna(subset=['score'])
    selected_columns = get_selected_columns_draws()
    data['is_draw'] = data['is_draw'].astype(int)
    # Ensure 'date_encoded' column exists
    if 'date_encoded' not in data.columns:
        # Define the reference date
        reference_date = pd.Timestamp('2020-08-11')

        # Calculate 'date_encoded' as days since the reference date
        data['date_encoded'] = (
            pd.to_datetime(
                data['Datum']) -
            reference_date).dt.days

    # Separate features and target (make sure you are working on a copy if
    # needed)
    X = data[selected_columns].copy()
    y = data[target_column]

    # Convert all numeric-like columns to numeric types, handling errors by
    # coercing
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

   # Convert all numeric-like columns (excluding problematic_cols that have
   # already been handled)
    for col in data.columns:
        try:
            # Convert string numbers with either dots or commas
            data[col] = (data[col].astype(str)
                         .str.strip()
                         .str.strip("'\"")
                         .str.replace(' ', '')
                         .str.replace(',', '.')
                         .replace('', '0')  # Replace empty strings with 0
                         .astype(float))
        except (ValueError, AttributeError) as e:
            print(f"Evaluation data: Could not convert column {col}: {str(e)}")
            data = data.drop(columns=[col], errors='ignore')
            continue

    # Separate features and target
    X = X[selected_columns]
    y = y

    # Add this before returning
    non_numeric_cols = X.select_dtypes(include=['object']).columns
    if len(non_numeric_cols) > 0:
        print(f"Warning: Found object columns: {list(non_numeric_cols)}")

    return X, y


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
        "venue_draw_rate",
        "home_defensive_activity",
        "away_expected_goals",
        "form_weighted_xg_diff",
        "home_draw_rate",
        "away_days_since_last_draw",
        "referee_encoded",
        "Away_fouls_mean",
        "home_fouls_rollingaverage",
        "Away_offsides_mean",
        "away_poisson_xG",
        "away_corners_mean",
        "away_defense_index",
        "home_corners_mean",
        "mid_season_factor",
        "home_defense_index",
        "Home_offsides_mean",
        "draw_propensity_score",
        "home_shot_on_target_rollingaverage",
        "possession_balance",
        "home_yellow_cards_mean",
        "historical_draw_tendency",
        "Away_saves_mean",
        "away_encoded",
        "venue_capacity",
        "away_possession_mean",
        "away_fouls_rollingaverage",
        "home_founded_year",
        "home_team_elo",
        "avg_league_position",
        "Home_saves_mean",
        "Home_fouls_mean",
        "away_founded_year",
        "away_yellow_cards_mean",
        "home_days_since_last_draw",
        "home_poisson_xG",
        "home_win_rate",
        "league_draw_rate_composite",
        "date_encoded",
        "Home_possession_mean",
        "away_corners_rollingaverage",
        "away_team_elo",
        "home_defense_weakness",
        "away_h2h_weighted",
        "fixture_id",
        "Home_draws",
        "home_corners_rollingaverage",
        "home_form_weighted_xg",
        "league_home_draw_rate",
        "elo_similarity_form_similarity",
        "seasonal_draw_pattern",
        "away_draw_rate",
        "weighted_h2h_draw_rate",
        "home_passing_efficiency",
        "elo_difference",
        "away_offensive_sustainability",
        "defensive_stability",
        "strength_equilibrium",
        "Home_passes_mean",
        "venue_draws",
        "away_shot_on_target_mean",
        "Away_passes_mean",
        "home_red_cards_rollingaverage",
        "away_saves_rollingaverage",
        "away_defense_weakness",
        "referee_goals_per_game",
        "xg_momentum_similarity",
        "home_yellow_cards_rollingaverage",
        "venue_encoded",
        "draw_probability_score",
        "home_saves_rollingaverage",
        "league_season_stage_draw_rate"
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
    
    try:
        # Load data with retry mechanism
        data = pd.read_excel(data_path)
        if data.empty:
            logger.error("Loaded dataset is empty", error_code=DataProcessingError.EMPTY_DATASET)
            raise ValueError("Dataset is empty")
            
        logger.info(f"Successfully loaded data with shape: {data.shape}")

        # Create target variable
        data['is_draw'] = (data['match_outcome'] == 2).astype(int)
        logger.info(f"Created target variable. Draw rate: {data['is_draw'].mean():.2%}")

        # Get selected columns
        selected_columns = get_selected_api_columns_draws()
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

    # Replace inf and nan values
    data = data.replace([np.inf, -np.inf], 0)
    data = data.fillna(0)

    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

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

    # Select features and target
    columns_to_drop = [
        'is_draw',
        'match_outcome',
        'home_goals',
        'away_goals',
        'total_goals',
        'score']
    X_train = train_data.drop(columns=columns_to_drop, errors='ignore')
    y_train = train_data['is_draw']
    X_test = test_data.drop(columns=columns_to_drop, errors='ignore')
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
def create_evaluation_sets_draws_api():
    """Load data from an Excel file and create evaluation sets for training.

    This function loads match data from the API prediction evaluation file and creates
    evaluation sets for draw prediction training. It handles data cleaning, type conversion,
    and feature selection.

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

        # Get selected columns
        selected_columns = get_selected_api_columns_draws()
        missing_columns = [col for col in selected_columns if col not in data.columns]
        if missing_columns:
            logger.error(
                f"Missing required columns: {missing_columns}",
                error_code=DataProcessingError.MISSING_REQUIRED_COLUMNS
            )
            raise ValueError(f"Missing required columns: {missing_columns}")

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
            'Home_points_cum', 'Home_team_matches', 'Home_draws', 'venue_encoded'
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
        selected_columns = get_selected_api_columns_draws()
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

        # Select only the required columns
        X = data[selected_columns]

        # Verify numeric conversion
        object_columns = []
        for col in selected_columns:
            if X[col].dtype == 'object':
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


@retry_on_error(max_retries=3, delay=1.0)
def get_real_scores_from_excel(running_ids: List[str]) -> Dict[str, Dict]:
    """Get real match scores from an Excel file.
    
    This function retrieves actual match results from the predictions evaluation Excel file
    for a given list of running IDs. It returns a dictionary containing match details
    including teams, date, league, score, and draw status.
    
    Args:
        running_ids (List[str]): List of running IDs to retrieve match results for.
        
    Returns:
        Dict[str, Dict]: Dictionary with running_id as key and match details as value.
            Each match details dictionary contains:
            - home_team (str): Name of the home team
            - away_team (str): Name of the away team
            - date (str): Match date
            - league (str): League name
            - score (str): Match score
            - is_draw (bool): Whether the match was a draw
            
    Raises:
        FileNotFoundError: If the data file cannot be found
        ValueError: If data validation fails
        Exception: For other processing errors
    """
    file_path = Path("./data/prediction/predictions_eval.xlsx")
    logger.info(f"Loading match results from: {file_path}")
    
    try:
        # Load Excel file
        df = pd.read_excel(file_path)
        if df.empty:
            logger.error("Loaded dataset is empty", error_code=DataProcessingError.EMPTY_DATASET)
            raise ValueError("Dataset is empty")
            
        logger.info(f"Successfully loaded data with shape: {df.shape}")
        
        # Verify required columns exist
        required_columns = ['running_id', 'home_team', 'away_team', 'date', 'league_y', 'score', 'is_draw']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(
                f"Missing required columns: {missing_columns}",
                error_code=DataProcessingError.MISSING_REQUIRED_COLUMNS
            )
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Filter matches by running_ids
        filtered_df = df[df['running_id'].isin(running_ids)]
        if filtered_df.empty:
            logger.warning(
                f"No matches found for provided running IDs",
                error_code=DataProcessingError.EMPTY_DATASET
            )
            return {}
            
        logger.info(f"Found {len(filtered_df)} matches")
        
        # Convert to dictionary with running_id as key
        results = {}
        for _, match in filtered_df.iterrows():
            try:
                running_id = match['running_id']
                results[running_id] = {
                    'home_team': match['home_team'],
                    'away_team': match['away_team'],
                    'date': match['date'],
                    'league': match['league_y'],
                    'score': match['score'],
                    'is_draw': bool(match['is_draw']),
                }
            except KeyError as e:
                logger.warning(
                    f"Error processing match {running_id}: {str(e)}",
                    error_code=DataProcessingError.INVALID_DATA_TYPE
                )
                continue
        
        if not results:
            logger.warning(
                "No valid matches found in dataset",
                error_code=DataProcessingError.EMPTY_DATASET
            )
        else:
            logger.info(f"Successfully processed {len(results)} matches")
        
        return results
    
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


@retry_on_error(max_retries=3, delay=2.0)
def get_real_scores_from_mongodb(running_ids: List[str]) -> Dict[str, Dict]:
    """Get real match scores from MongoDB.

    This function retrieves actual match results from the MongoDB database
    for a given list of running IDs. It connects to the local MongoDB instance
    and queries the aggregated_data collection.

    Args:
        running_ids (List[str]): List of running IDs to retrieve match results for.

    Returns:
        Dict[str, Dict]: Dictionary with running_id as key and match details as value.
            Each match details dictionary contains:
            - home_team (str): Name of the home team
            - away_team (str): Name of the away team
            - date (str): Match date
            - league (str): League name
            - score (str): Match score in format "X-Y"
            - is_draw (bool): Whether the match was a draw

    Raises:
        ConnectionError: If cannot connect to MongoDB
        ValueError: If data validation fails
        Exception: For other processing errors
    """
    logger.info(f"Retrieving scores for {len(running_ids)} matches from MongoDB")
    
    try:
        # Connect to MongoDB
        client = MongoClient('mongodb://localhost:27017/')
        db = client['football_data']
        collection = db['aggregated_data']
        logger.info("Successfully connected to MongoDB")

        # Query matches by running_ids
        matches = collection.find(
            {"running_id": {"$in": running_ids}},
            {
                "running_id": 1,
                "Home": 1,
                "Away": 1,
                "Score": 1,
                "league": 1,
                "Date": 1,
                "_id": 0
            }
        )

        # Convert to dictionary with running_id as key
        results = {}
        for match in matches:
            running_id = match['running_id']
            try:
                if isinstance(match['Score'], str):
                    home_score, away_score = map(
                        int, match['Score'].replace('–', '-').split('-'))
                else:
                    logger.warning(
                        f"Invalid score format for match {running_id}",
                        error_code=DataProcessingError.INVALID_DATA_TYPE
                    )
                    continue

                is_draw = home_score == away_score
                results[running_id] = {
                    'home_team': match['Home'],
                    'away_team': match['Away'],
                    'date': match['Date'],
                    'league': match['league'],
                    'score': match['Score'],
                    'is_draw': is_draw,
                }
            except (ValueError, KeyError) as e:
                logger.warning(
                    f"Error processing match {running_id}: {str(e)}",
                    error_code=DataProcessingError.INVALID_DATA_TYPE
                )
                continue

        if not results:
            logger.warning(
                "No valid matches found in MongoDB",
                error_code=DataProcessingError.EMPTY_DATASET
            )
        else:
            logger.info(f"Successfully retrieved {len(results)} matches")

        return results

    except ConnectionError as e:
        logger.error(
            f"MongoDB connection error: {str(e)}",
            error_code=DataProcessingError.MONGODB_CONNECTION_ERROR
        )
        raise
    except Exception as e:
        logger.error(
            f"Error retrieving scores from MongoDB: {str(e)}",
            error_code=DataProcessingError.FILE_CORRUPTED
        )
        raise
    finally:
        if 'client' in locals():
            client.close()
            logger.info("MongoDB connection closed")


@retry_on_error(max_retries=3, delay=1.0)
def import_training_data_goals(goal_type: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Import training data for goal prediction models.

    This function loads and preprocesses training data for goal prediction models.
    It handles data cleaning, numeric conversion, and train-test splitting.

    Args:
        goal_type (str): Type of goal prediction target. Valid values are:
            - 'home_goals': Number of goals scored by home team
            - 'away_goals': Number of goals scored by away team
            - 'total_goals': Total goals scored in the match

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: A tuple containing:
            - X_train (pd.DataFrame): Training features
            - y_train (pd.Series): Training targets
            - X_test (pd.DataFrame): Testing features
            - y_test (pd.Series): Testing targets

    Raises:
        FileNotFoundError: If the data file cannot be found
        ValueError: If goal_type is invalid or data validation fails
        Exception: For other processing errors
    """
    data_path = "data/training_data.xlsx"
    logger.info(f"Loading training data for {goal_type} prediction from: {data_path}")
    
    # Validate goal_type
    valid_goal_types = ['home_goals', 'away_goals', 'total_goals']
    if goal_type not in valid_goal_types:
        logger.error(
            f"Invalid goal_type: {goal_type}. Must be one of {valid_goal_types}",
            error_code=DataProcessingError.INVALID_DATA_TYPE
        )
        raise ValueError(f"Invalid goal_type: {goal_type}")

    try:
        # Load data from the Excel file
        data = pd.read_excel(data_path)
        if data.empty:
            logger.error("Loaded dataset is empty", error_code=DataProcessingError.EMPTY_DATASET)
            raise ValueError("Dataset is empty")
            
        logger.info(f"Successfully loaded data with shape: {data.shape}")

        # Convert goal_type column to integer
        try:
            data[goal_type] = data[goal_type].astype(int)
        except Exception as e:
            logger.error(
                f"Failed to convert {goal_type} to integer: {str(e)}",
                error_code=DataProcessingError.NUMERIC_CONVERSION_FAILED
            )
            raise ValueError(f"Invalid {goal_type} values")

        # Define columns to drop
        columns_to_drop = [
            'match_outcome',
            'home_goals',
            'away_goals',
            'total_goals',
            'score',
            'is_draw'
        ]

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

        # Replace inf values with 0
        data = data.replace([np.inf, -np.inf], 0)
        logger.info("Replaced infinite values with 0")

        # Split into train and test sets using 80/20 split
        logger.info("Splitting data into train and test sets")
        train_data, test_data = train_test_split(
            data,
            test_size=0.2,
            random_state=42,
            shuffle=True
        )

        # Prepare features and targets
        X_train = train_data.drop(columns=columns_to_drop, errors='ignore')
        y_train = train_data[goal_type]
        X_test = test_data.drop(columns=columns_to_drop, errors='ignore')
        y_test = test_data[goal_type]

        # Verify numeric conversion
        object_columns = X_train.select_dtypes(include=['object']).columns
        if not object_columns.empty:
            logger.error(
                f"Found {len(object_columns)} non-numeric columns: {list(object_columns)}",
                error_code=DataProcessingError.INVALID_DATA_TYPE
            )
            raise ValueError(f"Non-numeric columns found: {list(object_columns)}")

        # Final validation
        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Test set shape: {X_test.shape}")
        logger.info(f"Training mean {goal_type}: {y_train.mean():.2f}")
        logger.info(f"Test mean {goal_type}: {y_test.mean():.2f}")

        return X_train, y_train, X_test, y_test

    except FileNotFoundError:
        logger.error(
            f"Data file not found: {data_path}",
            error_code=DataProcessingError.FILE_NOT_FOUND
        )
        raise
    except pd.errors.EmptyDataError:
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


if __name__ == "__main__":

    # update_api_training_data_for_draws()
    # print("Training data updated successfully")
    # update_api_prediction_eval_data()
    # print("Prediction data updated successfully")
    update_api_prediction_data()
    print("Prediction data updated successfully")
    # fixtures = create_evaluation_sets_draws_api()
    # print(fixtures)
    # df = get_real_api_scores_from_excel()
    # print(df.shape)


    # sync_mlflow()
