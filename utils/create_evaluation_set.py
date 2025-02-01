import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import mlflow
import sys
from typing import Dict, Any, List, Tuple

from pathlib import Path
from pymongo import MongoClient

# Add project root to Python path
try:
    project_root = Path(__file__).parent.parent
    if not project_root.exists():
        # Handle network path by using raw string
        project_root = Path(r"\\".join(str(project_root).split("\\")))
    sys.path.append(str(project_root))
    print(f"Project root create_evaluation_set: {project_root}")
except Exception as e:
    print(f"Error setting project root path: {e}")
    # Fallback to current directory if path resolution fails
    sys.path.append(os.getcwd().parent)
    print(f"Current directory create_evaluation_set: {os.getcwd().parent}")

from utils.advanced_goal_features import AdvancedGoalFeatureEngineer
from utils.mlflow_utils import MLFlowConfig, MLFlowManager

# MLFLOW SETUP


def setup_mlflow_tracking(experiment_name: str) -> str:
    """Configure MLflow tracking for experiment monitoring.

    This function sets up MLflow tracking for experiment monitoring and model versioning.
    It configures the tracking URI, creates or gets the experiment, and ensures proper
    directory structure for MLflow artifacts.

    Args:
        experiment_name (str): Name of the MLflow experiment to create or get.

    Returns:
        str: Path to the mlruns directory where MLflow stores its data.

    Example:
        >>> mlruns_dir = setup_mlflow_tracking("xgboost_api_model")
        >>> print(f"MLflow tracking at: {mlruns_dir}")
        MLflow tracking at: ./mlruns

    Note:
        This function assumes a local MLflow tracking server. For production,
        you may want to configure a remote tracking server using environment
        variables or configuration files.
    """
    mlflow_manager = MLFlowManager()
    mlflow_manager.setup_experiment(experiment_name)
    return mlflow_manager.mlruns_dir


def sync_mlflow() -> None:
    """Synchronize MLflow data with shared storage.

    This function performs a two-way sync of MLflow data:
    1. Backs up local MLflow data to shared storage
    2. Syncs any updates from shared storage back to local

    This ensures consistency across different development environments
    and provides backup of experiment tracking data.

    Example:
        >>> sync_mlflow()
        Backing up MLflow data to shared storage...
        Syncing from shared storage...
        MLflow sync completed successfully.
    """
    mlflow_manager = MLFlowManager()
    mlflow_manager.backup_to_shared()
    mlflow_manager.sync_with_shared()

# GET TRAINING DATA FOR DRAWS


def get_selected_columns():

    selected_columns = ["season_encoded",
                        "league_encoded",
                        "home_encoded",
                        "away_encoded",
                        "venue_encoded",
                        "home_league_position",
                        "away_league_position",
                        "Home_points_cum",
                        "Away_points_cum",
                        "Home_team_matches",
                        "Away_team_matches",
                        "Home_goal_difference_cum",
                        "Away_goal_difference_cum",
                        "Home_draws",
                        "Away_draws",
                        "home_draw_rate",
                        "away_draw_rate",
                        "home_average_points",
                        "away_average_points",
                        "home_win_rate",
                        "away_win_rate",
                        "Home_wins",
                        "Away_wins",
                        "referee_foul_rate",
                        "referee_encoded",
                        "Home_saves_mean",
                        "Away_saves_mean",
                        "Home_fouls_mean",
                        "Away_fouls_mean",
                        "Home_possession_mean",
                        "away_possession_mean",
                        "Home_shot_on_target_mean",
                        "away_shot_on_target_mean",
                        "Home_offsides_mean",
                        "Away_offsides_mean",
                        "Home_passes_mean",
                        "Away_passes_mean",
                        "home_xG_rolling_rollingaverage",
                        "away_xG_rolling_rollingaverage",
                        "home_shot_on_target_rollingaverage",
                        "away_shot_on_target_rollingaverage",
                        "home_goal_rollingaverage",
                        "away_goal_rollingaverage",
                        "home_saves_rollingaverage",
                        "away_saves_rollingaverage",
                        "home_goal_difference_rollingaverage",
                        "away_goal_difference_rollingaverage",
                        "home_shots_on_target_accuracy_rollingaverage",
                        "away_shots_on_target_accuracy_rollingaverage",
                        "home_corners_rollingaverage",
                        "away_corners_rollingaverage",
                        "h2h_matches",
                        "h2h_draws",
                        "home_h2h_wins",
                        "away_h2h_wins",
                        "h2h_avg_goals",
                        "home_avg_attendance",
                        "away_avg_attendance",
                        "home_corners_mean",
                        "away_corners_mean",
                        "home_interceptions_mean",
                        "away_interceptions_mean",
                        "home_h2h_dominance",
                        "away_h2h_dominance",
                        "home_attack_strength",
                        "away_attack_strength",
                        "home_defense_weakness",
                        "away_defense_weakness",
                        "home_poisson_xG",
                        "away_poisson_xG",
                        "home_team_elo",
                        "away_team_elo",
                        "date_encoded",
                        "home_form_momentum",
                        "away_form_momentum",
                        "team_strength_diff",
                        "avg_league_position",
                        "form_difference",
                        "form_similarity",
                        "h2h_draw_rate",
                        "elo_difference",
                        "elo_similarity",
                        "combined_draw_rate",
                        "form_convergence",
                        "position_volatility",
                        "home_form_momentum_home_attack_strength_interaction",
                        "home_form_momentum_away_attack_strength_interaction",
                        "away_form_momentum_home_attack_strength_interaction",
                        "away_form_momentum_away_attack_strength_interaction",
                        "home_attack_strength_home_league_position_interaction",
                        "home_attack_strength_away_league_position_interaction",
                        "away_attack_strength_home_league_position_interaction",
                        "away_attack_strength_away_league_position_interaction",
                        "elo_similarity_form_similarity",
                        "league_draw_rate",
                        "season_progress",
                        "league_position_impact",
                        "league_competitiveness",
                        "xg_form_equilibrium",
                        "home_xg_momentum",
                        "away_xg_momentum",
                        "xg_momentum_similarity",
                        "home_form_weighted_xg",
                        "away_form_weighted_xg",
                        "form_weighted_xg_diff",
                        "home_attack_xg_power",
                        "away_attack_xg_power",
                        "attack_xg_equilibrium",
                        "home_xg_form",
                        "away_xg_form",
                        "xg_form_similarity",
                        "draw_xg_indicator",
                        "strength_equilibrium",
                        "form_stability",
                        "weighted_h2h_draw_rate",
                        "seasonal_draw_pattern",
                        "historical_draw_tendency",
                        "form_convergence_score",
                        "defensive_stability",
                        "position_equilibrium",
                        "goal_pattern_similarity",
                        "xg_equilibrium",
                        "possession_balance",
                        "mid_season_factor",
                        "league_home_draw_rate",
                        "league_away_draw_rate",
                        "league_season_stage_draw_rate",
                        "league_draw_rate_composite",
                        "form_position_interaction",
                        "strength_possession_interaction",
                        "draw_probability_score"]

    return selected_columns


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


def update_api_prediction_eval_data():
    """
    Update prediction data by adding advanced goal features and saving to predictions_eval.xlsx.
    """
    try:
        # Load existing prediction data
        data_path = "data/prediction/api_prediction_final.xlsx"
        data = pd.read_excel(data_path)

        # Initialize the feature engineer
        feature_engineer = AdvancedGoalFeatureEngineer()

        # Add advanced goal features
        updated_data = feature_engineer.add_goal_features(data)
        print(f"Updated prediction data shape: {updated_data.shape}")

        # Save updated data back to Excel
        updated_data.to_excel(data_path, index=False)

    except Exception as e:
        print(f"Error updating prediction data: {str(e)}")


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


def create_evaluation_sets_draws():
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
        # High Impact Features (Venue & Form) - Importance > 0.05
        'venue_draw_rate',                           # 0.0676
        'form_weighted_xg_diff',                     # 0.0572
        'home_draw_rate',                            # 0.0522

        # Medium-High Impact Features (Team Stats) - Importance 0.03-0.05
        'away_days_since_last_draw',                 # 0.0479
        'Away_offsides_mean',                        # 0.0377
        'Away_fouls_mean',                           # 0.0356
        'Home_offsides_mean',                        # 0.0340
        'away_corners_mean',                         # 0.0333
        'draw_propensity_score',                     # 0.0321
        'home_corners_mean',                         # 0.0312

        # Medium Impact Features (Game Context) - Importance 0.02-0.03
        'mid_season_factor',                         # 0.0298
        'home_shot_on_target_rollingaverage',        # 0.0287
        'away_encoded',                              # 0.0276
        'possession_balance',                        # 0.0265
        'home_defense_index',                        # 0.0254
        'away_possession_mean',                      # 0.0243
        'away_defense_index',                        # 0.0232
        'away_poisson_xG',                          # 0.0221

        # Lower-Medium Impact Features - Importance 0.015-0.02
        'historical_draw_tendency',                  # 0.0198
        'venue_capacity',                           # 0.0187
        'ref_goal_tendency',                        # 0.0176
        'Away_saves_mean',                          # 0.0165
        'Home_fouls_mean',                          # 0.0154
        'Home_saves_mean',                          # 0.0153

        # Low-Medium Impact Features - Importance 0.01-0.015
        'home_days_since_last_draw',                # 0.0148
        'home_corners_rollingaverage',              # 0.0147
        'home_team_elo',                            # 0.0146
        'avg_league_position',                      # 0.0145
        'Home_possession_mean',                     # 0.0144
        'league_draw_rate_composite',               # 0.0143

        # Low Impact Features - Importance 0.005-0.01
        'away_h2h_weighted',                        # 0.0098
        'away_draw_rate',                           # 0.0097
        'home_win_rate',                            # 0.0096
        'home_form_weighted_xg',                    # 0.0095
        'league_home_draw_rate',                    # 0.0094
        'seasonal_draw_pattern',                    # 0.0093
        'away_team_elo',                            # 0.0092
        'home_passing_efficiency',                  # 0.0091
        'Home_draws',                               # 0.0090
        'home_poisson_xG',                          # 0.0089
        'elo_similarity_form_similarity',           # 0.0088
        'away_corners_rollingaverage',              # 0.0087
        'venue_encoded',                            # 0.0086
        'defensive_stability',                      # 0.0085
        'home_defense_weakness',                    # 0.0084
        'away_shot_on_target_mean',                # 0.0083
        'away_saves_rollingaverage',               # 0.0082
        'elo_difference',                          # 0.0081
        'weighted_h2h_draw_rate',                  # 0.0080
        'strength_equilibrium',                    # 0.0079
        'venue_draws',                             # 0.0078
        'xg_momentum_similarity',                  # 0.0077
        'draw_probability_score',                  # 0.0076
        'Away_passes_mean',                        # 0.0075
        'Home_passes_mean',                        # 0.0074
        'venue_match_count'                        # 0.0073
    ]
    return selected_columns


def import_training_data_draws_api(
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
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

    Example:
        >>> X_train, y_train, X_test, y_test = import_training_data_draws_api()
        >>> print(f"Training samples: {len(X_train)}")
        Training samples: 8000
        >>> print(f"Draw rate in training: {y_train.mean():.2%}")
        Draw rate in training: 24.50%
    """
    data_path = "data/api_training_final.xlsx"
    data = pd.read_excel(data_path)

    # Create target variable
    data['is_draw'] = (data['match_outcome'] == 2).astype(int)

    # Get selected columns
    selected_columns = get_selected_api_columns_draws()

    # Replace inf and nan values
    data = data.replace([np.inf, -np.inf], np.nan)

    # Convert all columns to numeric
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Handle numeric conversion with detailed error handling
    for col in data.columns:
        try:
            if col in data.columns:
                data[col] = (
                    data[col]
                    .apply(lambda x: str(x) if pd.notnull(x) else '0')
                    .str.strip()
                    .str.replace('[^0-9.eE-]', '', regex=True)
                    .apply(lambda x: '0' if x in ['e', 'e-', 'e+'] else x)
                    .apply(lambda x: '1' + x if x.lower().startswith(('e', 'e-', 'e+')) else x)
                    .apply(lambda x: x.replace('-', 'e-', 1) if '-' in x and 'e' not in x.lower() else x)
                    .replace('', '0')
                    .pipe(pd.to_numeric, errors='coerce')
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0)
                )
        except Exception as e:
            print(f"Error converting column {col}: {str(e)}")
            continue

    # Define and convert integer columns
    int_columns = [
        'h2h_draws', 'home_h2h_wins', 'h2h_matches', 'Away_points_cum',
        'Home_points_cum', 'Home_team_matches', 'Home_draws', 'venue_encoded'
    ]

    # Convert integer columns back to int64
    for col in int_columns:
        if col in data.columns:
            data[col] = data[col].astype('int64')

    # Verify numeric conversion
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

    # Verify data types
    non_numeric_cols = X_train.select_dtypes(include=['object']).columns
    if len(non_numeric_cols) > 0:
        print(
            f"Warning: Found object columns in X_train: {list(non_numeric_cols)}")

    # Log integer column types
    for col in int_columns:
        if col in X_train.columns:
            print(f"Integer column {col}: {X_train[col].dtype}")

    return X_train, y_train, X_test, y_test


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
    for col in data.columns:
        try:
            if col in data.columns:
                # print(f"Converting column {col} (type: {data[col].dtype})")
                data[col] = (
                    data[col]
                    .apply(lambda x: str(x) if pd.notnull(x) else '0')
                    .str.strip()
                    .str.replace('[^0-9.eE-]', '', regex=True)
                    .apply(lambda x: '0' if x in ['e', 'e-', 'e+'] else x)
                    .apply(lambda x: '1' + x if x.lower().startswith(('e', 'e-', 'e+')) else x)
                    .apply(lambda x: x.replace('-', 'e-', 1) if '-' in x and 'e' not in x.lower() else x)
                    .replace('', '0')
                    .pipe(pd.to_numeric, errors='coerce')
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0)
                )
        except Exception as e:
            print(f"Error converting column {col}: {str(e)}")
            continue

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


def create_evaluation_sets_draws_api():
    """
    Load data from an Excel file and create evaluation sets for training.

    Parameters:
    - file_path: str, path to the Excel file.
    - target_column: str, the name of the target column in the dataset.

    Returns:
    - X_eval: pd.DataFrame, features for evaluation.
    - y_eval: pd.Series, target for evaluation.
    """
    file_path = "data/prediction/api_prediction_eval.xlsx"
    target_column = "is_draw"
    # Load data from the Excel file
    data = pd.read_excel(file_path)
    # Filter data where 'score' is not NA
    data = data.dropna(subset=['match_outcome'])
    data = data.replace([np.inf, -np.inf], np.nan)
    # data = data.fillna(0)
    selected_columns = get_selected_api_columns_draws()
    print(f"shape of data: {data.shape}")
    data['match_outcome'] = data['match_outcome'].astype(int)
    data['is_draw'] = (data['match_outcome'] == 2).astype(int)
    # print(data.head())
    # Ensure 'date_encoded' column exists
    if 'date_encoded' not in data.columns:
        # Define the reference date
        reference_date = pd.Timestamp('2020-08-11')
        # Calculate 'date_encoded' as days since the reference date
        data['date_encoded'] = (
            pd.to_datetime(
                data['Date']) -
            reference_date).dt.days

    # Verify all selected columns are present in data
    missing_columns = [
        col for col in selected_columns if col not in data.columns]
    if len(missing_columns) > 0:
        print(f"Warning: Missing required columns: {missing_columns}")
        # Add missing columns with default value 0
        for col in missing_columns:
            data[col] = 0
        print("Added missing columns with default value 0")

    # Define integer columns that should remain as int64
    int_columns = [
        'h2h_draws', 'home_h2h_wins', 'h2h_matches', 'Away_points_cum',
        'Home_points_cum', 'Home_team_matches', 'Home_draws', 'venue_encoded'
    ]

    # Convert integer columns back to int64
    for col in int_columns:
        if col in data.columns:
            data[col] = data[col].astype('int64')
    # Convert all numeric-like columns
    for col in selected_columns:
        try:
            if col in data.columns:
                # print(f"Converting column {col} (type: {data[col].dtype})")
                data[col] = (
                    data[col]
                    .apply(lambda x: str(x) if pd.notnull(x) else '0')
                    .str.strip()
                    .str.replace('[^0-9.eE-]', '', regex=True)
                    .apply(lambda x: '0' if x in ['e', 'e-', 'e+'] else x)
                    .apply(lambda x: '1' + x if x.lower().startswith(('e', 'e-', 'e+')) else x)
                    .apply(lambda x: x.replace('-', 'e-', 1) if '-' in x and 'e' not in x.lower() else x)
                    .replace('', '0')
                    .pipe(pd.to_numeric, errors='coerce')
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0)
                )
        except Exception as e:
            print(f"Error converting column {col}: {str(e)}")
            continue
    print(f"Converted data: {data.shape}")
    # Convert all numeric-like columns to numeric types, handling errors by coercing
    for col in data.columns:
        try:
            if col in data:  # Verify column exists
                data[col] = pd.to_numeric(data[col], errors='coerce')
        except Exception as e:
            print(f"Error converting column {col}: {str(e)}")
            data[col] = data[col].astype(float)
            continue

    # Add this before returning
    non_numeric_cols = data.select_dtypes(include=['object']).columns
    if len(non_numeric_cols) > 0:
        print(f"Warning: Found object columns: {list(non_numeric_cols)}")

    X = data[selected_columns]
    y = data['is_draw']

    return X, y


def create_prediction_set_api():
    """
    Load data from an Excel file and create evaluation sets for training.

    Parameters:
    - file_path: str, path to the Excel file.
    - target_column: str, the name of the target column in the dataset.

    Returns:
    - X_eval: pd.DataFrame, features for evaluation.
    - y_eval: pd.Series, target for evaluation.
    """
    file_path = "data/prediction/api_prediction_data_new.xlsx"
    # Load data from the Excel file
    data = pd.read_excel(file_path)
    # Filter data where 'score' is not NA

    selected_columns = get_selected_api_columns_draws()

    # Ensure 'date_encoded' column exists
    if 'date_encoded' not in data.columns:
        # Define the reference date
        reference_date = pd.Timestamp('2020-08-11')

        # Calculate 'date_encoded' as days since the reference date
        data['date_encoded'] = (
            pd.to_datetime(
                data['Datum']) -
            reference_date).dt.days

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
    X = data[selected_columns]

    # Add this before returning
    non_numeric_cols = X.select_dtypes(include=['object']).columns
    if len(non_numeric_cols) > 0:
        print(f"Warning: Found object columns: {list(non_numeric_cols)}")

    return X


# OTHER FUNCTIONS
def get_real_api_scores_from_excel(fixture_ids: List[str]) -> pd.DataFrame:
    """Get real match scores from an Excel file.
    
    Args:
        fixture_ids (List[str]): List of fixture IDs to retrieve.
        
    Returns:
        pd.DataFrame: DataFrame containing match results.
    """
    try:
        # Set up paths
        data_path = Path("./data/prediction/api_prediction_eval.xlsx")
        # Load Excel file
        df = pd.read_excel(data_path)
        # Filter rows where match_outcome is not NA
        df = df.dropna(subset=['match_outcome'])
        # Convert fixture_id column to integer type
        df['fixture_id'] = df['fixture_id'].astype(int)
        # Convert fixture_ids to integers
        fixture_ids = [int(fixture_id) for fixture_id in fixture_ids]
        # Filter matches by running_ids
        filtered_df = df[df['fixture_id'].isin(fixture_ids)]
        
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
            
            print(f"Retrieved {len(results_df)} real scores")
            return results_df
            
        except Exception as e:
            print(f"Error reading Excel file: {str(e)}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error reading Excel file: {str(e)}")
        return pd.DataFrame()


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
            
    Example:
        >>> running_ids = ['2023_EPL_001', '2023_EPL_002']
        >>> results = get_real_scores_from_excel(running_ids)
        >>> print(results['2023_EPL_001'])
        {
            'home_team': 'Arsenal',
            'away_team': 'Chelsea',
            'date': '2023-09-01',
            'league': 'Premier League',
            'score': '2-2',
            'is_draw': True
        }
    """
    try:
        # Set up paths
        data_path = Path("./data/prediction/predictions_eval.xlsx")
        
        # Load Excel file
        df = pd.read_excel(data_path)
        
        # Filter matches by running_ids
        filtered_df = df[df['running_id'].isin(running_ids)]
        
        # Convert to dictionary with running_id as key
        results = {}
        for _, match in filtered_df.iterrows():
            running_id = match['running_id']   
            results[running_id] = {
                'home_team': match['home_team'],
                'away_team': match['away_team'],
                'date': match['date'],
                'league': match['league_y'],
                'score': match['score'],
                'is_draw': match['is_draw'],
            }
        
        return results
    
    except Exception as e:
        print(f"Error reading Excel file: {str(e)}")
        return {}


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

    Example:
        >>> running_ids = ['2023_EPL_001', '2023_EPL_002']
        >>> results = get_real_scores_from_mongodb(running_ids)
        >>> print(results['2023_EPL_001'])
        {
            'home_team': 'Arsenal',
            'away_team': 'Chelsea',
            'date': '2023-09-01',
            'league': 'Premier League',
            'score': '2-2',
            'is_draw': True
        }
    """
    try:
        # Connect to MongoDB
        client = MongoClient('mongodb://localhost:27017/')
        db = client['football_data']
        collection = db['aggregated_data']

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
                        int, match['Score'].replace(
                            '–', '-').split('-'))
                else:
                    raise ValueError("Score is not a string")
            except ValueError as e:
                print(
                    f"Error parsing score for match {match['running_id']}: {str(e)}")
                home_score, away_score = None, None

            is_draw = None if home_score is None or away_score is None else home_score == away_score
            results[running_id] = {
                'home_team': match['Home'],
                'away_team': match['Away'],
                'date': match['Date'],
                'league': match['league'],
                'score': match['Score'],
                'is_draw': is_draw,
            }

        return results

    except Exception as e:
        print(f"Error connecting to MongoDB: {str(e)}")
        return {}


def create_evaluation_sets_goals(
        goal_type: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Create evaluation sets for goal prediction models.

    This function loads match data from an Excel file and creates evaluation sets
    for training goal prediction models. It handles data preprocessing including
    numeric conversion and missing value handling.

    Args:
        goal_type (str): Type of goal prediction target. Valid values are:
            - 'home_goals': Number of goals scored by home team
            - 'away_goals': Number of goals scored by away team
            - 'total_goals': Total goals scored in the match

    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple containing:
            - X_eval (pd.DataFrame): Features for evaluation
            - y_eval (pd.Series): Target variable (goal counts)

    Example:
        >>> X_eval, y_eval = create_evaluation_sets_goals('home_goals')
        >>> print(f"Features shape: {X_eval.shape}")
        Features shape: (1000, 120)
        >>> print(f"Target mean: {y_eval.mean():.2f}")
        Target mean: 1.47
    """
    file_path = "data/prediction/predictions_eval.xlsx"

    # Load data from the Excel file
    data = pd.read_excel(file_path)

    # Filter data where goal_type is not NA
    data = data.dropna(subset=[goal_type])

    # Convert numeric-like columns
    for col in data.columns:
        if data[col].dtype == 'object':
            try:
                data[col] = (data[col].astype(str)
                             .str.strip()
                             .str.strip("'\"")
                             .str.replace(' ', '')
                             .str.replace(',', '.')
                             .astype(float))
            except (AttributeError, ValueError) as e:
                print(f"Could not convert column {col}: {str(e)}")
                data = data.drop(columns=[col], errors='ignore')
                continue

    # Keep all columns for X except target-related ones
    columns_to_drop = [
        'match_outcome',
        'home_goals',
        'away_goals',
        'total_goals',
        'score',
        'is_draw']
    X = data.drop(columns=columns_to_drop, errors='ignore')
    y = data[goal_type]

    return X, y


def import_training_data_goals(
        goal_type: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
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

    Example:
        >>> X_train, y_train, X_test, y_test = import_training_data_goals('total_goals')
        >>> print(f"Training samples: {len(X_train)}")
        Training samples: 8000
        >>> print(f"Test samples: {len(X_test)}")
        Test samples: 2000
    """
    data_path = "data/training_data.xlsx"
    data = pd.read_excel(data_path)
    data[goal_type] = data[goal_type].astype(int)

    # Replace comma with dot for ALL numeric-like columns
    for col in data.columns:
        if data[col].dtype == 'object':  # Check if column is string type
            try:
                # Replace comma with dot and convert to float
                data[col] = (data[col].astype(str)
                             .str.strip()  # Remove leading/trailing whitespace
                             .str.strip("'\"")  # Remove quotes
                             .str.replace(' ', '')  # Remove any spaces
                             .str.replace(',', '.')  # Replace comma with dot
                             .astype(float))  # Convert to float

            except (AttributeError, ValueError) as e:
                print(f"Could not convert column {col}: {str(e)}")
                data = data.drop(columns=[col], errors='ignore')
                continue

    # Replace inf values with 0
    data = data.replace([np.inf, -np.inf], 0)

    # Split into train and test sets using 80/20 split
    train_data, test_data = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    # Prepare features and targets
    columns_to_drop = [
        'match_outcome',
        'home_goals',
        'away_goals',
        'total_goals',
        'score',
        'is_draw']
    X_train = train_data.drop(columns=columns_to_drop, errors='ignore')
    y_train = train_data[goal_type]
    X_test = test_data.drop(columns=columns_to_drop, errors='ignore')
    y_test = test_data[goal_type]

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":

    # update_api_training_data_for_draws()
    # print("Training data updated successfully")
    # update_api_prediction_eval_data()
    # print("Prediction data updated successfully")
    # update_api_prediction_data()
    # print("Prediction data updated successfully")
    fixtures = create_evaluation_sets_draws_api()
    print(fixtures)
    # df = get_real_api_scores_from_excel()
    # print(df.shape)


    # sync_mlflow()
