"""
PyCaret-based model training for soccer draw predictions.
Includes feature engineering with Featuretools, model optimization with PyCaret, and MLflow integration.
"""

# Standard library imports
import os
import sys
from pathlib import Path

# Third-party imports
import pandas as pd
import numpy as np
import featuretools as ft
from pycaret.classification import *
import mlflow
from sklearn.metrics import precision_score, recall_score, f1_score

# Add project root to Python path
try:
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
except Exception as e:
    print(f"Error setting project root path: {e}")
    sys.path.append(os.getcwd().parent)

# Local imports
from utils.create_evaluation_set import (
    import_feature_select_draws_api,
    import_training_data_draws_api,
    create_evaluation_sets_draws_api,
    setup_mlflow_tracking,
)

# Set up MLflow tracking
experiment_name = "pycaret_soccer_draw"
mlruns_dir = setup_mlflow_tracking(experiment_name)

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering using Featuretools.
    Args:
        df: Input DataFrame containing raw match data.
    Returns:
        Feature matrix with engineered features.
    """
    # Create EntitySet with proper configuration
    es = ft.EntitySet(id="fixture_id")
    es = es.add_dataframe(
        dataframe=df,
        dataframe_name="fixture_data",
        index="fixture_id"
    )

    # Generate features using Deep Feature Synthesis (DFS)
    feature_matrix, feature_defs = ft.dfs(
        entityset=es,
        target_dataframe_name="fixture_data",  # Use the same entity name as added
        max_depth=2,  # Adjust depth for feature complexity
        verbose=True,
    )

    return feature_matrix

def train_with_pycaret(feature_matrix: pd.DataFrame):
    """
    Train and optimize a model using PyCaret.
    Args:
        feature_matrix: Feature matrix with engineered features.
    """
    # Ensure the index is integer type
    feature_matrix.index = feature_matrix.index.astype(int)

    target_column = "is_draw"
    # Set up PyCaret environment
    clf_setup = setup(
        data=feature_matrix,
        target=target_column,
        session_id=42,
        normalize=True,
        fix_imbalance=True,
        feature_selection=True,
        remove_multicollinearity=True,
        verbose=False  # Changed from silent=True to verbose=False
    )

    # Compare models and select the best one based on precision
    best_model = compare_models(sort="Precision")

    # Tune the best model for precision
    tuned_model = tune_model(best_model, optimize="Precision")

    # Evaluate the tuned model
    evaluate_model(tuned_model)

    # Save the final model
    save_model(tuned_model, "soccer_draw_model_pycaret")

    return tuned_model

def main():
    # Load training data
    features_train, target_train, X_test, y_test = import_training_data_draws_api()
    features_val, target_val = create_evaluation_sets_draws_api()
    df = pd.concat([features_train, target_train], axis=1)

    # Perform feature engineering
    print("Performing feature engineering with Featuretools...")
    feature_matrix = feature_engineering(df)

    # Train and optimize model with PyCaret
    print("Training model with PyCaret...")
    with mlflow.start_run(run_name="pycaret_soccer_draw"):
        # Log dataset info
        mlflow.log_params({
            "train_samples": len(features_train),
            "test_samples": len(X_test),
            "n_features_original": features_train.shape[1],
            "n_features_engineered": feature_matrix.shape[1],
        })

        # Log feature engineering details
        mlflow.log_params({
            "feature_engineering_max_depth": 2,  # From ft.dfs
            "feature_engineering_verbose": True,  # From ft.dfs
        })

        # Log selected features
        selected_features = feature_matrix.columns.tolist()
        mlflow.log_param("selected_features", selected_features)

        # Train the model
        tuned_model = train_with_pycaret(feature_matrix)

        # Log PyCaret setup parameters
        mlflow.log_params({
            "pycaret_normalize": True,
            "pycaret_fix_imbalance": True,
            "pycaret_feature_selection": True,
            "pycaret_remove_multicollinearity": True,
        })

        # Log model hyperparameters
        mlflow.log_params(tuned_model.get_params())

        # Evaluate on test set
        try:
            test_predictions = predict_model(tuned_model, data=X_test)
            test_metrics = {
                "test_precision": precision_score(y_test, test_predictions['prediction_label'], zero_division=0),
                "test_recall": recall_score(y_test, test_predictions['prediction_label'], zero_division=0),
                "test_f1": f1_score(y_test, test_predictions['prediction_label'], zero_division=0),
            }
            print("Test metrics calculated successfully:", test_metrics)
        except KeyError as e:
            print(f"Error accessing prediction labels: {e}")
        except Exception as e:
            print(f"Error calculating test metrics: {e}")
        mlflow.log_metrics(test_metrics)

        # Log feature importance
        try:
            feature_importance = tuned_model.feature_importances_
            mlflow.log_dict(
                {feature: importance for feature, importance in zip(selected_features, feature_importance)},
                "feature_importance.json"
            )
        except AttributeError:
            print("Feature importance not available for this model.")

        # Log the model
        mlflow.log_artifact("soccer_draw_model_pycaret.pkl")
        # mlflow.log_artifact(mlruns_dir)

        print("Training completed. Model and metrics logged to MLflow.")

if __name__ == "__main__":
    main()