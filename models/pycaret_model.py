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
from mlflow.pyfunc import log_model
from mlflow.sklearn import log_model as sklearn_log_model
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
from utils.logger import ExperimentLogger

# Set up MLflow tracking
experiment_name = "pycaret_soccer_draw"
mlruns_dir = setup_mlflow_tracking(experiment_name)
logger = ExperimentLogger("pycaret_soccer_draw")

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Simplified feature engineering using built-in primitives."""
    # Keep target column for where clauses but exclude from features
    es = ft.EntitySet(id="fixture_id")
    es = es.add_dataframe(
        dataframe=df,
        dataframe_name="fixture_data",
        index="fixture_id"
    )

    # Use only built-in primitives
    trans_primitives = ['absolute', 'log', 'square', 'negate']
    agg_primitives = ['mean', 'std', 'max', 'min']

    # Replace these with your actual column names
    feature_a = 'Away_goal_difference_cum'  # Example feature
    feature_b = 'xg_momentum_similarity'  # Example feature

    # # Check if features exist before creating interaction
    # if feature_a in df.columns and feature_b in df.columns:
    #     seed_features = [ft.Feature(es['fixture_data'][feature_a]) * 
    #                      ft.Feature(es['fixture_data'][feature_b])]
    # else:
    #     seed_features = []

    # Targeted DFS with precision-focused constraints
    feature_matrix, feature_defs = ft.dfs(
        entityset=es,
        target_dataframe_name="fixture_data",
        trans_primitives=trans_primitives,
        agg_primitives=agg_primitives,
        max_depth=3,
        verbose=True,
        ignore_variables={'fixture_data': ['is_draw']},
        drop_contains=['%>', '&'],
        # seed_features=seed_features,
        cutoff_time=df[['fixture_id', 'is_draw']],
        max_features=100,
        primitive_options={
            'square': {'include_columns': df.filter(regex='_avg$|_mean$').columns.tolist()},
            'log': {'skipna': True, 'ignore_columns': df.filter(regex='_count$').columns.tolist()}
        }
    )

    # Precision-focused feature selection
    corr_matrix = feature_matrix.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    feature_matrix = feature_matrix.drop(columns=to_drop)

    # Ensure target alignment without index reset
    feature_matrix['is_draw'] = df.set_index('fixture_id')['is_draw']
    
    return feature_matrix.reset_index()

def train_with_pycaret(feature_matrix: pd.DataFrame):
    """
    Train and optimize a model using PyCaret.
    Args:
        feature_matrix: Feature matrix with engineered features (without target column).
    """
    # Ensure the index is integer type
    feature_matrix.index = feature_matrix.index.astype(int)

    # Check for missing values in the target column
    if feature_matrix['is_draw'].isnull().any():
        logger.warning(f"Found {feature_matrix['is_draw'].isnull().sum()} missing values in the target column 'is_draw'")
        # Drop rows with missing target values
        feature_matrix = feature_matrix.dropna(subset=['is_draw'])
        logger.info(f"Removed {feature_matrix['is_draw'].isnull().sum()} rows with missing target values")

    # Set up PyCaret environment
    clf_setup = setup(
        data=feature_matrix,
        target="is_draw",
        ignore_features=['is_draw'],
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
    
    logger.info(f"features_train: {features_train.shape}")
    logger.info(f"target_train: {target_train.shape}")

    # Perform feature engineering
    logger.info("Performing feature engineering with Featuretools...")
    feature_matrix = feature_engineering(df)

    # Train and optimize model with PyCaret
    logger.info("Training model with PyCaret...")
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
            "feature_engineering_max_depth": 3,  # From ft.dfs
            "feature_engineering_verbose": True,  # From ft.dfs
        })

        # Log selected features
        selected_features = [col for col in feature_matrix.columns.tolist() if col != 'is_draw']
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
            logger.info("Test metrics calculated successfully:", test_metrics)
        except KeyError as e:
            logger.error(f"Error accessing prediction labels: {e}")
        except Exception as e:
            logger.error(f"Error calculating test metrics: {e}")
        mlflow.log_metrics(test_metrics)

        # Evaluate on validation set
        try:
            val_predictions = predict_model(tuned_model, data=features_val)
            val_metrics = {
                "val_precision": precision_score(target_val, val_predictions['prediction_label'], zero_division=0),
                "val_recall": recall_score(target_val, val_predictions['prediction_label'], zero_division=0),
                "val_f1": f1_score(target_val, val_predictions['prediction_label'], zero_division=0),
            }
            logger.info("Validation metrics calculated successfully:", val_metrics)
        except KeyError as e:
            logger.error(f"Error accessing prediction labels: {e}")
        except Exception as e:
            logger.error(f"Error calculating validation metrics: {e}")
        mlflow.log_metrics(val_metrics)

        # Log feature importance
        try:
            feature_importance = tuned_model.feature_importances_
            mlflow.log_dict(
                {feature: importance for feature, importance in zip(selected_features, feature_importance)},
                "feature_importance.json"
            )
        except AttributeError:
            logger.warning("Feature importance not available for this model.")

        # Create input example using only the selected features that were used during training
        input_example = feature_matrix.iloc[:1].copy()
        
        # Convert all columns to numeric types to match training data
        input_example = input_example.apply(pd.to_numeric, errors='coerce')
        
        # Ensure column order matches training data
        input_example = input_example[tuned_model.feature_names_in_]
        
        # Create model signature with proper feature validation
        signature = mlflow.models.infer_signature(
            input_example,
            tuned_model.predict(input_example)
        )
        
        # Ensure the model_global folder exists
        model_global_path = Path("model_global")
        model_global_path.mkdir(parents=True, exist_ok=True)

        # Save the model to the model_global folder
        model_path = model_global_path / "soccer_draw_model_pycaret.pkl"
        save_model(tuned_model, str(model_path))  # Convert Path to string for PyCaret

        # Log the model using mlflow.sklearn
        mlflow.sklearn.log_model(
            tuned_model,
            "soccer_draw_model_pycaret",
            signature=signature,
            input_example=input_example
        )

        logger.info("Training completed. Model and metrics logged to MLflow.")

if __name__ == "__main__":
    main()