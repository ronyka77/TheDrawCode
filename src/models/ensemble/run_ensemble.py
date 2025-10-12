"""
Ensemble Model Runner

Main script for running the ensemble model training and evaluation.
"""

import json
import os
import random
import sys
import warnings
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.models
import mlflow.sklearn
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

# Filter scikit-learn parameter renaming warnings
warnings.filterwarnings("ignore", message=".*force_all_finite.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*ensure_all_finite.*", category=FutureWarning)

# Add project root to Python path
try:
    project_root = Path(__file__).parent.parent.parent
    if not project_root.exists():
        # Handle network path by using raw string
        project_root = Path(r"\\".join(str(project_root).split("\\")))
    sys.path.append(str(project_root))
    print(f"Project root run_ensemble: {project_root}")
except Exception as e:
    print(f"Error setting project root path: {e}")
    # Fallback to current directory if path resolution fails
    current_dir = Path(os.getcwd()).parent
    sys.path.append(str(current_dir))
    print(f"Current directory run_ensemble: {current_dir}")

# Set environment variables for Git
os.environ["GIT_PYTHON_GIT_EXECUTABLE"] = "C:/Program Files/Git/bin/git.exe"
# Set fixed seed and hash seed for determinism
SEED = 19
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
# Restrict parallel threads
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"
os.environ["VECLIB_MAXIMUM_THREADS"] = "16"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Local imports
from src.utils.logger import ExperimentLogger

experiment_name = "ensemble_model_improved"
logger = ExperimentLogger(experiment_name=experiment_name, log_dir="./logs/ensemble_model_improved")
from src.models.ensemble.data_utils import prepare_data
from src.models.ensemble.ensemble_model_20 import EnsembleModel
from src.models.StackedEnsemble.shared.data_loader_new import DataLoader
from src.utils.create_evaluation_set import (
    import_selected_features_ensemble_new,
    setup_mlflow_tracking,
)


def run_ensemble(
    extra_base_model_type: str = "random_forest",
    meta_learner_type: str = "tabnet",
    calibrate: bool = False,
    dynamic_weighting: bool = True,
    target_precision: float = 0.50,
    required_recall: float = 0.25,
    experiment_name: str = "ensemble_model_improved",
    logger: ExperimentLogger = logger,
):
    """
    Main function to run the ensemble model training and evaluation.
    Args:
        extra_base_model_type: Type of fourth base model ('random_forest', 'svm', or 'mlp')
        meta_learner_type: Type of meta-learner ('xgb', 'logistic', or 'mlp')
        calibrate: Whether to calibrate base model probabilities
        dynamic_weighting: Whether to use dynamic weighting for base model probabilities
        target_precision: Target precision for threshold tuning
        required_recall: Minimum required recall for threshold tuning
        experiment_name: Name of the MLflow experiment
    """

    # Set up MLflow tracking
    setup_mlflow_tracking(experiment_name)
    try:
        # Start MLflow run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"ensemble_run_{timestamp}"

        with mlflow.start_run(run_name=run_name):
            # Log run parameters
            mlflow.log_params(
                {
                    "extra_base_model_type": extra_base_model_type,
                    "meta_learner_type": meta_learner_type,
                    "calibrate": calibrate,
                    "dynamic_weighting": dynamic_weighting,
                    "target_precision": target_precision,
                    "required_recall": required_recall,
                }
            )

            logger.info("Starting ensemble model execution...")
            # Initialize variables to None to handle potential loading failures
            X_train, y_train, X_test, y_test, x_val, y_val = None, None, None, None, None, None

            try:
                X_train, y_train, X_test, y_test, x_val, y_val = DataLoader().load_data()
                # Convert all columns to float64 to ensure consistent data types
                X_train = X_train.astype("float64")
                X_test = X_test.astype("float64")
                x_val = x_val.astype("float64")
            except Exception as e:
                logger.error(f"Error loading time-based data: {str(e)}")
                logger.info("Falling back to standard data loading...")
                raise ValueError(f"Data loading failed: {str(e)}") from e

            # Add null checks for unbound variables
            if any(var is None for var in [X_train, y_train, X_test, y_test, x_val, y_val]):
                raise ValueError("Data loading failed - some variables were not initialized")

            # Log dataset sizes
            logger.info(
                f"Dataset sizes - Training: {X_train.shape}, Test: {X_test.shape}, Validation: {x_val.shape}"
            )
            mlflow.log_params(
                {
                    "train_size": len(X_train),
                    "test_size": len(X_test),
                    "val_size": len(x_val),
                    "positive_rate_train": y_train.mean(),
                    "positive_rate_test": y_test.mean(),
                    "positive_rate_val": y_val.mean(),
                }
            )

            # Feature selection
            logger.info("Selecting features...")
            # Add type guard for selected_features parameter
            try:
                features = import_selected_features_ensemble_new(model_type="all")
                if not isinstance(features, list):
                    raise TypeError(f"Expected features to be a list, got {type(features)}")
                if not features:
                    raise ValueError("No features selected")
            except Exception as e:
                logger.error(f"Error selecting features: {str(e)}")
                raise ValueError(f"Feature selection failed: {str(e)}") from e

            # Filter features for all datasets
            x_train_filtered = prepare_data(X_train, features)
            x_test_filtered = prepare_data(X_test, features)
            x_val_filtered = prepare_data(x_val, features)

            # Log the conversion
            mlflow.log_param("data_type_conversion", "all_columns_to_float64")
            logger.info(
                f"Data types after conversion: {x_train_filtered.dtypes.value_counts().to_dict()}"
            )
            # Create ensemble with configuration
            ensemble_model = EnsembleModel(
                logger=logger,
                extra_base_model_type=extra_base_model_type,
                meta_learner_type=meta_learner_type,
                calibrate=calibrate,
                dynamic_weighting=dynamic_weighting,
                target_precision=target_precision,
                required_recall=required_recall,
            )

            # Train the model
            logger.info("Training ensemble model...")
            training_results = ensemble_model.train(
                X_train=x_train_filtered,
                y_train=y_train,
                X_test=x_test_filtered,
                y_test=y_test,
                x_val=x_val_filtered,
                y_val=y_val,
                split_validation=False,  # Don't split again, we already have splits
            )
            log_all_model_params(ensemble_model)

            # Final metrics on validation set
            logger.info("Final metrics on validation set:")
            for metric, value in training_results.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {metric}: {value:.4f}")
            logger.info("Ensemble model execution completed successfully.")
            # Save model with signature to MLflow
            logger.info("Saving ensemble model with signature to MLflow...")
            input_example = x_val_filtered.iloc[0:1].copy()
            best_threshold = training_results["threshold"]
            # Get prediction for output example
            output_example = ensemble_model.predict_proba(input_example)

            # Infer model signature from input and output examples
            signature = mlflow.models.infer_signature(
                input_example, output_example, {"optimal_threshold": best_threshold}
            )

            # Register model with timestamp-based name following project guidelines
            model_name = f"ensemble_{datetime.now().strftime('%Y%m%d_%H%M')}"

            class EnsembleModelWrapper(BaseEstimator, ClassifierMixin):
                def __init__(self, model):
                    self.model = model

                def fit(self, X, y):
                    # This is just a wrapper, actual fitting is done elsewhere
                    return self

                def predict(self, X):
                    # Return class predictions (0 or 1)
                    probas = self.model.predict(X)
                    return probas

                def predict_proba(self, X):
                    # Return probability estimates
                    return self.model.predict_proba(X)

                def get_params(self, deep=True):
                    # Required for scikit-learn compatibility
                    return {"model": self.model}

                def set_params(self, **parameters):
                    # Required for scikit-learn compatibility
                    for parameter, value in parameters.items():
                        setattr(self, parameter, value)
                    return self

            # Wrap the ensemble model in a scikit-learn compatible wrapper
            model_wrapper = EnsembleModelWrapper(ensemble_model)

            # Log model with signature
            mlflow.sklearn.log_model(
                sk_model=model_wrapper,
                artifact_path="ensemble_model",
                signature=signature,
                registered_model_name=model_name,
                pip_requirements=["scikit-learn==1.6.1"],
            )
            logger.info(f"Model saved with signature and registered as: {model_name}")
            # Log the run ID for future reference
            active_run = mlflow.active_run()
            if active_run is not None:
                run_id = active_run.info.run_id
                logger.info(f"MLflow Run ID: {run_id}")
            else:
                logger.warning("No active MLflow run found")
            return ensemble_model

    except Exception as e:
        logger.error(f"Error in ensemble model execution: {str(e)}")
        raise


def get_model_params(model):
    """Attempt to extract parameters from a model using a standard method."""
    try:
        if hasattr(model, "get_params"):
            return model.get_params()
        elif hasattr(model, "get_config"):
            return model.get_config()
        else:
            # Fallback: serialize the model configuration to JSON if possible.
            return json.loads(model.to_json())
    except Exception as e:
        return {"error": str(e)}


def _extract_base_model_params(ensemble_model):
    """Extract parameters from base models."""
    base_models = {
        "model_xgb": "XGBoost",
        "model_tabnet": "TabNet",
        "model_lgb": "LightGBM",
        "model_mlp": "MLP",
        "model_extra": "Extra",
        "model_svm": "SVM"
    }

    params_dict = {}
    for attr_name, model_name in base_models.items():
        if hasattr(ensemble_model, attr_name):
            params_dict[model_name] = ensemble_model.get_model_params(
                getattr(ensemble_model, attr_name)
            )

    return params_dict


def _extract_calibrated_model_params(ensemble_model):
    """Extract parameters from calibrated models."""
    calibrated_models = {
        "model_xgb_calibrated": "XGBoost_calibrated",
        "model_tabnet_calibrated": "TabNet_calibrated",
        "model_lgb_calibrated": "LightGBM_calibrated",
        "model_extra_calibrated": "Extra_calibrated",
        "model_mlp_calibrated": "MLP_calibrated",
        "model_mlp_sklearn_calibrated": "MLP_sklearn_calibrated",
        "model_svm_calibrated": "SVM_calibrated"
    }

    params_dict = {}
    for attr_name, model_name in calibrated_models.items():
        model = getattr(ensemble_model, attr_name, None)
        if model is not None:
            params_dict[model_name] = ensemble_model.get_model_params(model)

    return params_dict


def log_all_model_params(ensemble_model):
    """
    Extracts and logs parameters for each base and extra model in the ensemble.

    Args:
        ensemble_model: Your trained ensemble model instance that contains attributes
                        like model_xgb, model_tabnet, model_lgb, and model_extra.
    """
    params_dict = {}

    # Extract base model parameters
    params_dict.update(_extract_base_model_params(ensemble_model))

    # Extract calibrated model parameters
    params_dict.update(_extract_calibrated_model_params(ensemble_model))

    # Extract meta-learner parameters
    if hasattr(ensemble_model, "meta_learner") and ensemble_model.meta_learner is not None:
        params_dict["MetaLearner"] = ensemble_model.get_model_params(ensemble_model.meta_learner)

    # Log the complete parameters dictionary as a JSON artifact to MLflow.
    mlflow.log_dict(params_dict, "ensemble_model_parameters.json")

    # Log summary parameters for UI comparison
    _log_model_summary_params(params_dict)


def _log_model_summary_params(params_dict):
    """Log first few parameters from each model for UI comparison."""
    for model_name, params in params_dict.items():
        if isinstance(params, dict):
            for key, value in list(params.items())[:3]:
                mlflow.log_param(f"{model_name}_{key}", str(value))


if __name__ == "__main__":
    # Run the ensemble model
    run_ensemble()
