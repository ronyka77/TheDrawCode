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
    sys.path.append(os.getcwd().parent)
    print(f"Current directory run_ensemble: {os.getcwd().parent}")

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
            try:
                logger.info("Loading data...")
                X_train, y_train, X_test, y_test, X_val, y_val = DataLoader().load_data()
                # Convert all columns to float64 to ensure consistent data types
                X_train = X_train.astype("float64")
                X_test = X_test.astype("float64")
                X_val = X_val.astype("float64")
            except Exception as e:
                logger.error(f"Error loading time-based data: {str(e)}")
                logger.info("Falling back to standard data loading...")

            # Log dataset sizes
            logger.info(
                f"Dataset sizes - Training: {X_train.shape}, Test: {X_test.shape}, Validation: {X_val.shape}"
            )
            mlflow.log_params(
                {
                    "train_size": len(X_train),
                    "test_size": len(X_test),
                    "val_size": len(X_val),
                    "positive_rate_train": y_train.mean(),
                    "positive_rate_test": y_test.mean(),
                    "positive_rate_val": y_val.mean(),
                }
            )

            # Feature selection
            logger.info("Selecting features...")
            features = import_selected_features_ensemble_new(model_type="all")

            # Filter features for all datasets
            X_train_filtered = prepare_data(X_train, features)
            X_test_filtered = prepare_data(X_test, features)
            X_val_filtered = prepare_data(X_val, features)

            # Log the conversion
            mlflow.log_param("data_type_conversion", "all_columns_to_float64")
            logger.info(
                f"Data types after conversion: {X_train_filtered.dtypes.value_counts().to_dict()}"
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
                X_train=X_train_filtered,
                y_train=y_train,
                X_test=X_test_filtered,
                y_test=y_test,
                X_val=X_val_filtered,
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
            input_example = X_val_filtered.iloc[0:1].copy()
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
            run_id = mlflow.active_run().info.run_id
            logger.info(f"MLflow Run ID: {run_id}")
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


def log_all_model_params(ensemble_model):
    """
    Extracts and logs parameters for each base and extra model in the ensemble.

    Args:
        ensemble_model: Your trained ensemble model instance that contains attributes
                        like model_xgb, model_tabnet, model_lgb, and model_extra.
    """
    params_dict = {}

    # Log parameters from each base model.
    if hasattr(ensemble_model, "model_xgb"):
        params_dict["XGBoost"] = ensemble_model.get_model_params(ensemble_model.model_xgb)
    if hasattr(ensemble_model, "model_tabnet"):
        params_dict["TabNet"] = ensemble_model.get_model_params(ensemble_model.model_tabnet)
    if hasattr(ensemble_model, "model_lgb"):
        params_dict["LightGBM"] = ensemble_model.get_model_params(ensemble_model.model_lgb)
    if hasattr(ensemble_model, "model_mlp"):
        params_dict["MLP"] = ensemble_model.get_model_params(ensemble_model.model_mlp)
    if hasattr(ensemble_model, "model_extra"):
        params_dict["Extra"] = ensemble_model.get_model_params(ensemble_model.model_extra)
    if hasattr(ensemble_model, "model_svm"):
        params_dict["SVM"] = ensemble_model.get_model_params(ensemble_model.model_svm)

    # Optionally, log calibrated versions if available.
    if (
        hasattr(ensemble_model, "model_xgb_calibrated")
        and ensemble_model.model_xgb_calibrated is not None
    ):
        params_dict["XGBoost_calibrated"] = ensemble_model.get_model_params(
            ensemble_model.model_xgb_calibrated
        )
    if (
        hasattr(ensemble_model, "model_tabnet_calibrated")
        and ensemble_model.model_tabnet_calibrated is not None
    ):
        params_dict["TabNet_calibrated"] = ensemble_model.get_model_params(
            ensemble_model.model_tabnet_calibrated
        )
    if (
        hasattr(ensemble_model, "model_lgb_calibrated")
        and ensemble_model.model_lgb_calibrated is not None
    ):
        params_dict["LightGBM_calibrated"] = ensemble_model.get_model_params(
            ensemble_model.model_lgb_calibrated
        )
    if (
        hasattr(ensemble_model, "model_extra_calibrated")
        and ensemble_model.model_extra_calibrated is not None
    ):
        params_dict["Extra_calibrated"] = ensemble_model.get_model_params(
            ensemble_model.model_extra_calibrated
        )
    if (
        hasattr(ensemble_model, "model_mlp_calibrated")
        and ensemble_model.model_mlp_calibrated is not None
    ):
        params_dict["MLP_calibrated"] = ensemble_model.get_model_params(
            ensemble_model.model_mlp_calibrated
        )
    if (
        hasattr(ensemble_model, "model_mlp_sklearn_calibrated")
        and ensemble_model.model_mlp_sklearn_calibrated is not None
    ):
        params_dict["MLP_sklearn_calibrated"] = ensemble_model.get_model_params(
            ensemble_model.model_mlp_sklearn_calibrated
        )
    if (
        hasattr(ensemble_model, "model_svm_calibrated")
        and ensemble_model.model_svm_calibrated is not None
    ):
        params_dict["SVM_calibrated"] = ensemble_model.get_model_params(
            ensemble_model.model_svm_calibrated
        )
    # Log additional settings (such as meta-learner parameters) if applicable.
    if hasattr(ensemble_model, "meta_learner") and ensemble_model.meta_learner is not None:
        params_dict["MetaLearner"] = ensemble_model.get_model_params(ensemble_model.meta_learner)

    # Log the complete parameters dictionary as a JSON artifact to MLflow.
    mlflow.log_dict(params_dict, "ensemble_model_parameters.json")
    # Optionally, also log some keys using mlflow.log_param for faster comparison in the UI.
    for model_name, params in params_dict.items():
        # For each top-level model, log a summary (e.g., only the first few keys).
        if isinstance(params, dict):
            for key, value in list(params.items())[:3]:
                mlflow.log_param(f"{model_name}_{key}", str(value))


if __name__ == "__main__":
    # Run the ensemble model
    run_ensemble()
