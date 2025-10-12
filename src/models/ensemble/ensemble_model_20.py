"""EnsembleModel Class - Version 0404

This file implements the ensemble model including:
- Base models: XGBoost, TabNet, LightGBM, RandomForest, and MLP (sklearn)

Author: Updated by AI assistant
"""

import json
import os
import pickle  # Needed for loading scaler
import random

import mlflow  # Ensure mlflow is imported for download_artifacts
import mlflow.artifacts  # Explicit import for artifacts
import mlflow.lightgbm
import mlflow.pyfunc
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd  # Add pandas import
import torch
from sklearn.base import BaseEstimator, ClassifierMixin

from src.models.ensemble.data_utils import prepare_data
from src.models.ensemble.diagnostics import analyze_prediction_errors, explain_predictions
from src.models.ensemble.evaluation import evaluate_model
from src.models.ensemble.meta_features_20 import (
    create_meta_dataframe,
    create_meta_features_optimized,
)
from src.models.ensemble.thresholds import tune_threshold_for_precision_optimized
from src.models.ensemble.training import hypertune_meta_learner, initialize_meta_learner
from src.models.ensemble.weights_0414 import compute_precision_focused_weights_optimized

# Import shared utility functions
from src.utils.logger import ExperimentLogger

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
# PyTorch specific reproducibility settings
torch.manual_seed(SEED)

# String constants for repeated messages
XGB_PREDICT_PROBA_FALLBACK_MSG = "XGBoost predict_proba not available, using pyfunc predict_proba"
PYTORCH_SCALER_PATH = "scaler/scaler_pytorch.pkl"


class EnsembleModel(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        logger=None,
        calibrate=False,
        calibration_method="sigmoid",
        individual_thresholding=False,
        meta_learner_type="xgb",
        dynamic_weighting=True,
        extra_base_model_type="random_forest",  # Removed, RF is now standard
        target_precision=0.50,
        required_recall=0.20,
    ):
        # Use provided logger or create a new one specific to this version
        self.logger = logger or ExperimentLogger(
            experiment_name="ensemble_model_0410", log_dir="./logs/ensemble_model_0410"
        )
        self.required_recall = required_recall  # For meta-learner

        self.target_precision = target_precision  # For dynamic weights

        # --- MLflow Run IDs for Base Models ---
        self.xgb_run_id = "a8f84a8a82ef44b7a64bc72d5d797a82"
        self.lgb_run_id = "ea8b1bb86aaa4faf9bf8c3d2d08145da"
        self.tabnet_run_id = "431df2695dd9431f8c088e15b675e8a3"
        self.extra_run_id = "625d925be2634d10b7da7f6a42576405"
        self.mlp_run_id = "ebb5dfa8d32c4409a7edb21f3fad09d0"
        self.pytorch_run_id = "bb59b9589aef4638a1e0d3aa406c7da7"
        self.svm_run_id = "8fded66e23fd422ab8d7db66687508dd"
        self.fnn_run_id = "c32a835ceac048e9af04334ebb3c3d57"

        # Minimum recalls for dynamic weighting (order: xgb, tabnet, lgb, rf, mlp, pytorch, svm)
        self.min_recalls = [0.25, 0.25, 0.30, 0.30, 0.25, 0.20, 0.25, 0.20]  # Added SVM recall

        # Meta-learner settings
        self.meta_learner_type = meta_learner_type
        self.optimal_threshold = 0.5  # Will be tuned
        self.individual_thresholding = individual_thresholding  # Unused currently?
        self.calibrate = calibrate  # Unused currently?
        self.calibration_method = calibration_method  # Unused currently?
        self.dynamic_weighting = dynamic_weighting
        num_models = 8  # Updated number of models (added SVM)
        if self.dynamic_weighting:
            # Adjusted for 7 base models
            self.dynamic_weights = {
                "xgb": 1 / num_models,
                "lgb": 1 / num_models,
                "tabnet": 1 / num_models,
                "extra": 1 / num_models,
                "mlp": 1 / num_models,
                "pytorch": 1 / num_models,  # Added pytorch
                "svm": 1 / num_models,  # Added SVM
                "fnn": 1 / num_models,  # Added FNN
            }

        # Placeholder attributes for models and features - will be populated by load_models
        self.meta_learner = None
        self.model_xgb = None
        self.model_lgb = None
        self.model_tabnet = None
        self.model_extra = None
        self.model_mlp = None
        self.model_mlp_scaler = None
        self.model_pytorch = None  # Added pytorch model placeholder
        self.model_pytorch_scaler = None  # Added pytorch scaler placeholder
        self.model_svm = None  # Added SVM model placeholder
        self.model_svm_scaler = None  # Added SVM scaler placeholder
        self.model_fnn = None  # Added FNN model placeholder
        self.model_fnn_scaler = None  # Added FNN scaler placeholder

        self.xgb_features = []
        self.lgb_features = []
        self.tabnet_features = []
        self.extra_features = []
        self.mlp_features = []
        self.pytorch_features = []  # Added pytorch feature list placeholder
        self.svm_features = []  # Added SVM feature list placeholder
        self.fnn_features = []  # Added FNN feature list placeholder

        # Load models on initialization
        self.load_models_from_mlflow()

    def train(
        self,
        X_train,
        y_train,
        x_val=None,
        y_val=None,
        X_test=None,
        y_test=None,
        split_validation=True,
        val_size=0.2,
    ) -> dict:
        self.logger.info("Starting ensemble model 0404 training...")
        # Data preparation
        # x_train_prepared = prepare_data(X_train, X_train.columns)
        # Add type assertions for DataFrame operations
        assert isinstance(x_val, pd.DataFrame), f"x_val must be a DataFrame, got {type(x_val)}"
        assert isinstance(X_test, pd.DataFrame), f"X_test must be a DataFrame, got {type(X_test)}"
        x_val_prepared = prepare_data(x_val, list(x_val.columns))
        x_test_prepared = prepare_data(X_test, list(X_test.columns))

        # Prepare feature subsets for each model
        self.logger.info("Preparing feature subsets for base models...")
        try:
            x_val_xgb = x_val_prepared[self.xgb_features]
        except KeyError:
            self.logger.warning(
                f"XGB features not found in x_val_prepared. Using all features. {self.xgb_features}"
            )
            x_val_xgb = x_val_prepared
        x_val_tabnet = x_val_prepared[self.tabnet_features]
        x_val_lgb = x_val_prepared[self.lgb_features]
        x_val_extra = x_val_prepared[self.extra_features]
        x_val_mlp = x_val_prepared[self.mlp_features]  # Prepare MLP features
        x_val_pytorch = x_val_prepared[self.pytorch_features]  # Added PyTorch features
        x_val_svm = x_val_prepared[self.svm_features]  # Added SVM features
        x_val_fnn = x_val_prepared[self.fnn_features]  # Added FNN features

        x_test_xgb = x_test_prepared[self.xgb_features]
        x_test_tabnet = x_test_prepared[self.tabnet_features]
        x_test_lgb = x_test_prepared[self.lgb_features]
        x_test_extra = x_test_prepared[self.extra_features]
        x_test_mlp = x_test_prepared[self.mlp_features]  # Prepare MLP features
        x_test_pytorch = x_test_prepared[self.pytorch_features]  # Added PyTorch features
        x_test_svm = x_test_prepared[self.svm_features]  # Added SVM features
        x_test_fnn = x_test_prepared[self.fnn_features]  # Added FNN features

        # Obtain predictions from base models on validation set
        self.logger.info("Obtaining validation predictions from base models...")
        # Add null checks before accessing model attributes
        if self.model_xgb is None:
            raise ValueError("XGBoost model not loaded")
        # XGBoost predict_proba might not be available on MLflow-loaded model, try pyfunc
        try:
            p_xgb_val = self.model_xgb.predict_proba(x_val_xgb)[:, 1]  # type: ignore
        except AttributeError:
            self.logger.warning("XGBoost predict_proba not available, using pyfunc predict_proba")
            # Load pyfunc version for prediction if direct model doesn't work
            xgb_uri = f"runs:/{self.xgb_run_id}/{'model'}"
            xgb_pyfunc = mlflow.pyfunc.load_model(xgb_uri)
            p_xgb_val = xgb_pyfunc.predict_proba(x_val_xgb)[:, 1]  # type: ignore

        if self.model_tabnet is None:
            raise ValueError("TabNet model not loaded")
        # TabNet input might need .values depending on saving format
        try:
            p_tabnet_val = self.model_tabnet.predict_proba(x_val_tabnet)[:, 1]  # type: ignore
        except TypeError:
            self.logger.warning("TabNet predict_proba failed on DataFrame, trying .values")
            p_tabnet_val = self.model_tabnet.predict_proba(x_val_tabnet.values)[:, 1]  # type: ignore

        if self.model_lgb is None:
            raise ValueError("LightGBM model not loaded")
        p_lgb_val = self.model_lgb.predict_proba(x_val_lgb)[:, 1]  # type: ignore

        if self.model_extra is None:
            raise ValueError("Extra Trees model not loaded")
        p_extra_val = self.model_extra.predict_proba(x_val_extra)[:, 1]  # type: ignore

        # MLP requires scaling
        if self.model_mlp is None:
            raise ValueError("MLP model not loaded")
        if self.model_mlp_scaler is None:
            raise ValueError("MLP scaler not loaded")
        x_val_mlp_scaled = self.model_mlp_scaler.transform(x_val_mlp)
        p_mlp_val = self.model_mlp.predict_proba(x_val_mlp_scaled)[:, 1]  # type: ignore

        # PyTorch model has scaler_ and device_ attached during creation
        if self.model_pytorch is None:
            raise ValueError("PyTorch model not loaded")
        p_pytorch_val = self.model_pytorch.predict_proba(x_val_pytorch)[:, 1]  # type: ignore

        # SVM requires scaling
        if self.model_svm is None:
            raise ValueError("SVM model not loaded")
        if self.model_svm_scaler is None:
            raise ValueError("SVM scaler not loaded")
        x_val_svm_scaled = self.model_svm_scaler.transform(x_val_svm)
        p_svm_val = self.model_svm.predict_proba(x_val_svm_scaled)[:, 1]  # type: ignore

        # FNN
        if self.model_fnn is None:
            raise ValueError("FNN model not loaded")
        p_fnn_val = self.model_fnn.predict_proba(x_val_fnn)[:, 1]  # type: ignore

        # Obtain predictions from base models on test set (used for meta-learner training)
        self.logger.info(
            "Obtaining test predictions from base models (for meta-learner training)..."
        )
        # Null checks already performed above for models
        try:
            p_xgb_test = self.model_xgb.predict_proba(x_test_xgb)[:, 1]  # type: ignore
        except AttributeError:
            self.logger.warning(XGB_PREDICT_PROBA_FALLBACK_MSG)
            xgb_uri = f"runs:/{self.xgb_run_id}/{'model'}"
            xgb_pyfunc = mlflow.pyfunc.load_model(xgb_uri)
            p_xgb_test = xgb_pyfunc.predict_proba(x_test_xgb)[:, 1]  # type: ignore
        try:
            p_tabnet_test = self.model_tabnet.predict_proba(x_test_tabnet)[:, 1]  # type: ignore
        except TypeError:
            p_tabnet_test = self.model_tabnet.predict_proba(x_test_tabnet.values)[:, 1]  # type: ignore
        # LightGBM
        p_lgb_test = self.model_lgb.predict_proba(x_test_lgb)[:, 1]  # type: ignore
        # Extra Trees
        p_extra_test = self.model_extra.predict_proba(x_test_extra)[:, 1]  # type: ignore
        # MLP requires scaling
        x_test_mlp_scaled = self.model_mlp_scaler.transform(x_test_mlp)
        p_mlp_test = self.model_mlp.predict_proba(x_test_mlp_scaled)[:, 1]  # type: ignore
        # PyTorch
        p_pytorch_test = self.model_pytorch.predict_proba(x_test_pytorch)[:, 1]  # type: ignore
        # SVM requires scaling
        x_test_svm_scaled = self.model_svm_scaler.transform(x_test_svm)
        p_svm_test = self.model_svm.predict_proba(x_test_svm_scaled)[:, 1]  # type: ignore
        # FNN
        p_fnn_test = self.model_fnn.predict_proba(x_test_fnn)[:, 1]  # type: ignore
        # Optionally calculate dynamic weights based on validation performance
        if self.dynamic_weighting:
            # Use test predictions for weights used during FINAL meta-learner TRAINING
            self.logger.info("Computing dynamic weights based on test performance...")
            self.dynamic_weights_train, self.thresholds_train = (
                compute_precision_focused_weights_optimized(
                    p_xgb_test,
                    p_tabnet_test,
                    p_lgb_test,
                    p_extra_test,
                    p_mlp_test,
                    p_pytorch_test,  # Added PyTorch
                    p_svm_test,  # Added SVM
                    p_fnn_test,  # Added FNN
                    y_test,
                    self.target_precision,
                    self.min_recalls,  # Should now have 7 elements
                    self.logger,
                )
            )
            # Ensure weights_0410 is imported and used
            self.logger.info("Computing dynamic weights based on validation performance...")
            self.dynamic_weights, self.thresholds = compute_precision_focused_weights_optimized(
                p_xgb_val,
                p_tabnet_val,
                p_lgb_val,
                p_extra_val,
                p_mlp_val,
                p_pytorch_val,  # Added PyTorch
                p_svm_val,  # Added SVM
                p_fnn_val,  # Added FNN
                y_val,
                self.target_precision,
                self.min_recalls,  # Should now have 7 elements
                self.logger,
            )

        # Validation meta-features (used for hypertuning meta-learner and final threshold tuning)
        self.logger.info("Creating validation meta-features...")
        meta_features_val = create_meta_features_optimized(
            p_xgb_val,
            p_tabnet_val,
            p_lgb_val,
            p_extra_val,
            p_mlp_val,
            p_pytorch_val,  # Added PyTorch
            p_svm_val,  # Added SVM
            p_fnn_val,  # Added FNN
            x_val_prepared,
            self.dynamic_weights if self.dynamic_weighting else None,
            self.thresholds if self.dynamic_weighting else None,
        )
        # Test meta-features (used for training the final meta-learner)
        self.logger.info("Creating test meta-features (for meta-learner training)...")
        meta_features_train = create_meta_features_optimized(
            p_xgb_test,
            p_tabnet_test,
            p_lgb_test,
            p_extra_test,
            p_mlp_test,
            p_pytorch_test,  # Added PyTorch
            p_svm_test,  # Added SVM
            p_fnn_test,  # Added FNN
            x_test_prepared,
            self.dynamic_weights_train if self.dynamic_weighting else None,
            self.thresholds_train if self.dynamic_weighting else None,
        )
        # Convert to DataFrame for better interpretability
        meta_df_val = create_meta_dataframe(meta_features_val)
        meta_df_train = create_meta_dataframe(meta_features_train)

        # Initialize and train meta-learner
        self.logger.info(f"Initializing meta-learner of type {self.meta_learner_type}...")
        self.meta_learner = initialize_meta_learner(self.meta_learner_type)

        # Train meta-learner using TEST meta-features and HYPERTUNE using VALIDATION meta-features
        self.logger.info("Hypertuning and training meta-learner...")
        # Fix DataFrame parameter passing to hypertune_meta_learner - convert to numpy arrays
        assert isinstance(meta_df_train, pd.DataFrame), (
            f"meta_df_train must be DataFrame, got {type(meta_df_train)}"
        )
        assert isinstance(meta_df_val, pd.DataFrame), (
            f"meta_df_val must be DataFrame, got {type(meta_df_val)}"
        )
        # Assert that y_test and y_val are not None (should be guaranteed by earlier checks)
        assert y_test is not None, "y_test should not be None at this point"
        assert y_val is not None, "y_val should not be None at this point"
        meta_train_np = meta_df_train.values
        meta_val_np = meta_df_val.values
        self.meta_learner = hypertune_meta_learner(
            meta_train_np,
            y_test,  # Train/evaluate HPO on test set
            meta_val_np,
            y_val,  # Use validation set for final HPO eval (or nested CV split)
            meta_learner_type=self.meta_learner_type,
            target_precision=self.target_precision,
            min_recall=self.required_recall,
        )
        # Tune threshold for optimal precision-recall trade-off using VALIDATION data
        self.logger.info(
            f"Tuning final threshold using validation data for target precision {self.target_precision}..."
        )
        # Initialize variables to avoid unbound variable issues
        meta_df_val_np = None
        if self.meta_learner_type == "tabnet":
            meta_df_val_np = meta_df_val.to_numpy()
            meta_val_probs = self.meta_learner.predict_proba(meta_df_val_np)[:, 1]  # type: ignore
        else:
            meta_val_probs = self.meta_learner.predict_proba(meta_df_val)[:, 1]  # type: ignore
        best_threshold, threshold_metrics = tune_threshold_for_precision_optimized(
            meta_val_probs,
            y_val,
            target_precision=self.target_precision,
            required_recall=self.required_recall,
            logger=self.logger,
        )

        self.optimal_threshold = best_threshold
        self.logger.info(f"Optimal threshold set to {self.optimal_threshold:.4f}")

        # Final evaluation on validation data using the tuned threshold
        self.logger.info("Performing final evaluation on validation data...")
        # Assert y_val is not None for evaluation
        assert y_val is not None, "y_val should not be None for evaluation"
        # evaluate_model expects DataFrame, so always pass meta_df_val
        eval_results = evaluate_model(
            self.meta_learner, meta_df_val, y_val, self.optimal_threshold, self.logger
        )
        eval_results.update(threshold_metrics)  # Add threshold metrics to final results
        self.logger.info(f"Validation evaluation results: {eval_results}")
        self.logger.info("Ensemble model 0404 training completed successfully.")

        return eval_results

    def predict_proba(self, X) -> np.ndarray:
        if self.meta_learner is None:
            raise ValueError("Model has not been trained. Call train() first.")

        # Use original columns from X for prepare_data
        x_prepared = prepare_data(X, X.columns)

        # Select features for each model
        x_xgb = x_prepared[self.xgb_features]
        x_tabnet = x_prepared[self.tabnet_features]
        x_lgb = x_prepared[self.lgb_features]
        x_extra = x_prepared[self.extra_features]
        x_mlp = x_prepared[self.mlp_features]
        x_pytorch = x_prepared[self.pytorch_features]  # Added PyTorch
        x_svm = x_prepared[self.svm_features]  # Added SVM
        x_fnn = x_prepared[self.fnn_features]  # Added FNN
        try:
            # Generate predictions with null guards
            if self.model_xgb is None:
                raise ValueError("XGBoost model not loaded")
            try:
                p_xgb = self.model_xgb.predict_proba(x_xgb)[:, 1]  # type: ignore
            except AttributeError:
                self.logger.warning(XGB_PREDICT_PROBA_FALLBACK_MSG)
                xgb_uri = f"runs:/{self.xgb_run_id}/{'model'}"
                xgb_pyfunc = mlflow.pyfunc.load_model(xgb_uri)
                p_xgb = xgb_pyfunc.predict_proba(x_xgb)[:, 1]  # type: ignore

            if self.model_lgb is None:
                raise ValueError("LightGBM model not loaded")
            p_lgb = self.model_lgb.predict_proba(x_lgb)[:, 1]  # type: ignore

            if self.model_extra is None:
                raise ValueError("Extra Trees model not loaded")
            p_extra = self.model_extra.predict_proba(x_extra)[:, 1]  # type: ignore

            if self.model_pytorch is None:
                raise ValueError("PyTorch model not loaded")
            p_pytorch = self.model_pytorch.predict_proba(x_pytorch)[:, 1]  # type: ignore

            if self.model_fnn is None:
                raise ValueError("FNN model not loaded")
            p_fnn = self.model_fnn.predict_proba(x_fnn)[:, 1]  # type: ignore

            # Handle potential TabNet input type error
            if self.model_tabnet is None:
                raise ValueError("TabNet model not loaded")
            try:
                p_tabnet = self.model_tabnet.predict_proba(x_tabnet)[:, 1]  # type: ignore
            except TypeError:
                p_tabnet = self.model_tabnet.predict_proba(x_tabnet.values)[:, 1]  # type: ignore

            # --- MLP Scaling and Prediction ---
            if self.model_mlp_scaler is None:
                raise ValueError("MLP scaler not loaded")
            # Ensure x_mlp is a DataFrame with correct columns before transform
            if not isinstance(x_mlp, pd.DataFrame):
                self.logger.warning(
                    "x_mlp is not a DataFrame before scaling. Attempting conversion."
                )
                # Assert that mlp_features is a list for DataFrame constructor
                assert isinstance(self.mlp_features, list), (
                    f"mlp_features must be list, got {type(self.mlp_features)}"
                )
                # Explicitly cast to list[str] for type checker
                mlp_columns: list[str] = list(self.mlp_features)
                x_mlp = pd.DataFrame(x_mlp, columns=mlp_columns)  # type: ignore
            # Re-select columns just in case order changed or to ensure DataFrame type
            x_mlp = x_mlp[self.mlp_features]
            x_mlp_scaled = self.model_mlp_scaler.transform(x_mlp)
            if self.model_mlp is None:
                raise ValueError("MLP model not loaded")
            p_mlp = self.model_mlp.predict_proba(x_mlp_scaled)[:, 1]  # type: ignore

            # --- SVM Scaling and Prediction ---
            if self.model_svm_scaler is None:
                raise ValueError("SVM scaler not loaded")
            # Ensure x_svm is a DataFrame with correct columns before transform
            if not isinstance(x_svm, pd.DataFrame):
                self.logger.warning(
                    "x_svm is not a DataFrame before scaling. Attempting conversion."
                )
                # Assert that svm_features is a list for DataFrame constructor
                assert isinstance(self.svm_features, list), (
                    f"svm_features must be list, got {type(self.svm_features)}"
                )
                # Explicitly cast to list[str] for type checker
                svm_columns: list[str] = list(self.svm_features)
                x_svm = pd.DataFrame(x_svm, columns=svm_columns)  # type: ignore
            # Re-select columns
            x_svm = x_svm[self.svm_features]
            x_svm_scaled = self.model_svm_scaler.transform(x_svm)
            if self.model_svm is None:
                raise ValueError("SVM model not loaded")
            p_svm = self.model_svm.predict_proba(x_svm_scaled)[:, 1]  # type: ignore

            meta_features = create_meta_features_optimized(
                p_xgb,
                p_tabnet,
                p_lgb,
                p_extra,
                p_mlp,
                p_pytorch,
                p_svm,  # Added SVM
                p_fnn,  # Added FNN
                x_prepared,
                self.dynamic_weights_train if self.dynamic_weighting else None,
                self.thresholds_train if self.dynamic_weighting else None,
            )
            # Ensure meta_features is a DataFrame before prediction if meta_learner expects it
            meta_df = create_meta_dataframe(meta_features)
            if self.meta_learner_type == "tabnet":
                meta_df_np = meta_df.to_numpy()
                meta_probs = self.meta_learner.predict_proba(meta_df_np)[:, 1]  # type: ignore
            else:
                meta_probs = self.meta_learner.predict_proba(meta_df)  # type: ignore
            # Handle both 1D and 2D probability arrays
            if len(meta_probs.shape) == 1:
                # Already 1D probabilities
                return meta_probs
            else:
                # Extract positive class probabilities from 2D array
                return meta_probs[:, 1]
        except Exception as e:
            self.logger.error(f"Error predicting probabilities: {e}")
            raise

    def predict(self, X) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return (probabilities >= self.optimal_threshold).astype(int)

    def explain_predictions(self, x_val) -> dict:
        # This might need updating if explain_predictions relies on specific base model types
        return explain_predictions(self, x_val, self.logger)

    def analyze_prediction_errors(self, x_val, y_val) -> dict:
        return analyze_prediction_errors(self, x_val, y_val, self.optimal_threshold, self.logger)

    def precision_filter(self, x, probabilities):
        # This logic might be too specific, consider making it more general or removing
        high_conf = probabilities > self.optimal_threshold
        x_high_conf = x[high_conf]
        if "home_form" in x_high_conf.columns and "away_form" in x_high_conf.columns:
            form_diff = abs(x_high_conf["home_form"] - x_high_conf["away_form"])
            likely_not_draw = form_diff > 0.5
            high_conf[high_conf] = ~likely_not_draw
        return high_conf

    def get_model_params(self, model):
        # Generic parameter getter
        try:
            if hasattr(model, "get_params"):
                return model.get_params()
            elif hasattr(model, "get_config"):
                return model.get_config()
            else:
                return json.loads(model.to_json())
        except Exception as e:
            return {"error": str(e)}

    def _load_sklearn_model_with_features(self, run_id, model_path, model_name, flavor="sklearn"):
        """Helper method to load sklearn-based models and extract features."""
        try:
            self.logger.info(f"Loading {model_name} model from run {run_id}...")
            uri = f"runs:/{run_id}/{model_path}"

            if flavor == "xgboost":
                model = mlflow.xgboost.load_model(uri)
            elif flavor == "lightgbm":
                model = mlflow.lightgbm.load_model(uri)
            else:
                model = mlflow.sklearn.load_model(uri)

            pyfunc = mlflow.pyfunc.load_model(uri)
            features = []
            if pyfunc.metadata.signature and pyfunc.metadata.signature.inputs:
                features = pyfunc.metadata.signature.inputs.input_names()
                self.logger.info(
                    f"Updated {model_name} feature signature: {len(features)} features"
                )
            else:
                self.logger.warning(f"No feature signature found for {model_name} model")
                if flavor == "lightgbm":
                    features = getattr(model, "feature_name_", [])
                else:
                    features = getattr(model, "feature_names_in_", [])

            return model, features
        except Exception as e:
            self.logger.error(f"Failed to load {model_name} model: {str(e)}")
            raise ValueError(f"Failed to load {model_name} model: {str(e)}") from e

    def _load_sklearn_model_with_scaler(self, run_id, model_path, scaler_path, model_name):
        """Helper method to load sklearn models with scalers."""
        try:
            self.logger.info(f"Loading {model_name} model from run {run_id}...")
            model_uri = f"runs:/{run_id}/{model_path}"
            model = mlflow.sklearn.load_model(model_uri)

            pyfunc = mlflow.pyfunc.load_model(model_uri)
            features = []
            if pyfunc.metadata.signature and pyfunc.metadata.signature.inputs:
                features = pyfunc.metadata.signature.inputs.input_names()
                self.logger.info(
                    f"Updated {model_name} feature signature: {len(features)} features"
                )
            else:
                self.logger.warning(f"No feature signature found for {model_name} model")
                features = getattr(model, "feature_names_in_", [])

            # Load the associated scaler
            self.logger.info(
                f"Loading {model_name} scaler artifact '{scaler_path}' from run {run_id}..."
            )
            scaler_local_path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path=scaler_path
            )
            with open(scaler_local_path, "rb") as f:
                scaler = pickle.load(f)
            self.logger.info(f"{model_name} scaler loaded successfully.")

            return model, scaler, features
        except Exception as e:
            self.logger.error(f"Failed to load {model_name} model or scaler: {str(e)}")
            raise ValueError(f"Failed to load {model_name} model or scaler: {str(e)}") from e

    def _load_pytorch_model_with_scaler(self, run_id, model_path, scaler_path, model_name):
        """Helper method to load PyTorch models and their scalers."""
        try:
            self.logger.info(f"Loading {model_name} model from run {run_id}...")
            model_uri = f"runs:/{run_id}/{model_path}"
            model = mlflow.pytorch.load_model(model_uri)

            # Load PyTorch model also as pyfunc to easily get signature
            pyfunc = mlflow.pyfunc.load_model(model_uri)
            features = []
            if pyfunc.metadata.signature and pyfunc.metadata.signature.inputs:
                features = pyfunc.metadata.signature.inputs.input_names()
                self.logger.info(
                    f"Updated {model_name} feature signature: {len(features)} features"
                )
            else:
                self.logger.warning(f"No feature signature found for {model_name} model.")
                features = []

            # Load the associated scaler
            self.logger.info(
                f"Loading {model_name} scaler artifact '{scaler_path}' from run {run_id}..."
            )
            scaler_local_path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path=scaler_path
            )
            with open(scaler_local_path, "rb") as f:
                scaler = pickle.load(f)
            self.logger.info(f"{model_name} scaler loaded successfully.")

            return model, scaler, features
        except Exception as e:
            self.logger.error(f"Failed to load {model_name} model or scaler: {str(e)}")
            raise ValueError(f"Failed to load {model_name} model or scaler: {str(e)}") from e

    def load_models_from_mlflow(
        self,
        xgb_path="model",
        lgb_path="model",
        tabnet_path="model_sklearn",
        extra_path="model",  # Path for Extra Trees model
        mlp_path="model",  # Artifact path for MLP model
        mlp_scaler_path="scaler/scaler_mlp.pkl",  # Artifact path for MLP scaler
        pytorch_path="model",
        pytorch_scaler_path=PYTORCH_SCALER_PATH,
        svm_path="model_svm",  # Artifact path for SVM model (from svm_model.py)
        svm_scaler_path="scaler_svm.pkl",  # Artifact path for SVM scaler (default name in svm_model.py)
        fnn_path="model",
        fnn_scaler_path=PYTORCH_SCALER_PATH,
    ):
        """
        Load pre-trained models (XGB, LGBM, TabNet, Extra Trees, MLP, PyTorch, SVM)
        and scalers from MLflow.
        Updates feature signatures for each model.
        """
        self.logger.info("Loading models and scaler from MLflow repository...")

        # Load XGBoost model
        self.model_xgb, self.xgb_features = self._load_sklearn_model_with_features(
            self.xgb_run_id, xgb_path, "XGBoost", "xgboost"
        )

        # Load LightGBM model
        self.model_lgb, self.lgb_features = self._load_sklearn_model_with_features(
            self.lgb_run_id, lgb_path, "LightGBM", "lightgbm"
        )

        # Load TabNet model
        self.model_tabnet, self.tabnet_features = self._load_sklearn_model_with_features(
            self.tabnet_run_id, tabnet_path, "TabNet", "sklearn"
        )

        # Load Random Forest model
        self.model_extra, self.extra_features = self._load_sklearn_model_with_features(
            self.extra_run_id, extra_path, "Extra Trees", "sklearn"
        )

        # Load MLP model and scaler
        self.model_mlp, self.model_mlp_scaler, self.mlp_features = (
            self._load_sklearn_model_with_scaler(self.mlp_run_id, mlp_path, mlp_scaler_path, "MLP")
        )

        # Load PyTorch model and scaler
        self.model_pytorch, self.model_pytorch_scaler, self.pytorch_features = (
            self._load_pytorch_model_with_scaler(
                self.pytorch_run_id, pytorch_path, pytorch_scaler_path, "PyTorch"
            )
        )

        # Load SVM model and scaler
        self.model_svm, self.model_svm_scaler, self.svm_features = (
            self._load_sklearn_model_with_scaler(self.svm_run_id, svm_path, svm_scaler_path, "SVM")
        )

        # Load FNN model and scaler
        self.model_fnn, self.model_fnn_scaler, self.fnn_features = (
            self._load_pytorch_model_with_scaler(self.fnn_run_id, fnn_path, fnn_scaler_path, "FNN")
        )

        self.logger.info("Base models loading complete.")
        # Consider setting self.selected_features based on intersection or a specific model
        # self.selected_features = self.xgb_features
        return True
