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
import mlflow.lightgbm
import mlflow.pyfunc
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin

from models.ensemble.data_utils import prepare_data
from models.ensemble.diagnostics import analyze_prediction_errors, explain_predictions
from models.ensemble.evaluation import evaluate_model
from models.ensemble.meta_features_0404 import create_meta_dataframe, create_meta_features_optimized
from models.ensemble.thresholds import tune_threshold_for_precision_optimized
from models.ensemble.training import hypertune_meta_learner, initialize_meta_learner
from models.ensemble.weights_0404 import compute_precision_focused_weights_optimized

# Import shared utility functions
from utils.logger import ExperimentLogger

# Set fixed seed and hash seed for determinism
SEED = 19
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Restrict parallel threads
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
# PyTorch specific reproducibility settings
torch.manual_seed(SEED)


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
        sampling_strategy=0.7,
        complexity_penalty=0.01,
        target_precision=0.50,
        required_recall=0.25,
        X_train=None,
    ):
        # Use provided logger or create a new one specific to this version
        self.logger = logger or ExperimentLogger(
            experiment_name="ensemble_model_0404", log_dir="./logs/ensemble_model_0404"
        )
        self.required_recall = required_recall
        self.sampling_strategy = sampling_strategy
        self.complexity_penalty = complexity_penalty
        self.target_precision = target_precision

        # --- MLflow Run IDs for Base Models ---
        self.xgb_run_id = "8371ab9f141c48b78033587e52dd5a08"  # Keep existing
        self.lgb_run_id = "99c15164c539454c86cb85ae36ab7033"  # Keep existing
        self.tabnet_run_id = "19a2b8f15feb44a68838eec74271acbc"  # Keep existing
        self.extra_run_id = "2830d0b8ebcb4c46809e6afab57da539"  # Keep existing (now standard)
        self.mlp_run_id = "30a6144dcb144396b2a7e0a784688ac5"
        # ------------------------------------

        # Minimum recalls for dynamic weighting (order: xgb, tabnet, lgb, rf, mlp)
        self.min_recalls = [0.30, 0.20, 0.20, 0.40, 0.30]  # Added MLP recall (adjust if needed)

        # Meta-learner settings
        self.meta_learner_type = meta_learner_type
        self.optimal_threshold = 0.5
        self.individual_thresholding = individual_thresholding  # Unused currently?
        self.calibrate = calibrate  # Unused currently?
        self.calibration_method = calibration_method  # Unused currently?
        self.dynamic_weighting = dynamic_weighting
        if self.dynamic_weighting:
            # Adjusted for 5 base models
            self.dynamic_weights = {
                "xgb": 1 / 5,
                "tabnet": 1 / 5,
                "lgb": 1 / 5,
                "extra": 1 / 5,
                "mlp": 1 / 5,
            }
        self.meta_learner = None

        # Placeholder attributes for models and features - will be populated by load_models
        self.model_xgb = None
        self.model_lgb = None
        self.model_tabnet = None
        self.model_extra = None  # Renamed from model_extra
        self.model_mlp = None
        self.model_mlp_scaler = None  # Scaler specific to MLP

        self.xgb_features = []
        self.lgb_features = []
        self.tabnet_features = []
        self.extra_features = []
        self.mlp_features = []

        # Load models on initialization
        self.load_models_from_mlflow()

    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        X_test=None,
        y_test=None,
        split_validation=True,
        val_size=0.2,
    ) -> dict:
        self.logger.info("Starting ensemble model 0404 training...")
        # Data preparation
        # Assume features are loaded during model loading, use a consistent set if needed
        # self.selected_features = self.xgb_features # Example: Use XGB features as the common set
        X_train_prepared = prepare_data(X_train, X_train.columns)
        if X_val is not None:
            X_val_prepared = prepare_data(X_val, X_val.columns)
        if X_test is not None:
            X_test_prepared = prepare_data(X_test, X_test.columns)

        if split_validation or X_val is None or y_val is None:
            from sklearn.model_selection import train_test_split

            self.logger.info("Splitting training data for validation...")
            X_train_prepared, X_val_prepared, y_train, y_val = train_test_split(
                X_train_prepared, y_train, test_size=val_size, random_state=SEED, stratify=y_train
            )

        # Prepare feature subsets for each model
        self.logger.info("Preparing feature subsets for base models...")
        X_val_xgb = X_val_prepared[self.xgb_features]
        X_val_tabnet = X_val_prepared[self.tabnet_features]
        X_val_lgb = X_val_prepared[self.lgb_features]
        X_val_extra = X_val_prepared[self.extra_features]
        X_val_mlp = X_val_prepared[self.mlp_features]  # Prepare MLP features

        X_test_xgb = X_test_prepared[self.xgb_features]
        X_test_tabnet = X_test_prepared[self.tabnet_features]
        X_test_lgb = X_test_prepared[self.lgb_features]
        X_test_extra = X_test_prepared[self.extra_features]
        X_test_mlp = X_test_prepared[self.mlp_features]  # Prepare MLP features

        # Obtain predictions from base models on validation set
        self.logger.info("Obtaining validation predictions from base models...")
        p_xgb_val = self.model_xgb.predict_proba(X_val_xgb)[:, 1]
        p_tabnet_val = self.model_tabnet.predict_proba(X_val_tabnet.values)[:, 1]
        p_lgb_val = self.model_lgb.predict_proba(X_val_lgb)[:, 1]
        p_extra_val = self.model_extra.predict_proba(X_val_extra)[:, 1]
        # MLP requires scaling
        X_val_mlp_scaled = self.model_mlp_scaler.transform(X_val_mlp)
        p_mlp_val = self.model_mlp.predict_proba(X_val_mlp_scaled)[:, 1]

        # Obtain predictions from base models on test set (used for meta-learner training)
        self.logger.info(
            "Obtaining test predictions from base models (for meta-learner training)..."
        )
        p_xgb_test = self.model_xgb.predict_proba(X_test_xgb)[:, 1]
        p_tabnet_test = self.model_tabnet.predict_proba(X_test_tabnet.values)[:, 1]
        p_lgb_test = self.model_lgb.predict_proba(X_test_lgb)[:, 1]
        p_extra_test = self.model_extra.predict_proba(X_test_extra)[:, 1]
        # MLP requires scaling
        X_test_mlp_scaled = self.model_mlp_scaler.transform(X_test_mlp)
        p_mlp_test = self.model_mlp.predict_proba(X_test_mlp_scaled)[:, 1]

        # Optionally calculate dynamic weights based on validation performance
        if self.dynamic_weighting:
            self.logger.info("Computing dynamic weights based on validation performance...")
            # Use validation predictions for weights used during meta-learner HYPERPARAMETER TUNING
            self.logger.info("Computing dynamic weights based on validation performance...")
            self.dynamic_weights, self.thresholds = compute_precision_focused_weights_optimized(
                p_xgb_val,
                p_tabnet_val,
                p_lgb_val,
                p_extra_val,
                p_mlp_val,  # Added MLP
                y_val,
                self.target_precision,
                self.min_recalls,
                self.logger,
            )
            # Use test predictions for weights used during FINAL meta-learner TRAINING
            # (assuming meta_features_train is used for fitting)
            self.logger.info("Computing dynamic weights based on test performance...")
            self.dynamic_weights_train, self.thresholds_train = (
                compute_precision_focused_weights_optimized(
                    p_xgb_test,
                    p_tabnet_test,
                    p_lgb_test,
                    p_extra_test,
                    p_mlp_test,  # Added MLP
                    y_test,
                    self.target_precision,
                    self.min_recalls,
                    self.logger,
                )
            )

        # Create meta-features from base model predictions
        # Validation meta-features (used for hypertuning meta-learner and final threshold tuning)
        self.logger.info("Creating validation meta-features...")
        meta_features_val = create_meta_features_optimized(
            p_xgb_val,
            p_tabnet_val,
            p_lgb_val,
            p_extra_val,
            p_mlp_val,  # Added MLP
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
            p_mlp_test,  # Added MLP
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
        self.meta_learner = hypertune_meta_learner(
            meta_df_train,
            y_test,  # Train/evaluate HPO on test set
            meta_df_val,
            y_val,  # Use validation set for final HPO eval (or nested CV split)
            meta_learner_type=self.meta_learner_type,
            target_precision=self.target_precision,
            min_recall=self.required_recall,
        )
        # Tune threshold for optimal precision-recall trade-off using VALIDATION data
        self.logger.info(
            f"Tuning final threshold using validation data for target precision {self.target_precision}..."
        )
        meta_val_probs = self.meta_learner.predict_proba(meta_df_val)[:, 1]
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
        X_prepared = prepare_data(X, X.columns)

        # Select features for each model
        X_xgb = X_prepared[self.xgb_features]
        X_tabnet = X_prepared[self.tabnet_features]
        X_lgb = X_prepared[self.lgb_features]
        X_extra = X_prepared[self.extra_features]
        X_mlp = X_prepared[self.mlp_features]

        try:
            # Generate predictions
            p_xgb = self.model_xgb.predict_proba(X_xgb)[:, 1]
            p_tabnet = self.model_tabnet.predict_proba(X_tabnet.values)[:, 1]
            p_lgb = self.model_lgb.predict_proba(X_lgb)[:, 1]
            p_extra = self.model_extra.predict_proba(X_extra)[:, 1]
            # Scale for MLP
            X_mlp_scaled = self.model_mlp_scaler.transform(X_mlp)
            p_mlp = self.model_mlp.predict_proba(X_mlp_scaled)[:, 1]

            # Create meta features (dynamic weights/thresholds are class attributes)
            meta_features = create_meta_features_optimized(
                p_xgb,
                p_tabnet,
                p_lgb,
                p_extra,
                p_mlp,  # Added MLP
                self.dynamic_weights if self.dynamic_weighting else None,
                self.thresholds if self.dynamic_weighting else None,
            )
            # Ensure meta_features is a DataFrame before prediction if meta_learner expects it
            meta_df = create_meta_dataframe(meta_features)

            meta_probs = self.meta_learner.predict_proba(meta_df)
            return meta_probs[:, 1]
        except Exception as e:
            self.logger.error(f"Error predicting probabilities: {e}", exc_info=True)
            raise

    def predict(self, X) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return (probabilities >= self.optimal_threshold).astype(int)

    def explain_predictions(self, X_val) -> dict:
        # This might need updating if explain_predictions relies on specific base model types
        return explain_predictions(self, X_val, self.logger)

    def analyze_prediction_errors(self, X_val, y_val) -> dict:
        return analyze_prediction_errors(self, X_val, y_val, self.optimal_threshold, self.logger)

    def precision_filter(self, X, probabilities):
        # This logic might be too specific, consider making it more general or removing
        high_conf = probabilities > self.optimal_threshold
        X_high_conf = X[high_conf]
        if "home_form" in X_high_conf.columns and "away_form" in X_high_conf.columns:
            form_diff = abs(X_high_conf["home_form"] - X_high_conf["away_form"])
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

    def load_models_from_mlflow(
        self,
        xgb_path="model",
        lgb_path="model",
        tabnet_path="model",
        extra_path="model",  # Path for Extra Trees model
        mlp_path="model_sklearn",  # Artifact path for MLP model
        mlp_scaler_path="scaler_sklearn.pkl",  # Artifact path for MLP scaler
    ):
        """
        Load pre-trained models (XGB, LGBM, TabNet, Extra Trees, MLP) and MLP scaler from MLflow.
        Updates feature signatures for each model.
        """
        self.logger.info("Loading models and scaler from MLflow repository...")

        # Load XGBoost model
        try:
            self.logger.info(f"Loading XGBoost model from run {self.xgb_run_id}...")
            xgb_uri = f"runs:/{self.xgb_run_id}/{xgb_path}"
            self.model_xgb = mlflow.xgboost.load_model(xgb_uri)
            xgb_pyfunc = mlflow.pyfunc.load_model(xgb_uri)
            if xgb_pyfunc.metadata.signature and xgb_pyfunc.metadata.signature.inputs:
                self.xgb_features = xgb_pyfunc.metadata.signature.inputs.input_names()
                self.logger.info(
                    f"Updated XGBoost feature signature: {len(self.xgb_features)} features"
                )
            else:
                self.logger.warning("No feature signature found for XGBoost model")
                self.xgb_features = getattr(self.model_xgb, "feature_names_in_", [])
        except Exception as e:
            self.logger.error(f"Failed to load XGBoost model: {str(e)}")
            raise ValueError(f"Failed to load XGBoost model: {str(e)}") from e

        # Load LightGBM model
        try:
            self.logger.info(f"Loading LightGBM model from run {self.lgb_run_id}...")
            lgb_uri = f"runs:/{self.lgb_run_id}/{lgb_path}"
            self.model_lgb = mlflow.lightgbm.load_model(lgb_uri)
            lgb_pyfunc = mlflow.pyfunc.load_model(lgb_uri)
            if lgb_pyfunc.metadata.signature and lgb_pyfunc.metadata.signature.inputs:
                self.lgb_features = lgb_pyfunc.metadata.signature.inputs.input_names()
                self.logger.info(
                    f"Updated LightGBM feature signature: {len(self.lgb_features)} features"
                )
            else:
                self.logger.warning("No feature signature found for LightGBM model")
                self.lgb_features = getattr(self.model_lgb, "feature_name_", [])
        except Exception as e:
            self.logger.error(f"Failed to load LightGBM model: {str(e)}")
            raise ValueError(f"Failed to load LightGBM model: {str(e)}") from e

        # Load TabNet model
        try:
            self.logger.info(f"Loading TabNet model from run {self.tabnet_run_id}...")
            tabnet_uri = f"runs:/{self.tabnet_run_id}/{tabnet_path}"
            self.model_tabnet = mlflow.sklearn.load_model(
                tabnet_uri
            )  # Assuming saved via sklearn flavor
            tabnet_pyfunc = mlflow.pyfunc.load_model(tabnet_uri)
            if tabnet_pyfunc.metadata.signature and tabnet_pyfunc.metadata.signature.inputs:
                self.tabnet_features = tabnet_pyfunc.metadata.signature.inputs.input_names()
                self.logger.info(
                    f"Updated TabNet feature signature: {len(self.tabnet_features)} features"
                )
            else:
                self.logger.warning("No feature signature found for TabNet model")
                self.tabnet_features = getattr(self.model_tabnet, "feature_names_in_", [])
        except Exception as e:
            self.logger.error(f"Failed to load TabNet model: {str(e)}")
            raise ValueError(f"Failed to load TabNet model: {str(e)}") from e

        # Load Random Forest model
        try:
            self.logger.info(f"Loading Extra Trees model from run {self.extra_run_id}...")
            extra_uri = f"runs:/{self.extra_run_id}/{extra_path}"
            self.model_extra = mlflow.sklearn.load_model(extra_uri)
            extra_pyfunc = mlflow.pyfunc.load_model(extra_uri)
            if extra_pyfunc.metadata.signature and extra_pyfunc.metadata.signature.inputs:
                self.extra_features = extra_pyfunc.metadata.signature.inputs.input_names()
                self.logger.info(
                    f"Updated Extra Trees signature: {len(self.extra_features)} features"
                )
            else:
                self.logger.warning("No feature signature found for Extra Trees model")
                self.extra_features = getattr(self.model_extra, "feature_names_in_", [])
        except Exception as e:
            self.logger.error(f"Failed to load Extra Trees model: {str(e)}")
            raise ValueError(f"Failed to load Extra Trees model: {str(e)}") from e

        # Load MLP (sklearn) model
        if not self.mlp_run_id or self.mlp_run_id == "PLACEHOLDER_MLP_RUN_ID":
            self.logger.warning(
                "MLP Run ID is not set or is a placeholder. Skipping MLP model loading."
            )
        else:
            try:
                self.logger.info(f"Loading MLP model from run {self.mlp_run_id}...")
                mlp_uri = f"runs:/{self.mlp_run_id}/{mlp_path}"
                self.model_mlp = mlflow.sklearn.load_model(mlp_uri)
                mlp_pyfunc = mlflow.pyfunc.load_model(mlp_uri)
                if mlp_pyfunc.metadata.signature and mlp_pyfunc.metadata.signature.inputs:
                    self.mlp_features = mlp_pyfunc.metadata.signature.inputs.input_names()
                    self.logger.info(
                        f"Updated MLP feature signature: {len(self.mlp_features)} features"
                    )
                else:
                    self.logger.warning("No feature signature found for MLP model")
                    self.mlp_features = getattr(self.model_mlp, "feature_names_in_", [])

                # Load the associated scaler
                self.logger.info(
                    f"Loading MLP scaler artifact '{mlp_scaler_path}' from run {self.mlp_run_id}..."
                )
                scaler_local_path = mlflow.artifacts.download_artifacts(
                    run_id=self.mlp_run_id, artifact_path=mlp_scaler_path
                )
                with open(scaler_local_path, "rb") as f:
                    self.model_mlp_scaler = pickle.load(f)
                self.logger.info("MLP scaler loaded successfully.")

            except Exception as e:
                self.logger.error(f"Failed to load MLP model or scaler: {str(e)}")
                # Decide if this should be fatal or just a warning
                raise ValueError(f"Failed to load MLP model or scaler: {str(e)}") from e

        self.logger.info("Base models loading complete.")
        # Consider setting self.selected_features based on intersection or a specific model
        # self.selected_features = self.xgb_features
        return True
