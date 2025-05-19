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
import pandas as pd  # Add pandas import
import sklearn
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler  # Needed for SVM scaler type check potentially

from src.models.ensemble.data_utils import prepare_data
from src.models.ensemble.diagnostics import analyze_prediction_errors, explain_predictions
from src.models.ensemble.evaluation import evaluate_model
from src.models.ensemble.meta_features_0414 import (
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
        target_precision=0.50,
        required_recall=0.25,
    ):
        # Use provided logger or create a new one specific to this version
        self.logger = logger or ExperimentLogger(
            experiment_name="ensemble_model_0410", log_dir="./logs/ensemble_model_0410"
        )
        self.required_recall = required_recall # For meta-learner

        self.target_precision = target_precision # For dynamic weights

        # --- MLflow Run IDs for Base Models ---
        self.xgb_run_id = "5f827508ce8346a99206d37f626912bf"
        self.lgb_run_id = "be439e143bd04b768309ca1f4e03199d"  
        self.tabnet_run_id = "e7d72ec3cd5c48ecb129630a50ed311d" 
        self.extra_run_id = "ec3e2fdbd57b4edab1988f7e39457de2" 
        self.mlp_run_id = "2d921ea19abd472f8650055e1b9f92c2"
        self.pytorch_run_id = "b42c959962574bcc9c41289e80aee7e4"
        self.svm_run_id = "2b10c9862c7d48db9a1922a3c30ba28e"  
        self.fnn_run_id = "9868d00ccb3c4629a2d2d00f100c8951"

        # Minimum recalls for dynamic weighting (order: xgb, tabnet, lgb, rf, mlp, pytorch, svm)
        self.min_recalls = [0.30, 0.20, 0.30, 0.40, 0.30, 0.30, 0.30, 0.30] # Added SVM recall

        # Meta-learner settings
        self.meta_learner_type = meta_learner_type
        self.optimal_threshold = 0.5 # Will be tuned
        self.individual_thresholding = individual_thresholding # Unused currently?
        self.calibrate = calibrate  # Unused currently?
        self.calibration_method = calibration_method # Unused currently?
        self.dynamic_weighting = dynamic_weighting
        num_models = 8 # Updated number of models (added SVM)
        if self.dynamic_weighting:
            # Adjusted for 7 base models
            self.dynamic_weights = {
                "xgb": 1 / num_models,
                "lgb": 1 / num_models,
                "tabnet": 1 / num_models,
                "extra": 1 / num_models,
                "mlp": 1 / num_models,
                "pytorch": 1 / num_models, # Added pytorch
                "svm": 1 / num_models, # Added SVM
                "fnn": 1 / num_models, # Added FNN
            }
        
        # Placeholder attributes for models and features - will be populated by load_models
        self.meta_learner = None
        self.model_xgb = None
        self.model_lgb = None
        self.model_tabnet = None
        self.model_extra = None 
        self.model_mlp = None
        self.model_mlp_scaler = None 
        self.model_pytorch = None       # Added pytorch model placeholder
        self.model_pytorch_scaler = None # Added pytorch scaler placeholder
        self.model_svm = None           # Added SVM model placeholder
        self.model_svm_scaler = None    # Added SVM scaler placeholder
        self.model_fnn = None           # Added FNN model placeholder
        self.model_fnn_scaler = None    # Added FNN scaler placeholder

        self.xgb_features = []
        self.lgb_features = []
        self.tabnet_features = []
        self.extra_features = []
        self.mlp_features = []
        self.pytorch_features = []      # Added pytorch feature list placeholder
        self.svm_features = []          # Added SVM feature list placeholder
        self.fnn_features = []          # Added FNN feature list placeholder

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
        # X_train_prepared = prepare_data(X_train, X_train.columns)
        X_val_prepared = prepare_data(X_val, X_val.columns)
        X_test_prepared = prepare_data(X_test, X_test.columns)

        # Prepare feature subsets for each model
        self.logger.info("Preparing feature subsets for base models...")
        try:
            X_val_xgb = X_val_prepared[self.xgb_features]
        except KeyError:
            self.logger.warning(f"XGB features not found in X_val_prepared. Using all features. {self.xgb_features}")
            X_val_xgb = X_val_prepared
        X_val_tabnet = X_val_prepared[self.tabnet_features]
        X_val_lgb = X_val_prepared[self.lgb_features]
        X_val_extra = X_val_prepared[self.extra_features]
        X_val_mlp = X_val_prepared[self.mlp_features]  # Prepare MLP features
        X_val_pytorch = X_val_prepared[self.pytorch_features] # Added PyTorch features
        X_val_svm = X_val_prepared[self.svm_features]         # Added SVM features
        X_val_fnn = X_val_prepared[self.fnn_features]         # Added FNN features

        X_test_xgb = X_test_prepared[self.xgb_features]
        X_test_tabnet = X_test_prepared[self.tabnet_features]
        X_test_lgb = X_test_prepared[self.lgb_features]
        X_test_extra = X_test_prepared[self.extra_features]
        X_test_mlp = X_test_prepared[self.mlp_features]  # Prepare MLP features
        X_test_pytorch = X_test_prepared[self.pytorch_features] # Added PyTorch features
        X_test_svm = X_test_prepared[self.svm_features]         # Added SVM features
        X_test_fnn = X_test_prepared[self.fnn_features]         # Added FNN features

        # Obtain predictions from base models on validation set
        self.logger.info("Obtaining validation predictions from base models...")
        p_xgb_val = self.model_xgb.predict_proba(X_val_xgb)[:, 1]
        # TabNet input might need .values depending on saving format
        try:
            p_tabnet_val = self.model_tabnet.predict_proba(X_val_tabnet)[:, 1] 
        except TypeError:
            self.logger.warning("TabNet predict_proba failed on DataFrame, trying .values")
            p_tabnet_val = self.model_tabnet.predict_proba(X_val_tabnet.values)[:, 1]
        p_lgb_val = self.model_lgb.predict_proba(X_val_lgb)[:, 1]
        p_extra_val = self.model_extra.predict_proba(X_val_extra)[:, 1]
        # MLP requires scaling
        X_val_mlp_scaled = self.model_mlp_scaler.transform(X_val_mlp)
        p_mlp_val = self.model_mlp.predict_proba(X_val_mlp_scaled)[:, 1]
        # PyTorch model has scaler_ and device_ attached during creation
        p_pytorch_val = self.model_pytorch.predict_proba(X_val_pytorch)[:, 1]
        # SVM requires scaling
        X_val_svm_scaled = self.model_svm_scaler.transform(X_val_svm)
        p_svm_val = self.model_svm.predict_proba(X_val_svm_scaled)[:, 1]
        # FNN
        p_fnn_val = self.model_fnn.predict_proba(X_val_fnn)[:, 1]

        # Obtain predictions from base models on test set (used for meta-learner training)
        self.logger.info(
            "Obtaining test predictions from base models (for meta-learner training)..."
        )
        p_xgb_test = self.model_xgb.predict_proba(X_test_xgb)[:, 1]
        try:
            p_tabnet_test = self.model_tabnet.predict_proba(X_test_tabnet)[:, 1]
        except TypeError:
            p_tabnet_test = self.model_tabnet.predict_proba(X_test_tabnet.values)[:, 1]
        # LightGBM
        p_lgb_test = self.model_lgb.predict_proba(X_test_lgb)[:, 1]
        # Extra Trees
        p_extra_test = self.model_extra.predict_proba(X_test_extra)[:, 1]
        # MLP requires scaling
        X_test_mlp_scaled = self.model_mlp_scaler.transform(X_test_mlp)
        p_mlp_test = self.model_mlp.predict_proba(X_test_mlp_scaled)[:, 1]
        # PyTorch 
        p_pytorch_test = self.model_pytorch.predict_proba(X_test_pytorch)[:, 1]
        # SVM requires scaling
        X_test_svm_scaled = self.model_svm_scaler.transform(X_test_svm)
        p_svm_test = self.model_svm.predict_proba(X_test_svm_scaled)[:, 1]
        # FNN
        p_fnn_test = self.model_fnn.predict_proba(X_test_fnn)[:, 1]
        # Optionally calculate dynamic weights based on validation performance
        if self.dynamic_weighting:
            # Use test predictions for weights used during FINAL meta-learner TRAINING
            self.logger.info("Computing dynamic weights based on test performance...")
            self.dynamic_weights_train, self.thresholds_train = compute_precision_focused_weights_optimized(
                p_xgb_test,
                p_tabnet_test,
                p_lgb_test,
                p_extra_test,
                p_mlp_test,  
                p_pytorch_test, # Added PyTorch
                p_svm_test,     # Added SVM
                p_fnn_test,     # Added FNN
                y_test,
                self.target_precision,
                self.min_recalls, # Should now have 7 elements
                self.logger,
            )
            # Ensure weights_0410 is imported and used
            self.logger.info("Computing dynamic weights based on validation performance...")
            self.dynamic_weights, self.thresholds = compute_precision_focused_weights_optimized(
                p_xgb_val,
                p_tabnet_val,
                p_lgb_val,
                p_extra_val,
                p_mlp_val,  
                p_pytorch_val, # Added PyTorch
                p_svm_val,     # Added SVM
                p_fnn_val,     # Added FNN
                y_val, 
                self.target_precision,
                self.min_recalls, # Should now have 7 elements
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
            p_pytorch_val, # Added PyTorch
            p_svm_val,     # Added SVM
            p_fnn_val,     # Added FNN
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
            p_pytorch_test, # Added PyTorch
            p_svm_test,     # Added SVM
            p_fnn_test,     # Added FNN
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
        if self.meta_learner_type == "tabnet":
            meta_df_val_np = meta_df_val.to_numpy()
            meta_val_probs = self.meta_learner.predict_proba(meta_df_val_np)[:, 1]
        else:
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
        if self.meta_learner_type == "tabnet":
            eval_results = evaluate_model(
                self.meta_learner, meta_df_val_np, y_val, self.optimal_threshold, self.logger
            )
        else:
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
        X_pytorch = X_prepared[self.pytorch_features] # Added PyTorch
        X_svm = X_prepared[self.svm_features]         # Added SVM
        X_fnn = X_prepared[self.fnn_features]         # Added FNN
        try:
            # Generate predictions
            p_xgb = self.model_xgb.predict_proba(X_xgb)[:, 1]
            p_lgb = self.model_lgb.predict_proba(X_lgb)[:, 1]
            p_extra = self.model_extra.predict_proba(X_extra)[:, 1]
            p_pytorch = self.model_pytorch.predict_proba(X_pytorch)[:, 1] 
            p_fnn = self.model_fnn.predict_proba(X_fnn)[:, 1]
            # Handle potential TabNet input type error
            try:
                p_tabnet = self.model_tabnet.predict_proba(X_tabnet)[:, 1]
            except TypeError:
                p_tabnet = self.model_tabnet.predict_proba(X_tabnet.values)[:, 1]
            
            # --- MLP Scaling and Prediction ---
            # Ensure X_mlp is a DataFrame with correct columns before transform
            if not isinstance(X_mlp, pd.DataFrame):
                self.logger.warning("X_mlp is not a DataFrame before scaling. Attempting conversion.")
                X_mlp = pd.DataFrame(X_mlp, columns=self.mlp_features)
            # Re-select columns just in case order changed or to ensure DataFrame type
            X_mlp = X_mlp[self.mlp_features]
            X_mlp_scaled = self.model_mlp_scaler.transform(X_mlp)
            p_mlp = self.model_mlp.predict_proba(X_mlp_scaled)[:, 1]

            # --- SVM Scaling and Prediction ---
            # Ensure X_svm is a DataFrame with correct columns before transform
            if not isinstance(X_svm, pd.DataFrame):
                self.logger.warning("X_svm is not a DataFrame before scaling. Attempting conversion.")
                X_svm = pd.DataFrame(X_svm, columns=self.svm_features)
            # Re-select columns
            X_svm = X_svm[self.svm_features]
            X_svm_scaled = self.model_svm_scaler.transform(X_svm)
            p_svm = self.model_svm.predict_proba(X_svm_scaled)[:, 1]

            meta_features = create_meta_features_optimized(
                p_xgb,
                p_tabnet,
                p_lgb,
                p_extra,
                p_mlp,  
                p_pytorch, 
                p_svm,          # Added SVM
                p_fnn,          # Added FNN
                self.dynamic_weights_train if self.dynamic_weighting else None,
                self.thresholds_train if self.dynamic_weighting else None,
            )
            # Ensure meta_features is a DataFrame before prediction if meta_learner expects it
            meta_df = create_meta_dataframe(meta_features)
            if self.meta_learner_type == "tabnet":
                meta_df_np = meta_df.to_numpy()
                meta_probs = self.meta_learner.predict_proba(meta_df_np)[:, 1]
            else:
                meta_probs = self.meta_learner.predict_proba(meta_df)
            # Handle both 1D and 2D probability arrays
            if len(meta_probs.shape) == 1:
                # Already 1D probabilities
                return meta_probs
            else:
                # Extract positive class probabilities from 2D array
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
        tabnet_path="model_sklearn",
        extra_path="model",  # Path for Extra Trees model
        mlp_path="model",  # Artifact path for MLP model
        mlp_scaler_path="scaler/scaler_mlp.pkl",  # Artifact path for MLP scaler
        pytorch_path="model",
        pytorch_scaler_path="scaler/scaler_pytorch.pkl",
        svm_path="model_svm",  # Artifact path for SVM model (from svm_model.py)
        svm_scaler_path="scaler_svm.pkl", # Artifact path for SVM scaler (default name in svm_model.py)
        fnn_path="model",
        fnn_scaler_path="scaler/scaler_pytorch.pkl",
    ):
        """
        Load pre-trained models (XGB, LGBM, TabNet, Extra Trees, MLP, PyTorch, SVM) 
        and scalers from MLflow.
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
                # Attempt to get features from the underlying sklearn model if possible
                self.mlp_features = getattr(self.model_mlp, "feature_names_in_", []) 

            # Load the associated MLP scaler
            self.logger.info(
                f"Loading MLP scaler artifact '{mlp_scaler_path}' from run {self.mlp_run_id}..."
            )
            scaler_local_path = mlflow.artifacts.download_artifacts(
                run_id=self.mlp_run_id, artifact_path=mlp_scaler_path
            )
            with open(scaler_local_path, "rb") as f:
                self.model_mlp_scaler = pickle.load(f)
            self.logger.info("MLP scaler loaded successfully.")
            if hasattr(self.model_mlp, 'scaler_') and hasattr(self.model_mlp, 'device_'):
                try: 
                    # Determine device (use CUDA if available, same logic as hypertuner)
                    pytorch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    self.model_mlp.scaler_ = self.model_mlp_scaler
                    self.model_mlp.device_ = pytorch_device
                    self.model_mlp.to(pytorch_device) # Ensure model is on the correct device
                    self.logger.info(f"Attached scaler and device ({pytorch_device}) to loaded MLP model.")
                except Exception as attach_e:
                    self.logger.warning(f"Could not attach scaler/device to MLP model: {attach_e}")
            else:
                self.logger.warning("Loaded MLP model does not have scaler_/device_ attributes for attachment.")
                # Ensure model is moved to the correct device anyway
                try:
                    pytorch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    self.model_mlp.to(pytorch_device)
                    self.logger.info(f"Moved loaded MLP model to device: {pytorch_device}")
                except Exception as move_e:
                    self.logger.error(f"Could not move MLP model to device: {move_e}")
        except Exception as e:
            self.logger.error(f"Failed to load MLP model or scaler: {str(e)}")
            raise ValueError(f"Failed to load MLP model or scaler: {str(e)}") from e
                
        # Load PyTorch model
        try:
            # Define artifact paths for PyTorch model and its scaler
            pytorch_model_path = "model" # Assuming artifact path is 'model'
            pytorch_scaler_path = "scaler/scaler_pytorch.pkl" # Assuming scaler saved in 'scaler' dir
            
            self.logger.info(f"Loading PyTorch model from run {self.pytorch_run_id}...")
            pytorch_uri = f"runs:/{self.pytorch_run_id}/{pytorch_model_path}"
            self.model_pytorch = mlflow.pytorch.load_model(pytorch_uri)
            
            # Load PyTorch model also as pyfunc to easily get signature
            pytorch_pyfunc = mlflow.pyfunc.load_model(pytorch_uri)
            if pytorch_pyfunc.metadata.signature and pytorch_pyfunc.metadata.signature.inputs:
                self.pytorch_features = pytorch_pyfunc.metadata.signature.inputs.input_names()
                self.logger.info(
                    f"Updated PyTorch feature signature: {len(self.pytorch_features)} features"
                )
            else:
                self.logger.warning("No feature signature found for PyTorch model.")
                self.pytorch_features = [] 

            # Load the associated PyTorch scaler
            self.logger.info(
                f"Loading PyTorch scaler artifact '{pytorch_scaler_path}' from run {self.pytorch_run_id}..."
            )
            scaler_local_path = mlflow.artifacts.download_artifacts(
                run_id=self.pytorch_run_id, artifact_path=pytorch_scaler_path
            )
            with open(scaler_local_path, "rb") as f:
                self.model_pytorch_scaler = pickle.load(f)
            self.logger.info("PyTorch scaler loaded successfully.")
            
            # Optional: Attach scaler and device to the loaded PyTorch model instance 
            # if its predict_proba method relies on them being attributes (like in the hypertuner)
            if hasattr(self.model_pytorch, 'scaler_') and hasattr(self.model_pytorch, 'device_'):
                try: 
                    # Determine device (use CUDA if available, same logic as hypertuner)
                    pytorch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    self.model_pytorch.scaler_ = self.model_pytorch_scaler
                    self.model_pytorch.device_ = pytorch_device
                    self.model_pytorch.to(pytorch_device) # Ensure model is on the correct device
                    self.logger.info(f"Attached scaler and device ({pytorch_device}) to loaded PyTorch model.")
                except Exception as attach_e:
                    self.logger.warning(f"Could not attach scaler/device to PyTorch model: {attach_e}")
            else:
                self.logger.warning("Loaded PyTorch model does not have scaler_/device_ attributes for attachment.")
                # Ensure model is moved to the correct device anyway
                try:
                    pytorch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    self.model_pytorch.to(pytorch_device)
                    self.logger.info(f"Moved loaded PyTorch model to device: {pytorch_device}")
                except Exception as move_e:
                    self.logger.error(f"Could not move PyTorch model to device: {move_e}")
        except Exception as e:
            self.logger.error(f"Failed to load PyTorch model or scaler: {str(e)}")
            raise ValueError(f"Failed to load PyTorch model or scaler: {str(e)}") from e

        # Load SVM (sklearn) model
        try:
            self.logger.info(f"Loading SVM model from run {self.svm_run_id}...")
            svm_uri = f"runs:/{self.svm_run_id}/{svm_path}"
            self.model_svm = mlflow.sklearn.load_model(svm_uri)
            svm_pyfunc = mlflow.pyfunc.load_model(svm_uri)
            if svm_pyfunc.metadata.signature and svm_pyfunc.metadata.signature.inputs:
                self.svm_features = svm_pyfunc.metadata.signature.inputs.input_names()
                self.logger.info(
                    f"Updated SVM feature signature: {len(self.svm_features)} features"
                )
            else:
                self.logger.warning("No feature signature found for SVM model")
                self.svm_features = getattr(self.model_svm, "feature_names_in_", [])

            # Load the associated SVM scaler
            self.logger.info(
                f"Loading SVM scaler artifact '{svm_scaler_path}' from run {self.svm_run_id}..."
            )
            scaler_local_path = mlflow.artifacts.download_artifacts(
                run_id=self.svm_run_id, artifact_path=svm_scaler_path
            )
            with open(scaler_local_path, "rb") as f:
                self.model_svm_scaler = pickle.load(f)
            self.logger.info("SVM scaler loaded successfully.")
            # Verify scaler type (optional)
            if not isinstance(self.model_svm_scaler, (StandardScaler, sklearn.preprocessing.RobustScaler)): # Add other expected scaler types if needed
                self.logger.warning(f"Loaded SVM scaler is of unexpected type: {type(self.model_svm_scaler).__name__}")
        except Exception as e:
            self.logger.error(f"Failed to load SVM model or scaler: {str(e)}")
            raise ValueError(f"Failed to load SVM model or scaler: {str(e)}") from e

        # Load FNN model
        try:
            self.logger.info(f"Loading FNN model from run {self.fnn_run_id}...")
            fnn_uri = f"runs:/{self.fnn_run_id}/{fnn_path}"
            self.model_fnn = mlflow.pytorch.load_model(fnn_uri)
            
            # Load PyTorch model also as pyfunc to easily get signature
            fnn_pyfunc = mlflow.pyfunc.load_model(fnn_uri)
            if fnn_pyfunc.metadata.signature and fnn_pyfunc.metadata.signature.inputs:
                self.fnn_features = fnn_pyfunc.metadata.signature.inputs.input_names()
                self.logger.info(
                    f"Updated FNN feature signature: {len(self.fnn_features)} features"
                )
            else:
                self.logger.warning("No feature signature found for FNN model.")
                self.fnn_features = [] 

            # Load the associated PyTorch scaler
            self.logger.info(
                f"Loading FNN scaler artifact '{fnn_scaler_path}' from run {self.fnn_run_id}..."
            )
            scaler_local_path = mlflow.artifacts.download_artifacts(
                run_id=self.fnn_run_id, artifact_path=fnn_scaler_path
            )
            with open(scaler_local_path, "rb") as f:
                self.model_fnn_scaler = pickle.load(f)
            self.logger.info("FNN scaler loaded successfully.")
            
            # Optional: Attach scaler and device to the loaded PyTorch model instance 
            # if its predict_proba method relies on them being attributes (like in the hypertuner)
            if hasattr(self.model_pytorch, 'scaler_') and hasattr(self.model_pytorch, 'device_'):
                try: 
                    # Determine device (use CUDA if available, same logic as hypertuner)
                    pytorch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    self.model_fnn.scaler_ = self.model_fnn_scaler
                    self.model_fnn.device_ = pytorch_device
                    self.model_fnn.to(pytorch_device) # Ensure model is on the correct device
                    self.logger.info(f"Attached scaler and device ({pytorch_device}) to loaded FNN model.")
                except Exception as attach_e:
                    self.logger.warning(f"Could not attach scaler/device to FNN model: {attach_e}")
            else:
                self.logger.warning("Loaded FNN model does not have scaler_/device_ attributes for attachment.")
                # Ensure model is moved to the correct device anyway
                try:
                    pytorch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    self.model_fnn.to(pytorch_device)
                    self.logger.info(f"Moved loaded FNN model to device: {pytorch_device}")
                except Exception as move_e:
                    self.logger.error(f"Could not move FNN model to device: {move_e}")
        except Exception as e:
            self.logger.error(f"Failed to load FNN model or scaler: {str(e)}")
            raise ValueError(f"Failed to load FNN model or scaler: {str(e)}") from e

        self.logger.info("Base models loading complete.")
        # Consider setting self.selected_features based on intersection or a specific model
        # self.selected_features = self.xgb_features
        return True
