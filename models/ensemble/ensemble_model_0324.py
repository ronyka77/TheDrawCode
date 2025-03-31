"""EnsembleModel Class

This file implements the ensemble model using updated base models:
- Base models: XGBoost, TabNet, and LightGBM
- Extra models: extended with CatBoost option

Author: Updated by AI assistant
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from pathlib import Path
import os
import sys
import json
import random
import time
import torch
from pytorch_tabnet.tab_model import TabNetClassifier


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
torch.use_deterministic_algorithms(True)  # Force deterministic algorithms
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from utils.logger import ExperimentLogger
from utils.create_evaluation_set import import_selected_features_ensemble, setup_mlflow_tracking
from models.ensemble.data_utils import prepare_data
from models.ensemble.meta_features import create_meta_features_optimized, create_meta_dataframe
from models.ensemble.diagnostics import explain_predictions, analyze_prediction_errors
from models.ensemble.training import train_base_models, hypertune_meta_learner, initialize_meta_learner
from models.ensemble.weights import compute_precision_focused_weights_optimized
from models.ensemble.thresholds import tune_threshold_for_precision_optimized
from models.ensemble.evaluation import evaluate_model

class EnsembleModel(BaseEstimator, ClassifierMixin):
    def __init__(self, 
                    logger=None, 
                    calibrate=False, 
                    calibration_method="sigmoid", 
                    individual_thresholding=False,
                    meta_learner_type='xgb', 
                    dynamic_weighting=True, 
                    extra_base_model_type='random_forest',
                    sampling_strategy=0.7, 
                    complexity_penalty=0.01, 
                    target_precision=0.50, 
                    required_recall=0.25, 
                    X_train=None):
        self.logger = logger or ExperimentLogger(experiment_name="ensemble_model_improved", log_dir="./logs/ensemble_model_improved")
        # Load selected features (for all models)
        # self.selected_features = import_selected_features_ensemble('all')
        self.required_recall = required_recall
        self.sampling_strategy = sampling_strategy
        self.complexity_penalty = complexity_penalty
        self.target_precision = target_precision
        self.xgb_run_id = 'ddb71abe2f0e4f91a3e5ffc3e4c10ffb'
        self.lgb_run_id = '8c33c9c3805449cb9048475e23d27914'
        self.tabnet_run_id = '7188f690235741fe83f872d44b229f6e'
        self.rf_run_id = 'f7be1d96055445b3abc1d7cb76033da2'
        self.min_recalls = [0.30, 0.20, 0.40, 0.40]
        # Meta-learner settings
        self.meta_learner_type = meta_learner_type
        self.optimal_threshold = 0.5
        self.individual_thresholding = individual_thresholding
        self.calibrate = calibrate
        self.calibration_method = calibration_method
        self.dynamic_weighting = dynamic_weighting
        if self.dynamic_weighting:
            self.dynamic_weights = {'xgb': 1/4, 'tabnet': 1/4, 'lgb': 1/4, 'extra': 1/4}
        self.meta_learner = None
        self.model_xgb_calibrated = None
        self.model_lgb_calibrated = None
        self.model_extra_calibrated = None
        self.extra_model_scaler = None
        self.load_models_from_mlflow()

    def train(self, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None, split_validation=True, val_size=0.2) -> dict:
        self.logger.info("Starting ensemble model training...")
        # Data preparation
        self.selected_features = X_train.columns
        X_train_prepared = prepare_data(X_train, self.selected_features)
        if X_val is not None:
            X_val_prepared = prepare_data(X_val, self.selected_features)
        if X_test is not None:
            X_test_prepared = prepare_data(X_test, self.selected_features)
        if split_validation or X_val is None or y_val is None:
            from sklearn.model_selection import train_test_split
            self.logger.info("Splitting training data for validation...")
            X_train_prepared, X_val_prepared, y_train, y_val = train_test_split(X_train_prepared, y_train, test_size=val_size, random_state=19, stratify=y_train)
        
        # X_combined = pd.concat([X_train_prepared, X_test_prepared], axis=0)
        # y_combined = pd.concat([y_train, y_test], axis=0)
        # Base models dictionary using updated keys
        base_models = {
            'xgb': self.model_xgb,
            'tabnet': self.model_tabnet,
            'lgb': self.model_lgb,
            'extra': self.model_extra
        } 
        # Prepare feature subsets
        X_val_prepared_xgb = X_val_prepared[self.xgb_features]
        X_val_prepared_tabnet = X_val_prepared[self.tabnet_features]
        X_val_prepared_lgb = X_val_prepared[self.lgb_features]
        X_val_prepared_rf = X_val_prepared[self.rf_features]
        X_train_prepared_xgb = X_train_prepared[self.xgb_features]
        X_train_prepared_tabnet = X_train_prepared[self.tabnet_features]
        X_train_prepared_lgb = X_train_prepared[self.lgb_features]
        X_train_prepared_rf = X_train_prepared[self.rf_features]
        X_test_prepared_xgb = X_test_prepared[self.xgb_features]
        X_test_prepared_tabnet = X_test_prepared[self.tabnet_features]
        X_test_prepared_lgb = X_test_prepared[self.lgb_features]
        X_test_prepared_rf = X_test_prepared[self.rf_features]

        # Obtain predictions from base models
        self.logger.info("Obtaining predictions from base models...")
        p_xgb = self.model_xgb.predict_proba(X_val_prepared_xgb)[:, 1]
        p_xgb_test = self.model_xgb.predict_proba(X_test_prepared_xgb)[:, 1]
        p_tabnet = self.model_tabnet.predict_proba(X_val_prepared_tabnet.values)[:, 1]
        p_tabnet_test = self.model_tabnet.predict_proba(X_test_prepared_tabnet.values)[:, 1]
        p_lgb = self.model_lgb.predict_proba(X_val_prepared_lgb)[:, 1]
        p_lgb_test = self.model_lgb.predict_proba(X_test_prepared_lgb)[:, 1]
        # Extra model predictions
        if self.extra_base_model_type in ['mlp', 'svm'] and self.extra_model_scaler is not None:
            X_val_scaled = self.extra_model_scaler.transform(X_val_prepared)
            X_test_scaled = self.extra_model_scaler.transform(X_test_prepared)
            if self.extra_base_model_type == 'mlp':
                p_extra = self.model_extra.predict(X_val_scaled, verbose=0).flatten()
                p_extra_test = self.model_extra.predict(X_test_scaled, verbose=0).flatten()
            else:
                p_extra = self.model_extra.predict_proba(X_val_scaled)[:, 1]
                p_extra_test = self.model_extra.predict_proba(X_test_scaled)[:, 1]
        else:
            p_extra = self.model_extra.predict_proba(X_val_prepared_rf)[:, 1]
            p_extra_test = self.model_extra.predict_proba(X_test_prepared_rf)[:, 1]
        
        # Optionally calculate dynamic weights based on validation performance
        if self.dynamic_weighting:
            self.logger.info("Computing dynamic weights based on validation performance...")
            # Combine validation and training predictions for weight computation
            self.logger.info(f"Combined dataset for weight computation: {len(p_xgb)} samples")
            self.dynamic_weights, self.thresholds = compute_precision_focused_weights_optimized(
                p_xgb, p_tabnet, p_lgb, p_extra, y_val, self.target_precision, self.min_recalls, self.logger
            )
            self.dynamic_weights_train, self.thresholds_train = compute_precision_focused_weights_optimized(
                p_xgb_test, p_tabnet_test, p_lgb_test, p_extra_test, y_test, self.target_precision, self.min_recalls, self.logger
            )
        # Create meta-features from base model predictions
        self.logger.info("Creating meta-features for meta-learner...")
        meta_features = create_meta_features_optimized(
            p_xgb, p_tabnet, p_lgb, p_extra, self.dynamic_weights if self.dynamic_weighting else None, self.thresholds if self.dynamic_weighting else None
        )
        meta_features_train = create_meta_features_optimized(
            p_xgb_test, p_tabnet_test, p_lgb_test, p_extra_test, self.dynamic_weights_train if self.dynamic_weighting else None, self.thresholds_train if self.dynamic_weighting else None
        )
        # Convert to DataFrame for better interpretability
        meta_df = create_meta_dataframe(meta_features)
        meta_df_train = create_meta_dataframe(meta_features_train)

        # Initialize and train meta-learner
        self.logger.info(f"Initializing meta-learner of type {self.meta_learner_type}...")
        self.meta_learner = initialize_meta_learner(self.meta_learner_type)

        # Train meta-learner
        self.logger.info("Training meta-learner...")
        self.meta_learner = hypertune_meta_learner(meta_df_train, y_test, meta_df, y_val, 
                                                    meta_learner_type=self.meta_learner_type, target_precision=self.target_precision, min_recall=self.required_recall)
        # Tune threshold for optimal precision-recall trade-off
        self.logger.info(f"Tuning threshold for target precision {self.target_precision}...")
        # Get meta-learner predictions on validation data
        meta_val_probs = self.meta_learner.predict_proba(meta_df)[:, 1]
        # Tune threshold
        best_threshold, threshold_metrics = tune_threshold_for_precision_optimized(
            meta_val_probs, y_val, 
            target_precision=self.target_precision,
            required_recall=self.required_recall,
            logger=self.logger
        )
        
        self.optimal_threshold = best_threshold
        threshold_metrics = threshold_metrics
        self.logger.info(f"Optimal threshold set to {self.optimal_threshold:.4f}")
        
        # Final evaluation on validation data
        self.logger.info("Performing final evaluation on validation data...")
        eval_results = evaluate_model(
            self.meta_learner, meta_df, y_val, self.optimal_threshold, self.logger
        )
        self.logger.info("Ensemble model training completed successfully.")
        
        return eval_results

    def predict_proba(self, X) -> np.ndarray:
        if self.meta_learner is None:
            raise ValueError("Model has not been trained. Call train() first.")
        X_prepared = prepare_data(X, X.columns)
        X_prepared_xgb = X_prepared[self.xgb_features]
        X_prepared_tabnet = X_prepared[self.tabnet_features]
        X_prepared_lgb = X_prepared[self.lgb_features]
        X_prepared_rf = X_prepared[self.rf_features]
        try:
            if self.extra_base_model_type in ['mlp', 'svm'] and self.extra_model_scaler is not None:
                X_scaled = self.extra_model_scaler.transform(X_prepared)
                if self.extra_base_model_type == 'mlp':
                    p_extra = self.model_extra.predict(X_scaled, verbose=0).flatten()
                else:
                    p_extra = self.model_extra.predict_proba(X_scaled)[:, 1]
            else:
                p_extra = self.model_extra.predict_proba(X_prepared_rf)[:, 1]
            xgb_model = self.model_xgb
            tabnet_model = self.model_tabnet
            lgb_model = self.model_lgb
            p_xgb = xgb_model.predict_proba(X_prepared_xgb)[:, 1]
            p_tabnet = tabnet_model.predict_proba(X_prepared_tabnet.values)[:, 1]
            p_lgb = lgb_model.predict_proba(X_prepared_lgb)[:, 1]
            meta_features = create_meta_features_optimized(p_xgb, p_tabnet, p_lgb, p_extra, self.dynamic_weights if self.dynamic_weighting else None, self.thresholds if self.thresholds else None)
            meta_probs = self.meta_learner.predict_proba(meta_features)
            return meta_probs[:, 1]
        except Exception as e:
            self.logger.error(f"Error predicting probabilities: {e}")
            raise

    def predict(self, X) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return (probabilities >= self.optimal_threshold).astype(int)

    def explain_predictions(self, X_val) -> dict:
        return explain_predictions(self, X_val, self.logger)

    def analyze_prediction_errors(self, X_val, y_val) -> dict:
        return analyze_prediction_errors(self, X_val, y_val, self.optimal_threshold, self.logger)

    def precision_filter(self, X, probabilities):
        high_conf = probabilities > self.optimal_threshold
        X_high_conf = X[high_conf]
        if 'home_form' in X_high_conf.columns and 'away_form' in X_high_conf.columns:
            form_diff = abs(X_high_conf['home_form'] - X_high_conf['away_form'])
            likely_not_draw = form_diff > 0.5
            high_conf[high_conf] = ~likely_not_draw
        return high_conf

    def get_model_params(self, model):
        try:
            if hasattr(model, "get_params"):
                return model.get_params()
            elif hasattr(model, "get_config"):
                return model.get_config()
            else:
                return json.loads(model.to_json())
        except Exception as e:
            return {"error": str(e)}

    def load_models_from_mlflow(self, 
                            xgb_path="model",
                            lgb_path="model", 
                            tabnet_path="model",
                            rf_path="model"):
        """
        Load pre-trained models from MLflow repository and update feature signatures.
        
        Args:
            xgb_path: Artifact path for XGBoost model within run
            lgb_path: Artifact path for LightGBM model within run
            tabnet_path: Artifact path for TabNet model within run
            rf_path: Artifact path for Random Forest model within run
            
        Raises:
            ValueError: If any model loading fails
        """
        import mlflow
        import mlflow.xgboost
        import mlflow.lightgbm
        import mlflow.sklearn
        import mlflow.pyfunc
        
        self.logger.info("Loading models from MLflow repository...")
        
        # Load XGBoost model
        try:
            self.logger.info(f"Loading XGBoost model from run {self.xgb_run_id}...")
            # Use direct artifact URI format instead of get_latest_versions
            xgb_uri = f"runs:/{self.xgb_run_id}/{xgb_path}"
            self.model_xgb = mlflow.xgboost.load_model(xgb_uri)
            
            # Also load as pyfunc to access metadata/signature
            xgb_pyfunc = mlflow.pyfunc.load_model(xgb_uri)
            
            # Extract feature signature from loaded model
            if xgb_pyfunc.metadata.signature and xgb_pyfunc.metadata.signature.inputs:
                self.xgb_features = xgb_pyfunc.metadata.signature.inputs.input_names()
                self.logger.info(f"Updated XGBoost feature signature with {len(self.xgb_features)} features")
            else:
                self.logger.warning("No feature signature found for XGBoost model")
                # Try to get feature names directly from the model
                if hasattr(self.model_xgb, 'feature_names'):
                    self.xgb_features = self.model_xgb.feature_names
                    self.logger.info(f"Retrieved {len(self.xgb_features)} feature names directly from XGBoost model")
        except Exception as e:
            self.logger.error(f"Failed to load XGBoost model: {str(e)}")
            raise ValueError(f"Failed to load XGBoost model: {str(e)}")
        
        # Load LightGBM model
        try:
            self.logger.info(f"Loading LightGBM model from run {self.lgb_run_id}...")
            # Use direct artifact URI format
            lgb_uri = f"runs:/{self.lgb_run_id}/{lgb_path}"
            self.model_lgb = mlflow.lightgbm.load_model(lgb_uri)
            
            # Also load as pyfunc to access metadata/signature
            lgb_pyfunc = mlflow.pyfunc.load_model(lgb_uri)
            
            # Extract feature signature from loaded model
            if lgb_pyfunc.metadata.signature and lgb_pyfunc.metadata.signature.inputs:
                self.lgb_features = lgb_pyfunc.metadata.signature.inputs.input_names()
                self.logger.info(f"Updated LightGBM feature signature with {len(self.lgb_features)} features")
            else:
                self.logger.warning("No feature signature found for LightGBM model")
                # Try to get feature names directly from the model
                if hasattr(self.model_lgb, 'feature_name_'):
                    self.lgb_features = self.model_lgb.feature_name_
                    self.logger.info(f"Retrieved {len(self.lgb_features)} feature names directly from LightGBM model")
        except Exception as e:
            self.logger.error(f"Failed to load LightGBM model: {str(e)}")
            raise ValueError(f"Failed to load LightGBM model: {str(e)}")
        
        # Load TabNet model (using sklearn flavor since TabNet is saved as sklearn)
        try:
            self.logger.info(f"Loading TabNet model from run {self.tabnet_run_id}...")
            # Use direct artifact URI format
            tabnet_uri = f"runs:/{self.tabnet_run_id}/{tabnet_path}"
            
            # Load TabNet model using sklearn flavor
            self.model_tabnet = mlflow.sklearn.load_model(tabnet_uri)
            
            # Also load as pyfunc to access metadata/signature
            tabnet_pyfunc = mlflow.pyfunc.load_model(tabnet_uri)
            
            # Extract feature signature from loaded model
            if tabnet_pyfunc.metadata.signature and tabnet_pyfunc.metadata.signature.inputs:
                self.tabnet_features = tabnet_pyfunc.metadata.signature.inputs.input_names()
                self.logger.info(f"Updated TabNet feature signature with {len(self.tabnet_features)} features")
            else:
                self.logger.warning("No feature signature found for TabNet model")
                # Try to get feature names directly from the model
                if hasattr(self.model_tabnet, 'feature_names_in_'):
                    self.tabnet_features = self.model_tabnet.feature_names_in_
                    self.logger.info(f"Retrieved {len(self.tabnet_features)} feature names directly from TabNet model")
                elif hasattr(self.model_tabnet, 'input_dim'):
                    feature_count = self.model_tabnet.input_dim
                    self.logger.warning(f"Using generic feature names for TabNet ({feature_count} features)")
                    self.tabnet_features = [f"feature_{i}" for i in range(feature_count)]
        except Exception as e:
            self.logger.error(f"Failed to load TabNet model: {str(e)}")
            raise ValueError(f"Failed to load TabNet model: {str(e)}")
        
        # Load Random Forest model
        try:
            self.logger.info(f"Loading Random Forest model from run {self.rf_run_id}...")
            # Use direct artifact URI format
            rf_uri = f"runs:/{self.rf_run_id}/{rf_path}"
            
            # Load RF model using sklearn flavor
            self.model_extra = mlflow.sklearn.load_model(rf_uri)
            
            # Also load as pyfunc to access metadata/signature
            rf_pyfunc = mlflow.pyfunc.load_model(rf_uri)
            
            # Extract feature signature from loaded model
            if rf_pyfunc.metadata.signature and rf_pyfunc.metadata.signature.inputs:
                self.rf_features = rf_pyfunc.metadata.signature.inputs.input_names()
                self.logger.info(f"Updated Random Forest feature signature with {len(self.rf_features)} features")
            else:
                self.logger.warning("No feature signature found for Random Forest model")
                # Try to get feature names directly from the model
                if hasattr(self.model_extra, 'feature_names_in_'):
                    self.rf_features = self.model_extra.feature_names_in_
                    self.logger.info(f"Retrieved {len(self.rf_features)} feature names directly from Random Forest model")
        except Exception as e:
            self.logger.error(f"Failed to load Random Forest model: {str(e)}")
            raise ValueError(f"Failed to load Random Forest model: {str(e)}")
        
        # Update extra_base_model_type to reflect the loaded model
        self.extra_base_model_type = 'random_forest'
        
        self.logger.info("All models successfully loaded from MLflow")
        self.selected_features = self.xgb_features  # Use XGBoost features as default selected features
        return True 