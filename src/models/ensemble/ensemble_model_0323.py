"""EnsembleModel Class

This file implements the ensemble model using updated base models:
- Base models: XGBoost, TabNet, and LightGBM
- Extra models: extended with CatBoost option

Author: Updated by AI assistant
"""

import json
import os
import random

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier

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

from src.models.ensemble.data_utils import prepare_data
from src.models.ensemble.diagnostics import analyze_prediction_errors, explain_predictions
from src.models.ensemble.evaluation import evaluate_model
from src.models.ensemble.meta_features import create_meta_dataframe, create_meta_features
from src.models.ensemble.thresholds import tune_threshold_for_precision
from src.models.ensemble.training import (
    hypertune_meta_learner,
    initialize_meta_learner,
    train_base_models,
)
from src.models.ensemble.weights import compute_precision_focused_weights
from src.utils.create_evaluation_set import import_selected_features_ensemble
from src.utils.logger import ExperimentLogger


class EnsembleModel(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        logger=None,
        calibrate=False,
        calibration_method="sigmoid",
        individual_thresholding=False,
        meta_learner_type="xgb",
        dynamic_weighting=True,
        extra_base_model_type="random_forest",
        sampling_strategy=0.7,
        complexity_penalty=0.01,
        target_precision=0.50,
        required_recall=0.25,
        X_train=None,
    ):
        self.logger = logger or ExperimentLogger(
            experiment_name="ensemble_model_improved", log_dir="./logs/ensemble_model_improved"
        )
        # Load selected features (for all models)
        self.selected_features = import_selected_features_ensemble("all")
        self.required_recall = required_recall
        self.sampling_strategy = sampling_strategy
        self.complexity_penalty = complexity_penalty
        self.target_precision = target_precision
        self.xgb_run_id = "4acba75e42b54fca8b716ec652eab968"
        self.lgb_run_id = "7f519dc3398e4a6ab7bf8409e936ad37"
        self.tabnet_run_id = "4e443703c98540858f303495f9ebca3a"
        self.rf_run_id = "5ee5fab32da74783944da445e3a20bb6"
        # Initialize base models: XGBoost, TabNet, LightGBM
        self.model_xgb = XGBClassifier(
            tree_method="hist",
            device="cpu",
            nthread=4,
            objective="binary:logistic",
            eval_metric=["aucpr", "error", "logloss"],
            verbosity=0,
            learning_rate=0.06,
            max_depth=12,
            min_child_weight=400,
            subsample=0.77,
            colsample_bytree=0.82,
            reg_alpha=53.8,
            reg_lambda=4.91,
            gamma=0.88,
            early_stopping_rounds=670,
            scale_pos_weight=2.28,
            seed=19,
        )

        self.model_tabnet = TabNetClassifier(
            optimizer_fn=torch.optim.Adam,
            optimizer_params={"lr": 0.05868852179579677},
            n_d=12,
            n_a=15,
            n_steps=9,
            gamma=1.5,
            lambda_sparse=6.883596605997507e-05,
            momentum=0.8600000000000001,
            mask_type="sparsemax",
            device_name="cpu",
            verbose=0,
            seed=19,
        )

        self.model_lgb = LGBMClassifier(
            objective="binary",
            metric=["binary_logloss", "auc"],
            verbose=-1,
            n_jobs=4,
            random_state=19,
            device="cpu",
            learning_rate=0.125,
            num_leaves=135,
            max_depth=5,
            min_child_samples=300,
            feature_fraction=0.67,
            bagging_fraction=0.65,
            bagging_freq=15,
            reg_alpha=7.1000000000000005,
            reg_lambda=10.9,
            min_split_gain=0.15000000000000002,
            early_stopping_rounds=650,
            path_smooth=0.13,
            cat_smooth=18.1,
            max_bin=570,
        )

        # Set feature sets; use xgboost features for tabnet as fallback
        self.xgb_features = import_selected_features_ensemble(model_type="xgb")
        self.tabnet_features = import_selected_features_ensemble(model_type="tabnet")
        self.lgb_features = import_selected_features_ensemble(model_type="lgbm")
        self.rf_features = import_selected_features_ensemble(model_type="rf")

        # Initialize extra model options, extended with CatBoost option
        self.extra_base_model_type = extra_base_model_type.lower()
        if self.extra_base_model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier

            self.model_extra = RandomForestClassifier(
                n_estimators=540,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=32,
                max_features=0.18,
                bootstrap=True,
                class_weight={0: 1.0, 1: 2.2},
                criterion="entropy",
                random_state=19,
                n_jobs=4,
            )
            self.logger.info("Extra base model initialized as RandomForestClassifier.")
        elif self.extra_base_model_type == "svm":
            from sklearn.svm import SVC

            self.model_extra = SVC(
                probability=True,
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced",
                random_state=19,
            )
            self.logger.info("Extra base model initialized as SVC.")
        elif self.extra_base_model_type == "mlp":
            from tensorflow import keras
            from tensorflow.keras import layers, regularizers

            self.model_extra = keras.Sequential()
            self.model_extra.add(layers.InputLayer(shape=(X_train.shape[1],)))
            for _ in range(1):
                self.model_extra.add(
                    layers.Dense(
                        120,
                        activation="tanh",
                        kernel_regularizer=regularizers.l1_l2(
                            l1=0.0006252292488020048, l2=0.0010179804312458536
                        ),
                    )
                )
                self.model_extra.add(layers.BatchNormalization())
                self.model_extra.add(layers.Dropout(0.2685783444324335))
            self.model_extra.add(layers.Dense(1, activation="sigmoid"))
            optimizer = keras.optimizers.Adam(
                learning_rate=0.0008586241362721754, beta_1=0.9, beta_2=0.999, epsilon=1e-8
            )
            self.model_extra.compile(
                optimizer=optimizer,
                loss="binary_crossentropy",
                metrics=["accuracy", keras.metrics.AUC(name="auc")],
            )
            self.logger.info("Extra base model initialized as MLPClassifier.")
        elif self.extra_base_model_type == "catboost":
            from catboost import CatBoostClassifier

            self.model_extra = CatBoostClassifier(
                learning_rate=0.021,
                depth=6,
                min_data_in_leaf=77,
                subsample=0.57,
                colsample_bylevel=0.50,
                reg_lambda=2.1,
                early_stopping_rounds=480,
                loss_function="Logloss",
                eval_metric="AUC",
                task_type="CPU",
                thread_count=4,
                verbose=-1,
            )
            self.logger.info("Extra base model initialized as CatBoostClassifier.")
        else:
            raise ValueError(f"Unknown extra_base_model_type: {self.extra_base_model_type}")

        # Meta-learner settings
        self.meta_learner_type = meta_learner_type
        self.optimal_threshold = 0.5
        self.individual_thresholding = individual_thresholding
        self.calibrate = calibrate
        self.calibration_method = calibration_method
        self.dynamic_weighting = dynamic_weighting
        if self.dynamic_weighting:
            self.dynamic_weights = {"xgb": 1 / 3, "tabnet": 1 / 3, "lgb": 1 / 3}
        self.meta_learner = None
        self.model_xgb_calibrated = None
        self.model_lgb_calibrated = None
        self.model_extra_calibrated = None
        self.extra_model_scaler = None

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
        self.logger.info("Starting ensemble model training...")
        # Data preparation
        X_train_prepared = prepare_data(X_train, self.selected_features)
        if X_val is not None:
            X_val_prepared = prepare_data(X_val, self.selected_features)
        if X_test is not None:
            X_test_prepared = prepare_data(X_test, self.selected_features)
        if split_validation or X_val is None or y_val is None:
            from sklearn.model_selection import train_test_split

            self.logger.info("Splitting training data for validation...")
            X_train_prepared, X_val_prepared, y_train, y_val = train_test_split(
                X_train_prepared, y_train, test_size=val_size, random_state=19, stratify=y_train
            )

        X_combined = pd.concat([X_train_prepared, X_test_prepared], axis=0)
        y_combined = pd.concat([y_train, y_test], axis=0)
        # Base models dictionary using updated keys
        base_models = {
            "xgb": self.model_xgb,
            "tabnet": self.model_tabnet,
            "lgb": self.model_lgb,
            "extra": self.model_extra,
        }
        trained_models = train_base_models(
            base_models, X_train_prepared, y_train, X_test_prepared, y_test, X_val_prepared, y_val
        )
        self.model_xgb = trained_models.get("xgb", self.model_xgb)
        self.model_tabnet = trained_models.get("tabnet", self.model_tabnet)
        self.model_lgb = trained_models.get("lgb", self.model_lgb)
        self.model_extra = trained_models.get("extra", self.model_extra)
        # Prepare feature subsets
        X_val_prepared_xgb = X_val_prepared[self.xgb_features]
        X_val_prepared_tabnet = X_val_prepared[self.tabnet_features]
        X_val_prepared_lgb = X_val_prepared[self.lgb_features]
        X_val_prepared_rf = X_val_prepared[self.rf_features]
        X_combined_xgb = X_combined[self.xgb_features]
        X_combined_tabnet = X_combined[self.tabnet_features]
        X_combined_lgb = X_combined[self.lgb_features]
        X_combined_rf = X_combined[self.rf_features]

        # Obtain predictions from base models
        self.logger.info("Obtaining predictions from base models...")
        p_xgb = self.model_xgb.predict_proba(X_val_prepared_xgb)[:, 1]
        p_xgb_train = self.model_xgb.predict_proba(X_combined_xgb)[:, 1]
        p_tabnet = self.model_tabnet.predict_proba(X_val_prepared_tabnet.values)[:, 1]
        p_tabnet_train = self.model_tabnet.predict_proba(X_combined_tabnet.values)[:, 1]
        p_lgb = self.model_lgb.predict_proba(X_val_prepared_lgb)[:, 1]
        p_lgb_train = self.model_lgb.predict_proba(X_combined_lgb)[:, 1]
        # Extra model predictions
        if self.extra_base_model_type in ["mlp", "svm"] and self.extra_model_scaler is not None:
            X_val_scaled = self.extra_model_scaler.transform(X_val_prepared)
            if self.extra_base_model_type == "mlp":
                p_extra = self.model_extra.predict(X_val_scaled, verbose=0).flatten()
            else:
                p_extra = self.model_extra.predict_proba(X_val_scaled)[:, 1]
        else:
            p_extra = self.model_extra.predict_proba(X_val_prepared_rf)[:, 1]
            p_extra_train = self.model_extra.predict_proba(X_combined_rf)[:, 1]

        # Optionally calculate dynamic weights based on validation performance
        if self.dynamic_weighting:
            self.logger.info("Computing dynamic weights based on validation performance...")
            # Combine validation and training predictions for weight computation
            self.logger.info(f"Combined dataset for weight computation: {len(p_xgb)} samples")
            self.dynamic_weights = compute_precision_focused_weights(
                p_lgb,
                p_tabnet,
                p_lgb,
                p_extra,
                y_val,
                self.target_precision,
                self.required_recall,
                self.logger,
            )
            self.dynamic_weights_train = compute_precision_focused_weights(
                p_xgb_train,
                p_tabnet_train,
                p_lgb_train,
                p_extra_train,
                y_combined,
                self.target_precision,
                self.required_recall,
                self.logger,
            )
        # Create meta-features from base model predictions
        self.logger.info("Creating meta-features for meta-learner...")
        meta_features = create_meta_features(
            p_xgb,
            p_tabnet,
            p_lgb,
            p_extra,
            self.dynamic_weights if self.dynamic_weighting else None,
        )
        meta_features_train = create_meta_features(
            p_xgb_train,
            p_tabnet_train,
            p_lgb_train,
            p_extra_train,
            self.dynamic_weights_train if self.dynamic_weighting else None,
        )
        # Convert to DataFrame for better interpretability
        meta_df = create_meta_dataframe(meta_features)
        meta_df_train = create_meta_dataframe(meta_features_train)

        # Initialize and train meta-learner
        self.logger.info(f"Initializing meta-learner of type {self.meta_learner_type}...")
        self.meta_learner = initialize_meta_learner(self.meta_learner_type)

        # Train meta-learner
        self.logger.info("Training meta-learner...")
        self.meta_learner = hypertune_meta_learner(
            meta_df_train,
            y_combined,
            meta_df,
            y_val,
            meta_learner_type=self.meta_learner_type,
            target_precision=self.target_precision,
            min_recall=self.required_recall,
        )
        # Tune threshold for optimal precision-recall trade-off
        self.logger.info(f"Tuning threshold for target precision {self.target_precision}...")
        # Get meta-learner predictions on validation data
        meta_val_probs = self.meta_learner.predict_proba(meta_df)[:, 1]
        # Tune threshold
        best_threshold, threshold_metrics = tune_threshold_for_precision(
            meta_val_probs,
            y_val,
            target_precision=self.target_precision,
            required_recall=self.required_recall,
            logger=self.logger,
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
            if self.extra_base_model_type in ["mlp", "svm"] and self.extra_model_scaler is not None:
                X_scaled = self.extra_model_scaler.transform(X_prepared)
                if self.extra_base_model_type == "mlp":
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
            meta_features = create_meta_features(
                p_xgb,
                p_tabnet,
                p_lgb,
                p_extra,
                self.dynamic_weights if self.dynamic_weighting else None,
            )
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
        if "home_form" in X_high_conf.columns and "away_form" in X_high_conf.columns:
            form_diff = abs(X_high_conf["home_form"] - X_high_conf["away_form"])
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

    def load_models_from_mlflow(
        self,
        xgb_run_id=None,
        lgb_run_id=None,
        tabnet_run_id=None,
        rf_run_id=None,
        xgb_path="model",
        lgb_path="model",
        tabnet_path="model",
        rf_path="model",
    ):
        """
        Load pre-trained models from MLflow repository and update feature signatures.

        Args:
            xgb_run_id: MLflow run ID for XGBoost model (required)
            lgb_run_id: MLflow run ID for LightGBM model (required)
            tabnet_run_id: MLflow run ID for TabNet model (required)
            rf_run_id: MLflow run ID for Random Forest model (required)
            xgb_path: Artifact path for XGBoost model within run
            lgb_path: Artifact path for LightGBM model within run
            tabnet_path: Artifact path for TabNet model within run
            rf_path: Artifact path for Random Forest model within run

        Raises:
            ValueError: If any required run ID is missing or model loading fails
        """
        import mlflow
        import mlflow.lightgbm
        import mlflow.pyfunc
        import mlflow.sklearn
        import mlflow.xgboost

        self.logger.info("Loading models from MLflow repository...")

        # Validate run IDs
        if not xgb_run_id:
            raise ValueError("XGBoost run ID is required")
        if not lgb_run_id:
            raise ValueError("LightGBM run ID is required")
        if not tabnet_run_id:
            raise ValueError("TabNet run ID is required")
        if not rf_run_id:
            raise ValueError("Random Forest run ID is required")

        # Load XGBoost model
        try:
            self.logger.info(f"Loading XGBoost model from run {xgb_run_id}...")
            mlflow_client = mlflow.tracking.MlflowClient()

            # Get the latest model version if multiple exist
            model_versions = mlflow_client.get_latest_versions(f"runs:/{xgb_run_id}/{xgb_path}")
            if model_versions:
                # Use the latest version
                model_versions[0]
                self.model_xgb = mlflow.xgboost.load_model(f"runs:/{xgb_run_id}/{xgb_path}")
            else:
                # Direct loading if no versions found
                self.model_xgb = mlflow.xgboost.load_model(f"runs:/{xgb_run_id}/{xgb_path}")

            # Update feature signature from model metadata
            model_info = mlflow_client.get_model_version_download_uri(
                f"runs:/{xgb_run_id}/{xgb_path}"
            )
            model_signature = mlflow.models.get_model_info(model_info).signature
            if model_signature and model_signature.inputs:
                self.xgb_features = model_signature.inputs.input_names()
                self.logger.info(
                    f"Updated XGBoost feature signature with {len(self.xgb_features)} features"
                )
            else:
                self.logger.warning("No feature signature found for XGBoost model")
        except Exception as e:
            self.logger.error(f"Failed to load XGBoost model: {str(e)}")
            raise ValueError(f"Failed to load XGBoost model: {str(e)}") from e

        # Load LightGBM model
        try:
            self.logger.info(f"Loading LightGBM model from run {lgb_run_id}...")

            # Get the latest model version if multiple exist
            model_versions = mlflow_client.get_latest_versions(f"runs:/{lgb_run_id}/{lgb_path}")
            if model_versions:
                # Use the latest version
                model_versions[0]
                self.model_lgb = mlflow.lightgbm.load_model(f"runs:/{lgb_run_id}/{lgb_path}")
            else:
                # Direct loading if no versions found
                self.model_lgb = mlflow.lightgbm.load_model(f"runs:/{lgb_run_id}/{lgb_path}")

            # Update feature signature from model metadata
            model_info = mlflow_client.get_model_version_download_uri(
                f"runs:/{lgb_run_id}/{lgb_path}"
            )
            model_signature = mlflow.models.get_model_info(model_info).signature
            if model_signature and model_signature.inputs:
                self.lgb_features = model_signature.inputs.input_names()
                self.logger.info(
                    f"Updated LightGBM feature signature with {len(self.lgb_features)} features"
                )
            else:
                self.logger.warning("No feature signature found for LightGBM model")
        except Exception as e:
            self.logger.error(f"Failed to load LightGBM model: {str(e)}")
            raise ValueError(f"Failed to load LightGBM model: {str(e)}") from e

        # Load TabNet model (using PyFunc for the TabNetWrapper)
        try:
            self.logger.info(f"Loading TabNet model from run {tabnet_run_id}...")

            # TabNet will be loaded as a PyFunc model since it uses a custom wrapper
            tabnet_pyfunc = mlflow.pyfunc.load_model(f"runs:/{tabnet_run_id}/{tabnet_path}")

            # Extract the underlying TabNetClassifier from the wrapper
            if hasattr(tabnet_pyfunc, "_model_impl") and hasattr(
                tabnet_pyfunc._model_impl, "model"
            ):
                # If loaded with mlflow.pyfunc.load_model
                self.model_tabnet = tabnet_pyfunc._model_impl.model
            elif hasattr(tabnet_pyfunc, "model"):
                # If the wrapper structure is directly accessible
                self.model_tabnet = tabnet_pyfunc.model
            else:
                # Fallback if structure is different
                self.logger.warning("TabNet wrapper structure is unexpected. Using as-is.")
                self.model_tabnet = tabnet_pyfunc

            # Update feature signature from model metadata
            model_info = mlflow_client.get_model_version_download_uri(
                f"runs:/{tabnet_run_id}/{tabnet_path}"
            )
            model_signature = mlflow.models.get_model_info(model_info).signature
            if model_signature and model_signature.inputs:
                self.tabnet_features = model_signature.inputs.input_names()
                self.logger.info(
                    f"Updated TabNet feature signature with {len(self.tabnet_features)} features"
                )
            else:
                self.logger.warning("No feature signature found for TabNet model")
        except Exception as e:
            self.logger.error(f"Failed to load TabNet model: {str(e)}")
            raise ValueError(f"Failed to load TabNet model: {str(e)}") from e

        # If signature-based extraction fails, try direct feature extraction
        if not hasattr(self, "tabnet_features") or not self.tabnet_features:
            if hasattr(self.model_tabnet, "input_dim"):
                # TabNet might store feature dimension but not names
                feature_count = self.model_tabnet.input_dim
                self.logger.warning(
                    f"No named features found for TabNet. Using {feature_count} unnamed features."
                )
                # Use existing features list or fallback
                if self.tabnet_features and len(self.tabnet_features) == feature_count:
                    self.logger.info("Using existing feature list for TabNet")
                else:
                    self.logger.warning("Using generic feature names for TabNet")
                    self.tabnet_features = [f"feature_{i}" for i in range(feature_count)]

        # Load Random Forest model
        try:
            self.logger.info(f"Loading Random Forest model from run {rf_run_id}...")

            # Get the latest model version if multiple exist
            model_versions = mlflow_client.get_latest_versions(f"runs:/{rf_run_id}/{rf_path}")
            if model_versions:
                # Use the latest version
                model_versions[0]
                self.model_extra = mlflow.sklearn.load_model(f"runs:/{rf_run_id}/{rf_path}")
            else:
                # Direct loading if no versions found
                self.model_extra = mlflow.sklearn.load_model(f"runs:/{rf_run_id}/{rf_path}")

            # Update feature signature from model metadata
            model_info = mlflow_client.get_model_version_download_uri(
                f"runs:/{rf_run_id}/{rf_path}"
            )
            model_signature = mlflow.models.get_model_info(model_info).signature
            if model_signature and model_signature.inputs:
                self.rf_features = model_signature.inputs.input_names()
                self.logger.info(
                    f"Updated Random Forest feature signature with {len(self.rf_features)} features"
                )
            else:
                self.logger.warning("No feature signature found for Random Forest model")
        except Exception as e:
            self.logger.error(f"Failed to load Random Forest model: {str(e)}")
            raise ValueError(f"Failed to load Random Forest model: {str(e)}") from e

        # Update extra_base_model_type to reflect the loaded model
        self.extra_base_model_type = "random_forest"

        self.logger.info("All models successfully loaded from MLflow")
        return True
