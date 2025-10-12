"""
EnsembleModel Class

Core implementation of the ensemble model with initialization methods
and high-level APIs.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Restrict parallel threads across various libraries
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


import random

import keras
import tensorflow as tf
from keras import layers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add project root to Python path
try:
    project_root = Path(__file__).parent.parent.parent
    if not project_root.exists():
        # Handle network path by using raw string
        project_root = Path(r"\\".join(str(project_root).split("\\")))
    sys.path.append(str(project_root))
    print(f"Project root ensemble_model: {project_root}")
except Exception as e:
    print(f"Error setting project root path: {e}")
    # Fallback to current directory if path resolution fails
    sys.path.append(str(Path(os.getcwd()).parent))
    print(f"Current directory ensemble_model: {Path(os.getcwd()).parent}")

# Local imports
# Module imports
from src.models.ensemble.calibration import analyze_calibration, calibrate_models
from src.models.ensemble.data_utils import prepare_data
from src.models.ensemble.diagnostics import (
    analyze_prediction_errors,
    detect_data_leakage,
    explain_predictions,
)
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

# Set random seeds for reproducibility
random_seed = 19
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
os.environ["PYTHONHASHSEED"] = str(random_seed)


class EnsembleModel(BaseEstimator, ClassifierMixin):
    """
    EnsembleModel trains a soft voting ensemble (XGBoost, CatBoost, LGBM) combined with stacking
    via a meta-learner. It applies threshold tuning on balanced data to yield a final model with a
    desired trade-off between precision and recall.

    Features:
        - Dynamic Weighting of Base Models: Combines base model probabilities using weights computed from validation precision.
        - Probability Calibration: Optionally calibrates each base model's probabilities.
        - Alternative Meta Learner: Supports different meta-learner types (e.g., Logistic Regression).
        - Model Diagnostics: Functions to analyze prediction errors and understand model behavior.
        - Feature Selection: Improved feature selection based on importance.
        - Class Imbalance Handling: Proper implementation of ADASYN for balancing training data.
        - Cross-Validation: Implementation of proper cross-validation techniques.
        - Precision Optimization: Enhanced threshold tuning focused on precision.
    """

    def __init__(
        self,
        logger: Optional[ExperimentLogger] = None,
        calibrate: bool = False,
        calibration_method: str = "sigmoid",
        individual_thresholding: bool = False,
        meta_learner_type: str = "xgb",
        dynamic_weighting: bool = True,
        extra_base_model_type: str = "random_forest",
        sampling_strategy: float = 0.7,
        complexity_penalty: float = 0.01,
        target_precision: float = 0.50,
        required_recall: float = 0.25,
        X_train: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize the EnsembleModel with configuration parameters.

        Args:
            logger: Logger instance for tracking experiment progress
            calibrate: Whether to calibrate base model probabilities
            calibration_method: Method for probability calibration ("sigmoid" or "isotonic")
            individual_thresholding: Whether to tune thresholds for each base model individually
            meta_learner_type: Type of meta-learner to use ("xgb", "logistic", or "mlp")
            dynamic_weighting: Whether to use dynamic weighting for base model probabilities
            extra_base_model_type: Type of fourth base model ("random_forest", "svm", or "mlp")
            sampling_strategy: ADASYN sampling strategy parameter
            complexity_penalty: Regularization parameter for preventing overfitting
            target_precision: Target precision for threshold tuning
            required_recall: Minimum required recall for threshold tuning
        """
        self.logger = logger or ExperimentLogger(
            experiment_name="ensemble_model_improved", log_dir="./logs/ensemble_model_improved"
        )
        # Load selected features (assumed common to all models)
        self.selected_features = list(import_selected_features_ensemble("all"))
        self.required_recall = required_recall
        self.sampling_strategy = sampling_strategy  # For ADASYN resampling
        self.complexity_penalty = complexity_penalty  # For regularization
        self.target_precision = target_precision  # For threshold tuning

        # Define base models with CPU-only settings with reduced complexity to avoid overfitting:
        self.model_xgb = XGBClassifier(  # 37.6%
            tree_method="hist",  # Required for CPU-only training per project rules
            device="cpu",
            nthread=4,
            objective="binary:logistic",
            eval_metric=["aucpr", "error", "logloss"],
            verbosity=0,
            learning_rate=0.05,
            max_depth=12,
            min_child_weight=390,
            subsample=0.7,
            colsample_bytree=0.79,
            reg_alpha=48.5,
            reg_lambda=5.73,
            gamma=0.98,
            early_stopping_rounds=660,
            scale_pos_weight=2.72,
            seed=19,
        )
        self.model_cat = CatBoostClassifier(  # 36.7%
            learning_rate=0.02,
            depth=9,
            min_data_in_leaf=70,
            subsample=0.95,
            colsample_bylevel=0.52,
            reg_lambda=4.570117165999504,
            leaf_estimation_iterations=12,
            bagging_temperature=3.6,
            scale_pos_weight=4.45,
            early_stopping_rounds=420,
            loss_function="Logloss",
            eval_metric="AUC",
            custom_metric=["Precision", "Recall"],
            task_type="CPU",
            thread_count=4,
            verbose=-1,
        )
        self.model_lgb = LGBMClassifier(  # 39.4%
            objective="binary",
            metric=["binary_logloss", "auc"],
            verbose=-1,
            n_jobs=4,
            random_state=19,
            device="cpu",
            learning_rate=0.11,
            num_leaves=145,
            max_depth=9,
            min_child_samples=170,
            feature_fraction=0.62,
            bagging_fraction=0.635,
            bagging_freq=8,
            reg_alpha=2.7,
            reg_lambda=8.3,
            min_split_gain=0.11,
            early_stopping_rounds=610,
            path_smooth=0.125,
            cat_smooth=16.8,
            max_bin=590,
        )
        self.xgb_features = list(import_selected_features_ensemble(model_type="xgb"))
        self.cat_features = list(import_selected_features_ensemble(model_type="cat"))
        self.lgb_features = list(import_selected_features_ensemble(model_type="lgbm"))
        self.rf_features = list(import_selected_features_ensemble(model_type="rf"))
        # Initialize the extra base model based on the selected type with reduced complexity
        self.extra_base_model_type = extra_base_model_type.lower()
        if self.extra_base_model_type == "random_forest":
            self.model_extra = RandomForestClassifier(
                n_estimators=180,
                max_depth=13,
                min_samples_split=10,
                min_samples_leaf=6,
                max_features=0.4,  # type: ignore
                bootstrap=True,
                class_weight={0: 1.0, 1: 2.0},
                criterion="entropy",
                random_state=19,
                n_jobs=4,
            )
            self.logger.info("Extra base model initialized as RandomForestClassifier.")
        elif self.extra_base_model_type == "svm":
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
            # Defer MLP initialization until training when we have data shape
            self.model_extra = None
            self.mlp_config = {
                "hidden_layers": 1,
                "neurons_per_layer": 120,
                "activation": "tanh",
                "l1_reg": 0.0006252292488020048,
                "l2_reg": 0.0010179804312458536,
                "dropout_rate": 0.2685783444324335,
                "learning_rate": 0.0008586241362721754,
                "beta_1": 0.9,
                "beta_2": 0.999,
                "epsilon": 1e-8,
            }
            self.logger.info("Extra base model configured as MLPClassifier (deferred initialization).")
        else:
            raise ValueError(f"Unknown extra_base_model_type: {self.extra_base_model_type}")

        # Set up meta-learner based on the chosen type
        self.meta_learner_type = meta_learner_type
        self.optimal_threshold = 0.5  # default (used for global tuning)
        # Flag to enable individual threshold tuning per base model
        self.individual_thresholding = individual_thresholding
        # Flag to control probability calibration of base models
        self.calibrate = calibrate
        self.calibration_method = calibration_method
        # Flag to control dynamic weighting during global probability voting
        self.dynamic_weighting = dynamic_weighting
        if self.dynamic_weighting:
            self.dynamic_weights = {
                "xgb": 1 / 3,
                "cat": 1 / 3,
                "lgb": 1 / 6,
                "extra": 1 / 6,
            }  # default weights

        # Placeholders for fitted models and scalers
        self.meta_learner = None
        self.model_xgb_calibrated = None
        self.model_cat_calibrated = None
        self.model_lgb_calibrated = None
        self.model_extra_calibrated = None
        self.extra_model_scaler = None
        self.mlp_config = None

    def _initialize_mlp_model(self, input_shape: int):
        """Initialize the Keras MLP model with the given input shape."""
        if self.mlp_config is None:
            raise ValueError("MLP configuration not found. This should not happen.")

        model = keras.Sequential()
        model.add(layers.InputLayer(shape=(input_shape,)))

        # Add hidden layers
        for _ in range(self.mlp_config["hidden_layers"]):
            model.add(
                layers.Dense(
                    self.mlp_config["neurons_per_layer"],
                    activation=self.mlp_config["activation"],
                    kernel_regularizer=regularizers.l1_l2(
                        l1=self.mlp_config["l1_reg"],
                        l2=self.mlp_config["l2_reg"]
                    ),
                )
            )
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(self.mlp_config["dropout_rate"]))

        # Output layer for binary classification
        model.add(layers.Dense(1, activation="sigmoid"))

        optimizer = keras.optimizers.Adam(
            learning_rate=self.mlp_config["learning_rate"],
            beta_1=self.mlp_config["beta_1"],
            beta_2=self.mlp_config["beta_2"],
            epsilon=self.mlp_config["epsilon"]
        )

        model.compile(
            optimizer=optimizer,  # type: ignore
            loss="binary_crossentropy",
            metrics=["accuracy", keras.metrics.AUC(name="auc")],
        )

        return model

    def _prepare_training_data(
        self,
        X_train,
        y_train,
        x_val,
        y_val,
        X_test,
        y_test,
        split_validation,
        val_size,
    ):
        """Prepare and validate training data."""
        x_train_prepared = prepare_data(X_train, self.selected_features)

        # Check for data leakage if validation data is provided
        if x_val is not None and X_test is not None:
            x_val_prepared = prepare_data(x_val, self.selected_features)
            x_test_prepared = prepare_data(X_test, self.selected_features)

            leakage_results = detect_data_leakage(
                x_train_prepared, x_test_prepared, x_val_prepared, self.logger
            )

            if leakage_results["overlap_percentage"] > 5.0:
                self.logger.warning(
                    f"Significant data leakage detected: {leakage_results['overlap_percentage']:.2f}%"
                )
        else:
            x_val_prepared = None
            x_test_prepared = None

        # Handle validation/test data splitting
        if split_validation or x_val is None or y_val is None:
            self.logger.info("Splitting training data for validation...")
            x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(
                x_train_prepared, y_train, test_size=val_size, random_state=19, stratify=y_train
            )
            x_train_prepared, y_train = x_train_split, y_train_split
            x_val_prepared, y_val = x_val_split, y_val_split
        elif x_val_prepared is None:
            x_val_prepared = prepare_data(x_val, self.selected_features)

        if X_test is None or y_test is None:
            self.logger.info("Creating test set from training data...")
            x_train_split, x_test_split, y_train_split, y_test_split = train_test_split(
                x_train_prepared, y_train, test_size=val_size, random_state=43, stratify=y_train
            )
            x_train_prepared, y_train = x_train_split, y_train_split
            x_test_prepared, y_test = x_test_split, y_test_split
        elif x_test_prepared is None:
            x_test_prepared = prepare_data(X_test, self.selected_features)

        return x_train_prepared, y_train, x_val_prepared, y_val, x_test_prepared, y_test

    def _train_base_models(self, x_train_resampled, y_train_resampled, x_val_prepared, y_val, x_test_prepared, y_test):
        """Train base models and optionally calibrate them."""
        # Initialize MLP model if needed
        if self.extra_base_model_type == "mlp" and self.model_extra is None:
            self.logger.info("Initializing MLP model with input shape...")
            self.model_extra = self._initialize_mlp_model(x_train_resampled.shape[1])

        # Initialize base models dictionary
        base_models = {
            "xgb": self.model_xgb,
            "cat": self.model_cat,
            "lgb": self.model_lgb,
            "extra": self.model_extra,
        }

        # For MLP or SVM, we need scaling
        if self.extra_base_model_type in ["mlp", "svm"]:
            self.logger.info(f"Applying StandardScaler for {self.extra_base_model_type} model...")
            self.extra_model_scaler = StandardScaler().fit(x_train_resampled)
            base_models["extra_scaler"] = self.extra_model_scaler

        # Train base models
        self.logger.info("Training base models...")
        trained_models = train_base_models(
            base_models,
            x_train_resampled,
            y_train_resampled,
            x_test_prepared,
            y_test,
            x_val_prepared,
            y_val,
        )

        # Prepare feature subsets
        x_val_prepared_xgb = x_val_prepared[self.xgb_features]
        x_val_prepared_cat = x_val_prepared[self.cat_features]
        x_val_prepared_lgb = x_val_prepared[self.lgb_features]
        x_val_prepared_rf = x_val_prepared[self.rf_features]

        x_train_prepared_xgb = x_train_resampled[self.xgb_features]
        x_train_prepared_cat = x_train_resampled[self.cat_features]
        x_train_prepared_lgb = x_train_resampled[self.lgb_features]
        x_train_prepared_rf = x_train_resampled[self.rf_features]

        x_test_prepared_xgb = x_test_prepared[self.xgb_features]
        x_test_prepared_cat = x_test_prepared[self.cat_features]
        x_test_prepared_lgb = x_test_prepared[self.lgb_features]
        x_test_prepared_rf = x_test_prepared[self.rf_features]

        x_combined_xgb = pd.concat([x_train_prepared_xgb, x_test_prepared_xgb], axis=0)
        x_combined_cat = pd.concat([x_train_prepared_cat, x_test_prepared_cat], axis=0)
        x_combined_lgb = pd.concat([x_train_prepared_lgb, x_test_prepared_lgb], axis=0)
        x_combined_rf = pd.concat([x_train_prepared_rf, x_test_prepared_rf], axis=0)

        # Update model references
        self.model_xgb = trained_models["xgb"]
        self.model_cat = trained_models["cat"]
        self.model_lgb = trained_models["lgb"]
        self.model_extra = trained_models["extra"]

        if "extra_scaler" in trained_models:
            self.extra_model_scaler = trained_models["extra_scaler"]

        # Optionally calibrate models
        if self.calibrate:
            self.logger.info(f"Calibrating base models using {self.calibration_method} method...")
            calibration_results = calibrate_models(
                trained_models,
                x_train_resampled,
                y_train_resampled,
                x_val_prepared,
                y_val,
                self.calibration_method,
                self.logger,
            )

            calibrated_models = calibration_results["calibrated_models"]

            # Store calibrated models
            self.model_xgb_calibrated = calibrated_models["xgb"]
            self.model_cat_calibrated = calibrated_models["cat"]
            self.model_lgb_calibrated = calibrated_models["lgb"]
            self.model_extra_calibrated = calibrated_models["extra"]

            # Analyze calibration effectiveness
            analyze_calibration(calibration_results["calibration_results"], y_val, self.logger)

        feature_data = (
            x_val_prepared_xgb, x_val_prepared_cat, x_val_prepared_lgb, x_val_prepared_rf,
            x_combined_xgb, x_combined_cat, x_combined_lgb, x_combined_rf
        )

        return trained_models, feature_data

    def _generate_base_predictions(self, x_train_prepared, y_train, x_test_prepared, y_test, x_val_prepared, feature_data):
        """Generate predictions from base models."""
        (
            x_val_prepared_xgb, x_val_prepared_cat, x_val_prepared_lgb, x_val_prepared_rf,
            x_combined_xgb, x_combined_cat, x_combined_lgb, x_combined_rf
        ) = feature_data

        # Use calibrated models if available
        xgb_model = self.model_xgb_calibrated if self.calibrate else self.model_xgb
        cat_model = self.model_cat_calibrated if self.calibrate else self.model_cat
        lgb_model = self.model_lgb_calibrated if self.calibrate else self.model_lgb
        extra_model = self.model_extra_calibrated if self.calibrate else self.model_extra

        # Ensure models are not None (should be initialized during training)
        assert xgb_model is not None, "XGBoost model not initialized"
        assert cat_model is not None, "CatBoost model not initialized"
        assert lgb_model is not None, "LightGBM model not initialized"
        assert extra_model is not None, "Extra model not initialized"

        # Combine features and handle indexes
        x_combined = pd.concat([x_train_prepared, x_test_prepared], axis=0)
        y_combined = pd.concat([y_train, y_test], axis=0)
        x_combined.reset_index(drop=True, inplace=True)
        y_combined.reset_index(drop=True, inplace=True)

        # Get predictions
        if self.extra_base_model_type in ["mlp", "svm"] and self.extra_model_scaler is not None:
            x_val_scaled = self.extra_model_scaler.transform(x_val_prepared)
            x_train_scaled = self.extra_model_scaler.transform(x_combined)

            if self.extra_base_model_type == "mlp":
                p_extra = extra_model.predict(x_val_scaled, verbose=0).flatten()  # type: ignore
                p_extra_train = extra_model.predict(x_train_scaled, verbose=0).flatten()  # type: ignore
            else:  # SVM case
                p_extra = np.asarray(extra_model.predict_proba(x_val_scaled))[:, 1]
                p_extra_train = np.asarray(extra_model.predict_proba(x_train_scaled))[:, 1]
        else:
            p_extra = np.asarray(extra_model.predict_proba(x_val_prepared_rf))[:, 1]
            p_extra_train = np.asarray(extra_model.predict_proba(x_combined_rf))[:, 1]

        p_xgb = np.asarray(xgb_model.predict_proba(x_val_prepared_xgb))[:, 1]
        p_xgb_train = np.asarray(xgb_model.predict_proba(x_combined_xgb))[:, 1]

        p_cat = np.asarray(cat_model.predict_proba(x_val_prepared_cat))[:, 1]
        p_cat_train = np.asarray(cat_model.predict_proba(x_combined_cat))[:, 1]

        p_lgb = np.asarray(lgb_model.predict_proba(x_val_prepared_lgb))[:, 1]
        p_lgb_train = np.asarray(lgb_model.predict_proba(x_combined_lgb))[:, 1]

        return {
            'p_xgb': p_xgb, 'p_cat': p_cat, 'p_lgb': p_lgb, 'p_extra': p_extra,
            'p_xgb_train': p_xgb_train, 'p_cat_train': p_cat_train,
            'p_lgb_train': p_lgb_train, 'p_extra_train': p_extra_train,
            'y_combined': y_combined
        }

    def _compute_dynamic_weights(self, predictions, y_val, y_combined):
        """Compute dynamic weights if enabled."""
        if self.dynamic_weighting:
            self.logger.info("Computing dynamic weights based on validation performance...")
            self.dynamic_weights = compute_precision_focused_weights(
                predictions['p_xgb'], predictions['p_cat'], predictions['p_lgb'], predictions['p_extra'],
                y_val, self.target_precision, self.required_recall, self.logger,
            )
            self.dynamic_weights_train = compute_precision_focused_weights(
                predictions['p_xgb_train'], predictions['p_cat_train'],
                predictions['p_lgb_train'], predictions['p_extra_train'],
                predictions['y_combined'], self.target_precision, self.required_recall, self.logger,
            )

    def _train_meta_learner(self, x_train_prepared, y_train, x_test_prepared, y_test, x_val_prepared, y_val, feature_data):
        """Train the meta-learner with predictions from base models."""
        # Generate base model predictions
        predictions = self._generate_base_predictions(
            x_train_prepared, y_train, x_test_prepared, y_test, x_val_prepared, feature_data
        )

        # Compute dynamic weights if enabled
        self._compute_dynamic_weights(predictions, y_val, predictions['y_combined'])

        # Create meta-features
        self.logger.info("Creating meta-features for meta-learner...")
        meta_features = create_meta_features(
            predictions['p_xgb'], predictions['p_cat'], predictions['p_lgb'], predictions['p_extra'],
            self.dynamic_weights if self.dynamic_weighting else None
        )
        meta_features_train = create_meta_features(
            predictions['p_xgb_train'], predictions['p_cat_train'],
            predictions['p_lgb_train'], predictions['p_extra_train'],
            self.dynamic_weights_train if self.dynamic_weighting else None,
        )

        # Convert to DataFrame
        meta_df = create_meta_dataframe(meta_features)
        meta_df_train = create_meta_dataframe(meta_features_train)

        # Initialize and train meta-learner
        self.logger.info(f"Initializing meta-learner of type {self.meta_learner_type}...")
        self.meta_learner = initialize_meta_learner(self.meta_learner_type)

        self.logger.info("Training meta-learner...")
        self.meta_learner = hypertune_meta_learner(
            meta_df_train.values,
            predictions['y_combined'].values if hasattr(predictions['y_combined'], 'values') else predictions['y_combined'],
            meta_df.values,
            y_val.values if hasattr(y_val, 'values') else y_val,
            meta_learner_type=self.meta_learner_type,
            target_precision=self.target_precision,
            min_recall=self.required_recall,
        )

        # Tune threshold
        self.logger.info(f"Tuning threshold for target precision {self.target_precision}...")
        meta_val_probs = self.meta_learner.predict_proba(meta_df)[:, 1]  # type: ignore
        best_threshold, _ = tune_threshold_for_precision(
            meta_val_probs, y_val,
            target_precision=self.target_precision,
            required_recall=self.required_recall,
            logger=self.logger
        )
        self.optimal_threshold = best_threshold
        self.logger.info(f"Optimal threshold set to {self.optimal_threshold:.4f}")

        # Final evaluation
        self.logger.info("Performing final evaluation on validation data...")
        eval_results = evaluate_model(
            self.meta_learner, meta_df,  # type: ignore
            y_val.values if hasattr(y_val, 'values') else y_val,
            self.optimal_threshold, self.logger  # type: ignore
        )
        return eval_results

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
        """
        Train the ensemble model including base models and meta-learner.

        Args:
            X_train: Training features
            y_train: Training target values
            x_val: Optional validation features
            y_val: Optional validation target values
            X_test: Optional test features
            y_test: Optional test target values
            split_validation: Whether to split training data for validation
            val_size: Size of validation split if split_validation is True

        Returns:
            Dictionary with training results and metrics
        """
        self.logger.info("Starting ensemble model training...")

        # Step 1: Data preparation and validation
        x_train_prepared, y_train, x_val_prepared, y_val, x_test_prepared, y_test = (
            self._prepare_training_data(
                X_train, y_train, x_val, y_val, X_test, y_test, split_validation, val_size
            )
        )

        # Step 2: Skip class imbalance handling - using original data
        self.logger.info("Skipping ADASYN resampling, using original training data...")
        x_train_resampled, y_train_resampled = x_train_prepared, y_train

        # Step 3: Train and optionally calibrate base models
        _, feature_data = self._train_base_models(
            x_train_resampled, y_train_resampled, x_val_prepared, y_val, x_test_prepared, y_test
        )

        # Extract feature data
        (
            x_val_prepared_xgb, x_val_prepared_cat, x_val_prepared_lgb, x_val_prepared_rf,
            x_combined_xgb, x_combined_cat, x_combined_lgb, x_combined_rf
        ) = feature_data

        # Step 4: Generate predictions and train meta-learner
        eval_results = self._train_meta_learner(
            x_train_prepared, y_train, x_test_prepared, y_test,
            x_val_prepared, y_val, feature_data
        )

        self.logger.info("Ensemble model training completed successfully.")
        return eval_results

    def predict_proba(self, X) -> np.ndarray:
        """
        Generate probability predictions from the ensemble.

        Args:
            X: Feature dataframe

        Returns:
            Array of predicted probabilities for class 1
        """
        if self.meta_learner is None:
            raise ValueError("Model has not been trained. Call train() first.")

        # Prepare input data
        x_prepared = prepare_data(X, X.columns)
        # Generate predictions from base models
        # Use calibrated models if available
        xgb_model = self.model_xgb_calibrated if self.calibrate else self.model_xgb
        cat_model = self.model_cat_calibrated if self.calibrate else self.model_cat
        lgb_model = self.model_lgb_calibrated if self.calibrate else self.model_lgb
        extra_model = self.model_extra_calibrated if self.calibrate else self.model_extra

        # Ensure models are not None (should be initialized during training)
        assert xgb_model is not None, "XGBoost model not initialized"
        assert cat_model is not None, "CatBoost model not initialized"
        assert lgb_model is not None, "LightGBM model not initialized"
        assert extra_model is not None, "Extra model not initialized"

        # Get predictions
        if self.extra_base_model_type in ["mlp", "svm"] and self.extra_model_scaler is not None:
            x_scaled = self.extra_model_scaler.transform(x_prepared)
            if self.extra_base_model_type == "mlp":
                p_extra = extra_model.predict(x_scaled).flatten()
            else:  # SVM case
                p_extra = np.asarray(extra_model.predict_proba(x_scaled))[:, 1]
        else:
            p_extra = np.asarray(extra_model.predict_proba(x_prepared))[:, 1]

        p_xgb = np.asarray(xgb_model.predict_proba(x_prepared))[:, 1]
        p_cat = np.asarray(cat_model.predict_proba(x_prepared))[:, 1]
        p_lgb = np.asarray(lgb_model.predict_proba(x_prepared))[:, 1]
        # Create meta-features
        meta_features = create_meta_features(
            p_xgb, p_cat, p_lgb, p_extra, self.dynamic_weights if self.dynamic_weighting else None
        )

        # Get meta-learner predictions
        meta_probs = self.meta_learner.predict_proba(meta_features)  # type: ignore

        return meta_probs[:, 1]

    def predict(self, X) -> np.ndarray:
        """
        Generate binary predictions using the optimal threshold.

        Args:
            X: Feature dataframe

        Returns:
            Array of binary predictions (0 or 1)
        """
        # Get probability predictions
        probabilities = self.predict_proba(X)

        # Apply threshold
        return (probabilities >= self.optimal_threshold).astype(int)

    def explain_predictions(self, x_val) -> dict:
        """
        Generate feature importance explanations using SHAP values.

        Args:
            x_val: Validation features

        Returns:
            Dictionary with explanation results
        """
        return explain_predictions(self, x_val, self.logger)

    def analyze_prediction_errors(self, x_val, y_val) -> dict:
        """
        Analyze prediction errors on the validation set.

        Args:
            x_val: Validation features
            y_val: Validation target values

        Returns:
            Dictionary with error analysis results
        """
        return analyze_prediction_errors(self, x_val, y_val, self.optimal_threshold, self.logger)  # type: ignore

    def precision_filter(self, x, probabilities):
        """
        Apply additional filtering to boost precision
        """
        # Get high-confidence predictions
        high_conf = probabilities > self.optimal_threshold

        # Get original features for these instances
        x_high_conf = x[high_conf]

        # Apply rule-based filters (examples)
        if "home_form" in x_high_conf.columns and "away_form" in x_high_conf.columns:
            form_diff = abs(x_high_conf["home_form"] - x_high_conf["away_form"])
            # Filter out likely non-draws (big form differences)
            likely_not_draw = form_diff > 0.5
            high_conf[high_conf] = ~likely_not_draw

        # Additional filters based on domain knowledge

        return high_conf

    def get_model_params(self, model):
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
