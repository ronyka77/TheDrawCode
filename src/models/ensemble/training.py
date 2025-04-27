"""
Model Training Utilities

Functions for training the ensemble models.
"""

import os
import random
import time
import warnings
from typing import Optional

import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
import torch
import torch.optim as optim
from lightgbm import LGBMClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from src.utils.logger import ExperimentLogger

logger = ExperimentLogger(
    experiment_name="ensemble_model_training", log_dir="./logs/ensemble_model_training"
)

from src.models.ensemble.ResNet import ResNetMetaLearner
from src.models.ensemble.thresholds import tune_threshold_for_precision_optimized
from src.utils.create_evaluation_set import import_selected_features_ensemble

# Filter scikit-learn parameter renaming warnings
warnings.filterwarnings("ignore", message=".*force_all_finite.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*ensure_all_finite.*", category=FutureWarning)

# Set random seeds for reproducibility
random_seed = 19
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
os.environ["PYTHONHASHSEED"] = str(random_seed)

# Restrict parallel threads across various libraries
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"
os.environ["VECLIB_MAXIMUM_THREADS"] = "16"
# PyTorch specific reproducibility settings
torch.manual_seed(random_seed)


def initialize_meta_learner(meta_learner_type: str = "xgb") -> object:
    """
    Initialize the meta learner based on the provided meta_learner_type.

    Args:
        meta_learner_type: Type of meta-learner ('xgb', 'logistic', 'mlp', 'tabnet', etc.)

    Returns:
        Initialized meta-learner model
    """
    logger.info(f"Initializing meta-learner of type: {meta_learner_type}")

    if meta_learner_type.lower() == "xgb":
        # XGBoost meta-learner with CPU settings and reduced complexity
        meta_learner = XGBClassifier(
            tree_method="hist",  # CPU-optimized
            device="cpu",
            n_jobs=4,
            objective="binary:logistic",
            learning_rate=0.05,
            n_estimators=200,  # Reduced from 500
            max_depth=4,  # Reduced from 6
            random_state=42,
            colsample_bytree=0.8,
            eval_metric=["logloss", "auc"],
            gamma=0.1,
            min_child_weight=10,
            reg_alpha=0.5,
            reg_lambda=1.0,
            subsample=0.8,
        )

        logger.info("XGBoost meta-learner initialized with CPU-optimized settings")
    elif meta_learner_type.lower() == "tabnet":
        # TabNet meta-learner with CPU settings

        params = {
            "n_d": 56,  # Dimension of the prediction layer
            "n_a": 12,  # Dimension of the attention layer
            "n_steps": 15,  # Number of steps in the architecture
            "gamma": 1.9500000000000002,  # Scaling coefficient for attention
            "lambda_sparse": 2.594352900973954e-07,  # Sparsity regularization
            "momentum": 0.725,
            "mask_type": "sparsemax",  # Used to compute sparse attention weights
            "device_name": "cuda",
            "verbose": 0,
            "seed": 19,
            "learning_rate": 0.012272866712824772,
        }
        # Handle optimizer params separately
        if "learning_rate" in params:
            params["optimizer_params"] = {"lr": params["learning_rate"]}
            # Optionally remove learning_rate from params to avoid passing it to TabNetClassifier
            del params["learning_rate"]
        if "eval_metric" in params:
            params.pop("eval_metric")
        if "patience" in params:
            params.pop("patience")
        if "max_epochs" in params:
            params.pop("max_epochs")
        if "batch_size" in params:
            params.pop("batch_size")
        if "virtual_batch_size" in params:
            params.pop("virtual_batch_size")

        meta_learner = TabNetClassifier(**params)
        logger.info("TabNet meta-learner initialized")
    elif meta_learner_type.lower() == "lgb":
        meta_learner = LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            n_jobs=4,
            random_state=19,
            device="cpu",
            bagging_fraction=0.7,
            bagging_freq=7,
            cat_smooth=9.5,
            feature_fraction=0.64,
            learning_rate=0.06999999999999999,
            max_bin=445,
            max_depth=5,
            metric=["binary_logloss", "average_precision", "auc"],
            min_child_samples=470,
            min_split_gain=0.51,
            num_leaves=152,
            path_smooth=0.315,
            reg_alpha=9.8,
            reg_lambda=4.85,
            verbose=-1,
        )
        logger.info("LightGBM meta-learner initialized with CPU-optimized settings")
    elif meta_learner_type.lower() == "logistic":
        # Logistic Regression meta-learner with L2 regularization
        meta_learner = LogisticRegression(
            penalty="l2",
            C=1.0,  # Inverse of regularization strength
            solver="lbfgs",
            max_iter=1000,
            tol=1e-4,
            random_state=42,
            class_weight="balanced",
            n_jobs=4,
        )

        logger.info("Logistic Regression meta-learner initialized with L2 regularization")
    elif meta_learner_type.lower() == "logistic_cv":
        # Logistic Regression with cross-validation for C parameter
        meta_learner = LogisticRegressionCV(
            penalty="l2",
            Cs=10,
            cv=5,
            solver="lbfgs",
            max_iter=1000,
            tol=1e-4,
            random_state=42,
            class_weight="balanced",
            n_jobs=4,
        )

        logger.info("LogisticRegressionCV meta-learner initialized with auto C selection")
    elif meta_learner_type.lower() == "mlp":
        # Neural Network meta-learner with reduced complexity
        meta_learner = MLPClassifier(
            hidden_layer_sizes=(20, 10),  # Reduced from (50, 20)
            activation="relu",
            solver="adam",
            alpha=0.01,  # Increased regularization
            learning_rate="adaptive",
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.2,
            batch_size=32,
            random_state=42,
        )

        logger.info("MLPClassifier meta-learner initialized with adaptive learning rate")
    elif meta_learner_type.lower() == "sgd":
        meta_learner = SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=0.0001,
            l1_ratio=0.5,
            class_weight=None,
            max_iter=1000,
            early_stopping=True,
            random_state=42,
        )

        logger.info("SGDClassifier meta-learner initialized with log_loss loss function")
    elif meta_learner_type.lower() == "resnet":
        meta_learner = ResNetMetaLearner()
        logger.info("ResNetMetaLearner meta-learner initialized")
    else:
        logger.error(f"Unknown meta_learner_type: {meta_learner_type}")
        raise ValueError(
            f"Unknown meta_learner_type: {meta_learner_type}. "
            f"Supported types: 'xgb', 'logistic', 'logistic_cv', 'mlp', 'resnet'"
        )

    # Log meta-learner parameters to MLflow
    meta_learner_params = meta_learner.get_params()

    # Flatten nested dictionaries for MLflow
    flattened_params = {}
    for key, value in meta_learner_params.items():
        if isinstance(value, (int, float, str, bool, type(None))):
            flattened_params[f"meta_learner_{key}"] = value

    mlflow.log_params(flattened_params)

    return meta_learner


def train_base_models(
    models: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
) -> dict:
    """
    Train base models with early stopping using evaluation data.

    Args:
        models: Dictionary of models to train
        X_train: Training features
        y_train: Training labels
        X_eval: Evaluation features for early stopping
        y_eval: Evaluation labels for early stopping

    Returns:
        Dictionary of trained models
    """

    trained_models = {}
    xgb_features = import_selected_features_ensemble(model_type="xgb")
    cat_features = import_selected_features_ensemble(model_type="cat")
    lgb_features = import_selected_features_ensemble(model_type="lgbm")
    rf_features = import_selected_features_ensemble(model_type="rf")
    tabnet_features = import_selected_features_ensemble(model_type="tabnet")

    for model_name, model in models.items():
        logger.info(f"Training base model: {model_name}")
        try:
            # Prepare training and evaluation data
            X_train_copy = X_train.copy()
            X_eval_copy = X_eval.copy()
            X_test_copy = X_test.copy()
            y_test_copy = y_test.copy()
            y_train_copy = y_train.copy()
            y_eval_copy = y_eval.copy()
            # Combine training and test data for comprehensive model training
            X_combined = pd.concat([X_train_copy, X_test_copy], axis=0)
            y_combined = pd.concat([y_train_copy, y_test_copy], axis=0)

            # Reset indices to avoid any potential issues
            X_combined.reset_index(drop=True, inplace=True)
            y_combined.reset_index(drop=True, inplace=True)
            X_train_copy = X_combined.copy()
            y_train_copy = y_combined.copy()
            logger.info(
                f"Combined training data shape: {X_train_copy.shape} and {y_train_copy.shape}"
            )
            # Handle model fitting based on model name
            if model_name == "extra":
                # Apply scaling if needed (for MLP or SVM models)
                if "extra_scaler" in models:
                    logger.info("Scaling training data")
                    scaler = models["extra_scaler"]
                    X_train_scaled = scaler.transform(X_train_copy)
                    X_eval_scaled = scaler.transform(X_eval_copy)

                    # Train with early stopping if model supports it
                    if hasattr(model, "early_stopping_rounds"):
                        model.fit(
                            X_train_scaled,
                            y_train_copy,
                            eval_set=[(X_eval_scaled, y_eval_copy)],
                            verbose=False,
                        )
                    else:
                        model.fit(X_train_scaled, y_train_copy)

                    trained_models[model_name] = model
                    trained_models["extra_scaler"] = scaler
                else:
                    X_train_rf = X_train_copy[rf_features]
                    logger.info("Training Random Forest model")
                    model.fit(X_train_rf, y_train_copy)
                    trained_models[model_name] = model
            elif model_name == "xgb":
                X_train_xgb = X_train_copy[xgb_features]
                X_eval_xgb = X_eval_copy[xgb_features]
                # XGBoost model with tree_method='hist' for CPU-only training
                model.fit(
                    X_train_xgb, y_train_copy, eval_set=[(X_eval_xgb, y_eval_copy)], verbose=False
                )
                trained_models[model_name] = model
            elif model_name == "tabnet":
                # TabNet specific training
                logger.info("Training TabNet model")
                X_train_tabnet = X_train_copy[tabnet_features]
                X_eval_tabnet = X_eval_copy[tabnet_features]
                # Convert to numpy arrays if pandas
                if hasattr(X_train_tabnet, "values"):
                    X_train_tabnet_np = X_train_tabnet.values
                    X_eval_tabnet_np = X_eval_tabnet.values
                    y_train_np = (
                        y_train_copy.values if hasattr(y_train_copy, "values") else y_train_copy
                    )
                    y_eval_np = (
                        y_eval_copy.values if hasattr(y_eval_copy, "values") else y_eval_copy
                    )
                else:
                    X_train_tabnet_np = X_train_tabnet
                    X_eval_tabnet_np = X_eval_tabnet
                    y_train_np = y_train_copy
                    y_eval_np = y_eval_copy

                # Train TabNet with early stopping on eval set
                model.fit(
                    X_train_tabnet_np,
                    y_train_np,
                    eval_set=[(X_eval_tabnet_np, y_eval_np)],
                    max_epochs=150,
                    patience=17,
                    eval_metric=["auc", "logloss"],
                )
                trained_models[model_name] = model
            elif model_name == "cat":
                X_train_cat = X_train_copy[cat_features]
                X_eval_cat = X_eval_copy[cat_features]
                # CatBoost model
                model.fit(
                    X_train_cat, y_train_copy, eval_set=(X_eval_cat, y_eval_copy), verbose=False
                )
                trained_models[model_name] = model
            elif model_name == "lgb":
                X_train_lgb = X_train_copy[lgb_features]
                X_eval_lgb = X_eval_copy[lgb_features]
                # LightGBM model
                model.fit(X_train_lgb, y_train_copy, eval_set=[(X_eval_lgb, y_eval_copy)])
                trained_models[model_name] = model
            else:
                # Generic model without specific handling
                model.fit(X_train_copy, y_train_copy)
                trained_models[model_name] = model

            logger.info(f"Successfully trained base model: {model_name}")
        except Exception as e:
            logger.error(f"Error training base model {model_name}: {str(e)}")
            raise

    return trained_models


def train_meta_learner(
    meta_learner,
    meta_features: np.ndarray,
    meta_targets: np.ndarray,
    eval_meta_features: Optional[np.ndarray] = None,
    eval_meta_targets: Optional[np.ndarray] = None,
    meta_learner_type: str = "xgb",
) -> object:
    """
    Train the meta-learner on meta-features from base models.

    Args:
        meta_learner: The meta-learner model to train
        meta_features: Meta-features for training
        meta_targets: Target values for training
        eval_meta_features: Evaluation meta-features for early stopping
        eval_meta_targets: Evaluation target values for early stopping
        meta_learner_type: Type of meta-learner model

    Returns:
        Trained meta-learner model
    """
    logger.info("Training meta-learner...")

    try:
        if meta_learner_type == "resnet":
            # Special handling for ResNet meta-learner
            meta_learner.fit(
                meta_features, meta_targets, X_val=eval_meta_features, y_val=eval_meta_targets
            )
            # ResNet has its own threshold tuning internally
            best_threshold = meta_learner.threshold

            # Get predictions using the tuned threshold
            y_proba = meta_learner.predict_proba(eval_meta_features)[:, 1]
            y_pred = (y_proba >= best_threshold).astype(int)

            # Calculate metrics with the model's threshold
            precision, recall = meta_learner._calculate_precision_recall(eval_meta_targets, y_pred)
            metrics = {"precision": precision, "recall": recall}

        # For models that support early stopping (XGBoost)
        elif hasattr(meta_learner, "early_stopping_rounds") or hasattr(
            meta_learner, "early_stopping"
        ):
            if eval_meta_features is not None and eval_meta_targets is not None:
                # Ensure both evaluation features and targets are provided
                if meta_learner_type == "lgb":
                    early_stopping_rounds = meta_learner.early_stopping_rounds
                    # LightGBM specific early stopping
                    meta_learner.fit(
                        meta_features,
                        meta_targets,
                        eval_set=[(eval_meta_features, eval_meta_targets)],
                        callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds)],
                    )
                else:
                    # Generic early stopping
                    meta_learner.fit(
                        meta_features,
                        meta_targets,
                        eval_set=[(eval_meta_features, eval_meta_targets)],
                        verbose=False,
                    )
            else:
                # Fall back to standard training if eval data is missing
                logger.warning(
                    "Evaluation data missing for early stopping. Falling back to standard training."
                )
                meta_learner.fit(meta_features, meta_targets)
        else:
            # Standard training for other models
            meta_learner.fit(meta_features, meta_targets)

        # Log meta-learner performance metrics to MLflow
        try:
            if meta_learner_type == "resnet":
                # ResNet has its own threshold tuning internally
                best_threshold = meta_learner.threshold

                # Get predictions using the tuned threshold
                y_proba = meta_learner.predict_proba(eval_meta_features)[:, 1]
                y_pred = (y_proba >= best_threshold).astype(int)

                # Calculate metrics with the model's threshold
                precision, recall = meta_learner._calculate_precision_recall(
                    eval_meta_targets, y_pred
                )
                metrics = {"precision": precision, "recall": recall}
                mlflow.log_metrics(metrics)
            elif hasattr(meta_learner, "evals_result"):
                # XGBoost-style metrics
                evals_result = meta_learner.evals_result()
                for metric_name, values in evals_result.get("validation_0", {}).items():
                    final_value = values[-1]
                    mlflow.log_metric(f"meta_learner_{metric_name}", final_value)
            elif hasattr(meta_learner, "best_score_"):
                # Scikit-learn style best score
                mlflow.log_metric("meta_learner_best_score", meta_learner.best_score_)
        except Exception as e:
            logger.warning(f"Could not log meta-learner training metrics: {str(e)}")

        logger.info("Meta-learner training completed successfully")

        return meta_learner

    except Exception as e:
        logger.error(f"Error training meta-learner: {str(e)}")
        raise


def hypertune_meta_learner(
    meta_features: np.ndarray,
    meta_targets: np.ndarray,
    eval_meta_features: Optional[np.ndarray] = None,
    eval_meta_targets: Optional[np.ndarray] = None,
    meta_learner_type="xgb",
    n_trials=200,
    timeout=900000,
    target_precision=0.5,
    min_recall=0.25,
):
    """
    Hypertune meta-learner using Optuna and optimize threshold for precision/recall balance.
    Args:
        meta_features: Meta-features for training
        meta_targets: Target values for training
        eval_meta_features: Evaluation meta-features for early stopping
        eval_meta_targets: Evaluation target values for early stopping
        meta_learner_type: Type of meta-learner ('xgb', 'logistic', 'mlp')
        n_trials: Number of Optuna trials
        timeout: Maximum time for optimization in seconds
        target_precision: Target precision for threshold optimization
        min_recall: Minimum required recall
    Returns:
        tuple: (best_meta_learner, best_threshold)
    """
    import os
    import random

    import lightgbm as lgb
    import numpy as np
    import tensorflow as tf
    from lightgbm import LGBMClassifier
    from pytorch_tabnet.tab_model import TabNetClassifier
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.neural_network import MLPClassifier
    from xgboost import XGBClassifier

    # Set random seeds for reproducibility
    random_seed = 19
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    
    # PyTorch specific reproducibility settings and optimizations
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.benchmark = True  # Auto-optimizes for hardware if input sizes don't change
        torch.backends.cudnn.deterministic = False  # Better performance, less deterministic
        # Enable TF32 for better performance on Ampere GPUs (RTX 30xx and newer)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    logger.info(f"Hyperparameter tuning for meta-learner type: {meta_learner_type}")
    best_model = None

    if meta_learner_type == "tabnet":
        # Convert to numpy arrays if pandas
        if hasattr(meta_features, "values"):
            # Use to_numpy() for modern pandas
            meta_features_np = meta_features.to_numpy()
            meta_targets_np = (
                meta_targets.to_numpy()
                if hasattr(meta_targets, "to_numpy")
                else meta_targets
            )
            eval_features_np = (
                eval_meta_features.to_numpy()
                if hasattr(eval_meta_features, "to_numpy")
                else eval_meta_features
            )
            eval_targets_np = (
                eval_meta_targets.to_numpy()
                if hasattr(eval_meta_targets, "to_numpy")
                else eval_meta_targets
            )
        else:
            meta_features_np = meta_features
            meta_targets_np = meta_targets
            eval_features_np = eval_meta_features
            eval_targets_np = eval_meta_targets

    def objective(trial):
        # Define hyperparameters based on meta-learner type
        if meta_learner_type == "xgb":
            base_params = {
                "tree_method": "hist",
                "objective": "binary:logistic",
                "n_jobs": 4,
                "eval_metric": ["aucpr", "error", "logloss"],
                # 'device': 'cpu',
                "random_state": 19,
            }

            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.10, step=0.005),
                "max_depth": trial.suggest_int("max_depth", 4, 13, step=1),
                "min_child_weight": trial.suggest_int("min_child_weight", 150, 800, step=10),
                "gamma": trial.suggest_float("gamma", 0.02, 4.0, step=0.02),
                "subsample": trial.suggest_float("subsample", 0.55, 0.95, step=0.01),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 0.90, step=0.01),
                "reg_alpha": trial.suggest_float("reg_alpha", 10.0, 70.0, step=0.1),
                "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0, step=0.01),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.8, 4.5, step=0.05),
                "early_stopping_rounds": trial.suggest_int(
                    "early_stopping_rounds", 400, 1200, step=20
                ),
            }
            params.update(base_params)
            meta_learner = XGBClassifier(**params)
            trial.set_user_attr("model", meta_learner)
        elif meta_learner_type == "tabnet":
            base_params = {
                "optimizer_fn": optim.Adam,
                "mask_type": "sparsemax",
                "eval_metric": ["logloss", "auc"],
                "verbose": 0,
                "seed": 19,
                "device_name": "cuda",
            }

            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 2e-1, log=True),
                "n_d": trial.suggest_int("n_d", 6, 64),
                "n_a": trial.suggest_int("n_a", 6, 64),
                "n_steps": trial.suggest_int("n_steps", 3, 20),
                "gamma": trial.suggest_float("gamma", 0.5, 3.7, step=0.05),
                "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-8, 1e-4, log=True),
                "momentum": trial.suggest_float("momentum", 0.7, 0.99, step=0.005),
                "mask_type": trial.suggest_categorical("mask_type", ["sparsemax", "entmax"]),
                "n_independent": trial.suggest_int("n_independent", 1, 7),
                "n_shared": trial.suggest_int("n_shared", 1, 7),
            }
            params.update(base_params)
            # Suggest fit parameters
            fit_params = {
                "max_epochs": trial.suggest_int("max_epochs", 50, 250, step=5),  # Fit param
                "patience": trial.suggest_int("patience", 4, 40, step=2),  # Fit param
                "batch_size": trial.suggest_int("batch_size", 1024, 24576),  # Fit param
                "virtual_batch_size": trial.suggest_int("virtual_batch_size", 128, 4096),  # Fit param
                "eval_metric": params.get("eval_metric", ["logloss", "auc"]),
            }
            # params.update(fit_params)
            train_params = params
            # Handle optimizer params separately
            if "learning_rate" in params:
                params["optimizer_params"] = {"lr": params["learning_rate"]}
                # Optionally remove learning_rate from params to avoid passing it to TabNetClassifier
                del params["learning_rate"]
            if "eval_metric" in train_params:
                train_params.pop("eval_metric")
            if "patience" in train_params:
                train_params.pop("patience")
            if "max_epochs" in train_params:
                train_params.pop("max_epochs")
            if "batch_size" in train_params:
                train_params.pop("batch_size")
            if "virtual_batch_size" in train_params:
                train_params.pop("virtual_batch_size")
            meta_learner = TabNetClassifier(**train_params)
            if torch.cuda.is_available():
                try:
                    if hasattr(meta_learner, 'network') and hasattr(torch, 'compile'):
                        logger.info("Applying torch.compile to TabNet network for GPU acceleration")
                        # Apply compilation with 'reduce-overhead' mode which is good for GPU performance
                        meta_learner.network = torch.compile(meta_learner.network, mode="reduce-overhead")
                        logger.info("Successfully applied torch.compile to TabNet network")
                except Exception as e:
                    logger.warning(f"Could not apply torch.compile: {str(e)}")
            # Update trial params with both training and model params
            params.update(fit_params)
            for key, value in params.items():
                trial.set_user_attr(key, value)
            trial.set_user_attr("model", meta_learner)
        elif meta_learner_type == "lgb":
            base_params = {
                "objective": "binary",
                "metric": ["binary_logloss", "auc"],
                "n_jobs": 4,
                "random_state": 19,
                "device": "cpu",
                "verbose": -1,
            }
            params = {
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.55, 0.65, step=0.005),
                "bagging_freq": trial.suggest_int("bagging_freq", 5, 14, step=1),
                "cat_smooth": trial.suggest_float("cat_smooth", 10.0, 30.0, step=0.1),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.50, 0.70, step=0.01),
                "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.20, step=0.005),
                "max_bin": trial.suggest_int("max_bin", 200, 700, step=10),
                "max_depth": trial.suggest_int("max_depth", 4, 10, step=1),
                "min_child_samples": trial.suggest_int("min_child_samples", 150, 320, step=10),
                "min_split_gain": trial.suggest_float("min_split_gain", 0.10, 0.20, step=0.01),
                "num_leaves": trial.suggest_int("num_leaves", 50, 150, step=5),
                "path_smooth": trial.suggest_float("path_smooth", 0.005, 0.60, step=0.005),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.5, 11.0, step=0.05),
                "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 5.0, step=0.05),
                "early_stopping_rounds": trial.suggest_int(
                    "early_stopping_rounds", 300, 700, step=10
                ),
            }
            params.update(base_params)
            meta_learner = LGBMClassifier(**params)
            trial.set_user_attr("model", meta_learner)
        elif meta_learner_type == "logistic":
            base_params = {
                "solver": "saga",  # Compatible with all penalties
                "max_iter": 1000,
                "random_state": 19,
            }
            params = {
                "C": trial.suggest_float("C", 0.001, 10.0, log=True),
                "penalty": trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"]),
                "solver": trial.suggest_categorical("solver", ["liblinear", "saga"]),
                "max_iter": trial.suggest_int("max_iter", 100, 1000, step=100),
            }

            if params["penalty"] == "elasticnet":
                params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
            params["class_weight"] = trial.suggest_categorical("class_weight", [None, "balanced"])
            if params["penalty"] == "elasticnet":
                params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)

            meta_learner = LogisticRegression(**params)
            trial.set_user_attr("model", meta_learner)
        elif meta_learner_type == "mlp":
            base_params = {"early_stopping": True, "random_state": 19}
            params = {
                "hidden_layer_sizes": (trial.suggest_int("hidden_units", 10, 100),),
                "alpha": trial.suggest_float("alpha", 0.0001, 0.1, log=True),
                "learning_rate_init": trial.suggest_float(
                    "learning_rate_init", 0.001, 0.1, log=True
                ),
                "max_iter": trial.suggest_int("max_iter", 100, 1000, step=100),
                "batch_size": trial.suggest_int("batch_size", 16, 128),
                "epochs": trial.suggest_int("epochs", 50, 200),
                "patience": trial.suggest_int("patience", 10, 50),
            }

            meta_learner = MLPClassifier(**params)
            trial.set_user_attr("model", meta_learner)
        elif meta_learner_type == "sgd":
            base_params = {"random_state": 19, "n_jobs": 4}
            params = {
                "loss": "modified_huber",
                "penalty": trial.suggest_categorical("penalty", ["l2", "elasticnet"]),
                "alpha": trial.suggest_float("alpha", 1e-6, 0.5, log=True),
                "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
                "learning_rate": trial.suggest_categorical(
                    "learning_rate", ["optimal", "constant", "adaptive", "invscaling"]
                ),
                "eta0": trial.suggest_float("eta0", 0.00001, 0.5, log=True),
                "max_iter": trial.suggest_int("max_iter", 400, 10000, step=200),
                "tol": trial.suggest_float("tol", 1e-6, 1e-2, log=True),
                "early_stopping": True,
                "validation_fraction": trial.suggest_float(
                    "validation_fraction", 0.05, 0.4, step=0.01
                ),
                "n_iter_no_change": trial.suggest_int("n_iter_no_change", 5, 50),
                "average": trial.suggest_categorical("average", [True, False]),
            }
            if params["penalty"] == "elasticnet":
                params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0, step=0.1)

            params.update(base_params)
            meta_learner = SGDClassifier(**params)
            trial.set_user_attr("model", meta_learner)
        elif meta_learner_type == "resnet":
            base_params = {
                "tree_method": "hist"  # Enforce CPU-only training
            }
            params = {
                "hidden_dim": trial.suggest_int("hidden_dim", 16, 128),
                "num_blocks": trial.suggest_int("num_blocks", 1, 4),
                "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
                "epochs": trial.suggest_int("epochs", 50, 300),
                "patience": trial.suggest_int("patience", 5, 30),
            }
            params.update(base_params)
            meta_learner = ResNetMetaLearner(**params)
            trial.set_user_attr("model", meta_learner)
        else:
            raise ValueError(f"Unsupported meta-learner type: {meta_learner_type}")

        # Train meta-learner
        try:
            if meta_learner_type == "tabnet":
                # Special handling for TabNet's training
                meta_learner.fit(
                    meta_features_np,
                    meta_targets_np,
                    eval_set=[(eval_features_np, eval_targets_np)],
                    max_epochs=params["max_epochs"],
                    patience=params["patience"],
                    batch_size=params["batch_size"],
                    virtual_batch_size=params["virtual_batch_size"],
                    eval_metric=params.get("eval_metric", ["logloss", "auc"]),
                    weights=1,  # Use automatic class weighting
                    drop_last=False,
                )
                # Get predictions on validation set
                y_proba = meta_learner.predict_proba(eval_features_np)[:, 1]
                # Find optimal threshold
                best_threshold, metrics = tune_threshold_for_precision_optimized(
                    y_proba,
                    eval_targets_np,
                    target_precision,
                    min_recall,  # Use numpy targets here
                )
                if metrics["recall"] < min_recall:
                    return -1.0
                return metrics["precision"]
            elif meta_learner_type == "resnet":
                # Special handling for ResNet meta-learner
                meta_learner.fit(
                    meta_features,
                    meta_targets,
                    X_val=eval_meta_features,
                    y_val=eval_meta_targets,
                    target_precision=target_precision,
                    min_recall=min_recall,
                )
                # ResNet has its own threshold tuning internally
                best_threshold = meta_learner.threshold

                # Get predictions using the tuned threshold
                y_proba = meta_learner.predict_proba(eval_meta_features)[:, 1]
                y_pred = (y_proba >= best_threshold).astype(int)

                # Calculate metrics with the model's threshold
                precision, recall = meta_learner._calculate_precision_recall(
                    eval_meta_targets, y_pred
                )
                metrics = {"precision": precision, "recall": recall}
            elif meta_learner_type == "lgb":
                # LightGBM specific early stopping
                early_stopping_rounds = params.pop("early_stopping_rounds")
                meta_learner.fit(
                    meta_features,
                    meta_targets,
                    eval_set=[(eval_meta_features, eval_meta_targets)],
                    callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds)],
                )
            elif (hasattr(meta_learner, "early_stopping_rounds")
                or hasattr(meta_learner, "early_stopping")
                and meta_learner_type != "sgd"
            ):
                meta_learner.fit(
                    meta_features,
                    meta_targets,
                    eval_set=[(eval_meta_features, eval_meta_targets)],
                    verbose=False,
                )
            else:
                meta_learner.fit(meta_features, meta_targets)

            # Get predictions on validation set (for non-ResNet models)
            if meta_learner_type not in ["resnet"]:
                if hasattr(meta_learner, "predict_proba"):
                    y_proba = meta_learner.predict_proba(eval_meta_features)[:, 1]
                else:
                    y_proba = meta_learner.predict(eval_meta_features)

                # Find optimal threshold
                best_threshold, metrics = tune_threshold_for_precision_optimized(
                    y_proba, eval_meta_targets, target_precision, min_recall
                )

            # If we couldn't find a threshold meeting criteria, penalize the score
            if metrics["recall"] < min_recall:
                return -1.0
            return metrics["precision"]

        except Exception as e:
            logger.error(f"Error in trial {trial.number}: {str(e)}")
            return -1.0

    # Initialize variables for batch training
    best_score = -float("inf")
    best_params = {}
    global_top_trials = []
    top_trials = []
    study_name = "ensemble_optimization"

    # Total trials to conduct
    total_trials = n_trials
    batch_size = 200
    num_batches = total_trials // batch_size
    if total_trials % batch_size != 0:
        num_batches += 1

    # Callback function for tracking top trials
    def callback(study, trial):
        nonlocal best_score, best_params, top_trials, best_model
        if trial.value > best_score:
            best_score = trial.value
            best_params = trial.params
            best_model = trial.user_attrs["model"]
            logger.info(f"Best params: {best_params}")
        # Create a record for the current trial
        current_run = (trial.value, trial.params, trial.number)
        top_trials.append(current_run)
        # Sort and keep only top 10 for this batch
        top_trials.sort(key=lambda x: x[0], reverse=True)
        top_trials[:] = top_trials[:10]
        if trial.number % 9 == 0:
            table_header = "| Rank | Trial # | Score | Parameters |"
            table_separator = "|------|---------|-------|------------|"
            table_rows = [
                f"| {i + 1} | {rec[2]} | {rec[0]:.4f} | {rec[1]} |"
                for i, rec in enumerate(top_trials)
            ]
            logger.info("Top trials in current batch:")
            logger.info(table_header)
            logger.info(table_separator)
            for row in table_rows:
                logger.info(row)

    # Loop over batches, resetting the sampler each time
    for batch in range(num_batches):
        # Generate a seed based on the current time
        random_seed = int(time.time())
        random_sampler = optuna.samplers.RandomSampler(seed=random_seed)

        # Create and run Optuna study with persistent storage
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=random_sampler
        )

        logger.info(
            f"Starting batch {batch + 1}/{num_batches} with new sampler (seed={random_seed})"
        )
        study.optimize(
            objective,
            n_trials=min(batch_size, total_trials - batch * batch_size),
            timeout=timeout,
            show_progress_bar=True,
            callbacks=[callback],
        )

        # Merge current batch's top trials with global_top_trials
        for trial_record in top_trials:
            global_top_trials.append(trial_record)
        # Keep only the best 10 across all batches
        global_top_trials.sort(key=lambda x: x[0], reverse=True)
        global_top_trials = global_top_trials[:10]

    # Get best parameters from global top trials
    if global_top_trials:
        best_score, best_params, best_trial_number = global_top_trials[0]
        logger.info(f"Best params: {best_params}")

    best_meta_learner = best_model

    # Get predictions and find optimal threshold
    if meta_learner_type == "tabnet":
        y_proba = best_meta_learner.predict_proba(eval_features_np)[:, 1]
    else:
        if hasattr(best_meta_learner, "predict_proba"):
            y_proba = best_meta_learner.predict_proba(eval_meta_features)[:, 1]
        else:
            y_proba = best_meta_learner.predict(eval_meta_features)

    # Find optimal threshold
    best_threshold, metrics = tune_threshold_for_precision_optimized(
        y_proba, eval_meta_targets, target_precision, min_recall
    )

    # Log final results
    logger.info(f"Final meta-learner: {meta_learner_type}")
    logger.info(f"Final threshold: {best_threshold:.4f}")
    logger.info(f"Final precision: {metrics['precision']:.4f}")
    logger.info(f"Final recall: {metrics['recall']:.4f}")
    # Log to MLflow
    mlflow.log_params(best_params)
    mlflow.log_param("meta_learner_type", meta_learner_type)
    mlflow.log_param("best_threshold", best_threshold)
    mlflow.log_metrics(
        {
            "final_precision": metrics["precision"],
            "final_recall": metrics["recall"],
            "final_f1": f1_score(eval_meta_targets, y_proba >= best_threshold),
        }
    )

    return best_meta_learner
