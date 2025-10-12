import os
import random

import numpy as np
import xgboost as xgb
from BorutaShap import BorutaShap

# Import own modules
from src.utils.logger import ExperimentLogger

# Define experiment name
experiment_name = "xgboost_soccer_prediction_25"
logger = ExperimentLogger(experiment_name)

# Import data at runtime to avoid global scope issues
from src.models.ensemble.data_utils import prepare_data
from src.models.StackedEnsemble.shared.data_loader_new import DataLoader
from src.utils.create_evaluation_set import import_selected_features_ensemble_new

# Set fixed seed and hash seed for determinism
SEED = 19
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Restrict parallel threads across various libraries
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

try:
    logger.info("Starting XGBoost model training")

    # Load data
    dataloader = DataLoader()
    X_train, y_train, X_test, y_test, X_eval, y_eval = dataloader.load_data()

    features = import_selected_features_ensemble_new(model_type="all")
    logger.info(f"Features: {len(features)}")
    X_train = prepare_data(X_train, features)
    X_test = prepare_data(X_test, features)
    X_eval = prepare_data(X_eval, features)

    # Log data shapes
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Testing data shape: {X_test.shape}")
    logger.info(f"Evaluation data shape: {X_eval.shape}")
    logger.info(
        f"Positive class ratio - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}, Eval: {y_eval.mean():.3f}"
    )

    # Define your XGBoost model
    params = {
        "alpha": 55.8,
        "colsample_bytree": 0.885,
        "eval_metric": ["aucpr", "error", "logloss"],
        "gamma": 4.43,
        "lambda": 6.94,
        "learning_rate": 0.15,
        "max_depth": 10,
        "min_child_weight": 635,
        "scale_pos_weight": 2.7,
        "subsample": 0.795,
    }
    xgb_clf = xgb.XGBClassifier(**params)

    # Run BorutaShap
    feature_selector = BorutaShap(
        model=xgb_clf,
        importance_measure="shap",  # or 'gini'
        classification=True,
        pvalue=0.10,
    )
    feature_selector.fit(
        X=X_train,
        y=y_train,
        n_trials=500,  # Number of Boruta iterations
        sample=False,  # Set to True for large datasets
        train_or_test="train",  # Use test set for SHAP values
        verbose=True,
    )

    # Get selected features
    selected_features = feature_selector.Subset().columns.tolist()
    print("Selected features:", selected_features)

    feature_selector.results_to_csv(filename="feature_importance")
    # Optionally, transform your data
    X_train_selected = feature_selector.transform(X_train)
except Exception as e:
    logger.error(f"Error: {e}")
    logger.error(f"Error type: {type(e)}")
    logger.error("Failed to run XGBoost model training")
    logger.error("Please check the data and model parameters")
    logger.error("Exiting the program")
