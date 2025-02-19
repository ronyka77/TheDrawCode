"""
EnsembleModel Module

This module implements an ensemble model that combines:
    • Soft Voting of base models: XGBoost, CatBoost, and LightGBM.
    • Stacking: A meta-learner is trained on the base model predictions.
    • Dynamic Weighting: Base model probabilities are combined by weighting each model based on its validation performance.
    • Probability Calibration: Base model probabilities can be calibrated using Platt scaling or isotonic regression.
    • Threshold Tuning and Balanced Data handling: The final probability is tuned using a grid search
        to achieve high precision while enforcing a minimum recall of 20%. Training is performed on
        balanced data using ADASYN oversampling.

Workflow:
1. Data Balancing: Use ADASYN to balance the training data.
2. Base Model Training: Train XGBoost, CatBoost, and LightGBM using CPU-only settings.
3. (Optional) Probability Calibration: Optionally calibrate base model probabilities.
4. Evaluation & Dynamic Weighting: Compute dynamic weights based on the validation performance.
5. Ensemble Prediction:
    • Soft Voting: Compute a weighted average of the base models' probabilities.
    • Stacking: Train a meta-learner on the base predictions.
    • Final Probability: Average the soft voting (with dynamic weights) and stacking probabilities.
6. Threshold Tuning: Use a grid search on the validation set to select the best threshold,
    ensuring recall is at least 15%.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from datetime import datetime
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.catboost
import mlflow.lightgbm
from mlflow.models.signature import infer_signature
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from pathlib import Path
import os
import sys
from imblearn.over_sampling import ADASYN
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import make_scorer
from sklearn.base import BaseEstimator, ClassifierMixin

# Add project root to Python path
try:
    project_root = Path(__file__).parent.parent
    if not project_root.exists():
        # Handle network path by using raw string
        project_root = Path(r"\\".join(str(project_root).split("\\")))
    sys.path.append(str(project_root))
    print(f"Project root ensemble_model: {project_root}")
except Exception as e:
    print(f"Error setting project root path: {e}")
    # Fallback to current directory if path resolution fails
    sys.path.append(os.getcwd().parent)
    print(f"Current directory ensemble_model: {os.getcwd().parent}")
    
os.environ['GIT_PYTHON_GIT_EXECUTABLE'] = "C:/Program Files/Git/bin/git.exe"

# Local imports
from utils.logger import ExperimentLogger
experiment_name = "ensemble_model_new"
logger = ExperimentLogger(experiment_name=experiment_name, log_dir='./logs/ensemble_model_new')

from utils.create_evaluation_set import create_ensemble_evaluation_set, import_selected_features_ensemble, import_training_data_ensemble, setup_mlflow_tracking
mlruns_dir = setup_mlflow_tracking(experiment_name)

class EnsembleModel(BaseEstimator, ClassifierMixin):
    """
    EnsembleModel trains a soft voting ensemble (XGBoost, CatBoost, LGBM) combined with stacking
    via a meta-learner. It applies threshold tuning on balanced data to yield a final model with a
    desired trade-off between precision and recall.
    
    New Features Added:
        - Dynamic Weighting of Base Models: Combines base model probabilities using weights computed from validation precision.
        - Probability Calibration: Optionally calibrates each base model's probabilities.
        - Alternative Meta Learner: Supports different meta-learner types (e.g., Logistic Regression).
    """
    def __init__(self, logger: ExperimentLogger = None, calibrate: bool = False, 
                    calibration_method: str = "sigmoid", individual_thresholding: bool = False,
                    meta_learner_type: str = 'xgb', dynamic_weighting: bool = True):
        self.logger = logger or ExperimentLogger(experiment_name="ensemble_model_new",
                                                    log_dir="./logs/ensemble_model_new")
        # Load selected features (assumed common to all models)
        self.selected_features = import_selected_features_ensemble('all')
        
        # Define base models with CPU-only settings:
        self.model_xgb = XGBClassifier(
            tree_method='hist',
            device='cpu',
            nthread=-1,
            objective='binary:logistic',
            learning_rate=0.08311627856474942,
            n_estimators=793,
            max_depth=5,
            random_state=138,
            colsample_bytree=0.9696188198654059,
            early_stopping_rounds=184,
            eval_metric=['error', 'auc', 'aucpr'],
            gamma=0.026573915810307977,
            min_child_weight=157,
            reg_alpha=0.00044056052563292266,
            reg_lambda=1.1040302522259011,
            scale_pos_weight=2.7864672907283436,
            subsample=0.3985764276456313,
            verbosity=0
        )
        self.model_cat = CatBoostClassifier(
            loss_function='Logloss',
            eval_metric='PRAUC',
            task_type='CPU',
            verbose=0,
            iterations=501,
            learning_rate=0.044415470772533286,
            depth=6,
            l2_leaf_reg=2.5412845413202043,
            border_count=128,
            subsample=0.7727569705672401,
            colsample_bylevel=0.9102359904680224,
            scale_pos_weight=2.12218845501254,
            min_data_in_leaf=225,
            early_stopping_rounds=335,
            leaf_estimation_iterations=4,
            grow_policy='Lossguide',
            random_seed=354
        )
        self.model_lgb = LGBMClassifier(
            objective='binary',
            metric=['binary_logloss', 'auc'],
            boosting_type='gbdt',
            device_type='cpu',
            verbose=-1,
            learning_rate=0.006225134508905533,
            max_depth=7,
            reg_lambda=0.4136861259036423,
            n_estimators=3902,
            num_leaves=55,
            early_stopping_rounds=382,
            feature_fraction=0.8438720741641329,
            bagging_freq=1,
            min_child_samples=35,
            bagging_fraction=0.8491638385070807,
            feature_fraction_bynode=0.9963248890782458,
            colsample_bytree=0.8641354771809475,
            min_child_weight=29,
            scale_pos_weight=2.118498238482285,
            random_state=229
        )
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
            self.dynamic_weights = {'xgb': 1/2, 'cat': 1/3, 'lgb': 1/6}  # default equal weights

    def _prepare_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensure X contains the required features and fill missing values."""
        X = X.copy()
        X = X.reindex(columns=self.selected_features, fill_value=0)
        return X.astype(np.float32)

    def _tune_individual_threshold(self, probs: np.ndarray, targets: pd.Series, 
                                    grid_start: float = 0.4, grid_stop: float = 0.7, grid_step: float = 0.01, 
                                    min_recall: float = 0.50) -> float:
        """
        Tune threshold for a single model's probabilities by iterating over a grid and selecting the value
        that maximizes precision while ensuring recall is at least min_recall.
        """
        best_threshold = 0.5
        best_precision = -np.inf
        for thresh in np.arange(grid_start, grid_stop, grid_step):
            preds = (probs >= thresh).astype(int)
            prec = precision_score(targets, preds, zero_division=0)
            rec = recall_score(targets, preds, zero_division=0)
            if rec >= min_recall and prec > best_precision:
                best_precision = prec
                best_threshold = thresh
        return best_threshold

    def _tune_threshold(self, probs: np.ndarray, targets: pd.Series, 
                        grid_start: float = 0.0, grid_stop: float = 1.0, grid_step: float = 0.01,
                        required_recall: float = 0.50, target_precision: float = 0.50) -> (float, dict):
        """
        Tune the global threshold by scanning a grid.
        In addition to ensuring a minimum recall, also try to select a threshold
        that produces precision of at least target_precision.
        """
        best_threshold = None
        best_precision = -np.inf
        max_recall = -np.inf
        max_recall_threshold = grid_start
        candidate_metrics = {}

        for thresh in np.arange(grid_start, grid_stop, grid_step):
            preds = (probs >= thresh).astype(int)
            prec = precision_score(targets, preds, zero_division=0)
            rec = recall_score(targets, preds, zero_division=0)
            # Track maximum recall for fallback.
            if rec > max_recall:
                max_recall = rec
                max_recall_threshold = thresh
            # Select candidate if recall meets the minimum and precision meets or exceeds target.
            if rec >= required_recall and prec >= best_precision:
                if prec > best_precision:
                    best_precision = prec
                    best_threshold = thresh
                    candidate_metrics = {
                        "precision": prec,
                        "recall": rec,
                        "f1": f1_score(targets, preds, zero_division=0)
                    }
        if best_threshold is None:
            # Fallback: choose the threshold with maximum recall, even if precision < target_precision.
            # Find threshold with maximum precision that still meets minimum recall requirement
            best_threshold = max_recall_threshold
            best_precision = precision_score(targets, (probs >= best_threshold).astype(int), zero_division=0)
            for thresh in np.arange(grid_start, grid_stop, grid_step):
                preds = (probs >= thresh).astype(int)
                prec = precision_score(targets, preds, zero_division=0)
                rec = recall_score(targets, preds, zero_division=0)
                if rec >= required_recall and prec > best_precision:
                    best_precision = prec
                    best_threshold = thresh
            candidate_metrics = {
                "precision": precision_score(targets, (probs >= best_threshold).astype(int), zero_division=0),
                "recall": max_recall,
                "f1": f1_score(targets, (probs >= best_threshold).astype(int), zero_division=0)
            }
            self.logger.info(f"No threshold achieved target precision; selected threshold with maximum recall. selected_threshold: {best_threshold}, max_recall: {max_recall}")
        return best_threshold, candidate_metrics

    def _compute_dynamic_weights(self, targets: pd.Series, p_xgb: np.ndarray, p_cat: np.ndarray, p_lgb: np.ndarray) -> dict:
        """
        Compute dynamic weights for each base model based on their precision on the validation set.
        The weights are normalized such that they sum to 1.
        """
        # Use a default threshold (e.g., 0.5) for initial binary predictions.
        preds_xgb = (p_xgb >= 0.5).astype(int)
        preds_cat = (p_cat >= 0.5).astype(int)
        preds_lgb = (p_lgb >= 0.5).astype(int)
        
        prec_xgb = precision_score(targets, preds_xgb, zero_division=0)
        prec_cat = precision_score(targets, preds_cat, zero_division=0)
        prec_lgb = precision_score(targets, preds_lgb, zero_division=0)
        
        self.logger.info("Validation precision for dynamic weighting:", 
                            extra={"xgb": prec_xgb, "cat": prec_cat, "lgb": prec_lgb})
        total = prec_xgb + prec_cat + prec_lgb + np.finfo(np.float32).eps  # add epsilon to avoid division by 0.
        weights = {
            'xgb': prec_xgb / total,
            'cat': prec_cat / total,
            'lgb': prec_lgb / total
        }
        self.logger.info("Dynamic weights computed:", extra=weights)
        return weights

    def _search_optimal_weights(self, p_xgb, p_cat, p_lgb, y_val):
        best_weights = {"xgb": 1/2, "cat": 1/3, "lgb": 1/6}
        best_precision = -np.inf
        # Define candidate weights in increments (for example, 0.1 steps)
        candidates = np.arange(0.0, 1.05, 0.1)
        for w_xgb in candidates:
            for w_cat in candidates:
                for w_lgb in candidates:
                    total = w_xgb + w_cat + w_lgb
                    # Only consider combinations with non-zero total weight
                    if total == 0:
                        continue
                    # Normalize weights.
                    norm_weights = {"xgb": w_xgb/total,
                                    "cat": w_cat/total,
                                    "lgb": w_lgb/total}
                    # Combine probabilities with current weights.
                    p_soft = (norm_weights["xgb"] * p_xgb +
                              norm_weights["cat"] * p_cat +
                              norm_weights["lgb"] * p_lgb)
                    # If using meta learner probabilities, combine as before:
                    meta_features = np.column_stack((p_xgb, p_cat, p_lgb))
                    p_stack = self.meta_learner.predict_proba(meta_features)[:, 1]
                    final_probs = (p_soft + p_stack) / 2.0
                    preds = (final_probs >= self.optimal_threshold).astype(int)
                    precision = precision_score(y_val, preds, zero_division=0)
                    recall = recall_score(y_val, preds, zero_division=0)
                    # Check if recall is above the minimum threshold, and precision improves.
                    if recall >= 0.30 and precision > best_precision:
                        best_precision = precision
                        best_weights = norm_weights
        return best_weights

    def _initialize_meta_learner(self):
        """
        Initialize the meta learner based on the provided meta_learner_type.
        """
        if self.meta_learner_type.lower() == 'logistic':
            # Cross-validated logistic regression finds optimal regularization automatically.
            self.meta_learner = LogisticRegressionCV(cv=5, penalty='l2', scoring='f1', solver='liblinear')
            self.logger.info("Meta learner initialized as LogisticRegressionCV.")
        elif self.meta_learner_type.lower() == 'xgb':
            from xgboost import XGBClassifier
            self.meta_learner = XGBClassifier(
                tree_method='hist',
                device='cpu',
                nthread=-1,
                objective='binary:logistic',
                learning_rate=0.05,
                n_estimators=300,
                max_depth=5,
                scale_pos_weight=2.19,  # adjust this if needed for cost sensitivity
                random_state=19
            )
            self.logger.info("Meta learner initialized as XGBClassifier.")
        elif self.meta_learner_type.lower() == 'mlp':
            from sklearn.neural_network import MLPClassifier
            self.meta_learner = MLPClassifier(hidden_layer_sizes=(50, 25), activation='relu',
                                                solver='adam', random_state=19, max_iter=200)
            self.logger.info("Meta learner initialized as MLPClassifier.")
        else:
            from sklearn.linear_model import LogisticRegression
            self.meta_learner = LogisticRegression(solver='liblinear', class_weight='balanced')
            self.logger.info("Meta learner initialized as standard LogisticRegression.")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame, y_test: pd.Series,
                    X_val: pd.DataFrame, y_val: pd.Series):
        """
        Train base models, fit meta-learner, update dynamic weights using the validation set, and tune the threshold.
        """
        # Prepare data with required features
        X_train = self._prepare_data(X_train)
        X_val = self._prepare_data(X_val)
        X_test = self._prepare_data(X_test)
        
        # Optionally balance the training data
        X_train_bal, y_train_bal = X_train, y_train
        
        # Train base models
        self.logger.info("Training base models...")
        self.logger.info("Training XGBoost...")
        self.model_xgb.fit(X_train_bal, y_train_bal, eval_set=[(X_test, y_test)], verbose=False)
        self.logger.info("Training CatBoost...")
        self.model_cat.fit(X_train_bal, y_train_bal, eval_set=(X_test, y_test))
        self.logger.info("Training LightGBM...")
        # Native LightGBM training with early stopping  
        self.model_lgb.fit(
            X_train_bal, y_train_bal,
            eval_set=(X_test, y_test)
        )
        
        # Enhanced version with automatic ensemble calibration
        self.calibrated_model = CalibratedClassifierCV(
            estimator=self.model_lgb,
            method='sigmoid',
            ensemble='auto',  # Automatically select best calibration method
            n_jobs=-1
        ).fit(
            X_train_bal,  # Use separate validation set for calibration
            y_train_bal, eval_set=(X_test, y_test)
        )
        self.logger.info("Base models trained successfully.")
        
        # Get base predictions on validation set.
        p_xgb = self.model_xgb.predict_proba(X_val)[:, 1]
        p_cat = self.model_cat.predict_proba(X_val)[:, 1]
        p_lgb = self.calibrated_model.predict_proba(X_val)[:, 1]
        
        # Form the meta-feature array.
        meta_features = np.column_stack((p_xgb, p_cat, p_lgb))
        
        # Initialize meta learner.
        self._initialize_meta_learner()
        
        # Fit the meta learner using the meta_feature array.
        self.meta_learner.fit(meta_features, y_val)
        self.logger.info("Meta learner fitted with meta features.")
        
        # Update dynamic weights based on validation probabilities.
        self.dynamic_weights = self._search_optimal_weights(p_xgb, p_cat, p_lgb, y_val)
        self.logger.info(f"XGBoost weight updated: {self.dynamic_weights['xgb']:.4f}")
        self.logger.info(f"CatBoost weight updated: {self.dynamic_weights['cat']:.4f}") 
        self.logger.info(f"LightGBM weight updated: {self.dynamic_weights['lgb']:.4f}")
        
        # Compute ensemble probabilities using dynamic weights.
        if self.dynamic_weighting:
            p_soft = (self.dynamic_weights['xgb'] * p_xgb +
                      self.dynamic_weights['cat'] * p_cat +
                      self.dynamic_weights['lgb'] * p_lgb)
        else:
            p_soft = (p_xgb + p_cat + p_lgb) / 3.0
        
        # Get meta learner predictions.
        p_stack = self.meta_learner.predict_proba(meta_features)[:, 1]
        final_probs = (p_soft + p_stack) / 2.0
        
        # Tune threshold (using the refined _tune_threshold method that enforces recall and precision criteria).
        self.optimal_threshold, tuning_metrics = self._tune_threshold(final_probs, y_val)
        self.logger.info("Threshold tuning completed, optimal_threshold: %.2f", self.optimal_threshold)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return final ensemble probability for the positive class.
        If individual thresholding is disabled and dynamic weighting is enabled,
        the base models' probabilities are combined using the calibrated dynamic weights.
        """
        X = self._prepare_data(X)
        p_xgb = self.model_xgb.predict_proba(X)[:, 1]
        p_cat = self.model_cat.predict_proba(X)[:, 1]
        p_lgb = self.calibrated_model.predict_proba(X)[:, 1]
        
        if self.dynamic_weighting and not self.individual_thresholding:
            p_soft = (self.dynamic_weights['xgb'] * p_xgb +
                      self.dynamic_weights['cat'] * p_cat +
                      self.dynamic_weights['lgb'] * p_lgb)
        else:
            p_soft = (p_xgb + p_cat + p_lgb) / 3.0
            
        meta_features = np.column_stack((p_xgb, p_cat, p_lgb))
        p_stack = self.meta_learner.predict_proba(meta_features)[:, 1]
        final_probs = (p_soft + p_stack) / 2.0
        return final_probs

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make final binary predictions.
        If individual thresholding is enabled, use the tuned thresholds and majority vote.
        Otherwise, use the global threshold on the ensemble probability (with dynamic weighting if enabled).
        """
        X = self._prepare_data(X)
        p_xgb = self.model_xgb.predict_proba(X)[:, 1]
        p_cat = self.model_cat.predict_proba(X)[:, 1]
        p_lgb = self.calibrated_model.predict_proba(X)[:, 1]
        
        if self.individual_thresholding and hasattr(self, 'tuned_thresholds'):
            preds_xgb = (p_xgb >= self.tuned_thresholds['xgb']).astype(int)
            preds_cat = (p_cat >= self.tuned_thresholds['cat']).astype(int)
            preds_lgb = (p_lgb >= self.tuned_thresholds['lgb']).astype(int)
            final_preds = ((preds_xgb + preds_cat + preds_lgb) >= 2).astype(int)
            return final_preds
        else:
            if self.dynamic_weighting:
                p_soft = (self.dynamic_weights['xgb'] * p_xgb +
                          self.dynamic_weights['cat'] * p_cat +
                          self.dynamic_weights['lgb'] * p_lgb)
            else:
                p_soft = (p_xgb + p_cat + p_lgb) / 3.0
                
            meta_features = np.column_stack((p_xgb, p_cat, p_lgb))
            p_stack = self.meta_learner.predict_proba(meta_features)[:, 1]
            final_probs = (p_soft + p_stack) / 2.0
            preds = (final_probs >= self.optimal_threshold).astype(int)
            return preds

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Evaluate the ensemble model on the provided dataset.
        This method computes key metrics (precision, recall, f1_score, AUC)
        and logs them. It also checks if the recall exceeds the required
        minimum threshold (default 15%).
        
        Returns:
            A dictionary with computed metrics.
        """
        # Compute predictions and probabilities.
        predictions = self.predict(X)
        probs = self.predict_proba(X)
        
        precision = precision_score(y, predictions, zero_division=0)
        recall = recall_score(y, predictions, zero_division=0)
        f1 = f1_score(y, predictions, zero_division=0)
        try:
            auc = roc_auc_score(y, probs)
        except Exception as e:
            auc = None
            self.logger.info("AUC computation failed.", extra={"error": str(e)})
        
        # Check if recall meets the required threshold.
        required_recall = 0.30
        recall_flag = recall >= required_recall
        
        # Log evaluation metrics.
        self.logger.info(f"Precision computed: {precision}")
        self.logger.info(f"Recall computed: {recall}")
        self.logger.info(f"F1 score computed: {f1}")
        self.logger.info(f"AUC computed: {auc}")
        self.logger.info(f"Optimal threshold: {self.optimal_threshold}")
        self.logger.info(f"Required recall threshold: {required_recall}")
        self.logger.info(f"Recall target status: {recall_flag}")
        if not recall_flag:
            self.logger.info("Recall below the required threshold.", extra={"observed_recall": recall, "required_recall": required_recall})
        else:
            self.logger.info("Recall meets the required threshold.", extra={"observed_recall": recall, "required_recall": required_recall})
        
        return {"precision": precision,
                "recall": recall,
                "f1_score": f1,
                "auc": auc,
                "optimal_threshold": self.optimal_threshold,
                "recall_flag": recall_flag}

if __name__ == "__main__":    
    # Load data (assuming these functions return pandas DataFrames/Series)
    selected_features = import_selected_features_ensemble('all')
    X_train, y_train, X_test, y_test = import_training_data_ensemble()
    X_val, y_val = create_ensemble_evaluation_set()
    
    # Use only selected features
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    X_val = X_val[selected_features]
    
    # Instantiate ensemble model with probability calibration, dynamic weighting, and an alternative meta learner
    ensemble_model = EnsembleModel(
        logger=logger, calibrate=True, calibration_method="sigmoid",
        meta_learner_type="xgb", dynamic_weighting=True
    )
    
    # Start MLflow run
    with mlflow.start_run(run_name="ensemble_model_new"):
        mlflow.log_params({
            "model_type": "ensemble",
            "calibrate": True,
            "calibration_method": "sigmoid",
            "meta_learner_type": "xgb",
            "dynamic_weighting": True,
            "threshold_tuning": True
        })
        
        # Train the ensemble model
        ensemble_model.train(X_train, y_train, X_test, y_test, X_val, y_val)
        
        # Evaluate on validation data
        predictions = ensemble_model.predict(X_val)
        final_probs = ensemble_model.predict_proba(X_val)
        evaluation_metrics = ensemble_model.evaluate(X_val, y_val)
        # Calculate metrics
        precision = precision_score(y_val, predictions, zero_division=0)
        recall = recall_score(y_val, predictions, zero_division=0)
        f1 = f1_score(y_val, predictions, zero_division=0)
        
        # Log metrics
        mlflow.log_metrics({
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "optimal_threshold": ensemble_model.optimal_threshold
        })
        
        # Log model
        signature = infer_signature(X_val.astype('float64'), predictions.astype('float64'))
        mlflow.sklearn.log_model(
            artifact_path="ensemble_model",
            sk_model=ensemble_model,
            signature=signature,
            registered_model_name=f"ensemble_{datetime.now().strftime('%Y%m%d_%H%M')}"
        )
        
        print("Ensemble predictions:", predictions)
        print("Optimal threshold:", ensemble_model.optimal_threshold)