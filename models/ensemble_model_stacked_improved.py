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
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, precision_score, recall_score, confusion_matrix, roc_curve, accuracy_score
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
from sklearn.metrics import precision_score, recall_score, roc_auc_score, precision_recall_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from pathlib import Path
import os
import sys
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.combine import SMOTEENN
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import make_scorer
from sklearn.base import BaseEstimator, ClassifierMixin
import shap
from typing import Dict, List, Tuple, Optional, Union

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
experiment_name = "ensemble_model_improved"
logger = ExperimentLogger(experiment_name=experiment_name, log_dir='./logs/ensemble_model_improved')

# Import data loading functions from utils
from utils.create_evaluation_set import import_selected_features_ensemble, setup_mlflow_tracking
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
        - Model Diagnostics: Functions to analyze prediction errors and understand model behavior.
        - Feature Selection: Improved feature selection based on importance.
        - Class Imbalance Handling: Proper implementation of ADASYN for balancing training data.
        - Cross-Validation: Implementation of proper cross-validation techniques.
        - Precision Optimization: Enhanced threshold tuning focused on precision.
    """
    def __init__(self, logger: ExperimentLogger = None, calibrate: bool = False, 
                    calibration_method: str = "sigmoid", individual_thresholding: bool = False,
                    meta_learner_type: str = 'xgb', dynamic_weighting: bool = True,
                    extra_base_model_type: str = 'random_forest', 
                    sampling_strategy: float = 0.7,
                    complexity_penalty: float = 0.01,
                    target_precision: float = 0.60,
                    required_recall: float = 0.40):
        self.logger = logger or ExperimentLogger(experiment_name="ensemble_model_improved",
                                                    log_dir="./logs/ensemble_model_improved")
        # Load selected features (assumed common to all models)
        self.selected_features = import_selected_features_ensemble('all')
        self.required_recall = required_recall
        self.sampling_strategy = sampling_strategy  # For ADASYN resampling
        self.complexity_penalty = complexity_penalty  # For regularization
        self.target_precision = target_precision  # For threshold tuning
        
        # Define base models with CPU-only settings with reduced complexity to avoid overfitting:
        self.model_xgb = XGBClassifier(
            tree_method='hist',
            device='cpu',
            n_jobs=-1,
            objective='binary:logistic',
            learning_rate=0.05,
            n_estimators=500,  # Reduced from 793
            max_depth=6,  # Reduced from 11
            random_state=98,
            colsample_bytree=0.7,
            early_stopping_rounds=300,
            eval_metric=['logloss', 'auc'],
            gamma=0.8,
            min_child_weight=50,
            reg_alpha=1.0,  # Increased regularization
            reg_lambda=2.0,  # Increased regularization
            scale_pos_weight=2.19,  # Adjusted for class imbalance
            subsample=0.8
        )
        self.model_cat = CatBoostClassifier(
            learning_rate=0.05,
            depth=6,  # Reduced from 10
            min_data_in_leaf=20,
            subsample=0.8,
            colsample_bylevel=0.8,
            reg_lambda=5.0,  # Increased regularization
            leaf_estimation_iterations=2,
            bagging_temperature=1.0,
            scale_pos_weight=2.19,  # Adjusted for class imbalance
            early_stopping_rounds=100,
            loss_function='Logloss',
            eval_metric='AUC',
            task_type='CPU',
            thread_count=-1,
            random_seed=26,
            verbose=100
        )
        self.model_lgb = LGBMClassifier(
            learning_rate=0.05,
            num_leaves=32,  # Reduced from 52
            max_depth=4,
            min_child_samples=100,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            reg_alpha=2.0,  # Increased regularization
            reg_lambda=2.0,  # Increased regularization
            min_split_gain=0.1,
            objective='binary',
            metric=['binary_logloss', 'auc'],
            verbose=-1,
            n_jobs=-1,
            device='cpu',
            early_stopping_rounds=300,
            random_state=19
        )
        
        # Initialize the extra base model based on the selected type with reduced complexity
        self.extra_base_model_type = extra_base_model_type.lower()
        if self.extra_base_model_type == 'random_forest':
            self.model_extra = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,  # Reduced from 24
                min_samples_split=10,
                min_samples_leaf=10,  # Increased from 5
                max_features='sqrt',  # Changed from None to reduce complexity
                bootstrap=True,
                class_weight='balanced',  # Added class weight
                criterion='entropy',
                random_state=42,
                n_jobs=-1
            )
            self.logger.info("Extra base model initialized as RandomForestClassifier.")
        elif self.extra_base_model_type == 'svm':
            self.model_extra = SVC(
                probability=True,
                kernel='rbf',
                C=1.0,
                gamma='scale',
                class_weight='balanced',
                random_state=42
            )
            self.logger.info("Extra base model initialized as SVC.")
        elif self.extra_base_model_type == 'mlp':
            self.model_extra = MLPClassifier(
                hidden_layer_sizes=(20),  # Reduced from 50
                activation='logistic',
                solver='adam',
                alpha=0.01,  # Increased regularization
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=200,  # Reduced from 324
                early_stopping=True,
                validation_fraction=0.2,
                batch_size=64,
                random_state=253
            )
            self.logger.info("Extra base model initialized as MLPClassifier.")
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
            self.dynamic_weights = {'xgb': 1/3, 'cat': 1/3, 'lgb': 1/6, 'extra': 1/6}  # default weights
            
    # PHASE 1: DIAGNOSTIC METHODS
    
    def detect_data_leakage(self, X_train: pd.DataFrame, X_test: pd.DataFrame, X_val: pd.DataFrame) -> Dict:
        """
        Check for potential data leakage between datasets by detecting duplicate rows.
        
        Args:
            X_train: Training dataset
            X_test: Test dataset
            X_val: Validation dataset
            
        Returns:
            Dictionary with overlap information
        """
        self.logger.info("Checking for data leakage between datasets...")
        
        # Create unique identifier for each row (converting to tuples)
        train_rows = set(X_train.apply(tuple, axis=1))
        test_rows = set(X_test.apply(tuple, axis=1))
        val_rows = set(X_val.apply(tuple, axis=1))
        
        # Calculate overlaps
        train_test_overlap = len(train_rows.intersection(test_rows))
        train_val_overlap = len(train_rows.intersection(val_rows))
        test_val_overlap = len(test_rows.intersection(val_rows))
        
        # Calculate percentages
        train_test_pct = train_test_overlap / len(train_rows) * 100 if len(train_rows) > 0 else 0
        train_val_pct = train_val_overlap / len(train_rows) * 100 if len(train_rows) > 0 else 0
        test_val_pct = test_val_overlap / len(test_rows) * 100 if len(test_rows) > 0 else 0
        
        results = {
            'train_test_overlap': train_test_overlap,
            'train_val_overlap': train_val_overlap,
            'test_val_overlap': test_val_overlap,
            'train_test_overlap_pct': train_test_pct,
            'train_val_overlap_pct': train_val_pct,
            'test_val_overlap_pct': test_val_pct
        }
        
        # Log results
        self.logger.info("Data leakage check completed", extra=results)
        
        # Issue warnings if significant overlap exists
        if train_test_pct > 1:
            self.logger.warning(f"WARNING: Training and test sets have {train_test_pct:.2f}% overlap!")
        if train_val_pct > 1:
            self.logger.warning(f"WARNING: Training and validation sets have {train_val_pct:.2f}% overlap!")
        if test_val_pct > 1:
            self.logger.warning(f"WARNING: Test and validation sets have {test_val_pct:.2f}% overlap!")
            
        return results
    
    def explain_predictions(self, X_val):
        """
        Generate feature importance explanations using SHAP values on validation data.
        Analyzes feature contributions to model predictions on the most recent data.
        
        Args:
            X_val: Validation features (most recent data) to explain
            
        Returns:
            Dictionary with explanation data
        """
        self.logger.info("Generating feature importance explanations for validation data (most recent data)...")
        
        # Initialize SHAP explainer for the XGBoost model
        try:
            import shap
            explainer = shap.TreeExplainer(self.model_xgb)
            
            # Calculate SHAP values - limit to 500 samples max for performance
            max_samples = min(500, len(X_val))
            X_sample = X_val.sample(max_samples, random_state=42) if len(X_val) > max_samples else X_val
            shap_values = explainer.shap_values(X_sample)
            
            # Get predictions on the sample
            probas = self.predict_proba(X_sample)
            preds = (probas >= self.optimal_threshold).astype(int)
            
            # Overall feature importance (mean absolute SHAP value)
            if isinstance(shap_values, list):
                # For multi-class models, use the class 1 (positive class) values
                shap_values_class1 = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                shap_values_class1 = shap_values
                
            # Calculate mean absolute SHAP values for each feature
            mean_abs_shap = np.abs(shap_values_class1).mean(axis=0)
            feature_importance = pd.DataFrame({
                'feature': X_val.columns,
                'importance': mean_abs_shap
            }).sort_values('importance', ascending=False)
            
            # Log overall feature importance
            self.logger.info("SHAP feature importance on validation data:", extra={
                'feature_importance': feature_importance.to_dict('records')
            })
            
            # Log to MLflow if active
            if mlflow.active_run() is not None:
                top_features = feature_importance.head(20)
                for idx, row in top_features.iterrows():
                    mlflow.log_metric(f"shap_importance_{row['feature']}", row['importance'])
            
            # Return explanations
            return {
                'feature_importance': feature_importance.to_dict('records'),
                'num_samples_analyzed': len(X_sample)
            }
                
        except Exception as e:
            self.logger.error(f"Error generating explanations: {str(e)}")
            return {
                'error': str(e),
                'feature_importance': None
            }

    def analyze_prediction_errors(self, X_val, y_val):
        """
        Analyze prediction errors on the validation set (most recent data).
        This provides insights into the model's performance on future data.
        
        Args:
            X_val: Validation features (most recent data)
            y_val: True validation labels
            
        Returns:
            Dictionary with error analysis metrics
        """
        self.logger.info("Analyzing prediction errors on validation set (most recent data)...")
        
        # Get predictions using the optimal threshold from test data
        proba = self.predict_proba(X_val)
        pred = (proba >= self.optimal_threshold).astype(int)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_val, pred).ravel()
        
        # Log confusion matrix
        conf_matrix = {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
        }
        
        # Calculate and log error distribution
        errors = np.abs(y_val - proba)
        error_stats = {
            'mean_error': float(np.mean(errors)),
            'median_error': float(np.median(errors)),
            'std_error': float(np.std(errors)),
            'max_error': float(np.max(errors)),
            'quartiles': [float(np.percentile(errors, q)) for q in [25, 50, 75]],
            'num_predictions': len(pred)
        }
        
        # Combine results
        results = {
            'confusion_matrix': conf_matrix,
            'error_distribution': error_stats
        }
        
        # Log error analysis results
        self.logger.info("Prediction error analysis on validation set:", extra=results)
        
        # Log key metrics to MLflow if active
        if mlflow.active_run() is not None:
            mlflow.log_metrics({
                'val_true_positives': tp,
                'val_false_positives': fp,
                'val_true_negatives': tn,
                'val_false_negatives': fn,
                'val_mean_error': error_stats['mean_error'],
                'val_median_error': error_stats['median_error']
            })
        
        return results

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
                        target_precision: float = 0.50) -> (float, dict):
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
            if rec >= self.required_recall and prec >= best_precision:
                if prec > best_precision:
                    best_precision = prec
                    best_threshold = thresh
                    candidate_metrics = {
                        "precision": prec,
                        "recall": rec,
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
                if rec >= self.required_recall and prec > best_precision:
                    best_precision = prec
                    best_threshold = thresh
            candidate_metrics = {
                "precision": precision_score(targets, (probs >= best_threshold).astype(int), zero_division=0),
                "recall": max_recall,
            }
            self.logger.info(f"No threshold achieved target precision; selected threshold with maximum recall. selected_threshold: {best_threshold}, max_recall: {max_recall}")
        return best_threshold, candidate_metrics

    def _compute_dynamic_weights(self, p_xgb: np.ndarray, p_cat: np.ndarray, 
                                p_lgb: np.ndarray, p_extra: np.ndarray, 
                                targets: pd.Series) -> dict:
        """
        Compute dynamic weights for each base model based on their precision on the validation set.
        The weights are normalized such that they sum to 1.
        
        Args:
            p_xgb: XGBoost predicted probabilities
            p_cat: CatBoost predicted probabilities
            p_lgb: LightGBM predicted probabilities
            p_extra: Extra model predicted probabilities
            targets: True labels
            
        Returns:
            Dictionary with normalized weights for each model
        """
        # Handle potential NaN values in predictions
        p_xgb = np.nan_to_num(p_xgb, nan=0.0, posinf=1.0, neginf=0.0)
        p_cat = np.nan_to_num(p_cat, nan=0.0, posinf=1.0, neginf=0.0)
        p_lgb = np.nan_to_num(p_lgb, nan=0.0, posinf=1.0, neginf=0.0)
        p_extra = np.nan_to_num(p_extra, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Determine optimal threshold for each model using a simple grid search
        # to find the best precision rather than using a fixed threshold
        def find_best_threshold(probs, labels):
            best_threshold = 0.5
            best_precision = 0
            best_recall = 0
            # Only search if we have positive samples
            if sum(labels) > 0:
                for thresh in np.linspace(0.3, 0.7, 9):  # 9 thresholds from 0.3 to 0.7
                    preds = (probs >= thresh).astype(int)
                    # Ensure we have both classes for meaningful metrics
                    if len(np.unique(preds)) < 2:
                        continue
                        
                    # Calculate metrics with zero_division=0 to avoid warnings
                    with np.errstate(divide='ignore', invalid='ignore'):
                        prec = precision_score(labels, preds, zero_division=0)
                        rec = recall_score(labels, preds, zero_division=0)
                    
                    # Handle NaN values
                    if np.isnan(prec):
                        continue
                        
                    if prec > best_precision:
                        best_precision = prec
                        best_threshold = thresh
                        best_recall = rec
            
            return best_threshold, best_precision, best_recall
        
        xgb_thresh, prec_xgb, rec_xgb = find_best_threshold(p_xgb, targets)
        cat_thresh, prec_cat, rec_cat = find_best_threshold(p_cat, targets)
        lgb_thresh, prec_lgb, rec_lgb = find_best_threshold(p_lgb, targets)
        extra_thresh, prec_extra, rec_extra = find_best_threshold(p_extra, targets)
        
        self.logger.info(f"Model thresholds: XGB={xgb_thresh:.2f}, CatBoost={cat_thresh:.2f}, "
                        f"LightGBM={lgb_thresh:.2f}, Extra={extra_thresh:.2f}")
        self.logger.info(f"Model precision: XGB={prec_xgb:.4f}, CatBoost={prec_cat:.4f}, "
                        f"LightGBM={prec_lgb:.4f}, Extra={prec_extra:.4f}")
        self.logger.info(f"Model recall: XGB={rec_xgb:.4f}, CatBoost={rec_cat:.4f}, "
                        f"LightGBM={rec_lgb:.4f}, Extra={rec_extra:.4f}")
        # Calculate scores using precision
        use_precision_for_weights = True
        if use_precision_for_weights:
            scores = {
                'xgb': prec_xgb,
                'cat': prec_cat,
                'lgb': prec_lgb,
                'extra': prec_extra
            }
            self.logger.info("Using precision for dynamic weights calculation")
        else:
            scores = {
                'xgb': rec_xgb,
                'cat': rec_cat,
                'lgb': rec_lgb,
                'extra': rec_extra
            }
            self.logger.info("Using recall for dynamic weights calculation")
        
        # If all scores are 0 or close to 0, use equal weights to avoid division issues
        if sum(scores.values()) < 1e-10:
            self.logger.warning("All model scores are approximately 0, using equal weights")
            weights = {
                'xgb': 0.25,
                'cat': 0.25,
                'lgb': 0.25,
                'extra': 0.25
            }
        else:
            # Avoid any potential division by zero by adding a small epsilon
            epsilon = 1e-10
            total = sum(scores.values()) + epsilon * len(scores)
            
            # Calculate initial weights
            raw_weights = {
                model: (score + epsilon) / total for model, score in scores.items()
            }
            
            # Ensure minimum weight for each model (ensure diversity)
            min_weight = 0.05
            for model in raw_weights:
                if raw_weights[model] < min_weight:
                    raw_weights[model] = min_weight
            
            # Re-normalize to sum to 1
            total_adjusted = sum(raw_weights.values())
            weights = {model: weight/total_adjusted for model, weight in raw_weights.items()}
        
        self.logger.info(f"Dynamic weights computed: {weights}")
        
        # Verify weights sum to approximately 1
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 1e-6:  # Check within reasonable floating point precision
            self.logger.warning(f"Dynamic weights sum ({weight_sum}) differs significantly from 1.0")
            # Normalize one more time to be sure
            weights = {k: v/weight_sum for k, v in weights.items()}
        
        return weights

    def _initialize_meta_learner(self) -> BaseEstimator:
        """
        Initialize the meta learner based on the provided meta_learner_type.
        
        Returns:
            Initialized meta-learner model
        """
        if self.meta_learner_type.lower() == 'logistic':
            meta_learner = LogisticRegressionCV(
                cv=5,                 
                scoring='precision',
                solver='liblinear',
                class_weight='balanced',
                max_iter=1000,
                random_state=42,
                n_jobs=-1,
                verbose=0,
                tol=1e-4
            )
            self.logger.info("Meta learner initialized as LogisticRegressionCV.")
        elif self.meta_learner_type.lower() == 'xgb':
            meta_learner = XGBClassifier(
                tree_method='hist',
                device='cpu',
                nthread=-1,
                objective='binary:logistic',
                learning_rate=0.05,
                n_estimators=300,
                max_depth=5,
                scale_pos_weight=2.19,
                random_state=19
            )
            self.logger.info("Meta learner initialized as XGBClassifier.")
        elif self.meta_learner_type.lower() == 'mlp':
            meta_learner = MLPClassifier(
                hidden_layer_sizes=(50, 25),
                activation='relu',
                solver='adam',
                random_state=19,
                max_iter=200
            )
            self.logger.info("Meta learner initialized as MLPClassifier.")
        else:
            meta_learner = LogisticRegression(
                solver='liblinear',
                class_weight='balanced'
            )
            self.logger.info("Meta learner initialized as standard LogisticRegression.")
        
        return meta_learner

    def train(self, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None, split_validation=True, val_size=0.2) -> Dict:
        """
        Train the ensemble model using a time-based evaluation strategy.
        
        Args:
            X_train: Training features (historical data)
            y_train: Training labels
            X_val: Validation features (most recent data for final evaluation)
            y_val: Validation labels
            X_test: Test features (intermediary data for hyperparameter tuning and threshold optimization)
            y_test: Test labels
            split_validation: Whether to split training data for validation/test if not provided
            val_size: Validation split size if splitting
            
        Returns:
            Dictionary with training results and metrics
        """
        self.logger.info("Starting ensemble model training with time-based evaluation strategy...")

        # Split data if test set not provided (though not recommended for time-based evaluation)
        if split_validation and (X_test is None or y_test is None):
            self.logger.warning("Creating random test split - not ideal for time-based evaluation")
            X_train, X_test, y_train, y_test = train_test_split(
                X_train, y_train, test_size=val_size, random_state=42, stratify=y_train
            )
            self.logger.info(f"Split training data with {val_size*100:.0f}% for test")

        # Check for data leakage between datasets
        if X_val is not None and y_val is not None and X_test is not None and y_test is not None:
            leakage_results = self.detect_data_leakage(X_train, X_test, X_val)
            if any(pct > 1.0 for pct in [leakage_results['train_test_overlap_pct'], 
                                        leakage_results['train_val_overlap_pct'],
                                        leakage_results['test_val_overlap_pct']]):
                self.logger.warning("Data leakage detected! This may affect the validity of your evaluation.")

        # Prepare data by filtering selected features and handling missing values
        X_train = self._prepare_data(X_train)
        X_test = self._prepare_data(X_test)
        X_val = self._prepare_data(X_val)

        # Store data for later use
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_val = X_val
        self.y_val = y_val

        # Apply ADASYN to handle class imbalance on training data
        X_train_resampled, y_train_resampled = self._apply_adasyn_resampling(X_train, y_train)
        self.logger.info(f"Using resampled training data with {len(X_train_resampled)} samples")

        # 1. Base Model Training on X_train with X_test for early stopping
        self.logger.info("Training base models on historical data (X_train) with test data (X_test) for early stopping...")

        # Train XGBoost model with X_test for early stopping
        self.logger.info("Training XGBoost model...")
        self.model_xgb.fit(
            X_train_resampled, y_train_resampled,
            eval_set=[(X_test, y_test)],  # Using X_test for early stopping
            verbose=False
        )

        # Train CatBoost model with X_test for early stopping
        self.logger.info("Training CatBoost model...")
        self.model_cat.fit(
            X_train_resampled, y_train_resampled,
            eval_set=[(X_test, y_test)],  # Using X_test for early stopping
            verbose=False
        )

        # Train LightGBM model with X_test for early stopping
        self.logger.info("Training LightGBM model...")
        self.model_lgb.fit(
            X_train_resampled, y_train_resampled,
            eval_set=[(X_test, y_test)]  # Using X_test for early stopping
        )
        
        # Train the extra model with scaling for non-tree-based models
        self.logger.info(f"Training extra base model ({self.extra_base_model_type})...")
        
        # Apply standardization for non-tree-based models
        if self.extra_base_model_type in ['svm', 'mlp']:
            self.extra_model_scaler = StandardScaler()
            X_train_scaled = self.extra_model_scaler.fit_transform(X_train_resampled)
            X_test_scaled = self.extra_model_scaler.transform(X_test)
            self.model_extra.fit(X_train_scaled, y_train_resampled)
        else:
            self.model_extra.fit(X_train_resampled, y_train_resampled)

        # 2. Calibrate probabilities using X_test if requested
        if self.calibrate:
            self.logger.info(f"Calibrating base model probabilities using {self.calibration_method} method on test data...")
            self._calibrate_models(X_train, y_train, X_test, y_test)

        # 3. Get base model predictions on test set for dynamic weighting and meta-learner
        self.logger.info("Generating base model predictions on test data for meta-learner training...")
        test_pred_proba_xgb = self.model_xgb.predict_proba(X_test)[:, 1]
        test_pred_proba_cat = self.model_cat.predict_proba(X_test)[:, 1]
        
        if hasattr(self, 'calibrated_lgb'):
            test_pred_proba_lgb = self.calibrated_lgb.predict_proba(X_test)[:, 1]
        else:
            test_pred_proba_lgb = self.model_lgb.predict_proba(X_test)[:, 1]
            
        if hasattr(self, 'extra_model_scaler'):
            test_pred_proba_extra = self.model_extra.predict_proba(self.extra_model_scaler.transform(X_test))[:, 1]
        else:
            test_pred_proba_extra = self.model_extra.predict_proba(X_test)[:, 1]

        # 4. Calculate dynamic weights based on test set performance
        if self.dynamic_weighting:
            self.logger.info("Computing dynamic weights based on test set performance...")
            self.dynamic_weights = self._compute_dynamic_weights(
                test_pred_proba_xgb, test_pred_proba_cat, test_pred_proba_lgb, test_pred_proba_extra, y_test
            )
            
            # Log dynamic weights
            self.logger.info(f"Dynamic weights from test data: {self.dynamic_weights}")
            
            # If running in MLflow, log the weights
            if mlflow.active_run() is not None:
                for model, weight in self.dynamic_weights.items():
                    mlflow.log_metric(f"weight_{model}", weight)

        # 5. Create meta-features for stacking from test set
        self.logger.info("Preparing meta-features for meta-learner training using test data...")
        meta_features_test = self._create_meta_features(
            test_pred_proba_xgb, test_pred_proba_cat, test_pred_proba_lgb, test_pred_proba_extra
        )

        # 6. Initialize and train the meta-learner on test set predictions
        self.logger.info(f"Training meta-learner on test data predictions...")
        self.meta_learner = self._initialize_meta_learner()
        self.meta_learner.fit(meta_features_test, y_test)

        # 7. Optimize prediction threshold using test set
        self.logger.info(f"Optimizing threshold on test data to reach target precision of {self.target_precision:.2f}...")
        test_proba = self.predict_proba(X_test)
        self.optimal_threshold = self._tune_threshold_for_precision(test_proba, y_test, 
                                                                target_precision=self.target_precision,
                                                                required_recall=self.required_recall)
        
        self.logger.info(f"Optimal threshold from test data: {self.optimal_threshold:.4f}")

        # 8. Final Evaluation on validation set (most recent data)
        self.logger.info("Performing final evaluation on most recent data (validation set)...")
        final_metrics = self.evaluate(X_val, y_val)
        
        # 9. Calculate feature importance for interpretability
        try:
            xgb_importance = self.model_xgb.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': xgb_importance
            }).sort_values('importance', ascending=False)
            
            # Log top important features
            self.logger.info("Feature importance analysis:", extra={
                'top_features': feature_importance.head(20).to_dict('records')
            })
            
            # Log to MLflow if active
            if mlflow.active_run() is not None:
                for idx, row in feature_importance.head(20).iterrows():
                    mlflow.log_metric(f"feature_importance_{row['feature']}", row['importance'])
                
        except Exception as e:
            self.logger.warning(f"Could not calculate feature importance: {str(e)}")
        
        return final_metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return probability predictions for the positive class.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of probability predictions
        """
        try:
            # Prepare data
            X = self._prepare_data(X)
            
            # Get base model predictions with robust error handling
            def get_model_preds(model, model_name, scaler=None):
                try:
                    # Check if we have a calibrated version
                    cal_model_name = f'calibrated_{model_name}'
                    if hasattr(self, cal_model_name) and getattr(self, cal_model_name) is not None:
                        use_model = getattr(self, cal_model_name)
                    else:
                        use_model = model
                        
                    # Apply scaling if needed
                    if scaler is not None:
                        X_pred = scaler.transform(X)
                        probs = use_model.predict_proba(X_pred)[:, 1]
                    else:
                        probs = use_model.predict_proba(X)[:, 1]
                        
                    # Handle NaN and infinite values
                    probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
                    return probs
                except Exception as e:
                    self.logger.error(f"Error in {model_name} prediction: {str(e)}")
                    return np.zeros(len(X))
            
            # Get predictions from all models
            p_xgb = get_model_preds(self.model_xgb, 'xgb')
            p_cat = get_model_preds(self.model_cat, 'cat')
            p_lgb = get_model_preds(self.model_lgb, 'lgb')
            p_extra = get_model_preds(self.model_extra, 'extra', 
                                        self.extra_model_scaler if hasattr(self, 'extra_model_scaler') else None)
            
            # Create meta-features for meta-learner prediction
            meta_features = self._create_meta_features(p_xgb, p_cat, p_lgb, p_extra)
            
            # Get meta-learner predictions
            try:
                meta_predictions = self.meta_learner.predict_proba(meta_features)[:, 1]
                meta_predictions = np.nan_to_num(meta_predictions, nan=0.5, posinf=1.0, neginf=0.0)
                
                # Option for stability: Test-time averaging with controlled noise
                # Only apply this when explicitly enabled and appropriate
                enable_tta = False  # Set to True to enable test-time augmentation
                
                if enable_tta and len(X) < 1000:  # Only for smaller prediction batches
                    n_iterations = 5
                    tta_predictions = np.zeros(len(X))
                    successful_iterations = 0
                    
                    for i in range(n_iterations):
                        try:
                            # Create deterministic noise based on sample index for reproducibility
                            # This produces consistent noise patterns for the same samples
                            noise_seeds = np.arange(len(X)) + i * len(X)
                            np.random.seed(42 + i)  # Fixed seed for reproducibility
                            
                            # Gentler noise that scales with probability (less noise near 0 and 1)
                            noise_level = 0.005  # Reduced from 0.01
                            # Scale noise to be gentler near 0 and 1
                            noise_scale = np.minimum(meta_features[:, 0], 1-meta_features[:, 0]) * 4
                            noise = np.random.normal(0, noise_level, size=len(X)) * noise_scale
                            
                            # Apply noise with clipping to ensure valid probabilities
                            noisy_meta_features = meta_features.copy()
                            for j in range(noisy_meta_features.shape[1]):
                                noisy_meta_features[:, j] = np.clip(
                                    noisy_meta_features[:, j] + noise, 0.0, 1.0
                                )
                            
                            # Get prediction with noisy features
                            iter_preds = self.meta_learner.predict_proba(noisy_meta_features)[:, 1]
                            iter_preds = np.nan_to_num(iter_preds, nan=0.5, posinf=1.0, neginf=0.0)
                            tta_predictions += iter_preds
                            successful_iterations += 1
                        except Exception as e:
                            self.logger.warning(f"Error in TTA iteration {i}: {str(e)}")
                            continue
                    
                    # Use TTA predictions if any were successful, otherwise fall back to original
                    if successful_iterations > 0:
                        meta_predictions = tta_predictions / successful_iterations
                        self.logger.info(f"Used test-time averaging with {successful_iterations} iterations")
                
            except Exception as e:
                self.logger.error(f"Error in meta-learner prediction: {str(e)}")
                # Fall back to weighted average of base models
                if self.dynamic_weighting:
                    meta_predictions = (
                        self.dynamic_weights['xgb'] * p_xgb +
                        self.dynamic_weights['cat'] * p_cat +
                        self.dynamic_weights['lgb'] * p_lgb +
                        self.dynamic_weights['extra'] * p_extra
                    )
                else:
                    meta_predictions = (p_xgb + p_cat + p_lgb + p_extra) / 4
                
                self.logger.warning("Falling back to weighted average of base models")
            
            # Final safety checks
            meta_predictions = np.nan_to_num(meta_predictions, nan=0.5, posinf=1.0, neginf=0.0)
            meta_predictions = np.clip(meta_predictions, 0.0, 1.0)
            
            return meta_predictions
            
        except Exception as e:
            self.logger.error(f"Critical error in predict_proba: {str(e)}")
            # Return default predictions in case of critical failure
            return np.full(len(X), 0.5)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make binary predictions using the ensemble model.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of binary predictions (0 or 1)
        """
        try:
            # Get probability predictions
            probs = self.predict_proba(X)
            
            # Apply optimal threshold
            if not hasattr(self, 'optimal_threshold') or self.optimal_threshold is None:
                self.logger.warning("Optimal threshold not set, using default threshold of 0.5")
                self.optimal_threshold = 0.5
                
            preds = (probs >= self.optimal_threshold).astype(int)
            
            return preds
            
        except Exception as e:
            self.logger.error(f"Error in predict method: {str(e)}")
            # Return conservative predictions (all 0) in case of failure
            return np.zeros(len(X), dtype=int)

    def evaluate(self, X_val, y_val):
        """
        Evaluate model performance on validation data (most recent data).
        This provides the final assessment of model performance for time-based evaluation.
        
        Args:
            X_val: Validation features (most recent data)
            y_val: Validation target values
            
        Returns:
            Dictionary with performance metrics and confidence intervals
        """
        self.logger.info("Evaluating model performance on validation data (most recent data)...")
        
        # Get predictions using the optimal threshold learned from test data
        y_proba = self.predict_proba(X_val)
        y_pred = (y_proba >= self.optimal_threshold).astype(int)
        
        # Calculate metrics with zero_division=0 to avoid warnings
        with np.errstate(divide='ignore', invalid='ignore'):
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            
            # Handle NaN values that might still occur
            precision = 0.0 if np.isnan(precision) else precision
            recall = 0.0 if np.isnan(recall) else recall
            
        # Calculate AUC with error handling
        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                auc = roc_auc_score(y_val, y_proba)
                if np.isnan(auc):
                    auc = 0.5  # Default to random classifier performance
        except Exception as e:
            self.logger.warning(f"Could not calculate AUC: {str(e)}")
            auc = 0.5
        
        # Generate confidence intervals using bootstrapping
        n_bootstraps = 1000
        max_attempts = n_bootstraps * 2  # Allow extra attempts to get valid samples
        bootstrap_precision = []
        bootstrap_recall = []
        bootstrap_auc = []
        valid_samples = 0
        attempts = 0
        
        self.logger.info(f"Calculating {n_bootstraps} bootstrap samples for confidence intervals...")
        
        while valid_samples < n_bootstraps and attempts < max_attempts:
            attempts += 1
            # Generate bootstrap indices
            indices = np.random.choice(len(y_val), len(y_val), replace=True)
            
            # Check if bootstrap sample contains both classes
            if len(np.unique(y_val.iloc[indices])) < 2:
                continue
                
            # Check if predictions have variation
            sample_preds = (y_proba[indices] >= self.optimal_threshold).astype(int)
            if len(np.unique(sample_preds)) < 2:
                continue
                
            # Sample validation data
            bootstrap_true = y_val.iloc[indices]
            bootstrap_pred = (y_proba[indices] >= self.optimal_threshold).astype(int)
            bootstrap_proba = y_proba[indices]
            
            # Calculate and store metrics
            try:
                with np.errstate(divide='ignore', invalid='ignore'):
                    bs_precision = precision_score(bootstrap_true, bootstrap_pred, zero_division=0)
                    bs_recall = recall_score(bootstrap_true, bootstrap_pred, zero_division=0)
                    bs_auc = roc_auc_score(bootstrap_true, bootstrap_proba)
                    
                    # Skip NaN or invalid values
                    if not (np.isnan(bs_precision) or np.isnan(bs_recall) or 
                            np.isnan(bs_auc)):
                        bootstrap_precision.append(bs_precision)
                        bootstrap_recall.append(bs_recall)
                        bootstrap_auc.append(bs_auc)
                        valid_samples += 1
            except Exception:
                continue
        
        # Calculate confidence intervals if we have enough valid samples
        conf_intervals = {}
        min_valid_samples = 100  # Minimum samples needed for reliable intervals
        
        if valid_samples >= min_valid_samples:
            conf_intervals = {
                'precision_ci': [
                    float(np.percentile(bootstrap_precision, 2.5)),
                    float(np.percentile(bootstrap_precision, 97.5))
                ],
                'recall_ci': [
                    float(np.percentile(bootstrap_recall, 2.5)),
                    float(np.percentile(bootstrap_recall, 97.5))
                ],
                'auc_ci': [
                    float(np.percentile(bootstrap_auc, 2.5)),
                    float(np.percentile(bootstrap_auc, 97.5))
                ]
            }
        else:
            self.logger.warning(f"Insufficient valid bootstrap samples ({valid_samples}/{n_bootstraps}) for confidence intervals")
        
        # Final metrics dictionary
        metrics = {
            'precision': float(precision),
            'recall': float(recall),
            'auc': float(auc),
            'optimal_threshold': float(self.optimal_threshold),
            'support_positives': int(np.sum(y_val)),
            'support_negatives': int(len(y_val) - np.sum(y_val)),
            'valid_bootstrap_samples': valid_samples
        }
        
        # Add confidence intervals if available
        metrics.update(conf_intervals)
        
        # Log results
        self.logger.info("Final validation metrics:", extra=metrics)
        
        # Log to MLflow if active
        if mlflow.active_run() is not None:
            metrics_for_mlflow = {
                'val_precision': precision,
                'val_recall': recall,
                'val_auc': auc,
                'val_positives': int(np.sum(y_val)),
                'val_negatives': int(len(y_val) - np.sum(y_val)),
                'val_bootstrap_samples': valid_samples
            }
            
            # Add confidence intervals if available
            if 'precision_ci' in conf_intervals:
                metrics_for_mlflow.update({
                    'val_precision_lower': conf_intervals['precision_ci'][0],
                    'val_precision_upper': conf_intervals['precision_ci'][1],
                    'val_recall_lower': conf_intervals['recall_ci'][0],
                    'val_recall_upper': conf_intervals['recall_ci'][1],
                    'val_auc_lower': conf_intervals['auc_ci'][0],
                    'val_auc_upper': conf_intervals['auc_ci'][1]
                })
            
            # Log all metrics with validation prefix
            mlflow.log_metrics(metrics_for_mlflow)
        
        return metrics
    
    def _calibrate_models(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """
        Calibrate all base models' probabilities using isotonic regression or Platt scaling.
        
        Args:
            X_test: Test features for calibration
            y_test: Test labels for calibration
        """
        self.logger.info(f"Calibrating model probabilities using {self.calibration_method} method...")
        
        try:
            # Calculate minimum samples needed for reliable calibration
            min_samples_required = 200  # Minimum samples needed for reliable calibration
            if len(X_test) < min_samples_required:
                self.logger.warning(f"Insufficient samples ({len(X_test)}) for reliable calibration. Skipping calibration.")
                return
                
            # Ensure X_test has the right features and handle missing/extra features
            X_test_prepared = self._prepare_data(X_test)
            X_train_prepared = self._prepare_data(X_train)
            # Function to calibrate a single model with robust error handling
            def calibrate_model(model, model_name, scaler=None):
                try:
                    self.logger.info(f"Calibrating {model_name} model...")
                    
                    # Create a frozen estimator to avoid retraining during calibration
                    from sklearn.base import clone
                    frozen_model = clone(model)
                    
                    # Determine appropriate CV strategy based on sample size
                    n_pos_samples = np.sum(y_test)
                    n_neg_samples = len(y_test) - n_pos_samples
                    min_class_samples = min(n_pos_samples, n_neg_samples)
                    min_samples_per_fold = 30  # Minimum samples per class per fold
                    
                    # Calculate max possible folds while ensuring min_samples_per_fold
                    max_folds = max(2, int(min_class_samples / min_samples_per_fold))
                    n_splits = min(5, max_folds)  # Cap at 5 folds
                    
                    if n_splits < 2 or min_class_samples < 2 * min_samples_per_fold:
                        self.logger.warning(f"Not enough samples for cross-validation calibration for {model_name}. Using 'prefit' mode.")
                        cv_strategy = 'prefit'
                        # For prefit mode, we use the existing trained model directly
                        frozen_model = model
                    else:
                        cv_strategy = n_splits
                        self.logger.info(f"Using {n_splits}-fold CV for {model_name} calibration")
                    
                    # Apply scaling if needed
                    X_calibration = self.extra_model_scaler.transform(X_test_prepared) if scaler is not None else X_test_prepared
                    X_train_prepared = self.extra_model_scaler.transform(X_train_prepared) if scaler is not None else X_train_prepared
                    # Create calibrator with appropriate settings
                    calibrated_model = CalibratedClassifierCV(
                        estimator=frozen_model,
                        method=self.calibration_method,
                        cv=cv_strategy,
                        n_jobs=-1,
                        ensemble=True
                    )
                    
                    # Fit calibration model
                    calibrated_model.fit(X_train_prepared, y_train, eval_set=[(X_calibration, y_test)])
                    
                    # Get probabilities before and after calibration
                    if scaler is not None:
                        X_pred = self.extra_model_scaler.transform(X_test_prepared)
                        prob_uncal = model.predict_proba(X_pred)[:, 1]
                        prob_cal = calibrated_model.predict_proba(X_pred)[:, 1]
                    else:
                        prob_uncal = model.predict_proba(X_test_prepared)[:, 1]
                        prob_cal = calibrated_model.predict_proba(X_test_prepared)[:, 1]
                    
                    # Remove NaNs and clip to valid range
                    prob_uncal = np.nan_to_num(prob_uncal, nan=0.0, posinf=1.0, neginf=0.0)
                    prob_cal = np.nan_to_num(prob_cal, nan=0.0, posinf=1.0, neginf=0.0)
                    
                    # Return calibrated model and probabilities
                    return calibrated_model, prob_uncal, prob_cal
                    
                except Exception as e:
                    self.logger.error(f"Error calibrating {model_name} model: {str(e)}")
                    return None, None, None
            
            # Calibrate each model
            models_to_calibrate = [
                (self.model_xgb, 'xgb', None),
                (self.model_cat, 'cat', None),
                (self.model_lgb, 'lgb', None),
                (self.model_extra, 'extra', self.extra_model_scaler if hasattr(self, 'extra_model_scaler') else None)
            ]
            
            calibration_results = {}
            
            for model, model_name, scaler in models_to_calibrate:
                result = calibrate_model(model, model_name, scaler)
                if result[0] is not None:
                    # Store calibrated model, uncalibrated and calibrated probabilities
                    setattr(self, f'calibrated_{model_name}', result[0])
                    calibration_results[model_name] = {
                        'uncalibrated': result[1],
                        'calibrated': result[2]
                    }
            
            # If no models were successfully calibrated, return
            if not calibration_results:
                self.logger.warning("No models were successfully calibrated.")
                return
            
            # Analyze calibration effectiveness with adaptive binning
            self._analyze_calibration(calibration_results, y_test)
            
            self.logger.info("Model calibration completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in model calibration process: {str(e)}")
            self.logger.warning("Using uncalibrated models due to calibration error")
            
    def _analyze_calibration(self, calibration_results, y_test):
        """
        Analyze the effectiveness of calibration with adaptive binning based on data distribution.
        
        Args:
            calibration_results: Dictionary with uncalibrated and calibrated probabilities per model
            y_test: True labels for test data
        """
        try:
            # Determine optimal bin count based on data size
            # Aim for ~30 samples per bin minimum
            min_samples_per_bin = 30
            max_bins = 20  # Cap at 20 bins maximum
            
            # Calculate adaptive bin count
            sample_count = len(y_test)
            bin_count = min(max_bins, max(5, sample_count // min_samples_per_bin))
            self.logger.info(f"Using {bin_count} bins for calibration analysis (adaptive)")
            
            for model_name, probs in calibration_results.items():
                uncal_probs = probs['uncalibrated']
                cal_probs = probs['calibrated']
                
                # Create adaptively-spaced bins based on data distribution
                # This ensures more bins in dense probability regions
                percentile_edges = np.linspace(0, 100, bin_count + 1)
                bin_edges = np.percentile(uncal_probs, percentile_edges)
                # Ensure bin edges are unique and span [0,1]
                bin_edges = np.unique(np.clip(bin_edges, 0, 1))
                actual_bin_count = len(bin_edges) - 1
                
                if actual_bin_count < 2:
                    self.logger.warning(f"Not enough unique probability values for {model_name} to create bins")
                    continue
                
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                # Initialize arrays with NaN to clearly indicate missing data
                uncal_actual = np.full(actual_bin_count, np.nan)
                cal_actual = np.full(actual_bin_count, np.nan)
                uncal_pred = np.full(actual_bin_count, np.nan)
                cal_pred = np.full(actual_bin_count, np.nan)
                bin_counts = np.zeros(actual_bin_count, dtype=int)
                
                # Analyze each bin
                for i in range(actual_bin_count):
                    # Get samples in this bin using half-open intervals [a, b)
                    if i == actual_bin_count - 1:  # Last bin includes the right edge
                        mask_uncal = (uncal_probs >= bin_edges[i]) & (uncal_probs <= bin_edges[i+1])
                        mask_cal = (cal_probs >= bin_edges[i]) & (cal_probs <= bin_edges[i+1])
                    else:
                        mask_uncal = (uncal_probs >= bin_edges[i]) & (uncal_probs < bin_edges[i+1])
                        mask_cal = (cal_probs >= bin_edges[i]) & (cal_probs < bin_edges[i+1])
                    
                    # For uncalibrated probabilities
                    count_uncal = np.sum(mask_uncal)
                    bin_counts[i] = count_uncal
                    
                    # Only calculate metrics if there are samples in this bin
                    if count_uncal >= min(5, len(y_test) // 20):  # At least 5 samples or 5% of data
                        uncal_actual[i] = np.mean(y_test[mask_uncal])
                        uncal_pred[i] = np.mean(uncal_probs[mask_uncal])
                    
                    # For calibrated probabilities (separate calculation)
                    count_cal = np.sum(mask_cal)
                    if count_cal >= min(5, len(y_test) // 20):
                        cal_actual[i] = np.mean(y_test[mask_cal])
                        cal_pred[i] = np.mean(cal_probs[mask_cal])
                
                # Calculate calibration error metrics without NaN values
                valid_uncal = ~np.isnan(uncal_actual) & ~np.isnan(uncal_pred)
                valid_cal = ~np.isnan(cal_actual) & ~np.isnan(cal_pred)
                
                # Expected Calibration Error (ECE)
                if np.any(valid_uncal) and np.any(valid_cal):
                    ece_uncal = np.average(
                        np.abs(uncal_actual[valid_uncal] - uncal_pred[valid_uncal]),
                        weights=bin_counts[valid_uncal]
                    )
                    ece_cal = np.average(
                        np.abs(cal_actual[valid_cal] - cal_pred[valid_cal]),
                        weights=bin_counts[valid_cal]
                    )
                    
                    self.logger.info(f"Calibration metrics for {model_name} model:")
                    self.logger.info(f"  ECE before calibration: {ece_uncal:.4f}")
                    self.logger.info(f"  ECE after calibration: {ece_cal:.4f}")
                    
                    # Log detailed calibration analysis
                    calibration_analysis = {
                        'bin_edges': bin_edges.tolist(),
                        'bin_centers': bin_centers.tolist(),
                        'bin_counts': bin_counts.tolist(),
                        'uncalibrated': {
                            'predicted': uncal_pred[valid_uncal].tolist(),
                            'actual': uncal_actual[valid_uncal].tolist(),
                            'ece': float(ece_uncal)
                        },
                        'calibrated': {
                            'predicted': cal_pred[valid_cal].tolist(),
                            'actual': cal_actual[valid_cal].tolist(),
                            'ece': float(ece_cal)
                        }
                    }
                    
                    self.logger.info(f"{model_name} calibration analysis:", extra={
                        'calibration_analysis': calibration_analysis
                    })
                    
                
        except Exception as e:
            self.logger.error(f"Error in calibration analysis: {str(e)}")

    def _create_meta_features(self, p_xgb: np.ndarray, p_cat: np.ndarray, 
                                p_lgb: np.ndarray, p_extra: np.ndarray) -> np.ndarray:
        """
        Create meta-features for the meta-learner by combining base model predictions.
        
        Args:
            p_xgb: XGBoost predicted probabilities
            p_cat: CatBoost predicted probabilities
            p_lgb: LightGBM predicted probabilities
            p_extra: Extra model predicted probabilities
            
        Returns:
            Meta-features for the meta-learner
        """
        # Handle potential NaN values in base model predictions
        p_xgb = np.nan_to_num(p_xgb, nan=0.0, posinf=1.0, neginf=0.0)
        p_cat = np.nan_to_num(p_cat, nan=0.0, posinf=1.0, neginf=0.0)
        p_lgb = np.nan_to_num(p_lgb, nan=0.0, posinf=1.0, neginf=0.0)
        p_extra = np.nan_to_num(p_extra, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Stack the base model probabilities
        base_probs = np.column_stack((p_xgb, p_cat, p_lgb, p_extra))
        
        # Calculate mean and variance
        mean_probs = np.mean(base_probs, axis=1, keepdims=True)
        var_probs = np.var(base_probs, axis=1, keepdims=True)
        
        # Calculate min, max, and range
        min_probs = np.min(base_probs, axis=1, keepdims=True)
        max_probs = np.max(base_probs, axis=1, keepdims=True)
        range_probs = max_probs - min_probs
        
        # Create soft voting result if dynamic weighting is enabled
        if self.dynamic_weighting:
            weighted_avg = (self.dynamic_weights['xgb'] * p_xgb +
                            self.dynamic_weights['cat'] * p_cat +
                            self.dynamic_weights['lgb'] * p_lgb +
                            self.dynamic_weights['extra'] * p_extra).reshape(-1, 1)
        else:
            weighted_avg = mean_probs
        
        # Combine all meta-features
        meta_features = np.hstack((
            base_probs,  # Base model probabilities
            mean_probs,  # Mean probability
            var_probs,   # Variance
            min_probs,   # Minimum
            max_probs,   # Maximum
            range_probs, # Range
            weighted_avg # Weighted average
        ))
        
        # Final safety check for any remaining NaN values
        meta_features = np.nan_to_num(meta_features, nan=0.0, posinf=1.0, neginf=0.0)
        
        return meta_features

    def _tune_threshold_for_precision(self, y_prob: np.ndarray, y_true: pd.Series, 
                                        target_precision: float = 0.60, 
                                        required_recall: float = 0.40,
                                        min_threshold: float = 0.1,
                                        max_threshold: float = 0.9,
                                        step: float = 0.01) -> float:
        """
        Tune the threshold to achieve a target precision with a minimum recall requirement.
        
        Args:
            y_prob: Predicted probabilities
            y_true: True labels
            target_precision: Desired precision level
            required_recall: Minimum required recall
            min_threshold: Minimum threshold to test
            max_threshold: Maximum threshold to test
            step: Step size for threshold search
            
        Returns:
            Optimal threshold
        """
        best_threshold = 0.5  # Default
        best_precision = 0
        best_recall = 0
        
        # Store results for analysis
        threshold_results = []
        
        # Test different thresholds
        for threshold in np.arange(min_threshold, max_threshold + step, step):
            y_pred = (y_prob >= threshold).astype(int)
            
            if sum(y_pred) == 0:  # No positive predictions
                continue
                
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            # Store results
            threshold_results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
            })
            
            # Check if precision meets target and recall meets requirement
            if recall >= required_recall:
                if precision > best_precision:
                    best_precision = precision
                    best_threshold = threshold
                    best_recall = recall
        
        # If no threshold meets both criteria, try to find a balance
        if best_threshold == 0.5 and best_precision == 0:
            self.logger.warning(f"Could not find threshold with precision >= {target_precision} and recall >= {required_recall}")
            
            # Find the threshold that maximizes precision
            best_result = max(threshold_results, key=lambda x: x['precision'])
            best_threshold = best_result['threshold']
            best_precision = best_result['precision']
            best_recall = best_result['recall']
            
            self.logger.info(f"Selected threshold {best_threshold:.4f} with precision {best_precision:.4f}, recall {best_recall:.4f}")
        
        # Log threshold tuning results
        self.logger.info("Threshold tuning results:", extra={
            'threshold_search_results': threshold_results,
            'best_threshold': best_threshold,
            'best_precision': best_precision,
            'best_recall': best_recall,
            'target_precision': target_precision,
            'required_recall': required_recall
        })
        
        # Log to MLflow if active
        if mlflow.active_run() is not None:
            mlflow.log_metrics({
                'threshold_tuning_precision': best_precision,
                'threshold_tuning_recall': best_recall,
            })
        
        return best_threshold

    def _apply_adasyn_resampling(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply ADASYN (Adaptive Synthetic Sampling) to handle class imbalance.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Resampled training data (features, labels)
        """
        self.logger.info(f"Applying ADASYN resampling with sampling_strategy={self.sampling_strategy}...")
        
        try:
            # Initialize ADASYN with sampling strategy
            adasyn = ADASYN(sampling_strategy=self.sampling_strategy, 
                            random_state=19, 
                            n_neighbors=5)
            
            # Apply resampling
            X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
            
            # Convert back to pandas DataFrame/Series
            X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
            y_resampled = pd.Series(y_resampled, name=y_train.name if hasattr(y_train, 'name') else 'target')
            
            # Log resampling results
            original_class_counts = np.bincount(y_train)
            resampled_class_counts = np.bincount(y_resampled)
            
            self.logger.info(f"Original class distribution: {original_class_counts}")
            self.logger.info(f"Resampled class distribution: {resampled_class_counts}")
            
            # Calculate imbalance ratios
            if original_class_counts[0] > 0 and original_class_counts[1] > 0:
                original_ratio = original_class_counts[0] / original_class_counts[1]
                resampled_ratio = resampled_class_counts[0] / resampled_class_counts[1]
                self.logger.info(f"Original imbalance ratio: {original_ratio:.2f}, Resampled ratio: {resampled_ratio:.2f}")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            self.logger.error(f"Error in ADASYN resampling: {str(e)}")
            self.logger.warning("Falling back to original data due to ADASYN error")
            return X_train, y_train

    def select_features_by_importance(self, X: pd.DataFrame, y: pd.Series, 
                                    importance_threshold: float = 0.01,
                                    n_iterations: int = 5) -> List[str]:
        """
        Select features based on importance scores from an XGBoost model.
        
        Args:
            X: Features DataFrame
            y: Target labels
            importance_threshold: Minimum importance score to keep a feature
            n_iterations: Number of iterations to run for stability
            
        Returns:
            List of selected feature names
        """
        self.logger.info(f"Selecting features with importance threshold {importance_threshold}...")
        
        # Initialize feature importance accumulator
        feature_importance_sum = np.zeros(X.shape[1])
        
        # Run multiple iterations for stability
        for i in range(n_iterations):
            # Train a simple XGBoost model
            model = XGBClassifier(
                tree_method='hist',
                device='cpu',
                n_jobs=-1,
                objective='binary:logistic',
                learning_rate=0.1,
                n_estimators=100,
                max_depth=4,
                random_state=19 + i
            )
            
            # Fit the model
            model.fit(X, y)
            
            # Accumulate feature importance scores
            feature_importance_sum += model.feature_importances_
        
        # Average the importance scores
        feature_importance = feature_importance_sum / n_iterations
        
        # Create DataFrame with feature names and importance scores
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        # Select features above threshold
        selected_features = importance_df[importance_df['importance'] >= importance_threshold]['feature'].tolist()
        
        # Log feature importance information
        self.logger.info("Feature importance scores:", extra={
            'top_20_features': importance_df.head(20).to_dict('records'),
            'importance_threshold': importance_threshold,
            'n_selected_features': len(selected_features),
            'selected_features': selected_features
        })
        
        # Log to MLflow if active
        if mlflow.active_run() is not None:
            mlflow.log_params({
                'importance_threshold': importance_threshold,
                'n_selected_features': len(selected_features)
            })
            
            # Log top features and their importance scores
            for idx, row in importance_df.head(20).iterrows():
                mlflow.log_metric(f"feature_importance_{row['feature']}", row['importance'])
        
        self.logger.info(f"Selected {len(selected_features)} features with importance >= {importance_threshold}")
        return selected_features

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Dict:
        """
        Perform stratified k-fold cross-validation.
        
        Args:
            X: Features DataFrame
            y: Target labels
            n_splits: Number of folds
            
        Returns:
            Dictionary with cross-validation results
        """
        self.logger.info(f"Performing {n_splits}-fold cross-validation...")
        
        # Initialize metrics storage
        metrics = {
            'precision': [],
            'recall': [],
            'auc': []
        }
        
        # Initialize k-fold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Perform cross-validation
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            self.logger.info(f"Training fold {fold}/{n_splits}")
            
            # Split data
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            
            # Train model
            self.train(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
            
            # Evaluate
            fold_metrics = self.evaluate(X_val_fold, y_val_fold)
            
            # Store metrics
            metrics['precision'].append(fold_metrics['precision'])
            metrics['recall'].append(fold_metrics['recall'])
            if fold_metrics['auc'] is not None:
                metrics['auc'].append(fold_metrics['auc'])
        
        # Calculate average metrics
        avg_metrics = {
            'precision': np.mean(metrics['precision']),
            'recall': np.mean(metrics['recall']),
            'auc': np.mean(metrics['auc']) if metrics['auc'] else None
        }
        
        # Calculate standard deviations
        std_metrics = {
            'precision_std': np.std(metrics['precision']),
            'recall_std': np.std(metrics['recall']),
            'auc_std': np.std(metrics['auc']) if metrics['auc'] else None
        }
        
        # Log results
        self.logger.info("Cross-validation results:", extra={
            'avg_metrics': avg_metrics,
            'std_metrics': std_metrics
        })
        
        return {
            'fold_metrics': metrics,
            'avg_metrics': avg_metrics,
            'std_metrics': std_metrics
        }

if __name__ == "__main__":
    # Set up project root for imports
    try:
        # Set environment variables and configurations
        os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
        
        logger.info("Starting ensemble model execution with time-based evaluation strategy...")
        
        # Import local modules after path setup
        from models.StackedEnsemble.shared.data_loader import DataLoader
        
        try:
            # Step 1: Load data with time-based splits
            # X_train, y_train = historical data (oldest)
            # X_test, y_test = intermediate data (for tuning)
            # X_val, y_val = most recent data (for final evaluation)
            logger.info("Loading data with time-based splits...")
            X_train, y_train, X_test, y_test, X_val, y_val = DataLoader().load_data()
            
            # Log dataset sizes
            logger.info(f"Dataset sizes - Training: {X_train.shape}, Test: {X_test.shape}, Validation: {X_val.shape}")
            
            # Step 2: Feature selection on training data only
            logger.info("Selecting features using training data only...")
            selected_features = import_selected_features_ensemble('all')
            
            # Step 3: Filter features for all datasets
            X_train_filtered = X_train[selected_features]
            X_test_filtered = X_test[selected_features]
            X_val_filtered = X_val[selected_features]
            
            # Create ensemble with time-based evaluation strategy configuration
            ensemble_model = EnsembleModel(
                extra_base_model_type='mlp',
                meta_learner_type='logistic',
                calibrate=True,
                dynamic_weighting=True,
                target_precision=0.60,  # Target precision
                required_recall=0.40    # Minimum recall requirement
            )
            
            # Step 5: Train the model using time-based evaluation strategy
            # - Train base models on X_train
            # - Use X_test for hyperparameter tuning and threshold optimization
            # - Final evaluation on X_val (most recent data)
            logger.info("Training ensemble model with time-based evaluation strategy...")
            training_results = ensemble_model.train(
                X_train=X_train_filtered, 
                y_train=y_train, 
                X_test=X_test_filtered, 
                y_test=y_test,
                X_val=X_val_filtered, 
                y_val=y_val,
                split_validation=False  # Don't split again, we already have time-based splits
            )
            
            # Step 6: Analyze prediction errors on validation set (most recent data)
            logger.info("Analyzing prediction errors on validation set (most recent data)...")
            error_analysis = ensemble_model.analyze_prediction_errors(X_val_filtered, y_val)
            
            # Step 7: Explain model predictions on validation set
            logger.info("Explaining model predictions on validation set...")
            explanation = ensemble_model.explain_predictions(X_val_filtered)
            
            # Step 8: Generate final validation metrics
            logger.info("Final metrics on validation set (most recent data):")
            for metric, value in training_results.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {metric}: {value:.4f}")
                    
            logger.info("Ensemble model execution completed successfully.")
            
        except Exception as e:
            logger.error(f"Error in ensemble model execution: {str(e)}")
            raise
            
    except Exception as e:
        print(f"Critical error: {str(e)}")
        raise