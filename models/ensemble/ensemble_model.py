"""
EnsembleModel Class

Core implementation of the ensemble model with initialization methods
and high-level APIs.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Dict, List, Tuple, Optional, Union
import os
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks

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
    sys.path.append(os.getcwd().parent)
    print(f"Current directory ensemble_model: {os.getcwd().parent}")

# Local imports
from utils.logger import ExperimentLogger
from utils.create_evaluation_set import import_selected_features_ensemble

# Module imports
from models.ensemble.calibration import calibrate_models, analyze_calibration
from models.ensemble.data_utils import prepare_data
from models.ensemble.meta_features import create_meta_features, create_meta_dataframe
from models.ensemble.diagnostics import detect_data_leakage, explain_predictions, analyze_prediction_errors
from models.ensemble.evaluation import evaluate_model, cross_validate
from models.ensemble.training import initialize_meta_learner, train_base_models, hypertune_meta_learner, train_meta_learner
from models.ensemble.weights import compute_dynamic_weights, compute_precision_focused_weights
from models.ensemble.thresholds import tune_threshold_for_precision, tune_threshold, tune_individual_threshold

# Set random seeds for reproducibility
random_seed = 19
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
os.environ['PYTHONHASHSEED'] = str(random_seed)

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
    def __init__(self, logger: ExperimentLogger = None, calibrate: bool = False, 
                calibration_method: str = "sigmoid", individual_thresholding: bool = False,
                meta_learner_type: str = 'xgb', dynamic_weighting: bool = True,
                extra_base_model_type: str = 'mlp', 
                sampling_strategy: float = 0.7,
                complexity_penalty: float = 0.01,
                target_precision: float = 0.50,
                required_recall: float = 0.25,
                X_train: pd.DataFrame = None):
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
        self.logger = logger or ExperimentLogger(experiment_name="ensemble_model_improved",
                                                log_dir="./logs/ensemble_model_improved")
        # Load selected features (assumed common to all models)
        self.selected_features = import_selected_features_ensemble('all')
        self.required_recall = required_recall
        self.sampling_strategy = sampling_strategy  # For ADASYN resampling
        self.complexity_penalty = complexity_penalty  # For regularization
        self.target_precision = target_precision  # For threshold tuning
        
        # Define base models with CPU-only settings with reduced complexity to avoid overfitting:
        self.model_xgb = XGBClassifier( #41.2%
            tree_method='hist',  # Required for CPU-only training per project rules
            device='cpu',
            nthread=-1,
            objective='binary:logistic',
            eval_metric=['aucpr', 'logloss', 'auc'],
            verbosity=0,
            learning_rate=0.03791741758266303,
            max_depth=6,
            min_child_weight=221,
            subsample=0.6321181883147767,
            colsample_bytree=0.8665027477545583,
            reg_alpha=0.009258121536305803,
            reg_lambda=10.625988960745355,
            gamma=1.3607609856733809,
            early_stopping_rounds=634,
            scale_pos_weight=3.8790295965457644,
            seed=19
        )
        self.model_cat = CatBoostClassifier( #39.7%
            learning_rate=0.05747334517009464,
            depth=6,
            min_data_in_leaf=26,
            subsample=0.7152201908359026,
            colsample_bylevel=0.35094802393061786,
            reg_lambda=1.0342319632749895,
            leaf_estimation_iterations=4,
            bagging_temperature=2.891201300013366,
            scale_pos_weight=5.295897250279237,
            early_stopping_rounds=201,
            loss_function='Logloss',
            eval_metric='AUC',
            task_type='CPU',
            thread_count=-1,
            verbose=-1
        )
        self.model_lgb = LGBMClassifier( #41.7%
            objective='binary',
            metric=['binary_logloss', 'average_precision', 'auc'],
            verbose=-1,
            n_jobs=-1,
            random_state=19,
            device='cpu',
            learning_rate=0.06999999999999999,
            num_leaves=92,
            max_depth=9,
            min_child_samples=305,
            feature_fraction=0.6,
            bagging_fraction=0.56,
            bagging_freq=8,
            reg_alpha=4.9,
            reg_lambda=1.3000000000000003,
            min_split_gain=0.16,
            early_stopping_rounds=368,
            path_smooth=0.47000000000000003,
            cat_smooth=14.0,
            max_bin=640
        )
        
        # Initialize the extra base model based on the selected type with reduced complexity
        self.extra_base_model_type = extra_base_model_type.lower()
        if self.extra_base_model_type == 'random_forest':
            self.model_extra = RandomForestClassifier(
                n_estimators=100,
                max_depth=24,
                min_samples_split=11,
                min_samples_leaf=5,
                max_features=None,
                bootstrap=True,
                class_weight=None,
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
            self.model_extra = keras.Sequential()
            self.model_extra.add(layers.InputLayer(shape=(X_train.shape[1],)))
            
            # Add hidden layers
            for _ in range(1):  # 1 hidden layer
                self.model_extra.add(layers.Dense(
                    120,  # 120 neurons per layer
                    activation='tanh',
                    kernel_regularizer=regularizers.l1_l2(
                        l1=0.0006252292488020048, 
                        l2=0.0010179804312458536
                    )
                ))
                self.model_extra.add(layers.BatchNormalization())
                self.model_extra.add(layers.Dropout(0.2685783444324335))  # Updated dropout rate
            
            # Output layer for binary classification
            self.model_extra.add(layers.Dense(1, activation='sigmoid'))
            optimizer = keras.optimizers.Adam(
                learning_rate=0.0008586241362721754,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-8
            )
            self.model_extra.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy', keras.metrics.AUC(name='auc')]
            )
            self.logger.info("Extra base model initialized as MLPClassifier.")
        elif self.extra_base_model_type == 'sgd':
            self.model_extra = SGDClassifier(
                loss='log_loss',
                penalty='elasticnet',
                alpha=0.0001,
                l1_ratio=0.5,
                class_weight=None,
                max_iter=1000,
                early_stopping=True,
                random_state=42
            )
            self.logger.info("Extra base model initialized as SGDClassifier.")
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
        
        # Placeholders for fitted models and scalers
        self.meta_learner = None
        self.model_xgb_calibrated = None
        self.model_cat_calibrated = None
        self.model_lgb_calibrated = None
        self.model_extra_calibrated = None
        self.extra_model_scaler = None

    def train(self, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None, 
                split_validation=True, val_size=0.2) -> Dict:
        """
        Train the ensemble model including base models and meta-learner.
        
        Args:
            X_train: Training features
            y_train: Training target values
            X_val: Optional validation features 
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
        X_train_prepared = prepare_data(X_train, self.selected_features)
        
        # Check for data leakage if validation data is provided
        if X_val is not None and X_test is not None:
            X_val_prepared = prepare_data(X_val, self.selected_features)
            X_test_prepared = prepare_data(X_test, self.selected_features)
            
            leakage_results = detect_data_leakage(
                X_train_prepared, X_test_prepared, X_val_prepared, self.logger
            )
            
            if leakage_results['overlap_percentage'] > 5.0:
                self.logger.warning(f"Significant data leakage detected: {leakage_results['overlap_percentage']:.2f}%")
        
        # If validation data is not provided, split the training data
        if split_validation or X_val is None or y_val is None:
            self.logger.info("Splitting training data for validation...")
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train_prepared, y_train, test_size=val_size, random_state=42, stratify=y_train
            )
            X_train_prepared, y_train = X_train_split, y_train_split
            X_val_prepared, y_val = X_val_split, y_val_split
        else:
            X_val_prepared = prepare_data(X_val, self.selected_features)
        
        # Split training data further if no test data is provided
        if X_test is None or y_test is None:
            self.logger.info("Creating test set from training data...")
            X_train_split, X_test_prepared, y_train_split, y_test = train_test_split(
                X_train_prepared, y_train, test_size=val_size, random_state=43, stratify=y_train
            )
            X_train_prepared, y_train = X_train_split, y_train_split
        else:
            X_test_prepared = prepare_data(X_test, self.selected_features)
        
        # Step 2: Skip class imbalance handling - using original data
        self.logger.info("Skipping ADASYN resampling, using original training data...")
        X_train_resampled, y_train_resampled = X_train_prepared, y_train
        
        # Step 3: Initialize base models dictionary
        base_models = {
            'xgb': self.model_xgb,
            'cat': self.model_cat,
            'lgb': self.model_lgb,
            'extra': self.model_extra
        }
        
        # For MLP or SVM, we need scaling
        if self.extra_base_model_type in ['mlp', 'svm']:
            self.logger.info(f"Applying StandardScaler for {self.extra_base_model_type} model...")
            self.extra_model_scaler = StandardScaler().fit(X_train_resampled)
            base_models['extra_scaler'] = self.extra_model_scaler
        
        # Step 4: Train base models
        self.logger.info("Training base models...")
        trained_models = train_base_models(
            base_models, X_train_resampled, y_train_resampled, X_test_prepared, y_test
        )
        
        # Update model references
        self.model_xgb = trained_models['xgb']
        self.model_cat = trained_models['cat']
        self.model_lgb = trained_models['lgb']
        self.model_extra = trained_models['extra']
        
        if 'extra_scaler' in trained_models:
            self.extra_model_scaler = trained_models['extra_scaler']
        
        # Step 5: Optionally calibrate models
        if self.calibrate:
            self.logger.info(f"Calibrating base models using {self.calibration_method} method...")
            calibration_results = calibrate_models(
                trained_models, X_train_resampled, y_train_resampled, 
                X_test_prepared, y_test, self.calibration_method, self.logger
            )
            
            calibrated_models = calibration_results['calibrated_models']
            
            # Store calibrated models
            self.model_xgb_calibrated = calibrated_models['xgb']
            self.model_cat_calibrated = calibrated_models['cat']
            self.model_lgb_calibrated = calibrated_models['lgb']
            self.model_extra_calibrated = calibrated_models['extra']
            
            # Analyze calibration effectiveness
            cal_analysis = analyze_calibration(
                calibration_results['calibration_results'], y_test, self.logger
            )
        
        # Step 6: Get base model predictions on validation data
        self.logger.info("Generating base model predictions on validation data...")
        
        # Use calibrated models if available
        xgb_model = self.model_xgb_calibrated if self.calibrate else self.model_xgb
        cat_model = self.model_cat_calibrated if self.calibrate else self.model_cat
        lgb_model = self.model_lgb_calibrated if self.calibrate else self.model_lgb
        extra_model = self.model_extra_calibrated if self.calibrate else self.model_extra
        
        # Get predictions
        if self.extra_base_model_type in ['mlp', 'svm'] and self.extra_model_scaler is not None:
            # Scale validation, training, and test data using the fitted scaler
            X_val_scaled = self.extra_model_scaler.transform(X_val_prepared)
            X_train_scaled = self.extra_model_scaler.transform(X_train_prepared)
            X_test_scaled = self.extra_model_scaler.transform(X_test_prepared)
            
            # Get probabilities from scaled data
            if self.extra_base_model_type == 'mlp':
                p_extra = extra_model.predict(X_val_scaled, verbose=0).flatten()
                p_extra_train = extra_model.predict(X_train_scaled, verbose=0).flatten()
                p_extra_test = extra_model.predict(X_test_scaled, verbose=0).flatten()
            else:  # SVM case
                p_extra = extra_model.predict_proba(X_val_scaled)[:, 1]
                p_extra_train = extra_model.predict_proba(X_train_scaled)[:, 1]
                p_extra_test = extra_model.predict_proba(X_test_scaled)[:, 1]
        else:
            p_extra = extra_model.predict_proba(X_val_prepared)[:, 1]
            p_extra_train = extra_model.predict_proba(X_train_prepared)[:, 1]
            p_extra_test = extra_model.predict_proba(X_test_prepared)[:, 1]
        
        p_xgb = xgb_model.predict_proba(X_val_prepared)[:, 1]
        p_xgb_train = xgb_model.predict_proba(X_train_prepared)[:, 1]
        p_xgb_test = xgb_model.predict_proba(X_test_prepared)[:, 1]
        
        p_cat = cat_model.predict_proba(X_val_prepared)[:, 1]
        p_cat_train = cat_model.predict_proba(X_train_prepared)[:, 1]
        p_cat_test = cat_model.predict_proba(X_test_prepared)[:, 1]
        
        p_lgb = lgb_model.predict_proba(X_val_prepared)[:, 1]
        p_lgb_train = lgb_model.predict_proba(X_train_prepared)[:, 1]
        p_lgb_test = lgb_model.predict_proba(X_test_prepared)[:, 1]
        
        # Step 7: Optionally calculate dynamic weights based on validation performance
        if self.dynamic_weighting:
            self.logger.info("Computing dynamic weights based on validation performance...")
            self.dynamic_weights = compute_precision_focused_weights(
                p_xgb, p_cat, p_lgb, p_extra, y_val, self.target_precision, self.required_recall, self.logger
            )
            self.dynamic_weights_train = compute_precision_focused_weights(
                p_xgb_train, p_cat_train, p_lgb_train, p_extra_train, y_train, self.target_precision, self.required_recall, self.logger
            )
            self.dynamic_weights_test = compute_precision_focused_weights(
                p_xgb_test, p_cat_test, p_lgb_test, p_extra_test, y_test, self.target_precision, self.required_recall, self.logger
            )
        
        # Step 8: Create meta-features from base model predictions
        self.logger.info("Creating meta-features for meta-learner...")
        meta_features = create_meta_features(
            p_xgb, p_cat, p_lgb, p_extra, self.dynamic_weights if self.dynamic_weighting else None
        )
        meta_features_train = create_meta_features(
            p_xgb_train, p_cat_train, p_lgb_train, p_extra_train, self.dynamic_weights_train if self.dynamic_weighting else None
        )
        meta_features_test = create_meta_features(
            p_xgb_test, p_cat_test, p_lgb_test, p_extra_test, self.dynamic_weights_test if self.dynamic_weighting else None
        )
        
        # Convert to DataFrame for better interpretability
        meta_df = create_meta_dataframe(meta_features)
        meta_df_train = create_meta_dataframe(meta_features_train)
        meta_df_test = create_meta_dataframe(meta_features_test)
        
        # Step 9: Initialize and train meta-learner
        self.logger.info(f"Initializing meta-learner of type {self.meta_learner_type}...")
        self.meta_learner = initialize_meta_learner(self.meta_learner_type)
        
        # Train meta-learner
        self.logger.info("Training meta-learner...")
        # self.meta_learner = train_meta_learner(self.meta_learner, meta_df, y_val)
        self.meta_learner = hypertune_meta_learner(meta_df_train, y_train, meta_df, y_val, 
                                                    meta_learner_type=self.meta_learner_type, target_precision=self.target_precision, min_recall=self.required_recall)
        # Step 10: Tune threshold for optimal precision-recall trade-off
        self.logger.info(f"Tuning threshold for target precision {self.target_precision}...")
        
        # Get meta-learner predictions on validation data
        meta_val_probs = self.meta_learner.predict_proba(meta_df)[:, 1]
        
        # Tune threshold
        best_threshold, threshold_metrics = tune_threshold_for_precision(
            meta_val_probs, y_val, 
            target_precision=self.target_precision,
            required_recall=self.required_recall,
            logger=self.logger
        )
        
        self.optimal_threshold = best_threshold
        threshold_metrics = threshold_metrics
        self.logger.info(f"Optimal threshold set to {self.optimal_threshold:.4f}")
        
        # Step 11: Final evaluation on validation data
        self.logger.info("Performing final evaluation on validation data...")
        eval_results = evaluate_model(
            self.meta_learner, meta_df, y_val, self.optimal_threshold, self.logger
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
        X_prepared = prepare_data(X, self.selected_features)
        
        # Generate predictions from base models
        # Use calibrated models if available
        xgb_model = self.model_xgb_calibrated if self.calibrate else self.model_xgb
        cat_model = self.model_cat_calibrated if self.calibrate else self.model_cat
        lgb_model = self.model_lgb_calibrated if self.calibrate else self.model_lgb
        extra_model = self.model_extra_calibrated if self.calibrate else self.model_extra
        
        # Get predictions
        if self.extra_base_model_type in ['mlp', 'svm'] and self.extra_model_scaler is not None:
            X_scaled = self.extra_model_scaler.transform(X_prepared)
            if self.extra_base_model_type == 'mlp':
                p_extra = extra_model.predict(X_scaled, verbose=0).flatten()
            else:  # SVM case
                p_extra = extra_model.predict_proba(X_scaled)[:, 1]
        else:
            p_extra = extra_model.predict_proba(X_prepared)[:, 1]
        
        p_xgb = xgb_model.predict_proba(X_prepared)[:, 1]
        p_cat = cat_model.predict_proba(X_prepared)[:, 1]
        p_lgb = lgb_model.predict_proba(X_prepared)[:, 1]
        
        # Create meta-features
        meta_features = create_meta_features(
            p_xgb, p_cat, p_lgb, p_extra, 
            self.dynamic_weights if self.dynamic_weighting else None
        )
        
        # Get meta-learner predictions
        meta_probs = self.meta_learner.predict_proba(meta_features)
        
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
    
    def explain_predictions(self, X_val) -> Dict:
        """
        Generate feature importance explanations using SHAP values.
        
        Args:
            X_val: Validation features
            
        Returns:
            Dictionary with explanation results
        """
        return explain_predictions(self, X_val, self.logger)
    
    def analyze_prediction_errors(self, X_val, y_val) -> Dict:
        """
        Analyze prediction errors on the validation set.
        
        Args:
            X_val: Validation features
            y_val: Validation target values
            
        Returns:
            Dictionary with error analysis results
        """
        return analyze_prediction_errors(self, X_val, y_val, self.optimal_threshold, self.logger)

    def precision_filter(self, X, probabilities):
        """
        Apply additional filtering to boost precision
        """
        # Get high-confidence predictions
        high_conf = probabilities > self.optimal_threshold
        
        # Get original features for these instances
        X_high_conf = X[high_conf]
        
        # Apply rule-based filters (examples)
        if 'home_form' in X_high_conf.columns and 'away_form' in X_high_conf.columns:
            form_diff = abs(X_high_conf['home_form'] - X_high_conf['away_form'])
            # Filter out likely non-draws (big form differences)
            likely_not_draw = form_diff > 0.5  
            high_conf[high_conf] = ~likely_not_draw
        
        # Additional filters based on domain knowledge
        
        return high_conf
