"""
EnsembleModel Module

This module implements an ensemble model that boosts precision and recall 
using two ensemble strategies:
  1. Soft Voting with weighted probability averaging.
  2. Stacking with a meta-learner (default = Logistic Regression).

Optional calibration is provided to adjust overconfident predictions.
The ensemble uses three base models:
  - XGBoost (XGBClassifier)
  - CatBoost (CatBoostClassifier)
  - LightGBM (LGBMClassifier)
  
For imbalanced datasets, focal loss parameters can be passed to LightGBM.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from pathlib import Path
import os
import sys
from imblearn.over_sampling import ADASYN
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import make_scorer

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
experiment_name = "stacked_ensemble_model"
logger = ExperimentLogger(experiment_name=experiment_name, log_dir='./logs/stacked_ensemble_model')


from utils.create_evaluation_set import create_ensemble_evaluation_set, import_selected_features_ensemble, import_training_data_ensemble, setup_mlflow_tracking
mlruns_dir = setup_mlflow_tracking(experiment_name)

def load_pretrained_model(run_id: str, model_type: str = "xgboost"):
    """Load the pretrained model from MLflow using run ID and model type.

    Supported model types: 'xgboost', 'catboost', and 'lightgbm'.
    """
    try:
        
        if model_type.lower() == "xgboost":
            model_uri = f"runs:/{run_id}/xgboost_api_model"
            return mlflow.xgboost.load_model(model_uri)
        elif model_type.lower() == "catboost":
            model_uri = f"runs:/{run_id}/catboost_model"
            return mlflow.catboost.load_model(model_uri)
        elif model_type.lower() == "lightgbm":
            model_uri = f"runs:/{run_id}/lightgbm_model"
            return mlflow.lightgbm.load_model(model_uri)
        elif model_type.lower() == "random_forest":
            model_uri = f"runs:/{run_id}/random_forest_model"
            return mlflow.sklearn.load_model(model_uri)
        elif model_type.lower() == "knn":
            model_uri = f"runs:/{run_id}/knn_model"
            return mlflow.sklearn.load_model(model_uri)
        elif model_type.lower() == "svm":
            model_uri = f"runs:/{run_id}/svm_model"
            return mlflow.sklearn.load_model(model_uri)
            

        else:
            logger.error(f"Unsupported model type: {model_type}")
            raise ValueError(f"Unsupported model type: {model_type}")
    except Exception as e:
        logger.warning(f"Could not load {model_type} model from run {run_id}, training new one. Error: {str(e)}")
        return None

def build_base_models(selected_features, calibrate: bool = False, calibration_method: str = "isotonic"):
    """Build base models with fallback to new models if pretrained not found"""
    pretrained_xgb_run_id = "cd82a16976744025a9793e57d9917368"
    pretrained_cat_run_id = "32"
    pretrained_lgbm_run_id = "53"
    pretrained_rf_run_id = "1234567890"  # Replace with actual run ID
    pretrained_knn_run_id = "1234567890"  # Replace with actual run ID
    pretrained_svm_run_id = "1234567890"  # Replace with actual run ID

    # Attempt to load with fallback
    models = []
    for model_type, run_id in [
        ("xgboost", pretrained_xgb_run_id),
        ("catboost", pretrained_cat_run_id), 
        ("lightgbm", pretrained_lgbm_run_id),
        ("random_forest", pretrained_rf_run_id),
        ("knn", pretrained_knn_run_id),
        ("svm", pretrained_svm_run_id)
    ]:
        model = load_pretrained_model(run_id, model_type)
        if model is None:
            logger.info(f"Training new {model_type} model")
            
            model = {
                "xgboost": XGBClassifier(**ModelTrainingFeatures().training_params['xgb']),
                "catboost": CatBoostClassifier(**ModelTrainingFeatures().training_params['cat']),
                "lightgbm": LGBMClassifier(**ModelTrainingFeatures().training_params['lgbm']),
                "random_forest": RandomForestClassifier(**ModelTrainingFeatures().training_params['rf']),
                "knn": KNeighborsClassifier(**ModelTrainingFeatures().training_params['knn']),
                "svm": SVC(**ModelTrainingFeatures().training_params['svm'])
            }[model_type]
        models.append(model)
    
    # Unpack models
    xgb_model, cat_model, lgbm_model, rf_model, knn_model, svm_model = models
    
    # Optional calibration for all models
    if calibrate:
        # Use the same calibration for consistency; note pretrained_xgb_model might already be calibrated
        xgb_model = CalibratedClassifierCV(xgb_model, method=calibration_method, cv=3)
        cat_model = CalibratedClassifierCV(cat_model, method=calibration_method, cv=3)
        lgbm_model = CalibratedClassifierCV(lgbm_model, method=calibration_method, cv=3)
        rf_model = CalibratedClassifierCV(rf_model, method=calibration_method, cv=3)
        knn_model = CalibratedClassifierCV(knn_model, method=calibration_method, cv=3)
        svm_model = CalibratedClassifierCV(svm_model, method=calibration_method, cv=3)
    return xgb_model, cat_model, lgbm_model, rf_model, knn_model, svm_model

class ModelTrainingFeatures:
    """Stores and manages model training parameters and feature sets for ensemble models"""
    
    def __init__(self):
        # Initialize with default feature sets from JSON
        self.feature_sets = self._load_default_features()
        self.training_params = {
            'xgb': {
                'learning_rate': 0.08311627856474942,
                'early_stopping_rounds': 184,
                'min_child_weight': 157,
                'gamma': 0.026573915810307977,
                'subsample': 0.3985764276456313,
                'colsample_bytree': 0.9696188198654059,
                'scale_pos_weight': 2.7864672907283436,
                'reg_alpha': 0.00044056052563292266,
                'reg_lambda': 1.1040302522259011,
                'max_depth': 5,
                'n_estimators': 793,
                'objective': 'binary:logistic',
                'tree_method': 'hist',
                'device': 'cpu',
                'eval_metric': ['error', 'auc', 'aucpr'],
                'verbosity': 0,
                'nthread': -1,
                'random_state': 138,
            },
            'cat': {
                'loss_function': 'Logloss',
                'eval_metric': 'AUC',
                'task_type': 'CPU',
                'auto_class_weights': 'Balanced',
                'grow_policy': 'SymmetricTree',
                'iterations': 4901,
                'learning_rate': 0.013288620619359776,
                'depth': 7,
                'l2_leaf_reg': 11.152891877342054,
                'border_count': 128,
                'subsample': 0.7437647532474002,
                'random_strength': 0.14171788543304523,
                'min_data_in_leaf': 47,   
                'early_stopping_rounds': 996,
                
            },
            'lgbm': {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'device_type': 'cpu',
                'verbose': 100,
                'learning_rate': 0.0037460291406696956,
                'max_depth': 6,
                'reg_lambda': 0.4103475806096283,
                'n_estimators': 3577,
                'num_leaves': 67,
                'early_stopping_rounds': 291,
                'feature_fraction': 0.8037822667865142,
                'bagging_freq': 2,
                'min_child_samples': 28,
                'bagging_fraction': 0.7718378995036574,
                'feature_fraction_bynode': 0.9996212749980057  # Note: feature_fraction takes precedence over colsample_bytree
            },
            'rf': {
                'n_estimators': 784,
                'max_depth': 6,
                'min_samples_split': 12,
                'min_samples_leaf': 2,
                'max_features': 0.9667000359567445,
                'bootstrap': False,
                'n_jobs': -1,
                'random_state': 42,
                'class_weight': {0: 1, 1: 2.19}
            },
            'knn': {
                'n_neighbors': 2,
                'weights': 'distance',
                'algorithm': 'ball_tree',
                'leaf_size': 14,
                'p': 1,
                'n_jobs': -1
            },
            'svm': {
                'C': 0.6814185192456121,
                'kernel': 'rbf',
                'gamma': 'scale',
                'class_weight': {0: 1, 1: 2.19},  # Updated for cost-sensitive learning
                'probability': True,
                'shrinking': True,
                'tol': 0.03600890837739052,
                'cache_size': 12000,
                'max_iter': 1000,
                'decision_function_shape': 'ovr'
            }
        }
        
    def _load_default_features(self) -> dict:
        """Load default feature sets from JSON file"""
        try:
            return import_selected_features_ensemble('all')
        except FileNotFoundError:
            logger.info(
                "Feature sets JSON file not found",
                error_code=DataProcessingError.FILE_NOT_FOUND
            )
            return {}
        except json.JSONDecodeError:
            logger.info(
                "Invalid JSON format in feature sets file",
                error_code=DataProcessingError.FILE_CORRUPTED
            )
            return {}

    def get_features(self, model_type: str) -> list:
        """Get feature set for specific model type"""
        return self.feature_sets.get(model_type, [])

    def get_training_params(self, model_type: str) -> dict:
        """Get training parameters for specific model type"""
        return self.training_params.get(model_type, {})

    def update_features(self, model_type: str, features: list):
        """Update feature set for specific model type"""
        if model_type in self.feature_sets:
            self.feature_sets[model_type] = features

    def update_training_params(self, model_type: str, params: dict):
        """Update training parameters for specific model type"""
        if model_type in self.training_params:
            self.training_params[model_type].update(params)

    def save_training_params(self, file_path: str = 'utils/training_params.json'):
        """Save current training parameters to JSON file"""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.training_params, f, indent=4)
        except Exception as e:
            logger.info(
                f"Error saving training parameters: {str(e)}",
                error_code=DataProcessingError.FILE_CORRUPTED
            )

class EnsembleModel:
    """
    Modified EnsembleModel that respects individual model feature requirements
    """
    def __init__(self, logger, selected_features, calibrate=True, calibration_method="sigmoid"):
        """
        Initialize the EnsembleModel.
        """
        self.calibrate = calibrate
        self.calibration_method = calibration_method   
        self.model_training_features = ModelTrainingFeatures()
        self.selected_features = selected_features
        self.training_params = self.model_training_features.training_params
    
        # Build base models
        xgb_model, cat_model, lgbm_model, rf_model, knn_model, svm_model = build_base_models(
            selected_features=self.selected_features,
            calibrate=self.calibrate, 
            calibration_method=self.calibration_method
        )

        self.base_models = [
            ("xgb", xgb_model),
            ("cat", cat_model),
            ("lgbm", lgbm_model),
            ("rf", rf_model),
            ("knn", knn_model),
            ("svm", svm_model)
        ]
        
        # Precision-optimized weights (original: {'xgb':1.5, 'cat':1.8, 'lgbm':1.7, 'rf':1.2, 'knn':0.9, 'svm':1.1})
        self.weights = {
            'rf': 2.5,   # Highest precision model (35.9%)
            'xgb': 2.0,  # Moderate precision but critical for recall
            'cat': 1.5,  # Baseline precision
            'lgbm': 0.8, # Recall anchor (keep some weight)
            'svm': 0.3,  # Minimize low-precision impact
            'knn': 0.2   # Minimal weight
        }
        self.logger = logger

        # Store feature sets for each model
        self.feature_sets = {
            'xgb': self.selected_features,
            'cat': self.selected_features,
            'lgbm': self.selected_features,
            'rf': self.selected_features,
            'knn': self.selected_features,
            'svm': self.selected_features
        }

        # Initialize an XGBoost-based meta learner for stacking
        self.meta_learner = XGBClassifier(tree_method='hist', device='cpu', random_state=42)

        # Existing initialization for custom scorer (now unused by meta learner)
        def recall_precision_balance(y_true, y_pred):
            recall = recall_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            return 0.7 * recall + 0.3 * precision  # Custom weighting

        custom_scorer = make_scorer(recall_precision_balance)

    def train(self, X_train, y_train, X_val, y_val):
        """Modified training that respects individual model features"""
        # Validate feature consistency
        # self._validate_features(X_train)

        # Get all unique columns across feature sets
        all_columns = self.selected_features
        print(f"all_columns: {all_columns}")
        # Ensure all columns are present in training data
        X_train = X_train.reindex(columns=list(all_columns), fill_value=0)
        X_val = X_val.reindex(columns=list(all_columns), fill_value=0)

        # Train each model using all columns
        for name, model in self.base_models:
            # Skip if model is already fitted (CalibratedClassifierCV)
            if hasattr(model, 'calibrated_classifiers_'):
                continue
                
            # For SVM - subsample large datasets
            if name == 'svm' and len(X_train) > 10000:
                print(f"Subsampling SVM training data from {len(X_train)} to 10000")
                svm_train_idx = np.random.choice(
                    len(X_train), 
                    size=10000, 
                    replace=False
                )
                model.fit(
                    X_train.iloc[svm_train_idx], 
                    y_train.iloc[svm_train_idx]
                )
            # For KNN - incremental training
            elif name == 'knn' and len(X_train) > 20000:
                print(f"Subsampling KNN training data from {len(X_train)} to 20000")
                sample_idx = np.random.choice(len(X_train), size=20000, replace=False)
                model.fit(
                    X_train.iloc[sample_idx],
                    y_train.iloc[sample_idx]
                )
            elif name == 'cat':
                print(f"Training {name} model")
                # Handle CatBoost feature weights separately
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=100  # Only print every 100th round
                    # feature_weights=[1.0] * X_train.shape[1]  # Equal weights for all features as default
                )
            elif name == 'rf':
                print(f"Training {name} model")
                try:
                    model.fit(
                        X_train,
                        y_train,
                        verbose=100,  # Only print every 100th round
                        eval_set=[(X_val, y_val)]
                    )
                except TypeError as e:
                    # Handle models that don't accept verbose/eval_set parameters
                    model.fit(X_train, y_train)
            elif name == 'lgbm':
                print(f"Training {name} model")
                try:
                    model.fit(
                        X_train,
                        y_train,
                        eval_set=[(X_val, y_val)]
                    )
                except TypeError as e:
                    # Handle models that don't accept verbose/eval_set parameters
                    self.logger.warning(f"Error training {name} model: {e}")
                    model.fit(X_train, y_train)
            else:
                print(f"Training {name} model")
                try:
                    model.fit(
                        X_train, 
                        y_train,
                        verbose=100,  # Only print every 100th round
                        eval_set=[(X_val, y_val)]
                    )
                except TypeError as e:
                    # Handle models that don't accept verbose/eval_set parameters
                    model.fit(X_train, y_train)

        # Calculate optimal threshold by maximizing F1 score
        proba = self.predict_proba(X_val)
        precisions, recalls, thresholds = precision_recall_curve(y_val, proba)

        # Find threshold that meets recall minimum
        viable_thresholds = [t for t, r in zip(thresholds, recalls[:-1]) if r >= 0.20]
        if viable_thresholds:
            # Select highest precision among viable thresholds
            viable_precisions = [p for p, t in zip(precisions[:-1], thresholds) if t in viable_thresholds]
            optimal_idx = np.argmax(viable_precisions)
            optimal_threshold = viable_thresholds[optimal_idx]
        else:
            # Fallback to 0.5 if no threshold meets recall
            optimal_threshold = 0.5
            self.logger.warning("No threshold met 20% recall minimum")

        self.optimal_threshold = optimal_threshold  # Store for prediction
        mlflow.log_param("optimal_threshold", optimal_threshold)

        # Dynamically adjust voting weights based on individual model precision on validation set
        new_weights = {}
        for name, model in self.base_models:
            X_model = X_val.reindex(columns=self.selected_features, fill_value=0)
            try:
                pred_prob = model.predict_proba(X_model)[:, 1]
                binary_preds = (pred_prob >= self.optimal_threshold).astype(int)
                prec = precision_score(y_val, binary_preds, zero_division=0)
            except Exception as e:
                self.logger.warning(f"Error computing precision for model {name}: {e}")
                prec = 0.1
            new_weights[name] = prec
        # Normalize weights so they sum to 1
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            for key in new_weights:
                new_weights[key] /= total_weight
        else:
            new_weights = self.weights  # fallback if total_weight is 0
        self.weights = new_weights
        mlflow.log_param("dynamic_weights", str(self.weights))

        # Train the meta-learner using base model predictions as features
        meta_features = []
        for name, model in self.base_models:
            X_model = X_val.reindex(columns=self.selected_features, fill_value=0)
            try:
                preds = model.predict_proba(X_model)[:, 1]
            except Exception as e:
                self.logger.warning(f"Error predicting with model {name} for meta-learner: {e}")
                preds = np.zeros(len(X_val))
            meta_features.append(preds)
        meta_features = np.column_stack(meta_features)
        self.meta_learner.fit(meta_features, y_val)
        mlflow.log_param("meta_learner_trained", True)

        # Convert probabilities to binary predictions using optimal threshold
        binary_preds = (proba >= self.optimal_threshold).astype(int)
        class1_precision = precision_score(y_val, binary_preds, labels=[1], zero_division=0)
        mlflow.log_metric("class1_precision", class1_precision)
        if class1_precision < 0.4:
            self.logger.warning("Class 1 precision below safety threshold")

        class1_recall = recall_score(y_val, binary_preds, labels=[1], zero_division=0)
        mlflow.log_metric("class1_recall", class1_recall)
        if class1_recall < 0.15:
            self.logger.warning("Recall below minimum threshold")

        # if precision_score(y_val, binary_preds) < 0.4:
        #     raise ValueError("Precision safety check failed")

    def predict(self, X):
        proba = self.predict_proba(X)
        
        # Use tuned threshold if available
        threshold = getattr(self, 'optimal_threshold', 0.55)  
        
        # Confidence-based prediction
        predictions = np.zeros_like(proba)
        predictions[proba >= threshold] = 1
        
        # Ensure minimum diversity
        if predictions.sum() < len(X) * 0.20:  # Enforce minimum 20% positives
            top_indices = np.argsort(proba)[-int(len(X)*0.20):]
            predictions[top_indices] = 1
        
        return predictions

    def predict_proba(self, X):
        """Modified prediction with feature alignment"""
        predictions = []
        model_features = self.selected_features
        for (name, model), weight in zip(self.base_models, self.weights):
            # Create feature matrix with proper columns
            X_model = X.reindex(columns=model_features, fill_value=0)
            
            # Validate dimensions before prediction
            if X_model.shape[1] != len(model_features):
                missing = set(model_features) - set(X_model.columns)
                raise ValueError(f"Feature mismatch in {name}: Missing {len(missing)} features")
            
            # Ensure correct data types
            X_model = X_model.astype(np.float32)
            
            preds = model.predict_proba(X_model)[:, 1]
            predictions.append(preds * self.weights[name])  # Explicitly use numerical weights from model registry
        
        return np.mean(predictions, axis=0)

    def _meta_predict(self, X_subset):
        # Generate meta features from base models and use meta learner for final prediction
        meta_features = []
        for name, model in self.base_models:
            X_model = X_subset.reindex(columns=self.selected_features, fill_value=0)
            preds = model.predict_proba(X_model)[:, 1]
            meta_features.append(preds)
        meta_features = np.column_stack(meta_features)
        return self.meta_learner.predict(meta_features)

    def _validate_features(self, X):
        """Enhanced validation per model"""
        # Ensure selected_features is a list/set for iteration
        all_required = list(self.selected_features) if not isinstance(self.selected_features, (list, set)) else self.selected_features
        
        # Check existence
        missing = set(all_required) - set(X.columns)
        
        if missing:
            raise ValueError(f"Missing features: {missing}")
        
        # Check dimensions for all features
        if len(all_required) != X.shape[1]:
            raise ValueError(f"Feature count mismatch: "
                            f"Expected {len(all_required)}, got {X.shape[1]}")

if __name__ == "__main__":
    # Load data using your existing utility functions
    selected_features = import_selected_features_ensemble('all')
    X_train, y_train, X_test, y_test = import_training_data_ensemble()
    X_val, y_val = create_ensemble_evaluation_set()
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    X_val = X_val[selected_features]
    
    print(f"X_train.shape: {X_train.shape}")
    print(f"X_test.shape: {X_test.shape}")
    print(f"X_val.shape: {X_val.shape}")
    
    # Initialize ensemble model 
    ensemble_model = EnsembleModel(
        logger=logger,
        selected_features=selected_features,
        calibrate=True,
        calibration_method="sigmoid"
    )

    # Set MLflow experiment and start run
    with mlflow.start_run(run_name="ensemble_training") as run:
        # Fit the ensemble model
        ensemble_model.train(X_train, y_train, X_test, y_test)
        
        # Make predictions and evaluate
        predictions = ensemble_model.predict(pd.DataFrame(X_val))
        proba = ensemble_model.predict_proba(pd.DataFrame(X_val))
        
        print("Classification Report:")
        print(classification_report(y_val, predictions)) 
        mlflow.end_run()