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
    pretrained_xgb_run_id = "12"
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
                'learning_rate': 0.00760022315772835,
                'early_stopping_rounds': 468,
                'min_child_weight': 135,
                'gamma': 2.6424389418822702,
                'subsample': 0.5507474166255953,
                'colsample_bytree': 0.8605669463072969,
                'scale_pos_weight': 2.3728539564011863,
                'reg_alpha': 0.00024338752831206268,
                'reg_lambda': 0.012210974240024406,
                'max_depth': 8,
                'n_estimators': 25840,
                'objective': 'binary:logistic',
                'tree_method': 'hist',
                'device': 'cpu',
                'eval_metric': ['error', 'auc', 'aucpr'],
                'verbosity': 0,
                'nthread': -1
            },
            'cat': {
                'iterations': 4209,
                'learning_rate': 0.0025294605477838576,
                'depth': 9,
                'l2_leaf_reg': 0.926979888847891,
                'border_count': 160,
                'subsample': 0.7396709709608331,
                'random_strength': 2.157330791523986,
                'auto_class_weights': 'Balanced',
                'grow_policy': 'SymmetricTree',
                'min_data_in_leaf': 79,
                'loss_function': 'Logloss',
                'early_stopping_rounds': 923,
                'task_type': 'CPU'
            },
            'lgbm': {
                'boosting_type': 'gbdt',  # Default boosting type that supports bagging
                'num_leaves': 31,
                'learning_rate': 0.005,  # Lower than XGB/CatBoost for stability
                'n_estimators': 15000,
                'scale_pos_weight': 2.5,  # Higher than XGB for class balance
                'min_child_samples': 200,
                'feature_fraction': 0.6,  # More conservative than XGB's 0.86
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'reg_alpha': 0.1,
                'reg_lambda': 0.5,
                'max_depth': 7,  # Shallower than XGB/CatBoost
                'objective': 'binary',
                'metric': 'binary_logloss',
                'device_type': 'cpu',
                'early_stopping_round': 500,
                'verbosity': -1
            },
            'rf': {
                'n_estimators': 1000,
                'max_depth': 12,  # Deeper than current 8
                'min_samples_split': 50,  # Higher for precision
                'min_samples_leaf': 20,
                'max_features': 'sqrt',  # Better than default for precision
                'class_weight': {0: 1, 1: 3},  # Explicit class weights
                'bootstrap': True,
                'n_jobs': -1,
                'max_samples': 0.5,  # Prevent overfitting
                'ccp_alpha': 0.01  # Cost-complexity pruning
            },
            'knn': {
                'n_neighbors': 50,  # Higher than current 5
                'weights': 'distance',
                'algorithm': 'brute',  # More precise than auto
                'leaf_size': 20,
                'p': 1,  # Manhattan distance
                'n_jobs': -1,
                'metric_params': {'w': 3}  # Weight class 1 higher
            },
            'svm': {
                'C': 0.5,  # Lower than current for tighter margin
                'kernel': 'sigmoid',  # Better for high dimensions
                'gamma': 'scale',
                'class_weight': 'balanced',
                'probability': True,
                'shrinking': True,
                'tol': 1e-4,
                'cache_size': 2000,  # Critical for large datasets
                'decision_function_shape': 'ovr',
                'break_ties': True  # Handle ambiguous cases
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
    def __init__(self, logger, selected_features, voting_method="soft", weights=None, calibrate=False, calibration_method="isotonic"):
        """
        Initialize the EnsembleModel.
        """
        self.calibrate = calibrate
        self.calibration_method = calibration_method   
        self.voting_method = voting_method
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
        
        # Setup custom weights if provided; default to equal weighting
        self.weights = weights if weights is not None else {
            'xgb': 1.5,
            'cat': 1.8, 
            'lgbm': 1.7,
            'rf': 1.2,
            'knn': 0.9,
            'svm': 1.1
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

        # Add to EnsembleModel __init__:
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
                    y_train.iloc[svm_train_idx],
                    eval_set=[(X_val, y_val)],
                    verbose=100
                )
            # For KNN - incremental training
            elif name == 'knn' and len(X_train) > 20000:
                print(f"Subsampling KNN training data from {len(X_train)} to 20000")
                sample_idx = np.random.choice(len(X_train), size=20000, replace=False)
                model.fit(
                    X_train.iloc[sample_idx],
                    y_train.iloc[sample_idx]
                )
            else:
                print(f"Training {name} model")
                # Handle CatBoost feature weights separately
                if name == 'catboost':
                    model.fit(
                        X_train,
                        y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=100,
                        feature_weights=[1.0] * X_train.shape[1]  # Equal weights for all features as default
                    )
                else:
                    print(f"Training {name} model")
                    try:
                        model.fit(
                            X_train, 
                            y_train,
                            eval_set=[(X_val, y_val)]
                        )
                    except TypeError as e:
                        # Handle models that don't accept verbose/eval_set parameters
                        model.fit(
                            X_train, 
                            y_train
                        )

        # Calculate optimal threshold using validation set
        proba = self.predict_proba(X_val)
        precisions, recalls, thresholds = precision_recall_curve(y_val, proba)
        optimal_idx = np.argmax(precisions >= 0.5)  # First threshold reaching 50% precision
        optimal_threshold = thresholds[optimal_idx]
        self.optimal_threshold = optimal_threshold  # Store for prediction
        mlflow.log_param("optimal_threshold", optimal_threshold)

        # Add to train method:
        # Convert probabilities to binary predictions using optimal threshold
        binary_preds = (proba >= optimal_threshold).astype(int)
        class1_precision = precision_score(y_val, binary_preds, labels=[1], zero_division=0)
        mlflow.log_metric("class1_precision", class1_precision)
        if class1_precision < 0.4:
            logger.warning("Class 1 precision below safety threshold")

        # Add to train method:
        class1_recall = recall_score(y_val, binary_preds, labels=[1], zero_division=0)
        mlflow.log_metric("class1_recall", class1_recall)
        if class1_recall < 0.15:
            logger.warning("Recall below minimum threshold")

        if precision_score(y_val, binary_preds) < 0.4:
            raise ValueError("Precision safety check failed")

        # After training
        print(f"XGB Features: {len(self.feature_sets['xgb'])}")  # Should be 99
        print(f"SVM Features: {len(self.feature_sets['svm'])}")   # Should be 50

    def predict(self, X):
        proba = self.predict_proba(X)
        
        # Use tuned threshold if available
        threshold = getattr(self, 'optimal_threshold', 0.55)  
        
        # Confidence-based prediction
        predictions = np.zeros_like(proba)
        predictions[proba >= threshold] = 1
        
        # Ensure minimum diversity
        if predictions.sum() < len(X) * 0.12:  # At least 12% positives
            top_indices = np.argsort(proba)[-int(len(X)*0.12):]
            predictions[top_indices] = 1
        
        return predictions
    
    def predict_proba(self, X):
        """Modified prediction with feature alignment"""
        predictions = []
        
        for (name, model), weight in zip(self.base_models, self.weights):
            model_features = self.feature_sets[name]
            
            # Create feature matrix with proper columns
            X_model = X.reindex(columns=model_features, fill_value=0)
            
            # Validate dimensions before prediction
            if X_model.shape[1] != len(model_features):
                missing = set(model_features) - set(X_model.columns)
                raise ValueError(f"Feature mismatch in {name}: Missing {len(missing)} features")
            
            # Ensure correct data types
            X_model = X_model.astype(np.float32)
            
            preds = model.predict_proba(X_model)[:, 1]
            predictions.append(preds * weight)
        
        return np.mean(predictions, axis=0)
    
    def _meta_predict(self, X_subset):
        # Use simple logistic regression check
        return (X_subset['draw_probability_score'] > 0.6).astype(int)

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
    # xgb_features = selected_features
    # cat_features = selected_features
    # lgbm_features = selected_features
    # rf_features = selected_features
    # knn_features = selected_features
    # svm_features = selected_features

    # all_features = list(
    #     set(xgb_features + cat_features + lgbm_features)
    # )
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
        weights={'xgb': 1.5, 'cat': 1.8, 'lgbm': 1.7, 'rf': 1.2, 'knn': 0.9, 'svm': 1.1},  # Boost LightGBM (better recall potential) using dict format
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