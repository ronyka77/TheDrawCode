# Standard library imports
from typing import Any, Dict, Optional, Tuple, Union, List
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, average_precision_score
)
from sklearn.model_selection import train_test_split
import json
import pickle
import glob
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
os.environ['GIT_PYTHON_GIT_EXECUTABLE'] = "C:/Program Files/Git/bin/git.exe"
import mlflow
import mlflow.xgboost
from datetime import datetime

# Local imports
from utils.logger import ExperimentLogger
from utils.create_val_set import get_selected_columns_for_4_5_over,create_evaluation_sets_over_4_5_goals

class XGBoostModelOver35(BaseEstimator, ClassifierMixin):
    """XGBoost model implementation for over 3.5 goals prediction."""
    
    def __init__(
        self,
        logger: Optional[ExperimentLogger] = None,
        categorical_features: Optional[List[str]] = None
    ) -> None:
        """Initialize XGBoost model."""
        self.logger = logger or ExperimentLogger()
        self.model_type = 'xgboost_over35_model'
        
        # Global training parameters optimized for over 3.5 goals
        self.global_params = {
            'learning_rate': 0.001,
            'max_depth': 3,
            'min_child_weight': 60,
            'gamma': 8.0,
            'subsample': 0.75,
            'colsample_bytree': 0.75,
            'scale_pos_weight': 3.0,
            'n_estimators': 10000,
            'reg_alpha': 2.0,
            'reg_lambda': 1.5,
            'objective': 'binary:logistic',
            'tree_method': 'hist',
            'eval_metric': ['error', 'auc', 'aucpr'],
            'early_stopping_rounds': 300,
            'verbosity': 0,
            'random_state': 42
        }
        
        self.model = None
        self.feature_importance = {}
        self.selected_features = get_selected_columns_for_4_5_over()
        self.categorical_features = categorical_features or []
        self.threshold = 0.53  # Default threshold optimized for over 4.5 goals

    def _validate_data(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Validate and format input data."""
        X_df = pd.DataFrame(X) if isinstance(X, np.ndarray) else X.copy()
        y_s = pd.Series(y) if isinstance(y, np.ndarray) and y is not None else y
        
        # Ensure all features are present
        if self.selected_features:
            missing_features = set(self.selected_features) - set(X_df.columns)
            if missing_features:
                raise ValueError(f"Missing features in input data: {missing_features}")
            X_df = X_df[self.selected_features]
        
        return X_df, y_s

    def _optimize_threshold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> float:
        """Find optimal prediction threshold using validation data."""
        best_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'threshold': 0.53}
        best_score = 0
        
        # Focus on higher thresholds for better precision in over 4.5 predictions
        for threshold in np.arange(0.3, 0.7, 0.01):
            y_pred = (y_prob >= threshold).astype(int)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            
            # Modified scoring to prioritize precision for over 4.5 goals
            if precision >= 0.35:  # Higher minimum precision requirement
                score = precision * 0.7 + recall * 0.3  # Weight precision more heavily
                if score > best_score:
                    best_score = score
                    best_metrics.update({
                        'precision': precision,
                        'recall': recall,
                        'f1': f1_score(y_true, y_pred, zero_division=0),
                        'threshold': threshold
                    })
        
        self.logger.info(f"Best threshold: {best_metrics['threshold']}")
        return best_metrics['threshold']

    def train(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> None:
        """Train the model."""
        X_train_df, y_train_s = self._validate_data(X_train, y_train)
        
        if X_val is not None and y_val is not None:
            X_val_df, y_val_s = self._validate_data(X_val, y_val)
        else:
            X_val_df, y_val_s = None, None
        
        # Initialize XGBClassifier with parameters
        self.model = xgb.XGBClassifier(**self.global_params)
        
        # Fit the model
        if X_val_df is not None and y_val_s is not None:
            self.model.fit(
                X_train_df,
                y_train_s,
                eval_set=[(X_train_df, y_train_s)],
                verbose=False
            )
            
            # Optimize threshold for over 4.5 goals prediction
            val_probs = self.model.predict_proba(X_val_df)[:, 1]
            self.threshold = self._optimize_threshold(y_val_s, val_probs)
        else:
            self.model.fit(X_train_df, y_train_s)
        
        # Store feature importance
        self.feature_importance = dict(zip(
            X_train_df.columns,
            self.model.feature_importances_
        ))

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict over 4.5 goals probabilities."""
        X_df, _ = self._validate_data(X)
        return self.model.predict_proba(X_df)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict using the optimized threshold for over 4.5 goals."""
        probas = self.predict_proba(X)[:, 1]
        return (probas >= self.threshold).astype(int)

    def analyze_predictions(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, Any]:
        """Analyze over 4.5 goals predictions."""
        X_df, y_s = self._validate_data(X, y)
        probas = self.predict_proba(X_df)[:, 1]
        predictions = (probas >= self.threshold).astype(int)
        
        analysis = {
            'metrics': {
                'precision': precision_score(y_s, predictions, zero_division=0),
                'recall': recall_score(y_s, predictions, zero_division=0),
                'f1': f1_score(y_s, predictions, zero_division=0),
                'accuracy': accuracy_score(y_s, predictions)
            },
            'probability_stats': {
                'mean': float(np.mean(probas)),
                'std': float(np.std(probas)),
                'min': float(np.min(probas)),
                'max': float(np.max(probas))
            },
            'class_distribution': {
                'predictions': pd.Series(predictions).value_counts().to_dict(),
                'actual': pd.Series(y_s).value_counts().to_dict()
            },
            'over45_rate': float(y_s.mean()),
            'predicted_rate': float(predictions.mean())
        }
        
        return analysis

    def save(self, path: str) -> None:
        """Save the model and its metadata."""
        try:
            # Save the model
            model_path = f"{path}_over45.model"
            self.model.save_model(model_path)
            
            # Save metadata
            metadata = {
                'global_params': self.global_params,
                'threshold': self.threshold,
                'selected_features': self.selected_features,
                'feature_importance': self.feature_importance
            }
            
            with open(f"{path}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error saving model: {str(e)}")
            raise

    def load(self, path: str) -> None:
        """Load the model and its metadata."""
        try:
            # Load the model
            model_path = f"{path}_over45.model"
            self.model = xgb.XGBClassifier()
            self.model.load_model(model_path)
            
            # Load metadata
            with open(f"{path}_metadata.json", 'r') as f:
                metadata = json.load(f)
                self.global_params = metadata.get('global_params', {})
                self.threshold = metadata.get('threshold', 0.53)
                self.selected_features = metadata.get('selected_features', get_selected_columns_for_4_5_over())
                self.feature_importance = metadata.get('feature_importance', {})
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading model: {str(e)}")
            raise 

    def predict_draw_probability(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Get over 4.5 probabilities."""
        X_df = pd.DataFrame(X)
        X_df = X_df[self.selected_features]
        return self.predict_proba(X_df)[:, 1]

    def _log_completion_metrics(
        self,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series]
    ) -> None:
        """Log completion metrics."""
        if X_val is not None and y_val is not None:
            analysis = self.analyze_predictions(X_val, y_val)
            
            # Log overall metrics
            if self.logger:
                self.logger.info(f"best_iteration: {self.model.best_iteration}")
                self.logger.info(f"best_score: {self.model.best_score}")
                self.logger.info(f"n_features: {len(self.feature_importance)}")
                self.logger.info(f"val_accuracy: {analysis['metrics']['accuracy']}")
                self.logger.info(f"val_precision: {analysis['metrics']['precision']}")
                self.logger.info(f"val_recall: {analysis['metrics']['recall']}")
                self.logger.info(f"val_f1: {analysis['metrics']['f1']}")

def import_training_data():
        """Import training data from data manager."""
        data_path = "data/training_data.csv"
        data = pd.read_csv(data_path)
        data['is_45over'] = ((data['home_goals'] + data['away_goals']) > 4.5).astype(float)
        
        # Define columns to drop
        columns_to_drop = [
            'match_outcome', 'home_goals', 'away_goals', 'total_goals',
            'Referee', 'Venue', 'Home', 'Away', 'Datum', 'league'  # Drop categorical columns
        ]
        
        # Drop specified columns
        data = data.drop(columns=columns_to_drop, errors='ignore')
        
        # Convert all remaining string/object columns to numeric
        for col in data.columns:
            if data[col].dtype == 'object' or data[col].dtype == 'datetime64[ns]':
                try:
                    # Replace comma with dot and convert to float
                    data[col] = (data[col].astype(str)
                            .str.strip()  # Remove leading/trailing whitespace
                            .str.strip("'\"")  # Remove quotes
                            .str.replace(' ', '')  # Remove any spaces
                            .str.replace(',', '.')  # Replace comma with dot
                            .astype(float))  # Convert to float
                    
                    # Debug: Print sample values after conversion
                    print(f"Column {col} sample values after conversion:")
                    # print(data[col].head())
                    
                except (AttributeError, ValueError) as e:
                    print(f"Could not convert column {col}: {str(e)}")
                    # Drop the column if it exists
                    if col in data.columns:
                        data.drop(columns=[col], inplace=True)
                    continue
        
        # Verify all columns are numeric
        non_numeric_cols = data.select_dtypes(exclude=['int64', 'float64']).columns
        if len(non_numeric_cols) > 0:
            print(f"Dropping non-numeric columns: {non_numeric_cols}")
            data = data.drop(columns=non_numeric_cols)
        
        # Split into train and test sets
        train_data, test_data = train_test_split(data, test_size=0.3, random_state=42, shuffle=True)
        selected_features = get_selected_columns_for_4_5_over()
        X_train = train_data[selected_features]
        y_train = train_data['is_45over']
        X_test = test_data[selected_features]
        y_test = test_data['is_45over']
        
        # Log the shape of the data
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        
        return X_train, y_train, X_test, y_test

def train_with_mlflow():
    """Train XGBoost model for over 4.5 goals prediction with MLflow tracking."""
    logger = ExperimentLogger()
    current_dir = os.getcwd()
    
    # Set up SQLite backend for MLflow
    if current_dir.startswith('\\\\'): 
        db_path = os.path.join(current_dir, 'mlflow.db')
        artifact_path = os.path.join(current_dir, 'mlflow_artifacts')
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    else:
        db_path = "./mlflow.db"
        artifact_path = os.path.join(project_root, "mlflow_artifacts")
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    
    # Ensure artifact directory exists
    os.makedirs(artifact_path, exist_ok=True)
    os.environ["MLFLOW_ARTIFACT_ROOT"] = artifact_path
    
    mlflow.set_experiment("over45_xgboost_model")
    
    # Import data
    X_train, y_train, X_test, y_test = import_training_data()
    X_val, y_val = create_evaluation_sets_over_4_5_goals()
    
    with mlflow.start_run(run_name=f"over45_xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log dataset info
        mlflow.log_params({
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "n_features_original": X_train.shape[1],
            "over45_ratio_train": (y_train == 1).mean(),
            "over45_ratio_val": (y_val == 1).mean()
        })
        
        # Create and train model
        xgb_model = XGBoostModelOver45(logger=logger)
        X_train = X_train[xgb_model.selected_features]
        X_val = X_val[xgb_model.selected_features]
        X_test = X_test[xgb_model.selected_features]
        
        # Log model parameters
        mlflow.log_params(xgb_model.global_params)
        mlflow.log_param("selected_features", xgb_model.selected_features)
        
        print("Starting training...")
        # Train the model
        xgb_model.train(X_train, y_train, X_val, y_val)
        
        # Create input example using only the selected features
        input_example = X_train.iloc[:1].copy()
        input_example = convert_int_columns(input_example)
    
        # Create model signature
        signature = mlflow.models.infer_signature(
            input_example,
            xgb_model.predict(input_example)
        )
        
        # Log the model
        mlflow.xgboost.log_model(
            xgb_model.model,
            "model_over45",
            signature=signature,
            input_example=input_example
        )
        
        # Analyze and log metrics
        train_analysis = xgb_model.analyze_predictions(X_train, y_train)
        val_analysis = xgb_model.analyze_predictions(X_val, y_val)
        test_analysis = xgb_model.analyze_predictions(X_test, y_test)
        
        logger.info(f"Train analysis: {train_analysis}")
        logger.info(f"Val analysis: {val_analysis}")
        logger.info(f"Test analysis: {test_analysis}")
        
        metrics_to_log = {
            "train_precision": train_analysis['metrics']['precision'],
            "train_recall": train_analysis['metrics']['recall'],
            "val_precision": val_analysis['metrics']['precision'],
            "val_recall": val_analysis['metrics']['recall'],
            "test_precision": test_analysis['metrics']['precision'],
            "test_recall": test_analysis['metrics']['recall']
        }
        
        mlflow.log_metrics(metrics_to_log)
        
        # Log analyses
        mlflow.log_dict(train_analysis, "train_analysis.json")
        mlflow.log_dict(val_analysis, "val_analysis.json")
        mlflow.log_dict(test_analysis, "test_analysis.json")

        logger.info(f"Training completed. MLflow run ID: {mlflow.active_run().info.run_id}")
        return xgb_model

def convert_int_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert integer columns to int32 for MLflow compatibility."""
    for col in df.columns:
        if df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    return df

if __name__ == "__main__":
    model = train_with_mlflow()
      