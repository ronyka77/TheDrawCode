"""
TabNet model implementation for binary classification with global training capabilities.
Provides methods for model training, prediction, and analysis with configurable parameters.
"""

# Standard library imports
import os

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  
os.environ['TORCHDYNAMO_DISABLE'] = '1'
# Third-party imports
import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import mlflow
import torch
import json

# Add project root to Python path
try:
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
except Exception as e:
    print(f"Error setting project root path: {e}")
    sys.path.append(os.getcwd().parent)
  

os.environ['TORCHDYNAMO_DISABLE'] = '1'

# Local imports
from utils.logger import ExperimentLogger
from utils.create_evaluation_set import get_selected_api_columns_draws, import_training_data_draws_api, create_evaluation_sets_draws_api, setup_mlflow_tracking

experiment_name = "tabnet_api_model"
mlruns_dir = setup_mlflow_tracking(experiment_name)

class TabNetModel:
    """TabNet model implementation for binary classification."""

    def __init__(self, logger: Optional[ExperimentLogger] = None):
        """Initialize TabNet model."""
        self.logger = logger or ExperimentLogger()
        self.model = None
        self.selected_features = get_selected_api_columns_draws()
        self.threshold = 0.55  # Default threshold for predictions

        # Global training parameters
        self.global_params = {
            'n_d': 8,
            'n_a': 8,
            'n_steps': 3,
            'gamma': 1.3,
            'cat_idxs': [],
            'cat_dims': [],
            'cat_emb_dim': 1,
        }

    def train(
        self,
        features_train: pd.DataFrame,
        target_train: pd.Series,
        features_val: Optional[pd.DataFrame] = None,
        target_val: Optional[pd.Series] = None,) -> None:
        """Train the TabNet model."""
        
        print(features_train.shape)
        
        # Drop NaN values from training and validation data
        features_train = features_train.dropna(subset=self.selected_features)
        target_train = target_train[features_train.index]
        
        print(features_train.shape)
        
        if features_val is not None:
            features_val = features_val.dropna(subset=self.selected_features)
            target_val = target_val[features_val.index]
        
        # Convert numeric columns to float
        for col in features_train.columns:
            if features_train[col].dtype == 'object':
                try:
                    features_train[col] = pd.to_numeric(features_train[col], errors='raise')
                except ValueError:
                    features_train = features_train.drop(columns=[col])
        
        # Convert data to numpy arrays
        X_train = np.array(features_train[self.selected_features])
        y_train = np.array(target_train)
        X_val = np.array(features_val[self.selected_features]) if features_val is not None else None
        y_val = np.array(target_val) if target_val is not None else None

        # Initialize TabNet model
        self.model = TabNetClassifier(
            n_d=self.global_params['n_d'],
            n_a=self.global_params['n_a'],
            n_steps=self.global_params['n_steps'],
            gamma=self.global_params['gamma'],
            cat_idxs=self.global_params['cat_idxs'],
            cat_dims=self.global_params['cat_dims'],
            cat_emb_dim=self.global_params['cat_emb_dim'],
            optimizer_fn=torch.optim.SGD,  # Use SGD instead of Adam
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size": 10, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type="sparsemax",
            verbose=1,
        )

        # Train the model
        self.model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_val, y_val)] if X_val is not None else None,
            eval_name=["val"] if X_val is not None else None,
            eval_metric=["auc", "logloss"],
            max_epochs=100,
            patience=10,
            batch_size=1024,
            virtual_batch_size=128,
        )

        # Save the model and metadata immediately after fitting
        if mlflow.active_run():
            # Log the model using MLflow's pyfunc functionality
            class TabNetWrapper(mlflow.pyfunc.PythonModel):
                def __init__(self, model):
                    self.model = model

                def predict(self, context, model_input):
                    return self.model.predict_proba(model_input)[:, 1]

            # Create a wrapper for the TabNet model
            tabnet_wrapper = TabNetWrapper(self.model)

            # Log the model
            mlflow.pyfunc.log_model(
                artifact_path="tabnet_model",
                python_model=tabnet_wrapper,
                registered_model_name="TabNet_API_Model",
            )

            # Log metadata
            mlflow.log_params(self.global_params)
            mlflow.log_metric("threshold", self.threshold)
            mlflow.log_dict(
                {"selected_features": self.selected_features},
                "selected_features.json",
            )

            print("Model and metadata logged to MLflow.")

        # Optimize threshold if validation data is provided
        if X_val is not None:
            val_probs = self.predict_proba(features_val)[:, 1]
            self.threshold = self._optimize_threshold(target_val, val_probs)

    def predict_proba(self, features_val: pd.DataFrame) -> np.ndarray:
        """Predict probabilities using the TabNet model."""
        features_df = pd.DataFrame(features_val)
        features_selected = features_df[self.selected_features]
        X = np.array(features_selected)
        return self.model.predict_proba(X)

    def predict(self, features_val: pd.DataFrame) -> np.ndarray:
        """Predict using the global threshold."""
        probas = self.predict_proba(features_val)[:, 1]
        return (probas >= self.threshold).astype(int)

    def _optimize_threshold(self, y_true: pd.Series, y_prob: np.ndarray) -> float:
        """Find optimal prediction threshold using validation data."""
        best_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'threshold': 0.55}
        best_score = 0

        for threshold in np.arange(0.3, 0.95, 0.01):
            y_pred = (y_prob >= threshold).astype(int)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)

            if precision >= 0.35:  # Higher minimum precision requirement
                score = precision * min(recall, 0.20)  # Lower recall cap
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

    def analyze_predictions(
        self,
        features_val: pd.DataFrame,
        target_val: pd.Series) -> Dict[str, Any]:
        """Analyze model predictions."""
        probas = self.predict_proba(features_val)[:, 1]
        predictions = (probas >= self.threshold).astype(int)

        analysis = {
            'metrics': {
                'precision': precision_score(target_val, predictions, zero_division=0),
                'recall': recall_score(target_val, predictions, zero_division=0),
                'f1': f1_score(target_val, predictions, zero_division=0),
            },
            'probability_stats': {
                'mean': float(np.mean(probas)),
                'std': float(np.std(probas)),
                'min': float(np.min(probas)),
                'max': float(np.max(probas)),
            },
            'class_distribution': {
                'predictions': pd.Series(predictions).value_counts().to_dict(),
                'actual': pd.Series(target_val).value_counts().to_dict(),
            },
            'draw_rate': float(target_val.mean()),
            'predicted_rate': float(predictions.mean()),
            'n_samples': len(target_val),
            'n_draws': int(target_val.sum()),
            'n_predicted': int(predictions.sum()),
            'n_correct': int(np.logical_and(target_val, predictions).sum()),
        }
        return analysis

    def save(self, path: str) -> None:
        """Save the model and its metadata to disk."""
        try:
            self.model.save_model(f"{path}_tabnet.zip")
            metadata = {
                'threshold': self.threshold,
                'selected_features': self.selected_features,
            }
            with open(f"{path}_metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error saving model: {str(e)}")
            raise

    def load(self, path: str) -> None:
        """Load the model and its metadata from disk."""
        try:
            self.model = TabNetClassifier()
            self.model.load_model(f"{path}_tabnet.zip")
            with open(f"{path}_metadata.json", 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                self.threshold = metadata.get('threshold', 0.55)
                self.selected_features = metadata.get('selected_features', [])
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading model: {str(e)}")
            raise

def train_with_mlflow():
    """Train TabNet model with MLflow tracking."""
    logger = ExperimentLogger(experiment_name="tabnet_api_model", log_dir='./logs/tabnet_model')

    features_train, target_train, features_test, target_test = import_training_data_draws_api()
    features_val, target_val = create_evaluation_sets_draws_api()

    with mlflow.start_run(run_name=f"tabnet_api_model"):
        # Log dataset info
        mlflow.log_params({
            "train_samples": len(features_train),
            "val_samples": len(features_val),
            "test_samples": len(features_test),
            "n_features_original": features_train.shape[1],
        })

        # Create and train model
        tabnet_model = TabNetModel(logger)
        tabnet_model.train(features_train, target_train, features_val, target_val)

        # Evaluate on test set
        test_analysis = tabnet_model.analyze_predictions(features_test, target_test)
        mlflow.log_metrics({
            "test_precision": test_analysis['metrics']['precision'],
            "test_recall": test_analysis['metrics']['recall'],
            "test_f1": test_analysis['metrics']['f1'],
        })
       

        logger.info("TabNet model training completed successfully.")

if __name__ == "__main__":
    train_with_mlflow()