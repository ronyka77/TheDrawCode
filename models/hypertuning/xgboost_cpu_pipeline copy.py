"""
XGBoost CPU-only pipeline implementation for the Soccer Prediction Project.
Designed to work alongside the H2O stacked ensemble while providing Windows compatibility.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
import mlflow

# Add project root to Python path
try:
    project_root = Path(__file__).parent.parent.parent
    if not project_root.exists():
        project_root = Path(r"\\".join(str(project_root).split("\\")))
    sys.path.append(str(project_root))
except Exception as e:
    print(f"Error setting project root path: {e}")
    sys.path.append(os.getcwd().parent.parent)

# Local imports
from utils.logger import ExperimentLogger
experiment_name = "xgboost_cpu_pipeline"
logger = ExperimentLogger(experiment_name=experiment_name, log_dir='./logs/xgboost_cpu_pipeline')

from utils.create_evaluation_set import (
    create_ensemble_evaluation_set,
    import_selected_features_ensemble,
    import_training_data_ensemble,
    setup_mlflow_tracking
)

mlruns_dir = setup_mlflow_tracking(experiment_name)

class XGBoostModel:
    """XGBoost model wrapper with memory optimization."""
    
    def __init__(
        self,
        params: Dict[str, Any],
        logger: Any,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50):
        self.params = params
        self.train_params = {k: v for k, v in params.items() if k not in ['n_estimators', 'early_stopping_rounds']}
        self.logger = logger
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None
        self._best_iteration = None
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, eval_set: List[Tuple[pd.DataFrame, pd.Series]]) -> None:
        """Train the model using pandas DataFrames directly."""
        try:
            # Create classifier
            self.model = xgb.XGBClassifier(
                **self.train_params,
                n_estimators=self.num_boost_round,
                early_stopping_rounds=self.early_stopping_rounds
            )
            
            # Train model
            self.model.fit(
                X_train, 
                y_train,
                eval_set=eval_set,
                verbose=False
            )
            
            # Store best iteration
            self._best_iteration = self.model.best_iteration
            
            # Log training results
            self.logger.info(f"Best iteration: {self._best_iteration}")
            evals_result = self.model.evals_result()
            if evals_result:
                for metric, values in evals_result['validation_0'].items():
                    best_value = min(values) if 'error' in metric else max(values)
                    self.logger.info(f"Best {metric}: {best_value:.4f}")
            
            # Force garbage collection
            import gc
            gc.collect()
            
        except Exception as e:
            self.logger.error(f"Error fitting XGBoost model: {str(e)}")
            raise
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using pandas DataFrame directly."""
        try:
            # Process in chunks for memory efficiency
            chunk_size = 10000
            predictions = []
            
            for i in range(0, len(X), chunk_size):
                chunk = X.iloc[i:i + chunk_size]
                chunk_preds = self.model.predict(chunk)
                predictions.append(chunk_preds)
            
            return np.concatenate(predictions)
            
        except Exception as e:
            self.logger.error(f"Error predicting with XGBoost model: {str(e)}")
            raise
            
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities using pandas DataFrame directly."""
        try:
            # Process in chunks for memory efficiency
            chunk_size = 10000
            probabilities = []
            
            for i in range(0, len(X), chunk_size):
                chunk = X.iloc[i:i + chunk_size]
                chunk_probs = self.model.predict_proba(chunk)
                probabilities.append(chunk_probs)
            
            return np.concatenate(probabilities)
            
        except Exception as e:
            self.logger.error(f"Error getting probabilities from XGBoost model: {str(e)}")
            raise
            
    def save(self, path: str) -> None:
        """Save the model using MLflow."""
        try:
            # Log model with MLflow
            # Create signature for model input/output validation
            input_example = pd.DataFrame(columns=self.model.feature_names)
            signature = mlflow.models.infer_signature(
                input_example,
                self.model.predict(input_example)
            )

            # Log model with MLflow, ensuring all parameters are serializable
            mlflow.xgboost.log_model(
                xgb_model=self.model,
                artifact_path=path,
                registered_model_name=f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M')}",
                signature=signature,
                metadata={
                    'best_iteration': int(self._best_iteration),  # Ensure serializable
                    'params': {k: str(v) for k, v in self.params.items()}  # Convert all params to strings
                }
            )
        except Exception as e:
            self.logger.error(f"Error saving XGBoost model: {str(e)}")
            raise
            
    def load(self, path: str) -> None:
        """Load the model from MLflow."""
        try:
            # Load model from MLflow
            self.model = mlflow.xgboost.load_model(path)
            
            # Get metadata from MLflow
            metadata = mlflow.get_run(path).data.params
            self._best_iteration = int(metadata.get('best_iteration', 0))
            self.params.update({k: v for k, v in metadata.items() 
                                if k not in ['best_iteration']})
                
        except Exception as e:
            self.logger.error(f"Error loading XGBoost model: {str(e)}")
            raise
                
        except Exception as e:
            self.logger.error(f"Error loading XGBoost model: {str(e)}")
            raise

class XGBoostCPUTrainer:
    """XGBoost trainer with CPU-only configuration and precision-focused tuning."""
    
    MIN_SAMPLES: int = 1000
    DEFAULT_THRESHOLD: float = 0.50
    TARGET_PRECISION: float = 0.5
    TARGET_RECALL: float = 0.15

    def __init__(
        self,
        logger: Optional[ExperimentLogger] = None,
        max_runtime_secs: int = 3600,
        seed: int = 42,
        max_models: int = 30,
        model_dir: Optional[str] = None):
        self.logger = logger or ExperimentLogger(
            experiment_name="xgboost_cpu_pipeline",
            log_dir='./logs/xgboost_cpu_pipeline'
        )
        self.max_runtime_secs = max_runtime_secs
        self.seed = seed
        self.max_models = max_models
        self.model_dir = model_dir or os.path.join(project_root, "models", "xgboost_cpu_pipeline")
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.models: List[XGBoostModel] = []
        self.threshold = self.DEFAULT_THRESHOLD
        self.best_precision = 0
        self.best_recall = 0
        self._check_logger()
        self._monitor_memory_usage()

    def _check_logger(self) -> None:
        """Verify logger is properly initialized and functional."""
        if not self.logger:
            raise ValueError("Logger not initialized")
        try:
            self.logger.info("Logger check: Initialization successful")
        except Exception as e:
            raise ValueError(f"Logger test failed: {str(e)}")

    def _monitor_memory_usage(self) -> None:
        """Monitor memory usage during training."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_usage_gb = memory_info.rss / (1024 * 1024 * 1024)
            
            self.logger.info(f"Current memory usage: {memory_usage_gb:.2f} GB")
            
            if memory_usage_gb > 12:
                self.logger.warning(f"High memory usage detected: {memory_usage_gb:.2f} GB")
                
        except ImportError:
            self.logger.warning("psutil not installed, memory monitoring disabled")
        except Exception as e:
            self.logger.error(f"Error monitoring memory: {str(e)}")

    def _setup_hyperparameters(self) -> List[Dict[str, Any]]:
        """Define hyperparameter configurations using memory-efficient sampling."""
        base_params = {
            "tree_method": "hist",
            "objective": "binary:logistic",
            "eval_metric": ["aucpr", "logloss"],
            "seed": self.seed
        }
        
        param_ranges = {
            "max_depth": [2, 3, 4, 5, 6, 7, 8],
            "eta": [0.005, 0.01, 0.015, 0.02, 0.025, 0.03],
            "subsample": [0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            "colsample_bytree": [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85],
            "min_child_weight": [25, 50, 100, 150, 200],
            "gamma": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
            "alpha": [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
            "lambda": [7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5],
            "n_estimators": [300, 500, 750, 1000, 1250],
            "early_stopping_rounds": [200, 300, 400, 500, 600]
        }
        
        np.random.seed(self.seed)
        param_list = []
        
        for _ in range(self.max_models):
            params = {
                key: np.random.choice(values)
                for key, values in param_ranges.items()
            }
            param_list.append({**base_params, **params})
        
        return param_list

    def _train_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series) -> None:
        """Train multiple XGBoost models with memory-efficient batch processing."""
        try:
            param_list = self._setup_hyperparameters()
            val_preds_sum = np.zeros(len(y_val))
            models_count = 0
            
            batch_size = 5
            for batch_start in range(0, len(param_list), batch_size):
                self._monitor_memory_usage()
                
                batch_end = min(batch_start + batch_size, len(param_list))
                batch_params = param_list[batch_start:batch_end]
                
                for params in batch_params:
                    self.logger.info(f"Training model {models_count + 1}/{len(param_list)}")
                    self.logger.info(f"Parameters: {params}")
                    train_params = {k: v for k, v in params.items() if k not in ['n_estimators', 'early_stopping_rounds']}
                    try:
                        model = XGBoostModel(
                            params=train_params,
                            logger=self.logger,
                            num_boost_round=params['n_estimators'],
                            early_stopping_rounds=params['early_stopping_rounds']
                        )
                        
                        model.fit(
                            X_train,
                            y_train,
                            eval_set=[(X_val, y_val)]
                        )
                        
                        val_preds = model.predict_proba(X_val)[:, 1]
                        optimal_threshold, metrics = self._find_optimal_threshold(val_preds, y_val)
                        
                        if metrics['precision'] > self.best_precision:
                            self.logger.info(f"New best precision: {metrics['precision']:.4f}")
                            self.best_precision = metrics['precision']
                        if metrics['recall'] > self.best_recall:
                            self.logger.info(f"New best recall: {metrics['recall']:.4f}")
                            self.best_recall = metrics['recall']
                        
                        if metrics['recall'] >= self.TARGET_RECALL:
                            self.models.append(model)
                            val_preds_sum += val_preds
                            models_count += 1
                            
                            self.logger.info(
                                f"Model {models_count} metrics - "
                                f"Precision: {metrics['precision']:.4f}, "
                                f"Recall: {metrics['recall']:.4f}"
                            )
                        
                    except Exception as model_error:
                        self.logger.error(f"Error training individual model: {str(model_error)}")
                        continue
                    
                import gc
                gc.collect()
            
            if not self.models:
                self.logger.warning("No models met the recall threshold")
                
            self.logger.info(f"Successfully trained {len(self.models)} models")
            
        except Exception as e:
            self.logger.error(f"Error training models: {str(e)}")
            raise

    def _evaluate_predictions(
        self,
        probabilities: np.ndarray,
        actuals: pd.Series,
        threshold: Optional[float] = None) -> Dict[str, float]:
        """Evaluate binary classification metrics."""
        try:
            threshold = threshold or self.threshold
            predictions = (probabilities >= threshold).astype(int)
            
            metrics = {
                'precision': precision_score(actuals, predictions, pos_label=1, zero_division=0),
                'recall': recall_score(actuals, predictions, pos_label=1, zero_division=0)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating predictions: {str(e)}")
            raise

    def _find_optimal_threshold(
        self,
        probabilities: np.ndarray,
        actuals: pd.Series) -> Tuple[float, Dict[str, float]]:
        """Find optimal prediction threshold prioritizing precision while ensuring recall."""
        try:
            self.logger.info("Finding optimal threshold...")
            best_metrics = {
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'threshold': self.DEFAULT_THRESHOLD
            }
            best_score = 0
            
            for threshold in np.arange(0.2, 0.80, 0.01):
                metrics = self._evaluate_predictions(probabilities, actuals, threshold)
                
                if metrics['recall'] >= self.TARGET_RECALL:
                    score = metrics['precision']
                    if score > best_score:
                        best_score = score
                        best_metrics = {**metrics, 'threshold': threshold}
            
            self.logger.info(
                f"Optimal threshold {best_metrics['threshold']:.3f}: "
                f"Precision={best_metrics['precision']:.4f}, "
                f"Recall={best_metrics['recall']:.4f}"
            )
            return best_metrics['threshold'], best_metrics
            
        except Exception as e:
            self.logger.error(f"Error in threshold optimization: {str(e)}")
            raise

    def _save_models(self) -> None:
        """Save trained models to disk."""
        try:
            for i, model in enumerate(self.models):
                model_path = os.path.join(self.model_dir, f"xgboost_{i}.json")
                model.save(model_path)
                self.logger.info(f"Saved model to: {model_path}")
                
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")
            raise

    def _log_training_results(self, test_metrics: Dict[str, float]) -> None:
        """Log training results to MLflow."""
        try:
            mlflow.log_params({
                "max_runtime_secs": self.max_runtime_secs,
                "seed": self.seed,
                "n_models": len(self.models),
                "optimal_threshold": self.threshold
            })
            
            mlflow.log_metrics({
                f"test_{k}": v for k, v in test_metrics.items()
            })
            
            mlflow.log_artifacts(self.model_dir, "models")
            
        except Exception as e:
            self.logger.error(f"Error logging results: {str(e)}")
            raise

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series) -> Dict[str, Any]:
        """Train multiple XGBoost models with validation-based threshold optimization."""
        try:
            self._train_models(X_train, y_train, X_val, y_val)
            
            if not self.models:
                raise ValueError("No models met the recall threshold")
            
            test_predictions = np.zeros(len(y_test))
            for model in self.models:
                test_predictions += model.predict_proba(X_test)[:, 1]
            test_predictions /= len(self.models)
            
            test_metrics = self._evaluate_predictions(test_predictions, y_test, self.threshold)
            
            self._save_models()
            self._log_training_results(test_metrics)
            
            return {
                'models': self.models,
                'threshold': self.threshold,
                'test_metrics': test_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}")
            raise

def train_xgboost_pipeline():
    """Main function to train the XGBoost models."""
    try:
        trainer = XGBoostCPUTrainer(
            logger=logger,
            max_runtime_secs=3600,
            seed=42
        )
        
        selected_features = import_selected_features_ensemble('all')
        X_train, y_train, X_test, y_test = import_training_data_ensemble()
        X_val, y_val = create_ensemble_evaluation_set()
        
        X_train = X_train[selected_features]
        X_val = X_val[selected_features]
        X_test = X_test[selected_features]
        
        with mlflow.start_run(
            run_name=f"xgboost_cpu_{datetime.now().strftime('%Y%m%d_%H%M')}"
        ):
            mlflow.log_params({
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "test_samples": len(X_test),
                "draw_ratio_train": (y_train == 1).mean(),
                "draw_ratio_val": (y_val == 1).mean(),
                "draw_ratio_test": (y_test == 1).mean()
            })
            
            results = trainer.train(
                X_train, y_train,
                X_val, y_val,
                X_test, y_test
            )
            
            print("\nModel Performance Summary:")
            print("-" * 80)
            print("Test Set Metrics:")
            for k, v in results['test_metrics'].items():
                print(f"{k:20}: {v:.4f}")
            
            return results
            
    except Exception as e:
        logger.error(f"Error during XGBoost training: {str(e)}")
        raise

if __name__ == "__main__":
    results = train_xgboost_pipeline()
    print("\nFinal Model Performance:")
    print("-" * 80)
    print("Test Set:")
    print(f"Precision: {results['test_metrics']['precision']:.4f}")
    print(f"Recall: {results['test_metrics']['recall']:.4f}")