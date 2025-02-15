"""
H2O Stacked Ensemble implementation with multiple boosting models and hyperparameter tuning.
Focuses on maximizing precision while maintaining minimum recall threshold.
"""

# Standard library imports
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
from datetime import datetime
import time

# Third-party imports
import numpy as np
import pandas as pd
import polars as pl
import mlflow
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
import h2o
from h2o.automl import H2OAutoML
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.xgboost import H2OXGBoostEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
from sklearn.base import BaseEstimator, ClassifierMixin

import xgboost as xgb

# Filter specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.base")
warnings.filterwarnings("ignore", category=UserWarning, module="h2o")

# Add project root to Python path
try:
    project_root = Path(__file__).parent.parent.parent
    if not project_root.exists():
        # Handle network path by using raw string
        project_root = Path(r"\\".join(str(project_root).split("\\")))
    sys.path.append(str(project_root))
    print(f"Project root h2o_stacked_ensemble_hypertuning: {project_root}")
except Exception as e:
    print(f"Error setting project root path: {e}")
    sys.path.append(os.getcwd().parent.parent)
    print(f"Current directory h2o_stacked_ensemble_hypertuning: {os.getcwd().parent.parent}")

# Local imports

from utils.logger import ExperimentLogger
experiment_name = "h2o_stacked_ensemble_hypertuning"
logger = ExperimentLogger(experiment_name=experiment_name, log_dir='./logs/h2o_stacked_ensemble_hypertuning')

from utils.create_evaluation_set import (
    create_ensemble_evaluation_set,
    import_selected_features_ensemble,
    import_training_data_ensemble,
    setup_mlflow_tracking
)

mlruns_dir = setup_mlflow_tracking(experiment_name)

# Add import at the top after other imports
from models.hypertuning.model_interface import ModelFactory, H2OModelWrapper, XGBoostModelWrapper

class H2OStackedEnsembleTrainer(BaseEstimator, ClassifierMixin):
    """H2O Stacked Ensemble trainer with precision-focused tuning and MLflow integration.
    
    This trainer combines multiple H2O boosting models (GBM, XGBoost) into a stacked ensemble,
    with hyperparameter tuning focused on maximizing precision while maintaining a minimum
    recall threshold of 15%.
    
    Attributes:
        MIN_SAMPLES: Minimum number of samples required for training
        DEFAULT_THRESHOLD: Default prediction threshold
        TARGET_PRECISION: Target precision threshold
        TARGET_RECALL: Target recall threshold
        MAX_MODELS: Maximum number of base models to train
    """
    
    MIN_SAMPLES: int = 1000
    DEFAULT_THRESHOLD: float = 0.50
    TARGET_PRECISION: float = 0.5
    TARGET_RECALL: float = 0.15
    MAX_MODELS: int = 20

    def __init__(
        self,
        logger: Optional[ExperimentLogger] = None,
        max_runtime_secs: int = 3600,
        seed: int = 42,
        model_dir: Optional[str] = None
    ):
        """Initialize the H2O Stacked Ensemble trainer.
        
        Args:
            logger: Optional logger instance
            max_runtime_secs: Maximum runtime in seconds for each base model
            seed: Random seed for reproducibility
            model_dir: Optional directory to save trained models
        """
        self.logger = logger or ExperimentLogger(
            experiment_name="h2o_stacked_ensemble_hypertuning",
            log_dir='./logs/h2o_stacked_ensemble_hypertuning'
        )
        self.max_runtime_secs = max_runtime_secs
        self.seed = seed
        self.model_dir = model_dir or os.path.join(project_root, "models", "h2o_stacked_ensemble_hypertuning")
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize model containers using wrappers
        self.gbm_models: List[H2OModelWrapper] = []
        self.xgb_models: List[XGBoostModelWrapper] = []
        self.ensemble_model = None
        self.threshold = self.DEFAULT_THRESHOLD
        self._check_logger()

    def _check_logger(self) -> None:
        """Verify logger is properly initialized and functional."""
        if not self.logger:
            raise ValueError("Logger not initialized")
        if not hasattr(self.logger, 'info'):
            raise ValueError("Logger missing required 'info' method")
        if not hasattr(self.logger, 'error'):
            raise ValueError("Logger missing required 'error' method")
        
        try:
            self.logger.info(
                f"Logger configuration - Experiment: {self.logger.experiment_name}, "
                f"Log directory: {self.logger.log_dir}"
            )
            self.logger.info("Logger check: Initialization successful")
        except Exception as e:
            raise ValueError(f"Logger test failed: {str(e)}")

    def _init_h2o(self) -> None:
        """Initialize H2O cluster with CPU-only configuration."""
        try:
            h2o.init(
                nthreads=8,  # Use 8 CPU cores
                max_mem_size="12G",  # 12GB memory limit
                name=f"h2o_stacked_{datetime.now().strftime('%Y%m%d_%H%M')}",
                port=54321 + np.random.randint(100),  # Random port to avoid conflicts
                enable_assertions=True
            )
            self.logger.info("H2O cluster initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize H2O cluster: {str(e)}")
            raise

    def _convert_to_h2o_frame(
        self,
        features: pd.DataFrame,
        target: Optional[pd.Series] = None,
        train: bool = True) -> h2o.H2OFrame:
        """Convert pandas DataFrame to H2OFrame with proper type handling.
        
        Args:
            features: Feature DataFrame
            target: Optional target Series
            train: Whether this is training data
            
        Returns:
            H2OFrame with features and optionally target
        """
        try:
            if target is not None:
                data = pd.concat([features, target.rename('is_draw')], axis=1)
            else:
                data = features.copy()
                
            h2o_frame = h2o.H2OFrame(data)
            
            if target is not None:
                h2o_frame['is_draw'] = h2o_frame['is_draw'].asfactor()  # Convert target to factor
                
            self.logger.info(f"Converted DataFrame to H2OFrame with shape: {h2o_frame.shape}")
            return h2o_frame
            
        except Exception as e:
            self.logger.error(f"Error converting to H2OFrame: {str(e)}")
            raise

    def _validate_data(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series) -> None:
        """Validate input data meets requirements.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Raises:
            ValueError: If data validation fails
        """
        try:
            # Check minimum samples
            if len(X_train) < self.MIN_SAMPLES:
                raise ValueError(f"Training set must have at least {self.MIN_SAMPLES} samples")
                
            # Check for missing values
            # Drop missing values from training data
            X_train = X_train.dropna()
            y_train = y_train[X_train.index]  # Align labels with remaining samples
            
            # Drop missing values from validation data
            X_val = X_val.dropna()
            y_val = y_val[X_val.index]  # Align labels with remaining samples
            
            # Log the number of dropped samples
            self.logger.info(f"Dropped {len(X_train) - len(X_train.dropna())} training samples with missing values")
            self.logger.info(f"Dropped {len(X_val) - len(X_val.dropna())} validation samples with missing values")
                
            # Check feature consistency
            if not all(col in X_val.columns for col in X_train.columns):
                raise ValueError("Validation features don't match training features")
                
            # Check target distribution
            train_pos = y_train.mean()
            val_pos = y_val.mean()
            self.logger.info(f"Target distribution - Train: {train_pos:.4f}, Val: {val_pos:.4f}")
            
            if train_pos == 0 or val_pos == 0:
                raise ValueError("Both training and validation sets must contain positive samples")
                
        except Exception as e:
            self.logger.error(f"Data validation failed: {str(e)}")
            raise

    def _setup_base_models(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Define hyperparameter search spaces for GBM models.
        
        Returns:
            Tuple of (gbm_params, empty_dict) dictionaries containing hyperparameter grids
        """
        # GBM hyperparameter grid with expanded parameter ranges
        gbm_params = {
            "max_depth": [2, 3, 4, 5],
            "learn_rate": [0.01, 0.0124, 0.015, 0.02],
            "sample_rate": [0.25, 0.30, 0.35, 0.40],
            "col_sample_rate": [0.65, 0.68, 0.71, 0.75],
            "min_rows": [5, 10, 15],
            "ntrees": [500, 579, 650],
            "stopping_rounds": [300],
            "stopping_metric": ["AUC"],
            "stopping_tolerance": [0.001],
            "seed": [self.seed]
        }
        
        return gbm_params, {}  # Return empty dict for xgb_params

    def _train_base_models(
        self,
        train_frame: h2o.H2OFrame,
        valid_frame: h2o.H2OFrame,
        predictors: List[str],
        response: str) -> None:
        """Train GBM base models using grid search.
        
        Args:
            train_frame: Training data as H2OFrame
            valid_frame: Validation data as H2OFrame
            predictors: List of predictor column names
            response: Name of response column
        """
        try:
            gbm_params, _ = self._setup_base_models()
            
            # Train GBM models using ModelFactory
            self.logger.info("Training GBM models...")
            gbm_grid = H2OGridSearch(
                model=H2OGradientBoostingEstimator(
                    score_each_iteration=False,
                    seed=self.seed
                ),
                hyper_params=gbm_params,
                search_criteria={
                    "strategy": "RandomDiscrete",
                    "max_models": self.MAX_MODELS,
                    "seed": self.seed
                }
            )
            
            gbm_grid.train(
                x=predictors,
                y=response,
                training_frame=train_frame,
                validation_frame=valid_frame,
                keep_cross_validation_predictions=True
            )
            
            # Convert grid models to wrappers
            for model in self._select_top_models(gbm_grid, valid_frame):
                wrapper = ModelFactory.create_model('h2o', {'model': model}, self.logger)
                self.gbm_models.append(wrapper)
            
            # If no GBM models meet the recall threshold, try XGBoost CPU pipeline
            if not self.gbm_models:
                self.logger.info("No GBM models met recall threshold, trying XGBoost CPU pipeline...")
                
                # Convert H2O frames to pandas
                train_df = h2o.as_list(train_frame, use_pandas=True)
                valid_df = h2o.as_list(valid_frame, use_pandas=True)
                
                # Separate features and target
                X_train = train_df[predictors]
                y_train = train_df[response]
                X_val = valid_df[predictors]
                y_val = valid_df[response]
                
                # Train XGBoost models using ModelFactory
                xgb_params = {
                    'tree_method': 'hist',
                    'objective': 'binary:logistic',
                    'eval_metric': ['auc', 'error'],
                    'seed': self.seed,
                    'n_estimators': 1000,
                    'early_stopping_rounds': 50
                }
                
                for _ in range(self.MAX_MODELS):
                    wrapper = ModelFactory.create_model('xgboost', {'params': xgb_params}, self.logger)
                    wrapper.fit(X_train, y_train, eval_set=[(X_val, y_val)])
                    
                    # Evaluate model
                    val_preds = wrapper.predict_proba(X_val)[:, 1]
                    metrics = self._evaluate_predictions(
                        pd.DataFrame({'p1': val_preds}),
                        valid_df[response]
                    )
                    
                    if metrics['recall'] >= self.TARGET_RECALL:
                        self.xgb_models.append(wrapper)
                        self.logger.info(
                            f"XGBoost model metrics - "
                            f"Precision: {metrics['precision']:.4f}, "
                            f"Recall: {metrics['recall']:.4f}"
                        )
            
            total_models = len(self.gbm_models) + len(self.xgb_models)
            if total_models == 0:
                raise ValueError("No models (GBM or XGBoost) met the recall threshold")
                
            self.logger.info(f"Selected {len(self.gbm_models)} GBM models and {len(self.xgb_models)} XGBoost models")
            
        except Exception as e:
            self.logger.error(f"Error training base models: {str(e)}")
            raise

    def _select_top_models(
        self,
        grid: H2OGridSearch,
        valid_frame: h2o.H2OFrame,
        top_n: int = 5) -> List[Union[H2OGradientBoostingEstimator, H2OXGBoostEstimator]]:
        """Select top performing models from grid search that meet recall threshold.
        
        Args:
            grid: Trained H2OGridSearch object
            valid_frame: Validation data as H2OFrame
            top_n: Number of top models to select
            
        Returns:
            List of selected models
        """
        try:
            selected_models = []
            grid_models = grid.get_grid(sort_by='auc', decreasing=True)
            
            for model in grid_models:
                # Get predictions and evaluate metrics
                preds = model.predict(valid_frame)
                metrics = self._evaluate_predictions(preds, valid_frame['is_draw'])
                
                # Only select models meeting recall threshold
                if metrics['recall'] >= self.TARGET_RECALL:
                    selected_models.append(model)
                    if len(selected_models) >= top_n:
                        break
                        
            return selected_models
            
        except Exception as e:
            self.logger.error(f"Error selecting top models: {str(e)}")
            raise

    def _find_optimal_threshold(
        self,
        model: Union[H2OGradientBoostingEstimator, H2OXGBoostEstimator, H2OStackedEnsembleEstimator],
        valid_frame: h2o.H2OFrame) -> Tuple[float, Dict[str, float]]:
        """Find optimal prediction threshold prioritizing precision while maintaining recall.
        
        Args:
            model: Trained H2O model
            valid_frame: Validation data as H2OFrame
            
        Returns:
            Tuple of (optimal threshold, metrics dictionary)
        """
        try:
            self.logger.info("Finding optimal threshold...")
            
            # Get predictions
            predictions = model.predict(valid_frame)
            preds_df = h2o.as_list(predictions, use_pandas=True)
            probs = preds_df['p1'].values
            
            actuals = h2o.as_list(valid_frame['is_draw'], use_pandas=True).values
            
            best_metrics = {
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'threshold': self.DEFAULT_THRESHOLD
            }
            best_score = 0
            
            # Search for optimal threshold
            for threshold in np.arange(0.4, 0.80, 0.01):
                preds = (probs >= threshold).astype(int)
                precision = precision_score(actuals, preds, pos_label=1, zero_division=0)
                recall = recall_score(actuals, preds, pos_label=1, zero_division=0)
                
                if recall >= self.TARGET_RECALL:
                    f1 = f1_score(actuals, preds, pos_label=1)
                    # Use precision as the score since we want to maximize it
                    score = precision
                    
                    if score > best_score:
                        best_score = score
                        best_metrics.update({
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'threshold': threshold
                        })
            
            self.logger.info(
                f"Optimal threshold {best_metrics['threshold']:.3f}: "
                f"Precision={best_metrics['precision']:.4f}, "
                f"Recall={best_metrics['recall']:.4f}"
            )
            return best_metrics['threshold'], best_metrics
            
        except Exception as e:
            self.logger.error(f"Error in threshold optimization: {str(e)}")
            raise

    def _evaluate_predictions(
        self,
        predictions: h2o.H2OFrame,
        actuals: h2o.H2OFrame,
        threshold: Optional[float] = None) -> Dict[str, float]:
        """Evaluate binary classification metrics for predictions.
        
        Args:
            predictions: Model predictions as H2OFrame
            actuals: Actual values as H2OFrame
            threshold: Optional classification threshold
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            preds_df = h2o.as_list(predictions, use_pandas=True)
            actuals_df = h2o.as_list(actuals, use_pandas=True)
            
            probs = preds_df['p1'].values
            actuals_arr = actuals_df.values
            
            threshold = threshold or self.threshold
            preds = (probs >= threshold).astype(int)
            
            metrics = {
                'precision': precision_score(actuals_arr, preds, pos_label=1, zero_division=0),
                'recall': recall_score(actuals_arr, preds, pos_label=1, zero_division=0),
                'f1': f1_score(actuals_arr, preds, pos_label=1)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating predictions: {str(e)}")
            raise

    def _build_stacked_ensemble(
        self,
        train_frame: h2o.H2OFrame,
        valid_frame: h2o.H2OFrame,
        predictors: List[str],
        response: str) -> None:
        """Build and train stacked ensemble from selected base models.
        
        Args:
            train_frame: Training data as H2OFrame
            valid_frame: Validation data as H2OFrame
            predictors: List of predictor column names
            response: Name of response column
        """
        try:
            total_models = len(self.gbm_models) + len(self.xgb_models)
            if total_models == 0:
                raise ValueError("No base models available for ensemble")
            
            # Get predictions from all models
            train_preds = []
            valid_preds = []
            
            # Get predictions from GBM models
            for model in self.gbm_models:
                train_preds.append(model.predict_proba(h2o.as_list(train_frame[predictors], use_pandas=True))[:, 1])
                valid_preds.append(model.predict_proba(h2o.as_list(valid_frame[predictors], use_pandas=True))[:, 1])
            
            # Get predictions from XGBoost models
            for model in self.xgb_models:
                train_preds.append(model.predict_proba(h2o.as_list(train_frame[predictors], use_pandas=True))[:, 1])
                valid_preds.append(model.predict_proba(h2o.as_list(valid_frame[predictors], use_pandas=True))[:, 1])
            
            # Convert predictions to H2O frames
            train_preds_h2o = h2o.H2OFrame(
                pd.DataFrame(np.column_stack(train_preds), columns=[f'pred_{i}' for i in range(total_models)])
            )
            valid_preds_h2o = h2o.H2OFrame(
                pd.DataFrame(np.column_stack(valid_preds), columns=[f'pred_{i}' for i in range(total_models)])
            )
            
            # Configure metalearner
            metalearner = H2OGeneralizedLinearEstimator(
                family="binomial",
                solver="coordinate_descent",
                alpha=[0.5],
                lambda_search=True,
                nlambdas=15,
                standardize=True,
                seed=self.seed
            )
            
            # Initialize and train stacked ensemble
            self.logger.info("Training stacked ensemble...")
            self.ensemble_model = H2OStackedEnsembleEstimator(
                base_models=self.gbm_models + [model._model for model in self.xgb_models],
                metalearner_algorithm="glm",
                metalearner_params=metalearner._parms,
                seed=self.seed
            )
            
            self.ensemble_model.train(
                x=list(train_preds_h2o.columns),
                y=response,
                training_frame=h2o.H2OFrame(pd.concat([
                    h2o.as_list(train_preds_h2o, use_pandas=True),
                    h2o.as_list(train_frame[response], use_pandas=True)
                ], axis=1)),
                validation_frame=h2o.H2OFrame(pd.concat([
                    h2o.as_list(valid_preds_h2o, use_pandas=True),
                    h2o.as_list(valid_frame[response], use_pandas=True)
                ], axis=1))
            )
            
            # Find optimal threshold for ensemble
            self.threshold, ensemble_metrics = self._find_optimal_threshold(
                self.ensemble_model,
                valid_frame
            )
            
            self.logger.info(
                f"Stacked ensemble performance - "
                f"Precision: {ensemble_metrics['precision']:.4f}, "
                f"Recall: {ensemble_metrics['recall']:.4f}, "
                f"F1: {ensemble_metrics['f1']:.4f}"
            )
            
        except Exception as e:
            self.logger.error(f"Error building stacked ensemble: {str(e)}")
            raise

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series) -> Dict[str, Any]:
        """Train stacked ensemble model with validation-based threshold optimization.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary containing ensemble model and performance metrics
        """
        try:
            # Validate input data
            self._validate_data(X_train, y_train, X_val, y_val)
            
            # Initialize H2O
            self._init_h2o()
            
            # Convert data to H2O frames
            train_frame = self._convert_to_h2o_frame(X_train, y_train)
            valid_frame = self._convert_to_h2o_frame(X_val, y_val)
            test_frame = self._convert_to_h2o_frame(X_test, y_test)
            
            # Get predictor and response names
            predictors = list(X_train.columns)
            response = 'is_draw'
            
            # Train base models
            self._train_base_models(train_frame, valid_frame, predictors, response)
            
            # Build and train stacked ensemble
            self._build_stacked_ensemble(train_frame, valid_frame, predictors, response)
            
            # Evaluate on test set
            test_metrics = self._evaluate_predictions(
                self.ensemble_model.predict(test_frame),
                test_frame['is_draw'],
                self.threshold
            )
            
            # Save models
            self._save_models()
            
            # Log results
            self._log_training_results(test_metrics)
            
            return {
                'ensemble_model': self.ensemble_model,
                'base_models': {
                    'gbm': self.gbm_models,
                    'xgb': self.xgb_models
                },
                'threshold': self.threshold,
                'test_metrics': test_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}")
            raise
        finally:
            h2o.cluster().shutdown()

    def _save_models(self) -> None:
        """Save trained models to disk."""
        try:
            # Save GBM models
            for i, model in enumerate(self.gbm_models):
                model_path = os.path.join(self.model_dir, f"gbm_{i}")
                model.save(model_path)
                self.logger.info(f"Saved GBM model to: {model_path}")
            
            # Save XGBoost models
            for i, model in enumerate(self.xgb_models):
                model_path = os.path.join(self.model_dir, f"xgb_{i}.json")
                model.save(model_path)
                self.logger.info(f"Saved XGBoost model to: {model_path}")
            
            # Save ensemble model
            if self.ensemble_model is not None:
                ensemble_path = os.path.join(self.model_dir, "stacked_ensemble")
                h2o.save_model(self.ensemble_model, path=ensemble_path, force=True)
                self.logger.info(f"Saved stacked ensemble to: {ensemble_path}")
                
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")
            raise

    def _log_training_results(self, test_metrics: Dict[str, float]) -> None:
        """Log training results to MLflow."""
        try:
            # Log hyperparameters
            mlflow.log_params({
                "max_runtime_secs": self.max_runtime_secs,
                "seed": self.seed,
                "n_gbm_models": len(self.gbm_models),
                "n_xgb_models": len(self.xgb_models),
                "optimal_threshold": self.threshold
            })
            
            # Log metrics
            mlflow.log_metrics({
                f"test_{k}": v for k, v in test_metrics.items()
            })
            
            # Log model files
            mlflow.log_artifacts(self.model_dir, "models")
            
        except Exception as e:
            self.logger.error(f"Error logging results: {str(e)}")
            raise

def train_stacked_ensemble():
    """Main function to train the stacked ensemble model."""
    try:
        # Initialize trainer
        trainer = H2OStackedEnsembleTrainer(
            logger=logger,
            max_runtime_secs=3600,
            seed=42
        )
        
        # Load data
        selected_features = import_selected_features_ensemble('all')
        X_train, y_train, X_test, y_test = import_training_data_ensemble()
        X_val, y_val = create_ensemble_evaluation_set()
        
        # Select features
        X_train = X_train[selected_features]
        X_val = X_val[selected_features]
        X_test = X_test[selected_features]
        
        # Start MLflow run
        with mlflow.start_run(
            run_name=f"h2o_stacked_ensemble_{datetime.now().strftime('%Y%m%d_%H%M')}"
        ):
            # Log data statistics
            mlflow.log_params({
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "test_samples": len(X_test),
                "draw_ratio_train": (y_train == 1).mean(),
                "draw_ratio_val": (y_val == 1).mean(),
                "draw_ratio_test": (y_test == 1).mean()
            })
            
            # Train model
            results = trainer.train(
                X_train, y_train,
                X_val, y_val,
                X_test, y_test
            )
            
            # Print performance summary
            print("\nModel Performance Summary:")
            print("-" * 80)
            print("Test Set Metrics:")
            for k, v in results['test_metrics'].items():
                if isinstance(v, (int, float)):
                    print(f"{k:20}: {v:.4f}")
            
            return results
            
    except Exception as e:
        logger.error(f"Error during stacked ensemble training: {str(e)}")
        raise

if __name__ == "__main__":
    results = train_stacked_ensemble()
    print("\nFinal Model Performance:")
    print("-" * 80)
    print("Test Set:")
    print(f"Precision: {results['test_metrics']['precision']:.4f}")
    print(f"Recall: {results['test_metrics']['recall']:.4f}")
    print(f"F1 Score: {results['test_metrics']['f1']:.4f}") 