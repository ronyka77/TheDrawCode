"""
H2O AutoML model implementation for binary classification with global training capabilities.
Provides methods for model training, prediction, and analysis with configurable parameters.
"""

# Standard library imports
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import time

# Third-party imports
import numpy as np
import pandas as pd
import polars as pl
import mlflow
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.base import BaseEstimator, ClassifierMixin
import h2o
from h2o.automl import H2OAutoML
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.xgboost import H2OXGBoostEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator

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
    print(f"Project root h2o_ensemble_hypertuning: {project_root}")
except Exception as e:
    print(f"Error setting project root path: {e}")
    sys.path.append(os.getcwd().parent.parent)
    print(f"Current directory h2o_ensemble_hypertuning: {os.getcwd().parent.parent}")

# Local imports
from utils.logger import ExperimentLogger
experiment_name = "automl_ensemble_hypertuning"
logger = ExperimentLogger(experiment_name=experiment_name, log_dir='./logs/automl_ensemble_hypertuning')

from utils.create_evaluation_set import (
    create_ensemble_evaluation_set,
    import_selected_features_ensemble,
    import_training_data_ensemble,
    setup_mlflow_tracking
)

mlruns_dir = setup_mlflow_tracking(experiment_name)

class H2OAutoMLTrainer(BaseEstimator, ClassifierMixin):
    """H2O AutoML trainer with MLflow integration and CPU-only training.
    
    Attributes:
        MIN_SAMPLES: Minimum number of samples required for training
        DEFAULT_THRESHOLD: Default prediction threshold
        TARGET_PRECISION: Target precision threshold
        TARGET_RECALL: Target recall threshold
    """
    
    MIN_SAMPLES: int = 1000
    DEFAULT_THRESHOLD: float = 0.50
    TARGET_PRECISION: float = 0.5
    TARGET_RECALL: float = 0.15

    def __init__(
        self,
        logger: Optional[ExperimentLogger] = None,
        max_runtime_secs: int = 3600,
        max_models: int = 20,
        seed: int = 42):
        """Initialize the H2O AutoML trainer.
        
        Args:
            logger: Optional logger instance
            max_runtime_secs: Maximum runtime in seconds
            max_models: Maximum number of models to train
            seed: Random seed for reproducibility
        """
        self.logger = logger or ExperimentLogger(
            experiment_name="automl_ensemble_hypertuning",
            log_dir='./logs/automl_ensemble_hypertuning'
        )
        self.max_runtime_secs = max_runtime_secs
        self.max_models = max_models
        self.seed = seed
        self.best_model = None
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
                nthreads=8,  # Use all CPU cores
                max_mem_size="12G",  # Adjust based on your system
                name=f"h2o_automl_{datetime.now().strftime('%Y%m%d_%H%M')}",
                port=54321 + np.random.randint(100), enable_assertions=True  # Random port to avoid conflicts
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
        """Convert pandas DataFrame to H2OFrame.
        
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

    def _find_optimal_threshold(
        self,
        model: h2o.automl.H2OAutoML,
        validation_frame: h2o.H2OFrame) -> Tuple[float, Dict[str, float]]:
        """Find optimal prediction threshold prioritizing precision.
        
        Args:
            model: Trained H2O model
            validation_frame: Validation data as H2OFrame
            
        Returns:
            Tuple of (optimal threshold, metrics dictionary)
        """
        try:
            self.logger.info("Finding optimal threshold...")
            # Get predictions and convert to polars
            predictions = model.predict(validation_frame)
            preds_pol = h2o_to_polars(predictions)
            # Extract p1 column using polars
            probs = preds_pol["p1"].to_numpy()
            # Convert validation data to polars and extract target
            valid_pol = h2o_to_polars(validation_frame)
            actuals = valid_pol["is_draw"].to_numpy()
            best_metrics = {
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'threshold': self.DEFAULT_THRESHOLD
            }
            best_score = 0
            
            # Search for optimal threshold
            for threshold in np.arange(0.4, 0.80, 0.01):
                self.logger.info(f"Evaluating threshold: {threshold}")
                preds = (probs >= threshold).astype(int)
                precision = precision_score(actuals, preds, zero_division=0)
                recall = recall_score(actuals, preds, zero_division=0)
                self.logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
                if recall >= 0.15:  # Minimum recall requirement
                    f1 = f1_score(actuals, preds)
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

    def _find_optimal_threshold_builtin(
        self,
        model: h2o.automl.H2OAutoML,
        validation_frame: h2o.H2OFrame) -> Tuple[float, Dict[str, float]]:
        """
        Find the optimal threshold using H2O's built-in metrics from thresholds_and_metric_scores().
        
        Args:
            model: Trained H2O model.
            validation_frame: Validation data as H2OFrame.
        
        Returns:
            Tuple of (optimal threshold, metrics dictionary).
        """
        try:
            self.logger.info("Finding optimal threshold via built-in H2O metrics...")
            # Get the performance object for the validation set
            perf = model.model_performance(validation_frame)
            # Get thresholds and corresponding metric scores as an H2OFrame
            metric_table = perf.thresholds_and_metric_scores()
            # Convert to Pandas for easier processing
            metric_df = metric_table.as_data_frame()
            
            # self.logger.info("Thresholds and Metric Scores from built-in H2O function:")
            # self.logger.info(metric_df.to_string())
            # Select rows where recall meets the minimum requirement (>= 15%)
            valid_metrics = metric_df[metric_df['recall'] >= 0.15]
            if valid_metrics.empty:
                self.logger.warning("No threshold meets the recall requirement; using default threshold.")
                fallback_f1 = perf.F1() if perf.F1() is not None else 0.0
                return self.DEFAULT_THRESHOLD, {
                    'precision': perf.precision(),
                    'recall': perf.recall(),
                    'f1': fallback_f1,
                    'threshold': self.DEFAULT_THRESHOLD
                }
            
            # Choose the row with the highest precision among those meeting the recall constraint
            best_row = valid_metrics.loc[valid_metrics['precision'].idxmax()]
            best_threshold = best_row['threshold']
            precision_val = best_row['precision']
            recall_val = best_row['recall']
            # Compute F1 score if missing from the leaderboard row
            if 'F1' in best_row and pd.notnull(best_row['F1']):
                computed_f1 = best_row['F1']
            else:
                computed_f1 = (2 * precision_val * recall_val) / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0.0
            
            best_metrics = {
                'precision': precision_val,
                'recall': recall_val,
                'f1': computed_f1,
                'threshold': best_threshold
            }
            
            self.logger.info(
                f"Optimal threshold (built-in): {best_threshold:.3f} - Precision: {best_metrics['precision']:.4f}, "
                f"Recall: {best_metrics['recall']:.4f}"
            )
            
            return best_threshold, best_metrics
        except Exception as e:
            self.logger.error(f"Exception during built-in threshold analysis: {str(e)}")
            raise

    def train(
        self,
        features_train: pd.DataFrame,
        target_train: pd.Series,
        features_val: pd.DataFrame,
        target_val: pd.Series,
        features_test: pd.DataFrame,
        target_test: pd.Series) -> Dict[str, Any]:
        """Train H2O AutoML model with validation-based threshold optimization.
        
        Args:
            features_train: Training features
            target_train: Training labels
            features_val: Validation features
            target_val: Validation labels
            features_test: Test features
            target_test: Test labels
            
        Returns:
            Dictionary containing best model and performance metrics
        """
        try:
            self.logger.info("Initializing H2O cluster...")
            self._init_h2o()
            
            # Convert data to H2O frames
            train_frame = self._convert_to_h2o_frame(features_train, target_train)
            valid_frame = self._convert_to_h2o_frame(features_val, target_val)
            test_frame = self._convert_to_h2o_frame(features_test, target_test)
            
            # Initialize and train AutoML
            aml = H2OAutoML(
                max_runtime_secs=self.max_runtime_secs,
                max_models=self.max_models,
                seed=self.seed,
                sort_metric='AUC',
                stopping_metric='AUC',
                balance_classes=True,
                nfolds=5,
                keep_cross_validation_predictions=True,
                keep_cross_validation_models=True,
                verbosity='info',
                include_algos=['DRF', 'GLM', 'GBM', 'StackedEnsemble', 'DeepLearning']
            )
            
            self.logger.info("Starting H2O AutoML training...")
            aml.train(
                x=train_frame.columns[:-1],  # All columns except target
                y='is_draw',
                training_frame=train_frame,
                validation_frame=test_frame,
                leaderboard_frame=valid_frame
            )
            # Save models to experiment folder
            model_dir = os.path.join(project_root, "models", "automl_ensemble_hypertuning")
            os.makedirs(model_dir, exist_ok=True)
            
            # Save leader model
            leader_model = aml.leader
            leader_model_path = h2o.save_model(
                model=leader_model,
                path=model_dir,
                force=True
            )
            self.logger.info(f"Saved leader model to: {leader_model_path}")
            # Save all models in leaderboard with proper type handling
            leaderboard_df = aml.leaderboard.as_data_frame()
            for model_row in leaderboard_df.itertuples(index=False):
                try:
                    model_id = str(model_row.model_id)  # Ensure model_id is string
                    if not model_id:  # Skip empty model IDs
                        continue
                        
                    # Get model and save with explicit type conversion
                    model = h2o.get_model(model_id)
                    model_path = h2o.save_model(
                        model=model,
                        path=model_dir,
                        force=True
                    )
                    self.logger.info(f"Successfully saved model {model_id} to: {model_path}")
                    
                    # Log model metrics for tracking
                    self.logger.info(f"Model {model_id} metrics: AUC={model.auc():.4f}, LogLoss={model.logloss():.4f}")
                    
                except Exception as e:
                    self.logger.warning(f"Error saving model {model_id}: {str(e)}")
                    self.logger.debug(f"Model details: {model_row}")
                    continue
            
            # Get best model and find optimal threshold
            self.best_model = aml.leader
            if self.best_model is not None:
                self.threshold, val_metrics = self._find_optimal_threshold_builtin(
                    self.best_model,
                    valid_frame
                )
            else:
                raise ValueError("No best model found in AutoML leaderboard")
            
            # Evaluate on test set
            self.logger.info("Evaluating model on test set...")
            if valid_frame is not None:
                test_metrics = self._evaluate_model(valid_frame)
            else:
                raise ValueError("Validation frame is None")
            
            # Log metrics and model info
            self.logger.info("Logging training results...")
            if aml is not None and valid_frame is not None:
                self._log_training_results(leaderboard_df, valid_frame, target_val, val_metrics, test_metrics)
            else:
                raise ValueError("AutoML object or validation frame is None")
            
            return {
                'best_model': self.best_model,
                'validation_metrics': val_metrics,
                'test_metrics': test_metrics,
                'threshold': self.threshold
            }
            
        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}")
            raise
        finally:
            h2o.cluster().shutdown()

    def _evaluate_model(self, test_frame: h2o.H2OFrame) -> Dict[str, float]:
        """Evaluate model performance on test data.
        
        Args:
            test_frame: Test data as H2OFrame
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.best_model.predict(test_frame)
        preds_pol = h2o_to_polars(predictions)
        probs = preds_pol.get_column("p1").to_numpy()

        test_pol = h2o_to_polars(test_frame)
        actuals = test_pol.get_column("is_draw").to_numpy()

        preds = (probs >= self.threshold).astype(int)
        
        metrics = {
            'precision': precision_score(actuals, preds, zero_division=0),
            'recall': recall_score(actuals, preds, zero_division=0),
            'f1': f1_score(actuals, preds)
        }
        
        return metrics

    def _log_training_results(
        self,
        leaderboard_df: pd.DataFrame,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        val_metrics: Dict[str, float],
        test_metrics: Dict[str, float]) -> None:
        """Log training results to MLflow.
        
        Args:
            aml: Trained AutoML object
            val_metrics: Validation metrics
            test_metrics: Test metrics
        """
        # Log validation metrics
        mlflow.log_metrics({
            f"val_{k}": v for k, v in val_metrics.items()
        })
        
        # Log test metrics
        mlflow.log_metrics({
            f"test_{k}": v for k, v in test_metrics.items()
        })
        
        # Log AutoML metadata
        mlflow.log_params({
            "max_runtime_secs": self.max_runtime_secs,
            "max_models": self.max_models,
            "seed": self.seed,
            # "n_models_trained": len(aml.leaderboard),
            # "best_model_type": aml.leader.model_id,
            "optimal_threshold": self.threshold
        })
        
        
        # Calculate recall for each model in leaderboard
        recall_scores = []
        precision_scores = []
        leaderboard_data = leaderboard_df 
        for model_id in leaderboard_data['model_id']:
            try:
                model = h2o.get_model(model_id)
                preds = model.predict(h2o.H2OFrame(X_val))
                recall = recall_score(
                    y_val, 
                    (preds['p1'].as_data_frame() >= self.threshold).astype(int),
                    zero_division=0
                )
                precision = precision_score(
                    y_val, 
                    (preds['p1'].as_data_frame() >= self.threshold).astype(int),
                    zero_division=0
                )
                recall_scores.append(recall)
                precision_scores.append(precision)
            except Exception as e:
                recall_scores.append(0.0)  # Use 0 as fallback value
                precision_scores.append(0.0)  # Use 0 as fallback value
        # Add recall column to leaderboard
        leaderboard_df['recall'] = recall_scores
        leaderboard_df['precision'] = precision_scores
        
        # Save and log leaderboard
        leaderboard_path = os.path.join(mlruns_dir, "leaderboard.csv")
        leaderboard_df.to_csv(os.path.join(project_root, "leaderboard.csv"), index=False)
        mlflow.log_artifact(leaderboard_path, "leaderboard.csv")

    def grid_search_tuning(self, X_train, y_train, X_test, y_test, X_val, y_val):
        """
        Perform grid search tuning using H2OGridSearch.
        """
        # Reinitialize the H2O cluster unconditionally to ensure an active connection.
        try:
            self.logger.info("Reinitializing H2O cluster for grid search tuning...")
            h2o.init(nthreads=8, max_mem_size="12G", enable_assertions=True,
                    name=f"h2o_automl_{datetime.now().strftime('%Y%m%d_%H%M')}",
                    port=54321 + np.random.randint(100))
        except Exception as e:
            self.logger.error(f"Failed to initialize H2O cluster for grid search tuning: {e}")
            raise

        # Convert input data to H2OFrames
        train_frame = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
        valid_frame = h2o.H2OFrame(pd.concat([X_val, y_val], axis=1))
        test_frame = h2o.H2OFrame(pd.concat([X_test, y_test], axis=1))
        
        # IMPORTANT: Convert the target column to a factor for classification
        train_frame["is_draw"] = train_frame["is_draw"].asfactor()
        valid_frame["is_draw"] = valid_frame["is_draw"].asfactor()
        test_frame["is_draw"] = test_frame["is_draw"].asfactor()    

        # Get predictor and response column names
        predictors = X_train.columns.tolist()
        response = 'is_draw'
        
        # Expanded hyperparameter grid for improved precision tuning
        hyper_params = {
            "max_depth": [3, 5, 7, 9],
            "learn_rate": [0.01, 0.05, 0.1],
            "ntrees": [50, 100, 200],
            "min_rows": [1, 5, 10, 15],
            "sample_rate": [0.7, 0.8, 0.9],
            "col_sample_rate": [0.6, 0.8, 1.0]
        }
        
        # Improved search criteria focused on AUC
        search_criteria = {
            "strategy": "RandomDiscrete",
            "max_models": 30,
            "seed": 42,
            "stopping_metric": "AUC",  # Now valid since the task is classification
            "stopping_tolerance": 0.005,
            "stopping_rounds": 5
        }
        
        # Set up a base estimator for grid search focused on AUC
        gbm_model = H2OGradientBoostingEstimator(
            stopping_metric="AUC",
            stopping_rounds=5,
            seed=42,
            balance_classes=True,
            score_tree_interval=10,           # Score every 10 trees for more granular AUC monitoring
            min_split_improvement=1e-4,         # Only allow splits if they improve the model by a small threshold
            nfolds=3,
            keep_cross_validation_predictions=True,
            keep_cross_validation_models=True
        )

        # Initialize H2O Deep Learning estimator with precision-focused configuration
        deep_learning_model = H2ODeepLearningEstimator(
            hidden=[64, 32],  # Deeper architecture with more capacity
            epochs=100,       # Increased training epochs
            activation='RectifierWithDropout',  # Add dropout for regularization
            stopping_metric='aucpr',  # Focus directly on precision
            stopping_rounds=10,      # More patience for precision improvement
            stopping_tolerance=0.0005,  # Tighter precision threshold
            nfolds=5,                # Increased cross-validation folds
            seed=42,                 # Random seed for reproducibility
            balance_classes=True,    # Balance class weights for imbalanced data
            variable_importances=True,  # Calculate feature importance
            export_weights_and_biases=True,  # Export model weights
            keep_cross_validation_predictions=True,
            keep_cross_validation_models=True,
            input_dropout_ratio=0.1,  # Add input layer dropout
            hidden_dropout_ratios=[0.2, 0.1],  # Layer-specific dropout
            l1=1e-4,  # L1 regularization
            l2=1e-4,  # L2 regularization
            max_w2=10.0  # Constrain weights to prevent overfitting
        )
        # Initialize Grid Search with the expanded hyperparameter grid and improved search criteria
        grid_gbm = H2OGridSearch(
            model=gbm_model,
            hyper_params=hyper_params,
            search_criteria=search_criteria
        )
        grid_deep_learning = H2OGridSearch(
            model=deep_learning_model,
            hyper_params=hyper_params,
            search_criteria=search_criteria
        )
        # Start grid search training asynchronously
        grid_gbm.train(x=predictors, y=response, training_frame=train_frame, validation_frame=valid_frame)
        
        # Once complete, get the sorted leaderboard as usual
        grid_leaderboard_gbm = grid_gbm.get_grid(sort_by="precision", decreasing=True)
        # self.logger.info(grid_leaderboard)
        self._log_grid_search_results(grid_leaderboard_gbm, X_val, y_val, "GBM")        
        # Train Deep Learning model
        grid_deep_learning.train(x=predictors, y=response, training_frame=train_frame, validation_frame=valid_frame)
        # Retrieve leaderboard sorted by AUC
        try:
            # Since deep_learning_model is not obtained via grid search, get_grid is unavailable.
            # Fallback: create a manual leaderboard using the model's own performance.
            deep_learning_leaderboard = pd.DataFrame([{
                "model_id": deep_learning_model.model_id,
                "auc": deep_learning_model.auc(),
                "logloss": deep_learning_model.logloss()
            }])
            self.logger.info("Deep Learning leaderboard created manually.")
            # self.logger.info(deep_learning_leaderboard)
            self._log_grid_search_results(deep_learning_leaderboard, X_val, y_val, "Deep Learning")
        except Exception as e:
            self.logger.error(f"Error while creating deep learning leaderboard: {e}")
        
        self.logger.info("Grid search completed. Best model parameters:")
        
        return grid_leaderboard

    def _log_grid_search_results(self, grid_leaderboard, X_val: pd.DataFrame, y_val: pd.Series, model_type: str) -> None:
        """Log grid search results and save leaderboard to CSV.
        
        Args:
            grid_leaderboard: The leaderboard from H2O grid search (expected as an H2OFrame).
            X_val: Validation features
            y_val: Validation targets
            model_type: A string identifier for the model type used in logging.
        
        Returns:
            None
        """
        try:
            # Convert leaderboard to pandas DataFrame if not already one.
            if isinstance(grid_leaderboard, h2o.H2OFrame):
                leaderboard_df = grid_leaderboard.as_data_frame(use_multi_threading=True)
            elif isinstance(grid_leaderboard, list):
                # If it's a list of models, build a DataFrame manually.
                leaderboard_data = [{"model_id": model.model_id} for model in grid_leaderboard]
                leaderboard_df = pd.DataFrame(leaderboard_data)
            else:
                self.logger.info("grid_leaderboard is already a DataFrame")
                leaderboard_df = grid_leaderboard  # already a DataFrame

            precision_scores = []
            recall_scores = []
            
            # Validate leaderboard structure and ensure model_id column exists
            if not isinstance(leaderboard_df, pd.DataFrame):
                self.logger.error(f"Invalid leaderboard type: {type(leaderboard_df)}. Expected pandas DataFrame")
                return
            
            if 'model_id' not in leaderboard_df.columns:
                self.logger.error("Leaderboard DataFrame missing required 'model_id' column")
                return
                
            for model_id in leaderboard_df['model_id']:
                try:
                        model = h2o.get_model(model_id)
                        preds = model.predict(h2o.H2OFrame(X_val))
                        preds_df = preds.as_data_frame()
                        
                        recall = recall_score(
                            y_val, 
                            (preds_df['p1'] >= self.threshold).astype(int),
                            zero_division=0
                        )
                        precision = precision_score(
                            y_val, 
                            (preds_df['p1'] >= self.threshold).astype(int),
                            zero_division=0
                        )
                        precision_scores.append(precision)
                        recall_scores.append(recall)
                        
                except Exception as e:
                    self.logger.warning(f"Error evaluating model {model_id}: {str(e)}")
                    precision_scores.append(0)
                    recall_scores.append(0)
                    continue
                
                leaderboard_df['precision'] = precision_scores
                leaderboard_df['recall'] = recall_scores

                # Log top 5 models.
                self.logger.info("Grid Search Leaderboard (Top 5 Models):")
                for i, row in leaderboard_df.head().iterrows():
                    self.logger.info(
                        f"Model {i+1}: Precision={row['precision']:.4f}, "
                        f"Recall={row['recall']:.4f}"
                    )

                leaderboard_path = os.path.join(project_root, f"grid_leaderboard_{model_type}.csv")
                leaderboard_df.to_csv(leaderboard_path, index=False)
                self.logger.info(f"Saved grid search leaderboard to {leaderboard_path}")
            else:
                self.logger.warning("Leaderboard DataFrame missing required 'model_id' column")

        except Exception as e:
            self.logger.error(f"Error logging grid search results: {str(e)}")

def h2o_to_polars(h2o_frame):
    """
    Convert an H2OFrame to a Polars DataFrame using as_arrow() for multi-threaded conversion.
    Falls back to the regular as_data_frame() conversion if an error occurs.
    """
    try:
        # First try direct conversion to Polars DataFrame
        
        try:
            # Attempt to use as_arrow() if available
            arrow_table = h2o_frame.as_arrow()  # Returns a pyarrow.Table
            df_pol = pl.from_arrow(arrow_table)
            return df_pol
        except AttributeError:
            # If as_arrow() not available, try converting to pandas first
            pandas_df = h2o_frame.as_data_frame()
            df_pol = pl.from_pandas(pandas_df)
            return df_pol
    except Exception as e:
        # Log the error and fallback to pandas DataFrame
        logger.error(f"Error during polars conversion: {str(e)}. Falling back to single-threaded conversion.")
        return h2o_frame.as_data_frame()  # single-threaded conversion

def train_automl_model():
    """Main function to train the H2O AutoML model."""
    try:
        # Initialize trainer
        trainer = H2OAutoMLTrainer(
            logger=logger,
            max_runtime_secs=3600,
            max_models=50
        )
        
        # Load data
        selected_features = import_selected_features_ensemble('all')
        X_train, y_train, X_test, y_test = import_training_data_ensemble()
        X_val, y_val = create_ensemble_evaluation_set()
        
        # Select features
        X_train = X_train[selected_features]
        X_val = X_val[selected_features]
        X_test = X_test[selected_features]
        predictors = list(X_train.columns)
        response = "is_draw"
        # Start MLflow run
        with mlflow.start_run(
            run_name=f"h2o_automl_{datetime.now().strftime('%Y%m%d_%H%M')}"
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
            # results = trainer.train(
            #     X_train, y_train,
            #     X_val, y_val,
            #     X_test, y_test
            # )
            # # Close current trainer session
            # trainer.logger.info("Closing current H2O AutoML session...")
            # h2o.cluster().shutdown()
            # Print performance summary
            # print("\nModel Performance Summary:")
            # print("-" * 80)
            # print("Validation Set Metrics:")
            # for k, v in results['validation_metrics'].items():
            #     if isinstance(v, (int, float)):
            #         print(f"{k:20}: {v:.4f}")
            
            # print("\nTest Set Metrics:")
            # for k, v in results['test_metrics'].items():
            #     if isinstance(v, (int, float)):
            #         print(f"{k:20}: {v:.4f}")
            
            
            # Initialize new trainer with same parameters
            trainer = H2OAutoMLTrainer(
                logger=logger,
                max_runtime_secs=3600,
                max_models=10,
                seed=42
            )
            trainer.logger.info("Initialized new H2O AutoML trainer instance")
            # Perform grid search
            grid_leaderboard = trainer.grid_search_tuning(
                X_train, y_train,
                X_test, y_test,
                X_val, y_val
            )
            
            
            return results
            
    except Exception as e:
        logger.error(f"Error during AutoML training: {str(e)}")
        raise

if __name__ == "__main__":
    results = train_automl_model()
    print("\nFinal Model Performance:")
    print("-" * 80)
    print("Validation Set:")
    print(f"Precision: {results['validation_metrics']['precision']:.4f}")
    print(f"Recall: {results['validation_metrics']['recall']:.4f}")
    print(f"F1 Score: {results['validation_metrics']['f1']:.4f}")
    print("\nTest Set:")
    print(f"Precision: {results['test_metrics']['precision']:.4f}")
    print(f"Recall: {results['test_metrics']['recall']:.4f}")
    print(f"F1 Score: {results['test_metrics']['f1']:.4f}") 