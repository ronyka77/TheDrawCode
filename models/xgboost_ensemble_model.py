# Standard library imports
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import json
import tempfile

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
import xgboost as xgb
from icecream import ic
import mlflow
import mlflow.xgboost
from mlflow.models import ModelSignature, infer_signature
from mlflow.tracking import MlflowClient

# Add project root to Python path
try:
    project_root = Path(__file__).parent.parent
    if not project_root.exists():
        # Handle network path by using raw string
        project_root = Path(r"\\".join(str(project_root).split("\\")))
    sys.path.append(str(project_root))
    print(f"Project root xgboost_ensemble_model: {project_root}")
except Exception as e:
    print(f"Error setting project root path: {e}")
    # Fallback to current directory if path resolution fails
    sys.path.append(os.getcwd().parent)
    print(f"Current directory xgboost_ensemble_model: {os.getcwd().parent}")
    
os.environ['GIT_PYTHON_GIT_EXECUTABLE'] = "C:/Program Files/Git/bin/git.exe"

# Local imports
from utils.logger import ExperimentLogger
from utils.create_evaluation_set import create_evaluation_sets_draws, import_training_data_draws_new, setup_mlflow_tracking

def setup_xgboost_temp_directory(logger: ExperimentLogger, project_root: Path) -> str:
    """Set up and verify XGBoost temporary directory.
    
    Args:
        logger: Logger instance for logging messages
        project_root: Project root path
        
    Returns:
        str: Path to the verified temporary directory
    """
    # Set up temp directory for XGBoost using project_root
    temp_dir = os.path.join(project_root, "temp", "xgboost")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Ensure temp_dir exists and is writable
    try:
        test_file = os.path.join(temp_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
    except Exception as e:
        logger.error(f"Temp directory {temp_dir} is not writable: {e}")
        # Fallback to system temp directory
        temp_dir = os.path.join(tempfile.gettempdir(), "xgboost_temp")
        os.makedirs(temp_dir, exist_ok=True)
        logger.info(f"Using fallback temp directory: {temp_dir}")
    
    # Set environment variables with verified temp directory
    os.environ.update({
        'XGBOOST_CACHE_DIR': temp_dir,
        'TMPDIR': temp_dir,
        'TEMP': temp_dir,
        'TMP': temp_dir
    })
    
    # Log the actual paths being used
    logger.info(f"Using temp directory: {temp_dir}")
    # logger.info(f"XGBOOST_CACHE_DIR: {os.environ.get('XGBOOST_CACHE_DIR')}")
    
    return temp_dir

class TwoStageEnsemble:
    """Two-stage ensemble model optimized for high precision predictions.
    
    First stage is optimized for recall, second stage for precision.
    Both stages must agree for a positive prediction.
    """
    
    def __init__(self, logger: Optional[ExperimentLogger] = None, temp_dir: str = None):
        self.logger = logger or ExperimentLogger()
        self.temp_dir = temp_dir
        
        # Stage 1 parameters (optimized)
        self.stage1_params = {
            'objective': 'binary:logistic',
            'tree_method': 'hist',
            'eta': 0.006865559224954327,
            'min_child_weight': 60,
            'gamma': 7.806498688082427,
            'subsample': 0.9078101439029029,
            'colsample_bytree': 0.7068967670968938,
            'scale_pos_weight': 2.8662440643452793,
            'max_depth': 5,
            'reg_alpha': 2.2327934748903853,
            'reg_lambda': 1.5146242746941245,
            'n_estimators': 11780,
            'early_stopping_rounds': 500
        }
        
        # Stage 2 parameters (optimized)
        self.stage2_params = {
            'objective': 'binary:logistic',
            'tree_method': 'hist',
            'eta': 0.008124767669595323,
            'min_child_weight': 69,
            'gamma': 2.953207382098794,
            'subsample': 0.7232914789349782,
            'colsample_bytree': 0.7646895411848675,
            'scale_pos_weight': 3.1368440226593757,
            'max_depth': 5,
            'reg_alpha': 1.487607571406742,
            'reg_lambda': 3.3472678825444158,
            'n_estimators': 18900,
            'early_stopping_rounds': 500
        }
        
        self.model1 = None
        self.model2 = None
        self.threshold1 = 0.22655221256782027
        self.threshold2 = 0.4925858455173652
        
    def fit(self, 
            X_train: pd.DataFrame, 
            y_train: pd.Series,
            X_val: pd.DataFrame,
            y_val: pd.Series) -> None:
        """Train both stages of the model."""
        try:
            # Create copies and convert integer columns to float64
            X_train_float = X_train.copy()
            X_val_float = X_val.copy()
            
            for col in X_train.select_dtypes(include=['int']).columns:
                X_train_float.loc[:, col] = X_train_float[col].astype('float64')
                X_val_float.loc[:, col] = X_val_float[col].astype('float64')
            
            # Train first stage model
            self.logger.info("Training first stage model...")
            self.model1 = xgb.XGBClassifier(**self.stage1_params)
            self.model1.fit(
                X_train_float, 
                y_train,
                eval_set=[(X_val_float, y_val)],
                verbose=False
            )
            
            # Get samples for second stage using .loc
            train_mask = self.model1.predict_proba(X_train_float)[:, 1] >= self.threshold1
            X_train_stage2 = X_train_float.loc[train_mask].copy()
            y_train_stage2 = y_train.loc[train_mask].copy()
            
            # Only train second stage if enough samples
            if len(y_train_stage2) > 1000:
                self.logger.info("Training second stage model...")
                self.model2 = xgb.XGBClassifier(**self.stage2_params)
                
                # Get validation samples using .loc
                val_mask = self.model1.predict_proba(X_val_float)[:, 1] >= self.threshold1
                X_val_stage2 = X_val_float.loc[val_mask].copy()
                y_val_stage2 = y_val.loc[val_mask].copy()
                
                self.model2.fit(
                    X_train_stage2, 
                    y_train_stage2,
                    eval_set=[(X_val_stage2, y_val_stage2)],
                    verbose=False
                )
            else:
                raise ValueError("Insufficient samples for second stage training")
                
        except Exception as e:
            ic(e)
            self.logger.error(f"Error during model training: {str(e)}")
            raise
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability predictions from both stages.
        
        Args:
            X: Features to predict
            
        Returns:
            Array of probabilities for both classes
        """
        # Convert integer columns to float64
        X_float = X.astype({col: 'float64' for col in X.select_dtypes(include=['int']).columns})
        
        # Get first stage predictions
        stage1_probs = self.model1.predict_proba(X_float)[:, 1]
        stage1_mask = stage1_probs >= self.threshold1
        
        # Initialize final probabilities
        final_probs = np.zeros(len(X))
        
        # Only get second stage predictions for samples that passed first stage
        if stage1_mask.any():
            stage2_probs = self.model2.predict_proba(X_float[stage1_mask])[:, 1]
            final_probs[stage1_mask] = stage2_probs
        
        # Ensure we return a 2D array with shape (n_samples, 2)
        return np.vstack((1 - final_probs, final_probs)).T
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make final predictions requiring both stages to agree.
        
        Args:
            X: Features to predict
            
        Returns:
            Array of binary predictions
        """
        probs = self.predict_proba(X)[:, 1]  # Get positive class probabilities
        return (probs >= self.threshold2).astype(int)
   
    def save(self, path: str) -> None:
        """Save the model and its metadata to disk."""
        try:
            # Save the model using sklearn's API
            model_path = f"{path}_hypertuned.json"
            self.model.get_booster().save_model(model_path)
            
            # Save metadata
            metadata = {
                'stage1_params': self.stage1_params,
                'stage2_params': self.stage2_params,
                'threshold1': self.threshold1,
                'threshold2': self.threshold2
            }
            
            with open(f"{path}_hypertuned_metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load(self, path: str) -> None:
        """Load the model and its metadata from disk."""
        try:
            # Load the model using sklearn's API
            model_path = f"{path}_hypertuned.json"
            self.model = xgb.XGBClassifier()
            self.model.load_model(model_path)
            
            # Load metadata
            try:
                with open(f"{path}_hypertuned_metadata.json", 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    self.stage1_params = metadata.get('stage1_params', {})
                    self.stage2_params = metadata.get('stage2_params', {})
                    self.threshold1 = metadata.get('threshold1', 0.5)
                    self.threshold2 = metadata.get('threshold2', 0.5)
            except FileNotFoundError:
                if self.logger:
                    self.logger.warning("Model metadata not found")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading model: {str(e)}")
            raise
            
class VotingEnsemble:
    """Voting ensemble requiring unanimous agreement for positive predictions."""
    
    def __init__(self, n_models: int = 5, logger: Optional[ExperimentLogger] = None, temp_dir: str = None):
        self.logger = logger or ExperimentLogger()
        self.n_models = n_models
        self.models: List[xgb.XGBClassifier] = []
        
        # Set XGBoost temp directory
        self.temp_dir = temp_dir
        
        # Base parameters (optimized)
        self.base_params = {
            'objective': 'binary:logistic',
            'tree_method': 'hist',
            'eta': 0.0027025804658614323,
            'min_child_weight': 88,
            'gamma': 7.058060647290297,
            'subsample': 0.9392182884712076,
            'colsample_bytree': 0.77474138934419,
            'scale_pos_weight': 3.7283950450408563,
            'max_depth': 5,
            'reg_alpha': 0.7560452842648453,
            'reg_lambda': 3.4087149623397104,
            'n_estimators': 19346,
            'early_stopping_rounds': 500
        }
        
        # Model-specific parameters
        self.model_params = [
            {'learning_rate': 0.01, 'min_child_weight': 50, 'gamma': 4, 'max_depth': 5},
            {'learning_rate': 0.01, 'min_child_weight': 45, 'gamma': 3.5, 'max_depth': 5},
            {'learning_rate': 0.01, 'min_child_weight': 55, 'gamma': 4.5, 'max_depth': 4},
            {'learning_rate': 0.01, 'min_child_weight': 40, 'gamma': 3, 'max_depth': 5},
            {'learning_rate': 0.01, 'min_child_weight': 60, 'gamma': 5, 'max_depth': 4}
        ]
        
        # Updated thresholds (optimized)
        self.thresholds = [
            0.5123415558451974,
            0.523915129253955,
            0.5235739487768032,
            0.5805933205011075,
            0.49127522750596514
        ]
        
    def fit(self, 
            X_train: pd.DataFrame, 
            y_train: pd.Series,
            X_val: pd.DataFrame,
            y_val: pd.Series) -> None:
        """Train all models in the ensemble."""
        try:
            # Create copy and convert integer columns
            X_train_float = X_train.copy()
            for col in X_train.select_dtypes(include=['int']).columns:
                X_train_float.loc[:, col] = X_train_float[col].astype('float64')
            
            for i in range(self.n_models):
                self.logger.info(f"Training model {i+1}/{self.n_models}")
                params = {**self.base_params, **self.model_params[i]}
                model = xgb.XGBClassifier(**params)
                
                # Different sampling for each model
                sample_idx = np.random.choice(
                    len(X_train), 
                    size=int(len(X_train) * 0.8), 
                    replace=True
                )
                
                # Use .loc for indexing
                X_train_sample = X_train_float.iloc[sample_idx].copy()
                y_train_sample = y_train.iloc[sample_idx].copy()
                
                model.fit(
                    X_train_sample, 
                    y_train_sample,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
                
                self.models.append(model)
                
        except Exception as e:
            ic(e)
            self.logger.error(f"Error during ensemble training: {str(e)}")
            raise
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability predictions from all models.
        
        Args:
            X: Features to predict
            
        Returns:
            Array of probabilities for both classes
        """
        # Convert integer columns to float64
        X_float = X.astype({col: 'float64' for col in X.select_dtypes(include=['int']).columns})
        
        # Get probabilities from each model
        all_probs = np.zeros((len(X), len(self.models)))
        for i, model in enumerate(self.models):
            probs = model.predict_proba(X_float)[:, 1]
            all_probs[:, i] = probs
        
        # Average probabilities across models
        final_probs = np.mean(all_probs, axis=1)
        
        # Return probabilities for both classes
        return np.column_stack((1 - final_probs, final_probs))
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions requiring majority agreement.
        
        Args:
            X: Features to predict
            
        Returns:
            Array of binary predictions
        """
        # Convert integer columns to float64
        X_float = X.astype({col: 'float64' for col in X.select_dtypes(include=['int']).columns})
        
        predictions = np.zeros((len(X), self.n_models))
        
        # Get predictions from each model
        for i, (model, threshold) in enumerate(zip(self.models, self.thresholds)):
            probs = model.predict_proba(X_float)[:, 1]
            predictions[:, i] = (probs >= threshold).astype(int)
        
        # Change to majority voting (3 out of 5)
        final_predictions = (predictions.sum(axis=1) >= 3).astype(int)
        return final_predictions

def train_ensemble_model() -> Dict[str, Any]:
    """Train and evaluate the ensemble model."""
    logger = ExperimentLogger()
    experiment_name = "xgboost_ensemble_draw_model"
    mlruns_dir = setup_mlflow_tracking(experiment_name)
   
    temp_dir = setup_xgboost_temp_directory(logger, project_root)
    
    try:
        # Load data
        X_train, y_train, X_test, y_test = import_training_data_draws_new()
        X_val, y_val = create_evaluation_sets_draws()
        
        ic(y_train.sum(), y_val.sum(), y_test.sum())  # Check class distribution
        
        logger.info("Starting ensemble model training")
    

        def combined_predict(X, two_stage, voting):
            """Make combined predictions from both models."""
            # Get probabilities for positive class only (class 1)
            two_stage_probs = two_stage.predict_proba(X)[:, 1]  # Get second column for positive class
            voting_probs = np.mean([model.predict_proba(X)[:, 1] for model in voting.models], axis=0)
            
            # Combined prediction using optimized thresholds
            combined = ((two_stage_probs >= two_stage.threshold2) & 
                       (voting_probs >= np.mean(voting.thresholds))).astype(int)
            
            ic(combined.sum(), "Combined positive predictions")
            return combined
        
        def evaluate_model(model, X, y, name):
            """Evaluate individual model performance."""
            preds = model.predict(X)
            metrics = {
                'precision': precision_score(y, preds, zero_division=0),
                'recall': recall_score(y, preds, zero_division=0),
                'f1': f1_score(y, preds, zero_division=0)
            }
            ic(f"{name} metrics:", metrics)
            return metrics
        
        with mlflow.start_run(run_name="xgboost_ensemble_draw_model") as run:
            # Add k-fold validation
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            val_precisions = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                # Train models
                two_stage = TwoStageEnsemble(logger, temp_dir=temp_dir)
                voting = VotingEnsemble(n_models=5, logger=logger, temp_dir=temp_dir)
                
                two_stage.fit(X_fold_train, y_fold_train, X_fold_val, y_fold_val)
                voting.fit(X_fold_train, y_fold_train, X_fold_val, y_fold_val)
                
                # Evaluate
                val_preds = combined_predict(X_fold_val, two_stage, voting)
                precision = precision_score(y_fold_val, val_preds)
                val_precisions.append(precision)
                ic(f"Fold {fold+1} Precision:", precision)
            
            # Train final models on full training set
            final_two_stage = TwoStageEnsemble(logger, temp_dir=temp_dir)
            final_voting = VotingEnsemble(n_models=5, logger=logger, temp_dir=temp_dir)
            
            final_two_stage.fit(X_train, y_train, X_val, y_val)
            final_voting.fit(X_train, y_train, X_val, y_val)
            
            # Evaluate individual models
            two_stage_metrics = evaluate_model(final_two_stage, X_val, y_val, "Two-stage")
            voting_metrics = evaluate_model(final_voting, X_val, y_val, "Voting")
            
            # Evaluate on validation set
            val_preds = combined_predict(X_val, final_two_stage, final_voting)
            ic(val_preds.sum(), "Validation positive predictions")
            ic(y_val.sum(), "Validation actual positives")
            
            val_metrics = {
                'precision': precision_score(y_val, val_preds, zero_division=0),
                'recall': recall_score(y_val, val_preds, zero_division=0),
                'f1': f1_score(y_val, val_preds, zero_division=0)
            }
            ic(val_metrics)
            
            # Evaluate on test set
            test_preds = combined_predict(X_test, final_two_stage, final_voting)
            ic(test_preds.sum(), "Test positive predictions")
            ic(y_test.sum(), "Test actual positives")
            
            test_metrics = {
                'precision': precision_score(y_test, test_preds, zero_division=0),
                'recall': recall_score(y_test, test_preds, zero_division=0),
                'f1': f1_score(y_test, test_preds, zero_division=0)
            }
            ic(test_metrics)
            
            # Log metrics
            mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
            
            # Create input example using first row of validation data
            input_example = X_val.iloc[[0]].copy()
            
            # Save the models using xgboost format instead of pyfunc
            mlflow.xgboost.log_model(
                final_two_stage.model1,
                "stage1_model",
                registered_model_name="stage1_model",
                input_example=input_example
            )
            
            if final_two_stage.model2 is not None:
                mlflow.xgboost.log_model(
                    final_two_stage.model2,
                    "stage2_model",
                    registered_model_name="stage2_model",
                    input_example=input_example
                )
            
            # Save voting ensemble components
            for i, model in enumerate(final_voting.models):
                mlflow.xgboost.log_model(
                    model,
                    f"voting_model_{i}",
                    registered_model_name=f"voting_model_{i}",
                    input_example=input_example
                )
            
            # Log parameters
            mlflow.log_params({
                "two_stage_threshold1": final_two_stage.threshold1,
                "two_stage_threshold2": final_two_stage.threshold2,
                "voting_thresholds": final_voting.thresholds
            })
            
            # Log feature names
            mlflow.log_param("feature_names", list(X_train.columns))
            
            # Save model artifacts
            model_info = {
                'two_stage_params': {
                    'stage1_params': final_two_stage.stage1_params,
                    'stage2_params': final_two_stage.stage2_params,
                    'threshold1': final_two_stage.threshold1,
                    'threshold2': final_two_stage.threshold2
                },
                'voting_params': {
                    'base_params': final_voting.base_params,
                    'model_params': final_voting.model_params,
                    'thresholds': final_voting.thresholds
                }
            }
    
            
            with open("model_info.json", "w") as f:
                json.dump(model_info, f)
            mlflow.log_artifact("model_info.json")
            
            return {
                'validation_metrics': val_metrics,
                'test_metrics': test_metrics,
                'models': {
                    'two_stage': final_two_stage,
                    'voting': final_voting
                },
                'run_id': run.info.run_id
            }
            
    except Exception as e:
        ic(e)  # Debug the exception
        logger.error(f"Error in ensemble training: {str(e)}")
        raise

if __name__ == "__main__":
    results = train_ensemble_model()
    
    # Print final results
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
    print(f"Run ID: {results['run_id']}")