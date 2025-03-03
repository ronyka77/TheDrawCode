"""
Soccer Prediction Model using MLJAR AutoML

A simplified implementation leveraging MLJAR AutoML for soccer draw prediction
with emphasis on precision metrics and threshold optimization.
"""

# -------------------------
# 1. IMPORTS AND SETUP
# -------------------------
import pandas as pd
import numpy as np
import os
import sys
import argparse
import logging
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from datetime import datetime
import shutil

# Add project root to Python path if needed
file = __file__
project_root = Path(file).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import MLJAR AutoML
try:
    from supervised import AutoML
except ImportError:
    print("Installing MLJAR AutoML...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mljar-supervised"])
    from supervised import AutoML

# Optional MLflow integration
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# -------------------------
# 2. LOGGING SETUP
# -------------------------
# Setup logging using ExperimentLogger
from utils.logger import ExperimentLogger
logger = ExperimentLogger(
    experiment_name="mljar_soccer_prediction",
    log_dir="logs/mljar_soccer_prediction"
)

# -------------------------
# 3. DATA LOADING
# -------------------------
def load_data_for_prediction():
    """
    Load data for soccer prediction using the existing project's DataLoader.
    Returns:
        tuple: (train_df, test_df, val_df) with target column included
    """
    logger.info("Loading data using project DataLoader")
    try:
        # Try to import from the existing project
        from models.StackedEnsemble.shared.data_loader import DataLoader
        # Initialize DataLoader
        data_loader = DataLoader()
        # Get train, test, and validation splits
        X_train, y_train, X_test, y_test, X_val, y_val = data_loader.load_data()
        # Convert all columns to float64 to avoid type compatibility issues
        logger.info("Converting all features to float64 type")
        X_train = X_train.astype('float64')
        X_test = X_test.astype('float64')
        X_val = X_val.astype('float64')
        y_train = y_train.astype('float64')
        y_test = y_test.astype('float64')
        y_val = y_val.astype('float64')
        # Log the conversion for tracking
        logger.info(f"Data types after conversion: {X_train.dtypes.value_counts().to_dict()}")
        # Create DataFrames with target included
        logger.info(f"Loaded data:")
        logger.info(f" - Training: {X_train.shape} with {y_train.sum()} positive samples ({y_train.mean()*100:.2f}%)")
        logger.info(f" - Testing: {X_test.shape} with {y_test.sum()} positive samples ({y_test.mean()*100:.2f}%)")
        logger.info(f" - Validation: {X_val.shape} with {y_val.sum()} positive samples ({y_val.mean()*100:.2f}%)")

        return X_train, y_train, X_test, y_test, X_val, y_val
    except ImportError:
        logger.error("Failed to import DataLoader from project")
        logger.info("Using sample data for demonstration")
        raise 

from sklearn.metrics import recall_score, precision_score, f1_score

def custom_metric(y_true, y_pred, y_pred_proba=None):
    """Custom metric that returns average precision but only if recall >= 0.20"""
    # For binary classification, we need probabilities for average precision
    if y_pred_proba is not None:
        prec = precision_score(y_true, y_pred_proba[:, 1])
    else:
        # Fall back to binary predictions if probabilities not available
        prec = precision_score(y_true, y_pred)
    
    # Calculate recall using binary predictions
    rec = recall_score(y_true, y_pred)
    
    # Return 0 if recall constraint not met
    if rec < 0.20:
        return 0.0
    else:
        return prec


# -------------------------
# 4. MODEL CLASS
# -------------------------
class SoccerPredictionModel:
    """Unified soccer prediction model using MLJAR AutoML"""
    
    def __init__(self, target_precision=0.4, min_recall=0.25):
        self.target_precision = target_precision
        self.min_recall = min_recall
        self.model = None
        self.threshold = 0.5  # Default threshold
    
    def load_data(self, X_train=None, y_train=None, X_test=None, y_test=None, X_val=None, y_val=None, 
                    train_path=None, val_path=None, test_path=None):
        """Load data from DataFrames or file paths"""
        # Load from paths if provided, otherwise use DataFrames
        self.X_val = X_val
        self.y_val = y_val
        # Concatenate X_train and X_test as requested
        self.X_train = pd.concat([X_train, X_test], axis=0)
        self.y_train = pd.concat([y_train, y_test], axis=0)
        logger.info(f"Concatenated X_train and X_test: new training shape {self.X_train.shape}")
        # Extract X and y
        logger.info(f"Data loaded successfully")
        logger.info(f"Training data shape: {self.X_train.shape}")
        logger.info(f"Validation data shape: {self.X_val.shape}")
        
        return self
    
    def train(self, mode="Compete", algorithms=None, metric="f1", total_time_limit=3600, validation_strategy="custom"):
        """Train models using MLJAR AutoML
        
        Args:
            mode (str): AutoML mode (Compete, Explain, Perform, etc.)
            algorithms (list): List of algorithms to use
            metric (str): Evaluation metric for optimization
            total_time_limit (int): Time limit for training in seconds
            validation_strategy (str): Validation strategy (auto, custom, kfold, split)
        
        Examples:
            # Using custom validation data (explicitly provided validation set)
            model.load_data(train_df=train_df, val_df=val_df, test_df=test_df)
            model.train(validation_strategy="custom")
            
            # Using k-fold cross-validation
            model.load_data(train_df=train_df, test_df=test_df)
            model.train(validation_strategy="kfold")
            
            # Using a random split for validation
            model.load_data(train_df=combined_train_df, test_df=test_df)
            model.train(validation_strategy="split")
            
            # Let AutoML decide the validation strategy
            model.load_data(train_df=train_df, test_df=test_df)
            model.train(validation_strategy="auto")
        
        Returns:
            SoccerPredictionModel: The trained model instance
        """
        algorithms = ["Xgboost", "LightGBM", "CatBoost", "Random Forest", "Neural Network"]
        
        logger.info(f"Training models with algorithms: {algorithms}")
        logger.info(f"Mode: {mode}, Metric: AUCPR")
        
        # Check if model already exists and force retraining
        if os.path.exists('MLjar_results'):
            logger.info("Removing existing model results to force retraining")
            shutil.rmtree('MLjar_results')
        
        # Initialize AutoML
        self.model = AutoML(
            mode='Explain', 
            # optuna_init_params={
            #     "direction": "maximize",
            #     "n_trials": 100,
            #     "timeout": 600,
            #     "n_jobs": -1,
            #     "random_state": 19
            # },                   # Compete mode uses stacked ensembles
            algorithms=algorithms,
            eval_metric='f1',   # Optimization metric
            results_path='MLjar_results',
            explain_level=2,              # Detailed model explanations
            ml_task="binary_classification",
            start_random_models=1,        # Start with 5 random models
            hill_climbing_steps=5,        # Fine-tune top models
            top_models_to_improve=3,
            features_selection=False,
            golden_features=False,         # Create interaction features
            kmeans_features=False,
            stack_models=True,            # Create stacked ensembles
            total_time_limit=total_time_limit,  # Time limit in seconds
            train_ensemble=True,          # Train ensemble from best models
            n_jobs=-1,                    # Use all available cores
            verbose=1,                    # Show progress information
            random_state=19,              # For reproducibility
            validation_strategy={
                "validation_type": "split",
                "train_ratio": 0.75,
                "shuffle": True
            }  
        )
        
        # self.model.eval_metric = custom_metric
        # Train model
        logger.info("Starting model training...")
        
        self.model.fit(self.X_train, self.y_train)
        
        logger.info("Model training completed")
        
        # Display model leaderboard
        leaderboard = self.model.get_leaderboard()
        logger.info("Model Leaderboard:")
        print(leaderboard)
        
        return self

    def optimize_threshold(self):
        """Find optimal threshold to achieve target precision with minimum recall"""
        if self.model is None:
            raise ValueError("Model must be trained before optimizing threshold")
        
        logger.info(f"Optimizing threshold for target precision: {self.target_precision}, minimum recall: {self.min_recall}")
        
        # Get feature names from the model
        try:
            # Try to get feature names from the model
            model_features = self.model.get_model_features()
            logger.info(f"Ensuring validation data columns match model features ({len(model_features)} features)")
            # Reorder validation data columns to match model features
            self.X_val = self.X_val[model_features]
        except:
            logger.warning("Could not get model features, attempting prediction with current column order")
        
        # Get predictions on validation set
        val_probs = self.model.predict_proba(self.X_val)[:, 1]
        
        # Try different thresholds
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_threshold = 0.5
        best_score = 0
        results = []
        
        for threshold in thresholds:
            # Apply threshold
            y_pred = (val_probs >= threshold).astype(int)
            
            # Calculate metrics
            prec = precision_score(self.y_val, y_pred, zero_division=0)
            rec = recall_score(self.y_val, y_pred, zero_division=0)
            f1 = f1_score(self.y_val, y_pred, zero_division=0)
            
            # Store results
            results.append({
                'threshold': threshold,
                'precision': prec,
                'recall': rec,
                'f1': f1
            })
            
            # Calculate custom score (precision if meets min recall)
            score = prec if rec >= self.min_recall else 0
            
            # Update best threshold if better score
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        self.threshold = best_threshold
        
        # Get metrics at best threshold
        y_pred_best = (val_probs >= self.threshold).astype(int)
        best_precision = precision_score(self.y_val, y_pred_best)
        best_recall = recall_score(self.y_val, y_pred_best)
        best_f1 = f1_score(self.y_val, y_pred_best)
        
        logger.info(f"Threshold Optimization Results:")
        logger.info(f"Optimized threshold: {self.threshold:.3f}")
        logger.info(f"Precision at this threshold: {best_precision:.3f}")
        logger.info(f"Recall at this threshold: {best_recall:.3f}")
        logger.info(f"F1 at this threshold: {best_f1:.3f}")
        
        # Create DataFrame with results for visualization
        results_df = pd.DataFrame(results)
        
        return self, results_df

    def evaluate(self):
        """Evaluate model on test data with optimized threshold"""
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        logger.info("Evaluating model on test data with optimized threshold")
        
        # Get predictions on test set
        test_probs = self.model.predict_proba(self.X_val)[:, 1]
        test_preds = (test_probs >= self.threshold).astype(int)
        
        # Calculate metrics
        prec = precision_score(self.y_val, test_preds, zero_division=0)
        rec = recall_score(self.y_val, test_preds, zero_division=0)
        f1 = f1_score(self.y_val, test_preds, zero_division=0)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(self.y_val, test_preds).ravel()
        
        metrics = {
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'threshold': self.threshold,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }
        
        logger.info(f"Evaluation Results:")
        logger.info(f"Precision: {prec:.3f}")
        logger.info(f"Recall: {rec:.3f}")
        logger.info(f"F1 Score: {f1:.3f}")
        logger.info(f"Confusion Matrix:")
        logger.info(f" TP: {tp}, FP: {fp}")
        logger.info(f" FN: {fn}, TN: {tn}")
        
        return metrics

    def predict(self, X):
        """Make predictions with optimized threshold"""
        if self.model is None:
            self.model = AutoML(
                results_path='MLjar_results'
            )
        
        # Get probabilities
        probs = self.model.predict_proba(X)[:, 1]
        
        # Apply threshold
        preds = (probs >= self.threshold).astype(int)
        
        # Return DataFrame with probabilities and predictions
        return pd.DataFrame({
            'probability': probs,
            'prediction': preds
        })
# -------------------------
# 5. MLFLOW INTEGRATION
# -------------------------
def setup_mlflow_tracking(experiment_name="mljar_soccer_prediction"):
    """Set up MLflow tracking"""
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not available, skipping experiment tracking")
        return None
    
    try:
        # Import MLFlowConfig and MLFlowManager from utils.mlflow_utils
        from utils.create_evaluation_set import setup_mlflow_tracking
        
        # Configure MLflow using the utility from create_evaluation_set
        mlflow_config = setup_mlflow_tracking(experiment_name)
        
        logger.info(f"MLflow tracking set up for experiment: {experiment_name}")
        return mlflow_config
    except Exception as e:
        logger.error(f"Error setting up MLflow: {str(e)}", 
                    extra={"error_code": "E302"})  # Using standard error code from DataProcessingError
        return None

def log_metrics_to_mlflow(metrics, model_path=None):
    """Log metrics to MLflow"""
    if not MLFLOW_AVAILABLE:
        return
    
    try:
        # Start run
        with mlflow.start_run(run_name='AutoML_tuning'):
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model
            if model_path and os.path.exists(model_path):
                mlflow.log_artifact(model_path)
            
            logger.info("Logged metrics to MLflow")
    except Exception as e:
        logger.error(f"Error logging to MLflow: {str(e)}")

# -------------------------
# 6. COMMAND LINE INTERFACE
# -------------------------
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Soccer Prediction using MLJAR AutoML")
    
    # Mode selection
    parser.add_argument("--mode", choices=["train", "predict", "evaluate"], default="train",
                        help="Operation mode: train a new model, make predictions, or evaluate existing model")
    
    # Model parameters
    parser.add_argument("--target-precision", type=float, default=0.4,
                        help="Target precision to achieve (default: 0.4)")
    parser.add_argument("--min-recall", type=float, default=0.25,
                        help="Minimum recall to maintain (default: 0.25)")
    parser.add_argument("--algorithms", type=str, default="Xgboost,LightGBM,CatBoost",
                        help="Comma-separated list of algorithms to use (default: Xgboost,LightGBM,CatBoost)")
    parser.add_argument("--time-limit", type=int, default=3600,
                        help="Time limit for training in seconds (default: 3600)")
    parser.add_argument("--validation-strategy", type=str, choices=["auto", "custom", "kfold", "split"], default="custom",
                        help="Validation strategy: auto, custom (use validation set), kfold, or split (default: custom)")
    
    # Model paths
    parser.add_argument("--model-path", type=str, default="soccer_model",
                        help="Path to save/load model (default: soccer_model)")
    
    # Data paths
    parser.add_argument("--train-path", type=str, default=None,
                        help="Path to training data (default: use built-in data loader)")
    parser.add_argument("--test-path", type=str, default=None,
                        help="Path to test data (default: use built-in data loader)")
    parser.add_argument("--val-path", type=str, default=None,
                        help="Path to validation data (default: use built-in data loader)")
    parser.add_argument("--predict-path", type=str, default=None,
                        help="Path to data for prediction (required for predict mode)")
    
    # MLflow options
    parser.add_argument("--experiment-name", type=str, default="mljar_soccer_prediction",
                        help="MLflow experiment name (default: mljar_soccer_prediction)")
    parser.add_argument("--disable-mlflow", action="store_true",
                        help="Disable MLflow tracking")
    
    return parser.parse_args()

# -------------------------
# 7. MAIN WORKFLOW
# -------------------------
def main():
    """Main function to run the workflow"""
    # Parse arguments
    args = parse_args()
    
    # Setup MLflow tracking if enabled
    if MLFLOW_AVAILABLE and not args.disable_mlflow:
        setup_mlflow_tracking(args.experiment_name)
    
    # Initialize model
    model = SoccerPredictionModel(
        target_precision=args.target_precision,
        min_recall=args.min_recall
    )
    X_train, y_train, X_test, y_test, X_val, y_val = load_data_for_prediction()
    # Execute based on mode
    if args.mode == "train":
        logger.info("Loading data from project data loader")
        model.load_data(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            X_val=X_val,
            y_val=y_val
        )
        
        # Train model with validation data
        model.train()
        
        # Optimize threshold
        model, threshold_results = model.optimize_threshold()
        
        # Evaluate model
        metrics = model.evaluate()
        
        # Log to MLflow if enabled
        if MLFLOW_AVAILABLE and not args.disable_mlflow:
            log_metrics_to_mlflow(metrics, args.model_path)
        
        logger.info("Training workflow completed successfully")
    
    elif args.mode == "evaluate":
        # Load saved model
        model.load(args.model_path)
        
        # Load test data
        if args.test_path:
            test_df = pd.read_csv(args.test_path)
            model.X_test = test_df.drop('target', axis=1)
            model.y_test = test_df['target']
        else:
            _, test_df, _ = load_data_for_prediction()
            model.X_test = test_df.drop('target', axis=1)
            model.y_test = test_df['target']
        
        # Evaluate model
        metrics = model.evaluate()
        
        # Log to MLflow if enabled
        if MLFLOW_AVAILABLE and not args.disable_mlflow:
            log_metrics_to_mlflow(metrics, args.model_path)
        
        logger.info("Evaluation workflow completed successfully")
    
    elif args.mode == "predict":
        # Check if prediction path is provided
        if not args.predict_path:
            logger.error("Prediction path must be provided for predict mode")
            return
        
        # Load saved model
        model.load(args.model_path)
        
        # Load prediction data
        predict_df = pd.read_csv(args.predict_path)
        
        # Make predictions
        logger.info(f"Making predictions on {len(predict_df)} samples")
        predictions = model.predict(predict_df)
        
        # Save predictions
        output_path = f"{os.path.splitext(args.predict_path)[0]}_predictions.csv"
        predictions.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
        
        # Count positive predictions
        positive_count = predictions['prediction'].sum()
        positive_pct = (positive_count / len(predictions)) * 100
        logger.info(f"Positive predictions: {positive_count} ({positive_pct:.2f}%)")
        logger.info("Prediction workflow completed successfully")
    
    else:
        logger.error(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()
