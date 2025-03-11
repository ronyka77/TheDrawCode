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
        # Create DataFrames with target included
        train_df = X_train.copy()
        train_df['target'] = y_train
        test_df = X_test.copy()
        test_df['target'] = y_test
        val_df = X_val.copy()
        val_df['target'] = y_val
        logger.info(f"Loaded data:")
        logger.info(f" - Training: {train_df.shape} with {train_df['target'].sum()} positive samples ({train_df['target'].mean()*100:.2f}%)")
        logger.info(f" - Testing: {test_df.shape} with {test_df['target'].sum()} positive samples ({test_df['target'].mean()*100:.2f}%)")
        logger.info(f" - Validation: {val_df.shape} with {val_df['target'].sum()} positive samples ({val_df['target'].mean()*100:.2f}%)")
        return train_df, test_df, val_df
    except ImportError:
        logger.error("Failed to import DataLoader from project")
        logger.info("Using sample data for demonstration")
        # Create sample data for demonstration
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        # Generate features
        X = np.random.randn(n_samples, n_features)
        # Generate target (imbalanced binary classification)
        y = np.zeros(n_samples)
        y[:int(n_samples * 0.2)] = 1  # 20% positive class
        # Create DataFrames
        columns = [f'feature_{i}' for i in range(n_features)]
        all_data = pd.DataFrame(X, columns=columns)
        all_data['target'] = y
        # Split into train, test, validation
        train_df = all_data[:int(n_samples * 0.7)]
        test_df = all_data[int(n_samples * 0.7):int(n_samples * 0.85)]
        val_df = all_data[int(n_samples * 0.85):]
        logger.info(f"Created sample data:")
        logger.info(f" - Training: {train_df.shape}")
        logger.info(f" - Testing: {test_df.shape}")
        logger.info(f" - Validation: {val_df.shape}")
        return train_df, test_df, val_df

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
    
    def load_data(self, train_df=None, val_df=None, test_df=None, 
                    train_path=None, val_path=None, test_path=None):
        """Load data from DataFrames or file paths"""
        # Load from paths if provided, otherwise use DataFrames
        self.train_df = pd.read_csv(train_path) if train_path else train_df
        self.val_df = pd.read_csv(val_path) if val_path else val_df
        self.test_df = pd.read_csv(test_path) if test_path else test_df
        self.train_df = pd.concat([self.train_df, self.test_df])
        # Extract X and y
        # Extract features and targets
        self.X_train = self.train_df.drop('target', axis=1)
        self.y_train = self.train_df['target']
        self.X_val = self.val_df.drop('target', axis=1)
        self.y_val = self.val_df['target']
        
        
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
        algorithms = ["Xgboost", "LightGBM", "CatBoost", "RandomForest", "NeuralNetwork"]
        
        logger.info(f"Training models with algorithms: {algorithms}")
        logger.info(f"Mode: {mode}, Metric: AUCPR")
        
        # Initialize AutoML
        self.model = AutoML(
            mode='Compete',                    # Compete mode uses stacked ensembles
            eval_metric='logloss',           # Optimization metric
            explain_level=2,              # Detailed model explanations
            ml_task="binary_classification",
            start_random_models=5,        # Start with 5 random models
            hill_climbing_steps=3,        # Fine-tune top models
            golden_features=True,         # Create interaction features
            stack_models='auto',            # Create stacked ensembles
            total_time_limit=total_time_limit,  # Time limit in seconds
            train_ensemble=True,          # Train ensemble from best models
            n_jobs=-1,                    # Use all available cores
            verbose=1,                    # Show progress information
            random_state=19,              # For reproducibility
            validation_strategy={
                "validation_type": "split",
                "train_ratio": 0.8,
                "shuffle": True
            }  # Validation strategy with 20% split
        )
        
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
            prec = precision_score(self.y_val, y_pred)
            rec = recall_score(self.y_val, y_pred)
            f1 = f1_score(self.y_val, y_pred)
            
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
        test_probs = self.model.predict_proba(self.X_test)[:, 1]
        test_preds = (test_probs >= self.threshold).astype(int)
        
        # Calculate metrics
        prec = precision_score(self.y_test, test_preds)
        rec = recall_score(self.y_test, test_preds)
        f1 = f1_score(self.y_test, test_preds)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(self.y_test, test_preds).ravel()
        
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
            raise ValueError("Model must be trained before prediction")
        
        # Get probabilities
        probs = self.model.predict_proba(X)[:, 1]
        
        # Apply threshold
        preds = (probs >= self.threshold).astype(int)
        
        # Return DataFrame with probabilities and predictions
        return pd.DataFrame({
            'probability': probs,
            'prediction': preds
        })

    def save(self, directory="soccer_model"):
        """Save model and configuration"""
        if self.model is None:
            raise ValueError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save model
        self.model.save(directory)
        
        # Save threshold
        with open(os.path.join(directory, "threshold.txt"), "w") as f:
            f.write(str(self.threshold))
        
        # Save configuration
        config = {
            'target_precision': self.target_precision,
            'min_recall': self.min_recall,
            'threshold': self.threshold,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        import json
        with open(os.path.join(directory, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
        
        logger.info(f"Model saved to {directory}")
        
        return self

    def load(self, directory="soccer_model"):
        """Load saved model and configuration"""
        if not os.path.exists(directory):
            raise ValueError(f"Model directory {directory} does not exist")
        
        logger.info(f"Loading model from {directory}")
        
        # Load model
        self.model = AutoML()
        self.model.load(directory)
        
        # Load threshold
        threshold_path = os.path.join(directory, "threshold.txt")
        if os.path.exists(threshold_path):
            with open(threshold_path, "r") as f:
                self.threshold = float(f.read().strip())
        
        # Load configuration if available
        config_path = os.path.join(directory, "config.json")
        if os.path.exists(config_path):
            import json
            with open(config_path, "r") as f:
                config = json.load(f)
                self.target_precision = config.get('target_precision', 0.4)
                self.min_recall = config.get('min_recall', 0.25)
                if 'threshold' in config:
                    self.threshold = config['threshold']
        logger.info(f"Model loaded with threshold: {self.threshold}")
        
        return self

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
    
    # Execute based on mode
    if args.mode == "train":
        # Load data
        if args.train_path and args.test_path and args.val_path:
            logger.info(f"Loading data from provided paths")
            model.load_data(
                train_path=args.train_path,
                test_path=args.test_path,
                val_path=args.val_path
            )
        else:
            logger.info("Loading data from project data loader")
            train_df, test_df, val_df = load_data_for_prediction()
            model.load_data(
                train_df=train_df,
                test_df=test_df,
                val_df=val_df
            )
        
        # Split algorithm string
        algorithms = [algo.strip() for algo in args.algorithms.split(",")]
        
        # Log validation data info
        if args.validation_strategy == "custom":
            logger.info(f"Using validation data with {len(model.X_val)} samples for early stopping")
        else:
            logger.info(f"Using {args.validation_strategy} validation strategy")
        
        # Train model with validation data
        model.train(
            algorithms=algorithms,
            total_time_limit=args.time_limit,
            validation_strategy=args.validation_strategy
        )
        
        # Optimize threshold
        model, threshold_results = model.optimize_threshold()
        
        # Evaluate model
        metrics = model.evaluate()
        
        # Save model
        model.save(args.model_path)
        
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
