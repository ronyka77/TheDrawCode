"""Script to run hyperparameter tuning for CatBoost model."""

import sys
from pathlib import Path
from typing import Dict, Any
import traceback
import time
import json
import os
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

os.environ["ARROW_S3_DISABLE"] = "1"

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from utils.logger import ExperimentLogger
    exp_logger = ExperimentLogger(experiment_name="catboost_hypertuning")
    from models.StackedEnsemble.base.tree_based.catboost_model import CatBoostModel
    from models.StackedEnsemble.shared.data_loader import DataLoader
    
except Exception as e:
    raise ImportError(f"Error importing modules: {str(e)}")

def run_catboost_hypertuning() -> Dict[str, Any]:
    """Run hyperparameter tuning for CatBoost model using soccer prediction data.
    
    Returns:
        Dictionary containing best parameters and performance metrics
    """
    start_time = time.time()
    results = {}
    
    try:
        exp_logger.info("Starting CatBoost hyperparameter tuning with soccer prediction data")
        
        # Load actual soccer prediction data
        exp_logger.info("Loading soccer prediction data")
        data_loader = DataLoader(experiment_name="catboost_hypertuning")
        X_train, y_train, X_test, y_test, X_val, y_val = data_loader.load_data()
        
        exp_logger.info(
            f"Data loaded successfully:"
            f"\n - Training set: {X_train.shape}"
            f"\n - Test set: {X_test.shape}"
            f"\n - Validation set: {X_val.shape}"
            f"\n - Draw rate (train): {(y_train == 1).mean():.2%}"
            f"\n - Draw rate (test): {(y_test == 1).mean():.2%}"
            f"\n - Draw rate (val): {(y_val == 1).mean():.2%}"
        )
        
        # Initialize model with CPU optimization
        exp_logger.info("Initializing model...")
        model = CatBoostModel(
            experiment_name="catboost_hypertuning",
            model_type="catboost",
            logger=exp_logger
        )
        exp_logger.info("Model initialized")
        
        # Run hyperparameter optimization
        exp_logger.info("Starting Optuna optimization for hyperparameter tuning")
        
        try:
            best_params = model.optimize_hyperparameters(
                X_train, y_train, X_val, y_val, X_test, y_test
            )
            optimization_time = time.time() - start_time
            exp_logger.info(f"Hyperparameter optimization completed in {optimization_time:.2f} seconds")
            
            # Convert integer parameters
            for key in ['depth', 'min_data_in_leaf', 'seed', 'n_estimators', 'random_seed', 'early_stopping_rounds']:
                if key in best_params:
                    best_params[key] = int(round(best_params[key]))
            
            # Train final model with best parameters
            exp_logger.info(f"Training final model with best parameters: {best_params}")
            train_start = time.time()
            final_model = CatBoostModel(
                experiment_name="catboost_hypertuning_final",
                model_type="catboost",
                logger=exp_logger
            )
            metrics = final_model.fit(X_train, y_train, X_val, y_val, X_test, y_test, **best_params)
            training_time = time.time() - train_start
            exp_logger.info(f"Final model training completed in {training_time:.2f} seconds")
            
            # Evaluate on validation set
            exp_logger.info("Evaluating final model on validation set")
            val_metrics = final_model.evaluate(X_val, y_val)
            exp_logger.info("Model evaluation completed")
            
            # Store all results
            results = {
                'best_params': best_params,
                'training_metrics': metrics,
                'validation_metrics': val_metrics,
                'timing': {
                    'optimization_time': optimization_time,
                    'training_time': training_time,
                    'total_time': time.time() - start_time
                },
                'data_stats': {
                    'train_shape': X_train.shape,
                    'test_shape': X_test.shape,
                    'val_shape': X_val.shape,
                    'draw_rates': {
                        'train': float((y_train == 1).mean()),
                        'test': float((y_test == 1).mean()),
                        'val': float((y_val == 1).mean())
                    }
                }
            }
            
        except Exception as e:
            error_msg = f"Error in optimization process: {str(e)}"
            exp_logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Save results to file
        results_path = project_root / "results" / "hypertuning"
        results_path.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = results_path / f"catboost_hypertuning_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_types(obj):
                if isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_types(item) for item in obj]
                return obj
            
            json.dump(convert_types(results), f, indent=2)
        exp_logger.info(f"Results saved to {results_file}")
        
        return results
        
    except Exception as e:
        exp_logger.error(f"Error during hyperparameter tuning: {str(e)}")
        traceback.print_exc()
        
        # Save error information
        results['error'] = {
            'message': str(e),
            'traceback': traceback.format_exc(),
            'time_of_failure': time.time() - start_time
        }
        
        # Try to save partial results if available
        if results:
            results_path = project_root / "results" / "hypertuning"
            results_path.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            error_file = results_path / f"catboost_hypertuning_error_{timestamp}.json"
            with open(error_file, 'w') as f:
                json.dump(convert_types(results), f, indent=2)
            exp_logger.info(f"Error results saved to {error_file}")
        
        raise

if __name__ == '__main__':
    try:
        exp_logger.info("Starting hyperparameter tuning script...")
        results = run_catboost_hypertuning()
        
        exp_logger.info("\nHyperparameter Tuning Results:")
        exp_logger.info("=" * 50)
        
        exp_logger.info("\nTiming Information:")
        exp_logger.info("-" * 30)
        for timing_key, timing_value in results['timing'].items():
            exp_logger.info(f"{timing_key}: {timing_value:.2f} seconds")
        
        exp_logger.info("\nData Statistics:")
        exp_logger.info("-" * 30)
        exp_logger.info("Dataset Shapes:")
        for key, shape in results['data_stats'].items():
            if 'shape' in key:
                exp_logger.info(f"{key}: {shape}")
        exp_logger.info("\nDraw Rates:")
        for key, rate in results['data_stats']['draw_rates'].items():
            exp_logger.info(f"{key}: {rate:.2%}")
        
        exp_logger.info("\nBest Parameters:")
        exp_logger.info("-" * 30)
        for param, value in results['best_params'].items():
            exp_logger.info(f"{param}: {value}")
        
        exp_logger.info("\nTraining Metrics:")
        exp_logger.info("-" * 30)
        for metric, value in results['training_metrics'].items():
            exp_logger.info(f"{metric}: {value:.4f}")
        
        exp_logger.info("\nValidation Metrics:")
        exp_logger.info("-" * 30)
        for metric, value in results['validation_metrics'].items():
            exp_logger.info(f"{metric}: {value:.4f}")
            
    except Exception as e:
        exp_logger.error(f"\nError: {str(e)}")
        traceback.print_exc()
        sys.exit(1) 