"""Script to run hyperparameter tuning for BERT model."""

import sys
from pathlib import Path
from typing import Dict, Any
import traceback
import time
import json
import os
import ray
from ray import tune
import numpy as np
import torch

os.environ["ARROW_S3_DISABLE"] = "1"
os.environ["RAY_DEDUP_LOGS"] = "0"

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from utils.logger import ExperimentLogger
    exp_logger = ExperimentLogger(experiment_name="bert_hypertuning")
    from models.StackedEnsemble.base.transformer_based.bert_model import BertModel
    from models.StackedEnsemble.shared.data_loader import DataLoader
    
except Exception as e:
    raise ImportError(f"Error importing modules: {str(e)}")

def run_bert_hypertuning() -> Dict[str, Any]:
    """Run hyperparameter tuning for BERT model using soccer prediction data.
    
    Returns:
        Dictionary containing best parameters and performance metrics
    """
    start_time = time.time()
    results = {}
    
    try:
        exp_logger.info("Starting BERT hyperparameter tuning with soccer prediction data")
        
        # Load actual soccer prediction data
        exp_logger.info("Loading soccer prediction data")
        data_loader = DataLoader(experiment_name="bert_hypertuning")
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
        model = BertModel(
            experiment_name="bert_hypertuning",
            model_type="bert",
            logger=exp_logger
        )
        exp_logger.info("Model initialized")
        
        # Run hyperparameter optimization
        exp_logger.info("Starting nested cross-validation for hyperparameter tuning")
        
        # Set up storage path for Ray
        storage_path = project_root / "ray_results"
        storage_path.mkdir(parents=True, exist_ok=True)
        
        # Configure Ray for local execution with proper resource management
        if not ray.is_initialized():
            ray.init(
                num_cpus=os.cpu_count(),  # Use all available CPUs
                _temp_dir=str(storage_path / "tmp"),  # Shorter temp directory
                ignore_reinit_error=True,
                include_dashboard=False,  # Disable dashboard for reduced overhead
                log_to_driver=True,  # Enable logging to driver for better debugging
            )
            exp_logger.info("Ray initialized with CPU-only configuration")
        
        try:
            best_params = model.optimize_hyperparameters(
                X_train, y_train, X_val, y_val, X_test, y_test
            )
            optimization_time = time.time() - start_time
            exp_logger.info(f"Hyperparameter optimization completed in {optimization_time:.2f} seconds")
            
            try:
                # Handle Ray ObjectRef by getting the actual value
                if hasattr(best_params, '_object_ref'):
                    exp_logger.info("Best parameters is a Ray ObjectRef")
                    best_params = ray.get(best_params)
                
                # Convert tuple to dict if needed
                if isinstance(best_params, tuple):
                    # Get parameter names from model config
                    model_config = model.config_loader.load_model_config('bert')
                    param_names = list(model_config.get('params', {}).keys())
                    
                    # Create dict with available parameters
                    best_params = dict(zip(param_names[:len(best_params)], best_params))
                    exp_logger.info(f"Converted tuple to dict with {len(best_params)} parameters")
                
                # Convert other non-dict types if possible
                elif not isinstance(best_params, dict):
                    if hasattr(best_params, '_asdict'):
                        exp_logger.info("Best parameters is a namedtuple")
                        best_params = best_params._asdict()
                    else:
                        error_msg = f"Cannot convert best_params of type {type(best_params)} to dict"
                        exp_logger.error(error_msg)
                        raise ValueError(error_msg)
                
                # Convert numpy types to Python types for JSON serialization
                best_params_log = {
                    k: v.item() if isinstance(v, (np.generic, np.number)) else v
                    for k, v in best_params.items() if not k.startswith('_')
                }
                exp_logger.info(f"Best parameters found: {json.dumps(best_params_log, indent=2)}")
            
            except Exception as e:
                error_msg = f"Error processing best parameters: {str(e)}"
                exp_logger.error(error_msg)
                raise ValueError(error_msg) from e
            
            # Convert integer parameters
            for key in ['per_device_train_batch_size', 'gradient_accumulation_steps', 'num_train_epochs', 'max_seq_length']:
                if key in best_params:
                    best_params[key] = int(round(best_params[key]))
            
            # Train final model with best parameters
            exp_logger.info(f"Training final model with best parameters: {best_params}")
            train_start = time.time()
            final_model = BertModel(
                experiment_name="bert_hypertuning_final",
                model_type="bert",
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
            
        finally:
            # Clean up Ray resources
            if ray.is_initialized():
                ray.shutdown()
                exp_logger.info("Ray resources cleaned up")
            
            # Clean up PyTorch CUDA memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Save results to file
        results_path = project_root / "results" / "hypertuning"
        results_path.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = results_path / f"bert_hypertuning_{timestamp}.json"
        
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
            error_file = results_path / f"bert_hypertuning_error_{timestamp}.json"
            with open(error_file, 'w') as f:
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
            exp_logger.info(f"Error results saved to {error_file}")
        
        raise

if __name__ == '__main__':
    try:
        exp_logger.info("Starting hyperparameter tuning script...")
        results = run_bert_hypertuning()
        
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