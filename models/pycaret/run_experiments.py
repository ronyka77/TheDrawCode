#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Soccer prediction experiment runner.

This script runs multiple experiments with different configurations
and compares the results.

Usage:
    python run_experiments.py --config-path "configs/experiments.yaml"
"""

import argparse
import logging
import os
import sys
import yaml
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import itertools

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import modules
from models.pycaret.main import main as run_pipeline
from utils.logger import ExperimentLogger

# Setup logging
logger = ExperimentLogger(experiment_name="soccer_prediction_experiments")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Soccer prediction experiment runner")
    parser.add_argument("--config-path", type=str, default="configs/experiments.yaml",
                        help="Path to experiment configuration file")
    parser.add_argument("--output-dir", type=str, default="experiment_results",
                        help="Directory to save experiment results")
    parser.add_argument("--parallel", action="store_true",
                        help="Run experiments in parallel")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Maximum number of parallel workers")
    return parser.parse_args()

def load_config(config_path):
    """
    Load experiment configuration.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise

def generate_experiment_configs(config):
    """
    Generate individual experiment configurations from a grid configuration.
    
    Args:
        config (dict): Grid configuration
        
    Returns:
        list: List of individual experiment configurations
    """
    base_config = config.get('base_config', {})
    grid_config = config.get('grid_config', {})
    
    # Generate parameter combinations
    param_names = list(grid_config.keys())
    param_values = list(grid_config.values())
    
    # Generate all combinations
    combinations = list(itertools.product(*param_values))
    
    # Create individual configs
    experiment_configs = []
    for i, combination in enumerate(combinations):
        # Create experiment config by combining base config with specific parameters
        exp_config = base_config.copy()
        
        # Add specific parameters
        for param_name, param_value in zip(param_names, combination):
            exp_config[param_name] = param_value
        
        # Add experiment ID
        exp_config['experiment_id'] = f"exp_{i+1}"
        
        experiment_configs.append(exp_config)
    
    logger.info(f"Generated {len(experiment_configs)} experiment configurations")
    return experiment_configs

def run_experiment(config, output_dir):
    """
    Run a single experiment with the given configuration.
    
    Args:
        config (dict): Experiment configuration
        output_dir (str): Directory to save results
        
    Returns:
        dict: Experiment results
    """
    experiment_id = config.get('experiment_id', f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    logger.info(f"Running experiment {experiment_id}")
    
    # Convert config to command line arguments
    args = argparse.Namespace()
    for key, value in config.items():
        setattr(args, key, value)
    
    # Set output directory
    experiment_output_dir = os.path.join(output_dir, experiment_id)
    os.makedirs(experiment_output_dir, exist_ok=True)
    args.output_dir = experiment_output_dir
    
    # Run pipeline
    try:
        result = run_pipeline(args)
        
        # Save experiment config and results
        config_path = os.path.join(experiment_output_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        result_path = os.path.join(experiment_output_dir, "results.json")
        with open(result_path, 'w') as f:
            # Convert any non-serializable objects to strings
            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
                    serializable_result[key] = value
                else:
                    serializable_result[key] = str(value)
            
            json.dump(serializable_result, f, indent=4)
        
        logger.info(f"Experiment {experiment_id} completed successfully")
        return {
            'experiment_id': experiment_id,
            'config': config,
            'result': result,
            'status': 'success'
        }
    except Exception as e:
        logger.error(f"Error in experiment {experiment_id}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Save error information
        error_path = os.path.join(experiment_output_dir, "error.txt")
        with open(error_path, 'w') as f:
            f.write(f"Error: {str(e)}\n\n")
            f.write(traceback.format_exc())
        
        return {
            'experiment_id': experiment_id,
            'config': config,
            'error': str(e),
            'status': 'error'
        }

def run_experiments_sequential(experiment_configs, output_dir):
    """
    Run experiments sequentially.
    
    Args:
        experiment_configs (list): List of experiment configurations
        output_dir (str): Directory to save results
        
    Returns:
        list: List of experiment results
    """
    results = []
    for config in experiment_configs:
        result = run_experiment(config, output_dir)
        results.append(result)
    return results

def run_experiments_parallel(experiment_configs, output_dir, max_workers=4):
    """
    Run experiments in parallel.
    
    Args:
        experiment_configs (list): List of experiment configurations
        output_dir (str): Directory to save results
        max_workers (int): Maximum number of parallel workers
        
    Returns:
        list: List of experiment results
    """
    from concurrent.futures import ProcessPoolExecutor
    
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_experiment, config, output_dir) for config in experiment_configs]
        for future in futures:
            result = future.result()
            results.append(result)
    
    return results

def summarize_results(results, output_dir):
    """
    Summarize experiment results.
    
    Args:
        results (list): List of experiment results
        output_dir (str): Directory to save summary
        
    Returns:
        pd.DataFrame: Summary dataframe
    """
    # Extract relevant information
    summary_data = []
    for result in results:
        experiment_id = result.get('experiment_id')
        config = result.get('config', {})
        status = result.get('status')
        
        # Extract metrics if available
        metrics = {}
        if status == 'success':
            result_data = result.get('result', {})
            if isinstance(result_data, dict):
                metrics = result_data.get('metrics', {})
        
        # Create summary row
        summary_row = {
            'experiment_id': experiment_id,
            'status': status
        }
        
        # Add configuration parameters
        for key, value in config.items():
            if key != 'experiment_id':
                summary_row[f"config_{key}"] = value
        
        # Add metrics
        for key, value in metrics.items():
            summary_row[f"metric_{key}"] = value
        
        summary_data.append(summary_row)
    
    # Create dataframe
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary
    summary_path = os.path.join(output_dir, "experiment_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    logger.info(f"Experiment summary saved to {summary_path}")
    
    return summary_df

def main():
    """Main function to run experiments."""
    args = parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = os.path.join(args.output_dir, f"experiments_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load configuration
        config = load_config(args.config_path)
        
        # Generate experiment configurations
        experiment_configs = generate_experiment_configs(config)
        
        # Save experiment configurations
        configs_path = os.path.join(output_dir, "experiment_configs.json")
        with open(configs_path, 'w') as f:
            json.dump(experiment_configs, f, indent=4)
        
        # Run experiments
        logger.info(f"Running {len(experiment_configs)} experiments")
        if args.parallel:
            logger.info(f"Running experiments in parallel with {args.max_workers} workers")
            results = run_experiments_parallel(experiment_configs, output_dir, args.max_workers)
        else:
            logger.info("Running experiments sequentially")
            results = run_experiments_sequential(experiment_configs, output_dir)
        
        # Summarize results
        summary_df = summarize_results(results, output_dir)
        
        # Log summary
        success_count = len([r for r in results if r.get('status') == 'success'])
        logger.info(f"Experiments completed: {len(results)} total, {success_count} successful")
        
        return {
            'success': True,
            'output_dir': output_dir,
            'summary': summary_df
        }
        
    except Exception as e:
        logger.error(f"Error in experiment runner: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'success': False,
            'error': str(e)
        }

if __name__ == "__main__":
    result = main()
    if result and result.get('success'):
        logger.info("Experiments executed successfully")
    else:
        logger.error("Experiment execution failed")
        sys.exit(1) 