"""
Integration Script for Specialized FNN with PyTorch Hypertuner

This script demonstrates how to use the Specialized FNN model with the 
existing PyTorch hypertuner workflow for soccer prediction.

Usage:
    python specialized_fnn_integration.py
"""
# These are the key imports we'll override
from src.models.StackedEnsemble.base.neural.pytorch_hypertuner_20 import (
    create_pytorch_model,
    load_hyperparameter_space,
)
from src.models.StackedEnsemble.base.neural.pytorch_hypertuner_20 import main as original_main

# Import the specialized FNN components
from src.models.StackedEnsemble.base.neural.specialized_fnn import (
    SpecializedFNN,
    add_specialized_hyperparameters,
    create_specialized_fnn,
)
from src.utils.logger import ExperimentLogger

# Setup logger
logger = ExperimentLogger("specialized_fnn_experiment_20")

def patched_create_pytorch_model(*args, **kwargs):
    """
    Replace the standard model creation with our specialized FNN.
    """
    logger.info("Creating specialized FNN model instead of default PyTorch model")
    return create_specialized_fnn(*args, **kwargs)

def patched_load_hyperparameter_space():
    """
    Load the standard hyperparameter space and extend it with FNN-specific parameters.
    """
    # Get the original hyperparameter space
    original_space = load_hyperparameter_space()
    
    # Add our specialized FNN hyperparameters
    enhanced_space = add_specialized_hyperparameters(original_space)
    
    logger.info("Enhanced hyperparameter space with specialized FNN parameters")
    return enhanced_space

def main():
    """
    Main execution function that uses the specialized FNN with the existing workflow.
    """
    # Temporarily patch functions to use our specialized model
    import src.models.StackedEnsemble.base.neural.pytorch_hypertuner as hypertuner
    
    global logger, experiment_name
    experiment_name = "specialized_fnn_experiment_20"
    logger.info("Starting specialized FNN experiment")
    # Save original functions and variables
    original_create_model = hypertuner.create_pytorch_model
    original_load_space = hypertuner.load_hyperparameter_space
    original_experiment_name = hypertuner.experiment_name
    
    try:
        # Patch with our specialized versions
        hypertuner.create_pytorch_model = patched_create_pytorch_model
        hypertuner.load_hyperparameter_space = patched_load_hyperparameter_space
        
        # Set custom experiment name
        hypertuner.experiment_name = experiment_name
        
        # Now run the original main function which will use our specialized components
        logger.info(f"Running hypertuner with specialized FNN model using experiment name: {hypertuner.experiment_name}")
        original_main()
        
    finally:
        # Restore original functions and variables
        hypertuner.create_pytorch_model = original_create_model
        hypertuner.load_hyperparameter_space = original_load_space
        hypertuner.experiment_name = original_experiment_name
    
    logger.info("Specialized FNN experiment completed")

if __name__ == "__main__":
    main() 