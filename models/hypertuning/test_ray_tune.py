"""
Simple test script to verify Ray Tune installation.
"""

import ray
from ray import tune
import os

def test_ray_tune():
    """Basic test function for Ray Tune."""
    print("Testing Ray Tune installation...")
    
    # Simple objective function
    def objective(config):
        return {"score": config["x"] ** 2}
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=2)
    
    # Run a simple optimization
    analysis = tune.run(
        objective,
        config={
            "x": tune.uniform(0, 10)
        },
        num_samples=2
    )
    
    print("Ray Tune test successful!")
    print(f"Best config: {analysis.get_best_config(metric='score', mode='min')}")
    
    ray.shutdown()

if __name__ == "__main__":
    test_ray_tune() 