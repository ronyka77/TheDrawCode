"""
Ensemble Model Package

This package implements a stacked ensemble model for soccer prediction
with dynamic weighting, calibration, and threshold optimization.
"""

__version__ = "2.1.0"
# Core class
from .ensemble_model import EnsembleModel

# Define public API
__all__ = [
    "EnsembleModel",
    "analyze_calibration",
    "calibrate_models",
    "analyze_prediction_errors",
    "explain_predictions",
    "evaluate_model",
    "create_meta_features",
    "run_ensemble",
    "tune_threshold",
    "compute_dynamic_weights",
]

# Utility functions (optional exports)
from .calibration import analyze_calibration as analyze_calibration
from .calibration import calibrate_models as calibrate_models
from .diagnostics import analyze_prediction_errors, explain_predictions
from .evaluation import evaluate_model as evaluate_model
from .meta_features import create_meta_features as create_meta_features

# Main execution
from .run_ensemble import run_ensemble
from .thresholds import tune_threshold as tune_threshold
from .weights import compute_dynamic_weights
