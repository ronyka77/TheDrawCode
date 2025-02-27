"""
Ensemble Model Package

This package implements a stacked ensemble model for soccer prediction
with dynamic weighting, calibration, and threshold optimization.
"""

__version__ = "2.1.0"

# Core class
from .ensemble_model import EnsembleModel

# Utility functions (optional exports)
from .calibration import calibrate_models, analyze_calibration
from .meta_features import create_meta_features
from .evaluation import evaluate_model
from .thresholds import tune_threshold
from .weights import compute_dynamic_weights
from .diagnostics import explain_predictions, analyze_prediction_errors

# Main execution
from .run_ensemble import run_ensemble
