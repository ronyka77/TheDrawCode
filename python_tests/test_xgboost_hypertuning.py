import pytest
import numpy as np
import pandas as pd
from models.hypertuning.xgboost_api_hypertuning import GlobalHypertuner
import optuna

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 1000
    X = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples)
    })
    y = (np.random.random(n_samples) > 0.8).astype(int)  # 20% positive class
    return X, y

def test_recall_threshold_enforcement(sample_data):
    """Test that trials with recall < 0.40 are pruned."""
    X, y = sample_data
    
    # Split data
    train_size = int(0.6 * len(X))
    val_size = int(0.2 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    hypertuner = GlobalHypertuner()
    study = optuna.create_study(direction="maximize")
    
    # Run a few trials
    n_trials = 5
    for _ in range(n_trials):
        trial = study.ask()
        try:
            value = hypertuner.objective(
                trial, X_train, y_train, X_val, y_val, X_test, y_test
            )
            study.tell(trial, value)
        except optuna.exceptions.TrialPruned:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
    
    # Check completed trials
    completed_trials = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
    for trial in completed_trials:
        assert trial.user_attrs["recall"] >= 0.40, \
            f"Trial completed with recall {trial.user_attrs['recall']} < 0.40"

def test_precision_optimization(sample_data):
    """Test that precision is optimized while maintaining recall threshold."""
    X, y = sample_data
    
    # Split data
    train_size = int(0.6 * len(X))
    val_size = int(0.2 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    hypertuner = GlobalHypertuner()
    
    # Run optimization
    best_params = hypertuner.tune_global_model(
        X_train, y_train, X_val, y_val, X_test, y_test,
        n_trials=10
    )
    
    # Get best trial
    study = hypertuner.study
    best_trial = study.best_trial
    
    # Assert recall threshold is met
    assert best_trial.user_attrs["recall"] >= 0.40, \
        "Best trial does not meet recall threshold"
    
    # Assert precision is optimized
    completed_trials = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
    best_precision = best_trial.user_attrs["precision"]
    
    for trial in completed_trials:
        if trial.number != best_trial.number:
            assert best_precision >= trial.user_attrs["precision"], \
                "Found trial with better precision than best trial" 