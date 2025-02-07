import optuna
from optuna import Trial, Study
from optuna.samplers import TPESampler
import mlflow
import numpy as np
from sklearn.metrics import precision_recall_curve
from optuna.pruners import ThresholdPruner
from imblearn.under_sampling import RandomUnderSampler
from pathlib import Path
import os
import sys

# Add project root to Python path
try:

    project_root = Path(__file__).parent.parent.parent
    if not project_root.exists():
        # Handle network path by using raw string
        project_root = Path(r"\\".join(str(project_root).split("\\")))

    sys.path.append(str(project_root))
    print(f"Project root ensemble_model: {project_root}")
except Exception as e:
    print(f"Error setting project root path: {e}")
    # Fallback to current directory if path resolution fails
    sys.path.append(os.getcwd().parent.parent)
    print(f"Current directory ensemble_model: {os.getcwd().parent.parent}")
    

os.environ['GIT_PYTHON_GIT_EXECUTABLE'] = "C:/Program Files/Git/bin/git.exe"

# Local imports
from utils.logger import ExperimentLogger
experiment_name = "ensemble_hypertuning"
logger = ExperimentLogger(experiment_name=experiment_name, log_dir='./logs/ensemble_hypertuning')
from utils.create_evaluation_set import create_ensemble_evaluation_set, import_selected_features_ensemble, import_training_data_ensemble, setup_mlflow_tracking
mlruns_dir = setup_mlflow_tracking(experiment_name)

from models.ensemble_model import EnsembleModel  # Your existing ensemble class

class EnsembleHyperTuner:
    def __init__(self, logger, selected_features, target_f2=0.5, n_trials=200):
        self.logger = logger
        self.selected_features = selected_features
        self.target_f2 = target_f2
        self.n_trials = n_trials
        self.best_params = None

    def tune(self, X_train, y_train, X_val, y_val):
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42, n_startup_trials=20)
        )
        
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, X_val, y_val),
            n_trials=self.n_trials,
            callbacks=[self._logging_callback]
        )
        
        self.best_params = study.best_params
        return study.best_params

    def objective(self, trial, X_train, y_train, X_val, y_val):
        # Suggest model weights
        if trial.number > 20:  # After initial exploration
            prev_weights = [t.params.get('xgb_weight', 1.0) for t in trial.study.trials]
            mean_weight = np.mean(prev_weights)
            weights = [
                trial.suggest_float('xgb_weight', max(0.3, mean_weight-0.5), min(2.5, mean_weight+0.5)),
                trial.suggest_float('cat_weight', 0.5, 2.5), 
                trial.suggest_float('lgbm_weight', 0.5, 2.5),
                trial.suggest_float('rf_weight', 0.1, 1.5),
                trial.suggest_float('knn_weight', 0.1, 1.0),
                trial.suggest_float('svm_weight', 0.1, 1.2)
            ]
        else:
            weights = [
                trial.suggest_float('xgb_weight', 0.5, 2.5),
                trial.suggest_float('cat_weight', 0.5, 2.5), 
                trial.suggest_float('lgbm_weight', 0.5, 2.5),
                trial.suggest_float('rf_weight', 0.1, 1.5),
                trial.suggest_float('knn_weight', 0.1, 1.0),
                trial.suggest_float('svm_weight', 0.1, 1.2)
            ]
        
        # Suggest feature subset selection
        n_features = trial.suggest_int('n_features', 15, 40)
        selected_features = self.selected_features
        
        # Modify training data accordingly
        # X_train_sub = self._select_features(X_train, selected_features, n_features)
        
        # Suggest calibration parameters
        calibrate_xgb = trial.suggest_categorical('calibrate_xgb', [True, False])
        calibrate_cat = trial.suggest_categorical('calibrate_cat', [True, False])
        
        # New model parameters
        trial.suggest_categorical('rf_max_depth', [5, 10, None])
        trial.suggest_int('knn_n_neighbors', 5, 50)
        trial.suggest_float('svm_C', 0.1, 10, log=True)
        
        # Update models with suggested params
        self.base_models[3][1].set_params(
            max_depth=trial.params['rf_max_depth']
        )
        self.base_models[4][1].set_params(
            n_neighbors=trial.params['knn_n_neighbors']
        )
        self.base_models[5][1].set_params(
            C=trial.params['svm_C']
        )
        
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        # Add intermediate reporting
        for epoch in range(10):
            partial_fit(epoch)
            trial.report(epoch_score, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        # Build ensemble with suggested params
        ensemble = EnsembleModel(
            logger=self.logger,
            weights=weights,
            calibrate=[calibrate_xgb, calibrate_cat, False],
            calibration_method="sigmoid"
        )
        ensemble.train(X_train, y_train, X_val, y_val)
        
        # Get predictions and optimize F2 score
        y_proba = ensemble.predict_proba(X_val)
        threshold, metrics = self._find_optimal_threshold(y_val, y_proba)
        
        # Store threshold as trial user attribute
        trial.set_user_attr("optimal_threshold", threshold)
        
        return metrics['f2'] 

    def _resample_data(self, X, y, trial):
        """Dynamic resampling based on trial parameters"""
        sampler_ratio = trial.suggest_float('sampler_ratio', 0.3, 0.7)
        sampler = RandomUnderSampler(sampling_strategy=sampler_ratio)
        return sampler.fit_resample(X, y) 

    def _logging_callback(self, study: Study, trial: Trial):
        self.logger.info(
            f"Trial {trial.number} finished with F2: {trial.value:.4f} "
            f"(Threshold: {trial.user_attrs.get('optimal_threshold', 0.5):.3f})"
        )
        
        if study.best_trial.number == trial.number:
            mlflow.log_params(trial.params)
            mlflow.log_param("optimal_threshold", trial.user_attrs["optimal_threshold"])
            mlflow.log_metrics({
                "best_precision": trial.user_attrs.get("precision", 0),
                "best_recall": trial.user_attrs.get("recall", 0)
            })

    def _find_optimal_threshold(self, y_true, y_proba):
        """Find threshold that maximizes F2 score (recall-focused)"""
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
        f2_scores = (5 * precisions * recalls) / (4 * precisions + recalls + 1e-8)
        optimal_idx = np.argmax(f2_scores)
        return thresholds[optimal_idx], {
            'threshold': thresholds[optimal_idx],
            'precision': precisions[optimal_idx],
            'recall': recalls[optimal_idx],
            'f2': f2_scores[optimal_idx]
        } 
        
if __name__ == "__main__":
    # Load data with selected features
    X_train, y_train, X_test, y_test = import_training_data_ensemble()
    X_val, y_val = create_ensemble_evaluation_set()
    selected_columns = import_selected_features_ensemble('all')
    X_train = X_train[selected_columns]
    X_val = X_val[selected_columns]
    
    # Initialize hypertuner
    tuner = EnsembleHyperTuner(logger, selected_features=selected_columns, target_f2=0.5, n_trials=200)
    
    # Perform tuning
    best_params = tuner.tune(X_train, y_train, X_val, y_val)
    logger.info(f"Best parameters: {best_params}")  
    
