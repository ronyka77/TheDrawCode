import sys
import os
from pathlib import Path
import optuna
import mlflow
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings

# Add project root to Python path
try:
    project_root = Path(__file__).parent.parent.parent
    if not project_root.exists():
        # Handle network path by using raw string
        project_root = Path(r"\\".join(str(project_root).split("\\")))
    sys.path.append(str(project_root))
    print(f"Project root catboost_model: {project_root}")
except Exception as e:
    print(f"Error setting project root path: {e}")
    sys.path.append(os.getcwd().parent.parent)
    print(f"Current directory catboost_model: {os.getcwd().parent.parent}")

# Local imports
from utils.logger import ExperimentLogger
experiment_name = "multi_model_hypertuning"
logger = ExperimentLogger(experiment_name=experiment_name, log_dir='./logs/multi_model_hypertuning')

from utils.create_evaluation_set import create_ensemble_evaluation_set, import_selected_features_ensemble, import_training_data_ensemble, setup_mlflow_tracking
mlruns_dir = setup_mlflow_tracking(experiment_name)

class MultiModelHypertuner:
    def __init__(self, logger=None):
        self.logger = logger if logger is not None else ExperimentLogger(
            experiment_name="multi_model_hypertuning", 
            log_dir="./logs/multi_model_hypertuning"
        )

    def tune_xgb(self, X_train, y_train, X_val, y_val, X_test, y_test, n_trials=15):
        def objective(trial):
            param = {
                'objective': 'binary:logistic',
                'tree_method': 'hist',
                'device': 'cpu',
                'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.05),  # Tighten around optimal 0.037
                'max_depth': trial.suggest_int('max_depth', 5, 7),  # Best was 5, explore nearby
                'n_estimators': trial.suggest_int('n_estimators', 1500, 2000),  # Optimal was 1765
                'min_child_weight': trial.suggest_int('min_child_weight', 100, 200),  # Best was 150
                'gamma': trial.suggest_float('gamma', 0.05, 0.15),  # Optimal 0.078 range
                'subsample': trial.suggest_float('subsample', 0.3, 0.5),  # Best was 0.372
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.9, 1.0),  # Optimal 0.96
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.5, 3.0),  # Best was 2.19
                'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 100, 200)  # Optimal 138
            }
            model = XGBClassifier(**param)

            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], 
                     verbose=False)
            preds = model.predict(X_val)
            precision = precision_score(y_val, preds, pos_label=1, zero_division=0)
            recall = recall_score(y_val, preds, pos_label=1, zero_division=0)
            self.logger.info(f"XGB precision: {precision}, recall: {recall}")
            # Modified scoring to prioritize recall
            return 0.7 * precision + 0.3 * recall  # More balanced weighting

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        self.logger.info(f"XGB tuning completed: Best precision: {study.best_value}")
        return study.best_params

    def tune_cat(self, X_train, y_train, X_val, y_val, X_test, y_test, n_trials=5):
        def objective(trial):
            param = {
                'loss_function': 'Logloss',
                'eval_metric': trial.suggest_categorical('eval_metric', ['Logloss', 'AUC', 'F1']),  # Test 3 common metrics
                'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 200, 700),
                'learning_rate': trial.suggest_float('learning_rate', 0.003, 0.05, log=True),
                'depth': trial.suggest_int('depth', 6, 9),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 12, log=True),
                'iterations': trial.suggest_int('iterations', 2500, 5000),
                'border_count': trial.suggest_int('border_count', 128, 224, step=32),
                'subsample': trial.suggest_float('subsample', 0.7, 0.8),
                'random_strength': trial.suggest_float('random_strength', 0.1, 5, log=True),
                'auto_class_weights': 'Balanced',
                'grow_policy': 'SymmetricTree',
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 40, 60),
                'verbose': False
            }
            model = CatBoostClassifier(**param)
            model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
            
            preds = model.predict(X_val)
            precision = precision_score(y_val, preds, pos_label=1, zero_division=0)
            recall = recall_score(y_val, preds, pos_label=1, zero_division=0)
            self.logger.info(f"CatBoost precision: {precision}, recall: {recall}")
            # Early prune if recall is too low
            if recall < 0.2:
                raise optuna.exceptions.TrialPruned()
                
            return 0.7 * precision + 0.3 * recall

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        self.logger.info(f"CatBoost tuning completed: Best precision: {study.best_value}")
        mlflow.log_params({f"cat_{k}": v for k, v in study.best_params.items()})
        
        return study.best_params

    def tune_lgbm(self, X_train, y_train, X_val, y_val, X_test, y_test, n_trials=15):
        def objective(trial):

            lgbm_params = {
                'objective': 'binary',
                'metric': ['focal_loss', 'auc'],
                'eval_metric': 'focal_loss',
                'verbose': -1,
                'random_state': 42,
                'boosting_type': 'gbdt',
                'device_type': 'cpu',
                'learning_rate': trial.suggest_float('learning_rate', 0.002, 0.05, log=True),
                'max_depth': trial.suggest_int('max_depth', 6, 9),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 15, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 2000, 5000),
                'num_leaves': trial.suggest_int('num_leaves', 31, 127),
                'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 50, 300),
                'subsample': trial.suggest_float('subsample', 0.7, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.8, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            }
            
            # Calculate class weights for imbalanced data
            # class_weights = {0: 1, 1: len(np.where(y_train == 0)[0])/len(np.where(y_train == 1)[0])}
            class_weights = {0: 1, 1: 2.19}
            model = LGBMClassifier(**lgbm_params, class_weight=class_weights)
            model.fit(X_train, y_train, eval_set=(X_test, y_test))
            

            preds = model.predict(X_val)
            precision = precision_score(y_val, preds, pos_label=1, zero_division=0)
            recall = recall_score(y_val, preds, pos_label=1, zero_division=0)
            self.logger.info(f"LightGBM precision: {precision}, recall: {recall}")
            # Early prune if recall is too low
            if recall < 0.2:

                raise optuna.exceptions.TrialPruned()
                
            return 0.7 * precision + 0.3 * recall

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        self.logger.info(f"LightGBM tuning completed: Best precision: {study.best_value}")
        mlflow.log_params({f"lgbm_{k}": v for k, v in study.best_params.items()})
        
        return study.best_params

    def tune_rf(self, X_train, y_train, X_val, y_val, X_test, y_test, n_trials=15):
        """Tune Random Forest hyperparameters using Optuna with precision optimization, leveraging all CPU cores for faster evaluation."""
        def objective(trial):

            rf_params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_float('max_features', 0.1, 1.0),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                # 'class_weight': {0: 1, 1: 5},
                'class_weight': {0: 1, 1: 2.19},
                'random_state': 42,
                'n_jobs': -1  # Use all available CPU cores for faster training


            }
            
            model = RandomForestClassifier(**rf_params)
            model.fit(X_train, y_train)
            
            preds = model.predict(X_val)
            precision = precision_score(y_val, preds, pos_label=1, zero_division=0)
            recall = recall_score(y_val, preds, pos_label=1, zero_division=0)
            self.logger.info(f"Random Forest precision: {precision}, recall: {recall}")
            # Early prune if recall is too low
            if recall < 0.2:

                raise optuna.exceptions.TrialPruned()
                
            return 0.7 * precision + 0.3 * recall
 
        study = optuna.create_study(direction="maximize")
        # Optimize using all available CPU cores in parallel
        study.optimize(objective, n_trials=n_trials, n_jobs=-1)
        
        self.logger.info(f"Random Forest tuning completed: Best precision: {study.best_value}")
        mlflow.log_params({f"rf_{k}": v for k, v in study.best_params.items()})
        
        return study.best_params

    def tune_knn(self, X_train, y_train, X_val, y_val, X_test, y_test, n_trials=5):
        """Tune K-Nearest Neighbors hyperparameters using Optuna with precision optimization.
           Updated parameter ranges and pruning threshold to reduce excessive pruning."""
        def objective(trial):
            knn_params = {
                'n_neighbors': trial.suggest_int('n_neighbors', 1, 50),  # Expanded range: lower bound reduced to 1
                'weights': trial.suggest_categorical('weights', ['distance', 'uniform']),
                'algorithm': trial.suggest_categorical('algorithm', ['ball_tree', 'brute']),  # Removed kd_tree for consistency
                'leaf_size': trial.suggest_int('leaf_size', 10, 40),
                'p': trial.suggest_int('p', 1, 2),
                'n_jobs': -1
            }
            # Use the full training data leveraging all available processors
            model = KNeighborsClassifier(**knn_params)
            model.fit(X_train, y_train)
            
            preds = model.predict(X_val)
            precision = precision_score(y_val, preds, pos_label=1, zero_division=0)
            recall = recall_score(y_val, preds, pos_label=1, zero_division=0)
            self.logger.info(f"KNN precision: {precision}, recall: {recall}")
            
            # Relaxed pruning threshold
            if recall < 0.15:  # From 0.2 to 0.15
                raise optuna.exceptions.TrialPruned()
                
            return 0.7 * precision + 0.3 * recall

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        best_params = {}
        try:
            best_precision = study.best_value
            best_params = study.best_params
            self.logger.info(f"KNN tuning completed: Best precision: {best_precision}")
            mlflow.log_params({f"knn_{k}": v for k, v in best_params.items()})
        except ValueError:
            self.logger.warning("KNN tuning did not complete any trial. No completed trials available.")
            mlflow.log_params({"knn_tuning": "no_completed_trials"})
        
        return best_params

    def tune_svm(self, X_train, y_train, X_val, y_val, X_test, y_test, n_trials=5):
        def objective(trial):
            svm_params = {
                'C': trial.suggest_float('C', 0.5, 2.0),
                'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf']),  # Linear first for speed
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'class_weight': {0: 1, 1: 2.19},
                'shrinking': True,
                'tol': trial.suggest_float('tol', 1e-2, 1e-1),  # Looser tolerance for faster convergence
                'cache_size': 12000,  # 12GB cache (75% of 16GB)
                'probability': True,
                'max_iter': 1000,  # Hard limit for safety
                'decision_function_shape': 'ovr'
            }

            # Remove subsampling - use full dataset with large cache
            model = SVC(**svm_params)
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_train, y_train)  # Use full training data
            except Exception as e:
                self.logger.warning(f"SVM fitting failed: {e}")
                raise optuna.exceptions.TrialPruned()
            
            preds = model.predict(X_val)
            precision = precision_score(y_val, preds, pos_label=1, zero_division=0)
            recall = recall_score(y_val, preds, pos_label=1, zero_division=0)
            self.logger.info(f"SVM precision: {precision}, recall: {recall}")

            # Use F2-score to prioritize recall
            f2_score = (5 * precision * recall) / (4 * precision + recall + 1e-8)
            score = 0.7 * precision + 0.3 * recall
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        self.logger.info(f"SVM tuning completed: Best score: {study.best_value}")
        mlflow.log_params({f"svm_{k}": v for k, v in study.best_params.items()})
        
        return study.best_params

    def tune_all_models(self, X_train, y_train, X_val, y_val, X_test, y_test, n_trials=10):
        tuning_results = {}
   
        # Log all results to MLflow
        with mlflow.start_run(nested=True):
            tuning_results['xgb'] = self.tune_xgb(X_train, y_train, X_val, y_val, X_test, y_test)
            tuning_results['cat'] = self.tune_cat(X_train, y_train, X_val, y_val, X_test, y_test, n_trials)
            tuning_results['lgbm'] = self.tune_lgbm(X_train, y_train, X_val, y_val, X_test, y_test)
            tuning_results['rf'] = self.tune_rf(X_train, y_train, X_val, y_val, X_test, y_test)
            tuning_results['knn'] = self.tune_knn(X_train, y_train, X_val, y_val, X_test, y_test)
            tuning_results['svm'] = self.tune_svm(X_train, y_train, X_val, y_val, X_test, y_test, n_trials)
        
        for model_type, params in tuning_results.items():
            mlflow.log_params({f"{model_type}_{k}": v for k, v in params.items()})
        
        return tuning_results 

if __name__ == "__main__":
    hypertuner = MultiModelHypertuner()
    X_train, y_train, X_test, y_test = import_training_data_ensemble()
    X_val, y_val = create_ensemble_evaluation_set()
    selected_features = import_selected_features_ensemble('all')
    X_train = X_train[selected_features]
    X_val = X_val[selected_features]
    X_test = X_test[selected_features]

    hypertuner.tune_all_models(X_train, y_train, X_val, y_val, X_test, y_test)
