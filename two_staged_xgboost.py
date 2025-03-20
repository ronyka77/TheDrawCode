"""
Two-Staged XGBoost Model for Binary Classification

This module implements a two-stage XGBoost classification pipeline:
    - Stage 1 (Stage1XGBClassifier): A base XGBoost classifier is trained on the raw features.
        It outputs the predicted probability for class 1, which is then used as the sole additional feature.
    - Stage 2 (Stage2XGBClassifier): A final XGBoost classifier is trained on the Stage 1 output.
        It undergoes hyperparameter and threshold optimization (with a recall constraint of 20%)
        aiming to boost the overall precision. Only the final Stage 2 model is logged to MLflow.
The joint hypertuning function optimizes a nested hyperparameter space:
    - A separate dictionary for Stage 1 hyperparameters.
    - A separate dictionary for Stage 2 hyperparameters.
    
Objective metric is Stage 2's precision provided yield recall is at least 20%.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import mlflow
import mlflow.models
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, log_loss
from utils.logger import ExperimentLogger
# Create a logger instance for internal logging (Stage 1 only)
experiment_name = "two_stage_xgboost"
log_dir = "logs/two_stage_xgboost"
logger = ExperimentLogger(experiment_name=experiment_name, log_dir=log_dir)

# Assume DataLoader is available from the existing module structure
from models.StackedEnsemble.shared.data_loader import DataLoader
from utils.create_evaluation_set import setup_mlflow_tracking

mlflow_tracking = setup_mlflow_tracking(experiment_name)

# Define minimum recall constraint for Stage 2
MIN_RECALL = 0.20
MIN_STAGE1_RECALL = 0.15
n_trials = 50
pip_requirements = ["xgboost==1.7.6", "optuna==3.3.0", "scikit-learn==1.3.2", "pandas==2.1.3", "numpy==1.26.1"]

base_params = {
            'objective': 'binary:logistic',
            'verbosity': 0,
            'nthread': -1,
            'seed': 19,
            'device': 'cpu',
            'tree_method': 'hist'
        }

###############################################
# Stage 1: Base XGBoost Classifier
###############################################
class Stage1XGBClassifier:
    def __init__(self, params):
        """
        Initialize Stage1XGBClassifier.
        
        Args:
            params (dict): Hyperparameters for Stage 1.
        """
        self.params = params
        self.optimal_threshold = 0.3
        self.model = xgb.XGBClassifier(**{**base_params, **self.params})
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the Stage 1 classifier.
        """
        if X_val is not None and y_val is not None:
            self.model.fit(X_train, y_train, 
                            eval_set=[(X_val, y_val)], 
                            verbose=False)
        else:
            self.model.fit(X_train, y_train)
        return self
    
    def optimize_threshold(self, X_val, y_val, min_threshold=0.1, max_threshold=0.9, step=0.01):
        """
        Optimize decision threshold on validation data.
        
        Args:
            X_val (array): Validation features.
            y_val (array): True labels.
            min_threshold (float): Minimum threshold to test.
            max_threshold (float): Maximum threshold to test.
            step (float): Step between thresholds.
            
        Returns:
            optimal_threshold (float), metrics (dict)
        """
        y_proba = self.predict_proba(X_val)
        thresholds = np.arange(min_threshold, max_threshold + step, step)
        best_thresh = self.optimal_threshold
        best_precision = 0.0
        best_metrics = {}
        
        for thresh in thresholds:
            preds = (y_proba >= thresh).astype(int)
            prec = precision_score(y_val, preds, zero_division=0)
            rec = recall_score(y_val, preds, zero_division=0)
            if rec >= MIN_STAGE1_RECALL and prec > best_precision:
                best_precision = prec
                best_thresh = thresh
                best_metrics = {
                    'precision': prec,
                    'recall': rec,
                    'f1': f1_score(y_val, preds, zero_division=0),
                    'log_loss': log_loss(y_val, np.column_stack((1-y_proba, y_proba)))
                }
                
        self.optimal_threshold = best_thresh
        return best_thresh, best_metrics
    
    def predict_proba(self, X):
        """
        Return predicted probabilities for class 1 as a 1D numpy array.
        """
        proba = self.model.predict_proba(X)[:, 1]
        return proba

###############################################
# Stage 2: Final XGBoost Classifier
###############################################
class Stage2XGBClassifier:
    def __init__(self, params):
        """
        Initialize Stage2XGBClassifier.
        
        Args:
            params (dict): Hyperparameters for Stage 2.
        """
        self.params = params
        self.model = xgb.XGBClassifier(**{**base_params, **self.params})
        self.optimal_threshold = 0.3
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train Stage 2 classifier on the provided data.
        """
        
        if X_val is not None and y_val is not None:
            self.model.fit(X_train, y_train, 
                            eval_set=[(X_val, y_val)], 
                            verbose=False)
        else:
            self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X):
        """
        Return predicted probabilities for class 1 as a 1D numpy array.
        """
        proba = self.model.predict_proba(X)[:, 1]
        return proba

    def optimize_threshold(self, X_val, y_val, min_threshold=0.1, max_threshold=0.9, step=0.01):
        """
        Optimize decision threshold on validation data.
        
        Args:
            X_val (array): Validation features.
            y_val (array): True labels.
            min_threshold (float): Minimum threshold to test.
            max_threshold (float): Maximum threshold to test.
            step (float): Step between thresholds.
            
        Returns:
            optimal_threshold (float), metrics (dict)
        """
        y_proba = self.predict_proba(X_val)
        thresholds = np.arange(min_threshold, max_threshold + step, step)
        best_thresh = self.optimal_threshold
        best_precision = 0.0
        best_metrics = {}
        
        for thresh in thresholds:
            preds = (y_proba >= thresh).astype(int)
            prec = precision_score(y_val, preds, zero_division=0)
            rec = recall_score(y_val, preds, zero_division=0)
            if rec >= MIN_RECALL and prec > best_precision:
                best_precision = prec
                best_thresh = thresh
                best_metrics = {
                    'precision': prec,
                    'recall': rec,
                    'f1': f1_score(y_val, preds, zero_division=0),
                    'log_loss': log_loss(y_val, np.column_stack((1-y_proba, y_proba)))
                }
                
        self.optimal_threshold = best_thresh
        return best_thresh, best_metrics
    
    def predict(self, X):
        """
        Predict binary labels using the optimized threshold.
        """
        proba = self.predict_proba(X)
        return (proba >= self.optimal_threshold).astype(int)

###############################################
# Joint Hyperparameter Tuning Function
###############################################
def joint_hypertune(X_train, y_train, X_test, y_test, X_eval, y_eval, n_trials=50):
    """
    Joint hypertuning objective for Stage 1 and Stage 2 classifiers.
    The objective:
        - Train Stage 1 with a given hyperparameter set.
        - Evaluate Stage 1 (log metrics via ExperimentLogger).
        - Generate Stage 1 predictions and use them as the only feature for Stage 2.
        - Train Stage 2, perform threshold optimization.
        - Return Stage 2 precision (if recall >= MIN_RECALL) as the objective metric.
    Args:
        X_train (DataFrame/array): Training features.
        y_train (array): Training labels.
        X_test (DataFrame/array): Test features.
        y_test (array): Test labels.
        X_eval (DataFrame/array): Evaluation features.
        y_eval (array): Evaluation labels.
        n_trials (int): Number of trials for Optuna.
        
    Returns:
        best_params (dict): Nested best hyperparameters for both stages.
        study (optuna.Study): The Optuna study object.
    """
    best_joint_score = 0.0
    best_joint_params = {}
    joint_trials = []
    def objective(trial):
        # Define hyperparameter spaces for each stage:
        # Stage 1 hyperparameters:
        stage1_params = {
            'learning_rate': trial.suggest_float('stage1_learning_rate', 0.005, 0.05, step=0.005),
            'max_depth': trial.suggest_int('stage1_max_depth', 3, 8, step=1),
            'early_stopping_rounds': trial.suggest_int('stage1_early_stopping_rounds', 50, 400, step=10),
            'reg_alpha': trial.suggest_float('stage1_reg_alpha', 0.05, 1.0, step=0.05),
            'reg_lambda': trial.suggest_float('stage1_reg_lambda', 0.05, 1.0, step=0.05),
            'eval_metric':  ['aucpr', 'error']
        }
        
        # Stage 2 hyperparameters:
        stage2_params = {
            'learning_rate': trial.suggest_float('stage2_learning_rate', 0.005, 0.05, step=0.005),
            'max_depth': trial.suggest_int('stage2_max_depth', 5, 10, step=1),
            'early_stopping_rounds': trial.suggest_int('stage2_early_stopping_rounds', 400, 1200, step=20),
            'reg_alpha': trial.suggest_float('stage2_reg_alpha', 10.0, 70.0, step=0.1),
            'reg_lambda': trial.suggest_float('stage2_reg_lambda', 1.0, 8.0, step=0.01),
            'eval_metric':  ['aucpr', 'error', 'logloss']
        }
        
        # Train Stage 1
        stage1 = Stage1XGBClassifier(stage1_params)
        stage1.train(X_train, y_train, X_test, y_test)
        # Evaluate Stage 1 using a lower threshold to favor high recall
        stage1_opt_thresh, stage1_metrics = stage1.optimize_threshold(X_eval, y_eval)
        stage1_preds = (stage1.predict_proba(X_eval) >= stage1_opt_thresh).astype(int)
        stage1_recall = stage1_metrics.get('recall', 0.0)
        stage1_precision = stage1_metrics.get('precision', 0.0)
        logger.info(f"Stage1 - Trial {trial.number} - Optimized Eval threshold {stage1_opt_thresh:.2f} - Precision: {stage1_precision:.4f}, Recall: {stage1_recall:.4f}")
        # Generate Stage 1 predictions and add them as new columns to the original DataFrames for Stage 2.
        X_train_stage2 = X_train.copy()
        X_train_stage2['stage1_pred'] = stage1.predict_proba(X_train).flatten()
        X_test_stage2 = X_test.copy()
        X_test_stage2['stage1_pred'] = stage1.predict_proba(X_test).flatten()
        X_eval_stage2 = X_eval.copy()
        X_eval_stage2['stage1_pred'] = stage1.predict_proba(X_eval).flatten()
        
        # Train Stage 2 on Stage 1 predictions.
        stage2 = Stage2XGBClassifier(stage2_params)
        stage2.train(X_train_stage2, y_train, X_test_stage2, y_test)
        
        # Optimize threshold for Stage 2.
        opt_thresh, stage2_metrics = stage2.optimize_threshold(X_eval_stage2, y_eval, min_threshold=0.1, max_threshold=0.9, step=0.01)
        trial.set_user_attr('stage2_threshold', opt_thresh)
        trial.set_user_attr('stage2_precision', stage2_metrics.get('precision', 0.0))
        trial.set_user_attr('stage2_recall', stage2_metrics.get('recall', 0.0))
        objective_value = stage2_metrics.get('precision', 0.0)
        return objective_value
    def stage2_callback(study, trial):
        """
        Callback function for logging Stage 2 performance metrics.
        Utilizes error handling to ensure that if any metric is missing,
        the corresponding value defaults to 0.0.
        """
        try:
            # Safely retrieve the Stage 2 metrics from trial.user_attrs, defaulting to 0.0 if not present.
            stage2_precision = trial.user_attrs.get('stage2_precision', 0.0)
            stage2_recall = trial.user_attrs.get('stage2_recall', 0.0)
            stage2_threshold = trial.user_attrs.get('stage2_threshold', 0.0)
            logger.info(f"Trial {trial.number} - Stage 2 Precision: {stage2_precision:.4f}, Recall: {stage2_recall:.4f}, Threshold: {stage2_threshold:.4f}")
        except Exception as e:
            logger.error(f"Error in stage2_callback for trial {trial.number}: {e}")
            # In case of any exception, log default values.
            logger.info(f"Trial {trial.number} - Stage 2 Precision: 0.0000, Recall: 0.0000, Threshold: 0.0000")
    def joint_trial_callback(study, trial):
        """
        Callback function for logging joint trial summary.
        Logs trial number, objective value, joint parameters (for Stage 1 and Stage 2),
        and Stage 2 performance metrics in a tabular format every 9 trials.
        """
        try:
            # Update and log the best joint trial using file_context_0 structure
            nonlocal best_joint_score
            nonlocal best_joint_params
            nonlocal joint_trials
            
            logger.info(f"Current best joint score: {best_joint_score:.4f}")
            if trial.value > best_joint_score:
                best_joint_score = trial.value
                best_joint_params = trial.params
                logger.info(f"New best joint score found in trial {trial.number}: {best_joint_score:.4f}")
            
            # Construct a record for the current joint trial including Stage 1 and Stage 2 parameters & Stage 2 metrics
            current_trial = {
                'trial_number': trial.number,
                'score': trial.value,
                'stage1_params': {
                    'learning_rate': trial.params.get('stage1_learning_rate'),
                    'max_depth': trial.params.get('stage1_max_depth'),
                    'n_estimators': trial.params.get('stage1_n_estimators'),
                    'early_stopping_rounds': trial.params.get('stage1_early_stopping_rounds'),
                    'reg_alpha': trial.params.get('stage1_reg_alpha'),
                    'reg_lambda': trial.params.get('stage1_reg_lambda'),
                    'eval_metric': trial.params.get('stage1_eval_metric')
                },
                'stage2_params': {
                    'learning_rate': trial.params.get('stage2_learning_rate'),
                    'max_depth': trial.params.get('stage2_max_depth'),
                    'n_estimators': trial.params.get('stage2_n_estimators'),
                    'early_stopping_rounds': trial.params.get('stage2_early_stopping_rounds'),
                    'reg_alpha': trial.params.get('stage2_reg_alpha'),
                    'reg_lambda': trial.params.get('stage2_reg_lambda'),
                    'eval_metric': trial.params.get('stage2_eval_metric')
                },
                'stage2_metrics': {
                    'threshold': trial.user_attrs.get('stage2_threshold', 0.0),
                    'precision': trial.user_attrs.get('stage2_precision', 0.0),
                    'recall': trial.user_attrs.get('stage2_recall', 0.0)
                }
            }
            
            # Append the current trial, sort the joint trials, and keep only the top 10
            joint_trials.append(current_trial)
            joint_trials.sort(key=lambda t: t['score'], reverse=True)
            joint_trials = joint_trials[:10]
            
            # Log a summary table every 9 trials
            if trial.number % 9 == 0:
                header = "| Rank | Trial # | Score | Stage1 Params | Stage2 Params | Stage2 Threshold | Stage2 Precision | Stage2 Recall |"
                separator = "|------|---------|-------|---------------|---------------|------------------|------------------|---------------|"
                logger.info("Joint Trial Update:")
                logger.info(header)
                logger.info(separator)
                for i, entry in enumerate(joint_trials):
                    row = (
                        f"| {i+1} | {entry['trial_number']} | {entry['score']:.4f} | "
                        f"{entry['stage1_params']} | {entry['stage2_params']} | "
                        f"{entry['stage2_metrics']['threshold']:.4f} | {entry['stage2_metrics']['precision']:.4f} | "
                        f"{entry['stage2_metrics']['recall']:.4f} |"
                    )
                    logger.info(row)
        except Exception as e:
            logger.error(f"Error in joint_trial_callback for trial {trial.number}: {e}")

    study = optuna.create_study(direction='maximize', 
        sampler=optuna.samplers.TPESampler(
            prior_weight=0.4,
            n_startup_trials=20,
            warn_independent_sampling=False,
            multivariate=True
        ))
    study.optimize(objective, 
        n_trials=n_trials, 
        timeout=3600,
        callbacks=[stage2_callback, joint_trial_callback]
    )
    best_params = study.best_trial.params
    
    logger.info(f"Best Joint Parameters: {best_params}")
    return best_params, study

###############################################
# Final Model Training & MLflow Logging
###############################################
def main():
    # Load data via existing DataLoader (assumed to return train and validation splits)
    dataloader = DataLoader()
    X_train, y_train, X_test, y_test, X_eval, y_eval = dataloader.load_data()  # Adjust according to your DataLoader's signature
    
    # Run joint hypertuning
    best_params, study = joint_hypertune(X_train, y_train, X_test, y_test, X_eval, y_eval, n_trials=n_trials)
    
    # Retrain the whole two-stage model using the best hyperparameters:
    # Stage 1 Training
    stage1 = Stage1XGBClassifier(best_params['stage1'])
    stage1.train(X_train, y_train, X_test, y_test)
    # Generate predictions for Stage 2 training
    X_train_stage2 = np.array(stage1.predict_proba(X_train)).reshape(-1, 1)
    X_test_stage2 = np.array(stage1.predict_proba(X_test)).reshape(-1, 1)
    X_eval_stage2 = np.array(stage1.predict_proba(X_eval)).reshape(-1, 1)
    
    # Stage 2 Training
    stage2 = Stage2XGBClassifier(best_params['stage2'])
    stage2.train(X_train_stage2, y_train, X_test_stage2, y_test)
    optimal_threshold, final_metrics = stage2.optimize_threshold(X_eval_stage2, y_eval)
    
    # Log final model and metrics to MLflow (only Stage 2 is logged)
    with mlflow.start_run(run_name=f"two_stage_xgb_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        mlflow.log_param("optimal_threshold", optimal_threshold)
        for metric_name, metric_value in final_metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        # Infer model signature using Stage 2 predictions on validation data
        input_example = X_eval_stage2.iloc[:5].copy() if hasattr(X_eval_stage2, 'iloc') else X_eval_stage2[:5].copy()
        if hasattr(input_example, 'dtypes'):
            for col in input_example.columns:
                if input_example[col].dtype.kind == 'i':
                    logger.info(f"Converting integer column '{col}' to float64 to handle potential missing values")
                    input_example[col] = input_example[col].astype('float64')

        signature = mlflow.models.infer_signature(input_example, stage2.model.predict(input_example))
        logger.info("Model signature created - check logs for any warnings about integer columns")
        mlflow.xgboost.log_model(
            stage2.model,
            artifact_path="model",
            pip_requirements=pip_requirements,  # Explicitly set requirements
            registered_model_name=f"two_stage_xgb_{datetime.now().strftime('%Y%m%d_%H%M')}",
            signature=signature
        )
        mlflow.end_run()
    
if __name__ == "__main__":
    main() 