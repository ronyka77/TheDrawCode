# Two-Stage XGBoost Model Implementation Guide

## Table of Contents
- [1. Introduction](#1-introduction)
- [2. Architecture Design](#2-architecture-design)
- [3. Implementation Details](#3-implementation-details)
- [4. Hyperparameter Optimization](#4-hyperparameter-optimization)
- [5. MLflow Integration](#5-mlflow-integration)
- [6. Evaluation Framework](#6-evaluation-framework)
- [7. Production Deployment](#7-production-deployment)
- [8. References](#8-references)

## 1. Introduction

This document outlines the implementation of a two-stage XGBoost model approach for soccer match prediction, targeting 50% precision and 20% recall. The approach uses a recall-optimized first stage followed by a precision-focused second stage filter.

### Objectives
- First stage: Achieve >40% recall with acceptable precision
- Second stage: Refine to 50% precision while maintaining >20% recall
- Ensure CPU-only training compatibility
- Implement robust logging and experiment tracking

## 2. Architecture Design

### 2.1 Overall Architecture

```
Training Data → [Stage 1: Recall Model] → High Recall Predictions → [Stage 2: Precision Model] → Final Predictions
```

### 2.2 First Stage Model (Recall-Optimized)

**Purpose:** Capture as many potential draws as possible (high recall)

**Key Configuration:**
```python
# First stage model configuration
first_stage_params = {
    'objective': 'binary:logistic',
    'tree_method': 'hist',  # CPU-optimized
    'learning_rate': 0.1,
    'max_depth': 7,
    'min_child_weight': 2,
    'scale_pos_weight': 3,  # Adjusted for recall focus
    'subsample': 0.85,
    'colsample_bytree': 0.8,
    'eval_metric': ['auc', 'logloss', 'error']
}

# Lower threshold for higher recall
first_stage_threshold = 0.3
```

### 2.3 Second Stage Model (Precision-Optimizer)

**Purpose:** Filter first-stage predictions to increase precision

**Key Configuration:**
```python
# Second stage model configuration
second_stage_params = {
    'objective': 'binary:logistic',
    'tree_method': 'hist',  # CPU-optimized
    'learning_rate': 0.03,
    'max_depth': 5,
    'min_child_weight': 4,  # Higher to prevent overfitting
    'scale_pos_weight': 1,  # Balanced for precision
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'eval_metric': ['auc', 'logloss', 'error']
}

# Higher threshold for better precision
second_stage_threshold = 0.6
```

## 3. Implementation Details

### 3.1 Data Preparation

```python
from utils.logger import ExperimentLogger
experiment_name="two_stage_xgboost"
# Initialize logger
logger = ExperimentLogger(
    experiment_name=experiment_name,
    log_dir="logs/two_stage_model"
)
from models.StackedEnsemble.shared.data_loader import DataLoader

# Log the start of the process
logger.start_run(run_name="data_preparation")
logger.log_params({"tree_method": "hist", "device": "cpu"})

# Create data loader and load data splits
data_loader = DataLoader(experiment_name=experiment_name)
X_train, y_train, X_test, y_test, X_val, y_val = data_loader.load_data()

# Store in appropriate variables for the two-stage model
train_data = X_train.copy()
train_data['is_draw'] = y_train
val_data = X_val.copy()
val_data['is_draw'] = y_val
test_data = X_test.copy()
test_data['is_draw'] = y_test

# Log data preparation completion
logger.log_metrics({
    "train_samples": len(train_data),
    "validation_samples": len(val_data),
    "test_samples": len(test_data),
    "positive_train_ratio": train_data['is_draw'].mean()
})
logger.end_run()
```

### 3.2 Feature Engineering

```python
# Stage 1 Features (Broader set for recall)
stage1_features = [
    # Core features
    'home_win_rate', 'away_win_rate', 'home_draw_rate', 'away_draw_rate',
    'home_goals_per_match', 'away_goals_per_match', 'home_xg', 'away_xg',
    
    # Form features
    'home_last5_points', 'away_last5_points', 'home_last5_goals_for', 'away_last5_goals_for',
    'home_last5_conceded', 'away_last5_conceded',
    
    # Historical matchup features
    'historical_draw_rate', 'last_encounter_draw',
    
    # League position features
    'position_difference', 'points_difference',
    
    # Temporal features
    'days_since_last_home_match', 'days_since_last_away_match',
    'matchweek', 'is_derby'
]

# Stage 2 Features (Precision-focused subset)
stage2_features = [
    # Core predictive features
    'home_win_rate', 'away_win_rate', 'home_draw_rate', 'away_draw_rate',
    'home_xg', 'away_xg',
    
    # First stage prediction
    'stage1_prediction', 'stage1_prediction_confidence',
    
    # Precision-enhancing features
    'goals_difference_consistency', 'xg_difference_consistency',
    'home_clean_sheet_rate', 'away_clean_sheet_rate',
    'home_defensive_stability', 'away_defensive_stability',
    
    # Historical precision indicators
    'similar_match_draw_rate', 'odds_implied_draw_probability'
]
```

### 3.3 First Stage Model Training

```python
import xgboost as xgb
import mlflow
from datetime import datetime

# Start logging for first stage
logger.start_run(run_name="stage1_training")

# Log feature set
logger.log_params({"num_features": len(stage1_features)})
logger.log_params(first_stage_params)

# Create DMatrix objects
dtrain = xgb.DMatrix(train_data[stage1_features], label=train_data['is_draw'])
dval = xgb.DMatrix(val_data[stage1_features], label=val_data['is_draw'])

# Setup MLflow tracking
with mlflow.start_run(run_name="stage1_model") as run:
    mlflow.log_params(first_stage_params)
    
    # Train with early stopping
    evals_result = {}
    model_stage1 = xgb.train(
        first_stage_params,
        dtrain,
        num_boost_round=5000,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=500,
        evals_result=evals_result,
        verbose_eval=100
    )
    
    # Log metrics
    for metric in ['auc', 'logloss', 'error']:
        mlflow.log_metric(f"val_{metric}", evals_result['val'][metric][-1])
    
    # Generate predictions
    val_preds = model_stage1.predict(dval)
    
    # Calculate recall-focused metrics
    from sklearn.metrics import precision_recall_curve, auc
    precision, recall, thresholds = precision_recall_curve(val_data['is_draw'], val_preds)
    
    # Find threshold that gives target recall
    target_recall = 0.4  # Aiming for 40%+ recall
    recall_threshold_idx = next(i for i, r in enumerate(recall) if r < target_recall)
    selected_threshold = thresholds[recall_threshold_idx]
    selected_precision = precision[recall_threshold_idx]
    
    # Log results
    mlflow.log_metric("selected_threshold", selected_threshold)
    mlflow.log_metric("selected_recall", recall[recall_threshold_idx])
    mlflow.log_metric("selected_precision", selected_precision)
    
    # Log model with signature
    signature = mlflow.models.infer_signature(
        val_data[stage1_features],
        model_stage1.predict(dval)
    )
    
    mlflow.xgboost.log_model(
        xgb_model=model_stage1,
        artifact_path="model_stage1",
        registered_model_name=f"xgboost_stage1_{datetime.now().strftime('%Y%m%d_%H%M')}",
        signature=signature
    )
    
    # Log feature importance
    feature_importance = model_stage1.get_score(importance_type='gain')
    mlflow.log_dict(feature_importance, "artifacts/feature_importance_stage1.json")

# End logging
logger.log_metrics({
    "stage1_recall": recall[recall_threshold_idx],
    "stage1_precision": selected_precision,
    "stage1_threshold": selected_threshold
})
logger.end_run()
```

### 3.4 Second Stage Model Training

```python
# Generate first stage predictions for second stage training
val_data['stage1_prediction'] = val_preds
val_data['stage1_prediction_confidence'] = abs(val_preds - 0.5) * 2
val_data['stage1_positive'] = (val_preds >= selected_threshold).astype(int)

# Filter validation set based on first stage predictions
val_filtered = val_data[val_data['stage1_positive'] == 1].copy()

# Start logging for second stage
logger.start_run(run_name="stage2_training")
logger.log_params({"filtered_samples": len(val_filtered)})
logger.log_params(second_stage_params)

# Create DMatrix objects for second stage
dtrain_stage2 = xgb.DMatrix(val_filtered[stage2_features], label=val_filtered['is_draw'])

# Split filtered data for validation
from sklearn.model_selection import train_test_split
stage2_train, stage2_val = train_test_split(
    val_filtered, test_size=0.3, stratify=val_filtered['is_draw']
)

dtrain_s2 = xgb.DMatrix(stage2_train[stage2_features], label=stage2_train['is_draw'])
dval_s2 = xgb.DMatrix(stage2_val[stage2_features], label=stage2_val['is_draw'])

# Setup MLflow tracking
with mlflow.start_run(run_name="stage2_model") as run:
    mlflow.log_params(second_stage_params)
    
    # Train with early stopping
    evals_result = {}
    model_stage2 = xgb.train(
        second_stage_params,
        dtrain_s2,
        num_boost_round=3000,
        evals=[(dtrain_s2, 'train'), (dval_s2, 'val')],
        early_stopping_rounds=300,
        evals_result=evals_result,
        verbose_eval=100
    )
    
    # Log metrics
    for metric in ['auc', 'logloss', 'error']:
        mlflow.log_metric(f"val_{metric}", evals_result['val'][metric][-1])
    
    # Generate predictions
    stage2_val_preds = model_stage2.predict(dval_s2)
    
    # Calculate precision-focused metrics
    precision, recall, thresholds = precision_recall_curve(stage2_val['is_draw'], stage2_val_preds)
    
    # Find threshold that gives target precision
    target_precision = 0.5  # Aiming for 50% precision
    precision_threshold_idx = next(i for i, p in enumerate(precision) if p < target_precision)
    selected_threshold_s2 = thresholds[precision_threshold_idx-1]
    selected_recall_s2 = recall[precision_threshold_idx-1]
    
    # Log results
    mlflow.log_metric("selected_threshold", selected_threshold_s2)
    mlflow.log_metric("selected_recall", selected_recall_s2)
    mlflow.log_metric("selected_precision", precision[precision_threshold_idx-1])
    
    # Log model with signature
    signature = mlflow.models.infer_signature(
        stage2_val[stage2_features],
        model_stage2.predict(dval_s2)
    )
    
    mlflow.xgboost.log_model(
        xgb_model=model_stage2,
        artifact_path="model_stage2",
        registered_model_name=f"xgboost_stage2_{datetime.now().strftime('%Y%m%d_%H%M')}",
        signature=signature
    )
    
    # Log feature importance
    feature_importance = model_stage2.get_score(importance_type='gain')
    mlflow.log_dict(feature_importance, "artifacts/feature_importance_stage2.json")

# End logging
logger.log_metrics({
    "stage2_recall": selected_recall_s2,
    "stage2_precision": precision[precision_threshold_idx-1],
    "stage2_threshold": selected_threshold_s2,
    "combined_model_recall": selected_recall_s2 * recall[recall_threshold_idx],
    "combined_model_precision": precision[precision_threshold_idx-1]
})
logger.end_run()
```

## 4. Hyperparameter Optimization

### 4.1 Optuna Framework Setup

```python
import optuna
from optuna.samplers import TPESampler
import numpy as np
from utils.logger import ExperimentLogger

# Initialize logger
logger = ExperimentLogger(
    experiment_name="two_stage_tuning",
    log_dir="logs/hyperparameter_tuning"
)
```

### 4.2 Stage 1 (Recall-Focused) Hypertuning

```python
def objective_stage1(trial):
    """
    Optuna objective function for Stage 1 model optimization.
    Focused on maximizing recall while maintaining reasonable precision.
    
    Parameters:
    -----------
    trial : optuna.Trial
        Optuna trial object
        
    Returns:
    --------
    float
        Composite score favoring recall
    """
    # Log trial start
    logger.start_run(run_name=f"stage1_trial_{trial.number}")
    
    # Define hyperparameters to tune with ranges similar to xgboost_model.py
    params = {
        'objective': 'binary:logistic',
        'tree_method': 'hist',  # CPU-optimized as per guidelines
        'eval_metric': 'logloss',
        
        # Core hyperparameters
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
        
        # Regularization parameters
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
        
        # Subsampling parameters
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        
        # Recall-focused parameters
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 5.0)
    }
    
    # Log parameters
    logger.log_params(params)
    
    # Create DMatrix objects
    dtrain = xgb.DMatrix(train_data[stage1_features], label=train_data['is_draw'])
    dval = xgb.DMatrix(val_data[stage1_features], label=val_data['is_draw'])
    
    # Train with early stopping
    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=500,
        evals_result=evals_result,
        verbose_eval=False
    )
    
    # Generate predictions
    val_preds = model.predict(dval)
    
    # Calculate recall-focused metrics
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(val_data['is_draw'], val_preds)
    
    # Find the point with recall >= 0.4 that maximizes precision
    valid_indices = np.where(recall >= 0.4)[0]
    if len(valid_indices) > 0:
        best_idx = valid_indices[np.argmax(precision[valid_indices])]
        best_threshold = thresholds[best_idx]
        best_precision = precision[best_idx]
        best_recall = recall[best_idx]
    else:
        # If no point reaches 0.4 recall, find maximum recall
        best_idx = np.argmax(recall)
        best_threshold = thresholds[best_idx]
        best_precision = precision[best_idx]
        best_recall = recall[best_idx]
    
    # Calculate PR-AUC
    from sklearn.metrics import auc
    pr_auc = auc(recall, precision)
    
    # Create composite score that favors recall
    # Weights: 60% recall, 30% precision, 10% PR-AUC
    composite_score = (0.6 * best_recall) + (0.3 * best_precision) + (0.1 * pr_auc)
    
    # Log metrics
    logger.log_metrics({
        'best_threshold': best_threshold,
        'best_precision': best_precision,
        'best_recall': best_recall,
        'pr_auc': pr_auc,
        'composite_score': composite_score
    })
    logger.end_run()
    
    # Return composite score
    return composite_score

# Create Optuna study with TPESampler
stage1_sampler = TPESampler(
    consider_prior=True,
    prior_weight=1.0,
    seed=42,
    n_startup_trials=10
)

stage1_study = optuna.create_study(
    direction="maximize",
    sampler=stage1_sampler,
    study_name="stage1_optimization"
)

# Start MLflow run for hyperparameter tuning
with mlflow.start_run(run_name="stage1_hyperopt") as run:
    mlflow.log_params({
        "n_trials": 50,
        "optimization_target": "recall_focused_composite",
        "recall_weight": 0.6,
        "precision_weight": 0.3,
        "auc_weight": 0.1
    })
    
    # Run optimization
    stage1_study.optimize(objective_stage1, n_trials=50, timeout=10800)  # 3 hours timeout
    
    # Log best parameters and scores
    mlflow.log_params({f"best_{k}": v for k, v in stage1_study.best_params.items()})
    mlflow.log_metric("best_composite_score", stage1_study.best_value)
    
    # Create hyperparameter importance plot
    try:
        from optuna.visualization import plot_param_importances
        import matplotlib.pyplot as plt
        
        fig = plot_param_importances(stage1_study)
        fig.write_image("stage1_param_importance.png")
        mlflow.log_artifact("stage1_param_importance.png", "hyperopt_plots")
    except:
        logger.log_warning("Failed to create hyperparameter importance plot")
```

### 4.3 Stage 2 (Precision-Focused) Hypertuning

```python
def objective_stage2(trial):
    """
    Optuna objective function for Stage 2 model optimization.
    Focused on maximizing precision while maintaining target recall.
    
    Parameters:
    -----------
    trial : optuna.Trial
        Optuna trial object
        
    Returns:
    --------
    float
        Composite score favoring precision
    """
    # Log trial start
    logger.start_run(run_name=f"stage2_trial_{trial.number}")
    
    # Use best parameters from Stage 1 to generate filtered dataset
    stage1_best_params = stage1_study.best_params.copy()
    stage1_best_params.update({
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'eval_metric': 'logloss'
    })
    
    # Train Stage 1 model with best params
    dtrain_s1 = xgb.DMatrix(train_data[stage1_features], label=train_data['is_draw'])
    dval_s1 = xgb.DMatrix(val_data[stage1_features], label=val_data['is_draw'])
    
    model_s1 = xgb.train(
        stage1_best_params,
        dtrain_s1,
        num_boost_round=3000,
        evals=[(dtrain_s1, 'train'), (dval_s1, 'val')],
        early_stopping_rounds=500,
        verbose_eval=False
    )
    
    # Generate Stage 1 predictions
    val_preds_s1 = model_s1.predict(dval_s1)
    
    # Find threshold to get target recall
    from sklearn.metrics import precision_recall_curve
    precision_s1, recall_s1, thresholds_s1 = precision_recall_curve(val_data['is_draw'], val_preds_s1)
    valid_indices = np.where(recall_s1 >= 0.4)[0]
    if len(valid_indices) > 0:
        idx = valid_indices[np.argmax(precision_s1[valid_indices])]
        stage1_threshold = thresholds_s1[idx]
    else:
        # Fallback
        stage1_threshold = 0.3
    
    # Filter validation set based on Stage 1 predictions
    val_data['stage1_prediction'] = val_preds_s1
    val_data['stage1_prediction_confidence'] = abs(val_preds_s1 - 0.5) * 2
    val_filtered = val_data[val_preds_s1 >= stage1_threshold].copy()
    
    # If filtered set is too small, abort trial
    if len(val_filtered) < 50 or val_filtered['is_draw'].sum() < 10:
        logger.log_warning(f"Insufficient samples after filtering: {len(val_filtered)} total, {val_filtered['is_draw'].sum()} positive")
        logger.end_run()
        return 0.0
    
    # Define hyperparameters for Stage 2
    params = {
        'objective': 'binary:logistic',
        'tree_method': 'hist',  # CPU-optimized
        'eval_metric': 'logloss',
        
        # Core hyperparameters
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 6),  # Shallower to prevent overfitting
        'min_child_weight': trial.suggest_int('min_child_weight', 2, 8),  # Higher to prevent overfitting
        'gamma': trial.suggest_float('gamma', 0.0, 2.0),
        
        # Regularization parameters (stronger)
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 20.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 20.0, log=True),
        
        # Subsampling parameters
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        
        # Precision-focused parameters
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.8, 3.0)
    }
    
    # Log parameters
    logger.log_params(params)
    
    # Split filtered data for training/validation
    from sklearn.model_selection import train_test_split
    stage2_train, stage2_val = train_test_split(
        val_filtered, test_size=0.3, stratify=val_filtered['is_draw'], random_state=42
    )
    
    # Create DMatrix objects for Stage 2
    dtrain_s2 = xgb.DMatrix(stage2_train[stage2_features], label=stage2_train['is_draw'])
    dval_s2 = xgb.DMatrix(stage2_val[stage2_features], label=stage2_val['is_draw'])
    
    # Train with early stopping
    evals_result = {}
    model = xgb.train(
        params,
        dtrain_s2,
        num_boost_round=2000,
        evals=[(dtrain_s2, 'train'), (dval_s2, 'val')],
        early_stopping_rounds=300,
        evals_result=evals_result,
        verbose_eval=False
    )
    
    # Generate predictions
    stage2_preds = model.predict(dval_s2)
    
    # Calculate precision-focused metrics
    precision, recall, thresholds = precision_recall_curve(stage2_val['is_draw'], stage2_preds)
    
    # Find the point with precision >= 0.5 that maximizes recall
    valid_indices = np.where(precision >= 0.5)[0]
    if len(valid_indices) > 0:
        best_idx = valid_indices[np.argmax(recall[valid_indices])]
        best_threshold = thresholds[best_idx]
        best_precision = precision[best_idx]
        best_recall = recall[best_idx]
    else:
        # If no point reaches 0.5 precision, find maximum precision
        best_idx = np.argmax(precision)
        best_threshold = thresholds[best_idx]
        best_precision = precision[best_idx]
        best_recall = recall[best_idx]
    
    # Calculate PR-AUC
    from sklearn.metrics import auc
    pr_auc = auc(recall, precision)
    
    # Calculate effective overall recall
    stage1_recall = len(val_filtered[val_filtered['is_draw'] == 1]) / len(val_data[val_data['is_draw'] == 1])
    combined_recall = stage1_recall * best_recall
    
    # Create composite score that favors precision
    # Weights: 60% precision, 20% stage2 recall, 10% combined recall, 10% PR-AUC
    composite_score = (0.6 * best_precision) + (0.2 * best_recall) + (0.1 * combined_recall) + (0.1 * pr_auc)
    
    # Check if we meet minimum combined recall of 20%
    if combined_recall < 0.2:
        # Apply penalty for not meeting minimum recall
        composite_score *= (combined_recall / 0.2)
    
    # Log metrics
    logger.log_metrics({
        'stage2_threshold': best_threshold,
        'stage2_precision': best_precision,
        'stage2_recall': best_recall,
        'combined_recall': combined_recall,
        'pr_auc': pr_auc,
        'composite_score': composite_score
    })
    logger.end_run()
    
    # Return composite score
    return composite_score

# Create Optuna study with TPESampler for Stage 2
stage2_sampler = TPESampler(
    consider_prior=True,
    prior_weight=1.0,
    seed=43,
    n_startup_trials=10
)

stage2_study = optuna.create_study(
    direction="maximize",
    sampler=stage2_sampler,
    study_name="stage2_optimization"
)

# Start MLflow run for hyperparameter tuning of Stage 2
with mlflow.start_run(run_name="stage2_hyperopt") as run:
    mlflow.log_params({
        "n_trials": 50,
        "optimization_target": "precision_focused_composite",
        "precision_weight": 0.6,
        "stage2_recall_weight": 0.2,
        "combined_recall_weight": 0.1,
        "auc_weight": 0.1
    })
    
    # Run optimization
    stage2_study.optimize(objective_stage2, n_trials=50, timeout=10800)  # 3 hours timeout
    
    # Log best parameters and scores
    mlflow.log_params({f"best_{k}": v for k, v in stage2_study.best_params.items()})
    mlflow.log_metric("best_composite_score", stage2_study.best_value)
    
    # Create hyperparameter importance plot
    try:
        from optuna.visualization import plot_param_importances
        import matplotlib.pyplot as plt
        
        fig = plot_param_importances(stage2_study)
        fig.write_image("stage2_param_importance.png")
        mlflow.log_artifact("stage2_param_importance.png", "hyperopt_plots")
    except:
        logger.log_warning("Failed to create hyperparameter importance plot")
```

### 4.4 End-to-End Optimization Workflow

```python
def run_full_hyperparameter_optimization():
    """
    Run the full two-stage hyperparameter optimization process
    """
    logger.start_run(run_name="full_hyperopt_workflow")
    
    # 1. Optimize Stage 1 model for recall
    logger.log_info("Starting Stage 1 (Recall-Focused) Optimization")
    stage1_study.optimize(objective_stage1, n_trials=50, timeout=10800)
    
    # Log best Stage 1 parameters
    logger.log_info(f"Best Stage 1 Parameters: {stage1_study.best_params}")
    logger.log_info(f"Best Stage 1 Composite Score: {stage1_study.best_value}")
    
    # 2. Optimize Stage 2 model for precision
    logger.log_info("Starting Stage 2 (Precision-Focused) Optimization")
    stage2_study.optimize(objective_stage2, n_trials=50, timeout=10800)
    
    # Log best Stage 2 parameters
    logger.log_info(f"Best Stage 2 Parameters: {stage2_study.best_params}")
    logger.log_info(f"Best Stage 2 Composite Score: {stage2_study.best_value}")
    
    # 3. Train the final models with best parameters
    
    # Best Stage 1 params
    stage1_best_params = stage1_study.best_params.copy()
    stage1_best_params.update({
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'eval_metric': 'logloss'
    })
    
    # Best Stage 2 params
    stage2_best_params = stage2_study.best_params.copy()
    stage2_best_params.update({
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'eval_metric': 'logloss'
    })
    
    # Log final parameter sets
    logger.log_params({
        'final_stage1_params': stage1_best_params,
        'final_stage2_params': stage2_best_params
    })
    
    # 4. Train final models and evaluate
    
    # [... Training code would go here ...]
    
    # 5. Summary of optimization results
    logger.log_info("Hyperparameter Optimization Complete!")
    logger.log_info(f"Target Metrics: Precision 50%, Recall 20%")
    logger.log_info(f"Final Performance: Precision {final_precision:.2f}%, Recall {final_recall:.2f}%")
    
    logger.end_run()
    
    return {
        'stage1_params': stage1_best_params,
        'stage2_params': stage2_best_params,
        'final_precision': final_precision,
        'final_recall': final_recall
    }
```

### 4.5 Hyperparameter Ranges Summary

| Parameter | Stage 1 (Recall) Range | Stage 2 (Precision) Range | Notes |
|-----------|------------------------|---------------------------|-------|
| learning_rate | 0.01 - 0.2 | 0.005 - 0.1 | Lower in Stage 2 for fine-tuning |
| max_depth | 3 - 8 | 2 - 6 | Shallower in Stage 2 to prevent overfitting |
| min_child_weight | 1 - 5 | 2 - 8 | Higher in Stage 2 for regularization |
| gamma | 0.0 - 1.0 | 0.0 - 2.0 | More pruning in Stage 2 |
| reg_alpha | 0.0 - 10.0 | 0.1 - 20.0 | Stronger L1 regularization in Stage 2 |
| reg_lambda | 0.1 - 10.0 | 0.5 - 20.0 | Stronger L2 regularization in Stage 2 |
| subsample | 0.6 - 1.0 | 0.5 - 0.9 | More aggressive in Stage 2 |
| colsample_bytree | 0.6 - 1.0 | 0.5 - 0.9 | More aggressive in Stage 2 |
| scale_pos_weight | 1.0 - 5.0 | 0.8 - 3.0 | More balanced in Stage 2 |

## 5. MLflow Integration

### 5.1 Experiment Structure

```python
# Create parent experiment for two-stage model
parent_experiment_id = mlflow.create_experiment(
    "two_stage_xgboost",
    tags={"model_type": "two_stage", "target_metric": "precision_recall_balance"}
)

# Create nested runs for each component
with mlflow.start_run(experiment_id=parent_experiment_id, run_name="complete_pipeline") as parent_run:
    # Log overall parameters
    mlflow.log_params({
        "target_precision": 0.5,
        "target_recall": 0.2,
        "stage1_focus": "recall",
        "stage2_focus": "precision"
    })
    
    # Track lineage of models
    mlflow.set_tag("model_lineage", "two_stage_xgboost_v1")
    
    # Child runs will be automatically nested under this parent
    # ... stage1 and stage2 runs can go here ...
    
    # Log final combined metrics
    mlflow.log_metrics({
        "final_precision": final_precision,
        "final_recall": final_recall,
        "f1_score": final_f1_score,
        "stage1_filtering_rate": stage1_positive_rate,
        "stage2_filtering_rate": stage2_positive_rate
    })
    
    # Register the combined model with proper naming convention
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    mlflow.register_model(
        model_uri=f"runs:/{parent_run.info.run_id}/combined_model",
        name=f"two_stage_xgboost_{timestamp}"
    )
```

### 5.2 Model Artifacts

The following artifacts are logged to MLflow:

- Feature importance plots for both models
- Precision-recall curves
- Threshold calibration curves
- Confusion matrices
- Model parameters and performance metrics
- Training history plots

## 6. Evaluation Framework

### 6.1 Performance Assessment

```python
# Apply the full two-stage pipeline to test data
def predict_two_stage(data, model1, model2, threshold1, threshold2, features1, features2):
    # First stage prediction
    dtest1 = xgb.DMatrix(data[features1])
    stage1_preds = model1.predict(dtest1)
    data['stage1_prediction'] = stage1_preds
    data['stage1_prediction_confidence'] = abs(stage1_preds - 0.5) * 2
    
    # Filter based on first stage
    stage1_positives = data[stage1_preds >= threshold1].copy()
    
    if len(stage1_positives) == 0:
        return pd.DataFrame(), pd.DataFrame()
    
    # Second stage prediction
    dtest2 = xgb.DMatrix(stage1_positives[features2])
    stage2_preds = model2.predict(dtest2)
    stage1_positives['stage2_prediction'] = stage2_preds
    
    # Final predictions
    final_positives = stage1_positives[stage2_preds >= threshold2].copy()
    
    return stage1_positives, final_positives

# Evaluate on test set
stage1_results, final_results = predict_two_stage(
    test_data, 
    model_stage1, 
    model_stage2,
    selected_threshold,
    selected_threshold_s2,
    stage1_features,
    stage2_features
)

# Calculate metrics
from sklearn.metrics import classification_report, confusion_matrix

# First stage metrics
stage1_precision = stage1_results['is_draw'].mean()
stage1_recall = len(stage1_results[stage1_results['is_draw'] == 1]) / len(test_data[test_data['is_draw'] == 1])

# Final metrics
final_precision = final_results['is_draw'].mean() if len(final_results) > 0 else 0
final_recall = len(final_results[final_results['is_draw'] == 1]) / len(test_data[test_data['is_draw'] == 1])

# Log evaluation
logger.start_run(run_name="test_evaluation")
logger.log_metrics({
    "stage1_precision": stage1_precision,
    "stage1_recall": stage1_recall,
    "final_precision": final_precision,
    "final_recall": final_recall,
    "stage1_filtering_rate": len(stage1_results) / len(test_data),
    "stage2_filtering_rate": len(final_results) / len(stage1_results) if len(stage1_results) > 0 else 0
})
logger.end_run()
```

### 6.2 Cross-Validation Stability

```python
from sklearn.model_selection import KFold

# K-fold cross-validation for stability testing
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_results = []
logger.start_run(run_name="cross_validation")

for fold, (train_idx, val_idx) in enumerate(kf.split(train_data)):
    # Split data
    fold_train = train_data.iloc[train_idx]
    fold_val = train_data.iloc[val_idx]
    
    # ... implement training for stage 1 and stage 2 ...
    
    # Track results
    cv_results.append({
        'fold': fold,
        'stage1_precision': s1_precision,
        'stage1_recall': s1_recall,
        'stage2_precision': s2_precision,
        'stage2_recall': s2_recall,
        'final_precision': final_precision,
        'final_recall': final_recall
    })
    
    logger.log_metrics({
        f"fold_{fold}_final_precision": final_precision,
        f"fold_{fold}_final_recall": final_recall
    })

# Calculate stability metrics
precision_std = np.std([r['final_precision'] for r in cv_results])
recall_std = np.std([r['final_recall'] for r in cv_results])

logger.log_metrics({
    "precision_mean": np.mean([r['final_precision'] for r in cv_results]),
    "precision_std": precision_std,
    "recall_mean": np.mean([r['final_recall'] for r in cv_results]),
    "recall_std": recall_std,
    "stability_score": 1 / (1 + precision_std + recall_std)
})
logger.end_run()
```

## 7. Production Deployment

### 7.1 Prediction API

```python
def two_stage_predict(match_data):
    """
    Apply the two-stage model to new match data
    
    Parameters:
    -----------
    match_data : DataFrame
        Features for the matches to predict
        
    Returns:
    --------
    DataFrame with original data and prediction columns
    """
    try:
        logger.start_run(run_name="prediction")
        
        # Load models from registry
        stage1_model = mlflow.xgboost.load_model("models:/xgboost_stage1_latest/Production")
        stage2_model = mlflow.xgboost.load_model("models:/xgboost_stage2_latest/Production")
        
        # First stage prediction
        dpredict1 = xgb.DMatrix(match_data[stage1_features])
        stage1_preds = stage1_model.predict(dpredict1)
        match_data['stage1_prediction'] = stage1_preds
        match_data['stage1_prediction_confidence'] = abs(stage1_preds - 0.5) * 2
        
        # Filter based on first stage
        stage1_positives = match_data[stage1_preds >= selected_threshold].copy()
        match_data['stage1_positive'] = (stage1_preds >= selected_threshold).astype(int)
        
        if len(stage1_positives) == 0:
            logger.log_metrics({"prediction_count": 0})
            logger.end_run()
            return match_data
        
        # Second stage prediction
        dpredict2 = xgb.DMatrix(stage1_positives[stage2_features])
        stage2_preds = stage2_model.predict(dpredict2)
        
        # Map predictions back to original dataframe
        match_data.loc[stage1_positives.index, 'stage2_prediction'] = stage2_preds
        match_data['stage2_positive'] = 0
        match_data.loc[stage1_positives.index, 'stage2_positive'] = (stage2_preds >= selected_threshold_s2).astype(int)
        
        # Final prediction
        match_data['final_prediction'] = match_data['stage1_positive'] & match_data['stage2_positive']
        
        logger.log_metrics({
            "prediction_count": len(match_data),
            "stage1_positive_count": match_data['stage1_positive'].sum(),
            "final_positive_count": match_data['final_prediction'].sum()
        })
        logger.end_run()
        
        return match_data
        
    except Exception as e:
        logger.log_error(f"Prediction error: {str(e)}")
        logger.end_run()
        raise
```

### 7.2 Monitoring and Maintenance

- Implement weekly model performance tracking
- Set up drift detection for input features and predictions
- Create automated retraining pipeline when performance drops below thresholds

## 8. References

- Project Documentation: [Project Plan](/docs/plan.md)
- Related Models: [XGBoost API Model](/models/xgboost_api_model.py)
- MLflow Guide: [MLflow Implementation Guide](/docs/guides/mlflow.md)
- Logging Standards: [Logger Implementation](/utils/logger.py)
