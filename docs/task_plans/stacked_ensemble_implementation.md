# Stacked Ensemble Implementation Plan

## Data Flow Strategy

### Data Split Usage
1. Training Data (`X_train`):
   - Primary training data for all base models
   - Used for nested cross-validation during hyperparameter tuning
   - Never used for performance evaluation or threshold optimization

2. Test Data (`X_test`):
   - Used as evaluation set during model training
   - Enables early stopping to prevent overfitting
   - Not used for final performance evaluation

3. Validation Data (`X_val`):
   - Completely held-out set
   - Used for:
     - Final model evaluation
     - Threshold optimization
     - Meta-feature generation for stacking
     - Meta-learner training

## Sprint 1: Infrastructure & Base Setup (2 weeks)

### Week 1: Project Structure & Core Utilities
1. Project Setup (2 days)
   ```bash
   # Create directory structure
   mkdir -p models/StackedEnsemble/{base/{tree_based,linear,neural,transformer,perpetual},stacking,shared,config/{model_configs,hyperparameter_spaces}}
   ```
   - [ ] Initialize git repository
   - [ ] Set up virtual environment
   - [ ] Create initial requirements.txt

2. Shared Components (3 days)
   ```python
   # shared/data_loader.py
   from utils.logger import ExperimentLogger
   from utils.create_evaluation_set import (
       create_ensemble_evaluation_set,
       import_selected_features_ensemble,
       import_training_data_ensemble
   )

   class DataLoader:
       def __init__(self):
           self.logger = ExperimentLogger(experiment_name="data_loader")
           self._cached_features = None

       def load_data(self):
           """Load and split data into train, test, and validation sets."""
           self.logger.info("Loading data splits")
           
           # Load selected features first
           if self._cached_features is None:
               self._cached_features = import_selected_features_ensemble('all')
               self.logger.info(f"Loaded {len(self._cached_features)} selected features")
           
           # Load training and test data
           X_train, y_train, X_test, y_test = import_training_data_ensemble()
           
           # Load validation data
           X_val, y_val = create_ensemble_evaluation_set()
           
           # Apply feature selection to all splits
           X_train = X_train[self._cached_features]
           X_test = X_test[self._cached_features]
           X_val = X_val[self._cached_features]
           
           self.logger.info(
               "Data split sizes:"
               f"\n - Train: {X_train.shape}"
               f"\n - Test: {X_test.shape}"
               f"\n - Validation: {X_val.shape}"
           )
           
           return X_train, y_train, X_test, y_test, X_val, y_val
       
       def get_feature_names(self) -> list:
           """Get the list of selected feature names.
           
           Returns:
               List of feature names
           """
           if self._cached_features is None:
               self._cached_features = import_selected_features_ensemble('all')
           return self._cached_features
   ```

3. Validation Framework (2 days)
   ```python
   # shared/validation.py
   class NestedCVValidator:
       def perform_nested_cv(self, X_train, y_train):
           """Perform nested CV using training data only."""
           outer_cv = StratifiedKFold(n_splits=5)
           inner_cv = StratifiedKFold(n_splits=3)
           
           for train_idx, val_idx in outer_cv.split(X_train, y_train):
               # Inner CV for hyperparameter tuning
               X_train_fold = X_train.iloc[train_idx]
               y_train_fold = y_train.iloc[train_idx]
               
               # Use Population Based Training for hyperparameter optimization
               best_params = self._optimize_inner_cv(
                   X_train_fold,
                   y_train_fold,
                   model,
                   param_space,
                   num_trials=8
               )

   def _optimize_inner_cv(self, X_train, y_train, model, param_space, num_trials=8):
       """Run inner cross-validation for hyperparameter tuning."""
       # Configure Population Based Training scheduler
       scheduler = tune.schedulers.PopulationBasedTraining(
           time_attr="training_iteration",
           metric="mean_precision",
           mode="max",
           perturbation_interval=4,
           hyperparam_mutations=param_space
       )

       def objective(config):
           try:
               mean_precision = self.cross_validate_model(config, X_train, y_train, model)
               tune.report(mean_precision=mean_precision)
               return mean_precision
           except Exception as e:
               self.logger.error(f"Error in objective function: {str(e)}")
               tune.report(mean_precision=float('-inf'))
               return float('-inf')

       # Run hyperparameter optimization
       analysis = tune.run(
           objective,
           config=param_space,
           num_samples=num_trials,
           scheduler=scheduler,
           resources_per_trial={"cpu": 1, "gpu": 0}
       )
   ```

4. MLflow & Logging Setup (2 days)
   ```python
   # shared/mlflow_utils.py
   from utils.logger import ExperimentLogger
   from utils.create_evaluation_set import setup_mlflow_tracking
   import mlflow
   
   class MLflowManager:
       def __init__(self, base_experiment_name: str):
           self.logger = ExperimentLogger(experiment_name=base_experiment_name)
           self.base_experiment_name = base_experiment_name
           
       def setup_model_experiment(self, model_type: str) -> str:
           """Setup MLflow experiment for specific model type."""
           experiment_name = f"{self.base_experiment_name}_{model_type}"
           mlruns_dir = setup_mlflow_tracking(experiment_name)
           self.logger.info(f"Set up MLflow experiment: {experiment_name}")
           return mlruns_dir
           
       def log_training_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
           """Log training metrics to MLflow."""
           mlflow.log_metrics(metrics, step=step)
           self.logger.info(f"Logged metrics at step {step}: {metrics}")
           
       def log_split_metrics(self, metrics: Dict[str, float], split: str):
           """Log metrics with split identifier."""
           self.mlflow_manager.log_metrics({
               f"{split}_{k}": v for k, v in metrics.items()
           })
   ```
   - [ ] Set up experiment hierarchy for each model type
   - [ ] Implement unified logging interface
   - [ ] Configure artifact storage
   - [ ] Set up model registry

4. Configuration System (2 days)
   ```yaml
   # config/model_configs/xgboost_config.yaml
   model:
     name: xgboost
     version: 1.0.0
     experiment_name: tree_based_xgboost
     cpu_config:
       tree_method: hist
       n_jobs: -1
     logging:
       log_dir: logs/xgboost
       metrics_tracking:
         - precision
         - recall
         - training_time
   ```
   - [ ] Create base configuration templates
   - [ ] Set up hyperparameter space definitions
   - [ ] Implement configuration loading utilities

### Week 2: Model Base Classes
1. Base Model Interface (3 days)
   ```python
   # base/model_interface.py
   from abc import ABC, abstractmethod
   from utils.logger import ExperimentLogger
   
   class BaseModel(ABC):
       def __init__(self, model_type: str, config_path: str):
           self.model_type = model_type
           self.logger = ExperimentLogger(f"ensemble_{model_type}")
           self.mlflow_manager = MLflowManager(f"ensemble_{model_type}")
           self.mlruns_dir = self.mlflow_manager.setup_model_experiment(model_type)
           
       @abstractmethod
       def _create_model(self, **kwargs) -> Any:
           """Create and return the actual model instance."""
           pass
           
       @abstractmethod
       def _convert_to_model_format(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
           """Convert data to model-specific format."""
           pass
       
       def fit(
           self,
           X_train: pd.DataFrame,
           y_train: pd.Series,
           X_test: pd.DataFrame,
           y_test: pd.Series
       ):
           """Train model using training data and test data for early stopping."""
           # Convert data
           train_data = self._convert_to_model_format(X_train, y_train)
           test_data = self._convert_to_model_format(X_test, y_test)
           
           # Train with early stopping on test set
           self.model.fit(
               train_data,
               eval_set=[test_data],
               early_stopping_rounds=self.early_stopping_rounds
           )
       
       def evaluate(self, X_val: pd.DataFrame, y_val: pd.Series):
           """Evaluate model on validation set."""
           val_data = self._convert_to_model_format(X_val, y_val)
           metrics = self._calculate_metrics(val_data)
           self.mlflow_manager.log_split_metrics(metrics, "validation")
           
       def predict(self, X: pd.DataFrame) -> np.ndarray:
           """Generate predictions for input data."""
           if not self.is_fitted:
               raise RuntimeError("Model must be fitted before making predictions")
           X_model = self._convert_to_model_format(X)
           return self._predict_model(X_model)
           
       @abstractmethod
       def _predict_model(self, X: Any) -> np.ndarray:
           """Model-specific prediction implementation."""
           pass
           
       def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
           """Generate probability predictions."""
           if not self.is_fitted:
               raise RuntimeError("Model must be fitted before making predictions")
           X_model = self._convert_to_model_format(X)
           return self._predict_proba_model(X_model)
           
       @abstractmethod
       def _predict_proba_model(self, X: Any) -> np.ndarray:
           """Model-specific probability prediction implementation."""
           pass
   ```
   - [ ] Implement resource monitoring
   - [ ] Set up performance tracking
   - [ ] Create monitoring dashboards

### Week 2: Hyperparameter Optimization Framework

1. Configuration System
   ```yaml
   # config/hyperparameter_spaces/lightgbm_space.yaml
   hyperparameters:
     learning_rate:
       distribution: log_uniform
       min: 0.001
       max: 0.1
     num_leaves:
       distribution: int_uniform
       min: 20
       max: 100
     # ... other parameters ...

   search_strategy:
     name: bayesian
     settings:
       max_trials: 40
       metric: precision
       mode: max
       early_stopping:
         type: asha
         grace_period: 500
   ```

2. Nested Cross-Validation Framework
   ```python
   # shared/validation.py
   class NestedCVValidator:
       def __init__(self, outer_splits=3, inner_splits=2):
           self.outer_splits = outer_splits
           self.inner_splits = inner_splits
           self.best_score = 0
           self.best_params = {}

       def optimize_hyperparameters(self, model, X, y, X_val, y_val, X_test, y_test, param_space):
           """Run nested cross-validation for hyperparameter optimization."""
           # Initialize outer CV
           outer_cv = StratifiedKFold(
               n_splits=self.outer_splits,
               shuffle=True,
               random_state=19
           )
           
           # Run inner optimization
           best_params, metrics = self._optimize_inner_cv(
               X, y, X_val, y_val, X_test, y_test,
               model, param_space
           )
           
           return best_params, metrics

       def _optimize_inner_cv(self, X_train, y_train, X_val, y_val, X_test, y_test, model, param_space):
           """Run inner cross-validation with Ray Tune."""
           # Configure Ray storage and resources
           storage_path = Path.cwd() / "ray_results"
           ray.init(
               num_cpus=os.cpu_count(),
               _temp_dir=str(storage_path / "tmp"),
               include_dashboard=False,
               log_to_driver=True
           )

           # Configure ASHA scheduler
           scheduler = tune.schedulers.ASHAScheduler(
               time_attr="training_iteration",
               metric="precision",
               mode="max",
               max_t=3000,
               grace_period=500
           )

           # Configure Bayesian optimization
           bayesopt = BayesOptSearch(
               metric="precision",
               mode="max",
               utility_kwargs={
                   "kind": "ucb",
                   "kappa": 2.5,
                   "xi": 0.0
               }
           )

           # Run optimization
           analysis = tune.run(
               self.objective,
               scheduler=scheduler,
               search_alg=bayesopt,
               num_samples=40,
               config=param_space,
               resources_per_trial={"cpu": 1, "gpu": 0}
           )

           return analysis.best_config
   ```

3. Model-Specific Hyperparameter Optimization
   ```python
   # base/tree_based/lightgbm_model.py
   class LightGBMModel(BaseModel):
       def optimize_hyperparameters(self, X, y, X_val, y_val, X_test, y_test):
           """Optimize hyperparameters using nested cross-validation."""
           # Initialize validator
           self.cv_validator = NestedCVValidator(logger=self.logger)
           
           # Prepare parameter space
           param_space = {}
           param_ranges = {}  # For logging
           
           for param, config in self.hyperparameter_space['hyperparameters'].items():
               if isinstance(config, dict) and 'distribution' in config:
                   if config['distribution'] == 'log_uniform':
                       min_val = max(config['min'], 1e-8)
                       max_val = max(config['max'], min_val + 1e-8)
                       param_space[param] = tune.uniform(min_val, max_val)
                   elif config['distribution'] == 'uniform':
                       param_space[param] = tune.uniform(config['min'], config['max'])
                   elif config['distribution'] == 'int_uniform':
                       min_val = max(1, int(config['min']))
                       max_val = max(min_val + 1, int(config['max']))
                       param_space[param] = tune.randint(min_val, max_val)
           
           # Add CPU-specific parameters
           param_space.update({
               'device': 'cpu',
               'force_row_wise': True,
               'deterministic': True
           })
           
           # Run optimization
           best_params, metrics = self.cv_validator.optimize_hyperparameters(
               self, X, y, X_val, y_val, X_test, y_test,
               param_space,
               self.hyperparameter_space.get('search_strategy', {})
           )
           
           return best_params, metrics
   ```

4. Threshold Optimization
   ```python
   def _optimize_threshold(self, X_val, y_val):
       """Optimize decision threshold based on precision-recall trade-off."""
       probas = self.predict_proba(X_val)
       thresholds = np.linspace(0.25, 0.8, 56)
       best_threshold = 0.3
       best_score = 0.0
       
       for threshold in thresholds:
           preds = (probas >= threshold).astype(int)
           metrics = calculate_metrics(y_val, preds, probas)
           
           if metrics['recall'] >= 0.15 and metrics['recall'] < 0.9 and metrics['precision'] > 0.30:
               score = metrics['precision']
               if score > best_score:
                   best_score = score
                   best_threshold = threshold
       
       return best_threshold
   ```

## Key Changes from Original Plan

1. Hyperparameter Optimization:
   - Switched from Population Based Training to Bayesian Optimization with ASHA scheduler
   - Added explicit CPU optimization constraints
   - Implemented proper parameter space conversion for Ray Tune
   - Added comprehensive logging of parameter ranges and results

2. Cross-Validation:
   - Implemented nested CV with outer (3 folds) and inner (2 folds) loops
   - Added proper handling of parameter types (int, float, categorical)
   - Improved error handling and resource cleanup

3. Threshold Optimization:
   - Added specific precision-recall trade-off constraints
   - Implemented fine-grained threshold search
   - Added proper metric calculation and validation

4. Configuration:
   - Added YAML-based configuration for both model and hyperparameter spaces
   - Implemented strict validation of configuration structure
   - Added CPU-specific parameter constraints

## Resource Management

1. Ray Tune Configuration:
   - Single CPU per trial
   - Disabled GPU usage
   - Proper cleanup of Ray resources
   - Shorter temporary directories for Windows compatibility

2. Memory Management:
   - Added proper cleanup in finally blocks
   - Implemented resource limits per trial
   - Added logging of resource usage

3. Error Handling:
   - Added comprehensive error catching and logging
   - Implemented fallback to default parameters
   - Added proper cleanup on errors

## Monitoring & Logging

1. Parameter Space Logging:
   - Added human-readable parameter ranges
   - Logged actual parameter values used
   - Added conversion of numpy types for JSON serialization

2. Results Logging:
   - Added comprehensive metrics tracking
   - Implemented proper MLflow integration
   - Added timing information for optimization process

3. Resource Monitoring:
   - Added CPU usage tracking
   - Implemented memory usage monitoring
   - Added disk space monitoring for Ray results

## Sprint 2: Base Model Implementation (3 weeks)

### Week 3-5: Model Implementations
For each model (XGBoost, LightGBM, CatBoost, etc.):
1. Model Training
   ```python
   def train(self, X_train, y_train, X_test, y_test):
       """Train with early stopping on test set and cross-validation."""
       # Configure cross-validation
       cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
       cv_scores = []

       # Train and evaluate on each fold
       for train_idx, val_idx in cv.split(X_train, y_train):
           X_train_fold = X_train.iloc[train_idx]
           X_val_fold = X_train.iloc[val_idx]
           y_train_fold = y_train.iloc[train_idx]
           y_val_fold = y_train.iloc[val_idx]

           # Train model with early stopping
           self.model.fit(
               X_train_fold, y_train_fold,
               eval_set=[(X_val_fold, y_val_fold)],
               early_stopping_rounds=100,
               verbose=False
           )

           # Get precision score
           metrics = self.evaluate(X_val_fold, y_val_fold)
           precision = metrics.get('precision', 0.0)
           recall = metrics.get('recall', 0.0)
           
           # Only consider precision if we have some positive predictions
           if recall == 0.0:
               cv_scores.append(0.0)  # Penalize models that predict all negatives
           else:
               cv_scores.append(precision)

       # Return the mean score
       return np.mean(cv_scores)
   ```

2. Hyperparameter Tuning
   ```python
   def tune_hyperparameters(self, X_train, y_train):
       # Configure hyperparameter space
       param_space = {
           'learning_rate': tune.loguniform(0.001, 0.1),
           'max_depth': tune.randint(3, 8),
           'min_child_weight': tune.randint(1, 7),
           'subsample': tune.uniform(0.6, 1.0),
           'colsample_bytree': tune.uniform(0.6, 1.0),
           'gamma': tune.loguniform(0.0, 5.0),
           'lambda': tune.loguniform(0.1, 10.0),
           'alpha': tune.loguniform(0.0, 10.0),
           'scale_pos_weight': tune.uniform(1.0, 5.0)
       }

       # Configure Population Based Training
       scheduler = tune.schedulers.PopulationBasedTraining(
           time_attr="training_iteration",
           metric="precision",
           mode="max",
           perturbation_interval=4,
           hyperparam_mutations=param_space
       )

       # Run optimization with CPU constraints
       analysis = tune.run(
           objective,
           config=param_space,
           scheduler=scheduler,
           num_samples=8,
           resources_per_trial={"cpu": 1, "gpu": 0},
           stop={"training_iteration": 1}
       )
   ```

## Sprint 3: Stacking Framework (2 weeks)

### Week 6: Meta Feature Generation
1. Base Model Predictions
   ```python
   # stacking/meta_feature_generator.py
   class MetaFeatureGenerator:
       def generate_meta_features(self, models, X_val, y_val):
           """Generate meta-features using validation set."""
           meta_features = pd.DataFrame()
           
           for model in models:
               # Get predictions on validation set
               pred_proba = model.predict_proba(X_val)
               meta_features[f"{model.name}_pred"] = pred_proba
           
           return meta_features, y_val
   ```

2. Threshold Optimization
   ```python
   # stacking/threshold_optimizer.py
   class ThresholdOptimizer:
       def optimize_threshold(self, y_val_true, y_val_pred_proba):
           """Optimize threshold using validation set."""
           thresholds = np.linspace(0, 1, 100)
           best_threshold = 0.5
           best_score = float('-inf')
           
           for threshold in thresholds:
               y_pred = (y_val_pred_proba >= threshold).astype(int)
               score = self._calculate_objective(y_val_true, y_pred)
               
               if score > best_score:
                   best_score = score
                   best_threshold = threshold
           
           return best_threshold
   ```

### Week 7: Meta Learning
1. Meta Learner Training
   ```python
   # stacking/meta_learner.py
   class MetaLearner:
       def train(self, meta_features_val, y_val):
           """Train meta learner on validation set meta-features."""
           self.model.fit(meta_features_val, y_val)
   ```

## Sprint 4: Analysis & Deployment (2 weeks)

### Week 8-9: Performance Analysis & Deployment
1. Performance Evaluation
   ```python
   # evaluation/performance_analyzer.py
   class PerformanceAnalyzer:
       def analyze_performance(self, X_val, y_val):
           """Analyze model performance on validation set."""
           # Calculate metrics
           metrics = calculate_metrics(y_val, y_pred)
           
           # Log to MLflow
           self.mlflow_manager.log_metrics(metrics)
   ```

## Dependencies & Critical Path

### Critical Dependencies
1. Data Loading → Model Training
2. Model Training → Meta Feature Generation
3. Meta Feature Generation → Meta Learning
4. All Components → Performance Analysis

### Parallel Tracks
- Base model implementations can be parallelized
- Monitoring setup can run alongside model development
- Documentation can be developed continuously

## Resource Requirements
- 2-3 developers for 9 weeks
- CPU-optimized development environment
- MLflow tracking server
- Monitoring infrastructure

## Risk Mitigation
1. Validate data split strategy early
2. Monitor for data leakage between splits
3. Implement comprehensive logging for all splits
4. Regular cross-validation checks 