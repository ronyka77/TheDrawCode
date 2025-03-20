import pymc as pm
import arviz as az
import numpy as np
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score
from utils.logger import ExperimentLogger
import mlflow
import optuna
from optuna.samplers import TPESampler
from datetime import datetime

logger = ExperimentLogger(experiment_name="soccer_prediction", log_dir="logs/soccer_prediction")



class BayesianMetaLearner:
    """
    Bayesian Logistic Regression implementation as meta learner for ensemble models.
    Uses PyMC for Bayesian inference and provides hyperparameter tuning capabilities.
    """
    def __init__(self, n_samples=1000, tune=1000, target_accept=0.8, prior_scale=3.0, class_weight=None, **kwargs):
        """
        Initialize the Bayesian Meta Learner
        
        Args:
            n_samples: Number of samples to draw in MCMC
            tune: Number of tuning steps in MCMC
            target_accept: Target acceptance rate for MCMC
            prior_scale: Scale parameter for the prior distributions
            class_weight: Optional class weights for imbalanced datasets
            **kwargs: Additional parameters to be stored in hyperparams
        """
        self.hyperparams = {
            'n_samples': n_samples,
            'tune': tune,
            'target_accept': target_accept,
            'prior_scale': prior_scale,
            'class_weight': class_weight
        }
        # Add any additional parameters
        self.hyperparams.update(kwargs)
        self.model = None
        self.trace = None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the Bayesian Logistic Regression model
        
        Args:
            X_train: Training features (base model predictions)
            y_train: Training target values
            X_val: Validation features
            y_val: Validation target values
        
        Returns:
            Self for method chaining
        """
        logger.info(f"Training Bayesian Logistic Regression with params: {self.hyperparams}")
        
        # Number of predictors (base models)
        n_predictors = X_train.shape[1]
        
        with pm.Model() as model:
            # Priors for weights
            weights = pm.Normal('weights', 
                                mu=0, 
                                sigma=self.hyperparams['prior_scale'], 
                                shape=n_predictors)
            
            # Prior for intercept
            intercept = pm.Normal('intercept', mu=0, sigma=self.hyperparams['prior_scale'])
            
            # Linear combination of inputs
            linear_pred = intercept + pm.math.dot(X_train, weights)
            
            # Likelihood (Bernoulli)
            pm.Bernoulli('likelihood', 
                        p=pm.math.sigmoid(linear_pred), 
                        observed=y_train)
            
            # Inference
            trace = pm.sample(self.hyperparams['n_samples'], 
                            tune=self.hyperparams['tune'],
                            target_accept=self.hyperparams['target_accept'],
                            return_inferencedata=True,
                            cores=1)  # Enforce CPU-only training
        
        self.model = model
        self.trace = trace
        
        # Evaluate if validation data is provided
        # if X_val is not None and y_val is not None:
        #     self.evaluate(X_val, y_val)
        
        return self

    def predict_proba(self, X):
        """
        Make probabilistic predictions returning the positive class probability.
        
        Args:
            X: Features to predict on.
        
        Returns:
            Predicted probabilities for class 1 as a 1D numpy array.
        """
        if self.model is None or self.trace is None:
            raise ValueError("Model has not been trained yet")
        
        with self.model:
            weights_posterior = self.trace.posterior.weights.values
            intercept_posterior = self.trace.posterior.intercept.values
            # Compute the posterior means across chains and samples.
            weights_mean = weights_posterior.mean(axis=(0, 1))
            intercept_mean = intercept_posterior.mean(axis=(0, 1))
            
            # Compute the linear prediction and clip for numerical stability.
            linear_pred = intercept_mean + np.dot(X, weights_mean)
            linear_pred = np.clip(linear_pred, -700, 700)
            p = 1.0 / (1.0 + np.exp(-linear_pred))
        return p

    def predict_positive_proba(self, X):
        """
        Make probabilistic predictions returning the probability for the positive class only.
        
        Args:
            X: Features to predict on.
        
        Returns:
            Predicted probabilities for class 1 as a 1D numpy array.
        """
        if self.model is None or self.trace is None:
            raise ValueError("Model has not been trained yet")
        
        with self.model:
            weights_posterior = self.trace.posterior.weights.values
            intercept_posterior = self.trace.posterior.intercept.values
            
            # Compute posterior means regardless of dimension
            weights_mean = weights_posterior.mean(axis=(0, 1))
            intercept_mean = intercept_posterior.mean(axis=(0, 1))
            
            # Compute linear predictions and clip for numerical stability
            linear_pred = intercept_mean + np.dot(X, weights_mean)
            linear_pred = np.clip(linear_pred, -700, 700)
            p = 1.0 / (1.0 + np.exp(-linear_pred))
        
        return p

    def predict(self, X, threshold=0.5):
        """
        Make class predictions using the probability for the positive class.
        
        Args:
            X: Features to predict on.
            threshold: Classification threshold.
        
        Returns:
            Predicted classes (0 or 1) as a 1D numpy array.
        """
        p = self.predict_positive_proba(X)
        return (p > threshold).astype(int)

    def evaluate(self, X_val, y_val, threshold=0.5):
        """
        Evaluate the Bayesian model performance
        
        Args:
            X_val: Validation features
            y_val: Validation targets
            threshold: Classification threshold
        
        Returns:
            Dictionary of metrics
        """
        y_pred_proba = self.predict_proba(X_val)
        # Use only the positive class probabilities; this yields a 1D array.
        y_pred = (y_pred_proba > threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        loss = log_loss(y_val, y_pred_proba)
        
        # Log metrics
        logger.info(f"Bayesian Meta Learner Evaluation: accuracy={accuracy:.4f}, "
                    f"precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}, loss={loss:.4f}")
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'log_loss': loss
        }
        
        return metrics

def objective_function(trial, X_train, y_train, X_val, y_val):
    """
    Objective function for Bayesian Meta Learner hyperparameter optimization
    
    Args:
        trial: Optuna trial object
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        
    Returns:
        Validation loss
    """
    # Define hyperparameters to tune
    hyperparams = {
        'n_samples': trial.suggest_int('n_samples', 500, 2000, step=500),
        'tune': trial.suggest_int('tune', 500, 2000, step=500),
        'target_accept': trial.suggest_float('target_accept', 0.7, 0.95, step=0.05),
        'prior_scale': trial.suggest_float('prior_scale', 1.0, 5.0, step=0.5)
    }
    
    try:
        # Train the model
        learner = BayesianMetaLearner(hyperparams=hyperparams)
        learner.train(X_train, y_train)
        
        # Evaluate
        metrics = learner.evaluate(X_val, y_val)
        
        # Optimize for validation loss
        return metrics['log_loss']
    except Exception as e:
        logger.error(f"Error in Bayesian meta learner training: {str(e)}")
        return float('inf')  # Return a large value if error occurs

def optimize_hyperparameters(X_train, y_train, X_val, y_val, n_trials=20, timeout=3600):
    """
    Optimize hyperparameters for Bayesian Meta Learner
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        n_trials: Number of optimization trials
        timeout: Maximum time in seconds for optimization
        
    Returns:
        Best hyperparameters and study object
    """
    logger.info(f"Starting Bayesian Meta Learner hyperparameter optimization with {n_trials} trials")
    
    # Create study object
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(
            consider_prior=True,
            prior_weight=0.3
        )
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective_function(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        timeout=timeout
    )
    
    # Get best parameters
    best_params = study.best_params
    logger.info(f"Best Bayesian Meta Learner parameters: {best_params}")
    
    return best_params, study

def train_with_optimal_parameters(X_train, y_train, X_val=None, y_val=None, params=None):
    """
    Train a Bayesian meta learner with optimal hyperparameters
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        n_trials: Number of optimization trials
        
    Returns:
        Trained BayesianMetaLearner
    """
    # Optimize hyperparameters
    best_params = params
    
    # Train with best hyperparameters
    learner = BayesianMetaLearner(hyperparams=best_params)
    learner.train(X_train, y_train, X_val, y_val)
    
    # Register model
    model_name = f"bayesian_meta_ensemble_{datetime.now().strftime('%Y%m%d_%H%M')}"
    mlflow.set_tag("model_type", "bayesian_meta_learner")
    return learner 

def optimize_threshold(self, X_val, y_val, target_precision=0.5, min_recall=0.25, logger=None):
    """
    Optimize the prediction threshold for the Bayesian meta-learner.
    
    Args:
        X_val: Validation features
        y_val: Validation targets
        target_precision: Target precision to achieve (default: 0.5)
        min_recall: Minimum required recall (default: 0.25)
        logger: Logger instance (optional)
        
    Returns:
        Tuple of (optimal_threshold, metrics_at_threshold)
    """
    from utils.logger import ExperimentLogger
    from models.ensemble.thresholds import tune_threshold_for_precision
    
    if logger is None:
        logger = ExperimentLogger(experiment_name="bayesian_meta_learner",
                                    log_dir="./logs/bayesian_meta_learner")
    
    logger.info("Optimizing threshold for Bayesian meta-learner...")
    
    # Get predictions on validation set
    y_proba = self.predict_proba(X_val)
    logger.info(f"y_proba: {y_proba}")
    # Find optimal threshold using the utility function
    optimal_threshold, metrics = tune_threshold_for_precision(
        y_proba, y_val, 
        target_precision=target_precision, 
        required_recall=min_recall,
        min_threshold=0.1,
        max_threshold=0.9,
        step=0.01,
        logger=logger
    )
    
    # Store the optimal threshold
    self.optimal_threshold = optimal_threshold
    
    # Log results
    logger.info(f"Optimal threshold: {optimal_threshold:.4f}")
    logger.info(f"Precision at threshold: {metrics['precision']:.4f}")
    logger.info(f"Recall at threshold: {metrics['recall']:.4f}")
    
    # Log to MLflow
    mlflow.log_param("bayesian_meta_learner_threshold", optimal_threshold)
    mlflow.log_metrics({
        "bayesian_precision_at_threshold": metrics['precision'],
        "bayesian_recall_at_threshold": metrics['recall'],
        "bayesian_f1_at_threshold": metrics['f1']
    })
    
    return optimal_threshold, metrics
