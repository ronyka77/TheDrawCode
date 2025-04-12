import os
import pickle
import random
from datetime import datetime

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd

# Set TensorFlow environment variables FIRST
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # No longer needed for GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # Reduce TensorFlow logging verbosity
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # Enable GPU memory growth
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # REMOVE this line to allow GPU usage
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import RobustScaler
from tensorflow import keras
from tensorflow.keras import callbacks, layers, regularizers  # type: ignore

# Logger and shared utilities
from src.utils.logger import ExperimentLogger

experiment_name = "tabnet_soccer_prediction"
logger = ExperimentLogger(experiment_name=experiment_name)

# Import shared utility functions
from src.models.StackedEnsemble.shared.data_loader import DataLoader
from src.models.StackedEnsemble.shared.hypertuner_utils import optimize_threshold
from src.utils.create_evaluation_set import import_selected_features_ensemble, setup_mlflow_tracking

experiment_name = "mlp_soccer_prediction"
logger = ExperimentLogger(experiment_name=experiment_name)


# Set random seeds for reproducibility
random_seed = 19
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
os.environ['PYTHONHASHSEED'] = str(random_seed)

# Configure Git executable path if available
git_executable = os.environ.get("GIT_PYTHON_GIT_EXECUTABLE")
if git_executable and os.path.exists(git_executable):
    import git
    git.refresh(git_executable)

mlflow_tracking = setup_mlflow_tracking(experiment_name)

# Global settings
min_recall = 0.20            # Minimum acceptable recall
n_trials = 100               # Fewer trials for MLP due to longer training times
pip_requirements = [
    f"tensorflow=={tf.__version__}",
    "scikit-learn", 
    f"mlflow=={mlflow.__version__}"
]
scaler = None  # Global scaler object

# Define base configurations
base_params = {
    'epochs': 100,
    'batch_size': 64,
    'verbose': 0,
    'metrics': ['accuracy', 'AUC']
}

# Verify GPU availability after TensorFlow import
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        logger.info(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs available.")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        logger.error(f"Error setting memory growth: {e}")
else:
    logger.warning("No GPUs detected by TensorFlow. Model will train on CPU.")

# Define the Wrapper Class
class KerasMLPWrapper(BaseEstimator, ClassifierMixin):
    """
    A wrapper for a fitted Keras Sequential model to provide
    a scikit-learn compatible predict_proba method.
    """
    def __init__(self, model, scaler):
        # Check if the model is a fitted Keras model
        # Keras models might not have model.built immediately after loading, check for weights
        if not isinstance(model, tf.keras.Model) or not model.weights:
            raise ValueError("Model must be a fitted Keras Model instance with weights.")
        self.model = model
        self.scaler = scaler
        # Infer classes_ if possible (assuming binary 0, 1)
        self.classes_ = np.array([0, 1])

    def predict(self, X):
        """Generates class predictions (thresholding at 0.5)."""
        # Ensure input is suitable for Keras model (e.g., numpy array)
        if isinstance(X, pd.DataFrame):
            X_input = X.values
        else:
            X_input = X

        probabilities = self.model.predict(X_input)
        # Flatten if shape is (n_samples, 1)
        if probabilities.ndim == 2 and probabilities.shape[1] == 1:
            probabilities = probabilities.flatten()
        return (probabilities >= 0.5).astype(int)

    def predict_proba(self, X):
        """Generates class probability estimates."""
        # Ensure input is suitable for Keras model
        if isinstance(X, pd.DataFrame):
            X_input = X.values
        else:
            X_input = X

        # Get probability of the positive class from the Keras model
        p1 = self.model.predict(X_input)
        # Flatten if shape is (n_samples, 1)
        if p1.ndim == 2 and p1.shape[1] == 1:
            p1 = p1.flatten()
        # Calculate probability of the negative class
        p0 = 1.0 - p1
        # Stack probabilities into shape [n_samples, 2]
        return np.vstack([p0, p1]).T

    # Add necessary methods for sklearn compatibility if needed further
    def get_params(self, deep=True):
        return {'model': self.model}

    def set_params(self, **params):
        if 'model' in params:
            self.model = params['model']
        return self

def load_hyperparameter_space():
    """
    Define hyperparameter space for MLP tuning.
    
    Returns:
        dict: Hyperparameter space configuration.
    """
    hyperparameter_space = {
        'learning_rate': {
            'type': 'float',
            'low': 1e-4,
            'high': 1e-2,
            'log': True
        },
        'hidden_layers': {
            'type': 'int',
            'low': 1,
            'high': 5
        },
        'neurons_per_layer': {
            'type': 'int',
            'low': 32,
            'high': 256,
            'step': 32
        },
        'dropout_rate': {
            'type': 'float',
            'low': 0.001,
            'high': 0.5,
            'step': 0.001
        },
        'activation': {
            'type': 'categorical',
            'choices': ['relu', 'elu', 'tanh']
        },
        'l1_regularization': {
            'type': 'float',
            'low': 0.0001,
            'high': 1e-2,
            'log': True
        },
        'l2_regularization': {
            'type': 'float',
            'low': 0.0001,
            'high': 1e-2,
            'log': True
        },
        'batch_size': {
            'type': 'int',
            'low': 16,
            'high': 128
        },
        'epochs': {
            'type': 'int',
            'low': 50,
            'high': 200,
            'step': 10
        },
        'patience': {
            'type': 'int',
            'low': 10,
            'high': 50
        },
        'class_weight_multiplier': {
            'type': 'float',
            'low': 1.0,
            'high': 5.0,
            'step': 0.05
        }
    }
    return hyperparameter_space

def preprocess_data(X_train, X_test, X_eval=None):
    is_scaler_loaded = False
    try:
        with open('src/models/scalers/scaler_mlp.pkl', 'rb') as f:
            scaler = pickle.load(f)
        logger.info("Loaded existing MLP scaler")
        is_scaler_loaded = True
    except Exception as e:
        logger.error(f"Error loading MLP scaler: {str(e)}")
        scaler = RobustScaler()
        logger.info("Created new MLP scaler")
        is_scaler_loaded = False
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    if X_eval is not None:
        X_eval_scaled = scaler.transform(X_eval)
    else:
        X_eval_scaled = None

    if not is_scaler_loaded:
        with open('src/models/scalers/scaler_mlp.pkl', 'wb') as f:
            pickle.dump(scaler, f)
    return X_train_scaled, X_test_scaled, X_eval_scaled, scaler

def create_model(model_params):
    """
    Create and compile a Keras MLP model based on provided hyperparameters.
    
    Args:
        model_params (dict): Hyperparameters for model configuration.
        
    Returns:
        keras.Model: Compiled MLP model.
    """
    try:
        input_dim = model_params.pop('input_dim')
        hidden_layers = model_params.pop('hidden_layers', 2)
        neurons_per_layer = model_params.pop('neurons_per_layer', 128)
        dropout_rate = model_params.pop('dropout_rate', 0.2)
        activation = model_params.pop('activation', 'relu')
        l1_reg = model_params.pop('l1_regularization', 0.0)
        l2_reg = model_params.pop('l2_regularization', 0.0)
        learning_rate = model_params.pop('learning_rate', 0.001)
        
        model = keras.Sequential()
        model.add(layers.InputLayer(shape=(input_dim,)))
        
        # Add hidden layers
        for _ in range(hidden_layers):
            model.add(layers.Dense(
                neurons_per_layer,
                activation=activation,
                kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg)
            ))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate))
        
        # Output layer for binary classification
        model.add(layers.Dense(1, activation='sigmoid'))
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        return model
    except Exception as e:
        logger.error(f"Error creating MLP model: {str(e)}")
        raise

def train_model(X_train, y_train, X_test, y_test, X_eval, y_eval, model_params):
    """
    Train the MLP model, wrap it, optimize threshold, return raw Keras model and metrics.
    """
    try:
        # Set the input dimension based on training data
        model_params['input_dim'] = X_train.shape[1]
        keras_model = create_model(model_params.copy())

        # Compute class weights
        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)
        class_weight = {0: 1.0, 1: (neg_count / pos_count) if pos_count > 0 else 1.0}
        class_weight[1] *= model_params.get('class_weight_multiplier', 1.0)

        # Early stopping callback
        early_stop = callbacks.EarlyStopping(
            monitor='val_auc', # Monitor validation AUC
            mode='max',
            patience=model_params.get('patience', 20),
            restore_best_weights=True,
            verbose=0
        )

        logger.info("Starting Keras model fitting...")
        keras_model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=model_params.get('epochs', 100),
            batch_size=model_params.get('batch_size', 64),
            class_weight=class_weight,
            callbacks=[early_stop],
            verbose=0
        )
        logger.info("Keras model fitting finished.")

        # Wrap the fitted Keras model
        logger.info("Wrapping fitted Keras model for threshold optimization.")
        wrapped_model = KerasMLPWrapper(keras_model, scaler)

        # Optimize threshold using the wrapped model
        best_threshold, threshold_metrics = optimize_threshold(wrapped_model, X_eval, y_eval, min_recall)

        # Return the original Keras model and the combined metrics
        return keras_model, threshold_metrics
    except Exception as e:
        logger.error(f"Error training MLP model: {str(e)}")
        raise

def optimize_hyperparameters(X_train, y_train, X_test, y_test, X_eval, y_eval, hyperparameter_space):
    """
    Run hyperparameter optimization using Optuna.
    """
    best_score = -float("inf")
    best_params = {}
    global_top_trials = []
    top_trials = []
    if not hyperparameter_space:
        hyperparameter_space = load_hyperparameter_space()

    def objective(trial):
        nonlocal best_score, best_params
        try:
            params = {}
            for param_name, param_config in hyperparameter_space.items():
                if param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name, param_config['low'], param_config['high'],
                        log=param_config.get('log', False), step=param_config.get('step')
                    )
                elif param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, param_config['low'], param_config['high']
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config['choices']
                    )
            # Train model - train_model now returns raw Keras model and metrics dict
            model, metrics = train_model(X_train, y_train, X_test, y_test, X_eval, y_eval, params.copy())

            precision = metrics.get('precision', 0.0)
            recall = metrics.get('recall', 0.0)
            score = precision if recall >= min_recall else 0.0 # Optimize for precision
            logger.info(f"  Trial {trial.number}: Score={score:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, AUC={metrics.get('auc', 0.0):.4f}")

            for metric_name, metric_value in metrics.items():
                # Ensure serializable for Optuna
                if isinstance(metric_value, (int, float, str, bool)) or metric_value is None:
                    trial.set_user_attr(metric_name, metric_value)
                elif isinstance(metric_value, np.generic):
                    trial.set_user_attr(metric_name, metric_value.item())
                else:
                    trial.set_user_attr(metric_name, str(metric_value))
            
            if score > 0.35:
                logger.info(f"Trial {trial.number} completed with score {score:.4f}")
                log_to_mlflow(model, metrics, params, experiment_name, scaler)
            return score
        except Exception as e:
            logger.error(f"Trial failed: {str(e)}")
            return 0.0

    def callback(study, trial):
        nonlocal best_score, best_params, top_trials
        logger.info(f"Current best score in this batch: {best_score:.4f}")
        if trial.value > best_score:
            best_score = trial.value
            best_params = trial.params
            logger.info(f"New best score found in trial {trial.number}: {best_score:.4f}")
        current_run = (trial.value, trial.params, trial.number)
        top_trials.append(current_run)
        top_trials.sort(key=lambda x: x[0], reverse=True)
        top_trials[:] = top_trials[:10]
        if trial.number % 9 == 0:
            table_rows = [
                f"| {i + 1} | {rec[2]} | {rec[0]:.4f} | {rec[1]} |"
                for i, rec in enumerate(top_trials)
            ]
            logger.info("Top trials in current batch:")
            for row in table_rows:
                logger.info(row)
        if trial.number % 100 == 0 and global_top_trials:
            table_rows = [
                f"| {i + 1} | {rec[2]} | {rec[0]:.4f} | {rec[1]} |"
                for i, rec in enumerate(global_top_trials[:10])
            ]
            logger.info("Global top trials:")
            for row in table_rows:
                logger.info(row)
        return best_score

    # --- Optuna Study Execution ---
    storage_url = "sqlite:///optuna_mlp.db"
    study_name = "mlp_optimization"
    total_trials = n_trials
    batch_size = 1000
    num_batches = total_trials // batch_size
    if total_trials % batch_size != 0:
        num_batches += 1

    logger.info(f"Starting Optuna study '{study_name}' with {total_trials} trials.")
    sampler = optuna.samplers.RandomSampler(seed=random_seed)
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
        storage=storage_url,
        load_if_exists=True,
        sampler=sampler,
    )
    for _ in range(num_batches):
        try:
            study.optimize(objective, n_trials=batch_size, callbacks=[callback])
        except KeyboardInterrupt:
            logger.info("Study interrupted by user. Saving current state...")
            study.save_state(f"{study_name}_interrupted.pkl")
            break
    best_params = study.best_trial.params
    logger.info(f"Best hyperparameters: {best_params}")
    best_params.update(base_params)
    return best_params

def hypertune_mlp(experiment_name):
    """
    Run hyperparameter tuning and final training for the MLP model with MLflow tracking.
    """
    try:
        X_train_scaled, X_test_scaled, X_eval_scaled, scaler = preprocess_data(X_train, X_test, X_eval)
        hyperparameter_space = load_hyperparameter_space()
        best_params = optimize_hyperparameters(X_train_scaled, y_train, X_test_scaled, y_test, X_eval_scaled, y_eval, hyperparameter_space)
        logger.info("Training final MLP model with best hyperparameters")
        model, metrics = train_model(X_train_scaled, y_train, X_test_scaled, y_test, X_eval_scaled, y_eval, best_params.copy())
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)
        input_example = X_eval_scaled[:5].copy() if hasattr(X_eval_scaled, 'iloc') else X_eval_scaled[:5].copy()
        
        
        log_to_mlflow(model, metrics, best_params, experiment_name, input_example, scaler)
        return best_params, metrics
    except Exception as e:
        logger.error(f"Error during hypertuning: {str(e)}")
        return None, None

def log_to_mlflow(model, metrics, params, experiment_name, scaler):
    """
    Log the final MLP model, its metrics, and parameters to MLflow.
    
    Returns:
        str: Run ID.
    """
    global X_eval
    try:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=f"mlp_final_{datetime.now().strftime('%Y%m%d_%H%M')}") as run:
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Wrap the fitted Keras model
            wrapped_model = KerasMLPWrapper(model, scaler)
            scaler_path = 'src/models/scalers/scaler_mlp.pkl'
            mlflow.log_artifact(scaler_path, artifact_path="scaler")
            # Create input example
            input_example = X_eval.iloc[:5].copy()
            # Identify and convert integer columns to float64 to prevent schema enforcement errors
            if hasattr(input_example, 'dtypes'):
                for col in input_example.columns:
                    if input_example[col].dtype.kind == 'i':
                        logger.info(f"Converting integer column '{col}' to float64 to handle potential missing values")
                        input_example[col] = input_example[col].astype('float64')
            
            signature = None
            if input_example is not None:
                try:
                    # Use wrapped model for predict_proba signature
                    signature = mlflow.models.infer_signature(
                        input_example,
                        wrapped_model.predict_proba(input_example)
                    )
                except Exception as sig_err:
                    logger.error(f"Failed to infer signature: {sig_err}")

            # Log the raw Keras model using mlflow.keras
            mlflow.sklearn.log_model(
                wrapped_model,
                artifact_path="model",
                pip_requirements=pip_requirements,
                registered_model_name=f"mlp_{datetime.now().strftime('%Y%m%d_%H%M')}",
                signature=signature
            )
            mlflow.end_run()
            run_id = run.info.run_id
            logger.info(f"MLflow run ID: {run_id}")
            return run_id
    except Exception as e:
        logger.error(f"Error logging to MLflow: {str(e)}")
        return None

def train_with_precision_target(X_train, y_train, X_test, y_test, X_eval, y_eval):
    """
    Train MLP model with focus on precision target.
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Testing features
        y_test: Testing labels
        X_eval: Evaluation features
        y_eval: Evaluation labels
    Returns:
        tuple: (best_model, best_metrics)
    """
    try:
        logger.info("Training model with precision target")
        params = base_params.copy()  # Inherits base MLP parameters
        # Specific parameters for this training run with advanced scheduling
        params.update({
            "learning_rate": 0.00022557466935731258,
            "hidden_layers": 5,
            "neurons_per_layer": 123,
            "dropout_rate": 0.213,
            "activation": "relu",
            "l1_regularization": 0.0033497010595414604,
            "l2_regularization": 0.0012056260409240693,
            "batch_size": 126,
            "epochs": 76,
            "patience": 24,
            "class_weight_multiplier": 1.5,
        })
        X_train_scaled, X_test_scaled, X_eval_scaled, scaler = preprocess_data(X_train, X_test, X_eval)
        # Train final model with best parameters
        logger.info("Training final model with best parameters")
        model, metrics = train_model(X_train_scaled, y_train, X_test_scaled, y_test, X_eval_scaled, y_eval, params)
        # Log to MLflow
        log_to_mlflow(model, metrics, params, experiment_name, scaler)
        return model, metrics, params
    except Exception as e:
        logger.error(f"Error during MLflow artifact logging: {str(e)}")
        return mlflow.active_run().info.run_id if mlflow.active_run() else None

def main():
    """
    Main execution function for MLP model training.
    """
    try:
        logger.info("Starting MLP model training")
        global X_train, y_train, X_test, y_test, X_eval, y_eval
        dataloader = DataLoader()
        X_train, y_train, X_test, y_test, X_eval, y_eval = dataloader.load_data()
        features = import_selected_features_ensemble(model_type="mlp")
        X_train = X_train[features]
        X_test = X_test[features]
        X_eval = X_eval[features]
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Testing data shape: {X_test.shape}")
        logger.info(f"Evaluation data shape: {X_eval.shape}")
        logger.info(f"Positive class ratio (Train): {np.mean(y_train):.3f}")
        best_params, metrics = hypertune_mlp(experiment_name)
        logger.info(f"Hypertuning completed with hyperparameters: {best_params}")
        logger.info(f"Hypertuning metrics: {metrics}")

        # Optional seed-based fine-tuning for improved precision
        train_with_precision_target(X_train, y_train, X_test, y_test, X_eval, y_eval)

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main() 