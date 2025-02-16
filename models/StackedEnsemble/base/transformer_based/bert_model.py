"""BERT model implementation with CPU optimization and hyperparameter tuning."""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

# Disable TensorFlow warnings and prevent TF from being loaded
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress tensorflow warnings
os.environ["ARROW_S3_DISABLE"] = "1"
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN custom operations
os.environ["USE_TORCH"] = "TRUE"  # Force PyTorch for transformers
os.environ["USE_TF"] = "FALSE"  # Disable TensorFlow for transformers

# Import only PyTorch-based transformers components
from transformers.models.bert import BertForSequenceClassification
from transformers import (
    BertTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    AdamW,
    get_linear_schedule_with_warmup
)

import joblib
import json
import sys
import ray
import time
import ray.tune as tune
import mlflow
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm

from utils.logger import ExperimentLogger
from models.StackedEnsemble.base.model_interface import BaseModel
from models.StackedEnsemble.utils.metrics import calculate_metrics
from models.StackedEnsemble.shared.validation import NestedCVValidator
from models.StackedEnsemble.shared.mlflow_utils import MLFlowManager

# Global validator actor name
VALIDATOR_ACTOR_NAME = "global_validator"

@ray.remote
class ValidatorActor:
    """Ray actor for validation that ensures single instance."""
    
    def __init__(self, logger=None, model_type='bert'):
        """Initialize validator actor.
        
        Args:
            logger: Logger instance
            model_type: Model type
        """
        # Create a new logger instance for the actor
        self.logger = logger or ExperimentLogger('bert_hypertuning')
        self.validator = NestedCVValidator(logger=self.logger, model_type=model_type)
        self.logger.info("Created new validator instance")
        
    def optimize_hyperparameters(self, model, X, y, X_val, y_val, X_test, y_test, param_space, search_strategy):
        """Run hyperparameter optimization."""
        try:
            # Ensure logger is available
            if not hasattr(self, 'logger') or self.logger is None:
                self.logger = ExperimentLogger('bert_hypertuning')
                self.validator.logger = self.logger
            
            self.logger.info("Starting hyperparameter optimization in validator actor")
            result = self.validator.optimize_hyperparameters(
                model, X, y, X_val, y_val, X_test, y_test, param_space, search_strategy
            )
            self.logger.info("Completed hyperparameter optimization in validator actor")
            return result
        except Exception as e:
            if hasattr(self, 'logger') and self.logger is not None:
                self.logger.error(f"Error in optimize_hyperparameters: {str(e)}")
            return self._get_default_params(param_space)
    
    def _get_default_params(self, param_space):
        """Get default parameters if optimization fails."""
        try:
            # Ensure logger is available
            if not hasattr(self, 'logger') or self.logger is None:
                self.logger = ExperimentLogger('bert_hypertuning')
                self.validator.logger = self.logger
            
            self.logger.info("Getting default parameters")
            params = self.validator._get_default_params(param_space)
            self.logger.info(f"Retrieved default parameters: {params}")
            return params
        except Exception as e:
            if hasattr(self, 'logger') and self.logger is not None:
                self.logger.error(f"Error getting default parameters: {str(e)}")
            return {
                'learning_rate': 2e-5,
                'per_device_train_batch_size': 16,
                'gradient_accumulation_steps': 2,
                'num_train_epochs': 3,
                'warmup_ratio': 0.1,
                'weight_decay': 0.01,
                'hidden_dropout_prob': 0.1,
                'attention_probs_dropout_prob': 0.1,
                'max_seq_length': 256,
                'device': 'cpu',
                'fp16': False
            }
    
    def get_info(self):
        """Get validator info."""
        return "Validator is set up"

class SoccerMatchDataset(Dataset):
    """Custom dataset for soccer match data."""
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
        """Initialize dataset.
        
        Args:
            texts: List of text inputs
            labels: List of labels
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BertModel(BaseModel):
    """BERT model implementation with CPU optimization."""
    
    _validator_actor = None  # Class-level validator actor reference
    
    @classmethod
    def get_validator_actor(cls, logger=None):
        """Get or create the validator actor."""
        if cls._validator_actor is None:
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    # Try to get existing actor
                    cls._validator_actor = ray.get_actor(VALIDATOR_ACTOR_NAME)
                    logger.info("Retrieved existing validator actor")
                    break
                except ValueError:
                    try:
                        # Create new actor with proper options
                        cls._validator_actor = ValidatorActor.options(
                            name=VALIDATOR_ACTOR_NAME,
                            lifetime="detached",  # Keep actor alive across failures
                            max_restarts=-1,  # Unlimited restarts
                            max_task_retries=3  # Retry failed tasks
                        ).remote(logger)
                        logger.info("Created new validator actor")
                        break
                    except Exception as e:
                        retry_count += 1
                        logger.warning(f"Attempt {retry_count} to create validator actor failed: {str(e)}")
                        if retry_count == max_retries:
                            raise RuntimeError("Failed to create validator actor after maximum retries")
                        time.sleep(1)  # Wait before retrying
        
        return cls._validator_actor

    def __init__(self, experiment_name: str = 'bert_experiment', model_type: str = "bert", logger: ExperimentLogger = ExperimentLogger('bert_experiment')):
        """Initialize BERT model.
        
        Args:
            experiment_name: Name for MLflow experiment tracking
            model_type: Type of model (e.g., 'bert')
            logger: Logger instance
        """
        # Get project root path
        project_root = Path(__file__).parent.parent.parent.parent.parent
        
        # Set up configuration paths
        self.config_path = os.path.join(
            project_root,
            "models",
            "StackedEnsemble",
            "config",
            "model_configs",
            "bert_config.yaml"
        )
        
        # Initialize base class
        super().__init__(model_type=model_type, experiment_name=experiment_name)
        self.logger = logger
        self.mlflow = MLFlowManager(experiment_name)
        self.model = None
        self.tokenizer = None
        self.best_threshold = 0.3
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(
                num_cpus=os.cpu_count(),
                ignore_reinit_error=True,
                include_dashboard=False,
                log_to_driver=True
            )
            self.logger.info("Ray initialized for hyperparameter tuning")
        
        # Get or create validator actor
        self.validator = self.get_validator_actor(self.logger)
        
        # Load model configuration
        self.model_config = self.config_loader.load_model_config(model_type)
        self.hyperparameter_space = self.config_loader.load_hyperparameter_space(model_type)
        self.best_params = {}
        self.best_score = 0
        
        self.logger.info(f"Initialized {model_type} model with experiment name: {experiment_name}")

    def _create_model(self, **kwargs) -> BertForSequenceClassification:
        """Create and configure BERT model instance.
        
        Args:
            **kwargs: Model parameters to override defaults
            
        Returns:
            Configured BERT classifier
        """
        params = {
            'model_name': 'bert-base-uncased',
            'num_labels': 2,
            'problem_type': 'single_label_classification',
            'hidden_dropout_prob': 0.1,
            'attention_probs_dropout_prob': 0.1
        }
        
        if self.model_config:
            params.update(self.model_config.get('params', {}))
        params.update(kwargs)
        
        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(params['model_name'])
        
        # Initialize model: this loads the base encoder from the checkpoint
        # and initializes a new classification head because of num_labels.
        model = BertForSequenceClassification.from_pretrained(
            params['model_name'],
            num_labels=params['num_labels'],
            problem_type=params['problem_type']
        )
        
        # Set dropout probabilities
        model.config.hidden_dropout_prob = params['hidden_dropout_prob']
        model.config.attention_probs_dropout_prob = params['attention_probs_dropout_prob']
        
        return model

    def _convert_to_model_format(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        max_length: int = 256) -> Tuple[Dataset, Optional[torch.Tensor]]:
        """Convert data to BERT format.
        
        Args:
            X: Feature matrix
            y: Optional target vector
            max_length: Maximum sequence length
            
        Returns:
            Tuple of (dataset, labels)
        """
        if X is None:
            raise ValueError("The feature dataset X must not be None.")
        
        # Convert features to text format
        texts = X.apply(lambda x: ' '.join([f"{col}:{val}" for col, val in x.items()]), axis=1)
        
        # Create dataset
        if y is not None:
            dataset = SoccerMatchDataset(texts, y.values, self.tokenizer, max_length)
        else:
            dataset = SoccerMatchDataset(texts, np.zeros(len(texts)), self.tokenizer, max_length)
        
        return dataset

    def _train_model(
        self,
        X: Any,
        y: Any,
        X_val: Any,
        y_val: Any,
        X_test: Any,
        y_test: Any,
        **kwargs) -> Dict[str, float]:
        """Train BERT model with early stopping.
        
        Args:
            X: Training features
            y: Training labels
            X_val: Validation features
            y_val: Validation labels
            X_test: Test features
            y_test: Test labels
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary of training metrics
        """
        # Create output directory if it doesn't exist
        output_dir = Path('./results/bert_model')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate optimal training settings
        num_examples = len(X)
        batch_size = kwargs.get('per_device_train_batch_size', 16)
        grad_acc_steps = kwargs.get('gradient_accumulation_steps', 4)
        effective_batch_size = batch_size * grad_acc_steps
        num_train_epochs = kwargs.get('num_train_epochs', 3)
        
        # Compute steps per epoch and base unit
        steps_per_epoch = max(1, num_examples // effective_batch_size)
        base_step = max(1, steps_per_epoch // 8)  # Base unit for step calculations
        
        # Set evaluation and saving steps based on the base step
        eval_steps = base_step
        save_steps = base_step  # initial computed value
        
        # If a custom save_steps is provided via kwargs, use it
        if 'save_steps' in kwargs:
            save_steps = kwargs['save_steps']
        
        # Enforce that save_steps is a multiple of eval_steps when load_best_model_at_end is True.
        # This avoids the ValueError: "--load_best_model_at_end requires the saving steps to be a round multiple of the evaluation steps"
        if save_steps % eval_steps != 0:
            adjusted_save_steps = eval_steps * round(save_steps / eval_steps)
            self.logger.info(
                f"Adjusting save_steps from {save_steps} to {adjusted_save_steps} "
                f"to be a multiple of eval_steps ({eval_steps})"
            )
            save_steps = adjusted_save_steps
        
        logging_steps = base_step
        max_steps = int(num_train_epochs * steps_per_epoch)
        warmup_steps = int(max_steps * kwargs.get('warmup_ratio', 0.1))
        
        self.logger.info(f"Training configuration:")
        self.logger.info(f"- Number of examples: {num_examples}")
        self.logger.info(f"- Batch size: {batch_size}")
        self.logger.info(f"- Gradient accumulation steps: {grad_acc_steps}")
        self.logger.info(f"- Effective batch size: {effective_batch_size}")
        self.logger.info(f"- Steps per epoch: {steps_per_epoch}")
        self.logger.info(f"- Base step unit: {base_step}")
        self.logger.info(f"- Eval/save/logging steps: {eval_steps}")
        self.logger.info(f"- Max steps: {max_steps}")
        self.logger.info(f"- Warmup steps: {warmup_steps}")
        
        # Set training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,  # Larger eval batch size
            gradient_accumulation_steps=grad_acc_steps,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            weight_decay=kwargs.get('weight_decay', 0.01),
            logging_dir=str(output_dir / 'logs'),
            logging_steps=logging_steps,
            eval_steps=eval_steps,
            save_steps=save_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="precision",
            greater_is_better=True,
            save_strategy="steps",
            eval_strategy="steps",
            remove_unused_columns=False,
            dataloader_num_workers=0,
            gradient_checkpointing=False,  # Gradient checkpointing disabled
            max_grad_norm=kwargs.get('max_grad_norm', 1.0),
            learning_rate=kwargs.get('learning_rate', 2e-5),
            fp16=False,
            report_to="none",
            # Additional settings for stability
            ddp_find_unused_parameters=False,
            seed=42,
            data_seed=42,
            group_by_length=True,
            optim="adamw_torch",  # Use PyTorch's AdamW
            lr_scheduler_type="linear",
            disable_tqdm=True,  # Disable progress bars for cleaner logs
            # Handle numerical stability
            bf16=False,
            half_precision_backend="auto",
            local_rank=-1
        )
        
        # Convert data to datasets
        train_dataset = self._convert_to_model_format(X, y, kwargs.get('max_seq_length', 256))
        val_dataset = self._convert_to_model_format(X_val, y_val, kwargs.get('max_seq_length', 256))
        test_dataset = self._convert_to_model_format(X_test, y_test, kwargs.get('max_seq_length', 256))
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=self._compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=3,
                    early_stopping_threshold=0.01
                )
            ],
            # Add data collator for dynamic padding
            data_collator=None  # Let Trainer use default collator
        )
        
        # Train model
        try:
            self.logger.info("Starting model training...")
            train_result = trainer.train()
            self.logger.info(f"Training completed. Metrics: {train_result.metrics}")
            
            # Evaluate on both validation and test sets
            self.logger.info("Evaluating on validation set...")
            val_metrics = trainer.evaluate(eval_dataset=val_dataset)
            
            self.logger.info("Evaluating on test set...")
            test_metrics = trainer.evaluate(eval_dataset=test_dataset)
            
            # Combine metrics
            metrics = {
                'val_' + k: v for k, v in val_metrics.items()
            }
            metrics.update({
                'test_' + k: v for k, v in test_metrics.items()
            })
            metrics.update({
                'train_' + k: v for k, v in train_result.metrics.items()
            })
            
            # Clean up checkpoints
            for checkpoint_dir in output_dir.glob('checkpoint-*'):
                if checkpoint_dir.is_dir():
                    try:
                        import shutil
                        shutil.rmtree(checkpoint_dir)
                    except Exception as e:
                        self.logger.warning(f"Failed to remove checkpoint directory {checkpoint_dir}: {e}")
            
            self.logger.info("Training completed successfully")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'auc': 0.0,
                'brier_score': 1.0
            }

    def _compute_metrics(self, pred):
        """Compute metrics for evaluation."""
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        probs = torch.softmax(torch.tensor(pred.predictions), dim=-1)[:, 1].numpy()
        
        # Calculate metrics
        metrics = calculate_metrics(labels, preds, probs)
        return metrics

    def _predict_model(self, X: Any) -> np.ndarray:
        """Generate predictions using trained model.
        
        Args:
            X: Features to predict
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before prediction")
        
        dataset = self._convert_to_model_format(X)
        trainer = Trainer(model=self.model)
        predictions = trainer.predict(dataset)
        probas = torch.softmax(torch.tensor(predictions.predictions), dim=-1)[:, 1].numpy()
        return (probas >= self.best_threshold).astype(int)

    def _predict_proba_model(self, X: Any) -> np.ndarray:
        """Generate probability predictions.
        
        Args:
            X: Features to predict
            
        Returns:
            Array of probability predictions
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before prediction")
        
        dataset = self._convert_to_model_format(X)
        trainer = Trainer(model=self.model)
        predictions = trainer.predict(dataset)
        return torch.softmax(torch.tensor(predictions.predictions), dim=-1)[:, 1].numpy()

    def _save_model_to_path(self, path: Path) -> None:
        """Save BERT model to specified path.
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise RuntimeError("No model to save")
        
        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(str(path))
        
        # Save tokenizer
        tokenizer_path = path.parent / "tokenizer"
        self.tokenizer.save_pretrained(str(tokenizer_path))
        
        # Save threshold
        threshold_path = path.parent / "threshold.json"
        with open(threshold_path, 'w') as f:
            json.dump({'threshold': self.best_threshold}, f)
            
        self.logger.info(f"Model saved to {path}")

    def _load_model_from_path(self, path: Path) -> None:
        """Load BERT model from specified path.
        
        Args:
            path: Path to load the model from
        """
        if not path.exists():
            raise FileNotFoundError(f"No model file found at {path}")
            
        # Load model
        self.model = BertForSequenceClassification.from_pretrained(str(path))
        
        # Load tokenizer
        tokenizer_path = path.parent / "tokenizer"
        self.tokenizer = BertTokenizer.from_pretrained(str(tokenizer_path))
        
        # Load threshold
        threshold_path = path.parent / "threshold.json"
        if threshold_path.exists():
            with open(threshold_path, 'r') as f:
                self.best_threshold = json.load(f)['threshold']
        else:
            self.best_threshold = 0.3
            
        self.logger.info(f"Model loaded with threshold {self.best_threshold}")

    def fit(
        self,
        X: Any,
        y: Any,
        X_val: Any,
        y_val: Any,
        X_test: Optional[Any] = None,
        y_test: Optional[Any] = None,
        **kwargs) -> Dict[str, float]:
        """Train the BERT model with early stopping.
        
        Args:
            X: Training features
            y: Training labels
            X_val: Validation features
            y_val: Validation labels
            X_test: Testing features (optional)
            y_test: Testing labels (optional)
            **kwargs: Additional hyperparameters
            
        Returns:
            Dictionary of training metrics
        """
        self.logger.info("Starting model training")
        
        try:
            # Initialize model with hyperparameters
            self.model = self._create_model(**kwargs)
            
            # Train model
            metrics = self._train_model(X, y, X_val, y_val, X_test, y_test, **kwargs)
            self.logger.info(f"Model training completed with metrics: {metrics}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}")
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'auc': 0.0,
                'brier_score': 1.0
            }

    def predict(self, X: Any) -> np.ndarray:
        """Make predictions using the trained model.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of predictions
        """
        return self._predict_model(X)
    
    def predict_proba(self, X: Any) -> np.ndarray:
        """Get prediction probabilities.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of prediction probabilities
        """
        return self._predict_proba_model(X)

    def evaluate(self, X_val: Any, y_val: Any, optimize_threshold: bool = True) -> Dict[str, float]:
        """Evaluate model performance."""
        if optimize_threshold:
            threshold = self._optimize_threshold(X_val, y_val)
        else:
            threshold = 0.3
        
        metrics = self._calculate_metrics(X_val, y_val, threshold)
        self.mlflow.log_metrics({f'val_{k}': v for k, v in metrics.items()})
        return metrics
    
    def _optimize_threshold(self, X_val: Any, y_val: Any) -> float:
        """Optimize decision threshold based on precision-recall trade-off.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Optimized threshold value
        """
        probas = self.predict_proba(X_val)
        precision, recall, thresholds = precision_recall_curve(y_val, probas)
        
        best_threshold = 0.3
        best_score = 0.0
        
        for i, threshold in enumerate(thresholds):
            if recall[i] >= 0.15 and recall[i] < 0.9 and precision[i] > 0.30:
                score = precision[i]
                if score > best_score:
                    self.logger.info(
                        f"Threshold: {threshold:.4f}, "
                        f"Precision: {precision[i]:.4f}, "
                        f"Recall: {recall[i]:.4f}"
                    )
                    best_score = score
                    best_threshold = threshold
        
        self.best_threshold = best_threshold
        self.logger.info(f"Optimized threshold: {best_threshold:.4f}")
        return best_threshold
    
    def _calculate_metrics(self, X: Any, y: Any, threshold: float = 0.3) -> Dict[str, float]:
        """Calculate performance metrics.
        
        Args:
            X: Features
            y: True labels
            threshold: Decision threshold
            
        Returns:
            Dictionary of metrics
        """
        if threshold is None:
            threshold = 0.3
            
        probas = self.predict_proba(X)
        preds = (probas >= threshold).astype(int)
        try:
            metrics = calculate_metrics(y, preds, probas)
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            return None
        return metrics

    def optimize_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        **kwargs) -> Dict[str, Any]:
        """Optimize hyperparameters using nested cross-validation.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            X_test: Testing features
            y_test: Testing labels
            **kwargs: Additional parameters for optimization
            
        Returns:
            Dictionary of best hyperparameters
        """
        self.logger.info("Starting hyperparameter optimization")
        
        # Prepare hyperparameter space
        param_space = self._prepare_parameter_space()
        self.logger.info("Parameter space prepared for optimization")
        
        try:
            # Get optimization results from the Ray actor
            self.logger.info("Starting hyperparameter optimization with Ray actor")
            
            # Log data shapes for debugging
            self.logger.info(
                f"Data shapes for optimization:"
                f"\n - Training: {X_train.shape}"
                f"\n - Validation: {X_val.shape}"
                f"\n - Test: {X_test.shape}"
            )
            
            # Start optimization with timeout
            self.logger.info("Submitting optimization task to Ray actor")
            optimization_future = self.validator.optimize_hyperparameters.remote(
                self, X_train, y_train, X_val, y_val, X_test, y_test,
                param_space, self.hyperparameter_space.get('search_strategy', {})
            )
            
            # Wait for results with timeout and logging
            try:
                self.logger.info("Waiting for optimization results...")
                best_params = ray.get(optimization_future, timeout=3600)  # 1 hour timeout
                self.logger.info("Received optimization results from Ray actor")
            except ray.exceptions.GetTimeoutError:
                self.logger.error("Optimization timed out after 1 hour")
                return self._get_default_params(param_space)
            except Exception as e:
                self.logger.error(f"Error getting optimization results: {str(e)}")
                return self._get_default_params(param_space)
            
            # Log successful completion
            self.logger.info("Hyperparameter optimization completed successfully")
            self.logger.info(f"Best parameters found: {best_params}")
            return best_params
                
        except Exception as e:
            self.logger.error(f"Error in hyperparameter optimization: {str(e)}")
            # Get default parameters
            try:
                self.logger.info("Attempting to get default parameters")
                default_params = ray.get(self.validator._get_default_params.remote(param_space))
                self.logger.info(f"Using default parameters: {default_params}")
                return default_params
            except Exception as inner_e:
                self.logger.error(f"Error getting default parameters: {str(inner_e)}")
                return {
                    'learning_rate': 2e-5,
                    'per_device_train_batch_size': 16,
                    'gradient_accumulation_steps': 2,
                    'num_train_epochs': 3,
                    'warmup_ratio': 0.1,
                    'weight_decay': 0.01,
                    'hidden_dropout_prob': 0.1,
                    'attention_probs_dropout_prob': 0.1,
                    'max_seq_length': 256,
                    'device': 'cpu',
                    'fp16': False
                }

    def _prepare_parameter_space(self) -> Dict[str, Any]:
        """Prepare hyperparameter space for optimization."""
        self.logger.info("Preparing parameter space for optimization")
        param_space = {}
        param_ranges = {}
        
        for param, config in self.hyperparameter_space['hyperparameters'].items():
            try:
                if isinstance(config, (int, float, str)):
                    param_space[param] = config
                    param_ranges[param] = f"fixed({config})"
                    self.logger.debug(f"Added fixed parameter {param}: {config}")
                elif isinstance(config, dict) and 'distribution' in config:
                    if config['distribution'] == 'log_uniform':
                        min_val = max(config['min'], 1e-8)
                        max_val = max(config['max'], min_val + 1e-8)
                        param_space[param] = tune.uniform(min_val, max_val)
                        param_ranges[param] = f"uniform({min_val:.2e}, {max_val:.2e})"
                        self.logger.debug(f"Added log_uniform parameter {param}")
                    elif config['distribution'] == 'uniform':
                        param_space[param] = tune.uniform(config['min'], config['max'])
                        param_ranges[param] = f"uniform({config['min']:.3f}, {config['max']:.3f})"
                        self.logger.debug(f"Added uniform parameter {param}")
                    elif config['distribution'] == 'int_uniform':
                        min_val = max(1, int(config['min']))
                        max_val = max(min_val + 1, int(config['max']))
                        param_space[param] = tune.randint(min_val, max_val)
                        param_ranges[param] = f"int_uniform({min_val}, {max_val})"
                        self.logger.debug(f"Added int_uniform parameter {param}")
            except Exception as e:
                self.logger.error(f"Error processing parameter {param}: {str(e)}")
                param_space[param] = config
                param_ranges[param] = f"default({config})"
        
        # Add CPU-specific parameters
        cpu_params = {
            'device': 'cpu',
            'fp16': False,
            'fp16_opt_level': 'O1',
            'max_grad_norm': 1.0,
            'num_workers': 4
        }
        param_space.update(cpu_params)
        self.logger.info("Added CPU-specific parameters")
        
        # Log the final parameter space
        self.logger.info("Final parameter space for optimization:")
        for param, range_str in param_ranges.items():
            self.logger.info(f"{param}: {range_str}")
            
        return param_space 