"""Configuration loading and validation utilities."""

from typing import Dict, Any, Optional
import os
from pathlib import Path
import yaml
from utils.logger import ExperimentLogger

class ConfigurationLoader:
    """Handles loading and validation of model configurations."""
    
    def __init__(self, model_type: str = None, logger: ExperimentLogger = None, experiment_name: str = None):
        """Initialize the configuration loader.
        
        Args:
            model_type: Type of model for configuration loading
            logger: Logger instance
            experiment_name: Name of the experiment for logging
        """
        self.model_type = model_type
        self.logger = logger or ExperimentLogger(experiment_name=experiment_name or f"{model_type}_config_loader")
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.base_config_path = self.project_root / "models" / "StackedEnsemble" / "config"

    def load_model_config(self, model_type: str = None) -> Dict[str, Any]:
        """Load model-specific configuration.
        
        Args:
            model_type: Type of model (e.g., 'catboost', 'bert', 'lightgbm', 'xgboost')
            
        Returns:
            Dictionary containing model configuration
        """
        model_type = model_type or self.model_type
        if not model_type:
            raise ValueError("Model type must be specified either in initialization or method call")
            
        config_path = self.base_config_path / "model_configs" / f"{model_type}_config.yaml"
        # self.logger.info(f"Loading configuration from {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Validate configuration
            self._validate_model_config(config, model_type)
            return config
            
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing configuration file: {e}")
            raise

    def load_hyperparameter_space(self, model_type: str = None) -> Dict[str, Any]:
        """Load hyperparameter search space configuration.
        
        Args:
            model_type: Type of model
            
        Returns:
            Dictionary containing hyperparameter search space
        """
        model_type = model_type or self.model_type
        if not model_type:
            raise ValueError("Model type must be specified either in initialization or method call")
            
        space_path = self.base_config_path / "hyperparameter_spaces" / f"{model_type}_space.yaml"
        # self.logger.info(f"Loading hyperparameter space from {space_path}")
        
        try:
            with open(space_path, 'r') as f:
                space = yaml.safe_load(f)
            
            # Validate search space
            self._validate_hyperparameter_space(space, model_type)
            return space
            
        except FileNotFoundError:
            self.logger.error(f"Hyperparameter space file not found: {space_path}")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing hyperparameter space file: {e}")
            raise
            
    def _validate_model_config(self, config: Dict[str, Any], model_type: str) -> None:
        """Validate model configuration structure.
        
        Args:
            config: Configuration dictionary to validate
            model_type: Type of model for specific validation rules
        """
        required_fields = {
            'model': {
                'name': str,
                'version': str,
                'experiment_name': str,
                'cpu_config': dict,
                'logging': {
                    'log_dir': str,
                    'metrics_tracking': list
                }
            }
        }
        
        self._validate_dict_structure(config, required_fields, f"{model_type} config")
        
    def _validate_hyperparameter_space(self, space: Dict[str, Any], model_type: str) -> None:
        """Validate hyperparameter search space structure.
        
        Args:
            space: Search space dictionary to validate
            model_type: Type of model for specific validation rules
        """
        required_fields = {
            'hyperparameters': dict,
            'optimization': {
                'n_trials': int,
                'timeout': int,
                'study_name': str,
                'direction': str,
                'metric': str
            }
        }
        
        self._validate_dict_structure(space, required_fields, f"{model_type} hyperparameter space")
        
    def _validate_dict_structure(
        self,
        data: Dict[str, Any],
        required_structure: Dict[str, Any],
        context: str
    ) -> None:
        """Validate dictionary structure against required fields.
        
        Args:
            data: Dictionary to validate
            required_structure: Required structure specification
            context: Context for error messages
        """
        for key, value_type in required_structure.items():
            if key not in data:
                raise ValueError(f"Missing required field '{key}' in {context}")
                
            if isinstance(value_type, dict):
                if not isinstance(data[key], dict):
                    raise ValueError(
                        f"Field '{key}' in {context} must be a dictionary"
                    )
                self._validate_dict_structure(data[key], value_type, f"{context}.{key}")
                
            elif isinstance(value_type, type):
                if not isinstance(data[key], value_type):
                    raise ValueError(
                        f"Field '{key}' in {context} must be of type {value_type.__name__}"
                    ) 