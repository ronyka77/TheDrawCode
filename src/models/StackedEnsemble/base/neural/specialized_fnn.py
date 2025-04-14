"""
Specialized Feed-Forward Neural Network for Soccer Draw Prediction

This module implements a specialized FNN architecture optimized for soccer predictions
with domain-specific enhancements including:
- Batch normalization for faster, more stable training
- Feature interaction layers to capture team relationship dynamics
- Residual connections for better gradient flow in deeper networks
- Soccer-specific feature processing (home/away/match context)
- Calibrated probability outputs

The model maintains compatibility with the existing PyTorch hypertuner workflow.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureInteractionLayer(nn.Module):
    """
    Learns pairwise interactions between features.
    
    This is particularly useful for soccer data where the interaction
    between features (e.g., home attack vs away defense) is important.
    """
    def __init__(self, input_dim, interaction_factors=8):
        super().__init__()
        self.interaction_factors = interaction_factors
        # Create factorized interaction parameters (for efficiency)
        self.factors = nn.Parameter(torch.randn(input_dim, interaction_factors))
        
    def forward(self, x):
        # Project features to a latent interaction space
        latent_factors = torch.matmul(x, self.factors)  # (batch_size, factors)
        # Compute interactions in this space (more efficient than pairwise)
        interactions = latent_factors.pow(2).sum(dim=1) - (latent_factors.pow(2).sum(dim=1))
        interactions = 0.5 * interactions.unsqueeze(1)  # (batch_size, 1)
        return interactions


class ResidualBlock(nn.Module):
    """
    Residual block with batch normalization and dropout.
    
    Helps with gradient flow in deeper networks and improves training stability.
    """
    def __init__(self, dim, dropout_rate=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        
    def forward(self, x):
        return x + self.block(x)  # Skip connection


class TemperatureScaling(nn.Module):
    """
    Learns to calibrate model probabilities through temperature scaling.
    
    This improves probability estimates, which is crucial for betting applications.
    """
    def __init__(self, init_temp=1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temp)
        
    def forward(self, logits):
        return logits / self.temperature


class SoccerFeatureProcessor(nn.Module):
    """
    Processes soccer-specific features with domain knowledge.
    
    Groups input features into:
    - Home team features
    - Away team features
    - Match context features
    """
    def __init__(self, input_dim, team_feature_pct=0.4):
        super().__init__()
        # Determine feature splits (approximate based on percentage)
        self.home_features = int(input_dim * team_feature_pct)
        self.away_features = int(input_dim * team_feature_pct)
        self.match_features = input_dim - self.home_features - self.away_features
        
        # Normalize each feature group separately
        self.home_norm = nn.BatchNorm1d(self.home_features)
        self.away_norm = nn.BatchNorm1d(self.away_features)
        self.match_norm = nn.BatchNorm1d(self.match_features) if self.match_features > 0 else None
        
    def forward(self, x):
        # Split features
        home_x = x[:, :self.home_features]
        away_x = x[:, self.home_features:self.home_features+self.away_features]
        
        # Apply normalization
        home_x = self.home_norm(home_x)
        away_x = self.away_norm(away_x)
        
        # Process match features if they exist
        if self.match_features > 0:
            match_x = x[:, self.home_features+self.away_features:]
            match_x = self.match_norm(match_x)
            # Concatenate all features
            return torch.cat([home_x, away_x, match_x], dim=1)
        else:
            # Concatenate just team features
            return torch.cat([home_x, away_x], dim=1)


class SpecializedFNN(nn.Module):
    """
    Specialized Feed-Forward Neural Network optimized for soccer prediction.
    
    Incorporates multiple enhancements:
    - Soccer-specific feature processing
    - Batch normalization for training stability
    - Feature interaction modeling
    - Residual connections
    - Calibrated probability outputs
    
    Compatible with the existing PyTorch hypertuner workflow.
    """
    def __init__(self, input_dim, **kwargs):
        super().__init__()
        # Extract parameters with sensible defaults
        hidden_units = kwargs.get("hidden_units", 64)
        num_layers = kwargs.get("num_layers", 2)
        dropout_rate = kwargs.get("dropout_rate", 0.3)
        team_feature_pct = kwargs.get("team_feature_pct", 0.4)  # 40% home, 40% away, 20% match
        use_residual = kwargs.get("use_residual", True if num_layers > 1 else False)
        use_interactions = kwargs.get("use_interactions", True)
        
        # Map activation function names to PyTorch classes
        activation_name = kwargs.get("activation_fn", "ReLU")
        if activation_name == "LeakyReLU":
            self.act_fn = nn.LeakyReLU()
        elif activation_name == "SiLU":
            self.act_fn = nn.SiLU()
        else:  # Default to ReLU
            self.act_fn = nn.ReLU()
            
        # Feature processor component
        self.feature_processor = SoccerFeatureProcessor(input_dim, team_feature_pct)
        
        # Feature interaction component (optional)
        self.use_interactions = use_interactions
        if use_interactions:
            self.interaction_layer = FeatureInteractionLayer(input_dim)
            # Adjust input dimension for the main network
            main_input_dim = input_dim + 1  # +1 for interaction output
        else:
            main_input_dim = input_dim
        
        # Main network layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(main_input_dim, hidden_units))
        layers.append(nn.BatchNorm1d(hidden_units))
        layers.append(self.act_fn)
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers with optional residual connections
        for _ in range(num_layers - 1):
            if use_residual:
                layers.append(ResidualBlock(hidden_units, dropout_rate))
            else:
                layers.append(nn.Linear(hidden_units, hidden_units))
                layers.append(nn.BatchNorm1d(hidden_units))
                layers.append(self.act_fn)
                layers.append(nn.Dropout(dropout_rate))
        
        # Output layer (logits)
        layers.append(nn.Linear(hidden_units, 1))
        
        # Combine all layers
        self.main_network = nn.Sequential(*layers)
        
        # Temperature scaling for probability calibration
        self.temperature_scaling = TemperatureScaling()
        
        # Add attributes to store scaler and device for predict_proba (required for hypertuner compatibility)
        self.scaler_ = None 
        self.device_ = None

    def forward(self, x):
        # Process features with domain knowledge
        processed_x = self.feature_processor(x)
        
        # Add interaction features if enabled
        if self.use_interactions:
            interactions = self.interaction_layer(x)
            x_with_interactions = torch.cat([processed_x, interactions], dim=1)
            logits = self.main_network(x_with_interactions)
        else:
            logits = self.main_network(processed_x)
            
        # Apply temperature scaling to logits
        calibrated_logits = self.temperature_scaling(logits)
        
        return calibrated_logits
    
    def predict_proba(self, X, batch_size=128):
        """
        Predict probabilities, mimicking scikit-learn interface.
        Requires scaler_ and device_ attributes to be set.
        Returns probabilities for both classes (0 and 1) in shape (N, 2).
        
        This method maintains compatibility with the existing hypertuner workflow.
        """
        if self.scaler_ is None or self.device_ is None:
            raise ValueError("Scaler and Device must be set on the model before calling predict_proba.")
        
        # Set model to evaluation mode
        self.eval()
        all_probs_class1 = []
        
        # Ensure X is DataFrame or Array that scaler expects
        X_scaled = self.scaler_.transform(X)
        
        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        # Process in batches to avoid memory issues
        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i:i+batch_size].to(self.device_)
                outputs = self(batch_X)
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_probs_class1.append(probs)
                
        # Concatenate probabilities for class 1
        probs_class1 = np.concatenate(all_probs_class1).flatten()
        
        # Calculate probabilities for class 0
        probs_class0 = 1.0 - probs_class1
        
        # Stack horizontally to get shape (N, 2)
        return np.column_stack((probs_class0, probs_class1))


# This function serves as a drop-in replacement for create_pytorch_model in the hypertuner
def create_specialized_fnn(model_params, input_dim, device, scaler):
    """
    Create and configure the specialized FNN model instance.
    
    This function matches the interface expected by the hypertuner workflow.
    
    Args:
        model_params (dict): Hyperparameters suggested by Optuna.
        input_dim (int): Number of input features.
        device (torch.device): The device (CPU or CUDA) to run the model on.
        scaler (StandardScaler): The fitted scaler instance.
        
    Returns:
        SpecializedFNN: Configured and device-placed FNN model instance.
    """
    # Extract relevant parameters for the model architecture
    architecture_params = {
        "num_layers": model_params.get("num_layers"),
        "hidden_units": model_params.get("hidden_units"),
        "activation_fn": model_params.get("activation_fn"), 
        "dropout_rate": model_params.get("dropout_rate"),
        "team_feature_pct": model_params.get("team_feature_pct", 0.4),
        "use_residual": model_params.get("use_residual", True),
        "use_interactions": model_params.get("use_interactions", True)
    }
    
    # Remove None values
    architecture_params = {k: v for k, v in architecture_params.items() if v is not None}
    
    # Create model instance
    model = SpecializedFNN(input_dim=input_dim, **architecture_params)
    model.to(device)
    
    # Attach scaler and device to the model instance (for predict_proba)
    model.scaler_ = scaler
    model.device_ = device
    
    return model


# Additional hyperparameters for the specialized FNN
def add_specialized_hyperparameters(hyperparameter_space):
    """
    Add specialized hyperparameters for the FNN model.
    
    Call this function to extend the existing hyperparameter space
    before passing it to Optuna for optimization.
    
    Args:
        hyperparameter_space (dict): Existing hyperparameter space.
        
    Returns:
        dict: Extended hyperparameter space with FNN-specific parameters.
    """
    specialized_params = {
        "team_feature_pct": {
            "type": "float",
            "low": 0.3,
            "high": 0.5,
            "step": 0.05
        },
        "use_residual": {
            "type": "categorical",
            "choices": [True, False]
        },
        "use_interactions": {
            "type": "categorical",
            "choices": [True, False]
        }
    }
    
    # Add specialized parameters to the existing space
    hyperparameter_space.update(specialized_params)
    return hyperparameter_space 