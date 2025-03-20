import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class ResidualBlock(nn.Module):
    """
    A simple residual block with two fully-connected layers
    """
    def __init__(self, input_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out += identity  # Residual connection
        out = self.relu(out)
        return out

class SmallResNet(nn.Module):
    """
    Small residual network for meta learning with 2-3 residual blocks
    """
    def __init__(self, input_dim, hidden_dim=32, num_blocks=2):
        super(SmallResNet, self).__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])
        
        # Output layer
        self.output = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.input_proj(x)
        x = nn.functional.relu(x)
        
        # Apply residual blocks
        for block in self.res_blocks:
            x = block(x)
            
        # Output probability
        x = self.output(x)
        x = self.sigmoid(x)
        return x

class ResNetMetaLearner(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible wrapper for the SmallResNet model
    Enables using the residual network as a meta learner in the ensemble
    """
    def __init__(self, input_dim=None, hidden_dim=32, num_blocks=2, lr=0.001, 
                    batch_size=64, epochs=100, patience=10, min_recall=0.2, **kwargs):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.min_recall = min_recall
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = 0.5
        self.device = torch.device("cpu")  # Following project requirements for CPU-only
        
        # Store any additional kwargs as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
        
    def fit(self, X, y, X_val=None, y_val=None):
        # If input_dim not specified, infer from input data
        if self.input_dim is None:
            self.input_dim = X.shape[1]
        
        # Initialize model
        self.model = SmallResNet(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_blocks=self.num_blocks
        ).to(self.device)
        
        # Prepare data
        X = self.scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y.reshape(-1, 1)).to(self.device)
        
        # Prepare validation data if provided
        if X_val is not None and y_val is not None:
            X_val = self.scaler.transform(X_val)
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val.reshape(-1, 1)).to(self.device)
            has_validation = True
        else:
            has_validation = False
        
        # Prepare data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()
        
        # Training loop
        best_val_metric = 0
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # Validation
            if has_validation:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_preds = val_outputs.cpu().numpy()
                    
                    # Find best threshold optimizing for precision with minimum recall
                    best_threshold, best_precision, best_recall = 0.5, 0, 0
                    
                    for threshold in np.linspace(0.1, 0.9, 81):
                        y_pred = (val_preds >= threshold).astype(int)
                        precision, recall = self._calculate_precision_recall(y_val, y_pred)
                        
                        if recall >= self.min_recall and precision > best_precision:
                            best_precision = precision
                            best_recall = recall
                            best_threshold = threshold
                    
                    # Early stopping logic
                    current_metric = best_precision if best_recall >= self.min_recall else 0
                    
                    if current_metric > best_val_metric:
                        best_val_metric = current_metric
                        patience_counter = 0
                        # Save best model
                        best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                        self.threshold = best_threshold
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= self.patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
                        
                    print(f"Epoch {epoch}, Loss: {epoch_loss/len(dataloader):.4f}, "
                            f"Precision: {best_precision:.4f}, Recall: {best_recall:.4f}, "
                            f"Threshold: {best_threshold:.2f}")
            else:
                print(f"Epoch {epoch}, Loss: {epoch_loss/len(dataloader):.4f}")
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            
        return self

    def predict_proba(self, X):
        # Scale input
        X = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            raw_preds = self.model(X_tensor).cpu().numpy()
            
        # Format as 2D array for sklearn compatibility
        probas = np.zeros((len(X), 2))
        probas[:, 1] = raw_preds.flatten()
        probas[:, 0] = 1 - probas[:, 1]
        
        return probas

    def predict(self, X):
        probas = self.predict_proba(X)[:, 1]
        return (probas >= self.threshold).astype(int)
   
    def _calculate_precision_recall(self, y_true, y_pred):
        # Calculate precision and recall
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        
        return precision, recall