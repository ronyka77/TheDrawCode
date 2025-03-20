"""
Feature Selector API for Soccer Prediction Project

This module provides a feature selection implementation using XGBoost importance metrics.
It is optimized for CPU-only training and integrates with the project's logging system.

Key Features:
- Multiple importance metrics (gain, weight, cover)
- Composite scoring system
- Detailed logging and analysis
- CPU-optimized XGBoost configuration
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import os
import sys

# Add project root to Python path with enhanced error handling
try:
    project_root = Path(__file__).parent.parent
    if not project_root.exists():
        # Handle network path by using raw string
        project_root = Path(r"\\".join(str(project_root).split("\\")))
    sys.path.append(str(project_root))
    print(f"Project root create_evaluation_set: {project_root}")
except Exception as e:
    print(f"Error setting project root path: {e}")
    # Fallback to current directory if path resolution fails
    sys.path.append(os.getcwd().parent)
    print(f"Current directory create_evaluation_set: {os.getcwd().parent}")
    
from utils.create_evaluation_set import import_feature_select_draws_api
from utils.logger import ExperimentLogger

class XGBoostFeatureSelector:
    """Feature selector using multiple XGBoost importance metrics.
    
    This class implements feature selection using a combination of XGBoost's
    importance metrics (gain, weight, cover) with a composite scoring system.
    It is optimized for CPU-only training and provides detailed logging.
    
    Attributes:
        importance_threshold (float): Minimum importance score to keep a feature
        min_features (int): Minimum number of features to retain
        logger (ExperimentLogger): Logger instance for tracking
        selected_features (List[str]): List of selected feature names
    """
    
    def __init__(self, importance_threshold: float = 0.001, min_features: int = 45):
        """Initialize the feature selector.
        
        Args:
            importance_threshold: Minimum importance score (default: 0.001)
            min_features: Minimum number of features to keep (default: 45)
        """
        self.importance_threshold = importance_threshold
        self.min_features = min_features
        self.logger = ExperimentLogger()
        self.selected_features = None
        
        # Log initialization
        self.logger.info(
            f"Initialized XGBoostFeatureSelector with threshold={importance_threshold}, "
            f"min_features={min_features}"
        )
        
    def select_features(self, X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> List[str]:
        """Select features using multiple XGBoost importance metrics.
        
        Args:
            X: Training features DataFrame
            y: Training target Series
            X_test: Test features DataFrame
            y_test: Test target Series
            
        Returns:
            List[str]: Selected feature names ordered by importance
            
        Raises:
            ValueError: If input data is invalid or empty
        """
        # Input validation
        if X.empty or y.empty:
            raise ValueError("Input data cannot be empty")
            
        self.logger.info(f"Starting feature selection with {len(X.columns)} initial features")
        
        # Initialize CPU-optimized XGBoost model
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            tree_method='hist',  # CPU-optimized algorithm
            n_estimators=10000,
            learning_rate=0.0022493372786333726,
            max_depth=4,
            min_child_weight=67,
            subsample=0.5806411377210506,
            colsample_bytree=0.8713636200673586,
            scale_pos_weight=1.2008613229403573,
            gamma=5.394482086803722,
            reg_alpha=4.209721316016628,
            reg_lambda=2.7388857488689484,
            early_stopping=300,
            verbosity=0,
            random_state=42
        )
        
        # Log model configuration
        self.logger.info("Model configuration set for CPU-optimized training")
        
        # Fit model with early stopping
        self.logger.info("Starting model training...")
        model.fit(
            X, 
            y, 
            eval_set=[(X_test, y_test)], 
            verbose=False
        )
        self.logger.info("Model training completed")
        
        # Get multiple importance metrics
        gain_scores = model.get_booster().get_score(importance_type='total_gain')
        weight_scores = model.get_booster().get_score(importance_type='weight')
        cover_scores = model.get_booster().get_score(importance_type='cover')
        
        # Combine importance scores
        importance_df = pd.DataFrame({
            'feature': list(gain_scores.keys()),
            'gain': list(gain_scores.values()),
            'weight': [weight_scores.get(f, 0) for f in gain_scores.keys()],
            'cover': [cover_scores.get(f, 0) for f in gain_scores.keys()]
        })
        
        # Add remaining features with zero importance
        missing_features = set(X.columns) - set(importance_df['feature'])
        if missing_features:
            self.logger.warning(f"Found {len(missing_features)} features with zero importance")
            missing_df = pd.DataFrame({
                'feature': list(missing_features),
                'gain': [0] * len(missing_features),
                'weight': [0] * len(missing_features),
                'cover': [0] * len(missing_features)
            })
            importance_df = pd.concat([importance_df, missing_df], ignore_index=True)
        
        # Normalize scores
        for col in ['gain', 'weight', 'cover']:
            importance_df[f'{col}_norm'] = importance_df[col] / (importance_df[col].sum() + 1e-10)
        
        # Calculate composite score with adjusted weights
        importance_df['composite_score'] = (
            importance_df['gain_norm'] * 0.5 +    # Emphasize gain
            importance_df['weight_norm'] * 0.3 +   # Moderate weight importance
            importance_df['cover_norm'] * 0.2      # Lower emphasis on coverage
        )
        
        # Sort by composite score
        importance_df = importance_df.sort_values('composite_score', ascending=False)
        
        # Ensure minimum number of features
        n_features = max(
            self.min_features, 
            int(len(importance_df[importance_df['composite_score'] > 0]) * 0.8)
        )
        selected_features = importance_df.head(n_features)['feature'].tolist()
        
        # Log feature selection results
        self.logger.info("\nFeature Selection Results:")
        self.logger.info("-" * 80)
        self.logger.info(f"Total features analyzed: {len(importance_df)}")
        self.logger.info(f"Features with non-zero importance: {len(importance_df[importance_df['composite_score'] > 0])}")
        self.logger.info(f"Features selected: {len(selected_features)}")
        
        # Log top features with scores
        self.logger.info("\nTop Selected Features:")
        self.logger.info("-" * 80)
        for i, (feature, score) in enumerate(
            importance_df[['feature', 'composite_score']].head(n_features).values, 1
        ):
            self.logger.info(f"{i:2d}. {feature:<40} {score:.4f}")
        
        self.selected_features = selected_features
        return selected_features

def load_data_without_selected_columns_api(data_path: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load and preprocess data without using predefined selected columns.
    
    Args:
        data_path: Path to the data file
        
    Returns:
        Tuple containing training and test features and targets
        
    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If data processing fails
    """
    logger = ExperimentLogger()
    logger.info(f"Loading data from: {data_path}")
    
    try:
        # Load data
        data = pd.read_excel(data_path)
        logger.info(f"Loaded data shape: {data.shape}")
        
        # Create target variable
        data['is_draw'] = (data['match_outcome'] == 2).astype(int)
        data.drop(columns=['match_outcome','home_goals','away_goals','Home','Away'], inplace=True)
        
        # Handle missing and infinite values
        data = data.replace([np.inf, -np.inf], 0)
        data = data.fillna(0)
        
        # Convert numeric columns
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            
        # Convert all numeric-like columns
        for col in data.columns:
            if not np.issubdtype(data[col].dtype, np.number):
                try:
                    data[col] = (data[col].astype(str)
                               .str.strip()
                               .str.strip("'\"")
                               .str.replace(' ', '')
                               .str.replace(',', '.')
                               .replace('', '0')
                               .astype(float))
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Could not convert column {col}: {str(e)}")
                    data = data.drop(columns=[col], errors='ignore')
                    continue
                    
        logger.info(f"Processed data shape: {data.shape}")
        
        # Split into train and test sets
        train_data, test_data = train_test_split(
            data,
            test_size=0.2,
            random_state=42,
            stratify=data['is_draw']
        )
        
        # Select features and target
        X_train = train_data.drop(columns=['is_draw'])
        y_train = train_data['is_draw']
        X_test = test_data.drop(columns=['is_draw'])
        y_test = test_data['is_draw']
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        return X_train, y_train, X_test, y_test
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def main():
    """Main function to run feature selection process."""
    logger = ExperimentLogger()
    logger.info("Starting feature selection process")
    
    try:
        # Load data
        data_path = "data/api_training_final.xlsx"
        X_train, y_train, X_test, y_test = import_feature_select_draws_api()
        
        # Drop non-numeric columns
        non_numeric_cols = [
            'Referee', 'draw', 'venue_name', 'Home', 'Away', 'away_win', 'Date',
            'referee_draw_rate', 'referee_draws', 'referee_match_count'
        ]

        X_train = X_train.drop(columns=non_numeric_cols, errors='ignore')
        X_test = X_test.drop(columns=non_numeric_cols, errors='ignore')
        
        # Initialize feature selector
        selector = XGBoostFeatureSelector(min_features=45)
        
        # Select features
        selected_features = selector.select_features(X_train, y_train, X_test, y_test)
        
        # Print results
        logger.info("\nFeature Selection Complete")
        logger.info("-" * 80)
        for i, feature in enumerate(selected_features, 1):
            logger.info(f"{i:2d}. {feature}")
        logger.info(f"\nTotal features selected: {len(selected_features)}")
        
    except Exception as e:
        logger.error(f"Error in feature selection process: {str(e)}")
        raise

if __name__ == "__main__":
    main()
