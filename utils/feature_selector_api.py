import pandas as pd
import numpy as np
from typing import List, Tuple
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import os
import sys

# Add project root to Python path
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
    def __init__(self, importance_threshold: float = 0.001, min_features: int = 45):
        self.importance_threshold = importance_threshold
        self.min_features = min_features
        self.logger = ExperimentLogger()
        self.selected_features = None
        
    def select_features(self, X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> List[str]:
        """Select features using multiple XGBoost importance metrics."""
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            tree_method='hist',
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
        
        # Fit model
        model.fit(X, y, eval_set=[(X_test, y_test)], verbose=False)
        
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
            importance_df['gain_norm'] * 0.5 +
            importance_df['weight_norm'] * 0.3 +
            importance_df['cover_norm'] * 0.2
        )
        
        # Sort by composite score
        importance_df = importance_df.sort_values('composite_score', ascending=False)
        
        # Ensure minimum number of features
        n_features = max(self.min_features, int(len(importance_df[importance_df['composite_score'] > 0]) * 0.8))
        selected_features = importance_df.head(n_features)['feature'].tolist()
        
        # Print feature importance analysis
        print("\nFeature Importance Analysis:")
        print("-" * 80)
        print(f"Total features analyzed: {len(importance_df)}")
        print(f"Features with non-zero importance: {len(importance_df[importance_df['composite_score'] > 0])}")
        print(f"Features selected: {len(selected_features)}")
        
        # Print top features with scores
        print("\nTop Selected Features:")
        print("-" * 80)
        for i, (feature, score) in enumerate(
            importance_df[['feature', 'composite_score']].head(n_features).values, 1
        ):
            print(f"{i:2d}. {feature:<40} {score:.4f}")
        
        self.selected_features = selected_features
        return selected_features

def load_data_without_selected_columns_api(data_path: str) -> pd.DataFrame:
    """Load data from the specified path without using selected columns."""
    data = pd.read_csv(data_path)
    
    # Create target variable
    data['is_draw'] = (data['match_outcome'] == 2).astype(int)
    data.drop(columns=['match_outcome','home_goals','away_goals','Home','Away'], inplace=True)
    
    # Replace inf and nan values
    data = data.replace([np.inf, -np.inf], 0)
    data = data.fillna(0)
    
    # Convert all numeric-like columns to numeric types, handling errors by coercing
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        
    # Convert all numeric-like columns
    for col in data.columns:
        if not np.issubdtype(data[col].dtype, np.number):
            try:
                # Convert string numbers with either dots or commas
                data[col] = (data[col].astype(str)
                           .str.strip()
                           .str.strip("'\"")
                           .str.replace(' ', '')
                           .str.replace(',', '.')
                           .replace('', '0')  # Replace empty strings with 0
                           .astype(float))
            except (ValueError, AttributeError) as e:
                print(f"Could not convert column {col}: {str(e)}")
                data = data.drop(columns=[col], errors='ignore')
                continue
            
    print(data.shape)
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
    
    return X_train, y_train, X_test, y_test


def main():
    # Load data
    data_path = "data/api_training_final.csv"
    X_train, y_train, X_test, y_test = import_feature_select_draws_api()
    
    # Drop non-numeric columns
    non_numeric_cols = ['Referee', 'Venue', 'Home', 'Away']
    X_train = X_train.drop(columns=non_numeric_cols, errors='ignore')
    X_test = X_test.drop(columns=non_numeric_cols, errors='ignore')
    
    # Initialize feature selector
    selector = XGBoostFeatureSelector(min_features=45)
    
    # Select features
    selected_features = selector.select_features(X_train, y_train, X_test, y_test)
    
    # Print selected features
    print("\nSelected Features:")
    print("-" * 80)
    for i, feature in enumerate(selected_features, 1):
        print(f"{i:2d}. {feature}")
    print(f"\nTotal features selected: {len(selected_features)}")

if __name__ == "__main__":
    main()
