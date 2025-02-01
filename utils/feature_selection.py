"""Enhanced feature selection system with composite scoring and validation.

This module implements an improved feature selection process with:
1. Optimized composite scoring
2. Correlation-based redundancy reduction
3. Stability selection via bootstrapping
4. Iterative feature elimination
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from itertools import product
import mlflow
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import sys

# Add project root to Python path
try:
    project_root = Path(__file__).parent.parent
    if not project_root.exists():
        # Handle network path by using raw string
        project_root = Path(r"\\".join(str(project_root).split("\\")))
    sys.path.append(str(project_root))
except Exception as e:
    print(f"Error setting project root path: {e}")
    sys.path.append(os.getcwd().parent)

# Import ExperimentLogger and evaluation set creation
from utils.logger import ExperimentLogger
from utils.create_evaluation_set import (
    import_feature_select_draws_api,
    setup_mlflow_tracking,
    DataProcessingError
)

class EnhancedFeatureSelector:
    """Enhanced feature selection with composite scoring and validation."""
    
    def __init__(
        self,
        n_bootstrap: int = 10,
        correlation_threshold: float = 0.90,
        target_features: Tuple[int, int] = (50, 80),
        random_state: int = 42,
        experiment_name: str = "feature_selection_optimization"
    ):
        """Initialize feature selector.
        
        Args:
            n_bootstrap: Number of bootstrap iterations
            correlation_threshold: Threshold for feature correlation
            target_features: Target range for number of features (min, max)
            random_state: Random state for reproducibility
            experiment_name: Name for MLflow experiment
        """
        self.n_bootstrap = n_bootstrap
        self.correlation_threshold = correlation_threshold
        self.target_features = target_features
        self.random_state = random_state
        
        # Initialize tracking
        self.feature_scores = {}
        self.stability_scores = {}
        self.selected_features = []
        self.correlation_groups = []
        
        # Set up logging and MLflow
        self.logger = ExperimentLogger(
            experiment_name=experiment_name,
            log_dir="logs/feature_selection"
        )
        
        # MLflow experiment setup
        self.experiment_name = experiment_name
        self.mlruns_dir = setup_mlflow_tracking(experiment_name)
        mlflow.set_experiment(self.experiment_name)
        
    def optimize_composite_weights(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model: xgb.XGBClassifier,
        weight_grid: Optional[Dict[str, List[float]]] = None
    ) -> Dict[str, float]:
        """Optimize feature importance weight combinations.
        
        Args:
            X: Feature DataFrame
            y: Target series
            model: XGBoost model instance
            weight_grid: Grid of weights to try for each metric
            
        Returns:
            Dictionary of optimal weights for each metric
        """
        self.logger.info("Starting composite weight optimization")
        
        # Default weight grid if none provided
        if weight_grid is None:
            weight_grid = {
                'gain': [0.4, 0.5, 0.6],
                'weight': [0.2, 0.3, 0.4],
                'cover': [0.1, 0.2, 0.3]
            }
            
        best_score = -np.inf
        best_weights = None
        
        # Generate all weight combinations
        weight_combinations = []
        for gain in weight_grid['gain']:
            for weight in weight_grid['weight']:
                for cover in weight_grid['cover']:
                    if abs(gain + weight + cover - 1.0) < 1e-10:  # Sum to 1
                        weight_combinations.append((gain, weight, cover))
        
        # Evaluate each combination
        for gain, weight, cover in weight_combinations:
            try:
                # Calculate composite scores
                importance_scores = self._calculate_composite_scores(
                    X, y, model, {'gain': gain, 'weight': weight, 'cover': cover}
                )
                
                # Select top features based on scores
                top_features = self._select_top_features(
                    importance_scores,
                    X,
                    min_features=self.target_features[0]
                )
                
                # Evaluate feature set
                score = self._evaluate_feature_set(X[top_features], y, model)
                
                # Log to MLflow
                mlflow.log_metrics({
                    'cv_score': score,
                    'n_features': len(top_features)
                })
                
                if score > best_score:
                    best_score = score
                    best_weights = {'gain': gain, 'weight': weight, 'cover': cover}
                    
                    # Log best weights
                    mlflow.log_metrics({
                        'best_gain_weight': gain,
                        'best_weight_weight': weight,
                        'best_cover_weight': cover,
                        'best_cv_score': score
                    })
                    
            except Exception as e:
                self.logger.error(f"Error evaluating weights {(gain, weight, cover)}: {str(e)}")
                continue
                
        # Log optimization results
        self.logger.info(f"Best weights found: {best_weights}")
        self.logger.info(f"Best cross-validation score: {best_score:.4f}")
        
        return best_weights
        
    def _calculate_composite_scores(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model: xgb.XGBClassifier,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate composite importance scores with given weights.
        
        Args:
            X: Feature DataFrame
            y: Target series
            model: XGBoost model instance
            weights: Dictionary of weights for each metric
            
        Returns:
            Dictionary of composite scores for each feature
        """
        # Fit model to get feature importance
        model.fit(X, y, eval_metric=['auc', 'aucpr'])
        
        # Get importance scores for each metric
        importance_metrics = {
            'gain': model.get_booster().get_score(importance_type='gain'),
            'weight': model.get_booster().get_score(importance_type='weight'),
            'cover': model.get_booster().get_score(importance_type='cover')
        }
        
        # Normalize scores
        scaler = MinMaxScaler()
        normalized_metrics = {}
        for metric, scores in importance_metrics.items():
            # Convert to DataFrame for normalization
            score_df = pd.DataFrame.from_dict(scores, orient='index', columns=[metric])
            normalized_metrics[metric] = scaler.fit_transform(score_df)
            
        # Calculate composite scores
        composite_scores = {}
        for feature in X.columns:
            score = 0
            for metric, weight in weights.items():
                if feature in importance_metrics[metric]:
                    score += weight * normalized_metrics[metric][feature]
            composite_scores[feature] = score
            
        return composite_scores
        
    def _select_top_features(
        self,
        scores: Dict[str, float],
        X: pd.DataFrame,
        min_features: int = 50
    ) -> List[str]:
        """Select top features based on composite scores.
        
        Args:
            scores: Dictionary of feature scores
            X: Feature DataFrame
            min_features: Minimum number of features to select
            
        Returns:
            List of selected feature names
        """
        # Sort features by score
        sorted_features = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Select features
        selected = [f[0] for f in sorted_features[:min_features]]
        
        # Log selection
        self.logger.info(f"Selected {len(selected)} features")
        return selected
        
    def _evaluate_feature_set(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model: xgb.XGBClassifier,
        cv: int = 5
    ) -> float:
        """Evaluate a feature set using cross-validation.
        
        Args:
            X: Feature DataFrame
            y: Target series
            model: XGBoost model instance
            cv: Number of cross-validation folds
            
        Returns:
            Mean cross-validation score
        """
        try:
            # Perform cross-validation
            scores = cross_val_score(
                model,
                X,
                y,
                cv=cv,
                scoring='average_precision'
            )
            return scores.mean()
        except Exception as e:
            self.logger.error(f"Error in cross-validation: {str(e)}")
            return -np.inf
            
    def analyze_correlations(
        self,
        X: pd.DataFrame,
        selected_features: List[str]
    ) -> List[List[str]]:
        """Analyze correlations between selected features.
        
        Args:
            X: Feature DataFrame
            selected_features: List of features to analyze
            
        Returns:
            List of correlated feature groups
        """
        # Calculate correlation matrix
        correlation_matrix = X[selected_features].corr()
        
        # Find highly correlated features
        correlated_groups = []
        processed_features = set()
        
        for i in range(len(selected_features)):
            if selected_features[i] in processed_features:
                continue
                
            correlated = []
            for j in range(i + 1, len(selected_features)):
                if abs(correlation_matrix.iloc[i, j]) >= self.correlation_threshold:
                    correlated.append(selected_features[j])
                    processed_features.add(selected_features[j])
                    
            if correlated:
                correlated.append(selected_features[i])
                correlated_groups.append(correlated)
                processed_features.add(selected_features[i])
                
        return correlated_groups
        
    def perform_stability_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model: xgb.XGBClassifier,
        weights: Dict[str, float],
        selection_threshold: float = 0.7
    ) -> Dict[str, float]:
        """Perform stability selection via bootstrapping.
        
        Args:
            X: Feature DataFrame
            y: Target series
            model: XGBoost model instance
            weights: Feature importance weights
            selection_threshold: Minimum selection frequency for stable features
            
        Returns:
            Dictionary of feature stability scores
        """
        self.logger.info(f"Starting stability selection with {self.n_bootstrap} iterations")
        
        feature_counts = {feature: 0 for feature in X.columns}
        feature_scores = {feature: [] for feature in X.columns}
        
      
        for i in range(self.n_bootstrap):
            try:
                # Create bootstrap sample
                indices = np.random.choice(
                    len(X),
                    size=len(X),
                    replace=True
                )
                X_boot = X.iloc[indices]
                y_boot = y.iloc[indices]
                
                # Calculate feature importance for this bootstrap
                importance_scores = self._calculate_composite_scores(
                    X_boot,
                    y_boot,
                    model,
                    weights
                )
                
                # Select top features
                selected = self._select_top_features(
                    importance_scores,
                    X_boot,
                    min_features=self.target_features[0]
                )
                
                # Update counts and scores
                for feature in selected:
                    feature_counts[feature] += 1
                    if feature in importance_scores:
                        feature_scores[feature].append(importance_scores[feature])
                        
                # Log progress
                if (i + 1) % 5 == 0:
                    self.logger.info(f"Completed {i + 1}/{self.n_bootstrap} iterations")
                    
            except Exception as e:
                self.logger.error(f"Error in bootstrap iteration {i}: {str(e)}")
                continue
        
        # Calculate stability scores
        stability_scores = {}
        for feature in X.columns:
            selection_freq = feature_counts[feature] / self.n_bootstrap
            mean_score = np.mean(feature_scores[feature]) if feature_scores[feature] else 0
            stability_scores[feature] = selection_freq * mean_score
            
        # Log stability results
        stable_features = {
            f: s for f, s in stability_scores.items()
            if feature_counts[f] / self.n_bootstrap >= selection_threshold
        }
        
        mlflow.log_metrics({
            'n_stable_features': len(stable_features),
            'mean_stability_score': np.mean(list(stability_scores.values()))
        })
        
        return stability_scores
            
    def perform_iterative_elimination(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model: xgb.XGBClassifier,
        step_size: int = 5,
        cv: int = 5,
        min_features: Optional[int] = None
    ) -> List[str]:
        """Perform iterative feature elimination with cross-validation.
        
        Args:
            X: Feature DataFrame
            y: Target series
            model: XGBoost model instance
            step_size: Number of features to remove in each iteration
            cv: Number of cross-validation folds
            min_features: Minimum number of features to retain
            
        Returns:
            List of selected feature names
        """
        self.logger.info("Starting iterative feature elimination")
        
        if min_features is None:
            min_features = self.target_features[0]
            
        remaining_features = list(X.columns)
        n_features = len(remaining_features)
        best_score = -np.inf
        best_features = remaining_features.copy()
        scores_history = []
        
        while len(remaining_features) > min_features:
            try:
                # Evaluate current feature set
                score = self._evaluate_feature_set(
                    X[remaining_features],
                    y,
                    model,
                    cv=cv
                )
                scores_history.append((len(remaining_features), score))
                
                # Update best score and features
                if score > best_score:
                    best_score = score
                    best_features = remaining_features.copy()
                    
                # Calculate feature importance
                importance_scores = self._calculate_composite_scores(
                    X[remaining_features],
                    y,
                    model,
                    weights={'gain': 0.5, 'weight': 0.3, 'cover': 0.2}
                )
                
                # Remove least important features
                sorted_features = sorted(
                    importance_scores.items(),
                    key=lambda x: x[1]
                )
                
                # Determine number of features to remove
                n_remove = min(step_size, len(remaining_features) - min_features)
                if n_remove <= 0:
                    break
                    
                # Remove features
                features_to_remove = [f[0] for f in sorted_features[:n_remove]]
                remaining_features = [f for f in remaining_features if f not in features_to_remove]
                
                # Log progress
                self.logger.info(
                    f"Removed {n_remove} features. "
                    f"Remaining: {len(remaining_features)}, "
                    f"Score: {score:.4f}"
                )
                
                # Log metrics directly without nested run
                mlflow.log_metrics({
                    'current_score': score,
                    'n_features': len(remaining_features)
                })
                
            except Exception as e:
                self.logger.error(f"Error in elimination iteration: {str(e)}")
                break
                
        # Log final results directly
        mlflow.log_metrics({
            'best_score': best_score,
            'n_selected_features': len(best_features)
        })
        
        return best_features
            
    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model: xgb.XGBClassifier
    ) -> List[str]:
        """Complete feature selection process combining all methods.
        
        Args:
            X: Feature DataFrame
            y: Target series
            model: XGBoost model instance
            
        Returns:
            List of selected feature names
        """
        self.logger.info("Starting complete feature selection process")
        
        try:
            # 1. Optimize composite weights
            best_weights = self.optimize_composite_weights(X, y, model)
            
            # 2. Perform stability selection
            stability_scores = self.perform_stability_selection(
                X, y, model, best_weights
            )
            
            # 3. Get stable features
            stable_features = [
                f for f, s in stability_scores.items()
                if s >= np.percentile(list(stability_scores.values()), 70)
            ]
            
            # 4. Analyze correlations among stable features
            correlation_groups = self.analyze_correlations(
                X[stable_features],
                stable_features
            )
            
            # 5. Remove redundant features from each correlation group
            unique_features = []
            for group in correlation_groups:
                # Keep feature with highest stability score
                best_feature = max(
                    group,
                    key=lambda x: stability_scores.get(x, 0)
                )
                unique_features.append(best_feature)
                
            # Add uncorrelated features
            uncorrelated = [
                f for f in stable_features
                if not any(f in group for group in correlation_groups)
            ]
            unique_features.extend(uncorrelated)
            
            # 6. Perform iterative elimination on remaining features
            final_features = self.perform_iterative_elimination(
                X[unique_features],
                y,
                model
            )
            
            # Store selected features
            self.selected_features = final_features
            
            return final_features
            
        except Exception as e:
            self.logger.error(f"Error in feature selection process: {str(e)}")
            raise

def run_feature_selection(
    experiment_name: str = "feature_selection_optimization"
) -> List[str]:
    """Run the complete feature selection process.
    
    Args:
        experiment_name: Name for MLflow experiment
        
    Returns:
        List of selected feature names
    """
    # Set up logging and MLflow
    logger = ExperimentLogger(
        experiment_name=experiment_name,
        log_dir="logs/feature_selection"
    )
    mlruns_dir = setup_mlflow_tracking(experiment_name)
    
    # Start single MLflow run for the entire process
    with mlflow.start_run(run_name=f"feature_selection_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        try:
            # Load data using feature selection import
            logger.info("Loading and preparing data")
            X_train, y_train, X_test, y_test = import_feature_select_draws_api()
            logger.info(f"Loaded training data with shape: {X_train.shape}")
            logger.info(f"Loaded test data with shape: {X_test.shape}")
            
            # Drop non-numeric columns
            non_numeric_cols = [
                'Referee', 'draw', 'venue_name', 'Home', 'Away', 'away_win', 'Date',
                'referee_draw_rate', 'referee_draws', 'referee_match_count'
            ]
            X_train = X_train.drop(columns=non_numeric_cols, errors='ignore')
            X_test = X_test.drop(columns=non_numeric_cols, errors='ignore')
            logger.info("Dropped non-numeric columns")
            
            # Initialize feature selector
            selector = EnhancedFeatureSelector(
                n_bootstrap=10,
                correlation_threshold=0.90,
                target_features=(50, 80),
                experiment_name=experiment_name
            )
            
            # Initialize model for feature selection
            model = xgb.XGBClassifier(
                tree_method='hist',
                device='cpu',
                random_state=42
            )
            
            # Run feature selection
            logger.info("Starting feature selection process")
            selected_features = selector.select_features(X_train, y_train, model)
            
            # Save results
            results_dir = Path("results/feature_selection")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save selected features
            features_path = results_dir / "selected_features.txt"
            with open(features_path, "w") as f:
                for feature in selected_features:
                    f.write(f"{feature}\n")
            logger.info(f"Saved selected features to {features_path}")
            mlflow.log_artifact(str(features_path))
            
            # Log metrics
            mlflow.log_metrics({
                'n_selected_features': len(selected_features),
                'train_shape': X_train.shape[0],
                'test_shape': X_test.shape[0]
            })
            
            # Log parameters
            mlflow.log_params({
                'n_bootstrap': selector.n_bootstrap,
                'correlation_threshold': selector.correlation_threshold,
                'target_features_min': selector.target_features[0],
                'target_features_max': selector.target_features[1],
                'random_state': selector.random_state
            })
            
            # Create input example and signature
            input_example = X_train[selected_features].head(1)
            signature = mlflow.models.infer_signature(
                X_train[selected_features],
                y_train
            )
            
            # Log model configuration
            mlflow.log_dict(
                {
                    'selected_features': selected_features,
                    'feature_importance': selector.feature_scores,
                    'stability_scores': selector.stability_scores
                },
                'model_config.json'
            )
            
            logger.info(f"Selected {len(selected_features)} features")
            logger.info("Feature selection completed successfully")
            
            return selected_features
            
        except Exception as e:
            error_code = getattr(e, 'error_code', DataProcessingError.FILE_CORRUPTED)
            logger.error(f"Error in feature selection process (code {error_code}): {str(e)}")
            raise

if __name__ == "__main__":
    try:
        selected_features = run_feature_selection()
        print("\nSelected features:")
        for i, feature in enumerate(selected_features, 1):
            print(f"{i}. {feature}")
    except Exception as e:
        print(f"Error: {str(e)}") 