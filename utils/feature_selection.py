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
from datetime import datetime
import os
import sys
from sklearn.metrics import precision_score, recall_score
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator

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
    create_evaluation_sets_draws_api,
    setup_mlflow_tracking,
    DataProcessingError
)

class EnhancedFeatureSelector(BaseEstimator):
    """Enhanced feature selection with composite scoring and validation."""
    
    def __init__(
        self,
        n_bootstrap: int = 10,
        correlation_threshold: float = 0.90,
        target_features: Tuple[int, int] = (50, 80),
        random_state: int = 42,
        experiment_name: str = "feature_selection_optimization",
        logger: ExperimentLogger = None
    ):
        """Initialize feature selector.
        
        Args:
            n_bootstrap: Number of bootstrap iterations
            correlation_threshold: Threshold for feature correlation
            target_features: Target range for number of features (min, max)
            random_state: Random state for reproducibility
            experiment_name: Name for MLflow experiment
        """
        super(EnhancedFeatureSelector, self).__init__()
        self.n_bootstrap = n_bootstrap
        self.correlation_threshold = correlation_threshold
        self.target_features = target_features
        self.random_state = random_state
        
        # Initialize tracking
        self.feature_scores = {}
        self.stability_scores = {}
        self.selected_features = []
        self.correlation_groups = []
        
        # Initialize logger
        self.logger = logger or ExperimentLogger('feature_selection')
        
        # MLflow experiment setup
        self.experiment_name = experiment_name
        self.mlruns_dir = setup_mlflow_tracking(experiment_name)
        mlflow.set_experiment(self.experiment_name)
        
    @property
    def _tags(self):
        # Manuális sklearn tags beállítás
        return {'allow_nan': True, 'requires_y': True}
        
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
                
                # Convert scores to DataFrame for proper indexing
                scores_df = pd.DataFrame.from_dict(importance_scores, orient='index', columns=['score'])
                
                # Select top features based on scores
                top_features = self._select_top_features(
                    scores_df['score'].to_dict(),  # Convert back to dict with proper indexing
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
                
        # Handle case where no valid weights are found
        if best_weights is None:
            self.logger.warning("No valid weights found, using defaults")
            best_weights = {'gain': 0.5, 'weight': 0.3, 'cover': 0.2}
            
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
        model.fit(X, y)
        
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
            normalized_metrics[metric] = pd.DataFrame(
                scaler.fit_transform(score_df),
                index=score_df.index
            )
        
        # Calculate composite scores
        composite_scores = {}
        for feature in X.columns:
            score = 0
            for metric, weight in weights.items():
                if feature in importance_metrics[metric]:
                    score += weight * normalized_metrics[metric].loc[feature].values[0]
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
            # Create a fresh model instance for cross-validation
            cv_model = xgb.XGBClassifier(
                **{k: v for k, v in model.get_params().items() 
                   if k != 'eval_metric'}  # Remove eval_metric for CV
            )
            
            # Perform cross-validation
            scores = cross_val_score(
                cv_model,
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
                # Note: eval_metric removed from fit() calls in _calculate_composite_scores
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

class PrecisionFocusedFeatureSelector(BaseEstimator):
    """Enhanced feature selector with class imbalance handling and calibration."""
    
    def __init__(self,
                 min_recall: float = 0.20,
                 target_precision: float = 0.50,
                 logger: ExperimentLogger = None,
                 handle_imbalance: bool = True,
                 calibrate_probas: bool = True):
        """Initialize with imbalance handling options.
        
        Args:
            min_recall: Minimum required recall
            target_precision: Target precision to achieve
            logger: Logger instance
            handle_imbalance: Whether to use SMOTE for imbalance
            calibrate_probas: Whether to calibrate probabilities
        """
        super(PrecisionFocusedFeatureSelector, self).__init__()
        self.min_recall = min_recall
        self.target_precision = target_precision
        self.logger = logger or ExperimentLogger('precision_focused_feature_selector')
        self.handle_imbalance = handle_imbalance
        self.calibrate_probas = calibrate_probas
        self.selected_features = []
        self.feature_scores = {}
        self.created_interactions = set()  # Track created interactions
        
    @property
    def _tags(self):
        # Manuális sklearn tags beállítás
        return {
            'allow_nan': True,
            'requires_y': True,
            'non_deterministic': False,
            'requires_fit': True,
            'preserves_dtype': [np.float64],
            'skip_validation': True
        }
        
    def _handle_class_imbalance(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply SMOTE to handle class imbalance."""
        try:
            if not self.handle_imbalance:
                return X, y
                
            self.logger.info("Applying SMOTE for class imbalance")
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            self.logger.info(f"Original class distribution: {pd.Series(y).value_counts(normalize=True)}")
            self.logger.info(f"Resampled class distribution: {pd.Series(y_resampled).value_counts(normalize=True)}")
            
            return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
            
        except Exception as e:
            self.logger.error(f"Error in class imbalance handling: {str(e)}")
            return X, y
            
    def _calibrate_model(self, model: xgb.XGBClassifier, X: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier:
        """Calibrate model probabilities using Platt scaling."""
        try:
            if not self.calibrate_probas:
                return model
                
            self.logger.info("Calibrating model probabilities")
            calibrated_model = CalibratedClassifierCV(
                model,
                cv=5,
                method='sigmoid'  # Platt scaling
            )
            calibrated_model.fit(X, y)
            
            return calibrated_model
            
        except Exception as e:
            self.logger.error(f"Error in model calibration: {str(e)}")
            return model
            
    def analyze_feature_importance(self,
                                 model: xgb.XGBClassifier,
                                 feature_names: List[str],
                                 X_val: pd.DataFrame,
                                 y_val: pd.Series) -> pd.DataFrame:
        """Analyze feature importance with enhanced precision focus."""
        try:
            self.logger.info("Handling class imbalance")
            X_balanced, y_balanced = self._handle_class_imbalance(X_val, y_val)
            
            self.logger.info("Calibrating model if enabled")
            calibrated_model = self._calibrate_model(model, X_balanced, y_balanced)
            
            self.logger.info("Getting base importance scores")
            importance_base = model.feature_importances_
            
            self.logger.info("Calculating precision impact scores with balanced data")
            precision_impact = self._calculate_precision_impact(
                calibrated_model, X_balanced, y_balanced, feature_names
            )
            
            # self.logger.info("Calculating interaction importance")
            # interaction_importance = self._calculate_interaction_importance(
            #     calibrated_model, X_balanced, y_balanced, feature_names
            # )
            
            self.logger.info("Combining scores with updated weights")
            combined_scores = (
                0.5 * precision_impact +  # Increased weight for precision impact
                0.3 * importance_base   # Base importance
                # 0.2 * interaction_importance  # New interaction component
            )
            
            self.logger.info("Creating and sorting importance DataFrame")
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_base,
                'precision_impact': precision_impact,
                # 'interaction_importance': interaction_importance,
                'combined_score': combined_scores
            }).sort_values('combined_score', ascending=False)
            
            return importance_df
            
        except Exception as e:
            self.logger.error(f"Error in feature importance analysis: {str(e)}")
            raise
            
    def _calculate_interaction_importance(self,
                                        model: xgb.XGBClassifier,
                                        X: pd.DataFrame,
                                        y: pd.Series,
                                        feature_names: List[str]) -> np.ndarray:
        """Calculate feature interaction importance."""
        try:
            interaction_scores = np.zeros(len(feature_names))
            
            
            for i, feature1 in enumerate(feature_names):
                for feature2 in feature_names[i+1:]:
                    # Create interaction feature
                    interaction_name = f'{feature1}_{feature2}'
                    X_interaction = X.copy()
                    X_interaction[interaction_name] = X[feature1] * X[feature2]
                    self.created_interactions.add(interaction_name)
                    
                    # Get predictions with interaction
                    base_score = model.score(X, y)
                    interaction_score = model.score(X_interaction, y)
                    
                    # Add interaction impact to both features
                    impact = interaction_score - base_score
                    if impact > 0:
                        interaction_scores[i] += impact
                        interaction_scores[feature_names.index(feature2)] += impact
                        
            # Log created interactions
            self.logger.info(f"Created {len(self.created_interactions)} interaction features")
            self.logger.debug(f"Interaction features: {sorted(self.created_interactions)}")
            
            return interaction_scores / interaction_scores.sum()
        
        except Exception as e:
            self.logger.error(f"Error in interaction importance calculation: {str(e)}")
            raise


    def _calculate_precision_impact(
        self,
        model: xgb.XGBClassifier,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: List[str]
    ) -> np.ndarray:
        """Calculate the precision impact of each feature by removing it and measuring the change.
        
        Args:
            model: Trained XGBoost model
            X: Feature DataFrame
            y: Target series
            feature_names: List of feature names to evaluate
            
        Returns:
            Array of precision impact scores for each feature
        """
        precision_impact = np.zeros(len(feature_names))
        
        # Get baseline precision
        baseline_precision = precision_score(y, model.predict(X))
        
        for i, feature in enumerate(feature_names):
            try:
                # Create copy of X without the current feature
                X_reduced = X.drop(columns=[feature])
                
                # Retrain model on reduced feature set
                reduced_model = xgb.XGBClassifier(
                    tree_method='hist',
                    device='cpu',
                    random_state=42
                )
                reduced_model.fit(X_reduced, y)
                
                # Calculate precision with reduced feature set
                reduced_precision = precision_score(y, reduced_model.predict(X_reduced))
                
                # Calculate precision impact
                precision_impact[i] = baseline_precision - reduced_precision
                
            except Exception as e:
                self.logger.error(f"Error calculating precision impact for feature {feature}: {str(e)}")
                precision_impact[i] = 0  # Default to 0 impact if error occurs
                
        # Normalize scores to sum to 1
        if precision_impact.sum() > 0:
            precision_impact = precision_impact / precision_impact.sum()
            
        return precision_impact

    def select_features(self,
                       importance_df: pd.DataFrame,
                       X_val: pd.DataFrame,
                       correlation_threshold: float = 0.85) -> List[str]:
        """Select optimal feature set with enhanced precision focus."""
        try:
            # Először ellenőrizzük az adatok minőségét
            self._validate_data_quality(X_val)
            
            # Candidate features validálása
            candidate_features = importance_df.sort_values(
                'combined_score', ascending=False
            )['feature'].tolist()
            
            valid_features = self._get_valid_features(X_val, candidate_features)
            
            if not valid_features:
                raise ValueError("No valid features found after validation")
            
            selected = []
            feature_groups = []
            
            self.logger.info(f"Starting feature selection with {len(valid_features)} valid features")
            
            for feature in valid_features:
                try:
                    if len(selected) == 0:
                        selected.append(feature)
                        feature_groups.append([feature])
                        continue
                        
                    # Biztonságos korrelációszámítás
                    correlations = self._calculate_safe_correlations(X_val, selected, feature)
                    
                    # Feature csoportosítás
                    self._group_feature(feature, correlations, selected, feature_groups, correlation_threshold)
                    
                except Exception as e:
                    self.logger.error(f"Error processing feature {feature}: {str(e)}")
                    continue
            
            final_features = self._select_best_features(feature_groups, importance_df)
            
            self.selected_features = final_features
            self.logger.info(f"Selected {len(final_features)} features")
            
            return final_features
            
        except Exception as e:
            self.logger.error(f"Error in feature selection: {str(e)}")
            raise

    def _validate_data_quality(self, X: pd.DataFrame) -> None:
        """Validate data quality and handle problematic values."""
        # Ellenőrizzük a NaN értékeket
        nan_cols = X.isna().sum()
        if nan_cols.any():
            self.logger.warning(f"Columns with NaN values: {nan_cols[nan_cols > 0]}")
            
        # Ellenőrizzük a konstans oszlopokat
        constant_cols = X.std() == 0
        if constant_cols.any():
            self.logger.warning(f"Constant columns detected: {X.columns[constant_cols].tolist()}")
            
        # Ellenőrizzük a végtelen értékeket
        inf_cols = np.isinf(X).sum()
        if inf_cols.any():
            self.logger.warning(f"Columns with infinite values: {inf_cols[inf_cols > 0]}")

    def _calculate_safe_correlations(self, X: pd.DataFrame, selected: List[str], feature: str) -> pd.Series:
        """Calculate correlations safely handling numerical issues."""
        try:
            # Kezeljük a NaN és végtelen értékeket
            X_clean = X.copy()
            X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
            
            # Töltsük ki a hiányzó értékeket az oszlop mediánjával
            for col in [feature] + selected:
                if X_clean[col].isna().any():
                    median_val = X_clean[col].median()
                    X_clean[col].fillna(median_val, inplace=True)
            
            # Számítsuk ki a korrelációkat
            correlations = abs(X_clean[selected].corrwith(X_clean[feature]))
            
            # Ellenőrizzük az eredményeket
            if correlations.isna().any():
                self.logger.warning(f"NaN correlations found for feature {feature}")
                correlations = correlations.fillna(0)
                
            return correlations
            
        except Exception as e:
            self.logger.error(f"Error in correlation calculation: {str(e)}")
            return pd.Series(0, index=selected)

    def _group_feature(self, 
                      feature: str, 
                      correlations: pd.Series, 
                      selected: List[str], 
                      feature_groups: List[List[str]], 
                      correlation_threshold: float) -> None:
        """Group features based on correlations."""
        group_found = False
        for group in feature_groups:
            if any(correlations[f] >= correlation_threshold for f in group):
                group.append(feature)
                group_found = True
                break
                
        if not group_found:
            selected.append(feature)
            feature_groups.append([feature])

    def _select_best_features(self, 
                             feature_groups: List[List[str]], 
                             importance_df: pd.DataFrame) -> List[str]:
        """Select best features from each group."""
        final_features = []
        for group in feature_groups:
            try:
                best_feature = max(group, key=lambda x: importance_df.loc[
                    importance_df['feature'] == x, 'combined_score'
                ].iloc[0])
                final_features.append(best_feature)
            except Exception as e:
                self.logger.error(f"Error selecting best feature from group {group}: {str(e)}")
                continue
        return final_features

def run_feature_selection(
    experiment_name: str = "feature_selection_optimization"
) -> List[str]:
    """Run the complete feature selection process with precision focus.
    
    Args:
        experiment_name: Name for MLflow experiment
        
    Returns:
        List of selected feature names
    """
    logger = ExperimentLogger(
        experiment_name=experiment_name,
        log_dir="logs/feature_selection"
    )
    
    try:
        # Load data
        logger.info("Loading and preparing data")
        X_train, y_train, X_test, y_test = import_feature_select_draws_api()
        X_val, y_val = create_evaluation_sets_draws_api(use_selected_columns=False)
        X_train, X_val, X_test = align_columns(logger, X_train, X_val, X_test)
        logger.info(f"Loaded data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # Initialize both feature selectors
        standard_selector = EnhancedFeatureSelector(
            n_bootstrap=10,
            correlation_threshold=0.85,
            target_features=(60, 100),
            experiment_name=experiment_name,
            logger=logger
        )
        
        precision_selector = PrecisionFocusedFeatureSelector(
            min_recall=0.20,
            target_precision=0.50,
            logger=logger
        )
        
        # Initialize base model for feature selection
        base_model = xgb.XGBClassifier(
            tree_method='hist',
            device='cpu',
            random_state=42,
            eval_metric=['auc', 'aucpr']
        )
        
        with mlflow.start_run(run_name=f"precision_focused_feature_selection_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            # First, get standard feature importance
            logger.info("Running standard feature selection")
            standard_features = standard_selector.select_features(X_train, y_train, base_model)
            logger.info(f"Standard features: {standard_features}")
            # Train model on selected features
            model = xgb.XGBClassifier(
                tree_method='hist',
                device='cpu',
                random_state=42
            )
            model.fit(
                X_train[standard_features],
                y_train,
                eval_set=[(X_val[standard_features], y_val)],
                verbose=False
            )
            
            # Now analyze precision impact
            logger.info("Analyzing precision impact of features")
            importance_df = precision_selector.analyze_feature_importance(
                model,
                standard_features,
                X_val[standard_features],
                y_val
            )
            
            # Select final feature set
            logger.info("Selecting final feature set with precision focus")
            final_features = precision_selector.select_features(
                importance_df,
                X_val,
                correlation_threshold=0.85
            )
            
            # Log results
            mlflow.log_metrics({
                'n_initial_features': len(standard_features),
                'n_final_features': len(final_features),
                'feature_reduction_percent': (1 - len(final_features)/len(standard_features)) * 100
            })
            
            # Save feature lists
            results_dir = Path("results/feature_selection")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save both feature sets
            with open(results_dir / "standard_features.txt", "w") as f:
                for feature in standard_features:
                    f.write(f"{feature}\n")
                    
            with open(results_dir / "precision_focused_features.txt", "w") as f:
                for feature in final_features:
                    f.write(f"{feature}\n")
                    
            # Evaluate feature sets
            logger.info("Evaluating feature sets")
            
            def evaluate_feature_set(features):
                model = xgb.XGBClassifier(
                    tree_method='hist',
                    device='cpu',
                    random_state=42
                )
                model.fit(
                    X_train[features],
                    y_train,
                    eval_set=[(X_val[features], y_val)],
                    verbose=False
                )
                
                # Get predictions
                val_probs = model.predict_proba(X_val[features])[:, 1]
                val_preds = (val_probs >= 0.5).astype(int)
                
                return {
                    'precision': precision_score(y_val, val_preds),
                    'recall': recall_score(y_val, val_preds)
                }
            
            # Compare metrics
            standard_metrics = evaluate_feature_set(standard_features)
            precision_metrics = evaluate_feature_set(final_features)
            
            mlflow.log_metrics({
                'standard_precision': standard_metrics['precision'],
                'standard_recall': standard_metrics['recall'],
                'precision_focused_precision': precision_metrics['precision'],
                'precision_focused_recall': precision_metrics['recall']
            })
            
            logger.info("\nFeature Selection Results:")
            logger.info(f"Standard Features: {len(standard_features)}")
            logger.info(f"Precision-Focused Features: {len(final_features)}")
            logger.info("\nMetrics Comparison:")
            logger.info(f"Standard - Precision: {standard_metrics['precision']:.4f}, Recall: {standard_metrics['recall']:.4f}")
            logger.info(f"Precision-Focused - Precision: {precision_metrics['precision']:.4f}, Recall: {precision_metrics['recall']:.4f}")
            
            return final_features
            
    except Exception as e:
        logger.error(f"Error in feature selection process: {str(e)}")
        raise

def align_columns(logger: ExperimentLogger, train_df: pd.DataFrame, test_df: pd.DataFrame, eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aligns the columns of the evaluation set with the training set by dropping
    columns that are missing in either DataFrame.
    
    Args:
        logger: ExperimentLogger instances
        train_df: Training DataFrame
        test_df: Test DataFrame
        eval_df: Evaluation DataFrame
        
    Returns:
        pd.DataFrame: Evaluation DataFrame with aligned columns
    """
    # Get common columns between all three DataFrames
    common_columns = set(train_df.columns).intersection(set(eval_df.columns)).intersection(set(test_df.columns))
    
    # Drop columns that are missing in either DataFrame
    # Preserve the original column order from the training set
    ordered_columns = [col for col in common_columns]
    logger.info(f"Aligned columns: {ordered_columns}")
    train_df = train_df[ordered_columns]
    test_df = test_df[ordered_columns]
    eval_df = eval_df[ordered_columns]
    
    return train_df, test_df, eval_df

def verify_interactions(train_df, test_df, eval_df, feature_names):
    """Verify that all necessary interaction features exist in all DataFrames."""
    expected_interactions = {
        f'{feature1}_{feature2}'
        for i, feature1 in enumerate(feature_names)
        for feature2 in feature_names[i+1:]
    }
    
    missing_in_train = expected_interactions - set(train_df.columns)
    missing_in_test = expected_interactions - set(test_df.columns)
    missing_in_eval = expected_interactions - set(eval_df.columns)
    
    if missing_in_train:
        print(f"Missing interactions in train: {missing_in_train}")
    if missing_in_test:
        print(f"Missing interactions in test: {missing_in_test}")
    if missing_in_eval:
        print(f"Missing interactions in eval: {missing_in_eval}")
        
    return not (missing_in_train or missing_in_test or missing_in_eval)

if __name__ == "__main__":
    try:
        selected_features = run_feature_selection()
        print("\nSelected features:")
        for i, feature in enumerate(selected_features, 1):
            print(f"{i}. {feature}")
    except Exception as e:
        print(f"Error: {str(e)}") 