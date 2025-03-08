import mlflow
import pandas as pd
import numpy as np
from pathlib import Path
import json
import os
import sys
from typing import Dict, Any, List, Tuple
import pymongo
import mlflow.sklearn
import mlflow.pyfunc
import mlflow.xgboost
import mlflow
from pymongo import MongoClient
from sklearn.metrics import recall_score, f1_score
from xgboost import XGBClassifier
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.create_evaluation_set import get_real_api_scores_from_excel, setup_mlflow_tracking, create_prediction_set_ensemble

experiment_name = "xgboost_soccer_prediction"
mlruns_dir = setup_mlflow_tracking(experiment_name)


class DrawPredictor:
    """Predictor class for draw predictions using the stacked model."""
    def __init__(self, model_uri: str):
        """Initialize predictor with model URI."""
        try:
            # Load model directly using mlflow.xgboost
            # Try loading with xgboost flavor first
            try:
                print(f"Loading model using xgboost flavor from {model_uri}")
                self.model = mlflow.xgboost.load_model(model_uri)
                print(f"Successfully loaded model using xgboost flavor from {model_uri}")
            except Exception as e:
                print(f"Failed to load model with xgboost flavor")
                raise e
            
            # Get model metadata for feature names
            model_info = mlflow.models.get_model_info(model_uri)
            if model_info.signature:
                # Parse signature inputs if they're in string format
                inputs = model_info.signature.inputs.to_dict()
                if isinstance(inputs, str):
                    inputs = json.loads(inputs)
                self.required_features = [col['name'] for col in inputs]
                print(f"Loaded {len(self.required_features)} feature names")
            else:
                print("Warning: No signature found in model metadata")
                self.required_features = []
            
            # Set threshold
            self.threshold = getattr(self.model, 'optimal_threshold', 0.27)
            print(f"Using threshold: {self.threshold:.2%}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error: {e}")
            raise

    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate input dataframe has all required columns."""
        missing_cols = set(self.required_features) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions and return results with probabilities."""
        # Validate input if needed
        if self.required_features:
            self._validate_input(df)
        
        try:
            # Initialize predictions and probabilities
            predictions = None
            pos_probas = None
            
            # First try direct XGBoost prediction
            if hasattr(self.model, '_Booster'):
                import xgboost as xgb
                # Convert DataFrame to DMatrix for XGBoost
                dmatrix = xgb.DMatrix(df)
                # Get raw predictions from the booster
                pos_probas = self.model._Booster.predict(dmatrix)
                predictions = (pos_probas >= self.threshold).astype(int)
            else:
                # Try sklearn-style prediction
                pos_probas = self.model.predict_proba(df)[:, 1]
                predictions = (pos_probas >= self.threshold).astype(int)
            
            if pos_probas is None or predictions is None:
                # Fallback to basic predict
                predictions = self.model.predict(df)
                pos_probas = predictions.astype(float)
                print("Warning: Using binary predictions instead of probabilities")
                
        except Exception as e:
            print(f"Error during prediction: {e}")
            # Initialize with zeros as last resort
            predictions = np.zeros(len(df))
            pos_probas = np.zeros(len(df))
        
        results = {
            'predictions': predictions.tolist(),
            'draw_probabilities': pos_probas.tolist(), 
            'threshold_used': self.threshold,
            'num_predictions': len(predictions),
            'positive_predictions': int(np.sum(predictions)),
            'prediction_rate': float(np.mean(predictions))
        }
        print(f"Prediction rate: {results['prediction_rate']}")
        return results

    def _find_optimal_threshold(
        self,
        model: XGBClassifier,
        features_val: pd.DataFrame,
        target_val: pd.Series) -> Tuple[float, Dict[str, float]]:
        """Find optimal prediction threshold prioritizing precision while maintaining recall.
        Args:
            model: Trained XGBoost model
            features_val: Validation features
            target_val: Validation targets
        Returns:
            Tuple of (optimal threshold, metrics dictionary)
        """
        try:
            prediction_df = features_val.copy()
            prediction_df = prediction_df[self.required_features]
            probas = self.model.predict_proba(prediction_df)[:, 1]
            best_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'threshold': 0.5}
            best_score = 0
            
            # Focus on higher thresholds for better precision, starting from 0.5
            for threshold in np.arange(0.2, 0.90, 0.01):
                preds = (probas >= threshold).astype(int)
                true_positives = ((preds == 1) & (target_val == 1)).sum()
                false_positives = ((preds == 1) & (target_val == 0)).sum()
                true_negatives = ((preds == 0) & (target_val == 0)).sum()
                false_negatives = ((preds == 0) & (target_val == 1)).sum()
                # Calculate metrics
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                # Only consider thresholds that meet minimum recall
                if recall >= 0.20:
                    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                    f1 = f1_score(target_val, preds)
                    # Modified scoring to prioritize precision
                    score = precision
                    
                    if score > best_score:
                        best_score = score
                        best_metrics.update({
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'threshold': threshold
                        })
            self.threshold = best_metrics['threshold']
            print(f"Optimal threshold set to {self.threshold}")        
            
            if best_metrics['recall'] < 0.20:
                print(
                    f"Could not find threshold meeting recall requirement. "
                    f"Best recall: {best_metrics['recall']:.4f}"
                    f"Best precision: {best_metrics['precision']:.4f}"
                )
            print(
                f"New best threshold {best_metrics['threshold']:.3f}: "
                f"Precision={best_metrics['precision']:.4f}, Recall={best_metrics['recall']:.4f}"
            )
            return self.threshold, best_metrics
            
        except Exception as e:
            print(f"Error in threshold optimization: {str(e)}")
            raise

def make_prediction(prediction_data, model_uri, real_scores_df) -> pd.DataFrame:
    """Make predictions and return results with probabilities."""
    try:
        # Initialize default values
        precision = 0.0
        draws_recall = 0.0
        
        # Initialize predictor
        predictor = DrawPredictor(model_uri)
        prediction_df = prediction_data.copy()
        
        # Ensure data types are compatible with model expectations
        # Convert numeric columns to float64 to match model expectations
        numeric_columns = prediction_df.select_dtypes(include=['number']).columns
        for col in numeric_columns:
            prediction_df[col] = prediction_df[col].astype('float64')
        
            
        # Add column validation
        predictor._validate_input(prediction_df)
        
        # Add dtype consistency check with proper DataFrame handling
        if not isinstance(prediction_data, pd.DataFrame):
            raise TypeError("prediction_data must be a pandas DataFrame")
        
        prediction_df = prediction_df[predictor.required_features]
        # Merge prediction data with real scores to get is_draw column
        predict_df = prediction_df.merge(
            real_scores_df[['fixture_id', 'is_draw']],
            on='fixture_id',
            how='left'
        )
        # Drop rows with NaN in is_draw column
        predict_df = predict_df.dropna(subset=['is_draw'])
        print(f"Merged prediction data with real scores and dropped NaN is_draw. Shape: {predict_df.shape}")
        # Make predictions first
        threshold, best_metrics = predictor._find_optimal_threshold(predictor.model, predict_df, predict_df['is_draw'])

        results = predictor.predict(prediction_df)
        print(f"Prediction successful...")
        # Add predictions to dataframe using .loc to avoid SettingWithCopyWarning
        prediction_df = prediction_df.copy()  # Create explicit copy
        prediction_df.loc[:, 'draw_predicted'] = results['predictions']
        prediction_df.loc[:, 'draw_probability'] = [round(prob, 2) for prob in results['draw_probabilities']]
        # Get real scores and merge - this is where the error occurs
        if 'fixture_id' in prediction_data.columns:
            print(f"prediction_data.columns: {prediction_data.shape}")
            # valid_fixture_ids = prediction_df['fixture_id'].dropna().astype('Int64').tolist()   
            if not real_scores_df.empty:  # Only proceed if we have real scores  
                # Ensure is_draw column exists and is properly formatted
                if 'is_draw' not in real_scores_df.columns:
                    if 'match_outcome' in real_scores_df.columns:
                        # Create is_draw from match_outcome if available
                        real_scores_df['is_draw'] = (real_scores_df['match_outcome'] == 2).astype(int)
                    else:
                        print("Warning: No match outcome data available in real scores")
                        real_scores_df['is_draw'] = None
                # Merge with validation data
                prediction_df = prediction_df.merge(
                    prediction_data[['fixture_id', 'league_name'] + [col for col in prediction_data.columns if col not in prediction_df.columns]], 
                    on='fixture_id', 
                    how='left'
                )
                # Merge with validation data
                matches_with_results = prediction_df.merge(
                    real_scores_df, 
                    on='fixture_id', 
                    how='left'
                )
                
                # Set fixture_id as the last column in dataset by reordering columns
                if 'fixture_id' in matches_with_results.columns:
                    # Get all columns except fixture_id
                    other_cols = [col for col in matches_with_results.columns if col != 'fixture_id']
                    # Reorder columns with fixture_id at the end
                    matches_with_results = matches_with_results[other_cols + ['fixture_id']]
                # Ensure is_draw is properly typed
                if 'is_draw' in matches_with_results.columns:
                    matches_with_results['is_draw'] = matches_with_results['is_draw'].fillna(-1).astype(int)
                if len(matches_with_results) > 0 and 'is_draw' in matches_with_results.columns:
                    # Filter out rows without valid is_draw values
                    valid_matches = matches_with_results[matches_with_results['is_draw'] != -1]
                    
                    if len(valid_matches) > 0:
                        # Calculate metrics using valid matches
                        true_positives = ((valid_matches['draw_predicted'] == 1) & 
                                        (valid_matches['is_draw'] == 1)).sum()
                        false_positives = ((valid_matches['draw_predicted'] == 1) & 
                                        (valid_matches['is_draw'] == 0)).sum()
                        true_negatives = ((valid_matches['draw_predicted'] == 0) & 
                                        (valid_matches['is_draw'] == 0)).sum()
                        false_negatives = ((valid_matches['draw_predicted'] == 0) & 
                                        (valid_matches['is_draw'] == 1)).sum()
                        
                        print(f"\nDetailed Metrics:")
                        print(f"Actual Draws: {valid_matches['is_draw'].sum()}")
                        print(f"Predicted Draws: {valid_matches['draw_predicted'].sum()}")
                        
                        # Calculate metrics
                        accuracy = (true_positives + true_negatives) / len(matches_with_results)
                        
                        if true_positives + false_negatives > 0:
                            draws_recall = true_positives / (true_positives + false_negatives)
                        
                        if true_positives + false_positives > 0:
                            precision = true_positives / (true_positives + false_positives)
                        
                        print(f"\nFinal Metrics:")
                        print(f"Accuracy: {accuracy:.2%}")
                        print(f"Precision: {precision:.2%}")
                        print(f"Recall: {draws_recall:.2%}")
                        print(f"Threshold: {predictor.threshold:.2%}")
                    else:
                        print("Warning: No match outcomes available for metric calculation")
            else:
                print("Warning: No real scores data available")
                matches_with_results = prediction_df.copy()
        matches_with_results = matches_with_results.loc[:, ~matches_with_results.columns.duplicated(keep='last')]
        return matches_with_results, precision, draws_recall
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        return pd.DataFrame(), 0.0, 0.0

def main():
    best_precision = 0
    best_model_uri = None
    best_predictions = pd.DataFrame()  # Initialize empty DataFrame
    predicted_df = pd.DataFrame()  # Initialize predicted_df
    # Model URIs to evaluate
    model_uris = [
        '1a3fd3f61ec144f5bc35da087aed1f33',
        'fd9e7ead05e5402b81b7aaba70de1fd7',
        '306fcb9a007447f0a50542d7d98837a9',
        '442f29de370c4a14969a6aa5de1b60b4',
        'c3b20fd4dff9466ab5deaccf6ebee4b3',
        '459e32871ab94727a4118fecf3d7f28d',
        'ce754a75ac274f0fbc6f0043b578dcec',
        '4cc45a815e3545bca50faba6ddba70d6',
        '55d72f8c1e4a42de94eb52793e83134d'
    ]
    # Get preprocessed prediction data using standardized function
    prediction_df = create_prediction_set_ensemble()
    prediction_data = prediction_df.copy()
    print(f"Loaded {len(prediction_data)} matches for prediction")
    
    try:
        # Get real scores with error handling
        real_scores_df = get_real_api_scores_from_excel()
        print(f"real_scores_df: {len(real_scores_df)}")
    except Exception as e:
        print(f"Error processing fixture IDs: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        # Create empty DataFrame to allow continuation
        real_scores_df = pd.DataFrame()
    # Evaluate each model
    for uri in model_uris:
        try:
            uri_full = f"runs:/{uri}/model"
            predicted_df, precision, draws_recall = make_prediction(prediction_data, uri_full, real_scores_df)
            # Add validation check
            if not isinstance(predicted_df, pd.DataFrame) or predicted_df.empty:
                print(f"Skipping invalid predictions from model {uri}")
                continue
                
            # Save individual model predictions
            model_output_path = Path(f"./data/prediction/predictions_model_{uri}.xlsx")
            # Reorder columns to place draw_predicted and draw_probability last
            cols = [col for col in predicted_df.columns if col not in ['draw_predicted', 'draw_probability']]
            cols.extend(['draw_predicted', 'draw_probability'])
            predicted_df = predicted_df[cols]
            predicted_df.to_excel(model_output_path, index=False)
            print(f"Predictions for model {uri} saved to: {model_output_path}")
                
            if precision > best_precision and draws_recall > 0.20:
                best_precision = precision
                best_model_uri = uri
                best_predictions = predicted_df.copy()
                print(f"New best model: {uri} with precision: {precision:.2%}")
                print(f"Draws recall: {draws_recall:.2%}")
        except Exception as e:
            print(f"Error evaluating model {uri}: {str(e)}")
            continue
    print(f"\nBest model URI: {best_model_uri}")
    print(f"Best precision: {best_precision:.2%}")
    
    # Handle empty predictions
    if best_predictions.empty:
        print("Warning: No valid predictions generated. Creating empty result.")
        predicted_df = pd.DataFrame(columns=['fixture_id', 'draw_predicted', 'draw_probability'])
    else:
        predicted_df = best_predictions
        # Reorder columns to place draw_predicted and draw_probability last
        cols = [col for col in predicted_df.columns if col not in ['draw_predicted', 'draw_probability']]
        cols.extend(['draw_predicted', 'draw_probability'])
        predicted_df = predicted_df[cols]


if __name__ == "__main__":
    main()