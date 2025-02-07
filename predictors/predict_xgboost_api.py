import mlflow
import pandas as pd
import numpy as np
from pathlib import Path
import json
import os
import sys
from typing import Dict, Any, List

import pymongo
import mlflow.xgboost
from pymongo import MongoClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.create_evaluation_set import get_selected_api_columns_draws, get_real_api_scores_from_excel, setup_mlflow_tracking, create_prediction_set_api


mlruns_dir = setup_mlflow_tracking("xgboost_draw_model")

class DrawPredictor:
    """Predictor class for draw predictions using the stacked model."""
    
    def __init__(self, model_uri: str):
        """Initialize predictor with model URI."""
        # Set up MLflow tracking URI based on current environment
        current_dir = os.getcwd()
        
        
        self.model = mlflow.xgboost.load_model(model_uri)
        self.required_columns = get_selected_api_columns_draws()
        # Load the threshold from the model if available
        self.threshold = 0.50
    
    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate input dataframe has all required columns."""
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions and return results with probabilities."""
        
        predict_df = df[self.required_columns]
        # self._validate_input(predict_df)
        # Get probabilities
        probas = self.model.predict_proba(predict_df)
        predictions = self.model.predict(predict_df)
   

        # Prepare results
        results = {
            'predictions': predictions.tolist(),
            'draw_probabilities': probas[:, 1].tolist(),
            'threshold_used': self.threshold,
            'num_predictions': len(predictions),
            'positive_predictions': int(predictions.sum()),
            'prediction_rate': float(predictions.mean())
        }
        print(f"results: {results['prediction_rate']}")
        return results
    
    def predict_single_match(self, row: pd.Series) -> Dict[str, Any]:
        """Make prediction for a single match."""
        df = pd.DataFrame([row])
        probas = self.model.predict_proba(df)
        prediction = self.model.predict(df)
        
        return {
            'is_draw': bool(prediction[0]),
            'draw_probability': float(probas[0, 1]),
            'confidence': float(max(probas[0]))
        }

def _preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data to handle number formats and missing values."""
    # Create a copy to avoid modifying original data
    df = df.copy()
    
    # Replace comma with dot for ALL numeric-like columns
    for col in df.columns:
        try:
            if col in df.columns:
                # print(f"Converting column {col} (type: {df[col].dtype})")
                df[col] = (
                    df[col]
                    .apply(lambda x: str(x) if pd.notnull(x) else '0')
                    .str.strip()
                    .str.replace('[^0-9.eE-]', '', regex=True)
                    .apply(lambda x: '0' if x in ['e', 'e-', 'e+'] else x)
                    .apply(lambda x: '1' + x if x.lower().startswith(('e', 'e-', 'e+')) else x)
                    .apply(lambda x: x.replace('-', 'e-', 1) if '-' in x and 'e' not in x.lower() else x)
                    .replace('', '0')
                    .pipe(pd.to_numeric, errors='coerce')
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0)
                )
        except Exception as e:
            print(f"Error converting column {col}: {str(e)}")
            continue
    
    return df

def make_prediction(prediction_data, model_uri, selected_columns, real_scores_df) -> pd.DataFrame:
    """Make predictions and return results with probabilities."""
    try:
        # Initialize default values
        precision = 0.0
        draws_recall = 0.0
        
        # Initialize predictor
        predictor = DrawPredictor(model_uri)
        prediction_df = prediction_data.copy()
        
        print(f"len(prediction_df): {len(prediction_df)}")
        
        # Make predictions first
        results = predictor.predict(prediction_data[selected_columns])
        
        # Add predictions to dataframe
        prediction_df['draw_predicted'] = results['predictions']
        prediction_df['draw_probability'] = [round(prob, 2) for prob in results['draw_probabilities']]
        prediction_df['fixture_id'] = prediction_df['fixture_id'].astype(int)
        # Get real scores and merge - this is where the error occurs
        if 'fixture_id' in prediction_df.columns:
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
                matches_with_results = prediction_df.merge(
                    real_scores_df, 
                    on='fixture_id', 
                    how='left'
                )
                
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
                        print(f"True Positives: {true_positives}")
                        print(f"False Positives: {false_positives}")
                        print(f"True Negatives: {true_negatives}")
                        print(f"False Negatives: {false_negatives}")
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
                    else:
                        print("Warning: No match outcomes available for metric calculation")
            else:
                print("Warning: No real scores data available")
                matches_with_results = prediction_df.copy()
        
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
        'd5d3379009334751ad89fdc835c2bbb2'
    ]

    # Get preprocessed prediction data using standardized function
    prediction_df = create_prediction_set_api()
    prediction_data = prediction_df.copy()
    print(f"Loaded {len(prediction_data)} matches for prediction")
    
    real_scores_df = get_real_api_scores_from_excel(prediction_data['fixture_id'].tolist())
    print(f"real_scores_df: {len(real_scores_df)}")
    
    # Get selected columns using standardized function
    selected_columns = get_selected_api_columns_draws()
    print(f"Number of selected columns: {len(selected_columns)}")
    
    # Evaluate each model
    for uri in model_uris:
        try:
            uri_full = f"runs:/{uri}/xgboost_api_model"
            predicted_df, precision, draws_recall = make_prediction(prediction_data, uri_full, selected_columns, real_scores_df)
            
            # Add validation check
            if not isinstance(predicted_df, pd.DataFrame) or predicted_df.empty:
                print(f"Skipping invalid predictions from model {uri}")
                continue
                
            if precision > best_precision and draws_recall > 0.1:
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
   
    # Save results
    output_path = Path("./data/prediction/predictions_xgboost_api.xlsx")
    predicted_df.to_excel(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")

if __name__ == "__main__":
    main()