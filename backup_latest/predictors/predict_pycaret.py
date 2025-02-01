import mlflow
import pandas as pd
import numpy as np
from pathlib import Path
import json
import os
import sys
from typing import Dict, Any, List

import pymongo
import mlflow.pyfunc
from pymongo import MongoClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.create_evaluation_set import get_selected_columns_draws, get_real_scores_from_excel, setup_mlflow_tracking

# Set up MLflow tracking
experiment_name = "pycaret_soccer_draw"
mlruns_dir = setup_mlflow_tracking(experiment_name)

class DrawPredictor:
    """Predictor class for draw predictions using the PyCaret model."""
    
    def __init__(self, model_uri: str):
        """Initialize predictor with model URI."""
        if not model_uri.startswith(("runs:/", "models:/", "file://")):
            raise ValueError(f"Invalid model URI: {model_uri}. Use 'runs:/', 'models:/', or 'file://'.")
        
        # If the model is directly in the artifacts folder, append the model name
        if model_uri.endswith("/"):
            model_uri += "soccer_draw_model_pycaret.pkl"
            
        print(f"Model URI before loading: {model_uri}")
        self.model = mlflow.pyfunc.load_model(model_uri)
        print(f"Model loaded successfully")
        
        # Load the selected features from MLflow
        self.selected_features = self._load_selected_features(model_uri)
        print(f"Selected features loaded successfully")
        

        # Load the threshold from the model if available
        self.threshold = getattr(self.model, 'threshold', 0.55)
    
    def _load_selected_features(self, model_uri: str) -> list:
        """Load selected features from the MLflow run."""
        try:
            # Extract run ID based on URI format
            if model_uri.startswith("runs:/"):
                run_id = model_uri.split("/")[1]
                print(f"Run ID: {run_id}")
            elif model_uri.startswith("models:/"):
                # For models:/ URIs, fetch the latest version and its run ID
                model_name = model_uri.split("/")[1]
                client = mlflow.tracking.MlflowClient()
                model_version = client.get_latest_versions(model_name, stages=["None"])[0]
                run_id = model_version.run_id
            elif model_uri.startswith("file://"):
                # For file:// URIs, assume the run ID is in the path
                run_id = model_uri.split("/")[-2]
            else:
                raise ValueError(f"Unsupported model URI format: {model_uri}")
            
            # Fetch the run details from MLflow
            client = mlflow.tracking.MlflowClient()
            run = client.get_run(run_id)
            
            # Retrieve the selected features from the run's parameters
            selected_features = run.data.params.get("selected_features", None)
            if not selected_features:
                raise ValueError(f"Selected features not found in MLflow run {run_id}.")
            
            # Parse the selected features from the string representation
            try:
                # Remove brackets and split by commas
                return selected_features.strip("[]").replace("'", "").split(", ")
            except Exception as e:
                raise ValueError(f"Failed to parse selected features: {str(e)}")
            
        except Exception as e:
            print(f"Error loading selected features: {str(e)}")
            raise
    
    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate input dataframe has all required columns."""
        missing_cols = set(self.selected_features) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions and return results with probabilities."""
        self._validate_input(df)
        predict_df = df[self.selected_features]
        
        # Get probabilities
        probas = self.model.predict_proba(predict_df)
        predictions = self.model.predict(predict_df)
        
        # Prepare results
        results = {
            'predictions': predictions.tolist(),
            'draw_probabilities': probas[:, 1].tolist(),  # Probability of class 1 (draw)
            'threshold_used': self.threshold,
            'num_predictions': len(predictions),
            'positive_predictions': int(predictions.sum()),
            'prediction_rate': float(predictions.mean())
        }
        
        return results
    
    def predict_single_match(self, row: pd.Series) -> Dict[str, Any]:
        """Make prediction for a single match."""
        df = pd.DataFrame([row])
        probas = self.model.predict_proba(df)
        prediction = self.model.predict(df)
        
        return {
            'is_draw': bool(prediction[0]),
            'draw_probability': float(probas[0, 1]),  # Probability of class 1 (draw)
            'confidence': float(max(probas[0]))  # Maximum probability across classes
        }

def _preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data to handle number formats and missing values."""
    # Create a copy to avoid modifying original data
    df = df.copy()
    
    # Replace comma with dot for ALL numeric-like columns
    for col in df.columns:
        if df[col].dtype == 'object':  # Check if column is string type
            try:     
                # Replace comma with dot and convert to float
                df[col] = (df[col].astype(str)
                          .str.strip()  # Remove leading/trailing whitespace
                          .str.strip("'\"")  # Remove quotes
                          .str.replace(' ', '')  # Remove any spaces
                          .str.replace(',', '.')  # Replace comma with dot
                          .astype(float))  # Convert to float
                
            except (AttributeError, ValueError) as e:
                print(f"Could not convert column {col}: {str(e)}")
                continue
    
    # Handle date encoding if needed
    if 'Datum' in df.columns:
        earliest_date = pd.to_datetime("2020-08-11")
        df['date_encoded'] = (pd.to_datetime(df['Datum']) - earliest_date).dt.days
        df['date_encoded'] = df['date_encoded'].astype(int)
    
    return df

def make_prediction(prediction_data, model_uri) -> pd.DataFrame:
    """Make predictions and return results with probabilities."""
    try:
        # Ensure the model URI is valid
        if not model_uri.startswith(("runs:/", "models:/", "file://")):
            raise ValueError(f"Invalid model URI: {model_uri}. Use 'runs:/', 'models:/', or 'file://'.")
        
        # Initialize predictor
        predictor = DrawPredictor(model_uri)
        
        prediction_df = prediction_data.copy()
        
        # Make predictions
        results = predictor.predict(prediction_data)
        
        # Add predictions to dataframe
        prediction_df['draw_predicted'] = results['predictions']
        prediction_df['draw_probability'] = [round(prob, 2) for prob in results['draw_probabilities']]
        prediction_df['date_encoded'] = prediction_data['date_encoded']
        
        # Get real scores from MongoDB
        if 'running_id' in prediction_df.columns:
            real_scores = get_real_scores_from_excel(prediction_df['running_id'].tolist())
            
            # Add real results to output
            real_scores_df = pd.DataFrame.from_dict(real_scores, orient='index')
            real_scores_df.reset_index(inplace=True)
            real_scores_df.rename(columns={'index': 'running_id'}, inplace=True)
            
            # Merge prediction_df with real_scores_df on running_id
            prediction_df = prediction_df.merge(real_scores_df, on='running_id', how='left')
            
            # Calculate accuracy metrics
            matches_with_results = prediction_df[prediction_df['is_draw'].notna()]
            if len(matches_with_results) > 0:
                accuracy = (matches_with_results['draw_predicted'] == 
                            matches_with_results['is_draw']).mean()
                draws_recall = ((matches_with_results['draw_predicted'] == 1) & 
                                (matches_with_results['is_draw'] == True)).sum() / (matches_with_results['is_draw'] == True).sum()
                print(f"\nPrediction Accuracy: {model_uri}")
                print("-" * 80)
                print(f"Total matches with results: {len(matches_with_results)}")
                print(f"Overall accuracy: {accuracy:.2%}")
                print(f"Draws recall: {draws_recall:.2%}")
                
                # Detailed analysis of predictions
                true_positives = ((matches_with_results['draw_predicted'] == 1) & 
                                (matches_with_results['is_draw'] == True)).sum()
                false_positives = ((matches_with_results['draw_predicted'] == 1) & 
                                    (matches_with_results['is_draw'] == False)).sum()
                
                if true_positives + false_positives > 0:
                    precision = true_positives / (true_positives + false_positives)
                    print(f"Draw prediction precision: {precision:.2%}")
        
        return prediction_df, precision, draws_recall
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise

def main():
    # Set up paths
    data_path = Path("./data/prediction/api_prediction_data_new.xlsx")
   
    # Model URIs to evaluate
    model_uris = [
        '6abfeb91b0d64646a35b4d2a47bd13f3'
    ]

    best_precision = 0
    best_model_uri = None
    best_predictions = None
    best_draws_recall = 0
    
    # Load and preprocess prediction data
    prediction_df = pd.read_excel(data_path)
    if 'match_outcome' in prediction_df.columns:
        prediction_df = prediction_df.drop(columns=['match_outcome'])
    prediction_data = _preprocess_data(prediction_df)
    print(f"Loaded {len(prediction_data)} matches for prediction")
    
    # Evaluate each model
    for uri in model_uris:
        try:
            uri_full = f"runs:/{uri}/soccer_draw_model_pycaret"
            predicted_df, precision, draws_recall = make_prediction(prediction_data, uri_full)
            
            if precision > best_precision and draws_recall > best_draws_recall:
                best_precision = precision
                best_model_uri = uri
                best_draws_recall = draws_recall
                best_predictions = predicted_df.copy()
                print(f"New best model: {uri} with precision: {precision:.2%}")
                print(f"Draws recall: {draws_recall:.2%}")
            else:
                print(f"No improvement for model {uri} with precision: {precision:.2%}")
                print(f"Draws recall: {draws_recall:.2%}")

        except Exception as e:
            print(f"Error evaluating model {uri}: {str(e)}")
            continue

    print(f"\nBest model URI: {best_model_uri}")
    print(f"Best precision: {best_precision:.2%}")
    # print(f"Draws recall: {best_draws_recall:.2%}")
    
    # Use best model for final predictions
    if best_model_uri:
        predicted_df = best_predictions
   
    # Save results
    output_path = Path("./data/prediction/api_predictions_pycaret.xlsx")
    predicted_df.to_excel(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")

if __name__ == "__main__":
    main()