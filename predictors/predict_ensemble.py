import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
from pathlib import Path
import json
import os
import sys
from typing import Dict, Any, List

import pymongo
from icecream import ic
from pymongo import MongoClient

# Add project root to Python path
try:
    project_root = Path(__file__).parent.parent
    if not project_root.exists():
        # Handle network path by using raw string
        project_root = Path(r"\\".join(str(project_root).split("\\")))
    sys.path.append(str(project_root))
    print(f"Project root: {project_root}")
except Exception as e:
    print(f"Error setting project root path: {e}")
    # Fallback to current directory if path resolution fails
    sys.path.append(os.getcwd().parent)
    print(f"Current directory: {os.getcwd().parent}")

from utils.create_evaluation_set import get_selected_columns_draws, setup_mlflow_tracking, create_evaluation_sets_draws, get_real_scores_from_excel

selected_columns = get_selected_columns_draws()

class DrawPredictor:
    """Predictor class for draw predictions using the stacked model."""
    
    def __init__(self, run_id: str):
        """Initialize predictor with run ID."""
        self.run_id = run_id
        model_path = Path(project_root) / "mlruns" / "2" / run_id / "artifacts"
        
        try:
            # Load models using xgboost flavor instead of pyfunc
            model_uri = f"file://{model_path.as_posix()}"
            two_stage_path = f"{model_uri}/stage2_model"
            # ic(f"Loading two-stage model from: {two_stage_path}")
            self.two_stage_model = mlflow.xgboost.load_model(two_stage_path)
            
            # Load voting models
            self.voting_models = []
            for i in range(5):
                voting_path = f"{model_uri}/voting_model_{i}"
                ic(f"Loading voting model {i}")
                model = mlflow.xgboost.load_model(voting_path)
                self.voting_models.append(model)
            
            # Load model info for thresholds - use direct file path for json
            model_info_path = model_path / "model_info.json"
            # ic(f"Loading model info from: {model_info_path}")
            with open(model_info_path) as f:
                model_info = json.load(f)
                
            self.two_stage_threshold1 = model_info['two_stage_params']['threshold1']
            self.two_stage_threshold2 = model_info['two_stage_params']['threshold2']
            self.voting_thresholds = model_info['voting_params']['thresholds']
            self.confidence_threshold = 0.65
            
        except Exception as e:
            ic(f"Error loading models: {str(e)}")
            raise
        
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions using both models."""
        # Get probabilities from two-stage model
        two_stage_probs = self.two_stage_model.predict_proba(df)[:, 1]  # Get probability of positive class
        
        # Get voting ensemble probabilities
        voting_probs = np.zeros((len(df), len(self.voting_models)))
        for i, model in enumerate(self.voting_models):
            probs = model.predict_proba(df)[:, 1]  # Get probability of positive class
            voting_probs[:, i] = probs
        
        # Average voting probabilities
        voting_mean_prob = np.mean(voting_probs, axis=1)
        
        # Combined prediction with confidence
        predictions = ((two_stage_probs >= self.two_stage_threshold2) & 
                      (voting_mean_prob >= self.confidence_threshold)).astype(int)
        
        result = {
            'predictions': predictions,
            'draw_probabilities': two_stage_probs,
            'voting_probabilities': voting_mean_prob,
            'confidence_score': np.minimum(two_stage_probs, voting_mean_prob),
            'threshold_used': self.two_stage_threshold2,
            'num_predictions': len(predictions),
            'positive_predictions': int(predictions.sum()),
            'prediction_rate': float(predictions.mean())
        }
        
        return result
    
    def predict_single_match(self, row: pd.Series) -> Dict[str, Any]:
        """Make prediction for a single match."""
        ic("Starting single match prediction")
        df = pd.DataFrame([row])
        
        # Get predictions from both models
        two_stage_prob = self.two_stage_model.predict(df)[0]  # Use predict()
        
        # Get voting predictions
        voting_probs = np.zeros(len(self.voting_models))
        for i, model in enumerate(self.voting_models):
            prob = model.predict(df)[0]  # Use predict()
            voting_probs[i] = (prob >= self.voting_thresholds[i]).astype(int)
        
        # Majority voting
        voting_pred = (voting_probs.sum() >= 3).astype(int)
        
        # Combined prediction
        two_stage_pred = (two_stage_prob >= self.two_stage_threshold2).astype(int)
        final_pred = (two_stage_pred & voting_pred).astype(int)
        
        result = {
            'is_draw': bool(final_pred),
            'draw_probability': float(two_stage_prob),
            'confidence': float(abs(two_stage_prob - 0.5) * 2)
        }
        return result

def _preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data to handle number formats and missing values."""
    ic("Starting preprocessing", df.shape)
    # ic("Initial columns:", df.columns)
    
    # Create a copy to avoid modifying original data
    df = df.copy()
    
    # Handle date encoding if needed
    if 'Datum' in df.columns:
        ic("Processing date encoding")
        earliest_date = pd.to_datetime("2020-08-11")
        df['date_encoded'] = (pd.to_datetime(df['Datum']) - earliest_date).dt.days
        df['date_encoded'] = df['date_encoded'].astype(int)
        ic("Date encoding complete")
    
    # Store non-numeric columns we want to keep
    keep_columns = ['running_id', 'Datum', 'Home', 'Away', 'league']
    stored_columns = {col: df[col].copy() for col in keep_columns if col in df.columns}
    
    # Replace comma with dot for numeric columns
    for col in df.columns:
        if col not in keep_columns and df[col].dtype == 'object':
            try:     
                # ic(f"Converting column {col}")
                # Replace comma with dot and convert to float
                df[col] = (df[col].astype(str)
                          .str.strip()
                          .str.strip("'\"")
                          .str.replace(' ', '')
                          .str.replace(',', '.')
                          .astype(float))
                # ic(f"Successfully converted {col}")
                
            except (AttributeError, ValueError) as e:
                ic(f"Failed to convert column {col}: {str(e)}")
                continue
    
    # Restore kept columns
    for col, values in stored_columns.items():
        df[col] = values
    
    # ic("Final preprocessed shape:", df.shape)
    # ic("Final columns:", df.columns)
    return df

def make_prediction(prediction_data, run_id: str) -> pd.DataFrame:
    """Make predictions and return results with probabilities."""
    try:
        # Initialize predictor
        model_path = Path(project_root) / "mlruns" / "2" / run_id / "artifacts"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        ic(f"Loading model from: {model_path}")
        predictor = DrawPredictor(run_id)
        
        # Keep a copy of running_id before prediction
        running_ids = prediction_data['running_id'].copy() if 'running_id' in prediction_data.columns else None
        
        # Make predictions
        results = predictor.predict(prediction_data)
        
        precision = 0.0
        
        # Get real scores from Excel only if we have running_ids
        if running_ids is not None:
            # ic("Getting real scores for validation")
            real_scores = get_real_scores_from_excel(running_ids.tolist())
            # ic(f"Retrieved {len(real_scores)} real scores")
            
            if real_scores:  # Only proceed if we got some real scores
                # Add real results to output
                real_scores_df = pd.DataFrame.from_dict(real_scores, orient='index')
                real_scores_df.reset_index(inplace=True)
                real_scores_df.rename(columns={'index': 'running_id'}, inplace=True)
                
                # Merge prediction_df with real_scores_df on running_id
                prediction_df = prediction_df.merge(real_scores_df, on='running_id', how='left')
                
                # Calculate accuracy metrics
                matches_with_results = prediction_df[prediction_df['is_draw'].notna()]
                # ic(f"Matches with known results: {len(matches_with_results)}")
                
                if len(matches_with_results) > 0:
                    true_positives = ((matches_with_results['draw_predicted'] == 1) & 
                                    (matches_with_results['is_draw'] == True)).sum()
                    false_positives = ((matches_with_results['draw_predicted'] == 1) & 
                                     (matches_with_results['is_draw'] == False)).sum()
                    
                    ic(f"True positives: {true_positives}")
                    ic(f"False positives: {false_positives}")
                    
                    if true_positives + false_positives > 0:
                        precision = true_positives / (true_positives + false_positives)
                        ic(f"Draw prediction precision: {precision:.2%}")
        else:
            ic("No running_ids provided, skipping real scores")
        return prediction_df, precision
        
    except Exception as e:
        ic(f"Error during prediction: {str(e)}")
        raise

def main():
    # Set up paths
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data/prediction/prediction_data.csv"
    experiment_name = "xgboost_ensemble_draw_model"
    setup_mlflow_tracking(experiment_name=experiment_name)
    
    # Model run IDs to evaluate
    run_ids = [
        "ce79e624e452469cb9dbc697a100f1c5",
        "8e369a543fcd437e8e89c521ec3ef91d",
        "9be0e3be588e44b79cb9728cead72789",
        "8722820d134d41548b26f71b3424bcb1",
        "62ec3a34222a465d92d2e33178a7ca84"
    ]

    best_precision = 0
    best_run_id = None
    best_predictions = None
    
    # Load and preprocess prediction data
    prediction_df = pd.read_csv(data_path, decimal=',')
    prediction_data = _preprocess_data(prediction_df)
    
    # Keep a copy of running_id before prediction
    running_ids = prediction_data['running_id'].copy() if 'running_id' in prediction_data.columns else None
    
    # Get selected columns for model input
    model_columns = get_selected_columns_draws()
    prediction_data_model = prediction_data[model_columns]
    print(f"Loaded {len(prediction_data)} matches for prediction")
    
    # Evaluate each model
    for run_id in run_ids:
        try:
            # Reset prediction_df for each iteration to avoid data corruption
            prediction_df = pd.read_csv(data_path, decimal=',')
            prediction_data = _preprocess_data(prediction_df)
            
            # Keep a copy of running_id before prediction
            running_ids = prediction_data['running_id'].copy() if 'running_id' in prediction_data.columns else None
            
            # Get selected columns for model input
            model_columns = get_selected_columns_draws()
            prediction_data_model = prediction_data[model_columns]
            
            # predicted_df, precision = make_prediction(prediction_data_model, run_id)
            # Use MLruns directory structure
            model_path = Path(f"{project_root}/mlruns/2/{run_id}/artifacts")
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found at {model_path}")
                
            # ic(f"Loading model from: {model_path}")
            predictor = DrawPredictor(run_id)
            
            # Make predictions
            results = predictor.predict(prediction_data_model)
            prediction_df['draw_predicted'] = results['predictions']
            prediction_df['draw_probability'] = [round(prob, 2) for prob in results['draw_probabilities']]
            precision = 0.0
            
            # Get real scores from Excel only if we have running_ids
            if running_ids is not None:
                # ic("Getting real scores for validation")
                real_scores = get_real_scores_from_excel(running_ids.tolist())
                ic(f"Retrieved {len(real_scores)} real scores")
                real_scores_df = pd.DataFrame.from_dict(real_scores, orient='index')
                # ic(f"Real scores DataFrame columns: {real_scores_df.columns}")
                real_scores_df.reset_index(inplace=True)
                real_scores_df.rename(columns={'index': 'running_id'}, inplace=True)
                real_scores_df['running_id'] = real_scores_df['running_id'].astype(int)
                ic(f"Running ID type in real_scores_df: {real_scores_df['running_id'].dtype}")
                ic(f"Running ID type in prediction_df: {prediction_df['running_id'].dtype}")
                
                if real_scores:  # Only proceed if we got some real scores
                    # Merge prediction_df with real_scores_df on running_id
                    prediction_df = prediction_df.merge(real_scores_df, on='running_id', how='left')
                    
                    # Calculate accuracy metrics
                    matches_with_results = prediction_df[prediction_df['is_draw'].notna()]
                    if len(matches_with_results) > 0:
                        accuracy = (matches_with_results['draw_predicted'] == 
                                    matches_with_results['is_draw']).mean()
                        draws_recall = ((matches_with_results['draw_predicted'] == 1) & 
                                        (matches_with_results['is_draw'] == True)).sum() / (matches_with_results['is_draw'] == True).sum()
                        print(f"\nPrediction Accuracy: {run_id}")
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
                        else:
                            print("No true positives or false positives found")
            else:
                ic("No running_ids provided, skipping real scores")
                
            
            if precision > best_precision:
                best_precision = precision
                best_run_id = run_id
                best_predictions = prediction_df.copy()
                print(f"New best model: {run_id} with precision: {precision:.2%}")

        except Exception as e:
            print(f"Error evaluating model {run_id}: {str(e)}")
            continue

    print(f"\nBest model run ID: {best_run_id}")
    print(f"Best precision: {best_precision:.2%}")
    
    # Use best model for final predictions
    if best_predictions is not None:
        # Save results
        output_path = project_root / "data/prediction/predictions_ensemble.xlsx"
        best_predictions.to_excel(output_path, index=False)
        print(f"\nPredictions saved to: {output_path}")

if __name__ == "__main__":
    main()