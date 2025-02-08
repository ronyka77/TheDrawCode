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
import mlflow.pyfunc
from pymongo import MongoClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.create_evaluation_set import get_real_api_scores_from_excel, setup_mlflow_tracking, create_prediction_set_ensemble

mlruns_dir = setup_mlflow_tracking("xgboost_draw_model")

class DrawPredictor:

    """Predictor class for draw predictions using the stacked model."""
    
    def __init__(self, model_uri: str):
        """Initialize predictor with model URI."""
        # Set up MLflow tracking URI based on current environment
        current_dir = os.getcwd()
        self.model = mlflow.xgboost.load_model(model_uri)
        try:
            self.test_model = mlflow.pyfunc.load_model(model_uri)
            # Get feature names from signature
            if self.test_model.metadata.signature:
                self.required_features = self.test_model.metadata.signature.inputs.input_names()
                print(f"Features from signature: {self.required_features}")
        except Exception as e:
            print(f"Error loading model: {e}")

        # Define expected input columns based on model requirements
        # self.expected_columns = [
        #     "position_equilibrium", "venue_match_count", "league_draw_rate", "possession_balance",
        #     "xg_form_similarity", "away_poisson_xG", "home_position_form", "away_team_elo",
        #     "venue_draws", "form_difference", "away_attack_strength_home_league_position_interaction",
        #     "home_poisson_xG", "away_set_piece_threat", "away_passing_efficiency", "away_h2h_weighted",
        #     "venue_draw_rate", "home_corners_mean", "home_draw_rate", "home_passing_efficiency",
        #     "Away_offsides_mean", "home_attack_xg_power", "away_average_points", "away_style_compatibility",
        #     "home_style_compatibility", "Home_shot_on_target_mean", "league_draw_rate_composite",
        #     "home_season_form", "home_xg_form", "Home_team_matches", "away_referee_impact",
        #     "strength_possession_interaction", "season_progress", "away_form_stability",
        #     "away_form_momentum_away_attack_strength_interaction", "Home_passes_mean", "home_goal_momentum",
        #     "home_h2h_wins", "away_scoring_efficiency", "goal_pattern_similarity", "home_form_weighted_xg",
        #     "Home_offsides_mean", "combined_draw_rate", "avg_league_position", "away_historical_strength",
        #     "home_yellow_cards_rollingaverage", "seasonal_draw_pattern", "draw_xg_indicator",
        #     "away_offensive_sustainability", "away_form_weighted_xg", "home_ref_interaction",
        #     "away_attack_conversion", "home_defense_weakness", "fixture_id", "league_home_draw_rate",
        #     "Home_possession_mean", "home_team_elo", "away_shots_on_target_accuracy_rollingaverage",
        #     "away_corners_mean", "home_yellow_cards_mean", "away_days_since_last_draw", "h2h_avg_goals",
        #     "home_goal_rollingaverage", "form_position_interaction", "away_fouls_rollingaverage",
        #     "draw_propensity_score", "Home_draws", "referee_goals_per_game", "away_attack_xg_power",
        #     "away_defensive_organization", "elo_similarity_form_similarity", "home_corners_rollingaverage",
        #     "away_crowd_resistance", "away_xg_momentum", "Away_saves_mean", "Home_goal_difference_cum",
        #     "xg_equilibrium", "home_attack_strength_home_league_position_interaction", "defensive_stability",
        #     "away_ref_interaction", "home_defense_index", "away_defense_index", "strength_equilibrium",
        #     "home_weighted_attack", "away_corners_rollingaverage", "away_shot_on_target_mean",
        #     "away_attack_strength", "away_saves_rollingaverage", "form_weighted_xg_diff", "away_encoded",
        #     "home_offensive_sustainability", "draw_probability_score", "venue_capacity", "Home_fouls_mean",
        #     "home_xG_rolling_rollingaverage", "away_possession_mean", "away_shot_on_target_rollingaverage",
        #     "away_draw_rate", "away_yellow_cards_mean", "home_xg_momentum"
        # ]
        # Load the threshold from the model if available
        self.threshold = 0.50
    
    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate input dataframe has all required columns."""
        missing_cols = set(self.required_features) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions and return results with probabilities."""
        
        # self._validate_input(predict_df)
        # Get probabilities
        probas = self.model.predict_proba(df)
        predictions = self.model.predict(df)
   
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

# def _preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
#     """Preprocess data to handle number formats and missing values."""
#     # Create a copy to avoid modifying original data
#     df = df.copy()
    
#     # Replace comma with dot for ALL numeric-like columns
#     for col in df.columns:
#         try:
#             if col in df.columns:
#                 df[col] = (
#                     df[col]
#                     .apply(lambda x: str(x) if pd.notnull(x) else '0')
#                     .str.strip()
#                     .str.replace('[^0-9.eE-]', '', regex=True)
#                     .apply(lambda x: '0' if x in ['e', 'e-', 'e+'] else x)
#                     .apply(lambda x: '1' + x if x.lower().startswith(('e', 'e-', 'e+')) else x)
#                     .apply(lambda x: x.replace('-', 'e-', 1) if '-' in x and 'e' not in x.lower() else x)
#                     .replace('', '0')
#                     .pipe(pd.to_numeric, errors='coerce')
#                     .replace([np.inf, -np.inf], np.nan)
#                     .fillna(0)
#                 )
#         except Exception as e:
#             print(f"Error converting column {col}: {str(e)}")
#             continue
    
#     return df

def make_prediction(prediction_data, model_uri, real_scores_df) -> pd.DataFrame:
    """Make predictions and return results with probabilities."""
    try:
        # Initialize default values
        precision = 0.0
        draws_recall = 0.0
        
        # Initialize predictor
        predictor = DrawPredictor(model_uri)
        prediction_df = prediction_data.copy()
        print(f"Prediction data len(prediction_df): {len(prediction_df)}")

         # Add column validation
        predictor._validate_input(prediction_df)
        
        # Add dtype consistency check with proper DataFrame handling
        if not isinstance(prediction_data, pd.DataFrame):
            raise TypeError("prediction_data must be a pandas DataFrame")
        
        prediction_df = prediction_data[predictor.required_features]
        # Make predictions first
        results = predictor.predict(prediction_df)
        print(f"Prediction successful...")

        # Add predictions to dataframe
        prediction_df['draw_predicted'] = results['predictions']
        prediction_df['draw_probability'] = [round(prob, 2) for prob in results['draw_probabilities']]
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
                    prediction_data, 
                    on='fixture_id', 
                    how='left'
                )

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
        'd0adbbe461e54cc2b981f7964cb68d96',
        '984ff71c0c704517a371048da7044202',
        '2ac6b3a56102419b80e9ee0c6b598b2d'
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
    
    # # Get selected columns using standardized function
    # selected_columns = import_selected_features_ensemble('all')
    # print(f"Number of selected columns: {len(selected_columns)}")
    
    # Evaluate each model
    for uri in model_uris:
        try:
            uri_full = f"runs:/{uri}/xgboost_api_model"
            predicted_df, precision, draws_recall = make_prediction(prediction_data, uri_full, real_scores_df)

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