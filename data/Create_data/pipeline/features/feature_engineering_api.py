import pandas as pd
import numpy as np
from typing import Dict

class FeatureEngineer:
    """Handle feature engineering for match outcome prediction"""
    
    def __init__(self, is_prediction: bool = False):
        self.is_prediction = is_prediction
        
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer all features including basic and draw-specific features."""
        try:
             # Calculate form momentum first
            df = self.calculate_form_momentum(df)
            
            # Add draw-specific features
            df = self.add_draw_features(df)
            
            # Add enhanced draw-specific features
            # df = FeatureEngineer.add_enhanced_draw_features(df)
            
            # Create interaction features
            df = self.create_interaction_features(df)
            
            # Create polynomial features
            df = self.create_polynomial_features(df)
            
            # Create composite features
            df = self.create_composite_features(df)
            
            # Then add draw-specific features
            df = self.engineer_draw_features(df)
            
            # Export the DataFrame to an Excel file
            output_file_path = "./data/feature_engineered_data.xlsx"
            try:
                df.to_excel(output_file_path, index=False)
                print(f"Feature engineered data successfully exported to {output_file_path}")
            except Exception as e:
                print(f"Error exporting feature engineered data to Excel: {str(e)}")
                raise
            
            return df
            
        except Exception as e:
            raise Exception(f"Error in engineer_all_features: {str(e)}")
        
    def engineer_draw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features specifically for draw prediction."""
        try:
            # Team Strength Equilibrium
            df['strength_equilibrium'] = 1 - abs(
                (df['home_attack_strength'] - df['away_attack_strength']) +
                (df['home_defense_weakness'] - df['away_defense_weakness'])
            )
            
            # Historical Draw Tendency
            df['historical_draw_tendency'] = (
                df['h2h_draw_rate'] * 0.4 +
                df['home_draw_rate'] * 0.3 +
                df['away_draw_rate'] * 0.3
            )
            
            # Form Convergence Score
            df['form_convergence_score'] = (
                (1 - abs(df['home_form_momentum'] - df['away_form_momentum'])) *
                df['form_similarity']
            )
            
            # Defensive Stability
            df['defensive_stability'] = (
                (df['Home_saves_mean'] + df['Away_saves_mean']) / 2 *
                (df['home_defense_weakness'] + df['away_defense_weakness']) / 2
            )
            
            # Position Equilibrium
            df['position_equilibrium'] = 1 - (
                abs(df['home_league_position'] - df['away_league_position']) / 20
            )
            
            # Goal Scoring Similarity
            df['goal_pattern_similarity'] = 1 - abs(
                df['home_goal_rollingaverage'] - df['away_goal_rollingaverage']
            )
            
            # xG Equilibrium
            df['xg_equilibrium'] = 1 - abs(
                df['home_xG_rolling_rollingaverage'] - 
                df['away_xG_rolling_rollingaverage']
            )
            
            # Possession Balance
            df['possession_balance'] = 1 - abs(
                df['Home_possession_mean'] - df['away_possession_mean']
            ) / 100
            
            # Draw-prone Period Detection
            df['mid_season_factor'] = 1 - abs(df['season_progress'] - 0.5) * 2
            
            # Create composite league draw tendency
            df['league_draw_rate_composite'] = (
                df['league_draw_rate'] * 0.3 +
                df['league_home_draw_rate'] * 0.2 +
                df['league_away_draw_rate'] * 0.2 +
                df['league_season_stage_draw_rate'] * 0.3
            )
            
            # Interaction Features
            df['form_position_interaction'] = df['form_convergence_score'] * df['position_equilibrium']
            df['strength_possession_interaction'] = df['strength_equilibrium'] * df['possession_balance']
            
            # Draw Probability Score (updated with new features)
            df['draw_probability_score'] = (
                df['historical_draw_tendency'] * 0.20 +
                df['form_convergence_score'] * 0.15 +
                df['strength_equilibrium'] * 0.15 +
                df['position_equilibrium'] * 0.15 +
                df['xg_equilibrium'] * 0.15 +
                df['league_draw_rate_composite'] * 0.20  # Increased weight for league factors
            )
            
            return df
            
        except Exception as e:
            raise Exception(f"Error in engineer_draw_features: {str(e)}")

    def add_league_specific_features(self,df: pd.DataFrame) -> pd.DataFrame:
        """Add league-specific draw features."""
        try:
            df = df.copy()
            
            # Calculate league-specific draw rates
            # league_draw_rates = df.groupby('league_encoded')['match_outcome'].apply(
            #     lambda x: (x == 1).mean()
            # ).to_dict()
            # df['league_draw_rate'] = df['league_encoded'].map(league_draw_rates)
            
            # League position impact (some leagues have stronger correlation)
            # df['league_position_impact'] = df.groupby('league_encoded').apply(
            #     lambda x: x['team_strength_diff'] * x['league_draw_rate']
            # )
            
            # League seasonality (some leagues have draw patterns in different parts of season)
            # df['season_progress'] = df.groupby(['league_encoded', 'season_encoded'])\
            #     .cumcount() / df.groupby(['league_encoded', 'season_encoded']).size()
            
            # League competitiveness score
            # df['league_competitiveness'] = df.groupby('league_encoded').apply(
            #     lambda x: 1 - x['team_strength_diff'].std() / x['team_strength_diff'].max()
            # )
            
            return df
        except Exception as e:
            raise Exception(f"Error in add_league_specific_features: {str(e)}")
   
    def add_enhanced_draw_features(self,df):
        """Add enhanced draw-specific features."""
        try:
            df = df.copy()
            
            # Fix form convergence calculation
            df['form_convergence'] = df.groupby('league_encoded').apply(
                lambda x: 1 - abs(x['home_form_momentum'].diff().rolling(5).mean() - 
                                x['away_form_momentum'].diff().rolling(5).mean())
            ).reset_index(level=0, drop=True)  # Reset the MultiIndex
            
            # League position volatility
            # df['position_volatility'] = df.groupby('league_encoded')['team_strength_diff']\
            #     .rolling(5).std().reset_index(level=0, drop=True).fillna(0)
            
            return df
        except Exception as e:
            raise Exception(f"Error in add_enhanced_draw_features: {str(e)}")
    
    def add_draw_features(self,df: pd.DataFrame) -> pd.DataFrame:
        """Add draw-specific features."""
        try:
            df = df.copy()
            
            # Team strength features (using only pre-match information)
            if all(col in df.columns for col in ['home_league_position', 'away_league_position']):
                # Convert to numeric
                df['home_league_position'] = pd.to_numeric(df['home_league_position'].astype(str).str.replace(',', '.'), errors='coerce')
                df['away_league_position'] = pd.to_numeric(df['away_league_position'].astype(str).str.replace(',', '.'), errors='coerce')
                
                df['team_strength_diff'] = abs(df['home_league_position'] - df['away_league_position'])
                df['avg_league_position'] = (df['home_league_position'] + df['away_league_position']) / 2
            
            # Form and momentum features (ensure these use only historical data)
            if all(col in df.columns for col in ['home_form_momentum', 'away_form_momentum']):
                # Convert to numeric
                df['home_form_momentum'] = pd.to_numeric(df['home_form_momentum'].astype(str).str.replace(',', '.'), errors='coerce')
                df['away_form_momentum'] = pd.to_numeric(df['away_form_momentum'].astype(str).str.replace(',', '.'), errors='coerce')
                
                df['form_difference'] = df['home_form_momentum'] - df['away_form_momentum']
                df['form_similarity'] = 1 / (1 + abs(df['form_difference'])).replace(np.inf, 1)
            
            # Historical matchup features (using only previous matches)
            if all(col in df.columns for col in ['h2h_matches', 'h2h_draws']):
                # Convert to numeric
                df['h2h_matches'] = pd.to_numeric(df['h2h_matches'].astype(str).str.replace(',', '.'), errors='coerce')
                df['h2h_draws'] = pd.to_numeric(df['h2h_draws'].astype(str).str.replace(',', '.'), errors='coerce')
                
                df['h2h_draw_rate'] = df['h2h_draws'] / df['h2h_matches'].replace(0, 1)
            
            # Team quality features (using only historical ratings)
            if all(col in df.columns for col in ['home_team_elo', 'away_team_elo']):
                # Convert to numeric
                df['home_team_elo'] = pd.to_numeric(df['home_team_elo'].astype(str).str.replace(',', '.'), errors='coerce')
                df['away_team_elo'] = pd.to_numeric(df['away_team_elo'].astype(str).str.replace(',', '.'), errors='coerce')
                
                df['elo_difference'] = abs(df['home_team_elo'] - df['away_team_elo'])
                df['elo_similarity'] = 1 / (1 + df['elo_difference'] / 100).replace(np.inf, 1)
            
            # Draw probability features (using only historical data)
            if all(col in df.columns for col in ['home_draw_rate', 'away_draw_rate']):
                # Convert to numeric
                df['home_draw_rate'] = pd.to_numeric(df['home_draw_rate'].astype(str).str.replace(',', '.'), errors='coerce')
                df['away_draw_rate'] = pd.to_numeric(df['away_draw_rate'].astype(str).str.replace(',', '.'), errors='coerce')
                
                df['combined_draw_rate'] = (df['home_draw_rate'] + df['away_draw_rate']) / 2
            
            # Fill NaN values with appropriate defaults
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(0)
            
            return df
            
        except Exception as e:
            raise Exception(f"Error in add_draw_features: {str(e)}")

    def create_interaction_features(self,df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between important predictors."""
        try:
            df = df.copy()
            
            # Identify key features for interactions
            form_features = [col for col in ['home_form_momentum', 'away_form_momentum'] if col in df.columns]
            strength_features = [col for col in ['home_attack_strength', 'away_attack_strength'] if col in df.columns]
            position_features = [col for col in ['home_league_position', 'away_league_position'] if col in df.columns]
            
            # Create interactions only if both features exist
            for feat1 in form_features:
                for feat2 in strength_features:
                    if feat1 in df.columns and feat2 in df.columns:
                        df[f'{feat1}_{feat2}_interaction'] = df[feat1] * df[feat2]
                    
            for feat1 in strength_features:
                for feat2 in position_features:
                    if feat1 in df.columns and feat2 in df.columns:
                        df[f'{feat1}_{feat2}_interaction'] = df[feat1] * df[feat2]
            
            return df
            
        except Exception as e:
            raise Exception(f"Error in create_interaction_features: {str(e)}")

    def create_polynomial_features(self,df: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
        """Create polynomial features for key predictors."""
        try:
            df = df.copy()
            
            # Select features for polynomial expansion
            key_features = [
                'draw_propensity_score',
                'elo_similarity',
                'xg_similarity',
                'form_similarity'
            ]
            
            # Create polynomial features only for existing columns
            existing_features = [f for f in key_features if f in df.columns]
            for feature in existing_features:
                for d in range(2, degree + 1):
                    df[f'{feature}_power_{d}'] = df[feature].pow(d)
            
            return df
            
        except Exception as e:
            raise Exception(f"Error in create_polynomial_features: {str(e)}")

    def calculate_form_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate form momentum for home and away teams based on recent performance.
        
        Form momentum considers recent match outcomes, weighted by recency and opponent strength.
        """
        try:
            df = df.copy()
            
            # Initialize momentum columns
            df['home_form_momentum'] = 0.0
            df['away_form_momentum'] = 0.0
            
            # Calculate base points for outcomes (win=3, draw=1, loss=0)
            if 'home_win_rate' in df.columns and 'away_win_rate' in df.columns:
                try:
                    df['home_form_momentum'] = (
                        df['home_win_rate'].astype(float) * 3 +  # Points from wins
                        df['home_draw_rate'].astype(float) * 1    # Points from draws
                    )
                    
                    df['away_form_momentum'] = (
                        df['away_win_rate'].astype(float) * 3 +  # Points from wins
                        df['away_draw_rate'].astype(float) * 1    # Points from draws
                    )
                except ValueError as ve:
                    raise ValueError(f"Error in calculate_form_momentum: {str(ve)}")
                
            # Adjust momentum based on league position (if available)
            if 'home_league_position' in df.columns and 'away_league_position' in df.columns:
                max_position = df[['home_league_position', 'away_league_position']].max().max()
                
                # Position adjustment factor (better position = higher momentum)
                df['home_form_momentum'] *= (1 + (max_position - df['home_league_position']) / max_position)
                df['away_form_momentum'] *= (1 + (max_position - df['away_league_position']) / max_position)
            
            # Adjust momentum based on goal difference (if available)
            if 'Home_goal_difference_cum' in df.columns and 'Away_goal_difference_cum' in df.columns:
                # Normalize goal differences to a small factor
                gd_factor = 0.1  # Small weight for goal difference
                df['home_form_momentum'] *= (1 + gd_factor * df['Home_goal_difference_cum'].clip(-10, 10))
                df['away_form_momentum'] *= (1 + gd_factor * df['Away_goal_difference_cum'].clip(-10, 10))
            
            # Scale momentums to a reasonable range (0-1)
            max_momentum = max(df['home_form_momentum'].max(), df['away_form_momentum'].max())
            if max_momentum > 0:
                df['home_form_momentum'] /= max_momentum
                df['away_form_momentum'] /= max_momentum
            
            return df
            
        except Exception as e:
            raise Exception(f"Error in calculate_form_momentum: {str(e)}")
  
    def create_composite_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite features combining poisson_xG and form metrics."""
        try:
            df = df.copy()
            
            # Poisson xG and Form Interactions
            df['xg_form_equilibrium'] = 1 - abs(
                (df['home_poisson_xG'] * df['home_form_momentum']) -
                (df['away_poisson_xG'] * df['away_form_momentum'])
            )
            
            # Expected Goals Momentum
            df['home_xg_momentum'] = df['home_poisson_xG'] * (1 + df['home_form_momentum'])
            df['away_xg_momentum'] = df['away_poisson_xG'] * (1 + df['away_form_momentum'])
            df['xg_momentum_similarity'] = 1 - abs(df['home_xg_momentum'] - df['away_xg_momentum'])
            
            # Form-Weighted Expected Goals
            try:
                df['home_form_weighted_xg'] = df['home_poisson_xG'].astype(float) * df['home_win_rate'].astype(float)
                df['away_form_weighted_xg'] = df['away_poisson_xG'].astype(float) * df['away_win_rate'].astype(float)
            except ValueError as ve:
                raise ValueError(f"Error in create_composite_features while calculating form-weighted xG: {str(ve)}")
            df['form_weighted_xg_diff'] = abs(df['home_form_weighted_xg'] - df['away_form_weighted_xg'])
            
            # Attack Strength and xG Combination
            df['home_attack_xg_power'] = df['home_attack_strength'] * df['home_poisson_xG']
            df['away_attack_xg_power'] = df['away_attack_strength'] * df['away_poisson_xG']
            df['attack_xg_equilibrium'] = 1 - abs(df['home_attack_xg_power'] - df['away_attack_xg_power'])
            
            # Rolling xG Form
            try:
                df['home_xg_form'] = df['home_xG_rolling_rollingaverage'].astype(float) * df['home_form_momentum'].astype(float)
                df['away_xg_form'] = df['away_xG_rolling_rollingaverage'].astype(float) * df['away_form_momentum'].astype(float)
            except ValueError as ve:
                raise ValueError(f"Error in create_composite_features while calculating rolling xG form: {str(ve)}")
            df['xg_form_similarity'] = 1 - abs(df['home_xg_form'] - df['away_xg_form'])
            
            # Create draw probability enhancers
            df['draw_xg_indicator'] = (
                (df['xg_form_equilibrium'] * 0.3) +
                (df['xg_momentum_similarity'] * 0.3) +
                (1 - df['form_weighted_xg_diff']) * 0.2 +
                (df['attack_xg_equilibrium'] * 0.2)
            )
            
            # Update draw probability score with new features
            if 'draw_probability_score' in df.columns:
                df['draw_probability_score'] = (
                    df['draw_probability_score'] * 0.7 +  # Original score
                    df['draw_xg_indicator'] * 0.3    # New xG-based indicators
                )
            
            return df
            
        except Exception as e:
            raise Exception(f"Error in create_composite_features: {str(e)}")
