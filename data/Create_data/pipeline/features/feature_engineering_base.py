import pandas as pd
import numpy as np
import logging

class FeatureEngineer_base:
    def __init__(self,
                 logger: logging.Logger,
                 target_variable: str,
                 is_prediction: bool = False):
        """Initialize feature engineer with thresholds and logger."""
        self.logger = logger
       
        self.target_variable = target_variable
        self.is_prediction = is_prediction
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer all features including basic and draw-specific features."""
        try:
            # Ensure league_encoded is integer type first
            if 'league_encoded' in df.columns:
                df['league_encoded'] = df['league_encoded'].astype(int)
            if 'date_encoded'  in df.columns:
                print('date_encoded exists in Dataframe')
                df['date_encoded'] = df['date_encoded'].astype(int)
            else:
                print('date_encoded does not exist in Dataframe')
                df['date_encoded'] = pd.to_datetime(df['datum']).astype(int)
            
            # Calculate form momentum first
            df = self.calculate_form_momentum(df)
            
            # Add draw-specific features
            df = self.add_draw_features(df)

            # Add enhanced draw-specific features
            df = self.add_enhanced_draw_features(df)
            
            # Create interaction features
            df = self.create_interaction_features(df)

            # Create polynomial features
            df = self.create_polynomial_features(df)
            
            # Add league-specific features
            df = self.add_league_specific_features(df)          
            
            # Create composite features combining xG and form
            df = self.create_composite_features(df)
                
            # Then add draw-specific features
            df = self.engineer_draw_features(df)
            
            df = self.add_advanced_features(df)
            # if self.is_prediction:
            #     # Export the DataFrame to an Excel file
            #     output_file_path = "./data/prediction/feature_engineered_data_prediction.xlsx"
            #     try:
            #         df.to_excel(output_file_path, index=False)
            #         print(f"Feature engineered data successfully exported to {output_file_path}")
            #     except Exception as e:
            #         print(f"Error exporting feature engineered data to Excel: {str(e)}")
            #         raise
            
            return df
            
        except Exception as e:
            raise Exception(f"Error in engineer_all_features: {str(e)}")
    
    def engineer_draw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features specifically for draw prediction."""
        try:
            df = df.copy()
            
            # Create team encodings if they don't exist
            if 'home_encoded' not in df.columns:
                home_teams = pd.Categorical(df['home_team'])
                df['home_encoded'] = home_teams.codes
                
            if 'away_encoded' not in df.columns:
                away_teams = pd.Categorical(df['away_team'])
                df['away_encoded'] = away_teams.codes
                
            # Team Strength Equilibrium
            if all(col in df.columns for col in ['home_attack_strength', 'away_attack_strength', 'home_defense_weakness', 'away_defense_weakness']):
                df['strength_equilibrium'] = 1 - abs(
                    (df['home_attack_strength'] - df['away_attack_strength']) +
                    (df['home_defense_weakness'] - df['away_defense_weakness'])
                )
            
            # Form Stability - New Feature
            if all(col in df.columns for col in ['league_encoded', 'home_form_momentum', 'away_form_momentum']):
                df['form_stability'] = 1 - df.groupby('league_encoded')[['home_form_momentum', 'away_form_momentum']].transform(
                    lambda x: x.std()
                ).mean(axis=1)
            
            # Weighted H2H Draw Rate - New Feature
            if all(col in df.columns for col in ['h2h_draw_rate', 'h2h_matches', 'league_draw_rate']):
                df['weighted_h2h_draw_rate'] = df.apply(
                    lambda row: (
                        row['h2h_draw_rate'] * 1.2 if row['h2h_matches'] >= 5
                        else row['h2h_draw_rate'] * 0.9 if row['h2h_matches'] >= 2
                        else row['league_draw_rate']
                    ),
                    axis=1
                )
            
            # Seasonal Draw Pattern - New Feature
            if all(col in df.columns for col in ['season_progress', 'league_draw_rate']):
                df['seasonal_draw_pattern'] = np.sin(2 * np.pi * df['season_progress']) * df['league_draw_rate']
            
            # Historical Draw Tendency - Updated weights
            if all(col in df.columns for col in ['weighted_h2h_draw_rate', 'home_draw_rate', 'away_draw_rate']):
                df['historical_draw_tendency'] = (
                    df['weighted_h2h_draw_rate'] * 0.4 +
                    df['home_draw_rate'] * 0.3 +
                    df['away_draw_rate'] * 0.3
                )
            
            # Form Convergence Score
            if all(col in df.columns for col in ['home_form_momentum', 'away_form_momentum', 'form_similarity']):
                df['form_convergence_score'] = (
                    (1 - abs(df['home_form_momentum'] - df['away_form_momentum'])) *
                    df['form_similarity']
                )
            
            # Defensive Stability
            if all(col in df.columns for col in ['Home_saves_mean', 'Away_saves_mean', 'home_defense_weakness', 'away_defense_weakness']):
                df['defensive_stability'] = (
                    (df['Home_saves_mean'] + df['Away_saves_mean']) / 2 *
                    (df['home_defense_weakness'] + df['away_defense_weakness']) / 2
                )
            
            # Position Equilibrium
            if all(col in df.columns for col in ['home_league_position', 'away_league_position']):
                df['position_equilibrium'] = 1 - (
                    abs(df['home_league_position'] - df['away_league_position']) / 20
                )
            
            # Goal Scoring Similarity
            if all(col in df.columns for col in ['home_goal_rollingaverage', 'away_goal_rollingaverage']):
                df['goal_pattern_similarity'] = 1 - abs(
                    df['home_goal_rollingaverage'] - df['away_goal_rollingaverage']
                )
            
            # xG Equilibrium
            if all(col in df.columns for col in ['home_xG_rolling_rollingaverage', 'away_xG_rolling_rollingaverage']):
                df['xg_equilibrium'] = 1 - abs(
                    df['home_xG_rolling_rollingaverage'] -
                    df['away_xG_rolling_rollingaverage']
                )
            
            # Possession Balance
            if all(col in df.columns for col in ['Home_possession_mean', 'away_possession_mean']):
                df['possession_balance'] = 1 - abs(
                    df['Home_possession_mean'] - df['away_possession_mean']
                ) / 100
            
            # Draw-prone Period Detection
            if 'season_progress' in df.columns:
                df['mid_season_factor'] = 1 - abs(df['season_progress'] - 0.5) * 2
            
            # Home-specific draw rates for each league
            if all(col in df.columns for col in ['league_encoded', 'home_encoded', 'match_outcome']):
                league_home_draw_rates = df.groupby(['league_encoded', 'home_encoded'])['match_outcome'].apply(
                    lambda x: (x == 1).mean()
                ).reset_index()
                league_home_draw_rates.columns = ['league_encoded', 'home_encoded', 'league_home_draw_rate']
                df = df.merge(league_home_draw_rates, on=['league_encoded', 'home_encoded'], how='left')
            
            # Away-specific draw rates for each league
            if all(col in df.columns for col in ['league_encoded', 'away_encoded', 'match_outcome']):
                league_away_draw_rates = (
                    df.groupby(['league_encoded', 'away_encoded'])['match_outcome']
                    .mean()
                    .reset_index()
                    .rename(columns={'match_outcome': 'league_away_draw_rate'})
                )
                df = df.merge(league_away_draw_rates, on=['league_encoded', 'away_encoded'], how='left')
            
            # Season stage draw rates
            if all(col in df.columns for col in ['season_encoded', 'date_encoded', 'league_encoded']):
                df['season_stage'] = df.groupby('season_encoded')['date_encoded'].transform(
                    lambda x: pd.qcut(x.rank(method='first'), q=3, labels=['1', '2', '3'])
                )
                league_season_stage_rates = (
                    df.groupby(['league_encoded', 'season_stage'], observed=False)['match_outcome']
                    .mean()
                    .reset_index()
                    .rename(columns={'match_outcome': 'league_season_stage_draw_rate'})
                )
                df = df.merge(league_season_stage_rates, on=['league_encoded', 'season_stage'], how='left')
                
                # Drop temporary columns
                df.drop(columns=['season_stage'], inplace=True, errors='ignore')
            
            # Drop other temporary columns if they exist
            for col in ['max_matches']:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)
            
            # Create composite league draw tendency with updated weights
            required_cols = ['league_draw_rate', 'league_home_draw_rate', 'league_away_draw_rate', 'league_season_stage_draw_rate']
            if all(col in df.columns for col in required_cols):
                df['league_draw_rate_composite'] = (
                    df['league_draw_rate'] * 0.3 +
                    df['league_home_draw_rate'] * 0.2 +
                    df['league_away_draw_rate'] * 0.2 +
                    df['league_season_stage_draw_rate'] * 0.3
                )
            
            # Interaction Features
            if all(col in df.columns for col in ['form_convergence_score', 'position_equilibrium']):
                df['form_position_interaction'] = df['form_convergence_score'] * df['position_equilibrium']
                
            if all(col in df.columns for col in ['strength_equilibrium', 'possession_balance']):
                df['strength_possession_interaction'] = df['strength_equilibrium'] * df['possession_balance']
            
            # Updated Draw Probability Score with new features and weights
            required_score_cols = [
                'historical_draw_tendency', 'form_convergence_score', 'strength_equilibrium',
                'position_equilibrium', 'xg_equilibrium', 'league_draw_rate_composite',
                'form_stability', 'seasonal_draw_pattern'
            ]
            if all(col in df.columns for col in required_score_cols):
                df['draw_probability_score'] = (
                    df['historical_draw_tendency'] * 0.15 +
                    df['form_convergence_score'] * 0.15 +
                    df['strength_equilibrium'] * 0.15 +
                    df['position_equilibrium'] * 0.10 +
                    df['xg_equilibrium'] * 0.15 +
                    df['league_draw_rate_composite'] * 0.15 +
                    df['form_stability'] * 0.10 +
                    df['seasonal_draw_pattern'] * 0.05
                )
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error in engineer_draw_features: {str(e)}")  # Log the error
            return df  # Return original dataframe on error
    
    def add_league_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add league-specific draw features."""
        try:
            df = df.copy()
            
            # Calculate league-specific draw rates
            if 'league_encoded' in df.columns and 'match_outcome' in df.columns:
                league_draw_rates = df.groupby('league_encoded')['match_outcome'].apply(
                    lambda x: (x == 1).mean()
                ).to_dict()
                df['league_draw_rate'] = df['league_encoded'].map(league_draw_rates)
            
            # Calculate season progress if possible
            if all(col in df.columns for col in ['league_encoded', 'season_encoded']):
                df['temp_key'] = df['league_encoded'].astype(str) + '_' + df['season_encoded'].astype(str)
                df['season_match_count'] = df.groupby('temp_key').cumcount() + 1
                df['max_matches'] = df.groupby('temp_key')['season_match_count'].transform('max')
                df['season_progress'] = df['season_match_count'] / df['max_matches']
                df.drop(['temp_key', 'season_match_count', 'max_matches'], axis=1, inplace=True)
            
            # Calculate league position impact if possible
            if all(col in df.columns for col in ['team_strength_diff', 'league_draw_rate']):
                df['league_position_impact'] = df['team_strength_diff'] * df['league_draw_rate']
            
            # Calculate league competitiveness if possible
            if 'team_strength_diff' in df.columns:
                league_competitiveness = df.groupby('league_encoded').apply(
                    lambda x: 1 - x['team_strength_diff'].std() / x['team_strength_diff'].max()
                ).to_dict()
                df['league_competitiveness'] = df['league_encoded'].map(league_competitiveness)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in add_league_specific_features: {str(e)}")
            return df  # Return original dataframe on error
    
    def add_enhanced_draw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add enhanced draw-specific features."""
        try:
            df = df.copy()
            
            # Fix form convergence calculation
            df['form_convergence'] = df.groupby('league_encoded').apply(
                lambda x: 1 - abs(x['home_form_momentum'].diff().rolling(5).mean() - 
                                x['away_form_momentum'].diff().rolling(5).mean())
            ).reset_index(level=0, drop=True)  # Reset the MultiIndex
            
            # Ensure form_convergence is numeric
            df['form_convergence'] = pd.to_numeric(df['form_convergence'], errors='coerce').fillna(0)
               
            # League position volatility
            df['position_volatility'] = df.groupby('league_encoded')['team_strength_diff']\
                .rolling(5).std().reset_index(level=0, drop=True).fillna(0)
            
            return df
        except Exception as e:
            raise Exception(f"Error in add_enhanced_draw_features: {str(e)}")
    
    def add_draw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add draw-specific features."""
        try:
            df = df.copy()
            
            # Validate required columns
            required_cols = ['home_league_position', 'away_league_position', 'home_form_momentum', 'away_form_momentum',
                         'h2h_matches', 'h2h_draws', 'home_team_elo', 'away_team_elo', 'home_draw_rate', 'away_draw_rate']
            if not all(col in df.columns for col in required_cols):
                missing_cols = set(required_cols) - set(df.columns)
                self.logger.error(f"Missing required columns for add_draw_features: {missing_cols}")
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Team strength features (using only pre-match information)
            if all(col in df.columns for col in ['home_league_position', 'away_league_position']):
                df['team_strength_diff'] = abs(df['home_league_position'] - df['away_league_position'])
                df['avg_league_position'] = (df['home_league_position'] + df['away_league_position']) / 2
            
            # Form and momentum features (ensure these use only historical data)
            if all(col in df.columns for col in ['home_form_momentum', 'away_form_momentum']):
                df['form_difference'] = df['home_form_momentum'] - df['away_form_momentum']
                df['form_similarity'] = 1 / (1 + abs(df['form_difference'])).replace(np.inf, 1)
            
            # Historical matchup features (using only previous matches)
            if all(col in df.columns for col in ['h2h_matches', 'h2h_draws']):
                df['h2h_draw_rate'] = df['h2h_draws'] / df['h2h_matches'].replace(0, 1)
            
            # Team quality features (using only historical ratings)
            if all(col in df.columns for col in ['home_team_elo', 'away_team_elo']):
                df['elo_difference'] = abs(df['home_team_elo'] - df['away_team_elo'])
                df['elo_similarity'] = 1 / (1 + df['elo_difference'] / 100).replace(np.inf, 1)
            
            # Draw probability features (using only historical data)
            if all(col in df.columns for col in ['home_draw_rate', 'away_draw_rate']):
                df['combined_draw_rate'] = (df['home_draw_rate'] + df['away_draw_rate']) / 2
            return df
            
        except Exception as e:
            raise Exception(f"Error in add_draw_features: {str(e)}")

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def create_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between key predictors.
        
        This function creates interaction terms by multiplying pairs of features
        to capture combined effects between different predictors.
        """
        try:
            df = df.copy()
            
            # Select features for interactions
            key_features = [
                'draw_probability_score',
                'elo_similarity',
                'xg_similarity',
                'form_similarity'
            ]
            
            # Create interaction features only for existing columns
            existing_features = [f for f in key_features if f in df.columns]
            
            # Create interactions between all pairs of features
            for i in range(len(existing_features)):
                for j in range(i + 1, len(existing_features)):
                    feat1 = existing_features[i]
                    feat2 = existing_features[j]
                    interaction_name = f'{feat1}_{feat2}'
                    df[interaction_name] = df[feat1] * df[feat2]
            
            return df
            
        except Exception as e:
            raise Exception(f"Error in create_interaction_features: {str(e)}")

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
                df['home_form_momentum'] = (
                    df['home_win_rate'] * 3 +  # Points from wins
                    df['home_draw_rate'] * 1    # Points from draws
                )
                
                df['away_form_momentum'] = (
                    df['away_win_rate'] * 3 +  # Points from wins
                    df['away_draw_rate'] * 1    # Points from draws
                )
                
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
            if all(col in df.columns for col in ['home_poisson_xG', 'home_form_momentum', 'away_poisson_xG', 'away_form_momentum']):
                df['xg_form_equilibrium'] = 1 - abs(
                    (df['home_poisson_xG'] * df['home_form_momentum']) -
                    (df['away_poisson_xG'] * df['away_form_momentum'])
                )
                
                # Expected Goals Momentum
                df['home_xg_momentum'] = df['home_poisson_xG'] * (1 + df['home_form_momentum'])
                df['away_xg_momentum'] = df['away_poisson_xG'] * (1 + df['away_form_momentum'])
                df['xg_momentum_similarity'] = 1 - abs(df['home_xg_momentum'] - df['away_xg_momentum'])
            
            # Form-Weighted Expected Goals (if win rates are available)
            if all(col in df.columns for col in ['home_win_rate', 'away_win_rate', 'home_poisson_xG', 'away_poisson_xG']):
                df['home_form_weighted_xg'] = df['home_poisson_xG'] * df['home_win_rate']
                df['away_form_weighted_xg'] = df['away_poisson_xG'] * df['away_win_rate']
                df['form_weighted_xg_diff'] = abs(df['home_form_weighted_xg'] - df['away_form_weighted_xg'])
            
            # Attack Strength and xG Combination
            if all(col in df.columns for col in ['home_attack_strength', 'away_attack_strength', 'home_poisson_xG', 'away_poisson_xG']):
                df['home_attack_xg_power'] = df['home_attack_strength'] * df['home_poisson_xG']
                df['away_attack_xg_power'] = df['away_attack_strength'] * df['away_poisson_xG']
                df['attack_xg_equilibrium'] = 1 - abs(df['home_attack_xg_power'] - df['away_attack_xg_power'])
            
            # Rolling xG Form
            if all(col in df.columns for col in ['home_xG_rolling_rollingaverage', 'away_xG_rolling_rollingaverage', 'home_form_momentum', 'away_form_momentum']):
                df['home_xg_form'] = df['home_xG_rolling_rollingaverage'] * df['home_form_momentum']
                df['away_xg_form'] = df['away_xG_rolling_rollingaverage'] * df['away_form_momentum']
                df['xg_form_similarity'] = 1 - abs(df['home_xg_form'] - df['away_xg_form'])
            
            # Create draw probability enhancers if all required features are available
            required_features = [
                'xg_form_equilibrium', 'xg_momentum_similarity', 'form_weighted_xg_diff',
                'attack_xg_equilibrium'
            ]
            if all(col in df.columns for col in required_features):
                df['draw_xg_indicator'] = (
                    (df['xg_form_equilibrium'] * 0.3) +
                    (df['xg_momentum_similarity'] * 0.3) +
                    (1 - df['form_weighted_xg_diff']) * 0.2 +
                    (df['attack_xg_equilibrium'] * 0.2)
                )
                
                # Update draw probability score if it exists
                if 'draw_probability_score' in df.columns:
                    df['draw_probability_score'] = (
                        df['draw_probability_score'] * 0.7 +  # Original score
                        df['draw_xg_indicator'] * 0.3    # New xG-based indicators
                    )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in create_composite_features: {str(e)}")
            return df  # Return original dataframe on error

    def add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds advanced features to the dataframe.
        """
        try:
            df = df.copy()
            
            df['is_draw'] = (df['match_outcome'] == 2).astype(int)
            
            all_teams = np.concatenate([df['home_encoded'].unique(), df['away_encoded'].unique()])
            all_teams = np.unique(all_teams)
            all_matches = df[['running_id','home_encoded','away_encoded','date_encoded','is_draw']]
            all_matches = all_matches.sort_values('date_encoded',ascending=True)
            
            # Momentum and Form Features
            print('goal_differential_trend')
            if all(col in df.columns for col in ['home_goal_difference_rollingaverage', 'away_goal_difference_rollingaverage']):
                df['goal_differential_trend'] = self.calculate_exponential_decay(df, 'home_goal_difference_rollingaverage') - \
                                                self.calculate_exponential_decay(df, 'away_goal_difference_rollingaverage')

            if 'is_draw' in df.columns:
                print('home_draw_streak_length')
                df['home_draw_streak_length'] = df.groupby('home_encoded')['is_draw'].cumsum()
                df['away_draw_streak_length'] = df.groupby('away_encoded')['is_draw'].cumsum()

            if all(col in df.columns for col in ['home_points_cum', 'Away_points_cum']):
                print('home_form_consistency_index')
                df['home_form_consistency_index'] = df.groupby('home_encoded')['home_points_cum'].rolling(window=5).var().reset_index(level=0, drop=True)
                df['away_form_consistency_index'] = df.groupby('away_encoded')['Away_points_cum'].rolling(window=5).var().reset_index(level=0, drop=True)

            if all(col in df.columns for col in ['home_points_cum', 'Away_points_cum']):
                print('home_points_acceleration')
                df['home_points_acceleration'] = df.groupby('home_encoded')['home_points_cum'].rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 2)[0]).reset_index(level=0, drop=True)
                df['away_points_acceleration'] = df.groupby('away_encoded')['Away_points_cum'].rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 2)[0]).reset_index(level=0, drop=True)

            # Timing and Context Features
            if 'is_draw' in df.columns and 'date_encoded' in df.columns:
                print('home_last_draw_date')
                df['home_last_draw_date'] = 0
                df['away_last_draw_date'] = 0
                df['home_last_match_date'] = 0
                df['away_last_match_date'] = 0
                
                for index, row in df.iterrows():
                    home_team = row['home_encoded']
                    away_team = row['away_encoded']
                    current_date = row['date_encoded']
                    
                    home_last_match = all_matches[
                        ((all_matches['home_encoded'] == home_team) | (all_matches['away_encoded'] == home_team)) &
                        (all_matches['date_encoded'] < current_date) 
                    ]
                    
                    away_last_match = all_matches[
                        ((all_matches['home_encoded'] == away_team) | (all_matches['away_encoded'] == away_team)) &
                        (all_matches['date_encoded'] < current_date) 
                    ]
                    
                    home_last_draw = home_last_match[home_last_match['is_draw'] == 1]['date_encoded'].max()
                    away_last_draw = away_last_match[away_last_match['is_draw'] == 1]['date_encoded'].max()
                    home_rest_days = current_date - home_last_match['date_encoded'].max()
                    away_rest_days = current_date - away_last_match['date_encoded'].max()
                    
                    df.at[index, 'home_last_draw_date'] = home_last_draw if not pd.isna(home_last_draw) else 0
                    df.at[index, 'away_last_draw_date'] = away_last_draw if not pd.isna(away_last_draw) else 0
                    df.at[index, 'home_rest_days'] = home_rest_days if not pd.isna(home_rest_days) else 0
                    df.at[index, 'away_rest_days'] = away_rest_days if not pd.isna(away_rest_days) else 0
                    
                df['home_days_since_last_draw'] = (df['date_encoded'] - df['home_last_draw_date'])
                df['away_days_since_last_draw'] = (df['date_encoded'] - df['away_last_draw_date'])
                
                df['home_rest_days'] = df['home_rest_days']
                df['away_rest_days'] = df['away_rest_days']
                df['rest_advantage'] = df['home_rest_days'] - df['away_rest_days']
                

            if 'date_encoded' in df.columns:
                print('season_phase')
                try:
                    df['season_phase'] = pd.cut(df['date_encoded'], bins=3, labels=['early', 'mid', 'late'])
                except Exception as e:
                    print(f"Error in season_phase: {str(e)}")
                    df['season_phase'] = pd.cut(df['date_encoded'], bins=3, labels=['early', 'mid', 'late'])
                
                try:
                    for phase in ['early', 'mid', 'late']:
                        print(f'home_performance_{phase}')
                        df[f'home_performance_{phase}'] = df.groupby(['home_encoded', 'season_phase'], observed=False)['Home_points_cum'].transform('mean')
                        df[f'away_performance_{phase}'] = df.groupby(['away_encoded', 'season_phase'], observed=False)['Away_points_cum'].transform('mean')
                except Exception as e:
                    print(f"Error in home_performance_{phase}: {str(e)}")

                print('home_fixture_congestion')
                try:    
                    df['home_fixture_congestion'] = df.groupby('home_encoded')['date_encoded'].rolling(window=14).count().reset_index(level=0, drop=True)
                    df['away_fixture_congestion'] = df.groupby('away_encoded')['date_encoded'].rolling(window=14).count().reset_index(level=0, drop=True)
                except Exception as e:
                    print(f"Error in home_fixture_congestion: {str(e)}")

            # Style Compatibility Metrics
            print('defensive_style_similarity')
            try:
                if all(col in df.columns for col in ['home_saves_rollingaverage', 'away_saves_rollingaverage', 'home_interceptions_rollingaverage', 'away_interceptions_rollingaverage', 'home_fouls_rollingaverage', 'away_fouls_rollingaverage']):
                    df['defensive_style_similarity'] = 1 - (abs(df['home_saves_rollingaverage'] - df['away_saves_rollingaverage']) + \
                                                        abs(df['home_interceptions_rollingaverage'] - df['away_interceptions_rollingaverage']) + \
                                                        abs(df['home_fouls_rollingaverage'] - df['away_fouls_rollingaverage']))
            except Exception as e:
                print(f"Error in defensive_style_similarity: {str(e)}")

            print('possession_style_clash')
            try:
                if all(col in df.columns for col in ['Home_possession_mean', 'away_possession_mean']):
                    df['possession_style_clash'] = abs(df['Home_possession_mean'] - df['away_possession_mean'])
            except Exception as e:
                print(f"Error in possession_style_clash: {str(e)}")

            print('tempo_differential')
            try:
                if all(col in df.columns for col in ['home_passes_rollingaverage', 'away_passes_rollingaverage', 'home_shots_rollingaverage', 'away_shots_rollingaverage', 'home_corners_rollingaverage', 'away_corners_rollingaverage']):
                    df['tempo_differential'] = abs(df['home_passes_rollingaverage'] - df['away_passes_rollingaverage']) + \
                                            abs(df['home_shots_rollingaverage'] - df['away_shots_rollingaverage']) + \
                                            abs(df['home_corners_rollingaverage'] - df['away_corners_rollingaverage'])
            except Exception as e:
                print(f"Error in tempo_differential: {str(e)}")


            print('home_tactical_flexibility')
            try:
                if all(col in df.columns for col in ['Home_possession_mean', 'away_possession_mean', 'home_shots_rollingaverage', 'away_shots_rollingaverage', 'home_passes_rollingaverage', 'away_passes_rollingaverage']):
                    df['home_tactical_flexibility'] = df['Home_possession_mean'] + \
                                                df['home_shots_rollingaverage'] + \
                                                df['home_passes_rollingaverage']
                    df['away_tactical_flexibility'] = df['away_possession_mean'] + \
                                                df['away_shots_rollingaverage'] + \
                                                df['away_passes_rollingaverage']
            except Exception as e:
                print(f"Error in home_tactical_flexibility: {str(e)}")

            # Historical Pattern Features
            try:
                if 'h2h_draws' in df.columns:
                    print('h2h_draw_frequency_weighted')
                    df['h2h_draw_frequency_weighted'] = self.calculate_exponential_decay(df, 'h2h_draws')
            except Exception as e:
                print(f"Error in h2h_draw_frequency_weighted: {str(e)}")


            try:
                if 'venue_encoded' in df.columns:
                    print('venue_draw_tendency')
                    df['venue_draw_tendency'] = df.groupby('venue_encoded')['is_draw'].transform('mean')
            except Exception as e:
                print(f"Error in venue_draw_tendency: {str(e)}")

            try:
                if 'referee_encoded' in df.columns:
                    print('referee_draw_pattern_index')
                    df['referee_draw_pattern_index'] = df.groupby('referee_encoded')['is_draw'].transform('mean')
            except Exception as e:
                print(f"Error in referee_draw_pattern_index: {str(e)}")

            try:
                if all(col in df.columns for col in ['h2h_draws', 'league_draw_rate_composite']):
                    print('matchup_draw_probability')
                    df['matchup_draw_probability'] = df['h2h_draws'] * 0.5 + df['league_draw_rate_composite'] * 0.5
            except Exception as e:
                print(f"Error in matchup_draw_probability: {str(e)}")

            try:
                print('defensive_equilibrium_score')
                if all(col in df.columns for col in ['home_saves_rollingaverage', 'away_saves_rollingaverage', 'home_interceptions_rollingaverage', 'away_interceptions_rollingaverage']):
                    df['defensive_equilibrium_score'] = 1 - (abs(df['home_saves_rollingaverage'] - df['away_saves_rollingaverage']) + \
                                                        abs(df['home_interceptions_rollingaverage'] - df['away_interceptions_rollingaverage']))
            except Exception as e:
                print(f"Error in defensive_equilibrium_score: {str(e)}")

            try:
                print('attacking_efficiency_balance')
                if all(col in df.columns for col in ['home_shots_rollingaverage', 'away_shots_rollingaverage', 'home_goals_rollingaverage', 'away_goals_rollingaverage']):
                    df['attacking_efficiency_balance'] = 1 - (abs(df['home_shots_rollingaverage'] - df['away_shots_rollingaverage']) + \
                                                        abs(df['home_goals_rollingaverage'] - df['away_goals_rollingaverage']))
            except Exception as e:
                print(f"Error in attacking_efficiency_balance: {str(e)}")

            try:
                print('pressure_resistance_ratio')
                if all(col in df.columns for col in ['Home_possession_mean', 'away_possession_mean', 'home_passes_rollingaverage', 'away_passes_rollingaverage']):
                    df['pressure_resistance_ratio'] = (df['Home_possession_mean'] / df['away_possession_mean']) * \
                                                (df['home_passes_rollingaverage'] / df['away_passes_rollingaverage'])
            except Exception as e:
                print(f"Error in pressure_resistance_ratio: {str(e)}")

            try:
                print('momentum_equilibrium_index')
                if all(col in df.columns for col in ['home_form_momentum', 'away_form_momentum']):
                    df['momentum_equilibrium_index'] = 1 - abs(df['home_form_momentum'] - df['away_form_momentum'])
            except Exception as e:
                print(f"Error in momentum_equilibrium_index: {str(e)}")

            # Composite Indicators
            try:
                if all(col in df.columns for col in ['is_draw', 'h2h_draws']):
                    print('draw_propensity_score')
                    df['draw_propensity_score'] = df['is_draw'].rolling(window=5).mean() * 0.6 + df['h2h_draws'] * 0.4
            except Exception as e:
                print(f"Error in draw_propensity_score: {str(e)}")

            try:
                if all(col in df.columns for col in ['home_points_cum', 'Away_points_cum']):
                    print('home_team_stability_index')
                    df['home_team_stability_index'] = df.groupby('home_encoded')['home_points_cum'].rolling(window=5).std().reset_index(level=0, drop=True)
                    df['away_team_stability_index'] = df.groupby('away_encoded')['Away_points_cum'].rolling(window=5).std().reset_index(level=0, drop=True)
            except Exception as e:
                print(f"Error in home_team_stability_index: {str(e)}")

            try:
                if all(col in df.columns for col in ['home_form_consistency_index', 'away_form_consistency_index', 'h2h_draws']):
                    print('matchup_volatility_score')
                    df['matchup_volatility_score'] = (df['home_form_consistency_index'] + df['away_form_consistency_index']) * 0.7 + \
                                                df['h2h_draws'] * 0.3
            except Exception as e:
                print(f"Error in matchup_volatility_score: {str(e)}")

            try:
                if all(col in df.columns for col in ['defensive_style_similarity', 'possession_style_clash', 'h2h_draw_frequency_weighted']):
                    print('tactical_deadlock_probability')
                    df['tactical_deadlock_probability'] = df['defensive_style_similarity'] * 0.4 + \
                                                    df['possession_style_clash'] * 0.3 + \
                                                    df['h2h_draw_frequency_weighted'] * 0.3
            except Exception as e:
                print(f"Error in tactical_deadlock_probability: {str(e)}")

            return df

        except Exception as e:
            self.logger.error(f"Error in add_advanced_features: {str(e)}")
            return df

    def calculate_exponential_decay(self, df: pd.DataFrame, column: str, alpha: float = 0.7) -> pd.Series:
        """
        Calculates exponential decay for a given column.
        """
        return df.groupby('home_encoded')[column].transform(lambda x: x.ewm(alpha=alpha).mean())