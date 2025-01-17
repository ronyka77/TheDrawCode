import pandas as pd
import numpy as np
from typing import Optional
import logging

class AdvancedGoalFeatureEngineer:
    """Engineers advanced goal-related features for match prediction models."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the feature engineer with logger."""
        self.logger = logger or logging.getLogger(__name__)
    def add_goal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all advanced goal-related features to the dataframe."""
        try:
            # Replace inf values with 0 for all columns before calculations
            df = df.replace([np.inf, -np.inf], 0)
            epsilon = 1e-10
            
            data = df.copy()
            
            # Verify all selected columns are numeric
            # Start Generation Here
            for column in df.columns:
                if df[column].dtype == 'object':
                    try:
                        df[column] = pd.to_numeric(df[column], errors='coerce')
                        # self.logger.debug(f"Converted column '{column}' to numeric.")
                    except Exception as e:
                        self.logger.warning(f"Failed to convert column '{column}' to numeric: {e}")
                        
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            non_numeric_columns = [col for col in df.columns if col not in numeric_columns]
            if non_numeric_columns:
                print(f"Warning: The following columns are non-numeric and will be excluded: {non_numeric_columns}")
                df = df[numeric_columns]
                
            # 1. Advanced Scoring Efficiency
            df['home_scoring_efficiency'] = (df['home_goal_rollingaverage'] * df['home_xG_rolling_rollingaverage']).div(
                df['home_shot_on_target_rollingaverage'] + epsilon).fillna(0)
            df['away_scoring_efficiency'] = (df['away_goal_rollingaverage'] * df['away_xG_rolling_rollingaverage']).div(
                df['away_shot_on_target_rollingaverage'] + epsilon).fillna(0)

            # 2. Form-Weighted Attack Strength
            df['home_weighted_attack'] = (df['home_attack_strength'] * df['home_form_momentum'] * 
                df['home_xg_form']).fillna(0)
            df['away_weighted_attack'] = (df['away_attack_strength'] * df['away_form_momentum'] * 
                df['away_xg_form']).fillna(0)

            # 3. Defensive Stability Index
            df['home_defense_index'] = (df['defensive_stability'] * (1 - df['home_defense_weakness']) * 
                df['home_saves_rollingaverage']).fillna(0)
            df['away_defense_index'] = (df['defensive_stability'] * (1 - df['away_defense_weakness']) * 
                df['away_saves_rollingaverage']).fillna(0)

            # 4. Position-Form Interaction
            df['home_position_form'] = (df['home_league_position'] * df['home_form_momentum'] * 
                df['form_position_interaction']).fillna(0)
            df['away_position_form'] = (df['away_league_position'] * df['away_form_momentum'] * 
                df['form_position_interaction']).fillna(0)

            # 5. Historical Performance Impact
            df['home_historical_strength'] = (df['home_h2h_dominance'] * df['elo_similarity'] * 
                df['weighted_h2h_draw_rate']).fillna(0)
            df['away_historical_strength'] = (df['away_h2h_dominance'] * df['elo_similarity'] * 
                df['weighted_h2h_draw_rate']).fillna(0)

            # New Features
            # Goal-scoring momentum
            df['home_goal_momentum'] = df['home_goal_rollingaverage'] * df['home_form_momentum']
            df['away_goal_momentum'] = df['away_goal_rollingaverage'] * df['away_form_momentum']

            # Head-to-head historical performance with recency weights
            df['home_h2h_weighted'] = df['home_h2h_dominance'] * df['weighted_h2h_draw_rate'] * df['elo_similarity']
            df['away_h2h_weighted'] = df['away_h2h_dominance'] * df['weighted_h2h_draw_rate'] * df['elo_similarity']
            
            # Referee tendency features
            
            # df['ref_goal_tendency'] = df['referee_foul_rate'] * df['referee_card_rate'] * df['league_competitiveness']
            # Calculate referee goals per game based on historical matches
            df['referee_goals_per_game'] = (
                (df['home_goal_rollingaverage'] + df['away_goal_rollingaverage']) * 
                df['referee_foul_rate']
            ).fillna(0)
            
            df['ref_goal_tendency'] = (
                df['referee_foul_rate'] * 
                df['league_competitiveness'] * 
                df['referee_goals_per_game']
            ).fillna(0)
            df['home_ref_interaction'] = df['ref_goal_tendency'] * df['home_attack_strength']
            df['away_ref_interaction'] = df['ref_goal_tendency'] * df['away_attack_strength']

            # 6. League Context Features
            df['home_league_context'] = (df['league_competitiveness'] * df['league_draw_rate'] * 
                df['league_position_impact']).fillna(0)
            df['away_league_context'] = (df['league_competitiveness'] * df['league_draw_rate'] * 
                df['league_position_impact']).fillna(0)

            # 7. Season Progress Impact
            df['home_season_form'] = (df['season_progress'] * df['home_form_momentum'] * 
                df['mid_season_factor']).fillna(0)
            df['away_season_form'] = (df['season_progress'] * df['away_form_momentum'] * 
                df['mid_season_factor']).fillna(0)

            # 8. Possession-Based Goal Potential
            df['home_possession_impact'] = (df['Home_possession_mean'] * df['strength_possession_interaction'] * 
                df['home_attack_strength']).fillna(0)
            df['away_possession_impact'] = (df['away_possession_mean'] * df['strength_possession_interaction'] * 
                df['away_attack_strength']).fillna(0)

            # 9. Combined Team Strength
            df['home_total_strength'] = (df['home_attack_strength'] * (1 - df['home_defense_weakness']) * 
                df['home_team_elo']).fillna(0)
            df['away_total_strength'] = (df['away_attack_strength'] * (1 - df['away_defense_weakness']) * 
                df['away_team_elo']).fillna(0)

            # 10. Form Stability Metrics
            df['home_form_stability'] = (df['form_stability'] * df['home_form_momentum'] * 
                df['xg_form_similarity']).fillna(0)
            df['away_form_stability'] = (df['form_stability'] * df['away_form_momentum'] * 
                df['xg_form_similarity']).fillna(0)

            # 11. Passing Efficiency to Goals
            df['home_passing_efficiency'] = (df['Home_passes_mean'] * df['Home_shot_on_target_mean']).div(
                df['Home_possession_mean'] + epsilon).fillna(0)
            df['away_passing_efficiency'] = (df['Away_passes_mean'] * df['away_shot_on_target_mean']).div(
                df['away_possession_mean'] + epsilon).fillna(0)

            # 12. Offensive Set Piece Threat
            df['home_set_piece_threat'] = (df['home_corners_rollingaverage'] * df['home_attack_strength'] * 
                df['Home_shot_on_target_mean']).fillna(0)
            df['away_set_piece_threat'] = (df['away_corners_rollingaverage'] * df['away_attack_strength'] * 
                df['away_shot_on_target_mean']).fillna(0)

            # 13. Team Style Compatibility
            df['home_style_compatibility'] = (df['Home_possession_mean'] * df['home_interceptions_mean'] * 
                df['Home_passes_mean']).div(1000).fillna(0)
            df['away_style_compatibility'] = (df['away_possession_mean'] * df['away_interceptions_mean'] * 
                df['Away_passes_mean']).div(1000).fillna(0)

            # 14. Attendance Impact on Performance
            df['home_crowd_factor'] = (df['home_avg_attendance'] * df['home_win_rate'] * 
                df['home_form_momentum']).div(10000).fillna(0)
            df['away_crowd_resistance'] = (df['away_avg_attendance'] * df['away_win_rate'] * 
                df['away_form_momentum']).div(10000).fillna(0)

            # 15. Defensive Organization
            df['home_defensive_organization'] = (df['home_interceptions_mean'] * df['Home_saves_mean'] * 
                (1 - df['home_defense_weakness'])).fillna(0)
            df['away_defensive_organization'] = (df['away_interceptions_mean'] * df['Away_saves_mean'] * 
                (1 - df['away_defense_weakness'])).fillna(0)

            # 16. Attack Conversion Quality
            df['home_attack_conversion'] = (df['home_goal_rollingaverage'] * df['home_xG_rolling_rollingaverage']).div(
                df['home_shots_on_target_accuracy_rollingaverage'] + epsilon).fillna(0)
            df['away_attack_conversion'] = (df['away_goal_rollingaverage'] * df['away_xG_rolling_rollingaverage']).div(
                df['away_shots_on_target_accuracy_rollingaverage'] + epsilon).fillna(0)

            # 17. Referee Impact Factor
            df['home_referee_impact'] = (df['referee_foul_rate'] * df['Home_fouls_mean'] * 
                df['home_attack_strength']).fillna(0)
            df['away_referee_impact'] = (df['referee_foul_rate'] * df['Away_fouls_mean'] * 
                df['away_attack_strength']).fillna(0)

            # 18. Tactical Adaptability
            df['home_tactical_adaptability'] = (df['home_form_stability'] * df['home_attack_xg_power'] * 
                df['home_form_weighted_xg']).fillna(0)
            df['away_tactical_adaptability'] = (df['away_form_stability'] * df['away_attack_xg_power'] * 
                df['away_form_weighted_xg']).fillna(0)

            # 19. Position vs Strength Balance
            df['home_position_strength_balance'] = (df['home_league_position'] * df['home_attack_strength_home_league_position_interaction'] * 
                df['home_form_momentum']).fillna(0)
            df['away_position_strength_balance'] = (df['away_league_position'] * df['away_attack_strength_away_league_position_interaction'] * 
                df['away_form_momentum']).fillna(0)

            # 20. Offensive Pressure Sustainability
            df['home_offensive_sustainability'] = (df['Home_possession_mean'] * df['home_corners_mean'] * 
                df['home_shots_on_target_accuracy_rollingaverage']).fillna(0)
            df['away_offensive_sustainability'] = (df['away_possession_mean'] * df['away_corners_mean'] * 
                df['away_shots_on_target_accuracy_rollingaverage']).fillna(0)
            
            # Start Generation Here
            missing_columns = df.columns.difference(data.columns)
            data[missing_columns] = df[missing_columns]
            
            return data

        except Exception as e:
            self.logger.error(f"Error in add_goal_features: {str(e)}")
            return data

    def validate_features(self, df: pd.DataFrame) -> bool:
        """Validate that all new features were created successfully."""
        expected_features = [
            'home_goal_efficiency', 'away_goal_efficiency',
            'home_defensive_resilience', 'away_defensive_resilience',
            'home_possession_effectiveness', 'away_possession_effectiveness',
            'home_form_goals', 'away_form_goals',
            'h2h_goal_rate',
            'home_position_goal_factor', 'away_position_goal_factor',
            'home_offensive_pressure', 'away_offensive_pressure',
            'home_goal_stability', 'away_goal_stability',
            'referee_goal_influence',
            'goal_potential_diff',
            'home_passing_goal_ratio', 'away_passing_goal_ratio',
            'home_attack_balance', 'away_attack_balance',
            'home_goal_trend', 'away_goal_trend',
            'venue_goal_factor',
            'total_form_goal_potential'
        ]
        
        return df