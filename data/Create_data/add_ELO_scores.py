import pandas as pd
import numpy as np
import os
import logging

class ELOCalculator:
    """
    Enhanced ELO Calculator with automatic league-specific K-factors.
    """
    def __init__(self, logger=None):
        """Initialize the ELOCalculator with required paths and settings."""
        self.logger = logger or logging.getLogger(__name__)
        
        # Set directories and files
        self.model_dir = "./models/"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Define data paths
        self.training_data_path = './data_files/model_data_training_newPoisson.xlsx'
        self.training_data_path_new = './data_files/model_data_training_withPoisson.xlsx'
        self.prediction_data_path = './data_files/model_data_prediction_newPoisson.xlsx'
        self.api_prediction_data_path = './data_files/api_football_prediction_newPoisson.xlsx'
        self.api_training_data_path = './data_files/api_football_training_newPoisson.xlsx'
        
        # Define export paths
        self.training_export_path = './data_files/model_data_training_newPoisson.xlsx'
        self.training_export_path_new = './data_files/model_data_training_withPoisson.xlsx'
        self.prediction_export_path = './data_files/model_data_prediction_newPoisson.xlsx'
        self.api_prediction_export_path = './data_files/api_football_prediction_newPoisson.xlsx'
        self.api_training_export_path = './data_files/api_football_training_newPoisson.xlsx'
        

        # ELO settings
        self.INITIAL_ELO = 1500
        self.league_k_factors = {}  # Will be calculated automatically
        self.elo_ratings = {}  # Dictionary to store ELO ratings
        self.current_season = None

    def calculate_league_k_factor(self, league_data):
        """
        Calculate K-factor for a specific league based on its competitiveness.
        Higher K-factor for more predictable leagues, lower for more volatile ones.
        """
        try:
            # Calculate league competitiveness metrics
            win_rate_std = np.std(pd.concat([league_data['home_win_rate'], 
                                           league_data['away_win_rate']]))
            
            points_std = np.std(pd.concat([league_data['home_average_points'], 
                                         league_data['away_average_points']]))
            
            goal_diff_std = np.std(pd.concat([league_data['Home_goal_difference_cum'], 
                                            league_data['Away_goal_difference_cum']]) / 
                                 pd.concat([league_data['Home_team_matches'], 
                                          league_data['Away_team_matches']]))
            
            # Normalize each metric between 0 and 1
            win_rate_factor = 1 - (win_rate_std / 0.5)  # 0.5 is max possible std for win rate
            points_factor = 1 - (points_std / 3)  # 3 is max points per match
            goal_diff_factor = 1 - (goal_diff_std / 5)  # 5 is a reasonable max std for goal diff
            
            # Combine factors to get K-factor between 20 and 40
            k_factor = 20 + (20 * (win_rate_factor + points_factor + goal_diff_factor) / 3)
            
            return round(k_factor, 2)
            
        except Exception as e:
            self.logger.error(f"Error calculating league K-factor: {str(e)}")
            return 30  # Default K-factor if calculation fails

    def calculate_expected_score(self, elo_a, elo_b):
        """Calculate expected score for team A based on ELO ratings."""
        return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

    def update_elo(self, elo_a, elo_b, score_a, k_factor):
        """Update ELO rating based on match result using league-specific K-factor."""
        expected_a = self.calculate_expected_score(elo_a, elo_b)
        new_elo_a = elo_a + k_factor * (score_a - expected_a)
        return new_elo_a

    def reset_season_ratings(self, league_teams):
        """Reset ELO ratings for all teams in a league at season start."""
        for team in league_teams:
            self.elo_ratings[team] = self.INITIAL_ELO

    def add_elo_scores(self, matches):
        """Add ELO scores to match data with league-specific calculations."""
        try:
            # Calculate K-factors for each league
            for league in matches['league_encoded'].unique():
                league_data = matches[matches['league_encoded'] == league]
                self.league_k_factors[league] = self.calculate_league_k_factor(league_data)
                self.logger.info(f"League {league} K-factor: {self.league_k_factors[league]}")
            
            # Initialize ELO ratings for all teams
            all_teams = pd.concat([matches['home_encoded'], matches['away_encoded']]).unique()
            for team in all_teams:
                self.elo_ratings[team] = self.INITIAL_ELO

            # Add columns for ELO ratings
            matches['home_team_elo'] = 0.0
            matches['away_team_elo'] = 0.0
            elo_count = 0
            # Process matches chronologically
            for index, row in matches.sort_values('Datum').iterrows():
                # Check for season change
                if row['season_encoded'] != self.current_season:
                    self.current_season = row['season_encoded']
                    league_teams = matches[matches['league_encoded'] == row['league_encoded']]
                    league_teams = pd.concat([league_teams['home_encoded'], 
                                           league_teams['away_encoded']]).unique()
                    self.reset_season_ratings(league_teams)

                home_team = row['home_encoded']
                away_team = row['away_encoded']
                league = row['league_encoded']
                k_factor = self.league_k_factors.get(league, 30)

                # Get current ELO ratings
                home_elo = self.elo_ratings[home_team]
                away_elo = self.elo_ratings[away_team]

                # Store pre-match ELO ratings
                matches.at[index, 'home_team_elo'] = home_elo
                matches.at[index, 'away_team_elo'] = away_elo

                # Update ratings if we have actual results
                if 'home_goals' in row and 'away_goals' in row:
                    home_goals = row['home_goals']
                    away_goals = row['away_goals']

                    # Determine match outcome
                    if home_goals > away_goals:
                        home_score, away_score = 1, 0
                    elif home_goals < away_goals:
                        home_score, away_score = 0, 1
                    else:
                        home_score = away_score = 0.5

                    # Update ELO ratings
                    new_home_elo = self.update_elo(home_elo, away_elo, home_score, k_factor)
                    new_away_elo = self.update_elo(away_elo, home_elo, away_score, k_factor)

                    # Save new ratings
                    self.elo_ratings[home_team] = new_home_elo
                    self.elo_ratings[away_team] = new_away_elo
                    elo_count += 1
            print(f"ELO created for {elo_count} matches")
            return matches

        except Exception as e:
            self.logger.error(f"Error in add_elo_scores: {str(e)}")
            raise

    def process_data(self):
        """Process all data files with ELO calculations."""
        try:
            # Define numeric columns
            numeric_columns = [
                'home_win_rate', 'away_win_rate',
                'home_average_points', 'away_average_points',
                'Home_goal_difference_cum', 'Away_goal_difference_cum',
                'Home_team_matches', 'Away_team_matches',
                'home_goals', 'away_goals',
                'home_poisson_xG', 'away_poisson_xG',
                'home_attack_strength', 'away_attack_strength',
                'home_defense_weakness', 'away_defense_weakness',
                'home_goal_rollingaverage', 'away_goal_rollingaverage',
                'home_saves_rollingaverage', 'away_saves_rollingaverage',
                'Home_possession_mean', 'away_possession_mean',
                'Home_passes_mean', 'Away_passes_mean'
            ]

            def convert_numeric_columns(df):
                """Convert numeric columns from string with commas to float."""
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = df[col].apply(lambda x: float(str(x).replace(',', '.')) if isinstance(x, str) else float(x))
                return df

            # Process training data
            self.logger.info("Processing training data...")
            training_data = pd.read_excel(self.training_data_path)
            training_data = convert_numeric_columns(training_data)
            training_data = training_data.sort_values('Datum')
            training_data = self.add_elo_scores(training_data)
            training_data.to_excel(self.training_export_path, index=False)
            self.logger.info("Training data processed and saved")
            
            # Process new training data with Poisson
            self.logger.info("Processing new training data with Poisson...")
            training_data_new = pd.read_excel(self.training_data_path_new)
            training_data_new = convert_numeric_columns(training_data_new)
            training_data_new = training_data_new.sort_values('Datum')
            training_data_new = self.add_elo_scores(training_data_new)
            training_data_new.to_excel(self.training_export_path_new, index=False)
            self.logger.info("Training data with Poisson processed and saved")

            # Process prediction data
            self.logger.info("Processing prediction data...")
            prediction_data = pd.read_excel(self.prediction_data_path)
            prediction_data = convert_numeric_columns(prediction_data)
            prediction_data = prediction_data.sort_values('Datum')
            prediction_data = self.add_elo_scores(prediction_data)
            prediction_data.to_excel(self.prediction_export_path, index=False)
            self.logger.info("Prediction data processed and saved")
            
            # Process API data
            self.logger.info("Processing API data...")
            api_data = pd.read_excel(self.api_data_path)
            api_data = convert_numeric_columns(api_data)
            api_data = api_data.sort_values('Datum')
            api_data = self.add_elo_scores(api_data)
            api_data.to_excel(self.api_export_path, index=False)
            self.logger.info("API data processed and saved")

        except Exception as e:
            self.logger.error(f"Error in process_data: {str(e)}")
            raise

def main():
    """Main execution function."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('./log/elo_calculator.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    try:
        calculator = ELOCalculator(logger=logger)
        calculator.process_data()
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
