import pymongo
from typing import List, Dict
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

class MongoDBFeatures:
    """
    A class to interact with MongoDB and retrieve fixtures where home.stats is not empty.
    """
    def __init__(self, logger=None):
        self.logger = logger
        self.mongo_uri = 'mongodb://192.168.0.77:27017/'
        self.client = pymongo.MongoClient(self.mongo_uri)
        self.db = self.client["api-football"]
        self.fixtures_collection = self.db["fixtures"]

    def get_fixtures_with_home_stats(self) -> List[Dict]:
        """
        Retrieves all fixtures from the MongoDB fixtures collection where home.stats is not empty.

        Returns:
            List[Dict]: List of fixtures with home.stats.
        """
        query = {
            "home.stats": {"$ne": {}},
            "score.fulltime.home": {"$ne": None},
            "score.fulltime.away": {"$ne": None},
            "match_outcome": {"$ne": None}
        }
        fixtures = self.fixtures_collection.find(query)
        fixture_list = list(fixtures)
        print(f"Found {len(fixture_list)} fixtures with home.stats.")
        return fixture_list
   
    def normalize_fixtures_data(self, fixtures_with_stats: List[Dict]) -> pd.DataFrame:
        """
        Normalizes the fixtures data and returns a pandas DataFrame.

        Args:
            fixtures_with_stats (List[Dict]): List of fixtures with home.stats.

        Returns:
            pd.DataFrame: Normalized fixtures data.
        """
        try:
            normalized_data = []
            
            for fixture in fixtures_with_stats:
                 # Debug print to see the structure
                print(f"Fixture structure: {fixture.keys()}")
                print(f"Home team data: {fixture['home'].keys()}")
                
                # Extract home team stats - add safety checks
                home_stats = fixture.get('home', {}).get('stats', {})
                if not home_stats:
                    print(f"Missing home stats for fixture: {fixture.get('fixture_id')}")
                    continue
                
                # Extract away team stats - add safety checks    
                away_stats = fixture.get('away', {}).get('stats', {})
                if not away_stats:
                    print(f"Missing away stats for fixture: {fixture.get('fixture_id')}")
                    continue
                    
                fixture_id = fixture['fixture_id']
                date = fixture['date']
                league_id = fixture['league_id']
                league_name = fixture['league_name']
                league_season = fixture['league_season']
                match_outcome = fixture['match_outcome']
                referee = fixture['referee']
                venue_id = fixture['venue_id']
                venue_name = fixture['venue_name']
                home_team_id = fixture['home']['team_id']
                home_team_name = fixture['home']['team_name']
                away_team_id = fixture['away']['team_id']
                away_team_name = fixture['away']['team_name']
                
                # GOALS
                home_goals = fixture['score']['fulltime']['home']
                away_goals = fixture['score']['fulltime']['away']
                home_halftime_goals = fixture['score']['halftime']['home']
                away_halftime_goals = fixture['score']['halftime']['away']
                
                # Extract home team stats
                home_shots_on_goal = home_stats.get('shots_on_goal')
                home_shots_off_goal = home_stats.get('shots_off_goal')
                home_total_shots = home_stats.get('total_shots')
                home_blocked_shots = home_stats.get('blocked_shots')
                home_shots_insidebox = home_stats.get('shots_insidebox')
                home_shots_outsidebox = home_stats.get('shots_outsidebox')
                home_expected_goals = home_stats.get('expected_goals')
                home_prevented_goals = home_stats.get('prevented_goals')
                home_fouls = home_stats.get('fouls')
                home_corner_kicks = home_stats.get('corner_kicks')
                home_offsides = home_stats.get('offsides')
                home_ball_possession = home_stats.get('ball_possession')
                home_yellow_cards = home_stats.get('yellow_cards')
                home_red_cards = home_stats.get('red_cards')
                home_goalkeeper_saves = home_stats.get('goalkeeper_saves')
                home_total_passes = home_stats.get('total_passes')
                home_passes = home_stats.get('passes_accurate')
                home_passes_accuracy = home_stats.get('passes_%')
                
                # Extract away team stats
                away_shots_on_goal = away_stats.get('shots_on_goal')
                away_shots_off_goal = away_stats.get('shots_off_goal')
                away_total_shots = away_stats.get('total_shots')
                away_blocked_shots = away_stats.get('blocked_shots')
                away_shots_insidebox = away_stats.get('shots_insidebox')
                away_shots_outsidebox = away_stats.get('shots_outsidebox')
                away_expected_goals = away_stats.get('expected_goals')
                away_prevented_goals = away_stats.get('prevented_goals')
                away_fouls = away_stats.get('fouls')
                away_corner_kicks = away_stats.get('corner_kicks')
                away_offsides = away_stats.get('offsides')
                away_ball_possession = away_stats.get('ball_possession')
                away_yellow_cards = away_stats.get('yellow_cards')
                away_red_cards = away_stats.get('red_cards')
                away_goalkeeper_saves = away_stats.get('goalkeeper_saves')
                away_total_passes = away_stats.get('total_passes')
                away_passes = away_stats.get('passes_accurate')
                away_passes_accuracy = away_stats.get('passes_%')
                
                normalized_data.append({
                    'fixture_id': fixture_id,
                    'date': date,
                    'league_id': league_id,
                    'league_season': league_season,
                    'league_name': league_name,
                    'referee': referee,
                    'venue_name': venue_name,
                    'venue_id': venue_id,
                    'home_goals': home_goals,
                    'away_goals': away_goals,
                    'home_halftime_goals': home_halftime_goals,
                    'away_halftime_goals': away_halftime_goals,
                    'match_outcome': match_outcome,
                    'home_team_id': home_team_id,
                    'home_team_name': home_team_name,
                    'away_team_id': away_team_id,
                    'away_team_name': away_team_name,
                    'home_shots_on_goal': home_shots_on_goal,
                    'home_prevented_goals': home_prevented_goals,
                    'home_shots_off_goal': home_shots_off_goal,
                    'home_total_shots': home_total_shots,
                    'home_blocked_shots': home_blocked_shots,
                    'home_shots_insidebox': home_shots_insidebox,
                    'home_shots_outsidebox': home_shots_outsidebox,
                    'home_expected_goals': home_expected_goals,
                    'home_fouls': home_fouls,
                    'home_corners': home_corner_kicks,
                    'home_offsides': home_offsides,
                    'home_ball_possession': home_ball_possession,
                    'home_yellow_cards': home_yellow_cards,
                    'home_red_cards': home_red_cards,
                    'home_saves': home_goalkeeper_saves,
                    'home_total_passes': home_total_passes,
                    'home_passes_accuracy': home_passes_accuracy,
                    'home_passes': home_passes,
                    'away_shots_on_goal': away_shots_on_goal,
                    'away_shots_off_goal': away_shots_off_goal,
                    'away_total_shots': away_total_shots,
                    'away_prevented_goals': away_prevented_goals,
                    'away_blocked_shots': away_blocked_shots,
                    'away_shots_insidebox': away_shots_insidebox,
                    'away_shots_outsidebox': away_shots_outsidebox,
                    'away_expected_goals': away_expected_goals,
                    'away_fouls': away_fouls,
                    'away_corners': away_corner_kicks,
                    'away_offsides': away_offsides,
                    'away_ball_possession': away_ball_possession,
                    'away_yellow_cards': away_yellow_cards,
                    'away_red_cards': away_red_cards,
                    'away_saves': away_goalkeeper_saves,
                    'away_total_passes': away_total_passes,
                    'away_passes_accuracy': away_passes_accuracy,
                    'away_passes': away_passes
                })
        
            fixtures_dataframe = pd.DataFrame(normalized_data)
            
            fixtures_dataframe['date'] = pd.to_datetime(fixtures_dataframe['date'], format='mixed', utc=True).dt.strftime('%Y-%m-%d %H:%M')
            
            for col in fixtures_dataframe.columns:
                if fixtures_dataframe[col].dtype == 'object':
                    if fixtures_dataframe[col].str.contains('%').any():
                        fixtures_dataframe[col] = fixtures_dataframe[col].str.rstrip('%').astype(float) / 100.0
                        if '_accuracy' not in col:
                            fixtures_dataframe.rename(columns={col: col + '_accuracy'}, inplace=True)
                        
            pd.set_option('future.no_silent_downcasting', True)
            fixtures_dataframe.fillna(0, inplace=True)
            fixtures_dataframe = fixtures_dataframe.infer_objects(copy=False)
            fixtures_dataframe.reset_index(drop=True, inplace=True)
            print(f"Normalized fixtures data: {fixtures_dataframe.shape}")
            return fixtures_dataframe
        except Exception as e:
            print(f"Error normalizing fixtures data: {e}")
            print(f"Error fixture: {fixture}")
            return pd.DataFrame()
    
    def export_to_excel(self, fixtures_dataframe: pd.DataFrame, file_path: str) -> None:
        """
        Exports the DataFrame to an Excel file, handling data types and formatting.

        Args:
            fixtures_dataframe (pd.DataFrame): The DataFrame to export.
            file_path (str): The path to save the Excel file.
        """
        try:
            fixtures_dataframe.to_excel(file_path, index=False)
            print(f"Data exported to {file_path}")
        except Exception as e:
            print(f"Error exporting data to Excel: {e}")
     
    def add_features(self, fixtures_dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Adds features to the fixtures DataFrame.

        Args:
            fixtures_dataframe (pd.DataFrame): The DataFrame to add features to.

        Returns:
            pd.DataFrame: The DataFrame with added features.
        """
        try:
            print("Start adding features...")
            fixtures_dataframe = self.load_and_prepare_data(fixtures_dataframe)
            
            print("Start adding cumulative sums...")
            fixtures_dataframe = self.add_cumulative_sums(fixtures_dataframe)
            
            print("Start adding rolling averages...")
            fixtures_dataframe = self.add_rolling_averages(fixtures_dataframe)
            
            print("Features added")
            return fixtures_dataframe
        except Exception as e:
            print(f"Error adding features to fixtures data: {e}")
            if self.logger:
                self.logger.error(f"Error adding features to fixtures data: {e}")
            return pd.DataFrame()
        
    def load_and_prepare_data(self, fixtures_dataframe: pd.DataFrame):
        try:
            print("Collecting data...")
            data = fixtures_dataframe
            print(data.shape)
            print("Dropping rows with missing match_outcome...")
            data = data.dropna(subset=['match_outcome'])
            print("Data Collected, Start Cleaning and Feature Engineering...")
            if self.logger:
                self.logger.info("Data Collected, Start Cleaning and Feature Engineering...")

            # Drop irrelevant columns
            data = data.drop(columns=['prediction_outcome', 'model_prediction'], errors='ignore')

            # Type conversions and extracting date components
            data['home_advantage'] = 1
            data['Date'] = pd.to_datetime(data['date'], errors='coerce')
            data['year'] = data['Date'].dt.year
            data['month'] = data['Date'].dt.month
            data['day_of_month'] = data['Date'].dt.day
            data['day_of_week'] = data['Date'].dt.dayofweek
            data['week_of_year'] = data['Date'].dt.isocalendar().week
            
            # Label encoding for categorical variables
            le = LabelEncoder()
            data['venue_encoded'] = data['venue_id']
            data['season_encoded'] = data['league_season']
            data['league_encoded'] = data['league_id']
            data['home_encoded'] = data['home_team_id']
            data['away_encoded'] = data['away_team_id']
            
            # First, sort the data based on the year, month, and day_of_month for each group.
            data = data.sort_values(by=['fixture_id'])

            # Numeric conversion and NaN handling
            data['home_xG'] = pd.to_numeric(data['home_expected_goals'], errors='coerce').fillna(0)
            data['away_xG'] = pd.to_numeric(data['away_expected_goals'], errors='coerce').fillna(0)

            print('Start possession and shooting...')
            if self.logger:
                self.logger.info('Start possession and shooting...')
            # Feature engineering for possession and shooting
            data['total_fouls'] = data['home_fouls'] + data['away_fouls']
            data['home_cards'] = data['home_yellow_cards'] + data['home_red_cards']
            data['away_cards'] = data['away_yellow_cards'] + data['away_red_cards']
            data['home_possession_shooting'] = data['home_passes_accuracy'] * data['home_shots_on_goal']
            data['away_possession_shooting'] = data['away_passes_accuracy'] * data['away_shots_on_goal']

            # Referee Stats
            if 'referee' in data.columns:
                data['referee_encoded'] = le.fit_transform(data['referee'].astype(str))
            else:
                raise KeyError("Error: 'referee' column not found in DataFrame.")

            referee_stats = data.groupby('referee_encoded')['total_fouls'].mean()
            data['referee_foul_rate'] = data['referee_encoded'].map(referee_stats)

            # Additional feature engineering
            print('Base feature engineering done, start additional features...')
            if self.logger:
                self.logger.info('Base feature engineering done, start additional features...')
            data['home_possession_diff'] = data['home_passes_accuracy'] - data['away_passes_accuracy']
            data['pass_accuracy_diff'] = data['home_passes'] - data['away_passes']
            data['home_shooting_accuracy'] = data['home_shots_on_goal'] / data['home_total_shots']
            data['away_shooting_accuracy'] = data['away_shots_on_goal'] / data['away_total_shots']
            data['home_shots_on_target_ratio'] = data['home_shots_on_goal'] / data['away_shots_on_goal']
            data['away_shots_on_target_ratio'] = data['away_shots_on_goal'] / data['home_shots_on_goal']
            data['home_saves_accuracy'] = data['home_saves'] / data['away_shots_on_goal']
            data['away_saves_accuracy'] = data['away_saves'] / data['home_shots_on_goal']

            # Defensive activity based on available data, including interceptions and duels won
            data['home_defensive_activity'] = data['home_blocked_shots'] + data['home_yellow_cards'] + data['home_red_cards'] + data['home_prevented_goals']
            data['away_defensive_activity'] = data['away_blocked_shots'] + data['away_yellow_cards'] + data['away_red_cards'] + data['away_prevented_goals']

            # Set-piece threat and foul impact
            data['home_set_piece_threat'] = data['home_corners'] + data['away_fouls']
            data['away_set_piece_threat'] = data['away_corners'] + data['home_fouls']

            # Outcome calculation
            print('Outcome calculation')
            if self.logger:
                self.logger.info('Outcome calculation')
            data['home_win'] = data['match_outcome'].apply(lambda x: 1 if x == 1 else 0)
            data['away_win'] = data['match_outcome'].apply(lambda x: 1 if x == 3 else 0)
            data['draw'] = data['match_outcome'].apply(lambda x: 1 if x == 2 else 0)
            
            # Points 
            print('Points calculation')
            if self.logger:
                self.logger.info('Points calculation')
            data['home_points'] = data['home_win'].apply(lambda x: 3 if x == 1 else 0) + data['draw']
            data['away_points'] = data['away_win'].apply(lambda x: 3 if x == 1 else 0) + data['draw']
            
            data['home_goal_difference'] = data['home_goals'] - data['away_goals']
            data['away_goal_difference'] = data['away_goals'] - data['home_goals']

            # Rolling averages and cumulative sums
            print('Start calculating Cumulative values...')
            if self.logger:
                self.logger.info('Start calculating Cumulative values...')
            data['home_points_cumulative'] = data.groupby(['home_encoded', 'season_encoded', 'league_encoded'])['home_points'].cumsum()
            data['away_points_cumulative'] = data.groupby(['away_encoded', 'season_encoded', 'league_encoded'])['away_points'].cumsum()
            data['home_goal_diff_cumulative'] = data.groupby(['home_encoded', 'season_encoded', 'league_encoded'])['home_goal_difference'].cumsum()
            data['away_goal_diff_cumulative'] = data.groupby(['away_encoded', 'season_encoded', 'league_encoded'])['away_goal_difference'].cumsum()
            return data 
        
        except Exception as e:
            print(f"Error in load_and_prepare_data: {e}")
            if self.logger:
                self.logger.error(f"Error in load_and_prepare_data: {e}")
            return pd.DataFrame()
    
    def add_cumulative_sums(self, dataframe):
        try:
            cumulative_columns = ['fixture_id','Date','home_encoded','away_encoded', 'season_encoded', 'league_encoded','home_saves','away_saves',
                                'home_shots_on_goal','away_shots_on_goal','home_passes_accuracy','away_passes_accuracy','home_fouls','away_fouls'
                                ,'home_points','away_points','home_goal_difference','away_goal_difference','home_win','away_win','draw','home_red_cards','away_red_cards','home_yellow_cards','away_yellow_cards']
            
            dataframe = dataframe.replace([np.inf, -np.inf], 0)
            
            cumsum_home_df = dataframe[cumulative_columns]
            cumsum_home_df = cumsum_home_df.copy()
            
            cumsum_home_df.loc[:,'home_points_cumsum'] = cumsum_home_df.groupby(['home_encoded', 'season_encoded', 'league_encoded'])['home_points'].cumsum() - cumsum_home_df['home_points']
            cumsum_home_df.loc[:,'home_saves_cumsum'] = cumsum_home_df.groupby(['home_encoded', 'season_encoded', 'league_encoded'])['home_saves'].cumsum() - cumsum_home_df['home_saves']
            cumsum_home_df.loc[:,'home_shots_on_target_cumsum'] = cumsum_home_df.groupby(['home_encoded', 'season_encoded', 'league_encoded'])['home_shots_on_goal'].cumsum() - cumsum_home_df['home_shots_on_goal']
            cumsum_home_df.loc[:,'home_passes_accuracy_cumsum'] = cumsum_home_df.groupby(['home_encoded', 'season_encoded', 'league_encoded'])['home_passes_accuracy'].cumsum() - cumsum_home_df['home_passes_accuracy']
            cumsum_home_df.loc[:,'home_fouls_cumsum'] = cumsum_home_df.groupby(['home_encoded', 'season_encoded', 'league_encoded'])['home_fouls'].cumsum() - cumsum_home_df['home_fouls']
            cumsum_home_df.loc[:,'home_goal_difference_cumsum'] = cumsum_home_df.groupby(['home_encoded', 'season_encoded', 'league_encoded'])['home_goal_difference'].cumsum() - cumsum_home_df['home_goal_difference']
            cumsum_home_df.loc[:,'home_win_cumsum'] = cumsum_home_df.groupby(['home_encoded', 'season_encoded', 'league_encoded'])['home_win'].cumsum() - cumsum_home_df['home_win']
            cumsum_home_df.loc[:,'draw_cumsum'] = cumsum_home_df.groupby(['home_encoded', 'season_encoded', 'league_encoded'])['draw'].cumsum() - cumsum_home_df['draw']
            cumsum_home_df.loc[:,'home_count'] = cumsum_home_df.groupby(['home_encoded', 'season_encoded', 'league_encoded'])['fixture_id'].cumcount()
            cumsum_home_df.loc[:,'red_cards_cumsum'] = cumsum_home_df.groupby(['home_encoded', 'season_encoded', 'league_encoded'])['home_red_cards'].cumsum() - cumsum_home_df['home_red_cards']
            cumsum_home_df.loc[:,'yellow_cards_cumsum'] = cumsum_home_df.groupby(['home_encoded', 'season_encoded', 'league_encoded'])['home_yellow_cards'].cumsum() - cumsum_home_df['home_yellow_cards']
            
            cumsum_away_df = dataframe[cumulative_columns]
            cumsum_away_df = cumsum_away_df.copy()
            
            cumsum_away_df.loc[:, 'away_points_cumsum'] = cumsum_away_df.groupby(['away_encoded', 'season_encoded', 'league_encoded'])['away_points'].cumsum() - cumsum_away_df['away_points']
            cumsum_away_df.loc[:, 'away_fouls_cumsum'] = cumsum_away_df.groupby(['away_encoded', 'season_encoded', 'league_encoded'])['away_fouls'].cumsum() - cumsum_away_df['away_fouls']
            cumsum_away_df.loc[:, 'away_saves_cumsum'] = cumsum_away_df.groupby(['away_encoded', 'season_encoded', 'league_encoded'])['away_saves'].cumsum() - cumsum_away_df['away_saves']
            cumsum_away_df.loc[:, 'away_shots_on_target_cumsum'] = cumsum_away_df.groupby(['away_encoded', 'season_encoded', 'league_encoded'])['away_shots_on_goal'].cumsum() - cumsum_away_df['away_shots_on_goal']
            cumsum_away_df.loc[:, 'away_passes_accuracy_cumsum'] = cumsum_away_df.groupby(['away_encoded', 'season_encoded', 'league_encoded'])['away_passes_accuracy'].cumsum() - cumsum_away_df['away_passes_accuracy']    
            cumsum_away_df.loc[:, 'away_goal_difference_cumsum'] = cumsum_away_df.groupby(['away_encoded', 'season_encoded', 'league_encoded'])['away_goal_difference'].cumsum() - cumsum_away_df['away_goal_difference']
            cumsum_away_df.loc[:, 'away_win_cumsum'] = cumsum_away_df.groupby(['away_encoded', 'season_encoded', 'league_encoded'])['away_win'].cumsum() - cumsum_away_df['away_win']   
            cumsum_away_df.loc[:, 'draw_cumsum'] = cumsum_away_df.groupby(['away_encoded', 'season_encoded', 'league_encoded'])['draw'].cumsum() - cumsum_away_df['draw']   
            cumsum_away_df.loc[:, 'away_count'] = cumsum_away_df.groupby(['away_encoded', 'season_encoded', 'league_encoded'])['fixture_id'].cumcount()
            cumsum_away_df.loc[:, 'red_cards_cumsum'] = cumsum_away_df.groupby(['away_encoded', 'season_encoded', 'league_encoded'])['away_red_cards'].cumsum() - cumsum_away_df['away_red_cards']
            cumsum_away_df.loc[:, 'yellow_cards_cumsum'] = cumsum_away_df.groupby(['away_encoded', 'season_encoded', 'league_encoded'])['away_yellow_cards'].cumsum() - cumsum_away_df['away_yellow_cards']
            
            cumsum_home_df = cumsum_home_df.sort_values(by=['fixture_id'])
            cumsum_away_df = cumsum_away_df.sort_values(by=['fixture_id'])
            print(f"Cumsum home: {len(cumsum_home_df)}")
            print(f"Cumsum away: {len(cumsum_away_df)}")
            
            path_cumsum_away = 'data/Create_data/data_files/base/cumsum_away_last_match_features.xlsx'
            path_cumsum_home = 'data/Create_data/data_files/base/cumsum_home_last_match_features.xlsx'
            cumsum_away_df.to_excel(path_cumsum_away, index=False)
            print(f"Cumulative sums away exported to Excel: {path_cumsum_away}")
            cumsum_home_df.to_excel(path_cumsum_home, index=False)
            print(f"Cumulative sums home exported to Excel: {path_cumsum_home}")

            dataframe['home_cumcount'] = dataframe.groupby(['home_encoded', 'season_encoded', 'league_encoded'])['fixture_id'].cumcount()
            dataframe['away_cumcount'] = dataframe.groupby(['away_encoded', 'season_encoded', 'league_encoded'])['fixture_id'].cumcount()
            print("Cumulative sums added")
            return dataframe
        except Exception as e:
            print(f"Error in add_cumulative_sums: {e}")
            if self.logger:
                self.logger.error(f"Error in add_cumulative_sums: {e}")
            return dataframe
    
    def add_rolling_averages(self, dataframe):
        try:
            rolling_columns = ['fixture_id','year','week_of_year','home_encoded','away_encoded', 'season_encoded', 'league_encoded','home_saves','away_saves',
                                'home_shots_on_goal','away_shots_on_goal','home_passes_accuracy','away_passes_accuracy','home_fouls','away_fouls'
                                ,'home_points','away_points','home_goal_difference','away_goal_difference','home_win','away_win','draw','home_red_cards','away_red_cards','home_yellow_cards','away_yellow_cards']
            
            dataframe = dataframe.replace([np.inf, -np.inf], 0)
            
            rolling_home_df = dataframe[rolling_columns].sort_values(by=['home_encoded', 'season_encoded', 'league_encoded', 'year', 'week_of_year'])
            rolling_away_df = dataframe[rolling_columns].sort_values(by=['away_encoded', 'season_encoded', 'league_encoded', 'year', 'week_of_year'])
            rolling_home_df = rolling_home_df.copy()
            rolling_away_df = rolling_away_df.copy()

            # Rolling sum for home dataframe
            home_df = rolling_home_df.shift(1).groupby(['home_encoded', 'season_encoded', 'league_encoded','year','week_of_year']) \
                        .rolling(window=4, min_periods=1, closed='left').sum().reset_index()

            # Rolling sum for away dataframe 
            away_df = rolling_away_df.shift(1).groupby(['away_encoded', 'season_encoded', 'league_encoded','year','week_of_year']) \
                        .rolling(window=4, min_periods=1, closed='left').sum().reset_index()

            print('Rolling dataframeframes ready, calculating values...')
            
            # Calculate rolling values 
            print('Rolling values: home_points_rolling')
            dataframe['home_points_rolling'] = dataframe.apply(lambda row: 
                self.calculate_rolling_values(row['home_encoded'], home_df,away_df,
                                        row['league_encoded'],row['season_encoded'],
                                        row['year'],row['week_of_year'],
                                        'home_points','away_points'), axis=1).ffill()
            print('Rolling values: home_saves_rolling')
            dataframe['home_saves_rolling'] = dataframe.apply(lambda row: 
                self.calculate_rolling_values(row['home_encoded'], home_df,away_df,
                                        row['league_encoded'],row['season_encoded'],
                                        row['year'],row['week_of_year'],
                                        'home_saves','away_saves'), axis=1).ffill()
            print('Rolling values: home_shots_on_target_rolling')
            dataframe['home_shots_on_target_rolling'] = dataframe.apply(lambda row: 
                self.calculate_rolling_values(row['home_encoded'], home_df,away_df,
                                        row['league_encoded'],row['season_encoded'],
                                        row['year'],row['week_of_year'],
                                        'home_shots_on_goal','away_shots_on_goal'), axis=1).ffill()
            print('Rolling values: home_passes_accuracy_rolling')
            dataframe['home_passes_accuracy_rolling'] = dataframe.apply(lambda row: 
                self.calculate_rolling_values(row['home_encoded'], home_df,away_df,
                                        row['league_encoded'],row['season_encoded'],
                                        row['year'],row['week_of_year'],
                                        'home_passes_accuracy','away_passes_accuracy'), axis=1).ffill()
            print('Rolling values: home_fouls_rolling')
            dataframe['home_fouls_rolling'] = dataframe.apply(lambda row: 
                self.calculate_rolling_values(row['home_encoded'], home_df,away_df,
                                        row['league_encoded'],row['season_encoded'],
                                        row['year'],row['week_of_year'],
                                        'home_fouls','away_fouls'), axis=1).ffill()
            print('Rolling values: home_goal_difference_rolling')
            dataframe['home_goal_difference_rolling'] = dataframe.apply(lambda row: 
                self.calculate_rolling_values(row['home_encoded'], home_df,away_df,
                                        row['league_encoded'],row['season_encoded'],
                                        row['year'],row['week_of_year'],
                                        'home_goal_difference','away_goal_difference'), axis=1).ffill()
            print('Rolling values: home_win_rolling')
            dataframe['home_win_rolling'] = dataframe.apply(lambda row: 
                self.calculate_rolling_values(row['home_encoded'], home_df,away_df,
                                        row['league_encoded'],row['season_encoded'],
                                        row['year'],row['week_of_year'],
                                        'home_win','away_win'), axis=1).ffill()
            print('Rolling values: home_draw_rolling')
            dataframe['home_draw_rolling'] = dataframe.apply(lambda row: 
                self.calculate_rolling_values(row['home_encoded'], home_df,away_df,
                                        row['league_encoded'],row['season_encoded'],
                                        row['year'],row['week_of_year'],
                                        'draw','draw'), axis=1).ffill()
            print('Rolling values: home_red_cards_rolling')
            dataframe['home_red_cards_rolling'] = dataframe.apply(lambda row: 
                self.calculate_rolling_values(row['home_encoded'], home_df,away_df,
                                        row['league_encoded'],row['season_encoded'],
                                        row['year'],row['week_of_year'],
                                        'home_red_cards','away_red_cards'), axis=1).ffill()
            print('Rolling values: home_yellow_cards_rolling')
            dataframe['home_yellow_cards_rolling'] = dataframe.apply(lambda row: 
                self.calculate_rolling_values(row['home_encoded'], home_df,away_df,
                                        row['league_encoded'],row['season_encoded'],
                                        row['year'],row['week_of_year'],
                                        'home_yellow_cards','away_yellow_cards'), axis=1).ffill()
            print('Rolling values: away_points_rolling')
            dataframe['away_points_rolling'] = dataframe.apply(lambda row: 
                self.calculate_rolling_values(row['away_encoded'], home_df,away_df,
                                        row['league_encoded'],row['season_encoded'],
                                        row['year'],row['week_of_year'],
                                        'home_points','away_points'), axis=1).ffill()
            print('Rolling values: away_saves_rolling')
            dataframe['away_saves_rolling'] = dataframe.apply(lambda row: 
                self.calculate_rolling_values(row['away_encoded'], home_df,away_df,
                                        row['league_encoded'],row['season_encoded'],
                                        row['year'],row['week_of_year'],
                                        'home_saves','away_saves'), axis=1).ffill()
            print('Rolling values: away_shots_on_target_rolling')
            dataframe['away_shots_on_target_rolling'] = dataframe.apply(lambda row: 
                self.calculate_rolling_values(row['away_encoded'], home_df,away_df,
                                        row['league_encoded'],row['season_encoded'],
                                        row['year'],row['week_of_year'],
                                        'home_shots_on_goal','away_shots_on_goal'), axis=1).ffill()
            print('Rolling values: away_passes_accuracy_rolling')
            dataframe['away_passes_accuracy_rolling'] = dataframe.apply(lambda row: 
                self.calculate_rolling_values(row['away_encoded'], home_df,away_df,
                                        row['league_encoded'],row['season_encoded'],
                                        row['year'],row['week_of_year'],
                                        'home_passes_accuracy','away_passes_accuracy'), axis=1).ffill()
            print('Rolling values: away_fouls_rolling')
            dataframe['away_fouls_rolling'] = dataframe.apply(lambda row: 
                self.calculate_rolling_values(row['away_encoded'], home_df,away_df,
                                        row['league_encoded'],row['season_encoded'],
                                        row['year'],row['week_of_year'],
                                        'home_fouls','away_fouls'), axis=1).ffill()
            print('Rolling values: away_goal_difference_rolling')
            dataframe['away_goal_difference_rolling'] = dataframe.apply(lambda row: 
                self.calculate_rolling_values(row['away_encoded'], home_df,away_df,
                                        row['league_encoded'],row['season_encoded'],
                                        row['year'],row['week_of_year'],
                                        'home_goal_difference','away_goal_difference'), axis=1).ffill()
            print('Rolling values: away_win_rolling')
            dataframe['away_win_rolling'] = dataframe.apply(lambda row: 
                self.calculate_rolling_values(row['away_encoded'], home_df,away_df,
                                        row['league_encoded'],row['season_encoded'],
                                        row['year'],row['week_of_year'],
                                        'home_win','away_win'), axis=1).ffill()
            print('Rolling values: away_draw_rolling')
            dataframe['away_draw_rolling'] = dataframe.apply(lambda row: 
                self.calculate_rolling_values(row['away_encoded'], home_df,away_df,
                                        row['league_encoded'],row['season_encoded'],
                                        row['year'],row['week_of_year'],
                                        'draw','draw'), axis=1).ffill()
            print('Rolling values: away_red_cards_rolling')
            dataframe['away_red_cards_rolling'] = dataframe.apply(lambda row: 
                self.calculate_rolling_values(row['away_encoded'], home_df,away_df,
                                        row['league_encoded'],row['season_encoded'],
                                        row['year'],row['week_of_year'],
                                        'home_red_cards','away_red_cards'), axis=1).ffill()
            print('Rolling values: away_yellow_cards_rolling')
            dataframe['away_yellow_cards_rolling'] = dataframe.apply(lambda row: 
                self.calculate_rolling_values(row['away_encoded'], home_df,away_df,
                                        row['league_encoded'],row['season_encoded'],
                                        row['year'],row['week_of_year'],
                                        'home_yellow_cards','away_yellow_cards'), axis=1).ffill()
            
            print("Rolling averages added")
            return dataframe
        except Exception as e:
            print(f"Error in add_rolling_averages: {e}")
            if self.logger:
                self.logger.error(f"Error in add_rolling_averages: {e}")
            return dataframe
    
    # Function to calculate rolling team form
    def calculate_rolling_values(self, team, home_df, away_df, league, season, year, week, home_column, away_column):
        try:
        
            # Filter based on team, league, and season from original DataFrame (not the rolled result)
            team_home_results_new = home_df[
                (home_df['home_encoded'] == team) & 
                (home_df['league_encoded'] == league) & 
                (home_df['season_encoded'] == season) &
                (home_df['week_of_year'] == week) &
                (home_df['year'] == year)
            ][home_column].sum()

            team_away_results_new = away_df[
                (away_df['away_encoded'] == team) & 
                (away_df['league_encoded'] == league) & 
                (away_df['season_encoded'] == season) &
                (away_df['week_of_year'] == week) &
                (away_df['year'] == year)
            ][away_column].sum()

            if team_home_results_new != 0 and team_away_results_new != 0:
                # Calculate rolling value
                rolling_value = (team_home_results_new + team_away_results_new) / 2
            else:
                rolling_value = team_home_results_new + team_away_results_new
                
            return rolling_value

        except Exception as e:
            print(f"Error calculating rolling values(probably column not in rolling_colums list): {e}")
            return 0

def main():
    mongodb_features = MongoDBFeatures()
    print("Getting fixtures with stats")
    fixtures_with_stats = mongodb_features.get_fixtures_with_home_stats()
    print("Normalizing fixtures data")
    fixtures_dataframe = mongodb_features.normalize_fixtures_data(fixtures_with_stats)
    fixtures_dataframe_final = mongodb_features.add_features(fixtures_dataframe)
    print(f"Final dataframe shape: {fixtures_dataframe_final.shape}")

    export_path = 'data/Create_data/data_files/base/api_football_last_match_features.xlsx'
    mongodb_features.export_to_excel(fixtures_dataframe_final, export_path)
    print("Data exported to Excel")

if __name__ == "__main__":
    main()
