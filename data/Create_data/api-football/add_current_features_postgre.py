import os
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import psycopg2
import requests
from dotenv import load_dotenv
from openpyxl import Workbook
from pandas import json_normalize
from psycopg2.extras import DictCursor
from sklearn.preprocessing import LabelEncoder

try:
    # Set the configuration key
    pd.set_option("future.no_silent_downcasting", True)
except KeyError as e:
    print(f"Configuration key not found: {e}")

load_dotenv()

class PostgreSQLFeatures:
    """
    A class to interact with PostgreSQL and retrieve fixtures data.
    """

    def __init__(self, logger=None):
        self.logger = logger
        # PostgreSQL connection parameters from environment variables
        self.db_host = os.getenv("POSTGRES_HOST")
        self.db_name = os.getenv("POSTGRES_DB")
        self.db_user = os.getenv("POSTGRES_USER")
        self.db_password = os.getenv("POSTGRES_PASSWORD")
        self.db_port = os.getenv("POSTGRES_PORT", "5432") # Default port for PostgreSQL                       

        self.conn = None
        self._connect_db()

        # Placeholder for other collections/tables until schemas are provided
        self.predictions_table_name = "api_football.predictions" 
        self.venues_collection = "api_football.venues" 
        self.team_stats_collection = "api_football.team_stats" 

    def _connect_db(self):
        """Establishes a connection to the PostgreSQL database."""
        if self.conn is None or self.conn.closed:
            try:
                self.conn = psycopg2.connect(
                    host=self.db_host,
                    database=self.db_name,
                    user=self.db_user,
                    password=self.db_password,
                    port=self.db_port
                )
                if self.logger:
                    self.logger.info("Successfully connected to PostgreSQL database.")
                print("Successfully connected to PostgreSQL database.")
            except psycopg2.Error as e:
                if self.logger:
                    self.logger.error(f"Error connecting to PostgreSQL database: {e}")
                print(f"Error connecting to PostgreSQL database: {e}")
                raise

    def _close_db(self):
        """Closes the database connection."""
        if self.conn and not self.conn.closed:
            self.conn.close()
            if self.logger:
                self.logger.info("PostgreSQL connection closed.")
            print("PostgreSQL connection closed.")

    def get_fixtures_with_home_stats(self) -> list[dict]:
        """
        Retrieves all fixtures from the PostgreSQL fixtures table.
        The condition "home.stats is not empty" from MongoDB needs to be translated.
        Assuming that if core stats columns are present and not null, it's equivalent.
        Also adding conditions for non-null scores as in the original query.
        The match_outcome column is missing in the provided DDL, this will need clarification.
        For now, I will assume it exists or can be derived.
        Returns:
            List[Dict]: List of fixtures.
        """
        self._connect_db()
        sql_query = """
        SELECT 
            fixture_id, referee, timezone, "date", "timestamp", 
            period_first, period_second, venue_id, venue_name, venue_city, 
            status_long, status_short, status_elapsed, status_extra, league_id, 
            league_name, league_country, league_logo, league_flag, league_season, 
            league_round, league_standings, home_team_id, home_team_name, home_team_logo, 
            home_team_winner as home_win, away_team_id, away_team_name, away_team_logo, away_team_winner as away_win, 
            home_goals, away_goals, home_halftime_goals, away_halftime_goals, 
            home_extratime_goals, away_extratime_goals, home_penalty_goals, 
            away_penalty_goals, home_shots_on_goal, home_shots_off_goal, home_total_shots, home_blocked_shots, 
            home_shots_insidebox, home_shots_outsidebox, home_fouls, home_corner_kicks, home_offsides, 
            home_ball_possession/100 as home_ball_possession_accuracy,
            home_yellow_cards, home_red_cards, home_goalkeeper_saves, home_total_passes, home_passes_accurate, 
            away_shots_on_goal, away_shots_off_goal, away_total_shots, away_blocked_shots, away_shots_insidebox, 
            away_shots_outsidebox, away_fouls, away_corner_kicks, away_offsides, 
            away_ball_possession/100 as away_ball_possession_accuracy, away_yellow_cards, 
            away_red_cards, away_goalkeeper_saves, away_total_passes, away_passes_accurate, 
            home_passes_percent/100 as home_passes_accuracy, away_passes_percent/100 as away_passes_accuracy
        FROM api_football.fixtures
        WHERE 
            home_goals IS NOT NULL and home_total_shots is not null;
        """

        df = pd.DataFrame()
        try:
            df = pd.read_sql_query(sql_query, self.conn)
            
            # Add match_outcome based on goals
            df['match_outcome'] = df.apply(lambda x: 
                '1' if x['home_goals'] > x['away_goals']
                else '3' if x['home_goals'] < x['away_goals'] 
                else '2', axis=1)
            
            # Rename passes_accurate columns to passes_accuracy
            df = df.rename(columns={
                'home_total_passes': 'home_passes',
                'away_total_passes': 'away_passes',
                'home_goalkeeper_saves': 'home_saves',
                'away_goalkeeper_saves': 'away_saves',
                'home_corner_kicks': 'home_corners',
                'away_corner_kicks': 'away_corners'
            })
            
            print(f"Found {len(df)} fixtures matching criteria from PostgreSQL.")
            if self.logger:
                self.logger.info(f"Found {len(df)} fixtures matching criteria from PostgreSQL.")
            
            return df
            
        except psycopg2.Error as e:
            print(f"Error executing query in get_fixtures_with_home_stats: {e}")
            if self.logger:
                self.logger.error(f"Error executing query in get_fixtures_with_home_stats: {e}")
            return []

    def get_fixtures_with_predictions(self) -> list[dict]:
        """
        Retrieves all predictions from the PostgreSQL api_football.predictions table.
        Returns:
            List[Dict]: List of prediction records.
        """
        self._connect_db()
        sql_query = """
        SELECT 
            fixture_id, winner_id, winner_name, win_or_draw, under_over, 
            goals_home, goals_away, advice, home_win_percent, draw_percent, 
            away_win_percent, updated_at, comparison_home_form, comparison_away_form, 
            comparison_home_att, comparison_away_att, comparison_home_def, comparison_away_def, 
            comparison_home_poisson_distribution, comparison_away_poisson_distribution, 
            comparison_home_h2h, comparison_away_h2h, comparison_home_goals, 
            comparison_away_goals, comparison_home_total, comparison_away_total, 
            h2h_home_wins, h2h_away_wins, h2h_draws, h2h_total_matches
        FROM api_football.predictions;
        """
        df = pd.DataFrame()
        try:
            df = pd.read_sql_query(sql_query, self.conn)
            print(f"Found {len(df)} predictions from PostgreSQL.")
            if self.logger:
                self.logger.info(f"Found {len(df)} predictions from PostgreSQL.")
        except (Exception, psycopg2.Error) as e:
            print(f"Error retrieving predictions from PostgreSQL: {e}")
            if self.logger:
                self.logger.error(f"Error retrieving predictions from PostgreSQL: {e}")
        return df

    def export_to_excel(self, fixtures_dataframe: pd.DataFrame, file_path: str) -> None:
        """
        Exports the DataFrame to an Excel file, handling data types and formatting.
        Args:
            fixtures_dataframe (pd.DataFrame): The DataFrame to export.
            file_path (str): The path to save the Excel file.
        """
        try:
            # Filter out rows where league_id is 235
            if "league_id" in fixtures_dataframe.columns:
                fixtures_dataframe = fixtures_dataframe[fixtures_dataframe["league_id"] != 235]
                print(
                    f"Filtered out {len(fixtures_dataframe[fixtures_dataframe['league_id'] == 235])} rows with league_id 235"
                )
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
            fixtures_dataframe = fixtures_dataframe.rename(columns={"date": "Date"})
            print("Features added")
            # Replace infinite values with NaN
            fixtures_dataframe = fixtures_dataframe.replace([np.inf, -np.inf], np.nan)
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
            data = data.dropna(subset=["match_outcome"])
            print("Data Collected, Start Cleaning and Feature Engineering...")
            if self.logger:
                self.logger.info("Data Collected, Start Cleaning and Feature Engineering...")
            # Drop irrelevant columns
            data = data.drop(columns=["prediction_outcome", "period_first", "period_second", "timestamp", "status_short", "status_elapsed", "status_extra", "status_elapsed", "league_logo", "league_flag", 
            "home_team_logo", "away_team_logo", "league_standings", "timezone", "model_prediction", "venue_city"], errors="ignore")

            # Type conversions and extracting date components
            data["home_advantage"] = 1
            data["Date"] = pd.to_datetime(data["date"], errors="coerce") 
            # Drop the 'date' column since we have 'Date'
            data = data.drop(columns=['date'], errors='ignore')
            data["year"] = data["Date"].dt.year
            data["month"] = data["Date"].dt.month
            data["day_of_month"] = data["Date"].dt.day
            data["day_of_week"] = data["Date"].dt.dayofweek
            data["week_of_year"] = data["Date"].dt.isocalendar().week
            
            # Ensure match_outcome is numeric before use
            if 'match_outcome' in data.columns:
                data["match_outcome"] = pd.to_numeric(data["match_outcome"], errors='coerce')

            # Label encoding for categorical variables
            le = LabelEncoder()
            data["venue_encoded"] = data["venue_id"]
            data["season_encoded"] = data["league_season"]
            data["league_encoded"] = data["league_id"]
            data["home_encoded"] = data["home_team_id"]
            data["away_encoded"] = data["away_team_id"]

            # First, sort the data based on the year, month, and day_of_month for each group.
            data = data.sort_values(by=["fixture_id"])

            shot_on_goal_weight = 0.20
            shot_inside_box_weight = 0.10 # Value for shots from dangerous areas
            corner_kick_weight = 0.03

            data['home_xG'] = (
                data.get('home_shots_on_goal', 0) * shot_on_goal_weight +
                data.get('home_shots_insidebox', 0) * shot_inside_box_weight +
                data.get('home_corner_kicks', 0) * corner_kick_weight
            )
            data['away_xG'] = (
                data.get('away_shots_on_goal', 0) * shot_on_goal_weight +
                data.get('away_shots_insidebox', 0) * shot_inside_box_weight +
                data.get('away_corner_kicks', 0) * corner_kick_weight
            )
            # Ensure xG is not negative (though unlikely with positive weights and counts)
            data['home_xG'] = data['home_xG'].clip(lower=0)
            data['away_xG'] = data['away_xG'].clip(lower=0)

            print("Start possession and shooting...")
            if self.logger:
                self.logger.info("Start possession and shooting...")
            # Feature engineering for possession and shooting
            data["total_fouls"] = data["home_fouls"] + data["away_fouls"]
            data["home_cards"] = data["home_yellow_cards"] + data["home_red_cards"]
            data["away_cards"] = data["away_yellow_cards"] + data["away_red_cards"]
            data["home_possession_shooting"] = (
                data["home_passes_accuracy"] * data["home_shots_on_goal"]
            )
            data["away_possession_shooting"] = (
                data["away_passes_accuracy"] * data["away_shots_on_goal"]
            )
            # Referee Stats
            if "referee" in data.columns:
                data["referee_encoded"] = le.fit_transform(data["referee"].astype(str))
            else:
                raise KeyError("Error: 'referee' column not found in DataFrame.")
            referee_stats = data.groupby("referee_encoded")["total_fouls"].mean()
            data["referee_foul_rate"] = data["referee_encoded"].map(referee_stats)
            # Additional feature engineering
            print("Base feature engineering done, start additional features...")
            if self.logger:
                self.logger.info("Base feature engineering done, start additional features...")
            data["home_possession_diff"] = (
                data["home_passes_accuracy"] - data["away_passes_accuracy"]
            )
            data["pass_accuracy_diff"] = data["home_passes"] - data["away_passes"]
            data["home_shooting_accuracy"] = data["home_shots_on_goal"] / data["home_total_shots"]
            data["away_shooting_accuracy"] = data["away_shots_on_goal"] / data["away_total_shots"]
            data["home_shots_on_target_ratio"] = (
                data["home_shots_on_goal"] / data["away_shots_on_goal"]
            )
            data["away_shots_on_target_ratio"] = (
                data["away_shots_on_goal"] / data["home_shots_on_goal"]
            )
            data["home_saves_accuracy"] = data["home_saves"] / data["away_shots_on_goal"]
            data["away_saves_accuracy"] = data["away_saves"] / data["home_shots_on_goal"]
            # Fill NaN values with 0 for defensive stats before calculating activity
            defensive_cols = ['home_blocked_shots', 'home_yellow_cards', 'home_red_cards', 'home_saves',
                            'away_blocked_shots', 'away_yellow_cards', 'away_red_cards', 'away_saves']
            data[defensive_cols] = data[defensive_cols].fillna(0)
            
            data["home_defensive_activity"] = (
                data["home_blocked_shots"].astype(float)
                + data["home_yellow_cards"].astype(float) 
                + data["home_red_cards"].astype(float)
                + data["home_saves"].astype(float)
            )
            data["away_defensive_activity"] = (
                data["away_blocked_shots"].astype(float)
                + data["away_yellow_cards"].astype(float)
                + data["away_red_cards"].astype(float) 
                + data["away_saves"].astype(float)
            )
            # Set-piece threat and foul impact
            data["home_set_piece_threat"] = data["home_corners"] + data["away_fouls"]
            data["away_set_piece_threat"] = data["away_corners"] + data["home_fouls"]
            # Outcome calculation
            print("Outcome calculation")
            if self.logger:
                self.logger.info("Outcome calculation")
            data["home_win"] = data["match_outcome"].apply(lambda x: 1 if x == 1 else 0)
            data["away_win"] = data["match_outcome"].apply(lambda x: 1 if x == 3 else 0)
            data["draw"] = data["match_outcome"].apply(lambda x: 1 if x == 2 else 0)
            # Points
            print("Points calculation")
            if self.logger:
                self.logger.info("Points calculation")
            data["home_points"] = (
                data["home_win"].apply(lambda x: 3 if x == 1 else 0) + data["draw"]
            )
            data["away_points"] = (
                data["away_win"].apply(lambda x: 3 if x == 1 else 0) + data["draw"]
            )

            data["home_goal_difference"] = data["home_goals"] - data["away_goals"]
            data["away_goal_difference"] = data["away_goals"] - data["home_goals"]
            # Rolling averages and cumulative sums
            print("Start calculating Cumulative values...")
            if self.logger:
                self.logger.info("Start calculating Cumulative values...")
            data["home_points_cumulative"] = data.groupby(
                ["home_encoded", "season_encoded", "league_encoded"]
            )["home_points"].cumsum()
            data["away_points_cumulative"] = data.groupby(
                ["away_encoded", "season_encoded", "league_encoded"]
            )["away_points"].cumsum()
            data["home_goal_diff_cumulative"] = data.groupby(
                ["home_encoded", "season_encoded", "league_encoded"]
            )["home_goal_difference"].cumsum()
            data["away_goal_diff_cumulative"] = data.groupby(
                ["away_encoded", "season_encoded", "league_encoded"]
            )["away_goal_difference"].cumsum()
            return data
        except Exception as e:
            print(f"Error in load_and_prepare_data: {e}")
            if self.logger:
                self.logger.error(f"Error in load_and_prepare_data: {e}")
            return pd.DataFrame()

    def get_future_matches(self) -> pd.DataFrame:
        """
        Retrieves all future fixtures (next 14 days, no scores) from the PostgreSQL 
        api_football.fixtures table and returns them as a pandas DataFrame.
        Args:
        Used to potentially merge some common columns.
        Returns:
            pd.DataFrame: DataFrame containing future fixtures.
        """
        self._connect_db()
        today = datetime.utcnow()
        two_weeks_date = today + timedelta(days=14)
        two_weeks_str = two_weeks_date.strftime("%Y-%m-%d %H:%M:%S")
        today_str = today.strftime("%Y-%m-%d %H:%M:%S")
        
        
        query_columns = [
            "fixture_id", "date", "league_id", "league_name", "league_season", 
            "referee", "venue_id", "venue_name", 
            "home_team_id", "home_team_name", "away_team_id", "away_team_name",
            "league_round" 
        ]
        select_cols_str = ", ".join([f'\"{col}\"' if col == "date" else col for col in query_columns])

        sql_query = f"""
        SELECT {select_cols_str}
        FROM api_football.fixtures
        WHERE "date" between '{today_str}' and '{two_weeks_str}' or home_total_shots is not null;
        """

        df_future = pd.DataFrame()
        try:
            df_future = pd.read_sql_query(sql_query, self.conn)
            print(f"Found {len(df_future)} future fixtures from PostgreSQL (next 14 days, no scores).")
            if self.logger:
                self.logger.info(f"Found {len(df_future)} future fixtures from PostgreSQL.")

            if df_future.empty:
                return pd.DataFrame() # Return empty if no future matches found

            df_future = df_future.rename(columns={"date": "Date"}) 

            # Ensure essential columns exist, fill with None or a sensible default if not from query
            expected_cols = [
                "fixture_id", "Date", "league_id", "league_season", "league_name", "referee",
                "venue_name", "venue_id", "home_team_id", "home_team_name", 
                "away_team_id", "away_team_name", "league_round"
            ]
            for col in expected_cols:
                if col not in df_future.columns:
                    df_future[col] = None # or np.nan or appropriate default
            
            # Ensure correct dtype for fixture_id if it was read as float from DB with NaNs (not typical for PK)
            if 'fixture_id' in df_future.columns:
                df_future["fixture_id"] = df_future["fixture_id"].astype(int)

        except (Exception, psycopg2.Error) as e:
            print(f"Error retrieving future matches: {e}")
            if self.logger:
                self.logger.error(f"Error retrieving future matches: {e}")
            # self._close_db() # Keep connection open
            return pd.DataFrame() # Return empty on error

        export_path = "data/Create_data/data_files/base/api_future_matches.xlsx"
        try:
            df_future.to_excel(export_path, index=False)
            print(f"Future matches exported to Excel: {export_path}")
        except Exception as e:
            print(f"Error exporting future matches to Excel: {e}")
            if self.logger:
                self.logger.error(f"Error exporting future matches to Excel: {e}")

        return df_future

    def export_venues(self) -> pd.DataFrame:
        """
        Exports venue data from PostgreSQL api_football.venues table to an Excel file.
        Returns:
            pd.DataFrame: DataFrame containing venue data.
        """
        self._connect_db()
        sql_query = """
        SELECT 
            team_id, team_name, team_code, team_country, team_founded, 
            team_national, team_logo, venue_id, venue_name, venue_address, 
            venue_city, venue_capacity, venue_surface, venue_image 
        FROM api_football.venues;
        """
        df = pd.DataFrame()
        try:
            df = pd.read_sql_query(sql_query, self.conn)
            print(f"Successfully retrieved {len(df)} venue records from api_football.venues.")
            if self.logger:
                self.logger.info(f"Successfully retrieved {len(df)} venue records from api_football.venues.")
            
            for col in df.columns:
                if df[col].isnull().any():
                    # print(f"Missing values found in column: {col}") # Optional: for debugging
                    if col.startswith("team_"):
                        df[col] = df[col].fillna("Unknown")
                    elif col == "venue_capacity":
                        df[col] = df[col].fillna(0)
                    elif col.startswith("venue_"):
                        df[col] = df[col].fillna("Unknown")
            
            export_path = "data/Create_data/data_files/base/api_venues.xlsx"
            df.to_excel(export_path, index=False)
            print(f"Normalized venues data exported to Excel: {export_path}")

        except (Exception, psycopg2.Error) as e:
            error_message = f"Error exporting venues data: {e}"
            print(error_message)
            if self.logger:
                self.logger.error(error_message)
            # self._close_db() # Keep connection open
        return df

    def flatten_team_stats(self) -> pd.DataFrame:
        """
        Retrieves team statistics from the PostgreSQL api_football.team_stats table.
        Returns:
            pd.DataFrame: DataFrame containing team statistics.
        """
        self._connect_db()
        sql_query = """
        SELECT 
            fixture_id, team_id, updated_at, league_id, league_name, league_country, 
            league_season, team_name, form, played_home, wins_home, draws_home, loses_home, 
            played_away, wins_away, draws_away, loses_away, played_total, wins_total, 
            draws_total, loses_total, goals_for_home, goals_for_avg_home, goals_for_away, 
            goals_for_avg_away, goals_for_total, goals_for_avg_total, goals_against_home, 
            goals_against_avg_home, goals_against_away, goals_against_avg_away, 
            goals_against_total, goals_against_avg_total, clean_sheet_home, 
            failed_to_score_home, clean_sheet_away, failed_to_score_away, 
            clean_sheet_total, failed_to_score_total, penalty_scored, penalty_missed, 
            penalty_total, streak_wins, streak_draws, streak_loses, biggest_wins_home, 
            biggest_wins_away, biggest_loses_home, biggest_loses_away, 
            biggest_goals_for_home, biggest_goals_for_away, biggest_goals_against_home, 
            biggest_goals_against_away 
        FROM api_football.team_stats;
        """
        df = pd.DataFrame()
        try:
            df = pd.read_sql_query(sql_query, self.conn)
            print(f"Successfully retrieved {len(df)} records from api_football.team_stats.")
            if self.logger:
                self.logger.info(f"Successfully retrieved {len(df)} records from api_football.team_stats.")
        except (Exception, psycopg2.Error) as e:
            error_message = f"Error retrieving data from api_football.team_stats: {e}"
            print(error_message)
            if self.logger:
                self.logger.error(error_message)
        # self._close_db() # Keep connection open if other methods will use it
        return df

def main():
    postgresql_features = PostgreSQLFeatures()
    print("Getting fixtures with stats")

    fixtures_dataframe = postgresql_features.get_fixtures_with_home_stats()
    print("Normalizing fixtures data")
    fixtures_dataframe_final = postgresql_features.add_features(fixtures_dataframe)
    print(f"Final dataframe shape: {fixtures_dataframe_final.shape}")

    export_path = "data/Create_data/data_files/base/api_football_current_features.xlsx"
    postgresql_features.export_to_excel(fixtures_dataframe_final, export_path)
    print("Data exported to Excel")

    print("Getting future matches")
    future_matches = postgresql_features.get_future_matches()
    print(f"Future matches shape: {future_matches.shape}")

    print("Exporting predictions")
    fixtures_with_predictions = postgresql_features.get_fixtures_with_predictions()
    export_path = "data/Create_data/data_files/base/predictions.xlsx"
    postgresql_features.export_to_excel(fixtures_with_predictions, export_path)

    print("Exporting venues")
    venues = postgresql_features.export_venues()
    print(f"Venues shape: {venues.shape}")

    print("Exporting team stats")
    team_stats = postgresql_features.flatten_team_stats()
    team_stats.to_excel("data/Create_data/data_files/base/api_team_stats.xlsx", index=False)
    print(f"Team stats shape: {team_stats.shape}")


if __name__ == "__main__":
    main()
