import logging
import os

import numpy as np
import pandas as pd
import python_calamine as calamine
from api_football.add_current_features_postgre import export_to_xlsx_fast


class ELOCalculator:
    """
    Enhanced ELO Calculator with automatic league-specific K-factors,
    allowing for prefixed values.
    """

    def __init__(self, logger=None):
        """Initialize the ELOCalculator with required paths and settings."""
        self.logger = logger or logging.getLogger(__name__)

        # Set directories and files
        self.model_dir = "./models/"
        os.makedirs(self.model_dir, exist_ok=True)

        # Define data paths
        # self.api_prediction_data_path = "./data_files/api_football_prediction_newPoisson.xlsx"
        self.api_prediction_data_path_new = (
            "./data_files/api_football_prediction_new_newPoisson.xlsx"
        )
        # self.api_training_data_path = "./data_files/api_football_training_newPoisson.xlsx"

        # Define export paths
        # self.api_prediction_export_path = "./data_files/api_football_prediction_newPoisson.xlsx"
        self.api_prediction_export_path_new = (
            "./data_files/api_football_prediction_new_newPoisson.xlsx"
        )
        # self.api_training_export_path = "./data_files/api_football_training_newPoisson.xlsx"

        # ELO settings
        self.INITIAL_ELO = 1500
        # Pre-defined K-factors for each league ID
        self.prefixed_k_factors = {
            2: 31.88,  # League 2
            39: 34.40,  # League 39
            40: 35.90,  # League 40
            41: 35.58,  # League 41
            61: 34.81,  # League 61
            62: 35.73,  # League 62
            71: 35.56,  # League 71
            72: 35.69,  # League 72
            78: 34.27,  # League 78
            79: 35.39,  # League 79
            88: 33.68,  # League 88
            89: 35.38,  # League 89
            94: 33.76,  # League 94
            98: 35.23,  # League 98
            103: 34.84,  # League 103
            106: 35.40,  # League 106
            113: 34.61,  # League 113
            119: 34.74,  # League 119
            128: 35.49,  # League 128
            135: 34.45,  # League 135
            136: 35.63,  # League 136
            140: 34.72,  # League 140
            141: 36.07,  # League 141
            169: 34.02,  # League 169
            172: 33.73,  # League 172
            179: 34.20,  # League 179
            188: 35.24,  # League 188
            203: 34.75,  # League 203
            207: 35.19,  # League 207
            210: 34.32,  # League 210
            218: 34.71,  # League 218
            239: 35.98,  # League 239
            244: 33.94,  # League 244
            253: 35.63,  # League 253
            262: 35.46,  # League 262
            271: 34.48,  # League 271
            283: 35.29,  # League 283
        }
        self.league_k_factors = {}  # Will be calculated or retrieved from prefixed
        self.elo_ratings = {}  # Dictionary to store ELO ratings
        self.current_season = None

    def calculate_league_k_factor(self, league_data):
        """
        Calculate K-factor for a specific league based on its competitiveness.
        Higher K-factor for more predictable leagues, lower for more volatile ones.
        """
        try:
            # Handle cases where team matches are zero to avoid division by zero
            league_data.loc[:, "Home_team_matches"] = league_data["Home_team_matches"].mask(
                league_data["Home_team_matches"] == 0, 1
            )
            league_data.loc[:, "Away_team_matches"] = league_data["Away_team_matches"].mask(
                league_data["Away_team_matches"] == 0, 1
            )

            # Calculate league competitiveness metrics
            win_rate_std = np.std(
                pd.concat([league_data["home_win_rate"], league_data["away_win_rate"]])
            )

            points_std = np.std(
                pd.concat([league_data["home_average_points"], league_data["away_average_points"]])
            )

            goal_diff_std = np.std(
                pd.concat(
                    [
                        league_data["Home_goal_difference_cum"],
                        league_data["Away_goal_difference_cum"],
                    ]
                )
                / pd.concat([league_data["Home_team_matches"], league_data["Away_team_matches"]])
            )

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
            # Calculate or retrieve K-factors for each league
            unique_leagues = matches["league_encoded"].unique()
            for league in unique_leagues:
                if league in self.prefixed_k_factors:
                    k_factor = self.prefixed_k_factors[league]
                    self.logger.info(f"Using prefixed K-factor for League {league}: {k_factor}")
                    self.league_k_factors[league] = k_factor
                else:
                    league_data = matches[matches["league_encoded"] == league]
                    k_factor = self.calculate_league_k_factor(league_data)
                    self.logger.info(f"Calculated K-factor for League {league}: {k_factor}")
                    self.league_k_factors[league] = k_factor

            # Initialize ELO ratings for all teams
            all_teams = pd.concat([matches["home_encoded"], matches["away_encoded"]]).unique()
            for team in all_teams:
                self.elo_ratings[team] = self.INITIAL_ELO
            # Add columns for ELO ratings
            matches["home_team_elo"] = 0.0
            matches["away_team_elo"] = 0.0
            elo_count = 0
            # Process matches chronologically
            for index, row in matches.sort_values("Date").iterrows():
                # Check for season change
                if row["season_encoded"] != self.current_season:
                    self.current_season = row["season_encoded"]
                    league_teams = matches[matches["league_encoded"] == row["league_encoded"]]
                    league_teams = pd.concat(
                        [league_teams["home_encoded"], league_teams["away_encoded"]]
                    ).unique()
                    self.reset_season_ratings(league_teams)
                home_team = row["home_encoded"]
                away_team = row["away_encoded"]
                league = row["league_encoded"]
                k_factor = self.league_k_factors.get(league, 30)
                # Get current ELO ratings
                home_elo = self.elo_ratings[home_team]
                away_elo = self.elo_ratings[away_team]
                # Store pre-match ELO ratings
                matches.at[index, "home_team_elo"] = home_elo
                matches.at[index, "away_team_elo"] = away_elo

                # Update ratings if we have actual results
                if "home_goals" in row and "away_goals" in row:
                    home_goals = row["home_goals"]
                    away_goals = row["away_goals"]
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
                "home_win_rate",
                "away_win_rate",
                "home_average_points",
                "away_average_points",
                "Home_goal_difference_cum",
                "Away_goal_difference_cum",
                "Home_team_matches",
                "Away_team_matches",
                "home_goals",
                "away_goals",
                "home_poisson_xG",
                "away_poisson_xG",
                "home_attack_strength",
                "away_attack_strength",
                "home_defense_weakness",
                "away_defense_weakness",
                "home_goal_rollingaverage",
                "away_goal_rollingaverage",
                "home_saves_rollingaverage",
                "away_saves_rollingaverage",
                "Home_possession_mean",
                "away_possession_mean",
                "Home_passes_mean",
                "Away_passes_mean",
            ]

            def convert_numeric_columns(df):
                """Convert numeric columns from string with commas to float."""
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = df[col].apply(
                            lambda x: float(str(x).replace(",", "."))
                            if isinstance(x, str)
                            else float(x)
                        )
                return df

            # # Process API data
            # self.logger.info("Processing API prediction data...")
            # api_prediction_data = load_excel_with_calamine(self.api_prediction_data_path)
            # api_prediction_copy = api_prediction_data.copy()
            # api_prediction_data = convert_numeric_columns(api_prediction_data)
            # api_prediction_data = api_prediction_data.sort_values("Date")
            # api_prediction_data = self.add_elo_scores(api_prediction_data)
            # api_prediction_copy = api_prediction_copy.merge(
            #     api_prediction_data[["fixture_id", "home_team_elo", "away_team_elo"]],
            #     on="fixture_id",
            #     how="left",
            # )
            # self.save_data_to_excel(
            #     api_prediction_copy, self.api_prediction_export_path, "API prediction data"
            # )
            # self.logger.info("API prediction data processed and saved")

            # Process API new data
            self.logger.info("Processing API prediction data...")
            api_prediction_data_new = load_excel_with_calamine(self.api_prediction_data_path_new)
            api_prediction_copy_new = api_prediction_data_new.copy()
            api_prediction_data_new = convert_numeric_columns(api_prediction_data_new)
            api_prediction_data_new = api_prediction_data_new.sort_values("Date")
            api_prediction_data_new = self.add_elo_scores(api_prediction_data_new)
            api_prediction_copy_new = api_prediction_copy_new.merge(
                api_prediction_data_new[["fixture_id", "home_team_elo", "away_team_elo"]],
                on="fixture_id",
                how="left",
            )
            self.save_data_to_excel(
                api_prediction_copy_new, self.api_prediction_export_path_new, "API prediction data"
            )
            self.logger.info("API prediction data processed and saved")

        except Exception as e:
            self.logger.error(f"Error in process_data: {str(e)}")
            raise

    def save_data_to_excel(self, df, output_path, type):
        """
        Save DataFrame to Excel using a memory-efficient approach.

        Args:
            df: DataFrame to save
            output_path: Path to save the Excel file
            type: Type of data being saved (for logging purposes)

        Returns:
            The original DataFrame
        """
        try:
            export_to_xlsx_fast(df, output_path)
            self.logger.info(f"Successfully exported {len(df)} rows to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to export {type} data: {str(e)}")
            # Try alternative format if Excel export fails
            if output_path.endswith(".xlsx"):
                alt_path = output_path.replace(".xlsx", ".csv")
                df.to_csv(alt_path, index=False)
                self.logger.info(f"Exported {type} data to alternative format: {alt_path}")

        return df


def load_excel_with_calamine(file_path, logger=None):
    """Load Excel file using calamine for improved performance with version 0.3.1."""
    if logger:
        logger.info(f"Loading data from {file_path} using calamine")
    try:
        # Load workbook with calamine
        workbook = calamine.load_workbook(file_path)
        # Get the first sheet (sheet_index=0)
        sheet_name = workbook.sheet_names[0]
        sheet = workbook.get_sheet_by_name(sheet_name)
        # Get sheet dimensions and data using correct calamine method
        rows = sheet.to_python()
        if not rows:
            return pd.DataFrame()

        # First row contains headers
        headers = rows[0]
        # Convert data to a list of dictionaries
        data = []
        for row in rows[1:]:
            # Make sure row is the same length as headers
            row_data = (
                row + [None] * (len(headers) - len(row))
                if len(row) < len(headers)
                else row[: len(headers)]
            )
            data.append(dict(zip(headers, row_data)))
        # Create DataFrame
        df = pd.DataFrame(data)
        # Replace NA values
        na_values = ["NaN", "N/A", "NA", "null", "None", "", "Infinity", "-Infinity", "inf", "-inf"]
        df = df.replace(na_values, np.nan)
        # Replace infinities
        df = df.replace([np.inf, -np.inf], np.nan)
        # Close workbook to release resources
        workbook.close()
        return df
    except Exception as e:
        if logger:
            logger.error(f"Error loading Excel file with calamine: {str(e)}")
        # Fallback to pandas if needed
        if logger:
            logger.info(f"Falling back to pandas for file: {file_path}")
        try:
            return pd.read_excel(file_path)
        except Exception as fallback_e:
            if logger:
                logger.error(f"Fallback to pandas also failed: {str(fallback_e)}")
            raise


def main():
    """Main execution function."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("./log/elo_calculator.log"), logging.StreamHandler()],
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
