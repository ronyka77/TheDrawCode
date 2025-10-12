import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import psycopg2
import requests
from dotenv import load_dotenv
from sqlalchemy import MetaData, Table, create_engine, delete, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError

load_dotenv()
# Add project root to Python path
try:
    project_root = Path(__file__).parent.parent.parent.parent
    if not project_root.exists():
        # Handle network path by using raw string
        project_root = Path(r"\\".join(str(project_root).split("\\")))
    sys.path.append(str(project_root))
    print(f"Project root get_fixtures: {project_root}")
except Exception as e:
    print(f"Error setting project root path: {e}")
    # Fallback to current directory if path resolution fails
    sys.path.append(os.getcwd().parent.parent.parent)
    print(f"Current directory get_fixtures: {os.getcwd().parent.parent.parent}")

from src.utils.logger import ExperimentLogger

# PostgreSQL engine setup
engine = create_engine("postgresql+psycopg2://postgres:ronaldo99@localhost:5432/api_football")
metadata = MetaData()

# Reflect the fixtures table
fixtures_table = Table("fixtures", metadata, autoload_with=engine, schema="api_football")
predictions_table = Table("predictions", metadata, autoload_with=engine, schema="api_football")
team_stats_table = Table("team_stats", metadata, autoload_with=engine, schema="api_football")
fixture_events_table = Table("events", metadata, autoload_with=engine, schema="api_football")


class ApiFootball:
    """
    A class to interact with the API-Football API and store data in PostgreSQL.
    """

    def __init__(self, api_key: str, logger: ExperimentLogger = None):
        self.api_key = api_key
        self.logger = logger or ExperimentLogger()
        self.base_url = "https://v3.football.api-sports.io/"
        self.headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": "v3.football.api-sports.io",
        }
        self.project_root = project_root
        self.data_dir = os.path.join(self.project_root, "data", "create_data", "api-football")
        os.makedirs(self.data_dir, exist_ok=True)

    def _get_request(self, endpoint: str, params: dict = None) -> dict:
        """
        Sends a GET request to the specified endpoint.
        Args:
            endpoint (str): The API endpoint to request.
            params (Dict, optional): Query parameters for the request. Defaults to None.
        Returns:
            Dict: The JSON response from the API.
        """
        url = self.base_url + endpoint
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error during API request to {url}: {e}")
            return {}

    def _save_json(self, data: Any, filename: str) -> None:
        """
        Saves data to a JSON file in the data directory.
        Args:
            data (Any): The data to save.
            filename (str): The name of the file to save to.
        """
        file_path = os.path.join(self.data_dir, filename)
        try:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=4)
            self.logger.info(f"Data saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving data to {file_path}: {e}")

    def _map_fixture_to_row(self, fixture: dict) -> dict:
        # Map/flatten the fixture dict to match the fixtures table columns
        def safe_get(d, *keys, default=None):
            for k in keys:
                if d is None:
                    return default
                d = d.get(k, default)
            return d

        row = {
            "fixture_id": safe_get(fixture, "fixture", "id"),
            "referee": safe_get(fixture, "fixture", "referee"),
            "timezone": safe_get(fixture, "fixture", "timezone"),
            "date": safe_get(fixture, "fixture", "date"),
            "timestamp": safe_get(fixture, "fixture", "timestamp"),
            "period_first": safe_get(fixture, "fixture", "periods", "first"),
            "period_second": safe_get(fixture, "fixture", "periods", "second"),
            "venue_id": safe_get(fixture, "fixture", "venue", "id"),
            "venue_name": safe_get(fixture, "fixture", "venue", "name"),
            "venue_city": safe_get(fixture, "fixture", "venue", "city"),
            "status_long": safe_get(fixture, "fixture", "status", "long"),
            "status_short": safe_get(fixture, "fixture", "status", "short"),
            "status_elapsed": safe_get(fixture, "fixture", "status", "elapsed"),
            "status_extra": safe_get(fixture, "fixture", "status", "extra"),
            "league_id": safe_get(fixture, "league", "id"),
            "league_name": safe_get(fixture, "league", "name"),
            "league_country": safe_get(fixture, "league", "country"),
            "league_logo": safe_get(fixture, "league", "logo"),
            "league_flag": safe_get(fixture, "league", "flag"),
            "league_season": safe_get(fixture, "league", "season"),
            "league_round": safe_get(fixture, "league", "round"),
            "league_standings": safe_get(fixture, "league", "standings"),
            "home_team_id": safe_get(fixture, "teams", "home", "id"),
            "home_team_name": safe_get(fixture, "teams", "home", "name"),
            "home_team_logo": safe_get(fixture, "teams", "home", "logo"),
            "home_team_winner": safe_get(fixture, "teams", "home", "winner"),
            "away_team_id": safe_get(fixture, "teams", "away", "id"),
            "away_team_name": safe_get(fixture, "teams", "away", "name"),
            "away_team_logo": safe_get(fixture, "teams", "away", "logo"),
            "away_team_winner": safe_get(fixture, "teams", "away", "winner"),
            "home_goals": safe_get(fixture, "goals", "home"),
            "away_goals": safe_get(fixture, "goals", "away"),
            "home_halftime_goals": safe_get(fixture, "score", "halftime", "home"),
            "away_halftime_goals": safe_get(fixture, "score", "halftime", "away"),
            "home_fulltime_goals": safe_get(fixture, "score", "fulltime", "home"),
            "away_fulltime_goals": safe_get(fixture, "score", "fulltime", "away"),
            "home_extratime_goals": safe_get(fixture, "score", "extratime", "home"),
            "away_extratime_goals": safe_get(fixture, "score", "extratime", "away"),
            "home_penalty_goals": safe_get(fixture, "score", "penalty", "home"),
            "away_penalty_goals": safe_get(fixture, "score", "penalty", "away"),
        }
        return row

    def upsert_fixture(self, row: dict):
        # Upsert a single fixture row into PostgreSQL
        stmt = pg_insert(fixtures_table).values(**row)
        update_dict = {c: stmt.excluded[c] for c in row.keys() if c != "fixture_id"}
        stmt = stmt.on_conflict_do_update(index_elements=["fixture_id"], set_=update_dict)
        try:
            with engine.begin() as conn:
                conn.execute(stmt)
            # self.logger.info(f"Upserted fixture {row.get('fixture_id')}")
        except SQLAlchemyError as e:
            self.logger.error(f"Error upserting fixture {row.get('fixture_id')}: {e}")

    def get_fixtures(self, league_id: int, season: int) -> int:
        endpoint = "fixtures"
        params = {"league": league_id, "season": season}
        response = self._get_request(endpoint, params)
        if response and "response" in response:
            fixtures = response["response"]
            for fixture in fixtures:
                # print(fixture)
                row = self._map_fixture_to_row(fixture)
                self.upsert_fixture(row)

            self.logger.info(
                f"Upserted {len(fixtures)} fixtures for league {league_id} season {season}"
            )
            return len(fixtures)
        else:
            self.logger.warning(f"No fixtures found for league {league_id} season {season}")
            return 0

    def get_fixtures_for_leagues(self) -> None:
        league_ids_file_path = os.path.join(
            project_root, "data", "create_data", "api_football", "league_ids.json"
        )
        seasons = [2025]
        try:
            with open(league_ids_file_path) as f:
                league_ids_data = json.load(f)
            num_leagues = len(league_ids_data)
            self.logger.info(f"Total number of leagues: {num_leagues}")
            for league_info in league_ids_data:
                league_id = league_info["league_id"]
                self.logger.info(f"Getting fixtures for league ID: {league_id}")
                for season in seasons:
                    fixtures = self.get_fixtures(league_id, season)
                    if fixtures:
                        self.logger.info(
                            f"Retrieved {fixtures} fixtures for league ID: {league_id} season: {season}"
                        )
                    else:
                        self.logger.warning(
                            f"No fixtures found for league ID: {league_id} season: {season}"
                        )
                        fixtures = self.get_fixtures(league_id, 2024)
                        self.logger.info(
                            f"Retrieved {fixtures} fixtures for league ID: {league_id} season: 2024"
                        )
        except Exception as e:
            self.logger.error(f"Error processing league IDs: {e}")

    def _parse_api_numeric(self, value: Any) -> Any:
        """Helper to parse numeric values that might be strings, ints, floats, or None."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return value
        try:
            # Attempt to remove % if it's a percentage string before float conversion
            str_value = str(value).strip()
            if str_value.endswith("%"):
                return float(str_value[:-1])
            return float(str_value)
        except (ValueError, TypeError):
            return None

    def _parse_team_stat_response(
        self, api_response_data: dict, fixture_id: int, target_team_id: Any
    ) -> dict:
        """Parses the teams/statistics API response (a dictionary) for a specific team's season stats,
        and filters the output to match the known columns in api_football.team_stats table."""
        known_team_stats_columns = [
            "fixture_id",
            "team_id",
            "updated_at",
            "league_id",
            "league_name",
            "league_country",
            "league_season",
            "team_name",
            "form",
            "played_home",
            "wins_home",
            "draws_home",
            "loses_home",
            "played_away",
            "wins_away",
            "draws_away",
            "loses_away",
            "played_total",
            "wins_total",
            "draws_total",
            "loses_total",
            "goals_for_home",
            "goals_for_avg_home",
            "goals_for_away",
            "goals_for_avg_away",
            "goals_for_total",
            "goals_for_avg_total",
            "goals_against_home",
            "goals_against_avg_home",
            "goals_against_away",
            "goals_against_avg_away",
            "goals_against_total",
            "goals_against_avg_total",
            "clean_sheet_home",
            "failed_to_score_home",
            "clean_sheet_away",
            "failed_to_score_away",
            "clean_sheet_total",
            "failed_to_score_total",
            "penalty_scored",
            "penalty_missed",
            "penalty_total",
            "streak_wins",
            "streak_draws",
            "streak_loses",
            "biggest_wins_home",
            "biggest_wins_away",
            "biggest_loses_home",
            "biggest_loses_away",
            "biggest_goals_for_home",
            "biggest_goals_for_away",
            "biggest_goals_against_home",
            "biggest_goals_against_away",
        ]

        if not api_response_data or not isinstance(api_response_data, dict):
            self.logger.warning(
                f"_parse_team_stat_response: Invalid api_response_data for team {target_team_id}, fixture_id context {fixture_id}"
            )
            return None

        # Ensure target_team_id is an int
        try:
            target_team_id_int = int(
                target_team_id[0] if isinstance(target_team_id, tuple) else target_team_id
            )
        except (ValueError, TypeError):
            self.logger.error(
                f"_parse_team_stat_response: Invalid target_team_id type. Got {target_team_id}"
            )
            return None

        # Extract main sections from API response
        league_api = api_response_data.get("league", {})
        team_api = api_response_data.get("team", {})
        fixtures_api = api_response_data.get("fixtures", {})
        goals_api = api_response_data.get("goals", {})
        biggest_api = api_response_data.get("biggest", {})
        penalty_api = api_response_data.get("penalty", {})
        clean_sheet_api = api_response_data.get("clean_sheet", {})
        failed_to_score_api = api_response_data.get("failed_to_score", {})

        if not team_api.get("id") == target_team_id_int:
            self.logger.warning(
                f"API response team ID {team_api.get('id')} does not match target_team_id {target_team_id_int}"
            )

        parsed_data = {
            "fixture_id": fixture_id,  # Contextual: links this season's stats to a fixture for fetching trigger
            "team_id": team_api.get("id", target_team_id_int),  # Prefer API's team_id if available
            "league_id": league_api.get("id"),
            "league_name": league_api.get("name"),
            "league_country": league_api.get("country"),
            "league_season": league_api.get("season"),  # This is the season of the stats
            "team_name": team_api.get("name"),
            "form": api_response_data.get("form"),
            "played_home": fixtures_api.get("played", {}).get("home"),
            "played_away": fixtures_api.get("played", {}).get("away"),
            "played_total": fixtures_api.get("played", {}).get("total"),
            "wins_home": fixtures_api.get("wins", {}).get("home"),
            "wins_away": fixtures_api.get("wins", {}).get("away"),
            "wins_total": fixtures_api.get("wins", {}).get("total"),
            "draws_home": fixtures_api.get("draws", {}).get("home"),
            "draws_away": fixtures_api.get("draws", {}).get("away"),
            "draws_total": fixtures_api.get("draws", {}).get("total"),
            "loses_home": fixtures_api.get("loses", {}).get("home"),
            "loses_away": fixtures_api.get("loses", {}).get("away"),
            "loses_total": fixtures_api.get("loses", {}).get("total"),
            "goals_for_home": goals_api.get("for", {}).get("total", {}).get("home"),
            "goals_for_avg_home": self._parse_api_numeric(
                goals_api.get("for", {}).get("average", {}).get("home")
            ),
            "goals_for_away": goals_api.get("for", {}).get("total", {}).get("away"),
            "goals_for_avg_away": self._parse_api_numeric(
                goals_api.get("for", {}).get("average", {}).get("away")
            ),
            "goals_for_total": goals_api.get("for", {}).get("total", {}).get("total"),
            "goals_for_avg_total": self._parse_api_numeric(
                goals_api.get("for", {}).get("average", {}).get("total")
            ),
            "goals_against_home": goals_api.get("against", {}).get("total", {}).get("home"),
            "goals_against_avg_home": self._parse_api_numeric(
                goals_api.get("against", {}).get("average", {}).get("home")
            ),
            "goals_against_away": goals_api.get("against", {}).get("total", {}).get("away"),
            "goals_against_avg_away": self._parse_api_numeric(
                goals_api.get("against", {}).get("average", {}).get("away")
            ),
            "goals_against_total": goals_api.get("against", {}).get("total", {}).get("total"),
            "goals_against_avg_total": self._parse_api_numeric(
                goals_api.get("against", {}).get("average", {}).get("total")
            ),
            "clean_sheet_home": clean_sheet_api.get("home"),
            "clean_sheet_away": clean_sheet_api.get("away"),
            "clean_sheet_total": clean_sheet_api.get("total"),
            "failed_to_score_home": failed_to_score_api.get("home"),
            "failed_to_score_away": failed_to_score_api.get("away"),
            "failed_to_score_total": failed_to_score_api.get("total"),
            "penalty_scored": penalty_api.get("scored", {}).get("total"),
            "penalty_missed": penalty_api.get("missed", {}).get("total"),
            "penalty_total": penalty_api.get("total"),
            "streak_wins": biggest_api.get("streak", {}).get("wins"),
            "streak_draws": biggest_api.get("streak", {}).get("draws"),
            "streak_loses": biggest_api.get("streak", {}).get("loses"),
            "biggest_wins_home": biggest_api.get("wins", {}).get("home"),
            "biggest_wins_away": biggest_api.get("wins", {}).get("away"),
            "biggest_loses_home": biggest_api.get("loses", {}).get("home"),
            "biggest_loses_away": biggest_api.get("loses", {}).get("away"),
            "biggest_goals_for_home": biggest_api.get("goals", {}).get("for", {}).get("home"),
            "biggest_goals_for_away": biggest_api.get("goals", {}).get("for", {}).get("away"),
            "biggest_goals_against_home": biggest_api.get("goals", {})
            .get("against", {})
            .get("home"),
            "biggest_goals_against_away": biggest_api.get("goals", {})
            .get("against", {})
            .get("away"),
            "updated_at": datetime.now(),
        }

        # Filter to only include keys that exist in the known_team_stats_columns list and are not None
        filtered_data = {
            k: v for k, v in parsed_data.items() if k in known_team_stats_columns and v is not None
        }

        if (
            not filtered_data.get("team_id")
            or not filtered_data.get("league_id")
            or not filtered_data.get("league_season")
        ):
            self.logger.warning(
                f"Filtered data for team {target_team_id_int} (fixture context {fixture_id}) is missing essential keys (team_id, league_id, league_season) after filtering."
            )
            return None

        return filtered_data

    def get_team_stats_for_fixtures(self):
        """Fetches season-aggregated statistics for each team linked to unprocessed fixtures and stores them in PostgreSQL."""
        try:
            self.logger.info("Starting to get season-aggregated team statistics...")

            query = text("""
                SELECT f.fixture_id, f.league_id, f.home_team_id, f.away_team_id, f.league_season, f.date
                FROM api_football.fixtures f
                left join api_football.team_stats ts on f.fixture_id = ts.fixture_id 
                WHERE date <= now() and ts.fixture_id  is null;
            """)

            fixtures_to_query_teams_for = []
            with engine.connect() as conn:
                result = conn.execute(query)
                fixtures_to_query_teams_for = result.fetchall()

            if not fixtures_to_query_teams_for:
                self.logger.info("No fixtures found to trigger team statistics processing.")
                return

            self.logger.info(
                f"Found {len(fixtures_to_query_teams_for)} fixtures to check for team statistics updates."
            )

            api_call_count_total = 0
            api_calls_in_current_batch = 0
            rate_limit_threshold_per_batch = 250
            pause_duration = 60
            batch_start_time = time.time()

            for i, fixture_row in enumerate(fixtures_to_query_teams_for):
                fixture_id = fixture_row.fixture_id
                league_id = fixture_row.league_id
                home_team_id = fixture_row.home_team_id
                away_team_id = fixture_row.away_team_id
                season = fixture_row.league_season
                date = fixture_row.date.strftime("%Y-%m-%d")

                if not all([fixture_id, league_id, home_team_id, away_team_id, season]):
                    self.logger.warning(
                        f"Skipping fixture {fixture_id} due to missing key DB information for team stat processing."
                    )
                    continue

                self.logger.info(
                    f"Processing fixture context {fixture_id} ({i + 1}/{len(fixtures_to_query_teams_for)}) for league {league_id}, season {season}"
                )
                teams_to_process_for_stats = [home_team_id, away_team_id]
                stats_for_db_batch = []

                for team_id_to_fetch in teams_to_process_for_stats:
                    try:
                        with engine.connect() as conn:
                            check_stmt = team_stats_table.select().where(
                                team_stats_table.c.team_id == team_id_to_fetch,
                                team_stats_table.c.fixture_id == fixture_id,
                            )
                            existing_stat_for_season = conn.execute(check_stmt).first()
                        if existing_stat_for_season:
                            self.logger.info(
                                f"Season stats for team {team_id_to_fetch}, league {league_id}, season {season} already exist in DB. Skipping API call."
                            )
                            continue
                    except SQLAlchemyError as e_check:
                        self.logger.error(
                            f"DB error checking existing season stats for team {team_id_to_fetch}, league {league_id}, season {season}: {e_check}"
                        )

                    if api_calls_in_current_batch >= rate_limit_threshold_per_batch:
                        elapsed_time_in_batch = time.time() - batch_start_time
                        if elapsed_time_in_batch < pause_duration:
                            sleep_time = pause_duration - elapsed_time_in_batch
                            self.logger.info(
                                f"Rate limit threshold hit. Sleeping for {sleep_time:.1f} seconds."
                            )
                            time.sleep(sleep_time)
                        api_calls_in_current_batch = 0
                        batch_start_time = time.time()

                    endpoint = "teams/statistics"
                    params = {
                        "league": league_id,
                        "team": team_id_to_fetch,
                        "season": season,
                        "date": date,
                    }

                    self.logger.info(
                        f"Calling API for team {team_id_to_fetch}, league {league_id}, season {season} (context fixture {fixture_id})"
                    )
                    response_json = self._get_request(endpoint, params)
                    api_calls_in_current_batch += 1
                    api_call_count_total += 1

                    if response_json and response_json.get("response"):
                        parsed_stats = self._parse_team_stat_response(
                            response_json["response"],
                            fixture_id,
                            team_id_to_fetch,
                            # league_id,
                            # season
                        )
                        if parsed_stats:
                            self.logger.info(
                                f"Parsed season stats for team {team_id_to_fetch}, league {league_id}"
                            )
                            if not parsed_stats.get("team_id") or parsed_stats.get("team_id") <= 0:
                                self.logger.warning(
                                    f"Skipping stats record - missing or invalid team_id for fixture {fixture_id}"
                                )
                                continue
                            stats_for_db_batch.append(parsed_stats)
                        else:
                            self.logger.warning(
                                f"Failed to parse API season stats for team {team_id_to_fetch}, L:{league_id}, S:{season}. Response: {response_json.get('response')}"
                            )
                    else:
                        self.logger.error(
                            f"API request failed for team {team_id_to_fetch}, L:{league_id}, S:{season}. Params: {params}. Response: {response_json}"
                        )

                if stats_for_db_batch:
                    try:
                        with engine.begin() as conn:
                            for stat_data in stats_for_db_batch:
                                conflict_elements = ["team_id", "fixture_id"]

                                values_to_insert = stat_data.copy()
                                values_to_insert["fixture_id"] = fixture_id

                                stmt = pg_insert(team_stats_table).values(**values_to_insert)
                                update_cols = {
                                    c.name: stmt.excluded[c.name]
                                    for c in team_stats_table.c
                                    if c.name not in conflict_elements
                                }
                                stmt = stmt.on_conflict_do_update(
                                    index_elements=conflict_elements, set_=update_cols
                                )
                                conn.execute(stmt)
                            self.logger.info(
                                f"Successfully upserted {len(stats_for_db_batch)} team season stats records (context fixture {fixture_id})."
                            )
                    except SQLAlchemyError as e_upsert:
                        self.logger.error(
                            f"DB error upserting batch season stats (context fixture {fixture_id}): {e_upsert}"
                        )

                if (i + 1) % 10 == 0:
                    self.logger.info(
                        f"--- Progress: Checked {i + 1}/{len(fixtures_to_query_teams_for)} fixture contexts. Total API calls this run: {api_call_count_total} ---"
                    )
                time.sleep(0.2)  # Shorter delay after each fixture context check

            self.logger.info(
                f"Finished get_team_stats_for_fixtures (season stats). Total API calls: {api_call_count_total}"
            )
        except Exception as e:
            self.logger.error(f"General error in get_team_stats_for_fixtures: {e}")

    def get_missing_statistics(self) -> None:
        """
        Gets statistics for fixtures that are missing statistics data.
        Handles API rate limiting by pausing after every 250 requests.
        """
        request_count = 0
        all_request_count = 0
        start_time = time.time()

        missing_fixture_ids = self.get_fixture_ids_without_statistics()
        missing_fixture_ids_count = len(missing_fixture_ids)
        print(f"Total number of missing fixture IDs: {missing_fixture_ids_count}")

        for fixture_id in missing_fixture_ids:
            self.get_statistics(fixture_id)
            request_count += 1
            all_request_count += 1
            print(f"Processed {all_request_count} of {missing_fixture_ids_count} fixtures")

            # Check if we've made 250 requests
            if request_count >= 250:
                elapsed_time = time.time() - start_time

                # If less than a minute has passed, wait
                if elapsed_time < 60:
                    wait_time = 60 - elapsed_time
                    print(f"Waiting for {wait_time:.1f} seconds")
                    time.sleep(wait_time)

                # Reset the counter and start time
                request_count = 0
                start_time = time.time()

    def get_statistics(self, fixture_id: int) -> dict:
        """
        Retrieves statistics for a specific fixture and stores them in PostgreSQL.
        Args:
            fixture_id (int): The ID of the fixture to get statistics for
        Returns:
            Dict: Statistics data for the specified fixture
        """
        endpoint = "fixtures/statistics"
        params = {"fixture": fixture_id}

        response = self._get_request(endpoint, params)

        if response and "response" in response and response["response"]:
            raw_statistics = response["response"]
            statistics = {"fixture_id": fixture_id, "home": {"stats": {}}, "away": {"stats": {}}}

            try:
                # Assuming the first element is home and the second is away
                if len(raw_statistics) >= 2:
                    statistics["home"]["team_id"] = raw_statistics[0]["team"]["id"]
                    statistics["home"]["team_name"] = raw_statistics[0]["team"]["name"]
                    statistics["home"]["team_logo"] = raw_statistics[0]["team"]["logo"]
                    statistics["away"]["team_id"] = raw_statistics[1]["team"]["id"]
                    statistics["away"]["team_name"] = raw_statistics[1]["team"]["name"]
                    statistics["away"]["team_logo"] = raw_statistics[1]["team"]["logo"]
                    # Process statistics for both teams
                    for i, team in enumerate(["home", "away"]):
                        for stat in raw_statistics[i]["statistics"]:
                            key = stat["type"].lower().replace(" ", "_")
                            statistics[team]["stats"][key] = stat["value"]
                else:
                    self.logger.warning(f"Insufficient statistics data for fixture {fixture_id}")
                    print(f"json: {raw_statistics}")
            except (IndexError, KeyError, TypeError) as e:
                self.logger.error(f"Error processing statistics for fixture {fixture_id}: {e}")
                time.sleep(3)
                return {}

            # Map statistics to fixture table columns
            def stat_map(prefix, stats):
                mapping = {
                    "shots_on_goal": f"{prefix}_shots_on_goal",
                    "shots_off_goal": f"{prefix}_shots_off_goal",
                    "total_shots": f"{prefix}_total_shots",
                    "blocked_shots": f"{prefix}_blocked_shots",
                    "shots_insidebox": f"{prefix}_shots_insidebox",
                    "shots_outsidebox": f"{prefix}_shots_outsidebox",
                    "fouls": f"{prefix}_fouls",
                    "corner_kicks": f"{prefix}_corner_kicks",
                    "offsides": f"{prefix}_offsides",
                    "ball_possession": f"{prefix}_ball_possession",
                    "yellow_cards": f"{prefix}_yellow_cards",
                    "red_cards": f"{prefix}_red_cards",
                    "goalkeeper_saves": f"{prefix}_goalkeeper_saves",
                    "total_passes": f"{prefix}_total_passes",
                    "passes_accurate": f"{prefix}_passes_accurate",
                    "passes_%": f"{prefix}_passes_percent",
                    "expected_goals": f"{prefix}_expected_goals",
                }
                result = {}
                for k, v in mapping.items():
                    val = stats.get(k)
                    # Convert ball_possession from '55%' to float if needed
                    if k == "ball_possession" and isinstance(val, str) and val.endswith("%"):
                        try:
                            val = float(val.replace("%", ""))
                        except Exception:
                            val = None
                    # Convert passes_percent from '55%' to float if needed
                    if (
                        k == "passes_percent"
                        or k == "passes_%"
                        and isinstance(val, str)
                        and val.endswith("%")
                    ):
                        try:
                            val = float(val.replace("%", ""))
                        except Exception:
                            val = None
                    result[v] = val
                return result

            home_stats = stat_map("home", statistics["home"]["stats"])
            away_stats = stat_map("away", statistics["away"]["stats"])

            # Prepare update dict
            update_dict = {**home_stats, **away_stats}

            # Update the fixture row in PostgreSQL
            stmt = (
                update(fixtures_table)
                .where(fixtures_table.c.fixture_id == fixture_id)
                .values(**update_dict)
            )
            try:
                with engine.begin() as conn:
                    conn.execute(stmt)
                self.logger.info(f"Statistics for fixture {fixture_id} updated in PostgreSQL.")
            except Exception as e:
                self.logger.error(
                    f"Error updating statistics for fixture {fixture_id} in PostgreSQL: {e}"
                )
            return statistics
        else:
            self.logger.warning(f"No statistics found for fixture {fixture_id}")
            try:
                # Get fixture date from PostgreSQL
                query = text(
                    "SELECT date FROM api_football.fixtures WHERE fixture_id = :fixture_id"
                )
                with engine.connect() as conn:
                    result = conn.execute(query, {"fixture_id": fixture_id}).first()
                    if result:
                        fixture_date = datetime.strptime(str(result[0]), "%Y-%m-%d %H:%M:%S")
                        if fixture_date < datetime.now() - timedelta(days=7):
                            # Delete old fixture from PostgreSQL
                            delete_query = text(
                                "DELETE FROM api_football.fixtures WHERE fixture_id = :fixture_id"
                            )
                            with engine.begin() as conn:
                                result = conn.execute(delete_query, {"fixture_id": fixture_id})
                                if result.rowcount == 1:
                                    self.logger.info(
                                        f"Fixture {fixture_id} dropped from PostgreSQL due to date constraint."
                                    )
                                else:
                                    self.logger.warning(
                                        f"Fixture {fixture_id} was NOT deleted (rowcount={result.rowcount})."
                                    )
                            return {}
            except Exception as e:
                self.logger.error(f"Error checking date for fixture {fixture_id}: {e}")
            time.sleep(1)
            return {}

    def get_fixture_ids_without_statistics(self) -> list[int]:
        """
        Retrieves fixture IDs from PostgreSQL where statistics columns are empty, the date is today or earlier,
        and the league ID is one of the specified IDs.
        Returns:
            List[int]: List of fixture IDs without statistics that meet the criteria.
        """
        try:
            # Load league IDs from JSON file
            league_ids_file_path = os.path.join(
                project_root, "data", "create_data", "api_football", "league_ids.json"
            )
            with open(league_ids_file_path) as f:
                league_ids_data = json.load(f)
            target_league_ids = [item["league_id"] for item in league_ids_data]
            league_mapping = {item["league_id"]: item["league_name"] for item in league_ids_data}

            today = (datetime.now() - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M")
            print(today)

            # Query to find fixtures without statistics
            query = text("""
                SELECT f.fixture_id, f.league_id, f.league_name 
                FROM api_football.fixtures f
                left join api_football.team_stats ts on f.fixture_id = ts.fixture_id 
                WHERE date <= now() and ts.fixture_id  is null
            """)

            league_counts = {}
            fixture_ids = []

            with engine.connect() as conn:
                result = conn.execute(query, {"today": today, "league_ids": target_league_ids})

                for row in result:
                    fixture_ids.append(row.fixture_id)
                    league_id = row.league_id
                    if league_id:
                        league_counts[league_id] = league_counts.get(league_id, 0) + 1

            # Log league statistics
            self.logger.info("--- Final League Counts for Fixtures Without Statistics ---")
            for league_id, count in league_counts.items():
                league_name = league_mapping.get(league_id, f"Unknown League ({league_id})")
                self.logger.info(f"League: {league_name} ({league_id}) - Count: {count}")
            self.logger.info("--- End Final League Counts ---")

            return fixture_ids

        except Exception as e:
            self.logger.error(f"Error retrieving fixture IDs without statistics: {e}")
            return []

    def get_prediction_for_fixture(self, fixture_id: int) -> bool:
        """
        Fetches prediction data for a specific fixture_id from the API,
        transforms it, and upserts it into the api_football.predictions table.
        Args:
            fixture_id (int): The ID of the fixture to get predictions for.
        Returns:
            bool: True if the operation was successful, False otherwise.
        """
        api_url = f"{self.base_url}/predictions?fixture={fixture_id}"
        try:
            response = requests.get(api_url, headers=self.headers)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)

            api_data = response.json()

            if not api_data.get("response"):
                message = f"No prediction data found in API response for fixture ID: {fixture_id}"
                self.logger.warning(message)
                print(message)
                return False

            # The main object containing all prediction related data for the fixture
            prediction_api_obj = api_data["response"][0]

            # Extract parts of the API response
            predictions_part = prediction_api_obj.get("predictions", {})
            teams_api_part = prediction_api_obj.get("teams", {})
            comparison_part = prediction_api_obj.get("comparison", {})
            h2h_list_api = prediction_api_obj.get("h2h", [])

            fixture_home_team_id = teams_api_part.get("home", {}).get("id")
            fixture_away_team_id = teams_api_part.get("away", {}).get("id")

            # Helper to safely get and convert percentage strings (e.g., "50%" or "50.5") to float
            def parse_percent(percent_val):
                if percent_val is None:
                    return None
                if isinstance(percent_val, (int, float)):
                    return float(percent_val)
                try:
                    s_val = str(percent_val).replace("%", "")
                    return float(s_val)
                except ValueError:
                    return None

            # Prepare data for database, matching the provided schema
            data_to_upsert = {
                "fixture_id": fixture_id,
                # From 'predictions' part
                "winner_id": predictions_part.get("winner", {}).get("id"),
                "winner_name": predictions_part.get("winner", {}).get("name"),
                "win_or_draw": predictions_part.get("win_or_draw"),
                "under_over": predictions_part.get("under_over"),
                "goals_home": predictions_part.get("goals", {}).get("home"),
                "goals_away": predictions_part.get("goals", {}).get("away"),
                "advice": predictions_part.get("advice"),
                "home_win_percent": parse_percent(predictions_part.get("percent", {}).get("home")),
                "draw_percent": parse_percent(predictions_part.get("percent", {}).get("draw")),
                "away_win_percent": parse_percent(predictions_part.get("percent", {}).get("away")),
                # From 'comparison' part - parse percentages to remove % signs
                "comparison_home_form": parse_percent(comparison_part.get("form", {}).get("home")),
                "comparison_away_form": parse_percent(comparison_part.get("form", {}).get("away")),
                "comparison_home_att": parse_percent(comparison_part.get("att", {}).get("home")),
                "comparison_away_att": parse_percent(comparison_part.get("att", {}).get("away")),
                "comparison_home_def": parse_percent(comparison_part.get("def", {}).get("home")),
                "comparison_away_def": parse_percent(comparison_part.get("def", {}).get("away")),
                "comparison_home_poisson_distribution": parse_percent(
                    comparison_part.get("poisson_distribution", {}).get("home")
                ),
                "comparison_away_poisson_distribution": parse_percent(
                    comparison_part.get("poisson_distribution", {}).get("away")
                ),
                "comparison_home_h2h": parse_percent(comparison_part.get("h2h", {}).get("home")),
                "comparison_away_h2h": parse_percent(comparison_part.get("h2h", {}).get("away")),
                "comparison_home_goals": parse_percent(
                    comparison_part.get("goals", {}).get("home")
                ),
                "comparison_away_goals": parse_percent(
                    comparison_part.get("goals", {}).get("away")
                ),
                "comparison_home_total": parse_percent(
                    comparison_part.get("total", {}).get("home")
                ),
                "comparison_away_total": parse_percent(
                    comparison_part.get("total", {}).get("away")
                ),
                "updated_at": datetime.now(),
            }

            # Calculate H2H stats
            h2h_home_wins_calc, h2h_away_wins_calc, h2h_draws_calc = 0, 0, 0
            if (
                fixture_home_team_id is not None
                and fixture_away_team_id is not None
                and h2h_list_api
            ):
                for match in h2h_list_api:
                    match_teams = match.get("teams", {})
                    match_goals = match.get("goals", {})

                    h2h_match_home_id = match_teams.get("home", {}).get("id")

                    h2h_gh = match_goals.get("home")
                    h2h_ga = match_goals.get("away")

                    if isinstance(h2h_gh, (int, float)) and isinstance(h2h_ga, (int, float)):
                        if h2h_gh == h2h_ga:
                            h2h_draws_calc += 1
                        elif h2h_match_home_id == fixture_home_team_id:
                            if h2h_gh > h2h_ga:
                                h2h_home_wins_calc += 1
                            else:
                                h2h_away_wins_calc += 1
                        elif h2h_match_home_id == fixture_away_team_id:
                            if h2h_ga > h2h_gh:
                                h2h_home_wins_calc += 1
                            else:
                                h2h_away_wins_calc += 1

            data_to_upsert["h2h_home_wins"] = h2h_home_wins_calc
            data_to_upsert["h2h_away_wins"] = h2h_away_wins_calc
            data_to_upsert["h2h_draws"] = h2h_draws_calc
            data_to_upsert["h2h_total_matches"] = len(h2h_list_api) if h2h_list_api else 0

            # Upsert to PostgreSQL
            stmt = pg_insert(predictions_table).values(**data_to_upsert)
            update_dict = {c: stmt.excluded[c] for c in data_to_upsert.keys() if c != "fixture_id"}
            stmt = stmt.on_conflict_do_update(index_elements=["fixture_id"], set_=update_dict)
            try:
                with engine.begin() as conn:
                    conn.execute(stmt)
                message = f"Successfully upserted prediction for fixture ID: {fixture_id}"
                self.logger.info(message)
                print(message)
                return True
            except SQLAlchemyError as e:
                message = f"Error upserting prediction for fixture ID {fixture_id}: {e}"
                self.logger.error(message)
                print(message)
                return False

        except requests.exceptions.RequestException as e:
            message = f"API request failed for fixture ID {fixture_id}: {e}"
            self.logger.error(message)
            print(message)
            return False
        except (Exception, psycopg2.Error) as e:
            message = f"Error processing or upserting prediction for fixture ID {fixture_id}: {e}"
            self.logger.error(message)
            print(message)
            if self.conn and not self.conn.closed:
                try:
                    self.conn.rollback()
                except psycopg2.Error as rb_e:
                    print(f"Rollback failed: {rb_e}")
            return False

    def get_fixture_ids_without_predictions(self) -> list[int]:
        """
        Retrieves fixture IDs from PostgreSQL where predictions do not exist.

        Returns:
            List[int]: List of fixture IDs without predictions.
        """
        try:
            query = text("""
                SELECT f.fixture_id 
                FROM api_football.fixtures f
                LEFT JOIN api_football.predictions p ON f.fixture_id = p.fixture_id
                WHERE p.fixture_id IS NULL
                AND f.date <= NOW() + INTERVAL '2 weeks'
            """)

            fixture_ids = []
            with engine.connect() as conn:
                result = conn.execute(query)
                fixture_ids = [row[0] for row in result]

            self.logger.info(f"Found {len(fixture_ids)} fixtures without predictions")
            # Process predictions for each fixture ID
            for fixture_id in fixture_ids:
                success = self.get_prediction_for_fixture(fixture_id)
                if not success:
                    self.logger.warning(f"Failed to get prediction for fixture ID: {fixture_id}")
                    continue
            return fixture_ids

        except Exception as e:
            self.logger.error(f"Error retrieving fixture IDs without predictions: {e}")
            return []

    def update_venues(self) -> None:
        """
        Updates venue data by:
        1. Getting all home team IDs where venue is missing
        2. Fetching team and venue information from teams API
        3. Upserting to venues table in PostgreSQL
        """
        try:
            # Get all home team IDs where venue is missing
            query = text("""
                SELECT DISTINCT f.home_team_id 
                FROM api_football.fixtures f
                LEFT JOIN api_football.venues v ON f.home_team_id = v.team_id
                WHERE v.team_id IS NULL
            """)

            missing_team_ids = []
            with engine.connect() as conn:
                result = conn.execute(query)
                missing_team_ids = [row[0] for row in result]

            self.logger.info(f"Found {len(missing_team_ids)} teams with missing venue data")

            if missing_team_ids:
                headers = {
                    "x-rapidapi-key": self.api_key,
                    "x-rapidapi-host": "v3.football.api-sports.io",
                }

                for team_id in missing_team_ids:
                    # Fetch team data from API
                    url = f"https://v3.football.api-sports.io/teams?id={team_id}"
                    response = requests.get(url, headers=headers)

                    if response.status_code == 200:
                        team_data = response.json()
                        if team_data["results"] > 0:
                            team = team_data["response"][0]
                            venue = team["venue"]

                            # Upsert to venues table
                            upsert_query = text("""
                                INSERT INTO api_football.venues (
                                    team_id, team_name, team_code, team_country, team_founded,
                                    team_national, team_logo, venue_id, venue_name, venue_address,
                                    venue_city, venue_capacity, venue_surface, venue_image, updated_at
                                ) VALUES (
                                    :team_id, :team_name, :team_code, :team_country, :team_founded,
                                    :team_national, :team_logo, :venue_id, :venue_name, :venue_address,
                                    :venue_city, :venue_capacity, :venue_surface, :venue_image, :updated_at
                                )
                                ON CONFLICT (team_id) DO UPDATE SET
                                    team_name = EXCLUDED.team_name,
                                    team_code = EXCLUDED.team_code,
                                    team_country = EXCLUDED.team_country,
                                    team_founded = EXCLUDED.team_founded,
                                    team_national = EXCLUDED.team_national,
                                    team_logo = EXCLUDED.team_logo,
                                    venue_id = EXCLUDED.venue_id,
                                    venue_name = EXCLUDED.venue_name,
                                    venue_address = EXCLUDED.venue_address,
                                    venue_city = EXCLUDED.venue_city,
                                    venue_capacity = EXCLUDED.venue_capacity,
                                    venue_surface = EXCLUDED.venue_surface,
                                    venue_image = EXCLUDED.venue_image,
                                    updated_at = EXCLUDED.updated_at
                            """)

                            params = {
                                "team_id": team["team"]["id"],
                                "team_name": team["team"]["name"],
                                "team_code": team["team"]["code"],
                                "team_country": team["team"]["country"],
                                "team_founded": team["team"]["founded"],
                                "team_national": team["team"]["national"],
                                "team_logo": team["team"]["logo"],
                                "venue_id": venue["id"],
                                "venue_name": venue["name"],
                                "venue_address": venue["address"],
                                "venue_city": venue["city"],
                                "venue_capacity": venue["capacity"],
                                "venue_surface": venue["surface"],
                                "venue_image": venue["image"],
                                "updated_at": datetime.now(),
                            }

                            with engine.begin() as conn:
                                conn.execute(upsert_query, params)

                            self.logger.info(f"Updated venue data for team ID: {team_id}")
                    else:
                        self.logger.error(
                            f"Error getting team data for ID {team_id}: {response.status_code}"
                        )
            self.logger.info("Venue update completed")
        except Exception as e:
            self.logger.error(f"Error updating venues: {e}")

    def delete_old_unscored_fixtures(self) -> None:
        """
        Deletes fixtures from PostgreSQL that are older than 7 days and have no score.
        """
        try:
            # Calculate cutoff date (7 days ago)
            cutoff_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

            # First count the number of fixtures that match criteria
            with engine.connect() as conn:
                result = conn.execute(
                    text("""
                    SELECT COUNT(*) 
                    FROM api_football.fixtures 
                    WHERE date < :cutoff_date AND home_goals IS NULL
                """),
                    {"cutoff_date": cutoff_date},
                )
                count = result.scalar()

            self.logger.info(f"Found {count} old unscored fixtures to delete")

            # Then delete them
            with engine.begin() as conn:
                result = conn.execute(
                    text("""
                    DELETE FROM api_football.fixtures 
                    WHERE date < :cutoff_date AND home_goals IS NULL
                """),
                    {"cutoff_date": cutoff_date},
                )
                deleted_count = result.rowcount

            self.logger.info(f"Deleted {deleted_count} old unscored fixtures")
            print(f"Deleted {deleted_count} old unscored fixtures")

        except Exception as e:
            self.logger.error(f"Error deleting old unscored fixtures: {e}")

    def _map_api_event_to_row(self, event_data: dict, fixture_id_context: int) -> dict:
        """Maps a single event object from the API response to a dictionary for the fixture_events table."""
        if not event_data or not isinstance(event_data, dict):
            self.logger.warning(
                f"_map_api_event_to_row: Invalid event_data for fixture_id {fixture_id_context}"
            )
            return None

        # Helper to safely access nested dictionary keys
        def safe_get(d, *keys, default=None):
            for k in keys:
                if not isinstance(d, dict):
                    return default
                d = d.get(k)
            return d if d is not None else default

        # Get the raw event detail
        raw_event_detail = safe_get(event_data, "detail")
        event_type = safe_get(event_data, "type")

        # Handle missing event_detail with appropriate defaults based on event type
        if raw_event_detail is None and event_type:
            # Map event types to default details when API doesn't provide them
            default_details = {
                "Var": "VAR Check",
                "var": "VAR Check",  # Handle case variations
                "VAR": "VAR Check",
                "Goal": "Goal",
                "Card": "Card",
                "subst": "Substitution",
                "Substitution": "Substitution",
            }

            event_detail = default_details.get(event_type, f"{event_type} Event")

            # Log when we apply a default value for tracking
            self.logger.info(
                f"Applied default event_detail '{event_detail}' for event_type '{event_type}' in fixture {fixture_id_context}"
            )
        else:
            event_detail = raw_event_detail

        mapped_event = {
            "fixture_id": fixture_id_context,
            "time_elapsed": safe_get(event_data, "time", "elapsed"),
            "time_extra": safe_get(event_data, "time", "extra"),
            "team_id": safe_get(event_data, "team", "id"),
            "team_name": safe_get(event_data, "team", "name"),
            "player_id": safe_get(event_data, "player", "id"),
            "player_name": safe_get(event_data, "player", "name"),
            "assist_player_id": safe_get(event_data, "assist", "id"),
            "assist_player_name": safe_get(event_data, "assist", "name"),
            "event_type": event_type,
            "event_detail": event_detail,
            "event_comments": safe_get(event_data, "comments"),
            "updated_at": datetime.now(),
        }

        if not mapped_event["team_id"] or not mapped_event["event_type"]:
            self.logger.warning(
                f"Essential event data missing for fixture {fixture_id_context}: {event_data}"
            )
            return None

        # Final validation that event_detail is not None (database constraint)
        if mapped_event["event_detail"] is None:
            self.logger.warning(
                f"Could not determine event_detail for event_type '{event_type}' in fixture {fixture_id_context}. Original data: {event_data}"
            )
            return None

        return mapped_event

    def get_and_upsert_fixture_events(self, fixture_id: int) -> bool:
        """
        Fetches event data for a specific fixture_id from the API,
        deletes existing events for this fixture, and inserts the new events.
        Args:
            fixture_id (int): The ID of the fixture to get events for.
        Returns:
            bool: True if the operation was successful, False otherwise.
        """
        endpoint = "fixtures/events"
        params = {"fixture": fixture_id}

        self.logger.info(f"Fetching events for fixture_id: {fixture_id}")
        api_response = self._get_request(endpoint, params)

        if not api_response or "response" not in api_response or not api_response["response"]:
            self.logger.warning(
                f"No event data found in API response for fixture ID: {fixture_id}. API Response: {api_response}"
            )
            return True

        events_from_api = api_response["response"]
        events_to_insert = []

        for event_data in events_from_api:
            mapped_row = self._map_api_event_to_row(event_data, fixture_id)
            if mapped_row:
                events_to_insert.append(mapped_row)

        if not events_to_insert:
            self.logger.info(
                f"No valid events parsed to insert for fixture_id: {fixture_id} after mapping."
            )
            return True  # No valid events to insert, consider it done.

        try:
            with engine.begin() as conn:
                # Delete existing events for this fixture_id to prevent duplicates/stale data
                delete_stmt = delete(fixture_events_table).where(
                    fixture_events_table.c.fixture_id == fixture_id
                )
                delete_result = conn.execute(delete_stmt)
                self.logger.info(
                    f"Deleted {delete_result.rowcount} existing event(s) for fixture_id: {fixture_id}"
                )

                # Bulk insert the new events
                conn.execute(fixture_events_table.insert(), events_to_insert)
                self.logger.info(
                    f"Successfully inserted {len(events_to_insert)} events for fixture_id: {fixture_id}"
                )
            return True
        except SQLAlchemyError as e:
            self.logger.error(f"Database error processing events for fixture_id {fixture_id}: {e}")
            return False
        except Exception as e:
            self.logger.error(
                f"Unexpected error processing events for fixture_id {fixture_id}: {e}"
            )
            return False

    def get_events_for_missing_fixtures(self):
        """
        Identifies finished fixtures without events and fetches/stores their event data.
        Includes API rate limiting.
        """
        self.logger.info(
            "Starting to get fixture events for finished fixtures missing event data..."
        )

        fixtures_to_query = []
        try:
            query = text("""
                SELECT DISTINCT f.fixture_id
                FROM api_football.fixtures f
                LEFT JOIN api_football.events e ON f.fixture_id = e.fixture_id
                WHERE f.home_fulltime_goals IS NOT NULL
                    AND f.date <= NOW() 
                    AND e.fixture_id IS NULL;
            """)

            with engine.connect() as conn:
                result = conn.execute(query)
                fixtures_to_query = [row.fixture_id for row in result]

            if not fixtures_to_query:
                self.logger.info("No finished fixtures found missing event data.")
                return

            self.logger.info(
                f"Found {len(fixtures_to_query)} finished fixtures missing event data."
            )

        except SQLAlchemyError as e:
            self.logger.error(f"DB error querying for fixtures missing events: {e}")
            return
        except Exception as e:
            self.logger.error(f"Unexpected error querying for fixtures missing events: {e}")
            return

        api_call_count_total = 0
        api_calls_in_current_batch = 0
        rate_limit_threshold_per_batch = 250  # As used in get_team_stats_for_fixtures
        pause_duration = 60  # Seconds
        batch_start_time = time.time()

        processed_count = 0
        for fixture_id in fixtures_to_query:
            if api_calls_in_current_batch >= rate_limit_threshold_per_batch:
                elapsed_time_in_batch = time.time() - batch_start_time
                if elapsed_time_in_batch < pause_duration:
                    sleep_time = pause_duration - elapsed_time_in_batch
                    self.logger.info(
                        f"Event fetching rate limit: Sleeping for {sleep_time:.1f} seconds."
                    )
                    time.sleep(sleep_time)
                api_calls_in_current_batch = 0
                batch_start_time = time.time()

            success = self.get_and_upsert_fixture_events(fixture_id)
            api_calls_in_current_batch += 1
            api_call_count_total += 1

            if success:
                processed_count += 1
            else:
                self.logger.warning(
                    f"Failed to get/upsert events for fixture_id: {fixture_id}. Will not retry in this run."
                )

            if (api_call_count_total % 10 == 0) or (api_call_count_total == len(fixtures_to_query)):
                self.logger.info(
                    f"--- Event Fetching Progress: Attempted {api_call_count_total}/{len(fixtures_to_query)} fixtures. "
                    f"Successfully processed: {processed_count}. ---"
                )

            time.sleep(0.1)  # Small delay between individual fixture event calls

        self.logger.info(
            f"Finished getting fixture events. Total API calls: {api_call_count_total}. "
            f"Successfully processed fixtures with events: {processed_count}."
        )


def main():
    api_key = os.getenv("API_FOOTBALL_API_KEY")
    if not api_key:
        print("API_FOOTBALL_API_KEY not found.")
        return
    logger = ExperimentLogger("get_fixtures")
    api_football = ApiFootball(api_key, logger)
    api_football.get_fixtures_for_leagues()
    api_football.get_missing_statistics()
    api_football.delete_old_unscored_fixtures()
    api_football.get_fixture_ids_without_predictions()
    api_football.get_team_stats_for_fixtures()
    api_football.get_events_for_missing_fixtures()
    api_football.update_venues()


if __name__ == "__main__":
    main()
