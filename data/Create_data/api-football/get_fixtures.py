import requests
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import pymongo
from datetime import datetime

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

from utils.logger import ExperimentLogger

class ApiFootball:
    """
    A class to interact with the API-Football API and store data in MongoDB.
    """
    def __init__(self, api_key: str, logger: ExperimentLogger = None):
        self.api_key = api_key
        self.logger = logger or ExperimentLogger()
        self.base_url = "https://v3.football.api-sports.io/"
        self.headers = {
            'x-rapidapi-key': self.api_key,
            'x-rapidapi-host': "v3.football.api-sports.io"
        }
        self.project_root = project_root
        self.data_dir = os.path.join(self.project_root, "data", "api-football")
        os.makedirs(self.data_dir, exist_ok=True)

        # MongoDB setup
        self.mongo_uri = 'mongodb://192.168.0.77:27017/'
        self.client = pymongo.MongoClient(self.mongo_uri)
        self.db = self.client["api-football"]  # Database name
        self.fixtures_collection = self.db["fixtures"] # Collection name
        self.leagues_collection = self.db["leagues"] # Collection name

    def _get_request(self, endpoint: str, params: Dict = None) -> Dict:
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

    def get_leagues(self, league_id: int = None) -> List[Dict]:
        """
        Retrieves all available leagues from the API and stores them in MongoDB.

        Args:
            league_id (int, optional): The ID of a specific league to retrieve. Defaults to None.

        Returns:
            List[Dict]: A list of available leagues.
        """
        endpoint = "leagues"
        params = {}
        if league_id:
            params['id'] = league_id
        response = self._get_request(endpoint, params)
        
        if response and 'response' in response:
            leagues = response['response']
            
            # Store in MongoDB with flattened structure
            for league in leagues:
                flattened_league = {
                    "league_id": league["league"]["id"],
                    "league_name": league["league"]["name"], 
                    "league_type": league["league"]["type"],
                    "league_logo": league["league"]["logo"],
                    "country_name": league["country"]["name"],
                    "country_code": league["country"]["code"],
                    "country_flag": league["country"]["flag"],
                    "seasons": league["seasons"]
                }
                
                self.leagues_collection.update_one(
                    {"league_id": flattened_league["league_id"]},
                    {"$set": flattened_league},
                    upsert=True
                )
            
            return leagues
        else:
            self.logger.warning("No leagues data found.")
            return []
    
    def get_fixtures(self, league_id: int, season: int) -> List[Dict]:
        """
        Retrieves fixtures for a specific league and season.

        Args:
            league_id (int): The ID of the league to get fixtures for
            season (int): The season year to get fixtures for

        Returns:
            List[Dict]: A list of fixtures for the specified league and season
        """
        endpoint = "fixtures"
        params = {
            'league': league_id,
            'season': season
        }
        
        response = self._get_request(endpoint, params)
        
        if response and 'response' in response:
            fixtures = response['response']
             
            # Insert fixtures into MongoDB
            for fixture in fixtures:
                transformed_fixture = {
                    "fixture_id": fixture["fixture"]["id"],
                    "date": fixture["fixture"]["date"],
                    "referee": fixture["fixture"]["referee"],
                    "venue_name": fixture["fixture"]["venue"]["name"],
                    "venue_id": fixture["fixture"]["venue"]["id"],
                    "league_id": fixture["league"]["id"],
                    "league_name": fixture["league"]["name"],
                    "league_season": fixture["league"]["season"],
                    "league_round": fixture["league"]["round"],
                    "home": {
                        "team_id": fixture["teams"]["home"]["id"],
                        "team_name": fixture["teams"]["home"]["name"],
                        "team_logo": fixture["teams"]["home"]["logo"],
                        "stats": {}
                    },
                    "away": {
                        "team_id": fixture["teams"]["away"]["id"],
                        "team_name": fixture["teams"]["away"]["name"],
                        "team_logo": fixture["teams"]["away"]["logo"],
                        "stats": {}
                    },
                    "goals": {
                        "home": fixture["goals"]["home"],
                        "away": fixture["goals"]["away"]
                    },
                    "score": {
                        "halftime": {
                            "home": fixture["score"]["halftime"]["home"],
                            "away": fixture["score"]["halftime"]["away"]
                        },
                        "fulltime": {
                            "home": fixture["score"]["fulltime"]["home"],
                            "away": fixture["score"]["fulltime"]["away"]
                        }
                    },
                    "match_outcome": 1 if fixture["fixture"]["status"]["short"] == "FT" and fixture["goals"]["home"] > fixture["goals"]["away"] else (2 if fixture["fixture"]["status"]["short"] == "FT" and fixture["goals"]["home"] == fixture["goals"]["away"] else (3 if fixture["fixture"]["status"]["short"] == "FT" else None))
                }
                
                self.fixtures_collection.update_one(
                    {"fixture_id": transformed_fixture["fixture_id"]},
                    {"$set": transformed_fixture},
                    upsert=True
                )
                
            return fixtures
        else:
            self.logger.warning(f"No fixtures found for league {league_id} season {season}")
            return []
    
    def get_statistics(self, fixture_id: int) -> Dict:
        """
        Retrieves statistics for a specific fixture and stores them in MongoDB.

        Args:
            fixture_id (int): The ID of the fixture to get statistics for

        Returns:
            Dict: Statistics data for the specified fixture
        """
        endpoint = "fixtures/statistics"
        params = {
            'fixture': fixture_id
        }
        
        response = self._get_request(endpoint, params)
        
        if response and 'response' in response:
            raw_statistics = response['response']

            # Transform statistics into simplified format
            statistics = {
                'fixture_id': fixture_id,
                'home': {
                    'team_id': raw_statistics[0]['team']['id'],
                    'team_name': raw_statistics[0]['team']['name'],
                    'team_logo': raw_statistics[0]['team']['logo'],
                    'stats': {}
                },
                'away': {
                    'team_id': raw_statistics[1]['team']['id'], 
                    'team_name': raw_statistics[1]['team']['name'],
                    'team_logo': raw_statistics[1]['team']['logo'],
                    'stats': {}
                }
            }

            # Process statistics for both teams
            for i, team in enumerate(['home', 'away']):
                for stat in raw_statistics[i]['statistics']:
                    key = stat['type'].lower().replace(' ', '_')
                    statistics[team]['stats'][key] = stat['value']

            # Insert simplified statistics into MongoDB
            try:
                self.fixtures_collection.update_one(
                    {"fixture_id": fixture_id},
                    {"$set": {"home.stats": statistics['home']['stats'], "away.stats": statistics['away']['stats']}}
                )
                self.logger.info(f"Statistics for fixture {fixture_id} inserted into MongoDB.")
            except Exception as e:
                self.logger.error(f"Error inserting statistics for fixture {fixture_id} into MongoDB: {e}")

            return statistics
        else:
            self.logger.warning(f"No statistics found for fixture {fixture_id}")
            return {}
    
    def get_fixture_ids_without_statistics(self) -> List[int]:
        """
        Retrieves fixture IDs from MongoDB where home/stats is empty and the date is today or earlier.

        Returns:
            List[int]: List of fixture IDs without statistics.
        """
        today = datetime.now().strftime('%Y-%m-%d %H:%M')
        print(today)
        query = {
            "date": {"$lte": today},
            "home.stats": {}
        }
        fixtures = self.fixtures_collection.find(query, {"fixture_id": 1})
        fixture_ids = [fixture["fixture_id"] for fixture in fixtures]
        self.logger.info(f"Found {len(fixture_ids)} fixtures without statistics.")
        return fixture_ids

    def _save_json(self, data: Any, filename: str) -> None:
        """
        Saves data to a JSON file in the data directory.

        Args:
            data (Any): The data to save.
            filename (str): The name of the file to save to.
        """
        file_path = os.path.join(self.data_dir, filename)
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
            self.logger.info(f"Data saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving data to {file_path}: {e}")

    def get_league_ids(self) -> None:
        """
        Retrieves league IDs, names, and countries from the leagues.json file and saves them to a new JSON file.
        """
        leagues_file_path = os.path.join(self.data_dir, 'leagues', 'leagues.json')
        try:
            with open(leagues_file_path, 'r') as f:
                leagues_data = json.load(f)

            league_info = []
            for item in leagues_data:
                league_id = item['league']['id']
                league_name = item['league']['name']
                country_name = item['country']['name']
                league_info.append({
                    'league_id': league_id,
                    'league_name': league_name,
                    'country': country_name
                })

            self._save_json(league_info, 'league_ids.json')
        except Exception as e:
            self.logger.error(f"Error processing leagues data: {e}")

    def get_fixtures_for_leagues(self) -> None:
        """
        Retrieves fixtures for each league ID found in 'league_ids.json'.
        """
        league_ids_file_path = os.path.join(self.data_dir, 'league_ids.json')
        seasons = [2022, 2023, 2024]
        try:
            with open(league_ids_file_path, 'r') as f:
                league_ids_data = json.load(f)
                
            num_leagues = len(league_ids_data)
            self.logger.info(f"Total number of leagues: {num_leagues}")
            
            for league_info in league_ids_data:
                league_id = league_info['league_id']
                self.logger.info(f"Getting fixtures for league ID: {league_id}")
                for season in seasons:
                    fixtures = self.get_fixtures(league_id, season)
                    if fixtures:
                        self.logger.info(f"Retrieved fixtures for league ID: {league_id} season: {season}")
                    else:
                        self.logger.warning(f"No fixtures found for league ID: {league_id} season: {season}")

        except Exception as e:
            self.logger.error(f"Error processing league IDs: {e}")


def main():
    api_key = '9d97f6f9804b592c86be814e246a077d'
    if not api_key:
        print("API_FOOTBALL_KEY not found.")
        return
    
    logger = ExperimentLogger()
    api_football = ApiFootball(api_key, logger)
    api_football.get_fixtures_for_leagues()

    # leagues = api_football.get_leagues()
    
    # fixtures = api_football.get_fixtures(61, 2024)

if __name__ == "__main__":
    main()
