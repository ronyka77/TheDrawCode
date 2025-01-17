import requests
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import pymongo

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
            
            # Store in MongoDB
            for league in leagues:
                self.leagues_collection.update_one(
                    {"league.id": league["league"]["id"]},
                    {"$set": league},
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
        
        fixtures_dir = os.path.join(self.data_dir, "fixtures")
        if not os.path.exists(fixtures_dir):
            os.makedirs(fixtures_dir)
            
        fixtures_file = f"fixtures_{league_id}_{season}.json"
        
        if response and 'response' in response:
            fixtures = response['response']
            self._save_json(fixtures, os.path.join("fixtures", fixtures_file))
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
                self.fixtures_collection.insert_one(statistics)
                self.logger.info(f"Statistics for fixture {fixture_id} inserted into MongoDB.")
            except Exception as e:
                self.logger.error(f"Error inserting statistics for fixture {fixture_id} into MongoDB: {e}")

            return statistics
        else:
            self.logger.warning(f"No statistics found for fixture {fixture_id}")
            return {}
    
    def get_seasons_by_league(self, league_id: int) -> List[int]:
        """
        Retrieves all available seasons for a given league from the API.

        Args:
            league_id (int): The league for which to retrieve seasons.

        Returns:
            List[int]: A list of available seasons for the given league.
        """
        leagues = self.get_leagues(league_id)
        if leagues:
            seasons = []
            for league in leagues:
                if 'seasons' in league:
                    seasons.extend([season['year'] for season in league['seasons']])
            self._save_json(seasons, f"seasons_{league_id}.json")
            return seasons
        else:
            self.logger.warning(f"No seasons data found for league {league_id}.")
            return []

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

def main():
    api_key = '9d97f6f9804b592c86be814e246a077d'
    if not api_key:
        print("API_FOOTBALL_KEY not found in environment variables.")
        return
    
    logger = ExperimentLogger()
    api_football = ApiFootball(api_key, logger)
    
    # Example: Get statistics for a fixture and store in MongoDB
    statistics = api_football.get_statistics(1213754)
    # print(statistics)

if __name__ == "__main__":
    main()
