import requests
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import pymongo
from datetime import datetime, timedelta
import time
import pandas as pd

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
        self.data_dir = os.path.join(self.project_root, "data", "create_data", "api-football")
        os.makedirs(self.data_dir, exist_ok=True)
        # MongoDB setup
        self.mongo_uri = 'mongodb://192.168.0.75:27017/'
        self.client = pymongo.MongoClient(self.mongo_uri)
        self.db = self.client["api-football"]  # Database name
        self.fixtures_collection = self.db["fixtures"] # Collection name
        self.predictions_collection = self.db["predictions"] # Collection name
        self.leagues_collection = self.db["leagues"] # Collection name
        self.venues_collection = self.db["venues"] # Collection name

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
                    "date": datetime.strptime(fixture["fixture"]["date"], "%Y-%m-%dT%H:%M:%S%z").strftime("%Y-%m-%d %H:%M"),
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
                        
                    },
                    "away": {
                        "team_id": fixture["teams"]["away"]["id"],
                        "team_name": fixture["teams"]["away"]["name"],
                        "team_logo": fixture["teams"]["away"]["logo"],
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
                
                # Update only specific fields, preserving existing stats
                self.fixtures_collection.update_one(
                    {"fixture_id": transformed_fixture["fixture_id"]},
                    {"$set": {
                        "date": transformed_fixture["date"],
                        "referee": transformed_fixture["referee"],
                        "venue_name": transformed_fixture["venue_name"],
                        "venue_id": transformed_fixture["venue_id"],
                        "league_id": transformed_fixture["league_id"],
                        "league_name": transformed_fixture["league_name"],
                        "league_season": transformed_fixture["league_season"],
                        "league_round": transformed_fixture["league_round"],
                        "home.team_id": transformed_fixture["home"]["team_id"],
                        "home.team_name": transformed_fixture["home"]["team_name"],
                        "home.team_logo": transformed_fixture["home"]["team_logo"],
                        "away.team_id": transformed_fixture["away"]["team_id"],
                        "away.team_name": transformed_fixture["away"]["team_name"],
                        "away.team_logo": transformed_fixture["away"]["team_logo"],
                        "goals": transformed_fixture["goals"],
                        "score": transformed_fixture["score"],
                        "match_outcome": transformed_fixture["match_outcome"]
                    }},
                    upsert=True
                )
                
            return len(fixtures)
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
        
        if response and 'response' in response and response['response']:
            raw_statistics = response['response']
            statistics = {
                'fixture_id': fixture_id,
                'home': {
                    'stats': {}
                },
                'away': {
                    'stats': {}
                }
            }
            
            try:
                # Assuming the first element is home and the second is away
                if len(raw_statistics) >= 2:
                    statistics['home']['team_id'] = raw_statistics[0]['team']['id']
                    statistics['home']['team_name'] = raw_statistics[0]['team']['name']
                    statistics['home']['team_logo'] = raw_statistics[0]['team']['logo']
                    statistics['away']['team_id'] = raw_statistics[1]['team']['id']
                    statistics['away']['team_name'] = raw_statistics[1]['team']['name']
                    statistics['away']['team_logo'] = raw_statistics[1]['team']['logo']
                    # Process statistics for both teams
                    for i, team in enumerate(['home', 'away']):
                        for stat in raw_statistics[i]['statistics']:
                            key = stat['type'].lower().replace(' ', '_')
                            statistics[team]['stats'][key] = stat['value']
                else:
                    self.logger.warning(f"Insufficient statistics data for fixture {fixture_id}")
                    print(f"json: {raw_statistics}")
            except (IndexError, KeyError, TypeError) as e:
                self.logger.error(f"Error processing statistics for fixture {fixture_id}: {e}")
                time.sleep(3)
                return {}
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
            try:
                fixture_date_str = self.fixtures_collection.find_one({"fixture_id": fixture_id}, {"date": 1})["date"]
                fixture_date = datetime.strptime(fixture_date_str, '%Y-%m-%d %H:%M')
                if fixture_date < datetime(2025, 2, 15):
                    self.fixtures_collection.delete_one({"fixture_id": fixture_id})
                    self.logger.info(f"Fixture {fixture_id} dropped from MongoDB due to date constraint.")
                    return {}
            except Exception as e:
                self.logger.error(f"Error checking date for fixture {fixture_id}: {e}")
            time.sleep(1)
            return {}

    def get_fixture_ids_without_statistics(self) -> List[int]:
        """
        Retrieves fixture IDs from MongoDB where home/stats is empty, the date is today or earlier,
        and the league ID is one of the specified IDs.
        Returns:
            List[int]: List of fixture IDs without statistics that meet the criteria.
        """
        
        league_ids_file_path = os.path.join(project_root, 'data', 'create_data', 'api-football', 'league_ids.json')
        with open(league_ids_file_path, 'r') as f:
            league_ids_data = json.load(f)
        target_league_ids = [item['league_id'] for item in league_ids_data]
        
        today = (datetime.now() - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M')
        print(today)
        query = {
            "date": {"$lte": today},
            "league_id": {"$in": target_league_ids},
            "home.stats": {},
            "score.fulltime.home": {"$ne": None}
        }
        fixtures = self.fixtures_collection.find(query, {"fixture_id": 1})
        fixture_ids = [fixture["fixture_id"] for fixture in fixtures]
        self.logger.info(f"Found {len(fixture_ids)} fixtures without statistics for league IDs {target_league_ids}.")
        return fixture_ids

    def get_fixture_ids_without_predictions(self) -> List[int]:
        """
        Retrieves fixture IDs from MongoDB where there is no corresponding prediction document,
        the date is today or earlier, and the league ID is one of the specified IDs.
        
        Returns:
            List[int]: List of fixture IDs without predictions that meet the criteria.
        """
        # Read fixture IDs from Excel file
        excel_path=os.path.join(project_root, 'data', 'Create_data', 'data_files', 'base', 'api_future_matches.xlsx')
        try:
            df = pd.read_excel(excel_path)
            fixture_ids = df['fixture_id'].tolist()
            self.logger.info(f"Successfully read {len(fixture_ids)} fixture IDs from Excel file")
        except Exception as e:
            self.logger.error(f"Error reading fixture IDs from Excel file: {e}")
            fixture_ids = []
        
        # Get fixture IDs that already have predictions
        existing_predictions = self.predictions_collection.distinct("fixture_id", {"fixture_id": {"$in":    fixture_ids}})
        
        # Get difference between all fixtures and those with predictions
        fixtures_without_predictions = list(set(fixture_ids) - set(existing_predictions))
        
        self.logger.info(f"Found {len(fixtures_without_predictions)} fixtures without predictions.")
        return fixtures_without_predictions

    def get_fixtures_for_leagues(self) -> None:
        """
        Retrieves fixtures for each league ID found in 'league_ids.json'.
        """
        league_ids_file_path = os.path.join(project_root, 'data', 'create_data', 'api-football', 'league_ids.json')
        seasons = [2025]
        try:
            with open(league_ids_file_path, 'r') as f:
                league_ids_data = json.load(f)
            # league_ids_data = [{'league_id': 40}] #IF YOU WANT TO GET FIXTURES FOR A SPECIFIC LEAGUE
            
            num_leagues = len(league_ids_data)
            self.logger.info(f"Total number of leagues: {num_leagues}")
            for league_info in league_ids_data:
                league_id = league_info['league_id']
                self.logger.info(f"Getting fixtures for league ID: {league_id}")
                for season in seasons:
                    fixtures = self.get_fixtures(league_id, season)
                    if fixtures:
                        self.logger.info(f"Retrieved {fixtures} fixtures for league ID: {league_id} season: {season}")
                    else:
                        self.logger.warning(f"No fixtures found for league ID: {league_id} season: {season}")
                        fixtures = self.get_fixtures(league_id, 2024)
                        self.logger.info(f"Retrieved {fixtures} fixtures for league ID: {league_id} season: 2024")


        except Exception as e:
            self.logger.error(f"Error processing league IDs: {e}")

    def get_statistics_for_fixtures(self) -> None:
        """
        Retrieves statistics for each fixture ID found in the fixtures.json files.
        """
        fixture_ids = self.get_fixture_ids_without_statistics()
        prediction_ids = self.get_fixture_ids_without_predictions()
        fixture_id_count = len(fixture_ids)
        request_count = 0
        all_request_count = 0
        start_time = time.time()
        
        for fixture_id in fixture_ids:
            self.get_statistics(fixture_id)
            request_count += 1
            all_request_count += 1
            print(f"Processed {all_request_count} of {fixture_id_count} fixtures")
            
            # Check if we've made 250 requests
            if request_count >= 250:
                elapsed_time = time.time() - start_time
                
                # If less than a minute has passed, wait
                if elapsed_time < 60:
                    print(f"Waiting for {60 - elapsed_time} seconds")
                    time.sleep(60 - elapsed_time)
                    
                # Reset the counter and start time
                request_count = 0
                start_time = time.time()
        request_count = 0
        all_request_count = 0
        start_time = time.time()
        for fixture_id in prediction_ids:
            self.get_predictions_for_fixture(fixture_id)
            request_count += 1
            all_request_count += 1
            print(f"Processed {all_request_count} of {fixture_id_count} fixtures")
            
            # Check if we've made 250 requests
            if request_count >= 250:
                elapsed_time = time.time() - start_time
                # If less than a minute has passed, wait
                if elapsed_time < 60:
                    print(f"Waiting for {60 - elapsed_time} seconds")
                    time.sleep(60 - elapsed_time)
                    
                # Reset the counter and start time
                request_count = 0
                start_time = time.time()

    def get_league_statistics_from_mongodb(self) -> Dict[int, int]:
        """
        Retrieves statistics from MongoDB showing count of fixtures per league where FT score is not null.
        
        Returns:
            Dict[int, int]: Dictionary mapping league_id to count of fixtures with non-null FT scores
        """
        try:
            # Aggregation pipeline to count non-null FT scores per league
            pipeline = [
                {
                    "$match": {
                        "score.fulltime.home": {"$exists": True, "$ne": None}
                    }
                },
                {
                    "$group": {
                        "_id": "$league_id",
                        "count": {"$sum": 1}
                    }
                }
            ]
            
            # Execute aggregation using self.fixtures_collection
            results = self.fixtures_collection.aggregate(pipeline)
            
            # Convert results to dictionary
            league_counts = {result['_id']: result['count'] for result in results}
            
            # Print as table
            print("\nLeague Statistics:")
            print("+------------+------------+")
            print("| League ID  | Fixture Count |")
            print("+------------+------------+")
            for league_id, count in sorted(league_counts.items(), key=lambda item: item[0]):
                print(f"| {str(league_id).center(10)} | {str(count).center(12)} |")
            print("+------------+------------+")
            
            return league_counts
            
        except Exception as e:
            self.logger.error(f"Error retrieving league statistics from MongoDB: {e}")
            return {}

    def delete_fixtures_not_in_leagues(self) -> None:
        """
        Deletes fixtures from MongoDB where league_id is not in league_ids.json.
        """
        try:
            # Load league IDs from JSON file
            league_ids_dir = os.path.join(self.project_root, "data", "Create_data", "api-football")
            league_ids_path = os.path.join(league_ids_dir, "league_ids.json")
            with open(league_ids_path, 'r') as f:
                league_ids = [league['league_id'] for league in json.load(f)]
                
            print(f"league_ids: {league_ids}")
            if len(league_ids) > 0:
                # Delete fixtures where league_id is not in the list
                result = self.fixtures_collection.delete_many(
                    {"league_id": {"$nin": league_ids}}
                )
                result = self.fixtures_collection.delete_many(
                    {
                        "$or": [
                            {"league_id": {"$nin": league_ids}},
                            {
                                "score.fulltime.home": None,
                                "date": {"$lt": "2024-10-01"}
                            }
                        ]
                    }
                )
                
                self.logger.info(f"Deleted {result.deleted_count} fixtures with invalid league IDs")
                print(f"Deleted {result.deleted_count} fixtures with invalid league IDs")
            else:
                self.logger.warning("No league IDs found in league_ids.json")
                print("No league IDs found in league_ids.json")
                
        except Exception as e:
            self.logger.error(f"Error deleting fixtures: {e}")

    def get_teams_missing_venues(self) -> List[int]:
        """
        Retrieves team IDs from fixtures collection that don't have corresponding venue data.
        
        Returns:
            List of team IDs that need venue information
        """
        try:
            # Get all unique team IDs from fixtures collection
            team_ids = self.fixtures_collection.distinct("teams.home.id") + \
                        self.fixtures_collection.distinct("teams.away.id")
            team_ids = list(set(team_ids))
            
            # Get all team IDs that have venue data
            teams_with_venues = self.venues_collection.distinct("team_id")
            
            # Find team IDs that don't have venue data
            missing_teams = list(set(team_ids) - set(teams_with_venues))
            
            self.logger.info(f"Found {len(missing_teams)} teams missing venue data")
            print(f"Found {len(missing_teams)} teams missing venue data")
            
            return missing_teams
            
        except Exception as e:
            self.logger.error(f"Error finding teams missing venue data: {e}")
            return []

    def get_teams_for_leagues(self) -> None:
        """
        Retrieves teams for each league ID found in 'league_ids.json' and stores them in a JSON file.
        """
        league_ids_file_path = os.path.join(project_root, 'data', 'create_data', 'api-football', 'league_ids.json')
        output_file_path = os.path.join(project_root, 'data', 'create_data', 'api-football', 'teams.json')
        
        try:
            with open(league_ids_file_path, 'r') as f:
                league_ids_data = json.load(f)
            
            all_teams = []
            
            for league_info in league_ids_data:
                league_id = league_info['league_id']
                self.logger.info(f"Getting teams for league ID: {league_id}")
                
                # Make API request to get teams
                url = "https://v3.football.api-sports.io/teams"
                params = {
                    "league": league_id,
                    "season": 2024
                }
                headers = {
                    "x-rapidapi-key": self.api_key,
                    "x-rapidapi-host": "v3.football.api-sports.io"
                }
                
                response = requests.get(url, headers=headers, params=params)
                
                if response.status_code == 200:
                    teams_data = response.json()
                    if teams_data['results'] > 0:
                        all_teams.extend(teams_data['response'])
                        self.logger.info(f"Retrieved {len(teams_data['response'])} teams for league ID: {league_id}")
                        
                        # Insert/update venues in MongoDB
                        for team in teams_data['response']:
                            try:
                                # Create document with team and venue data
                                venue_doc = {
                                    'team_id': team['team']['id'],
                                    'team': {
                                        'id': team['team']['id'],
                                        'name': team['team']['name'],
                                        'code': team['team']['code'],
                                        'country': team['team']['country'],
                                        'founded': team['team']['founded'],
                                        'national': team['team']['national'],
                                        'logo': team['team']['logo']
                                    },
                                    'venue': {
                                        'id': team['venue']['id'],
                                        'name': team['venue']['name'],
                                        'address': team['venue']['address'],
                                        'city': team['venue']['city'],
                                        'capacity': team['venue']['capacity'],
                                        'surface': team['venue']['surface'],
                                        'image': team['venue']['image']
                                    }
                                }
                                
                                # Upsert into venues collection using team_id as unique key
                                self.venues_collection.update_one(
                                    {'team_id': team['team']['id']},
                                    {'$set': venue_doc},
                                    upsert=True
                                )
                                # self.logger.info(f"Updated venue data for team ID: {team['team']['id']}")
                                
                            except Exception as e:
                                self.logger.error(f"Error updating venue data for team ID {team['team']['id']}: {e}")
                                
                    else:
                        self.logger.warning(f"No teams found for league ID: {league_id}")
                else:
                    self.logger.error(f"Error getting teams for league ID {league_id}: {response.status_code}")
                
                # Respect API rate limits
                time.sleep(5)
            
            # Save all teams to JSON file
            with open(output_file_path, 'w') as f:
                json.dump(all_teams, f, indent=4)
            
            self.logger.info(f"Saved {len(all_teams)} teams to {output_file_path}")
            
        except Exception as e:
            self.logger.error(f"Error processing league IDs: {e}")

    def get_predictions_for_fixture(self, fixture_id):
        """
        Get prediction data for a specific fixture ID and upsert to MongoDB
        
        Args:
            fixture_id (int): The ID of the fixture to get predictions for
        """
        try:
            # Make API request
            url = f"/predictions?fixture={fixture_id}"
            response = requests.get(
                f"{self.base_url}{url}",
                headers=self.headers
            )
            if response.status_code == 200:
                predictions_data = response.json()
                
                if predictions_data.get('response'):
                    prediction = predictions_data['response'][0]
                    
                    # Create prediction document
                    prediction_doc = {
                        'fixture_id': fixture_id,
                        'predictions': {
                            'winner': prediction.get('predictions', {}).get('winner', {}),
                            'win_or_draw': prediction.get('predictions', {}).get('win_or_draw'),
                            'under_over': prediction.get('predictions', {}).get('under_over'),
                            'goals': prediction.get('predictions', {}).get('goals', {}),
                            'advice': prediction.get('predictions', {}).get('advice'),
                            'percent': prediction.get('predictions', {}).get('percent', {})
                        },
                        'league': {
                            'id': prediction.get('league', {}).get('id'),
                            'name': prediction.get('league', {}).get('name'),
                            'country': prediction.get('league', {}).get('country'),
                            'logo': prediction.get('league', {}).get('logo'),
                            'flag': prediction.get('league', {}).get('flag'),
                            'season': prediction.get('league', {}).get('season')
                        },
                        'teams': {
                            'home': {
                                'id': prediction.get('teams', {}).get('home', {}).get('id'),
                                'name': prediction.get('teams', {}).get('home', {}).get('name'),
                                'logo': prediction.get('teams', {}).get('home', {}).get('logo'),
                                'last_5': prediction.get('teams', {}).get('home', {}).get('last_5', {}),
                                'league': prediction.get('teams', {}).get('home', {}).get('league', {})
                            },
                            'away': {
                                'id': prediction.get('teams', {}).get('away', {}).get('id'),
                                'name': prediction.get('teams', {}).get('away', {}).get('name'),
                                'logo': prediction.get('teams', {}).get('away', {}).get('logo'),
                                'last_5': prediction.get('teams', {}).get('away', {}).get('last_5', {}),
                                'league': prediction.get('teams', {}).get('away', {}).get('league', {})
                            }
                        },
                        'comparison': prediction.get('comparison', {}),
                        'h2h': prediction.get('h2h', []),
                        'updated_at': datetime.now()
                    }
                    
                    # Upsert to predictions collection
                    self.predictions_collection.update_one(
                        {'fixture_id': fixture_id},
                        {'$set': prediction_doc},
                        upsert=True
                    )
                    
                    self.logger.info(f"Updated prediction data for fixture ID: {fixture_id}")
                    
                else:
                    self.logger.warning(f"No prediction data found for fixture ID: {fixture_id}")
            else:
                self.logger.error(f"Error getting prediction data for fixture ID {fixture_id}: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error getting prediction data for fixture ID {fixture_id}: {e}")

def main():
    api_key = '9d97f6f9804b592c86be814e246a077d'
    if not api_key:
        print("API_FOOTBALL_KEY not found.")
        return

    logger = ExperimentLogger()
    api_football = ApiFootball(api_key, logger)

    api_football.get_fixtures_for_leagues()

    api_football.get_statistics_for_fixtures()

    api_football.delete_fixtures_not_in_leagues()

    api_football.get_teams_for_leagues()

    api_football.get_teams_missing_venues()

if __name__ == "__main__":
    main()
