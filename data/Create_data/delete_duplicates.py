import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging
from typing import List, Dict, Any
from mongo.database import MongoClient
from logging_config import LoggerSetup
import pandas as pd


class DuplicateHandler:
    """Class to handle finding and removing duplicate entries in MongoDB collections."""

    def __init__(self, collection_name: str = 'fixtures'):
        """Initialize DuplicateHandler with collection name and setup logging.
        
        Args:
            collection_name: Name of MongoDB collection to process. Defaults to 'fixtures'.
        """
        self.collection_name = collection_name
        # Initialize logger before anything else
        self.logger = LoggerSetup.setup_logger(
            name='delete_duplicates',
            log_file='./log/delete_duplicates.log',
            level=logging.INFO
        )
        try:
            self.client = MongoClient()
            # Test database connection immediately
            with self.client.get_database() as db:
                collection = db[self.collection_name]
                doc_count = collection.count_documents({})
                self.logger.info(f"Successfully connected to database. Found {doc_count} documents.")
        except Exception as e:
            self.logger.error(f"Failed to initialize database connection: {str(e)}")
            raise

    @staticmethod 
    def standardize_name(name: str) -> str:
        """Standardize team names by removing common abbreviations and special characters."""
        if not name:
            return ""
            
        replacements = {
            # Common symbols
            "'": "",
            "-": "",
            ".": "",
            "_": "",
            "&": "and",
            "+": "and",
            "/": "",
            "\\": "",
            "(": "",
            ")": "",
            "[": "",
            "]": "",
            "{": "",
            "}": "",
            "*": "",
            "#": "",
            "@": "",
            "$": "",
            "%": "",
            "^": "",
            "|": "",
            "~": "",
            "`": "",
            "å": "a",
            "à": "a",
            "æ": "ae",
            "ë": "e",
            "ì": "i",
            "í": "i",
            "î": "i",
            "ø": "o",
            "ö": "o",
            "õ": "o",
            "ù": "u",
            "ú": "u",
            "û": "u",
            "ý": "y",
            "ÿ": "y",
            "ž": "z",
            "š": "s",
            "â": "a",
            "ã": "a",
            "ä": "a",
            "ß": "ss",
            "ç": "c",      
            "é": "e",
            "É": "E",
            "è": "e",
            "ê": "e",
            "ï": "i",
            "î": "i",
            "ò": "o",
            "á": "a",
            "ô": "o",
            "ő": "o",
            "ó": "o",
            "ű": "u",
            "ü": "u",
            "ñ": "n", 
            "ą": "a",
            "ć": "c", 
            "ę": "e",
            "ł": "l",
            "ń": "n",
            "ś": "s",
            "ș": "s",
            "ă": "a",
            "ț": "t",
            "ă": "a",
            "ș": "s",
            "ț": "t",
            "ă": "a",
            "â": "a",
            "î": "i",
            "æ": "ae",
            "ø": "o",
            "å": "a",
            "ź": "z",
            "ż": "z",
            "  ":" ",
            "Atle Tucuman": "Atl Tucuman",
            "Bayern Munich II": "Bayern II",
            "Ba�ak�ehir tr": "Basaksehir",
            "B SAD": "BSAD",
            "Arminia Bielefeld": "Arminia",
            "Amazonas FC": "Amazonas",
            "Aston Villa eng": "Aston Villa",
            "at Sturm Graz": "Sturm Graz",
            "at RB Salzburg": "RB Salzburg",
            "AC Ajaccio": "Ajaccio",
            "AC Milan": "Milan",
            "AS Roma": "Roma",
            "Athletic Club": "Ath Bilbao",
            "Atletico Madrid": "Atl Madrid", 
            "Atl Madrid es": "Atl Madrid",
            "Atl  Madrid": "Atl Madrid",
            "Atl Goianiense":"Atletico GO",
            "Atletico Mineiro":"Atletico MG",
            "Athletic Club": "Ath Bilbao",
            "Athletic Bilbao": "Ath Bilbao",
            "Athletico Paranaense": "Athletico-PR",
            "Barcelona es": "Barcelona",
            "Bayern Munich de": "Bayern Munich",
            "de Bayern Munich": "Bayern Munich",
            "Bayer Leverkusen": "Leverkursen",
            "B. Monchengladbach": "Gladbach",
            "B Monchengladbach": "Gladbach",
            "Birmingham City": "Birmingham",
            "Burton Albion": "Burton",
            "Burgos CF": "Burgos",
            "ch Young Boys": "Young Boys",
            "Cadiz CF": "Cadiz",
            "Cádiz": "Cadiz",
            "CD Mirandes": "Mirandes",
            "Cardiff City": "Cardiff",
            "Coventry City": "Coventry",
            "Clermont Foot": "Clermont",
            "Crawley Town": "Crawley",
            "Chapeconense SC": "Chapecoense",
            "Darmstadt 98": "Darmstadt",
            "Derby County": "Derby",
            "Defensa y Justicia": "Defensa y Just",
            "Dep Riestra": "Deportivo Riestra",
            "Dep La Coruna": "La Coruna",
            "Dortmund de": "Dortmund",
            "de RB Leipzig": "RB Leipzig",
            "de Leverkusen": "Leverkusen",
            "Estudiantes L.P.": "Estudiantes",
            "Erzgebirge Aue" : "Aue",
            "Eint Frankfurt": "Eintracht Frankfurt",
            "es Girona": "Girona",
            "es Atl Madrid": "Atl Madrid",
            "es Barcelona": "Barcelona",
            "es Real Madrid": "Real Madrid",
            "Girona es": "Girona",
            "eng Manchester City": "Manchester City",
            "Manchester City eng": "Manchester City",
            "eng Arsenal": "Arsenal",
            "eng Aston Villa": "Aston Villa",
            "eng Liverpool": "Liverpool",
            "Leeds United": "Leeds",
            "Flamengo RJ": "Flamengo",
            "FC Barcelona": "Barcelona",
            "FC Bayern": "Bayern Munich", 
            "FC Emmen": "Emmen",
            "FC Internazionale": "Inter Milan",
            "FC Porto": "Porto",
            "FC Volendam": "Volendam",
            "FC Eindhoven": "Eindhoven FC",
            "FC Cartagena SAD": "Cartagena",
            "fr Brest": "Brest",
            "Brest fr": "Brest",
            "fr Monaco": "Monaco",
            "Monaco fr": "Monaco",
            "fr PSG": "PSG",
            "PSG fr": "PSG",
            "Fortuna Sittard": "Sittard",
            "Go Ahead Eag":"GA Eagles",
            "G A Eagles":"GA Eagles",
            "Granada CF": "Granada",
            "Gimnasia L P": "Gimnasia LP",
            "Gimnasia-LP": "Gimnasia LP",
            "Hellas Verona": "Verona",
            "Helmond Sport": "Helmond",
            "Heracles Almelo": "Heracles",
            "Hertha BSC":"Hertha Berlin",
            "Hannover 96": "Hannover",
            "Hull City": "Hull",
            "it Bologna": "Bologna",
            "Bologna it": "Bologna",
            "it Juventus": "Juventus",
            "Juventus it": "Juventus",
            "it Milan": "Milan",
            "Milan it": "Milan",
            "it Inter": "Inter",
            "Inter it": "Inter",
            "it Atalanta": "Atalanta",
            "Ipswich Town": "Ipswich",
            "Ingolstadt 04": "Ingolstadt",
            "Jahn Rburg": "Regensburg",
            "Leverkusen": "Leverkursen",
            "Leverkursen de": "Leverkursen",
            "Luton Town": "Luton",
            "Lincoln City": "Lincoln",
            "Lille fr": "Lille",
            "Legia Warsaw": "Legia",
            "fr Lille": "Lille",
            "Liverpool eng": "Liverpool",
            "Leicester City": "Leicester",
            "Kawa Frontale": "Kawasaki Frontale",
            "Kyoto Sanga": "Kyoto",
            "Karlsruher SC": "Karlsruher",
            "Köln": "Koln",
            "FC Koln": "Koln",
            "Newcastle Utd": "Newcastle",
            "NEC Nijmegen": "Nijmegen",
            "nl Feyenoord": "Feyenoord",
            "Feyenoord nl": "Feyenoord",
            "Norwich City": "Norwich",
            "Nottham Forest": "Nottingham",
            "Mainz 05": "Mainz",
            "Malmo FF": "Malmo",
            "Mansfield Town": "Mansfield",
            "MVV Maastricht": "Maastricht",
            "Oxford United": "Oxford Utd",
            "Paris SG": "PSG",
            "Paris Saint Germain": "PSG",
            "Paris SG fr": "PSG",
            "Paris S-G": "PSG",
            "Paris S G": "PSG",
            "Paris S-G fr": "PSG",
            "P'borough Utd": "Peterborough",
            "Paderborn 07": "Paderborn",
            "Pborough Utd": "Peterborough",
            "pt Benfica": "Benfica",
            "PSV Eindhoven": "PSV",
            "Racing Santander": "Racing Sant",
            "Racing Club Ferrol": "Racing Ferrol",
            "R Oviedo": "Oviedo",
            "R  Oviedo": "Oviedo",
            "RB Salzburg au": "RB Salzburg",
            "RB Leipzig de": "RB Leipzig",
            "Real Madrid es": "Real Madrid",
            "RKC Waalwijk": "Waalwijk",
            "RKS Rakow": "Rakow",
            "Roda JC": "Roda",
            "Rodez Aveyron": "Rodez",
            "Rotherham Utd": "Rotherham",
            "Sparta Rdam": "Sparta Rotterdam",
            "Sparta Prague cz": "Sparta Prague",
            "Sporting CP pt": "Sporting CP",
            "Sporting Gijon": "Gijon",
            "Stuttgart de": "Stuttgart",
            "Schalke 04": "Schalke",
            "Sarmiento Junin": "Sarmiento",
            "Saint-Étienne": "St Etienne",
            "Saint Etienne": "St Etienne",
            "SaintEtienne": "St Etienne",
            "Stoke City": "Stoke",
            "Swansea City": "Swansea",
            "Wigan Athletic": "Wigan",
            "Sheffield Weds": "Sheffield Wed",
            "Varnamo Sodra FF": "Varnamo",
            "Vila Nova FC": "Vila Nova",
            "Vfl Osnabruck": "Osnabruck",
            "VVV Venlo": "Venlo",
            "VVVVenlo": "Venlo",
            "Young Boys ch": "Young Boys",
            "Ch Young Boys": "Young Boys",
            "Wisla Krakow": "Wisla",
        }
        name = str(name).strip()
        for old, new in replacements.items():
            name = name.replace(old, new)
        return name.strip()

    def find_duplicates(self) -> List[Dict[str, Any]]:
        """Find duplicate entries in the collection based on standardized team names and dates.
        
        Returns:
            List of documents that are duplicates
        """
        try:
            duplicates = []
            
            with self.client.get_database() as db:
                collection = db[self.collection_name]
                
                # Create index on Odd_Home for faster sorting
                collection.create_index([("Odd_Home", -1)])
                
                # Get all documents and verify we have data
                # Only fetch fields we need
                documents = list(collection.find({}, {
                    'unique_id': 1,
                    '_id': 1, 
                    'Home': 1,
                    'Away': 1,
                    'Date': 1,
                    'Odd_Home': 1,
                    'Odd_Away': 1,
                    'Odds_Draw': 1
                }))
                
                if not documents:
                    self.logger.warning("No documents found in collection")
                    return []
                    
                self.logger.info(f"Processing {len(documents)} documents. This may take a few minutes...")
                print(f"Processing {len(documents)} documents. This may take a few minutes...")
                
                # Pre-standardize all team names to avoid repeated processing
                for doc in documents:
                    # Update MongoDB document with standardized names
                    std_home = self.standardize_name(doc.get('Home', ''))
                    std_away = self.standardize_name(doc.get('Away', ''))
                    
                    collection.update_one(
                        {'_id': doc['_id']},
                        {'$set': {
                            'Home': std_home,
                            'Away': std_away
                        }}
                    )
                    
                    # Update the document in memory too since we'll use it later
                    doc['Home'] = std_home
                    doc['Away'] = std_away
                    
                    try:
                        doc['date_obj'] = pd.to_datetime(doc.get('Date', ''))
                    except:
                        self.logger.error(f"Invalid date format for document {doc.get('unique_id')}: {doc.get('Date')}")
                        doc['date_obj'] = None
                
                # Filter out documents with invalid dates
                documents = [doc for doc in documents if doc['date_obj'] is not None]
                
                # Sort documents by unique_id
                documents.sort(key=lambda x: x.get('unique_id', ''))
                self.logger.info(f"Sorted {len(documents)} documents by unique_id")
            
                # Create a dictionary to track unique combinations
                seen_matches = {}
                
                # Process in batches for better memory management
                batch_size = 1000
                for i in range(0, len(documents), batch_size):
                    batch = documents[i:i + batch_size]
                    self.logger.info(f"Processing batch {i // batch_size + 1} of {len(documents) // batch_size + 1}")
                    print(f"Processing batch {i // batch_size + 1} of {len(documents) // batch_size + 1}")  
                    for doc in batch:
                        # Check for matches within ±2 days
                        match_found = False
                        for key, existing in seen_matches.items():
                            existing_date = existing['date']
                            date_diff = abs((doc['date_obj'] - existing_date).days)
                            
                            if (date_diff <= 2 and 
                                doc['Home'] == existing['home'] and 
                                doc['Away'] == existing['away']):
                                # This is a duplicate
                                duplicates.append(doc)
                                match_found = True
                                
                                # Merge odds data from duplicate into original document
                                original_id = existing['_id']
                                odds_update = {k: doc[k] for k in ['Odd_Home', 'Odd_Away', 'Odds_Draw'] 
                                             if k in doc and doc[k]}
                                
                                if odds_update:
                                    collection.update_one(
                                        {'_id': original_id},
                                        {'$set': odds_update}
                                    )
                                break
                                
                        if not match_found:
                            # No duplicate found, add to seen matches
                            match_key = f"{doc['date_obj']}_{doc['Home']}_{doc['Away']}"
                            seen_matches[match_key] = {
                                '_id': doc['_id'],
                                'date': doc['date_obj'],
                                'home': doc['Home'],
                                'away': doc['Away']
                            }
            
            self.logger.info(f"Found {len(duplicates)} duplicate entries in {self.collection_name}")
            print(f"Found {len(duplicates)} duplicate entries in {self.collection_name}")
            return duplicates
        except Exception as e:
            self.logger.error(f"Error finding duplicates in {self.collection_name}: {str(e)}")
            raise

    def delete_duplicates(self) -> None:
        """Delete duplicate entries from the collection."""
        try:
            self.logger.info("Starting duplicate deletion process")
            duplicates = self.find_duplicates()
            
            if not duplicates:
                self.logger.info(f"No duplicates found in {self.collection_name}")
                return
                
            with self.client.get_database() as db:
                collection = db[self.collection_name]
                
                # Delete each duplicate document
                duplicate_ids = [doc['_id'] for doc in duplicates]
                
                result = collection.delete_many({'_id': {'$in': duplicate_ids}})
                
                self.logger.info(f"Successfully deleted {result.deleted_count} duplicate documents from {self.collection_name}")
                
        except Exception as e:
            self.logger.error(f"Error deleting duplicates from {self.collection_name}: {str(e)}")
            raise

if __name__ == "__main__":
    # This code block runs when the script is executed directly (not imported as a module)
    # Create a DuplicateHandler instance to manage the 'fixtures' collection
    duplicate_handler = DuplicateHandler(collection_name='fixtures')
    
    # Call delete_duplicates() to find and remove any duplicate entries
    # This will:
    # 1. Connect to MongoDB
    # 2. Find duplicate entries based on date and team names
    # 3. Delete the duplicate documents while preserving the original entries
    duplicate_handler.delete_duplicates()
