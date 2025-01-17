from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import logging
import pandas as pd
from datetime import datetime, timedelta
import random
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logging_config import LoggerSetup
from mongo.database import MongoClient  # Import MongoClient from database.py
from delete_duplicates import DuplicateHandler
from datetime import datetime, timedelta


logger = LoggerSetup.setup_logger(
    name='odds_scraper',
    log_file='./log/odds_scraper.log', 
    level=logging.INFO
)

user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.90 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.5112.79 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.90 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.5112.79 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.90 Safari/537.36"]

def initialize_mongodb():
    """Initialize MongoDB client and return a collection."""
    client = MongoClient()  # Use the MongoClient from database.py
    return client.get_collection("odds_data")  # Return odds_data collection for storing scraped data


def delete_future_odds():
    """Delete odds data from MongoDB for dates after January 1st, 2024."""
    try:
        client = MongoClient()  # Use the MongoClient from database.py
        collection = client.get_collection("odds_data")
        
        # Delete documents where date is after Jan 1, 2024
        result = collection.delete_many({
            "Date": {"$gt": "2024-01-01"}
        })
        
        logger.info(f"Successfully deleted {result.deleted_count} documents with dates after 2024-01-01")
        
    except Exception as e:
        logger.error(f"Error deleting future odds data: {e}")
        raise

# Execute the deletion 
# delete_future_odds() #Execute this when you want to delete this year odds data

# URL for the main site to scrape and constant league identifiers for URL generation
MAIN_URL = "https://www.oddsportal.com/"

# Dictionary defining leagues and their country identifiers for URL structure
LEAGUES = {
    "champions-league": "europe",
    "laliga": "spain",
    "laliga2": "spain",
    "bundesliga": "germany",
    "2-bundesliga": "germany",
    "3-liga": "germany",
    "serie-a": "italy",
    "serie-b": "italy",
    "ligue-1": "france",
    "ligue-2": "france",
    "premier-league": "england",
    "championship": "england",
    "league-one": "england",
    "eredivisie":"netherlands",
    "eerste-divisie":"netherlands",
    "liga-portugal":"portugal",
    "superliga":"denmark",
    "superliga":"romania",
    "super-lig":"turkey",
    # "allsvenskan":"sweden",
    "ekstraklasa":"poland"
}

# Define years for historical data scraping
# YEARS = ["2020-2021", "2021-2022", "2022-2023", "2023-2024"] # Use this for all historical data scraping
YEARS = [] #Only for current season
# Define column names for the DataFrame where scraped data will be stored
columns = ['Date', 'Time', 'Home', 'Away', 'Odd_Home', 'Odds_Draw', 'Odd_Away']

def generate_urls(leagues, years):
    """Generate URLs for both current and historical seasons based on league structure."""
    urls = []
    for league, country in leagues.items():
        # URL for the current active season (without a specific year)
        urls.append({
            "url": f"https://www.oddsportal.com/soccer/{country}/{league}/results/",
            "league": league
        })
        
        # URL for future matches
        urls.append({
            "url": f"https://www.oddsportal.com/soccer/{country}/{league}/",
            "league": league
        })
        
        # URLs for past seasons with specified years
        for year in years:
            urls.append({
                "url": f"https://www.oddsportal.com/soccer/{country}/{league}-{year}/results/",
                "league": league
            })
    
    # Define a dictionary to organize special URLs by league
    special_urls_by_league = {
        "brazil": [
            "https://www.oddsportal.com/football/brazil/serie-b/",
            "https://www.oddsportal.com/football/brazil/serie-b/results/",
            "https://www.oddsportal.com/football/brazil/serie-a-betano/",
            "https://www.oddsportal.com/football/brazil/serie-a-betano/results/"
            # "https://www.oddsportal.com/football/brazil/serie-a-2022/results/",
            # "https://www.oddsportal.com/football/brazil/serie-a-2021/results/",
            # "https://www.oddsportal.com/football/brazil/serie-a-2020/results/"
        ],
        "colombia": [
            "https://www.oddsportal.com/football/colombia/primera-a/",
            "https://www.oddsportal.com/football/colombia/primera-a/results/"
            # "https://www.oddsportal.com/football/colombia/primera-a-2023/results/",
            # "https://www.oddsportal.com/football/colombia/primera-a-2022/results/",
            # "https://www.oddsportal.com/football/colombia/primera-a-2021/results/",
            # "https://www.oddsportal.com/football/colombia/primera-a-2020/results/"
        ],
        "argentina": [
            "https://www.oddsportal.com/football/argentina/torneo-betano/",
            "https://www.oddsportal.com/football/argentina/torneo-betano/results/",
            # "https://www.oddsportal.com/football/argentina/liga-profesional-2023/results/",
            # "https://www.oddsportal.com/football/argentina/liga-profesional-2022/results/",
            # "https://www.oddsportal.com/football/argentina/liga-profesional-2021/results/",
            # "https://www.oddsportal.com/football/argentina/superliga-2019-2020/results/"
        ],
        "japan": [
            "https://www.oddsportal.com/football/japan/j1-league/",
            "https://www.oddsportal.com/football/japan/j1-league/results/",
            # "https://www.oddsportal.com/football/japan/j1-league-2023/results/",
            # "https://www.oddsportal.com/football/japan/j1-league-2022/results/",
            # "https://www.oddsportal.com/football/japan/j1-league-2021/results/",
            # "https://www.oddsportal.com/football/japan/j1-league-2020/results/"
        ],
        # "netherlands": [
        # #     "https://www.oddsportal.com/football/netherlands/eerste-divisie-2023/results/",
        #     "https://www.oddsportal.com/football/netherlands/eerste-divisie-2022-2023/results/",
        #     "https://www.oddsportal.com/football/netherlands/eerste-divisie-2021-2022/results/"
        # #     "https://www.oddsportal.com/football/netherlands/eerste-divisie-2020-2021/results/",
        # #     "https://www.oddsportal.com/football/netherlands/eredivisie-2020-2021/results/"
        # ],
        # "romania": [
        #     "https://www.oddsportal.com/football/romania/liga-1-2023-2024/results/",
        #     "https://www.oddsportal.com/football/romania/liga-1-2022-2023/results/",
        #     "https://www.oddsportal.com/football/romania/liga-1-2021-2022/results/",
        #     "https://www.oddsportal.com/football/romania/liga-1-2020-2021/results/"
        # ],
        # "portugal": [
        #     "https://www.oddsportal.com/football/portugal/primeira-liga-2023-2024/results/",
        #     "https://www.oddsportal.com/football/portugal/primeira-liga-2022-2023/results/",
            # "https://www.oddsportal.com/football/portugal/liga-portugal-2021-2022/results/",
        #     "https://www.oddsportal.com/football/portugal/primeira-liga-2020-2021/results/"
        # ],
        # "poland": [
            # "https://www.oddsportal.com/football/poland/ekstraklasa-2023-2024/results/",
            # "https://www.oddsportal.com/football/poland/ekstraklasa-2022-2023/results/",
            # "https://www.oddsportal.com/football/poland/ekstraklasa-2021-2022/results/",
        #     "https://www.oddsportal.com/football/poland/ekstraklasa-2020-2021/results/"
        # ]
        # "england": [
        #     "https://www.oddsportal.com/football/england/league-one-2020-2021/results/"
        # ],
        # "italy": [
        #     "https://www.oddsportal.com/football/italy/serie-b-2020-2021/results/"
        # ]
        
        
    }

    # Loop through the dictionary and append each URL with its league to `urls`
    for league, urls_list in special_urls_by_league.items():
        for url in urls_list:
            urls.append({"url": url, "league": league})
    
    logger.info(f"urls: {urls}")
    return urls  # Return all generated URLs for scraping

urls = generate_urls(LEAGUES, YEARS)  # Generate URLs using LEAGUES and YEARS

def initialize_driver() -> webdriver.Chrome:
    """
    Initialize the Chrome WebDriver with custom options for stable headless operation.

    Returns:
        webdriver.Chrome: An instance of the Chrome WebDriver with specified options.
    """
    chrome_options = Options()
    chrome_options.add_argument(f"user-agent={random.choice(user_agents)}")
    chrome_options.add_argument("--headless=new")  # Use new headless mode
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-logging")
    chrome_options.add_argument("--log-level=3")
    chrome_options.add_argument("--silent")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--ignore-certificate-errors")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    
    try:
        service = Service(ChromeDriverManager().install())
        service.log_path = "NUL"  # Suppress service logs
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Additional settings to avoid detection
        driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": random.choice(user_agents)})
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        return driver
    except Exception as e:
        logger.error(f"Failed to initialize WebDriver: {e}")
        raise

# Instantiate the WebDriver for use in scraping
driver = initialize_driver()

def approve_cookie(max_retries=3):
    """Handle cookie consent on the main website with retries."""
    for attempt in range(max_retries):
        try:
            driver.get(MAIN_URL)
            time.sleep(2 + attempt)  # Increase wait time with each retry
            
            cookie_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.ID, 'onetrust-accept-btn-handler'))
            )
            cookie_button.click()
            logger.info("Cookie consent handled successfully")
            return
        except Exception as e:
            logger.warning(f"Cookie approval attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                logger.error("All cookie approval attempts failed")
                # Continue anyway as the cookie popup might not appear
                return
            time.sleep(2)  # Wait before retrying

def get_pagination(url, max_retries=3):
    """
    Retrieve the number of pages with improved reliability and debugging.
    """
    for attempt in range(max_retries):
        try:
            driver.get(url)
            time.sleep(2)  # Allow time for page load
            
            # Scroll to bottom to ensure all elements are loaded
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
            
            soup = BeautifulSoup(driver.page_source, "lxml")
            pagination_links = soup.select('a[data-number]')
            
            if pagination_links:
                max_page = int(pagination_links[-1].text)
                logger.info(f"URL: {url}")
                logger.info(f"Found pagination links: {[link.text for link in pagination_links]}")
                logger.info(f"Max page number: {max_page}")
                return max_page
            else:
                logger.warning(f"Attempt {attempt + 1}: No pagination links found for {url}")
                
                # Debug information
                logger.debug("Page source excerpt:")
                logger.debug(soup.prettify()[:1000])  # First 1000 chars of HTML
                
                # Check if we're being rate limited or blocked
                if "rate limit" in soup.text.lower() or "blocked" in soup.text.lower():
                    logger.warning("Possible rate limiting detected")
                    time.sleep(5 * (attempt + 1))  # Exponential backoff
                
                # Try an alternative pagination selector
                alt_pagination = soup.select('.pagination a')
                if alt_pagination:
                    logger.info("Found alternative pagination elements")
                    logger.info(f"Alternative elements: {[link.text for link in alt_pagination]}")
        
        except Exception as e:
            logger.error(f"Pagination error on attempt {attempt + 1}: {str(e)}")
            time.sleep(3)
    
    logger.error(f"Failed to get pagination after {max_retries} attempts for {url}")
    return 1  # Default to 1 page if all attempts fail

def load_page_and_scroll(url, page):
    """Load the page and scroll to the bottom to load all elements."""
    try:
        full_url = f"{url}#/page/{page}" if page > 1 else url
        driver.get(full_url)
        
        # Wait for initial load
        time.sleep(3)
        
        # Scroll multiple times to ensure all content loads
        for _ in range(3):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
        
        # Find all event rows
        events = driver.find_elements(By.CLASS_NAME, 'eventRow')
        if not events:
            logger.warning(f"No events found on {full_url}")
            return []
            
        logger.info(f"Found {len(events)} events on page {page}")
        return events
        
    except Exception as e:
        logger.error(f"Error loading page {page}: {str(e)}")
        return []

def parse_date(event):
    """Extract and format date from event row."""
    try:
        dates = [dv.text for dv in event.find_all('div', class_='truncate')]
        if dates:
            date_text = dates[1] 
            date_part = date_text.split(' - ')[0].strip() if date_text.split(' - ')[0].strip() != '1' else dates[0].split(' - ')[0].strip()
            for date_format in ("%d %B %Y", "%d %b %Y", "%d %m %Y"):
                if "Yesterday" in date_part:  
                    datum = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
                    logger.info(f"Date found: {datum}")
                    return datum
                elif "Tomorrow" in date_part:
                    datum = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
                    logger.info(f"Date found: {datum}")
                    return datum
                elif "Today" in date_part:
                    datum = datetime.now().strftime("%Y-%m-%d")
                    logger.info(f"Date found: {datum}")
                    return datum
                else:
                    try:
                        datum = datetime.strptime(date_part, date_format).strftime("%Y-%m-%d")
                        logger.info(f"Date found: {datum}")
                        return datum
                    except ValueError:
                        # logger.info(f"Wrong date format: {date_part}")
                        pass
    except Exception as e:
        logger.error(f"Failed to parse date: {e}")
    # logger.warning(f"Date format not recognized for '{date_text}'. Skipping event.")
    return None  # Return None if date parsing fails

def parse_odds(event):
    """Extract odds information from event row."""
    try:
        odds = [p.text for p in event.find_all('p')]
        if len(odds) == 7:  # Adjust if necessary
            odds = odds[1:]
        return odds
    except Exception as e:
        logger.warning(f"Odds data is incomplete: {e}")
        return None

def process_event(event, league, actual_date):
    """Process a single event and return structured data if parsing is successful."""
    event_html = event.get_attribute('outerHTML')
    event = BeautifulSoup(event_html, "lxml")
    
    # Parse odds
    odds = parse_odds(event)
    if not odds or len(odds) < 6:
        return None  # Skip this event if odds data is incomplete

    # Extract home and away teams, generate unique_id
    home_team, away_team = odds[1], odds[2]
    home_team = DuplicateHandler.standardize_name(home_team)
    away_team = DuplicateHandler.standardize_name(away_team)
    unique_id = f"{actual_date}_{home_team}_{away_team}"
    
    # Create event data dictionary
    return {
        "unique_id": unique_id,
        "League": league,
        'Date': actual_date,
        'Time': odds[0],
        'Home': home_team,
        'Away': away_team,
        'Odd_Home': odds[3],
        'Odds_Draw': odds[4],
        'Odd_Away': odds[5]
    }
    
def scrape_page(url, page, league, retries=3, delay=5):
    """Enhanced page scraping with better error handling and debugging."""
    for attempt in range(retries):
        try:
            full_url = f"{url}#/page/{page}"
            logger.info(f"Attempting to scrape: {full_url}")
            
            driver.get(full_url)
            time.sleep(2)  # Initial wait
            
            # Scroll multiple times to ensure all content loads
            for _ in range(3):
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1)
            
            # Wait for events to be present
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, 'eventRow'))
                )
            except Exception as e:
                logger.warning(f"Timeout waiting for events on page {page}: {e}")
                
                # Debug information
                soup = BeautifulSoup(driver.page_source, "lxml")
                logger.debug(f"Current URL: {driver.current_url}")
                logger.debug(f"Page title: {driver.title}")
                logger.debug("Page source excerpt:")
                logger.debug(soup.prettify()[:1000])
                
                if attempt < retries - 1:
                    continue
            
            event_elements = driver.find_elements(By.CLASS_NAME, 'eventRow')
            
            if not event_elements:
                logger.warning(f"No events found on page {page} (attempt {attempt + 1})")
                if attempt < retries - 1:
                    time.sleep(delay * (attempt + 1))
                    continue
            
            # Process events as before...
            data = []
            actual_date = None
            
            for event_element in event_elements:
                event_html = event_element.get_attribute('outerHTML')
                event = BeautifulSoup(event_html, "lxml")
                date = parse_date(event)
                if date:
                    actual_date = date
                if actual_date is None:
                    logger.warning(f"No date found for event {event_element}")
                    continue
                event_data = process_event(event_element, league, actual_date)
                if event_data:
                    data.append(event_data)
            
            if data:
                logger.info(f"Successfully scraped {len(data)} events from page {page}")
                return data
            else:
                logger.warning(f"No data found on attempt {attempt + 1} for {url} page {page}")
                time.sleep(delay)  # Wait before retrying if data is missing
            
        except Exception as e:
            logger.error(f"Error scraping page {page} (attempt {attempt + 1}): {str(e)}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    
    # Log if all retries failed and return empty data
    logger.error(f"Failed to scrape data from {url} page {page} after {retries} attempts")
    return []

def validate_data(data):
    """Basic validation to ensure data has expected fields."""
    required_fields = {"Date", "Odd_Home"}
    return all(required_fields.issubset(event.keys()) for event in data)

def insert_to_mongodb(data):
    """Insert or update records in MongoDB based on unique Date, Home, and Away fields."""
    collection = initialize_mongodb()  # Access MongoDB collection
    if data:
        for record in data:
            # Home = DuplicateHandler.standardize_name(record['Home'])
            # Away = DuplicateHandler.standardize_name(record['Away'])
            print(f"{record['Date']}-{record['Home']}-{record['Away']}")
            
            # Define the filter to check if a matching record exists
            filter_query = {
                'Date': record['Date'],
                'Home': record['Home'],
                'Away': record['Away']
            }
            
            # Use the upsert option to insert if not found, or update if it exists
            update_query = {"$set": record}  # Updates fields with new data if record exists
            collection.update_one(filter_query, update_query, upsert=True)
        
        logger.info(f"Processed {len(data)} records with upsert (insert/update) into MongoDB")
    else:
        logger.warning("No data to insert.")

def main():
    """Main execution function."""
    try:
        approve_cookie()
        collection = initialize_mongodb()
        
        for url_data in urls:
            url = url_data["url"]
            league = url_data["league"]
            
            try:
                max_pages = get_pagination(url)
                logger.info(f"Processing {league}: {url} - {max_pages} pages")
                
                for page in range(1, max_pages + 1):
                    events = load_page_and_scroll(url, page)
                    if not events:
                        continue
                        
                    for event in events:
                        try:
                            actual_date = parse_date(BeautifulSoup(event.get_attribute('outerHTML'), "lxml"))
                            if not actual_date:
                                continue
                                
                            event_data = process_event(event, league, actual_date)
                            if event_data:
                                collection.update_one(
                                    {
                                        'Date': event_data['Date'],
                                        'Home': event_data['Home'],
                                        'Away': event_data['Away']
                                    },
                                    {'$set': event_data},
                                    upsert=True
                                )
                                
                        except Exception as e:
                            logger.error(f"Error processing individual event: {str(e)}")
                            continue
                            
                    time.sleep(2)  # Wait between pages
                    
            except Exception as e:
                logger.error(f"Error processing URL {url}: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Main execution error: {str(e)}")
        raise
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
