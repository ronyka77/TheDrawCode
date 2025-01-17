@echo off
echo Starting data processing sequence...

REM Set Python path - adjust this to your Python installation path
call conda init
call conda activate soccerpredictor_env
pushd "\\192.168.0.77\Betting\Draw_Prediction_Projects\TheDrawCode\data\Create_data"

echo Running fbref_get_data.py...
python scraping\fbref_get_data.py
if errorlevel 1 (
    echo Error running fbref_get_data.py
    pause
    exit /b 1
)
echo fbref_get_data.py completed successfully.

echo Running odds_scraper.py...
python scraping\odds_scraper.py
if errorlevel 1 (
    echo Error running fbref_scraper.py
    pause
    exit /b 1
)
echo odd_scraper.py completed successfully.

echo Running fbref_scraper.py...
python scraping\fbref_scraper.py
if errorlevel 1 (
    echo Error running fbref_scraper.py
    pause
    exit /b 1
)
echo fbref_scraper.py completed successfully.

echo Running merge_odds.py...
python scraping\merge_odds.py
if errorlevel 1 (
    echo Error running merge_odds.py
    pause
    exit /b 1
)
echo merge_odds.py completed successfully.

echo Running aggregation.py...
python aggregation.py
if errorlevel 1 (
    echo Error running aggregation.py
    pause
    exit /b 1
)
echo aggregation.py completed successfully.

echo All processes completed successfully!
pause