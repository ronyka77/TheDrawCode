@echo off
echo Starting model processing sequence...

REM Set Python path and activate environment
call conda init
call conda activate soccerpredictor_env
pushd "\\192.168.0.77\Betting\Draw_Prediction_Projects\TheDrawCode\data\Create_data"

echo Running add_poisson_xG.py...
python add_poisson_xG.py
if errorlevel 1 (
    echo Error running add_poisson_xG.py
    pause
    exit /b 1
)
echo add_poisson_xG.py completed successfully.

echo Running add_ELO_scores.py...
python add_ELO_scores.py
if errorlevel 1 (
    echo Error running add_ELO_scores.py
    pause
    exit /b 1
)
echo add_ELO_scores.py completed successfully.

echo All model processing completed successfully!
pause 