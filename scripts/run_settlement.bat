@echo off
REM Scheduled bet-settlement wrapper (registered in Windows Task Scheduler, every 4h).
REM Must run from the project root so credentials.env and data/betting/bet_log.csv resolve.
cd /d "C:\Python\T20_Win_Probability_Model"
echo ============================================================ >> "logs\settlement.log"
echo [%DATE% %TIME%] Running scheduled settlement... >> "logs\settlement.log"
"C:\Users\GNAlb\AppData\Local\Python\pythoncore-3.14-64\python.exe" "scripts\reconcile_legacy_bets.py" --max-age-hours 72 >> "logs\settlement.log" 2>&1
echo [%DATE% %TIME%] Done (exit %ERRORLEVEL%). >> "logs\settlement.log"
