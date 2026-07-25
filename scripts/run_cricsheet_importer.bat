@echo off
REM ===================================================================
REM Automated Cricsheet Game Importer & Model Retrainer
REM ===================================================================

cd /d "c:\Python\T20_Win_Probability_Model"

if not exist "logs" mkdir "logs"

echo =================================================================== >> "logs\cricsheet_importer.log"
echo Cricsheet Importer run started at %DATE% %TIME% >> "logs\cricsheet_importer.log"
echo =================================================================== >> "logs\cricsheet_importer.log"

"C:\Users\GNAlb\AppData\Local\Python\pythoncore-3.14-64\python.exe" -u scripts/import_and_retrain.py --days 2 >> "logs\cricsheet_importer.log" 2>&1

echo Cricsheet Importer run completed at %DATE% %TIME% >> "logs\cricsheet_importer.log"
echo. >> "logs\cricsheet_importer.log"
