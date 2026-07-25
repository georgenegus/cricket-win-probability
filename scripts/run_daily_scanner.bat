@echo off
REM ===================================================================
REM Daily Automated Scanner Execution Script for TheInsideEdge
REM ===================================================================

cd /d "c:\Python\T20_Win_Probability_Model"

if not exist "logs" mkdir "logs"

echo =================================================================== >> "logs\daily_scanner.log"
echo Scanner run started at %DATE% %TIME% >> "logs\daily_scanner.log"
echo =================================================================== >> "logs\daily_scanner.log"

"C:\Users\GNAlb\AppData\Local\Python\pythoncore-3.14-64\python.exe" -u -m src.betting.scanner >> "logs\daily_scanner.log" 2>&1

echo Scanner run completed at %DATE% %TIME% >> "logs\daily_scanner.log"
echo. >> "logs\daily_scanner.log"
