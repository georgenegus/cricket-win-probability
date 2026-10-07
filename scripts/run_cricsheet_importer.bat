@echo off
REM ===================================================================
REM Automated Cricsheet Game Importer & Model Retrainer
REM ===================================================================

cd /d "%~dp0.."

REM Use the per-user Python install when present, otherwise the py launcher.
set "PYTHON_EXE=%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=py"

if not exist "logs" mkdir "logs"

echo =================================================================== >> "logs\cricsheet_importer.log"
echo Cricsheet Importer run started at %DATE% %TIME% >> "logs\cricsheet_importer.log"
echo =================================================================== >> "logs\cricsheet_importer.log"

"%PYTHON_EXE%" -u scripts/import_and_retrain.py --days 2 >> "logs\cricsheet_importer.log" 2>&1

echo Cricsheet Importer run completed at %DATE% %TIME% >> "logs\cricsheet_importer.log"
echo. >> "logs\cricsheet_importer.log"
