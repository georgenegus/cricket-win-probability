@echo off
REM ===================================================================
REM Daily Automated Scanner Execution Script for TheInsideEdge
REM ===================================================================

cd /d "%~dp0.."

REM Use the per-user Python install when present, otherwise the py launcher.
set "PYTHON_EXE=%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=py"

if not exist "logs" mkdir "logs"

echo =================================================================== >> "logs\daily_scanner.log"
echo Scanner run started at %DATE% %TIME% >> "logs\daily_scanner.log"
echo =================================================================== >> "logs\daily_scanner.log"

"%PYTHON_EXE%" -u -m src.betting.scanner >> "logs\daily_scanner.log" 2>&1

echo Scanner run completed at %DATE% %TIME% >> "logs\daily_scanner.log"
echo. >> "logs\daily_scanner.log"
