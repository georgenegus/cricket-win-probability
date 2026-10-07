@echo off
REM Scheduled bet-settlement wrapper (registered in Windows Task Scheduler, every 4h).
REM Runs from the project root (this file's parent folder) so credentials.env and data/betting/bet_log.csv resolve.
cd /d "%~dp0.."

REM Use the per-user Python install when present, otherwise the py launcher.
set "PYTHON_EXE=%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=py"
echo ============================================================ >> "logs\settlement.log"
echo [%DATE% %TIME%] Running scheduled settlement... >> "logs\settlement.log"
"%PYTHON_EXE%" "scripts\reconcile_legacy_bets.py" --max-age-hours 72 >> "logs\settlement.log" 2>&1
echo [%DATE% %TIME%] Done (exit %ERRORLEVEL%). >> "logs\settlement.log"
