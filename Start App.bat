@echo off
setlocal
cd /d "%~dp0"
title Product Image Quality Agent

echo ==============================================
echo    PRODUCT IMAGE QUALITY AGENT
echo ==============================================
echo.

REM --- App already running? Sirf browser kholo ---
netstat -an | findstr ":8000" | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo The app is already running - opening it in your browser...
    start "" http://127.0.0.1:8000/
    ping -n 4 127.0.0.1 >nul
    exit /b 0
)

if not exist ".venv\Scripts\python.exe" goto SETUP
goto RUN

:SETUP
echo ==============================================
echo    FIRST TIME SETUP (one time only)
echo =============================================
echo Installing everything needed - this takes 5 to 10 minutes.
echo Please keep this window open and wait.
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
if not exist ".venv\Scripts\python.exe" goto FAIL
echo.
echo Setup finished successfully!
echo.
goto RUN

:FAIL
echo.
echo ===========================================================
echo  SETUP DID NOT FINISH.
echo.
echo  Most likely Python is not installed on this computer.
echo.
echo  1. Install Python 3.10 or newer from:
echo     https://www.python.org/downloads/
echo     IMPORTANT: tick "Add Python to PATH" in the installer.
echo.
echo  2. If Windows shows a security warning when you double-click
echo     this file, click "More info" then "Run anyway".
echo.
echo  3. Then double-click "Start App.bat" again.
echo ===========================================================
pause
exit /b 1

:RUN
echo Starting the app... your browser will open automatically.
echo.
echo  - Keep this window open (it runs the app).
echo  - Close this window to stop the app.
echo.
start "" powershell -NoProfile -Command "Start-Sleep -Seconds 6; Start-Process 'http://127.0.0.1:8000/'"
".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
echo.
echo The app has stopped. Double-click "Start App.bat" to start it again.
pause
exit /b 0
