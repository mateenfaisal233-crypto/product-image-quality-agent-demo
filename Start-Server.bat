@echo off
cd /d "%~dp0"
echo Starting Product Image Quality Agent server...
echo Browser kholein: http://127.0.0.1:8000
start "" http://127.0.0.1:8000
".venv\Scripts\python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port 8000
pause
