@echo off
cd /d "%~dp0"
echo Starting FaceMakeIt at http://localhost:8000
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
pause
