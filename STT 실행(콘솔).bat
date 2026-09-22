@echo off
cd /d "%~dp0"
echo Running in debug mode (console stays open)...
".venv\Scripts\python.exe" main.py
pause
