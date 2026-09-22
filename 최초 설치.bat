@echo off
cd /d "%~dp0"
if not exist ".venv" python -m venv .venv
".venv\Scripts\python.exe" -m pip install -r requirements.txt
".venv\Scripts\python.exe" download_model.py
echo.
echo Setup done. Run "STT 실행.bat".
pause
