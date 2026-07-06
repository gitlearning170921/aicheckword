@echo off
chcp 65001 >nul
cd /d "%~dp0..\.."

echo [start_all] Web 8501 + API 8000
if exist "venv\Scripts\activate.bat" call venv\Scripts\activate.bat
start "aicheckword-api-logs" cmd /k "cd /d %CD% && call dev\local-run\start_api.bat --in-window"
streamlit run src/app.py
pause
