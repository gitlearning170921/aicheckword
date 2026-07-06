@echo off
chcp 65001 >nul
cd /d "%~dp0..\.."
title AI审核工具 - Web UI

echo ========================================
echo   aicheckword - Streamlit Web UI
echo   http://127.0.0.1:8501
echo ========================================
echo.

if exist "venv\Scripts\activate.bat" call venv\Scripts\activate.bat
streamlit run src/app.py
pause
