@echo off
chcp 65001 >nul
cd /d "%~dp0..\.."
title AI审核工具 - Web UI（局域网）

echo ========================================
echo   aicheckword - Streamlit 局域网模式
echo   本机: http://127.0.0.1:8501
echo   局域网: http://本机IP:8501
echo ========================================
echo.

if exist "venv\Scripts\activate.bat" call venv\Scripts\activate.bat
streamlit run src/app.py --server.address=0.0.0.0
pause
