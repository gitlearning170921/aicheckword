@echo off
chcp 65001 >nul
title AI审核工具 - 启动全部(Web+API)
cd /d "%~dp0"

echo ========================================
echo   注册文档审核工具 - 启动全部服务
echo ========================================
echo.
echo [INFO] Web UI: http://127.0.0.1:8501
echo [INFO] API:    http://127.0.0.1:8000
echo.

REM Activate venv if exists (for current window)
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Start API in new window (prefer restart_api.bat so user has one entry)
start "aicheckword-api-logs" cmd /k "cd /d %~dp0 && call restart_api.bat"

REM Start Streamlit in current window
streamlit run src/app.py
pause

