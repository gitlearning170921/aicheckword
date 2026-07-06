@echo off
chcp 65001 >nul
title aicheckword - API
setlocal

if /i not "%~1"=="--in-window" (
    start "aicheckword-api-logs" cmd /k call "%~f0" --in-window %*
    exit /b 0
)

set "PORT=%~2"
if "%PORT%"=="" set "PORT=8000"
set "PORT=%PORT:"=%"

cd /d "%~dp0..\.."
if exist "venv\Scripts\activate.bat" call venv\Scripts\activate.bat

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] python not in PATH
    pause & exit /b 1
)

echo [start_api] port %PORT%  Swagger: http://127.0.0.1:%PORT%/docs
call "%~dp0stop_api.bat" "%PORT%" 1

set API_PORT=%PORT%
set PYTHONUTF8=1
python -m src.api.server
if errorlevel 1 py -3 -m src.api.server

echo [WARN] API exited.
pause
endlocal
