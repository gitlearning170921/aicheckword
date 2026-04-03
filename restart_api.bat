@echo off
chcp 65001 >nul
title AI审核工具 - 重启API
setlocal enabledelayedexpansion

REM Usage:
REM   restart_api.bat            -> restart API on port 8000
REM   restart_api.bat 9000       -> restart API on port 9000

set "PORT=%~1"
if "%PORT%"=="" set "PORT=8000"

echo [INFO] Target API port: %PORT%

REM Move to project root (where this .bat is located)
cd /d "%~dp0"

REM Try activate venv first
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating venv...
    call venv\Scripts\activate.bat
)

where python >nul 2>&1
set "HAS_PYTHON=%errorlevel%"
where py >nul 2>&1
set "HAS_PY=%errorlevel%"
if not "%HAS_PYTHON%"=="0" if not "%HAS_PY%"=="0" (
    echo [ERROR] Neither `python` nor `py` was found in PATH.
    echo [HINT] Install Python or add it to PATH, then retry.
    endlocal
    pause
    exit /b 1
)

echo [INFO] Looking for existing process on port %PORT%...
set "FOUND_PID="
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /r /c:":%PORT% .*LISTENING"') do (
    set "FOUND_PID=%%p"
    echo [INFO] Stopping PID !FOUND_PID! ...
    taskkill /PID !FOUND_PID! /F >nul 2>&1
    if !errorlevel! equ 0 (
        echo [OK] PID !FOUND_PID! stopped.
    ) else (
        echo [WARN] Failed to stop PID !FOUND_PID! (may already exit).
    )
)

if "%FOUND_PID%"=="" (
    echo [INFO] No listening process found on port %PORT%.
)

echo [INFO] Opening API log window...
echo [INFO] Keep that window open to view logs.

start "aicheckword-api-logs" cmd /k "cd /d %~dp0 && set API_PORT=%PORT% && set PYTHONUTF8=1 && echo [INFO] API_PORT=%PORT% && echo [INFO] Starting: python -m src.api.server && (python -m src.api.server || (echo [WARN] python failed, trying py -3 ... && py -3 -m src.api.server)) & echo. & echo [WARN] API process exited. & pause"

echo [DONE] API restart command executed.
endlocal
pause