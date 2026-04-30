@echo off
chcp 65001 >nul
title AI��˹��� - ����API
setlocal

REM Ensure a persistent log window when double-clicked.
REM If not already running inside a dedicated console, relaunch in a new cmd /k window.
if /i not "%~1"=="--in-window" (
    rem Relaunch in a persistent window; avoid tricky nested quotes.
    start "aicheckword-api-logs" cmd /k call "%~f0" --in-window %*
    exit /b 0
)

REM Usage:
REM   restart_api.bat                 -> restart API on port 8000
REM   restart_api.bat 9000            -> restart API on port 9000

set "PORT=%~2"
if "%PORT%"=="" set "PORT=8000"
set "PORT=%PORT:"=%"
set "PORT=%PORT:"=%"

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

echo [INFO] Stopping API on port %PORT% (if any)...
call "%~dp0stop_api.bat" "%PORT%" 1

echo.
echo ========================================
echo   Starting API (logs in this window)
echo ========================================
echo [INFO] API_PORT=%PORT%
echo [INFO] Base URL: http://127.0.0.1:%PORT%
echo [INFO] Swagger:  http://127.0.0.1:%PORT%/docs
echo [INFO] Health:   http://127.0.0.1:%PORT%/status
echo ========================================
echo.

set API_PORT=%PORT%
set PYTHONUTF8=1

echo [INFO] Starting: python -m src.api.server
python -m src.api.server
if errorlevel 1 (
    echo [WARN] python failed, trying: py -3 -m src.api.server
    py -3 -m src.api.server
)

echo.
echo [WARN] API process exited.
endlocal
pause