@echo off
chcp 65001 >nul
cd /d "%~dp0..\.."
setlocal

set "PORT=%~1"
if "%PORT%"=="" set "PORT=8000"
set "PORT=%PORT:"=%"

echo [stop_api] port %PORT%
set "FOUND=0"
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":%PORT%" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1 && (
        echo [OK] stopped PID %%a
        set "FOUND=1"
    )
)
if "%FOUND%"=="0" echo 未发现监听 %PORT% 的进程
if "%2"=="" pause
endlocal
