@echo off
chcp 65001 >nul
cd /d "%~dp0"
setlocal

REM Usage:
REM   stop_api.bat        -> stop API on port 8000
REM   stop_api.bat 9000   -> stop API on port 9000

set "PORT=%~1"
if "%PORT%"=="" set "PORT=8000"
set "PORT=%PORT:"=%"

echo ========================================
echo   ע���ĵ���˹��� - ֹͣ API ����
echo ========================================
echo.
echo [INFO] Target API port: %PORT%

set "FOUND=0"
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":%PORT%" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1 && (
        echo [OK] ��ֹͣ API ���� (PID %%a)
        set "FOUND=1"
    )
)

if "%FOUND%"=="0" (
    echo δ���������е� API ����
) else (
    echo.
    echo API ������ֹͣ
)

if "%2"=="" (
    REM no extra args -> keep window
    pause
)
endlocal
