@echo off
chcp 65001 >nul
cd /d "%~dp0..\.."

echo [stop_all] Streamlit 8501 + API 8000
set FOUND=0
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8501" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a 2>nul && set FOUND=1
)
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8000" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a 2>nul && set FOUND=1
)
if %FOUND%==0 echo 未发现运行中的服务
if "%1"=="" pause
