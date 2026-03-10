@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo   注册文档审核工具 - 停止服务
echo ========================================
echo.

set FOUND=0

:: 停止 Streamlit (端口 8501)，netstat 最后一列为 PID
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8501" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a 2>nul && (
        echo [OK] 已停止 Streamlit (PID %%a)
        set FOUND=1
    )
)

:: 停止 API 服务 (端口 8000)
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8000" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a 2>nul && (
        echo [OK] 已停止 API 服务 (PID %%a)
        set FOUND=1
    )
)

if %FOUND%==0 (
    echo 未发现运行中的服务
) else (
    echo.
    echo 服务已停止
)
if "%1"=="" pause
