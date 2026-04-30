@echo off
chcp 65001 >nul
title AI审核工具 - 重启全部(Web+API)
cd /d "%~dp0"

echo ========================================
echo   注册文档审核工具 - 重启全部服务
echo ========================================
echo.

call "%~dp0stop.bat" 1
timeout /t 2 /nobreak >nul

start "aicheckword-api-logs" cmd /k "cd /d %~dp0 && call restart_api.bat"
start "aicheckword-web" cmd /k "cd /d %~dp0 && call start.bat"

echo 已在新窗口中启动 Web + API
pause

