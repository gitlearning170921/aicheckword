@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo   注册文档审核工具 - 重启服务
echo ========================================
echo.

:: 先停止（传入参数以跳过 pause）
call "%~dp0stop.bat" 1
timeout /t 2 /nobreak >nul
echo.

:: 再启动
start "AI审核工具" cmd /k "%~dp0start.bat"
echo 已在新窗口中启动服务
pause
