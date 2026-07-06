@echo off
chcp 65001 >nul
cd /d "%~dp0..\.."

call "%~dp0stop_all.bat" 1
timeout /t 2 /nobreak >nul
start "aicheckword-api-logs" cmd /k "cd /d %CD% && call dev\local-run\start_api.bat --in-window"
start "aicheckword-web" cmd /k "cd /d %CD% && call dev\local-run\start_streamlit.bat"
echo 已重启 Web + API
pause
