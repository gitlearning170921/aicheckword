@echo off
chcp 65001 >nul
cd /d "%~dp0..\.."

call "%~dp0stop_all.bat" 1
timeout /t 2 /nobreak >nul
start "aicheckword-web" cmd /k call "%~dp0start_streamlit.bat"
echo 已在新窗口启动 Streamlit
pause
