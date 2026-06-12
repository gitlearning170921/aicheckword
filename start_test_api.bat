@echo off
echo [INFO] 已合并到 restart_api.bat；请在 .env 中设置 AIWORD_ENV=test
call "%~dp0restart_api.bat" --in-window
