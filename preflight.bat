@echo off
chcp 65001 >nul
cd /d "%~dp0"
python scripts\preflight_cursor.py
set EXITCODE=%ERRORLEVEL%
if %EXITCODE% neq 0 exit /b %EXITCODE%
exit /b 0
