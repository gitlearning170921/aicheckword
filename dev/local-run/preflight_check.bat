@echo off
chcp 65001 >nul
cd /d "%~dp0..\.."
python scripts\preflight_cursor.py
exit /b %ERRORLEVEL%
