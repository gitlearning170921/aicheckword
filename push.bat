@echo off
title Git push
cd /d "%~dp0"

echo ========================================
echo   git push only (no add/commit)
echo ========================================
echo.

git push %*
set "ERR=%ERRORLEVEL%"

echo.
if %ERR% neq 0 (
    echo [FAILED] exit code: %ERR%
) else (
    echo [OK] push finished.
)
echo.
echo ========================================
echo   See output above. Press any key to close.
echo ========================================
pause
exit /b %ERR%
