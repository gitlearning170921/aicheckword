@echo off
cd /d "%~dp0..\.."
echo [git-no_tag] 仅 git push
git push %*
pause
exit /b %ERRORLEVEL%
