@echo off
chcp 65001 >nul
cd /d "%~dp0..\.."

echo ========================================
echo  [git-no_tag] aicheckword 日常提交（不打 tag）
echo  仓库: %CD%
echo ========================================
echo 发版 tag 由 aiword\dev\git-tag_release\release.bat 统一打
echo.

if "%~1"=="" (
    set /p MSG="请输入提交说明: "
) else (
    set "MSG=%~1"
)
if "%MSG%"=="" ( echo 已取消 & pause & exit /b 1 )

git add -A
git commit -m "%MSG%"
if errorlevel 1 ( echo 无变更或提交失败 & pause & exit /b 1 )
git push
if errorlevel 1 ( pause & exit /b 1 )
echo 完成（未打 tag）。
pause
