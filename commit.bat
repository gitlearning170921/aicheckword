@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo   提交代码到 Git
echo ========================================
echo.

if "%~1"=="" (
    set /p MSG="请输入提交说明: "
) else (
    set "MSG=%~1"
)

if "%MSG%"=="" (
    echo 未输入提交说明，已取消。
    pause
    exit /b 1
)

echo.
echo 执行: git add .
git add .
echo.
echo 执行: git commit -m "%MSG%"
git commit -m "%MSG%"
if errorlevel 1 (
    echo.
    echo 提交失败或没有变更可提交。
    pause
    exit /b 1
)
echo.
echo 执行: git push
git push
echo.
echo 提交完成。
pause
