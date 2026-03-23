@echo off
chcp 65001 >nul
title AI审核工具 - Web UI（局域网可访问）
cd /d "%~dp0"

echo ========================================
echo   注册文档审核工具 - 局域网模式
echo ========================================
echo.
echo 本机浏览器请打开:  http://localhost:8501
echo 或:               http://127.0.0.1:8501
echo.
echo 其他电脑请打开:  http://本机局域网IP:8501
echo （在 cmd 执行 ipconfig 查看 IPv4 地址；防火墙需放行 8501）
echo.
echo 注意: 终端里若显示 http://0.0.0.0:8501 请忽略，勿在浏览器输入 0.0.0.0
echo ========================================
echo.

if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

streamlit run src/app.py --server.address=0.0.0.0
pause
