@echo off
chcp 65001 >nul
title AI审核工具 - Web UI
cd /d "%~dp0"

echo ========================================
echo   注册文档审核工具 - 启动服务
echo ========================================
echo.

:: 激活虚拟环境（如有）
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

:: 若已安装 protobuf 3.20.x 仍报 Descriptors 错误，可取消下一行（纯 Python 解析，略慢）
:: set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

streamlit run src/app.py
pause
