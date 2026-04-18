@echo off
chcp 65001 >nul 2>&1 & REM 切换 UTF-8 编码（支持中文输出）
set "LAUNCH_DIR=%cd%"
cd /d "%~dp0"

REM 检查 venv 是否存在
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] 虚拟环境不存在！请先创建：
    echo   python -m venv venv
    echo   call venv\Scripts\activate.bat
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)

REM 激活虚拟环境并运行
call venv\Scripts\activate.bat
echo [OK] 已激活虚拟环境: %VIRTUAL_ENV%

python agents\chat_agent.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Agent 异常退出 (code: %errorlevel%)
)

pause
