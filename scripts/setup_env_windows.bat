@echo off
REM Windows环境配置脚本
REM 用于设置和安装项目依赖

echo 正在配置Windows环境...

REM 检查是否安装了Python
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python 3.9或更高版本
    pause
    exit /b 1
)

REM 检查是否安装了uv
uv --version >nul 2>&1
if errorlevel 1 (
    echo 正在安装uv包管理器...
    python -m pip install uv
)

REM 创建虚拟环境并安装依赖
echo 正在创建虚拟环境并安装依赖...
uv venv
if errorlevel 1 (
    echo 虚拟环境创建失败
    pause
    exit /b 1
)

REM 激活虚拟环境并安装项目依赖
call .venv\Scripts\activate.bat
uv sync

REM 检查是否有.env文件，如果不存在则复制示例文件
if not exist ".env" (
    echo 创建环境配置文件...
    copy .env.example .env
)

echo Windows环境配置完成！
echo.
echo 要激活虚拟环境，请运行: call .venv\Scripts\activate.bat
echo 要启动服务，请运行: uv run python main.py
pause