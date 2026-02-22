#!/bin/bash

# Linux环境配置脚本
# 用于设置和安装项目依赖

echo "正在配置Linux环境..."

# 检查是否安装了Python
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python，请先安装Python 3.9或更高版本"
    exit 1
fi

# 检查是否安装了uv
if ! command -v uv &> /dev/null; then
    echo "正在安装uv包管理器..."
    pip install uv || python3 -m pip install uv
fi

# 创建虚拟环境并安装依赖
echo "正在创建虚拟环境并安装依赖..."
uv venv
if [ $? -ne 0 ]; then
    echo "虚拟环境创建失败"
    exit 1
fi

# 激活虚拟环境并安装项目依赖
source .venv/bin/activate
uv sync

# 检查是否有.env文件，如果不存在则复制示例文件
if [ ! -f ".env" ]; then
    echo "创建环境配置文件..."
    cp .env.example .env
fi

echo "Linux环境配置完成！"
echo ""
echo "要激活虚拟环境，请运行: source .venv/bin/activate"
echo "要启动服务，请运行: uv run python main.py"