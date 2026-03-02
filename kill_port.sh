#!/bin/bash
# 脚本：强制杀掉占用 8000 端口的进程

PORT=${1:-8000}

echo "🔍 查找占用端口 $PORT 的进程..."

if [ "$(uname)" = "Darwin" ] || [ "$(uname -s)" = "Linux" ]; then
    # macOS / Linux 环境
    PID=$(lsof -ti:$PORT)

    if [ -z "$PID" ]; then
        echo "✅ 端口 $PORT 未被占用"
        exit 0
    fi

    echo "⚠️  发现进程 PID: $PID"

    # 显示进程信息
    echo "📋 进程详情："
    ps -p $PID -o pid,comm,user,etime

    # 强制杀掉进程
    echo "💀 强制终止进程..."
    kill -9 $PID

    if [ $? -eq 0 ]; then
        echo "✅ 端口 $PORT 已释放"
    else
        echo "❌ 终止进程失败，请使用 sudo 权限重试"
        exit 1
    fi

elif [ "$(uname)" = "MINGW64_NT"* ] || [ "$(uname)" = "MSYS_NT"* ] || [ "$(uname)" = "CYGWIN_NT"* ]; then
    # Windows 环境（Git Bash / MSYS2）
    PID=$(netstat -ano | grep ":$PORT " | grep "LISTENING" | awk '{print $5}' | head -1)

    if [ -z "$PID" ]; then
        echo "✅ 端口 $PORT 未被占用"
        exit 0
    fi

    echo "⚠️  发现进程 PID: $PID"

    # 显示进程信息
    echo "📋 进程详情："
    tasklist | findstr "$PID"

    # 强制杀掉进程
    echo "💀 强制终止进程..."
    taskkill //F //PID $PID

    if [ $? -eq 0 ]; then
        echo "✅ 端口 $PORT 已释放"
    else
        echo "❌ 终止进程失败，请以管理员身份运行"
        exit 1
    fi

elif [ "$(expr substr $(uname -s) 1 7)" = "Windows" ]; then
    # 原生 Windows 环境（使用 PowerShell）
    powershell -Command "Get-Process -Id (Get-NetTCPConnection -LocalPort $PORT).OwningProcess -ErrorAction SilentlyContinue | Stop-Process -Force"
    if [ $? -eq 0 ]; then
        echo "✅ 端口 $PORT 已释放"
    else
        echo "✅ 端口 $PORT 未被占用或已释放"
    fi
else
    echo "❌ 未知操作系统，无法确定如何终止进程"
    exit 1
fi
