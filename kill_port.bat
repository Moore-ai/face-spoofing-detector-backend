:: 脚本：强制杀掉占用 8000 端口的进程
@echo off
setlocal enabledelayedexpansion

chcp 65001 >nul

:: 默认端口 8000，允许通过参数指定
set PORT=%~1
if "%PORT%"=="" set PORT=8000

echo.
echo ========================================
echo   端口 8000 进程终止工具
echo ========================================
echo.

echo [查找] 正在查找占用端口 %PORT% 的进程...

:: 查找占用指定端口的进程
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%PORT% " ^| findstr "LISTENING"') do (
    set PID=%%a
    goto FOUND
)

echo [信息] 端口 %PORT% 未被占用
echo [完成] 操作成功
goto END

:FOUND
echo [发现] 进程 PID: %PID%

:: 显示进程信息
echo.
echo [进程] 进程详情：
tasklist /FI "PID eq %PID%" /FO TABLE

echo.
choice /C YN /M "确定要强制终止此进程吗？"
if errorlevel 2 goto CANCEL

echo.
echo [终止] 强制终止进程...

:: 强制杀掉进程
taskkill /F /PID %PID%

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [成功] 端口 %PORT% 已释放
    echo [完成] 操作成功
) else (
    echo.
    echo [错误] 终止进程失败
    echo [提示] 请尝试以管理员身份运行此脚本
    exit /b 1
)

goto END

:ABORT
echo.
echo [取消] 用户取消操作
goto END

:END
echo.
pause
