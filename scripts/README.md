# 脚本目录

此目录包含用于在不同操作系统上快速设置开发环境的脚本。

## 脚本列表

### Windows
- `setup_env_windows.bat` - 用于在Windows系统上设置开发环境

使用方法：
```cmd
setup_env_windows.bat
```

### Linux
- `setup_env_linux.sh` - 用于在Linux系统上设置开发环境

使用方法：
```bash
chmod +x setup_env_linux.sh
./setup_env_linux.sh
```

### macOS
- `setup_env_macos.sh` - 用于在macOS系统上设置开发环境

使用方法：
```bash
chmod +x setup_env_macos.sh
./setup_env_macos.sh
```

## 功能说明

这些脚本会自动执行以下操作：
1. 检查Python 3.9+是否已安装
2. 安装`uv`包管理器（如果未安装）
3. 创建虚拟环境
4. 安装项目依赖
5. 如果不存在，则从`.env.example`创建`.env`文件

## 注意事项

- 在运行脚本前，请确保您具有足够的权限
- 对于Linux/macOS系统，可能需要运行`chmod +x script_name.sh`给脚本添加执行权限
- 如果脚本无法正常工作，可以参考主README.md手动安装依赖