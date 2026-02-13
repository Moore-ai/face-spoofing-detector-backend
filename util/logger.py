import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(level: int = logging.INFO) -> None:
    """配置应用日志"""
    # 创建日志目录
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # 统一的日志格式
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 根日志配置
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 检查是否已有控制台处理器，避免重复添加
    has_console_handler = any(
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
        for handler in root_logger.handlers
    )

    if not has_console_handler:
        # 控制台输出
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        root_logger.addHandler(console)

    # 文件输出（文件名包含启动时间，精确到秒）
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"{timestamp}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
