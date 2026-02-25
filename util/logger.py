import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any


class JSONFormatter(logging.Formatter):
    """JSON 日志格式化器

    将日志记录为 JSON 格式，便于 ELK 等日志分析工具采集
    """

    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录为 JSON"""
        log_data: dict[str, Any] = {
            "timestamp": record.created,
            "timestamp_iso": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # 添加额外字段
        if hasattr(record, "req_id"):
            log_data["req_id"] = record.req_id
        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id

        # 添加位置信息（调试时有用）
        if record.levelno >= logging.ERROR:
            log_data["location"] = {
                "filename": record.filename,
                "lineno": record.lineno,
                "funcName": record.funcName,
            }

        # 添加异常堆栈
        if record.exc_info:
            log_data["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False)


class FileFormatter(logging.Formatter):
    """文件日志格式化器

    人类可读的传统格式，用于文件输出
    """

    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录"""
        return (
            f"{self.formatTime(record)} - {record.name} - "
            f"{record.levelname} - {record.getMessage()}"
        )


def setup_logging(
    level: int = logging.INFO,
    json_format: bool = False,
    log_dir: str | None = None,
) -> None:
    """配置应用日志

    Args:
        level: 日志级别
        json_format: 是否使用 JSON 格式（默认 False，使用传统格式）
        log_dir: 日志目录，默认使用 logs/

    注意：日志仅输出到文件，不输出到控制台
    """
    # 创建日志目录
    if log_dir:
        log_path = Path(log_dir)
    else:
        log_path = Path("logs")
    log_path.mkdir(exist_ok=True)

    # 根日志配置
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 清除现有处理器（避免重复）
    root_logger.handlers.clear()

    # 选择格式化器
    if json_format:
        file_formatter = JSONFormatter()
    else:
        file_formatter = FileFormatter()

    # 文件输出（文件名包含启动时间，精确到秒）
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_path / f"{timestamp}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # 配置审计日志记录器
    audit_logger = logging.getLogger("audit")
    audit_logger.setLevel(logging.INFO)

    # 审计日志使用独立的 JSON 格式文件
    audit_formatter = JSONFormatter()
    audit_file = log_path / f"{timestamp}_audit.log"
    audit_handler = logging.FileHandler(audit_file, encoding="utf-8")
    audit_handler.setFormatter(audit_formatter)
    audit_logger.addHandler(audit_handler)

    # 避免审计日志传播到根日志器（避免重复记录）
    audit_logger.propagate = False
