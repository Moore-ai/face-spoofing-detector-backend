"""
配置管理模块
从环境变量和.env文件读取配置，提供默认值
"""

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


# 配置模块级日志（使用 NullHandler，避免在不需要日志时输出到控制台）
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# 加载.env文件（如果存在）
env_path = Path(__file__).parent.parent / ".env"

if env_path.exists():
    loaded = load_dotenv(dotenv_path=env_path)
    if loaded:
        logger.info(f"Loaded environment variables from: {env_path}")
    else:
        logger.warning(f"Failed to load environment variables from: {env_path}")
else:
    logger.error(f"Environment file not found: {env_path}")
    logger.info("Using default configuration values")


class Config:
    """应用配置类"""

    # 模型路径
    MODEL_SINGLE_PATH: str = os.getenv("MODEL_SINGLE_PATH", "models/single_model.pth")
    MODEL_FUSION_PATH: str = os.getenv("MODEL_FUSION_PATH", "models/fusion_model.pth")

    # 服务器配置
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))

    # 日志配置
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # 推理设备
    DEVICE: str = os.getenv("DEVICE", "cuda")

    # 模型输入尺寸
    INPUT_SIZE: int = int(os.getenv("INPUT_SIZE", "112"))

    # 认证与安全配置
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
    JWT_EXPIRATION_HOURS: int = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))
    ADMIN_USERNAME: str = os.getenv("ADMIN_USERNAME", "admin")
    ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "your-admin-password-change-in-production")
    DEFAULT_ACTIVATION_CODE: str = os.getenv("DEFAULT_ACTIVATION_CODE", "ACT-DEV-DEFAULT-KEY")

    # 调试模式配置
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"
    DEBUG_DELAY_PER_IMAGE: float = float(os.getenv("DEBUG_DELAY_PER_IMAGE", "0.5"))
    DEBUG_DELAY_PER_PAIR: float = float(os.getenv("DEBUG_DELAY_PER_PAIR", "0.8"))
    DEBUG_FAILURE_RATE: float = float(os.getenv("DEBUG_FAILURE_RATE", "0.0"))

    # 日志配置（新增）
    LOG_JSON_FORMAT: bool = os.getenv("LOG_JSON_FORMAT", "false").lower() == "true"
    LOG_TO_CONSOLE: bool = os.getenv("LOG_TO_CONSOLE", "true").lower() == "true"
    LOG_REQUEST_BODY: bool = os.getenv("LOG_REQUEST_BODY", "false").lower() == "true"
    LOG_RESPONSE_BODY: bool = os.getenv("LOG_RESPONSE_BODY", "false").lower() == "true"
    AUDIT_LOG_ENABLED: bool = os.getenv("AUDIT_LOG_ENABLED", "true").lower() == "true"

    # 重试配置
    RETRY_ENABLED: bool = os.getenv("RETRY_ENABLED", "true").lower() == "true"
    RETRY_MAX_ATTEMPTS: int = int(os.getenv("RETRY_MAX_ATTEMPTS", "3"))
    RETRY_DELAY_SECONDS: float = float(os.getenv("RETRY_DELAY_SECONDS", "1.0"))
    RETRY_EXPONENTIAL_BACKOFF: bool = os.getenv("RETRY_EXPONENTIAL_BACKOFF", "true").lower() == "true"
    RETRY_MAX_DELAY_SECONDS: float = float(os.getenv("RETRY_MAX_DELAY_SECONDS", "10.0"))

    # 数据库配置（新增）
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./db/detections.db")

    # 图片存储配置
    STORAGE_TYPE: str = os.getenv("STORAGE_TYPE", "local")  # "local" 或 "s3"
    STORAGE_LOCAL_PATH: str = os.getenv("STORAGE_LOCAL_PATH", "storage/images")
    STORAGE_QUOTA_BYTES: Optional[int] = None
    STORAGE_RETENTION_DAYS: int = int(os.getenv("STORAGE_RETENTION_DAYS", "30"))

    # S3 存储配置（可选）
    S3_BUCKET: str = os.getenv("S3_BUCKET", "")
    S3_REGION: str = os.getenv("S3_REGION", "us-east-1")
    S3_ENDPOINT_URL: Optional[str] = os.getenv("S3_ENDPOINT_URL")
    S3_ACCESS_KEY: Optional[str] = os.getenv("S3_ACCESS_KEY")
    S3_SECRET_KEY: Optional[str] = os.getenv("S3_SECRET_KEY")

    # 图片自动保存配置（方案 A：服务端控制策略）
    STORAGE_AUTO_SAVE: bool = os.getenv("STORAGE_AUTO_SAVE", "true").lower() == "true"
    STORAGE_SAVE_STRATEGY: str = os.getenv("STORAGE_SAVE_STRATEGY", "error_only")  # never, always, error_only, smart
    STORAGE_SAVE_ERROR_RATE: float = float(os.getenv("STORAGE_SAVE_ERROR_RATE", "1.0"))
    STORAGE_SAVE_FAKE_RATE: float = float(os.getenv("STORAGE_SAVE_FAKE_RATE", "0.1"))
    STORAGE_SAVE_REAL_RATE: float = float(os.getenv("STORAGE_SAVE_REAL_RATE", "0.01"))
    STORAGE_SAVE_LOW_CONFIDENCE_THRESHOLD: float = float(os.getenv("STORAGE_SAVE_LOW_CONFIDENCE_THRESHOLD", "0.6"))
    STORAGE_MAX_PER_TASK: int = int(os.getenv("STORAGE_MAX_PER_TASK", "10"))

    # 图片压缩配置
    IMAGE_COMPRESS_ENABLED: bool = os.getenv("IMAGE_COMPRESS_ENABLED", "true").lower() == "true"
    IMAGE_COMPRESS_QUALITY: int = int(os.getenv("IMAGE_COMPRESS_QUALITY", "75"))
    IMAGE_COMPRESS_TYPE: str = os.getenv("IMAGE_COMPRESS_TYPE", "opencv")  # opencv, pillow, resize
    IMAGE_COMPRESS_MAX_WIDTH: Optional[int] = int(os.getenv("IMAGE_COMPRESS_MAX_WIDTH") or 0) or None
    IMAGE_COMPRESS_MAX_HEIGHT: Optional[int] = int(os.getenv("IMAGE_COMPRESS_MAX_HEIGHT") or 0) or None

    # 任务调度器配置
    TASK_SCHEDULER_MAX_WORKERS: int = int(os.getenv("TASK_SCHEDULER_MAX_WORKERS", "1"))
    TASK_SCHEDULER_ENABLE_BATCH_PROCESSING: bool = os.getenv("TASK_SCHEDULER_ENABLE_BATCH_PROCESSING", "true").lower() == "true"


# 导出配置实例
settings = Config()
