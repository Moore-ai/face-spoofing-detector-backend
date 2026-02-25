"""
配置管理模块
从环境变量和.env文件读取配置，提供默认值
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv


# 配置模块级日志（简单配置，确保加载时的日志能被记录）
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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


# 导出配置实例
settings = Config()
