from contextlib import asynccontextmanager
from typing import Annotated
from fastapi import FastAPI, Depends

import logging

logger = logging.getLogger(__name__)

from service import infer_service
from util.websocket_manager import ConnectionManager, connection_manager
from util.config import settings


# 全局变量
_infer_service: infer_service.InferService


async def initialize_inference_service():
    """初始化推理服务"""
    global _infer_service
    logger.info("Initializing inference service...")

    _infer_service = infer_service.InferService.create(
        settings.MODEL_SINGLE_PATH,
        settings.MODEL_FUSION_PATH,
    )
    logger.info("Inference service initialized successfully")


async def initialize_database():
    """初始化数据库"""
    try:
        logger.info("Initializing database...")
        from db import db_manager

        db_manager.initialize()
        db_manager.create_tables()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}", exc_info=True)
        # 数据库初始化失败不影响服务启动，继续运行


async def initialize_image_storage():
    """初始化图片存储"""
    try:
        logger.info("Initializing image storage...")
        from util.storage import storage_manager

        storage_manager.initialize(
            storage_type=settings.STORAGE_TYPE,
            storage_path=settings.STORAGE_LOCAL_PATH,
            quota_bytes=settings.STORAGE_QUOTA_BYTES,
        )
        logger.info("Image storage initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize image storage: {e}", exc_info=True)
        # 存储初始化失败不影响服务启动，继续运行


async def initialize_image_auto_save_service():
    """初始化图片自动存储服务"""
    try:
        logger.info("Initializing image auto-save service...")
        from service.image_auto_save_service import image_auto_save_service
        from util.image_compressor import ImageCompressionConfig
        from util.storage import storage_manager

        # 创建压缩配置
        compression_config = ImageCompressionConfig(
            enabled=settings.IMAGE_COMPRESS_ENABLED,
            compressor_type=settings.IMAGE_COMPRESS_TYPE,
            quality=settings.IMAGE_COMPRESS_QUALITY,
            max_width=settings.IMAGE_COMPRESS_MAX_WIDTH,
            max_height=settings.IMAGE_COMPRESS_MAX_HEIGHT,
        )

        image_auto_save_service.initialize(
            storage_backend=storage_manager,
            compression_config=compression_config,
        )
        logger.info("Image auto-save service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize image auto-save service: {e}", exc_info=True)
        # 自动存储服务初始化失败不影响服务启动，继续运行


async def cleanup_database():
    """清理数据库连接"""
    try:
        from db import db_manager
        db_manager.close()
        logger.info("Database connection closed")
    except Exception as e:
        logger.error(f"Error closing database connection: {e}")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """应用生命周期管理"""
    # 初始化阶段
    try:
        await initialize_inference_service()
        await initialize_database()
        await initialize_image_storage()
        await initialize_image_auto_save_service()
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}", exc_info=True)
        raise

    yield

    # 清理阶段
    logger.info("Shutting down")
    await cleanup_database()


# 依赖注入器
def create_infer_service():
    """创建推理服务依赖"""
    return _infer_service


def create_connection_manager():
    """创建连接管理器依赖"""
    return connection_manager


InferServiceDep = Annotated[infer_service.InferService, Depends(create_infer_service)]
ConnectionManagerDep = Annotated[ConnectionManager, Depends(create_connection_manager)]
