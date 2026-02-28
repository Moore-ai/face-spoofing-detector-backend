from contextlib import asynccontextmanager
from typing import Annotated
from fastapi import FastAPI, Depends

import logging

logger = logging.getLogger(__name__)

from service import infer_service
from util.websocket_manager import ConnectionManager, connection_manager


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """应用生命周期管理"""
    global _infer_service
    try:
        logger.info("Initializing inference service...")
        from util.config import settings

        _infer_service = infer_service.InferService.create(
            settings.MODEL_SINGLE_PATH,
            settings.MODEL_FUSION_PATH,
        )
        logger.info("Inference service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize inference service: {e}", exc_info=True)
        raise

    # 初始化数据库
    try:
        logger.info("Initializing database...")
        from db import db_manager

        db_manager.initialize()
        db_manager.create_tables()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}", exc_info=True)
        # 数据库初始化失败不影响服务启动，继续运行

    yield
    logger.info("Shutting down")

    # 关闭数据库连接
    try:
        from db import db_manager
        db_manager.close()
        logger.info("Database connection closed")
    except Exception as e:
        logger.error(f"Error closing database connection: {e}")


# infer_service
def create_infer_service():
    return _infer_service


def create_connection_manager():
    return connection_manager


InferServiceDep = Annotated[infer_service.InferService, Depends(create_infer_service)]
ConnectionManagerDep = Annotated[ConnectionManager, Depends(create_connection_manager)]
