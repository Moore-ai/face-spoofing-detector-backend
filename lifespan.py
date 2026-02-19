from contextlib import asynccontextmanager
from typing import Annotated
from fastapi import FastAPI, Depends

import logging

logger = logging.getLogger(__name__)

from service import infer_service
from util.websocket_manager import ConnectionManager, connection_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
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

    yield
    logger.info("Shutting down")


# infer_service
def create_infer_service():
    return _infer_service


def create_connection_manager():
    return connection_manager


InferServiceDep = Annotated[infer_service.InferService, Depends(create_infer_service)]
ConnectionManagerDep = Annotated[ConnectionManager, Depends(create_connection_manager)]
