from contextlib import asynccontextmanager
from typing import Annotated
from fastapi import FastAPI, Depends

import os
import logging

logger = logging.getLogger(__name__)

from service import infer_service


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


InferServiceDep = Annotated[infer_service.InferService, Depends(create_infer_service)]
