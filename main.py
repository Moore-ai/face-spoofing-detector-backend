from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
import uvicorn
import logging
import traceback
from util.logger import setup_logging

from lifespan import lifespan
from router import register_routers

from util.config import settings

setup_logging(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger(__name__)

app = FastAPI(lifespan=lifespan)
register_routers(app)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP 异常处理器，记录客户端错误"""
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")

    log_msg = (
        f"HTTP {exc.status_code} error at {request.method} {request.url.path} "
        f"from {client_ip} (UA: {user_agent}): {exc.detail}"
    )

    if exc.status_code >= 500:
        logger.error(log_msg)
    elif exc.status_code >= 400:
        logger.warning(log_msg)

    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器，捕获所有未处理的异常并记录到日志"""
    error_msg = (
        f"Unhandled exception at {request.method} {request.url.path}: {str(exc)}"
    )
    logger.error(error_msg)
    logger.error(traceback.format_exc())

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "服务器内部错误",
            "path": request.url.path,
        },
    )


if __name__ == "__main__":
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
