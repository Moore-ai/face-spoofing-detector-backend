from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import traceback
from util.logger import setup_logging
from util.config import settings
from middleware.logging_middleware import RequestLoggingMiddleware, AuditLogMiddleware
from middleware.rate_limiter import RateLimitMiddleware

from lifespan import lifespan
from router import register_routers

# 初始化日志配置
setup_logging(
    level=getattr(logging, settings.LOG_LEVEL),
    json_format=settings.LOG_JSON_FORMAT,
)
logger = logging.getLogger(__name__)

app = FastAPI(lifespan=lifespan)

# =========== 中间件注册 ===========
# 注意：中间件注册顺序很重要，最先注册的在最外层

# 请求日志中间件（最外层，记录所有请求）
app.add_middleware(
    RequestLoggingMiddleware,
    log_request_body=settings.LOG_REQUEST_BODY,
    log_response_body=settings.LOG_RESPONSE_BODY,
)

# 审计日志中间件（记录关键操作）
if settings.AUDIT_LOG_ENABLED:
    app.add_middleware(AuditLogMiddleware)

# 速率限制中间件
app.add_middleware(RateLimitMiddleware, enabled=True)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应配置具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
register_routers(app)


# =========== 异常处理 ===========

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP 异常处理器，记录客户端错误"""
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    api_key = request.headers.get("x-api-key", "")
    api_key_prefix = api_key[:12] + "..." if api_key and len(api_key) > 12 else api_key

    log_msg = (
        f"HTTP {exc.status_code} error at {request.method} {request.url.path} "
        f"from {client_ip} (UA: {user_agent}): {exc.detail}"
    )

    if exc.status_code >= 500:
        logger.error(log_msg)
    elif exc.status_code >= 400:
        logger.warning(log_msg)

    # 记录审计日志（认证失败、未授权等）
    if exc.status_code in [401, 403]:
        from util.audit import audit_logger_instance
        audit_logger_instance.log_auth_failed(
            reason=exc.detail,
            client_ip=client_ip,
            path=request.url.path,
        )

    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器，捕获所有未处理的异常并记录到日志"""
    client_ip = request.client.host if request.client else "unknown"
    api_key = request.headers.get("x-api-key", "")
    api_key_prefix = api_key[:12] + "..." if api_key and len(api_key) > 12 else None

    error_msg = (
        f"Unhandled exception at {request.method} {request.url.path}: {str(exc)}"
    )
    logger.error(error_msg)
    logger.error(traceback.format_exc())

    # 记录审计日志
    from util.audit import audit_logger_instance
    audit_logger_instance.log_error(
        error_message=str(exc),
        path=request.url.path,
        actor_id=f"api_key:{api_key_prefix}" if api_key_prefix else None,
        client_ip=client_ip,
    )

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
