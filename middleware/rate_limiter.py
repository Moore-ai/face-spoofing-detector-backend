"""
速率限制中间件模块

实现基于滑动窗口的请求限流：
- 支持基于 IP 或 API Key 的限流
- 可配置不同端点的限流策略
- 返回标准限流响应头
"""

import time
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from fastapi import Request, HTTPException
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware

from util.config import settings

logger = logging.getLogger(__name__)

@dataclass
class RateLimitConfig:
    """限流配置"""

    requests: int  # 允许的请求数
    window_seconds: int  # 时间窗口（秒）

    @classmethod
    def per_minute(cls, n: int) -> "RateLimitConfig":
        """每分钟 n 次"""
        return cls(requests=n, window_seconds=60)

    @classmethod
    def per_second(cls, n: int) -> "RateLimitConfig":
        """每秒 n 次"""
        return cls(requests=n, window_seconds=1)

    @classmethod
    def per_hour(cls, n: int) -> "RateLimitConfig":
        """每小时 n 次"""
        return cls(requests=n, window_seconds=3600)


# 默认限流配置
DEFAULT_RATE_LIMIT = RateLimitConfig.per_minute(60)  # 默认每分钟 60 次
AUTH_RATE_LIMIT = RateLimitConfig.per_minute(10)  # 认证端点更严格
INFER_RATE_LIMIT = RateLimitConfig.per_minute(30)  # 推理端点


@dataclass
class RequestRecord:
    """请求记录"""

    timestamps: list[float] = field(default_factory=list)

    def cleanup(self, window_seconds: int) -> None:
        """清理过期的时间戳"""
        cutoff = time.time() - window_seconds
        self.timestamps = [t for t in self.timestamps if t > cutoff]

    def add_request(self) -> int:
        """添加请求并返回当前窗口内的请求数"""
        now = time.time()
        self.timestamps.append(now)
        return len(self.timestamps)


class RateLimiter:
    """速率限制器"""

    def __init__(self):
        # 按 key 存储请求记录：{identifier: RequestRecord}
        self._records: dict[str, RequestRecord] = defaultdict(RequestRecord)

    def _get_identifier(self, request: Request) -> str:
        """获取限流标识符（基于 IP 或 API Key）"""
        # 优先使用 API Key（如果有）
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api_key:{api_key}"

        # 否则使用 IP 地址
        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"

    def _get_rate_limit_for_path(self, path: str) -> RateLimitConfig:
        """根据路径获取限流配置"""
        if path.startswith("/auth"):
            return AUTH_RATE_LIMIT
        elif path.startswith("/infer"):
            return INFER_RATE_LIMIT
        else:
            return DEFAULT_RATE_LIMIT

    def check_rate_limit(
        self,
        request: Request,
        override_config: Optional[RateLimitConfig] = None,
    ) -> tuple[bool, int, int]:
        """
        检查请求是否超过限制

        Returns:
            (allowed, remaining, retry_after):
            - allowed: 是否允许请求
            - remaining: 剩余请求数
            - retry_after: 重试等待时间（秒）
        """
        identifier = self._get_identifier(request)
        config = override_config or self._get_rate_limit_for_path(request.url.path)

        record = self._records[identifier]
        record.cleanup(config.window_seconds)

        current_count = len(record.timestamps)
        remaining = max(0, config.requests - current_count)

        if current_count >= config.requests:
            # 计算重试等待时间
            if record.timestamps:
                oldest = min(record.timestamps)
                retry_after = int(oldest + config.window_seconds - time.time()) + 1
            else:
                retry_after = config.window_seconds

            return False, 0, retry_after

        # 允许请求，记录时间戳
        record.add_request()
        remaining = max(0, config.requests - len(record.timestamps))

        return True, remaining, 0


# 全局速率限制器实例
rate_limiter = RateLimiter()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """速率限制中间件"""

    def __init__(
        self,
        app,
        enabled: bool = True,
        default_limit: RateLimitConfig = DEFAULT_RATE_LIMIT,
    ):
        super().__init__(app)
        self.enabled = enabled
        self.default_limit = default_limit

    async def dispatch(self, request: Request, call_next):
        # 如果未启用，直接放行
        if not self.enabled:
            return await call_next(request)

        # 检查速率限制
        allowed, remaining, retry_after = rate_limiter.check_rate_limit(request)

        if not allowed:
            # 记录审计日志
            try:
                from util.audit import audit_logger_instance
                api_key = request.headers.get("X-API-Key")
                api_key_prefix = api_key[:12] + "..." if api_key and len(api_key) > 12 else None
                client_ip = request.client.host if request.client else "unknown"

                audit_logger_instance.log_rate_limited(
                    client_ip=client_ip,
                    path=request.url.path,
                    api_key_prefix=api_key_prefix,
                )
            except Exception as e:
                # 审计日志失败不影响主流程
                logger.debug(f"审计日志记录失败：{e}")

            # 返回 429 Too Many Requests
            response = Response(
                content='{"detail": "请求过于频繁，请稍后重试"}',
                status_code=429,
                media_type="application/json",
            )
            response.headers["X-RateLimit-Limit"] = str(
                self.default_limit.requests
            )
            response.headers["X-RateLimit-Remaining"] = "0"
            response.headers["X-RateLimit-Reset"] = str(retry_after)
            response.headers["Retry-After"] = str(retry_after)
            return response

        # 继续处理请求
        response = await call_next(request)

        # 添加限流响应头
        response.headers["X-RateLimit-Limit"] = str(
            self.default_limit.requests
        )
        response.headers["X-RateLimit-Remaining"] = str(remaining)

        return response


# 依赖注入版本的速率限制（用于特定路由）
async def rate_limit_dependency(
    request: Request,
    limit: Optional[int] = None,
    window: Optional[int] = None,
) -> None:
    """
    速率限制依赖（用于特定路由）

    用法:
        @router.get("/sensitive")
        async def sensitive_endpoint(
            _: Annotated[None, Depends(rate_limit_dependency)]
        ):
            ...
    """
    if limit is not None and window is not None:
        config = RateLimitConfig(requests=limit, window_seconds=window)
        allowed, _, retry_after = rate_limiter.check_rate_limit(
            request, override_config=config
        )
    else:
        allowed, _, retry_after = rate_limiter.check_rate_limit(request)

    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="请求过于频繁，请稍后重试",
            headers={
                "Retry-After": str(retry_after),
            },
        )
