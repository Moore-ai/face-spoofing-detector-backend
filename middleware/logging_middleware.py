"""
请求日志中间件

记录所有 HTTP 请求的详细信息：
- 请求方法、路径、耗时
- 响应状态码
- 用户认证信息（API Key / JWT）
- 客户端 IP、User-Agent

支持 JSON 格式输出，便于 ELK 等日志分析工具采集
"""

import json
import logging
import time
from typing import Any, Callable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """请求日志中间件

    记录每个请求的详细信息，包括：
    - 请求时间戳
    - 请求方法、路径、查询参数
    - 响应状态码
    - 请求耗时（毫秒）
    - 客户端 IP、User-Agent
    - 认证信息（API Key 前缀、JWT 用户 ID）
    - 请求/响应大小（可选）
    """

    def __init__(
        self,
        app: ASGIApp,
        log_request_body: bool = False,
        log_response_body: bool = False,
    ):
        """
        Args:
            app: FastAPI 应用
            log_request_body: 是否记录请求体（默认 False，生产环境建议关闭）
            log_response_body: 是否记录响应体（默认 False，生产环境建议关闭）
        """
        super().__init__(app)
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """处理请求日志"""
        # 请求开始时间
        start_time = time.perf_counter()

        # 提取请求信息
        method = request.method
        path = request.url.path
        query_params = dict(request.query_params)
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")

        # 提取认证信息
        auth_info = self._extract_auth_info(request)

        # 记录请求体（可选）
        request_body = None
        if self.log_request_body and method in ["POST", "PUT", "PATCH"]:
            request_body = await self._get_request_body(request)

        # 执行请求
        response = await call_next(request)

        # 计算耗时
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # 记录响应体（可选）
        response_body = None
        if self.log_response_body:
            response_body = await self._get_response_body(response)

        # 构建日志数据
        log_data: dict[str, Any] = {
            "timestamp": time.time(),
            "type": "request_log",
            "method": method,
            "path": path,
            "query_params": query_params,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
            "client_ip": client_ip,
            "user_agent": user_agent,
            "auth": auth_info,
        }

        # 添加请求/响应体（如果启用）
        if request_body is not None:
            log_data["request_body"] = request_body
        if response_body is not None:
            log_data["response_body"] = response_body

        # 根据状态码选择日志级别
        if response.status_code >= 500:
            log_func = logger.error
        elif response.status_code >= 400:
            log_func = logger.warning
        else:
            log_func = logger.info

        # 记录 JSON 格式日志
        log_func(json.dumps(log_data, ensure_ascii=False))

        return response

    def _extract_auth_info(self, request: Request) -> dict[str, Any]:
        """从请求中提取认证信息（脱敏处理）"""
        auth_info: dict[str, Any] = {"authenticated": False}

        # 尝试提取 API Key
        api_key = request.headers.get("x-api-key")
        if api_key:
            # 只显示前缀用于追踪
            auth_info["api_key_prefix"] = api_key[:12] + "..." if len(api_key) > 12 else api_key
            auth_info["authenticated"] = True
            auth_info["auth_type"] = "api_key"

        # 尝试提取 JWT Token
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            # JWT Token 显示前 20 个字符
            auth_info["token_prefix"] = token[:20] + "..." if len(token) > 20 else token
            auth_info["authenticated"] = True
            auth_info["auth_type"] = "jwt"

        return auth_info

    async def _get_request_body(self, request: Request) -> str | None:
        """获取请求体（脱敏处理）"""
        try:
            body = await request.body()
            body_str = body.decode("utf-8")

            # 尝试解析 JSON 并脱敏敏感字段
            if request.headers.get("content-type") == "application/json":
                try:
                    data = json.loads(body_str)
                    # 脱敏敏感字段
                    for field in ["password", "api_key", "token", "secret"]:
                        if field in data and isinstance(data[field], str):
                            data[field] = self._redact_sensitive_value(data[field])
                    return json.dumps(data, ensure_ascii=False)
                except json.JSONDecodeError:
                    pass

            return body_str
        except Exception:
            return None

    async def _get_response_body(self, response: Response) -> str | None:
        """获取响应体"""
        try:
            body_parts = []
            async for chunk in response.body_iterator:  # type: ignore[attr-defined]
                body_parts.append(chunk)

            # 重新构建响应体
            body = b"".join(body_parts)
            body_str = body.decode("utf-8")

            # 重新设置响应体的 body_iterator（因为已经被读取了）
            from starlette.responses import StreamingResponse

            new_response = StreamingResponse(
                iter([body]),
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )
            # 替换原始响应的 body_iterator  # type: ignore[attr-defined]
            response.body_iterator = new_response.body_iterator  # type: ignore[attr-defined]

            return body_str
        except Exception:
            return None

    def _redact_sensitive_value(self, value: str, visible_chars: int = 4) -> str:
        """脱敏敏感值"""
        if len(value) <= visible_chars:
            return "*" * len(value)
        return value[:visible_chars] + "*" * (len(value) - visible_chars)


class AuditLogMiddleware(BaseHTTPMiddleware):
    """审计日志中间件

    专注于记录关键操作的审计信息：
    - 认证相关操作（登录、激活码换取 API Key）
    - 管理操作（创建/删除激活码）
    - 推理请求（用于使用统计）

    审计日志与普通请求日志分离，便于独立分析
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.audit_logger = logging.getLogger("audit")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """处理审计日志"""
        # 定义需要审计的路径前缀
        audit_paths = [
            "/auth/",
            "/infer/",
        ]

        path = request.url.path

        # 检查是否需要审计
        needs_audit = any(path.startswith(prefix) for prefix in audit_paths)

        if not needs_audit:
            return await call_next(request)

        # 请求开始时间
        start_time = time.perf_counter()

        # 提取用户信息
        client_ip = request.client.host if request.client else "unknown"
        auth_info = self._extract_user_identity(request)

        # 执行请求
        response = await call_next(request)

        # 计算耗时
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # 构建审计日志数据
        audit_data: dict[str, Any] = {
            "timestamp": time.time(),
            "type": "audit_log",
            "event": self._classify_event(path, request.method),
            "user_id": auth_info.get("user_id"),
            "auth_type": auth_info.get("auth_type"),
            "api_key_prefix": auth_info.get("api_key_prefix"),
            "action": f"{request.method} {path}",
            "status": "success" if response.status_code < 400 else "failure",
            "status_code": response.status_code,
            "duration_ms": duration_ms,
            "client_ip": client_ip,
            "path": path,
            "method": request.method,
        }

        # 记录审计日志
        if response.status_code >= 400:
            self.audit_logger.warning(json.dumps(audit_data, ensure_ascii=False))
        else:
            self.audit_logger.info(json.dumps(audit_data, ensure_ascii=False))

        return response

    def _extract_user_identity(self, request: Request) -> dict[str, Any]:
        """提取用户身份信息"""
        identity: dict[str, Any] = {"user_id": None, "auth_type": None}

        # API Key
        api_key = request.headers.get("x-api-key")
        if api_key:
            identity["api_key_prefix"] = api_key[:12] + "..." if len(api_key) > 12 else api_key
            identity["auth_type"] = "api_key"
            identity["user_id"] = f"api_key:{api_key[:12]}"

        # JWT Token
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            identity["auth_type"] = "jwt"
            identity["user_id"] = "jwt_user"

        return identity

    def _classify_event(self, path: str, method: str) -> str:
        """根据路径和方法分类事件类型"""
        # 精确匹配
        exact_matches = {
            ("/auth/activate", "POST"): "ACTIVATE_CODE",
            ("/auth/token", "POST"): "JWT_LOGIN",
            ("/auth/activation-codes", "POST"): "CREATE_ACTIVATION_CODE",
            ("/auth/activation-codes", "GET"): "LIST_ACTIVATION_CODES",
            ("/infer/single", "POST"): "SINGLE_INFERENCE",
            ("/infer/fusion", "POST"): "FUSION_INFERENCE",
        }

        if (path, method) in exact_matches:
            return exact_matches[(path, method)]

        # 前缀匹配（用于带参数的路径）
        prefix_matches = {
            ("/auth/activation-codes/", "DELETE"): "DELETE_ACTIVATION_CODE",
            ("/auth/activation-codes/", "PUT"): "UPDATE_ACTIVATION_CODE",
            ("/auth/activation-codes/", "POST"): "DEACTIVATE_ACTIVATION_CODE",
            ("/infer/task/", "GET"): "QUERY_TASK_STATUS",
        }

        for prefix, method_pair in prefix_matches.items():
            if path.startswith(prefix[0]) and method == prefix[1]:
                return method_pair

        return f"UNKNOWN_{method}"
