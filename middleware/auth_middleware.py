"""
认证中间件模块

提供 FastAPI 依赖注入式的认证机制：
- 支持 API Key 和 JWT Token 两种认证方式
- 可配置白名单路由
- 自动从请求头提取认证信息
"""

from fastapi import Request, HTTPException, Depends, WebSocket, WebSocketException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Annotated

from util.auth import AuthCredentials, validate_api_key, verify_token

# HTTP Bearer Token 提取器
http_bearer = HTTPBearer(auto_error=False)


# 白名单路由配置（这些路由不需要认证）
WHITELIST_PATHS = [
    "/health",  # 健康检查
    "/auth/token",  # 获取 JWT Token（管理员登录）
    "/auth/activate",  # 激活码换取 API Key
    "/infer/ws",  # WebSocket 连接（认证在连接内部处理）
    "/docs",  # Swagger UI
    "/redoc",  # ReDoc
    "/openapi.json",  # OpenAPI schema
]


def is_whitelisted_path(path: str) -> bool:
    """检查路径是否在白名单中"""
    # 精确匹配
    if path in WHITELIST_PATHS:
        return True

    # 前缀匹配（用于某些特殊路径）
    whitelisted_prefixes = [
        "/docs",
        "/redoc",
    ]
    for prefix in whitelisted_prefixes:
        if path.startswith(prefix):
            return True

    return False


async def extract_auth_from_request(request: Request) -> AuthCredentials:
    """从请求中提取认证信息

    支持两种认证方式：
    1. X-API-Key 请求头
    2. Authorization: Bearer <token>
    """
    # 优先尝试 API Key
    api_key = request.headers.get("X-API-Key")
    if api_key:
        credentials = AuthCredentials.from_api_key(api_key)
        if credentials.authenticated:
            return credentials

    # 然后尝试 JWT Token
    auth_header = request.headers.get("Authorization")
    if auth_header:
        parts = auth_header.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1]
            credentials = AuthCredentials.from_token(token)
            if credentials.authenticated:
                return credentials

    return AuthCredentials()


async def get_current_user(
    request: Request,
) -> AuthCredentials:
    """获取当前认证用户（用于需要认证的路由）

    如果路径在白名单中，返回未认证状态
    否则必须认证成功，否则抛出 401 错误
    """
    # 检查是否在白名单中
    if is_whitelisted_path(request.url.path):
        return AuthCredentials()

    # 提取并验证认证信息
    credentials: AuthCredentials = await extract_auth_from_request(request)

    if not credentials.authenticated:
        raise HTTPException(
            status_code=401,
            detail="未授权访问",
            headers={"WWW-Authenticate": "Bearer or API-Key"},
        )

    return credentials


async def get_current_user_optional(
    request: Request,
) -> AuthCredentials:
    """获取当前认证用户（可选认证）

    无论是否认证成功都返回，调用方自行判断
    用于部分开放但需要知道用户身份的场景
    """
    return await extract_auth_from_request(request)


def require_permissions(*permissions: str):
    """权限要求装饰器工厂

    用法:
        @router.get("/admin")
        @require_permissions("admin")
        async def admin_endpoint(auth: AuthCredentials):
            ...
    """
    required = set(permissions)

    async def permission_checker(
        auth: Annotated[AuthCredentials, Depends(get_current_user)],
    ) -> AuthCredentials:
        if not auth.authenticated:
            raise HTTPException(status_code=401, detail="未授权访问")

        user_permissions = set(auth.permissions)
        if not required.issubset(user_permissions):
            missing = required - user_permissions
            raise HTTPException(
                status_code=403,
                detail=f"缺少权限：{', '.join(missing)}",
            )

        return auth

    return permission_checker


class AuthMiddleware:
    """认证中间件类（用于全局注册）

    注意：主要使用 get_current_user 依赖注入方式
    此中间件用于额外的全局处理（如日志记录）
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        # 仅处理 HTTP 请求
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)

        # 记录认证状态（用于审计日志）
        path = request.url.path
        is_whitelisted = is_whitelisted_path(path)

        if not is_whitelisted:
            credentials = await extract_auth_from_request(request)
            if not credentials.authenticated:
                # 记录未授权访问尝试
                import logging
                logger = logging.getLogger(__name__)
                client_ip = request.client.host if request.client else "unknown"
                logger.warning(
                    f"未授权访问尝试：{request.method} {path} from {client_ip}"
                )

        await self.app(scope, receive, send)


# WebSocket 认证辅助函数
async def get_websocket_auth(websocket: WebSocket) -> AuthCredentials:
    """从 WebSocket 连接中提取认证信息"""
    # 从查询参数获取 API Key 或 Token
    api_key = websocket.query_params.get("api_key")
    token = websocket.query_params.get("token")

    if api_key:
        credentials = AuthCredentials.from_api_key(api_key)
        if credentials.authenticated:
            return credentials

    if token:
        credentials = AuthCredentials.from_token(token)
        if credentials.authenticated:
            return credentials

    return AuthCredentials()


async def websocket_auth(websocket: WebSocket) -> AuthCredentials:
    """WebSocket 认证（在 accept 之前调用）"""
    credentials = await get_websocket_auth(websocket)

    if not credentials.authenticated:
        raise WebSocketException(
            code=4001,  # WebSocket 自定义错误码
            reason="未授权访问",
        )

    return credentials
