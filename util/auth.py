"""
认证与授权工具模块

提供两种认证方式：
- API Key: 适用于客户端请求（通过激活码换取）
- JWT Token: 适用于管理后台认证（管理员登录）
"""

import os
import secrets
from datetime import datetime, timedelta
from functools import wraps

import jwt
from pydantic import BaseModel

from schemas.activation import ActivationCodeInfo
from schemas.auth import APIKeyInfo


# JWT 配置
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))

# API Key 配置（API Key 通过激活码换取，不直接从数据库加载）
# 格式：{api_key: {name, created_at, permissions, user_id}}
API_KEYS_DB: dict[str, dict] = {}

# 激活码配置
# 格式：{code: {user_id, name, created_at, is_active, max_uses, current_uses, expires_at, permissions}}
ACTIVATION_CODES_DB: dict[str, dict] = {}

# 初始化默认激活码（仅用于开发环境）
DEFAULT_ACTIVATION_CODE = os.getenv("DEFAULT_ACTIVATION_CODE", "ACT-DEV-DEFAULT-KEY")
if DEFAULT_ACTIVATION_CODE:
    default_code_info = ActivationCodeInfo(
        code=DEFAULT_ACTIVATION_CODE,
        user_id="default_user",
        name="default_activation_code",
        created_at=datetime.utcnow(),
        is_active=True,
        max_uses=100,  # 开发环境默认 100 次使用
        current_uses=0,
        expires_at=None,
        permissions=["read", "write"],
    )
    ACTIVATION_CODES_DB[DEFAULT_ACTIVATION_CODE] = default_code_info.model_dump()


class TokenPayload(BaseModel):
    """JWT Token 载荷"""

    sub: str  # 用户 ID
    exp: datetime  # 过期时间
    iat: datetime  # 签发时间
    permissions: list[str] = []  # 权限列表


def generate_api_key() -> str:
    """生成新的 API Key"""
    return f"sk_{secrets.token_urlsafe(32)}"


def register_api_key(
    api_key: str,
    name: str,
    permissions: list[str] | None = None,
    user_id: str = "",
) -> APIKeyInfo:
    """注册新的 API Key"""
    if permissions is None:
        permissions = ["read", "write"]

    info = APIKeyInfo(
        name=name,
        created_at=datetime.utcnow(),
        permissions=permissions,
        user_id=user_id,
    )
    API_KEYS_DB[api_key] = info.model_dump()
    return info


def validate_api_key(api_key: str) -> APIKeyInfo | None:
    """验证 API Key 是否有效"""
    if not api_key:
        return None

    key_info = API_KEYS_DB.get(api_key)
    if key_info:
        return APIKeyInfo(**key_info)
    return None


def create_token(
    user_id: str,
    permissions: list[str] | None = None,
    expires_delta: timedelta | None = None,
) -> str:
    """创建 JWT Token"""
    now = datetime.utcnow()

    if expires_delta is None:
        expires_delta = timedelta(hours=JWT_EXPIRATION_HOURS)

    payload = TokenPayload(
        sub=user_id,
        exp=now + expires_delta,
        iat=now,
        permissions=permissions or ["read", "write"],
    )

    token = jwt.encode(payload.model_dump(), JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token


def verify_token(token: str) -> TokenPayload | None:
    """验证 JWT Token"""
    try:
        payload_dict = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return TokenPayload(**payload_dict)
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


class AuthCredentials:
    """认证凭据封装"""

    def __init__(
        self,
        authenticated: bool = False,
        user_id: str | None = None,
        permissions: list[str] | None = None,
        auth_type: str | None = None,  # "api_key" or "jwt"
    ):
        self.authenticated = authenticated
        self.user_id = user_id
        self.permissions = permissions or []
        self.auth_type = auth_type

    @classmethod
    def from_api_key(cls, api_key: str) -> "AuthCredentials":
        """从 API Key 创建凭据"""
        key_info = validate_api_key(api_key)
        if key_info:
            return cls(
                authenticated=True,
                user_id=f"api_key:{key_info.name}",
                permissions=key_info.permissions,
                auth_type="api_key",
            )
        return cls()

    @classmethod
    def from_token(cls, token: str) -> "AuthCredentials":
        """从 JWT Token 创建凭据"""
        payload = verify_token(token)
        if payload:
            return cls(
                authenticated=True,
                user_id=payload.sub,
                permissions=payload.permissions,
                auth_type="jwt",
            )
        return cls()

    def has_permission(self, permission: str) -> bool:
        """检查是否有指定权限"""
        return permission in self.permissions


def require_permission(permission: str):
    """权限检查装饰器（用于未来扩展）"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 从 kwargs 中获取 auth_credentials
            auth_credentials: AuthCredentials | None = kwargs.get("auth_credentials")
            if not auth_credentials or not auth_credentials.has_permission(permission):
                from fastapi import HTTPException

                raise HTTPException(
                    status_code=403,
                    detail=f"缺少权限：{permission}",
                )
            return await func(*args, **kwargs)

        return wrapper

    return decorator


# ==================== 激活码管理功能 ====================

def generate_activation_code() -> str:
    """生成新的激活码"""
    prefix = "ACT"
    unique_part = secrets.token_urlsafe(16).replace("_", "").replace("-", "")[:16]
    return f"{prefix}-{unique_part[:8]}-{unique_part[8:]}".upper()


def create_activation_code(
    name: str = "",
    max_uses: int = 1,  # 默认 1 次使用
    expires_in_hours: int | None = None,
    permissions: list[str] | None = None,
) -> ActivationCodeInfo:
    """创建新的激活码

    Args:
        name: 激活码名称/描述
        max_uses: 最大使用次数（必须 >= 1，默认 1 次）
        expires_in_hours: 过期时间（小时），None 表示永不过期
        permissions: 权限列表，默认 ["read", "write"]

    Returns:
        ActivationCodeInfo: 激活码信息
    """
    if permissions is None:
        permissions = ["read", "write"]

    if max_uses < 1:
        raise ValueError("max_uses 必须 >= 1")

    code = generate_activation_code()
    now = datetime.utcnow()

    # 计算过期时间
    expires_at = None
    if expires_in_hours:
        expires_at = now + timedelta(hours=expires_in_hours)

    info = ActivationCodeInfo(
        code=code,
        user_id=f"user_{code}",
        name=name,
        created_at=now,
        is_active=True,
        max_uses=max_uses,
        current_uses=0,
        expires_at=expires_at,
        permissions=permissions,
    )

    ACTIVATION_CODES_DB[code] = info.model_dump()
    return info


def validate_activation_code(code: str) -> ActivationCodeInfo | None:
    """验证激活码是否有效"""
    if not code:
        return None

    code_info = ACTIVATION_CODES_DB.get(code)
    if not code_info:
        return None

    # 检查是否启用
    if not code_info.get("is_active", True):
        return None

    # 检查是否过期
    expires_at = code_info.get("expires_at")
    if expires_at:
        # expires_at 可能是 datetime 对象或 ISO 格式字符串
        if isinstance(expires_at, str):
            expires_at = datetime.fromisoformat(expires_at)
        if expires_at < datetime.utcnow():
            return None

    # 检查使用次数
    max_uses = code_info.get("max_uses")
    current_uses = code_info.get("current_uses", 0)
    if max_uses is None or current_uses >= max_uses:
        return None

    return ActivationCodeInfo(**code_info)


def increment_activation_code_usage(code: str) -> bool:
    """增加激活码使用次数"""
    if code not in ACTIVATION_CODES_DB:
        return False

    ACTIVATION_CODES_DB[code]["current_uses"] = ACTIVATION_CODES_DB[code].get("current_uses", 0) + 1
    return True


def activate_with_code(code: str) -> tuple[str, ActivationCodeInfo] | None:
    """使用激活码换取 API Key

    返回：(api_key, activation_code_info) 或 None（如果激活码无效）
    """
    code_info = validate_activation_code(code)
    if not code_info:
        return None

    # 生成 API Key
    api_key = generate_api_key()

    # 注册 API Key，关联到激活码的用户
    register_api_key(
        api_key=api_key,
        name=f"activated_{code}",
        permissions=code_info.permissions,
        user_id=code_info.user_id,
    )

    # 增加使用次数
    increment_activation_code_usage(code)

    return api_key, code_info


def get_all_activation_codes() -> list[ActivationCodeInfo]:
    """获取所有激活码"""
    codes = []
    for code, info in ACTIVATION_CODES_DB.items():
        # 确保字典中包含 code 字段
        info_with_code = {**info, "code": code}
        codes.append(ActivationCodeInfo(**info_with_code))
    return codes


def deactivate_activation_code(code: str) -> bool:
    """禁用激活码"""
    if code not in ACTIVATION_CODES_DB:
        return False

    ACTIVATION_CODES_DB[code]["is_active"] = False
    return True


def delete_activation_code(code: str) -> bool:
    """删除激活码"""
    if code not in ACTIVATION_CODES_DB:
        return False

    del ACTIVATION_CODES_DB[code]
    return True


def update_activation_code(
    code: str,
    is_active: bool | None = None,
    max_uses: int | None = None,
    permissions: list[str] | None = None,
) -> ActivationCodeInfo | None:
    """更新激活码配置"""
    if code not in ACTIVATION_CODES_DB:
        return None

    if is_active is not None:
        ACTIVATION_CODES_DB[code]["is_active"] = is_active
    if max_uses is not None:
        ACTIVATION_CODES_DB[code]["max_uses"] = max_uses
    if permissions is not None:
        ACTIVATION_CODES_DB[code]["permissions"] = permissions

    return ActivationCodeInfo(**ACTIVATION_CODES_DB[code])
