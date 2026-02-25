"""
认证路由控制器（管理后台专用）

提供以下端点：
- POST /auth/token - 获取 JWT Token（管理员登录）
- GET /auth/me - 获取当前用户信息
- POST /auth/refresh - 刷新 Token

注意：此控制器仅用于管理后台认证，客户端请使用激活码模式
"""

import logging
from datetime import timedelta
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Header

from schemas.auth import (
    TokenRequest,
    TokenResponse,
    UserInfo,
)
from util.auth import (
    create_token,
    verify_token,
    AuthCredentials,
    JWT_EXPIRATION_HOURS,
)
from middleware.auth_middleware import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(tags=["认证"])


@router.post("/token", response_model=TokenResponse)
async def login_for_access_token(request: TokenRequest):
    """获取 JWT Token

    使用用户名和密码登录，返回 JWT Token。

    > 注意：当前为简化实现，仅接受预定义的管理员账户。
    > 生产环境应连接数据库进行用户认证。
    """
    # 简单实现：接受固定的管理员账户（生产环境应连接数据库）
    # 可通过环境变量设置管理员凭据
    import os

    admin_username = os.getenv("ADMIN_USERNAME", "admin")
    admin_password = os.getenv("ADMIN_PASSWORD", "admin123")

    if request.username != admin_username or request.password != admin_password:
        raise HTTPException(
            status_code=401,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 创建 Token
    access_token_expires = timedelta(hours=JWT_EXPIRATION_HOURS)
    access_token = create_token(
        user_id=request.username,
        permissions=["read", "write", "admin"],
        expires_delta=access_token_expires,
    )

    logger.info(f"用户 {request.username} 登录成功")

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=int(access_token_expires.total_seconds()),
        permissions=["read", "write", "admin"],
    )


@router.get("/me", response_model=UserInfo)
async def get_current_user_info(
    auth: Annotated[AuthCredentials, Depends(get_current_user)],
):
    """获取当前认证用户的信息"""
    return UserInfo(
        user_id=auth.user_id or "anonymous",
        auth_type=auth.auth_type or "none",
        permissions=auth.permissions,
        authenticated=auth.authenticated,
    )


@router.post("/refresh")
async def refresh_token(
    authorization: Annotated[Optional[str], Header()] = None,
):
    """刷新 Token

    使用当前 Token 换取新的 Token，延长有效期。
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="缺少 Authorization 头")

    # 提取 Token
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="无效的 Authorization 头")

    token = parts[1]

    # 验证旧 Token
    payload = verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Token 已过期或无效")

    # 创建新 Token
    access_token_expires = timedelta(hours=JWT_EXPIRATION_HOURS)
    new_token = create_token(
        user_id=payload.sub,
        permissions=payload.permissions,
        expires_delta=access_token_expires,
    )

    return TokenResponse(
        access_token=new_token,
        token_type="bearer",
        expires_in=int(access_token_expires.total_seconds()),
        permissions=payload.permissions,
    )
