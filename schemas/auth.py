"""
认证相关的数据模型定义

JWT Token 仅用于管理后台认证，客户端请使用激活码模式
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class TokenRequest(BaseModel):
    """JWT Token 请求（管理员登录）

    仅用于管理后台认证
    """

    username: str = Field(..., description="管理员用户名")
    password: str = Field(..., description="管理员密码")


class TokenResponse(BaseModel):
    """JWT Token 响应（管理后台）"""

    access_token: str = Field(..., description="访问令牌")
    token_type: str = Field(default="bearer", description="令牌类型")
    expires_in: int = Field(..., description="过期时间（秒）")
    permissions: list[str] = Field(default_factory=list, description="权限列表")


class APIKeyInfo(BaseModel):
    """API Key 信息（用于内部存储）"""

    name: str = Field(..., description="API Key 名称")
    created_at: datetime = Field(..., description="创建时间")
    permissions: list[str] = Field(default_factory=list, description="权限列表")
    user_id: str = Field(default="", description="关联的用户 ID")


class UserInfo(BaseModel):
    """当前用户信息"""

    user_id: str = Field(..., description="用户 ID")
    auth_type: str = Field(..., description="认证类型：api_key 或 jwt")
    permissions: list[str] = Field(default_factory=list, description="权限列表")
    authenticated: bool = Field(default=False, description="是否已认证")


class AuthErrorResponse(BaseModel):
    """认证错误响应"""

    detail: str = Field(..., description="错误详情")
    status_code: int = Field(..., description="状态码")
