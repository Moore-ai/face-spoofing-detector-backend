"""
激活码相关的数据模型定义

激活码模式：
- 管理员生成激活码时设置最大使用次数
- 用户输入激活码换取 API Key
- 激活码达到使用次数上限后自动失效
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class ActivateRequest(BaseModel):
    """激活码验证请求"""

    code: str = Field(..., description="激活码", examples=["ACT-001-XXXXX-YYYYY"])


class ActivateResponse(BaseModel):
    """激活码验证响应"""

    api_key: str = Field(..., description="API Key")
    message: str = Field(default="激活成功", description="消息")
    expires_at: Optional[datetime] = Field(None, description="API Key 过期时间")


class ActivationCodeInfo(BaseModel):
    """激活码信息"""

    code: str = Field(..., description="激活码")
    user_id: str = Field(..., description="关联的用户 ID")
    name: str = Field(default="", description="激活码名称/描述")
    created_at: datetime = Field(..., description="创建时间")
    is_active: bool = Field(default=True, description="是否启用")
    max_uses: int = Field(..., description="最大使用次数")
    current_uses: int = Field(default=0, description="当前使用次数")
    expires_at: Optional[datetime] = Field(None, description="过期时间")
    permissions: list[str] = Field(default_factory=list, description="权限列表")
    priority: int = Field(default=0, ge=0, le=100, description="任务优先级，范围 0-100，值越大优先级越高")


class ActivationCodeCreate(BaseModel):
    """创建激活码请求"""

    name: str = Field(default="", description="激活码名称/描述")
    max_uses: int = Field(..., ge=1, description="最大使用次数（必须 >= 1）")
    expires_in_hours: Optional[int] = Field(None, description="过期时间（小时）")
    permissions: list[str] = Field(default=["read", "write"], description="权限列表")
    priority: int = Field(default=0, ge=0, le=100, description="任务优先级，范围 0-100，值越大优先级越高")


class ActivationCodeListResponse(BaseModel):
    """激活码列表响应"""

    activation_codes: list[ActivationCodeInfo] = Field(
        default_factory=list,
        description="激活码列表",
    )


class ActivationCodeGenerateResponse(BaseModel):
    """生成激活码响应"""

    code: str = Field(..., description="激活码")
    user_id: str = Field(..., description="关联的用户 ID")
    name: str = Field(default="", description="激活码名称")
    created_at: datetime = Field(..., description="创建时间")
    is_active: bool = Field(default=True, description="是否启用")
    max_uses: int = Field(..., description="最大使用次数")
    expires_at: Optional[datetime] = Field(None, description="过期时间")
    priority: int = Field(default=0, description="任务优先级")


class ActivationCodeUpdate(BaseModel):
    """更新激活码请求"""

    is_active: Optional[bool] = Field(None, description="是否启用")
    max_uses: Optional[int] = Field(None, ge=1, description="最大使用次数")
    permissions: Optional[list[str]] = Field(None, description="权限列表")
    priority: Optional[int] = Field(None, ge=0, le=100, description="任务优先级")
