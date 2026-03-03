"""
激活码路由控制器

提供以下端点：
- POST /auth/activate - 激活码换取 API Key
- GET /auth/activation-codes - 列出所有激活码（需要 admin 权限）
- POST /auth/activation-codes - 创建新激活码（需要 admin 权限）
- PUT /auth/activation-codes/{code} - 更新激活码（需要 admin 权限）
- DELETE /auth/activation-codes/{code} - 删除激活码（需要 admin 权限）
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from schemas.activation import (
    ActivateRequest,
    ActivateResponse,
    ActivationCodeCreate,
    ActivationCodeInfo,
    ActivationCodeListResponse,
    ActivationCodeGenerateResponse,
    ActivationCodeUpdate,
)
from util.auth import (
    activate_with_code,
    get_all_activation_codes,
    create_activation_code,
    update_activation_code,
    delete_activation_code,
    deactivate_activation_code,
)
from util.auth import AuthCredentials
from middleware.auth_middleware import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(tags=["激活码"])


@router.post("/activate", response_model=ActivateResponse)
async def activate(request: ActivateRequest):
    """激活码换取 API Key

    用户输入激活码，验证通过后返回 API Key。
    API Key 用于后续请求的认证。
    """
    result = activate_with_code(request.code)

    if not result:
        raise HTTPException(
            status_code=400,
            detail="激活码无效、已过期或已被禁用",
        )

    api_key, code_info = result

    # 计算 API Key 过期时间（与激活码保持一致）
    expires_at = None
    if code_info.expires_at:
        from datetime import datetime
        expires_at = datetime.fromisoformat(str(code_info.expires_at))

    logger.info(f"激活码 {request.code[:8]}... 激活成功，用户 ID: {code_info.user_id}")

    return ActivateResponse(
        api_key=api_key,
        message="激活成功",
        expires_at=expires_at,
    )


@router.get("/activation-codes", response_model=ActivationCodeListResponse)
async def list_activation_codes(
    auth: Annotated[AuthCredentials, Depends(get_current_user)],
):
    """列出所有激活码

    需要 admin 权限。
    """
    if "admin" not in auth.permissions:
        raise HTTPException(status_code=403, detail="需要 admin 权限")

    codes = get_all_activation_codes()

    # 隐藏敏感信息 - 创建脱敏后的副本
    masked_codes: list[ActivationCodeInfo] = []
    for code in codes:
        if len(code.code) > 12:
            # 创建脱敏后的副本
            masked_code = code.model_copy(update={"code": f"{code.code[:8]}...{code.code[-4:]}"})
        else:
            masked_code = code
        masked_codes.append(masked_code)

    return ActivationCodeListResponse(activation_codes=masked_codes)


@router.post("/activation-codes", response_model=ActivationCodeGenerateResponse)
async def generate_activation_code(
    request: ActivationCodeCreate,
    auth: Annotated[AuthCredentials, Depends(get_current_user)],
):
    """生成新的激活码

    需要 admin 权限。
    """
    if "admin" not in auth.permissions:
        raise HTTPException(status_code=403, detail="需要 admin 权限")

    code_info = create_activation_code(
        name=request.name,
        max_uses=request.max_uses,
        expires_in_hours=request.expires_in_hours,
        permissions=request.permissions,
        priority=request.priority,  # 使用请求中的优先级
    )

    logger.info(f"用户 {auth.user_id} 生成了新激活码：{code_info.code} (priority={code_info.priority})")

    return ActivationCodeGenerateResponse(
        code=code_info.code,
        user_id=code_info.user_id,
        name=code_info.name,
        created_at=code_info.created_at,
        is_active=code_info.is_active,
        max_uses=code_info.max_uses,
        expires_at=code_info.expires_at,
    )


@router.put("/activation-codes/{code}")
async def update_activation_code_endpoint(
    code: str,
    request: ActivationCodeUpdate,
    auth: Annotated[AuthCredentials, Depends(get_current_user)],
):
    """更新激活码配置

    需要 admin 权限。
    """
    if "admin" not in auth.permissions:
        raise HTTPException(status_code=403, detail="需要 admin 权限")

    # 检查激活码是否存在
    from util.auth import ACTIVATION_CODES_DB
    if code not in ACTIVATION_CODES_DB:
        raise HTTPException(status_code=404, detail=f"激活码不存在：{code}")

    code_info = update_activation_code(
        code=code,
        is_active=request.is_active,
        max_uses=request.max_uses,
        permissions=request.permissions,
        priority=request.priority,
    )

    if not code_info:
        raise HTTPException(status_code=404, detail=f"更新失败：{code}")

    logger.info(f"用户 {auth.user_id} 更新了激活码：{code}")

    return {
        "message": f"激活码 {code} 已更新",
        "code_info": code_info.model_dump(),
    }


@router.delete("/activation-codes/{code}")
async def delete_activation_code_endpoint(
    code: str,
    auth: Annotated[AuthCredentials, Depends(get_current_user)],
):
    """删除激活码

    需要 admin 权限。
    """
    if "admin" not in auth.permissions:
        raise HTTPException(status_code=403, detail="需要 admin 权限")

    # 检查激活码是否存在
    from util.auth import ACTIVATION_CODES_DB
    if code not in ACTIVATION_CODES_DB:
        raise HTTPException(status_code=404, detail=f"激活码不存在：{code}")

    success = delete_activation_code(code)

    if not success:
        raise HTTPException(status_code=400, detail="删除失败")

    logger.info(f"用户 {auth.user_id} 删除了激活码：{code}")

    return {"message": f"激活码 {code} 已删除"}


@router.post("/activation-codes/{code}/deactivate")
async def deactivate_activation_code_endpoint(
    code: str,
    auth: Annotated[AuthCredentials, Depends(get_current_user)],
):
    """禁用激活码

    需要 admin 权限。
    """
    if "admin" not in auth.permissions:
        raise HTTPException(status_code=403, detail="需要 admin 权限")

    # 检查激活码是否存在
    from util.auth import ACTIVATION_CODES_DB
    if code not in ACTIVATION_CODES_DB:
        raise HTTPException(status_code=404, detail=f"激活码不存在：{code}")

    success = deactivate_activation_code(code)

    if not success:
        raise HTTPException(status_code=400, detail="禁用失败")

    logger.info(f"用户 {auth.user_id} 禁用了激活码：{code}")

    return {"message": f"激活码 {code} 已禁用"}
