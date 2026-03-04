"""
应用配置管理控制器
提供查询、更新、回滚配置的 API 端点
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, Query

from service.config_service import ConfigService
from schemas.config import (
    AppConfigResponse,
    LoggingConfigResponse,
    RetryConfigResponse,
    DebugConfigResponse,
    StorageSaveStrategyResponse,
    ImageCompressConfigResponse,
    LoggingConfigUpdateRequest,
    RetryConfigUpdateRequest,
    DebugConfigUpdateRequest,
    StorageSaveStrategyUpdateRequest,
    ImageCompressConfigUpdateRequest,
)
from util.auth import AuthCredentials
from middleware.auth_middleware import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()


def require_admin(
    auth: AuthCredentials = Depends(get_current_user),
) -> AuthCredentials:
    """要求管理员权限（JWT Token）"""
    if not auth.authenticated:
        raise HTTPException(status_code=401, detail="未授权访问")
    if auth.auth_type != "jwt":
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return auth


@router.get(
    "/config",
    response_model=AppConfigResponse,
    tags=["系统配置"],
    dependencies=[Depends(require_admin)],
)
async def get_app_config(
    auth: AuthCredentials = Depends(require_admin),
) -> AppConfigResponse:
    """
    获取应用配置（仅管理员）

    需要管理员权限（JWT Token）

    返回所有支持热更新配置组的当前配置
    """
    return AppConfigResponse(
        logging=LoggingConfigResponse(**ConfigService.get_logging_config()),
        retry=RetryConfigResponse(**ConfigService.get_retry_config()),
        debug=DebugConfigResponse(**ConfigService.get_debug_config()),
        storage_save_strategy=StorageSaveStrategyResponse(**ConfigService.get_storage_save_strategy_config()),
        image_compress=ImageCompressConfigResponse(**ConfigService.get_image_compress_config()),
    )


# ============= 日志配置 =============

@router.get(
    "/config/logging",
    response_model=LoggingConfigResponse,
    tags=["系统配置"],
    dependencies=[Depends(require_admin)],
)
async def get_logging_config(
    auth: AuthCredentials = Depends(require_admin),
) -> LoggingConfigResponse:
    """获取日志配置（仅管理员）"""
    return LoggingConfigResponse(**ConfigService.get_logging_config())


@router.put(
    "/config/logging",
    response_model=LoggingConfigResponse,
    tags=["系统配置"],
    dependencies=[Depends(require_admin)],
)
async def update_logging_config(
    request: LoggingConfigUpdateRequest,
    auth: AuthCredentials = Depends(require_admin),
) -> LoggingConfigResponse:
    """更新日志配置（仅管理员）

    - **log_level**: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - **log_to_console**: 是否输出到控制台
    - **log_json_format**: 是否 JSON 格式
    - **log_request_body**: 是否记录请求体
    - **log_response_body**: 是否记录响应体
    - **audit_log_enabled**: 是否启用审计日志

    注意：只更新运行时配置，不持久化到.env 文件
    """
    return LoggingConfigResponse(**ConfigService.update_logging_config(
        log_level=request.log_level,
        log_to_console=request.log_to_console,
        log_json_format=request.log_json_format,
        log_request_body=request.log_request_body,
        log_response_body=request.log_response_body,
        audit_log_enabled=request.audit_log_enabled,
    ))


# ============= 重试配置 =============

@router.get(
    "/config/retry",
    response_model=RetryConfigResponse,
    tags=["系统配置"],
    dependencies=[Depends(require_admin)],
)
async def get_retry_config(
    auth: AuthCredentials = Depends(require_admin),
) -> RetryConfigResponse:
    """获取重试配置（仅管理员）"""
    return RetryConfigResponse(**ConfigService.get_retry_config())


@router.put(
    "/config/retry",
    response_model=RetryConfigResponse,
    tags=["系统配置"],
    dependencies=[Depends(require_admin)],
)
async def update_retry_config(
    request: RetryConfigUpdateRequest,
    auth: AuthCredentials = Depends(require_admin),
) -> RetryConfigResponse:
    """更新重试配置（仅管理员）

    注意：只更新运行时配置，不持久化到.env 文件
    """
    return RetryConfigResponse(**ConfigService.update_retry_config(
        retry_enabled=request.retry_enabled,
        retry_max_attempts=request.retry_max_attempts,
        retry_delay_seconds=request.retry_delay_seconds,
        retry_exponential_backoff=request.retry_exponential_backoff,
        retry_max_delay_seconds=request.retry_max_delay_seconds,
    ))


# ============= 调试配置 =============

@router.get(
    "/config/debug",
    response_model=DebugConfigResponse,
    tags=["系统配置"],
    dependencies=[Depends(require_admin)],
)
async def get_debug_config(
    auth: AuthCredentials = Depends(require_admin),
) -> DebugConfigResponse:
    """获取调试配置（仅管理员）"""
    return DebugConfigResponse(**ConfigService.get_debug_config())


@router.put(
    "/config/debug",
    response_model=DebugConfigResponse,
    tags=["系统配置"],
    dependencies=[Depends(require_admin)],
)
async def update_debug_config(
    request: DebugConfigUpdateRequest,
    auth: AuthCredentials = Depends(require_admin),
) -> DebugConfigResponse:
    """更新调试配置（仅管理员）

    注意：只更新运行时配置，不持久化到.env 文件
    """
    return DebugConfigResponse(**ConfigService.update_debug_config(
        debug_mode=request.debug_mode,
        debug_delay_per_image=request.debug_delay_per_image,
        debug_delay_per_pair=request.debug_delay_per_pair,
        debug_failure_rate=request.debug_failure_rate,
    ))


# ============= 存储策略配置 =============

@router.get(
    "/config/storage-strategy",
    response_model=StorageSaveStrategyResponse,
    tags=["系统配置"],
    dependencies=[Depends(require_admin)],
)
async def get_storage_strategy_config(
    auth: AuthCredentials = Depends(require_admin),
) -> StorageSaveStrategyResponse:
    """获取存储策略配置（仅管理员）"""
    return StorageSaveStrategyResponse(**ConfigService.get_storage_save_strategy_config())


@router.put(
    "/config/storage-strategy",
    response_model=StorageSaveStrategyResponse,
    tags=["系统配置"],
    dependencies=[Depends(require_admin)],
)
async def update_storage_strategy_config(
    request: StorageSaveStrategyUpdateRequest,
    auth: AuthCredentials = Depends(require_admin),
) -> StorageSaveStrategyResponse:
    """更新存储策略配置（仅管理员）

    注意：只更新运行时配置，不持久化到.env 文件
    """
    return StorageSaveStrategyResponse(**ConfigService.update_storage_save_strategy_config(
        storage_auto_save=request.storage_auto_save,
        storage_save_strategy=request.storage_save_strategy,
        storage_save_error_rate=request.storage_save_error_rate,
        storage_save_fake_rate=request.storage_save_fake_rate,
        storage_save_real_rate=request.storage_save_real_rate,
        storage_max_per_task=request.storage_max_per_task,
    ))


# ============= 图片压缩配置 =============

@router.get(
    "/config/compress",
    response_model=ImageCompressConfigResponse,
    tags=["系统配置"],
    dependencies=[Depends(require_admin)],
)
async def get_compress_config(
    auth: AuthCredentials = Depends(require_admin),
) -> ImageCompressConfigResponse:
    """获取图片压缩配置（仅管理员）"""
    return ImageCompressConfigResponse(**ConfigService.get_image_compress_config())


@router.put(
    "/config/compress",
    response_model=ImageCompressConfigResponse,
    tags=["系统配置"],
    dependencies=[Depends(require_admin)],
)
async def update_compress_config(
    request: ImageCompressConfigUpdateRequest,
    auth: AuthCredentials = Depends(require_admin),
) -> ImageCompressConfigResponse:
    """更新图片压缩配置（仅管理员）

    注意：只更新运行时配置，不持久化到.env 文件
    """
    return ImageCompressConfigResponse(**ConfigService.update_image_compress_config(
        compress_enabled=request.compress_enabled,
        compress_quality=request.compress_quality,
        compress_type=request.compress_type,
    ))


# ============= 配置历史与回滚 =============

@router.get(
    "/config/history",
    tags=["系统配置"],
    dependencies=[Depends(require_admin)],
)
async def get_config_history(
    limit: int = Query(10, ge=1, le=50, description="返回最近 N 条记录"),
    auth: AuthCredentials = Depends(require_admin),
):
    """获取配置变更历史（仅管理员）"""
    return {"history": ConfigService.get_config_history(limit)}


@router.post(
    "/config/rollback/{config_type}/{index}",
    tags=["系统配置"],
    dependencies=[Depends(require_admin)],
)
async def rollback_config(
    config_type: str,
    index: int,
    auth: AuthCredentials = Depends(require_admin),
):
    """
    回滚配置到历史版本（仅管理员）

    - **config_type**: 配置类型 (logging, retry, debug, storage_save_strategy, image_compress)
    - **index**: 历史记录索引（从 0 开始）

    注意：回滚会应用历史配置的旧值到当前运行时
    """
    valid_types = ["logging", "retry", "debug", "storage_save_strategy", "image_compress"]
    if config_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"无效的配置类型，必须是：{', '.join(valid_types)}"
        )

    try:
        result = ConfigService.rollback_config(config_type, index)
        return {
            "success": True,
            "message": f"已将 {config_type} 配置回滚到历史版本 #{index}",
            "config": result,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
