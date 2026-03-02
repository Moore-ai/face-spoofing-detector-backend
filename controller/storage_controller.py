"""
图片存储管理控制器
提供图片查询、删除、统计等 API 端点（仅管理员）
注意：图片上传由服务端根据存储策略自动执行，不向客户端开放
"""

import logging
from typing import Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy.orm import Session

from db import get_db_session
from util.storage import storage_manager, StorageError, LocalStorageBackend
from util.auth import AuthCredentials
from middleware.auth_middleware import get_current_user
from schemas.storage import (
    ImageQueryResponse,
    ImageListResponse,
    StorageStatsResponse,
    StorageConfigResponse,
    ImageDeleteResponse,
    StorageCleanupRequest,
    StorageQuotaUpdateRequest,
    ImageBatchDownloadRequest,
    ImageIdListResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def get_db() -> Session:
    """获取数据库会话的依赖注入"""
    db_gen = get_db_session()
    return next(db_gen)


def require_auth(
    auth: AuthCredentials = Depends(get_current_user),
) -> AuthCredentials:
    """要求认证"""
    if not auth.authenticated:
        raise HTTPException(status_code=401, detail="未授权访问，需要 API Key 或 JWT Token")
    return auth


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
    "/images",
    response_model=ImageListResponse,
    tags=["图片存储"],
    dependencies=[Depends(require_admin)],
)
async def query_images(
    task_id: Optional[str] = Query(None, description="任务 ID 过滤"),
    start_date: Optional[datetime] = Query(None, description="开始日期"),
    end_date: Optional[datetime] = Query(None, description="结束日期"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    auth: AuthCredentials = Depends(require_admin),
) -> ImageListResponse:
    """
    查询图片列表（仅管理员）

    - **task_id**: 按任务 ID 过滤（可选）
    - **start_date**: 查询开始日期（可选）
    - **end_date**: 查询结束日期（可选）
    - **page**: 页码，从 1 开始
    - **page_size**: 每页数量，最大 100

    需要管理员权限（JWT Token）

    注意：图片上传由服务端根据存储策略自动执行，不向客户端开放
    """
    try:
        # 查询图片列表
        result = storage_manager.query_images(
            task_id=task_id,
            start_date=start_date,
            end_date=end_date,
            page=page,
            page_size=page_size,
        )

        # 转换为响应格式
        items = []
        for item in result["items"]:
            items.append(
                ImageQueryResponse(
                    image_id=item.get("image_id", ""),
                    task_id=item.get("task_id", ""),
                    image_type=item.get("image_type", ""),
                    modality=item.get("modality", ""),
                    storage_path=item.get("storage_path", ""),
                    storage_type=item.get("storage_type", "local"),
                    file_size=item.get("file_size", 0),
                    content_type=item.get("content_type", "image/jpeg"),
                    created_at=datetime.fromisoformat(item.get("created_at", "")),
                    metadata=item.get("metadata"),
                )
            )

        return ImageListResponse(
            total=result["total"],
            page=result["page"],
            page_size=result["page_size"],
            items=items,
        )

    except Exception as e:
        logger.error(f"Failed to query images: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"查询失败：{str(e)}")


@router.get(
    "/images/{image_id}",
    response_model=ImageQueryResponse,
    tags=["图片存储"],
    dependencies=[Depends(require_admin)],
)
async def get_image_info(
    image_id: str,
    auth: AuthCredentials = Depends(require_admin),
) -> ImageQueryResponse:
    """
    获取单个图片信息（仅管理员）

    - **image_id**: 图片唯一 ID

    需要管理员权限（JWT Token）
    """
    try:
        metadata = storage_manager.backend.get_metadata(image_id) if storage_manager.backend else None

        if not metadata:
            raise HTTPException(status_code=404, detail=f"图片 {image_id} 不存在")

        return ImageQueryResponse(
            image_id=metadata.get("image_id", ""),
            task_id=metadata.get("task_id", ""),
            image_type=metadata.get("image_type", ""),
            modality=metadata.get("modality", ""),
            storage_path=metadata.get("storage_path", ""),
            storage_type=metadata.get("storage_type", "local"),
            file_size=metadata.get("file_size", 0),
            content_type=metadata.get("content_type", "image/jpeg"),
            created_at=datetime.fromisoformat(metadata.get("created_at", "")),
            metadata=metadata.get("metadata"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"查询失败：{str(e)}")


@router.delete(
    "/images",
    response_model=ImageDeleteResponse,
    tags=["图片存储"],
    dependencies=[Depends(require_admin)],
)
async def delete_images(
    image_ids: Optional[str] = Query(None, description="要删除的图片 ID 列表（逗号分隔）"),
    task_ids: Optional[str] = Query(None, description="要删除的任务 ID 列表（逗号分隔）"),
    older_than_days: Optional[int] = Query(None, description="删除早于 N 天的图片", ge=1),
    auth: AuthCredentials = Depends(require_admin),
) -> ImageDeleteResponse:
    """
    删除图片（仅管理员）

    - **image_ids**: 要删除的图片 ID 列表，用逗号分隔（可选）
    - **task_ids**: 要删除的任务 ID 列表，用逗号分隔（可选）
    - **older_than_days**: 删除早于 N 天的图片（可选）

    需要管理员权限（JWT Token）

    注意：image_ids、task_ids、older_than_days 只能使用其中一个
    """
    # 验证参数
    params = [image_ids, task_ids, older_than_days]
    provided_params = [p for p in params if p is not None]

    if len(provided_params) != 1:
        raise HTTPException(
            status_code=400, detail="必须指定且只能指定 image_ids、task_ids 或 older_than_days 其中一个参数"
        )

    try:
        deleted_count = 0
        freed_size = 0

        if image_ids:
            # 删除指定图片
            ids = [id.strip() for id in image_ids.split(",")]
            for image_id in ids:
                if storage_manager.delete_image(image_id):
                    deleted_count += 1
                    # 获取删除前的大小
                    metadata = storage_manager.backend.get_metadata(image_id) if storage_manager.backend else None
                    if metadata:
                        freed_size += metadata.get("file_size", 0)

        elif task_ids:
            # 删除指定任务的图片
            ids = [id.strip() for id in task_ids.split(",")]
            result = storage_manager.cleanup_images(older_than_days=1, task_ids=ids)
            deleted_count = result["deleted_count"]
            freed_size = result["freed_size_bytes"]

        elif older_than_days:
            # 删除旧图片
            result = storage_manager.cleanup_images(older_than_days=older_than_days)
            deleted_count = result["deleted_count"]
            freed_size = result["freed_size_bytes"]

        return ImageDeleteResponse(
            success=True,
            deleted_count=deleted_count,
            freed_size_bytes=freed_size,
            message=f"成功删除 {deleted_count} 张图片，释放 {freed_size / (1024 * 1024):.2f} MB 空间",
        )

    except StorageError as e:
        logger.error(f"Storage error: {e}")
        raise HTTPException(status_code=500, detail=f"删除失败：{str(e)}")
    except Exception as e:
        logger.error(f"Failed to delete images: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"删除失败：{str(e)}")


@router.get(
    "/stats",
    response_model=StorageStatsResponse,
    tags=["图片存储"],
    dependencies=[Depends(require_admin)],
)
async def get_storage_stats(
    auth: AuthCredentials = Depends(require_admin),
) -> StorageStatsResponse:
    """
    获取存储统计信息（仅管理员）

    返回总图片数、总存储空间、配额使用情况、按类型/模态统计等

    需要管理员权限（JWT Token）
    """
    try:
        stats = storage_manager.get_stats()

        return StorageStatsResponse(
            total_images=stats["total_images"],
            total_size_bytes=stats["total_size_bytes"],
            total_size_mb=stats["total_size_mb"],
            quota_bytes=stats["quota_bytes"],
            quota_used_percent=stats["quota_used_percent"],
            by_type=stats["by_type"],
            by_modality=stats["by_modality"],
        )

    except StorageError as e:
        logger.error(f"Storage error: {e}")
        raise HTTPException(status_code=500, detail=f"获取统计失败：{str(e)}")
    except Exception as e:
        logger.error(f"Failed to get storage stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取统计失败：{str(e)}")


@router.get(
    "/config",
    response_model=StorageConfigResponse,
    tags=["图片存储"],
    dependencies=[Depends(require_admin)],
)
async def get_storage_config(
    auth: AuthCredentials = Depends(require_admin),
) -> StorageConfigResponse:
    """
    获取存储配置（仅管理员）

    需要管理员权限（JWT Token）
    """
    from util.config import settings

    return StorageConfigResponse(
        storage_type=settings.STORAGE_TYPE,
        storage_path=settings.STORAGE_LOCAL_PATH,
        quota_bytes=settings.STORAGE_QUOTA_BYTES,
        retention_days=settings.STORAGE_RETENTION_DAYS,
    )


@router.put(
    "/config",
    response_model=StorageConfigResponse,
    tags=["图片存储"],
    dependencies=[Depends(require_admin)],
)
async def update_storage_config(
    request: StorageQuotaUpdateRequest,
    auth: AuthCredentials = Depends(require_admin),
) -> StorageConfigResponse:
    """
    更新存储配置（仅管理员）

    - **quota_bytes**: 新的存储配额（字节）
    - **retention_days**: 保留天数

    需要管理员权限（JWT Token）

    注意：这只更新运行时配置，不持久化到.env 文件
    """
    from util.config import settings

    # 更新配置（注意：这只更新运行时配置，不持久化到.env 文件）
    if request.quota_bytes is not None:
        settings.STORAGE_QUOTA_BYTES = request.quota_bytes
        # 安全更新后端配额（仅本地存储支持）
        if storage_manager.backend and isinstance(storage_manager.backend, LocalStorageBackend):
            storage_manager.backend.update_quota(request.quota_bytes)

    if request.retention_days is not None:
        settings.STORAGE_RETENTION_DAYS = request.retention_days

    return StorageConfigResponse(
        storage_type=settings.STORAGE_TYPE,
        storage_path=settings.STORAGE_LOCAL_PATH,
        quota_bytes=settings.STORAGE_QUOTA_BYTES,
        retention_days=settings.STORAGE_RETENTION_DAYS,
    )


@router.post(
    "/cleanup",
    response_model=ImageDeleteResponse,
    tags=["图片存储"],
    dependencies=[Depends(require_admin)],
)
async def cleanup_storage(
    request: StorageCleanupRequest,
    auth: AuthCredentials = Depends(require_admin),
) -> ImageDeleteResponse:
    """
    清理存储空间（仅管理员）

    - **older_than_days**: 清理 N 天前的图片
    - **task_ids**: 指定任务 ID 列表（可选，指定后只清理这些任务的图片）

    需要管理员权限（JWT Token）
    """
    try:
        result = storage_manager.cleanup_images(
            older_than_days=request.older_than_days,
            task_ids=request.task_ids,
        )

        return ImageDeleteResponse(
            success=True,
            deleted_count=result["deleted_count"],
            freed_size_bytes=result["freed_size_bytes"],
            message=f"清理完成：删除 {result['deleted_count']} 张图片，释放 {result['freed_size_bytes'] / (1024 * 1024):.2f} MB 空间",
        )

    except StorageError as e:
        logger.error(f"Storage error: {e}")
        raise HTTPException(status_code=500, detail=f"清理失败：{str(e)}")
    except Exception as e:
        logger.error(f"Failed to cleanup storage: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"清理失败：{str(e)}")


@router.post(
    "/images/download",
    tags=["图片存储"],
    dependencies=[Depends(require_admin)],
)
async def batch_download_images(
    request: ImageBatchDownloadRequest,
    auth: AuthCredentials = Depends(require_admin),
) -> Response:
    """
    批量下载压缩后的图片（仅管理员）

    - **image_ids**: 要下载的图片 ID 列表

    需要管理员权限（JWT Token）

    返回 ZIP 压缩包，包含所有请求的图片文件
    """
    import io
    import zipfile

    if not request.image_ids:
        raise HTTPException(status_code=400, detail="image_ids 不能为空")

    if len(request.image_ids) > 100:
        raise HTTPException(status_code=400, detail="最多只能同时下载 100 张图片")

    try:
        # 创建 ZIP 文件
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for image_id in request.image_ids:
                # 获取图片数据
                image_data = storage_manager.get_image(image_id)

                if not image_data:
                    logger.warning(f"Image {image_id} not found")
                    continue

                # 获取元数据以确定文件扩展名
                metadata = storage_manager.backend.get_metadata(image_id) # type: ignore
                content_type = metadata.get("content_type", "image/jpeg") if metadata else "image/jpeg"

                # 确定文件扩展名
                ext_map = {
                    "image/jpeg": ".jpg",
                    "image/jpg": ".jpg",
                    "image/png": ".png",
                    "image/gif": ".gif",
                    "image/webp": ".webp",
                }
                ext = ext_map.get(content_type, ".bin")

                # 写入 ZIP 文件
                zip_file.writestr(f"{image_id}{ext}", image_data)
                logger.info(f"Added image {image_id} to zip ({len(image_data)} bytes)")

        # 准备响应
        zip_buffer.seek(0)
        zip_data = zip_buffer.getvalue()

        return Response(
            content=zip_data,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="images_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.zip"',
                "Content-Length": str(len(zip_data)),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to batch download images: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"批量下载失败：{str(e)}")


@router.get(
    "/image-ids",
    response_model=ImageIdListResponse,
    tags=["图片存储"],
    dependencies=[Depends(require_admin)],
)
async def get_all_image_ids(
    auth: AuthCredentials = Depends(require_admin),
) -> ImageIdListResponse:
    """
    获取所有压缩图片的 ID 列表（仅管理员）

    需要管理员权限（JWT Token）

    返回所有已存储图片的 ID 列表，可用于批量下载或其他操作
    """
    try:
        # 查询所有图片（不分页）
        result = storage_manager.query_images(
            page=1,
            page_size=10000,  # 设置一个较大的值以获取所有图片
        )

        # 提取 image_id 列表
        image_ids = [item.get("image_id", "") for item in result.get("items", [])]

        return ImageIdListResponse(
            total=result.get("total", len(image_ids)),
            image_ids=image_ids,
        )

    except Exception as e:
        logger.error(f"Failed to get all image IDs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取图片 ID 列表失败：{str(e)}")
