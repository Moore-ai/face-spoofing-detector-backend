"""
检测结果历史记录控制器
提供查询、删除历史记录的 API 端点
"""

import logging
from typing import Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from db import get_db_session
from service.history_service import DetectionHistoryService
from schemas.history import (
    HistoryQueryResponse,
    HistoryStatsResponse,
)
from util.auth import AuthCredentials
from middleware.auth_middleware import get_current_user

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


@router.get("/history", response_model=HistoryQueryResponse, tags=["历史记录"])
async def query_history(
    client_id: Optional[str] = Query(None, description="客户端 ID 过滤"),
    mode: Optional[str] = Query(None, description="模式过滤 (single/fusion)"),
    status: Optional[str] = Query(None, description="状态过滤 (逗号分隔的列表)"),
    days: Optional[int] = Query(None, description="最近 N 天的记录"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    db: Session = Depends(get_db),
    auth: AuthCredentials = Depends(require_auth),
):
    """
    查询检测历史记录

    - **client_id**: 按客户端 ID 过滤（可选）
    - **mode**: 按模式过滤，single 或 fusion（可选）
    - **status**: 按状态过滤，多个状态用逗号分隔（可选）
    - **days**: 查询最近 N 天的记录（可选）
    - **page**: 页码，从 1 开始
    - **page_size**: 每页数量，最大 100

    需要 API Key 或 JWT Token 认证
    """
    # 解析状态列表
    status_list = None
    if status:
        status_list = [s.strip() for s in status.split(",")]

    # 计算日期范围
    start_date = None
    end_date = None
    if days:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

    try:
        # 如果是普通用户（API Key 认证），只能查询自己的记录
        if auth.auth_type == "api_key":
            # 从 auth 凭据中提取 API Key 哈希（与保存时保持一致）
            # auth.user_id 格式为 "api_key:{name}"，我们提取 name 部分作为哈希
            if auth.user_id and auth.user_id.startswith("api_key:"):
                api_key_hash = auth.user_id.replace("api_key:", "")[:32]
            else:
                api_key_hash = str(auth.user_id or "unknown")[:32]

            result = DetectionHistoryService.query_tasks(
                db=db,
                api_key_hash=api_key_hash,
                client_id=client_id,
                mode=mode,
                status=status_list,
                start_date=start_date,
                end_date=end_date,
                page=page,
                page_size=page_size,
            )
        else:
            # JWT Token 认证（管理员）可以查询所有记录或按条件过滤
            result = DetectionHistoryService.query_tasks(
                db=db,
                client_id=client_id,
                mode=mode,
                status=status_list,
                start_date=start_date,
                end_date=end_date,
                page=page,
                page_size=page_size,
            )

        return result

    except Exception as e:
        logger.error(f"Failed to query history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"查询失败：{str(e)}")


@router.get("/history/stats", response_model=HistoryStatsResponse, tags=["历史记录"])
async def get_history_stats(
    client_id: Optional[str] = Query(None, description="客户端 ID 过滤"),
    mode: Optional[str] = Query(None, description="模式过滤 (single/fusion)"),
    status: Optional[str] = Query(None, description="状态过滤 (逗号分隔的列表)"),
    days: Optional[int] = Query(None, description="最近 N 天的统计"),
    db: Session = Depends(get_db),
    auth: AuthCredentials = Depends(require_auth),
):
    """
    获取检测历史统计信息

    - **client_id**: 按客户端 ID 过滤（可选）
    - **mode**: 按模式过滤（可选）
    - **status**: 按状态过滤，逗号分隔（可选）
    - **days**: 统计最近 N 天的数据（可选，不指定则统计所有）

    返回总任务数、总推理数、真假人脸计数、成功率、平均处理时间等

    需要 API Key 或 JWT Token 认证
    """
    # 计算日期范围
    start_date = None
    end_date = None
    if days:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

    # 解析状态列表
    status_list = None
    if status:
        status_list = [s.strip() for s in status.split(",")]

    try:
        # 如果是普通用户，只能查看自己的统计
        if auth.auth_type == "api_key":
            # 从 auth 凭据中提取 API Key 哈希（与保存时保持一致）
            if auth.user_id and auth.user_id.startswith("api_key:"):
                api_key_hash = auth.user_id.replace("api_key:", "")[:32]
            else:
                api_key_hash = str(auth.user_id or "unknown")[:32]

            result = DetectionHistoryService.get_stats(
                db=db,
                api_key_hash=api_key_hash,
                mode=mode,
                status=status_list,
                start_date=start_date,
                end_date=end_date,
            )
        else:
            # 管理员可以查看所有统计或按条件过滤
            result = DetectionHistoryService.get_stats(
                db=db,
                client_id=client_id,
                mode=mode,
                status=status_list,
                start_date=start_date,
                end_date=end_date,
            )

        return result

    except Exception as e:
        logger.error(f"Failed to get stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取统计失败：{str(e)}")


@router.delete("/history", tags=["历史记录"])
async def delete_history(
    task_ids: Optional[str] = Query(None, description="要删除的任务 ID 列表（逗号分隔）"),
    days_ago: Optional[int] = Query(None, description="删除早于 N 天的记录", ge=1),
    db: Session = Depends(get_db),
    auth: AuthCredentials = Depends(require_auth),
):
    """
    删除检测历史记录

    - **task_ids**: 要删除的任务 ID 列表，用逗号分隔（可选）
    - **days_ago**: 删除早于 N 天的记录（可选）

    需要 API Key 或 JWT Token 认证。普通用户只能删除自己的记录，管理员可以删除所有记录。

    注意：task_ids 和 days_ago 只能使用其中一个
    """
    # 验证参数
    if not task_ids and not days_ago:
        raise HTTPException(status_code=400, detail="必须指定 task_ids 或 days_ago 参数")

    if task_ids and days_ago:
        raise HTTPException(status_code=400, detail="task_ids 和 days_ago 只能使用其中一个")

    try:
        deleted_count = 0

        if task_ids:
            # 删除指定任务
            ids = [tid.strip() for tid in task_ids.split(",")]

            # 普通用户只能删除自己的记录，管理员可以删除所有记录
            if auth.auth_type == "api_key":
                # 从 auth 凭据中提取 API Key 哈希（与保存时保持一致）
                api_key_hash = None
                if auth.user_id and auth.user_id.startswith("api_key:"):
                    api_key_hash = auth.user_id.replace("api_key:", "")[:32]
                else:
                    api_key_hash = str(auth.user_id or "unknown")[:32]

                deleted_count = DetectionHistoryService.delete_tasks(
                    db=db, task_ids=ids, api_key_hash=api_key_hash
                )
            else:
                # JWT 管理员直接删除
                deleted_count = DetectionHistoryService.delete_tasks(db=db, task_ids=ids)
        elif days_ago:
            # 删除旧记录
            deleted_count = DetectionHistoryService.delete_tasks(
                db=db, older_than_days=days_ago
            )

        return {
            "deleted_count": deleted_count,
            "message": f"成功删除 {deleted_count} 条记录",
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to delete history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"删除失败：{str(e)}")


@router.get("/history/task/{task_id}", tags=["历史记录"])
async def get_task_history(
    task_id: str,
    db: Session = Depends(get_db),
    auth: AuthCredentials = Depends(require_auth),
):
    """
    获取单个任务的详细历史记录

    - **task_id**: 任务 ID

    需要 API Key 或 JWT Token 认证，用户只能查看自己的任务（管理员除外）
    """
    try:
        task = DetectionHistoryService.get_task_by_id(db=db, task_id=task_id)

        if not task:
            raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")

        # 权限检查：非管理员只能查看自己的任务
        if auth.auth_type == "api_key":
            # 从 auth 凭据中提取 API Key 哈希（与保存时保持一致）
            if auth.user_id and auth.user_id.startswith("api_key:"):
                api_key_hash = auth.user_id.replace("api_key:", "")[:32]
            else:
                api_key_hash = str(auth.user_id or "unknown")[:32]

            task_api_key_hash = getattr(task, 'api_key_hash', None)
            if task_api_key_hash is not None and task_api_key_hash != api_key_hash:
                raise HTTPException(status_code=403, detail="无权查看此任务")

        # 转换为响应格式
        results = []
        for result in task.results:
            results.append(
                {
                    "mode": result.mode,
                    "modality": result.modality,
                    "result": result.result,
                    "confidence": result.confidence,
                    "probabilities": [result.prob_real, result.prob_fake],
                    "processing_time": result.processing_time,
                    "image_index": result.image_index,
                    "error": result.error_message,
                    "retry_count": result.retry_count,
                }
            )

        return {
            "task_id": task.task_id,
            "client_id": task.client_id,
            "mode": task.mode,
            "status": task.status,
            "total_items": task.total_items,
            "successful_items": task.successful_items,
            "failed_items": task.failed_items,
            "real_count": task.real_count,
            "fake_count": task.fake_count,
            "error_count": task.error_count,
            "elapsed_time_ms": task.elapsed_time_ms,
            "created_at": task.created_at.isoformat(),
            "completed_at": task.completed_at.isoformat() if task.completed_at is not None else None,
            "results": results,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"查询失败：{str(e)}")
