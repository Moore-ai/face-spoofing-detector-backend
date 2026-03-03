"""
推理路由控制器

提供以下端点：
- POST /infer/single - 单模态检测
- POST /infer/fusion - 融合模态检测
- GET /infer/task/{task_id} - 查询任务状态
- WebSocket /infer/ws - 实时进度推送
"""

import base64
import io
import logging
import asyncio

import numpy as np
from fastapi import APIRouter, WebSocket, Header, HTTPException, Depends, Query
from PIL import Image
from starlette.websockets import WebSocketDisconnect

from typing import Annotated, Optional, List

from lifespan import InferServiceDep, ConnectionManagerDep
from schemas.detection import (
    AsyncTaskResponse,
    DetectionResultItem,
    TaskStatusResponse,
    SingleModeRequest,
    ImagePair,
    FusionModeRequest,
    TaskQueueStatusResponse,
)
from service.progress_service import progress_tracker, TaskProgress
from service.image_auto_save_service import image_auto_save_service
from service.task_scheduler import task_scheduler
from util.auth import AuthCredentials
from middleware.auth_middleware import get_current_user
from db import db_manager

logger = logging.getLogger(__name__)

router = APIRouter()


def decode_base64_image(base64_str: str) -> np.ndarray:
    """将 base64 字符串解码为 numpy 图像数组"""
    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data))
    return np.array(img)


async def run_single_detection_task(
    task_id: str,
    images: list[str],
    infer_service: InferServiceDep,
    connection_manager: ConnectionManagerDep,
    api_key_hash: Optional[str] = None,
    client_id: Optional[str] = None,
) -> None:
    """后台执行单模态检测任务"""
    try:
        decoded_images: list[np.ndarray] = []

        # 解码所有图像并追踪进度
        for i, img_base64 in enumerate(images):
            # 检查取消请求
            if await progress_tracker.check_cancellation(task_id):
                logger.info(f"任务 {task_id} 在解码第 {i} 张图片时被取消")
                await progress_tracker.confirm_cancel(task_id)
                await connection_manager.broadcast_by_task(
                    {
                        "type": "task_cancelled",
                        "data": {
                            "task_id": task_id,
                            "status": "cancelled",
                            "message": "任务已取消",
                            "processed_items": i,
                            "total_items": len(images),
                        },
                    },
                    task_id,
                )
                return

            decoded_images.append(decode_base64_image(img_base64))

        # 定义进度回调函数
        async def progress_callback(current: int, total: int, result: DetectionResultItem) -> None:
            # 每次处理完一张图片后检查取消
            if await progress_tracker.check_cancellation(task_id):
                logger.info(f"任务 {task_id} 在处理第 {current} 张图片时被取消")
                return

            await progress_tracker.update_progress(
                task_id=task_id,
                result=result,
                message=f"完成第 {current}/{total} 张图片的推理",
            )

        # 批量检测（带进度回调）
        batch_results = await infer_service.detect_single_batch(
            decoded_images, progress_callback=progress_callback
        )

        # 再次检查是否被取消（可能在处理过程中被取消）
        if await progress_tracker.check_cancellation(task_id):
            await progress_tracker.confirm_cancel(task_id)
            await connection_manager.broadcast_by_task(
                {
                    "type": "task_cancelled",
                    "data": {
                        "task_id": task_id,
                        "status": "cancelled",
                        "message": "任务已取消",
                        "processed_items": len(batch_results),
                        "total_items": len(decoded_images),
                    },
                },
                task_id,
            )
            logger.info(f"任务 {task_id} 已取消，已处理 {len(batch_results)}/{len(decoded_images)} 项")
            return

        logger.info(f"单模态检测任务 {task_id} 完成，共处理 {len(batch_results)} 张图片")

        # 获取任务状态
        task = await progress_tracker.get_task(task_id)

        if task is None:
            raise HTTPException(status_code=404, detail="任务不存在")

        # 发送任务完成通知
        await connection_manager.broadcast_by_task(
            {
                "type": "task_completed" if task.failed_items == 0 else "task_partial_failure",
                "data": {
                    "task_id": task_id,
                    "status": task.status,
                    "message": task.message,
                    "total_items": len(decoded_images),
                    "processed_items": len(batch_results),
                    "successful_items": len(batch_results) - task.failed_items,
                    "failed_items": task.failed_items,
                    "errors": task.error_items if task.error_items else None,
                },
            },
            task_id,
        )

        # 保存历史记录到数据库
        await save_task_history(
            task_id=task_id,
            client_id=client_id,
            api_key_hash=api_key_hash,
            task=task,
            mode="single",
        )

        # 自动保存图片（根据策略）
        await save_images_auto(
            task_id=task_id,
            client_id=client_id,
            api_key_hash=api_key_hash,
            images=images,
            task=task,
        )

    except Exception as e:
        await progress_tracker.fail_task(task_id, str(e))
        logger.error(f"单模态检测任务 {task_id} 失败：{e}", exc_info=True)

        # 发送失败通知给对应的客户
        await connection_manager.broadcast_by_task(
            {
                "type": "task_failed",
                "data": {
                    "task_id": task_id,
                    "status": "failed",
                    "message": str(e),
                },
            },
            task_id,
        )

    finally:
        # 立即清理连接映射，任务数据持久保存供查询
        await connection_manager.unregister_task(task_id)


async def run_fusion_detection_task(
    task_id: str,
    pairs: list[ImagePair],
    infer_service: InferServiceDep,
    connection_manager: ConnectionManagerDep,
    api_key_hash: Optional[str] = None,
    client_id: Optional[str] = None,
) -> None:
    """后台执行融合模态检测任务"""
    try:
        decoded_pairs: list[tuple[np.ndarray, np.ndarray]] = []

        # 解码所有图像对并追踪进度
        for i, pair in enumerate(pairs):
            # 检查取消请求
            if await progress_tracker.check_cancellation(task_id):
                logger.info(f"任务 {task_id} 在解码第 {i} 对图像时被取消")
                await progress_tracker.confirm_cancel(task_id)
                await connection_manager.broadcast_by_task(
                    {
                        "type": "task_cancelled",
                        "data": {
                            "task_id": task_id,
                            "status": "cancelled",
                            "message": "任务已取消",
                            "processed_items": i,
                            "total_items": len(pairs),
                        },
                    },
                    task_id,
                )
                return

            decoded_pairs.append(
                (decode_base64_image(pair.rgb), decode_base64_image(pair.ir))
            )

        # 定义进度回调函数
        async def progress_callback(current: int, total: int, result: DetectionResultItem) -> None:
            # 每次处理完一对图像后检查取消
            if await progress_tracker.check_cancellation(task_id):
                logger.info(f"任务 {task_id} 在处理第 {current} 对图像时被取消")
                return

            await progress_tracker.update_progress(
                task_id=task_id,
                result=result,
                message=f"完成第 {current}/{total} 对图像的推理",
            )

        # 批量检测（带进度回调）
        batch_results = await infer_service.detect_fusion_batch(
            decoded_pairs, progress_callback=progress_callback
        )

        # 再次检查是否被取消（可能在处理过程中被取消）
        if await progress_tracker.check_cancellation(task_id):
            await progress_tracker.confirm_cancel(task_id)
            await connection_manager.broadcast_by_task(
                {
                    "type": "task_cancelled",
                    "data": {
                        "task_id": task_id,
                        "status": "cancelled",
                        "message": "任务已取消",
                        "processed_items": len(batch_results),
                        "total_items": len(decoded_pairs),
                    },
                },
                task_id,
            )
            logger.info(f"任务 {task_id} 已取消，已处理 {len(batch_results)}/{len(decoded_pairs)} 项")
            return

        logger.info(f"融合模态检测任务 {task_id} 完成，共处理 {len(batch_results)} 对图像")

        # 获取任务状态
        task = await progress_tracker.get_task(task_id)

        if task is None:
            raise HTTPException(status_code=404, detail="任务不存在")

        # 发送任务完成通知
        await connection_manager.broadcast_by_task(
            {
                "type": "task_completed" if task.failed_items == 0 else "task_partial_failure",
                "data": {
                    "task_id": task_id,
                    "status": task.status,
                    "message": task.message,
                    "total_items": len(decoded_pairs),
                    "processed_items": len(batch_results),
                    "successful_items": len(batch_results) - task.failed_items,
                    "failed_items": task.failed_items,
                    "errors": task.error_items if task.error_items else None,
                },
            },
            task_id,
        )

        # 保存历史记录到数据库
        await save_task_history(
            task_id=task_id,
            client_id=client_id,
            api_key_hash=api_key_hash,
            task=task,
            mode="fusion",
        )

        # 自动保存图片（根据策略）
        await save_images_auto(
            task_id=task_id,
            client_id=client_id,
            api_key_hash=api_key_hash,
            images=[pair.rgb for pair in pairs],  # 只保存 RGB 图片
            task=task,
        )

    except Exception as e:
        await progress_tracker.fail_task(task_id, str(e))
        logger.error(f"融合模态检测任务 {task_id} 失败：{e}", exc_info=True)

        # 发送失败通知给对应的客户
        await connection_manager.broadcast_by_task(
            {
                "type": "task_failed",
                "data": {
                    "task_id": task_id,
                    "status": "failed",
                    "message": str(e),
                },
            },
            task_id,
        )

    finally:
        # 立即清理连接映射，任务数据持久保存供查询
        await connection_manager.unregister_task(task_id)


async def save_task_history(
    task_id: str,
    client_id: Optional[str],
    api_key_hash: Optional[str],
    task: TaskProgress,
    mode: str = "single",
) -> None:
    """保存任务历史记录到数据库"""
    try:
        db = db_manager.get_session()
        try:
            # 将结果转换为字典列表
            results_data = []
            for result in task.all_results:
                results_data.append({
                    "mode": result.mode,
                    "modality": getattr(result, "modality", None),
                    "result": result.result,
                    "confidence": result.confidence,
                    "probabilities": result.probabilities.tolist() if hasattr(result.probabilities, "tolist") else result.probabilities,
                    "processing_time": result.processing_time,
                    "image_index": result.image_index,
                    "error": result.error,
                    "retry_count": result.retry_count,
                    "success": result.success,
                })

            # 调用历史服务保存
            from service.history_service import DetectionHistoryService

            DetectionHistoryService.save_task(
                db=db,
                task_id=task_id,
                client_id=client_id,
                api_key_hash=api_key_hash,
                mode=mode,
                status=task.status,
                total_items=task.total_items,
                successful_items=task.completed_items - task.failed_items,
                failed_items=task.failed_items,
                real_count=task.real_count,
                fake_count=task.fake_count,
                error_count=task.failed_items,
                elapsed_time_ms=task.elapsed_time_ms,
                results=results_data,
            )
            logger.info(f"任务 {task_id} 历史记录已保存 (mode={mode})")
        finally:
            db.close()
    except Exception as e:
        logger.error(f"保存任务历史记录失败 {task_id}: {e}", exc_info=True)
        # 不抛出异常，避免影响主流程


async def save_images_auto(
    task_id: str,
    client_id: Optional[str],
    api_key_hash: Optional[str],
    images: List[str],
    task: TaskProgress,
) -> None:
    """自动保存图片（根据策略）"""
    try:
        # 准备结果数据
        results = []
        for result in task.all_results:
            results.append({
                "mode": result.mode,
                "modality": getattr(result, "modality", "rgb"),
                "result": result.result,
                "confidence": result.confidence,
                "image_index": result.image_index,
                "error": result.error,
                "retry_count": result.retry_count,
                "success": result.success,
                "processing_time": result.processing_time,
            })

        # 调用自动存储服务
        await image_auto_save_service.save_images_from_results(
            task_id=task_id,
            client_id=client_id,
            api_key_hash=api_key_hash,
            results=results,
            original_images=images,
        )
    except Exception as e:
        logger.error(f"自动保存图片失败 {task_id}: {e}", exc_info=True)
        # 不抛出异常，避免影响主流程


@router.post("/single", response_model=AsyncTaskResponse)
async def detect_single_mode(
    request: SingleModeRequest,
    infer_service: InferServiceDep,
    connection_manager: ConnectionManagerDep,
    auth: Annotated[AuthCredentials, Depends(get_current_user)],
    x_client_id: str = Header(..., description="WebSocket 分配的客户端 ID"),
):
    """单模态推理端点（异步模式）

    需要先在 WebSocket 连接中获取 client_id，然后通过 Header 传入。
    需要 API Key 认证（通过 X-API-Key 请求头或 Authorization: Bearer）。
    立即返回 task_id，后台异步执行检测任务。
    任务进度会自动推送到对应的 WebSocket 客户端。

    任务会根据 API Key 的优先级被调度执行，高优先级任务优先处理。
    """
    # 验证认证
    if not auth.authenticated:
        raise HTTPException(status_code=401, detail="未授权访问，需要 API Key 或 JWT Token")

    # 验证 client_id 是否存在
    if x_client_id not in connection_manager.active_connections:
        return AsyncTaskResponse(
            task_id="",
            message=f"无效的 client_id: {x_client_id}，请先建立 WebSocket 连接",
        )

    # 创建任务（使用 API Key 的优先级）
    task_id = await progress_tracker.create_task(
        total_items=len(request.images),
        client_id=x_client_id,
        priority=auth.priority,  # 使用 API Key 的优先级
    )
    await progress_tracker.start_task(task_id)

    # 注册任务到客户
    if not await connection_manager.register_task(x_client_id, task_id):
        await progress_tracker.fail_task(task_id, "任务注册失败，该任务可能已被其他客户注册")
        return AsyncTaskResponse(
            task_id=task_id,
            message="任务注册失败",
        )

    # 获取 API Key 哈希用于历史记录
    api_key_hash = None
    if auth.authenticated and auth.auth_type == "api_key":
        # 从 auth 凭据中提取 API Key 哈希
        # 由于 auth.user_id 格式为 "api_key:{name}"，我们直接使用 name 作为标识
        if auth.user_id and auth.user_id.startswith("api_key:"):
            api_key_hash = auth.user_id.replace("api_key:", "")[:32]  # 截断作为哈希
        else:
            api_key_hash = str(auth.user_id or "unknown")[:32]

    # 提交任务到调度器（按优先级调度执行）
    success = await task_scheduler.submit_task(
        task_id=task_id,
        task_type="single",
        task_data={
            "task_id": task_id,
            "images": request.images,
            "api_key_hash": api_key_hash,
            "client_id": x_client_id,
        },
        priority=auth.priority,
        callback=lambda data: run_single_detection_task(
            task_id=data["task_id"],
            images=data["images"],
            infer_service=infer_service,
            connection_manager=connection_manager,
            api_key_hash=data["api_key_hash"],
            client_id=data["client_id"],
        ),
    )

    if not success:
        await progress_tracker.fail_task(task_id, "任务提交失败，调度器未就绪")
        return AsyncTaskResponse(
            task_id=task_id,
            message="任务提交失败，请稍后重试",
        )

    return AsyncTaskResponse(
        task_id=task_id,
        message="单模态检测任务已创建，等待调度执行",
    )


@router.post("/fusion", response_model=AsyncTaskResponse)
async def detect_fusion_mode(
    request: FusionModeRequest,
    infer_service: InferServiceDep,
    connection_manager: ConnectionManagerDep,
    auth: Annotated[AuthCredentials, Depends(get_current_user)],
    x_client_id: str = Header(..., description="WebSocket 分配的客户端 ID"),
):
    """融合模态推理端点（异步模式）

    需要先在 WebSocket 连接中获取 client_id，然后通过 Header 传入。
    需要 API Key 认证（通过 X-API-Key 请求头或 Authorization: Bearer）。
    立即返回 task_id，后台异步执行检测任务。
    任务进度会自动推送到对应的 WebSocket 客户端。

    任务会根据 API Key 的优先级被调度执行，高优先级任务优先处理。
    """
    # 验证认证
    if not auth.authenticated:
        raise HTTPException(status_code=401, detail="未授权访问，需要 API Key 或 JWT Token")

    # 验证 client_id 是否存在
    if x_client_id not in connection_manager.active_connections:
        return AsyncTaskResponse(
            task_id="",
            message=f"无效的 client_id: {x_client_id}，请先建立 WebSocket 连接",
        )

    # 创建任务（使用 API Key 的优先级）
    task_id = await progress_tracker.create_task(
        total_items=len(request.pairs),
        client_id=x_client_id,
        priority=auth.priority,  # 使用 API Key 的优先级
    )
    await progress_tracker.start_task(task_id)

    # 注册任务到客户
    if not await connection_manager.register_task(x_client_id, task_id):
        await progress_tracker.fail_task(task_id, "任务注册失败，该任务可能已被其他客户注册")
        return AsyncTaskResponse(
            task_id=task_id,
            message="任务注册失败",
        )

    # 获取 API Key 哈希用于历史记录
    api_key_hash = None
    if auth.authenticated and auth.auth_type == "api_key":
        if auth.user_id and auth.user_id.startswith("api_key:"):
            api_key_hash = auth.user_id.replace("api_key:", "")[:32]
        else:
            api_key_hash = str(auth.user_id or "unknown")[:32]

    # 提交任务到调度器（按优先级调度执行）
    success = await task_scheduler.submit_task(
        task_id=task_id,
        task_type="fusion",
        task_data={
            "task_id": task_id,
            "pairs": request.pairs,
            "api_key_hash": api_key_hash,
            "client_id": x_client_id,
        },
        priority=auth.priority,
        callback=lambda data: run_fusion_detection_task(
            task_id=data["task_id"],
            pairs=data["pairs"],
            infer_service=infer_service,
            connection_manager=connection_manager,
            api_key_hash=data["api_key_hash"],
            client_id=data["client_id"],
        ),
    )

    if not success:
        await progress_tracker.fail_task(task_id, "任务提交失败，调度器未就绪")
        return AsyncTaskResponse(
            task_id=task_id,
            message="任务提交失败，请稍后重试",
        )

    return AsyncTaskResponse(
        task_id=task_id,
        message="融合模态检测任务已创建，等待调度执行",
    )


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    connection_manager: ConnectionManagerDep,
):
    """WebSocket 端点 - 实时进度推送

    连接建立后，后端生成 client_id 并发送给客户端。
    支持通过查询参数进行认证：
    - ?api_key=sk_xxx 或 ?token=jwt_xxx
    任务进度会自动推送给对应客户端，无需订阅。
    """
    # WebSocket 认证（可选）
    from middleware.auth_middleware import get_websocket_auth
    auth = await get_websocket_auth(websocket)

    # 如果提供了认证信息但无效，拒绝连接
    if (websocket.query_params.get("api_key") or websocket.query_params.get("token")) and not auth.authenticated:
        await websocket.close(code=4001, reason="未授权访问")
        return

    client_id = await connection_manager.connect(websocket)

    await websocket.send_json(
        {
            "type": "connected",
            "client_id": client_id,
        }
    )

    try:
        while True:
            # 保持连接活跃，不处理消息
            await websocket.receive_text()
    except WebSocketDisconnect:
        logger.debug(f"WebSocket 客户端 {client_id} 正常断开连接")
    except Exception as e:
        logger.error(f"WebSocket 客户端 {client_id} 连接异常：{e}")
    finally:
        await connection_manager.disconnect(client_id)


@router.get("/task/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: str,
    auth: Annotated[AuthCredentials, Depends(get_current_user)],
):
    """获取任务状态和结果

    需要 API Key 认证。
    返回任务的当前状态、进度信息和所有检测结果。
    """
    # 验证认证
    if not auth.authenticated:
        raise HTTPException(status_code=401, detail="未授权访问，需要 API Key 或 JWT Token")

    task = await progress_tracker.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")

    # 构建响应
    return TaskStatusResponse(
        task_id=task.task_id,
        status=task.status,
        total_items=task.total_items,
        completed_items=task.completed_items,
        failed_items=task.failed_items,
        progress_percentage=round(task.progress_percentage, 2),
        real_count=task.real_count,
        fake_count=task.fake_count,
        error_count=task.failed_items,
        elapsed_time_ms=task.elapsed_time_ms,
        message=task.message,
        results=task.all_results if task.status in ("completed", "partial_failure", "failed") else None,
        current_result=task.current_result,
        errors=task.error_items if task.error_items else None,
        priority=task.priority,
    )


@router.delete("/task/{task_id}", response_model=TaskStatusResponse)
async def cancel_task_endpoint(
    task_id: str,
    auth: Annotated[AuthCredentials, Depends(get_current_user)],
):
    """取消任务

    需要 API Key 认证。
    只有正在运行的任务可以被取消，已完成或已失败的任务无法取消。
    """
    # 验证认证
    if not auth.authenticated:
        raise HTTPException(status_code=401, detail="未授权访问，需要 API Key 或 JWT Token")

    task = await progress_tracker.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    # 尝试取消任务
    success = await progress_tracker.cancel_task(task_id, "用户请求取消")
    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"无法取消任务，当前状态：{task.status}",
        )

    # 等待一小段时间让任务处理取消逻辑
    await asyncio.sleep(0.1)

    # 返回更新后的任务状态
    updated_task = await progress_tracker.get_task(task_id)
    assert updated_task
    return TaskStatusResponse(
        task_id=updated_task.task_id,
        status=updated_task.status,
        total_items=updated_task.total_items,
        completed_items=updated_task.completed_items,
        failed_items=updated_task.failed_items,
        progress_percentage=round(updated_task.progress_percentage, 2),
        real_count=updated_task.real_count,
        fake_count=updated_task.fake_count,
        error_count=updated_task.failed_items,
        elapsed_time_ms=updated_task.elapsed_time_ms,
        message=updated_task.message,
        results=None,  # 取消时不返回完整结果
        current_result=updated_task.current_result,
        errors=None,
        priority=updated_task.priority,
    )


@router.get("/tasks", response_model=List[TaskStatusResponse])
async def list_tasks(
    auth: Annotated[AuthCredentials, Depends(get_current_user)],
    x_client_id: str = Header(..., description="WebSocket 分配的客户端 ID"),
    status: Optional[str] = Query(None, description="按状态过滤：pending, running, completed, partial_failure, failed, cancelled"),
):
    """获取当前客户端的所有任务列表

    需要 API Key 认证。
    返回该客户端创建的所有任务，支持按状态过滤。
    """
    # 验证认证
    if not auth.authenticated:
        raise HTTPException(status_code=401, detail="未授权访问")

    # 获取该客户端的所有任务
    tasks = await progress_tracker.get_client_tasks(x_client_id)

    # 按状态过滤
    if status:
        tasks = [t for t in tasks if t.status == status]

    # 转换为响应格式
    return [
        TaskStatusResponse(
            task_id=task.task_id,
            status=task.status,
            total_items=task.total_items,
            completed_items=task.completed_items,
            failed_items=task.failed_items,
            progress_percentage=round(task.progress_percentage, 2),
            real_count=task.real_count,
            fake_count=task.fake_count,
            error_count=task.failed_items,
            elapsed_time_ms=task.elapsed_time_ms,
            message=task.message,
            results=None,  # 列表不返回完整结果
            current_result=task.current_result,
            errors=task.error_items if task.error_items else None,
            priority=task.priority,
        )
        for task in tasks
    ]


@router.get("/queue/status", response_model=TaskQueueStatusResponse)
async def get_queue_status(
    auth: Annotated[AuthCredentials, Depends(get_current_user)],
):
    """获取任务队列状态（仅管理员）

    需要 JWT Token 认证。
    返回调度器状态、队列大小、工作线程数等信息。
    """
    # 验证认证（仅管理员）
    if not auth.authenticated or auth.auth_type != "jwt":
        raise HTTPException(status_code=403, detail="仅管理员可访问")

    status = await task_scheduler.get_queue_status()
    return TaskQueueStatusResponse(**status)
