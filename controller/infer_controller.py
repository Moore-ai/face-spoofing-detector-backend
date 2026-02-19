import asyncio
import base64
import io
import logging

import numpy as np
from fastapi import APIRouter, WebSocket, BackgroundTasks, Header, HTTPException
from PIL import Image
from pydantic import BaseModel
from starlette.websockets import WebSocketDisconnect

from lifespan import InferServiceDep, ConnectionManagerDep
from schemas.detection import AsyncTaskResponse, DetectionResultItem, TaskStatusResponse
from service.progress_service import progress_tracker

logger = logging.getLogger(__name__)

router = APIRouter()


class SingleModeRequest(BaseModel):
    mode: str
    modality: str
    images: list[str]


class ImagePair(BaseModel):
    rgb: str
    ir: str


class FusionModeRequest(BaseModel):
    mode: str
    pairs: list[ImagePair]


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
) -> None:
    """后台执行单模态检测任务"""
    try:
        decoded_images: list[np.ndarray] = []

        # 解码所有图像并追踪进度
        for _, img_base64 in enumerate(images):
            decoded_images.append(decode_base64_image(img_base64))

        # 定义进度回调函数
        def progress_callback(current: int, total: int, result: DetectionResultItem) -> None:
            progress_tracker.update_progress(
                task_id=task_id,
                result=result,
                message=f"完成第 {current}/{total} 张图片的推理",
            )

        # 批量检测（带进度回调）
        batch_results = infer_service.detect_single_batch(
            decoded_images, progress_callback=progress_callback
        )
        logger.info(f"单模态检测任务 {task_id} 完成，共处理 {len(batch_results)} 张图片")

        # 发送任务完成通知
        await connection_manager.broadcast_by_task(
            {
                "type": "task_completed",
                "data": {
                    "task_id": task_id,
                    "status": "completed",
                    "message": "任务已完成，可查询检测结果",
                    "total_items": len(decoded_images),
                    "processed_items": len(batch_results),
                },
            },
            task_id,
        )

    except Exception as e:
        progress_tracker.fail_task(task_id, str(e))
        logger.error(f"单模态检测任务 {task_id} 失败: {e}", exc_info=True)

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
        connection_manager.unregister_task(task_id)


async def run_fusion_detection_task(
    task_id: str,
    pairs: list[ImagePair],
    infer_service: InferServiceDep,
    connection_manager: ConnectionManagerDep,
) -> None:
    """后台执行融合模态检测任务"""
    try:
        decoded_pairs: list[tuple[np.ndarray, np.ndarray]] = []

        # 解码所有图像对并追踪进度
        for _, pair in enumerate(pairs):
            decoded_pairs.append(
                (decode_base64_image(pair.rgb), decode_base64_image(pair.ir))
            )

        # 定义进度回调函数
        def progress_callback(current: int, total: int, result: DetectionResultItem) -> None:
            progress_tracker.update_progress(
                task_id=task_id,
                result=result,
                message=f"完成第 {current}/{total} 对图像的推理",
            )

        # 批量检测（带进度回调）
        batch_results = infer_service.detect_fusion_batch(
            decoded_pairs, progress_callback=progress_callback
        )

        logger.info(f"融合模态检测任务 {task_id} 完成，共处理 {len(batch_results)} 对图像")

        # 发送任务完成通知
        await connection_manager.broadcast_by_task(
            {
                "type": "task_completed",
                "data": {
                    "task_id": task_id,
                    "status": "completed",
                    "message": "任务已完成，可查询检测结果",
                    "total_items": len(decoded_pairs),
                    "processed_items": len(batch_results),
                },
            },
            task_id,
        )

    except Exception as e:
        progress_tracker.fail_task(task_id, str(e))
        logger.error(f"融合模态检测任务 {task_id} 失败: {e}", exc_info=True)

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
        connection_manager.unregister_task(task_id)


@router.post("/single", response_model=AsyncTaskResponse)
async def detect_single_mode(
    request: SingleModeRequest,
    background_tasks: BackgroundTasks,
    infer_service: InferServiceDep,
    connection_manager: ConnectionManagerDep,
    x_client_id: str = Header(..., description="WebSocket 分配的客户端 ID"),
):
    """单模态推理端点（异步模式）

    需要先在 WebSocket 连接中获取 client_id，然后通过 Header 传入。
    立即返回 task_id，后台异步执行检测任务。
    任务进度会自动推送到对应的 WebSocket 客户端。
    """
    # 验证 client_id 是否存在
    if x_client_id not in connection_manager.active_connections:
        return AsyncTaskResponse(
            task_id="",
            message=f"无效的 client_id: {x_client_id}，请先建立 WebSocket 连接",
        )

    # 创建任务
    task_id = progress_tracker.create_task(total_items=len(request.images))
    progress_tracker.start_task(task_id)

    # 注册任务到客户
    if not connection_manager.register_task(x_client_id, task_id):
        progress_tracker.fail_task(task_id, "任务注册失败，该任务可能已被其他客户注册")
        return AsyncTaskResponse(
            task_id=task_id,
            message="任务注册失败",
        )

    # 后台异步执行检测
    background_tasks.add_task(
        run_single_detection_task,
        task_id=task_id,
        images=request.images,
        infer_service=infer_service,
        connection_manager=connection_manager,
    )

    return AsyncTaskResponse(
        task_id=task_id,
        message="单模态检测任务已创建，正在后台处理",
    )


@router.post("/fusion", response_model=AsyncTaskResponse)
async def detect_fusion_mode(
    request: FusionModeRequest,
    background_tasks: BackgroundTasks,
    infer_service: InferServiceDep,
    connection_manager: ConnectionManagerDep,
    x_client_id: str = Header(..., description="WebSocket 分配的客户端 ID"),
):
    """融合模态推理端点（异步模式）

    需要先在 WebSocket 连接中获取 client_id，然后通过 Header 传入。
    立即返回 task_id，后台异步执行检测任务。
    任务进度会自动推送到对应的 WebSocket 客户端。
    """
    # 验证 client_id 是否存在
    if x_client_id not in connection_manager.active_connections:
        return AsyncTaskResponse(
            task_id="",
            message=f"无效的 client_id: {x_client_id}，请先建立 WebSocket 连接",
        )

    # 创建任务
    task_id = progress_tracker.create_task(total_items=len(request.pairs))
    progress_tracker.start_task(task_id)

    # 注册任务到客户
    if not connection_manager.register_task(x_client_id, task_id):
        progress_tracker.fail_task(task_id, "任务注册失败，该任务可能已被其他客户注册")
        return AsyncTaskResponse(
            task_id=task_id,
            message="任务注册失败",
        )

    # 后台异步执行检测
    background_tasks.add_task(
        run_fusion_detection_task,
        task_id=task_id,
        pairs=request.pairs,
        infer_service=infer_service,
        connection_manager=connection_manager,
    )

    return AsyncTaskResponse(
        task_id=task_id,
        message="融合模态检测任务已创建，正在后台处理",
    )


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    connection_manager: ConnectionManagerDep,
):
    """WebSocket 端点 - 实时进度推送

    连接建立后，后端生成 client_id 并发送给客户端。
    任务进度会自动推送给对应客户端，无需订阅。
    """
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
        logger.error(f"WebSocket 客户端 {client_id} 连接异常: {e}")
    finally:
        connection_manager.disconnect(client_id)


@router.get("/task/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """获取任务状态和结果

    返回任务的当前状态、进度信息和所有检测结果。
    """
    task = progress_tracker.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")

    # 构建响应
    return TaskStatusResponse(
        task_id=task.task_id,
        status=task.status,
        total_items=task.total_items,
        completed_items=task.completed_items,
        progress_percentage=round(task.progress_percentage, 2),
        real_count=task.real_count,
        fake_count=task.fake_count,
        elapsed_time_ms=task.elapsed_time_ms,
        message=task.message,
        results=task.all_results if task.status in ("completed", "failed") else None,
        current_result=task.current_result,
    )
