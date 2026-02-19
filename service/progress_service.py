"""
进度追踪服务
管理批量检测任务的进度状态
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from util.websocket_manager import connection_manager
from schemas.detection import DetectionResultItem

logger = logging.getLogger(__name__)


@dataclass
class TaskProgress:
    """任务进度数据类"""

    task_id: str
    total_items: int
    completed_items: int = 0
    status: str = ""  # pending, running, completed, failed
    current_result: Optional[DetectionResultItem] = None  # 当前检测结果
    all_results: List[DetectionResultItem] = field(default_factory=list)  # 所有检测结果
    real_count: int = 0
    fake_count: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    message: str = ""

    @property
    def progress_percentage(self) -> float:
        """计算进度百分比"""
        if self.total_items == 0:
            return 0.0
        return (self.completed_items / self.total_items) * 100

    @property
    def elapsed_time_ms(self) -> int:
        """计算已用时间（毫秒）"""
        if self.start_time is None:
            return 0
        end = self.end_time or time.time()
        return int((end - self.start_time) * 1000)

    def set_result(self, detection_result: DetectionResultItem) -> None:
        """添加单次检测结果"""
        self.current_result = detection_result
        self.all_results.append(detection_result)
        self.completed_items += 1
        if detection_result.result == "real":
            self.real_count += 1
        else:
            self.fake_count += 1

    def to_dict(self, include_all_results: bool = False) -> dict:
        """转换为字典格式用于 JSON 序列化

        Args:
            include_all_results: 是否包含所有结果列表，默认只包含当前结果
        """
        result = {
            "task_id": self.task_id,
            "total_items": self.total_items,
            "completed_items": self.completed_items,
            "progress_percentage": round(self.progress_percentage, 2),
            "status": self.status,
            "current_result": self.current_result.model_dump() if self.current_result else None,
            "real_count": self.real_count,
            "fake_count": self.fake_count,
            "elapsed_time_ms": self.elapsed_time_ms,
            "message": self.message,
        }

        if include_all_results:
            result["all_results"] = [r.model_dump() for r in self.all_results]

        return result


class ProgressTracker:
    """进度追踪器"""

    def __init__(self):
        # 存储所有任务: {task_id: TaskProgress}
        self.progresses: Dict[str, TaskProgress] = {}

    def create_task(self, total_items: int) -> str:
        """创建新任务"""
        task_id = str(uuid.uuid4())
        task = TaskProgress(
            task_id=task_id,
            total_items=total_items,
            status="pending",
        )
        self.progresses[task_id] = task
        logger.info(f"创建任务 {task_id}, 总计 {total_items} 项")
        return task_id

    def get_task(self, task_id: str) -> Optional[TaskProgress]:
        """获取任务状态"""
        return self.progresses.get(task_id)

    def start_task(self, task_id: str) -> None:
        """开始任务"""
        if task_id in self.progresses:
            self.progresses[task_id].status = "running"
            self.progresses[task_id].start_time = time.time()
            logger.info(f"任务 {task_id} 开始执行")

    def update_progress(
        self,
        task_id: str,
        result: DetectionResultItem,
        message: str = "",
    ) -> None:
        """更新当前处理的项"""
        if task_id not in self.progresses:
            return

        task = self.progresses[task_id]
        task.set_result(result)
        if message:
            task.message = message

        if task.completed_items == task.total_items:
            task.status = "completed"
            task.end_time = time.time()

        self._notify_progress(task)

    def fail_task(self, task_id: str, message: str) -> None:
        """标记任务失败"""
        if task_id not in self.progresses:
            return

        task = self.progresses[task_id]
        task.status = "failed"
        task.message = message
        task.end_time = time.time()
        logger.error(f"任务 {task_id} 失败: {message}")

    def _notify_progress(self, task: TaskProgress) -> None:
        """通知进度更新（WebSocket + 回调）"""
        if task.task_id not in self.progresses:
            return

        msg = {
            "type": "progress_update",
            "data": task.to_dict(),
        }

        # WebSocket 广播
        try:
            import asyncio

            asyncio.create_task(connection_manager.broadcast_by_task(msg, task.task_id))
        except Exception as e:
            logger.error(f"WebSocket 广播失败: {e}")

    def cleanup_task(self, task_id: str) -> None:
        """清理完成的任务数据"""
        if task_id in self.progresses:
            del self.progresses[task_id]
        logger.debug(f"清理任务 {task_id}")


# 全局进度追踪器实例
progress_tracker = ProgressTracker()
