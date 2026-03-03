"""
任务调度器 - 基于优先级的任务队列

实现真正的优先级任务调度，高优先级任务优先执行
"""

import asyncio
import heapq
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from enum import Enum

from service.progress_service import progress_tracker

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """任务队列中的状态"""
    QUEUED = "queued"  # 等待调度
    RUNNING = "running"  # 正在执行
    CANCELLED = "cancelled"  # 已取消
    COMPLETED = "completed"  # 已完成


@dataclass(order=True)
class PrioritizedTask:
    """带优先级的任务包装"""

    # 优先级：负数是为了使用最小堆实现最大优先级
    priority: int  # 存储为负数，这样优先级越高（数值越大）越先执行
    timestamp: float  # 任务创建时间戳，用于相同优先级时的排序
    task_id: str = field(compare=False)
    task_type: str = field(compare=False)  # "single" 或 "fusion"
    task_data: Any = field(compare=False)  # 任务数据
    callback: Optional[Callable] = field(compare=False, default=None)  # 完成回调
    status: TaskStatus = field(compare=False, default=TaskStatus.QUEUED)

    def __post_init__(self):
        # 将优先级转换为负数用于最小堆
        if isinstance(self.priority, int):
            object.__setattr__(self, 'priority', -self.priority)


class TaskQueue:
    """优先级任务队列"""

    def __init__(self):
        # 使用 heapq 实现优先级队列（最小堆）
        self._queue: list[PrioritizedTask] = []
        self._task_map: dict[str, PrioritizedTask] = {}
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)  # 用于等待新任务

    async def enqueue(
        self,
        task_id: str,
        task_type: str,
        task_data: Any,
        priority: int = 0,
        callback: Optional[Callable] = None,
    ) -> bool:
        """
        将任务加入队列

        Args:
            task_id: 任务 ID
            task_type: 任务类型 ("single" 或 "fusion")
            task_data: 任务数据
            priority: 优先级 (0-100)，值越大优先级越高
            callback: 完成回调

        Returns:
            bool: 是否成功加入队列
        """
        async with self._lock:
            # 检查任务是否已存在
            if task_id in self._task_map:
                logger.warning(f"任务 {task_id} 已在队列中")
                return False

            # 创建优先级任务
            task = PrioritizedTask(
                priority=priority,  # 会自动转换为负数
                timestamp=time.time(),
                task_id=task_id,
                task_type=task_type,
                task_data=task_data,
                callback=callback,
            )

            heapq.heappush(self._queue, task)
            self._task_map[task_id] = task

            logger.info(
                f"任务加入队列: task_id={task_id}, "
                f"type={task_type}, priority={priority}, "
                f"queue_size={len(self._queue)}"
            )

            # 通知等待的调度器
            self._condition.notify()

            return True

    async def dequeue(self, timeout: Optional[float] = None) -> Optional[PrioritizedTask]:
        """
        从队列中取出最高优先级的任务

        Args:
            timeout: 等待超时时间（秒），None 表示无限等待

        Returns:
            PrioritizedTask | None: 最高优先级任务，队列为空且超时时返回 None
        """
        async with self._condition:
            # 如果队列为空，等待新任务
            if not self._queue:
                if timeout is None:
                    await self._condition.wait()
                else:
                    try:
                        await asyncio.wait_for(self._condition.wait(), timeout)
                    except asyncio.TimeoutError:
                        return None

            # 从队列中取出任务
            if self._queue:
                task = heapq.heappop(self._queue)
                del self._task_map[task.task_id]
                task.status = TaskStatus.RUNNING
                logger.debug(f"任务出队: task_id={task.task_id}, priority={-task.priority}")
                return task

            return None

    async def cancel(self, task_id: str) -> bool:
        """取消队列中的任务"""
        async with self._lock:
            if task_id not in self._task_map:
                return False

            task = self._task_map[task_id]
            task.status = TaskStatus.CANCELLED

            # 从队列中移除任务
            self._queue.remove(task)
            heapq.heapify(self._queue)  # 重新堆化
            del self._task_map[task_id]

            logger.info(f"任务已从队列中取消: task_id={task_id}")
            return True

    async def get_queue_size(self) -> int:
        """获取队列中的任务数量"""
        async with self._lock:
            return len(self._queue)

    async def get_all_tasks(self) -> list:
        """获取所有任务（按优先级排序）"""
        async with self._lock:
            # 返回副本，避免外部修改
            return sorted(self._queue, key=lambda t: (-t.priority, t.timestamp))

    async def get_task(self, task_id: str) -> Optional[PrioritizedTask]:
        """获取指定任务"""
        async with self._lock:
            return self._task_map.get(task_id)

    async def clear(self) -> int:
        """清空队列，返回被清空的任务数量"""
        async with self._lock:
            count = len(self._queue)
            self._queue.clear()
            self._task_map.clear()
            logger.info(f"队列已清空，共 {count} 个任务")
            return count


class TaskScheduler:
    """任务调度器"""

    def __init__(
        self,
        max_workers: int = 1,
        enable_batch_processing: bool = True,
    ):
        """
        初始化任务调度器

        Args:
            max_workers: 最大工作线程数（并发执行的任务数）
            enable_batch_processing: 是否启用批量处理优化
        """
        self.task_queue = TaskQueue()
        self.max_workers = max_workers
        self.enable_batch_processing = enable_batch_processing
        self._workers: list[asyncio.Task] = []
        self._is_running: bool = False
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """启动调度器"""
        async with self._lock:
            if self._is_running:
                logger.warning("调度器已在运行中")
                return

            self._is_running = True

            # 启动工作线程
            for i in range(self.max_workers):
                worker = asyncio.create_task(self._worker_loop(i))
                self._workers.append(worker)

            logger.info(
                f"任务调度器已启动，工作线程数: {self.max_workers}, "
                f"批量处理: {'启用' if self.enable_batch_processing else '禁用'}"
            )

    async def stop(self) -> None:
        """停止调度器"""
        async with self._lock:
            if not self._is_running:
                return

            self._is_running = False

            # 取消所有工作线程
            for worker in self._workers:
                worker.cancel()

            # 等待工作线程结束
            await asyncio.gather(*self._workers, return_exceptions=True)
            self._workers.clear()

            # 清空队列
            await self.task_queue.clear()

            logger.info("任务调度器已停止")

    async def submit_task(
        self,
        task_id: str,
        task_type: str,
        task_data: Any,
        priority: int = 0,
        callback: Optional[Callable] = None,
    ) -> bool:
        """
        提交任务到调度器

        Args:
            task_id: 任务 ID
            task_type: 任务类型 ("single" 或 "fusion")
            task_data: 任务数据
            priority: 优先级 (0-100)
            callback: 完成回调

        Returns:
            bool: 是否成功提交
        """
        if not self._is_running:
            logger.error("调度器未运行，无法提交任务")
            return False

        return await self.task_queue.enqueue(
            task_id=task_id,
            task_type=task_type,
            task_data=task_data,
            priority=priority,
            callback=callback,
        )

    async def cancel_task(self, task_id: str) -> bool:
        """
        取消任务

        如果任务已在队列中，直接取消；
        如果任务正在执行，设置取消标志（需要任务自身检查）
        """
        # 先尝试从队列中取消
        if await self.task_queue.cancel(task_id):
            return True

        # 如果不在队列中，可能是正在执行，设置取消标志
        return await progress_tracker.cancel_task(task_id, "用户取消")

    async def get_queue_status(self) -> dict:
        """获取队列状态"""
        queue_size = await self.task_queue.get_queue_size()

        return {
            "is_running": self._is_running,
            "queue_size": queue_size,
            "max_workers": self.max_workers,
            "active_workers": len([w for w in self._workers if not w.done()]),
        }

    async def _worker_loop(self, worker_id: int) -> None:
        """
        工作线程循环

        从队列中取出任务并执行
        """
        logger.info(f"工作线程 {worker_id} 已启动")

        try:
            while self._is_running:
                # 从队列中获取任务（等待最多 1 秒）
                task = await self.task_queue.dequeue(timeout=1.0)

                if task is None:
                    continue

                # 执行任务
                await self._execute_task(worker_id, task)

        except asyncio.CancelledError:
            logger.info(f"工作线程 {worker_id} 已取消")
        except Exception as e:
            logger.error(f"工作线程 {worker_id} 异常: {e}", exc_info=True)

    async def _execute_task(self, worker_id: int, task: PrioritizedTask) -> None:
        """
        执行任务

        这里需要与现有的后台任务执行逻辑集成
        """
        logger.info(
            f"工作线程 {worker_id} 正在执行任务: "
            f"task_id={task.task_id}, type={task.task_type}, "
            f"priority={-task.priority}"
        )

        try:
            # 执行回调（如果有）
            if task.callback:
                await task.callback(task.task_data)

            # 更新任务状态为已完成
            task.status = TaskStatus.COMPLETED

            logger.info(f"工作线程 {worker_id} 任务完成: task_id={task.task_id}")

        except Exception as e:
            logger.error(f"工作线程 {worker_id} 任务执行失败: {e}", exc_info=True)
            task.status = TaskStatus.CANCELLED


# 全局调度器实例
task_scheduler = TaskScheduler(
    max_workers=1,  # 默认单线程，确保任务按优先级顺序执行
    enable_batch_processing=True,
)
