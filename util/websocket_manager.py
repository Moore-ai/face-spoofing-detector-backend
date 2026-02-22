"""
WebSocket 连接管理工具
管理客户端连接和消息广播
"""

import asyncio
import logging
import uuid
from typing import Dict, Set

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """WebSocket 连接管理器

    架构规则：
    1. client_id 由后端生成
    2. 一个任务仅对应一个客户
    3. 一个客户可以对应多个任务

    线程安全：所有公共方法都使用 asyncio.Lock 保护
    """

    def __init__(self):
        # 存储活跃连接：{client_id: WebSocket}
        self.active_connections: Dict[str, WebSocket] = {}
        # 任务到客户的映射：{task_id: client_id}
        # 一个任务只能属于一个客户
        self.task_to_client: Dict[str, str] = {}
        # 客户到任务的映射：{client_id: Set[task_id]}
        # 一个客户可以有多个任务
        self.client_to_tasks: Dict[str, Set[str]] = {}
        # 异步锁保护共享数据
        self._lock = asyncio.Lock()

    def generate_client_id(self) -> str:
        """生成唯一的客户端 ID"""
        return str(uuid.uuid4())

    async def connect(self, websocket: WebSocket) -> str:
        """接受新的 WebSocket 连接，返回生成的 client_id"""
        await websocket.accept()
        client_id = self.generate_client_id()
        async with self._lock:
            self.active_connections[client_id] = websocket
            self.client_to_tasks[client_id] = set()
        logger.info(f"WebSocket 客户端已连接：{client_id}")
        return client_id

    async def disconnect(self, client_id: str) -> None:
        """断开 WebSocket 连接"""
        async with self._lock:
            if client_id in self.active_connections:
                del self.active_connections[client_id]

                # 清理该客户的所有任务映射
                if client_id in self.client_to_tasks:
                    tasks = self.client_to_tasks[client_id].copy()
                    for task_id in tasks:
                        if task_id in self.task_to_client:
                            del self.task_to_client[task_id]
                    del self.client_to_tasks[client_id]

        logger.info(f"WebSocket 客户端已断开：{client_id}")

    async def send_message(self, client_id: str, message: dict) -> bool:
        """向指定客户端发送消息"""
        async with self._lock:
            if client_id not in self.active_connections:
                return False
            websocket = self.active_connections[client_id]
        try:
            await websocket.send_json(message)
            return True
        except Exception as e:
            logger.error(f"发送消息到 {client_id} 失败：{e}")
            return False

    async def broadcast_to_client(self, client_id: str, message: dict) -> bool:
        """向指定客户发送消息"""
        return await self.send_message(client_id, message)

    async def broadcast_by_task(self, message: dict, task_id: str) -> bool:
        """根据任务 ID 广播消息给对应的客户

        一个任务只对应一个客户
        """
        async with self._lock:
            if task_id not in self.task_to_client:
                logger.warning(f"任务 {task_id} 没有关联的客户")
                return False

            client_id = self.task_to_client[task_id]

        return await self.send_message(client_id, message)

    async def register_task(self, client_id: str, task_id: str) -> bool:
        """为客户注册一个任务

        规则：
        - 如果任务已被其他客户注册，返回 False
        - 否则建立映射关系，返回 True
        """
        async with self._lock:
            # 检查任务是否已被其他客户注册
            if task_id in self.task_to_client:
                existing_client = self.task_to_client[task_id]
                if existing_client != client_id:
                    logger.warning(
                        f"任务 {task_id} 已被客户 {existing_client} 注册，"
                        f"无法为客户 {client_id} 注册"
                    )
                    return False
                return True  # 已注册，幂等

            # 建立映射
            self.task_to_client[task_id] = client_id
            if client_id in self.client_to_tasks:
                self.client_to_tasks[client_id].add(task_id)

        logger.info(f"任务 {task_id} 已注册到客户 {client_id}")
        return True

    async def unregister_task(self, task_id: str) -> None:
        """注销任务"""
        async with self._lock:
            if task_id in self.task_to_client:
                client_id = self.task_to_client[task_id]
                del self.task_to_client[task_id]

                if client_id in self.client_to_tasks:
                    self.client_to_tasks[client_id].discard(task_id)

        logger.info(f"任务 {task_id} 已注销")

    async def get_client_tasks(self, client_id: str) -> Set[str]:
        """获取客户的所有任务"""
        async with self._lock:
            return self.client_to_tasks.get(client_id, set()).copy()

    async def get_task_client(self, task_id: str) -> str | None:
        """获取任务对应的客户"""
        async with self._lock:
            return self.task_to_client.get(task_id)


# 全局连接管理器实例
connection_manager = ConnectionManager()
