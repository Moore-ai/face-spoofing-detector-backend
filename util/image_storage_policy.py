"""
图片存储策略模块
服务端控制存储策略，而非客户端请求
"""

import logging
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from util.image_compressor import ImageCompressor
from util.storage import StorageManager

logger = logging.getLogger(__name__)


class StoreStrategy(Enum):
    """存储策略类型"""
    NEVER = "never"           # 从不存储
    ALWAYS = "always"         # 总是存储
    ERROR_ONLY = "error_only"  # 只存储错误结果
    SMART = "smart"           # 智能策略（按置信度采样）


@dataclass
class StoragePolicy:
    """
    存储策略配置

    服务端控制何时保存图片，基于检测结果而非客户端请求
    """
    strategy: StoreStrategy = StoreStrategy.ERROR_ONLY

    # 智能策略参数（当 strategy=SMART 时有效）
    save_error_rate: float = 1.0      # 错误结果保存率（默认 100%）
    save_fake_rate: float = 0.1       # 伪造样本保存率（默认 10%）
    save_real_rate: float = 0.01      # 真实样本保存率（默认 1%）
    save_low_confidence_threshold: float = 0.6  # 低置信度阈值，低于此值的样本优先保存

    # 过滤参数
    min_processing_time_ms: int = 0   # 最小处理时间（用于过滤异常）
    max_storage_per_task: int = 10    # 每个任务最多保存图片数

    def should_save(
        self,
        result: str,
        confidence: float,
        is_error: bool = False,
    ) -> bool:
        """
        判断是否应该保存当前结果

        Args:
            result: 检测结果 ("real", "fake", "error")
            confidence: 置信度 (0-1)
            is_error: 是否推理错误

        Returns:
            是否保存
        """
        import random

        if self.strategy == StoreStrategy.NEVER:
            return False

        if self.strategy == StoreStrategy.ALWAYS:
            return True

        if self.strategy == StoreStrategy.ERROR_ONLY:
            return is_error or result == "error"

        if self.strategy == StoreStrategy.SMART:
            # 错误结果
            if is_error or result == "error":
                return random.random() < self.save_error_rate

            # 低置信度样本优先保存（用于模型优化）
            if confidence < self.save_low_confidence_threshold:
                return True

            # 伪造样本
            if result == "fake":
                return random.random() < self.save_fake_rate

            # 真实样本
            if result == "real":
                return random.random() < self.save_real_rate

        return False


@dataclass
class ImageStorageContext:
    """
    图片存储上下文

    封装存储决策所需的所有信息
    """
    task_id: str
    client_id: Optional[str]
    api_key_hash: Optional[str]

    # 图片信息
    image_data: str  # Base64 编码
    image_type: str  # "original" 或 "processed"
    modality: str    # "rgb", "ir", "fusion"
    image_index: int  # 批次中的索引

    # 检测结果
    result: str       # "real", "fake", "error"
    confidence: float
    is_error: bool = False
    error_message: Optional[str] = None

    # 元数据
    metadata: dict = field(default_factory=dict)


class ImageStorageManager:
    """
    图片存储管理器

    基于策略决定是否保存图片，并调用存储后端执行保存
    """

    def __init__(
        self,
        policy: StoragePolicy,
        storage_backend: Optional[StorageManager] = None,
        compressor: Optional[ImageCompressor] = None,
    ):
        """
        初始化

        Args:
            policy: 存储策略
            storage_backend: 存储后端（util.storage.StorageManager）
            compressor: 图片压缩器
        """
        self.policy: StoragePolicy = policy
        self.storage_backend: Optional[StorageManager] = storage_backend
        self.compressor: Optional[ImageCompressor] = compressor
        self._saved_count_per_task: dict = {}  # 跟踪每个任务的保存数量

    def reset_task_counts(self):
        """重置任务计数（定期调用）"""
        self._saved_count_per_task.clear()

    def should_store(self, context: ImageStorageContext) -> bool:
        """
        判断是否应该存储图片

        Args:
            context: 存储上下文

        Returns:
            是否存储
        """
        # 检查任务级别的数量限制
        current_count = self._saved_count_per_task.get(context.task_id, 0)
        if current_count >= self.policy.max_storage_per_task:
            logger.debug(
                f"Task {context.task_id} reached max storage limit "
                f"({self.policy.max_storage_per_task})"
            )
            return False

        # 应用存储策略
        should_save = self.policy.should_save(
            result=context.result,
            confidence=context.confidence,
            is_error=context.is_error,
        )

        if should_save:
            logger.debug(
                f"Will store image: task={context.task_id}, "
                f"result={context.result}, confidence={context.confidence:.2f}"
            )
        else:
            logger.debug(
                f"Skip storing image: task={context.task_id}, "
                f"result={context.result}, confidence={context.confidence:.2f}"
            )

        return should_save

    async def store_if_needed(
        self,
        context: ImageStorageContext,
    ) -> Optional[dict]:
        """
        根据策略决定是否存储图片

        Args:
            context: 存储上下文

        Returns:
            存储结果或 None（未存储）
        """
        if not self.should_store(context):
            return None

        if not self.storage_backend:
            logger.warning("Storage backend not initialized")
            return None

        try:
            # 压缩图片（如果配置了压缩器）
            image_data = context.image_data
            if self.compressor:
                try:
                    image_data = self.compressor.compress_from_base64(
                        context.image_data,
                        quality=75,  # 默认质量
                    )
                except Exception as e:
                    logger.warning(f"Compression failed, using original: {e}")

            # 保存图片
            result = self.storage_backend.save_image(
                task_id=context.task_id,
                image_data=image_data,
                image_type=context.image_type,
                modality=context.modality,
                metadata={
                    "result": context.result,
                    "confidence": context.confidence,
                    "is_error": context.is_error,
                    "error_message": context.error_message,
                    "image_index": context.image_index,
                    "client_id": context.client_id,
                    "api_key_hash": context.api_key_hash,
                    **context.metadata,
                },
            )

            # 更新计数
            self._saved_count_per_task[context.task_id] = \
                self._saved_count_per_task.get(context.task_id, 0) + 1

            logger.info(
                f"Stored image: task={context.task_id}, "
                f"image_id={result.get('image_id')}, "
                f"result={context.result}"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to store image: {e}", exc_info=True)
            return None

    def get_saved_count(self, task_id: str) -> int:
        """获取任务已保存的图片数量"""
        return self._saved_count_per_task.get(task_id, 0)


class StoragePolicyFactory:
    """存储策略工厂"""

    @staticmethod
    def create_from_config(
        strategy_name: str,
        **kwargs,
    ) -> StoragePolicy:
        """
        从配置创建存储策略

        Args:
            strategy_name: 策略名称 ("never", "always", "error_only", "smart")
            **kwargs: 额外参数

        Returns:
            存储策略实例
        """
        strategy_map = {
            "never": StoreStrategy.NEVER,
            "always": StoreStrategy.ALWAYS,
            "error_only": StoreStrategy.ERROR_ONLY,
            "smart": StoreStrategy.SMART,
        }

        strategy = strategy_map.get(strategy_name.lower(), StoreStrategy.ERROR_ONLY)

        return StoragePolicy(
            strategy=strategy,
            save_error_rate=kwargs.get("save_error_rate", 1.0),
            save_fake_rate=kwargs.get("save_fake_rate", 0.1),
            save_real_rate=kwargs.get("save_real_rate", 0.01),
            save_low_confidence_threshold=kwargs.get(
                "save_low_confidence_threshold", 0.6
            ),
            max_storage_per_task=kwargs.get("max_storage_per_task", 10),
        )
