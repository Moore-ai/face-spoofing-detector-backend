"""
图片自动存储服务
在推理完成后根据策略自动保存图片
"""

import logging
from typing import Optional, List, Dict, Any

from util.config import settings
from util.image_compressor import ImageCompressionConfig
from util.image_storage_policy import (
    StoragePolicy,
    StoragePolicyFactory,
    ImageStorageManager,
    ImageStorageContext,
)
from util.storage import StorageManager

logger = logging.getLogger(__name__)


class ImageAutoSaveService:
    """
    图片自动存储服务

    根据配置的存储策略，在推理完成后自动决定是否保存图片
    支持图片压缩以减少存储空间
    """

    def __init__(self):
        self._initialized = False
        self._storage_policy: Optional[StoragePolicy] = None
        self._storage_manager: Optional[ImageStorageManager] = None
        self._compression_config: Optional[ImageCompressionConfig] = None

    def initialize(
        self,
        storage_backend: Optional[StorageManager] = None,
        policy: Optional[StoragePolicy] = None,
        compression_config: Optional[ImageCompressionConfig] = None,
    ):
        """
        初始化自动存储服务

        Args:
            storage_backend: 存储后端（util.storage.StorageManager）
            policy: 存储策略，默认从配置读取
            compression_config: 压缩配置，默认从配置读取
        """
        if self._initialized:
            logger.warning("ImageAutoSaveService already initialized")
            return

        logger.info("Initializing image auto-save service...")

        # 创建存储策略
        if policy is None:
            policy = StoragePolicyFactory.create_from_config(
                strategy_name=settings.STORAGE_SAVE_STRATEGY,
                save_error_rate=settings.STORAGE_SAVE_ERROR_RATE,
                save_fake_rate=settings.STORAGE_SAVE_FAKE_RATE,
                save_real_rate=settings.STORAGE_SAVE_REAL_RATE,
                save_low_confidence_threshold=settings.STORAGE_SAVE_LOW_CONFIDENCE_THRESHOLD,
                max_storage_per_task=settings.STORAGE_MAX_PER_TASK,
            )
        self._storage_policy = policy

        # 创建压缩配置
        if compression_config is None:
            compression_config = ImageCompressionConfig(
                enabled=settings.IMAGE_COMPRESS_ENABLED,
                compressor_type=settings.IMAGE_COMPRESS_TYPE,
                quality=settings.IMAGE_COMPRESS_QUALITY,
                max_width=settings.IMAGE_COMPRESS_MAX_WIDTH,
                max_height=settings.IMAGE_COMPRESS_MAX_HEIGHT,
            )
        self._compression_config = compression_config

        # 创建压缩器
        compressor = None
        if compression_config.enabled:
            compressor = compression_config.get_compressor()
            if compressor:
                logger.info(f"Image compression enabled: type={compression_config.compressor_type}, quality={compression_config.quality}")
            else:
                logger.warning("Image compression enabled but no compressor available")

        # 创建存储管理器
        self._storage_manager = ImageStorageManager(
            policy=policy,
            storage_backend=storage_backend,
            compressor=compressor,
        )

        self._initialized = True
        logger.info(
            f"Image auto-save service initialized: "
            f"strategy={policy.strategy.value}, "
            f"compression={compression_config.enabled}"
        )

    async def save_images_from_results(
        self,
        task_id: str,
        client_id: Optional[str],
        api_key_hash: Optional[str],
        results: List[Dict[str, Any]],
        original_images: Optional[List[str]] = None,  # Base64 编码的原始图片
    ):
        """
        从推理结果自动保存图片

        Args:
            task_id: 任务 ID
            client_id: 客户端 ID
            api_key_hash: API Key 哈希
            results: 推理结果列表
            original_images: 原始图片列表（Base64 编码）
        """
        if not self._initialized:
            return

        if not settings.STORAGE_AUTO_SAVE:
            return

        try:
            saved_count = 0
            skip_count = 0

            for i, result in enumerate(results):
                # 获取原始图片（如果有）
                image_data = None
                if original_images and i < len(original_images):
                    image_data = original_images[i]

                if not image_data:
                    logger.debug(f"No image data for index {i}, skipping")
                    skip_count += 1
                    continue

                # 创建存储上下文
                context = ImageStorageContext(
                    task_id=task_id,
                    client_id=client_id,
                    api_key_hash=api_key_hash,
                    image_data=image_data,
                    image_type="original",
                    modality=result.get("modality", "rgb"),
                    image_index=result.get("image_index", i),
                    result=result.get("result", "error"),
                    confidence=result.get("confidence", 0.0),
                    is_error=result.get("result") == "error" or result.get("success", True) is False,
                    error_message=result.get("error"),
                    metadata={
                        "processing_time": result.get("processing_time", 0),
                        "retry_count": result.get("retry_count", 0),
                    },
                )

                # 根据策略决定是否保存
                assert self._storage_manager
                result_dict = await self._storage_manager.store_if_needed(context)
                if result_dict:
                    saved_count += 1
                else:
                    skip_count += 1

            logger.info(
                f"Auto-save completed for task {task_id}: "
                f"saved={saved_count}, skipped={skip_count}"
            )

        except Exception as e:
            logger.error(f"Failed to auto-save images for task {task_id}: {e}", exc_info=True)

    def get_saved_count(self, task_id: str) -> int:
        """获取任务已保存的图片数量"""
        if not self._initialized:
            return 0
        assert self._storage_manager
        return self._storage_manager.get_saved_count(task_id)


# 全局自动存储服务实例
image_auto_save_service = ImageAutoSaveService()
