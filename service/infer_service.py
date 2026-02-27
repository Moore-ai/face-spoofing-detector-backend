import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Callable, Awaitable, Tuple, Optional

import numpy as np

from inferencer.inferencer_factory import FaceAntiSpoofingInferencer
from inferencer.base_inferencer import (
    SingleBaseInferencer,
    FusionBaseInferencer,
    ModelType,
)
from inferencer.debug_inferencer import DebugSingleInferencer, DebugFusionInferencer
from util.result_parser import parse_fusion_prediction, parse_single_prediction
from schemas.detection import DetectionResultItem
from util.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """重试配置"""

    enabled: bool = True
    max_attempts: int = 3
    delay_seconds: float = 1.0
    exponential_backoff: bool = True
    max_delay_seconds: float = 10.0


@dataclass
class InferenceResult:
    """单次推理结果包装"""

    success: bool
    result: Optional[DetectionResultItem] = None
    error: Optional[str] = None
    retry_count: int = 0


class InferService:
    @staticmethod
    def create(
        single_model_path: str,
        fusion_model_path: str,
    ):
        logger.info(f"Loading single model from: {single_model_path}")
        logger.info(f"Loading fusion model from: {fusion_model_path}")

        # 检查是否启用调试模式
        if settings.DEBUG_MODE:
            logger.info("=== 调试模式已启用 - 使用虚拟推理器 ===")
            single_inferencer = DebugSingleInferencer(
                input_size=settings.INPUT_SIZE,
                delay_per_image=settings.DEBUG_DELAY_PER_IMAGE,
                failure_rate=settings.DEBUG_FAILURE_RATE,
            )
            fusion_inferencer = DebugFusionInferencer(
                input_size=settings.INPUT_SIZE,
                delay_per_pair=settings.DEBUG_DELAY_PER_PAIR,
                failure_rate=settings.DEBUG_FAILURE_RATE,
            )
            logger.info(f"Debug inferencers created successfully (failure_rate={settings.DEBUG_FAILURE_RATE})")
        else:
            # 生产模式 - 加载真实模型
            try:
                single_inferencer = FaceAntiSpoofingInferencer.create(
                    model_path=single_model_path,
                    modal_type=ModelType.SINGLE,
                    input_size=settings.INPUT_SIZE,
                    device=settings.DEVICE,
                )
                logger.info("Single model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load single model: {e}", exc_info=True)
                raise

            try:
                fusion_inferencer = FaceAntiSpoofingInferencer.create(
                    model_path=fusion_model_path,
                    modal_type=ModelType.FUSION,
                    input_size=settings.INPUT_SIZE,
                    device=settings.DEVICE,
                )
                logger.info("Fusion model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load fusion model: {e}", exc_info=True)
                raise

        return InferService(
            _single_model_inferencer=single_inferencer,
            _fusion_model_inferencer=fusion_inferencer,
        )

    def __init__(
        self,
        _single_model_inferencer: SingleBaseInferencer,
        _fusion_model_inferencer: FusionBaseInferencer,
    ) -> None:
        self.single_model_inferencer = _single_model_inferencer
        self.fusion_model_inferencer = _fusion_model_inferencer

    @staticmethod
    def _get_retry_config() -> RetryConfig:
        """从配置获取重试参数"""
        return RetryConfig(
            enabled=settings.RETRY_ENABLED,
            max_attempts=settings.RETRY_MAX_ATTEMPTS,
            delay_seconds=settings.RETRY_DELAY_SECONDS,
            exponential_backoff=settings.RETRY_EXPONENTIAL_BACKOFF,
            max_delay_seconds=settings.RETRY_MAX_DELAY_SECONDS,
        )

    @staticmethod
    def _calculate_delay(attempt: int, config: RetryConfig) -> float:
        """计算重试延迟（支持指数退避）"""
        if config.exponential_backoff:
            delay = config.delay_seconds * (2**attempt)
            return min(delay, config.max_delay_seconds)
        return config.delay_seconds

    async def _infer_with_retry(
        self,
        infer_func: Callable[[], np.ndarray],
        parse_func: Callable[[np.ndarray, float], DetectionResultItem],
        image_index: int,
        mode: str,
        config: RetryConfig,
    ) -> InferenceResult:
        """带重试机制的推理执行"""
        last_error: Optional[Exception] = None

        for attempt in range(config.max_attempts):
            start_time = time.time()
            try:
                logits = infer_func()
                result = parse_func(logits, start_time)
                result.image_index = image_index
                result.retry_count = attempt
                result.success = True

                if attempt > 0:
                    logger.info(f"图片 {image_index} 在第 {attempt + 1} 次尝试成功")

                return InferenceResult(success=True, result=result, retry_count=attempt)

            except Exception as e:
                last_error = e
                logger.warning(
                    f"图片 {image_index} 推理失败 (尝试 {attempt + 1}/{config.max_attempts}): {e}"
                )

                if attempt < config.max_attempts - 1 and config.enabled:
                    delay = self._calculate_delay(attempt, config)
                    logger.info(f"将在 {delay:.1f}s 后重试图片 {image_index}...")
                    await asyncio.sleep(delay)

        # 所有尝试都失败
        error_msg = str(last_error) if last_error else "未知错误"
        logger.error(f"图片 {image_index} 在 {config.max_attempts} 次尝试后仍失败: {error_msg}")

        error_result = DetectionResultItem(
            mode=mode,
            result="error",
            confidence=0.0,
            probabilities=np.array([0.0, 0.0]),
            processing_time=0,
            image_index=image_index,
            error=error_msg,
            retry_count=config.max_attempts,
            success=False,
        )

        return InferenceResult(success=False, result=error_result, error=error_msg)

    def predict_single(self, image: np.ndarray) -> np.ndarray:
        """执行单模态推理"""
        return self.single_model_inferencer.predict(image)

    def predict_fusion(self, rgb_image: np.ndarray, ir_image: np.ndarray) -> np.ndarray:
        """执行融合模态推理"""
        return self.fusion_model_inferencer.predict(rgb_image, ir_image)

    async def detect_single_batch(
        self,
        images: list[np.ndarray],
        progress_callback: Callable[[int, int, DetectionResultItem], Awaitable[None]] | None = None,
    ) -> list[tuple[DetectionResultItem, int]]:
        """
        批量单模态检测（带重试机制）

        Args:
            images: 解码后的图像数组列表
            progress_callback: 异步进度回调函数，接收 (current_index, total_count, result) 参数

        Returns:
            每个图像的检测结果列表，每项为 (解析结果字典，处理时间毫秒)
        """
        config = self._get_retry_config()
        results = []

        for i, image in enumerate(images):
            infer_result = await self._infer_with_retry(
                infer_func=lambda img=image: self.predict_single(img),
                parse_func=lambda logits, start: parse_single_prediction(logits, start),
                image_index=i,
                mode="single",
                config=config,
            )

            result = infer_result.result
            results.append((result, result.processing_time)) # type: ignore

            # 调用进度回调
            if progress_callback:
                await progress_callback(i + 1, len(images), result) # type: ignore

        return results

    async def detect_fusion_batch(
        self,
        image_pairs: list[Tuple[np.ndarray, np.ndarray]],
        progress_callback: Callable[[int, int, DetectionResultItem], Awaitable[None]] | None = None,
    ) -> list[tuple[DetectionResultItem, int]]:
        """
        批量融合模态检测（带重试机制）

        Args:
            image_pairs: 解码后的图像对列表，每项为 (rgb_image, ir_image)
            progress_callback: 异步进度回调函数，接收 (current_index, total_count, result) 参数

        Returns:
            每个图像对的检测结果列表，每项为 (解析结果字典，处理时间毫秒)
        """
        config = self._get_retry_config()
        results = []

        for i, (rgb_image, ir_image) in enumerate(image_pairs):
            infer_result = await self._infer_with_retry(
                infer_func=lambda rgb=rgb_image, ir=ir_image: self.predict_fusion(rgb, ir),
                parse_func=lambda logits, start: parse_fusion_prediction(logits, start),
                image_index=i,
                mode="fusion",
                config=config,
            )

            result = infer_result.result
            results.append((result, result.processing_time)) # type: ignore

            # 调用进度回调
            if progress_callback:
                await progress_callback(i + 1, len(image_pairs), result) # type: ignore

        return results
