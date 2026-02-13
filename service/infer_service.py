import logging
import time
from typing import Tuple

import numpy as np

from inferencer.inferencer_factory import FaceAntiSpoofingInferencer
from inferencer.base_inferencer import (
    SingleBaseInferencer,
    FusionBaseInferencer,
    ModelType,
)
from util.result_parser import parse_fusion_prediction, parse_single_prediction

logger = logging.getLogger(__name__)


class InferService:
    @staticmethod
    def create(
        single_model_path: str,
        fusion_model_path: str,
    ):
        from util.config import settings

        logger.info(f"Loading single model from: {single_model_path}")
        logger.info(f"Loading fusion model from: {fusion_model_path}")
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

    def predict_single(self, image: np.ndarray) -> np.ndarray:
        """执行单模态推理"""
        return self.single_model_inferencer.predict(image)

    def predict_fusion(self, rgb_image: np.ndarray, ir_image: np.ndarray) -> np.ndarray:
        """执行融合模态推理"""
        return self.fusion_model_inferencer.predict(rgb_image, ir_image)

    def detect_single_batch(
        self,
        images: list[np.ndarray],
    ) -> list[Tuple[dict, int]]:
        """
        批量单模态检测

        Args:
            images: 解码后的图像数组列表

        Returns:
            每个图像的检测结果列表，每项为 (解析结果字典, 处理时间毫秒)
        """
        results = []

        for image in images:
            start_time = time.time()

            logits = self.predict_single(image)
            parsed = parse_single_prediction(logits)

            processing_time = int((time.time() - start_time) * 1000)
            results.append((parsed, processing_time))

        return results

    def detect_fusion_batch(
        self,
        image_pairs: list[Tuple[np.ndarray, np.ndarray]],
    ) -> list[Tuple[dict, int]]:
        """
        批量融合模态检测

        Args:
            image_pairs: 解码后的图像对列表，每项为 (rgb_image, ir_image)

        Returns:
            每个图像对的检测结果列表，每项为 (解析结果字典, 处理时间毫秒)
        """
        results = []

        for rgb_image, ir_image in image_pairs:
            start_time = time.time()

            logits = self.predict_fusion(rgb_image, ir_image)
            parsed = parse_fusion_prediction(logits)

            processing_time = int((time.time() - start_time) * 1000)
            results.append((parsed, processing_time))

        return results
