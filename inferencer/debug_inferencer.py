"""
调试模式推理器
不加载真实模型，生成虚拟的推理结果
用于开发测试和前端联调
"""

import asyncio
import logging
import numpy as np
from typing import Tuple

from inferencer.base_inferencer import BaseInferencer, FusionBaseInferencer
from util.config import settings

logger = logging.getLogger(__name__)


class DebugSingleInferencer(BaseInferencer):
    """
    调试模式单模态推理器

    不加载真实模型，模拟推理过程：
    - 70% 概率返回 real（真实人脸）
    - 30% 概率返回 fake（攻击人脸）
    - 模拟处理延迟
    """

    def __init__(
        self,
        input_size: int = 112,
        num_classes: int = 2,
        delay_per_image: float = 0.5,
    ):
        """
        初始化调试推理器

        Args:
            input_size: 输入图像尺寸（默认 112）
            num_classes: 分类类别数（默认 2）
            delay_per_image: 每张图像的模拟处理延迟（秒）
        """
        # 不调用父类 __init__，避免加载模型
        self.input_size = input_size
        self.num_classes = num_classes
        self.delay_per_image = delay_per_image
        self.model_path = "debug_mode"
        self.model_format = None
        self.model = None

        logger.info(
            f"DebugSingleInferencer 已初始化 (input_size={input_size}, "
            f"delay={delay_per_image}s)"
        )

    def _load_model(self):
        """调试模式无需加载模型"""
        pass

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        调试模式预处理 - 仅验证输入

        Args:
            image: 输入图像

        Returns:
            原样返回输入图像
        """
        # 简单验证输入
        if image is None or image.size == 0:
            raise ValueError("调试模式：输入图像为空")
        return image

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """
        调试模式推理 - 生成虚拟结果

        Args:
            input_data: 输入数据（未使用）

        Returns:
            虚拟的 logits 输出
        """
        # 70% 概率为 real
        is_real = np.random.random() > 0.3

        # 生成合理的置信度
        if is_real:
            confidence = np.random.uniform(0.75, 0.99)
        else:
            confidence = np.random.uniform(0.55, 0.85)

        # 构造 logits（[real_logit, fake_logit]）
        # 使用 log 概率近似
        if is_real:
            logits = np.array([[confidence, 1 - confidence]])
        else:
            logits = np.array([[1 - confidence, confidence]])

        return logits

    async def predict_async(
        self,
        image_base64: str,
        modality: str = "rgb",
    ) -> Tuple[str, float, np.ndarray]:
        """
        异步预测（供推理服务调用）

        Args:
            image_base64: base64 编码的图像
            modality: 模态类型 ("rgb" 或 "ir")

        Returns:
            (result, confidence, probabilities)
            - result: "real" 或 "fake"
            - confidence: 置信度
            - probabilities: [real_prob, fake_prob]
        """
        # 模拟处理延迟
        await asyncio.sleep(self.delay_per_image)

        # 生成虚拟结果
        is_real = np.random.random() > 0.3

        if is_real:
            confidence = float(np.random.uniform(0.75, 0.99))
            result = "real"
        else:
            confidence = float(np.random.uniform(0.55, 0.85))
            result = "fake"

        probabilities = np.array([confidence, 1 - confidence])

        logger.debug(
            f"调试推理结果：{result} (confidence={confidence:.3f}, "
            f"modality={modality})"
        )

        return result, confidence, probabilities


class DebugFusionInferencer(FusionBaseInferencer):
    """
    调试模式融合模态推理器

    不加载真实模型，模拟 RGB+IR 融合推理过程：
    - 70% 概率返回 real
    - 30% 概率返回 fake
    - 模拟处理延迟（比单模态稍长）
    """

    def __init__(
        self,
        input_size: int = 112,
        num_classes: int = 2,
        delay_per_pair: float = 0.8,
    ):
        """
        初始化调试推理器

        Args:
            input_size: 输入图像尺寸（默认 112）
            num_classes: 分类类别数（默认 2）
            delay_per_pair: 每对图像的模拟处理延迟（秒）
        """
        # 不调用父类 __init__，避免加载模型
        self.input_size = input_size
        self.num_classes = num_classes
        self.delay_per_pair = delay_per_pair
        self.model_path = "debug_mode"
        self.model_format = None
        self.model = None

        logger.info(
            f"DebugFusionInferencer 已初始化 (input_size={input_size}, "
            f"delay={delay_per_pair}s)"
        )

    def _load_model(self):
        """调试模式无需加载模型"""
        pass

    def preprocess(
        self,
        rgb_image: np.ndarray,
        ir_image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        调试模式预处理 - 仅验证输入

        Args:
            rgb_image: RGB 图像
            ir_image: IR 图像

        Returns:
            原样返回输入图像
        """
        if rgb_image is None or rgb_image.size == 0:
            raise ValueError("调试模式：RGB 图像为空")
        if ir_image is None or ir_image.size == 0:
            raise ValueError("调试模式：IR 图像为空")
        return rgb_image, ir_image

    def infer(
        self,
        input_data: Tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """
        调试模式推理 - 生成虚拟结果

        Args:
            input_data: (RGB, IR) 图像对（未使用）

        Returns:
            虚拟的 logits 输出
        """
        # 70% 概率为 real
        is_real = np.random.random() > 0.3

        # 融合模态通常置信度更高
        if is_real:
            confidence = np.random.uniform(0.80, 0.99)
        else:
            confidence = np.random.uniform(0.60, 0.90)

        if is_real:
            logits = np.array([[confidence, 1 - confidence]])
        else:
            logits = np.array([[1 - confidence, confidence]])

        return logits

    async def predict_async(
        self,
        rgb_base64: str,
        ir_base64: str,
    ) -> Tuple[str, float, np.ndarray]:
        """
        异步预测（供推理服务调用）

        Args:
            rgb_base64: base64 编码的 RGB 图像
            ir_base64: base64 编码的 IR 图像

        Returns:
            (result, confidence, probabilities)
            - result: "real" 或 "fake"
            - confidence: 置信度
            - probabilities: [real_prob, fake_prob]
        """
        # 模拟处理延迟（融合模式稍长）
        await asyncio.sleep(self.delay_per_pair)

        # 生成虚拟结果
        is_real = np.random.random() > 0.3

        if is_real:
            confidence = float(np.random.uniform(0.80, 0.99))
            result = "real"
        else:
            confidence = float(np.random.uniform(0.60, 0.90))
            result = "fake"

        probabilities = np.array([confidence, 1 - confidence])

        logger.debug(
            f"调试融合推理结果：{result} (confidence={confidence:.3f})"
        )

        return result, confidence, probabilities
