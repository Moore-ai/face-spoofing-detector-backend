import logging
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import os

logger = logging.getLogger(__name__)


class ModelFormat(Enum):
    """模型格式枚举"""

    PYTORCH = "pth"
    ONNX = "onnx"
    RKNN = "rknn"


class ModelType(Enum):
    """模态类型枚举"""

    SINGLE = "single"  # 单模态：RGB 或 IR
    FUSION = "fusion"  # 融合模态：RGB + IR


class BaseInferencer(ABC):
    """
    推理器抽象基类
    所有具体推理器必须继承此类并实现抽象方法
    """

    def __init__(self, model_path: str, input_size: int = 112, num_classes: int = 2):
        """
        初始化推理器

        Args:
            model_path: 模型文件路径
            input_size: 输入图像尺寸（默认112）
            num_classes: 分类类别数（默认2：活体/非活体）
        """
        self.model_path = model_path
        self.input_size = input_size
        self.num_classes = num_classes
        logger.info(f"Initializing inferencer with model: {model_path}")

        # 验证模型文件存在
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        self.model_format = self._detect_model_format()
        logger.info(f"Detected model format: {self.model_format.value}")
        self.model = None

        # 加载模型
        try:
            self._load_model()
            logger.info(f"Model loaded successfully: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}", exc_info=True)
            raise

    def _detect_model_format(self) -> ModelFormat:
        """根据文件后缀检测模型格式"""
        ext = os.path.splitext(self.model_path)[1].lower().replace(".", "")
        try:
            return ModelFormat(ext)
        except ValueError:
            logger.error(f"Unsupported model format: {ext}")
            raise ValueError(f"不支持的模型格式: {ext}，仅支持 pth/onnx/rknn")

    @abstractmethod
    def _load_model(self):
        """加载模型（子类必须实现）"""
        pass

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理（子类必须实现）

        Args:
            image: 输入图像，BGR格式 (H, W, C)

        Returns:
            预处理后的数组，准备输入模型
        """
        pass

    @abstractmethod
    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """
        执行推理（子类必须实现）

        Args:
            input_data: 预处理后的输入数据

        Returns:
            模型原始输出 (logits)
        """
        pass

    def predict(self, *images: np.ndarray) -> np.ndarray:
        """
        端到端预测（便捷方法）

        Args:
            *images: 输入图像（单模态1张，融合模态2张）

        Returns:
            模型原始输出 (logits)，shape: (1, num_classes)
        """
        preprocessed = self.preprocess(*images)
        return self.infer(preprocessed)


SingleBaseInferencer = BaseInferencer


class FusionBaseInferencer(BaseInferencer):
    """
    融合模态推理器抽象基类（RGB + IR）
    """

    @abstractmethod
    def preprocess(self, rgb_image: np.ndarray, ir_image: np.ndarray) -> np.ndarray:
        """
        融合模态图像预处理（子类必须实现）

        Args:
            rgb_image: RGB图像，BGR格式 (H, W, 3)
            ir_image: IR图像，BGR格式 (H, W, 3)

        Returns:
            预处理后的融合数据，准备输入模型
        """
        pass
