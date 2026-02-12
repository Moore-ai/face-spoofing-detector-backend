from enum import Enum

from base_inferencer import BaseInferencer
from fusion_model_inferencer import FusionModalInferencer
from single_model_inferencer import SingleModalInferencer


class ModelType(Enum):
    """模型类型枚举"""
    SINGLE = "single"
    FUSION = "fusion"


class FaceAntiSpoofingInferencer:
    """
    人脸活体检测推理器工厂类
    根据模型类型自动选择对应的推理器
    """

    @staticmethod
    def create(
        model_path: str,
        modal_type: ModelType,
        input_size: int = 112,
        num_classes: int = 2,
        device: str = "cuda",
        rknn_target: str = "rv1126",
    ) -> BaseInferencer:
        """
        创建推理器实例

        Args:
            model_path: 模型文件路径
            modal_type: 模态类型（ModalType.SINGLE 或 ModalType.FUSION）
            input_size: 输入图像尺寸
            num_classes: 分类类别数
            device: PyTorch设备
            rknn_target: RKNN目标设备

        Returns:
            对应的推理器实例
        """
        if modal_type == ModelType.SINGLE:
            return SingleModalInferencer(
                model_path=model_path,
                input_size=input_size,
                num_classes=num_classes,
                device=device,
                rknn_target=rknn_target,
            )
        elif modal_type == ModelType.FUSION:
            return FusionModalInferencer(
                model_path=model_path,
                input_size=input_size,
                num_classes=num_classes,
                device=device,
                rknn_target=rknn_target,
            )
        else:
            raise ValueError(f"不支持的模态类型: {modal_type}")
