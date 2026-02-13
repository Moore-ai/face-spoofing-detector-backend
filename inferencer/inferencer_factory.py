import logging
from typing import Union, overload, Literal

from inferencer.base_inferencer import BaseInferencer, FusionBaseInferencer, ModelType
from inferencer.fusion_model_inferencer import FusionModalInferencer
from inferencer.single_model_inferencer import SingleModalInferencer

logger = logging.getLogger(__name__)


class FaceAntiSpoofingInferencer:
    """
    人脸活体检测推理器工厂类
    根据模型类型自动选择对应的推理器
    """

    @overload
    @staticmethod
    def create(
        model_path: str,
        modal_type: Literal[ModelType.SINGLE],
        input_size: int = 112,
        num_classes: int = 2,
        device: str = "cuda",
        rknn_target: str = "rv1126",
    ) -> BaseInferencer: ...

    @overload
    @staticmethod
    def create(
        model_path: str,
        modal_type: Literal[ModelType.FUSION],
        input_size: int = 112,
        num_classes: int = 2,
        device: str = "cuda",
        rknn_target: str = "rv1126",
    ) -> FusionBaseInferencer: ...

    @staticmethod
    def create(
        model_path: str,
        modal_type: ModelType,
        input_size: int = 112,
        num_classes: int = 2,
        device: str = "cuda",
        rknn_target: str = "rv1126",
    ) -> Union[BaseInferencer, FusionBaseInferencer]:
        """
        创建推理器实例

        Args:
            model_path: 模型文件路径
            modal_type: 模态类型（ModalType.SINGLE 或 ModalType.FUSION）
            input_size: 输入图像尺寸
            num_classes: 分类类别数
            device: PyTorch设备，默认从配置读取
            rknn_target: RKNN目标设备

        Returns:
            对应的推理器实例
        """
        logger.info(f"Creating inferencer for {modal_type.value} model: {model_path}")
        try:
            if modal_type == ModelType.SINGLE:
                inferencer = SingleModalInferencer(
                    model_path=model_path,
                    input_size=input_size,
                    num_classes=num_classes,
                    device=device,
                    rknn_target=rknn_target,
                )
                logger.info(f"Single modal inferencer created: {model_path}")
                return inferencer
            elif modal_type == ModelType.FUSION:
                inferencer = FusionModalInferencer(
                    model_path=model_path,
                    input_size=input_size,
                    num_classes=num_classes,
                    device=device,
                    rknn_target=rknn_target,
                )
                logger.info(f"Fusion modal inferencer created: {model_path}")
                return inferencer
            else:
                logger.error(f"Unsupported modal type: {modal_type}")
                raise ValueError(f"不支持的模态类型: {modal_type}")
        except Exception as e:
            logger.error(
                f"Failed to create inferencer for {modal_type.value} model {model_path}: {e}",
                exc_info=True,
            )
            raise
