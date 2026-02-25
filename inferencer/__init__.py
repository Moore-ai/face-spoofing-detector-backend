"""推理器模块"""

from inferencer.base_inferencer import (
    BaseInferencer,
    FusionBaseInferencer,
    ModelFormat,
    ModelType,
)
from inferencer.single_model_inferencer import SingleModalInferencer
from inferencer.fusion_model_inferencer import FusionModalInferencer
from inferencer.debug_inferencer import DebugSingleInferencer, DebugFusionInferencer
from inferencer.inferencer_factory import FaceAntiSpoofingInferencer

__all__ = [
    # 基类
    "BaseInferencer",
    "FusionBaseInferencer",
    "ModelFormat",
    "ModelType",
    # 真实模型推理器
    "SingleModalInferencer",
    "FusionModalInferencer",
    # 调试模式推理器
    "DebugSingleInferencer",
    "DebugFusionInferencer",
    # 工厂
    "FaceAntiSpoofingInferencer",
]
