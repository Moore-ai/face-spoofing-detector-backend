"""
检测相关的数据模型定义
"""

from pydantic import BaseModel, field_serializer
import numpy as np
from typing import List, Optional


class DetectionResultItem(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    mode: str
    result: str  # "real" 或 "fake"
    confidence: float
    probabilities: np.ndarray
    processing_time: int

    @field_serializer("probabilities")
    def serialize_probabilities(self, value: np.ndarray) -> list[float]:
        """将 numpy 数组序列化为 Python 列表"""
        return value.tolist()


class AsyncTaskResponse(BaseModel):
    """异步任务响应（立即返回 task_id）"""

    task_id: str
    message: str = "任务已创建，正在后台处理"


class TaskStatusResponse(BaseModel):
    """任务状态查询响应"""

    task_id: str
    status: str  # pending, running, completed, failed
    total_items: int
    completed_items: int
    progress_percentage: float
    real_count: int
    fake_count: int
    elapsed_time_ms: int
    message: str
    results: Optional[List[DetectionResultItem]] = None
    current_result: Optional[DetectionResultItem] = None


class SingleModeRequest(BaseModel):
    """单模态推理请求"""

    mode: str
    modality: str  # "rgb" 或 "ir"
    images: list[str]  # base64 编码的图像列表


class ImagePair(BaseModel):
    """融合模态图像对"""

    rgb: str  # base64 编码的 RGB 图像
    ir: str  # base64 编码的 IR 图像


class FusionModeRequest(BaseModel):
    """融合模态推理请求"""

    mode: str
    pairs: list[ImagePair]  # 图像对列表
