"""
检测相关的数据模型定义
"""

from pydantic import BaseModel, field_serializer, Field
import numpy as np
from typing import List, Optional


class DetectionResultItem(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    mode: str
    result: str  # "real", "fake", 或 "error"
    confidence: float
    probabilities: np.ndarray
    processing_time: int
    # 新增字段
    image_index: Optional[int] = None  # 批次中的索引
    error: Optional[str] = None  # 错误信息
    retry_count: int = 0  # 重试次数
    success: bool = True  # 是否成功

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
    status: str  # pending, running, completed, partial_failure, failed, cancelled
    total_items: int
    completed_items: int
    failed_items: int = 0  # 失败项数量
    progress_percentage: float
    real_count: int
    fake_count: int
    error_count: int = 0  # 错误计数
    elapsed_time_ms: int
    message: str
    results: Optional[List[DetectionResultItem]] = None
    current_result: Optional[DetectionResultItem] = None
    errors: Optional[List[dict]] = None  # 错误详情列表
    priority: int = Field(default=0, ge=0, le=100, description="任务优先级，范围 0-100，值越大优先级越高")


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


class TaskQueueStatusResponse(BaseModel):
    """任务队列状态响应"""

    is_running: bool
    queue_size: int
    max_workers: int
    active_workers: int
