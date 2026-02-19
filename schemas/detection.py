"""
检测相关的数据模型定义
"""

from pydantic import BaseModel
import numpy as np
from typing import List, Optional


class DetectionResultItem(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    mode: str
    result: str  # "real" 或 "fake"
    confidence: float
    probabilities: np.ndarray
    processing_time: int


class AsyncTaskResponse(BaseModel):
    """异步任务响应（立即返回task_id）"""

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

