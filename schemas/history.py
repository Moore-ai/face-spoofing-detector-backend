"""
检测结果历史记录相关的数据模型
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class HistoryResultItem(BaseModel):
    """单条检测历史记录"""

    mode: str  # "single" 或 "fusion"
    modality: Optional[str] = None  # "rgb", "ir" 或 None（融合模式）
    result: str  # "real", "fake", 或 "error"
    confidence: float
    probabilities: List[float]  # [real_prob, fake_prob]
    processing_time: int  # 毫秒
    image_index: Optional[int] = None  # 批次中的索引
    error: Optional[str] = None  # 错误信息（如果有）
    retry_count: int = 0  # 重试次数


class HistoryTaskItem(BaseModel):
    """任务历史记录"""

    task_id: str
    client_id: Optional[str] = None  # 客户端 ID
    api_key_hash: Optional[str] = None  # API Key 的哈希（用于追踪用户，不存储原始 key）
    mode: str  # "single" 或 "fusion"
    status: str  # "completed", "partial_failure", "failed"
    total_items: int
    successful_items: int
    failed_items: int
    real_count: int
    fake_count: int
    elapsed_time_ms: int
    created_at: datetime
    completed_at: datetime
    results: Optional[List[HistoryResultItem]] = None


class HistoryQueryResponse(BaseModel):
    """历史记录查询响应"""

    total: int
    page: int
    page_size: int
    total_pages: int
    items: List[HistoryTaskItem]


class HistoryStatsResponse(BaseModel):
    """历史统计信息响应"""

    total_tasks: int
    total_inferences: int
    total_real: int
    total_fake: int
    total_errors: int
    success_rate: float  # 成功率
    avg_processing_time_ms: float  # 平均处理时间
    date_range: dict  # {"start": str, "end": str}


class HistoryDeleteResponse(BaseModel):
    """历史记录删除响应"""

    deleted_count: int
    message: str
