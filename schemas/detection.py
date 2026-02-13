"""
检测相关的数据模型定义
"""

from pydantic import BaseModel


class DetectionResultItem(BaseModel):
    id: str
    result: str  # "real" 或 "fake"
    confidence: float
    timestamp: str
    processing_time: int  # 毫秒


class BatchDetectionResult(BaseModel):
    results: list[DetectionResultItem]
    total_count: int
    real_count: int
    fake_count: int
    average_confidence: float
