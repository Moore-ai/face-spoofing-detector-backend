"""
数据库模型定义
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship
from db import Base


class DetectionTask(Base):
    """检测任务表"""

    __tablename__ = "detection_tasks"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    task_id = Column(String(64), unique=True, index=True, nullable=False)
    client_id = Column(String(64), index=True, nullable=True)  # 客户端 ID
    api_key_hash = Column(String(64), index=True, nullable=True)  # API Key 哈希

    # 任务信息
    mode = Column(String(16), nullable=False)  # "single" 或 "fusion"
    status = Column(String(32), nullable=False, default="pending")  # 任务状态
    total_items = Column(Integer, nullable=False, default=0)
    successful_items = Column(Integer, nullable=False, default=0)
    failed_items = Column(Integer, nullable=False, default=0)

    # 统计信息
    real_count = Column(Integer, nullable=False, default=0)
    fake_count = Column(Integer, nullable=False, default=0)
    error_count = Column(Integer, nullable=False, default=0)
    elapsed_time_ms = Column(Integer, nullable=False, default=0)

    # 时间戳
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # 关联的结果
    results = relationship(
        "DetectionResult",
        back_populates="task",
        cascade="all, delete-orphan",
        lazy="selectin",  # 预加载
    )

    def __repr__(self):
        return f"<DetectionTask(task_id='{self.task_id}', status='{self.status}')>"


class DetectionResult(Base):
    """检测结果表"""

    __tablename__ = "detection_results"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    task_id = Column(String(64), ForeignKey("detection_tasks.task_id", ondelete="CASCADE"), nullable=False)
    image_index = Column(Integer, nullable=True)  # 批次中的索引

    # 检测信息
    mode = Column(String(16), nullable=False)  # "single" 或 "fusion"
    modality = Column(String(16), nullable=True)  # "rgb", "ir" 或 None
    result = Column(String(16), nullable=False)  # "real", "fake", 或 "error"
    confidence = Column(Float, nullable=False)
    prob_real = Column(Float, nullable=False)  # real 概率
    prob_fake = Column(Float, nullable=False)  # fake 概率
    processing_time = Column(Integer, nullable=False)  # 毫秒

    # 错误信息
    is_error = Column(Boolean, nullable=False, default=False)
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, nullable=False, default=0)

    # 创建时间
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # 关联的任务
    task = relationship("DetectionTask", back_populates="results")

    def __repr__(self):
        return f"<DetectionResult(task_id='{self.task_id}', index={self.image_index}, result='{self.result}')>"
