"""
图片存储管理相关数据模型
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class ImageUploadRequest(BaseModel):
    """图片上传请求"""
    task_id: str = Field(..., description="关联的任务 ID")
    image_type: str = Field(..., description="图片类型：original(原始图), processed(处理图)")
    modality: str = Field(..., description="模态：rgb, ir, fusion")
    image_data: str = Field(..., description="Base64 编码的图片数据")
    metadata: Optional[dict] = Field(None, description="额外元数据")


class ImageUploadResponse(BaseModel):
    """图片上传响应"""
    success: bool
    image_id: str = Field(..., description="图片唯一 ID")
    storage_path: str = Field(..., description="存储路径")
    file_size: int = Field(..., description="文件大小（字节）")
    message: str = Field("", description="响应消息")


class ImageQueryResponse(BaseModel):
    """图片查询响应"""
    image_id: str
    task_id: str
    image_type: str
    modality: str
    storage_path: str
    storage_type: str  # local, s3, etc.
    file_size: int
    content_type: str
    created_at: datetime
    metadata: Optional[dict] = None


class ImageListResponse(BaseModel):
    """图片列表响应"""
    total: int = Field(..., description="总数")
    page: int = Field(..., description="当前页码")
    page_size: int = Field(..., description="每页数量")
    items: List[ImageQueryResponse] = Field(default_factory=list, description="图片列表")


class StorageStatsResponse(BaseModel):
    """存储统计响应"""
    total_images: int = Field(..., description="总图片数")
    total_size_bytes: int = Field(..., description="总存储空间（字节）")
    total_size_mb: float = Field(..., description="总存储空间（MB）")
    quota_bytes: Optional[int] = Field(None, description="存储配额（字节）")
    quota_used_percent: float = Field(..., description="配额使用百分比")
    by_type: dict = Field(default_factory=dict, description="按类型统计")
    by_modality: dict = Field(default_factory=dict, description="按模态统计")


class StorageConfigResponse(BaseModel):
    """存储配置响应"""
    storage_type: str = Field(..., description="存储类型：local, s3")
    storage_path: str = Field(..., description="本地存储路径")
    quota_bytes: Optional[int] = Field(None, description="存储配额")
    retention_days: int = Field(30, description="图片保留天数")


class ImageDeleteResponse(BaseModel):
    """图片删除响应"""
    success: bool
    deleted_count: int = Field(..., description="删除的图片数量")
    freed_size_bytes: int = Field(..., description="释放的存储空间（字节）")
    message: str = Field("", description="响应消息")


class StorageCleanupRequest(BaseModel):
    """存储清理请求"""
    older_than_days: int = Field(..., ge=1, description="清理 N 天前的图片")
    task_ids: Optional[List[str]] = Field(None, description="指定任务 ID 列表")


class StorageQuotaUpdateRequest(BaseModel):
    """存储配额更新请求"""
    quota_bytes: Optional[int] = Field(None, description="新的存储配额（字节）")
    retention_days: Optional[int] = Field(None, description="保留天数")


class ImageBatchDownloadRequest(BaseModel):
    """批量下载图片请求"""
    image_ids: List[str] = Field(..., description="要下载的图片 ID 列表")


class ImageIdListResponse(BaseModel):
    """图片 ID 列表响应"""
    total: int = Field(..., description="总图片数")
    image_ids: List[str] = Field(default_factory=list, description="图片 ID 列表")
