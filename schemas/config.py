"""
应用配置相关数据模型
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal


class LoggingConfigResponse(BaseModel):
    """日志配置响应"""
    log_level: str = Field(..., description="日志级别")
    log_to_console: bool = Field(default=False, description="是否输出到控制台")
    log_json_format: bool = Field(default=False, description="是否 JSON 格式")
    log_request_body: bool = Field(default=False, description="是否记录请求体")
    log_response_body: bool = Field(default=False, description="是否记录响应体")
    audit_log_enabled: bool = Field(default=True, description="是否启用审计日志")


class RetryConfigResponse(BaseModel):
    """重试配置响应"""
    retry_enabled: bool = Field(default=True, description="是否启用重试")
    retry_max_attempts: int = Field(default=3, ge=1, le=10, description="最大重试次数")
    retry_delay_seconds: float = Field(default=1.0, ge=0, description="重试基础延迟")
    retry_exponential_backoff: bool = Field(default=True, description="是否启用指数退避")
    retry_max_delay_seconds: float = Field(default=10.0, ge=0, description="最大重试延迟")


class DebugConfigResponse(BaseModel):
    """调试配置响应"""
    debug_mode: bool = Field(default=False, description="调试模式")
    debug_delay_per_image: float = Field(default=0.5, ge=0, description="单模态每图延迟")
    debug_delay_per_pair: float = Field(default=0.8, ge=0, description="融合模态每对延迟")
    debug_failure_rate: float = Field(default=0.0, ge=0, le=1, description="失败模拟率")


class StorageSaveStrategyResponse(BaseModel):
    """存储策略配置响应"""
    storage_auto_save: bool = Field(default=True, description="是否自动保存")
    storage_save_strategy: Literal["never", "always", "error_only", "smart"] = Field(
        default="error_only", description="存储策略"
    )
    storage_save_error_rate: float = Field(default=1.0, ge=0, le=1, description="错误结果保存率")
    storage_save_fake_rate: float = Field(default=0.1, ge=0, le=1, description="伪造样本保存率")
    storage_save_real_rate: float = Field(default=0.01, ge=0, le=1, description="真实样本保存率")
    storage_max_per_task: int = Field(default=10, ge=1, description="每任务最多保存数")


class ImageCompressConfigResponse(BaseModel):
    """图片压缩配置响应"""
    compress_enabled: bool = Field(default=True, description="是否启用压缩")
    compress_quality: int = Field(default=75, ge=1, le=100, description="压缩质量")
    compress_type: Literal["opencv", "pillow", "resize"] = Field(default="opencv", description="压缩器类型")


class AppConfigResponse(BaseModel):
    """应用配置总响应"""
    logging: LoggingConfigResponse
    retry: RetryConfigResponse
    debug: DebugConfigResponse
    storage_save_strategy: StorageSaveStrategyResponse
    image_compress: ImageCompressConfigResponse


class LoggingConfigUpdateRequest(BaseModel):
    """日志配置更新请求"""
    log_level: Optional[Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]] = Field(
        None, description="日志级别"
    )
    log_to_console: Optional[bool] = Field(None, description="是否输出到控制台")
    log_json_format: Optional[bool] = Field(None, description="是否 JSON 格式")
    log_request_body: Optional[bool] = Field(None, description="是否记录请求体")
    log_response_body: Optional[bool] = Field(None, description="是否记录响应体")
    audit_log_enabled: Optional[bool] = Field(None, description="是否启用审计日志")


class RetryConfigUpdateRequest(BaseModel):
    """重试配置更新请求"""
    retry_enabled: Optional[bool] = Field(None, description="是否启用重试")
    retry_max_attempts: Optional[int] = Field(None, ge=1, le=10, description="最大重试次数")
    retry_delay_seconds: Optional[float] = Field(None, ge=0, description="重试基础延迟")
    retry_exponential_backoff: Optional[bool] = Field(None, description="是否指数退避")
    retry_max_delay_seconds: Optional[float] = Field(None, ge=0, description="最大重试延迟")


class DebugConfigUpdateRequest(BaseModel):
    """调试配置更新请求"""
    debug_mode: Optional[bool] = Field(None, description="调试模式")
    debug_delay_per_image: Optional[float] = Field(None, ge=0, description="单模态每图延迟")
    debug_delay_per_pair: Optional[float] = Field(None, ge=0, description="融合模态每对延迟")
    debug_failure_rate: Optional[float] = Field(None, ge=0, le=1, description="失败模拟率")


class StorageSaveStrategyUpdateRequest(BaseModel):
    """存储策略配置更新请求"""
    storage_auto_save: Optional[bool] = Field(None, description="是否自动保存")
    storage_save_strategy: Optional[Literal["never", "always", "error_only", "smart"]] = Field(
        None, description="存储策略"
    )
    storage_save_error_rate: Optional[float] = Field(None, ge=0, le=1, description="错误结果保存率")
    storage_save_fake_rate: Optional[float] = Field(None, ge=0, le=1, description="伪造样本保存率")
    storage_save_real_rate: Optional[float] = Field(None, ge=0, le=1, description="真实样本保存率")
    storage_max_per_task: Optional[int] = Field(None, ge=1, description="每任务最多保存数")


class ImageCompressConfigUpdateRequest(BaseModel):
    """图片压缩配置更新请求"""
    compress_enabled: Optional[bool] = Field(None, description="是否启用压缩")
    compress_quality: Optional[int] = Field(None, ge=1, le=100, description="压缩质量")
    compress_type: Optional[Literal["opencv", "pillow", "resize"]] = Field(None, description="压缩器类型")
