"""
健康检查控制器
提供系统健康状态检查，包括模型、GPU、磁盘和依赖服务状态
"""

import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from util.auth import AuthCredentials
from middleware.auth_middleware import get_current_user
from util.config import settings

router = APIRouter()


# ============================================
# 认证依赖
# ============================================

def require_admin(
    auth: AuthCredentials = Depends(get_current_user),
) -> AuthCredentials:
    """要求管理员权限（JWT Token）"""
    if not auth.authenticated:
        raise HTTPException(status_code=401, detail="未授权访问")
    if auth.auth_type != "jwt":
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return auth


# ============================================
# 响应模型
# ============================================

class HealthResponse(BaseModel):
    """基础健康检查响应"""
    status: str
    content: str


class ModelStatus(BaseModel):
    """模型状态"""
    name: str
    path: str
    exists: bool
    file_size_mb: Optional[float] = None


class GPUStatus(BaseModel):
    """GPU 状态"""
    available: bool
    device_name: Optional[str] = None
    memory_total_mb: Optional[float] = None
    memory_free_mb: Optional[float] = None
    memory_used_mb: Optional[float] = None
    memory_used_percent: Optional[float] = None


class DiskStatus(BaseModel):
    """磁盘状态"""
    path: str
    total_gb: float
    used_gb: float
    free_gb: float
    used_percent: float


class DatabaseStatus(BaseModel):
    """数据库状态"""
    connected: bool
    database_url: str
    error_message: Optional[str] = None


class StorageStatus(BaseModel):
    """存储服务状态"""
    initialized: bool
    storage_type: str
    quota_bytes: Optional[int] = None


class DetailedHealthResponse(BaseModel):
    """详细健康检查响应"""
    status: str
    debug_mode: bool = False

    # 模型状态
    models: list[ModelStatus] = []

    # GPU 状态
    gpu: Optional[GPUStatus] = None

    # 磁盘状态
    disk: Optional[DiskStatus] = None

    # 数据库状态
    database: Optional[DatabaseStatus] = None

    # 存储服务状态
    storage: Optional[StorageStatus] = None

    # 整体健康信息
    healthy_components: list[str] = []
    unhealthy_components: list[str] = []


# ============================================
# 健康检查函数
# ============================================

def check_model_status() -> list[ModelStatus]:
    """检查模型文件状态"""
    models = []

    # 检查单模态模型
    single_model_path = Path(settings.MODEL_SINGLE_PATH)
    single_exists = single_model_path.exists()
    single_size = single_model_path.stat().st_size / (1024 * 1024) if single_exists else None

    models.append(ModelStatus(
        name="single_model",
        path=settings.MODEL_SINGLE_PATH,
        exists=single_exists,
        file_size_mb=round(single_size, 2) if single_size else None
    ))

    # 检查融合模型
    fusion_model_path = Path(settings.MODEL_FUSION_PATH)
    fusion_exists = fusion_model_path.exists()
    fusion_size = fusion_model_path.stat().st_size / (1024 * 1024) if fusion_exists else None

    models.append(ModelStatus(
        name="fusion_model",
        path=settings.MODEL_FUSION_PATH,
        exists=fusion_exists,
        file_size_mb=round(fusion_size, 2) if fusion_size else None
    ))

    return models


def check_gpu_status() -> GPUStatus:
    """检查 GPU 状态"""
    try:
        import torch

        if not torch.cuda.is_available():
            return GPUStatus(available=False)

        # 获取 GPU 信息
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0) if device_count > 0 else None

        # 获取显存信息（单位：字节）
        memory_total = torch.cuda.get_device_properties(0).total_memory if device_count > 0 else 0
        memory_allocated = torch.cuda.memory_allocated(0) if device_count > 0 else 0
        memory_reserved = torch.cuda.memory_reserved(0) if device_count > 0 else 0
        memory_free = memory_total - memory_reserved

        # 转换为 MB
        memory_total_mb = memory_total / (1024 * 1024)
        memory_free_mb = memory_free / (1024 * 1024)
        memory_used_mb = memory_reserved / (1024 * 1024)
        memory_used_percent = (memory_reserved / memory_total * 100) if memory_total > 0 else 0

        return GPUStatus(
            available=True,
            device_name=device_name,
            memory_total_mb=round(memory_total_mb, 2),
            memory_free_mb=round(memory_free_mb, 2),
            memory_used_mb=round(memory_used_mb, 2),
            memory_used_percent=round(memory_used_percent, 2)
        )

    except Exception as e:
        # 发生异常时返回不可用状态
        return GPUStatus(available=False)


def check_disk_status(path: str = "storage/images") -> Optional[DiskStatus]:
    """检查磁盘空间状态"""
    try:
        # 获取磁盘使用情况
        usage = shutil.disk_usage(path)

        total_gb = usage.total / (1024 ** 3)
        used_gb = usage.used / (1024 ** 3)
        free_gb = usage.free / (1024 ** 3)
        used_percent = (usage.used / usage.total * 100) if usage.total > 0 else 0

        return DiskStatus(
            path=path,
            total_gb=round(total_gb, 2),
            used_gb=round(used_gb, 2),
            free_gb=round(free_gb, 2),
            used_percent=round(used_percent, 2)
        )

    except Exception as e:
        # 如果存储目录不存在，检查父目录
        try:
            parent_path = str(Path(path).parent)
            usage = shutil.disk_usage(parent_path)

            total_gb = usage.total / (1024 ** 3)
            used_gb = usage.used / (1024 ** 3)
            free_gb = usage.free / (1024 ** 3)
            used_percent = (usage.used / usage.total * 100) if usage.total > 0 else 0

            return DiskStatus(
                path=parent_path,
                total_gb=round(total_gb, 2),
                used_gb=round(used_gb, 2),
                free_gb=round(free_gb, 2),
                used_percent=round(used_percent, 2)
            )
        except Exception:
            return None


def check_database_status() -> DatabaseStatus:
    """检查数据库连接状态"""
    try:
        from db import db_manager

        # 检查是否已初始化
        if not db_manager._initialized:
            return DatabaseStatus(
                connected=False,
                database_url=settings.DATABASE_URL,
                error_message="Database not initialized"
            )

        # 尝试获取会话以测试连接
        try:
            from sqlalchemy import text
            session = db_manager.get_session()
            # 执行简单查询测试连接
            session.execute(text("SELECT 1"))
            session.close()
            return DatabaseStatus(
                connected=True,
                database_url=settings.DATABASE_URL
            )
        except Exception as e:
            return DatabaseStatus(
                connected=False,
                database_url=settings.DATABASE_URL,
                error_message=str(e)
            )

    except Exception as e:
        return DatabaseStatus(
            connected=False,
            database_url=settings.DATABASE_URL,
            error_message=str(e)
        )


def check_storage_status() -> StorageStatus:
    """检查存储服务状态"""
    try:
        from util.storage import storage_manager

        return StorageStatus(
            initialized=storage_manager._initialized,
            storage_type=settings.STORAGE_TYPE,
            quota_bytes=settings.STORAGE_QUOTA_BYTES
        )

    except Exception:
        return StorageStatus(
            initialized=False,
            storage_type=settings.STORAGE_TYPE,
            quota_bytes=settings.STORAGE_QUOTA_BYTES
        )


# ============================================
# API 端点
# ============================================

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """简单健康检查端点"""
    return HealthResponse(
        status="ok",
        content="Hello, from fastapi",
    )


@router.get(
    "/health/detailed",
    response_model=DetailedHealthResponse,
    dependencies=[Depends(require_admin)],
)
async def detailed_health_check():
    """详细健康检查端点（需要管理员权限）

    返回系统各组件的健康状态：
    - 模型文件状态
    - GPU 状态
    - 磁盘空间
    - 数据库连接
    - 存储服务状态
    """
    healthy_components: list[str] = []
    unhealthy_components: list[str] = []

    # 1. 检查模型状态
    models = check_model_status()
    for model in models:
        if model.exists:
            healthy_components.append(f"model:{model.name}")
        else:
            unhealthy_components.append(f"model:{model.name}")

    # 2. 检查 GPU 状态
    gpu = check_gpu_status()
    if gpu.available:
        healthy_components.append("gpu")
    else:
        unhealthy_components.append("gpu")

    # 3. 检查磁盘状态
    disk = check_disk_status(settings.STORAGE_LOCAL_PATH)
    if disk and disk.free_gb > 1:  # 至少 1GB 可用
        healthy_components.append("disk")
    elif disk:
        unhealthy_components.append("disk")

    # 4. 检查数据库状态
    database = check_database_status()
    if database.connected:
        healthy_components.append("database")
    else:
        unhealthy_components.append("database")

    # 5. 检查存储服务状态
    storage = check_storage_status()
    if storage.initialized:
        healthy_components.append("storage")
    else:
        unhealthy_components.append("storage")

    # 确定整体状态
    if unhealthy_components:
        overall_status = "degraded" if healthy_components else "unhealthy"
    else:
        overall_status = "healthy"

    return DetailedHealthResponse(
        status=overall_status,
        debug_mode=settings.DEBUG_MODE,
        models=models,
        gpu=gpu,
        disk=disk,
        database=database,
        storage=storage,
        healthy_components=healthy_components,
        unhealthy_components=unhealthy_components,
    )
