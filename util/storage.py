"""
图片存储管理模块
支持本地存储和对象存储（S3 兼容）
"""

import os
import uuid
import base64
import shutil
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

from util.config import settings

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """存储后端抽象基类"""

    @abstractmethod
    def save(self, image_id: str, image_data: bytes, metadata: dict) -> dict:
        """保存图片"""
        pass

    @abstractmethod
    def load(self, image_id: str) -> Optional[bytes]:
        """加载图片"""
        pass

    @abstractmethod
    def delete(self, image_id: str) -> bool:
        """删除图片"""
        pass

    @abstractmethod
    def exists(self, image_id: str) -> bool:
        """检查图片是否存在"""
        pass

    @abstractmethod
    def get_metadata(self, image_id: str) -> Optional[dict]:
        """获取图片元数据"""
        pass

    @abstractmethod
    def list_images(
        self,
        task_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict:
        """列出图片"""
        pass

    @abstractmethod
    def get_stats(self) -> dict:
        """获取存储统计信息"""
        pass

    @abstractmethod
    def cleanup(self, older_than_days: int, task_ids: Optional[List[str]] = None) -> dict:
        """清理旧图片"""
        pass


class LocalStorageBackend(StorageBackend):
    """本地文件存储后端"""

    def __init__(self, base_path: str, quota_bytes: Optional[int] = None):
        """
        初始化本地存储

        Args:
            base_path: 基础存储路径
            quota_bytes: 存储配额（字节），None 表示无限制
        """
        self.base_path = Path(base_path)
        self.quota_bytes = quota_bytes
        self.metadata_file = self.base_path / "_metadata.json"
        self.metadata: Dict[str, dict] = {}

        # 确保存储目录存在
        self.base_path.mkdir(parents=True, exist_ok=True)

        # 加载元数据
        self._load_metadata()

        logger.info(f"LocalStorageBackend initialized: {self.base_path}")

    def _load_metadata(self):
        """加载元数据文件"""
        import json

        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded metadata for {len(self.metadata)} images")
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
                self.metadata = {}
        else:
            self.metadata = {}

    def _save_metadata(self):
        """保存元数据文件"""
        import json

        try:
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def _get_image_path(self, image_id: str) -> Path:
        """获取图片存储路径"""
        # 使用两级目录结构避免单目录文件过多
        prefix1 = image_id[:2]
        prefix2 = image_id[2:4]
        dir_path = self.base_path / prefix1 / prefix2
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path / f"{image_id}.bin"

    def _get_current_size(self) -> int:
        """获取当前存储总大小"""
        total_size = 0
        for path in self.base_path.rglob("*.bin"):
            try:
                total_size += path.stat().st_size
            except Exception:
                pass
        return total_size

    def _check_quota(self, required_bytes: int) -> bool:
        """检查配额是否足够"""
        if self.quota_bytes is None:
            return True

        current_size = self._get_current_size()
        return (current_size + required_bytes) <= self.quota_bytes

    def update_quota(self, quota_bytes: Optional[int]) -> None:
        """更新存储配额"""
        self.quota_bytes = quota_bytes
        logger.info(f"Storage quota updated to: {quota_bytes} bytes")

    def save(self, image_id: str, image_data: bytes, metadata: dict) -> dict:
        """保存图片"""
        # 检查配额
        if not self._check_quota(len(image_data)):
            raise StorageQuotaExceededError(
                f"Storage quota exceeded. Required: {len(image_data)} bytes"
            )

        # 获取存储路径
        image_path = self._get_image_path(image_id)

        # 保存图片数据
        try:
            with open(image_path, "wb") as f:
                f.write(image_data)
        except Exception as e:
            logger.error(f"Failed to save image {image_id}: {e}")
            raise StorageError(f"Failed to save image: {e}")

        # 更新元数据
        file_size = image_path.stat().st_size
        meta = {
            "image_id": image_id,
            "storage_path": str(image_path.relative_to(self.base_path)),
            "file_size": file_size,
            "content_type": metadata.get("content_type", "image/jpeg"),
            "created_at": datetime.utcnow().isoformat(),
            **metadata,
        }
        self.metadata[image_id] = meta
        self._save_metadata()

        logger.info(f"Saved image {image_id}: {file_size} bytes")

        return {
            "image_id": image_id,
            "storage_path": str(image_path),
            "file_size": file_size,
            "relative_path": meta["storage_path"],
        }

    def load(self, image_id: str) -> Optional[bytes]:
        """加载图片"""
        image_path = self._get_image_path(image_id)

        if not image_path.exists():
            return None

        try:
            with open(image_path, "rb") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to load image {image_id}: {e}")
            return None

    def delete(self, image_id: str) -> bool:
        """删除图片"""
        image_path = self._get_image_path(image_id)

        try:
            if image_path.exists():
                file_size = image_path.stat().st_size
                image_path.unlink()
                logger.info(f"Deleted image {image_id}: {file_size} bytes")

                # 清理空目录
                self._cleanup_empty_dirs(image_path.parent)

                # 更新元数据
                if image_id in self.metadata:
                    del self.metadata[image_id]
                    self._save_metadata()

                return True
            else:
                logger.warning(f"Image {image_id} not found for deletion")
                return False
        except Exception as e:
            logger.error(f"Failed to delete image {image_id}: {e}")
            return False

    def _cleanup_empty_dirs(self, dir_path: Path):
        """清理空目录"""
        try:
            while dir_path != self.base_path:
                if not any(dir_path.iterdir()):
                    dir_path.rmdir()
                    logger.debug(f"Removed empty directory: {dir_path}")
                    dir_path = dir_path.parent
                else:
                    break
        except Exception as e:
            logger.debug(f"Error cleaning up directories: {e}")

    def exists(self, image_id: str) -> bool:
        """检查图片是否存在"""
        return self._get_image_path(image_id).exists()

    def get_metadata(self, image_id: str) -> Optional[dict]:
        """获取图片元数据"""
        return self.metadata.get(image_id)

    def list_images(
        self,
        task_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict:
        """列出图片"""
        # 过滤元数据
        filtered = []
        for image_id, meta in self.metadata.items():
            # 按任务 ID 过滤
            if task_id and meta.get("task_id") != task_id:
                continue

            # 按日期过滤
            try:
                created_at = datetime.fromisoformat(meta["created_at"])
                if start_date and created_at < start_date:
                    continue
                if end_date and created_at > end_date:
                    continue
            except Exception:
                pass

            filtered.append(meta)

        # 分页
        total = len(filtered)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        items = filtered[start_idx:end_idx]

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "items": items,
        }

    def get_stats(self) -> dict:
        """获取存储统计信息"""
        total_size = self._get_current_size()
        total_images = len(self.metadata)

        # 按类型统计
        by_type: Dict[str, dict] = {}
        by_modality: Dict[str, dict] = {}

        for meta in self.metadata.values():
            # 按类型统计
            image_type = meta.get("image_type", "unknown")
            if image_type not in by_type:
                by_type[image_type] = {"count": 0, "size": 0}
            by_type[image_type]["count"] += 1
            by_type[image_type]["size"] += meta.get("file_size", 0)

            # 按模态统计
            modality = meta.get("modality", "unknown")
            if modality not in by_modality:
                by_modality[modality] = {"count": 0, "size": 0}
            by_modality[modality]["count"] += 1
            by_modality[modality]["size"] += meta.get("file_size", 0)

        # 计算配额使用率
        quota_used_percent = 0.0
        if self.quota_bytes:
            quota_used_percent = (total_size / self.quota_bytes) * 100

        return {
            "total_images": total_images,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "quota_bytes": self.quota_bytes,
            "quota_used_percent": round(quota_used_percent, 2),
            "by_type": by_type,
            "by_modality": by_modality,
        }

    def cleanup(self, older_than_days: int, task_ids: Optional[List[str]] = None) -> dict:
        """清理旧图片"""
        cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)
        deleted_count = 0
        freed_size = 0

        # 找出需要删除的图片
        to_delete = []
        for image_id, meta in self.metadata.items():
            # 如果指定了任务 ID 列表，只删除这些任务的图片
            if task_ids:
                if meta.get("task_id") not in task_ids:
                    continue
            else:
                # 否则按日期清理
                try:
                    created_at = datetime.fromisoformat(meta["created_at"])
                    if created_at >= cutoff_date:
                        continue
                except Exception:
                    continue

            to_delete.append((image_id, meta.get("file_size", 0)))

        # 删除图片
        for image_id, file_size in to_delete:
            if self.delete(image_id):
                deleted_count += 1
                freed_size += file_size

        logger.info(f"Cleaned up {deleted_count} images, freed {freed_size} bytes")

        return {
            "deleted_count": deleted_count,
            "freed_size_bytes": freed_size,
        }


class S3StorageBackend(StorageBackend):
    """S3 对象存储后端（预留实现）"""

    def __init__(
        self,
        bucket: str,
        region: str,
        endpoint_url: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
    ):
        """
        初始化 S3 存储

        Args:
            bucket: S3 桶名
            region: AWS 区域
            endpoint_url: S3 兼容端点（用于 MinIO 等）
            access_key: 访问密钥
            secret_key: 密钥
        """
        self.bucket = bucket
        self.region = region
        self.endpoint_url = endpoint_url
        self.access_key = access_key
        self.secret_key = secret_key

        # 延迟导入 boto3，避免未安装时的导入错误
        self.s3_client = None
        self._init_client()

        logger.info(f"S3StorageBackend initialized: {bucket}@{region}")

    def _init_client(self):
        """初始化 S3 客户端"""
        try:
            import boto3
            from botocore.config import Config

            config = Config(
                retries={"max_attempts": 3},
                connect_timeout=5,
            )

            self.s3_client = boto3.client(
                "s3",
                region_name=self.region,
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                config=config,
            )

            # 确保桶存在
            self._ensure_bucket_exists()

        except ImportError:
            logger.error("boto3 not installed. Install with: pip install boto3")
            raise StorageError("boto3 library not installed")
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            raise StorageError(f"Failed to initialize S3: {e}")

    def _ensure_bucket_exists(self):
        """确保 S3 桶存在"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket) # type: ignore
        except Exception:
            # 桶不存在，尝试创建
            try:
                self.s3_client.create_bucket( # type: ignore
                    Bucket=self.bucket,
                    CreateBucketConfiguration={"LocationConstraint": self.region},
                )
                logger.info(f"Created S3 bucket: {self.bucket}")
            except Exception as e:
                logger.error(f"Failed to create S3 bucket: {e}")
                raise

    def _get_object_key(self, image_id: str) -> str:
        """获取 S3 对象键"""
        # 使用两级前缀结构
        prefix1 = image_id[:2]
        prefix2 = image_id[2:4]
        return f"images/{prefix1}/{prefix2}/{image_id}.bin"

    def save(self, image_id: str, image_data: bytes, metadata: dict) -> dict:
        """保存图片到 S3"""
        key = self._get_object_key(image_id)

        try:
            # 准备元数据
            s3_metadata = {}
            for k, v in metadata.items():
                if isinstance(v, str):
                    s3_metadata[k] = v
                else:
                    s3_metadata[k] = str(v)

            # 上传
            self.s3_client.put_object( # type: ignore
                Bucket=self.bucket,
                Key=key,
                Body=image_data,
                Metadata=s3_metadata,
                ContentType=metadata.get("content_type", "application/octet-stream"),
            )

            logger.info(f"Saved image {image_id} to S3: {len(image_data)} bytes")

            return {
                "image_id": image_id,
                "storage_path": f"s3://{self.bucket}/{key}",
                "file_size": len(image_data),
            }

        except Exception as e:
            logger.error(f"Failed to save image {image_id} to S3: {e}")
            raise StorageError(f"Failed to save to S3: {e}")

    def load(self, image_id: str) -> Optional[bytes]:
        """从 S3 加载图片"""
        key = self._get_object_key(image_id)

        try:
            response = self.s3_client.get_object( # type: ignore
                Bucket=self.bucket,
                Key=key,
            )
            return response["Body"].read()
        except Exception as e:
            logger.error(f"Failed to load image {image_id} from S3: {e}")
            return None

    def delete(self, image_id: str) -> bool:
        """从 S3 删除图片"""
        key = self._get_object_key(image_id)

        try:
            self.s3_client.delete_object( # type: ignore
                Bucket=self.bucket,
                Key=key,
            )
            logger.info(f"Deleted image {image_id} from S3")
            return True
        except Exception as e:
            logger.error(f"Failed to delete image {image_id} from S3: {e}")
            return False

    def exists(self, image_id: str) -> bool:
        """检查图片是否存在于 S3"""
        key = self._get_object_key(image_id)

        try:
            self.s3_client.head_object( # type: ignore
                Bucket=self.bucket,
                Key=key,
            )
            return True
        except Exception:
            return False

    def get_metadata(self, image_id: str) -> Optional[dict]:
        """获取 S3 图片元数据"""
        key = self._get_object_key(image_id)

        try:
            response = self.s3_client.head_object( # type: ignore
                Bucket=self.bucket,
                Key=key,
            )
            return response.get("Metadata", {})
        except Exception:
            return None

    def list_images(
        self,
        task_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict:
        """列出 S3 中的图片"""
        # S3 列表操作较为耗时，这里简化实现
        # 生产环境建议使用单独的数据库记录元数据
        prefix = "images/"

        try:
            response = self.s3_client.list_objects_v2( # type: ignore
                Bucket=self.bucket,
                Prefix=prefix,
                MaxKeys=page_size,
            )

            items = []
            for obj in response.get("Contents", []):
                # 从键名提取 image_id
                key = obj["Key"]
                if key.endswith(".bin"):
                    image_id = key.split("/")[-1].replace(".bin", "")

                    # 获取元数据
                    metadata = self.get_metadata(image_id)

                    # 过滤
                    if task_id and metadata.get("task_id") != task_id: # type: ignore
                        continue

                    items.append(
                        {
                            "image_id": image_id,
                            "storage_path": f"s3://{self.bucket}/{key}",
                            "file_size": obj["Size"],
                            "created_at": obj["LastModified"].isoformat(),
                            **(metadata or {}),
                        }
                    )

            return {
                "total": len(items),
                "page": page,
                "page_size": page_size,
                "items": items[:page_size],
            }

        except Exception as e:
            logger.error(f"Failed to list images from S3: {e}")
            return {
                "total": 0,
                "page": page,
                "page_size": page_size,
                "items": [],
            }

    def get_stats(self) -> dict:
        """获取 S3 存储统计"""
        # 简化实现，生产环境建议使用 CloudWatch API
        prefix = "images/"
        total_size = 0
        total_count = 0

        try:
            paginator = self.s3_client.get_paginator("list_objects_v2") # type: ignore
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    total_count += 1
                    total_size += obj["Size"]
        except Exception as e:
            logger.error(f"Failed to get S3 stats: {e}")

        return {
            "total_images": total_count,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "quota_bytes": None,  # S3 无配额限制
            "quota_used_percent": 0.0,
            "by_type": {},
            "by_modality": {},
        }

    def cleanup(self, older_than_days: int, task_ids: Optional[List[str]] = None) -> dict:
        """清理 S3 中的旧图片"""
        # 简化实现
        cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)
        deleted_count = 0
        freed_size = 0

        prefix = "images/"

        try:
            paginator = self.s3_client.get_paginator("list_objects_v2") # type: ignore
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    if obj["LastModified"].replace(tzinfo=None) < cutoff_date:
                        key = obj["Key"]
                        self.s3_client.delete_object( # type: ignore
                            Bucket=self.bucket,
                            Key=key,
                        )
                        deleted_count += 1
                        freed_size += obj["Size"]
        except Exception as e:
            logger.error(f"Failed to cleanup S3 images: {e}")

        return {
            "deleted_count": deleted_count,
            "freed_size_bytes": freed_size,
        }


class StorageError(Exception):
    """存储操作异常"""

    pass


class StorageQuotaExceededError(StorageError):
    """存储配额超限异常"""

    pass


class StorageManager:
    """存储管理器（单例模式）"""

    _instance: Optional["StorageManager"] = None

    def __init__(self):
        self.backend: Optional[StorageBackend] = None
        self._initialized = False

    @classmethod
    def get_instance(cls) -> "StorageManager":
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize(
        self,
        storage_type: str = "local",
        storage_path: Optional[str] = None,
        quota_bytes: Optional[int] = None,
        s3_config: Optional[dict] = None,
    ):
        """
        初始化存储管理器

        Args:
            storage_type: 存储类型 - "local" 或 "s3"
            storage_path: 本地存储路径（local 模式）
            quota_bytes: 存储配额（字节）
            s3_config: S3 配置字典（s3 模式）
        """
        if self._initialized:
            logger.warning("StorageManager already initialized")
            return

        logger.info(f"Initializing storage manager: type={storage_type}")

        if storage_type == "local":
            path = storage_path or settings.STORAGE_LOCAL_PATH
            quota = quota_bytes or settings.STORAGE_QUOTA_BYTES
            self.backend = LocalStorageBackend(path, quota)
        elif storage_type == "s3":
            if not s3_config:
                s3_config = {
                    "bucket": settings.S3_BUCKET,
                    "region": settings.S3_REGION,
                    "endpoint_url": settings.S3_ENDPOINT_URL,
                    "access_key": settings.S3_ACCESS_KEY,
                    "secret_key": settings.S3_SECRET_KEY,
                }
            self.backend = S3StorageBackend(**s3_config)
        else:
            raise ValueError(f"Unknown storage type: {storage_type}")

        self._initialized = True
        logger.info("Storage manager initialized successfully")

    def save_image(
        self,
        task_id: str,
        image_data: str,
        image_type: str,
        modality: str,
        metadata: Optional[dict] = None,
    ) -> dict:
        """
        保存图片

        Args:
            task_id: 关联的任务 ID
            image_data: Base64 编码的图片数据
            image_type: 图片类型 - "original" 或 "processed"
            modality: 模态 - "rgb", "ir", "fusion"
            metadata: 额外元数据

        Returns:
            保存结果字典
        """
        if not self.backend:
            raise StorageError("Storage manager not initialized")

        # 生成图片 ID
        image_id = str(uuid.uuid4())

        # 解码 Base64
        try:
            if "," in image_data:
                # 移除 data:image/jpeg;base64, 前缀
                image_data = image_data.split(",", 1)[1]
            decoded_data = base64.b64decode(image_data)
        except Exception as e:
            logger.error(f"Failed to decode image data: {e}")
            raise StorageError(f"Invalid image data: {e}")

        # 准备元数据
        meta = {
            "task_id": task_id,
            "image_type": image_type,
            "modality": modality,
            "content_type": "image/jpeg",
        }
        if metadata:
            meta.update(metadata)

        # 保存
        result = self.backend.save(image_id, decoded_data, meta)

        logger.info(f"Saved image {image_id} for task {task_id}")

        return {
            "image_id": image_id,
            **result,
        }

    def get_image(self, image_id: str) -> Optional[bytes]:
        """获取图片数据"""
        if not self.backend:
            raise StorageError("Storage manager not initialized")
        return self.backend.load(image_id)

    def delete_image(self, image_id: str) -> bool:
        """删除图片"""
        if not self.backend:
            raise StorageError("Storage manager not initialized")
        return self.backend.delete(image_id)

    def query_images(
        self,
        task_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict:
        """查询图片列表"""
        if not self.backend:
            raise StorageError("Storage manager not initialized")
        return self.backend.list_images(task_id, start_date, end_date, page, page_size)

    def get_stats(self) -> dict:
        """获取存储统计"""
        if not self.backend:
            raise StorageError("Storage manager not initialized")
        return self.backend.get_stats()

    def cleanup_images(
        self, older_than_days: int, task_ids: Optional[List[str]] = None
    ) -> dict:
        """清理旧图片"""
        if not self.backend:
            raise StorageError("Storage manager not initialized")
        return self.backend.cleanup(older_than_days, task_ids)


# 全局存储管理器实例
storage_manager = StorageManager.get_instance()
