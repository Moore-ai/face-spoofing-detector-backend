"""
图片压缩模块
支持多种压缩策略，使用工厂模式
"""

import io
import logging
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


class ImageCompressor(ABC):
    """图片压缩器抽象基类"""

    @abstractmethod
    def compress(self, image_data: bytes, quality: int = 85) -> bytes:
        """
        压缩图片

        Args:
            image_data: 原始图片数据（JPEG 格式）
            quality: 压缩质量（1-100，越高越清晰）

        Returns:
            压缩后的图片数据
        """
        pass

    @abstractmethod
    def compress_from_base64(self, image_base64: str, quality: int = 85) -> str:
        """
        压缩 Base64 编码的图片

        Args:
            image_base64: Base64 编码的图片数据
            quality: 压缩质量（1-100）

        Returns:
            压缩后的 Base64 图片数据
        """
        pass


class OpenCVCompressor(ImageCompressor):
    """OpenCV 图片压缩器"""

    def __init__(self):
        try:
            import cv2
            self.cv2 = cv2
        except ImportError:
            logger.error("OpenCV not installed. Install with: pip install opencv-python")
            raise ImportError("OpenCV is required for OpenCVCompressor")

    def compress(self, image_data: bytes, quality: int = 85) -> bytes:
        """使用 OpenCV 压缩 JPEG 图片"""
        try:
            # 解码图片
            np_array = np.frombuffer(image_data, np.uint8)
            image = self.cv2.imdecode(np_array, self.cv2.IMREAD_COLOR)

            if image is None:
                logger.warning("Failed to decode image for compression")
                return image_data

            # 压缩参数
            encode_params = [
                self.cv2.IMWRITE_JPEG_QUALITY,
                max(1, min(100, quality)),  # 限制在 1-100
                self.cv2.IMWRITE_JPEG_OPTIMIZE, 1,
                self.cv2.IMWRITE_JPEG_PROGRESSIVE, 1,
            ]

            # 编码压缩
            success, encoded = self.cv2.imencode('.jpg', image, encode_params)

            if not success:
                logger.warning("Failed to encode compressed image")
                return image_data

            original_size = len(image_data)
            compressed_size = len(encoded)
            compression_ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0

            logger.debug(
                f"Image compressed: {original_size} -> {compressed_size} bytes "
                f"({compression_ratio:.1f}% reduction, quality={quality})"
            )

            return encoded.tobytes()

        except Exception as e:
            logger.error(f"Compression failed: {e}")
            return image_data  # 返回原始数据

    def compress_from_base64(self, image_base64: str, quality: int = 85) -> str:
        """压缩 Base64 图片"""
        import base64

        try:
            # 移除可能的 data:image/jpeg;base64, 前缀
            if ',' in image_base64:
                image_base64 = image_base64.split(',', 1)[1]

            # 解码 Base64
            image_data = base64.b64decode(image_base64)

            # 压缩
            compressed_data = self.compress(image_data, quality)

            # 重新编码为 Base64
            return base64.b64encode(compressed_data).decode('utf-8')

        except Exception as e:
            logger.error(f"Base64 compression failed: {e}")
            return image_base64


class PillowCompressor(ImageCompressor):
    """Pillow 图片压缩器"""

    def __init__(self):
        try:
            from PIL import Image
            self.Image = Image
        except ImportError:
            logger.error("Pillow not installed. Install with: pip install Pillow")
            raise ImportError("Pillow is required for PillowCompressor")

    def compress(self, image_data: bytes, quality: int = 85) -> bytes:
        """使用 Pillow 压缩 JPEG 图片"""
        try:
            # 打开图片
            img = self.Image.open(io.BytesIO(image_data))

            # 转换为 RGB（如果需要）
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')

            # 压缩参数
            output = io.BytesIO()
            img.save(
                output,
                format='JPEG',
                quality=max(1, min(100, quality)),  # 限制在 1-100
                optimize=True,
                progressive=True,
            )

            compressed_data = output.getvalue()
            output.close()

            original_size = len(image_data)
            compressed_size = len(compressed_data)
            compression_ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0

            logger.debug(
                f"Image compressed (Pillow): {original_size} -> {compressed_size} bytes "
                f"({compression_ratio:.1f}% reduction, quality={quality})"
            )

            return compressed_data

        except Exception as e:
            logger.error(f"Pillow compression failed: {e}")
            return image_data  # 返回原始数据

    def compress_from_base64(self, image_base64: str, quality: int = 85) -> str:
        """压缩 Base64 图片"""
        import base64

        try:
            # 移除可能的 data:image/jpeg;base64, 前缀
            if ',' in image_base64:
                image_base64 = image_base64.split(',', 1)[1]

            # 解码 Base64
            image_data = base64.b64decode(image_base64)

            # 压缩
            compressed_data = self.compress(image_data, quality)

            # 重新编码为 Base64
            return base64.b64encode(compressed_data).decode('utf-8')

        except Exception as e:
            logger.error(f"Pillow Base64 compression failed: {e}")
            return image_base64


class ResizeCompressor(ImageCompressor):
    """调整尺寸压缩器（可结合质量压缩）"""

    def __init__(self, compressor: Optional[ImageCompressor] = None):
        """
        初始化

        Args:
            compressor: 底层压缩器，默认使用 OpenCVCompressor
        """
        try:
            import cv2
            self.cv2 = cv2
        except ImportError:
            logger.error("OpenCV not installed")
            raise ImportError("OpenCV is required for ResizeCompressor")

        self.compressor = compressor or OpenCVCompressor()

    def compress(
        self,
        image_data: bytes,
        quality: int = 85,
        max_width: Optional[int] = None,
        max_height: Optional[int] = None,
    ) -> bytes:
        """
        调整尺寸并压缩图片

        Args:
            image_data: 原始图片数据
            quality: 压缩质量（1-100）
            max_width: 最大宽度（可选）
            max_height: 最大高度（可选）

        Returns:
            压缩后的图片数据
        """
        try:
            # 解码图片
            np_array = np.frombuffer(image_data, np.uint8)
            image = self.cv2.imdecode(np_array, self.cv2.IMREAD_COLOR)

            if image is None:
                logger.warning("Failed to decode image for resize")
                return image_data

            original_height, original_width = image.shape[:2]

            # 计算新尺寸
            new_width, new_height = original_width, original_height

            if max_width and original_width > max_width:
                scale = max_width / original_width
                new_width = max_width
                new_height = int(original_height * scale)

            if max_height and new_height > max_height:
                scale = max_height / new_height
                new_height = max_height
                new_width = int(new_width * scale)

            # 如果需要调整尺寸
            if new_width != original_width or new_height != original_height:
                image = self.cv2.resize(
                    image,
                    (new_width, new_height),
                    interpolation=self.cv2.INTER_AREA,
                )
                logger.debug(
                    f"Image resized: {original_width}x{original_height} -> "
                    f"{new_width}x{new_height}"
                )

            # 压缩参数
            encode_params = [
                self.cv2.IMWRITE_JPEG_QUALITY,
                max(1, min(100, quality)),
                self.cv2.IMWRITE_JPEG_OPTIMIZE, 1,
            ]

            # 编码压缩
            success, encoded = self.cv2.imencode('.jpg', image, encode_params)

            if not success:
                logger.warning("Failed to encode resized image")
                return image_data

            return encoded.tobytes()

        except Exception as e:
            logger.error(f"Resize compression failed: {e}")
            return image_data

    def compress_from_base64(self, image_base64: str, quality: int = 85, **kwargs) -> str:
        """压缩 Base64 图片（支持尺寸调整参数）"""
        import base64

        try:
            # 移除可能的 data:image/jpeg;base64, 前缀
            if ',' in image_base64:
                image_base64 = image_base64.split(',', 1)[1]

            # 解码 Base64
            image_data = base64.b64decode(image_base64)

            # 压缩（支持额外参数）
            compressed_data = self.compress(image_data, quality, **kwargs)

            # 重新编码为 Base64
            return base64.b64encode(compressed_data).decode('utf-8')

        except Exception as e:
            logger.error(f"Resize Base64 compression failed: {e}")
            return image_base64


class CompressorFactory:
    """压缩器工厂类"""

    _compressors: dict = {}
    _default_compressor: Optional[ImageCompressor] = None

    @classmethod
    def register(cls, name: str, compressor_class: type) -> None:
        """
        注册压缩器

        Args:
            name: 压缩器名称
            compressor_class: 压缩器类
        """
        cls._compressors[name] = compressor_class
        logger.info(f"Registered compressor: {name}")

    @classmethod
    def create(cls, name: str, **kwargs) -> ImageCompressor:
        """
        创建压缩器实例

        Args:
            name: 压缩器名称

        Returns:
            压缩器实例
        """
        default = cls.get_default()
        assert default
        if name not in cls._compressors:
            logger.warning(f"Unknown compressor: {name}, using default")
            return default

        try:
            return cls._compressors[name](**kwargs)
        except Exception as e:
            logger.error(f"Failed to create compressor {name}: {e}")
            return default

    @classmethod
    def get_default(cls) -> Optional[ImageCompressor]:
        """
        获取默认压缩器

        Returns:
            默认压缩器实例，如果不可用则返回 None
        """
        if cls._default_compressor is None:
            # 尝试使用 OpenCV（性能更好）
            try:
                cls._default_compressor = OpenCVCompressor()
                logger.info("Using OpenCVCompressor as default")
            except ImportError:
                # 回退到 Pillow
                try:
                    cls._default_compressor = PillowCompressor()
                    logger.info("Using PillowCompressor as default (OpenCV not available)")
                except ImportError:
                    logger.warning("No compressor available (OpenCV and Pillow not installed)")
                    cls._default_compressor = None

        return cls._default_compressor

    @classmethod
    def get_available_compressors(cls) -> list:
        """
        获取可用的压缩器列表

        Returns:
            压缩器名称列表
        """
        available = []

        # 检查 OpenCV
        try:
            available.append("opencv")
        except ImportError:
            pass

        # 检查 Pillow
        try:
            available.append("pillow")
        except ImportError:
            pass

        # 添加注册的压缩器
        available.extend(cls._compressors.keys())

        return available


# 注册默认压缩器
CompressorFactory.register("opencv", OpenCVCompressor)
CompressorFactory.register("pillow", PillowCompressor)
CompressorFactory.register("resize", ResizeCompressor)


class ImageCompressionConfig:
    """图片压缩配置"""

    def __init__(
        self,
        enabled: bool = True,
        compressor_type: str = "opencv",
        quality: int = 75,  # 默认质量，稍低于原始质量以减少存储
        max_width: Optional[int] = None,
        max_height: Optional[int] = None,
    ):
        """
        初始化压缩配置

        Args:
            enabled: 是否启用压缩
            compressor_type: 压缩器类型
            quality: 压缩质量（1-100）
            max_width: 最大宽度（仅 resize 压缩器有效）
            max_height: 最大高度（仅 resize 压缩器有效）
        """
        self.enabled = enabled
        self.compressor_type = compressor_type
        self.quality = max(1, min(100, quality))
        self.max_width = max_width
        self.max_height = max_height

    def get_compressor(self) -> Optional[ImageCompressor]:
        """
        获取压缩器实例

        Returns:
            压缩器实例或 None
        """
        if not self.enabled:
            return None

        if self.max_width or self.max_height:
            return CompressorFactory.create(
                "resize",
                compressor=CompressorFactory.create(self.compressor_type),
            )

        return CompressorFactory.create(self.compressor_type)
