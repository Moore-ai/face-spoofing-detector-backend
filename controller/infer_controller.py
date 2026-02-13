import base64
import io

import numpy as np
from fastapi import APIRouter
from PIL import Image
from pydantic import BaseModel

from lifespan import InferServiceDep
from schemas.detection import BatchDetectionResult
from util.batch_result_builder import build_batch_detection_result

router = APIRouter()


def decode_base64_image(base64_str: str) -> np.ndarray:
    """将 base64 字符串解码为 numpy 图像数组"""
    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data))
    return np.array(img)


class SingleModeRequest(BaseModel):
    mode: str
    modality: str
    images: list[str]


class ImagePair(BaseModel):
    rgb: str
    ir: str


class FusionModeRequest(BaseModel):
    mode: str
    pairs: list[ImagePair]


class BatchDetectRequest(BaseModel):
    image_paths: list[str]
    mode: str


@router.post("/single", response_model=BatchDetectionResult)
async def detect_single_mode(
    request: SingleModeRequest,
    service: InferServiceDep,
):
    """单模态推理端点"""
    # 解码所有图像
    images = [decode_base64_image(img_base64) for img_base64 in request.images]

    # 调用服务层进行批量检测
    batch_results = service.detect_single_batch(images)

    # 构建并返回结果
    return build_batch_detection_result(batch_results)


@router.post("/fusion", response_model=BatchDetectionResult)
async def detect_fusion_mode(
    request: FusionModeRequest,
    service: InferServiceDep,
):
    """融合模态推理端点"""
    # 解码所有图像对
    image_pairs = [
        (decode_base64_image(pair.rgb), decode_base64_image(pair.ir))
        for pair in request.pairs
    ]

    # 调用服务层进行批量检测
    batch_results = service.detect_fusion_batch(image_pairs)

    # 构建并返回结果
    return build_batch_detection_result(batch_results)
