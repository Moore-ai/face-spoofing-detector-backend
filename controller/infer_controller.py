from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


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


@router.post("/single", response_model=BatchDetectionResult)
async def detect_single_mode(request: SingleModeRequest):
    # 这里实现推理逻辑
    # 目前返回一个示例响应
    dummy_item = DetectionResultItem(
        id="0",
        result="real",
        confidence=0.95,
        timestamp="2026-02-12T10:30:00Z",
        processing_time=45,
    )
    response = BatchDetectionResult(
        results=[dummy_item],
        total_count=1,
        real_count=1,
        fake_count=0,
        average_confidence=0.95,
    )
    return response


@router.post("/fusion", response_model=BatchDetectionResult)
async def detect_fusion_mode(request: FusionModeRequest):
    # 这里实现融合模式推理逻辑
    # 目前返回一个示例响应
    dummy_item = DetectionResultItem(
        id="0",
        result="fake",
        confidence=0.88,
        timestamp="2026-02-12T10:30:00Z",
        processing_time=52,
    )
    response = BatchDetectionResult(
        results=[dummy_item],
        total_count=1,
        real_count=0,
        fake_count=1,
        average_confidence=0.88,
    )
    return response


@router.post("/batch", response_model=BatchDetectionResult)
async def batch_detect(request: BatchDetectRequest):
    # 这里实现批量检测推理逻辑
    # 目前返回一个示例响应
    total = len(request.image_paths)
    results: list[DetectionResultItem] = []
    real_count = 0
    fake_count = 0

    for i, _ in enumerate(request.image_paths):
        is_real = i % 2 == 0
        result = DetectionResultItem(
            id=str(i),
            result="real" if is_real else "fake",
            confidence=0.92 if is_real else 0.87,
            timestamp="2026-02-12T10:30:00Z",
            processing_time=40 + i * 2,
        )
        results.append(result)
        if is_real:
            real_count += 1
        else:
            fake_count += 1

    avg_confidence = (
        sum(r.confidence for r in results) / len(results) if results else 0.0
    )

    response = BatchDetectionResult(
        results=results,
        total_count=total,
        real_count=real_count,
        fake_count=fake_count,
        average_confidence=avg_confidence,
    )
    return response
