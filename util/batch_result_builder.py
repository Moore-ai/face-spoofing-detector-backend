"""
批量检测结果构建工具
用于将检测的中间结果转换为最终的响应格式
"""

from datetime import datetime, timezone
from typing import Tuple

from schemas.detection import BatchDetectionResult, DetectionResultItem


def build_batch_detection_result(
    batch_results: list[Tuple[dict, int]],
) -> BatchDetectionResult:
    """
    将批量检测结果转换为响应格式

    Args:
        batch_results: 检测结果列表，每项为 (解析结果字典, 处理时间毫秒)
            解析结果字典需包含: is_living(bool), confidence(float)

    Returns:
        格式化后的批量检测结果
    """
    results: list[DetectionResultItem] = []
    real_count = 0
    fake_count = 0
    total_confidence = 0.0

    for i, (parsed, processing_time) in enumerate(batch_results):
        is_real = parsed["is_living"]
        result_str = "real" if is_real else "fake"

        item = DetectionResultItem(
            id=str(i),
            result=result_str,
            confidence=parsed["confidence"],
            timestamp=datetime.now(timezone.utc).isoformat(),
            processing_time=processing_time,
        )
        results.append(item)
        total_confidence += parsed["confidence"]

        if is_real:
            real_count += 1
        else:
            fake_count += 1

    total = len(batch_results)
    avg_confidence = total_confidence / total if total > 0 else 0.0

    return BatchDetectionResult(
        results=results,
        total_count=total,
        real_count=real_count,
        fake_count=fake_count,
        average_confidence=avg_confidence,
    )
