"""
推理结果解析工具
用于解析模型输出的 logits，转换为概率和预测类别
"""

import numpy as np
from typing import Tuple
import time

from schemas.detection import DetectionResultItem


def _parse_prediction_result(
    logits: np.ndarray,
) -> Tuple[int, float, np.ndarray]:
    """
    解析模型预测结果

    Args:
        logits: 模型原始输出，shape: (1, num_classes) 或 (num_classes,)

    Returns:
        Tuple[类别索引, 置信度, 概率分布]
        - 类别索引: 预测的类别 (0=非活体, 1=活体)
        - 置信度: 预测类别的概率值
        - 概率分布: 所有类别的 softmax 概率
    """
    # 确保是一维数组
    if logits.ndim > 1:
        logits = logits.flatten()

    # 计算 softmax 概率
    exp_logits = np.exp(logits - np.max(logits))  # 数值稳定性处理
    probabilities = exp_logits / np.sum(exp_logits)

    # 获取预测类别和置信度
    predicted_class = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_class])

    return predicted_class, confidence, probabilities


def is_living(
    logits: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[bool, float]:
    """
    判断是否为活体

    Args:
        logits: 模型原始输出
        threshold: 置信度判定阈值，默认0.5

    Returns:
        Tuple[是否为活体, 置信度]
    """
    predicted_class, confidence, _ = _parse_prediction_result(logits)
    is_live = predicted_class == 1 and confidence >= threshold
    return is_live, confidence


def parse_single_prediction(
    logits: np.ndarray,
    start_time: float,
    threshold: float = 0.5,
) -> DetectionResultItem:
    predicted_class, confidence, probabilities = _parse_prediction_result(logits)
    live_flag = predicted_class == 1 and confidence >= threshold

    processing_time = int((time.time() - start_time) * 1000)

    return DetectionResultItem(
        mode="single",
        result="real" if live_flag else "fake",
        confidence=confidence,
        probabilities=probabilities,
        processing_time=processing_time,
    )


def parse_fusion_prediction(
    logits: np.ndarray,
    start_time: float,
    threshold: float = 0.5,
) -> DetectionResultItem:
    result = parse_single_prediction(logits, start_time, threshold)
    result.mode = "fusion"
    return result
