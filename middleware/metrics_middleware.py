"""
Prometheus 指标中间件
收集并暴露服务监控指标，包括：
- 请求量 QPS（按端点统计）
- 请求延迟（P50/P90/P99）
- 推理延迟（单张图片推理耗时）
- 任务成功率（按模态统计）
- 按激活码统计客户端使用量
"""

import time
import logging
import threading
from typing import Callable, Awaitable, Literal
from collections import deque
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from util.config import settings

logger = logging.getLogger(__name__)

# 类型别名
TaskMode = Literal["single", "fusion"]
TaskStatus = Literal["completed", "partial_failure", "failed", "cancelled"]

# 滑动窗口大小（用于计算成功率）
SUCCESS_RATE_WINDOW_SIZE = 100

# 线程安全的成功率计算器
class SuccessRateCalculator:
    """计算任务成功率的滑动窗口实现"""

    def __init__(self, window_size: int = SUCCESS_RATE_WINDOW_SIZE):
        self._window_size = window_size
        self._lock = threading.Lock()
        # 按模式分开统计
        self._windows: dict[str, deque] = {
            "single": deque(maxlen=window_size),
            "fusion": deque(maxlen=window_size),
        }

    def record(self, mode: str, success: bool) -> None:
        """记录任务结果

        Args:
            mode: 任务模式 ("single" 或 "fusion")
            success: 是否成功
        """
        with self._lock:
            if mode in self._windows:
                self._windows[mode].append(1 if success else 0)

    def get_rate(self, mode: str) -> float:
        """获取成功率

        Args:
            mode: 任务模式

        Returns:
            成功率百分比 (0-100)
        """
        with self._lock:
            window = self._windows.get(mode)
            if not window or len(window) == 0:
                return 0.0
            return (sum(window) / len(window)) * 100


# 全局成功率计算器
_success_rate_calculator = SuccessRateCalculator()

# ============================================
# Prometheus 指标定义
# ============================================

# 请求计数器
# 标签：method (HTTP 方法), endpoint (端点路径), status (状态码)
request_counter = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

# 请求延迟直方图（秒）
# 标签：method (HTTP 方法), endpoint (端点路径)
# 桶：5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s, 10s, +Inf
request_latency = Histogram(
    "http_request_latency_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float("inf")),
)

# 推理延迟直方图（毫秒）
# 标签：modality (single/fusion), result (real/fake/error)
inference_latency = Histogram(
    "inference_latency_milliseconds",
    "Inference latency in milliseconds",
    ["modality", "result"],
    buckets=(5, 10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, float("inf")),
)

# 推理计数器
# 标签：modality (single/fusion), result (real/fake/error)
inference_counter = Counter(
    "inference_total",
    "Total inference requests",
    ["modality", "result"],
)

# 任务计数器
# 标签：mode (single/fusion), status (completed/partial_failure/failed/cancelled)
task_counter = Counter(
    "task_total",
    "Total tasks processed",
    ["mode", "status"],
)

# 任务成功率仪表盘
task_success_rate = Gauge(
    "task_success_rate",
    "Task success rate (percentage)",
    ["mode"],
)

# 任务耗时直方图（毫秒）
# 标签：mode (single/fusion), status (completed/failed/cancelled)
task_duration = Histogram(
    "task_duration_milliseconds",
    "Task execution duration in milliseconds",
    ["mode", "status"],
    buckets=(100, 250, 500, 750, 1000, 1500, 2000, 3000, 5000, 7500, 10000, float("inf")),
)

# 活跃连接数（WebSocket）
active_connections = Gauge(
    "websocket_active_connections",
    "Number of active WebSocket connections",
)

# API Key 使用量计数器
# 标签：api_key_hash (API Key 哈希，前 8 位)
api_key_usage = Counter(
    "api_key_usage_total",
    "Total API key usage",
    ["api_key_hash"],
)

# 激活码使用量计数器
# 标签：activation_code_id (激活码 ID)
activation_code_usage = Counter(
    "activation_code_usage_total",
    "Total activation code usage",
    ["activation_code_id"],
)

# 速率限制触发计数器
# 标签：endpoint (端点路径), limit_type (ip/api_key)
rate_limit_counter = Counter(
    "rate_limit_total",
    "Total rate limit triggered",
    ["endpoint", "limit_type"],
)

# 错误计数器
# 标签：endpoint (端点路径), error_type (错误类型)
error_counter = Counter(
    "error_total",
    "Total errors",
    ["endpoint", "error_type"],
)

# 系统信息仪表盘
system_info = Gauge(
    "system_info",
    "System information",
    ["version", "device"],
)

# 设置系统信息
system_info.labels(version="0.1.0", device=settings.DEVICE).set(1)


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Prometheus 指标收集中间件

    自动收集以下指标：
    - HTTP 请求总数（按端点、方法、状态码分类）
    - HTTP 请求延迟（P50/P90/P99）
    - API Key 使用量
    - 速率限制触发次数
    - 错误数
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """处理每个请求，收集指标"""
        # 记录开始时间
        start_time = time.time()

        # 获取请求信息
        method = request.method
        endpoint = self._normalize_endpoint(request.url.path)

        # 获取 API Key（如果有）
        api_key = request.headers.get("x-api-key", "")
        api_key_hash = self._hash_api_key(api_key) if api_key else "anonymous"

        try:
            # 执行请求
            response = await call_next(request)

            # 记录请求延迟
            duration = time.time() - start_time
            request_latency.labels(method=method, endpoint=endpoint).observe(duration)

            # 记录请求计数
            request_counter.labels(method=method, endpoint=endpoint, status=response.status_code).inc()

            # 记录 API Key 使用量
            if api_key:
                api_key_usage.labels(api_key_hash=api_key_hash).inc()

            # 记录错误
            if response.status_code >= 500:
                error_counter.labels(endpoint=endpoint, error_type="server_error").inc()
            elif response.status_code >= 400:
                error_counter.labels(endpoint=endpoint, error_type="client_error").inc()

            # 记录速率限制
            if response.status_code == 429:
                limit_type = "api_key" if api_key else "ip"
                rate_limit_counter.labels(endpoint=endpoint, limit_type=limit_type).inc()

            return response

        except Exception as e:
            # 记录异常
            duration = time.time() - start_time
            request_latency.labels(method=method, endpoint=endpoint).observe(duration)
            request_counter.labels(method=method, endpoint=endpoint, status=500).inc()
            error_counter.labels(endpoint=endpoint, error_type="exception").inc()

            if api_key:
                api_key_usage.labels(api_key_hash=api_key_hash).inc()

            logger.error(f"Error processing request: {e}")
            raise

    def _normalize_endpoint(self, path: str) -> str:
        """
        标准化端点路径，避免高基数标签问题

        例如：
        - /infer/task/123 -> /infer/task/{task_id}
        - /history/task/456 -> /history/task/{task_id}
        - /storage/images/abc -> /storage/images/{image_id}
        """
        # 任务 ID 路径
        if path.startswith("/infer/task/"):
            return "/infer/task/{task_id}"
        if path.startswith("/history/task/"):
            return "/history/task/{task_id}"
        if path.startswith("/storage/images/"):
            return "/storage/images/{image_id}"
        if path.startswith("/auth/activation-codes/"):
            return "/auth/activation-codes/{code}"

        return path

    def _hash_api_key(self, api_key: str) -> str:
        """
        哈希 API Key 用于指标标签

        返回前 8 位字符作为标识
        """
        import hashlib
        if not api_key:
            return "anonymous"
        # 使用 SHA256 哈希的前 8 位
        hash_obj = hashlib.sha256(api_key.encode())
        return hash_obj.hexdigest()[:8]


# ============================================
# 推理指标记录工具函数
# ============================================

def record_inference(
    modality: str,
    result: str,
    latency_ms: float,
) -> None:
    """
    记录单次推理指标

    Args:
        modality: 模态类型 ("single" 或 "fusion")
        result: 推理结果 ("real", "fake", "error")
        latency_ms: 推理延迟（毫秒）
    """
    inference_latency.labels(modality=modality, result=result).observe(latency_ms)
    inference_counter.labels(modality=modality, result=result).inc()


# 有效的模式列表
VALID_TASK_MODES = ("single", "fusion")
# 有效的状态列表
VALID_TASK_STATUSES = ("completed", "partial_failure", "failed", "cancelled")


def record_task_completion(
    mode: TaskMode | str,
    status: TaskStatus | str,
    duration_ms: int | None = None,
    task_id: str | None = None,
) -> None:
    """记录任务完成指标（生产级实现）

    Args:
        mode: 任务模式 ("single" 或 "fusion")
        status: 任务状态 ("completed", "partial_failure", "failed", "cancelled")
        duration_ms: 任务执行耗时（毫秒），可选
        task_id: 任务 ID，可选，用于日志追踪

    Raises:
        ValueError: 当 mode 或 status 无效时
    """
    # 参数验证（支持 str 类型输入）
    if mode not in VALID_TASK_MODES:
        raise ValueError(f"Invalid task mode: {mode}. Must be one of {VALID_TASK_MODES}")
    if status not in VALID_TASK_STATUSES:
        raise ValueError(f"Invalid task status: {status}. Must be one of {VALID_TASK_STATUSES}")

    # 类型 narrowing：确保 status 为 TaskStatus 类型
    _validated_status: TaskStatus = status  # type: ignore[attr-defined]

    try:
        # 1. 记录任务计数
        task_counter.labels(mode=mode, status=_validated_status).inc()

        # 2. 记录任务耗时（如果提供）
        if duration_ms is not None and duration_ms >= 0:
            task_duration.labels(mode=mode, status=_validated_status).observe(duration_ms)

        # 3. 更新成功率（使用滑动窗口计算）
        # 成功：completed 视为成功，partial_failure 视为部分成功
        # 失败：failed, cancelled 视为失败
        if status in ("completed", "partial_failure"):
            _success_rate_calculator.record(mode, True)
        else:
            _success_rate_calculator.record(mode, False)

        # 更新成功率 Gauge
        success_rate = _success_rate_calculator.get_rate(mode)
        task_success_rate.labels(mode=mode).set(success_rate)

        # 4. 记录详细日志
        log_msg = f"Task completed: mode={mode}, status={status}"
        if task_id:
            log_msg += f", task_id={task_id}"
        if duration_ms is not None:
            log_msg += f", duration_ms={duration_ms}"
        log_msg += f", success_rate={success_rate:.1f}%"
        logger.info(log_msg)

    except Exception as e:
        # 确保指标记录失败不影响主流程
        logger.error(f"Failed to record task metrics: {e}")
        raise


def record_websocket_connection(delta: int) -> None:
    """
    记录 WebSocket 连接数变化

    Args:
        delta: 连接数变化 (+1 或 -1)
    """
    active_connections.inc(delta)


def record_activation_code_usage(code_id: str) -> None:
    """
    记录激活码使用量

    Args:
        code_id: 激活码 ID
    """
    activation_code_usage.labels(activation_code_id=code_id).inc()


# ============================================
# 指标查询工具函数
# ============================================

def get_latest_metrics() -> bytes:
    """
    获取最新 Prometheus 格式的指标数据

    Returns:
        Prometheus 格式的指标文本（字节）
    """
    return generate_latest()


def get_metrics_content_type() -> str:
    """
    获取 Prometheus 指标的内容类型

    Returns:
        Content-Type 字符串
    """
    return CONTENT_TYPE_LATEST
