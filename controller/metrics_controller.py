"""
Prometheus 指标控制器
暴露/metrics 端点供 Prometheus 抓取
"""

from fastapi import APIRouter, Response
from middleware.metrics_middleware import get_latest_metrics, get_metrics_content_type

router = APIRouter()


@router.get("/metrics")
async def metrics():
    """
    Prometheus 指标暴露端点

    返回 Prometheus 文本格式的指标数据，包括：
    - http_requests_total: HTTP 请求总数（按方法、端点、状态码分类）
    - http_request_latency_seconds: HTTP 请求延迟直方图
    - inference_total: 推理总数（按模态、结果分类）
    - inference_latency_milliseconds: 推理延迟直方图
    - task_total: 任务总数（按模式、状态分类）
    - task_success_rate: 任务成功率
    - websocket_active_connections: 活跃 WebSocket 连接数
    - api_key_usage_total: API Key 使用量
    - activation_code_usage_total: 激活码使用量
    - rate_limit_total: 速率限制触发次数
    - error_total: 错误总数
    - system_info: 系统信息

    Returns:
        Prometheus 文本格式的指标数据
    """
    metrics_data = get_latest_metrics()
    return Response(
        content=metrics_data,
        media_type=get_metrics_content_type(),
    )
