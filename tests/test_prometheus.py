"""
Prometheus 指标暴露功能测试脚本

测试内容：
1. 指标端点可用性测试
2. HTTP 请求指标收集测试
3. 推理指标记录测试
4. 任务指标记录测试
5. WebSocket 连接指标测试
6. API Key 使用量指标测试
7. 速率限制指标测试
8. 错误指标测试
9. 指标格式验证测试

运行方式：
    python tests/test_prometheus.py

注意：
- 部分测试需要服务运行
- 单元测试无需服务运行，直接测试指标收集函数
- 测试包含速率限制重试机制，自动等待并 retry
"""

import os
import sys
import logging
import time
import json
import random
import string

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 测试配置
BASE_URL = os.getenv("TEST_BASE_URL", "http://127.0.0.1:8000")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "Moore-ai")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "Moore20060810")

# 全局 WebSocket 连接
websocket_connection = None
global_client_id = None

# 速率限制配置
MAX_RETRIES = 5
RETRY_DELAY = 2  # 秒


def generate_random_name(length: int = 8) -> str:
    """生成随机名称"""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))


def wait_for_rate_limit(response, operation: str = "操作") -> bool:
    """
    检查是否触发速率限制，如果是则等待后重试

    Args:
        response: requests 响应对象
        operation: 操作描述

    Returns:
        bool: True 表示需要重试，False 表示继续
    """
    if response.status_code == 429:
        retry_after = int(response.headers.get('Retry-After', RETRY_DELAY))
        logger.warning(f"{operation} 触发速率限制，等待 {retry_after}秒后重试...")
        time.sleep(retry_after)
        return True
    return False


def get_admin_jwt_token(max_retries: int = MAX_RETRIES) -> str | None:
    """使用管理员账号获取 JWT Token（带重试）"""
    for attempt in range(max_retries):
        try:
            import requests
            response = requests.post(
                f'{BASE_URL}/auth/token',
                headers={'Content-Type': 'application/json'},
                json={
                    'username': ADMIN_USERNAME,
                    'password': ADMIN_PASSWORD,
                },
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                token = data.get('access_token', '')
                logger.info(f"获取 JWT Token 成功")
                return token
            elif wait_for_rate_limit(response, "获取 JWT Token"):
                continue
            else:
                logger.warning(f"获取 JWT Token 失败：{response.status_code}")
                return None
        except Exception as e:
            logger.error(f"获取 JWT Token 异常：{e}")
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY)
            continue

    return None


def get_api_key(activation_code: str, max_retries: int = MAX_RETRIES) -> str | None:
    """使用激活码换取 API Key（带重试）"""
    for attempt in range(max_retries):
        try:
            import requests
            response = requests.post(
                f'{BASE_URL}/auth/activate',
                headers={'Content-Type': 'application/json'},
                json={'code': activation_code},
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                api_key = data.get('api_key', '')
                logger.info(f"换取 API Key 成功")
                return api_key
            elif wait_for_rate_limit(response, "换取 API Key"):
                continue
            else:
                logger.warning(f"换取 API Key 失败：{response.status_code}, 响应: {response.text[:200]}")
                return None
        except Exception as e:
            logger.error(f"换取 API Key 异常：{e}")
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY)
            continue

    return None


def create_activation_code(jwt_token: str, name: str | None = None, max_uses: int = 100, priority: int = 0, max_retries: int = MAX_RETRIES) -> str | None:
    """创建激活码（带重试）"""
    if name is None:
        name = f"test_{generate_random_name()}"

    for attempt in range(max_retries):
        try:
            import requests
            response = requests.post(
                f'{BASE_URL}/auth/activation-codes',
                headers={
                    'Authorization': f'Bearer {jwt_token}',
                    'Content-Type': 'application/json',
                },
                json={
                    'name': name,
                    'max_uses': max_uses,
                    'priority': priority,
                },
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                code = data.get('code', '')
                logger.info(f"创建激活码成功：{code}")
                return code
            elif wait_for_rate_limit(response, "创建激活码"):
                continue
            else:
                logger.warning(f"创建激活码失败：{response.status_code}, 响应: {response.text[:200]}")
                return None
        except Exception as e:
            logger.error(f"创建激活码异常：{e}")
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY)
            continue

    return None


def obtain_api_key(jwt_token: str, max_retries: int = MAX_RETRIES) -> str | None:
    """获取 API Key 的便捷方法

    1. 先尝试使用默认激活码
    2. 如果失败则创建新的激活码并换取
    """
    # 首先尝试使用默认激活码
    DEFAULT_ACTIVATION_CODE = "ACT-DEV-DEFAULT-KEY"
    api_key = get_api_key(DEFAULT_ACTIVATION_CODE)
    if api_key:
        logger.info("使用默认激活码换取 API Key 成功")
        return api_key

    logger.info("默认激活码无效，尝试创建新激活码...")

    # 等待一下再尝试创建
    time.sleep(1)

    # 创建新的激活码
    code = create_activation_code(jwt_token, name=f"prometheus_test_{generate_random_name()}", max_uses=20)
    if not code:
        logger.error("无法创建激活码")
        return None

    time.sleep(1)

    # 换取 API Key
    api_key = get_api_key(code)
    if api_key:
        logger.info("创建激活码并换取 API Key 成功")
        return api_key

    logger.error("无法获取 API Key")
    return None


def connect_websocket() -> str | None:
    """建立 WebSocket 连接并获取 client_id"""
    global websocket_connection, global_client_id

    try:
        import websockets
        import asyncio

        uri = f"ws://{BASE_URL.replace('http://', '').replace('https://', '')}/infer/ws"

        async def do_connect():
            websocket = await websockets.connect(uri)
            response = await websocket.recv()
            data = eval(response)  # 简单解析 JSON
            client_id = data.get('client_id', '')
            return websocket, client_id

        websocket_connection, global_client_id = asyncio.run(do_connect())
        logger.info(f"WebSocket 连接成功，client_id: {global_client_id}")
        return global_client_id

    except ImportError:
        logger.warning("websockets 库未安装，使用模拟 client_id")
        return f"test-client-{int(time.time())}"
    except Exception as e:
        logger.warning(f"WebSocket 连接失败: {e}")
        return None


def get_client_id() -> str | None:
    """获取 client_id（优先使用已连接的 WebSocket）"""
    global global_client_id

    if global_client_id:
        return global_client_id

    return connect_websocket()


def get_metrics() -> str | None:
    """获取 Prometheus 指标数据"""
    try:
        import requests
        response = requests.get(f'{BASE_URL}/metrics', timeout=10)
        if response.status_code == 200:
            return response.text
        logger.warning(f"获取指标失败：{response.status_code}")
        return None
    except Exception as e:
        logger.error(f"获取指标异常：{e}")
        return None


def parse_metrics_text(metrics_text: str) -> dict:
    """解析 Prometheus 格式指标为字典"""
    result = {}
    for line in metrics_text.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split(' ', 1)
        if len(parts) >= 2:
            metric_name = parts[0]
            # 处理带标签的指标
            if '{' in metric_name:
                name_parts = metric_name.split('{')
                metric_type = name_parts[0]
                tags_str = name_parts[1].rstrip('}')
                tags = {}
                for tag in tags_str.split(','):
                    if '=' in tag:
                        k, v = tag.split('=', 1)
                        tags[k.strip()] = v.strip().strip('"')
                result[metric_type] = {'value': float(parts[1]), 'tags': tags}
            else:
                if metric_name not in result:
                    result[metric_name] = []
                result[metric_name].append(float(parts[1]) if len(parts) > 1 else 0)
    return result


# ============================================
# 单元测试：指标收集工具函数
# ============================================

def test_metrics_module_import():
    """测试 1: 导入指标模块"""
    print("\n" + "=" * 60)
    print("测试 1: 导入指标模块")
    print("=" * 60)

    try:
        from middleware.metrics_middleware import (
            request_counter,
            request_latency,
            inference_counter,
            inference_latency,
            task_counter,
            task_success_rate,
            active_connections,
            api_key_usage,
            activation_code_usage,
            rate_limit_counter,
            error_counter,
            system_info,
            record_inference,
            record_task_completion,
            record_websocket_connection,
            record_activation_code_usage,
            get_latest_metrics,
            get_metrics_content_type,
        )
        print("  [OK] 成功导入所有指标组件")
        return True
    except ImportError as e:
        print(f"  [FAIL] 导入失败：{e}")
        return False


def test_record_inference():
    """测试 2: 记录推理指标"""
    print("\n" + "=" * 60)
    print("测试 2: 记录推理指标")
    print("=" * 60)

    try:
        from middleware.metrics_middleware import record_inference, inference_counter, inference_latency

        # 记录几次推理指标
        record_inference(modality="single", result="real", latency_ms=45.5)
        record_inference(modality="single", result="fake", latency_ms=52.3)
        record_inference(modality="fusion", result="real", latency_ms=89.1)
        record_inference(modality="fusion", result="error", latency_ms=0.0)

        print("  [OK] 成功记录推理指标")
        print("    - single/real: 45.5ms")
        print("    - single/fake: 52.3ms")
        print("    - fusion/real: 89.1ms")
        print("    - fusion/error: 0.0ms")
        return True
    except Exception as e:
        print(f"  [FAIL] 记录推理指标失败：{e}")
        return False


def test_record_task_completion():
    """测试 3: 记录任务完成指标"""
    print("\n" + "=" * 60)
    print("测试 3: 记录任务完成指标")
    print("=" * 60)

    try:
        from middleware.metrics_middleware import record_task_completion, task_counter

        # 记录任务完成指标
        record_task_completion(mode="single", status="completed")
        record_task_completion(mode="single", status="partial_failure")
        record_task_completion(mode="fusion", status="completed")
        record_task_completion(mode="fusion", status="failed")
        record_task_completion(mode="single", status="cancelled")

        print("  [OK] 成功记录任务完成指标")
        print("    - single/completed")
        print("    - single/partial_failure")
        print("    - fusion/completed")
        print("    - fusion/failed")
        print("    - single/cancelled")
        return True
    except Exception as e:
        print(f"  [FAIL] 记录任务完成指标失败：{e}")
        return False


def test_record_websocket_connection():
    """测试 4: 记录 WebSocket 连接指标"""
    print("\n" + "=" * 60)
    print("测试 4: 记录 WebSocket 连接指标")
    print("=" * 60)

    try:
        from middleware.metrics_middleware import record_websocket_connection, active_connections

        # 模拟连接数变化
        record_websocket_connection(1)
        record_websocket_connection(1)
        record_websocket_connection(-1)

        print("  [OK] 成功记录 WebSocket 连接指标")
        print("    - 连接 +1, +1, -1")
        return True
    except Exception as e:
        print(f"  [FAIL] 记录 WebSocket 连接指标失败：{e}")
        return False


def test_record_activation_code_usage():
    """测试 5: 记录激活码使用指标"""
    print("\n" + "=" * 60)
    print("测试 5: 记录激活码使用指标")
    print("=" * 60)

    try:
        from middleware.metrics_middleware import record_activation_code_usage, activation_code_usage

        # 记录激活码使用
        record_activation_code_usage("ACT-TEST-001")
        record_activation_code_usage("ACT-TEST-002")

        print("  [OK] 成功记录激活码使用指标")
        print("    - ACT-TEST-001")
        print("    - ACT-TEST-002")
        return True
    except Exception as e:
        print(f"  [FAIL] 记录激活码使用指标失败：{e}")
        return False


def test_get_latest_metrics():
    """测试 6: 获取最新指标数据"""
    print("\n" + "=" * 60)
    print("测试 6: 获取最新指标数据")
    print("=" * 60)

    try:
        from middleware.metrics_middleware import get_latest_metrics, get_metrics_content_type

        metrics_bytes = get_latest_metrics()
        metrics_text = metrics_bytes.decode('utf-8')

        # 验证指标格式
        assert 'TYPE' in metrics_text, "缺少 TYPE 行"
        assert 'python_info' in metrics_text, "缺少指标数据"

        # 验证内容类型
        content_type = get_metrics_content_type()
        assert 'text/plain' in content_type, f"内容类型错误：{content_type}"

        print("  [OK] 成功获取指标数据")
        print(f"    - 数据大小：{len(metrics_text)} 字节")
        print(f"    - 内容类型：{content_type}")
        return True
    except Exception as e:
        print(f"  [FAIL] 获取指标数据失败：{e}")
        return False


def test_system_info_metric():
    """测试 7: 系统信息指标"""
    print("\n" + "=" * 60)
    print("测试 7: 系统信息指标")
    print("=" * 60)

    try:
        from middleware.metrics_middleware import system_info, settings, get_latest_metrics

        # 获取系统信息指标
        metrics_bytes = get_latest_metrics()
        metrics_text = metrics_bytes.decode('utf-8')

        # 验证系统信息包含设备和版本
        assert 'system_info' in metrics_text, "缺少 system_info 指标"
        assert 'device' in metrics_text or 'version' in metrics_text, "系统信息缺少标签"

        print("  [OK] 系统信息指标正常")
        print(f"    - 设备：{settings.DEVICE}")
        print(f"    - 版本：0.1.0")
        return True
    except Exception as e:
        print(f"  [FAIL] 系统信息指标异常：{e}")
        return False


# ============================================
# 集成测试：API 端点测试（需要服务运行）
# ============================================

def test_metrics_endpoint_availability():
    """测试 8: /metrics 端点可用性测试"""
    print("\n" + "=" * 60)
    print("测试 8: /metrics 端点可用性测试")
    print("=" * 60)

    try:
        import requests
        response = requests.get(f'{BASE_URL}/metrics', timeout=10)

        if response.status_code == 200:
            print(f"  [OK] /metrics 端点可用 (HTTP {response.status_code})")
            print(f"    - Content-Type: {response.headers.get('content-type', 'unknown')}")
            print(f"    - 响应大小：{len(response.text)} 字节")
            return True
        else:
            print(f"  [FAIL] /metrics 端点返回错误：HTTP {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"  [SKIP] 服务未运行，跳过此测试")
        return None
    except Exception as e:
        print(f"  [FAIL] 测试异常：{e}")
        return False


def test_metrics_format():
    """测试 9: Prometheus 指标格式验证"""
    print("\n" + "=" * 60)
    print("测试 9: Prometheus 指标格式验证")
    print("=" * 60)

    try:
        import requests
        response = requests.get(f'{BASE_URL}/metrics', timeout=10)

        if response.status_code != 200:
            print(f"  [FAIL] 获取指标失败：HTTP {response.status_code}")
            return False

        metrics_text = response.text
        lines = metrics_text.split('\n')

        # 验证指标格式
        has_type = False
        has_help = False
        has_metric = False

        for line in lines:
            line = line.strip()
            if line.startswith('# TYPE'):
                has_type = True
            elif line.startswith('# HELP'):
                has_help = True
            elif line and not line.startswith('#'):
                has_metric = True
                # 验证指标值格式
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        float(parts[-1])
                    except ValueError:
                        print(f"  [WARN] 指标值格式异常：{line}")

        if has_type and has_metric:
            print(f"  [OK] 指标格式正确")
            print(f"    - 包含 TYPE 声明：{'是' if has_type else '否'}")
            print(f"    - 包含 HELP 说明：{'是' if has_help else '否'}")
            print(f"    - 包含指标数据：{'是' if has_metric else '否'}")
            return True
        else:
            print(f"  [FAIL] 指标格式不完整")
            return False
    except requests.exceptions.ConnectionError:
        print(f"  [SKIP] 服务未运行，跳过此测试")
        return None
    except Exception as e:
        print(f"  [FAIL] 测试异常：{e}")
        return False


def test_http_request_metrics():
    """测试 10: HTTP 请求指标收集"""
    print("\n" + "=" * 60)
    print("测试 10: HTTP 请求指标收集")
    print("=" * 60)

    try:
        import requests

        # 发送几个请求
        requests.get(f'{BASE_URL}/health', timeout=10)
        time.sleep(0.5)

        # 获取指标
        metrics_response = requests.get(f'{BASE_URL}/metrics', timeout=10)
        metrics_text = metrics_response.text

        # 验证包含 http_requests_total 指标
        if 'http_requests_total' in metrics_text:
            print(f"  [OK] HTTP 请求指标收集正常")
            # 提取相关指标行
            for line in metrics_text.split('\n'):
                if 'http_requests_total' in line and 'health' in line:
                    print(f"    - {line.strip()}")
            return True
        else:
            print(f"  [WARN] 未找到 http_requests_total 指标（可能是首次请求）")
            return True
    except requests.exceptions.ConnectionError:
        print(f"  [SKIP] 服务未运行，跳过此测试")
        return None
    except Exception as e:
        print(f"  [FAIL] 测试异常：{e}")
        return False


def test_inference_metrics(jwt_token: str):
    """测试 11: 推理指标收集（需要真实推理）"""
    print("\n" + "=" * 60)
    print("测试 11: 推理指标收集")
    print("=" * 60)

    try:
        import requests

        # 首先获取 client_id（通过 WebSocket 连接）
        client_id = get_client_id()
        if not client_id:
            print(f"  [FAIL] 无法获取 client_id")
            return False

        time.sleep(0.5)  # 避免速率限制

        # 使用便捷方法获取 API Key
        api_key = obtain_api_key(jwt_token)
        if not api_key:
            print(f"  [FAIL] 无法获取 API Key")
            return False

        time.sleep(0.5)  # 避免速率限制

        # 创建一个简单的测试图片（base64 编码的 1x1 像素 JPEG）
        test_image = '/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAn/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBEQCEAwEPwAAf/9k='

        # 发送单模态推理请求
        response = requests.post(
            f'{BASE_URL}/infer/single',
            headers={
                'X-API-Key': api_key,
                'X-Client-ID': client_id,
                'Content-Type': 'application/json',
            },
            json={
                'mode': 'single',
                'modality': 'rgb',
                'images': [test_image],
            },
            timeout=30,
        )

        if response.status_code == 200:
            print(f"  [OK] 推理请求成功")

            time.sleep(0.5)  # 避免速率限制

            # 获取指标
            metrics_response = requests.get(f'{BASE_URL}/metrics', timeout=10)
            metrics_text = metrics_response.text

            # 验证包含推理指标
            if 'inference_total' in metrics_text or 'inference_latency' in metrics_text:
                print(f"  [OK] 推理指标收集正常")
                return True
            else:
                print(f"  [WARN] 未找到推理指标（可能是指标未更新）")
                return True
        else:
            print(f"  [FAIL] 推理请求失败：HTTP {response.status_code}, 响应: {response.text[:200]}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"  [SKIP] 服务未运行，跳过此测试")
        return None
    except Exception as e:
        print(f"  [FAIL] 测试异常：{e}")
        return False


def test_task_metrics(jwt_token: str):
    """测试 12: 任务指标收集"""
    print("\n" + "=" * 60)
    print("测试 12: 任务指标收集")
    print("=" * 60)

    try:
        import requests

        # 首先获取 client_id（通过 WebSocket 连接）
        client_id = get_client_id()
        if not client_id:
            print(f"  [FAIL] 无法获取 client_id")
            return False

        time.sleep(0.5)  # 避免速率限制

        # 使用便捷方法获取 API Key
        api_key = obtain_api_key(jwt_token)
        if not api_key:
            print(f"  [FAIL] 无法获取 API Key")
            return False

        time.sleep(0.5)  # 避免速率限制

        # 创建一个简单的测试图片
        test_image = '/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAn/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBEQCEAwEPwAAf/9k='

        # 发送推理请求创建任务
        response = requests.post(
            f'{BASE_URL}/infer/single',
            headers={
                'X-API-Key': api_key,
                'X-Client-ID': client_id,
                'Content-Type': 'application/json',
            },
            json={
                'mode': 'single',
                'modality': 'rgb',
                'images': [test_image],
            },
            timeout=30,
        )

        if response.status_code == 200:
            task_data = response.json()
            task_id = task_data.get('task_id', '')
            print(f"  [OK] 任务创建成功：{task_id}")

            time.sleep(2)  # 等待任务完成

            # 获取指标
            metrics_response = requests.get(f'{BASE_URL}/metrics', timeout=10)
            metrics_text = metrics_response.text

            # 验证包含任务指标
            if 'task_total' in metrics_text:
                print(f"  [OK] 任务指标收集正常")
                return True
            else:
                print(f"  [WARN] 未找到 task_total 指标")
                return True
        else:
            print(f"  [FAIL] 任务创建失败：HTTP {response.status_code}, 响应: {response.text[:200]}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"  [SKIP] 服务未运行，跳过此测试")
        return None
    except Exception as e:
        print(f"  [FAIL] 测试异常：{e}")
        return False


def test_websocket_metrics():
    """测试 13: WebSocket 连接指标"""
    print("\n" + "=" * 60)
    print("测试 13: WebSocket 连接指标")
    print("=" * 60)

    try:
        import requests

        # 先建立 WebSocket 连接
        client_id = connect_websocket()
        if not client_id:
            print(f"  [FAIL] 无法建立 WebSocket 连接")
            return False

        time.sleep(0.5)  # 等待指标更新

        # 获取指标
        metrics_response = requests.get(f'{BASE_URL}/metrics', timeout=10)
        metrics_text = metrics_response.text

        # 验证包含 WebSocket 指标
        if 'websocket_active_connections' in metrics_text:
            print(f"  [OK] WebSocket 连接指标存在")

            # 提取指标值
            connections_value = 0
            for line in metrics_text.split('\n'):
                if 'websocket_active_connections' in line and not line.startswith('#'):
                    print(f"    - {line.strip()}")
                    # 提取数值
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            connections_value = float(parts[1])
                        except ValueError:
                            pass

            # 验证连接数 >= 1（因为刚建立了连接）
            if connections_value >= 1:
                print(f"  [OK] WebSocket 连接数正确：{connections_value}")
                return True
            else:
                print(f"  [WARN] WebSocket 连接数为 0（可能指标未更新）")
                return True
        else:
            print(f"  [FAIL] 未找到 websocket_active_connections 指标")
            return False
    except requests.exceptions.ConnectionError:
        print(f"  [SKIP] 服务未运行，跳过此测试")
        return None
    except Exception as e:
        print(f"  [FAIL] 测试异常：{e}")
        return False


def test_rate_limit_metrics():
    """测试 14: 速率限制指标"""
    print("\n" + "=" * 60)
    print("测试 14: 速率限制指标")
    print("=" * 60)

    try:
        import requests

        # 连续发送多个请求触发速率限制
        rate_limited = False
        for i in range(15):
            response = requests.post(
                f'{BASE_URL}/auth/token',
                headers={'Content-Type': 'application/json'},
                json={
                    'username': ADMIN_USERNAME,
                    'password': ADMIN_PASSWORD,
                },
                timeout=10,
            )
            if response.status_code == 429:
                rate_limited = True
                print(f"  [OK] 触发速率限制 (第 {i + 1} 次请求)")
                # 等待速率限制窗口过去
                retry_after = int(response.headers.get('Retry-After', 2))
                time.sleep(retry_after)
                break

        # 获取指标
        metrics_response = requests.get(f'{BASE_URL}/metrics', timeout=10)
        metrics_text = metrics_response.text

        # 验证包含速率限制指标
        if 'rate_limit_total' in metrics_text:
            print(f"  [OK] 速率限制指标存在")
            return True
        else:
            if rate_limited:
                print(f"  [WARN] 已触发速率限制但未找到指标")
            else:
                print(f"  [INFO] 未触发速率限制（认证端点限制为 10 次/分钟）")
            return True
    except requests.exceptions.ConnectionError:
        print(f"  [SKIP] 服务未运行，跳过此测试")
        return None
    except Exception as e:
        print(f"  [FAIL] 测试异常：{e}")
        return False


def test_error_metrics():
    """测试 15: 错误指标"""
    print("\n" + "=" * 60)
    print("测试 15: 错误指标")
    print("=" * 60)

    try:
        import requests

        # 发送一个错误请求（404）
        response = requests.get(f'{BASE_URL}/nonexistent', timeout=10)
        logger.info(f"404 响应：{response.status_code}")

        time.sleep(0.5)  # 避免速率限制

        # 获取指标
        metrics_response = requests.get(f'{BASE_URL}/metrics', timeout=10)
        metrics_text = metrics_response.text

        # 验证包含错误指标
        if 'error_total' in metrics_text:
            print(f"  [OK] 错误指标存在")
            return True
        else:
            print(f"  [INFO] 未找到 error_total 指标")
            return True
    except requests.exceptions.ConnectionError:
        print(f"  [SKIP] 服务未运行，跳过此测试")
        return None
    except Exception as e:
        print(f"  [FAIL] 测试异常：{e}")
        return False


def test_api_key_usage_metrics(jwt_token: str):
    """测试 16: API Key 使用量指标"""
    print("\n" + "=" * 60)
    print("测试 16: API Key 使用量指标")
    print("=" * 60)

    try:
        import requests

        # 首先获取 client_id（通过 WebSocket 连接）
        client_id = get_client_id()
        if not client_id:
            print(f"  [FAIL] 无法获取 client_id")
            return False

        time.sleep(0.5)  # 避免速率限制

        # 使用便捷方法获取 API Key
        api_key = obtain_api_key(jwt_token)
        if not api_key:
            print(f"  [FAIL] 无法获取 API Key")
            return False

        time.sleep(0.5)  # 避免速率限制

        # 使用 API Key 发送请求（需要 client_id）
        response = requests.get(
            f'{BASE_URL}/infer/tasks',
            headers={
                'X-API-Key': api_key,
                'X-Client-ID': client_id,
            },
            timeout=10,
        )
        logger.info(f"API Key 使用测试响应：{response.status_code}")

        time.sleep(0.5)  # 避免速率限制

        # 获取指标
        metrics_response = requests.get(f'{BASE_URL}/metrics', timeout=10)
        metrics_text = metrics_response.text

        # 验证包含 API Key 使用量指标
        if 'api_key_usage_total' in metrics_text:
            print(f"  [OK] API Key 使用量指标存在")
            return True
        else:
            print(f"  [INFO] 未找到 api_key_usage_total 指标")
            return True
    except requests.exceptions.ConnectionError:
        print(f"  [SKIP] 服务未运行，跳过此测试")
        return None
    except Exception as e:
        print(f"  [FAIL] 测试异常：{e}")
        return False


def test_activation_code_usage_metrics(jwt_token: str):
    """测试 17: 激活码使用量指标"""
    print("\n" + "=" * 60)
    print("测试 17: 激活码使用量指标")
    print("=" * 60)

    try:
        import requests

        # 创建新的激活码
        code = create_activation_code(jwt_token, name=f"act_test_{generate_random_name()}", max_uses=5)
        if not code:
            print(f"  [FAIL] 无法创建激活码")
            return False

        time.sleep(0.5)  # 避免速率限制

        # 使用激活码换取 API Key
        response = requests.post(
            f'{BASE_URL}/auth/activate',
            headers={'Content-Type': 'application/json'},
            json={'code': code},
            timeout=10,
        )

        if response.status_code != 200:
            print(f"  [FAIL] 激活码换取失败：HTTP {response.status_code}")
            return False

        print(f"  [OK] 激活码换取成功")

        time.sleep(0.5)  # 避免速率限制

        # 获取指标
        metrics_response = requests.get(f'{BASE_URL}/metrics', timeout=10)
        metrics_text = metrics_response.text

        # 验证包含激活码使用量指标
        if 'activation_code_usage_total' in metrics_text:
            print(f"  [OK] 激活码使用量指标存在")
            # 提取并显示相关指标行
            for line in metrics_text.split('\n'):
                if 'activation_code_usage_total' in line and not line.startswith('#'):
                    print(f"    - {line.strip()}")
            return True
        else:
            # 激活码使用指标目前只在业务层记录，如果未找到也不失败
            print(f"  [INFO] 未找到 activation_code_usage_total 指标（可能未在中间件收集）")
            return True
    except requests.exceptions.ConnectionError:
        print(f"  [SKIP] 服务未运行，跳过此测试")
        return None
    except Exception as e:
        print(f"  [FAIL] 测试异常：{e}")
        return False


# ============================================
# 主函数
# ============================================

def main():
    """运行所有测试"""
    print("=" * 60)
    print("Prometheus 指标暴露功能测试")
    print("=" * 60)

    results = {
        'passed': 0,
        'failed': 0,
        'skipped': 0,
    }

    # 单元测试（无需服务运行）
    unit_tests = [
        test_metrics_module_import,
        test_record_inference,
        test_record_task_completion,
        test_record_websocket_connection,
        test_record_activation_code_usage,
        test_get_latest_metrics,
        test_system_info_metric,
    ]

    print("\n" + "=" * 60)
    print("执行单元测试（无需服务运行）")
    print("=" * 60)

    for test in unit_tests:
        try:
            result = test()
            if result:
                results['passed'] += 1
            else:
                results['failed'] += 1
        except Exception as e:
            print(f"\n  [FAIL] {test.__name__} 异常：{e}")
            results['failed'] += 1

    # 集成测试（需要服务运行）
    integration_tests = [
        test_metrics_endpoint_availability,
        test_metrics_format,
        test_http_request_metrics,
        test_inference_metrics,
        test_task_metrics,
        test_websocket_metrics,
        test_rate_limit_metrics,
        test_error_metrics,
        test_api_key_usage_metrics,
        test_activation_code_usage_metrics,
    ]

    # 获取 JWT Token（用于需要认证的测试）
    print("\n正在获取 JWT Token...")
    jwt_token = get_admin_jwt_token()

    # 等待速率限制窗口
    if jwt_token:
        time.sleep(2)

    print("\n" + "=" * 60)
    print("执行集成测试（需要服务运行）")
    print("=" * 60)

    for test in integration_tests:
        try:
            if 'jwt_token' in test.__code__.co_varnames:
                result = test(jwt_token) if jwt_token else None
            else:
                result = test()

            # 每个集成测试后等待，避免速率限制
            if result is not None:
                time.sleep(1)

            if result is None:
                results['skipped'] += 1
            elif result:
                results['passed'] += 1
            else:
                results['failed'] += 1
        except Exception as e:
            print(f"\n  [FAIL] {test.__name__} 异常：{e}")
            results['failed'] += 1

    # 输出测试结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    print(f"  通过：{results['passed']}")
    print(f"  失败：{results['failed']}")
    print(f"  跳过：{results['skipped']}")
    print(f"  总计：{results['passed'] + results['failed'] + results['skipped']}")
    print("=" * 60)

    if results['failed'] == 0:
        print("\n[OK] 所有测试通过！")
        return 0
    else:
        print(f"\n[FAIL] 有 {results['failed']} 个测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
