"""
WebSocket 客户端测试脚本
用于测试实时进度推送功能

使用示例：
    python test_websocket_client.py                    # 使用默认配置
    python test_websocket_client.py --url http://localhost:8000  # 指定URL
    python test_websocket_client.py --skip-fusion      # 跳过融合模态测试
    python test_websocket_client.py --help             # 显示帮助信息
"""

import asyncio
import argparse
import json
import sys
import time

try:
    import websockets
    import requests
except ImportError as e:
    print(f"错误: 缺少必要的依赖包 - {e}")
    print("请安装所需依赖:")
    print("  pip install requests websockets")
    print("或使用 uv:")
    print("  uv pip install requests websockets")
    sys.exit(1)


# 默认配置
DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_WS_URL = "ws://localhost:8000"

# 全局配置
BASE_URL = DEFAULT_BASE_URL
WS_URL = DEFAULT_WS_URL
TEST_CONFIG = {}


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="WebSocket 客户端测试脚本")
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_BASE_URL,
        help=f"服务器URL (默认: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--skip-fusion",
        action="store_true",
        help="跳过融合模态测试",
    )
    parser.add_argument(
        "--skip-single",
        action="store_true",
        help="跳过单模态测试",
    )
    parser.add_argument(
        "--skip-websocket",
        action="store_true",
        help="跳过WebSocket基本连接测试",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="跳过错误场景测试",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="任务等待超时时间（秒） (默认: 10.0)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细输出",
    )
    return parser.parse_args()


def check_service_health() -> bool:
    """检查服务是否健康"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"服务健康检查: {health_data}")
            return True
        else:
            print(f"服务健康检查失败: 状态码 {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"无法连接到服务 {BASE_URL}: {e}")
        print("请确保服务已启动并运行在正确的地址和端口")
        return False


def setup_test_config(args: argparse.Namespace) -> None:
    """根据命令行参数设置测试配置"""
    global BASE_URL, WS_URL, TEST_CONFIG

    BASE_URL = args.url
    # 从HTTP URL推导WebSocket URL
    if BASE_URL.startswith("http://"):
        WS_URL = BASE_URL.replace("http://", "ws://")
    elif BASE_URL.startswith("https://"):
        WS_URL = BASE_URL.replace("https://", "wss://")
    else:
        WS_URL = f"ws://{BASE_URL.split('://')[-1]}" if "://" in BASE_URL else f"ws://{BASE_URL}"

    TEST_CONFIG.update({
        "skip_fusion": args.skip_fusion,
        "skip_single": args.skip_single,
        "skip_websocket": args.skip_websocket,
        "skip_errors": args.skip_errors,
        "timeout": args.timeout,
        "verbose": args.verbose,
    })

    print(f"测试配置:")
    print(f"  - 服务器 URL: {BASE_URL}")
    print(f"  - WebSocket URL: {WS_URL}")
    print(f"  - 超时时间: {args.timeout}秒")
    print(f"  - 跳过融合模态: {args.skip_fusion}")
    print(f"  - 跳过单模态: {args.skip_single}")
    print(f"  - 跳过WebSocket测试: {args.skip_websocket}")
    print(f"  - 跳过错误场景: {args.skip_errors}")
    print()


class TestResult:
    """测试结果统计"""
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.errors = []

    def add_passed(self, test_name: str):
        self.total += 1
        self.passed += 1
        if TEST_CONFIG.get("verbose"):
            print(f"✓ {test_name}: 通过")

    def add_failed(self, test_name: str, error: str):
        self.total += 1
        self.failed += 1
        self.errors.append(f"{test_name}: {error}")
        print(f"✗ {test_name}: 失败 - {error}")

    def add_skipped(self, test_name: str, reason: str):
        self.total += 1
        self.skipped += 1
        print(f"⏭ {test_name}: 跳过 - {reason}")

    def print_summary(self):
        print("\n" + "=" * 50)
        print("测试结果摘要")
        print("=" * 50)
        print(f"总计: {self.total}")
        print(f"通过: {self.passed}")
        print(f"失败: {self.failed}")
        print(f"跳过: {self.skipped}")

        if self.failed > 0:
            print(f"\n失败详情:")
            for error in self.errors:
                print(f"  - {error}")

        success_rate = (self.passed / self.total * 100) if self.total > 0 else 0
        print(f"\n成功率: {success_rate:.1f}%")

        return self.failed == 0


async def test_websocket_progress():
    """测试 WebSocket 实时进度推送"""
    ws_endpoint = f"{WS_URL}/infer/ws"

    print(f"连接到 WebSocket: {ws_endpoint}")

    async with websockets.connect(ws_endpoint) as websocket:
        # 1. 接收连接确认和 client_id
        print("\n1. 等待连接确认...")
        connect_response = await websocket.recv()
        connect_data = json.loads(connect_response)
        print(f"连接响应: {connect_data}")

        if connect_data.get("type") != "connected" or "client_id" not in connect_data:
            raise Exception("未收到有效的连接确认")

        client_id = connect_data["client_id"]
        print(f"获取到 client_id: {client_id}")

        # 2. 测试无 WebSocket 连接的请求（应该失败）
        print("\n2. 测试无 WebSocket 连接的请求...")
        response = requests.post(
            f"{BASE_URL}/infer/single",
            json={
                "mode": "single",
                "modality": "rgb",
                "images": [],  # 空列表会立即返回错误
            },
            headers={"X-Client-ID": "invalid_client_id"}
        )
        print(f"无效 client_id 响应: 状态码={response.status_code}, 响应={response.text}")

        # 3. 发送空的检测请求（应该立即返回）
        print("\n3. 发送空图像检测请求...")
        response = requests.post(
            f"{BASE_URL}/infer/single",
            json={
                "mode": "single",
                "modality": "rgb",
                "images": [],  # 空列表会立即返回错误
            },
            headers={"X-Client-ID": client_id}
        )
        print(f"空图像请求响应: 状态码={response.status_code}")
        if response.status_code == 200:
            print(f"响应数据: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        else:
            print(f"错误: {response.text}")

        # 4. 等待可能的进度消息
        print("\n4. 等待 3 秒接收可能的进度消息...")
        try:
            for i in range(3):
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    print(f"收到消息[{i+1}]: {message}")
                except asyncio.TimeoutError:
                    print(f"等待 {i+1} 秒: 无消息")
        except Exception as e:
            print(f"接收消息异常: {e}")

        print("\nWebSocket 测试完成")


async def test_task_status_polling():
    """测试任务状态轮询"""
    print("\n测试任务状态轮询...")

    # 测试一个不存在的任务
    task_id = "non-existent-task"
    response = requests.get(f"{BASE_URL}/infer/task/{task_id}")

    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")


async def test_real_detection():
    """测试实际的图像检测流程"""
    print("\n测试实际的图像检测流程...")

    # 1. 建立 WebSocket 连接获取 client_id
    ws_endpoint = f"{WS_URL}/infer/ws"

    async with websockets.connect(ws_endpoint) as websocket:
        # 接收连接确认
        connect_response = await websocket.recv()
        connect_data = json.loads(connect_response)

        if connect_data.get("type") != "connected" or "client_id" not in connect_data:
            raise Exception("未收到有效的连接确认")

        client_id = connect_data["client_id"]
        print(f"获取到 client_id: {client_id}")

        # 2. 准备测试图像 (1x1 白色像素 PNG)
        # 创建一个最小的 PNG 图像 base64
        test_image_base64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        )

        # 3. 发送单模态检测请求
        print("\n发送单模态检测请求...")
        response = requests.post(
            f"{BASE_URL}/infer/single",
            json={
                "mode": "single",
                "modality": "rgb",
                "images": [test_image_base64],  # 单个测试图像
            },
            headers={"X-Client-ID": client_id}
        )

        print(f"检测请求响应: 状态码={response.status_code}")
        if response.status_code != 200:
            raise Exception(f"检测请求失败: 状态码={response.status_code}, 响应={response.text}")

        task_data = response.json()
        task_id = task_data.get("task_id")
        if not task_id:
            raise Exception("未收到有效的 task_id")

        print(f"任务创建成功, task_id: {task_id}")

        # 4. 接收进度更新
        print("\n等待进度更新...")
        progress_received = False
        start_time = time.time()

        timeout = TEST_CONFIG.get("timeout", 10.0)
        while time.time() - start_time < timeout:  # 最多等待指定秒数
            try:
                # 设置较短的超时，以便循环检查
                message = await asyncio.wait_for(websocket.recv(), timeout=0.5)
                message_data = json.loads(message)

                if message_data.get("type") == "progress_update":
                    progress_data = message_data.get("data", {})
                    print(f"进度更新: {json.dumps(progress_data, indent=2, ensure_ascii=False)}")
                    progress_received = True

                    # 如果任务完成，跳出循环
                    if progress_data.get("status") == "completed":
                        print("任务完成!")
                        break

                elif message_data.get("type") == "task_completed":
                    print(f"任务完成通知: {message_data.get('data', {})}")
                    break

                elif message_data.get("type") == "task_failed":
                    print(f"任务失败通知: {message_data.get('data', {})}")
                    break

            except asyncio.TimeoutError:
                # 继续等待
                continue
            except Exception as e:
                print(f"接收消息异常: {e}")
                break

        if not progress_received:
            print("警告: 未收到进度更新消息")

        # 5. 查询任务状态
        print("\n查询任务状态...")
        response = requests.get(f"{BASE_URL}/infer/task/{task_id}")
        print(f"任务状态查询: 状态码={response.status_code}")
        if response.status_code == 200:
            print(f"任务状态: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        else:
            print(f"错误: {response.text}")

        print("\n实际检测测试完成")


async def test_fusion_detection():
    """测试融合模态检测流程"""
    print("\n测试融合模态检测流程...")

    # 1. 建立 WebSocket 连接获取 client_id
    ws_endpoint = f"{WS_URL}/infer/ws"

    async with websockets.connect(ws_endpoint) as websocket:
        # 接收连接确认
        connect_response = await websocket.recv()
        connect_data = json.loads(connect_response)

        if connect_data.get("type") != "connected" or "client_id" not in connect_data:
            raise Exception("未收到有效的连接确认")

        client_id = connect_data["client_id"]
        print(f"获取到 client_id: {client_id}")

        # 2. 准备测试图像对 (相同的 1x1 白色像素 PNG)
        test_image_base64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        )

        # 3. 发送融合模态检测请求
        print("\n发送融合模态检测请求...")
        response = requests.post(
            f"{BASE_URL}/infer/fusion",
            json={
                "mode": "fusion",
                "pairs": [
                    {
                        "rgb": test_image_base64,
                        "ir": test_image_base64
                    }
                ],
            },
            headers={"X-Client-ID": client_id}
        )

        print(f"融合检测请求响应: 状态码={response.status_code}")
        if response.status_code != 200:
            raise Exception(f"检测请求失败: 状态码={response.status_code}, 响应={response.text}")

        task_data = response.json()
        task_id = task_data.get("task_id")
        if not task_id:
            raise Exception("未收到有效的 task_id")

        print(f"融合任务创建成功, task_id: {task_id}")

        # 4. 接收进度更新
        print("\n等待进度更新...")
        progress_received = False
        start_time = time.time()

        timeout = TEST_CONFIG.get("timeout", 10.0)
        while time.time() - start_time < timeout:  # 最多等待指定秒数
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=0.5)
                message_data = json.loads(message)

                if message_data.get("type") == "progress_update":
                    progress_data = message_data.get("data", {})
                    print(f"进度更新: {json.dumps(progress_data, indent=2, ensure_ascii=False)}")
                    progress_received = True

                    if progress_data.get("status") == "completed":
                        print("融合任务完成!")
                        break

                elif message_data.get("type") == "task_completed":
                    print(f"融合任务完成通知: {message_data.get('data', {})}")
                    break

                elif message_data.get("type") == "task_failed":
                    print(f"融合任务失败通知: {message_data.get('data', {})}")
                    break

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"接收消息异常: {e}")
                break

        if not progress_received:
            print("警告: 未收到进度更新消息")

        # 5. 查询任务状态
        print("\n查询融合任务状态...")
        response = requests.get(f"{BASE_URL}/infer/task/{task_id}")
        print(f"融合任务状态查询: 状态码={response.status_code}")
        if response.status_code == 200:
            print(f"融合任务状态: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        else:
            print(f"错误: {response.text}")

        print("\n融合检测测试完成")


async def test_error_scenarios():
    """测试错误场景"""
    print("\n测试错误场景...")

    # 1. 测试无效的 client_id
    print("\n1. 测试无效的 client_id...")
    response = requests.post(
        f"{BASE_URL}/infer/single",
        json={
            "mode": "single",
            "modality": "rgb",
            "images": [],
        },
        headers={"X-Client-ID": "invalid_client_id_123"}
    )
    print(f"无效 client_id 响应: 状态码={response.status_code}, 响应={response.text}")

    # 2. 测试无效的 base64 数据
    print("\n2. 测试无效的 base64 数据...")
    # 先获取有效的 client_id
    ws_endpoint = f"{WS_URL}/infer/ws"
    try:
        async with websockets.connect(ws_endpoint) as websocket:
            connect_response = await websocket.recv()
            connect_data = json.loads(connect_response)
            client_id = connect_data.get("client_id") if connect_data.get("type") == "connected" else None

            if client_id:
                response = requests.post(
                    f"{BASE_URL}/infer/single",
                    json={
                        "mode": "single",
                        "modality": "rgb",
                        "images": ["invalid_base64_data!!!"],
                    },
                    headers={"X-Client-ID": client_id}
                )
                print(f"无效 base64 响应: 状态码={response.status_code}, 响应={response.text}")

                # 3. 测试无效的 modality
                print("\n3. 测试无效的 modality...")
                response = requests.post(
                    f"{BASE_URL}/infer/single",
                    json={
                        "mode": "single",
                        "modality": "invalid_modality",  # 无效的模态
                        "images": [],
                    },
                    headers={"X-Client-ID": client_id}
                )
                print(f"无效 modality 响应: 状态码={response.status_code}, 响应={response.text}")

    except Exception as e:
        print(f"连接 WebSocket 失败: {e}")

    # 4. 测试不存在的任务查询
    print("\n4. 测试不存在的任务查询...")
    response = requests.get(f"{BASE_URL}/infer/task/nonexistent-task-123456")
    print(f"不存在的任务查询: 状态码={response.status_code}, 响应={response.text}")

    print("\n错误场景测试完成")


def test_endpoints():
    """测试 HTTP 端点"""
    print("\n测试 API 端点...")

    # 健康检查
    print("\n1. 健康检查...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"健康检查: {response.json()}")

    # API 文档
    print("\n2. API 文档...")
    print(f"Swagger UI: {BASE_URL}/docs")
    print(f"ReDoc: {BASE_URL}/redoc")


async def run_test_with_result(test_func, test_name: str, test_result: TestResult, skip_check: bool = False) -> None:
    """运行测试并记录结果"""
    if skip_check:
        test_result.add_skipped(test_name, "根据配置跳过")
        return

    try:
        await test_func()
        test_result.add_passed(test_name)
    except Exception as e:
        error_msg = str(e)
        if TEST_CONFIG.get("verbose"):
            import traceback
            error_msg = f"{error_msg}\n{traceback.format_exc()}"
        test_result.add_failed(test_name, error_msg)


async def run_all_tests(test_result: TestResult) -> None:
    """运行所有测试"""
    print("=" * 50)
    print("WebSocket 进度推送测试")
    print("=" * 50)

    # 1. 服务健康检查
    print("\n[1/6] 检查服务健康状态...")
    if not check_service_health():
        print("错误: 服务不健康，停止测试")
        return

    # 2. 测试 HTTP 端点
    print("\n[2/6] 测试 HTTP 端点...")
    try:
        test_endpoints()
        test_result.add_passed("HTTP端点测试")
    except Exception as e:
        test_result.add_failed("HTTP端点测试", str(e))

    # 3. 测试任务状态轮询
    print("\n[3/6] 测试任务状态轮询...")
    try:
        await test_task_status_polling()
        test_result.add_passed("任务状态轮询测试")
    except Exception as e:
        test_result.add_failed("任务状态轮询测试", str(e))

    # 4. 测试基本 WebSocket 连接
    print("\n[4/6] 测试基本 WebSocket 连接...")
    await run_test_with_result(
        test_websocket_progress,
        "基本WebSocket连接测试",
        test_result,
        TEST_CONFIG.get("skip_websocket", False)
    )

    # 5. 测试单模态检测流程
    print("\n[5/6] 测试单模态检测流程...")
    await run_test_with_result(
        test_real_detection,
        "单模态检测测试",
        test_result,
        TEST_CONFIG.get("skip_single", False)
    )

    # 6. 测试融合模态检测流程
    print("\n[6/6] 测试融合模态检测流程...")
    await run_test_with_result(
        test_fusion_detection,
        "融合模态检测测试",
        test_result,
        TEST_CONFIG.get("skip_fusion", False)
    )

    # 7. 测试错误场景
    if not TEST_CONFIG.get("skip_errors", False):
        print("\n[7/6] 测试错误场景...")
        await run_test_with_result(
            test_error_scenarios,
            "错误场景测试",
            test_result,
            False
        )


async def main():
    """主函数 - 解析参数并运行测试"""
    # 解析命令行参数
    args = parse_arguments()
    setup_test_config(args)

    # 创建测试结果统计
    test_result = TestResult()

    # 运行所有测试
    try:
        await run_all_tests(test_result)
    except Exception as e:
        print(f"\n测试过程中发生未预期错误: {e}")
        import traceback
        traceback.print_exc()
        test_result.add_failed("测试框架", str(e))

    # 打印测试摘要
    all_passed = test_result.print_summary()

    # 根据测试结果设置退出码
    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
