"""
推理功能测试脚本（异步版本）

使用 asyncio 和 aiohttp 实现异步推理请求，
配合 websockets 实现实时进度监听。

测试推理服务的各项功能：
1. WebSocket 连接与 client_id 获取
2. 单模态推理（RGB/IR）
3. 融合模态推理
4. 任务状态查询
5. 推理结果验证
6. WebSocket 进度推送（异步监听）

使用前请确保服务已启动：
    python main.py

并且已有有效的 API Key（可通过 test_auth.py 获取）

运行测试：
    # 方式 1：使用命令行参数（推荐）
    python tests/test_infer.py --api-key sk_xxxxx

    # 方式 2：使用环境变量
    set API_KEY=sk_xxxxx && python tests/test_infer.py

    # 方式 3：使用默认配置（需修改脚本中的 DEFAULT_API_KEY）
    python tests/test_infer.py

获取 API Key：
    python tests/test_auth.py
"""

import asyncio
import json
import base64
import time
import argparse
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

try:
    import aiohttp
    import websockets
    from PIL import Image
    import numpy as np
except ImportError:
    print("请安装异步依赖：pip install aiohttp websockets pillow numpy")
    exit(1)

BASE_URL = "http://127.0.0.1:8000"

# 测试配置 - 可通过命令行参数或环境变量覆盖
DEFAULT_API_KEY = "sk_your_api_key_here"

# 线程池用于同步/异步转换
_executor = ThreadPoolExecutor(max_workers=4)


def create_test_image_base64(width=112, height=112, color=(255, 128, 64)):
    """创建测试图像并转换为 base64"""
    img = Image.new('RGB', (width, height), color)
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')


def print_section(title):
    """打印分割线"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(test_name: str, success: bool, details="") -> bool:
    """打印测试结果"""
    status = "[PASS]" if success else "[FAIL]"
    print(f"\n[{test_name}] {status}")
    if details:
        print(f"    {details}")
    return success


# ==================== 异步 WebSocket 连接 ====================

async def test_websocket_connect(api_key):
    """异步测试 WebSocket 连接并获取 client_id"""
    print_section("WebSocket 连接测试")

    try:
        ws_url = f"ws://127.0.0.1:8000/infer/ws?api_key={api_key}"
        ws = await websockets.connect(ws_url, open_timeout=5)

        # 接收连接确认消息
        message = await asyncio.wait_for(ws.recv(), timeout=5)
        data = json.loads(message)

        if data.get("type") == "connected" and data.get("client_id"):
            client_id = data["client_id"]
            print_result("WebSocket 连接", True, f"Client ID: {client_id}")
            return ws, client_id
        else:
            print_result("WebSocket 连接", False, f"响应格式错误：{data}")
            await ws.close()
            return None, None

    except Exception as e:
        print_result("WebSocket 连接", False, f"错误：{e}")
        return None, None


# ==================== 异步推理请求 ====================

async def send_inference_request(session, endpoint, payload, api_key, client_id):
    """发送异步推理请求"""
    url = f"{BASE_URL}{endpoint}"
    headers = {
        "X-API-Key": api_key,
        "X-Client-ID": client_id,
        "Content-Type": "application/json"
    }

    async with session.post(url, json=payload, headers=headers) as response:
        data = await response.json()
        return response.status, data


async def send_status_request(session, task_id, api_key):
    """发送异步状态查询请求"""
    url = f"{BASE_URL}/infer/task/{task_id}"
    headers = {"X-API-Key": api_key}

    async with session.get(url, headers=headers) as response:
        data = await response.json()
        return response.status, data


# ==================== WebSocket 进度监听器 ====================

class ProgressListener:
    """WebSocket 进度监听器类"""

    def __init__(self, ws):
        self.ws = ws
        self.progress_messages = []
        self.completed_message = None
        self.failed_message = None
        self.listening = True

    async def listen(self):
        """异步监听 WebSocket 消息"""
        while self.listening:
            try:
                message = await asyncio.wait_for(self.ws.recv(), timeout=0.5)
                data = json.loads(message)
                msg_type = data.get("type")

                if msg_type == "progress_update":
                    progress_data = data.get("data", {})
                    self.progress_messages.append(progress_data)
                    print(f"\n  [进度更新] {progress_data.get('message', '')}")
                    print(f"           进度：{progress_data.get('progress_percentage', 0)}%")

                elif msg_type == "task_completed":
                    self.completed_message = data
                    print(f"\n  [任务完成] 收到完成通知")
                    break

                elif msg_type == "task_failed":
                    self.failed_message = data
                    print(f"\n  [任务失败] {data.get('data', {}).get('message', '')}")
                    break

            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                break
            except Exception as e:
                print(f"    接收消息异常：{e}")
                break

    def get_result(self):
        """返回监听结果"""
        return {
            "success": len(self.progress_messages) > 0 or self.completed_message is not None,
            "progress_count": len(self.progress_messages),
            "completed": self.completed_message is not None,
            "failed": self.failed_message is not None
        }


# ==================== 单模态推理测试 ====================

async def test_single_rgb_inference(ws, client_id, api_key):
    """异步测试单模态 RGB 推理"""
    print_section("单模态 RGB 推理测试")

    if not ws or not client_id:
        return print_result("单模态 RGB 推理", False, "WebSocket 未连接"), None

    try:
        # 创建测试图像
        rgb_base64 = create_test_image_base64(color=(255, 128, 64))
        images = [rgb_base64] * 3  # 3 张相同的图像

        async with aiohttp.ClientSession() as session:
            # 发送推理请求（不等待，立即开始监听）
            task_future = asyncio.create_task(
                send_inference_request(
                    session, "/infer/single",
                    {"mode": "single", "modality": "rgb", "images": images},
                    api_key, client_id
                )
            )

            # 立即开始监听 WebSocket 进度
            listener = ProgressListener(ws)
            listen_task = asyncio.create_task(listener.listen())

            # 等待推理请求返回
            status, data = await task_future

            if status == 200:
                task_id = data.get("task_id")
                print_result(
                    "单模态 RGB 推理请求",
                    True,
                    f"Task ID: {task_id}\n消息：{data.get('message', '')}"
                )

                # 继续监听直到完成
                await listen_task

                # 返回结果
                result = listener.get_result()
                print_result(
                    "WebSocket 进度监听",
                    result["success"],
                    f"收到 {result['progress_count']} 条进度更新"
                )
                return task_id, result
            else:
                print_result(
                    "单模态 RGB 推理请求",
                    False,
                    f"状态码：{status}, 错误：{data}"
                )
                return None, None

    except Exception as e:
        print_result("单模态 RGB 推理请求", False, f"错误：{e}")
        return None, None


async def test_single_ir_inference(ws, client_id, api_key):
    """异步测试单模态 IR 推理"""
    print_section("单模态 IR 推理测试")

    if not ws or not client_id:
        return print_result("单模态 IR 推理", False, "WebSocket 未连接"), None

    try:
        # 创建测试图像（模拟 IR 图像，使用灰度）
        ir_base64 = create_test_image_base64(color=(128, 128, 128))
        images = [ir_base64] * 2  # 2 张相同的图像

        async with aiohttp.ClientSession() as session:
            # 发送推理请求
            task_future = asyncio.create_task(
                send_inference_request(
                    session, "/infer/single",
                    {"mode": "single", "modality": "ir", "images": images},
                    api_key, client_id
                )
            )

            # 监听进度
            listener = ProgressListener(ws)
            listen_task = asyncio.create_task(listener.listen())

            # 等待推理请求返回
            status, data = await task_future

            if status == 200:
                task_id = data.get("task_id")
                print_result(
                    "单模态 IR 推理请求",
                    True,
                    f"Task ID: {task_id}\n消息：{data.get('message', '')}"
                )

                # 继续监听直到完成
                await listen_task

                # 返回结果
                result = listener.get_result()
                print_result(
                    "WebSocket 进度监听",
                    result["success"],
                    f"收到 {result['progress_count']} 条进度更新"
                )
                return task_id, result
            else:
                print_result(
                    "单模态 IR 推理请求",
                    False,
                    f"状态码：{status}, 错误：{data}"
                )
                return None, None

    except Exception as e:
        print_result("单模态 IR 推理请求", False, f"错误：{e}")
        return None, None


# ==================== 融合模态推理测试 ====================

async def test_fusion_inference(ws, client_id, api_key):
    """异步测试融合模态推理"""
    print_section("融合模态推理测试")

    if not ws or not client_id:
        return print_result("融合模态推理", False, "WebSocket 未连接"), None

    try:
        # 创建测试图像对
        rgb_base64 = create_test_image_base64(color=(255, 128, 64))
        ir_base64 = create_test_image_base64(color=(128, 128, 128))

        pairs = [
            {"rgb": rgb_base64, "ir": ir_base64},
            {"rgb": rgb_base64, "ir": ir_base64}
        ]

        async with aiohttp.ClientSession() as session:
            # 发送推理请求
            task_future = asyncio.create_task(
                send_inference_request(
                    session, "/infer/fusion",
                    {"mode": "fusion", "pairs": pairs},
                    api_key, client_id
                )
            )

            # 监听进度
            listener = ProgressListener(ws)
            listen_task = asyncio.create_task(listener.listen())

            # 等待推理请求返回
            status, data = await task_future

            if status == 200:
                task_id = data.get("task_id")
                print_result(
                    "融合模态推理请求",
                    True,
                    f"Task ID: {task_id}\n消息：{data.get('message', '')}"
                )

                # 继续监听直到完成
                await listen_task

                # 返回结果
                result = listener.get_result()
                print_result(
                    "WebSocket 进度监听",
                    result["success"],
                    f"收到 {result['progress_count']} 条进度更新"
                )
                return task_id, result
            else:
                print_result(
                    "融合模态推理请求",
                    False,
                    f"状态码：{status}, 错误：{data}"
                )
                return None, None

    except Exception as e:
        print_result("融合模态推理请求", False, f"错误：{e}")
        return None, None


# ==================== 任务状态查询测试 ====================

async def test_task_status(task_id, api_key):
    """异步测试任务状态查询"""
    print_section(f"任务状态查询 - {task_id[:8]}...")

    if not task_id:
        return print_result("任务状态查询", False, "Task ID 为空")

    try:
        max_retries = 10
        retry_count = 0

        async with aiohttp.ClientSession() as session:
            while retry_count < max_retries:
                status, data = await send_status_request(session, task_id, api_key)

                if status == 200:
                    task_status = data.get("status")
                    progress = data.get("progress_percentage", 0)

                    print(f"\n  状态：{task_status}")
                    print(f"  进度：{progress}%")
                    print(f"  已完成：{data.get('completed_items', 0)}/{data.get('total_items', 0)}")

                    if task_status == "completed":
                        real_count = data.get("real_count", 0)
                        fake_count = data.get("fake_count", 0)
                        print(f"  真实人脸：{real_count}")
                        print(f"  伪造人脸：{fake_count}")

                        return print_result(
                            "任务状态查询",
                            True,
                            f"任务已完成，共处理 {data.get('completed_items', 0)} 张图像"
                        )
                    elif task_status == "failed":
                        return print_result(
                            "任务状态查询",
                            False,
                            f"任务失败：{data.get('message', '未知错误')}"
                        )

                elif status == 404:
                    return print_result(
                        "任务状态查询",
                        False,
                        "任务不存在或已过期"
                    )
                else:
                    return print_result(
                        "任务状态查询",
                        False,
                        f"状态码：{status}, 错误：{data}"
                    )

                retry_count += 1
                await asyncio.sleep(1)

        return print_result(
            "任务状态查询",
            False,
            f"任务超时（等待 {max_retries} 秒）"
        )

    except Exception as e:
        print_result("任务状态查询", False, f"错误：{e}")
        return None


# ==================== 主测试流程 ====================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="推理功能测试脚本（异步版本）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用 API Key 参数
  python tests/test_infer.py --api-key sk_xxxxx

  # 使用简写
  python tests/test_infer.py -k sk_xxxxx

  # 使用环境变量
  set API_KEY=sk_xxxxx && python tests/test_infer.py

  # 使用默认配置（修改脚本中的 DEFAULT_API_KEY）
  python tests/test_infer.py
        """
    )

    parser.add_argument(
        "-k", "--api-key",
        type=str,
        default=None,
        help="API Key (优先级：命令行参数 > 环境变量 > 默认配置)"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="服务器地址 (默认：127.0.0.1)"
    )

    parser.add_argument(
        "--port",
        type=str,
        default="8000",
        help="服务器端口 (默认：8000)"
    )

    return parser.parse_args()


async def run_tests(api_key):
    """异步运行所有测试"""
    global BASE_URL

    print("\n" + "=" * 60)
    print("  推理功能测试（异步版本）")
    print("=" * 60)
    print(f"\n  服务器：{BASE_URL}")
    print(f"  API Key: {api_key[:15]}..." if api_key else "  API Key: 未设置")

    # 检查 API Key 配置
    if api_key == "sk_your_api_key_here":
        print("\n[WARN] 请先配置 API Key！")
        print("\n获取 API Key 的方法:")
        print("  1. 运行认证测试获取 API Key:")
        print("     python tests/test_auth.py")
        print("\n  2. 使用激活码换取 API Key:")
        print("     curl -X POST http://127.0.0.1:8000/auth/activate \\")
        print("       -H \"Content-Type: application/json\" \\")
        print("       -d '{\"code\": \"ACT-XXXXX\"}'")
        print("\n  3. 使用环境变量:")
        print("     set API_KEY=sk_xxxxx && python tests/test_infer.py")
        print("\n  4. 使用命令行参数:")
        print("     python tests/test_infer.py --api-key sk_xxxxx")
        return [], []

    test_results = []
    websocket_results = []

    # 1. WebSocket 连接
    ws, client_id = await test_websocket_connect(api_key)

    if ws and client_id:
        # 2. 单模态 RGB 推理测试（包含进度监听）
        task_id_1, ws_result = await test_single_rgb_inference(ws, client_id, api_key)
        if task_id_1:
            websocket_results.append(ws_result)
            # 查询状态
            status_result = await test_task_status(task_id_1, api_key)
            test_results.append(status_result)

        # 3. 单模态 IR 推理测试
        task_id_2, ws_result = await test_single_ir_inference(ws, client_id, api_key)
        if task_id_2:
            websocket_results.append(ws_result)
            status_result = await test_task_status(task_id_2, api_key)
            test_results.append(status_result)

        # 4. 融合模态推理测试
        task_id_3, ws_result = await test_fusion_inference(ws, client_id, api_key)
        if task_id_3:
            websocket_results.append(ws_result)
            status_result = await test_task_status(task_id_3, api_key)
            test_results.append(status_result)

        # 关闭 WebSocket 连接
        if ws:
            await ws.close()
            print("\n[INFO] WebSocket 连接已关闭")

    return test_results, websocket_results


def main():
    """主函数"""
    args = parse_args()

    # 确定 API Key（优先级：命令行参数 > 环境变量 > 默认配置）
    api_key = args.api_key or __import__("os").getenv("API_KEY") or DEFAULT_API_KEY

    # 更新 BASE_URL
    global BASE_URL
    BASE_URL = f"http://{args.host}:{args.port}"

    # 运行异步测试
    test_results, websocket_results = asyncio.run(run_tests(api_key))

    # 打印汇总
    print_section("测试汇总")
    test_passed = sum(1 for r in test_results if r)
    test_total = len(test_results)
    ws_passed = sum(1 for r in websocket_results if r and r.get("success"))
    ws_total = len(websocket_results)

    print(f"\n  推理测试：{test_passed}/{test_total}")
    print(f"  进度监听：{ws_passed}/{ws_total}")

    total_passed = test_passed + ws_passed
    total_tests = test_total + ws_total

    if total_passed == total_tests and total_tests > 0:
        print("\n  [OK] 所有测试通过!")
    elif total_tests == 0:
        print("\n  [WARN] 未执行任何测试（可能 WebSocket 连接失败）")
    else:
        print(f"\n  [WARN] {total_tests - total_passed} 个测试失败")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
    except Exception as e:
        print(f"\n测试异常：{e}")
        import traceback
        traceback.print_exc()
