"""
历史记录调试测试脚本 - 版本 2

修复测试逻辑，确保正确测试历史记录的保存和查询
"""

import sys
import requests
import base64
import time
import websocket
import json

# 配置
BASE_URL = "http://127.0.0.1:8000"

# 默认激活码
DEFAULT_ACTIVATION_CODE = "ACT-DEV-DEFAULT-KEY"


def get_api_key_from_activation_code(code=DEFAULT_ACTIVATION_CODE):
    """使用激活码换取 API Key"""
    try:
        response = requests.post(
            f"{BASE_URL}/auth/activate",
            json={"code": code},
            timeout=5
        )
        response.raise_for_status()
        data = response.json()
        print(f"[激活] 激活码换取 API Key: {data.get('api_key', '')[:20]}...")
        return data.get("api_key")
    except Exception as e:
        print(f"[激活] 失败：{e}")
        return None


def create_mock_image():
    """创建一个简单的模拟图片（1x1 像素的白色 PNG）"""
    png_data = bytes([
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
        0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
        0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
        0x54, 0x08, 0xD7, 0x63, 0xF8, 0xFF, 0xFF, 0x3F,
        0x00, 0x05, 0xFE, 0x02, 0xFE, 0xDC, 0xCC, 0x59,
        0xE7, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,
        0x44, 0xAE, 0x42, 0x60, 0x82
    ])
    return base64.b64encode(png_data).decode('utf-8')


def connect_websocket(api_key):
    """连接 WebSocket 获取 client_id，保持连接"""
    try:
        ws = websocket.create_connection(f"ws://127.0.0.1:8000/infer/ws?api_key={api_key}")
        result = ws.recv()
        client_id = json.loads(result).get('client_id')
        print(f"WebSocket 连接成功，client_id: {client_id}")
        return ws, client_id
    except Exception as e:
        print(f"WebSocket 连接失败：{e}")
        return None, f"test_{time.time()}"


def query_history(api_key, client_id=None):
    """查询历史记录"""
    headers = {"X-API-Key": api_key}
    params = {"page": 1, "page_size": 20}
    if client_id:
        params["client_id"] = client_id

    response = requests.get(f"{BASE_URL}/history", headers=headers, params=params, timeout=10)
    print(f"[查询历史] 状态码：{response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"[查询历史] total={data.get('total')}, items_count={len(data.get('items', []))}")
        if data.get('items'):
            for item in data['items'][:3]:
                task_id = item.get('task_id', 'N/A')
                api_key_hash = item.get('api_key_hash', 'N/A')
                hash_str = api_key_hash[:16] + '...' if api_key_hash and api_key_hash != 'N/A' else 'N/A'
                print(f"  - task_id={task_id}, api_key_hash={hash_str}")
        return data
    else:
        print(f"[查询历史] 错误：{response.text}")
    return None


def do_inference(api_key, client_id):
    """执行单次推理"""
    headers = {"X-API-Key": api_key, "X-Client-ID": client_id}
    image = create_mock_image()

    response = requests.post(
        f"{BASE_URL}/infer/single",
        headers=headers,
        json={"mode": "single", "modality": "rgb", "images": [image]},
        timeout=30
    )
    print(f"[推理] 状态码：{response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"[推理] task_id={data.get('task_id')}, message={data.get('message')}")
        return data.get('task_id')
    else:
        print(f"[推理] 错误：{response.text}")
    return None


def main():
    print("=" * 60)
    print("历史记录调试测试 - 版本 2")
    print("=" * 60)

    # 步骤 1: 获取 API Key
    print("\n[步骤 1] 使用激活码换取 API Key...")
    api_key = get_api_key_from_activation_code()
    if not api_key:
        print("无法获取 API Key，退出测试")
        return 1
    print(f"API Key: {api_key[:20]}...")

    # 步骤 2: 连接 WebSocket 获取 client_id（保持连接）
    print("\n[步骤 2] 连接 WebSocket 获取 client_id...")
    ws, client_id = connect_websocket(api_key)
    if not ws:
        print("使用模拟 client_id，但推理可能会失败")

    try:
        # 步骤 3: 查询当前历史记录（推理前）
        print("\n[步骤 3] 推理前查询历史记录...")
        history_before = query_history(api_key, client_id)
        count_before = history_before.get('total', 0) if history_before else 0

        # 步骤 4: 执行推理
        print("\n[步骤 4] 执行推理...")
        task_id = do_inference(api_key, client_id)

        if task_id:
            # 等待任务完成
            print("等待任务完成...")
            time.sleep(5)

            # 步骤 5: 查询历史记录（推理后）
            print("\n[步骤 5] 推理后查询历史记录...")
            history_after = query_history(api_key, client_id)
            count_after = history_after.get('total', 0) if history_after else 0

            # 步骤 6: 验证结果
            print("\n" + "=" * 60)
            print("测试结果")
            print("=" * 60)
            print(f"推理前记录数：{count_before}")
            print(f"推理后记录数：{count_after}")
            print(f"新增记录数：{count_after - count_before}")

            if count_after > count_before:
                print("\n[PASS] 测试通过！新记录已保存。")
                return 0
            else:
                print("\n[FAIL] 测试失败！新记录未被保存。")
                print("\n可能的原因:")
                print("1. 任务调度器未执行任务")
                print("2. api_key_hash 不一致")
                print("3. client_id 不匹配")
                return 1
        else:
            print("\n[FAIL] 推理失败，未获取到 task_id")
            return 1
    finally:
        # 关闭 WebSocket 连接
        if ws:
            ws.close()
            print("\nWebSocket 连接已关闭")


if __name__ == "__main__":
    sys.exit(main())
