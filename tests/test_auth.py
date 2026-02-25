"""
推理 API 认证与功能测试脚本

测试内容：
1. 健康检查（无需认证）
2. JWT Token 获取（管理员登录）
3. 激活码管理（生成、列出、更新、删除）
4. 激活码换取 API Key
5. 推理端点认证（单模态、融合模态）
6. 任务状态查询
7. WebSocket 连接测试

使用前请确保服务已启动：
    python main.py

运行测试：
    python tests/test_auth.py
"""

import requests
import websocket
import json

BASE_URL = "http://127.0.0.1:8000"

# 测试配置
ADMIN_USERNAME = "Moore-ai"
ADMIN_PASSWORD = "Moore20060810"


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


# ==================== 健康检查 ====================

def test_health():
    """测试健康检查端点（无需认证）"""
    print_section("健康检查")
    try:
        response = requests.get(f"{BASE_URL}/health")
        return print_result(
            "健康检查",
            response.status_code == 200,
            f"响应：{response.json()}"
        )
    except Exception as e:
        return print_result("健康检查", False, f"错误：{e}")


# ==================== JWT Token 测试 ====================

def test_get_jwt_token():
    """测试获取 JWT Token（管理员登录）"""
    print_section("JWT Token 获取")
    try:
        response = requests.post(
            f"{BASE_URL}/auth/token",
            json={"username": ADMIN_USERNAME, "password": ADMIN_PASSWORD}
        )
        if response.status_code == 200:
            data = response.json()
            token = data["access_token"]
            return print_result(
                "JWT Token 获取",
                True,
                f"Token: {token[:50]}...\n过期时间：{data['expires_in']}秒"
            ), token
        else:
            return print_result(
                "JWT Token 获取",
                False,
                f"状态码：{response.status_code}, 错误：{response.json()}"
            ), None
    except Exception as e:
        return print_result("JWT Token 获取", False, f"错误：{e}"), None


def test_get_me(token):
    """测试使用 Token 获取用户信息"""
    print_section("用户信息查询")
    try:
        response = requests.get(
            f"{BASE_URL}/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        return print_result(
            "用户信息查询",
            response.status_code == 200,
            f"响应：{response.json()}"
        )
    except Exception as e:
        return print_result("用户信息查询", False, f"错误：{e}")


# ==================== 激活码管理测试 ====================

def test_create_activation_code(token):
    """测试创建激活码"""
    print_section("创建激活码")
    try:
        response = requests.post(
            f"{BASE_URL}/auth/activation-codes",
            json={
                "name": "test_user",
                "max_uses": 10,
                "expires_in_hours": 24
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        if response.status_code == 200:
            data = response.json()
            return print_result(
                "创建激活码",
                True,
                f"激活码：{data['code']}\n最大使用次数：{data['max_uses']}"
            ), data["code"]
        else:
            return print_result(
                "创建激活码",
                False,
                f"状态码：{response.status_code}, 错误：{response.json()}"
            ), None
    except Exception as e:
        return print_result("创建激活码", False, f"错误：{e}"), None


def test_list_activation_codes(token):
    """测试列出所有激活码"""
    print_section("列出激活码")
    try:
        response = requests.get(
            f"{BASE_URL}/auth/activation-codes",
            headers={"Authorization": f"Bearer {token}"}
        )
        if response.status_code == 200:
            data = response.json()
            codes = data.get("activation_codes", [])
            return print_result(
                "列出激活码",
                True,
                f"激活码数量：{len(codes)}"
            )
        else:
            return print_result(
                "列出激活码",
                False,
                f"状态码：{response.status_code}, 错误：{response.json()}"
            )
    except Exception as e:
        return print_result("列出激活码", False, f"错误：{e}")


def test_update_activation_code(token, code):
    """测试更新激活码配置"""
    print_section("更新激活码")
    try:
        response = requests.put(
            f"{BASE_URL}/auth/activation-codes/{code}",
            json={"max_uses": 20},
            headers={"Authorization": f"Bearer {token}"}
        )
        if response.status_code == 200:
            data = response.json()
            return print_result(
                "更新激活码",
                True,
                f"响应：{data.get('message', '')}"
            )
        else:
            return print_result(
                "更新激活码",
                False,
                f"状态码：{response.status_code}, 错误：{response.json()}"
            )
    except Exception as e:
        return print_result("更新激活码", False, f"错误：{e}")


def test_deactivate_activation_code(token, code):
    """测试禁用激活码"""
    print_section("禁用激活码")
    try:
        response = requests.post(
            f"{BASE_URL}/auth/activation-codes/{code}/deactivate",
            headers={"Authorization": f"Bearer {token}"}
        )
        if response.status_code == 200:
            return print_result(
                "禁用激活码",
                True,
                f"响应：{response.json().get('message', '')}"
            )
        else:
            return print_result(
                "禁用激活码",
                False,
                f"状态码：{response.status_code}, 错误：{response.json()}"
            )
    except Exception as e:
        return print_result("禁用激活码", False, f"错误：{e}")


# ==================== 激活码换取 API Key ====================

def test_activate_code(activation_code):
    """测试激活码换取 API Key"""
    print_section("激活码换取 API Key")
    try:
        response = requests.post(
            f"{BASE_URL}/auth/activate",
            json={"code": activation_code}
        )
        if response.status_code == 200:
            data = response.json()
            return print_result(
                "激活码换取 API Key",
                True,
                f"API Key: {data['api_key'][:30]}...\n过期时间：{data.get('expires_at', '永不过期')}"
            ), data["api_key"]
        else:
            return print_result(
                "激活码换取 API Key",
                False,
                f"状态码：{response.status_code}, 错误：{response.json()}"
            ), None
    except Exception as e:
        return print_result("激活码换取 API Key", False, f"错误：{e}"), None


# ==================== 推理端点测试 ====================

def test_infer_without_auth():
    """测试未认证访问推理端点（应该被拒绝）"""
    print_section("未认证访问推理端点（预期失败）")
    try:
        response = requests.post(
            f"{BASE_URL}/infer/single",
            json={"mode": "single", "modality": "rgb", "images": []}
        )
        return print_result(
            "未认证访问拦截",
            response.status_code == 401,
            f"状态码：{response.status_code} (预期 401)"
        )
    except Exception as e:
        return print_result("未认证访问拦截", False, f"错误：{e}")


def test_task_status_without_auth():
    """测试未认证访问任务状态（应该被拒绝）"""
    print_section("未认证访问任务状态（预期失败）")
    try:
        response = requests.get(f"{BASE_URL}/infer/task/non-existent-task")
        return print_result(
            "未认证任务查询拦截",
            response.status_code == 401,
            f"状态码：{response.status_code} (预期 401)"
        )
    except Exception as e:
        return print_result("未认证任务查询拦截", False, f"错误：{e}")


def test_infer_with_api_key(api_key):
    """测试使用 API Key 访问推理端点"""
    print_section("使用 API Key 访问推理端点")
    headers = {"X-API-Key": api_key}

    # 测试单模态推理（需要 WebSocket client_id，这里只测试认证）
    try:
        response = requests.post(
            f"{BASE_URL}/infer/single",
            json={"mode": "single", "modality": "rgb", "images": []},
            headers=headers
        )
        # 由于没有 client_id，会返回无效 client_id 错误，但认证应该通过
        return print_result(
            "单模态推理认证",
            response.status_code in (200, 400, 422),
            f"状态码：{response.status_code}, 响应：{response.json()}"
        )
    except Exception as e:
        return print_result("单模态推理认证", False, f"错误：{e}")


def test_fusion_with_api_key(api_key):
    """测试使用 API Key 访问融合模态推理端点"""
    print_section("使用 API Key 访问融合模态推理端点")
    headers = {"X-API-Key": api_key}

    try:
        response = requests.post(
            f"{BASE_URL}/infer/fusion",
            json={"mode": "fusion", "pairs": []},
            headers=headers
        )
        return print_result(
            "融合模态推理认证",
            response.status_code in (200, 400, 422),
            f"状态码：{response.status_code}, 响应：{response.json()}"
        )
    except Exception as e:
        return print_result("融合模态推理认证", False, f"错误：{e}")


# ==================== WebSocket 测试 ====================

def test_websocket_connection(api_key):
    """测试 WebSocket 连接"""
    print_section("WebSocket 连接测试")
    try:
        ws_url = f"ws://127.0.0.1:8000/infer/ws?api_key={api_key}"
        ws = websocket.create_connection(ws_url, timeout=5)

        # 接收连接确认消息
        message = ws.recv()
        data = json.loads(message)

        ws.close()

        if data.get("type") == "connected" and data.get("client_id"):
            return print_result(
                "WebSocket 连接",
                True,
                f"Client ID: {data['client_id']}"
            )
        else:
            return print_result(
                "WebSocket 连接",
                False,
                f"响应格式错误：{data}"
            )
    except Exception as e:
        return print_result("WebSocket 连接", False, f"错误：{e}")


def test_websocket_without_auth():
    """测试 WebSocket 无认证连接"""
    print_section("WebSocket 无认证连接测试")
    try:
        ws_url = "ws://127.0.0.1:8000/infer/ws"
        ws = websocket.create_connection(ws_url, timeout=5)

        # 接收连接确认消息
        message = ws.recv()
        data = json.loads(message)

        ws.close()

        # 无认证时也应该能连接（可选认证）
        if data.get("type") == "connected" and data.get("client_id"):
            return print_result(
                "WebSocket 无认证连接",
                True,
                f"Client ID: {data['client_id']} (可选认证)"
            )
        else:
            return print_result(
                "WebSocket 无认证连接",
                False,
                f"响应格式错误：{data}"
            )
    except Exception as e:
        return print_result("WebSocket 无认证连接", False, f"错误：{e}")


# ==================== 主测试流程 ====================

def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("  推理 API 认证与功能测试")
    print("=" * 60)

    test_results = []
    total_tests = 0
    executed_tests = 0

    # 1. 健康检查
    total_tests += 1
    executed_tests += 1
    test_results.append(("健康检查", test_health()))

    # 2. 未认证访问测试
    total_tests += 2
    executed_tests += 2
    test_results.append(("未认证访问拦截", test_infer_without_auth()))
    test_results.append(("未认证任务查询拦截", test_task_status_without_auth()))

    # 3. JWT Token 测试
    total_tests += 1
    token_result, token = test_get_jwt_token()
    executed_tests += 1
    test_results.append(("JWT Token 获取", token_result))

    if token:
        executed_tests += 1
        test_results.append(("用户信息查询", test_get_me(token)))
        total_tests += 1

        # 4. 激活码管理测试
        total_tests += 1
        create_result, activation_code = test_create_activation_code(token)
        executed_tests += 1
        test_results.append(("创建激活码", create_result))

        if activation_code:
            total_tests += 3
            executed_tests += 3
            test_results.append(("列出激活码", test_list_activation_codes(token)))
            test_results.append(("更新激活码", test_update_activation_code(token, activation_code)))

            # 5. 激活码换取 API Key
            activate_result, api_key = test_activate_code(activation_code)
            test_results.append(("激活码换取 API Key", activate_result))

            if api_key:
                # 6. 推理端点测试
                total_tests += 2
                executed_tests += 2
                test_results.append(("单模态推理认证", test_infer_with_api_key(api_key)))
                test_results.append(("融合模态推理认证", test_fusion_with_api_key(api_key)))

                # 7. WebSocket 测试
                total_tests += 1
                executed_tests += 1
                test_results.append(("WebSocket 连接", test_websocket_connection(api_key)))

        # 测试禁用激活码（无论前面是否成功，都尝试执行）
        if activation_code and token:
            total_tests += 1
            executed_tests += 1
            test_results.append(("禁用激活码", test_deactivate_activation_code(token, activation_code)))
    else:
        # Token 获取失败时，跳过依赖 Token 的测试
        print("\n[WARN] JWT Token 获取失败，跳过后续依赖 Token 的测试")

    # WebSocket 无认证测试
    total_tests += 1
    executed_tests += 1
    test_results.append(("WebSocket 无认证连接", test_websocket_without_auth()))

    # 打印汇总
    print_section("测试汇总")
    passed = sum(1 for _, result in test_results if result)
    failed = len(test_results) - passed
    skipped = total_tests - executed_tests

    print(f"\n  通过：{passed}")
    print(f"  失败：{failed}")
    if skipped > 0:
        print(f"  跳过：{skipped}")
    print(f"  总计：{executed_tests} (执行) / {total_tests} (计划)")

    if failed == 0:
        print("\n  [OK] 所有已执行的测试通过!")
    else:
        print(f"\n  [WARN] {failed} 个测试失败")

    # 显示失败的
    print("\n  详细结果:")
    for name, result in test_results:
        status = "通过" if result else "失败"
        print(f"    [{status}] {name}")

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
