"""
健康检查增强功能测试脚本

测试内容：
1. 简单健康检查端点 /health (无需认证)
2. 详细健康检查端点 /health/detailed (需要管理员 JWT)
"""

import sys
import requests

# 配置
BASE_URL = "http://127.0.0.1:8000"

# 管理员账号 (从 .env 读取)
ADMIN_USERNAME = "Moore-ai"
ADMIN_PASSWORD = "Moore20060810"


def get_jwt_token():
    """获取管理员 JWT Token"""
    try:
        response = requests.post(
            f"{BASE_URL}/auth/token",
            json={"username": ADMIN_USERNAME, "password": ADMIN_PASSWORD},
            timeout=5
        )
        response.raise_for_status()
        data = response.json()
        return data.get("access_token")
    except Exception as e:
        print(f"Failed to get JWT token: {e}")
        return None


def test_simple_health_check():
    """测试 1: 简单健康检查端点 (无需认证)"""
    print("\n" + "=" * 50)
    print("Test 1: Simple health check endpoint /health")
    print("=" * 50)

    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        response.raise_for_status()

        data = response.json()
        print(f"Status code: {response.status_code}")
        print(f"Response: {data}")

        assert data["status"] == "ok", f"Expected status 'ok', got '{data['status']}'"
        assert "content" in data, "Missing 'content' field"

        print("[PASS] Test passed!")
        return True

    except requests.exceptions.RequestException as e:
        print(f"[FAIL] Test failed: {e}")
        return False


def test_detailed_health_check_without_auth():
    """测试 2: 详细健康检查端点 (未认证，应该返回 401)"""
    print("\n" + "=" * 50)
    print("Test 2: Detailed health check without auth (should fail)")
    print("=" * 50)

    try:
        response = requests.get(f"{BASE_URL}/health/detailed", timeout=5)

        # 应该返回 401 或 403
        if response.status_code in (401, 403):
            print(f"Status code: {response.status_code} (expected)")
            print(f"Response: {response.json()}")
            print("[PASS] Test passed! (Correctly rejected unauthenticated request)")
            return True
        else:
            print(f"[FAIL] Unexpected status code: {response.status_code}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"[FAIL] Test failed: {e}")
        return False


def test_detailed_health_check_with_admin():
    """测试 3: 详细健康检查端点 (管理员认证)"""
    print("\n" + "=" * 50)
    print("Test 3: Detailed health check with admin JWT")
    print("=" * 50)

    # 获取管理员 token
    token = get_jwt_token()
    if not token:
        print("[FAIL] Could not get admin JWT token")
        return False

    print(f"Got JWT token: {token[:20]}...")

    try:
        response = requests.get(
            f"{BASE_URL}/health/detailed",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10
        )
        response.raise_for_status()

        data = response.json()
        print(f"Status code: {response.status_code}")
        print(f"Overall status: {data['status']}")

        # 检查模型状态
        print("\n[Models]:")
        for model in data.get("models", []):
            exists_str = "[OK]" if model["exists"] else "[MISSING]"
            size_str = f"{model['file_size_mb']} MB" if model.get("file_size_mb") else "N/A"
            print(f"  - {model['name']}: {exists_str}, size: {size_str}")

        # 检查 GPU 状态
        print("\n[GPU]:")
        gpu = data.get("gpu", {})
        if gpu.get("available"):
            print(f"  - Available: Yes")
            print(f"  - Device: {gpu.get('device_name')}")
            print(f"  - Memory free: {gpu.get('memory_free_mb')} MB")
        else:
            print(f"  - Available: No")

        # 检查磁盘状态
        print("\n[Disk]:")
        disk = data.get("disk", {})
        if disk:
            print(f"  - Free: {disk.get('free_gb')} GB")

        # 检查数据库状态
        print("\n[Database]:")
        db = data.get("database", {})
        connected_str = "[OK]" if db.get("connected") else "[FAIL]"
        print(f"  - Status: {connected_str}")

        # 检查存储服务状态
        print("\n[Storage]:")
        storage = data.get("storage", {})
        init_str = "[OK]" if storage.get("initialized") else "[FAIL]"
        print(f"  - Status: {init_str}")

        # 检查健康组件
        print("\n[Healthy components]:")
        for comp in data.get("healthy_components", []):
            print(f"  - {comp}")

        print("\n[Unhealthy components]:")
        for comp in data.get("unhealthy_components", []):
            print(f"  - {comp}")

        # 验证响应字段
        assert "status" in data, "Missing 'status' field"
        assert "models" in data, "Missing 'models' field"
        assert "gpu" in data, "Missing 'gpu' field"
        assert "disk" in data, "Missing 'disk' field"
        assert "database" in data, "Missing 'database' field"
        assert "storage" in data, "Missing 'storage' field"

        print("\n[PASS] Test passed!")
        return True

    except requests.exceptions.RequestException as e:
        print(f"[FAIL] Test failed: {e}")
        return False
    except AssertionError as e:
        print(f"[FAIL] Assertion failed: {e}")
        return False


def main():
    """运行所有测试"""
    print("=" * 50)
    print("Health Check Enhancement Tests")
    print("=" * 50)

    results = []

    # 测试 1: 简单健康检查 (无需认证)
    results.append(("Simple health check", test_simple_health_check()))

    # 测试 2: 详细健康检查 (未认证，应该拒绝)
    results.append(("Detailed health without auth", test_detailed_health_check_without_auth()))

    # 测试 3: 详细健康检查 (管理员认证)
    results.append(("Detailed health with admin", test_detailed_health_check_with_admin()))

    # 总结
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{name}: {status}")

    print(f"\nTotal: {passed}/{total} passed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
