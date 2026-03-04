"""
配置热更新功能测试脚本

测试内容：
1. 获取应用配置（需 JWT）
2. 未认证访问（应返回 401）
3. 更新日志配置
4. 更新重试配置
5. 更新调试配置
6. 更新存储策略配置
7. 更新图片压缩配置
8. 获取配置历史
9. 回滚配置

使用前请确保服务已启动：
    python main.py

运行测试：
    python tests/test_config_hotreload.py
"""

import requests
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


# ==================== 认证辅助函数 ====================

def get_jwt_token():
    """获取 JWT Token"""
    try:
        response = requests.post(
            f"{BASE_URL}/auth/token",
            json={"username": ADMIN_USERNAME, "password": ADMIN_PASSWORD}
        )
        if response.status_code == 200:
            return response.json()["access_token"]
        else:
            print(f"获取 JWT Token 失败：{response.json()}")
            return None
    except Exception as e:
        print(f"获取 JWT Token 异常：{e}")
        return None


# ==================== 配置查询测试 ====================

def test_get_app_config(token):
    """测试获取应用配置"""
    print_section("获取应用配置")
    try:
        response = requests.get(
            f"{BASE_URL}/system/config",
            headers={"Authorization": f"Bearer {token}"}
        )
        return print_result(
            "获取应用配置",
            response.status_code == 200,
            f"配置：{json.dumps(response.json(), indent=2, ensure_ascii=False)}"
        )
    except Exception as e:
        return print_result("获取应用配置", False, f"错误：{e}")


def test_get_app_config_unauthorized():
    """测试未认证访问配置"""
    print_section("未认证访问配置")
    try:
        response = requests.get(f"{BASE_URL}/system/config")
        return print_result(
            "未认证访问 (应返回 401)",
            response.status_code == 401,
            f"状态码：{response.status_code}, 响应：{response.json()}"
        )
    except Exception as e:
        return print_result("未认证访问", False, f"错误：{e}")


# ==================== 日志配置测试 ====================

def test_get_logging_config(token):
    """测试获取日志配置"""
    print_section("日志配置查询")
    try:
        response = requests.get(
            f"{BASE_URL}/system/config/logging",
            headers={"Authorization": f"Bearer {token}"}
        )
        return print_result(
            "获取日志配置",
            response.status_code == 200,
            f"配置：{json.dumps(response.json(), indent=2, ensure_ascii=False)}"
        )
    except Exception as e:
        return print_result("获取日志配置", False, f"错误：{e}")


def test_update_logging_config(token):
    """测试更新日志配置"""
    print_section("更新日志配置")
    try:
        # 更新日志级别为 DEBUG
        response = requests.put(
            f"{BASE_URL}/system/config/logging",
            headers={"Authorization": f"Bearer {token}"},
            json={"log_level": "DEBUG", "log_to_console": True}
        )
        if response.status_code == 200:
            data = response.json()
            return print_result(
                "更新日志配置",
                True,
                f"新配置：{json.dumps(data, indent=2, ensure_ascii=False)}"
            )
        else:
            return print_result(
                "更新日志配置",
                False,
                f"状态码：{response.status_code}, 错误：{response.json()}"
            )
    except Exception as e:
        return print_result("更新日志配置", False, f"错误：{e}")


# ==================== 重试配置测试 ====================

def test_get_retry_config(token):
    """测试获取重试配置"""
    print_section("重试配置查询")
    try:
        response = requests.get(
            f"{BASE_URL}/system/config/retry",
            headers={"Authorization": f"Bearer {token}"}
        )
        return print_result(
            "获取重试配置",
            response.status_code == 200,
            f"配置：{json.dumps(response.json(), indent=2, ensure_ascii=False)}"
        )
    except Exception as e:
        return print_result("获取重试配置", False, f"错误：{e}")


def test_update_retry_config(token):
    """测试更新重试配置"""
    print_section("更新重试配置")
    try:
        response = requests.put(
            f"{BASE_URL}/system/config/retry",
            headers={"Authorization": f"Bearer {token}"},
            json={"retry_max_attempts": 5, "retry_delay_seconds": 2.0}
        )
        if response.status_code == 200:
            data = response.json()
            return print_result(
                "更新重试配置",
                True,
                f"新配置：{json.dumps(data, indent=2, ensure_ascii=False)}"
            )
        else:
            return print_result(
                "更新重试配置",
                False,
                f"状态码：{response.status_code}, 错误：{response.json()}"
            )
    except Exception as e:
        return print_result("更新重试配置", False, f"错误：{e}")


# ==================== 调试配置测试 ====================

def test_get_debug_config(token):
    """测试获取调试配置"""
    print_section("调试配置查询")
    try:
        response = requests.get(
            f"{BASE_URL}/system/config/debug",
            headers={"Authorization": f"Bearer {token}"}
        )
        return print_result(
            "获取调试配置",
            response.status_code == 200,
            f"配置：{json.dumps(response.json(), indent=2, ensure_ascii=False)}"
        )
    except Exception as e:
        return print_result("获取调试配置", False, f"错误：{e}")


def test_update_debug_config(token):
    """测试更新调试配置"""
    print_section("更新调试配置")
    try:
        response = requests.put(
            f"{BASE_URL}/system/config/debug",
            headers={"Authorization": f"Bearer {token}"},
            json={"debug_failure_rate": 0.1, "debug_delay_per_image": 0.3}
        )
        if response.status_code == 200:
            data = response.json()
            return print_result(
                "更新调试配置",
                True,
                f"新配置：{json.dumps(data, indent=2, ensure_ascii=False)}"
            )
        else:
            return print_result(
                "更新调试配置",
                False,
                f"状态码：{response.status_code}, 错误：{response.json()}"
            )
    except Exception as e:
        return print_result("更新调试配置", False, f"错误：{e}")


# ==================== 存储策略配置测试 ====================

def test_get_storage_strategy_config(token):
    """测试获取存储策略配置"""
    print_section("存储策略配置查询")
    try:
        response = requests.get(
            f"{BASE_URL}/system/config/storage-strategy",
            headers={"Authorization": f"Bearer {token}"}
        )
        return print_result(
            "获取存储策略配置",
            response.status_code == 200,
            f"配置：{json.dumps(response.json(), indent=2, ensure_ascii=False)}"
        )
    except Exception as e:
        return print_result("获取存储策略配置", False, f"错误：{e}")


def test_update_storage_strategy_config(token):
    """测试更新存储策略配置"""
    print_section("更新存储策略配置")
    try:
        response = requests.put(
            f"{BASE_URL}/system/config/storage-strategy",
            headers={"Authorization": f"Bearer {token}"},
            json={"storage_save_fake_rate": 0.2, "storage_max_per_task": 20}
        )
        if response.status_code == 200:
            data = response.json()
            return print_result(
                "更新存储策略配置",
                True,
                f"新配置：{json.dumps(data, indent=2, ensure_ascii=False)}"
            )
        else:
            return print_result(
                "更新存储策略配置",
                False,
                f"状态码：{response.status_code}, 错误：{response.json()}"
            )
    except Exception as e:
        return print_result("更新存储策略配置", False, f"错误：{e}")


# ==================== 图片压缩配置测试 ====================

def test_get_compress_config(token):
    """测试获取图片压缩配置"""
    print_section("图片压缩配置查询")
    try:
        response = requests.get(
            f"{BASE_URL}/system/config/compress",
            headers={"Authorization": f"Bearer {token}"}
        )
        return print_result(
            "获取图片压缩配置",
            response.status_code == 200,
            f"配置：{json.dumps(response.json(), indent=2, ensure_ascii=False)}"
        )
    except Exception as e:
        return print_result("获取图片压缩配置", False, f"错误：{e}")


def test_update_compress_config(token):
    """测试更新图片压缩配置"""
    print_section("更新图片压缩配置")
    try:
        response = requests.put(
            f"{BASE_URL}/system/config/compress",
            headers={"Authorization": f"Bearer {token}"},
            json={"compress_quality": 80, "compress_type": "pillow"}
        )
        if response.status_code == 200:
            data = response.json()
            return print_result(
                "更新图片压缩配置",
                True,
                f"新配置：{json.dumps(data, indent=2, ensure_ascii=False)}"
            )
        else:
            return print_result(
                "更新图片压缩配置",
                False,
                f"状态码：{response.status_code}, 错误：{response.json()}"
            )
    except Exception as e:
        return print_result("更新图片压缩配置", False, f"错误：{e}")


# ==================== 配置历史与回滚测试 ====================

def test_get_config_history(token):
    """测试获取配置历史"""
    print_section("配置历史查询")
    try:
        response = requests.get(
            f"{BASE_URL}/system/config/history?limit=10",
            headers={"Authorization": f"Bearer {token}"}
        )
        if response.status_code == 200:
            data = response.json()
            history = data.get("history", [])
            return print_result(
                "获取配置历史",
                True,
                f"历史记录数量：{len(history)}\n历史：{json.dumps(history, indent=2, ensure_ascii=False)}"
            )
        else:
            return print_result(
                "获取配置历史",
                False,
                f"状态码：{response.status_code}, 错误：{response.json()}"
            )
    except Exception as e:
        return print_result("获取配置历史", False, f"错误：{e}")


def test_rollback_config(token):
    """测试回滚配置"""
    print_section("配置回滚")
    try:
        # 首先获取历史记录
        history_response = requests.get(
            f"{BASE_URL}/system/config/history?limit=5",
            headers={"Authorization": f"Bearer {token}"}
        )
        if history_response.status_code != 200:
            return print_result("获取历史记录（回滚用）", False, f"状态码：{history_response.status_code}")

        history = history_response.json().get("history", [])
        if len(history) < 2:
            return print_result("配置回滚", False, "历史记录不足 2 条，无法测试回滚")

        # 找到第一条 logging 配置历史记录
        logging_history_index = None
        for i, record in enumerate(history):
            if record.get("config_type") == "logging":
                logging_history_index = i
                break

        if logging_history_index is None:
            return print_result("配置回滚", False, "未找到 logging 配置历史记录")

        # 回滚到该版本
        rollback_response = requests.post(
            f"{BASE_URL}/system/config/rollback/logging/{logging_history_index}",
            headers={"Authorization": f"Bearer {token}"}
        )

        if rollback_response.status_code == 200:
            data = rollback_response.json()
            return print_result(
                "配置回滚",
                True,
                f"回滚成功：{data.get('message')}\n回滚后配置：{json.dumps(data.get('config', {}), indent=2, ensure_ascii=False)}"
            )
        else:
            return print_result(
                "配置回滚",
                False,
                f"状态码：{rollback_response.status_code}, 错误：{rollback_response.json()}"
            )
    except Exception as e:
        return print_result("配置回滚", False, f"错误：{e}")


def test_rollback_invalid_config(token):
    """测试回滚无效配置类型"""
    print_section("无效配置回滚")
    try:
        response = requests.post(
            f"{BASE_URL}/system/config/rollback/invalid_type/0",
            headers={"Authorization": f"Bearer {token}"}
        )
        return print_result(
            "无效配置类型回滚 (应返回 400)",
            response.status_code == 400,
            f"状态码：{response.status_code}, 响应：{response.json()}"
        )
    except Exception as e:
        return print_result("无效配置类型回滚", False, f"错误：{e}")


# ==================== 主测试流程 ====================

def run_all_tests():
    """运行所有测试"""
    print_section("配置热更新功能测试")
    print(f"目标地址：{BASE_URL}")
    print(f"管理员账号：{ADMIN_USERNAME}")

    results = []

    # 1. 未认证测试
    results.append(test_get_app_config_unauthorized())

    # 2. 获取 JWT Token
    token = get_jwt_token()
    if not token:
        print("\n[ERROR] 无法获取 JWT Token，后续测试无法进行")
        return

    # 3. 配置查询测试
    results.append(test_get_app_config(token))
    results.append(test_get_logging_config(token))
    results.append(test_get_retry_config(token))
    results.append(test_get_debug_config(token))
    results.append(test_get_storage_strategy_config(token))
    results.append(test_get_compress_config(token))

    # 4. 配置更新测试
    results.append(test_update_logging_config(token))
    results.append(test_update_retry_config(token))
    results.append(test_update_debug_config(token))
    results.append(test_update_storage_strategy_config(token))
    results.append(test_update_compress_config(token))

    # 5. 配置历史与回滚测试
    results.append(test_get_config_history(token))
    results.append(test_rollback_config(token))
    results.append(test_rollback_invalid_config(token))

    # 打印测试总结
    print_section("测试总结")
    passed = sum(results)
    total = len(results)
    print(f"通过：{passed}/{total}")
    if passed == total:
        print("[SUCCESS] 所有测试通过！")
    else:
        print(f"[WARNING] {total - passed} 个测试失败")


if __name__ == "__main__":
    run_all_tests()
