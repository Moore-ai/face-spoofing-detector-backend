"""
激活码管理与测试交互脚本

功能：
1. 创建激活码（需要管理员 JWT Token）
2. 测试激活码是否有效（通过 /auth/activate 端点）

使用前请确保服务已启动：
    python main.py

运行脚本：
    python tests/create_act.py
"""

import requests

BASE_URL = "http://127.0.0.1:8000"

# 测试配置
ADMIN_USERNAME = "Moore-ai"
ADMIN_PASSWORD = "Moore20060810"


def print_section(title: str):
    """打印分割线"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_menu():
    """打印菜单"""
    print("\n" + "-" * 40)
    print("  激活码管理菜单")
    print("-" * 40)
    print("  1. 创建激活码")
    print("  2. 测试激活码是否有效")
    print("  3. 列出所有激活码")
    print("  4. 禁用激活码")
    print("  5. 删除激活码")
    print("  0. 退出")
    print("-" * 40)


def get_admin_token() -> str | None:
    """获取管理员 JWT Token"""
    try:
        response = requests.post(
            f"{BASE_URL}/auth/token",
            json={"username": ADMIN_USERNAME, "password": ADMIN_PASSWORD}
        )
        if response.status_code == 200:
            data = response.json()
            return data["access_token"]
        else:
            print(f"[错误] 获取 Token 失败：{response.json()}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"[错误] 请求失败：{e}")
        return None


def create_activation_code(token: str) -> str | None:
    """创建激活码"""
    print("\n--- 创建激活码 ---")

    name = input("用户名 (name): ").strip()
    if not name:
        name = "tmp"

    try:
        max_uses = int(input("最大使用次数 (默认 10): ").strip() or "10")
        expires_in_hours = int(input("过期时间 (小时，默认 24): ").strip() or "24")
        priority = int(input("任务优先级 0-100 (默认 50, VIP 建议 80+): ").strip() or "50")

        # 验证优先级范围
        if priority < 0 or priority > 100:
            print(f"[警告] 优先级超出范围 (0-100)，已自动调整为 50")
            priority = 50
    except ValueError:
        print("[错误] 输入无效，使用默认值")
        max_uses = 10
        expires_in_hours = 24
        priority = 50

    try:
        response = requests.post(
            f"{BASE_URL}/auth/activation-codes",
            json={
                "name": name,
                "max_uses": max_uses,
                "expires_in_hours": expires_in_hours,
                "priority": priority,
            },
            headers={"Authorization": f"Bearer {token}"}
        )

        if response.status_code == 200:
            data = response.json()
            print(f"\n[成功] 激活码创建成功!")
            print(f"  激活码：{data['code']}")
            print(f"  用户：{data['user_id']}")
            print(f"  最大使用次数：{data['max_uses']}")
            print(f"  优先级：{data.get('priority', 0)} (范围 0-100，值越大优先级越高)")
            print(f"  过期时间：{data.get('expires_at', '永不过期')}")
            return data["code"]
        else:
            print(f"[错误] 创建失败：{response.json()}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"[错误] 请求失败：{e}")
        return None


def test_activation_code(code: str | None = None) -> bool:
    """测试激活码是否有效"""
    print("\n--- 测试激活码 ---")

    if code is None:
        code = input("请输入激活码：").strip()

    if not code:
        print("[错误] 激活码不能为空")
        return False

    try:
        response = requests.post(
            f"{BASE_URL}/auth/activate",
            json={"code": code}
        )

        if response.status_code == 200:
            data = response.json()
            print(f"\n[成功] 激活码有效!")
            print(f"  API Key: {data['api_key'][:30]}...")
            print(f"  过期时间：{data.get('expires_at', '永不过期')}")
            return True
        elif response.status_code == 400:
            error = response.json().get("detail", "未知错误")
            print(f"\n[失败] 激活码无效或已过期：{error}")
            return False
        elif response.status_code == 404:
            print(f"\n[失败] 激活码不存在")
            return False
        else:
            print(f"[错误] 请求失败，状态码：{response.status_code}")
            print(f"  响应：{response.json()}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"[错误] 请求失败：{e}")
        return False


def list_activation_codes(token: str):
    """列出所有激活码"""
    print("\n--- 所有激活码 ---")

    try:
        response = requests.get(
            f"{BASE_URL}/auth/activation-codes",
            headers={"Authorization": f"Bearer {token}"}
        )

        if response.status_code == 200:
            data = response.json()
            codes = data.get("activation_codes", [])

            if not codes:
                print("  暂无激活码")
                return

            print(f"\n激活码总数：{len(codes)}\n")
            for i, code_info in enumerate(codes, 1):
                status = "有效" if code_info.get("is_active") else "已禁用"
                expires = code_info.get("expires_at", "永不过期")
                print(f"  {i}. {code_info['code']}")
                print(f"     用户：{code_info['user_id']}")
                print(f"     状态：{status}")
                print(f"     使用：{code_info['current_uses']}/{code_info['max_uses']}")
                print(f"     过期：{expires}")
                print()
        else:
            print(f"[错误] 获取失败：{response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"[错误] 请求失败：{e}")


def deactivate_activation_code(token: str):
    """禁用激活码"""
    print("\n--- 禁用激活码 ---")

    code = input("请输入要禁用的激活码：").strip()
    if not code:
        print("[错误] 激活码不能为空")
        return

    try:
        response = requests.post(
            f"{BASE_URL}/auth/activation-codes/{code}/deactivate",
            headers={"Authorization": f"Bearer {token}"}
        )

        if response.status_code == 200:
            print(f"\n[成功] 激活码已禁用")
        elif response.status_code == 404:
            print(f"\n[失败] 激活码不存在")
        else:
            print(f"[错误] 操作失败：{response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"[错误] 请求失败：{e}")


def delete_activation_code(token: str):
    """删除激活码"""
    print("\n--- 删除激活码 ---")

    code = input("请输入要删除的激活码：").strip()
    if not code:
        print("[错误] 激活码不能为空")
        return

    confirm = input(f"确认删除激活码 {code}? (y/N): ").strip().lower()
    if confirm != "y":
        print("已取消")
        return

    try:
        response = requests.delete(
            f"{BASE_URL}/auth/activation-codes/{code}",
            headers={"Authorization": f"Bearer {token}"}
        )

        if response.status_code == 200:
            print(f"\n[成功] 激活码已删除")
        elif response.status_code == 404:
            print(f"\n[失败] 激活码不存在")
        else:
            print(f"[错误] 操作失败：{response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"[错误] 请求失败：{e}")


def main():
    """主函数"""
    print_section("激活码管理与测试工具")

    # 尝试获取管理员 Token
    print("\n正在获取管理员 Token...")
    token = get_admin_token()

    if not token:
        print("\n[警告] 无法获取管理员 Token，请检查:")
        print(f"  1. 服务是否已启动 (http://127.0.0.1:8000)")
        print(f"  2. 管理员账号密码是否正确")
        print(f"     当前配置：{ADMIN_USERNAME} / {ADMIN_PASSWORD}")
        print("\n部分功能将不可用 (需要 Token 的操作)")

    created_codes = []  # 记录本次会话创建的激活码

    while True:
        print_menu()
        choice = input("请选择操作 (0-5): ").strip()

        if choice == "0":
            print("\n再见!")
            break

        elif choice == "1":
            if not token:
                print("\n[错误] 需要管理员 Token，请先登录")
                token = get_admin_token()
                if not token:
                    continue
            code = create_activation_code(token)
            print(code)
            if code:
                created_codes.append(code)

        elif choice == "2":
            # 如果有刚创建的激活码，优先测试
            if created_codes:
                print(f"\n发现未测试的激活码：{created_codes[-1]}")
                test = input("是否测试最后一个创建的激活码？(Y/n): ").strip().lower()
                if test != "n":
                    test_activation_code(created_codes[-1])
                    continue
            test_activation_code()

        elif choice == "3":
            if not token:
                print("\n[错误] 需要管理员 Token")
                token = get_admin_token()
                if not token:
                    continue
            list_activation_codes(token)

        elif choice == "4":
            if not token:
                print("\n[错误] 需要管理员 Token")
                token = get_admin_token()
                if not token:
                    continue
            deactivate_activation_code(token)

        elif choice == "5":
            if not token:
                print("\n[错误] 需要管理员 Token")
                token = get_admin_token()
                if not token:
                    continue
            delete_activation_code(token)

        else:
            print("\n[错误] 无效选择，请输入 0-5")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序已中断")
    except Exception as e:
        print(f"\n程序异常：{e}")
        import traceback
        traceback.print_exc()
