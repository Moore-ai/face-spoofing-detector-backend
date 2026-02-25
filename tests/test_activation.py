"""
认证系统完整测试脚本
"""

import sys

print('=' * 60)
print('认证系统功能测试')
print('=' * 60)

# 1. 测试模块导入
print('\n[1] 测试模块导入...')
from util.auth import (
    activate_with_code,
    create_activation_code,
    validate_activation_code,
    get_all_activation_codes,
    deactivate_activation_code,
    delete_activation_code,
    ACTIVATION_CODES_DB,
    API_KEYS_DB,
)
from schemas.activation import (
    ActivateRequest,
    ActivateResponse,
    ActivationCodeCreate,
    ActivationCodeInfo,
)
print('    [OK] 模块导入成功')

# 2. 测试生成激活码
print('\n[2] 测试生成激活码...')
code_info = create_activation_code(name='test_user', max_uses=5, expires_in_hours=24)
print(f'    激活码：{code_info.code}')
print(f'    用户 ID: {code_info.user_id}')
print(f'    最大使用：{code_info.max_uses}')
print('    [OK] 生成成功')

# 3. 测试验证激活码
print('\n[3] 测试验证激活码...')
result = validate_activation_code(code_info.code)
assert result is not None, "激活码验证失败"
print(f'    验证结果：{result.code}')
print('    [OK] 验证成功')

# 4. 测试激活码换取 API Key
print('\n[4] 测试激活码换取 API Key...')
result = activate_with_code(code_info.code)
assert result is not None, "激活码换取 API Key 失败"
api_key, code_info = result
print(f'    API Key: {api_key[:20]}...')
print(f'    使用次数：{code_info.current_uses}')
print('    [OK] 换取成功')

# 5. 测试获取所有激活码
print('\n[5] 测试获取所有激活码...')
all_codes = get_all_activation_codes()
print(f'    激活码总数：{len(all_codes)}')
print('    [OK] 获取成功')

# 6. 测试禁用激活码
print('\n[6] 测试禁用激活码...')
success = deactivate_activation_code(code_info.code)
assert success is True, "禁用激活码失败"
result = validate_activation_code(code_info.code)
assert result is None, "禁用后应该无效"
print(f'    禁用后验证：无效 (预期)')
print('    [OK] 禁用成功')

# 7. 测试应用启动
print('\n[7] 测试应用启动...')
from main import app
from fastapi.routing import APIRoute
routes = [r.path for r in app.routes if isinstance(r, APIRoute)]
assert '/auth/activate' in routes, "缺少激活码端点"
assert '/auth/activation-codes' in routes, "缺少激活码管理端点"
print(f'    路由数量：{len(app.routes)}')
print('    [OK] 应用启动成功')

print('\n' + '=' * 60)
print('所有测试通过!')
print('=' * 60)
