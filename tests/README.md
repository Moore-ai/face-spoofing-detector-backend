# 测试脚本使用指南

本文档说明如何运行后端服务的测试脚本。

## 测试脚本概述

| 脚本 | 说明 | 认证测试 | 推理测试 |
|------|------|----------|----------|
| `test_auth.py` | 认证系统完整测试 | ✅ | ❌ |
| `test_infer.py` | 推理功能测试 | ❌ | ✅ |
| `test_activation.py` | 激活码单元测试（离线） | ✅ | ❌ |
| `test_websocket_client.py` | WebSocket 客户端测试 | ❌ | ✅ |

## 前置条件

### 1. 启动服务

```bash
# 方式 1：直接运行
python main.py

# 方式 2：使用 uv
uv run python main.py

# 方式 3：开发模式（热重载）
uv run uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### 2. 安装测试依赖

```bash
# WebSocket 测试需要
pip install websocket-client

# 或使用 uv
uv pip install websocket-client
```

## 测试脚本说明

### test_auth.py - 认证系统测试

**测试内容**：
- 健康检查
- JWT Token 获取（管理员登录）
- 激活码管理（创建、列出、更新、禁用、删除）
- 激活码换取 API Key
- 推理端点认证（未认证拦截）
- WebSocket 连接测试

**运行方式**：

```bash
python tests/test_auth.py
```

**配置说明**：

在 `tests/test_auth.py` 中修改管理员凭据：

```python
ADMIN_USERNAME = "your_username"
ADMIN_PASSWORD = "your_password"
```

**输出示例**：

```
============================================================
  推理 API 认证与功能测试
============================================================

============================================================
  健康检查
============================================================

[健康检查] ✅ 通过
    响应：{'status': 'ok'}

============================================================
  未认证访问推理端点（预期失败）
============================================================

[未认证访问拦截] ✅ 通过
    状态码：401 (预期 401)

============================================================
  JWT Token 获取
============================================================

[JWT Token 获取] ✅ 通过
    Token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
    过期时间：86400 秒

...

============================================================
  测试汇总
============================================================

  通过：12/12

  🎉 所有测试通过!
```

---

### test_infer.py - 推理功能测试

**测试内容**：
- WebSocket 连接与 client_id 获取
- 单模态 RGB 推理
- 单模态 IR 推理
- 融合模态推理
- 任务状态查询
- WebSocket 进度推送

**运行方式**：

```bash
python tests/test_infer.py
```

**配置说明**：

在 `tests/test_infer.py` 中配置 API Key：

```python
API_KEY = "sk_your_api_key_here"  # 替换为实际 API Key
```

获取 API Key 的方法：

1. 运行 `test_auth.py` 获取 API Key
2. 或通过 API 手动获取：
   ```bash
   # 1. 获取 JWT Token
   curl -X POST http://127.0.0.1:8000/auth/token \
     -H "Content-Type: application/json" \
     -d '{"username": "admin", "password": "your_password"}'

   # 2. 创建激活码
   curl -X POST http://127.0.0.1:8000/auth/activation-codes \
     -H "Authorization: Bearer YOUR_JWT_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"name": "test", "max_uses": 10}'

   # 3. 激活码换取 API Key
   curl -X POST http://127.0.0.1:8000/auth/activate \
     -H "Content-Type: application/json" \
     -d '{"code": "ACT-XXXXX-YYYYY"}'
   ```

**输出示例**：

```
============================================================
  推理功能测试
============================================================

============================================================
  WebSocket 连接测试
============================================================

[WebSocket 连接] ✅ 通过
    Client ID: 550e8400-e29b-41d4-a716-446655440000

============================================================
  单模态 RGB 推理测试
============================================================

[单模态 RGB 推理请求] ✅ 通过
    Task ID: task-uuid-123
    消息：单模态检测任务已创建，正在后台处理

============================================================
  WebSocket 进度推送测试
============================================================

  收到消息类型：progress_update
    进度：33.33%
    消息：完成第 1/3 张图片的推理

...

============================================================
  测试汇总
============================================================

  通过：6/6

  🎉 所有推理测试通过!
```

---

### test_activation.py - 激活码单元测试

**测试内容**：
- 模块导入测试
- 激活码生成
- 激活码验证
- 激活码换取 API Key
- 激活码禁用/删除

**运行方式**：

```bash
python tests/test_activation.py
```

**说明**：此脚本测试内存中的激活码功能，不需要服务运行。

---

### test_websocket_client.py - WebSocket 客户端测试

**测试内容**：
- WebSocket 连接建立
- 进度消息接收
- 任务完成通知

**运行方式**：

```bash
python tests/test_websocket_client.py
```

---

## 测试流程图

```
┌─────────────────────────────────────────────────────────┐
│                   测试准备                               │
│  1. 启动服务：python main.py                            │
│  2. 安装依赖：pip install websocket-client              │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│              运行认证测试（test_auth.py）                │
│  1. 健康检查 ✓                                          │
│  2. 获取 JWT Token ✓                                    │
│  3. 创建激活码 ✓                                        │
│  4. 激活码换取 API Key ✓                                │
│  5. 未认证访问拦截 ✓                                    │
│  6. WebSocket 连接 ✓                                    │
└─────────────────────────────────────────────────────────┘
                         ↓
              复制获取的 API Key
                         ↓
┌─────────────────────────────────────────────────────────┐
│              运行推理测试（test_infer.py）               │
│  1. WebSocket 连接 ✓                                    │
│  2. 单模态 RGB 推理 ✓                                    │
│  3. 单模态 IR 推理 ✓                                     │
│  4. 融合模态推理 ✓                                      │
│  5. 任务状态查询 ✓                                      │
│  6. WebSocket 进度推送 ✓                                │
└─────────────────────────────────────────────────────────┘
```

---

## 常见问题

### 1. 连接被拒绝

**错误**：`ConnectionRefusedError` 或 `Max retries exceeded`

**解决**：
- 确保服务已启动并运行在 `http://127.0.0.1:8000`
- 检查端口是否被占用

### 2. 认证失败

**错误**：`401 Unauthorized`

**解决**：
- 检查 API Key 是否正确
- 检查 JWT Token 是否过期
- 确认激活码是否有效

### 3. WebSocket 连接失败

**错误**：`WebSocketConnectionClosedException`

**解决**：
- 检查 WebSocket URL 是否正确
- 确认 API Key 通过查询参数传递
- 检查防火墙设置

### 4. 任务超时

**错误**：`任务超时（等待 10 秒）`

**解决**：
- 增加 `test_infer.py` 中的 `max_retries` 值
- 检查推理服务是否正常
- 确认模型文件是否存在

### 5. 模块导入错误

**错误**：`ModuleNotFoundError: No module named 'websocket'`

**解决**：
```bash
pip install websocket-client
# 或
uv pip install websocket-client
```

---

## 自动化测试建议

### 创建测试脚本（推荐）

```bash
#!/bin/bash
# run_tests.sh

echo "启动服务..."
python main.py &
SERVER_PID=$!
sleep 3

echo "运行认证测试..."
python tests/test_auth.py

echo "运行推理测试..."
python tests/test_infer.py

echo "清理..."
kill $SERVER_PID
```

### 使用 pytest 运行

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_auth.py::test_health

# 带覆盖率运行
pytest --cov=.
```

---

## 相关文件

- `tests/test_auth.py` - 认证系统测试
- `tests/test_infer.py` - 推理功能测试
- `tests/test_activation.py` - 激活码单元测试
- `tests/test_websocket_client.py` - WebSocket 测试
- `AUTH.md` - 认证系统使用指南
- `README.md` - 项目文档
