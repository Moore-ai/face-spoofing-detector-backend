# 人脸活体检测后端服务

基于 FastAPI 的人脸活体检测（Face Anti-Spoofing）推理服务，支持单模态和融合模态图像检测，提供实时进度推送和异步任务管理。

**部署架构**：
- **前端**：Tauri 桌面应用（React + Rust），部署在用户端
- **后端**：Python FastAPI 推理服务，集中部署
- **多客户端**：支持多个桌面客户端同时连接后端服务

**认证系统**：
- 采用**激活码模式**，用户输入激活码换取 API Key
- 支持 API Key 认证（客户端请求）和 JWT Token（管理后台）
- 速率限制防止 API 滥用

## 功能特性

- **多模态支持**：支持单模态（RGB/IR）和融合模态（RGB+IR）检测
- **多模型格式**：支持 PyTorch (.pth)、ONNX (.onnx)、RKNN (.rknn) 模型
- **RESTful API**：基于 FastAPI 构建，支持异步处理
- **WebSocket 实时推送**：实时推送任务进度更新和完成通知
- **异步任务管理**：后台批量处理，支持任务状态查询
- **进度追踪服务**：实时统计任务进度、结果数量和分类
- **灵活配置**：通过环境变量或 `.env` 文件配置所有参数
- **完整日志**：详细的日志记录，仅输出到文件
- **请求审计**：关键操作审计日志，JSON 格式便于 ELK 采集
- **全局异常处理**：统一的错误处理和日志记录机制
- **依赖注入架构**：模块化设计，易于测试和维护
- **认证与授权**：激活码模式，支持多客户端管理和权限控制
- **速率限制**：防止 API 滥用，保护服务稳定性
- **重试机制**：推理失败自动重试，支持指数退避策略
- **部分失败支持**：批量任务支持部分失败状态，错误可追踪
- **任务优先级调度**：基于优先级的任务队列，高优先级任务优先执行
- **图片存储管理**：支持本地/S3 存储，提供图片上传、查询、删除、统计 API
- **存储配额管理**：防止存储溢出，支持手动清理过期图片
- **Prometheus 监控**：暴露丰富的指标数据，支持 QPS、延迟、成功率等统计
- **健康检查增强**：详细健康检查，支持模型、GPU、磁盘、数据库、存储状态监控
- **配置热更新**：支持运行时动态修改日志、重试、调试、存储策略等配置

## 技术栈

- **Web 框架**：FastAPI（支持异步、WebSocket）
- **深度学习**：PyTorch、ONNX Runtime
- **图像处理**：OpenCV、Pillow、NumPy
- **模型架构**：MobileNetV2（可扩展）
- **实时通信**：WebSocket（进度推送）
- **数据验证**：Pydantic（类型安全）
- **配置管理**：pydantic-settings（环境变量）
- **日志系统**：Python logging（文件 + 控制台）
- **测试框架**：pytest（单元测试和集成测试）
- **API 文档**：Swagger UI、ReDoc（自动生成）
- **依赖管理**：uv（推荐）或 pip
- **监控指标**：Prometheus 指标暴露

## 快速开始

### 1. 安装依赖

```bash
# 使用 uv（推荐）
uv sync

# 或使用 pip
uv pip install -e .
```

### 2. 使用自动化脚本配置环境（推荐）

我们提供了针对不同操作系统的自动化配置脚本，可一键完成环境配置：

#### Windows:
```cmd
scripts/setup_env_windows.bat
```

#### Linux/macOS:
```bash
# 使脚本可执行
chmod +x scripts/setup_env_*.sh

# 运行对应系统的脚本
./scripts/setup_env_linux.sh    # Linux
./scripts/setup_env_macos.sh    # macOS
```

脚本会自动：
- 检查 Python 版本
- 安装 uv 包管理器（如未安装）
- 创建虚拟环境
- 安装项目依赖
- 从.env.example 创建.env 配置文件

### 3. 手动配置环境（可选）

如果您不想使用自动化脚本，也可以手动配置环境：

```bash
# 复制示例配置文件
cp .env.example .env

# 编辑 .env 文件，修改为你的配置
notepad .env  # Windows
# 或
vim .env      # Linux/Mac
```

### 4. 准备模型

将训练好的模型文件放入 `models/` 目录：

```
models/
├── single_model.pth    # 单模态模型
└── fusion_model.pth    # 融合模态模型
```

支持的模型格式：
- `.pth` - PyTorch 模型
- `.onnx` - ONNX 模型
- `.rknn` - RKNN 模型（嵌入式设备）

### 5. 启动服务

```bash
# 方式 1：直接运行（生产模式）
python main.py

# 方式 2：使用 uv
uv run python main.py

# 方式 3：开发模式（支持热重载）
uv run uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

服务启动后，访问：
- API 文档：http://127.0.0.1:8000/docs
- 健康检查：http://127.0.0.1:8000/health
- ReDoc 文档：http://127.0.0.1:8000/redoc

## 配置说明

所有配置项均可在 `.env` 文件中修改，或通过环境变量覆盖。

### 模型配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `MODEL_SINGLE_PATH` | `models/single_model.pth` | 单模态模型路径 |
| `MODEL_FUSION_PATH` | `models/fusion_model.pth` | 融合模态模型路径 |
| `INPUT_SIZE` | `112` | 模型输入尺寸（正方形） |

### 服务器配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `HOST` | `127.0.0.1` | 监听地址，`0.0.0.0` 允许外部访问 |
| `PORT` | `8000` | 监听端口 |

### 推理配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `DEVICE` | `cuda` | 推理设备，`cuda` 或 `cpu` |

### 日志配置（新增）

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `LOG_LEVEL` | `INFO` | 日志级别：`DEBUG`/`INFO`/`WARNING`/`ERROR`/`CRITICAL` |
| `LOG_JSON_FORMAT` | `false` | 是否使用 JSON 格式日志 |
| `LOG_TO_CONSOLE` | `true` | 是否输出到控制台（开发环境建议开启） |
| `LOG_REQUEST_BODY` | `false` | 是否记录请求体（生产环境建议关闭） |
| `LOG_RESPONSE_BODY` | `false` | 是否记录响应体（生产环境建议关闭） |
| `AUDIT_LOG_ENABLED` | `true` | 是否启用审计日志 |

**日志文件位置**：
- 普通日志：`logs/{timestamp}.log`
- 审计日志：`logs/{timestamp}_audit.log`（JSON 格式）

**注意**：日志默认输出到控制台和文件。

### 认证配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `JWT_SECRET_KEY` | `your-secret-key...` | JWT 密钥（生产环境务必修改） |
| `JWT_EXPIRATION_HOURS` | `24` | JWT Token 过期时间（小时） |
| `ADMIN_USERNAME` | `admin` | 管理员账户（用于管理后台） |
| `ADMIN_PASSWORD` | `your-admin-password...` | 管理员密码（生产环境务必修改） |
| `DEFAULT_ACTIVATION_CODE` | `ACT-DEV-DEFAULT-KEY` | 默认激活码（仅开发环境） |

### 调试模式配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `DEBUG_MODE` | `false` | 启用调试模式（无需模型文件） |
| `DEBUG_DELAY_PER_IMAGE` | `0.5` | 单模态每张图像延迟（秒） |
| `DEBUG_DELAY_PER_PAIR` | `0.8` | 融合模态每对图像延迟（秒） |
| `DEBUG_FAILURE_RATE` | `0.0` | 模拟失败率（用于测试重试机制） |

### 重试配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `RETRY_ENABLED` | `true` | 是否启用推理失败重试 |
| `RETRY_MAX_ATTEMPTS` | `3` | 最大重试次数（包含首次尝试） |
| `RETRY_DELAY_SECONDS` | `1.0` | 重试基础延迟（秒） |
| `RETRY_EXPONENTIAL_BACKOFF` | `true` | 是否启用指数退避 |
| `RETRY_MAX_DELAY_SECONDS` | `10.0` | 最大重试延迟（秒） |

### 数据库配置（新增）

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `DATABASE_URL` | `sqlite:///./db/detections.db` | 数据库连接 URL |

**支持的数据库类型**：
- SQLite（默认）：`sqlite:///./db/detections.db`
- PostgreSQL：`postgresql://user:password@host:port/dbname`

**注意**：
- 首次启动时会自动创建数据库表
- 数据库文件存储在 `db/` 目录
- 历史记录会在推理完成后自动保存

### 图片存储配置（新增）

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `STORAGE_TYPE` | `local` | 存储类型：`local`（本地）或 `s3`（对象存储） |
| `STORAGE_LOCAL_PATH` | `storage/images` | 本地存储基础路径 |
| `STORAGE_QUOTA_BYTES` | `` | 存储配额（字节），空表示无限制 |
| `STORAGE_RETENTION_DAYS` | `30` | 图片保留天数（清理时使用） |
| `STORAGE_AUTO_SAVE` | `true` | 是否自动保存图片（推理完成后） |
| `STORAGE_SAVE_STRATEGY` | `error_only` | 存储策略：`never`/`always`/`error_only`/`smart` |
| `STORAGE_SAVE_ERROR_RATE` | `1.0` | 错误结果保存率（智能策略） |
| `STORAGE_SAVE_FAKE_RATE` | `0.1` | 伪造样本保存率（智能策略） |
| `STORAGE_SAVE_REAL_RATE` | `0.01` | 真实样本保存率（智能策略） |
| `STORAGE_SAVE_LOW_CONFIDENCE_THRESHOLD` | `0.6` | 低置信度阈值（智能策略） |
| `STORAGE_MAX_PER_TASK` | `10` | 每任务最大保存数 |
| `IMAGE_COMPRESS_ENABLED` | `true` | 是否启用图片压缩 |
| `IMAGE_COMPRESS_QUALITY` | `75` | 压缩质量（1-100） |
| `IMAGE_COMPRESS_TYPE` | `opencv` | 压缩器类型：`opencv`/`pillow`/`resize` |
| `IMAGE_COMPRESS_MAX_WIDTH` | `` | 最大宽度（resize 压缩器） |
| `IMAGE_COMPRESS_MAX_HEIGHT` | `` | 最大高度（resize 压缩器） |

**S3 对象存储配置**（可选，使用本地存储时无需配置）：

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `S3_BUCKET` | `` | S3 桶名 |
| `S3_REGION` | `us-east-1` | AWS 区域 |
| `S3_ENDPOINT_URL` | `` | S3 兼容端点（用于 MinIO、阿里云 OSS 等） |
| `S3_ACCESS_KEY` | `` | S3 访问密钥 |
| `S3_SECRET_KEY` | `` | S3 密钥 |

**注意**：
- 本地存储采用两级目录结构：`storage/images/{prefix1}/{prefix2}/{image_id}.bin`
- 存储配额用于防止存储溢出
- 清理过期图片需要手动调用 API 端点

### Prometheus 监控配置（新增）

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `PROMETHEUS_ENABLED` | `true` | 是否启用 Prometheus 指标收集 |
| `PROMETHEUS_METRICS_PATH` | `/metrics` | Prometheus 指标端点路径 |

**收集的指标**：
- `http_requests_total` - HTTP 请求总数（按方法、端点、状态码分类）
- `http_request_latency_seconds` - HTTP 请求延迟直方图
- `inference_total` - 推理总数（按模态、结果分类）
- `inference_latency_milliseconds` - 推理延迟直方图
- `task_total` - 任务总数（按模式、状态分类）
- `task_success_rate` - 任务成功率
- `websocket_active_connections` - 活跃 WebSocket 连接数
- `api_key_usage_total` - API Key 使用量
- `activation_code_usage_total` - 激活码使用量
- `rate_limit_total` - 速率限制触发次数
- `error_total` - 错误总数
- `system_info` - 系统信息

**使用示例**：
```bash
# 访问/metrics 端点获取 Prometheus 格式指标
curl http://127.0.0.1:8000/metrics

# Prometheus 配置文件（prometheus.yml）
scrape_configs:
  - job_name: 'heli_code_backend'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:8000']
```

### 配置热更新（新增）

系统支持运行时动态修改配置，无需重启服务。

| 端点 | 方法 | 说明 | 认证要求 |
|------|------|------|----------|
| `/system/config` | GET | 获取所有配置组 | **需要 JWT（管理员）** |
| `/system/config/logging` | GET/PUT | 日志配置查询/更新 | **需要 JWT（管理员）** |
| `/system/config/retry` | GET/PUT | 重试配置查询/更新 | **需要 JWT（管理员）** |
| `/system/config/debug` | GET/PUT | 调试配置查询/更新 | **需要 JWT（管理员）** |
| `/system/config/storage-strategy` | GET/PUT | 存储策略配置查询/更新 | **需要 JWT（管理员）** |
| `/system/config/compress` | GET/PUT | 图片压缩配置查询/更新 | **需要 JWT（管理员）** |
| `/system/config/history` | GET | 获取配置变更历史 | **需要 JWT（管理员）** |
| `/system/config/rollback/{type}/{index}` | POST | 回滚配置到历史版本 | **需要 JWT（管理员）** |

**支持热更新的配置类型**：
| 配置类型 | 配置项 | 生效时机 |
|----------|--------|----------|
| 日志配置 | `log_level`, `log_to_console`, `log_json_format` 等 | 立即生效 |
| 重试配置 | `retry_enabled`, `retry_max_attempts` 等 | 下次推理时应用 |
| 调试配置 | `debug_mode`, `debug_failure_rate` 等 | 下次推理时应用 |
| 存储策略 | `storage_auto_save`, `storage_save_strategy` 等 | 下次保存图片时应用 |
| 压缩配置 | `compress_enabled`, `compress_quality` 等 | 下次压缩图片时应用 |

**使用示例**：
```bash
# 获取当前配置
curl http://127.0.0.1:8000/system/config \
  -H "Authorization: Bearer {jwt_token}"

# 更新日志级别为 DEBUG
curl -X PUT http://127.0.0.1:8000/system/config/logging \
  -H "Authorization: Bearer {jwt_token}" \
  -H "Content-Type: application/json" \
  -d '{"log_level": "DEBUG"}'

# 获取配置历史
curl "http://127.0.0.1:8000/system/config/history?limit=5" \
  -H "Authorization: Bearer {jwt_token}"

# 回滚配置到历史版本 #0
curl -X POST http://127.0.0.1:8000/system/config/rollback/logging/0 \
  -H "Authorization: Bearer {jwt_token}"
```

**注意**：
- 配置变更仅对当前运行实例有效，不会写入 `.env` 文件
- 服务重启后配置恢复到 `.env` 中的值
- 配置历史记录存储在内存中，服务重启后清空

## 认证系统使用指南

### 激活码模式流程

```
1. 管理员生成激活码 → 2. 用户输入激活码 → 3. 换取 API Key → 4. 后续请求携带 API Key
```

### 速率限制策略

| 端点类型 | 限制 | 说明 |
|----------|------|------|
| 认证端点（`/auth/*`） | 10 次/分钟 | 防止暴力破解 |
| 推理端点（`/infer/*`） | 30 次/分钟 | 防止 API 滥用 |
| 其他端点 | 60 次/分钟 | 默认限制 |

响应头中包含限流信息：
- `X-RateLimit-Limit`: 总请求数限制
- `X-RateLimit-Remaining`: 剩余请求数
- `X-RateLimit-Reset`: 窗口重置时间（秒）
- `Retry-After`: 重试等待时间（秒）

### 1. 管理员生成激活码

```bash
# 1. 获取 JWT Token（管理员登录）
curl -X POST http://127.0.0.1:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'

# 2. 使用 Token 生成激活码
# 注意：max_uses 是必需字段，必须 >= 1
curl -X POST http://127.0.0.1:8000/auth/activation-codes \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "user_001",
    "max_uses": 100,
    "expires_in_hours": 720
  }'
```

**参数说明**：
- `name`: 激活码名称/描述（可选）
- `max_uses`: **最大使用次数（必需，必须 >= 1）**
- `expires_in_hours`: 过期时间（小时，可选，None 表示永不过期）
- `permissions`: 权限列表（可选，默认 ["read", "write"]）

### 2. 用户激活客户端

```bash
curl -X POST http://127.0.0.1:8000/auth/activate \
  -H "Content-Type: application/json" \
  -d '{"code": "ACT-XXXXX-YYYYY"}'
```

响应：
```json
{
  "api_key": "sk_xxxxx...",
  "message": "激活成功"
}
```

### 3. 使用 API Key 访问推理端点

```bash
# 单模态推理
curl -H "X-API-Key: sk_xxxxx..." \
  -H "X-Client-ID: your-client-id-from-websocket" \
  -X POST http://127.0.0.1:8000/infer/single \
  -H "Content-Type: application/json" \
  -d '{"mode": "single", "modality": "rgb", "images": [...]}'

# 融合模态推理
curl -H "X-API-Key: sk_xxxxx..." \
  -H "X-Client-ID: your-client-id-from-websocket" \
  -X POST http://127.0.0.1:8000/infer/fusion \
  -H "Content-Type: application/json" \
  -d '{"mode": "fusion", "pairs": [{"rgb": "base64_rgb", "ir": "base64_ir"}]}'

# 查询任务状态
curl -H "X-API-Key: sk_xxxxx..." \
  http://127.0.0.1:8000/infer/task/{task_id}
```

### 审计日志

系统自动记录关键操作的审计日志：

**认证事件**：
- `auth.activate_code` - 激活码换取 API Key
- `auth.jwt_login` - JWT 登录
- `auth.failed` - 认证失败

**管理事件**：
- `activation_code.created` - 创建激活码
- `activation_code.deleted` - 删除激活码
- `activation_code.updated` - 更新激活码
- `activation_code.deactivated` - 禁用激活码

**推理事件**：
- `inference.single` - 单模态推理
- `inference.fusion` - 融合模态推理
- `inference.task_query` - 任务状态查询

**系统事件**：
- `system.rate_limited` - 被速率限制
- `system.error` - 系统错误

审计日志存储在 `logs/{timestamp}_audit.log`，采用 JSON 格式，便于日志分析工具采集。

## API 接口

### 认证相关端点

| 端点 | 方法 | 说明 | 认证要求 |
|------|------|------|----------|
| `/auth/token` | POST | 获取 JWT Token（管理员登录） | 无 |
| `/auth/activate` | POST | 激活码换取 API Key | 无 |
| `/auth/me` | GET | 获取当前用户信息 | 需要认证 |
| `/auth/refresh` | POST | 刷新 Token | 需要 Token |
| `/auth/activation-codes` | GET | 列出所有激活码 | 需要 admin |
| `/auth/activation-codes` | POST | 创建新激活码 | 需要 admin |
| `/auth/activation-codes/{code}` | PUT | 更新激活码配置 | 需要 admin |
| `/auth/activation-codes/{code}` | DELETE | 删除激活码 | 需要 admin |
| `/auth/activation-codes/{code}/deactivate` | POST | 禁用激活码 | 需要 admin |

**注意**：
- `max_uses` 是创建激活码时的必需字段，必须 >= 1
- 激活码达到使用次数上限后自动失效

### 推理相关端点

| 端点 | 方法 | 说明 | 认证要求 |
|------|------|------|----------|
| `/infer/single` | POST | 单模态检测 | **需要 API Key** |
| `/infer/fusion` | POST | 融合模态检测 | **需要 API Key** |
| `/infer/ws` | WebSocket | 实时进度推送 | 内置认证 |
| `/infer/task/{task_id}` | GET | 查询任务状态 | **需要 API Key** |
| `/infer/task/{task_id}` | DELETE | 取消任务 | **需要 API Key** |
| `/infer/tasks` | GET | 获取当前客户端的任务列表 | **需要 API Key** |
| `/infer/queue/status` | GET | 获取任务队列状态 | **需要 JWT（管理员）** |

**任务优先级调度**（新增）：
- 所有推理任务通过优先级任务调度器执行
- 高优先级任务优先处理（VIP 客户）
- 优先级范围：0-100（值越大优先级越高）
- 通过激活码设置 API Key 的优先级

### 健康检查端点

| 端点 | 方法 | 说明 | 认证要求 |
|------|------|------|----------|
| `/health` | GET | 简单健康检查 | 无 |
| `/health/detailed` | GET | 详细健康检查 | **需要 JWT（管理员）** |

**详细健康检查返回字段**：
- `status`: 整体状态（healthy/degraded/unhealthy）
- `models`: 模型文件状态列表
  - `name`: 模型名称
  - `path`: 模型文件路径
  - `exists`: 是否存在
  - `file_size_mb`: 文件大小（MB）
- `gpu`: GPU 状态
  - `available`: 是否可用
  - `device_name`: 设备名称
  - `memory_total_mb`: 显存总量
  - `memory_used_mb`: 显存已用
  - `memory_free_mb`: 显存可用
  - `memory_used_percent`: 显存使用率
- `disk`: 磁盘状态
  - `total_gb`: 总容量
  - `used_gb`: 已使用
  - `free_gb`: 可用空间
  - `used_percent`: 使用率
- `database`: 数据库状态
  - `connected`: 是否已连接
  - `error_message`: 错误信息（如有）
- `storage`: 存储服务状态
  - `initialized`: 是否已初始化
  - `storage_type`: 存储类型
  - `quota_bytes`: 存储配额
- `healthy_components`: 健康组件列表
- `unhealthy_components`: 不健康组件列表

### 历史记录相关端点（新增）

| 端点 | 方法 | 说明 | 认证要求 |
|------|------|------|----------|
| `/history` | GET | 查询历史记录（分页） | **需要 API Key 或 JWT** |
| `/history/stats` | GET | 获取统计信息 | **需要 API Key 或 JWT** |
| `/history/task/{task_id}` | GET | 查询单个任务详情 | **需要 API Key 或 JWT** |
| `/history` | DELETE | 删除历史记录 | **需要 JWT（管理员）** |

**历史记录查询参数**：
- `client_id` - 按客户端 ID 过滤（可选）
- `mode` - 按模式过滤：`single` 或 `fusion`（可选）
- `status` - 按状态过滤：`completed`, `partial_failure`, `failed`（可选，逗号分隔）
- `days` - 查询最近 N 天的记录（可选）
- `page` - 页码，从 1 开始（默认 1）
- `page_size` - 每页数量，最大 100（默认 20）

**统计信息响应**：
- `total_tasks` - 总任务数
- `total_inferences` - 总推理次数
- `total_real` - 真实人脸计数
- `total_fake` - 伪造人脸计数
- `total_errors` - 错误计数
- `success_rate` - 成功率（%）
- `avg_processing_time_ms` - 平均处理时间（毫秒）

### 智能图片存储管理（新增）- 方案 A：服务端控制策略

**架构说明**：
- **服务端自动存储**：推理完成后根据策略自动保存有意义的图片
- **客户端无上传权限**：不提供图片上传端点，存储由服务端控制
- **智能存储策略**：默认只存储错误/失败结果，支持智能采样
- **图片压缩**：支持多种压缩算法，减少存储空间

**存储策略类型**：
| 策略类型 | 说明 | 适用场景 |
|----------|------|----------|
| `never` | 从不存储 | 完全不保存图片 |
| `always` | 总是存储 | 测试环境，不推荐生产使用 |
| `error_only` | **只存储错误/失败结果** | **生产环境推荐** |
| `smart` | 智能采样 | 模型优化阶段 |

| 端点 | 方法 | 说明 | 认证要求 |
|------|------|------|----------|
| `/storage/images` | GET | 查询图片列表 | **需要 JWT（管理员）** |
| `/storage/images/{image_id}` | GET | 获取单个图片信息 | **需要 JWT（管理员）** |
| `/storage/images` | DELETE | 删除图片 | **需要 JWT（管理员）** |
| `/storage/stats` | GET | 获取存储统计 | **需要 JWT（管理员）** |
| `/storage/config` | GET | 获取存储配置 | **需要 JWT（管理员）** |
| `/storage/config` | PUT | 更新存储配置 | **需要 JWT（管理员）** |
| `/storage/cleanup` | POST | 清理过期图片 | **需要 JWT（管理员）** |
| `/storage/image-ids` | GET | 获取所有图片 ID 列表 | **需要 JWT（管理员）** |
| `/storage/images/download` | POST | 批量下载图片（ZIP） | **需要 JWT（管理员）** |

**注意**：所有存储相关端点**仅限管理员**（JWT Token）访问，客户端无法上传图片。

**查询参数**：
- `task_id` - 按任务 ID 过滤（可选）
- `start_date` - 查询开始日期（可选）
- `end_date` - 查询结束日期（可选）
- `page` - 页码，从 1 开始（默认 1）
- `page_size` - 每页数量，最大 100（默认 20）

**存储统计响应**：
- `total_images` - 总图片数
- `total_size_bytes` - 总存储空间（字节）
- `total_size_mb` - 总存储空间（MB）
- `quota_bytes` - 存储配额（字节）
- `quota_used_percent` - 配额使用百分比
- `by_type` - 按类型统计（original/processed）
- `by_modality` - 按模态统计（rgb/ir/fusion）

**新增端点说明**：

1. **GET `/storage/image-ids`** - 获取所有图片 ID 列表
   - 返回所有已存储图片的 ID 列表
   - 用于批量下载或其他批量操作
   - 响应格式：
     ```json
     {
       "total": 150,
       "image_ids": ["uuid-1", "uuid-2", ...]
     }
     ```

2. **POST `/storage/images/download`** - 批量下载图片
   - 请求参数：
     ```json
     {
       "image_ids": ["uuid-1", "uuid-2", "uuid-3"]
     }
     ```
   - 限制：最多同时下载 100 张图片
   - 返回：ZIP 压缩包（`application/zip`），包含所有请求的图片文件
   - 自动跳过不存在的图片
   - 根据图片类型使用正确的文件扩展名（.jpg, .png, .gif, .webp）

**配置示例**：
```bash
# 启用自动存储（推理完成后自动保存）
STORAGE_AUTO_SAVE=true

# 设置存储策略（推荐：error_only）
STORAGE_SAVE_STRATEGY=error_only

# 启用图片压缩
IMAGE_COMPRESS_ENABLED=true
IMAGE_COMPRESS_QUALITY=75
IMAGE_COMPRESS_TYPE=opencv
```

## 实时进度推送

服务提供 WebSocket 连接，用于实时推送任务进度更新、完成通知和失败通知。

### WebSocket 连接

```http
GET /infer/ws
```

**连接流程：**
1. 客户端连接到 WebSocket 端点
2. 服务端返回连接确认和唯一的 `client_id`
3. 客户端使用此 `client_id` 提交检测请求
4. 服务端推送任务进度更新到对应客户端

**连接响应：**
```json
{
  "type": "connected",
  "client_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### 进度更新消息

任务执行过程中，服务端会推送进度更新：

```json
{
  "type": "progress_update",
  "data": {
    "task_id": "task_123456",
    "total_items": 10,
    "completed_items": 3,
    "failed_items": 0,
    "progress_percentage": 30.0,
    "status": "running",
    "current_result": {
      "mode": "single",
      "result": "real",
      "confidence": 0.95,
      "probabilities": [0.95, 0.05],
      "processing_time": 45,
      "success": true,
      "retry_count": 0
    },
    "real_count": 2,
    "fake_count": 1,
    "error_count": 0,
    "elapsed_time_ms": 1500,
    "message": "完成第 3/10 张图片的推理"
  }
}
```

### 任务完成通知

任务完成后，服务端推送完成通知：

**全部成功：**
```json
{
  "type": "task_completed",
  "data": {
    "task_id": "task_123456",
    "status": "completed",
    "message": "任务已完成",
    "total_items": 10,
    "successful_items": 10,
    "failed_items": 0
  }
}
```

**部分失败：**
```json
{
  "type": "task_partial_failure",
  "data": {
    "task_id": "task_123456",
    "status": "partial_failure",
    "message": "任务完成，2 项失败",
    "total_items": 10,
    "successful_items": 8,
    "failed_items": 2,
    "errors": [
      {
        "index": 3,
        "error": "模型推理失败：CUDA out of memory",
        "retry_count": 3
      },
      {
        "index": 7,
        "error": "图像解码失败：无效的 base64 编码",
        "retry_count": 3
      }
    ]
  }
}
```

### 任务状态说明

| 状态 | 说明 |
|------|------|
| `pending` | 任务已创建，等待处理 |
| `running` | 任务正在处理中 |
| `completed` | 任务完成，所有项目成功 |
| `partial_failure` | 任务完成，部分项目失败 |
| `failed` | 任务失败（任务级别错误） |

## 项目结构

```
backend/
├── main.py                      # 应用入口点，包含全局异常处理
├── lifespan.py                  # 应用生命周期管理，依赖注入配置（含数据库初始化）
├── router.py                    # 路由注册
├── controller/                  # FastAPI 控制器（API 端点）
│   ├── health_controller.py     # 健康检查端点（含详细健康检查）
│   ├── infer_controller.py      # 推理接口（单模态/融合模态/WebSocket）
│   ├── auth_controller.py       # 认证路由（JWT Token、API Key）
│   ├── activation_controller.py # 激活码路由
│   ├── history_controller.py    # 历史记录查询/统计/删除 ✅
│   ├── storage_controller.py    # 图片存储管理 ✅
│   ├── metrics_controller.py     # Prometheus 指标暴露 ✅
│   └── config_controller.py     # 配置热更新管理 ✅
├── service/                     # 业务逻辑层
│   ├── infer_service.py         # 主推理服务，协调控制器和推理器
│   ├── progress_service.py      # 进度追踪服务
│   ├── history_service.py       # 历史记录持久化服务 ✅
│   ├── image_auto_save_service.py # 图片自动存储服务 ✅
│   └── config_service.py        # 配置热更新管理服务 ✅
├── inferencer/                  # 模型推理实现层
│   ├── base_inferencer.py       # 抽象基类，定义推理接口
│   ├── inferencer_factory.py    # 工厂类，创建推理器实例
│   ├── single_model_inferencer.py   # 单模态推理器
│   └── fusion_model_inferencer.py   # 融合模态推理器
├── util/                        # 工具模块
│   ├── config.py                # 配置管理
│   ├── logger.py                # 日志配置
│   ├── result_parser.py         # 模型输出解析器
│   ├── websocket_manager.py     # WebSocket 连接管理
│   ├── auth.py                  # 认证工具（API Key、JWT、激活码）
│   ├── image_compressor.py      # 图片压缩模块（OpenCV/Pillow/Resize）✅
│   ├── image_storage_policy.py  # 图片存储策略模块 ✅
│   └── storage.py               # 图片存储管理（本地/S3）✅
├── db/                          # 数据库模块 ✅
│   ├── __init__.py              # 数据库连接管理
│   └── models.py                # 数据库模型定义（DetectionTask, DetectionResult, StoredImage）
├── schemas/                     # 数据模型（Pydantic）
│   ├── detection.py             # 检测请求/响应
│   ├── auth.py                  # 认证相关模型
│   ├── activation.py            # 激活码相关模型
│   ├── history.py               # 历史记录相关模型 ✅
│   ├── storage.py               # 图片存储相关模型 ✅
│   └── config.py                # 配置热更新相关模型 ✅
├── middleware/                  # 中间件
│   ├── auth_middleware.py       # 认证中间件
│   ├── rate_limiter.py          # 速率限制中间件
│   ├── logging_middleware.py    # 请求日志和审计日志中间件
│   └── metrics_middleware.py    # Prometheus 指标收集中间件
├── tests/                       # 测试文件
│   ├── README.md                # 测试使用指南
│   ├── test_auth.py             # 认证系统完整测试
│   ├── test_infer.py            # 推理功能测试
│   ├── test_activation.py       # 激活码测试
│   ├── test_websocket_client.py # WebSocket 测试
│   ├── test_history.py          # 历史记录功能测试 ✅
│   ├── test_storage.py          # 图片存储功能测试 ✅
│   ├── test_task_management.py  # 批量任务管理增强测试 ✅
│   ├── test_task_scheduler.py   # 优先级任务调度器测试 ✅
│   ├── test_prometheus.py       # Prometheus 指标测试 ✅
│   ├── test_health_check.py     # 健康检查增强测试 ✅
│   └── test_config_hotreload.py # 配置热更新功能测试 ✅
├── models/                      # 模型文件存储（.gitignore）
├── db/                          # SQLite 数据库文件（自动生成） ✅
├── storage/                     # 图片存储目录（自动生成） ✅
├── logs/                        # 日志文件
├── .env                         # 环境变量配置（本地，不提交）
└── .env.example                 # 环境变量示例
```

## 开发指南

### 代码规范

- **类型注解**：所有函数参数和返回值必须添加类型注解
- **命名规范**：
  - 变量/函数：`snake_case`
  - 类：`PascalCase`
  - 常量：`UPPER_SNAKE_CASE`
- **导入顺序**：标准库 → 第三方库 → 本地模块
- **日志记录**：使用 `logger = logging.getLogger(__name__)`

### 代码检查

```bash
# 代码检查
ruff check .

# 自动修复
ruff check --fix .

# 格式化代码
ruff format .
```

### 运行测试

```bash
# 使用 pytest 运行所有测试
pytest

# 详细输出
pytest -v

# 覆盖率测试
pytest --cov=.

# 运行特定测试文件
pytest tests/test_activation.py
pytest tests/test_websocket_client.py
pytest tests/test_auth.py
pytest tests/test_infer.py
```

### 运行测试脚本（直接运行）

```bash
# 认证系统完整测试（需要服务运行）
python tests/test_auth.py

# 推理功能测试（需要服务运行 + API Key 配置）
python tests/test_infer.py

# 激活码单元测试（无需服务运行）
python tests/test_activation.py

# WebSocket 客户端测试
python tests/test_websocket_client.py

# 历史记录功能测试（无需服务运行）
python tests/test_history.py

# 图片存储功能测试（无需服务运行）
python tests/test_storage.py

# 批量任务管理测试（需要服务运行）
python tests/test_task_management.py

# 任务调度器功能测试（无需服务运行）
python tests/test_task_scheduler.py

# Prometheus 指标测试（部分需要服务运行）
python tests/test_prometheus.py

# 健康检查增强测试（需要服务运行）
python tests/test_health_check.py

# 配置热更新测试（需要服务运行）
python tests/test_config_hotreload.py

# 健康检查增强测试（需要服务运行）
python tests/test_health_check.py
```

**测试说明**：
- `tests/test_auth.py` - 认证系统完整测试（JWT、激活码、API Key、WebSocket 连接）
- `tests/test_infer.py` - 推理功能测试（单模态、融合模态、任务查询、进度推送）
- `tests/test_activation.py` - 激活码单元测试（离线测试）
- `tests/test_websocket_client.py` - WebSocket 客户端测试
- `tests/test_history.py` - 历史记录功能测试（数据库 CRUD、查询、统计）
- `tests/test_storage.py` - 图片存储功能测试（上传、查询、删除、统计、清理、**批量下载**、**图片压缩**、**存储配额**）
- `tests/test_task_management.py` - 批量任务管理增强功能测试
- `tests/test_task_scheduler.py` - 优先级任务调度器功能测试
- `tests/test_prometheus.py` - Prometheus 指标暴露功能测试
  - 单元测试：指标模块导入、推理指标记录、任务指标记录、WebSocket 连接指标、激活码使用指标
  - 集成测试：/metrics 端点可用性、指标格式验证、HTTP 请求指标、推理指标、任务指标、速率限制指标、错误指标、API Key 使用量指标
- `tests/test_history.py` - 历史记录功能测试（数据库 CRUD、查询、统计）
- `tests/test_storage.py` - 图片存储功能测试（上传、查询、删除、统计、清理、批量下载、图片压缩、存储配额）
- `tests/test_task_management.py` - 批量任务管理增强功能测试
- `tests/test_task_scheduler.py` - 优先级任务调度器功能测试
- `tests/test_health_check.py` - 健康检查增强功能测试
  - 测试简单健康检查端点
  - 测试未授权访问被拒绝
  - 测试管理员 JWT 访问详细健康检查
- `tests/test_config_hotreload.py` - 配置热更新功能测试
  - 获取应用配置（需 JWT）
  - 未认证访问（应返回 401）
  - 更新日志配置
  - 更新重试配置
  - 更新调试配置
  - 获取配置历史
  - 回滚配置
- `tests/README.md` - 测试使用指南

## 注意事项

1. **模型文件**：不要将模型文件提交到 Git，使用 `.gitignore` 忽略 `models/` 目录
2. **环境变量**：`.env` 文件包含敏感配置，不要提交到版本控制
3. **日志文件**：日志仅输出到文件（`logs/` 目录），建议定期清理
4. **审计日志**：JSON 格式存储在 `logs/{timestamp}_audit.log`，便于日志分析
5. **GPU 内存**：如果使用 CUDA，确保显存足够加载模型
6. **并发处理**：服务基于 FastAPI，支持异步并发处理请求
7. **WebSocket 连接**：检测请求需要先建立 WebSocket 连接获取 `client_id`
8. **任务持久存储**：任务完成后数据持久保存，可随时查询历史任务
9. **API Key 安全**：
   - 生产环境务必修改默认配置
   - 定期轮换激活码
   - 禁用不再使用的激活码
10. **速率限制**：触发限流时会自动记录审计日志
11. **重试机制**：
    - 单张图片推理失败自动重试（默认最多 3 次）
    - 重试延迟按指数增长（1s, 2s, 4s...）
    - 部分失败不会导致整个任务失败
    - 可通过 `DEBUG_FAILURE_RATE` 测试重试逻辑
12. **任务状态**：
    - `completed` - 所有项目成功
    - `partial_failure` - 部分项目失败
    - `failed` - 任务级别错误（如模型未加载）
13. **图片存储**：
    - 本地存储采用两级目录结构，避免单目录文件过多
    - 存储配额用于防止存储溢出，建议根据磁盘空间合理设置
    - 清理过期图片需手动调用 `POST /storage/cleanup` 端点
    - S3 存储需要安装 `boto3` 库：`pip install boto3`
14. **配置热更新**：
    - 配置变更仅对当前运行实例有效，不会写入 `.env` 文件
    - 服务重启后配置恢复到 `.env` 中的值
    - `log_level` 更新后立即生效，其他配置在下次使用时自动应用
    - 配置历史记录存储在内存中，服务重启后清空
    - 所有配置端点需要管理员 JWT Token 认证

## 许可证

[Apache License 2.0](LICENSE)

## 贡献

欢迎提交 Issue 和 Pull Request。

## 联系方式

如有问题，请在 GitHub Issues 中提交。
