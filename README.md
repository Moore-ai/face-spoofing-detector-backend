# 人脸活体检测后端服务

基于 FastAPI 的人脸活体检测（Face Anti-Spoofing）推理服务，支持单模态和融合模态图像检测，提供实时进度推送和异步任务管理。

## 功能特性

- **多模态支持**：支持单模态（RGB/IR）和融合模态（RGB+IR）检测
- **多模型格式**：支持 PyTorch (.pth)、ONNX (.onnx)、RKNN (.rknn) 模型
- **RESTful API**：基于 FastAPI 构建，支持异步处理
- **WebSocket实时推送**：实时推送任务进度更新和完成通知
- **异步任务管理**：后台批量处理，支持任务状态查询
- **进度追踪服务**：实时统计任务进度、结果数量和分类
- **灵活配置**：通过环境变量或 `.env` 文件配置所有参数
- **完整日志**：详细的日志记录，支持文件和控制台双输出
- **全局异常处理**：统一的错误处理和日志记录机制
- **依赖注入架构**：模块化设计，易于测试和维护

## 技术栈

- **Web 框架**：FastAPI（支持异步、WebSocket）
- **深度学习**：PyTorch、ONNX Runtime
- **图像处理**：OpenCV、Pillow、NumPy
- **模型架构**：MobileNetV2（可扩展）
- **实时通信**：WebSocket（进度推送）
- **数据验证**：Pydantic（类型安全）
- **配置管理**：pydantic-settings（环境变量）
- **日志系统**：Python logging（文件+控制台）
- **测试框架**：pytest（单元测试和集成测试）
- **API 文档**：Swagger UI、ReDoc（自动生成）
- **依赖管理**：uv（推荐）或 pip

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
- 检查Python版本
- 安装uv包管理器（如未安装）
- 创建虚拟环境
- 安装项目依赖
- 从.env.example创建.env配置文件

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

### 3. 准备模型

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

### 4. 启动服务

```bash
# 方式1：直接运行（生产模式）
python main.py

# 方式2：使用 uv
uv run python main.py

# 方式3：开发模式（支持热重载）
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

### 日志配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `LOG_LEVEL` | `INFO` | 日志级别：`DEBUG`/`INFO`/`WARNING`/`ERROR`/`CRITICAL` |

### 环境变量示例

```bash
# 使用 CPU 推理
DEVICE=cpu python main.py

# 指定不同端口
PORT=8080 python main.py

# 开发模式（详细日志）
LOG_LEVEL=DEBUG python main.py
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
    "progress_percentage": 30.0,
    "status": "running",
    "current_result": {
      "mode": "single",
      "result": "real",
      "confidence": 0.95,
      "probabilities": [0.95, 0.05],
      "processing_time": 45
    },
    "real_count": 2,
    "fake_count": 1,
    "elapsed_time_ms": 1500,
    "message": "完成第 3/10 张图片的推理"
  }
}
```

### 任务完成通知

任务完成后推送：

```json
{
  "type": "task_completed",
  "data": {
    "task_id": "task_123456",
    "status": "completed",
    "message": "任务已完成，可查询检测结果",
    "total_items": 10,
    "processed_items": 10
  }
}
```

### 任务失败通知

任务失败时推送：

```json
{
  "type": "task_failed",
  "data": {
    "task_id": "task_123456",
    "status": "failed",
    "message": "图像解码失败"
  }
}
```

## API 接口

### 1. 单模态检测（异步）

**请求**

```http
POST /infer/single
Content-Type: application/json
X-Client-ID: 550e8400-e29b-41d4-a716-446655440000

{
  "mode": "single",
  "modality": "rgb",
  "images": ["base64_encoded_image_1", "base64_encoded_image_2"]
}
```

> **注意**：需要先建立 WebSocket 连接获取 `client_id`，并通过 `X-Client-ID` 请求头传入。

**响应**

```json
{
  "task_id": "task_123456",
  "message": "单模态检测任务已创建，正在后台处理"
}
```

### 2. 融合模态检测（异步）

**请求**

```http
POST /infer/fusion
Content-Type: application/json
X-Client-ID: 550e8400-e29b-41d4-a716-446655440000

{
  "mode": "fusion",
  "pairs": [
    {
      "rgb": "base64_encoded_rgb_image",
      "ir": "base64_encoded_ir_image"
    }
  ]
}
```

**响应**

```json
{
  "task_id": "task_123456",
  "message": "融合模态检测任务已创建，正在后台处理"
}
```

### 3. 任务状态查询

**请求**

```http
GET /infer/task/{task_id}
```

**响应**

```json
{
  "task_id": "task_123456",
  "status": "completed",
  "total_items": 10,
  "completed_items": 10,
  "progress_percentage": 100.0,
  "real_count": 8,
  "fake_count": 2,
  "elapsed_time_ms": 4500,
  "message": "任务已完成",
  "results": [
    {
      "mode": "single",
      "result": "real",
      "confidence": 0.92,
      "probabilities": [0.92, 0.08],
      "processing_time": 45
    }
  ],
  "current_result": {
    "mode": "single",
    "result": "real",
    "confidence": 0.92,
    "probabilities": [0.92, 0.08],
    "processing_time": 45
  }
}
```

> **注意**：`results` 字段仅在任务状态为 `completed` 或 `failed` 时返回。

### 4. 健康检查

```http
GET /health
```

## 项目结构

```
backend/
├── main.py                      # 应用入口点，包含全局异常处理
├── lifespan.py                  # 应用生命周期管理，依赖注入配置
├── router.py                    # 路由注册和设置
├── controller/                  # FastAPI 控制器（API 端点）
│   ├── health_controller.py     # 健康检查端点
│   └── infer_controller.py      # 推理接口（单模态/融合模态/WebSocket）
├── service/                     # 业务逻辑层
│   ├── infer_service.py         # 主推理服务，协调控制器和推理器
│   └── progress_service.py      # 进度追踪服务，管理批量检测任务进度
├── inferencer/                  # 模型推理实现层
│   ├── base_inferencer.py       # 抽象基类，定义推理接口
│   ├── inferencer_factory.py    # 工厂类，根据模型格式创建推理器实例
│   ├── single_model_inferencer.py   # 单模态推理器
│   └── fusion_model_inferencer.py   # 融合模态推理器
├── util/                        # 工具模块
│   ├── config.py                # 配置管理（基于pydantic-settings）
│   ├── logger.py                # 日志配置，支持文件和控制台输出
│   ├── result_parser.py         # 模型输出解析器
│   ├── websocket_manager.py     # WebSocket 连接管理，客户端映射
│   └── model/                   # 模型架构定义（MobileNetV2）
├── schemas/                     # 数据模型（Pydantic）
│   └── detection.py             # 检测请求/响应数据模型
├── tests/                       # 测试文件
│   └── test_websocket_client.py # WebSocket 客户端测试脚本
├── models/                      # 模型文件存储（.gitignore）
├── logs/                        # 日志文件，按启动时间命名
├── .env                         # 环境变量配置（本地，不提交）
└── .env.example                 # 环境变量示例模板
```

## 架构说明

### 依赖注入架构
- **lifespan.py**: 应用生命周期管理，初始化全局服务实例
- **依赖注入**: 使用 FastAPI `Depends` 注入 `InferService` 和 `ConnectionManager`
- **工厂模式**: `InferencerFactory` 根据模型格式创建相应推理器
- **分层架构**: Controllers → Services → Inferencers → Utilities

### 实时进度推送系统
1. **WebSocket 连接**: 客户端连接到 `/infer/ws` 获取 `client_id`
2. **任务注册**: 检测请求使用 `client_id` 注册任务到对应客户端
3. **进度追踪**: `ProgressTracker` 管理任务状态和统计信息
4. **实时推送**: `WebSocketManager` 向特定客户端广播进度更新
5. **状态查询**: 支持通过 REST API 查询任务状态和结果

### 任务生命周期
1. **pending**: 任务创建，等待执行
2. **running**: 任务开始执行，接收进度更新
3. **completed**: 任务完成，包含所有检测结果
4. **failed**: 任务失败，包含错误信息

### 异步处理流程
1. 客户端建立 WebSocket 连接
2. 提交检测请求（单模态/融合模态）
3. 立即返回 `task_id`，后台异步处理
4. 实时接收进度更新（WebSocket）
5. 任务完成后可通过 REST API 查询完整结果

## 日志说明

日志文件存储在 `logs/` 目录下，文件名包含启动时间：

```
logs/
├── 2026-02-13_14-30-52.log     # 示例：2026年2月13日 14:30:52 启动
└── 2026-02-13_15-21-08.log
```

日志级别：
- **ERROR** - 错误信息（模型加载失败、未处理异常等）
- **WARNING** - 警告信息（HTTP 4xx 错误等）
- **INFO** - 一般信息（服务启动、模型加载成功、请求处理等）
- **DEBUG** - 调试信息（开发模式使用）

日志包含内容：
- 时间戳
- 模块名
- 日志级别
- 详细消息
- 异常堆栈（ERROR 级别）

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

#### 单元测试
```bash
# 运行所有测试
pytest

# 详细输出
pytest -v

# 覆盖率测试
pytest --cov=.

# 运行特定测试文件
pytest tests/test_websocket_client.py
```

#### WebSocket 集成测试
服务启动后，运行 WebSocket 客户端测试：

```bash
# 基本测试（需要服务运行在默认地址）
python tests/test_websocket_client.py

# 指定服务器地址
python tests/test_websocket_client.py --url http://192.168.1.100:8000

# 跳过部分测试场景
python tests/test_websocket_client.py --skip-fusion --skip-errors

# 详细输出和长超时
python tests/test_websocket_client.py --verbose --timeout 30.0

# 显示帮助信息
python tests/test_websocket_client.py --help
```

#### 测试功能覆盖
- ✅ WebSocket 连接和 client_id 分配
- ✅ 无效客户端 ID 验证
- ✅ 单模态检测任务创建和进度推送
- ✅ 融合模态检测任务创建和进度推送
- ✅ 任务状态查询 API
- ✅ 错误场景处理（无效 base64、无效 modality 等）
- ✅ 服务健康检查

## 注意事项

1. **模型文件**：不要将模型文件提交到 Git，使用 `.gitignore` 忽略 `models/` 目录
2. **环境变量**：`.env` 文件包含敏感配置，不要提交到版本控制
3. **日志文件**：日志文件会随时间增长，建议定期清理或使用日志轮转
4. **GPU 内存**：如果使用 CUDA，确保显存足够加载模型
5. **并发处理**：服务基于 FastAPI，支持异步并发处理请求
6. **WebSocket 连接**：检测请求需要先建立 WebSocket 连接获取 `client_id`
7. **任务持久存储**：任务完成后数据持久保存，可随时查询历史任务
8. **客户端映射**：每个 WebSocket 连接对应一个客户端，断开连接后任务不再推送

## 故障排查

### 1. 模型文件不存在

```
ERROR - Model file not found: models/single_model.pth
```

**解决**：将模型文件放入 `models/` 目录，或修改 `.env` 中的路径配置

### 2. CUDA 不可用

```
CUDA is not available, using CPU
```

**解决**：安装 CUDA 版本的 PyTorch，或在 `.env` 中设置 `DEVICE=cpu`

### 3. 端口被占用

```
Address already in use
```

**解决**：修改 `.env` 中的 `PORT` 为其他端口，或关闭占用该端口的程序

### 4. WebSocket 连接失败

```
Connection refused / Cannot connect to WebSocket
```

**解决**：
- 确保服务已启动并运行在正确的地址和端口
- 检查防火墙设置，确保 WebSocket 端口可访问
- 验证 URL 格式：`ws://localhost:8000/infer/ws`（HTTP）或 `wss://...`（HTTPS）

### 5. 无效的 client_id 错误

```
无效的 client_id: xxx，请先建立 WebSocket 连接
```

**解决**：
1. 先连接到 `/infer/ws` WebSocket 端点
2. 接收 `client_id`（包含在 `connected` 消息中）
3. 通过 `X-Client-ID` 请求头传入 `client_id`

### 6. 任务状态查询返回 404

```
任务不存在或已过期
```

**解决**：
- 确保使用正确的 `task_id`（任务ID区分大小写）
- 检查任务是否已成功创建（可能创建失败）
- 验证任务是否处于 pending、running、completed 或 failed 状态
- 注意：任务数据持久保存，不会自动清理

## 许可证

[Apache License 2.0](LICENSE)

## 贡献

欢迎提交 Issue 和 Pull Request。

## 联系方式

如有问题，请在 GitHub Issues 中提交。
