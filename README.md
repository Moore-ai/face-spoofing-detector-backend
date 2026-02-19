# 人脸活体检测后端服务

基于 FastAPI 的人脸活体检测（Face Anti-Spoofing）推理服务，支持单模态和融合模态图像检测。

## 功能特性

- **多模态支持**：支持单模态（RGB/IR）和融合模态（RGB+IR）检测
- **多模型格式**：支持 PyTorch (.pth)、ONNX (.onnx)、RKNN (.rknn) 模型
- **RESTful API**：基于 FastAPI 构建，支持异步处理
- **灵活配置**：通过环境变量或 `.env` 文件配置所有参数
- **完整日志**：详细的日志记录，支持文件和控制台双输出
- **全局异常处理**：统一的错误处理和日志记录机制

## 技术栈

- **Web 框架**：FastAPI
- **深度学习**：PyTorch、ONNX Runtime
- **图像处理**：OpenCV、Pillow、NumPy
- **模型支持**：MobileNetV2（可扩展）
- **日志**：Python logging
- **依赖管理**：uv（推荐）或 pip

## 快速开始

### 1. 安装依赖

```bash
# 使用 uv（推荐）
uv sync

# 或使用 pip
uv pip install -e .
```

### 2. 配置环境

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
# 方式1：直接运行
python main.py

# 方式2：使用 uv
uv run python main.py
```

服务启动后，访问：
- API 文档：http://127.0.0.1:8000/docs
- 健康检查：http://127.0.0.1:8000/health

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

## API 接口

### 1. 单模态检测

**请求**

```http
POST /infer/single
Content-Type: application/json

{
  "mode": "single",
  "modality": "rgb",
  "images": ["base64_encoded_image_1", "base64_encoded_image_2"]
}
```

**响应**

```json
{
  "results": [
    {
      "id": "0",
      "result": "real",
      "confidence": 0.95,
      "timestamp": "2026-02-13T10:30:00+00:00",
      "processing_time": 45
    }
  ],
  "total_count": 1,
  "real_count": 1,
  "fake_count": 0,
  "average_confidence": 0.95
}
```

### 2. 融合模态检测

**请求**

```http
POST /infer/fusion
Content-Type: application/json

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

与单模态检测格式相同。

### 3. 健康检查

```http
GET /health
```

## 项目结构

```
backend/
├── main.py                      # 应用入口
├── lifespan.py                  # 生命周期管理、依赖注入
├── router.py                    # 路由注册
├── controller/                  # 控制器（API 端点）
│   ├── health_controller.py     # 健康检查
│   └── infer_controller.py      # 推理接口
├── service/                     # 业务逻辑
│   └── infer_service.py         # 推理服务
├── inferencer/                  # 推理器实现
│   ├── base_inferencer.py       # 抽象基类
│   ├── inferencer_factory.py    # 工厂类
│   ├── single_model_inferencer.py   # 单模态推理器
│   └── fusion_model_inferencer.py   # 融合模态推理器
├── util/                        # 工具模块
│   ├── config.py                # 配置管理
│   ├── logger.py                # 日志配置
│   ├── result_parser.py         # 结果解析
│   └── batch_result_builder.py  # 批量结果构建
├── schemas/                     # 数据模型
│   └── detection.py             # 检测相关模型
├── models/                      # 模型文件（不提交到版本控制）
├── logs/                        # 日志文件（按启动时间命名）
├── .env                         # 环境变量（本地配置，不提交）
└── .env.example                 # 环境变量示例
```

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

```bash
# 运行所有测试
pytest

# 详细输出
pytest -v

# 覆盖率测试
pytest --cov=.
```

## 注意事项

1. **模型文件**：不要将模型文件提交到 Git，使用 `.gitignore` 忽略 `models/` 目录
2. **环境变量**：`.env` 文件包含敏感配置，不要提交到版本控制
3. **日志文件**：日志文件会随时间增长，建议定期清理或使用日志轮转
4. **GPU 内存**：如果使用 CUDA，确保显存足够加载模型
5. **并发处理**：服务基于 FastAPI，支持异步并发处理请求

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

## 许可证

[Apache License 2.0](LICENSE)

## 贡献

欢迎提交 Issue 和 Pull Request。

## 联系方式

如有问题，请在 GitHub Issues 中提交。
