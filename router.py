
from fastapi import FastAPI

from controller import (
    health_controller,
    infer_controller,
    auth_controller,
    activation_controller,
    history_controller,
    storage_controller,
)

import logging
logger = logging.getLogger(__name__)

ROUTER_CONFIGS = [
    {
        "router": health_controller.router,
        "prefix": "",
        "tag": ["健康测试"],
        "dependencies": [],
    },
    {
        "router": auth_controller.router,
        "prefix": "/auth",
        "tag": ["认证"],
        "dependencies": [],
    },
    {
        "router": activation_controller.router,
        "prefix": "/auth",
        "tag": ["激活码"],
        "dependencies": [],
    },
    {
        "router": infer_controller.router,
        "prefix": "/infer",
        "tag": ["模型推理"],
        "dependencies": [],  # 移除全局依赖，WebSocket 不支持常规依赖注入
    },
    {
        "router": history_controller.router,
        "prefix": "",
        "tag": ["历史记录"],
        "dependencies": [],
    },
    {
        "router": storage_controller.router,
        "prefix": "/storage",
        "tag": ["图片存储"],
        "dependencies": [],
    },
]

def register_routers(app: FastAPI):
    for config in ROUTER_CONFIGS:
        app.include_router(
            config["router"], 
            prefix=config.get("prefix", ""), 
            tags=config["tag"],
            dependencies=config.get("dependencies", []),
        )
        logger.info(f"📦 注册路由: {config.get('prefix', '')} -> {config.get('tag', '')}")