
from fastapi import FastAPI

from controller import (
    health_controller,
    infer_controller
)

ROUTER_CONFIGS = [
    {
        "router": health_controller.router,
        "prefix": "",
        "tag": ["健康测试"],
        "dependencies": [],
    },
    {
        "router": infer_controller.router,
        "prefix": "/infer",
        "tag": ["模型推理"],
        "dependencies": [],
    }
]

def register_routers(app: FastAPI):
    for config in ROUTER_CONFIGS:
        app.include_router(
            config["router"], 
            prefix=config.get("prefix", ""), 
            tags=config["tag"],
            dependencies=config.get("dependencies", []),
        )
        print(f"📦 注册路由: {config.get('prefix', '')} -> {config.get('tag', '')}")