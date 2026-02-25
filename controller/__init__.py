"""
Controller 模块
"""

from controller.health_controller import router as health_router
from controller.infer_controller import router as infer_router
from controller.auth_controller import router as auth_router
from controller.activation_controller import router as activation_router

__all__ = ["health_router", "infer_router", "auth_router", "activation_router"]
