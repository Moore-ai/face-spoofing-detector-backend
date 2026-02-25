"""
Schema 模块
"""

from schemas.detection import (
    DetectionResultItem,
    AsyncTaskResponse,
    TaskStatusResponse,
    SingleModeRequest,
    ImagePair,
    FusionModeRequest,
)
from schemas.auth import (
    TokenRequest,
    TokenResponse,
    UserInfo,
    AuthErrorResponse,
)
from schemas.activation import (
    ActivateRequest,
    ActivateResponse,
    ActivationCodeInfo,
    ActivationCodeCreate,
    ActivationCodeListResponse,
    ActivationCodeGenerateResponse,
    ActivationCodeUpdate,
)

__all__ = [
    "DetectionResultItem",
    "AsyncTaskResponse",
    "TaskStatusResponse",
    "SingleModeRequest",
    "ImagePair",
    "FusionModeRequest",
    "TokenRequest",
    "TokenResponse",
    "UserInfo",
    "AuthErrorResponse",
    "ActivateRequest",
    "ActivateResponse",
    "ActivationCodeInfo",
    "ActivationCodeCreate",
    "ActivationCodeListResponse",
    "ActivationCodeGenerateResponse",
    "ActivationCodeUpdate",
]
