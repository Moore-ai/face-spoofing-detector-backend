"""
审计日志工具模块

提供关键操作的审计记录功能：
- 认证事件（登录、激活码换取 API Key）
- 管理操作（创建/删除/禁用激活码）
- 推理请求统计
- 异常情况记录

审计日志与普通日志分离，便于独立分析和合规审查
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)
audit_logger = logging.getLogger("audit")


class AuditEventType(str, Enum):
    """审计事件类型枚举"""

    # 认证相关
    AUTH_ACTIVATE_CODE = "auth.activate_code"  # 激活码换取 API Key
    AUTH_JWT_LOGIN = "auth.jwt_login"  # JWT 登录
    AUTH_API_KEY_CREATED = "auth.api_key_created"  # API Key 创建
    AUTH_FAILED = "auth.failed"  # 认证失败

    # 激活码管理
    ACTIVATION_CODE_CREATED = "activation_code.created"  # 创建激活码
    ACTIVATION_CODE_DELETED = "activation_code.deleted"  # 删除激活码
    ACTIVATION_CODE_UPDATED = "activation_code.updated"  # 更新激活码
    ACTIVATION_CODE_DEACTIVATED = "activation_code.deactivated"  # 禁用激活码
    ACTIVATION_CODE_LISTED = "activation_code.listed"  # 列出激活码

    # 推理相关
    INFERENCE_SINGLE = "inference.single"  # 单模态推理
    INFERENCE_FUSION = "inference.fusion"  # 融合模态推理
    INFERENCE_TASK_QUERY = "inference.task_query"  # 任务状态查询

    # 系统相关
    SYSTEM_ERROR = "system.error"  # 系统错误
    SYSTEM_RATE_LIMITED = "system.rate_limited"  # 被速率限制


@dataclass
class AuditEvent:
    """审计事件数据类"""

    event_type: AuditEventType
    actor_id: str | None = None  # 执行者 ID（用户 ID、API Key 前缀等）
    actor_type: str | None = None  # 执行者类型（user, api_key, system）
    status: str = "success"  # success, failure, error
    status_code: int | None = None  # HTTP 状态码
    duration_ms: int | None = None  # 耗时（毫秒）
    client_ip: str | None = None  # 客户端 IP
    user_agent: str | None = None  # User-Agent
    path: str | None = None  # 请求路径
    method: str | None = None  # 请求方法
    details: dict[str, Any] = field(default_factory=dict)  # 附加详情
    timestamp: float = field(default_factory=time.time)  # 时间戳

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "timestamp": self.timestamp,
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.timestamp)),
            "type": "audit_log",
            "event": self.event_type.value,
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "status": self.status,
            "status_code": self.status_code,
            "duration_ms": self.duration_ms,
            "client_ip": self.client_ip,
            "user_agent": self.user_agent,
            "path": self.path,
            "method": self.method,
            "details": self.details,
        }

    def to_json(self) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False)


class AuditLogger:
    """审计日志记录器"""

    def __init__(self, logger: logging.Logger | None = None):
        """
        Args:
            logger: 日志记录器，默认使用 audit_logger
        """
        self.logger = logger or audit_logger

    def log(self, event: AuditEvent) -> None:
        """记录审计事件"""
        json_log = event.to_json()

        if event.status in ["failure", "error"]:
            self.logger.warning(json_log)
        else:
            self.logger.info(json_log)

    # ========== 便捷方法 ==========

    def log_auth_activate(
        self,
        activation_code: str,
        api_key: str | None = None,
        status: str = "success",
        status_code: int | None = None,
        duration_ms: int | None = None,
        client_ip: str | None = None,
    ) -> None:
        """记录激活码换取 API Key 事件"""
        event = AuditEvent(
            event_type=AuditEventType.AUTH_ACTIVATE_CODE,
            actor_id=f"code:{activation_code[:12]}" if activation_code else None,
            actor_type="activation_code",
            status=status,
            status_code=status_code,
            duration_ms=duration_ms,
            client_ip=client_ip,
            details={
                "activation_code_prefix": activation_code[:12] + "..." if activation_code and len(activation_code) > 12 else activation_code,
                "api_key_issued": api_key is not None,
            },
        )
        self.log(event)

    def log_auth_jwt_login(
        self,
        username: str | None = None,
        status: str = "success",
        status_code: int | None = None,
        duration_ms: int | None = None,
        client_ip: str | None = None,
    ) -> None:
        """记录 JWT 登录事件"""
        event = AuditEvent(
            event_type=AuditEventType.AUTH_JWT_LOGIN,
            actor_id=username,
            actor_type="user",
            status=status,
            status_code=status_code,
            duration_ms=duration_ms,
            client_ip=client_ip,
            details={"username": username},
        )
        self.log(event)

    def log_auth_failed(
        self,
        reason: str,
        client_ip: str | None = None,
        path: str | None = None,
    ) -> None:
        """记录认证失败事件"""
        event = AuditEvent(
            event_type=AuditEventType.AUTH_FAILED,
            actor_id=f"ip:{client_ip}" if client_ip else None,
            actor_type="anonymous",
            status="failure",
            client_ip=client_ip,
            path=path,
            details={"reason": reason},
        )
        self.log(event)

    def log_activation_code_created(
        self,
        code: str,
        max_uses: int,
        expires_in_hours: int | None = None,
        created_by: str | None = None,
        client_ip: str | None = None,
    ) -> None:
        """记录创建激活码事件"""
        event = AuditEvent(
            event_type=AuditEventType.ACTIVATION_CODE_CREATED,
            actor_id=created_by,
            actor_type="admin",
            status="success",
            client_ip=client_ip,
            details={
                "code_prefix": code[:12] + "..." if code and len(code) > 12 else code,
                "max_uses": max_uses,
                "expires_in_hours": expires_in_hours,
            },
        )
        self.log(event)

    def log_activation_code_deleted(
        self,
        code: str,
        deleted_by: str | None = None,
        client_ip: str | None = None,
    ) -> None:
        """记录删除激活码事件"""
        event = AuditEvent(
            event_type=AuditEventType.ACTIVATION_CODE_DELETED,
            actor_id=deleted_by,
            actor_type="admin",
            status="success",
            client_ip=client_ip,
            details={
                "code_prefix": code[:12] + "..." if code and len(code) > 12 else code,
            },
        )
        self.log(event)

    def log_activation_code_updated(
        self,
        code: str,
        updated_fields: list[str],
        updated_by: str | None = None,
        client_ip: str | None = None,
    ) -> None:
        """记录更新激活码事件"""
        event = AuditEvent(
            event_type=AuditEventType.ACTIVATION_CODE_UPDATED,
            actor_id=updated_by,
            actor_type="admin",
            status="success",
            client_ip=client_ip,
            details={
                "code_prefix": code[:12] + "..." if code and len(code) > 12 else code,
                "updated_fields": updated_fields,
            },
        )
        self.log(event)

    def log_activation_code_deactivated(
        self,
        code: str,
        deactivated_by: str | None = None,
        client_ip: str | None = None,
    ) -> None:
        """记录禁用激活码事件"""
        event = AuditEvent(
            event_type=AuditEventType.ACTIVATION_CODE_DEACTIVATED,
            actor_id=deactivated_by,
            actor_type="admin",
            status="success",
            client_ip=client_ip,
            details={
                "code_prefix": code[:12] + "..." if code and len(code) > 12 else code,
            },
        )
        self.log(event)

    def log_inference(
        self,
        modality: str,
        item_count: int,
        api_key_prefix: str | None = None,
        task_id: str | None = None,
        status: str = "success",
        status_code: int | None = None,
        duration_ms: int | None = None,
        client_ip: str | None = None,
    ) -> None:
        """记录推理事件"""
        event_type = (
            AuditEventType.INFERENCE_SINGLE if modality == "single" else AuditEventType.INFERENCE_FUSION
        )
        event = AuditEvent(
            event_type=event_type,
            actor_id=f"api_key:{api_key_prefix}" if api_key_prefix else None,
            actor_type="api_key",
            status=status,
            status_code=status_code,
            duration_ms=duration_ms,
            client_ip=client_ip,
            details={
                "modality": modality,
                "item_count": item_count,
                "task_id": task_id,
            },
        )
        self.log(event)

    def log_task_query(
        self,
        task_id: str,
        api_key_prefix: str | None = None,
        status: str = "success",
        status_code: int | None = None,
        duration_ms: int | None = None,
        client_ip: str | None = None,
    ) -> None:
        """记录任务状态查询事件"""
        event = AuditEvent(
            event_type=AuditEventType.INFERENCE_TASK_QUERY,
            actor_id=f"api_key:{api_key_prefix}" if api_key_prefix else None,
            actor_type="api_key",
            status=status,
            status_code=status_code,
            duration_ms=duration_ms,
            client_ip=client_ip,
            details={"task_id": task_id},
        )
        self.log(event)

    def log_rate_limited(
        self,
        client_ip: str,
        path: str,
        api_key_prefix: str | None = None,
    ) -> None:
        """记录被速率限制事件"""
        event = AuditEvent(
            event_type=AuditEventType.SYSTEM_RATE_LIMITED,
            actor_id=f"ip:{client_ip}",
            actor_type="client",
            status="failure",
            client_ip=client_ip,
            path=path,
            details={
                "api_key_prefix": api_key_prefix,
            },
        )
        self.log(event)

    def log_error(
        self,
        error_message: str,
        path: str | None = None,
        actor_id: str | None = None,
        client_ip: str | None = None,
        extra_details: dict | None = None,
    ) -> None:
        """记录系统错误事件"""
        event = AuditEvent(
            event_type=AuditEventType.SYSTEM_ERROR,
            actor_id=actor_id,
            actor_type="system",
            status="error",
            client_ip=client_ip,
            path=path,
            details={
                "error_message": error_message,
                **(extra_details or {}),
            },
        )
        self.log(event)


# 全局审计日志记录器实例
audit_logger_instance = AuditLogger()


# ========== 便捷函数 ==========

def log_auth_activate(
    activation_code: str,
    api_key: str | None = None,
    status: str = "success",
    status_code: int | None = None,
    duration_ms: int | None = None,
    client_ip: str | None = None,
) -> None:
    """便捷函数：记录激活码换取 API Key 事件"""
    audit_logger_instance.log_auth_activate(
        activation_code, api_key, status, status_code, duration_ms, client_ip
    )


def log_auth_failed(
    reason: str,
    client_ip: str | None = None,
    path: str | None = None,
) -> None:
    """便捷函数：记录认证失败事件"""
    audit_logger_instance.log_auth_failed(reason, client_ip, path)


def log_inference(
    modality: str,
    item_count: int,
    api_key_prefix: str | None = None,
    task_id: str | None = None,
    status: str = "success",
    status_code: int | None = None,
    duration_ms: int | None = None,
    client_ip: str | None = None,
) -> None:
    """便捷函数：记录推理事件"""
    audit_logger_instance.log_inference(
        modality, item_count, api_key_prefix, task_id, status, status_code, duration_ms, client_ip
    )


def log_rate_limited(
    client_ip: str,
    path: str,
    api_key_prefix: str | None = None,
) -> None:
    """便捷函数：记录被速率限制事件"""
    audit_logger_instance.log_rate_limited(client_ip, path, api_key_prefix)
