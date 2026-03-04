"""
应用配置管理服务
提供配置的读取、更新、回滚功能
"""

import logging
from typing import Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


# 配置历史记录（内存存储）
_config_history: list[dict[str, Any]] = []
_MAX_HISTORY_LENGTH = 50  # 最多保留 50 条历史记录


class ConfigService:
    """配置管理服务"""

    @staticmethod
    def get_logging_config() -> dict:
        """获取日志配置"""
        from util.config import settings

        return {
            "log_level": settings.LOG_LEVEL,
            "log_to_console": settings.LOG_TO_CONSOLE,
            "log_json_format": settings.LOG_JSON_FORMAT,
            "log_request_body": settings.LOG_REQUEST_BODY,
            "log_response_body": settings.LOG_RESPONSE_BODY,
            "audit_log_enabled": settings.AUDIT_LOG_ENABLED,
        }

    @staticmethod
    def update_logging_config(
        log_level: Optional[str] = None,
        log_to_console: Optional[bool] = None,
        log_json_format: Optional[bool] = None,
        log_request_body: Optional[bool] = None,
        log_response_body: Optional[bool] = None,
        audit_log_enabled: Optional[bool] = None,
    ) -> dict:
        """更新日志配置"""
        from util.config import settings

        old_config = ConfigService.get_logging_config()
        changes = {}

        if log_level is not None and log_level != settings.LOG_LEVEL:
            old_level = settings.LOG_LEVEL
            settings.LOG_LEVEL = log_level
            # 动态更新 logging 级别
            logging.getLogger().setLevel(getattr(logging, log_level, logging.INFO))
            changes["log_level"] = {"old": old_level, "new": log_level}

        if log_to_console is not None and log_to_console != settings.LOG_TO_CONSOLE:
            changes["log_to_console"] = {"old": settings.LOG_TO_CONSOLE, "new": log_to_console}
            settings.LOG_TO_CONSOLE = log_to_console

        if log_json_format is not None and log_json_format != settings.LOG_JSON_FORMAT:
            changes["log_json_format"] = {"old": settings.LOG_JSON_FORMAT, "new": log_json_format}
            settings.LOG_JSON_FORMAT = log_json_format

        if log_request_body is not None and log_request_body != settings.LOG_REQUEST_BODY:
            changes["log_request_body"] = {"old": settings.LOG_REQUEST_BODY, "new": log_request_body}
            settings.LOG_REQUEST_BODY = log_request_body

        if log_response_body is not None and log_response_body != settings.LOG_RESPONSE_BODY:
            changes["log_response_body"] = {"old": settings.LOG_RESPONSE_BODY, "new": log_response_body}
            settings.LOG_RESPONSE_BODY = log_response_body

        if audit_log_enabled is not None and audit_log_enabled != settings.AUDIT_LOG_ENABLED:
            changes["audit_log_enabled"] = {"old": settings.AUDIT_LOG_ENABLED, "new": audit_log_enabled}
            settings.AUDIT_LOG_ENABLED = audit_log_enabled

        # 记录历史
        if changes:
            ConfigService._record_history("logging", old_config, changes)
            logger.info(f"日志配置已更新：{changes}")

        return ConfigService.get_logging_config()

    @staticmethod
    def get_retry_config() -> dict:
        """获取重试配置"""
        from util.config import settings

        return {
            "retry_enabled": settings.RETRY_ENABLED,
            "retry_max_attempts": settings.RETRY_MAX_ATTEMPTS,
            "retry_delay_seconds": settings.RETRY_DELAY_SECONDS,
            "retry_exponential_backoff": settings.RETRY_EXPONENTIAL_BACKOFF,
            "retry_max_delay_seconds": settings.RETRY_MAX_DELAY_SECONDS,
        }

    @staticmethod
    def update_retry_config(
        retry_enabled: Optional[bool] = None,
        retry_max_attempts: Optional[int] = None,
        retry_delay_seconds: Optional[float] = None,
        retry_exponential_backoff: Optional[bool] = None,
        retry_max_delay_seconds: Optional[float] = None,
    ) -> dict:
        """更新重试配置"""
        from util.config import settings

        old_config = ConfigService.get_retry_config()
        changes = {}

        if retry_enabled is not None and retry_enabled != settings.RETRY_ENABLED:
            changes["retry_enabled"] = {"old": settings.RETRY_ENABLED, "new": retry_enabled}
            settings.RETRY_ENABLED = retry_enabled

        if retry_max_attempts is not None and retry_max_attempts != settings.RETRY_MAX_ATTEMPTS:
            changes["retry_max_attempts"] = {"old": settings.RETRY_MAX_ATTEMPTS, "new": retry_max_attempts}
            settings.RETRY_MAX_ATTEMPTS = retry_max_attempts

        if retry_delay_seconds is not None and retry_delay_seconds != settings.RETRY_DELAY_SECONDS:
            changes["retry_delay_seconds"] = {"old": settings.RETRY_DELAY_SECONDS, "new": retry_delay_seconds}
            settings.RETRY_DELAY_SECONDS = retry_delay_seconds

        if retry_exponential_backoff is not None and retry_exponential_backoff != settings.RETRY_EXPONENTIAL_BACKOFF:
            changes["retry_exponential_backoff"] = {"old": settings.RETRY_EXPONENTIAL_BACKOFF, "new": retry_exponential_backoff}
            settings.RETRY_EXPONENTIAL_BACKOFF = retry_exponential_backoff

        if retry_max_delay_seconds is not None and retry_max_delay_seconds != settings.RETRY_MAX_DELAY_SECONDS:
            changes["retry_max_delay_seconds"] = {"old": settings.RETRY_MAX_DELAY_SECONDS, "new": retry_max_delay_seconds}
            settings.RETRY_MAX_DELAY_SECONDS = retry_max_delay_seconds

        if changes:
            ConfigService._record_history("retry", old_config, changes)
            logger.info(f"重试配置已更新：{changes}")

        return ConfigService.get_retry_config()

    @staticmethod
    def get_debug_config() -> dict:
        """获取调试配置"""
        from util.config import settings

        return {
            "debug_mode": settings.DEBUG_MODE,
            "debug_delay_per_image": settings.DEBUG_DELAY_PER_IMAGE,
            "debug_delay_per_pair": settings.DEBUG_DELAY_PER_PAIR,
            "debug_failure_rate": settings.DEBUG_FAILURE_RATE,
        }

    @staticmethod
    def update_debug_config(
        debug_mode: Optional[bool] = None,
        debug_delay_per_image: Optional[float] = None,
        debug_delay_per_pair: Optional[float] = None,
        debug_failure_rate: Optional[float] = None,
    ) -> dict:
        """更新调试配置"""
        from util.config import settings

        old_config = ConfigService.get_debug_config()
        changes = {}

        if debug_mode is not None and debug_mode != settings.DEBUG_MODE:
            changes["debug_mode"] = {"old": settings.DEBUG_MODE, "new": debug_mode}
            settings.DEBUG_MODE = debug_mode

        if debug_delay_per_image is not None and debug_delay_per_image != settings.DEBUG_DELAY_PER_IMAGE:
            changes["debug_delay_per_image"] = {"old": settings.DEBUG_DELAY_PER_IMAGE, "new": debug_delay_per_image}
            settings.DEBUG_DELAY_PER_IMAGE = debug_delay_per_image

        if debug_delay_per_pair is not None and debug_delay_per_pair != settings.DEBUG_DELAY_PER_PAIR:
            changes["debug_delay_per_pair"] = {"old": settings.DEBUG_DELAY_PER_PAIR, "new": debug_delay_per_pair}
            settings.DEBUG_DELAY_PER_PAIR = debug_delay_per_pair

        if debug_failure_rate is not None and debug_failure_rate != settings.DEBUG_FAILURE_RATE:
            changes["debug_failure_rate"] = {"old": settings.DEBUG_FAILURE_RATE, "new": debug_failure_rate}
            settings.DEBUG_FAILURE_RATE = debug_failure_rate

        if changes:
            ConfigService._record_history("debug", old_config, changes)
            logger.info(f"调试配置已更新：{changes}")

        return ConfigService.get_debug_config()

    @staticmethod
    def get_storage_save_strategy_config() -> dict:
        """获取存储策略配置"""
        from util.config import settings

        return {
            "storage_auto_save": settings.STORAGE_AUTO_SAVE,
            "storage_save_strategy": settings.STORAGE_SAVE_STRATEGY,
            "storage_save_error_rate": settings.STORAGE_SAVE_ERROR_RATE,
            "storage_save_fake_rate": settings.STORAGE_SAVE_FAKE_RATE,
            "storage_save_real_rate": settings.STORAGE_SAVE_REAL_RATE,
            "storage_max_per_task": settings.STORAGE_MAX_PER_TASK,
        }

    @staticmethod
    def update_storage_save_strategy_config(
        storage_auto_save: Optional[bool] = None,
        storage_save_strategy: Optional[str] = None,
        storage_save_error_rate: Optional[float] = None,
        storage_save_fake_rate: Optional[float] = None,
        storage_save_real_rate: Optional[float] = None,
        storage_max_per_task: Optional[int] = None,
    ) -> dict:
        """更新存储策略配置"""
        from util.config import settings

        old_config = ConfigService.get_storage_save_strategy_config()
        changes = {}

        if storage_auto_save is not None and storage_auto_save != settings.STORAGE_AUTO_SAVE:
            changes["storage_auto_save"] = {"old": settings.STORAGE_AUTO_SAVE, "new": storage_auto_save}
            settings.STORAGE_AUTO_SAVE = storage_auto_save

        if storage_save_strategy is not None and storage_save_strategy != settings.STORAGE_SAVE_STRATEGY:
            changes["storage_save_strategy"] = {"old": settings.STORAGE_SAVE_STRATEGY, "new": storage_save_strategy}
            settings.STORAGE_SAVE_STRATEGY = storage_save_strategy

        if storage_save_error_rate is not None and storage_save_error_rate != settings.STORAGE_SAVE_ERROR_RATE:
            changes["storage_save_error_rate"] = {"old": settings.STORAGE_SAVE_ERROR_RATE, "new": storage_save_error_rate}
            settings.STORAGE_SAVE_ERROR_RATE = storage_save_error_rate

        if storage_save_fake_rate is not None and storage_save_fake_rate != settings.STORAGE_SAVE_FAKE_RATE:
            changes["storage_save_fake_rate"] = {"old": settings.STORAGE_SAVE_FAKE_RATE, "new": storage_save_fake_rate}
            settings.STORAGE_SAVE_FAKE_RATE = storage_save_fake_rate

        if storage_save_real_rate is not None and storage_save_real_rate != settings.STORAGE_SAVE_REAL_RATE:
            changes["storage_save_real_rate"] = {"old": settings.STORAGE_SAVE_REAL_RATE, "new": storage_save_real_rate}
            settings.STORAGE_SAVE_REAL_RATE = storage_save_real_rate

        if storage_max_per_task is not None and storage_max_per_task != settings.STORAGE_MAX_PER_TASK:
            changes["storage_max_per_task"] = {"old": settings.STORAGE_MAX_PER_TASK, "new": storage_max_per_task}
            settings.STORAGE_MAX_PER_TASK = storage_max_per_task

        if changes:
            ConfigService._record_history("storage_save_strategy", old_config, changes)
            logger.info(f"存储策略配置已更新：{changes}")

        return ConfigService.get_storage_save_strategy_config()

    @staticmethod
    def get_image_compress_config() -> dict:
        """获取图片压缩配置"""
        from util.config import settings

        return {
            "compress_enabled": settings.IMAGE_COMPRESS_ENABLED,
            "compress_quality": settings.IMAGE_COMPRESS_QUALITY,
            "compress_type": settings.IMAGE_COMPRESS_TYPE,
        }

    @staticmethod
    def update_image_compress_config(
        compress_enabled: Optional[bool] = None,
        compress_quality: Optional[int] = None,
        compress_type: Optional[str] = None,
    ) -> dict:
        """更新图片压缩配置"""
        from util.config import settings

        old_config = ConfigService.get_image_compress_config()
        changes = {}

        if compress_enabled is not None and compress_enabled != settings.IMAGE_COMPRESS_ENABLED:
            changes["compress_enabled"] = {"old": settings.IMAGE_COMPRESS_ENABLED, "new": compress_enabled}
            settings.IMAGE_COMPRESS_ENABLED = compress_enabled

        if compress_quality is not None and compress_quality != settings.IMAGE_COMPRESS_QUALITY:
            changes["compress_quality"] = {"old": settings.IMAGE_COMPRESS_QUALITY, "new": compress_quality}
            settings.IMAGE_COMPRESS_QUALITY = compress_quality

        if compress_type is not None and compress_type != settings.IMAGE_COMPRESS_TYPE:
            changes["compress_type"] = {"old": settings.IMAGE_COMPRESS_TYPE, "new": compress_type}
            settings.IMAGE_COMPRESS_TYPE = compress_type

        if changes:
            ConfigService._record_history("image_compress", old_config, changes)
            logger.info(f"图片压缩配置已更新：{changes}")

        return ConfigService.get_image_compress_config()

    @staticmethod
    def _record_history(config_type: str, old_config: dict, changes: dict) -> None:
        """记录配置历史"""
        global _config_history

        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "config_type": config_type,
            "old_config": old_config,
            "changes": changes,
        }

        _config_history.append(record)

        # 限制历史记录长度
        if len(_config_history) > _MAX_HISTORY_LENGTH:
            _config_history = _config_history[-_MAX_HISTORY_LENGTH:]

    @staticmethod
    def get_config_history(limit: int = 10) -> list[dict]:
        """获取配置历史"""
        return _config_history[-limit:]

    @staticmethod
    def rollback_config(
        config_type: str,
        target_index: int,
    ) -> dict:
        """回滚配置到历史版本"""
        global _config_history

        if target_index < 0 or target_index >= len(_config_history):
            raise ValueError(f"无效的历史记录索引：{target_index}")

        target_record = _config_history[target_index]

        if target_record["config_type"] != config_type:
            raise ValueError(f"配置类型不匹配：期望 {config_type}, 实际 {target_record['config_type']}")

        # 根据配置类型调用对应的更新方法
        if config_type == "logging":
            old = target_record["old_config"]
            return ConfigService.update_logging_config(
                log_level=old["log_level"],
                log_to_console=old["log_to_console"],
                log_json_format=old["log_json_format"],
                log_request_body=old["log_request_body"],
                log_response_body=old["log_response_body"],
                audit_log_enabled=old["audit_log_enabled"],
            )
        elif config_type == "retry":
            old = target_record["old_config"]
            return ConfigService.update_retry_config(
                retry_enabled=old["retry_enabled"],
                retry_max_attempts=old["retry_max_attempts"],
                retry_delay_seconds=old["retry_delay_seconds"],
                retry_exponential_backoff=old["retry_exponential_backoff"],
                retry_max_delay_seconds=old["retry_max_delay_seconds"],
            )
        elif config_type == "debug":
            old = target_record["old_config"]
            return ConfigService.update_debug_config(
                debug_mode=old["debug_mode"],
                debug_delay_per_image=old["debug_delay_per_image"],
                debug_delay_per_pair=old["debug_delay_per_pair"],
                debug_failure_rate=old["debug_failure_rate"],
            )
        elif config_type == "storage_save_strategy":
            old = target_record["old_config"]
            return ConfigService.update_storage_save_strategy_config(
                storage_auto_save=old["storage_auto_save"],
                storage_save_strategy=old["storage_save_strategy"],
                storage_save_error_rate=old["storage_save_error_rate"],
                storage_save_fake_rate=old["storage_save_fake_rate"],
                storage_save_real_rate=old["storage_save_real_rate"],
                storage_max_per_task=old["storage_max_per_task"],
            )
        elif config_type == "image_compress":
            old = target_record["old_config"]
            return ConfigService.update_image_compress_config(
                compress_enabled=old["compress_enabled"],
                compress_quality=old["compress_quality"],
                compress_type=old["compress_type"],
            )
        else:
            raise ValueError(f"不支持的配置类型：{config_type}")
