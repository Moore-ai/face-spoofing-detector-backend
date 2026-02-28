"""
数据库连接管理模块
支持 SQLite 和 PostgreSQL
"""

import logging
from typing import Optional, Generator
import hashlib

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session

from util.config import settings

logger = logging.getLogger(__name__)

# SQLAlchemy 基类
Base = declarative_base()


class DatabaseManager:
    """数据库管理器"""

    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self._initialized = False

    def initialize(self, database_url: Optional[str] = None):
        """
        初始化数据库连接

        Args:
            database_url: 数据库连接 URL
                - SQLite: "sqlite:///./db/detections.db"
                - PostgreSQL: "postgresql://user:password@host:port/dbname"
        """
        if self._initialized:
            logger.warning("Database already initialized")
            return

        # 使用默认 SQLite 数据库
        if not database_url:
            database_url = getattr(settings, "DATABASE_URL", "sqlite:///./db/detections.db")

        # 确保 database_url 不为 None
        assert database_url is not None, "DATABASE_URL 不能为空"

        logger.info(f"Initializing database connection: {database_url}")

        # 创建引擎
        if database_url.startswith("sqlite"):
            self.engine = create_engine(
                database_url,
                connect_args={"check_same_thread": False},  # SQLite 需要
                echo=False,
            )
        else:
            # PostgreSQL 或其他数据库
            self.engine = create_engine(
                database_url,
                pool_size=10,
                max_overflow=20,
                echo=False,
            )

        # 创建会话工厂
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
        )

        self._initialized = True
        logger.info("Database connection initialized successfully")

    def create_tables(self):
        """创建所有表"""
        if not self._initialized:
            raise RuntimeError("Database not initialized")

        logger.info("Creating database tables...")
        # 导入模型以确保它们被注册到 Base metadata
        from db import models  # noqa: F401
        models  # type: ignore # 避免 linter 警告
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created successfully")

    def get_session(self) -> Session:
        """获取数据库会话"""
        if not self._initialized:
            raise RuntimeError("Database not initialized")
        if self.SessionLocal is None:
            raise RuntimeError("SessionLocal not initialized")

        session = self.SessionLocal()
        try:
            return session
        except Exception:
            session.close()
            raise

    def close(self):
        """关闭数据库连接"""
        if self.engine:
            self.engine.dispose()
            self._initialized = False
            logger.info("Database connection closed")


def hash_api_key(api_key: str) -> str:
    """
    对 API Key 进行哈希（用于追踪，不存储原始 key）

    Args:
        api_key: 原始 API Key

    Returns:
        SHA256 哈希值（截断）
    """
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]


# 全局数据库管理器实例
db_manager = DatabaseManager()


def get_db_session() -> Generator[Session, None, None]:
    """
    获取数据库会话的依赖注入函数

    Yields:
        SQLAlchemy Session
    """
    session = db_manager.get_session()
    try:
        yield session
    finally:
        session.close()
