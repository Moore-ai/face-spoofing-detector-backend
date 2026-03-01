"""
检测结果持久化服务
提供存储、查询、统计、删除历史记录的功能
"""

import logging
from typing import Optional, List, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_

from db.models import DetectionTask, DetectionResult
from schemas.history import (
    HistoryTaskItem,
    HistoryResultItem,
    HistoryQueryResponse,
    HistoryStatsResponse,
)

logger = logging.getLogger(__name__)


class DetectionHistoryService:
    """检测结果历史服务"""

    @staticmethod
    def save_task(
        db: Session,
        task_id: str,
        client_id: Optional[str],
        api_key_hash: Optional[str],
        mode: str,
        status: str,
        total_items: int,
        successful_items: int,
        failed_items: int,
        real_count: int,
        fake_count: int,
        error_count: int,
        elapsed_time_ms: int,
        results: List[dict],
    ) -> DetectionTask:
        """
        保存任务历史记录

        Args:
            db: 数据库会话
            task_id: 任务 ID
            client_id: 客户端 ID
            api_key_hash: API Key 哈希
            mode: 模式 ("single" 或 "fusion")
            status: 任务状态
            total_items: 总项目数
            successful_items: 成功项目数
            failed_items: 失败项目数
            real_count: 真实人脸计数
            fake_count: 伪造人脸计数
            error_count: 错误计数
            elapsed_time_ms: 耗时（毫秒）
            results: 检测结果列表（每项是一个字典）

        Returns:
            保存的任务记录
        """
        try:
            # 创建任务记录
            task = DetectionTask(
                task_id=task_id,
                client_id=client_id,
                api_key_hash=api_key_hash,
                mode=mode,
                status=status,
                total_items=total_items,
                successful_items=successful_items,
                failed_items=failed_items,
                real_count=real_count,
                fake_count=fake_count,
                error_count=error_count,
                elapsed_time_ms=elapsed_time_ms,
                completed_at=datetime.utcnow(),
            )
            db.add(task)

            # 创建结果记录
            for result_data in results:
                result = DetectionResult(
                    task_id=task_id,
                    image_index=result_data.get("image_index"),
                    mode=result_data.get("mode", mode),
                    modality=result_data.get("modality"),
                    result=result_data.get("result"),
                    confidence=result_data.get("confidence", 0.0),
                    prob_real=result_data.get("probabilities", [0.0, 0.0])[0],
                    prob_fake=result_data.get("probabilities", [0.0, 0.0])[1],
                    processing_time=result_data.get("processing_time", 0),
                    is_error=result_data.get("result") == "error" or not result_data.get("success", True),
                    error_message=result_data.get("error"),
                    retry_count=result_data.get("retry_count", 0),
                )
                db.add(result)

            db.commit()
            logger.info(f"Saved task {task_id} with {len(results)} results to database")
            return task

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to save task {task_id}: {e}", exc_info=True)
            raise

    @staticmethod
    def get_task_by_id(db: Session, task_id: str) -> Optional[DetectionTask]:
        """
        根据任务 ID 获取历史记录

        Args:
            db: 数据库会话
            task_id: 任务 ID

        Returns:
            任务记录，如果不存在则返回 None
        """
        return db.query(DetectionTask).filter(DetectionTask.task_id == task_id).first()

    @staticmethod
    def query_tasks(
        db: Session,
        client_id: Optional[str] = None,
        api_key_hash: Optional[str] = None,
        mode: Optional[str] = None,
        status: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> HistoryQueryResponse:
        """
        查询历史记录

        Args:
            db: 数据库会话
            client_id: 客户端 ID 过滤
            api_key_hash: API Key 哈希过滤
            mode: 模式过滤 ("single" 或 "fusion")
            status: 状态过滤列表
            start_date: 开始日期
            end_date: 结束日期
            page: 页码（从 1 开始）
            page_size: 每页数量

        Returns:
            分页查询结果
        """
        query = db.query(DetectionTask)

        # 应用过滤条件
        if client_id:
            query = query.filter(DetectionTask.client_id == client_id)
        if api_key_hash:
            query = query.filter(DetectionTask.api_key_hash == api_key_hash)
        if mode:
            query = query.filter(DetectionTask.mode == mode)
        if status:
            query = query.filter(DetectionTask.status.in_(status))
        if start_date:
            query = query.filter(DetectionTask.created_at >= start_date)
        if end_date:
            query = query.filter(DetectionTask.created_at <= end_date)

        # 获取总数
        total = query.count()

        # 分页查询
        offset = (page - 1) * page_size
        tasks = query.order_by(DetectionTask.created_at.desc()).offset(offset).limit(page_size).all()

        # 转换为响应格式
        items = []
        for task in tasks:
            result_items = []
            for result in task.results:
                result_items.append(
                    HistoryResultItem(
                        mode=result.mode,
                        modality=result.modality,
                        result=result.result,
                        confidence=result.confidence,
                        probabilities=[result.prob_real, result.prob_fake],
                        processing_time=result.processing_time,
                        image_index=result.image_index,
                        error=result.error_message,
                        retry_count=result.retry_count,
                    )
                )

            items.append(
                HistoryTaskItem(
                    task_id=task.task_id, # type: ignore
                    client_id=task.client_id, # type: ignore
                    api_key_hash=task.api_key_hash, # type: ignore
                    mode=task.mode, # type: ignore
                    status=task.status, # type: ignore
                    total_items=int(task.total_items), # type: ignore
                    successful_items=int(task.successful_items), # type: ignore
                    failed_items=int(task.failed_items), # type: ignore
                    real_count=task.real_count, # type: ignore
                    fake_count=task.fake_count, # type: ignore
                    elapsed_time_ms=task.elapsed_time_ms, # type: ignore
                    created_at=task.created_at, # type: ignore
                    completed_at=task.completed_at, # type: ignore
                    results=result_items,
                )
            )

        total_pages = (total + page_size - 1) // page_size

        return HistoryQueryResponse(
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            items=items,
        )

    @staticmethod
    def get_stats(
        db: Session,
        client_id: Optional[str] = None,
        api_key_hash: Optional[str] = None,
        mode: Optional[str] = None,
        status: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> HistoryStatsResponse:
        """
        获取统计信息

        Args:
            db: 数据库会话
            client_id: 客户端 ID 过滤
            api_key_hash: API Key 哈希过滤
            mode: 模式过滤 ("single" 或 "fusion")
            status: 状态过滤列表
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            统计信息
        """
        # 应用过滤条件
        filter_conditions = []
        if client_id:
            filter_conditions.append(DetectionTask.client_id == client_id)
        if api_key_hash:
            filter_conditions.append(DetectionTask.api_key_hash == api_key_hash)
        if mode:
            filter_conditions.append(DetectionTask.mode == mode)
        if status:
            filter_conditions.append(DetectionTask.status.in_(status))
        if start_date:
            filter_conditions.append(DetectionTask.created_at >= start_date)
        if end_date:
            filter_conditions.append(DetectionTask.created_at <= end_date)

        # 聚合统计（使用 SQLAlchemy 传统方式）
        stats = db.query(
            func.count(DetectionTask.id).label("total_tasks"),
            func.sum(DetectionTask.total_items).label("total_inferences"),
            func.sum(DetectionTask.real_count).label("total_real"),
            func.sum(DetectionTask.fake_count).label("total_fake"),
            func.sum(DetectionTask.error_count).label("total_errors"),
            func.sum(DetectionTask.successful_items).label("total_successful"),
            func.avg(DetectionTask.elapsed_time_ms).label("avg_time"),
        ).filter(*filter_conditions).first()

        # 获取日期范围
        date_range_query = db.query(
            func.min(DetectionTask.created_at).label("start_date"),
            func.max(DetectionTask.created_at).label("end_date"),
        ).filter(*filter_conditions)
        date_range = date_range_query.first()

        # 计算成功率：成功推理数 / 总推理数 * 100
        # 成功推理数 = real_count + fake_count（不包括 error）
        # 总推理数 = real_count + fake_count + error_count
        total_real = stats.total_real or 0  # type: ignore
        total_fake = stats.total_fake or 0  # type: ignore
        total_errors = stats.total_errors or 0  # type: ignore

        total_valid = total_real + total_fake
        total_all = total_valid + total_errors

        success_rate = (total_all > 0) and (total_valid / total_all * 100) or 0.0

        return HistoryStatsResponse(
            total_tasks=stats.total_tasks or 0,  # type: ignore
            total_inferences=stats.total_inferences or 0,  # type: ignore
            total_real=total_real,
            total_fake=total_fake,
            total_errors=total_errors,
            success_rate=round(success_rate, 2),
            avg_processing_time_ms=round(stats.avg_time or 0, 2),  # type: ignore
            date_range={
                "start": date_range.start_date.isoformat() if date_range.start_date else None,  # type: ignore
                "end": date_range.end_date.isoformat() if date_range.end_date else None,  # type: ignore
            },
        )

    @staticmethod
    def delete_tasks(
        db: Session,
        task_ids: Optional[List[str]] = None,
        client_id: Optional[str] = None,
        api_key_hash: Optional[str] = None,
        older_than_days: Optional[int] = None,
    ) -> int:
        """
        删除历史记录

        Args:
            db: 数据库会话
            task_ids: 要删除的任务 ID 列表
            client_id: 客户端 ID 过滤
            api_key_hash: API Key 哈希过滤（用于普通用户删除自己的记录）
            older_than_days: 删除早于指定天数的记录

        Returns:
            删除的记录数

        Note:
            参数互斥，只能使用其中一个
        """
        try:
            query = db.query(DetectionTask)

            if task_ids:
                query = query.filter(DetectionTask.task_id.in_(task_ids))
                # 如果提供了 api_key_hash，额外过滤确保用户只能删除自己的记录
                if api_key_hash:
                    query = query.filter(DetectionTask.api_key_hash == api_key_hash)
            elif client_id:
                query = query.filter(DetectionTask.client_id == client_id)
            elif older_than_days:
                cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)
                query = query.filter(DetectionTask.created_at < cutoff_date)
            else:
                raise ValueError("必须指定删除条件：task_ids, client_id, 或 older_than_days")

            count = query.count()
            query.delete(synchronize_session=False)
            db.commit()

            logger.info(f"Deleted {count} task(s) from database")
            return count

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to delete tasks: {e}", exc_info=True)
            raise

    @staticmethod
    def get_user_usage_stats(
        db: Session,
        api_key_hash: str,
        days: int = 30,
    ) -> dict:
        """
        获取用户使用统计（用于激活码使用分析）

        Args:
            db: 数据库会话
            api_key_hash: API Key 哈希
            days: 统计天数

        Returns:
            使用统计字典
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        stats = (
            db.query(
                func.count(DetectionTask.id).label("total_tasks"),
                func.sum(DetectionTask.total_items).label("total_inferences"),
                func.sum(DetectionTask.successful_items).label("total_successful"),
            )
            .filter(
                and_(
                    DetectionTask.api_key_hash == api_key_hash,
                    DetectionTask.created_at >= cutoff_date,
                )
            )
            .first()
        )

        return {
            "total_tasks": stats.total_tasks or 0, # type: ignore
            "total_inferences": stats.total_inferences or 0, # type: ignore
            "total_successful": stats.total_successful or 0, # type: ignore
            "period_days": days,
        }
