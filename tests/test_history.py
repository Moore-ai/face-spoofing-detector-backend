"""
检测结果持久化功能测试脚本

测试内容：
1. 数据库初始化
2. 保存历史记录
3. 查询历史记录
4. 获取统计信息
5. 删除历史记录

运行方式：
    python tests/test_history.py

注意：
- 此测试无需服务运行，直接测试数据库层
- 测试数据存储在测试数据库中，测试完成后会清理
"""

import os
import sys
import logging

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 设置测试环境变量
os.environ['DATABASE_URL'] = 'sqlite:///./db/test_detections.db'


def test_database_initialization():
    """测试 1: 数据库初始化"""
    logger.info("=" * 60)
    logger.info("测试 1: 数据库初始化")
    logger.info("=" * 60)

    from db import db_manager

    # 先删除旧的测试数据库（如果存在）
    import os
    test_db_path = "db/test_detections.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
        logger.info(f"已删除旧的测试数据库：{test_db_path}")

    # 初始化数据库
    db_manager.initialize()
    db_manager.create_tables()

    logger.info("✓ 数据库初始化成功")
    return True


def test_save_history():
    """测试 2: 保存历史记录"""
    logger.info("=" * 60)
    logger.info("测试 2: 保存历史记录")
    logger.info("=" * 60)

    from db import db_manager, hash_api_key
    from service.history_service import DetectionHistoryService

    # 获取数据库会话
    db = db_manager.get_session()

    try:
        # 准备测试数据
        task_id = "test_task_001"
        client_id = "test_client_001"
        api_key = "test_api_key_12345"
        api_key_hash = hash_api_key(api_key)

        results = [
            {
                "mode": "single",
                "modality": "rgb",
                "result": "real",
                "confidence": 0.95,
                "probabilities": [0.95, 0.05],
                "processing_time": 45,
                "image_index": 0,
                "error": None,
                "retry_count": 0,
                "success": True,
            },
            {
                "mode": "single",
                "modality": "rgb",
                "result": "fake",
                "confidence": 0.85,
                "probabilities": [0.15, 0.85],
                "processing_time": 52,
                "image_index": 1,
                "error": None,
                "retry_count": 0,
                "success": True,
            },
            {
                "mode": "single",
                "modality": "rgb",
                "result": "error",
                "confidence": 0.0,
                "probabilities": [0.0, 0.0],
                "processing_time": 0,
                "image_index": 2,
                "error": "模型推理失败",
                "retry_count": 3,
                "success": False,
            },
        ]

        # 保存历史记录
        task = DetectionHistoryService.save_task(
            db=db,
            task_id=task_id,
            client_id=client_id,
            api_key_hash=api_key_hash,
            mode="single",
            status="partial_failure",
            total_items=3,
            successful_items=2,
            failed_items=1,
            real_count=1,
            fake_count=1,
            error_count=1,
            elapsed_time_ms=150,
            results=results,
        )

        logger.info(f"✓ 历史记录保存成功：task_id={task_id}")
        logger.info(f"  - 任务状态：{task.status}")
        logger.info(f"  - 总项目数：{task.total_items}")
        logger.info(f"  - 成功：{task.successful_items}, 失败：{task.failed_items}")

        return True

    finally:
        db.close()


def test_query_history():
    """测试 3: 查询历史记录"""
    logger.info("=" * 60)
    logger.info("测试 3: 查询历史记录")
    logger.info("=" * 60)

    from db import db_manager
    from service.history_service import DetectionHistoryService

    db = db_manager.get_session()

    try:
        # 查询所有记录
        result = DetectionHistoryService.query_tasks(
            db=db,
            page=1,
            page_size=10,
        )

        logger.info(f"✓ 查询成功")
        logger.info(f"  - 总记录数：{result.total}")
        logger.info(f"  - 当前页数：{result.page}")
        logger.info(f"  - 每页数量：{result.page_size}")

        # 显示第一条记录
        if result.items:
            item = result.items[0]
            logger.info(f"  - 第一条记录:")
            logger.info(f"    task_id: {item.task_id}")
            logger.info(f"    status: {item.status}")
            logger.info(f"    results count: {len(item.results)}") # type: ignore

        # 按模式过滤查询
        result_single = DetectionHistoryService.query_tasks(
            db=db,
            mode="single",
            page=1,
            page_size=10,
        )
        logger.info(f"✓ 按模式过滤查询成功 (single): {result_single.total} 条记录")

        return True

    finally:
        db.close()


def test_get_stats():
    """测试 4: 获取统计信息"""
    logger.info("=" * 60)
    logger.info("测试 4: 获取统计信息")
    logger.info("=" * 60)

    from db import db_manager
    from service.history_service import DetectionHistoryService

    db = db_manager.get_session()

    try:
        stats = DetectionHistoryService.get_stats(db=db)

        logger.info(f"✓ 统计信息获取成功")
        logger.info(f"  - 总任务数：{stats.total_tasks}")
        logger.info(f"  - 总推理数：{stats.total_inferences}")
        logger.info(f"  - 真实人脸：{stats.total_real}")
        logger.info(f"  - 伪造人脸：{stats.total_fake}")
        logger.info(f"  - 错误数：{stats.total_errors}")
        logger.info(f"  - 成功率：{stats.success_rate}%")
        logger.info(f"  - 平均处理时间：{stats.avg_processing_time_ms}ms")

        return True

    finally:
        db.close()


def test_delete_history():
    """测试 5: 删除历史记录"""
    logger.info("=" * 60)
    logger.info("测试 5: 删除历史记录")
    logger.info("=" * 60)

    from db import db_manager
    from service.history_service import DetectionHistoryService

    db = db_manager.get_session()

    try:
        # 删除测试任务
        deleted_count = DetectionHistoryService.delete_tasks(
            db=db,
            task_ids=["test_task_001"],
        )

        logger.info(f"✓ 删除成功：{deleted_count} 条记录")

        # 验证删除
        result = DetectionHistoryService.query_tasks(db=db, page=1, page_size=10)
        logger.info(f"✓ 删除后剩余记录：{result.total} 条")

        return True

    finally:
        db.close()


def cleanup_test_database():
    """清理测试数据库"""
    logger.info("=" * 60)
    logger.info("清理测试数据库")
    logger.info("=" * 60)

    from db import db_manager

    # 先关闭数据库连接
    try:
        db_manager.close()
    except Exception:
        pass

    # 等待一下确保连接已关闭
    import time
    time.sleep(0.5)

    import os
    test_db_path = "db/test_detections.db"
    if os.path.exists(test_db_path):
        try:
            os.remove(test_db_path)
            logger.info(f"✓ 测试数据库已删除：{test_db_path}")
        except PermissionError:
            logger.warning(f"无法删除测试数据库（可能被占用）: {test_db_path}")


def main():
    """运行所有测试"""
    logger.info("=" * 60)
    logger.info("检测结果持久化功能测试")
    logger.info("=" * 60)

    try:
        # 运行测试
        test_database_initialization()
        test_save_history()
        test_query_history()
        test_get_stats()
        test_delete_history()

        logger.info("=" * 60)
        logger.info("所有测试通过 ✓")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"测试失败：{e}", exc_info=True)
        return 1

    finally:
        # 清理测试数据库
        cleanup_test_database()

    return 0


if __name__ == "__main__":
    sys.exit(main())
