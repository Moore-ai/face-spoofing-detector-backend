"""
任务调度器功能测试
测试优先级任务调度器的正确性
"""

import asyncio
from service.task_scheduler import task_scheduler


async def test_task_scheduler():
    """测试任务调度器"""
    print("=" * 60)
    print("任务调度器功能测试")
    print("=" * 60)

    # 1. 启动调度器
    print("\n[1] 启动任务调度器...")
    await task_scheduler.start()

    status = await task_scheduler.get_queue_status()
    print(f"  调度器状态: {status}")

    # 2. 提交多个不同优先级的任务
    print("\n[2] 提交多个不同优先级的任务...")

    tasks_data = [
        {"task_id": "task_low", "priority": 30, "description": "低优先级任务"},
        {"task_id": "task_medium", "priority": 70, "description": "中优先级任务"},
        {"task_id": "task_high", "priority": 90, "description": "高优先级任务"},
        {"task_id": "task_vip", "priority": 100, "description": "VIP 优先级任务"},
    ]

    async def dummy_callback(task_data):
        """模拟任务执行"""
        print(f"  执行任务: {task_data['task_id']} ({task_data['description']})")
        await asyncio.sleep(0.1)  # 模拟任务处理时间
        print(f"  任务完成: {task_data['task_id']}")

    # 提交任务
    for task_data in tasks_data:
        success = await task_scheduler.submit_task(
            task_id=task_data["task_id"],
            task_type="test",
            task_data=task_data,
            priority=task_data["priority"],
            callback=dummy_callback
        )
        if success:
            print(f"  [OK] 已提交: {task_data['description']} (priority={task_data['priority']})")
        else:
            print(f"  [FAIL] 提交失败: {task_data['description']}")

    # 等待队列处理
    await asyncio.sleep(0.5)

    queue_size = await task_scheduler.task_queue.get_queue_size()
    print(f"\n[3] 队列状态:")
    print(f"  队列大小: {queue_size}")

    # 4. 获取所有任务
    print("\n[4] 获取队列中的所有任务...")
    all_tasks = await task_scheduler.task_queue.get_all_tasks()
    for task in all_tasks:
        print(f"  任务: {task.task_id}, 优先级: {-task.priority}, 状态: {task.status}")

    # 5. 测试取消任务
    print("\n[5] 测试取消队列中的任务...")
    cancel_success = await task_scheduler.cancel_task("task_low")
    if cancel_success:
        print("  [OK] 成功取消任务")
    else:
        print("  [FAIL] 取消任务失败")

    # 6. 获取队列状态
    status = await task_scheduler.get_queue_status()
    print(f"\n[6] 最终队列状态:")
    print(f"  {status}")

    # 7. 停止调度器
    print("\n[7] 停止调度器...")
    await task_scheduler.stop()
    print("  [OK] 调度器已停止")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_task_scheduler())
