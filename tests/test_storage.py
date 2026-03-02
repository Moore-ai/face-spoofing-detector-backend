"""
图片存储功能测试脚本

测试内容：
1. 存储初始化
2. 上传图片
3. 查询图片列表
4. 获取存储统计
5. 删除图片
6. 清理过期图片
7. 获取所有图片 ID 列表
8. 图片压缩功能测试
9. 存储配额测试
10. 批量下载功能测试（API 层）

运行方式：
    python tests/test_storage.py

注意：
- 此测试无需服务运行，直接测试存储层
- 测试数据存储在测试存储目录中，测试完成后会清理
- 如需测试 API 端点，请先启动服务并配置 JWT_TOKEN
"""

import os
import sys
import logging
import base64

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 设置测试环境变量
os.environ['STORAGE_TYPE'] = 'local'
os.environ['STORAGE_LOCAL_PATH'] = 'storage/test_images'
os.environ['STORAGE_QUOTA_BYTES'] = '104857600'  # 100MB
os.environ['IMAGE_COMPRESS_ENABLED'] = 'true'
os.environ['IMAGE_COMPRESS_TYPE'] = 'opencv'
os.environ['IMAGE_COMPRESS_QUALITY'] = '75'


def create_test_image_data() -> str:
    """创建测试图片数据（简单的 base64 编码）"""
    # 创建一个简单的 1x1 像素的 JPEG 图片（最小有效 JPEG）
    # 实际使用中应该是真实的图片数据
    test_data = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7telerit(telerit(telerit(telerit(telerit(telerit(telerit(telerit\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a%&\'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xd5\xff\xd9'
    return base64.b64encode(test_data).decode('utf-8')


def create_larger_test_image_data(width: int = 100, height: int = 100) -> str:
    """创建较大的测试图片数据（用于测试压缩）"""
    try:
        import numpy as np
        import cv2

        # 创建一个渐变的彩色图片
        img = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                img[i, j] = [
                    int(255 * i / height),  # B
                    int(255 * j / width),   # G
                    128                      # R
                ]

        # 编码为 JPEG
        success, encoded = cv2.imencode('.jpg', img, [
            cv2.IMWRITE_JPEG_QUALITY,
            90,  # 使用较高质量以便测试压缩效果
        ])

        if success:
            return base64.b64encode(encoded.tobytes()).decode('utf-8')
        else:
            logger.warning("Failed to encode test image, using fallback")
    except Exception as e:
        logger.warning(f"Failed to create larger test image: {e}, using fallback")

    # 回退到简单的测试图片
    return create_test_image_data()


def test_storage_initialization():
    """测试 1: 存储初始化"""
    logger.info("=" * 60)
    logger.info("测试 1: 存储初始化")
    logger.info("=" * 60)

    from util.storage import storage_manager

    # 先删除测试目录（如果存在）
    import shutil
    test_dir = "storage/test_images"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        logger.info(f"已删除旧的测试存储目录：{test_dir}")

    # 初始化存储管理器
    storage_manager.initialize(
        storage_type="local",
        storage_path=test_dir,
        quota_bytes=104857600,  # 100MB
    )

    logger.info("✓ 存储管理器初始化成功")
    return True


def test_upload_image():
    """测试 2: 上传图片"""
    logger.info("=" * 60)
    logger.info("测试 2: 上传图片")
    logger.info("=" * 60)

    from util.storage import storage_manager

    # 创建测试图片数据
    image_data = create_test_image_data()
    task_id = "test_task_001"

    # 上传图片
    result = storage_manager.save_image(
        task_id=task_id,
        image_data=image_data,
        image_type="original",
        modality="rgb",
        metadata={"test": True, "description": "测试图片"},
    )

    logger.info(f"✓ 图片上传成功")
    logger.info(f"  - image_id: {result['image_id']}")
    logger.info(f"  - storage_path: {result['storage_path']}")
    logger.info(f"  - file_size: {result['file_size']} 字节")

    # 上传多张图片用于后续测试
    logger.info("上传额外两张图片用于后续测试...")
    for i in range(2):
        storage_manager.save_image(
            task_id=task_id,
            image_data=image_data,
            image_type="processed" if i % 2 == 0 else "original",
            modality="ir" if i % 2 == 0 else "rgb",
            metadata={"batch_index": i},
        )
    logger.info("✓ 额外图片上传成功")

    return result['image_id']


def test_query_images(image_id):
    """测试 3: 查询图片"""
    logger.info("=" * 60)
    logger.info("测试 3: 查询图片")
    logger.info("=" * 60)

    from util.storage import storage_manager

    # 查询单个图片
    metadata = storage_manager.backend.get_metadata(image_id) # type: ignore
    if metadata:
        logger.info(f"✓ 查询图片成功")
        logger.info(f"  - task_id: {metadata.get('task_id')}")
        logger.info(f"  - image_type: {metadata.get('image_type')}")
        logger.info(f"  - modality: {metadata.get('modality')}")
        logger.info(f"  - file_size: {metadata.get('file_size')} 字节")
    else:
        logger.error("查询图片失败：未找到元数据")
        return False

    # 查询图片列表
    logger.info("查询任务图片列表...")
    result = storage_manager.query_images(
        task_id="test_task_001",
        page=1,
        page_size=10,
    )

    logger.info(f"✓ 查询列表成功")
    logger.info(f"  - 总数：{result['total']}")
    logger.info(f"  - 页码：{result['page']}")
    logger.info(f"  - 每页数量：{result['page_size']}")

    return True


def test_get_stats():
    """测试 4: 获取存储统计"""
    logger.info("=" * 60)
    logger.info("测试 4: 获取存储统计")
    logger.info("=" * 60)

    from util.storage import storage_manager

    stats = storage_manager.get_stats()

    logger.info(f"✓ 统计信息获取成功")
    logger.info(f"  - 总图片数：{stats['total_images']}")
    logger.info(f"  - 总存储空间：{stats['total_size_mb']} MB")
    logger.info(f"  - 配额：{stats['quota_bytes'] / (1024 * 1024):.0f} MB")
    logger.info(f"  - 配额使用率：{stats['quota_used_percent']:.2f}%")
    logger.info(f"  - 按类型统计：{stats['by_type']}")
    logger.info(f"  - 按模态统计：{stats['by_modality']}")

    return True


def test_delete_image(image_id):
    """测试 5: 删除图片"""
    logger.info("=" * 60)
    logger.info("测试 5: 删除图片")
    logger.info("=" * 60)

    from util.storage import storage_manager

    # 删除指定图片
    deleted = storage_manager.delete_image(image_id)

    if deleted:
        logger.info(f"✓ 图片删除成功：{image_id}")
    else:
        logger.error(f"删除图片失败：{image_id}")
        return False

    # 验证删除
    metadata = storage_manager.backend.get_metadata(image_id) # type: ignore
    if metadata is None:
        logger.info("✓ 验证删除成功：元数据已移除")
    else:
        logger.error("验证失败：元数据仍然存在")
        return False

    return True


def test_cleanup_images():
    """测试 6: 清理过期图片"""
    logger.info("=" * 60)
    logger.info("测试 6: 清理过期图片")
    logger.info("=" * 60)

    from util.storage import storage_manager

    # 由于测试图片都是刚上传的，不会被清理
    # 这里只是测试清理功能是否正常工作
    result = storage_manager.cleanup_images(
        older_than_days=1,  # 清理 1 天前的图片
    )

    logger.info(f"✓ 清理完成")
    logger.info(f"  - 删除图片数：{result['deleted_count']}")
    logger.info(f"  - 释放空间：{result['freed_size_bytes']} 字节")

    return True


def test_get_all_image_ids():
    """测试 7: 获取所有图片 ID 列表"""
    logger.info("=" * 60)
    logger.info("测试 7: 获取所有图片 ID 列表")
    logger.info("=" * 60)

    from util.storage import storage_manager

    # 获取所有图片 ID
    result = storage_manager.query_images(
        page=1,
        page_size=10000,
    )

    image_ids = [item.get("image_id", "") for item in result.get("items", [])]

    logger.info(f"✓ 获取图片 ID 列表成功")
    logger.info(f"  - 总图片数：{result.get('total', 0)}")
    logger.info(f"  - 图片 ID 数量：{len(image_ids)}")
    logger.info(f"  - ID 列表：{image_ids[:5]}..." if len(image_ids) > 5 else f"  - ID 列表：{image_ids}")

    return True


def test_image_compression():
    """测试 8: 图片压缩功能测试"""
    logger.info("=" * 60)
    logger.info("测试 8: 图片压缩功能测试")
    logger.info("=" * 60)

    from util.image_compressor import CompressorFactory, ImageCompressionConfig

    # 创建较大的测试图片
    image_data = create_larger_test_image_data(100, 100)
    original_bytes = len(base64.b64decode(image_data))
    logger.info(f"原始图片大小：{original_bytes} 字节")

    # 测试 OpenCV 压缩器
    try:
        compressor = CompressorFactory.create("opencv")
        compressed_data = compressor.compress_from_base64(image_data, quality=75)
        compressed_bytes = len(base64.b64decode(compressed_data))
        compression_ratio = (1 - compressed_bytes / original_bytes) * 100 if original_bytes > 0 else 0

        logger.info(f"✓ OpenCV 压缩器测试成功")
        logger.info(f"  - 压缩后大小：{compressed_bytes} 字节")
        logger.info(f"  - 压缩率：{compression_ratio:.1f}%")

        # 验证压缩后的图片可以解码
        import cv2
        import numpy as np
        np_array = np.frombuffer(base64.b64decode(compressed_data), np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        if img is not None:
            logger.info(f"  - 压缩图片验证成功：{img.shape}")
        else:
            logger.error("  - 压缩图片验证失败：无法解码")
            return False

    except Exception as e:
        logger.error(f"OpenCV 压缩器测试失败：{e}")
        return False

    # 测试 Pillow 压缩器（如果可用）
    try:
        compressor = CompressorFactory.create("pillow")
        compressed_data = compressor.compress_from_base64(image_data, quality=75)
        compressed_bytes = len(base64.b64decode(compressed_data))
        compression_ratio = (1 - compressed_bytes / original_bytes) * 100

        logger.info(f"✓ Pillow 压缩器测试成功")
        logger.info(f"  - 压缩后大小：{compressed_bytes} 字节")
        logger.info(f"  - 压缩率：{compression_ratio:.1f}%")

    except ImportError:
        logger.info("  - Pillow 未安装，跳过测试")
    except Exception as e:
        logger.warning(f"Pillow 压缩器测试失败：{e}")

    return True


def test_storage_quota():
    """测试 9: 存储配额测试"""
    logger.info("=" * 60)
    logger.info("测试 9: 存储配额测试")
    logger.info("=" * 60)

    from util.storage import storage_manager, StorageQuotaExceededError

    # 获取当前统计
    stats = storage_manager.get_stats()
    current_size = stats['total_size_bytes']
    quota = stats['quota_bytes']

    logger.info(f"当前存储使用：{current_size} 字节")
    logger.info(f"当前配额限制：{quota} 字节")

    # 测试配额更新
    if storage_manager.backend:
        # 设置一个很小的配额来测试超限
        original_quota = storage_manager.backend.quota_bytes  # type: ignore
        storage_manager.backend.update_quota(100)  # type: ignore

        logger.info("配额已更新为 100 字节，测试超限...")

        # 尝试上传一张图片（应该会失败）
        image_data = create_test_image_data()
        try:
            storage_manager.save_image(
                task_id="quota_test",
                image_data=image_data,
                image_type="original",
                modality="rgb",
            )
            logger.warning("  - 配额超限测试失败：应该抛出异常")
            # 恢复原始配额
            storage_manager.backend.update_quota(original_quota)  # type: ignore
            return False
        except StorageQuotaExceededError:
            logger.info("✓ 配额超限测试成功：正确抛出异常")
        except Exception as e:
            logger.error(f"配额超限测试异常：{e}")
            storage_manager.backend.update_quota(original_quota)  # type: ignore
            return False

        # 恢复原始配额
        storage_manager.backend.update_quota(original_quota)  # type: ignore
        logger.info(f"配额已恢复为 {original_quota} 字节")

    return True


def test_api_endpoints():
    """测试 10: API 端点测试（需要服务运行）"""
    logger.info("=" * 60)
    logger.info("测试 10: API 端点测试（需要服务运行）")
    logger.info("=" * 60)

    # 检查是否配置了 JWT Token
    jwt_token = os.environ.get('JWT_TOKEN')
    if not jwt_token:
        logger.info("  - 未配置 JWT_TOKEN，跳过 API 端点测试")
        logger.info("  - 如需测试 API 端点，请设置环境变量 JWT_TOKEN")
        return True

    base_url = os.environ.get('API_BASE_URL', 'http://127.0.0.1:8000')

    try:
        import requests

        headers = {
            'Authorization': f'Bearer {jwt_token}',
            'Content-Type': 'application/json',
        }

        # 测试 1: 获取所有图片 ID
        logger.info("测试 GET /storage/image-ids ...")
        response = requests.get(f'{base_url}/storage/image-ids', headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()
            logger.info(f"✓ /storage/image-ids 响应成功")
            logger.info(f"  - 总图片数：{data.get('total', 0)}")
            logger.info(f"  - 返回 ID 数：{len(data.get('image_ids', []))}")
        else:
            logger.error(f"✗ /storage/image-ids 失败：{response.status_code}")
            return False

        # 测试 2: 获取存储统计
        logger.info("测试 GET /storage/stats ...")
        response = requests.get(f'{base_url}/storage/stats', headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()
            logger.info(f"✓ /storage/stats 响应成功")
            logger.info(f"  - 总图片数：{data.get('total_images', 0)}")
            logger.info(f"  - 总存储空间：{data.get('total_size_mb', 0)} MB")
        else:
            logger.error(f"✗ /storage/stats 失败：{response.status_code}")
            return False

        # 测试 3: 批量下载（如果有图片）
        response = requests.get(f'{base_url}/storage/image-ids', headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            image_ids = data.get('image_ids', [])[:3]  # 最多测试 3 张

            if image_ids:
                logger.info(f"测试 POST /storage/images/download (下载 {len(image_ids)} 张图片) ...")
                response = requests.post(
                    f'{base_url}/storage/images/download',
                    headers=headers,
                    json={'image_ids': image_ids},
                    timeout=10,
                )

                if response.status_code == 200:
                    logger.info(f"✓ /storage/images/download 响应成功")
                    logger.info(f"  - ZIP 文件大小：{len(response.content)} 字节")
                else:
                    logger.error(f"✗ /storage/images/download 失败：{response.status_code}")
                    return False
            else:
                logger.info("  - 没有图片可供下载测试，跳过")

        logger.info("✓ 所有 API 端点测试通过")
        return True

    except ImportError:
        logger.warning("  - requests 库未安装，跳过 API 测试")
        return True
    except requests.exceptions.ConnectionError:
        logger.warning("  - 无法连接到服务，跳过 API 测试")
        return True
    except Exception as e:
        logger.error(f"API 测试失败：{e}")
        return False


def cleanup_test_storage():
    """清理测试存储"""
    logger.info("=" * 60)
    logger.info("清理测试存储")
    logger.info("=" * 60)

    from util.storage import storage_manager

    # 关闭存储管理器
    try:
        storage_manager.backend = None
        storage_manager._initialized = False
    except Exception:
        pass

    # 删除测试目录
    import shutil
    test_dir = "storage/test_images"
    if os.path.exists(test_dir):
        try:
            shutil.rmtree(test_dir)
            logger.info(f"✓ 测试存储目录已删除：{test_dir}")
        except Exception as e:
            logger.warning(f"无法删除测试存储目录：{e}")


def main():
    """运行所有测试"""
    logger.info("=" * 60)
    logger.info("图片存储功能测试")
    logger.info("=" * 60)
    logger.info("")

    test_results = {
        '存储初始化': False,
        '上传图片': False,
        '查询图片': False,
        '存储统计': False,
        '删除图片': False,
        '清理过期图片': False,
        '获取所有 ID': False,
        '图片压缩': False,
        '存储配额': False,
        'API 端点': False,
    }

    try:
        # 测试 1: 存储初始化
        test_results['存储初始化'] = test_storage_initialization()

        # 测试 2: 上传图片
        image_id = test_upload_image()
        test_results['上传图片'] = True

        # 测试 3: 查询图片
        test_results['查询图片'] = test_query_images(image_id)

        # 测试 4: 获取存储统计
        test_results['存储统计'] = test_get_stats()

        # 测试 5: 删除图片
        test_results['删除图片'] = test_delete_image(image_id)

        # 测试 6: 清理过期图片
        test_results['清理过期图片'] = test_cleanup_images()

        # 测试 7: 获取所有图片 ID
        test_results['获取所有 ID'] = test_get_all_image_ids()

        # 测试 8: 图片压缩功能
        test_results['图片压缩'] = test_image_compression()

        # 测试 9: 存储配额
        test_results['存储配额'] = test_storage_quota()

        # 测试 10: API 端点（需要服务运行）
        test_results['API 端点'] = test_api_endpoints()

        # 打印测试结果摘要
        logger.info("")
        logger.info("=" * 60)
        logger.info("测试结果摘要")
        logger.info("=" * 60)

        passed = sum(1 for v in test_results.values() if v)
        total = len(test_results)

        for test_name, result in test_results.items():
            status = "✓ 通过" if result else "✗ 失败"
            logger.info(f"  {test_name}: {status}")

        logger.info("")
        logger.info(f"总计：{passed}/{total} 测试通过")

        if passed == total:
            logger.info("所有测试通过 ✓")
        else:
            logger.warning(f"有 {total - passed} 个测试失败")

        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"测试失败：{e}", exc_info=True)
        return 1

    finally:
        # 清理测试存储
        cleanup_test_storage()

    return 0 if all(test_results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
