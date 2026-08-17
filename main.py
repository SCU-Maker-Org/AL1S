#!/usr/bin/env python3
"""
AL1S-Bot 主程序入口
"""
import atexit
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
# 禁用 sentence-transformers 的多进程以避免 semaphore 泄漏
os.environ.setdefault("SENTENCE_TRANSFORMERS_DISABLE_MULTIPROCESSING", "1")
# 设置 OMP 线程数为1以减少多进程问题
os.environ.setdefault("OMP_NUM_THREADS", "1")
# 禁用 HuggingFace 下载进度条（必须在程序启动时设置）
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
# 设置下载超时和重试（全局设置）
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
os.environ.setdefault("HF_HUB_DOWNLOAD_RETRIES", "5")
import signal
import sys

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from loguru import logger

from src.bot import AL1SBot
from src.config import config


def _cleanup_process_pools():
    """清理可能存在的 joblib/loky 进程池和其他资源，避免退出时的 semaphore 警告。"""
    try:
        # 清理 joblib/loky 执行器
        from joblib.externals.loky import get_reusable_executor

        executor = get_reusable_executor()
        if executor is not None:
            executor.shutdown(wait=True, kill_workers=True)
            logger.debug("已清理 loky 执行器")
    except Exception as e:
        logger.debug(f"清理 loky 执行器时出现异常: {e}")

    try:
        # 清理 sentence-transformers 相关的多进程资源
        import sys

        if "sentence_transformers" in sys.modules:
            try:
                # 尝试清理 sentence-transformers 的内部进程池
                import sentence_transformers

                if hasattr(sentence_transformers, "_pool"):
                    pool = getattr(sentence_transformers, "_pool")
                    if pool is not None:
                        pool.close()
                        pool.join()
                        logger.debug("已清理 sentence-transformers 进程池")
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"清理 sentence-transformers 资源时出现异常: {e}")

    try:
        # 清理其他资源
        import gc
        import multiprocessing
        import os
        import threading

        # 强制垃圾回收
        gc.collect()

        # 清理多进程相关资源
        try:
            multiprocessing.util._cleanup_tests()
        except Exception:
            pass

        # 清理可能的 semaphore 资源
        try:
            import glob
            import tempfile

            temp_dir = tempfile.gettempdir()
            # 查找并清理可能的 loky semaphore 文件
            loky_files = glob.glob(os.path.join(temp_dir, "loky-*"))
            for file in loky_files:
                try:
                    if os.path.exists(file):
                        os.unlink(file)
                except Exception:
                    pass
        except Exception:
            pass

        logger.debug("资源清理完成")
    except Exception as e:
        logger.debug(f"资源清理时出现异常: {e}")


atexit.register(_cleanup_process_pools)


def signal_handler(signum, frame):
    """信号处理器"""
    logger.info(f"收到信号 {signum}，正在关闭机器人...")
    try:
        _cleanup_process_pools()
    except Exception:
        pass
    sys.exit(0)


def main():
    """主函数"""
    try:
        # 配置日志（根据配置中的 app.debug 决定级别）
        debug_mode = bool(
            getattr(getattr(config, "app", None), "debug", False)
            or getattr(config, "debug", False)
        )

        # 移除默认控制台 sink（默认是 DEBUG）
        logger.remove()

        # 控制台输出
        logger.add(
            sys.stderr,
            level="DEBUG" if debug_mode else "INFO",
            enqueue=True,
        )

        # 文件输出
        logger.add(
            "logs/bot.log",
            rotation="1 day",
            retention="7 days",
            level="DEBUG" if debug_mode else "INFO",
            enqueue=True,
        )

        logger.info("正在启动 AL1S-Bot...")

        # 设置信号处理器
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # 创建机器人实例
        bot = AL1SBot()

        # 启动机器人
        bot.start()

    except KeyboardInterrupt:
        logger.info("收到键盘中断信号")
    except Exception as e:
        logger.error(f"机器人运行失败: {e}")
        raise


if __name__ == "__main__":
    try:
        # 创建logs目录
        Path("logs").mkdir(exist_ok=True)

        # 运行主程序
        main()
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error(f"程序启动失败: {e}")
        sys.exit(1)
