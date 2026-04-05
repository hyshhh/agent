"""日志工具 — 基于 loguru"""

import sys
from loguru import logger


def setup_logger(level: str = "INFO", log_file: str | None = None):
    """配置全局日志器"""
    logger.remove()  # 移除默认处理器

    # 控制台输出（带颜色）
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{module}</cyan>:<cyan>{function}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # 文件输出（可选）
    if log_file:
        logger.add(
            log_file,
            level="DEBUG",
            rotation="10 MB",
            retention="7 days",
            encoding="utf-8",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {module}:{function}:{line} | {message}",
        )

    return logger


def get_logger():
    """获取全局 logger 实例"""
    return logger
