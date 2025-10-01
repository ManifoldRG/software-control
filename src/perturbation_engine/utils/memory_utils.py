"""
Memory utilities for monitoring and cleanup
"""

import gc
import logging
from typing import Optional

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def log_memory_usage(
    context: str, logger: Optional[logging.Logger] = None, threshold_mb: int = 1000
) -> float:
    """
    Log current memory usage for monitoring

    Args:
        context: Description of when/where this is called
        logger: Logger instance (if None, creates a default logger)
        threshold_mb: Memory threshold in MB to trigger garbage collection

    Returns:
        Memory usage in MB
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if not PSUTIL_AVAILABLE:
        logger.debug("psutil not available, cannot get memory info")
        return 0.0

    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        logger.info(f"Memory usage at {context}: {memory_mb:.1f} MB")

        # Force garbage collection if memory usage is high
        if memory_mb > threshold_mb:
            logger.warning(f"High memory usage detected: {memory_mb:.1f} MB, forcing garbage collection")
            gc.collect()

        return memory_mb
    except Exception as e:
        logger.debug(f"Could not get memory info: {e}")
        return 0.0


def force_garbage_collection(logger: Optional[logging.Logger] = None) -> None:
    """
    Force garbage collection and log memory usage

    Args:
        logger: Logger instance (if None, creates a default logger)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        gc.collect()
        log_memory_usage("After garbage collection", logger)
    except Exception as e:
        logger.error(f"Error during garbage collection: {e}")
