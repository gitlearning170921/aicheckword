"""审核报告相关性能诊断：设置 audit_perf_log=true 或环境变量 AUDIT_PERF_LOG=1 后输出分阶段耗时（毫秒）。"""

import logging
import os
import time
from typing import Any, Callable, Optional, TypeVar

from config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


def audit_perf_enabled() -> bool:
    if getattr(settings, "audit_perf_log", False):
        return True
    return (os.environ.get("AUDIT_PERF_LOG") or "").strip().lower() in ("1", "true", "yes", "on")


def audit_perf_log(phase: str, elapsed_ms: float, extra: str = "") -> None:
    if not audit_perf_enabled():
        return
    msg = f"[audit_perf] {phase}: {elapsed_ms:.1f} ms"
    if extra:
        msg += f" | {extra}"
    logger.info(msg)


def audit_perf_time_block(phase: str, fn: Callable[[], T], extra: str = "") -> T:
    """执行 fn 并记录耗时（毫秒）。"""
    t0 = time.perf_counter()
    try:
        return fn()
    finally:
        if audit_perf_enabled():
            ms = (time.perf_counter() - t0) * 1000.0
            audit_perf_log(phase, ms, extra)
