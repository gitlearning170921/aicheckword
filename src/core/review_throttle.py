"""
云端审核稳定性：全进程串行 pacing，避免 DeepSeek 等 API 连续请求叠峰导致限流、连接重置与本地 CPU/内存尖峰。

Streamlit 主线程虽多为单线程执行用户脚本，但长文档分块、嵌入检索等仍可能与其它逻辑交错；用全局锁保证「两次 LLM 调用」之间最小间隔。
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from config import settings

_lock = threading.Lock()
_last_llm_mono: float = 0.0


_DEFAULT_DEEPSEEK_PACING = 0.9


def _effective_interval_sec(provider: str) -> float:
    p = (provider or "").strip().lower()
    if p != "deepseek":
        return 0.0
    v = float(getattr(settings, "review_llm_min_interval_sec", 0.0) or 0.0)
    # 负数：关闭 pacing；0：使用内置默认
    if v < 0:
        return 0.0
    if v == 0.0:
        return _DEFAULT_DEEPSEEK_PACING
    return v


def wait_before_llm_call(provider: Optional[str] = None) -> None:
    """在发起一次 Chat 请求前调用；DeepSeek 默认约 0.9s 间隔。"""
    prov = (provider or getattr(settings, "provider", "") or "").strip().lower()
    interval = _effective_interval_sec(prov)
    if interval <= 0:
        return
    global _last_llm_mono
    with _lock:
        now = time.monotonic()
        wait = interval - (now - _last_llm_mono)
        if wait > 0:
            time.sleep(wait)
        _last_llm_mono = time.monotonic()
