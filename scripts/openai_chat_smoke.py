"""OpenAI 兼容对话冒烟（不走向量化）。用法：set OPENAI_API_KEY=... && python scripts/openai_chat_smoke.py"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main() -> int:
    from config import settings
    from src.core.llm_factory import invoke_chat_direct, resolve_model_for_provider

    key = (os.environ.get("OPENAI_API_KEY") or settings.openai_api_key or "").strip()
    if not key:
        print("SKIP: 未配置 OPENAI_API_KEY")
        return 0
    settings.openai_api_key = key
    settings.provider = "openai"
    base = (os.environ.get("OPENAI_BASE_URL") or settings.openai_base_url or "").strip()
    if base:
        settings.openai_base_url = base.rstrip("/")
    model = (os.environ.get("OPENAI_MODEL") or resolve_model_for_provider("openai") or "gpt-4o-mini").strip()
    settings.llm_model = model

    print(f"provider=openai model={settings.llm_model} base={settings.openai_base_url}")
    text = invoke_chat_direct("回复 OK 两个字母即可。", provider="openai", temperature=0)
    print("reply:", (text or "")[:200])
    print("OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"FAIL: {e}", file=sys.stderr)
        raise SystemExit(1)
