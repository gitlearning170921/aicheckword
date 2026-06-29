"""
Claude 提供方冒烟：验证 create_chat_llm 与 invoke_chat_direct。

用法（在 aicheckword 根目录）：
  set ANTHROPIC_API_KEY=sk-ant-...
  python scripts/claude_smoke.py

可选环境变量：
  CLAUDE_BASE_URL   默认 https://api.anthropic.com
  CLAUDE_MODEL      默认 claude-sonnet-4-20250514
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main() -> int:
    from config import settings
    from src.core.llm_factory import create_chat_llm, invoke_chat_direct

    key = (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY") or "").strip()
    if key:
        settings.claude_api_key = key
    if not (settings.claude_api_key or "").strip():
        print("SKIP: 未配置 ANTHROPIC_API_KEY / CLAUDE_API_KEY / settings.claude_api_key")
        return 0

    base = (os.environ.get("CLAUDE_BASE_URL") or "").strip()
    if base:
        settings.claude_base_url = base.rstrip("/")

    model = (os.environ.get("CLAUDE_MODEL") or "claude-sonnet-4-20250514").strip()
    settings.provider = "claude"
    settings.llm_model = model

    prompt = "请用一句话介绍你自己（测试连通性，中文回答）。"
    print(f"[invoke_chat_direct] provider=claude model={model} base={settings.claude_base_url}")
    text1 = invoke_chat_direct(prompt, provider="claude", temperature=0.1)
    print("reply:", (text1 or "")[:500])

    print(f"[create_chat_llm] model={model}")
    llm = create_chat_llm(temperature=0.1)
    msg = llm.invoke(prompt)
    content = getattr(msg, "content", msg)
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif hasattr(block, "text"):
                parts.append(str(block.text))
        content = "\n".join(p for p in parts if p.strip())
    print("invoke:", str(content)[:500])
    print("OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"FAIL: {e}", file=sys.stderr)
        raise SystemExit(1)
