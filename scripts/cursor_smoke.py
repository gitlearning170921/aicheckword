"""
Cursor Cloud Agents 冒烟（aicheckword 侧栏「Cursor Agent (Cloud API)」）。

用法（在 aicheckword 根目录，会从 settings / 库已保存配置读取 cursor_*）：
  python scripts/cursor_smoke.py

可选环境变量覆盖：
  CURSOR_API_KEY
  CURSOR_REPOSITORY   如 your-org/your-repo 或 https://github.com/org/repo
  CURSOR_REF          默认 main
  CURSOR_API_BASE     默认 https://api.cursor.com
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main() -> int:
    from config import settings
    from src.core.cursor_agent import complete_task

    ak = (os.environ.get("CURSOR_API_KEY") or settings.cursor_api_key or "").strip()
    repo = (os.environ.get("CURSOR_REPOSITORY") or settings.cursor_repository or "").strip()
    if ak:
        settings.cursor_api_key = ak
    if repo:
        settings.cursor_repository = repo
    ref = (os.environ.get("CURSOR_REF") or settings.cursor_ref or "main").strip()
    settings.cursor_ref = ref
    base = (os.environ.get("CURSOR_API_BASE") or settings.cursor_api_base or "").strip()
    if base:
        settings.cursor_api_base = base.rstrip("/")

    if not ak or not repo:
        print("SKIP: 需要 cursor_api_key 与 cursor_repository（侧栏或环境变量）")
        return 0

    print(
        f"base={settings.cursor_api_base or 'https://api.cursor.com'} "
        f"repo={settings.cursor_repository} ref={settings.cursor_ref}"
    )
    text = complete_task("只回复两个字母：OK", poll_interval=2.0, timeout=300)
    print("reply:", (text or "")[:300])
    print("OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"FAIL: {e}", file=sys.stderr)
        raise SystemExit(1)
