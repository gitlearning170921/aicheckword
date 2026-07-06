"""题干中误写入的开卷链接 HTML 清理（button / anchor / 半截属性碎片）。"""

from __future__ import annotations

import html as html_module
import re
from typing import Any


def strip_broken_open_book_html(stem: str) -> str:
    """将题干恢复为纯文本 + 《文件名》引用，供前端重新 linkify。"""
    s = html_module.unescape(str(stem or ""))

    def _inner_text(raw: str) -> str:
        return re.sub(r"</?[^>]+>", "", str(raw or "")).strip()

    def _anchor_repl(m: re.Match[str]) -> str:
        href = str(m.group(1) or "").strip()
        inner = _inner_text(m.group(2))
        if inner:
            return inner
        if href and not href.startswith(("http://", "https://", "/", "#", "javascript:")):
            name = href.strip("《》")
            return f"《{name}》" if name else ""
        return inner

    # 1) 完整 <a>（href 为文件名或 title 为开卷查阅）
    s = re.sub(
        r'<a\b[^>]*href\s*=\s*["\']([^"\']*)["\'][^>]*>(.*?)</a>',
        _anchor_repl,
        s,
        flags=re.I | re.S,
    )
    s = re.sub(
        r'<a\b[^>]*title\s*=\s*["\']开卷查阅[:：]点击展开全文["\'][^>]*>(.*?)</a>',
        lambda m: _inner_text(m.group(1)),
        s,
        flags=re.I | re.S,
    )
    s = re.sub(r"<a\b[^>]*>", "", s, flags=re.I)
    s = re.sub(r"</a\s*>", "", s, flags=re.I)
    s = re.sub(r'\s*href\s*=\s*["\'][^"\']*["\']', "", s, flags=re.I)

    # 2) 完整 <button class=exam-open-book-link>
    s = re.sub(
        r"<button[^>]*exam-open-book-link[^>]*>(.*?)</button>",
        lambda m: _inner_text(m.group(1)).strip("《》"),
        s,
        flags=re.I | re.S,
    )
    s = re.sub(r"<button\b[^>]*>", "", s, flags=re.I)
    s = re.sub(r"</button\s*>", "", s, flags=re.I)

    # 3) 游离属性碎片
    s = re.sub(r'\s*data-open-book-file\s*=\s*["\'][^"\']*["\']', "", s, flags=re.I)
    s = re.sub(r'\s*class\s*=\s*["\'][^"]*exam-open-book-link[^"\']*["\']', "", s, flags=re.I)
    s = re.sub(r'\s*type\s*=\s*["\']button["\']', "", s, flags=re.I)
    s = re.sub(r'"?\s*title\s*=\s*["\']开卷查阅[:：]点击展开全文["\']\s*[>》]?', "", s, flags=re.I)

    # 4) 折叠「裸名《同名》」重复
    s = re.sub(r"(审核点清单[-:][\w.\-]+)\s*《\1》", r"《\1》", s)
    s = re.sub(r"《(审核点清单[-:][\w.\-]+)》\s*\1(?![\w.\-])", r"《\1》", s)
    return s.strip()


def strip_open_book_html_in_tree(obj: Any) -> int:
    """递归清理 JSON 树中的 stem/title 字段；返回修改的字段数。"""
    changed = 0
    if isinstance(obj, dict):
        for key, val in list(obj.items()):
            if key in ("stem", "title", "stem_snapshot") and isinstance(val, str):
                cleaned = strip_broken_open_book_html(val)
                if cleaned != val:
                    obj[key] = cleaned
                    changed += 1
            elif isinstance(val, (dict, list)):
                changed += strip_open_book_html_in_tree(val)
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                changed += strip_open_book_html_in_tree(item)
    return changed
