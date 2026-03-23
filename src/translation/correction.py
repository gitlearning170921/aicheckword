"""
翻译校正：针对已翻译文档做自动修复（不依赖再次调用 LLM）。

当前规则：
- 同词异译统一（如 Occasional / Sometimes -> Occasional）
- 常见截断词修复（如 Occasiona -> Occasional）
- 数值表达模式修复（如 10-2 -> 10^-2，保留语义）
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class CorrectionStats:
    total_blocks: int = 0
    changed_blocks: int = 0
    term_unified: int = 0
    truncation_fixed: int = 0
    numeric_fixed: int = 0


_TERM_GROUPS_EN: Dict[str, List[str]] = {
    "Occasional": ["Sometimes", "Sometime", "Occasionally (level)"],
    "Infrequent": ["Uncommon"],
    "Very rare": ["Extremely rare"],
}

_TRUNC_FIX_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bOccasiona\b", re.IGNORECASE), "Occasional"),
    (re.compile(r"\bInfrequen\b", re.IGNORECASE), "Infrequent"),
]

# 10-2 -> 10^-2；仅在明显是科学计数时触发
_NUMERIC_FIX_PATTERN = re.compile(r"(?<!\^)\b10\s*-\s*(\d+)\b")


def _fix_numeric_expr(text: str) -> Tuple[str, int]:
    n = 0

    def _repl(m):
        nonlocal n
        n += 1
        return f"10^-{m.group(1)}"

    return _NUMERIC_FIX_PATTERN.sub(_repl, text), n


def _unify_terms_en(text: str, doc_all_text: str) -> Tuple[str, int]:
    changed = 0
    out = text
    for canonical, variants in _TERM_GROUPS_EN.items():
        # 只有文档里出现 canonical 才做归一，避免强行覆盖
        if re.search(rf"\b{re.escape(canonical)}\b", doc_all_text, re.IGNORECASE):
            for v in variants:
                pat = re.compile(rf"\b{re.escape(v)}\b", re.IGNORECASE)
                out2, n = pat.subn(canonical, out)
                if n:
                    out = out2
                    changed += n
    return out, changed


def correct_text(text: str, target_lang: str, doc_all_text: str = "") -> Tuple[str, Dict[str, int]]:
    if not text:
        return text, {"term_unified": 0, "truncation_fixed": 0, "numeric_fixed": 0}

    out = text
    term_unified = truncation_fixed = numeric_fixed = 0
    tlang = (target_lang or "en").lower()

    if tlang in ("en", "de"):
        # 数值表达优先修复
        out, numeric_fixed = _fix_numeric_expr(out)

    if tlang == "en":
        out, term_unified = _unify_terms_en(out, doc_all_text or "")
        for pat, repl in _TRUNC_FIX_PATTERNS:
            out2, n = pat.subn(repl, out)
            if n:
                out = out2
                truncation_fixed += n

    return out, {
        "term_unified": term_unified,
        "truncation_fixed": truncation_fixed,
        "numeric_fixed": numeric_fixed,
    }

