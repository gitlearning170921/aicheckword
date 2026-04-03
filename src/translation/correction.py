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
from typing import Dict, List, Set, Tuple


@dataclass
class CorrectionStats:
    total_blocks: int = 0
    changed_blocks: int = 0
    term_unified: int = 0
    truncation_fixed: int = 0
    numeric_fixed: int = 0
    manual_replaced: int = 0


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


def _adapt_replacement_case(matched: str, replacement: str) -> str:
    """按匹配词的大小写形式调整替换词（句首大写、全大写、全小写等）。"""
    if not matched or not replacement:
        return replacement
    r = replacement
    if matched.isupper():
        return r.upper()
    if matched.islower():
        return r.lower()
    if len(matched) > 1 and matched[0].isupper() and matched[1:].islower():
        if len(r) > 1:
            return r[0].upper() + r[1:].lower()
        return r.upper()
    if matched[0].isupper() and r and r[0].islower():
        return r[0].upper() + r[1:]
    return r


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


def apply_manual_replacements(
    text: str,
    pairs: List[Tuple[str, str]],
    target_lang: str,
) -> Tuple[str, int]:
    """
    手工替换：将错误译文 A 替换为正确译文 B。
    英/德目标下对纯拉丁错误词使用整词边界，避免误伤；否则按字面串替换。
    """
    if not text or not pairs:
        return text, 0
    tlang = (target_lang or "en").lower()
    out = text
    total = 0
    for wrong, right in pairs:
        w, r = (wrong or "").strip(), (right or "").strip()
        if not w or w == r:
            continue
        n = 0
        if tlang in ("en", "de") and re.fullmatch(r"[a-zA-Z][a-zA-Z0-9._-]*", w):
            pat = re.compile(r"\b" + re.escape(w) + r"\b", re.IGNORECASE)

            def _repl(m: re.Match) -> str:
                return _adapt_replacement_case(m.group(0), r)

            out, n = pat.subn(_repl, out)
        else:
            pat = re.compile(re.escape(w))
            out, n = pat.subn(r, out)
        total += n
    return out, total


def _dedupe_glossary_entries(entries: List[Tuple[str, str]], tlang: str) -> List[Tuple[str, str]]:
    """同次校正输入内：相同中文 + 相同译文（英德忽略大小写）只保留一条。"""
    seen: Set[Tuple[str, str]] = set()
    out: List[Tuple[str, str]] = []
    for zh, tgt in entries:
        z, t = (zh or "").strip(), (tgt or "").strip()
        if not z or not t:
            continue
        key = (z, t.casefold()) if tlang in ("en", "de") else (z, t)
        if key in seen:
            continue
        seen.add(key)
        out.append((z, t))
    return out


def _glossary_pair_already_in_kb(
    kb,
    zh: str,
    tgt: str,
    lang_label: str,
    tlang: str,
) -> bool:
    """知识库词条分类中是否已有相同「中文词条 + 该语种译法」。"""
    try:
        hits = kb.search_by_category(zh, "glossary", top_k=24)
    except Exception:
        return False
    needle_zh = f"词条（中文）：{zh}"
    line_need = f"{lang_label}：{tgt}"
    for doc in hits:
        text = getattr(doc, "page_content", None) or ""
        if needle_zh not in text:
            continue
        if line_need in text:
            return True
        if tlang in ("en", "de") and line_need.lower() in text.lower():
            return True
    return False


def save_glossary_correction_entries(
    collection_name: str,
    entries: List[Tuple[str, str]],
    target_lang: str,
) -> int:
    """
    将「中文词条 → 目标语译法」写入当前知识库的词条（glossary）分类。
    entries: (source_zh, target_translation)
    同次输入会去重；知识库中已存在的同对词条不会重复写入。
    """
    if not collection_name or not entries:
        return 0
    from datetime import datetime

    from src.core.knowledge_base import KnowledgeBase
    from src.core.langchain_compat import Document

    tlang = (target_lang or "en").lower()
    lang_label = {"en": "英文", "de": "德文", "zh": "中文"}.get(tlang, "目标语")
    kb = KnowledgeBase(collection_name=collection_name)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved = 0
    unique = _dedupe_glossary_entries(entries, tlang)
    for zh, tgt in unique:
        if _glossary_pair_already_in_kb(kb, zh, tgt, lang_label, tlang):
            continue
        content = (
            f"词条（中文）：{zh}\n"
            f"{lang_label}：{tgt}\n"
            f"来源：文档翻译校正写入，供后续翻译检索优先采用。"
        )
        doc = Document(
            page_content=content,
            metadata={"category": "glossary"},
        )
        fn = f"translation_correction_glossary_{stamp}_{saved}.txt"
        kb.add_documents([doc], file_name=fn, category="glossary")
        saved += 1
    return saved


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

