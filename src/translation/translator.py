"""
调用 LLM 逐句翻译：支持中文↔英文、中文↔德文；可选公司信息固定译法。
FDA 医疗器械语境：正式、技术含义一致、不意译；跨批次术语表保证全文一致。
"""
from typing import List, Optional, Dict, Any
import sys
import re
import unicodedata
from pathlib import Path

# 项目根加入 path，便于 import config / src.core
_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import settings
from src.core.llm_factory import invoke_chat_direct

# 每批最多送多少句（避免单次请求过长）
BATCH_SIZE = 10

# 句内换行占位，避免批次里「一句多行」被模型当成多句导致漏译/错位
_LINE_BREAK_PH = "⟦BR⟧"
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_LETTER_RE = re.compile(r"[a-zA-Z\u00c0-\u024f]")


def _escape_br(s: str) -> str:
    return (s or "").replace("\n", _LINE_BREAK_PH)


def _unescape_br(s: str) -> str:
    return (s or "").replace(_LINE_BREAK_PH, "\n")

# 目标语言：en 英文, de 德文, zh 中文
TARGET_LANG_EN = "en"
TARGET_LANG_DE = "de"
TARGET_LANG_ZH = "zh"

FDA_SYSTEM_HINT_EN = """You are a professional translator for FDA medical device regulatory documents.
Rules: Translate Chinese to English. Be formal, accurate, and consistent with technical/regulatory terminology. Do not paraphrase or add explanations. Do not merge or split sentences. Preserve line breaks (\\n), list numbering (e.g. 1. 2. 3. or 一、二、), and paragraph structure in your output; output exactly one translation per input line when given multiple lines. If a line is already in English, keep it unchanged.
Punctuation: Output must use Western/English punctuation only—comma, period, semicolon, colon, parentheses, and straight quotes. Do not leave Chinese punctuation (e.g. 、，。；：？！「」【】（）《》) in the English text; use comma for Chinese 、 or ， between list items, period for 。, etc.
Terminology: If the same Chinese term or clearly equivalent risk/probability level wording appears in different places (narrative text, table headers, table cells), use the SAME English word everywhere—do not alternate synonyms (e.g. do not mix "Occasional" and "Sometimes" for the same level). Match introductory lists and table rows. Output complete words—never truncate (e.g. "Occasional" not "Occasiona").
Risk occurrence probability (ISO 14971 / regulatory style): use ONE fixed English label per level everywhere—prose and tables must match. Typical mapping: 频繁/经常 → Frequent; 有时/偶发/偶然 (mid-range levels) → Occasional — use "Occasional", NOT "Sometimes", for table rows and headings; 少见/不常见 → Infrequent if distinct, else align with your scale; 罕见 → Rare; 极少/非常少 → Very rare. If introductory text lists "Occasional", table cells for the same level must also say "Occasional"."""

FDA_SYSTEM_HINT_DE = """You are a professional translator for FDA medical device regulatory documents.
Rules: Translate Chinese to German. Be formal, accurate, and consistent with technical/regulatory terminology. Do not paraphrase or add explanations. Do not merge or split sentences. Preserve line breaks (\\n), list numbering (e.g. 1. 2. 3. or 一、二、), and paragraph structure in your output; output exactly one translation per input line when given multiple lines. If a line is already in German, keep it unchanged.
Punctuation: Use Western punctuation (comma, period, semicolon, colon, ASCII parentheses and quotes). Replace any Chinese punctuation (、，。；：？！「」【】（）《》) with German-standard Western equivalents; use comma for enumeration (Chinese 、 or ，).
Terminology: Use identical German terms everywhere for the same source concept across narrative and tables; no synonym mixing. Output complete words, never truncated.
Risk probability levels: same Chinese level (有时/偶发/偶然 etc.) must map to ONE German term in both prose and tables (e.g. prefer "Gelegentlich" consistently, not alternating near-synonyms)."""

FDA_SYSTEM_HINT_ZH = """You are a professional translator for FDA medical device regulatory documents.
Rules: Translate English or German to Chinese. Be formal, accurate, and consistent with technical/regulatory terminology. Do not paraphrase or add explanations. Do not merge or split sentences. Preserve line breaks (\\n), list numbering (e.g. 1. 2. 3.), and paragraph structure in your output; output exactly one translation per input line when given multiple lines. If a line is already in Chinese, keep it unchanged.
Terminology: Same English/German term must map to the same Chinese everywhere in narrative and tables; no synonym drift. Output complete characters, no truncation."""


def _system_hint_for_target(target_lang: str) -> str:
    if target_lang == TARGET_LANG_DE:
        return FDA_SYSTEM_HINT_DE
    if target_lang == TARGET_LANG_ZH:
        return FDA_SYSTEM_HINT_ZH
    return FDA_SYSTEM_HINT_EN


def _direction_hint(target_lang: str) -> str:
    if target_lang == TARGET_LANG_DE:
        return "Translate the following Chinese sentences to German."
    if target_lang == TARGET_LANG_ZH:
        return "Translate the following English or German sentences to Chinese."
    return "Translate the following Chinese sentences to English."


def _normalize_cache_key(s: str) -> str:
    """统一空白与 Unicode 形式，便于相同原文命中同一缓存。"""
    if not s:
        return ""
    t = unicodedata.normalize("NFC", s.strip())
    return " ".join(t.split())


# 风险发生概率等级：不同中文表述在表格与正文中应译成同一英文（避免 Occasional / Sometimes 混用）
_RISK_ZH_ALIAS_GROUPS = [
    frozenset({"偶发", "偶然", "有时", "偶尔"}),
    frozenset({"频繁", "经常"}),
    frozenset({"少见", "不常见"}),
    frozenset({"罕见"}),
    frozenset({"极少", "非常少"}),
]


def _risk_alias_lookup(running_glossary: Dict[str, str], s: str) -> Optional[str]:
    """短词级：同组中文共用已出现的译文。"""
    nk = _normalize_cache_key(s)
    if not nk or len(nk) > 8:
        return None
    for g in _RISK_ZH_ALIAS_GROUPS:
        if nk not in g:
            continue
        for key in g:
            hit = running_glossary.get(key)
            if hit:
                return hit
    return None


def _risk_alias_propagate(running_glossary: Dict[str, str], src: str, translation: str) -> None:
    nk = _normalize_cache_key(src)
    if not nk or len(nk) > 8:
        return
    for g in _RISK_ZH_ALIAS_GROUPS:
        if nk not in g:
            continue
        for alias in g:
            running_glossary.setdefault(alias, translation)
        break


def _fix_common_en_truncations(text: str) -> str:
    """修复模型偶发截断的常见监管用语。"""
    if not text:
        return text
    t = text
    t = re.sub(r"\bOccasiona\b(?![a-z])", "Occasional", t, flags=re.IGNORECASE)
    t = re.sub(r"\bInfrequen\b(?![a-z])", "Infrequent", t, flags=re.IGNORECASE)
    return t


# 译成英文/德文后：残留中文标点统一为西式（顿号、逗号→ comma，句号→ period）
_ZH_PUNCT_TO_WESTERN = str.maketrans({
    "\u3001": ",",  # 、 enumeration comma → comma
    "\uff0c": ",",  # ， fullwidth comma
    "\u3002": ".",  # 。 ideographic full stop
    "\uff1b": ";",  # ；
    "\uff1a": ":",  # ：
    "\uff1f": "?",  # ？
    "\uff01": "!",  # ！
    "\u201c": '"',  # “
    "\u201d": '"',  # ”
    "\u2018": "'",  # ‘
    "\u2019": "'",  # ’
    "\uff08": "(",  # （
    "\uff09": ")",  # ）
    "\u3010": "[",  # 【
    "\u3011": "]",  # 】
    "\u300c": '"',  # 「
    "\u300d": '"',  # 」
    "\u300e": "'",  # 『
    "\u300f": "'",  # 』
    "\u300a": '"',  # 《
    "\u300b": '"',  # 》
    "\u3008": "<",  # 〈
    "\u3009": ">",  # 〉
    "\uff0e": ".",  # ． fullwidth full stop
    "\u22ef": "...",  # ⋯
})


def _zh_punctuation_to_western(text: str) -> str:
    """将译文中的中文/全角标点转为英文常用标点（列举用逗号，句末用句号）。"""
    if not text:
        return text
    t = text.replace("……", "...").replace("。。", ".")
    t = t.translate(_ZH_PUNCT_TO_WESTERN)
    # 常见半角中文标点（若模型直接输出）
    t = t.replace("、", ",").replace("，", ",").replace("。", ".").replace("；", ";")
    t = t.replace("：", ":").replace("？", "?").replace("！", "!")
    t = t.replace("（", "(").replace("）", ")").replace("【", "[").replace("】", "]")
    t = t.replace("「", '"').replace("」", '"').replace("『", "'").replace("』", "'")
    t = t.replace("《", '"').replace("》", '"')
    # 轻度整理连续逗号
    t = re.sub(r",\s*,+", ", ", t)
    return t


def _cache_lookup(cache: Dict[str, str], s: str) -> Optional[str]:
    if not s:
        return None
    if s in cache:
        return cache[s]
    nk = _normalize_cache_key(s)
    if nk in cache:
        return cache[nk]
    return None


def _cache_store(cache: Dict[str, str], s: str, translation: str) -> None:
    """同时存原文键与规范化键，提高命中率。"""
    cache[s] = translation
    nk = _normalize_cache_key(s)
    if nk and nk != s:
        cache[nk] = translation


def _format_running_glossary(
    gloss: Dict[str, str],
    target_lang: str,
    max_chars: int = 6500,
) -> str:
    """将本任务已译片段拼成提示块；长键优先，便于子串级一致。"""
    if not gloss:
        return ""
    items = sorted(gloss.items(), key=lambda x: len(x[0]), reverse=True)
    lines: List[str] = []
    n = 0
    for k, v in items:
        if not k or not v:
            continue
        line = f"{k} → {v}"
        if n + len(line) + 1 > max_chars:
            break
        lines.append(line)
        n += len(line) + 1
    if not lines:
        return ""
    if target_lang == TARGET_LANG_ZH:
        hdr = (
            "【本任务已确定的译法，后文必须严格沿用，禁止同义替换；"
            "正文与表格中同一概念须用同一译词。】"
        )
    elif target_lang == TARGET_LANG_DE:
        hdr = (
            "Bereits in diesem Auftrag festgelegte Übersetzungen – exakt wiederverwenden, "
            "keine Synonyme; Tabellen und Fließtext müssen übereinstimmen."
        )
    else:
        hdr = (
            "Running glossary for THIS job — reuse EXACTLY the same English for each source line below; "
            "do NOT use synonyms. Narrative text and table cells must match for the same concept."
        )
    return hdr + "\n" + "\n".join(lines)


def _company_overrides_prompt(company_overrides: Optional[Dict[str, str]]) -> str:
    """将公司信息配置拼成提示语，供 LLM 在翻译时优先采用。"""
    if not company_overrides or not isinstance(company_overrides, dict):
        return ""
    parts = []
    labels = {
        "company_name": "Company name / 公司名称",
        "address": "Address / 地址",
        "contact": "Contact person / 联系人",
        "phone": "Phone / 电话",
        "fax": "Fax / 传真",
        "email": "Email / 邮箱",
    }
    for key, label in labels.items():
        val = (company_overrides.get(key) or "").strip()
        if val:
            parts.append(f"{label}: {val}")
    if not parts:
        return ""
    return "When translating company-related information, use exactly these terms:\n" + "\n".join(parts)


def _get_kb_context(
    collection_name: str,
    query_prefix: str,
    top_k_glossary: int = 8,
    top_k_regulation: int = 5,
    top_k_case: int = 3,
) -> str:
    """从知识库检索词条、法规、案例，拼成参考上下文。"""
    try:
        from src.core.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(collection_name=collection_name)
    except Exception:
        return ""
    parts = []
    query = (query_prefix or "")[:2000]
    if not query:
        return ""
    try:
        glossary_docs = kb.search_by_category(query, "glossary", top_k=top_k_glossary)
        if glossary_docs:
            parts.append("【Preferred terminology (use when applicable)】")
            for d in glossary_docs:
                parts.append((d.page_content or "").strip())
        reg_docs = kb.search_by_category(query, "regulation", top_k=top_k_regulation)
        if reg_docs:
            parts.append("【Regulatory reference style】")
            for d in reg_docs[:3]:
                parts.append((d.page_content or "").strip()[:500])
        case_docs = kb.search(query, top_k=top_k_case)
        if case_docs:
            parts.append("【Reference phrasing】")
            for d in case_docs[:2]:
                parts.append((d.page_content or "").strip()[:400])
    except Exception:
        pass
    if not parts:
        return ""
    return "\n\n".join(parts)


def _parse_batch_model_output(out: str, n: int) -> List[str]:
    """
    解析模型输出：优先「序号|译文」每行一条（句内换行已用 ⟦BR⟧ 转义）；
    否则按非空行顺序回填（兼容旧行为）。
    """
    result = [""] * n
    pipe_hit: Dict[int, str] = {}
    for ln in out.strip().splitlines():
        ln = ln.strip()
        if not ln or "|" not in ln:
            continue
        num_str, rest = ln.split("|", 1)
        try:
            idx = int(num_str.strip()) - 1
            if 0 <= idx < n:
                pipe_hit[idx] = _unescape_br(rest.strip())
        except ValueError:
            continue
    for i in range(n):
        if i in pipe_hit:
            result[i] = pipe_hit[i]
    # 顺序回填：模型未用 pipe 格式时
    plain: List[str] = []
    for ln in out.strip().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if "|" in ln:
            head, _ = ln.split("|", 1)
            if head.strip().isdigit():
                continue
        if re.match(r"^\d+\.\s+", ln):
            ln = re.sub(r"^\d+\.\s+", "", ln, count=1)
        plain.append(_unescape_br(ln))
    pi = 0
    for i in range(n):
        if result[i].strip():
            continue
        if pi < len(plain):
            result[i] = plain[pi]
            pi += 1
    return result


def _translate_single_fallback(
    sentence: str,
    context: str,
    provider: Optional[str],
    target_lang: str,
    company_overrides: Optional[Dict[str, str]],
    consistency_block: str,
) -> str:
    """单句补译：批次漏译或仍含中文时使用。"""
    esc = _escape_br(sentence)
    user_content = ""
    if (context or "").strip():
        user_content += "Reference (use for terminology):\n" + context.strip()[:2000] + "\n\n"
    if (consistency_block or "").strip():
        user_content += consistency_block.strip()[:4000] + "\n\n"
    ch = _company_overrides_prompt(company_overrides)
    if ch:
        user_content += ch + "\n\n"
    user_content += (
        _direction_hint(target_lang)
        + " Translate the text after '1|' completely. Output exactly one line starting with '1|' then the translation. "
        + "Preserve ⟦BR⟧ as line breaks in the translation if present in the source.\n\n1|"
        + esc
    )
    full_prompt = _system_hint_for_target(target_lang) + "\n\n" + user_content
    try:
        out = invoke_chat_direct(full_prompt, temperature=0.0, provider=provider)
    except Exception:
        return sentence
    for ln in out.strip().splitlines():
        ln = ln.strip()
        if ln.startswith("1|"):
            return _unescape_br(ln[2:].lstrip())
    # 无 1| 时取首段有意义行
    for ln in out.strip().splitlines():
        ln = ln.strip()
        if ln and "|" in ln:
            return _unescape_br(ln.split("|", 1)[-1].strip())
        if ln:
            return _unescape_br(ln)
    return sentence


def _translate_batch(
    sentences: List[str],
    context: str,
    provider: Optional[str] = None,
    target_lang: str = TARGET_LANG_EN,
    company_overrides: Optional[Dict[str, str]] = None,
    consistency_block: str = "",
) -> List[str]:
    """将一批句子翻译成目标语言，返回等长列表。句内换行用 ⟦BR⟧ 保护，避免漏译。"""
    if not sentences:
        return []
    n = len(sentences)
    user_content = ""
    if context.strip():
        user_content += "Reference (use for terminology and style):\n" + context.strip() + "\n\n"
    company_hint = _company_overrides_prompt(company_overrides)
    if company_hint:
        user_content += company_hint + "\n\n"
    if (consistency_block or "").strip():
        user_content += (consistency_block.strip() + "\n\n")
    user_content += (
        _direction_hint(target_lang)
        + "\nEach INPUT line has the form \"N|text\" where N is the item number (1.."
        + str(n)
        + "). The token ⟦BR⟧ inside text means a line break inside that item; keep ⟦BR⟧ in your translation where needed.\n"
        + "OUTPUT: exactly "
        + str(n)
        + " lines. Line i must be \"i|\" followed by the translation of input item i only. "
        + "Do not merge items. Do not output extra lines.\n\nINPUT:\n"
    )
    for i, s in enumerate(sentences):
        user_content += f"{i + 1}|{_escape_br(s)}\n"
    full_prompt = _system_hint_for_target(target_lang) + "\n\n" + user_content
    try:
        out = invoke_chat_direct(full_prompt, temperature=0.0, provider=provider)
    except Exception as e:
        raise RuntimeError(f"翻译接口调用失败: {e}") from e
    result = _parse_batch_model_output(out, n)
    if target_lang == TARGET_LANG_EN:
        result = [_fix_common_en_truncations(x) for x in result]
    # 不再用中文原文填充缺行，避免「译成英文后仍留中文」
    return result


def _repair_untranslated_lines(
    sentences: List[str],
    results: List[str],
    target_lang: str,
    context: str,
    provider: Optional[str],
    company_overrides: Optional[Dict[str, str]],
    consistency_block: str,
    cache: Dict[str, str],
    running_glossary: Dict[str, str],
) -> None:
    """目标为英/德时输出仍含中文、或目标为中文时仍含西文：单句补译并写回 results 原地。"""
    t = target_lang or TARGET_LANG_EN
    for i in range(len(results)):
        src = sentences[i] if i < len(sentences) else ""
        out = results[i] or ""
        if not (src or "").strip():
            continue
        need = False
        if t in (TARGET_LANG_EN, TARGET_LANG_DE):
            if _CJK_RE.search(src):
                if not out.strip() or _CJK_RE.search(out):
                    need = True
        elif t == TARGET_LANG_ZH:
            # 外→中：仍含较长西文片段时补译（避免产品型号等短拉丁误触发）
            if _LATIN_LETTER_RE.search(src):
                if not out.strip():
                    need = True
                elif _LATIN_LETTER_RE.search(out) and len(re.findall(r"[a-zA-Z]{3,}", out)) >= 3:
                    need = True
        if not need:
            continue
        fixed = _translate_single_fallback(
            src, context, provider, t, company_overrides, consistency_block
        )
        if t == TARGET_LANG_EN:
            fixed = _fix_common_en_truncations(fixed)
        if t in (TARGET_LANG_EN, TARGET_LANG_DE):
            fixed = _zh_punctuation_to_western(fixed)
        results[i] = fixed
        _cache_store(cache, src, fixed)
        nk = _normalize_cache_key(src)
        if nk:
            running_glossary[nk] = fixed
        _risk_alias_propagate(running_glossary, src, fixed)


def translate_sentences(
    sentences: List[str],
    collection_name: Optional[str] = None,
    use_kb: bool = True,
    cache: Optional[Dict[str, str]] = None,
    provider: Optional[str] = None,
    target_lang: str = TARGET_LANG_EN,
    company_overrides: Optional[Dict[str, str]] = None,
    kb_query_extra: Optional[str] = None,
    running_glossary: Optional[Dict[str, str]] = None,
) -> List[str]:
    """
    将句子列表翻译为目标语言。target_lang: en 英文, de 德文, zh 中文。
    可选知识库做术语/风格参考；可选 company_overrides 固定公司名称/地址/联系人/电话等译法。
    kb_query_extra：追加到知识库检索 query（如项目名称、产品名），与文档审核项目上下文一致。
    cache：相同原文在同一任务内复用译法（含规范化空白后的键）。
    running_glossary：跨批次累积「原文→译文」，注入后续批次提示，减少同词异译（如 Occasional/Sometimes）。
    返回与 sentences 等长的列表。
    """
    if not sentences:
        return []
    cache = cache if cache is not None else {}
    if running_glossary is None:
        running_glossary = {}
    tlang = target_lang or TARGET_LANG_EN
    context = ""
    if use_kb and collection_name:
        prefix = " ".join(sentences[:3])
        extra = (kb_query_extra or "").strip()
        if extra:
            prefix = (extra + " " + prefix).strip()
        context = _get_kb_context(collection_name, prefix)
    results: List[str] = []
    i = 0
    while i < len(sentences):
        batch = sentences[i : i + BATCH_SIZE]
        batch_results: List[str] = []
        to_call: List[str] = []
        need_indices: List[int] = []
        for s in batch:
            hit = _cache_lookup(cache, s)
            if hit is None:
                hit = _risk_alias_lookup(running_glossary, s)
            if hit is not None:
                if tlang == TARGET_LANG_EN:
                    hit = _fix_common_en_truncations(hit)
                if tlang in (TARGET_LANG_EN, TARGET_LANG_DE):
                    hit = _zh_punctuation_to_western(hit)
                batch_results.append(hit)
            else:
                need_indices.append(len(batch_results))
                batch_results.append("")
                to_call.append(s)
        if to_call:
            consistency_block = _format_running_glossary(running_glossary, tlang)
            # 同一批次内完全相同原文只请求一次模型，避免同批异译
            uniq_order = list(dict.fromkeys(to_call))
            trans = _translate_batch(
                uniq_order,
                context,
                provider=provider,
                target_lang=tlang,
                company_overrides=company_overrides,
                consistency_block=consistency_block,
            )
            trans_map: Dict[str, str] = {}
            for ui, src_u in enumerate(uniq_order):
                en_u = trans[ui] if ui < len(trans) else src_u
                if tlang == TARGET_LANG_EN:
                    en_u = _fix_common_en_truncations(en_u)
                if tlang in (TARGET_LANG_EN, TARGET_LANG_DE):
                    en_u = _zh_punctuation_to_western(en_u)
                trans_map[src_u] = en_u
            for k, src in enumerate(to_call):
                en = trans_map.get(src, src)
                _cache_store(cache, src, en)
                nk = _normalize_cache_key(src)
                if nk:
                    running_glossary[nk] = en
                _risk_alias_propagate(running_glossary, src, en)
                if k < len(need_indices):
                    batch_results[need_indices[k]] = en
        results.extend(batch_results)
        i += len(batch)
    _repair_untranslated_lines(
        sentences,
        results,
        tlang,
        context,
        provider,
        company_overrides,
        _format_running_glossary(running_glossary, tlang),
        cache,
        running_glossary,
    )
    if tlang in (TARGET_LANG_EN, TARGET_LANG_DE):
        results = [_zh_punctuation_to_western(r) for r in results]
    if tlang == TARGET_LANG_EN:
        results = [_fix_common_en_truncations(r) for r in results]
    return results
