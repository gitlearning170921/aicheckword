"""
分句与占位：按句号/分号/问号/换行等分句；根据目标语言标记「需翻译」。
混合中/英/德时按语言拆成 runs，只翻译目标语种外的片段，保留已有英/德文与□等符号。
"""
import re
from typing import List, Tuple, Optional

from .models import TextBlock, SegmentResult


# 中文→外文时：句号、分号、问号、感叹号、逗号、顿号、换行均分句，避免长段漏译
_TERMINATORS = re.compile(r"([。；？！，、\n]+)")
# 目标为中文时：中英文标点都分句，便于混合段落正确拆句
_TERMINATORS_ZH = re.compile(r"([.?!。；？！，、\n]+)")
# 含中文字符（含 CJK 标点区）
_HAS_CJK = re.compile(r"[\u4e00-\u9fff]")
# CJK 连续块（用于按语言拆分，保留括号等与中文连在一起的标点）
_CJK_RUN = re.compile(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]+")
# 含拉丁字母或常见西文（英文、德文等），用于「译成中文」时识别待译内容
_HAS_LATIN = re.compile(r"[a-zA-Z\u00c0-\u024f\u1e00-\u1eff]")
# 科学计数/范围类表达：翻译时应保持原样（避免改变含义）
_SCIENTIFIC_EXPR = re.compile(r"^\s*[<>=~\-\+\d\.\^\u2070-\u209fEe×xX/]+\s*$")


def _has_chinese(s: str) -> bool:
    return bool(s and _HAS_CJK.search(s))


def _has_latin_or_foreign(s: str) -> bool:
    """是否含拉丁/英文/德文等字母，用于目标为中文时标记需翻译。"""
    return bool(s and _HAS_LATIN.search(s))


def _split_language_runs(text: str) -> List[Tuple[str, bool]]:
    """
    将混合中/英/德的文本按「CJK 连续块」与「非 CJK」拆成 (run_text, is_cjk) 列表。
    用于只翻译中文片段、保留原有英文/德文与□等符号。
    """
    if not (text or text.strip()):
        return []
    parts = _CJK_RUN.split(text)
    separators = _CJK_RUN.findall(text)
    runs: List[Tuple[str, bool]] = []
    for i, p in enumerate(parts):
        if p:
            runs.append((p, False))  # 非 CJK（英文、德文、□、数字等）
        if i < len(separators):
            runs.append((separators[i], True))  # CJK 块
    return runs


def _need_translate(sentence: str, target_lang: str) -> bool:
    """
    根据目标语言判断该句是否需翻译。
    - target_lang 为 en/de：含中文则需翻译（中文→英文/德文）；
    - target_lang 为 zh：含拉丁/西文则需翻译（英文/德文→中文），纯中文保留。
    """
    if not (sentence or sentence.strip()):
        return False
    if _SCIENTIFIC_EXPR.match(sentence or ""):
        return False
    if (target_lang or "").strip().lower() == "zh":
        return _has_latin_or_foreign(sentence)
    return _has_chinese(sentence)


def segment_chinese_sentences(text: str, target_lang: str = "en") -> List[SegmentResult]:
    """
    将一段文本拆成句子级片段。need_translate 由 target_lang 决定（见 _need_translate）。
    分段时保留每段首尾空白与换行，便于回填时恢复序号、换行、分段。
    """
    if not (text or text.strip()):
        return [SegmentResult(text=text or "", need_translate=False)]
    terminators = _TERMINATORS_ZH if (target_lang or "").strip().lower() == "zh" else _TERMINATORS
    parts = terminators.split(text)
    results: List[SegmentResult] = []
    i = 0
    while i < len(parts):
        chunk = parts[i]
        if terminators.match(chunk):
            if results:
                results[-1].text += chunk
            else:
                results.append(SegmentResult(text=chunk, need_translate=False))
            i += 1
            continue
        sentence = chunk
        if i + 1 < len(parts) and terminators.match(parts[i + 1]):
            sentence += parts[i + 1]
            i += 1
        # 保留首尾空白与换行，便于回填时恢复格式；仅用 strip 后的内容判断是否需翻译
        content = sentence.strip()
        if content:
            results.append(SegmentResult(
                text=sentence,
                need_translate=_need_translate(content, target_lang),
            ))
        i += 1
    return results


def blocks_to_sentences(blocks: List[TextBlock], target_lang: str = "en") -> Tuple[List[List[SegmentResult]], List[str]]:
    """
    对每个 block 的 original_text 分句，返回 segment_map 与需翻译的句子列表。
    若句段同时含中文与英文/德文，则拆成 language runs，只把「需译语种」送入翻译，保留其余。
    """
    segment_map: List[List[SegmentResult]] = []
    to_translate: List[str] = []
    tlang = (target_lang or "en").strip().lower()
    for block in blocks:
        segs = segment_chinese_sentences(block.original_text, target_lang=tlang or "en")
        for seg in segs:
            if seg.need_translate and seg.text.strip() and _has_chinese(seg.text) and _has_latin_or_foreign(seg.text):
                seg.runs = _split_language_runs(seg.text)
        segment_map.append(segs)
        for seg in segs:
            if not seg.need_translate or not seg.text.strip():
                continue
            if seg.runs:
                for run_text, is_cjk in seg.runs:
                    need_run = (tlang in ("en", "de") and is_cjk) or (tlang == "zh" and not is_cjk)
                    if need_run and (run_text.strip() or run_text):
                        to_translate.append(run_text.strip() or run_text)
            else:
                to_translate.append(seg.text.strip())
    return segment_map, to_translate


def _segment_prefix_suffix(seg: SegmentResult) -> tuple:
    """取句段首尾空白（含换行），便于回填时保留序号、换行、分段。"""
    s = seg.text
    stripped = s.strip()
    if not stripped:
        return (s, "")
    start = len(s) - len(s.lstrip())
    end = len(s) - len(s.rstrip())
    return (s[:start], s[-end:] if end else "")


def _run_prefix_suffix(run_text: str) -> Tuple[str, str]:
    """取单段 run 的首尾空白，便于回填时保留□等符号。"""
    s = run_text
    stripped = s.strip()
    if not stripped:
        return (s, "")
    start = len(s) - len(s.lstrip())
    end = len(s) - len(s.rstrip())
    return (s[:start], s[-end:] if end else "")


def apply_translations_to_blocks(
    blocks: List[TextBlock],
    segment_map: List[List[SegmentResult]],
    translations: List[str],
    target_lang: str = "en",
) -> None:
    """
    将翻译结果按 to_translate 顺序填回 blocks；保留每段首尾空白与换行。
    若句段有 runs（混合语种），只替换需译的 run，保留原有英/德文与□等。
    """
    t_idx = 0
    tlang = (target_lang or "en").strip().lower()
    for bi, block in enumerate(blocks):
        segs = segment_map[bi] if bi < len(segment_map) else []
        new_parts = []
        for seg in segs:
            if seg.runs:
                for run_text, is_cjk in seg.runs:
                    need_run = (tlang in ("en", "de") and is_cjk) or (tlang == "zh" and not is_cjk)
                    if need_run and (run_text.strip() or run_text):
                        prefix, suffix = _run_prefix_suffix(run_text)
                        if t_idx < len(translations):
                            new_parts.append(prefix + (translations[t_idx] or run_text.strip()) + suffix)
                            t_idx += 1
                        else:
                            new_parts.append(run_text)
                    else:
                        new_parts.append(run_text)
            elif seg.need_translate and seg.text.strip():
                prefix, suffix = _segment_prefix_suffix(seg)
                if t_idx < len(translations):
                    new_parts.append(prefix + (translations[t_idx] or seg.text.strip()) + suffix)
                    t_idx += 1
                else:
                    new_parts.append(seg.text)
            else:
                new_parts.append(seg.text)
        block.translated_text = "".join(new_parts)
