"""
程序级基础质量扫描：序号跳号、重复文案、关键字段前后不一致、引用文件/标准号不一致。

用于补足 LLM 审核时容易跳过的「人眼一眼能看出」的问题。误报优先保守：
只报区间内部缺口、足够长的重复段、同标签不同取值、同标准号不同年代。
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

_CN_DIGIT = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}

_HEADER_FOOTER_HINTS = (
    "机密",
    "confidential",
    "受控",
    "页眉",
    "页脚",
    "第 页",
    "page of",
)

_FIELD_SPECS: Sequence[Tuple[str, re.Pattern[str]]] = (
    ("产品名称", re.compile(r"(?:产品名称|Product\s*Name)\s*[:：]\s*(.+)", re.I)),
    ("项目名称", re.compile(r"(?:项目名称|Project\s*Name)\s*[:：]\s*(.+)", re.I)),
    ("型号", re.compile(r"(?:规格型号|产品型号|型号规格|Model(?:\s*No\.?)?)\s*[:：]\s*(.+)", re.I)),
    ("软件版本", re.compile(r"(?:软件版本|Software\s*Version)\s*[:：]\s*(.+)", re.I)),
    ("文件版本", re.compile(r"(?:文件版本|文档版本|Document\s*Version)\s*[:：]\s*(.+)", re.I)),
    ("适用范围", re.compile(r"(?:适用范围|Scope(?:\s*of\s*Application)?)\s*[:：]\s*(.+)", re.I)),
    ("预期用途", re.compile(r"(?:预期用途|Intended\s*Use)\s*[:：]\s*(.+)", re.I)),
    ("软件安全级别", re.compile(r"(?:软件安全级别|Software\s*Safety\s*Class)\s*[:：]\s*(.+)", re.I)),
    ("发布版本", re.compile(r"(?:发布版本|Release\s*Version)\s*[:：]\s*(.+)", re.I)),
)

_HEADING_LINE = re.compile(
    r"^[ \t]*(?:第\s*)?(\d{1,2}(?:\.\d{1,2}){1,4}|\d{1,2})(?:\s*[章节条款篇]|[、.．]\s+|\s+)([^\n]{2,80})$",
    re.M,
)
_CHAPTER_CN = re.compile(r"第\s*([0-9一二三四五六七八九十两〇零]+)\s*章")
_LIST_LEAD = re.compile(r"^[ \t]*(?:[（(])?(\d{1,2})(?:[）)]|[、.．)])\s+\S")
_STD_REF = re.compile(
    r"\b((?:GB/T|GB|YY/T|YY|ISO|IEC|EN)\s*[\d.]+)(?:[-—](\d{4}))?",
    re.I,
)
_BOOK_TITLE = re.compile(r"《([^》]{2,80})》")
_TITLE_VER = re.compile(
    r"《([^》]{2,80})》\s*[（(]?\s*(?:V|v|版本|Rev\.?)\s*([0-9]+(?:\.[0-9]+)*)",
)
_DOC_NO_TITLE = re.compile(
    r"\b([A-Z]{2,8}\d*(?:\.\d+)?[-–][A-Z0-9]{1,8}[-–]?\d{0,4})\s*[《“\"]([^》\"”]{2,60})",
    re.I,
)


@dataclass
class BasicQualityFinding:
    category: str
    severity: str
    location: str
    description: str
    regulation_ref: str
    suggestion: str
    modify_docs: List[str] = field(default_factory=list)


def _cn_numeral_to_int(raw: str) -> Optional[int]:
    s = (raw or "").strip()
    if not s:
        return None
    if s.isdigit():
        n = int(s)
        return n if 1 <= n <= 99 else None
    if s == "十":
        return 10
    if s.startswith("十"):
        rest = _cn_numeral_to_int(s[1:])
        return 10 + (rest or 0) if rest is not None or not s[1:] else None
    if "十" in s:
        a, b = s.split("十", 1)
        tens = _cn_numeral_to_int(a or "一")
        ones = _cn_numeral_to_int(b) if b else 0
        if tens is None or ones is None:
            return None
        return tens * 10 + ones
    if len(s) == 1 and s in _CN_DIGIT:
        return _CN_DIGIT[s]
    return None


def _internal_gaps(nums: Sequence[int], *, max_span: int = 40) -> List[int]:
    uniq = sorted({int(n) for n in nums if isinstance(n, int) and n >= 0})
    if len(uniq) < 2:
        return []
    lo, hi = uniq[0], uniq[-1]
    if hi - lo > max_span or hi == lo:
        return []
    present = set(uniq)
    return [i for i in range(lo, hi + 1) if i not in present]


def _norm_value(s: str) -> str:
    t = re.sub(r"\s+", " ", (s or "").strip())
    t = t.strip(" ，,;；。.!！？?|｜\t\"'“”")
    t = re.split(r"[|｜\t]", t, maxsplit=1)[0].strip()
    if len(t) > 80:
        t = t[:80]
    return t


def _norm_para(s: str) -> str:
    t = re.sub(r"\s+", "", (s or "").strip().lower())
    t = re.sub(r"[，。、“”‘’：:；;,.!?！？()（）\[\]【】<>《》\-_/\\]+", "", t)
    return t


def _looks_like_heading_title(title: str) -> bool:
    t = (title or "").strip()
    if len(t) < 2:
        return False
    if re.match(r"^[\d.\-_/\\]+$", t):
        return False
    if re.match(r"^(?:20\d{2}|19\d{2})[.\-/]", t):
        return False
    return True


def _scan_heading_gaps(text: str) -> List[BasicQualityFinding]:
    out: List[BasicQualityFinding] = []
    by_parent: Dict[str, List[int]] = defaultdict(list)
    samples: Dict[str, List[str]] = defaultdict(list)
    for m in _HEADING_LINE.finditer(text or ""):
        num, title = m.group(1), m.group(2)
        if not _looks_like_heading_title(title):
            continue
        parts = [int(x) for x in num.split(".") if x.isdigit()]
        if not parts or parts[0] > 40:
            continue
        # 形如 2.0 / 1.0 更像版本号，不当章节
        if len(parts) == 2 and parts[1] == 0:
            continue
        parent = ".".join(str(x) for x in parts[:-1]) if len(parts) > 1 else ""
        sib = parts[-1]
        if sib <= 0:
            continue
        key = parent or "_"
        if sib not in by_parent[key]:
            by_parent[key].append(sib)
            samples[key].append(f"{num} {title.strip()[:24]}")
    for parent, sibs in by_parent.items():
        gaps = _internal_gaps(sibs, max_span=25)
        if not gaps:
            continue
        loc_prefix = f"{parent}." if parent and parent != "_" else ""
        shown = "、".join(samples[parent][:6])
        missing = "、".join(f"{loc_prefix}{g}" for g in gaps[:8])
        out.append(
            BasicQualityFinding(
                category="格式规范",
                severity="medium",
                location=f"章节编号 {loc_prefix or ''}*",
                description=(
                    f"同级章节/条款编号不连续，缺失 {missing}。"
                    f"已识别到的邻近条目包括：{shown}。"
                ),
                regulation_ref="受控文件结构与编号连续性（通用文件控制要求）",
                suggestion=f"请按同级连续编号补全或重排缺失项（{missing}），并同步修订目录与正文交叉引用。",
            )
        )
        if len(out) >= 8:
            break

    chapters = []
    for m in _CHAPTER_CN.finditer(text or ""):
        n = _cn_numeral_to_int(m.group(1))
        if n:
            chapters.append(n)
    ch_gaps = _internal_gaps(chapters, max_span=20)
    if ch_gaps:
        missing = "、".join(f"第{g}章" for g in ch_gaps[:8])
        out.append(
            BasicQualityFinding(
                category="完整性",
                severity="medium",
                location="章标题",
                description=f"正文「第X章」编号不连续，缺失 {missing}。",
                regulation_ref="受控文件结构完整性（通用文件控制要求）",
                suggestion=f"请核对目录与正文，补全或修正缺失章节（{missing}）。",
            )
        )
    return out


def _scan_list_runs(text: str) -> List[BasicQualityFinding]:
    out: List[BasicQualityFinding] = []
    run: List[Tuple[int, str]] = []

    def flush() -> None:
        nonlocal run
        if len(run) >= 3:
            nums = [n for n, _ in run]
            gaps = _internal_gaps(nums, max_span=15)
            # 仅当序列大体递增且存在内部跳号时报告
            if gaps and nums[-1] >= nums[0] + 2:
                sample = " → ".join(t[:18] for _, t in run[:5])
                missing = "、".join(str(g) for g in gaps[:8])
                out.append(
                    BasicQualityFinding(
                        category="格式规范",
                        severity="medium",
                        location=run[0][1][:40],
                        description=(
                            f"连续条目序号不连贯，缺失序号 {missing}。"
                            f"邻近条目：{sample}。"
                        ),
                        regulation_ref="清单/条款序号连续性（通用文件控制要求）",
                        suggestion=f"请将该组条目按 1,2,3… 连续编号，补上或删除跳号（缺失 {missing}）。",
                    )
                )
        run = []

    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            flush()
            continue
        m = _LIST_LEAD.match(line)
        if not m:
            flush()
            continue
        n = int(m.group(1))
        if n <= 0 or n > 40:
            flush()
            continue
        if not run:
            run = [(n, line[:60])]
            continue
        prev = run[-1][0]
        if n == 1 and prev >= 1:
            flush()
            run = [(n, line[:60])]
            continue
        if n >= prev:
            run.append((n, line[:60]))
            if len(run) > 30:
                flush()
            continue
        flush()
        run = [(n, line[:60])]
    flush()
    return out[:6]


def _scan_duplicate_paragraphs(text: str) -> List[BasicQualityFinding]:
    blocks = [
        re.sub(r"\s+", " ", blk.strip())
        for blk in re.split(r"\n\s*\n", text or "")
        if blk.strip()
    ]
    long_blocks = [b for b in blocks if len(b) >= 36]
    if len(long_blocks) >= 2:
        units = long_blocks
    else:
        units = [s.strip() for s in re.split(r"[。！？]", text or "") if len(s.strip()) >= 36]

    counts: Dict[str, int] = defaultdict(int)
    samples: Dict[str, str] = {}
    for p in units:
        if any(h in p.lower() for h in _HEADER_FOOTER_HINTS):
            continue
        if "编制" in p and "审核" in p and "批准" in p:
            continue
        key = _norm_para(p)
        if len(key) < 24:
            continue
        counts[key] += 1
        samples.setdefault(key, p)

    out: List[BasicQualityFinding] = []
    for key, n in counts.items():
        sample = samples[key]
        L = len(sample)
        if n < 2 or L < 36:
            continue
        if L < 48 and n >= 4:
            continue
        loc = sample[:48] + ("…" if len(sample) > 48 else "")
        out.append(
            BasicQualityFinding(
                category="准确性",
                severity="medium" if L >= 80 or n >= 3 else "low",
                location=loc,
                description=(
                    f"文档中出现 {n} 处相同或几乎相同的文案（约 {L} 字），"
                    f"疑似复制粘贴残留。原文摘录：「{sample[:120]}{'…' if L > 120 else ''}」"
                ),
                regulation_ref="文件内容唯一性/防重复（通用文件控制要求）",
                suggestion="请删除重复段落，或改写为交叉引用（见第X节），避免同一段正文在多处原样重复。",
            )
        )
        if len(out) >= 6:
            break
    return out


def _scan_field_inconsistencies(text: str) -> List[BasicQualityFinding]:
    out: List[BasicQualityFinding] = []
    for label, pat in _FIELD_SPECS:
        values: List[str] = []
        seen_norm = []
        for m in pat.finditer(text or ""):
            val = _norm_value(m.group(1))
            if len(val) < 2:
                continue
            nv = re.sub(r"^v", "", val, flags=re.I).strip()
            if nv not in seen_norm:
                seen_norm.append(nv)
                values.append(val)
        if len(values) < 2:
            continue
        shown = "」与「".join(values[:4])
        out.append(
            BasicQualityFinding(
                category="一致性",
                severity="high" if label in ("产品名称", "型号", "适用范围", "预期用途") else "medium",
                location=f"字段「{label}」",
                description=(
                    f"同一文档内「{label}」前后取值不一致，分别出现：「{shown}」。"
                    "属于明显的前后描述不一致。"
                ),
                regulation_ref="关键信息全文一致（通用文件控制/注册资料一致性）",
                suggestion=f"请统一全文「{label}」为与项目资料一致的唯一表述，并同步封面、页眉、修订页与正文。",
            )
        )
        if len(out) >= 8:
            break
    return out


def _core_title(title: str) -> str:
    t = re.sub(r"[\s（()）]", "", title or "")
    t = re.sub(r"(?:试行|修订|最新|受控|正式)版?", "", t)
    t = re.sub(r"v?\d+(?:\.\d+)*", "", t, flags=re.I)
    return t.lower()


def _scan_citation_inconsistencies(text: str) -> List[BasicQualityFinding]:
    out: List[BasicQualityFinding] = []
    by_std: Dict[str, set] = defaultdict(set)
    for m in _STD_REF.finditer(text or ""):
        base = re.sub(r"\s+", "", m.group(1).upper())
        year = (m.group(2) or "").strip()
        if year:
            by_std[base].add(year)
    for base, years in by_std.items():
        if len(years) < 2:
            continue
        ys = "、".join(sorted(years))
        out.append(
            BasicQualityFinding(
                category="准确性",
                severity="medium",
                location=f"引用标准 {base}",
                description=f"同一标准号 {base} 在文中出现了不同年代号：{ys}。引用文件不一致。",
                regulation_ref="规范性引用文件年代号一致（通用标准化/文件控制要求）",
                suggestion=f"请统一 {base} 的年代号（以现行有效版本或项目受控清单为准），并同步引用文件清单与正文。",
            )
        )
        if len(out) >= 5:
            break

    ver_map: Dict[str, set] = defaultdict(set)
    for m in _TITLE_VER.finditer(text or ""):
        ver_map[_core_title(m.group(1))].add(m.group(2))
    for core, vers in ver_map.items():
        if not core or len(vers) < 2:
            continue
        out.append(
            BasicQualityFinding(
                category="一致性",
                severity="medium",
                location=f"引用文件《{core}》",
                description=f"同一引用文件出现了不同版本号：{'、'.join(sorted(vers))}。",
                regulation_ref="引用文件版本一致（通用文件控制要求）",
                suggestion="请在引用文件清单与正文中统一该文件的受控版本，删除过期版本引用或标明作废关系。",
            )
        )
        if len(out) >= 8:
            break

    no_map: Dict[str, set] = defaultdict(set)
    for m in _DOC_NO_TITLE.finditer(text or ""):
        no_map[m.group(1).upper()].add(_norm_value(m.group(2)))
    for no, titles in no_map.items():
        titles = {t for t in titles if t}
        if len(titles) < 2:
            continue
        shown = "」与「".join(list(titles)[:3])
        out.append(
            BasicQualityFinding(
                category="一致性",
                severity="medium",
                location=f"文件编号 {no}",
                description=f"同一文件编号 {no} 对应了不同文件名：「{shown}」。引用文件不一致。",
                regulation_ref="受控文件编号与名称一一对应（通用文件控制要求）",
                suggestion=f"请核对编号 {no} 的正确文件名，统一清单与正文引用。",
            )
        )
        if len(out) >= 10:
            break

    # 引用文件清单 vs 正文：《》标题
    list_sec = _extract_reference_section(text or "")
    if list_sec:
        listed = {_core_title(t) for t in _BOOK_TITLE.findall(list_sec) if len(t) >= 4}
        body = (text or "")
        # 去掉清单段再取正文引用，避免自己比自己
        body_wo = body.replace(list_sec, " ", 1)
        cited = {_core_title(t) for t in _BOOK_TITLE.findall(body_wo) if len(t) >= 4}
        if len(listed) >= 3:
            missing_in_list = [c for c in cited if c and c not in listed]
            # 仅报正文有、清单无（更像漏列）；清单有正文未引用常见且不一定是错
            for c in missing_in_list[:4]:
                out.append(
                    BasicQualityFinding(
                        category="完整性",
                        severity="low",
                        location="引用文件 / 正文交叉引用",
                        description=f"正文引用了《{c}》，但未出现在「引用文件/规范性引用文件/参考文献」清单中。",
                        regulation_ref="引用文件清单完整性（通用文件控制要求）",
                        suggestion=f"请将《{c}》补入引用文件清单，或改正文为清单中的受控文件名。",
                    )
                )
    return out[:10]


def _extract_reference_section(text: str) -> str:
    m = re.search(
        r"(?:^|\n)\s*(?:\d+(?:\.\d+)*\s*)?(?:规范性引用文件|引用文件|参考文献)[^\n]{0,40}\n"
        r"(.{80,6000}?)(?=\n\s*(?:\d+(?:\.\d+)*\s*)?(?:目的|范围|职责|定义|术语|职责与权限)\b|\Z)",
        text,
        re.S,
    )
    return m.group(1) if m else ""


def scan_basic_quality_issues(text: str, file_name: str = "") -> List[BasicQualityFinding]:
    blob = text or ""
    if len(blob.strip()) < 20:
        return []
    fn = (file_name or "").strip()
    findings: List[BasicQualityFinding] = []
    findings.extend(_scan_heading_gaps(blob))
    findings.extend(_scan_list_runs(blob))
    findings.extend(_scan_duplicate_paragraphs(blob))
    findings.extend(_scan_field_inconsistencies(blob))
    findings.extend(_scan_citation_inconsistencies(blob))
    if fn:
        for f in findings:
            if not f.modify_docs:
                f.modify_docs = [fn]
    # 去重
    seen = set()
    uniq: List[BasicQualityFinding] = []
    for f in findings:
        key = (f.category, f.location[:40], f.description[:80])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(f)
    return uniq[:24]


BASIC_QUALITY_PROMPT_BLOCK = """
- **基础质量（必须先查，禁止因「太浅」而跳过）**：
  1. **序号连贯**：章节号、条款号、列表序号、表格「序号」列是否连续；有无跳号、重号、目录与正文编号不一致。
  2. **重复文案**：是否存在整段或接近整段的复制粘贴重复（同一段话在多处原样出现）。
  3. **前后描述一致**：产品名称、型号、版本、适用范围、预期用途、软件安全级别、发布版本等同一信息在封面/修订页/正文/表格中是否同一表述。
  4. **引用文件一致**：「引用文件/规范性引用文件/参考文献」清单与正文引用的文件名、文件编号、标准号及年代号是否一致（清单有正文无、正文有清单无、同号不同年、同编号不同文件名）。
  上述问题须逐条输出审核点（category 用格式规范/一致性/完整性/准确性），location 引用原文或章节号；不得只写法规层面结论而忽略这些明显问题。
""".strip()
