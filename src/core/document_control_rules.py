"""从《文件控制程序》等知识库文本解析受控文件编号规则（对齐 QP 4.2.4 结构）。"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# QP 4.2.4 缺省结构（知识库未命中细节时回退；程序文件更新后仍以 KB 检索结果覆盖 excerpt）
_DEFAULT_PROCEDURE_RULES: List[Dict[str, Any]] = [
    {
        "docTypeCode": "QM",
        "name": "质量手册",
        "level": 1,
        "renderTemplate": "{type}{seq:03d}",
        "prefixSource": "fixed",
        "fixedPrefix": "QM",
        "seqStart": 1,
        "seqPad": 3,
        "example": "QM001",
        "autoAllocatable": True,
        "needsProjectCode": False,
        "sheetCategory": "程序文件",
    },
    {
        "docTypeCode": "QP",
        "name": "程序文件（二级）",
        "level": 2,
        "renderTemplate": "",
        "prefixSource": "fixed",
        "fixedPrefix": "QP",
        "example": "QP4.2.3",
        "autoAllocatable": False,
        "needsProjectCode": False,
        "manualHint": "按标准条款编号，例如 QP4.2.3",
        "sheetCategory": "程序文件",
    },
    {
        "docTypeCode": "SMP",
        "name": "管理性文件（三级）",
        "level": 3,
        "renderTemplate": "",
        "prefixSource": "fixed",
        "fixedPrefix": "SMP",
        "example": "SMP5.1-01",
        "autoAllocatable": False,
        "needsProjectCode": False,
        "manualHint": "按条款与顺序号，例如 SMP5.1-01",
        "sheetCategory": "程序文件",
    },
    {
        "docTypeCode": "SOP",
        "name": "操作性文件（四级）",
        "level": 4,
        "renderTemplate": "{prefix}-{type}{seq:03d}",
        "prefixSource": "from_project_code",
        "seqStart": 1,
        "seqPad": 3,
        "example": "XXX-SOP001",
        "autoAllocatable": True,
        "needsProjectCode": True,
        "sheetCategory": "SOP",
    },
    {
        "docTypeCode": "SRS",
        "name": "技术文件（设计/规格）",
        "level": 5,
        "renderTemplate": "{prefix}-{subtype}-{seq:03d}",
        "prefixSource": "from_project_code",
        "seqStart": 1,
        "seqPad": 3,
        "example": "PACS-SRS-001",
        "autoAllocatable": True,
        "needsProjectCode": True,
        "needsSubtype": True,
        "subtypeFromTitle": True,
        "sheetCategory": "DHF",
    },
    {
        "docTypeCode": "WL",
        "name": "外来文件",
        "level": 0,
        "renderTemplate": "",
        "example": "沿用标准号/文件号",
        "autoAllocatable": False,
        "needsProjectCode": False,
        "manualHint": "沿用原标准号或文件号",
        "sheetCategory": "四级表单",
    },
    {
        "docTypeCode": "QR",
        "name": "质量记录",
        "level": 0,
        "renderTemplate": "{type}-{ref}-{seq:02d}",
        "prefixSource": "fixed",
        "fixedPrefix": "QR",
        "seqStart": 1,
        "seqPad": 2,
        "example": "QR-QP4.2.4-01",
        "autoAllocatable": False,
        "needsProjectCode": False,
        "needsParentRef": True,
        "manualHint": "QR-二/三级文件编号-顺序号，例如 QR-QP4.2.4-01",
        "sheetCategory": "四级表单",
    },
]

_SECTION_MARKERS = (
    (r"质量手册的编号规则", "QM"),
    (r"程序文件的编号规则", "QP"),
    (r"管理性文件的编号规则", "SMP"),
    (r"操作性文件的编号规则", "SOP"),
    (r"技术文件编号规则", "SRS"),
    (r"外来文件编号", "WL"),
    (r"记录表格编号规则", "QR"),
)


def _clone_rule(base: Dict[str, Any]) -> Dict[str, Any]:
    return dict(base)


def _extract_example_after(blob: str, start: int, code: str) -> str:
    window = blob[start : start + 360]
    next_sec = re.search(r"(?:的编号规则|编号规则如下|编号：)", window[30:])
    if next_sec:
        window = window[: 30 + next_sec.start()]
    for pat in (
        rf"例如[：:]\s*({re.escape(code)}[^\n；;。]*)",
        rf"(?:即为|如)[：:]\s*({re.escape(code)}[^\n；;。]*)",
    ):
        m = re.search(pat, window, re.I)
        if m:
            return (m.group(1) or "").strip()
    return ""


def _extract_excerpt(blob: str, marker: str, code: str) -> str:
    m = re.search(marker, blob)
    if not m:
        return ""
    start = m.start()
    return blob[start : start + 220].replace("\n", " ").strip()


def parse_document_control_procedure_rules(text: str) -> List[Dict[str, Any]]:
    """解析知识库中的《文件控制程序》编号章节；返回带业务语义的规则列表。"""
    blob = str(text or "").strip()
    by_code = {r["docTypeCode"]: _clone_rule(r) for r in _DEFAULT_PROCEDURE_RULES}
    if not blob:
        return list(by_code.values())

    for marker, code in _SECTION_MARKERS:
        m = re.search(marker, blob)
        if not m:
            continue
        rule = by_code.get(code)
        if not rule:
            continue
        excerpt = _extract_excerpt(blob, marker, code)
        if excerpt:
            rule["kbRuleExcerpt"] = excerpt
        example = _extract_example_after(blob, m.start(), code)
        if example:
            rule["example"] = example
        elif code == "WL":
            rule["example"] = "沿用标准号/文件号"

    # 知识库命中《文件控制程序》但未逐条解析时，仍返回完整缺省规则集（excerpt 可能为空）
    if "文件控制程序" in blob or "文件和资料的编号方法" in blob:
        for rule in by_code.values():
            rule.setdefault("kbRuleExcerpt", rule.get("kbRuleExcerpt") or f"{rule.get('name') or ''}：{rule.get('example') or ''}")

    return [by_code[c] for c in (r["docTypeCode"] for r in _DEFAULT_PROCEDURE_RULES) if c in by_code]


def merge_kb_rules_with_fallback(
    kb_text: str,
    *,
    source_file: str = "",
) -> List[Dict[str, Any]]:
    rules = parse_document_control_procedure_rules(kb_text)
    if source_file:
        for rule in rules:
            rule.setdefault("kbSourceFile", source_file)
    return rules
