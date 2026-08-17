# -*- coding: utf-8 -*-
"""知识库文件分类常量。

internal_control（内部管控文件）供文控/版本任务等内部制度使用，
不参与审核点生成、初稿生成、文档审核、审核后修改等业务检索。
"""
from __future__ import annotations

from typing import Dict, FrozenSet, Optional


# 分类键 → 中文标签（训练 UI / 统计）
CATEGORY_LABELS: Dict[str, str] = {
    "regulation": "法规文件",
    "program": "程序文件",
    "project_case": "项目案例文件",
    "glossary": "词条",
    "internal_control": "内部管控文件",
}

CATEGORY_VALUES: Dict[str, str] = {v: k for k, v in CATEGORY_LABELS.items()}

# 业务检索默认排除（checklist / draft / review / audit-modify）
AUDIT_EXCLUDED_CATEGORIES: FrozenSet[str] = frozenset({"internal_control"})

# 已入库文档迁入 internal_control 时的文件名匹配（仅用于一次性迁移，不强制新训练归类）
INTERNAL_CONTROL_FILENAME_NEEDLES = (
    "yy-iw-020",
    "yyiw020",
    "iw-020",
    "iw020",
    "医疗软件质量合规管理",
)


def is_audit_excluded_category(category: Optional[str]) -> bool:
    return str(category or "").strip().lower() in AUDIT_EXCLUDED_CATEGORIES


def is_internal_control_filename(file_name: str) -> bool:
    """判断展示文件名是否匹配应迁移的内部管控制度（仅迁移用，训练不强制）。"""
    n = str(file_name or "").strip()
    if not n:
        return False
    low = n.casefold().replace("－", "-").replace(" ", "")
    for needle in INTERNAL_CONTROL_FILENAME_NEEDLES:
        if needle.casefold().replace(" ", "") in low:
            return True
    if "医疗软件质量合规管理" in n:
        return True
    return False
