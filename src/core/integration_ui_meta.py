# -*- coding: utf-8 -*-
"""集成 UI 元数据（无 Streamlit 依赖），供 aiword 与 integration API 共用。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .db import (
    get_project,
    get_review_extra_instructions,
    get_review_system_prompt,
    get_review_user_prompt,
)


def post_audit_form_meta_defaults_from_report(report: dict[str, Any]) -> dict[str, Any]:
    """从报告 ``_review_meta`` 提取审核后修改表单默认值（对齐 ``app._post_audit_form_meta_defaults`` 的报告部分）。"""
    if not isinstance(report, dict):
        report = {}
    base = report.get("_review_meta") if isinstance(report.get("_review_meta"), dict) else {}
    if (not base) and report.get("batch") and isinstance(report.get("reports"), list):
        for sub in report.get("reports") or []:
            if isinstance(sub, dict) and isinstance(sub.get("_review_meta"), dict) and sub.get(
                "_review_meta"
            ):
                base = dict(sub["_review_meta"])
                break
    out: dict[str, Any] = {}
    for k, v in dict(base).items():
        if v is None:
            out[k] = ""
        elif isinstance(v, (dict, list)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def merge_project_into_post_audit_meta(
    meta: dict[str, Any],
    project: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """用 aicheckword 项目记录补全 post-audit 维度（当 _review_meta 缺字段时）。"""
    if not isinstance(meta, dict):
        meta = {}
    if not isinstance(project, dict) or not project:
        return meta
    pairs = (
        ("project_id", "id"),
        ("project_name", "name"),
        ("product_name", "product_name"),
        ("model", "model"),
        ("model_en", "model_en"),
        ("registration_country", "registration_country"),
        ("registration_type", "registration_type"),
        ("registration_component", "registration_component"),
        ("project_form", "project_form"),
        ("document_language", "document_language"),
        ("scope_of_application", "scope_of_application"),
    )
    for mk, pk in pairs:
        if not str(meta.get(mk) or "").strip():
            val = project.get(pk)
            if val is not None and str(val).strip():
                meta[mk] = str(val).strip() if pk != "id" else int(val)
    return meta


def postaudit_target_key(nm: str) -> str:
    s = str(nm or "").strip()
    if not s:
        return ""
    base = Path(s).name
    m = re.match(r"^\d+_(.+)$", base)
    return (m.group(1) if m else base).strip().lower()


def collapse_remediation_to_template(
    text_by_target: dict[str, str],
    template_key: str,
    base_display_name: str,
) -> dict[str, str]:
    """将 immediate-remediation 的多目标键折叠为与 template_file_names 一致的单一键。"""
    tk = str(template_key or "").strip() or str(base_display_name or "").strip()
    if not tk:
        return dict(text_by_target or {})
    tn = postaudit_target_key(tk)
    bn = postaudit_target_key(base_display_name)
    parts: list[str] = []
    for src, txt in (text_by_target or {}).items():
        t = str(txt or "").strip()
        if not t:
            continue
        sn = postaudit_target_key(src)
        if (
            not tn
            or sn == tn
            or (bn and (sn == bn or bn in sn or sn in bn))
            or len(text_by_target) == 1
        ):
            parts.append(t)
        else:
            parts.append(f"【审核目标：{src}】\n{t}")
    if not parts:
        parts = [str(v).strip() for v in text_by_target.values() if str(v).strip()]
    merged = "\n\n".join(parts).strip()
    return {tk: merged} if merged else {}


def build_kb_query_extra_from_project(
    project_id: int,
    *,
    registration_countries: Optional[List[str]] = None,
    registration_types: Optional[List[str]] = None,
    registration_components: Optional[List[str]] = None,
    project_forms: Optional[List[str]] = None,
) -> str:
    """与 Streamlit 翻译页按项目检索知识库时的 hint 拼接一致。"""
    proj = get_project(int(project_id))
    if not proj:
        return ""
    hint_parts = [
        proj.get("name") or "",
        proj.get("product_name") or "",
        proj.get("name_en") or "",
        proj.get("product_name_en") or "",
        proj.get("model") or "",
        proj.get("model_en") or "",
        (proj.get("scope_of_application") or "")[:500],
        " ".join(registration_countries or []),
        " ".join(registration_types or []),
        " ".join(registration_components or []),
        " ".join(project_forms or []),
    ]
    return " ".join(p for p in hint_parts if p).strip()


def audit_prompt_defaults() -> dict[str, str]:
    """当前库内审核提示词默认值（与 Streamlit ③ 文档审核页加载时一致）。"""
    sys_p = (get_review_system_prompt() or "").strip()
    usr_p = (get_review_user_prompt() or "").strip()
    extra = (get_review_extra_instructions() or "").strip()
    return {
        "system_prompt": sys_p,
        "user_prompt": usr_p,
        "extra_instructions": extra,
    }


def project_row_for_integration(project: dict[str, Any]) -> dict[str, Any]:
    """供 aiword 填入 audit/draft payload 的项目字段子集。"""
    if not isinstance(project, dict):
        return {}
    return {
        "project_id": int(project.get("id") or 0) or None,
        "project_name": str(project.get("name") or "").strip(),
        "project_name_en": str(project.get("name_en") or "").strip(),
        "product_name": str(project.get("product_name") or "").strip(),
        "product_name_en": str(project.get("product_name_en") or "").strip(),
        "model": str(project.get("model") or "").strip(),
        "model_en": str(project.get("model_en") or "").strip(),
        "registration_country": str(project.get("registration_country") or "").strip(),
        "registration_country_en": str(
            project.get("registration_country_en") or ""
        ).strip(),
        "registration_type": str(project.get("registration_type") or "").strip(),
        "registration_component": str(
            project.get("registration_component") or ""
        ).strip(),
        "project_form": str(project.get("project_form") or "").strip(),
        "document_language": str(project.get("document_language") or "").strip(),
        "scope_of_application": str(project.get("scope_of_application") or "").strip(),
    }
