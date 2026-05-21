"""
集成通用元数据：供 aiword 审核/翻译/审核后修改等页下拉与默认值。

GET /api/integration/common/bootstrap?collection=regulations
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query

from src.core.db import (
    REGISTRATION_COMPONENTS,
    REGISTRATION_TYPES,
    get_dimension_options,
    get_translation_config,
    list_project_cases,
    list_projects,
)

router = APIRouter(prefix="/api/integration/common", tags=["integration-common"])

# 与 src/app.py 侧栏文档语言选项一致（勿从 db 导入，该常量仅在 app 中定义）
_DOC_LANG_VALUE_TO_LABEL = {"": "不指定", "zh": "中文版", "en": "英文版", "both": "中英文"}
_DOC_LANG_ORDER = ("", "zh", "en", "both")


def _rows_from_strings(items: List[str], *, allow_empty: bool = False) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if allow_empty:
        out.append({"value": "", "label": "不指定"})
    for it in items or []:
        s = str(it or "").strip()
        if not s:
            continue
        out.append({"value": s, "label": s})
    return out


@router.get("/bootstrap")
def integration_common_bootstrap(
    collection: str = Query("regulations"),
):
    """项目列表 + 注册维度 + 文档语言 + 翻译公司默认配置（与 Streamlit 侧栏一致）。"""
    try:
        dims = get_dimension_options() or {}
    except Exception:
        dims = {}
    try:
        projects = list_projects(collection) or []
    except Exception:
        projects = []
    try:
        cases_raw = list_project_cases(collection) or []
    except Exception:
        cases_raw = []
    try:
        tcfg = get_translation_config() or {}
    except Exception:
        tcfg = {}

    countries = list(dims.get("registration_countries") or ["中国", "美国", "欧盟"])
    forms = list(dims.get("project_forms") or ["Web", "APP", "PC"])

    project_rows: List[Dict[str, Any]] = []
    for p in projects:
        if not isinstance(p, dict):
            continue
        try:
            pid = int(p.get("id") or 0)
        except (TypeError, ValueError):
            continue
        if pid <= 0:
            continue
        name = str(p.get("name") or "").strip()
        product = str(p.get("product_name") or "").strip()
        label = name or f"项目#{pid}"
        if product:
            label = f"{label} · {product}"
        label = f"{label} (ID:{pid})"
        project_rows.append(
            {
                "id": pid,
                "label": label,
                "name": name,
                "productName": product,
                "productNameEn": str(p.get("product_name_en") or "").strip(),
                "registrationCountry": str(p.get("registration_country") or "").strip(),
                "registrationCountryEn": str(
                    p.get("registration_country_en") or ""
                ).strip(),
                "registrationType": str(p.get("registration_type") or "").strip(),
                "registrationComponent": str(
                    p.get("registration_component") or ""
                ).strip(),
                "projectForm": str(p.get("project_form") or "").strip(),
                "model": str(p.get("model") or "").strip(),
                "modelEn": str(p.get("model_en") or "").strip(),
            }
        )

    doc_lang_rows = [
        {"value": k, "label": _DOC_LANG_VALUE_TO_LABEL.get(k, k or "不指定")}
        for k in _DOC_LANG_ORDER
    ]

    case_rows: List[Dict[str, Any]] = []
    for c in cases_raw:
        if not isinstance(c, dict):
            continue
        try:
            cid = int(c.get("id") or 0)
        except (TypeError, ValueError):
            continue
        if cid <= 0:
            continue
        cn = str(c.get("case_name") or "").strip()
        pn = str(c.get("product_name") or "").strip()
        label = f"ID:{cid}"
        if cn:
            label += f" | {cn}"
        if pn:
            label += f" · {pn}"
        case_rows.append(
            {
                "id": cid,
                "label": label,
                "productName": pn,
                "productNameEn": str(c.get("product_name_en") or "").strip(),
                "registrationCountry": str(c.get("registration_country") or "").strip(),
                "registrationCountryEn": str(
                    c.get("registration_country_en") or ""
                ).strip(),
                "documentLanguage": str(c.get("document_language") or "").strip(),
                "registrationType": str(c.get("registration_type") or "").strip(),
                "projectForm": str(c.get("project_form") or "").strip(),
            }
        )

    company_cfg = (
        tcfg.get("company_config")
        if isinstance(tcfg.get("company_config"), dict)
        else {}
    ) or {}

    return {
        "ok": True,
        "collection": collection,
        "projects": project_rows,
        "cases": case_rows,
        "documentLanguages": doc_lang_rows,
        "registrationCountries": _rows_from_strings(countries, allow_empty=True),
        "registrationTypes": _rows_from_strings(REGISTRATION_TYPES, allow_empty=True),
        "registrationComponents": _rows_from_strings(
            REGISTRATION_COMPONENTS, allow_empty=True
        ),
        "projectForms": _rows_from_strings(forms, allow_empty=True),
        "targetLangDefault": (tcfg.get("target_lang") or "en"),
        "supportedTargetLangs": [
            {"value": "en", "label": "英文（en）"},
            {"value": "de", "label": "德文（de）"},
            {"value": "zh", "label": "中文（zh）"},
        ],
        "companyConfig": company_cfg,
    }
