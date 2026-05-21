# -*- coding: utf-8 -*-
"""与 Streamlit 第三步「文档审核」对齐的 review_context 增强（供 integration API 复用）。"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.core.db import get_knowledge_docs_by_case_id, list_project_cases
from src.core.document_loader import extract_section_outline_from_texts, load_single_file

_DOC_LANG_VALUE_TO_LABEL = {"": "不指定", "zh": "中文版", "en": "英文版", "both": "中英文"}
_COUNTRY_CN_TO_EN = {"中国": "China", "美国": "USA", "欧盟": "EU", "欧洲": "Europe"}


def _as_list(v: Any) -> List[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str) and v.strip():
        return [v.strip()]
    return []


def match_project_case_for_review(
    collection: str,
    doc_text: str,
    dimension_filters: dict,
) -> Optional[dict]:
    """与 app._match_project_case_for_review 一致：先产品名命中，再按注册维度匹配。"""
    cases = list_project_cases(collection) or []
    if not cases:
        return None
    doc_text = (doc_text or "").strip()
    for c in cases:
        product_name = (c.get("product_name") or "").strip()
        product_name_en = (c.get("product_name_en") or "").strip()
        scope = (c.get("scope_of_application") or "").strip()
        name_match = (product_name and product_name in doc_text) or (
            product_name_en and product_name_en in doc_text
        )
        if name_match:
            if not scope or scope in doc_text:
                return c
            if scope and len(scope) > 50:
                for part in (scope[:80], scope[50:130]):
                    if part.strip() and part.strip() in doc_text:
                        return c
            return c
    sel_countries = _as_list(dimension_filters.get("registration_country"))
    sel_types = _as_list(dimension_filters.get("registration_type"))
    sel_components = _as_list(dimension_filters.get("registration_component"))
    sel_forms = _as_list(dimension_filters.get("project_form"))
    if not (sel_countries or sel_types or sel_components or sel_forms):
        return None
    match_countries = set(sel_countries) | {
        _COUNTRY_CN_TO_EN.get(x, x) for x in sel_countries if x
    }
    for c in cases:
        if sel_countries:
            cc = (c.get("registration_country") or "").strip()
            cc_en = (c.get("registration_country_en") or "").strip()
            if not (cc in match_countries or cc_en in match_countries):
                continue
        if sel_types and (c.get("registration_type") or "") not in sel_types:
            continue
        if sel_components and (c.get("registration_component") or "") not in sel_components:
            continue
        if sel_forms and (c.get("project_form") or "") not in sel_forms:
            continue
        return c
    return None


def build_case_context_text(collection: str, matched_case: dict) -> str:
    """构建与 Streamlit 审核页相同的 case_context_text。"""
    case_lang = matched_case.get("document_language") or ""
    case_lang_label = _DOC_LANG_VALUE_TO_LABEL.get(case_lang, "不指定")
    case_ctx = (
        f"\n\n【过往项目案例参考】\n"
        f"案例文档语言：{case_lang_label}\n"
        f"案例名称：{matched_case.get('case_name', '')}\n"
        f"案例名称（英文）：{matched_case.get('case_name_en', '')}\n"
        f"产品名称：{matched_case.get('product_name', '')}\n"
        f"产品名称（英文）：{matched_case.get('product_name_en', '')}\n"
        f"注册国家：{matched_case.get('registration_country', '')}\n"
        f"注册国家（英文）：{matched_case.get('registration_country_en', '')}\n"
        f"注册类别：{matched_case.get('registration_type', '')}\n"
        f"注册组成：{matched_case.get('registration_component', '')}\n"
        f"项目形态：{matched_case.get('project_form', '')}\n"
    )
    scope = (matched_case.get("scope_of_application") or "").strip()
    if scope:
        case_ctx += f"产品适用范围：{scope}\n"
    try:
        case_chunks = get_knowledge_docs_by_case_id(
            collection, int(matched_case["id"]), limit=500
        )
        if case_chunks:
            outline = extract_section_outline_from_texts(
                [c.get("content") or "" for c in case_chunks]
            )
            if outline.strip():
                case_ctx += (
                    "\n【历史案例文档章节参考】\n以下为案例库中该案例文档的章节结构，"
                    "请据此检查待审文档是否具备应有章节；缺失的章节须作为「文档内容完整性」"
                    "审核点列出，并指明应补充的章节名称或位置。\n\n"
                    + outline.strip()
                    + "\n"
                )
                case_ctx += (
                    "\n**完整性审核执行要求**：请按上表章节**逐条**在待审文档中查找对应或等价章节；"
                    "每发现一处缺失或标题/层级明显不一致，单列一条审核点，不得合并为多处以「若干章节缺失」一笔带过。"
                    "location 须写清缺失章节名称或待审文档中应出现的位置。\n"
                )
    except Exception:
        pass
    case_ctx += "\n请参考上述案例经验审核当前文档，如有类似问题请重点关注。"
    return case_ctx


def peek_document_text(
    file_path: str,
    *,
    display_name: str = "",
    max_chars: int = 15000,
    force_ocr_refresh: bool = False,
) -> str:
    try:
        docs = load_single_file(
            file_path,
            force_ocr_refresh=force_ocr_refresh,
            ocr_cache_file_name=(display_name or "").strip() or None,
        )
        blob = "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)
        return (blob or "")[: max(0, int(max_chars))]
    except Exception:
        return ""


def enrich_review_context_for_integration(
    ctx: Dict[str, Any],
    *,
    collection: str,
    project_id: Optional[int],
    doc_text_for_case_match: str = "",
    auto_match_case: bool = True,
) -> Dict[str, Any]:
    """
    补齐 integration 审核与 Streamlit 页面对齐的上下文键：
    - _filter_by_registration_type（按项目时启用）
    - case_context_text（自动匹配过往案例 + 章节完整性要求）
    """
    out = dict(ctx or {})
    if project_id:
        out["_filter_by_registration_type"] = True
    if not auto_match_case:
        return out
    text = (doc_text_for_case_match or "").strip()
    if not text:
        return out
    dim = {
        "registration_country": _as_list(out.get("registration_country")),
        "registration_type": _as_list(out.get("registration_type")),
        "registration_component": _as_list(out.get("registration_component")),
        "project_form": _as_list(out.get("project_form")),
    }
    matched = match_project_case_for_review(collection, text, dim)
    if matched:
        out["case_context_text"] = build_case_context_text(collection, matched)
        out["_matched_case_id"] = matched.get("id")
        out["_matched_case_name"] = matched.get("case_name") or matched.get("product_name")
    return out
