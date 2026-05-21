# -*- coding: utf-8 -*-
"""初稿集成页 UI 元数据（无 Streamlit 依赖），与 ``src/app.py`` 初稿页同源。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.core.db import get_project, get_project_case_file_names, list_project_cases, list_projects
from src.core.integration_ui_meta import project_row_for_integration

DOC_LANG_VALUE_TO_LABEL: Dict[str, str] = {
    "": "不指定",
    "zh": "中文版",
    "en": "英文版",
    "both": "中英文",
}
DOC_LANG_ORDER: tuple[str, ...] = ("", "zh", "en", "both")

DRAFT_AUTHOR_ROLE_KEYS: tuple[str, ...] = (
    "",
    "pm",
    "pjm",
    "rm",
    "rdm",
    "ui",
    "qa",
    "cm",
    "ra",
    "prod",
)

DRAFT_AUTHOR_ROLE_LABELS: tuple[str, ...] = (
    "（未指定）通用技术编写",
    "产品经理",
    "项目经理",
    "风险经理",
    "研发经理",
    "UI设计师",
    "测试工程师",
    "配置管理员",
    "注册工程师",
    "生产专员",
)

DRAFT_STRATEGY_OPTIONS: tuple[dict[str, str], ...] = (
    {
        "value": "change",
        "label": "注册变更：对照参考在基础文件上自动识别新增/细化/删除（保留版式与未涉及原文）",
    },
    {
        "value": "reuse",
        "label": "新项目复用：按参考文件全量更新内容（保留格式章节不变）",
    },
)


def infer_draft_author_role_key(
    file_names: list,
    *,
    registration_type: str = "",
    project_form: str = "",
) -> str:
    """根据待生成文件名与案例注册类别/项目形态推断 author_role（与 Streamlit 初稿页一致）。"""
    scores = {k: 0 for k in DRAFT_AUTHOR_ROLE_KEYS}

    def _idx(k: str) -> int:
        try:
            return DRAFT_AUTHOR_ROLE_KEYS.index(k)
        except ValueError:
            return 0

    rt = (registration_type or "").strip()
    pf = (project_form or "").strip()
    high_risk_reg = any(x in rt for x in ("三类", "Ⅲ", "Ⅱb", "Ⅱa"))

    for fn in file_names or []:
        s = (fn or "").strip()
        if not s:
            continue
        low = s.lower()

        def _hit(*parts: str) -> bool:
            for p in parts:
                if not p:
                    continue
                if all(ord(c) < 128 for c in p):
                    if p.lower() in low:
                        return True
                else:
                    if p in s:
                        return True
            return False

        if _hit(
            "测试用例",
            "test case",
            "test execution",
            "system test",
            "system testing",
            "确认测试",
            "集成测试",
            "单元测试",
            "unit test",
            "integration test",
            "verification plan",
            "verification report",
            "validation plan",
            "validation report",
            "测试报告",
            "测试计划",
            "测试方案",
            "测试",
            "验证",
            "确认",
            "V&V",
            "IQ",
            "OQ",
            "PQ",
        ):
            scores["qa"] += 3
        if _hit(
            "traceability",
            "追溯",
            "rtm",
            "可追溯性",
            "traceability analysis",
            "追溯矩阵",
            "追溯分析",
        ):
            scores["qa"] += 2
        if _hit(
            "risk",
            "ras",
            "rmp",
            "rmr",
            "风险分析",
            "风险管理",
            "risk analysis",
            "risk management",
            "风险评估",
            "风险控制",
            "风险报告",
            "风险",
            "hazard",
            "fmea",
            "fta",
        ):
            scores["rm"] += 3
            if high_risk_reg:
                scores["rm"] += 1
        if _hit(
            "urs",
            "用户需求",
            "product requirement",
            "产品需求",
            "市场需求",
            "prd",
            "mrd",
            "需求",
            "user needs",
            "user requirement",
        ):
            scores["pm"] += 3
        if _hit(
            "srs",
            "软件需求规范",
            "requirement specification",
            "软件需求说明书",
            "软件需求",
            "software requirement",
            "software requirements",
        ):
            scores["rdm"] += 3
        if _hit(
            "architecture",
            "ads",
            "架构",
            "详细设计",
            "概要设计",
            "design specification",
            "sdd",
            "网络安全",
            "cybersecurity",
            "cyber security",
            "设计说明",
            "设计规范",
            "软件设计",
            "设计",
        ):
            scores["rdm"] += 2
        if _hit("software description", "软件描述", "软件研究"):
            scores["rdm"] += 2
        if _hit("audit", "审计", "日志", "权限", "access control", "编码规范"):
            scores["rdm"] += 1
        if _hit(
            "instruction",
            "ifu",
            "说明书",
            "使用说明",
            "udn",
            "user manual",
            "用户手册",
            "instructions for use",
            "产品技术要求",
            "注册申报",
            "注册申请",
            "注册自检",
            "技术审评",
            "临床评价",
        ):
            scores["ra"] += 2
        if _hit("label", "标签", "包装标识"):
            scores["ra"] += 1
        if _hit(
            "milestone",
            "计划",
            "project plan",
            "schedule",
            "开发计划",
            "项目计划",
            "进度计划",
            "立项",
            "里程碑",
        ):
            scores["pjm"] += 2
        if _hit(
            "config",
            "配置管理",
            "release",
            "baseline",
            "configuration",
            "version control",
            "版本控制",
            "变更管理",
            "变更控制",
            "配置项",
            "配置",
            "scm",
            "cm plan",
        ):
            scores["cm"] += 3
        if _hit(
            "interface",
            "界面",
            " ui",
            "usability",
            "可用性",
            "交互",
            "user experience",
            "用户体验",
            "ux",
        ):
            scores["ui"] += 2
        if _hit(
            "生产",
            "production",
            "manufacturing",
            "制造",
            "生产工艺",
            "生产放行",
            "工艺规程",
            "bom",
        ):
            scores["prod"] += 2
        if _hit("预期用途", "适应症", "intended use", "产品特性", "产品定义"):
            scores["pm"] += 1

    if max(scores.values()) == 0:
        if pf and any(x in pf for x in ("软件", "APP", "Web", "PC", "独立")):
            return DRAFT_AUTHOR_ROLE_KEYS[_idx("rdm")]
        return DRAFT_AUTHOR_ROLE_KEYS[0]

    tie_break = ["qa", "rm", "rdm", "ra", "pm", "pjm", "ui", "cm", "prod", ""]
    best = max(scores.values())
    for k in tie_break:
        if scores.get(k, 0) == best:
            return DRAFT_AUTHOR_ROLE_KEYS[_idx(k)]
    return DRAFT_AUTHOR_ROLE_KEYS[0]


def infer_draft_author_role_idx(
    file_names: list,
    *,
    registration_type: str = "",
    project_form: str = "",
) -> int:
    """返回 author_role 下拉索引（供 Streamlit ``draft_author_role_idx`` 使用）。"""
    key = infer_draft_author_role_key(
        file_names,
        registration_type=registration_type,
        project_form=project_form,
    )
    try:
        return DRAFT_AUTHOR_ROLE_KEYS.index(key)
    except ValueError:
        return 0


def format_project_option_label(p: dict[str, Any]) -> str:
    try:
        nm = str((p or {}).get("name") or "").strip() or "未命名"
    except Exception:
        nm = "未命名"
    try:
        pid = int((p or {}).get("id") or 0)
    except (TypeError, ValueError):
        pid = 0
    pc = str((p or {}).get("project_code") or "").strip()
    suf = f" · {pc}" if pc else ""
    head = f"{nm} (ID:{pid}){suf}"
    prod = str((p or {}).get("product_name") or "").strip() or str(
        (p or {}).get("product_name_en") or ""
    ).strip()
    rcn = str((p or {}).get("registration_country") or "").strip()
    rce = str((p or {}).get("registration_country_en") or "").strip()
    if rcn and rce and rcn != rce:
        cshow = f"{rcn} / {rce}"
    elif rcn:
        cshow = rcn
    else:
        cshow = rce
    reg_type = str((p or {}).get("registration_type") or "").strip()
    extras: list[str] = []
    if prod:
        extras.append(f"产品:{prod}")
    if cshow:
        extras.append(f"国家:{cshow}")
    if reg_type:
        extras.append(f"类别:{reg_type}")
    if not extras:
        return head
    return f"{head} | " + " | ".join(extras)


def format_case_option_label(c: dict[str, Any]) -> str:
    name = (str(c.get("case_name") or "").strip()) or "—"
    product = (str(c.get("product_name") or "").strip()) or "—"
    country = (str(c.get("registration_country") or "").strip()) or "—"
    lang_val = str(c.get("document_language") or "").strip()
    lang_label = DOC_LANG_VALUE_TO_LABEL.get(lang_val, lang_val or "—")
    return f"{name}（{product} · {country} · {lang_label}）"


def _integration_collection_rows(collection_ids: Optional[List[str]] = None) -> List[Dict[str, str]]:
    ids = [x.strip() for x in (collection_ids or ["regulations"]) if str(x).strip()]
    if not ids:
        ids = ["regulations"]
    rows: List[Dict[str, str]] = []
    for cid in ids:
        if cid == "regulations":
            lab = "法规/通用知识库（regulations）"
        else:
            lab = f"知识库「{cid}」"
        rows.append({"id": cid, "label": lab})
    return rows


def project_draft_defaults(project_id: int) -> dict[str, Any]:
    """选中 aicheckword 项目时填入初稿/审核 payload 的字段子集。"""
    proj = get_project(int(project_id)) or {}
    row = project_row_for_integration(proj)
    doc_lang = str(row.get("document_language") or "").strip().lower()
    return {
        **row,
        "document_language_value": doc_lang,
        "document_language_label": DOC_LANG_VALUE_TO_LABEL.get(doc_lang, doc_lang or "不指定"),
    }


def build_draft_page_bootstrap(
    collection: str,
    *,
    base_case_id: Optional[int] = None,
    template_file_names: Optional[List[str]] = None,
    collection_ids: Optional[List[str]] = None,
) -> dict[str, Any]:
    """供 ``GET /api/integration/draft/page-bootstrap`` 与 ``/meta`` 扩展字段使用。"""
    coll = (collection or "regulations").strip() or "regulations"
    projects_raw = list_projects(coll) or []
    cases_raw = list_project_cases(coll) or []
    templates_raw: List[str] = []
    bcid = int(base_case_id or 0)
    if bcid > 0:
        templates_raw = list(
            get_project_case_file_names(coll, bcid) or []
        )

    tpl_for_infer = list(template_file_names or templates_raw)
    sel_case: Optional[dict[str, Any]] = None
    if bcid > 0:
        for c in cases_raw:
            if isinstance(c, dict) and int(c.get("id") or 0) == bcid:
                sel_case = c
                break

    rt = str((sel_case or {}).get("registration_type") or "")
    pf = str((sel_case or {}).get("project_form") or "")
    suggested_author = infer_draft_author_role_key(
        tpl_for_infer,
        registration_type=rt,
        project_form=pf,
    )

    project_rows: List[Dict[str, Any]] = []
    for p in projects_raw:
        if not isinstance(p, dict):
            continue
        try:
            pid = int(p.get("id") or 0)
        except (TypeError, ValueError):
            continue
        if pid <= 0:
            continue
        project_rows.append(
            {
                "id": pid,
                "label": format_project_option_label(p),
                "name": str(p.get("name") or "").strip(),
                "productName": str(p.get("product_name") or "").strip(),
                "productNameEn": str(p.get("product_name_en") or "").strip(),
                "registrationCountry": str(p.get("registration_country") or "").strip(),
                "registrationCountryEn": str(
                    p.get("registration_country_en") or ""
                ).strip(),
                "registrationType": str(p.get("registration_type") or "").strip(),
            }
        )

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
        case_rows.append(
            {
                "id": cid,
                "label": f"ID:{cid} | {format_case_option_label(c)}",
                "productName": str(c.get("product_name") or "").strip(),
                "productNameEn": str(c.get("product_name_en") or "").strip(),
                "registrationCountry": str(c.get("registration_country") or "").strip(),
                "registrationCountryEn": str(
                    c.get("registration_country_en") or ""
                ).strip(),
                "documentLanguage": str(c.get("document_language") or "").strip().lower(),
                "registrationType": str(c.get("registration_type") or "").strip(),
                "projectForm": str(c.get("project_form") or "").strip(),
                "caseName": str(c.get("case_name") or "").strip(),
                "caseNameEn": str(c.get("case_name_en") or "").strip(),
            }
        )

    template_rows = [
        {"id": str(name or "").strip(), "label": str(name or "").strip()}
        for name in templates_raw
        if str(name or "").strip()
    ]

    return {
        "collection": coll,
        "collections": _integration_collection_rows(collection_ids),
        "documentLanguages": [
            {"value": k, "label": DOC_LANG_VALUE_TO_LABEL[k]} for k in DOC_LANG_ORDER
        ],
        "draftStrategies": [
            {"value": x["value"], "label": x["label"]} for x in DRAFT_STRATEGY_OPTIONS
        ],
        "authorRoles": [
            {"value": DRAFT_AUTHOR_ROLE_KEYS[i], "label": DRAFT_AUTHOR_ROLE_LABELS[i]}
            for i in range(
                min(len(DRAFT_AUTHOR_ROLE_KEYS), len(DRAFT_AUTHOR_ROLE_LABELS))
            )
        ],
        "suggestedAuthorRole": suggested_author,
        "booleanOptions": [
            {
                "id": "inplace_patch",
                "label": "就地修改（保留基础文件格式，推荐用于注册递交版式）",
                "default": True,
            },
            {
                "id": "save_as_case",
                "label": "将本次生成结果写入案例库（project_cases）",
                "default": True,
            },
            {
                "id": "multi_base_auto_route",
                "label": "多份基础/多份参考时由 AI 自动分配（推荐：自动匹配改哪几份 Base、参考内容如何拆分）",
                "default": True,
            },
            {
                "id": "docx_track_changes",
                "label": "就地修改导出 Word 时使用修订标记（插入/删除，便于在 Word 中审阅修订）",
                "default": True,
            },
        ],
        "templateScopeModes": [
            {"value": "selected", "label": "仅生成下方所选模板文件（可多选）"},
            {
                "value": "all",
                "label": "生成该案例下全部模板文件（与 aicheckword「需显式确认全选」等效）",
            },
        ],
        "projectModes": [
            {"value": "existing", "label": "使用已有项目（不新建）"},
            {"value": "new", "label": "新建项目"},
        ],
        "projects": project_rows,
        "cases": case_rows,
        "templates": template_rows,
        "projectsRaw": projects_raw,
        "casesRaw": cases_raw,
        "templateFileNames": templates_raw,
    }
