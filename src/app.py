"""Streamlit Web UI：注册文档审核工具"""

import sys
import os
from pathlib import Path

# 保证项目根在 path 中（运行 streamlit run src/app.py 时需能 import config 与 src）
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# 尽早设置：避免 tiktoken 下载 cl100k_base 时走代理导致 ProxyError/SSLEOFError（须在任何可能触发 tiktoken 的 import 之前）
_tiktoken_host = "openaipublic.blob.core.windows.net"
for _np in ("NO_PROXY", "no_proxy"):
    _v = os.environ.get(_np, "")
    if _tiktoken_host not in _v:
        os.environ[_np] = f"{_v},{_tiktoken_host}".lstrip(",")

import json
import copy
import gc
import pickle
import html
import time
import tempfile
import zipfile
from datetime import datetime
from typing import Optional, Tuple
import uuid
import shutil
import traceback
import inspect
from difflib import SequenceMatcher
import warnings
from pathlib import Path

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["POSTHOG_DISABLED"] = "true"
warnings.filterwarnings("ignore", category=DeprecationWarning)
import logging
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb").setLevel(logging.WARNING)

import streamlit as st
import pandas as pd

from src.core.cursor_skills_rules_updater import apply_patch_updates

def _st_button_compat(label: str, *, key: str, **kwargs) -> bool:
    """兼容旧版 Streamlit：无 use_container_width 等参数时自动降级。"""
    safe = {k: v for k, v in kwargs.items() if k in ("help", "disabled", "on_click", "args", "kwargs")}
    try:
        return st.button(label, key=key, use_container_width=True, **safe)
    except TypeError:
        try:
            return st.button(label, key=key, **safe)
        except TypeError:
            return st.button(label, key=key)


def _st_radio_compat(
    label: str,
    options,
    *,
    key: str,
    index: Optional[int] = None,
    horizontal: bool = True,
    **kwargs,
):
    """兼容旧版 Streamlit：无 radio(horizontal=...) 时退回纵向排列。"""
    kw = {"key": key, **kwargs}
    if index is not None:
        kw["index"] = index
    if horizontal:
        try:
            return st.radio(label, options, horizontal=True, **kw)
        except TypeError:
            pass
    return st.radio(label, options, **kw)


def _st_main_page_nav(labels: list, *, state_key: str, title: str = "选择功能"):
    """顶部功能入口：有 horizontal radio 时用横向单选；极旧版（如 Streamlit 1.9）用两行按钮 + 状态变量。"""
    if not labels:
        raise ValueError("_st_main_page_nav: empty labels")
    if state_key not in st.session_state or st.session_state[state_key] not in labels:
        st.session_state[state_key] = labels[0]
    try:
        return st.radio(title, labels, horizontal=True, key=state_key)
    except TypeError:
        pass
    st.markdown(f"**{title}**")
    half = (len(labels) + 1) // 2
    row_a, row_b = labels[:half], labels[half:]
    ra = st.columns(len(row_a))
    for i, lab in enumerate(row_a):
        with ra[i]:
            if _st_button_compat(lab, key=f"{state_key}_rowa_{i}"):
                st.session_state[state_key] = lab
    if row_b:
        rb = st.columns(len(row_b))
        for j, lab in enumerate(row_b):
            with rb[j]:
                if _st_button_compat(lab, key=f"{state_key}_rowb_{j}"):
                    st.session_state[state_key] = lab
    cur = st.session_state[state_key]
    st.caption(f"当前：**{cur}**")
    return cur


def _st_run_tabs_or_pick(labels: list, *, radio_label: str, session_key: str, tab_bodies: list):
    """有 st.tabs 时用原生标签页；否则用横向按钮行模拟（比纵向 radio 更接近「横向 Tab」）。

    说明：Streamlit 1.9 等极旧版本无 st.tabs、无 radio(horizontal=)，原先会退化为纵向列表。
    """
    if len(labels) != len(tab_bodies):
        raise ValueError("_st_run_tabs_or_pick: labels and tab_bodies length mismatch")
    tabs_fn = getattr(st, "tabs", None)
    if callable(tabs_fn):
        widgets = tabs_fn(labels)
        for i, w in enumerate(widgets):
            with w:
                tab_bodies[i]()
        return
    idx_key = f"{session_key}_poly_tab_idx"
    if idx_key not in st.session_state:
        st.session_state[idx_key] = 0
    n = len(labels)
    st.caption(radio_label)
    btn_cols = st.columns(n)
    for i, lab in enumerate(labels):
        with btn_cols[i]:
            if _st_button_compat(lab, key=f"{session_key}_poly_tb_{i}"):
                st.session_state[idx_key] = i
    idx = int(st.session_state.get(idx_key) or 0)
    idx = max(0, min(idx, n - 1))
    st.session_state[idx_key] = idx
    tab_bodies[idx]()


from config import settings
from config.settings import (
    get_pdf_ocr_llm_model,
    get_provider_sidebar_slot,
    maybe_seed_provider_sidebar_presets_from_legacy,
    openai_form_base_url_default_from_settings,
    pdf_ocr_llm_model_field_available,
    repair_provider_sidebar_presets_urls,
    sanitize_openai_form_base_url,
    set_pdf_ocr_llm_model,
    upsert_provider_sidebar_slot,
)
from src.core.agent import ReviewAgent
from src.core.document_draft_generator import DocumentDraftGenerator
from src.core.display_filename import sanitize_audit_report_dict
from src.core.document_loader import (
    load_single_file,
    split_documents,
    LOADER_MAP,
    is_archive,
    extract_archive,
    is_deprecated_path,
    extract_section_outline_from_texts,
)
from src.core.langchain_compat import Document as LCDocument
from src.core.knowledge_base import _add_batch_with_retry, annotate_main_knowledge_documents
from src.core.audit_perf import audit_perf_enabled, audit_perf_log, audit_perf_time_block
from src.core.db import (
    load_app_settings,
    save_app_settings,
    add_operation_log,
    get_current_model_info,
    get_operation_logs,
    get_operation_summary,
    save_audit_report,
    get_audit_reports,
    get_audit_report_by_id,
    get_audit_reports_by_file_name,
    get_audit_report_file_names,
    update_audit_report,
    invalidate_audit_reports_list_cache,
    save_audit_correction,
    get_knowledge_stats,
    get_checkpoint_stats,
    get_knowledge_stats_by_category,
    clear_knowledge_docs,
    append_knowledge_docs,
    save_audit_checklist,
    get_audit_checklists,
    get_audit_checklist_by_id,
    update_audit_checklist,
    delete_audit_checklist,
    get_dimension_options,
    save_dimension_options,
    list_projects,
    get_project,
    create_project,
    update_project,
    delete_project,
    get_project_knowledge_stats,
    clear_project_knowledge_docs,
    get_existing_file_names,
    get_existing_checkpoint_file_names,
    get_existing_project_file_names,
    list_ocr_cache_file_names,
    create_project_case,
    list_project_cases,
    get_project_case_file_names,
    delete_project_case,
    get_project_case,
    update_project_case,
    update_knowledge_docs_case_id,
    get_knowledge_docs_by_case_id,
    upsert_draft_file_skills_rules,
    get_draft_file_skills_rules,
    delete_draft_file_skills_rules,
    get_review_extra_instructions,
    update_review_extra_instructions,
    get_review_system_prompt,
    get_review_user_prompt,
    update_review_prompts,
    get_prompt_by_key,
    update_prompt_by_key,
    get_translation_config,
    save_translation_config,
    PROMPT_KEYS,
    REGISTRATION_TYPES,
    REGISTRATION_TYPE_OPTIONS,
    REGISTRATION_COMPONENTS,
    OP_TYPE_TRAIN_BATCH,
    OP_TYPE_TRAIN,
    OP_TYPE_REVIEW_BATCH,
    OP_TYPE_REVIEW,
    OP_TYPE_REVIEW_TEXT,
    OP_TYPE_CORRECTION,
    OP_TYPE_GENERATE_CHECKLIST,
    OP_TYPE_TRAIN_CHECKLIST,
    OP_TYPE_TRAIN_PROJECT,
    OP_TYPE_TRAIN_PROJECT_ERROR,
    OP_TYPE_TRANSLATION,
    OP_TYPE_TRANSLATION_ERROR,
)
from src.core.audit_handoff import (
    build_immediate_audit_point_records,
    build_immediate_audit_remediation_by_target,
)
from src.core.report_export import (
    report_to_html,
    report_to_docx,
    report_to_pdf,
    report_to_excel,
    report_to_docx_with_comments,
    report_todo_to_csv,
    report_todo_to_pdf,
    report_todo_to_docx,
    report_todo_to_excel,
)
from src.core.draft_export import export_like_base
from src.streamlit_compat import streamlit_divider, streamlit_rerun
from src.system_config_ui import render_system_config_page


def _resolve_draft_artifact_path(raw_path: str) -> Path:
    """把历史记录里保存的产物路径尽量解析为当前机器可读的绝对路径。

    典型问题：
    - 记录里是绝对路径，但服务迁机/换盘符后原路径不存在；
    - 记录里误存了相对路径（相对项目根或相对 uploads 根）。
    """
    s = (raw_path or "").strip()
    if not s:
        return Path("")
    p = Path(s)
    # 1) 原样存在：直接返回（尽量 resolve，便于后续比较）
    try:
        if p.is_file():
            return p
    except Exception:
        pass

    # 2) 相对 uploads_path（常见：把 uploads 下的子路径入库）
    try:
        up = settings.uploads_path
        cand = (up / p).resolve()
        if cand.is_file():
            return cand
    except Exception:
        pass

    # 3) 相对项目根（常见：误把相对仓库路径写入日志）
    try:
        cand2 = (_root / p).resolve()
        if cand2.is_file():
            return cand2
    except Exception:
        pass

    # 4) 仅按文件名在 draft_outputs 下兜底搜索（最后手段：避免历史记录完全不可用）
    try:
        name = Path(s).name
        if name:
            drafts_dir = (settings.uploads_path / "draft_outputs").resolve()
            if drafts_dir.is_dir():
                hits = list(drafts_dir.rglob(name))
                # 取最近修改的一个（同名文件可能多次生成）
                hits = [h for h in hits if h.is_file()]
                if hits:
                    hits.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    return hits[0]
    except Exception:
        pass

    return p


def _read_download_bytes(resolved: Path) -> Tuple[Optional[bytes], str]:
    """读取下载字节；失败返回 (None, 错误说明)。"""
    try:
        if not resolved or not str(resolved).strip():
            return None, "路径为空"
        if not resolved.is_file():
            return None, f"文件不存在或不可访问：{resolved}"
        data = resolved.read_bytes()
        if not data:
            return None, f"文件为空：{resolved}"
        return data, ""
    except Exception as e:
        return None, f"读取失败：{type(e).__name__}: {e}"


def _draft_download_button(*, label: str, raw_path: str, mime: str, key: str) -> None:
    """历史/结果区统一下载：避免 Streamlit /media/*.bin 404（常见原因：路径失效或读取失败仍渲染按钮）。"""
    rp = _resolve_draft_artifact_path(raw_path)
    # Streamlit 的 /media 下载是临时资源：同一 widget key 在 rerun 后可能仍被浏览器复用旧 URL。
    # 用“文件签名（mtime+size）”拼进 key：文件不变时 key 稳定；文件更新/重导出后 key 自动变化，触发重新注册。
    try:
        if rp.is_file():
            stt = rp.stat()
            sig = f"{int(getattr(stt, 'st_mtime_ns', int(stt.st_mtime * 1e9)))}_{int(stt.st_size)}"
        else:
            sig = uuid.uuid4().hex[:10]
    except Exception:
        sig = uuid.uuid4().hex[:10]
    _key = f"{key}__{sig}"
    data, err = _read_download_bytes(rp)
    if data is None:
        st.caption(f"下载不可用：{err}")
        try:
            st.caption(f"原始记录路径：{raw_path}")
        except Exception:
            pass
        return
    st.download_button(
        label,
        data=data,
        file_name=rp.name,
        mime=mime,
        key=_key,
    )


def _make_ttl_cache(ttl_sec=5):
    """简易 TTL 缓存：将数据缓存在 st.session_state 中，避免每次前端交互都重复走 DB 或初始化。

    说明：
    - key 由函数名 + 参数组成，ttl 到期后自动失效；
    - 相比简单闭包缓存，本实现在 Streamlit 多次 rerun 下也能复用缓存，适合「切换输入框/Tab」等纯前端操作场景。
    """

    def decorator(func):
        cache_key = f"_ttl_cache_{func.__name__}"

        def wrapper(*args, **kwargs):
            store = st.session_state.setdefault(cache_key, {})
            key = (args, tuple(sorted((k, v) for k, v in kwargs.items())))
            now = time.time()
            if key in store:
                val, expiry = store[key]
                if now < expiry:
                    return val
            val = func(*args, **kwargs)
            store[key] = (val, now + ttl_sec)
            return val

        def _clear():
            if cache_key in st.session_state:
                st.session_state.pop(cache_key, None)

        wrapper.clear = _clear
        return wrapper

    return decorator


# 侧栏统计缓存，减少切换功能时的 DB 与初始化开销（ttl 秒后失效）
@_make_ttl_cache(ttl_sec=5)
def _cached_knowledge_stats(collection: str):
    return get_knowledge_stats(collection)


@_make_ttl_cache(ttl_sec=5)
def _cached_checkpoint_stats(collection: str):
    return get_checkpoint_stats(collection)


@_make_ttl_cache(ttl_sec=8)
def _cached_audit_feedback_stats(collection: str):
    """误报/纠正独立库 `{collection}_audit_feedback` 块数（与第二步清单库区分）。"""
    return get_knowledge_stats(f"{collection}_audit_feedback")


@_make_ttl_cache(ttl_sec=5)
def _cached_knowledge_stats_by_category(collection: str):
    return get_knowledge_stats_by_category(collection)


@_make_ttl_cache(ttl_sec=8)
def _cached_operation_logs(op_type, collection_filter, limit: int):
    return get_operation_logs(op_type=op_type, collection=collection_filter, limit=limit)


def _invalidate_operation_logs_cache() -> None:
    """写入 operation_logs 后调用，避免「操作记录」页 TTL 缓存看不到刚写入的行。"""
    try:
        _cached_operation_logs.clear()
    except Exception:
        pass


@_make_ttl_cache(ttl_sec=12)
def _cached_list_projects(collection: str):
    return list_projects(collection)


@_make_ttl_cache(ttl_sec=12)
def _cached_list_project_cases(collection: str):
    return list_project_cases(collection)


@_make_ttl_cache(ttl_sec=30)
def _cached_dimension_options():
    return get_dimension_options()


# 注册国家中英文对应，用于维度匹配时中/英文均可匹配
_COUNTRY_CN_TO_EN = {"中国": "China", "美国": "USA", "欧盟": "EU", "欧洲": "Europe"}

# 文档语言选项：全系统统一为四选一，保证逻辑闭环
# 使用处：① 第一步-项目案例(上传/目录)案例文档语言 ② 第一步-生成审核点 ③ 第二步-导入审核点 ④ 第三步-待审文档语言/待审文本语言
# 取值：不指定→""、中文版→zh、英文版→en、中英文→both；存库/上下文均用取值，展示用 DOC_LANG_VALUE_TO_LABEL
DOC_LANG_OPTIONS = ["不指定", "中文版", "英文版", "中英文"]
DOC_LANG_LABEL_TO_VALUE = {"不指定": "", "中文版": "zh", "英文版": "en", "中英文": "both"}
DOC_LANG_VALUE_TO_LABEL = {"": "不指定", "zh": "中文版", "en": "英文版", "both": "中英文"}

# 与「生成初稿」页编写人员身份下拉顺序一致：["", "pm", "pjm", "rm", "rdm", "ui", "qa", "cm", "ra", "prod"]
_DRAFT_AUTHOR_ROLE_KEYS = ["", "pm", "pjm", "rm", "rdm", "ui", "qa", "cm", "ra", "prod"]


def _infer_draft_author_role_idx(
    file_names: list,
    *,
    registration_type: str = "",
    project_form: str = "",
) -> int:
    """根据待生成文件名与模板案例上的注册类别/项目形态，推断「编写人员身份」下拉索引。"""
    scores = {k: 0 for k in _DRAFT_AUTHOR_ROLE_KEYS}

    def _idx(k: str) -> int:
        try:
            return _DRAFT_AUTHOR_ROLE_KEYS.index(k)
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

        # 测试/验证/确认 → 测试工程师（qa）
        if _hit(
            "测试用例", "test case", "test execution", "system test", "system testing",
            "确认测试", "集成测试", "单元测试", "unit test", "integration test",
            "verification plan", "verification report", "validation plan", "validation report",
            "测试报告", "测试计划", "测试方案", "测试", "验证", "确认", "V&V", "IQ", "OQ", "PQ",
        ):
            scores["qa"] += 3
        if _hit("traceability", "追溯", "rtm", "可追溯性", "traceability analysis", "追溯矩阵", "追溯分析"):
            scores["qa"] += 2
        # 风险 → 风险经理（rm）
        if _hit(
            "risk", "ras", "rmp", "rmr", "风险分析", "风险管理", "risk analysis",
            "risk management", "风险评估", "风险控制", "风险报告", "风险",
            "hazard", "fmea", "fta",
        ):
            scores["rm"] += 3
            if high_risk_reg:
                scores["rm"] += 1
        # 用户需求/产品预期用途 → 产品经理（pm）
        if _hit(
            "urs", "用户需求", "product requirement", "产品需求", "市场需求",
            "prd", "mrd", "需求", "user needs", "user requirement",
        ):
            scores["pm"] += 3
        # 软件需求/设计/架构 → 研发经理（rdm）
        if _hit(
            "srs", "软件需求规范", "requirement specification", "软件需求说明书", "软件需求",
            "software requirement", "software requirements",
        ):
            scores["rdm"] += 3
        if _hit(
            "architecture", "ads", "架构", "详细设计", "概要设计", "design specification", "sdd",
            "网络安全", "cybersecurity", "cyber security", "设计说明", "设计规范", "软件设计",
            "设计",
        ):
            scores["rdm"] += 2
        if _hit("software description", "软件描述", "软件研究"):
            scores["rdm"] += 2
        if _hit("audit", "审计", "日志", "权限", "access control", "编码规范"):
            scores["rdm"] += 1
        # 说明书/标签 → 注册工程师（ra）
        if _hit(
            "instruction", "ifu", "说明书", "使用说明", "udn", "user manual", "用户手册",
            "instructions for use", "产品技术要求", "注册申报", "注册申请", "注册自检",
            "技术审评", "临床评价",
        ):
            scores["ra"] += 2
        if _hit("label", "标签", "包装标识"):
            scores["ra"] += 1
        # 计划/进度 → 项目经理（pjm）
        if _hit(
            "milestone", "计划", "project plan", "schedule", "开发计划",
            "项目计划", "进度计划", "立项", "里程碑",
        ):
            scores["pjm"] += 2
        # 配置管理 → 配置管理员（cm）
        if _hit(
            "config", "配置管理", "release", "baseline", "configuration",
            "version control", "版本控制", "变更管理", "变更控制", "配置项",
            "配置", "scm", "cm plan",
        ):
            scores["cm"] += 3
        # 界面/可用性 → UI 设计师（ui）
        if _hit(
            "interface", "界面", " ui", "usability", "可用性",
            "交互", "user experience", "用户体验", "ux",
        ):
            scores["ui"] += 2
        # 生产/制造 → 生产专员（prod）
        if _hit(
            "生产", "production", "manufacturing", "制造", "生产工艺",
            "生产放行", "工艺规程", "bom",
        ):
            scores["prod"] += 2
        if _hit("预期用途", "适应症", "intended use", "产品特性", "产品定义"):
            scores["pm"] += 1

    if max(scores.values()) == 0:
        if pf and any(x in pf for x in ("软件", "APP", "Web", "PC", "独立")):
            return _idx("rdm")
        return 0

    tie_break = ["qa", "rm", "rdm", "ra", "pm", "pjm", "ui", "cm", "prod", ""]
    best = max(scores.values())
    for k in tie_break:
        if scores.get(k, 0) == best:
            return _idx(k)
    return 0


def _format_case_option(c: dict) -> str:
    """项目案例下拉项显示：案例名（产品 · 国家 · 文档语言），便于区分同名案例。"""
    name = (c.get("case_name") or "").strip() or "—"
    product = (c.get("product_name") or "").strip() or "—"
    country = (c.get("registration_country") or "").strip() or "—"
    lang_val = (c.get("document_language") or "").strip()
    lang_label = DOC_LANG_VALUE_TO_LABEL.get(lang_val, lang_val or "—")
    return f"{name}（{product} · {country} · {lang_label}）"


def _match_project_case_for_review(
    collection: str,
    doc_text: str,
    dimension_filters: dict,
) -> Optional[dict]:
    """先按产品名称+适用范围匹配过往项目案例，否则按维度匹配。支持中文或英文匹配。"""
    cases = _cached_list_project_cases(collection)
    if not cases:
        return None
    doc_text = (doc_text or "").strip()
    # 1) 优先：产品名称（中文或英文）在文档中出现
    for c in cases:
        product_name = (c.get("product_name") or "").strip()
        product_name_en = (c.get("product_name_en") or "").strip()
        scope = (c.get("scope_of_application") or "").strip()
        name_match = (product_name and product_name in doc_text) or (product_name_en and product_name_en in doc_text)
        if name_match:
            if not scope or scope in doc_text:
                return c
            if scope and len(scope) > 50:
                for part in (scope[:80], scope[50:130]):
                    if part.strip() and part.strip() in doc_text:
                        return c
            return c
    # 2) 按维度匹配（注册国家：选中中文则同时匹配该国家的英文，案例的中文或英文任一侧匹配即可）
    sel_countries = dimension_filters.get("registration_country") or []
    sel_types = dimension_filters.get("registration_type") or []
    sel_components = dimension_filters.get("registration_component") or []
    sel_forms = dimension_filters.get("project_form") or []
    if not (sel_countries or sel_types or sel_components or sel_forms):
        return None
    match_countries = set(sel_countries) | {_COUNTRY_CN_TO_EN.get(x, x) for x in sel_countries if x}
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


def _provider_ready() -> bool:
    """当前 provider 是否就绪"""
    p = (settings.provider or "").strip().lower()
    if p == "ollama":
        return True
    if p == "cursor":
        return bool(settings.cursor_api_key and settings.cursor_repository)
    if p == "openai":
        return bool(settings.openai_api_key)
    if p == "deepseek":
        return bool(settings.deepseek_api_key or settings.openai_api_key)
    if p == "lingyi":
        return bool(settings.lingyi_api_key or settings.openai_api_key)
    if p == "gemini":
        return bool(settings.gemini_api_key or settings.google_api_key)
    if p == "tongyi":
        return bool(settings.dashscope_api_key)
    if p == "baidu":
        return bool(settings.qianfan_ak and settings.qianfan_sk)
    return bool(settings.openai_api_key)


def _require_provider() -> bool:
    """检查 AI 服务是否就绪"""
    if _provider_ready():
        return True
    p = (settings.provider or "").strip().lower()
    if p == "ollama":
        st.warning("⚠️ Ollama 服务未启动，请确保已安装并运行 Ollama（ollama serve）。")
    elif p == "cursor":
        st.warning("⚠️ Cursor 模式下请填写 API Key 和 GitHub 仓库地址（Cursor Dashboard → Integrations）。")
    elif p == "gemini":
        st.warning("⚠️ Gemini 模式下请在 .env 中配置 GEMINI_API_KEY 或 GOOGLE_API_KEY。")
    elif p == "tongyi":
        st.warning("⚠️ 通义模式下请配置 DASHSCOPE_API_KEY。")
    elif p == "baidu":
        st.warning("⚠️ 文心模式下请配置 QIANFAN_AK 与 QIANFAN_SK。")
    elif p in ("deepseek", "lingyi"):
        st.warning("⚠️ 请填写 API Key；DeepSeek / 零一为 OpenAI 兼容接口，可同时填写 Base URL。")
    else:
        st.warning("⚠️ 请先在左侧边栏填写 OpenAI API Key 或选择其他已配置的提供方。")
    return False


def init_agent():
    """初始化或获取 Agent（不触发 OpenAI 连接）"""
    collection = st.session_state.get("collection_name", "regulations")
    if "agent" not in st.session_state or st.session_state.get("_col") != collection:
        st.session_state.agent = ReviewAgent(collection)
        st.session_state._col = collection
    return st.session_state.agent


def _default_llm_for_provider(p: str) -> str:
    return {
        "ollama": "qwen2.5",
        "openai": "gpt-4o-mini",
        "deepseek": "deepseek-chat",
        "lingyi": "yi-large",
        "gemini": "gemini-1.5-flash",
        "tongyi": "qwen-plus",
        "baidu": "ERNIE-Bot-4",
        "cursor": "gpt-4o-mini",
    }.get((p or "").strip().lower(), "qwen2.5")


def _default_embed_for_provider(p: str) -> str:
    return "text-embedding-3-small" if (p or "").strip().lower() == "openai" else "nomic-embed-text"


def _default_pdf_ocr_for_provider(p: str) -> str:
    return "qwen-vl-plus" if (p or "").strip().lower() in ("tongyi", "deepseek") else ""


def _sidebar_repair_openai_form_base_url_session(provider: str) -> None:
    """同 tab 下 session 里曾缓存错误 Base URL 时纠正（切换逻辑已跳过 hydrate 的情况）。"""
    if provider not in ("openai", "deepseek", "lingyi"):
        return
    bk = f"sidebar_base_url_{provider}_v2"
    cur = (st.session_state.get(bk) or "").strip()
    cand = cur or openai_form_base_url_default_from_settings(provider)
    fixed = sanitize_openai_form_base_url(provider, cand)
    if fixed != cur:
        st.session_state[bk] = fixed
    if provider == "deepseek":
        if (settings.deepseek_base_url or "").strip() != fixed:
            settings.deepseek_base_url = fixed
    elif provider == "lingyi":
        if (settings.lingyi_base_url or "").strip() != fixed:
            settings.lingyi_base_url = fixed
    else:
        if (settings.openai_base_url or "").strip() != fixed:
            settings.openai_base_url = fixed


def _sidebar_hydrate_on_provider_tab_change(provider: str) -> None:
    """切换侧栏「服务提供方」时：从 provider_sidebar_presets 整槽恢复到 session_state 与 settings。"""
    st.session_state["_sidebar_provider_tab"] = provider
    slot = get_provider_sidebar_slot(provider)
    if provider == "cursor":
        dlm = (slot.get("llm_model") or "").strip() or (settings.llm_model or _default_llm_for_provider("cursor")).strip()
        dem = (slot.get("embedding_model") or "").strip() or (settings.embedding_model or _default_embed_for_provider("cursor")).strip()
        settings.llm_model = dlm
        settings.embedding_model = dem
        if pdf_ocr_llm_model_field_available():
            docr = (slot.get("pdf_ocr_llm_model") or "").strip()
            if not docr:
                docr = (get_pdf_ocr_llm_model() or "").strip()
            if not docr:
                _emb = (settings.cursor_embedding or "ollama").strip().lower()
                docr = _default_pdf_ocr_for_provider("openai") if _emb == "openai" else ""
            st.session_state[f"sidebar_pdf_ocr_llm_model_{provider}_v2"] = docr
            set_pdf_ocr_llm_model(docr)
        return

    dlm = (slot.get("llm_model") or "").strip() or _default_llm_for_provider(provider)
    dem = (slot.get("embedding_model") or "").strip() or _default_embed_for_provider(provider)
    st.session_state[f"sidebar_llm_model_{provider}_v2"] = dlm
    st.session_state[f"sidebar_embed_model_{provider}_v2"] = dem
    settings.llm_model = dlm
    settings.embedding_model = dem

    if pdf_ocr_llm_model_field_available() and provider in ("ollama", "openai", "deepseek", "lingyi", "tongyi"):
        docr = (slot.get("pdf_ocr_llm_model") or "").strip()
        if not docr:
            docr = (get_pdf_ocr_llm_model() or "").strip() or _default_pdf_ocr_for_provider(provider)
        st.session_state[f"sidebar_pdf_ocr_llm_model_{provider}_v2"] = docr
        set_pdf_ocr_llm_model(docr)

    if provider == "ollama":
        ou = (slot.get("ollama_base_url") or "").strip() or (settings.ollama_base_url or "http://localhost:11434").strip()
        st.session_state[f"sidebar_ollama_base_url_{provider}_v2"] = ou
        settings.ollama_base_url = ou

    if provider in ("openai", "deepseek", "lingyi"):
        bu_raw = (slot.get("base_url") or "").strip() or openai_form_base_url_default_from_settings(provider)
        bu = sanitize_openai_form_base_url(provider, bu_raw)
        st.session_state[f"sidebar_base_url_{provider}_v2"] = bu
        if provider == "deepseek":
            settings.deepseek_base_url = bu
        elif provider == "lingyi":
            settings.lingyi_base_url = bu
        else:
            settings.openai_base_url = bu


def _sidebar_fill_missing_widget_keys(provider: str) -> None:
    """同提供方下热重载等导致 session 丢键时，用预设补全，不覆盖已有输入。"""
    if provider == "cursor":
        if pdf_ocr_llm_model_field_available():
            ok = f"sidebar_pdf_ocr_llm_model_{provider}_v2"
            if ok not in st.session_state:
                slot_c = get_provider_sidebar_slot(provider)
                docr = (slot_c.get("pdf_ocr_llm_model") or "").strip() or (get_pdf_ocr_llm_model() or "").strip()
                if not docr:
                    _emb = (settings.cursor_embedding or "ollama").strip().lower()
                    docr = _default_pdf_ocr_for_provider("openai") if _emb == "openai" else ""
                st.session_state[ok] = docr
        return
    slot = get_provider_sidebar_slot(provider)
    lk = f"sidebar_llm_model_{provider}_v2"
    if lk not in st.session_state:
        st.session_state[lk] = (slot.get("llm_model") or "").strip() or _default_llm_for_provider(provider)
    ek = f"sidebar_embed_model_{provider}_v2"
    if ek not in st.session_state:
        st.session_state[ek] = (slot.get("embedding_model") or "").strip() or _default_embed_for_provider(provider)
    if pdf_ocr_llm_model_field_available() and provider in ("ollama", "openai", "deepseek", "lingyi", "tongyi"):
        ok = f"sidebar_pdf_ocr_llm_model_{provider}_v2"
        if ok not in st.session_state:
            docr = (slot.get("pdf_ocr_llm_model") or "").strip() or (get_pdf_ocr_llm_model() or "").strip() or _default_pdf_ocr_for_provider(provider)
            st.session_state[ok] = docr
    if provider == "ollama":
        bk = f"sidebar_ollama_base_url_{provider}_v2"
        if bk not in st.session_state:
            st.session_state[bk] = (slot.get("ollama_base_url") or "").strip() or (settings.ollama_base_url or "http://localhost:11434").strip()
    if provider in ("openai", "deepseek", "lingyi"):
        bk = f"sidebar_base_url_{provider}_v2"
        if bk not in st.session_state:
            _raw = (slot.get("base_url") or "").strip() or openai_form_base_url_default_from_settings(provider)
            st.session_state[bk] = sanitize_openai_form_base_url(provider, _raw)


def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.title("⚙️ 设置")

        # 首次从 DB 载入配置到 settings（仅一次）
        if not st.session_state.get("db_settings_loaded"):
            db_conf = None
            try:
                db_conf = load_app_settings()
            except Exception as _e:
                # DB 连接失败不应阻塞页面加载（常见：MySQL 不可达/超时）
                st.warning(f"⚠️ 数据库连接失败，已使用本地默认配置继续运行：{_e}")
            if db_conf:
                settings.provider = db_conf.get("provider") or settings.provider
                settings.openai_api_key = db_conf.get("openai_api_key") or settings.openai_api_key
                settings.openai_base_url = db_conf.get("openai_base_url") or settings.openai_base_url
                settings.ollama_base_url = db_conf.get("ollama_base_url") or settings.ollama_base_url
                settings.cursor_api_key = db_conf.get("cursor_api_key") or settings.cursor_api_key
                settings.cursor_api_base = db_conf.get("cursor_api_base") or settings.cursor_api_base
                settings.cursor_repository = db_conf.get("cursor_repository") or settings.cursor_repository
                settings.cursor_ref = db_conf.get("cursor_ref") or settings.cursor_ref
                settings.cursor_embedding = db_conf.get("cursor_embedding") or settings.cursor_embedding
                settings.llm_model = db_conf.get("llm_model") or settings.llm_model
                # 库列常为历史空串：勿用 "" 覆盖 pydantic-settings 已从 .env 注入的 PDF_OCR_LLM_MODEL
                if pdf_ocr_llm_model_field_available():
                    _db_ocr = db_conf.get("pdf_ocr_llm_model")
                    if _db_ocr is not None and str(_db_ocr).strip():
                        set_pdf_ocr_llm_model(str(_db_ocr).strip())
                settings.embedding_model = db_conf.get("embedding_model") or settings.embedding_model
                settings.deepseek_api_key = db_conf.get("deepseek_api_key") or settings.deepseek_api_key
                settings.deepseek_base_url = db_conf.get("deepseek_base_url") or settings.deepseek_base_url
                settings.lingyi_api_key = db_conf.get("lingyi_api_key") or settings.lingyi_api_key
                settings.lingyi_base_url = db_conf.get("lingyi_base_url") or settings.lingyi_base_url
                settings.gemini_api_key = db_conf.get("gemini_api_key") or settings.gemini_api_key
                settings.google_api_key = db_conf.get("gemini_api_key") or settings.google_api_key
                settings.dashscope_api_key = db_conf.get("dashscope_api_key") or settings.dashscope_api_key
                settings.qianfan_ak = db_conf.get("qianfan_ak") or settings.qianfan_ak
                settings.qianfan_sk = db_conf.get("qianfan_sk") or settings.qianfan_sk
                v = db_conf.get("cursor_verify_ssl")
                vv = v if isinstance(v, bool) else (v != 0 and v != "0")
                settings.cursor_verify_ssl = vv
                settings.llm_verify_ssl = vv
                from config.cursor_overrides import _cursor_overrides
                _cursor_overrides["verify_ssl"] = vv
                t = db_conf.get("cursor_trust_env")
                tt = t if isinstance(t, bool) else (t != 0 and t != "0")
                settings.cursor_trust_env = tt
                settings.llm_trust_env = tt
                _cursor_overrides["trust_env"] = tt
                # 全量 JSON 最后覆盖兼容列（迁机后主要从库恢复）
                raw_json = db_conf.get("runtime_settings_json")
                if raw_json:
                    try:
                        parsed = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
                        if isinstance(parsed, dict) and parsed:
                            from config.runtime_settings import (
                                apply_runtime_config_dict,
                                sync_cursor_overrides_from_settings,
                            )

                            apply_runtime_config_dict(parsed)
                            sync_cursor_overrides_from_settings()
                    except Exception:
                        pass
            try:
                repair_provider_sidebar_presets_urls()
            except Exception:
                pass
            st.session_state["db_settings_loaded"] = True
            st.session_state["current_provider"] = (settings.provider or "ollama").strip().lower()
            if not st.session_state.get("_presets_seeded_from_legacy"):
                maybe_seed_provider_sidebar_presets_from_legacy()
                st.session_state["_presets_seeded_from_legacy"] = True

        # --- AI 服务配置 ---
        st.subheader("AI 服务")

        # (显示名, provider 值)，按推荐使用优先级排序
        _provider_list = [
            ("DeepSeek (OpenAI 兼容)", "deepseek"),
            ("OpenAI", "openai"),
            ("零一万物 (OpenAI 兼容)", "lingyi"),
            ("Google Gemini", "gemini"),
            ("阿里通义千问", "tongyi"),
            ("百度文心一言", "baidu"),
            ("Ollama (本地免费)", "ollama"),
            ("Cursor Agent (Cloud API)", "cursor"),
        ]
        provider_options = [x[0] for x in _provider_list]
        # 用 current_provider 驱动默认选中，选完后立即写入，下次 rerun 即生效，避免需点两次
        _cur = (st.session_state.get("current_provider") or settings.provider or "ollama").strip().lower()
        current_idx = next((i for i, (_, v) in enumerate(_provider_list) if v == _cur), 0)
        provider_choice = st.selectbox(
            "服务提供方",
            provider_options,
            index=min(current_idx, len(provider_options) - 1),
            help="Gemini/通义/文心 需安装对应依赖；密钥可写在 .env，侧栏保存会写入当前内存。",
        )
        _provider = _provider_list[provider_options.index(provider_choice)][1]
        is_ollama = _provider == "ollama"
        is_cursor = _provider == "cursor"
        is_openai_form = _provider in ("openai", "deepseek", "lingyi")  # 共用 Key + Base URL

        settings.provider = _provider
        st.session_state["current_provider"] = _provider
        _prev_tab = st.session_state.get("_sidebar_provider_tab")
        if _prev_tab != _provider:
            _sidebar_hydrate_on_provider_tab_change(_provider)
        _sidebar_fill_missing_widget_keys(_provider)
        if _provider in ("openai", "deepseek", "lingyi"):
            _sidebar_repair_openai_form_base_url_session(_provider)

        # 所有 AI 服务通用：不校验 SSL、不使用系统代理（代理/证书异常导致 SSLEOFError 时可勾选）
        from config.cursor_overrides import _cursor_overrides, get_llm_verify_ssl, get_llm_trust_env
        llm_no_ssl = st.checkbox(
            "不校验 SSL（所有 AI 服务）",
            value=not get_llm_verify_ssl(),
            key="llm_no_ssl",
            help="代理或证书异常导致 SSLEOFError 时可勾选；对 OpenAI/DeepSeek/零一/Cursor/Ollama 等均生效。",
        )
        _cursor_overrides["verify_ssl"] = not llm_no_ssl
        llm_no_proxy = st.checkbox(
            "不使用系统代理（所有 AI 服务）",
            value=not get_llm_trust_env(),
            key="llm_no_proxy",
            help="直连 API、不走系统代理；代理导致 SSL EOF 时可勾选。",
        )
        _cursor_overrides["trust_env"] = not llm_no_proxy

        # 提供方或通用配置变化时重置 agent 缓存，使下次请求（审核/生成审核点等）使用新配置
        _cfg_key = (_provider, get_llm_verify_ssl(), get_llm_trust_env())
        _last_cfg = st.session_state.get("_llm_config_key")
        if _last_cfg is not None and _last_cfg != _cfg_key and "agent" in st.session_state:
            try:
                st.session_state.agent.reset_clients()
            except Exception:
                pass
        st.session_state["_llm_config_key"] = _cfg_key

        # 不再强制二次 rerun：否则 Streamlit 可能复用旧组件位置，导致未保存时显示错位。

        if is_cursor:
            cursor_api_key = st.text_input(
                "Cursor API Key",
                value=settings.cursor_api_key,
                type="password",
                key=f"sidebar_api_key_{_provider}",
                help="Cursor Dashboard → Integrations 创建",
            )
            cursor_repo = st.text_input(
                "GitHub 仓库地址",
                value=settings.cursor_repository,
                key=f"sidebar_cursor_repo_{_provider}",
                help="必填，如 https://github.com/your-org/your-repo（Agent 会基于该仓库运行）",
            )
            cursor_ref = st.text_input(
                "分支/标签",
                value=settings.cursor_ref,
                key=f"sidebar_cursor_ref_{_provider}",
                help="默认 main",
            )
            cursor_embed = st.selectbox(
                "向量化使用",
                ["ollama", "openai"],
                index=0 if (settings.cursor_embedding or "ollama").lower() == "ollama" else 1,
                key=f"sidebar_cursor_embed_{_provider}",
                help="知识库向量化仍需要 Ollama 或 OpenAI",
            )
            llm_model = settings.llm_model
            embed_model = settings.embedding_model
        elif is_ollama:
            ollama_url = st.text_input(
                "Ollama 地址",
                key=f"sidebar_ollama_base_url_{_provider}_v2",
                help="默认 http://localhost:11434，通常不用改",
            )
            _lm_key = f"sidebar_llm_model_{_provider}_v2"
            llm_model = st.text_input(
                "审核模型",
                key=_lm_key,
                help="推荐 qwen2.5（中文好）、llama3.1、mistral 等",
            )
            embed_model = st.text_input(
                "向量化模型",
                key=f"sidebar_embed_model_{_provider}_v2",
                help="推荐 nomic-embed-text、bge-m3 等",
            )
        elif is_openai_form:
            # 按 provider 使用独立 key，避免切换提供方时 DeepSeek/零一/OpenAI 的 API Key 互相填充
            _api_key_value = settings.deepseek_api_key if _provider == "deepseek" else (settings.lingyi_api_key if _provider == "lingyi" else settings.openai_api_key)
            api_key = st.text_input(
                "API Key",
                value=_api_key_value,
                type="password",
                key=f"sidebar_api_key_{_provider}",
                help="DeepSeek / 零一万物 为 OpenAI 兼容接口",
            )
            base_url = st.text_input(
                "API Base URL",
                key=f"sidebar_base_url_{_provider}_v2",
                help="DeepSeek 填 https://api.deepseek.com/v1（404 时请确认无误）；零一默认 https://api.lingyiwanwu.com/v1",
            )
            _model_help = "DeepSeek 填 deepseek-chat 或 deepseek-chat-v2 等（404 多为模型名或 Base URL 错误）；零一如 yi-large 等"
            _lm_key = f"sidebar_llm_model_{_provider}_v2"
            llm_model = st.text_input(
                "审核模型",
                key=_lm_key,
                help=_model_help,
            )
            _embed_help = "向量化仍用 Ollama 或 OpenAI；选 Cursor 时可在 Cursor 区块选择。"
            if _provider == "deepseek":
                _embed_help = "DeepSeek 不提供 embeddings 接口，向量化将使用 **Ollama**；请确保 Ollama 已启动并在此填写向量化模型（如 nomic-embed-text）。"
            elif _provider == "lingyi":
                _embed_help = "零一万物向量化将使用 **Ollama**；请确保 Ollama 已启动并在此填写向量化模型（如 nomic-embed-text）。"
            embed_model = st.text_input(
                "向量化模型",
                key=f"sidebar_embed_model_{_provider}_v2",
                help=_embed_help,
            )
        elif _provider == "gemini":
            st.caption("密钥请配置环境变量 GEMINI_API_KEY 或 GOOGLE_API_KEY（.env）。")
            _lm_key = f"sidebar_llm_model_{_provider}_v2"
            llm_model = st.text_input(
                "审核模型",
                key=_lm_key,
                help="如 gemini-1.5-flash、gemini-1.5-pro",
            )
            embed_model = st.text_input(
                "向量化模型",
                help="向量建议仍用 Ollama 或 OpenAI Embedding",
                key=f"sidebar_embed_model_{_provider}_v2",
            )
        elif _provider == "tongyi":
            st.caption("密钥请配置 DASHSCOPE_API_KEY（.env 或系统环境变量）。")
            _lm_key = f"sidebar_llm_model_{_provider}_v2"
            llm_model = st.text_input(
                "审核模型",
                key=_lm_key,
                help="如 qwen-plus、qwen-max",
            )
            embed_model = st.text_input(
                "向量化模型",
                help="通义向量可用 text-embedding-v3，需单独接 Embedding API；当前可先 Ollama",
                key=f"sidebar_embed_model_{_provider}_v2",
            )
        elif _provider == "baidu":
            st.caption("密钥请配置 QIANFAN_AK、QIANFAN_SK（.env）。")
            _lm_key = f"sidebar_llm_model_{_provider}_v2"
            llm_model = st.text_input(
                "审核模型",
                key=_lm_key,
                help="千帆模型名",
            )
            embed_model = st.text_input(
                "向量化模型",
                help="向量可继续用 Ollama/OpenAI",
                key=f"sidebar_embed_model_{_provider}_v2",
            )

        # 扫描件 PDF：OCR 使用独立多模态模型（可与「审核模型」不同）；仅当 Settings 含该字段时展示
        pdf_ocr_llm_model = ""
        _pdf_ocr_providers = ("ollama", "openai", "deepseek", "lingyi", "tongyi", "cursor")
        if pdf_ocr_llm_model_field_available() and _provider in _pdf_ocr_providers:
            _ocr_help = (
                "文本层为空的 PDF 会逐页调用多模态做 OCR；各提供方独立保存（与审核模型可不同，点「保存配置」写入预设）。"
                "留空时：通义/DeepSeek（需配 DASHSCOPE_API_KEY）用 qwen-vl-plus；Ollama 用上方审核模型；OpenAI/零一用上方审核模型。"
                " 建议通义/DeepSeek OCR 填 qwen-vl-plus 或 qwen-vl-max；Ollama：qwen2.5vl；OpenAI：gpt-4o-mini。"
            )
            if _provider == "cursor":
                _ocr_help += (
                    " Cursor：OCR 实际走「向量化使用」— 选 **ollama** 时用本地多模态；选 **openai** 时用 OpenAI 兼容 Vision（需 OPENAI_API_KEY）。"
                )
            if _provider == "deepseek":
                _ocr_help += " DeepSeek 本身不支持传图 OCR，已自动走通义 VL（需 DASHSCOPE_API_KEY）。"
            _ocr_widget_key = f"sidebar_pdf_ocr_llm_model_{_provider}_v2"
            pdf_ocr_llm_model = st.text_input(
                "PDF / 扫描件 OCR 模型（多模态）",
                key=_ocr_widget_key,
                help=_ocr_help,
            )

        if st.button("💾 保存配置"):
            from config.cursor_overrides import _cursor_overrides as _co
            settings.provider = _provider
            settings.llm_model = llm_model
            set_pdf_ocr_llm_model(pdf_ocr_llm_model or "")
            settings.embedding_model = embed_model
            # 通用 HTTP 选项：写入 settings 并持久化到 DB（仍用 cursor_* 列存）
            _v = _co.get("verify_ssl") if _co.get("verify_ssl") is not None else get_llm_verify_ssl()
            _t = _co.get("trust_env") if _co.get("trust_env") is not None else get_llm_trust_env()
            settings.llm_verify_ssl = settings.cursor_verify_ssl = _v
            settings.llm_trust_env = settings.cursor_trust_env = _t
            if is_cursor:
                settings.cursor_api_key = cursor_api_key
                settings.cursor_repository = cursor_repo
                settings.cursor_ref = cursor_ref
                settings.cursor_embedding = cursor_embed
            elif is_ollama:
                settings.ollama_base_url = ollama_url
            elif is_openai_form:
                _bu_save = sanitize_openai_form_base_url(_provider, (base_url or "").strip())
                if _provider == "deepseek":
                    settings.deepseek_api_key = api_key
                    settings.deepseek_base_url = _bu_save
                elif _provider == "lingyi":
                    settings.lingyi_api_key = api_key
                    settings.lingyi_base_url = _bu_save
                else:
                    settings.openai_api_key = api_key
                    settings.openai_base_url = _bu_save
            _preset_updates = {
                "llm_model": (llm_model or "").strip(),
                "embedding_model": (embed_model or "").strip(),
            }
            if pdf_ocr_llm_model_field_available() and _provider in (
                "ollama",
                "openai",
                "deepseek",
                "lingyi",
                "tongyi",
                "cursor",
            ):
                _preset_updates["pdf_ocr_llm_model"] = (pdf_ocr_llm_model or "").strip()
            if is_ollama:
                _preset_updates["ollama_base_url"] = (ollama_url or "").strip()
            if is_openai_form:
                _preset_updates["base_url"] = sanitize_openai_form_base_url(_provider, (base_url or "").strip())
            upsert_provider_sidebar_slot(_provider, _preset_updates)
            save_app_settings(
                provider=settings.provider,
                openai_api_key=settings.openai_api_key,
                openai_base_url=settings.openai_base_url,
                ollama_base_url=settings.ollama_base_url,
                cursor_api_key=settings.cursor_api_key,
                cursor_api_base=settings.cursor_api_base,
                cursor_repository=settings.cursor_repository,
                cursor_ref=settings.cursor_ref,
                cursor_embedding=settings.cursor_embedding,
                cursor_verify_ssl=_v,
                cursor_trust_env=_t,
                llm_model=settings.llm_model,
                pdf_ocr_llm_model=get_pdf_ocr_llm_model(),
                embedding_model=settings.embedding_model,
                deepseek_api_key=getattr(settings, "deepseek_api_key", "") or "",
                deepseek_base_url=getattr(settings, "deepseek_base_url", "") or "",
                lingyi_api_key=getattr(settings, "lingyi_api_key", "") or "",
                lingyi_base_url=getattr(settings, "lingyi_base_url", "") or "",
                gemini_api_key=getattr(settings, "gemini_api_key", "") or getattr(settings, "google_api_key", "") or "",
                dashscope_api_key=getattr(settings, "dashscope_api_key", "") or "",
                qianfan_ak=getattr(settings, "qianfan_ak", "") or "",
                qianfan_sk=getattr(settings, "qianfan_sk", "") or "",
            )
            try:
                from src.core.db import persist_settings_dual_write

                persist_settings_dual_write()
            except Exception as _pe:
                st.warning(f"全量配置写入 runtime_settings_json 失败（兼容列已保存），可稍后在系统配置页重试：{_pe}")
            if "agent" in st.session_state:
                st.session_state.agent.reset_clients()
            st.success("配置已保存，页面将刷新。")
            streamlit_rerun()

        if _provider_ready():
            if settings.is_cursor:
                label = "Cursor Agent 模式 ✓"
            elif settings.is_ollama:
                label = "Ollama 本地模式 ✓"
            else:
                label = "OpenAI 模式 ✓"
            st.success(label)
        else:
            st.error("AI 服务未就绪")

        st.markdown("---")

        # --- 知识库 ---
        st.subheader("知识库")
        collection = st.text_input(
            "知识库名称",
            value=st.session_state.get("collection_name", "regulations"),
            help="不同项目可使用不同的知识库名称",
        )
        st.session_state.collection_name = collection

        # 侧栏仅用缓存统计，不初始化 Agent，避免切换功能时卡顿
        try:
            reg_stats = _cached_knowledge_stats(collection)
            cp_stats = _cached_checkpoint_stats(collection)
            fb_stats = _cached_audit_feedback_stats(collection)
            by_cat = _cached_knowledge_stats_by_category(collection)
        except Exception:
            reg_stats = {}
            cp_stats = {}
            fb_stats = {}
            by_cat = {}

        st.caption("法规知识库（第一步）— 以数据库为准")
        st.metric("法规向量块数", reg_stats.get("total_chunks", 0))

        st.caption("审核点知识库（第二步）— 以数据库为准")
        st.metric("清单向量块（第二步训练）", cp_stats.get("total_chunks", 0))
        st.metric("误报/纠正反馈块（独立库）", fb_stats.get("total_chunks", 0))
        st.caption(
            f"反馈数据集合：`{collection}_audit_feedback`，与清单向量 `{collection}_checkpoints` 分离；"
            "清空「全部」时删除反馈库；仅清空「审核点知识库」时**保留**反馈。"
        )

        try:
            st.caption("训练统计（按类型，以数据库为准）")
            st.markdown(f"**全部** {by_cat.get('total_files', 0)} 个文件 / {by_cat.get('total_chunks', 0)} 块")
            bc = by_cat.get("by_category") or {}
            for cat_key, label in CATEGORY_LABELS.items():
                c = bc.get(cat_key, {})
                st.caption(f"{label}: {c.get('files', 0)} 文件 / {c.get('chunks', 0)} 块")
        except Exception:
            pass

        if st.button("🔄 刷新统计", help="立即刷新上方统计与操作记录缓存"):
            _cached_knowledge_stats.clear()
            _cached_checkpoint_stats.clear()
            _cached_audit_feedback_stats.clear()
            _cached_knowledge_stats_by_category.clear()
            _cached_operation_logs.clear()
            st.experimental_rerun()

        clear_target = st.selectbox("清空目标", ["全部", "法规知识库", "审核点知识库"], key="clear_target")
        if st.button("🗑️ 清空知识库"):
            which_map = {"全部": "all", "法规知识库": "regulations", "审核点知识库": "checkpoints"}
            agent = init_agent()
            agent.clear_knowledge(which_map[clear_target])
            _cached_knowledge_stats.clear()
            _cached_checkpoint_stats.clear()
            _cached_audit_feedback_stats.clear()
            _cached_knowledge_stats_by_category.clear()
            st.success(f"已清空：{clear_target}")
            st.experimental_rerun()

        st.markdown("---")
        with st.expander("📐 维度选项配置", expanded=False):
            st.caption("注册国家、项目形态、国家→法规关键词均可在此配置，用于项目、审核点生成与审核维度。")
            dims = _cached_dimension_options()
            countries_str = st.text_area(
                "注册国家（每行一个，默认：中国、美国、欧盟）",
                value="\n".join(dims.get("registration_countries", ["中国", "美国", "欧盟"])),
                height=80,
                key="dim_countries",
            )
            forms_str = st.text_area(
                "项目形态（每行一个，默认：Web、APP、PC）",
                value="\n".join(dims.get("project_forms", ["Web", "APP", "PC"])),
                height=60,
                key="dim_forms",
            )
            _existing_kw = dims.get("country_extra_keywords") or {}
            _default_kw = {"CE": ["MDR"], "\u6b27\u76df": ["MDR"]}
            _kw_to_show = _existing_kw if _existing_kw else _default_kw
            kw_str = st.text_area(
                "\u56fd\u5bb6\u2192\u6cd5\u89c4\u5173\u952e\u8bcd\u6620\u5c04\uff08JSON\uff0c\u4e0d\u533a\u5206\u5927\u5c0f\u5199\uff09",
                value=json.dumps(_kw_to_show, ensure_ascii=False, indent=2),
                height=100,
                key="dim_country_keywords",
                help='\u6ce8\u518c\u56fd\u5bb6\u9009\u62e9 CE \u65f6\uff0c\u5ba1\u6838/\u751f\u6210\u5ba1\u6838\u70b9\u4f1a\u81ea\u52a8\u6269\u5c55\u68c0\u7d22 MDR \u7b49\u5173\u952e\u8bcd\u3002\u683c\u5f0f\u793a\u4f8b\uff1a{"CE": ["MDR"], "\u6b27\u76df": ["MDR"], "FDA": ["21 CFR Part 820"]}',
            )
            if st.button("保存维度选项", key="save_dims"):
                cl = [x.strip() for x in countries_str.split("\n") if x.strip()]
                fl = [x.strip() for x in forms_str.split("\n") if x.strip()]
                kw_parsed = None
                try:
                    kw_parsed = json.loads(kw_str) if kw_str.strip() else {}
                    if not isinstance(kw_parsed, dict):
                        st.warning("\u56fd\u5bb6\u2192\u6cd5\u89c4\u5173\u952e\u8bcd\u683c\u5f0f\u5e94\u4e3a JSON \u5bf9\u8c61\uff0c\u5df2\u5ffd\u7565\u3002")
                        kw_parsed = None
                except json.JSONDecodeError:
                    st.warning("\u56fd\u5bb6\u2192\u6cd5\u89c4\u5173\u952e\u8bcd JSON \u683c\u5f0f\u9519\u8bef\uff0c\u8bf7\u68c0\u67e5\u3002")
                    kw_parsed = None
                if cl and fl:
                    save_dimension_options(registration_countries=cl, project_forms=fl, country_extra_keywords=kw_parsed)
                    st.success("已保存")
                else:
                    st.warning("请至少各保留一项")

        st.markdown("---")
        st.subheader("API 服务")
        st.code(f"http://localhost:{settings.api_port}", language="text")
        st.caption("启动命令：`python -m src.api.server`")

        st.markdown("---")
        st.caption("注册文档审核工具 v1.0")


def _save_uploaded_file(file) -> str:
    """将上传文件保存到临时目录，返回路径"""
    suffix = Path(file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.read())
        return tmp.name


def _scan_directory_files(dir_path):
    """扫描目录中所有支持格式的文件；路径中含「废弃」的文件或目录跳过。"""
    path = Path(dir_path)
    files = []
    for ext in LOADER_MAP:
        for fp in path.rglob("*" + ext):
            if not is_deprecated_path(fp):
                files.append(fp)
    return files


def _expand_uploads(uploaded_files):
    """
    将上传列表展开：压缩包解压后加入其内文档，普通文件直接加入。
    返回 (items, temp_dirs)。items = [(path, display_name, is_from_archive), ...]
    """
    items = []
    temp_dirs = []

    for file in uploaded_files:
        if is_deprecated_path(file.name):
            continue
        tmp_path = _save_uploaded_file(file)
        if is_archive(Path(tmp_path)):
            try:
                temp_dir, doc_paths = extract_archive(tmp_path)
                temp_dirs.append(temp_dir)
                Path(tmp_path).unlink(missing_ok=True)
                archive_name = Path(file.name).stem
                for fp in doc_paths:
                    try:
                        rel = fp.relative_to(temp_dir)
                        display = f"{archive_name}/{rel}"
                    except ValueError:
                        display = fp.name
                    items.append((str(fp), display, True))
            except Exception as e:
                items.append((tmp_path, file.name + f" (解压失败: {e})", False))
        else:
            items.append((tmp_path, file.name, False))

    return items, temp_dirs


def _format_time(seconds):
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        m, s = divmod(int(seconds), 60)
        return f"{m}m{s}s"


def _format_review_exception(e: BaseException) -> str:
    """将审核相关异常转为可展示说明（部分异常 str(e) 与 repr(e) 均为空，界面会显示空白）。"""
    msg = (str(e) or "").strip()
    name = type(e).__name__
    _no_detail = f"{name}（未返回具体说明，常见于超时、连接中断或网关无响应；请检查网络、API Key、模型服务与 Cursor/代理配置）"
    if msg:
        headline = msg if msg.startswith(name) else f"{name}: {msg}"
    else:
        try:
            r = (repr(e) or "").strip()
            if r and r not in (f"{name}()", "''", '""', "None"):
                headline = f"{name}: {r}"
            else:
                headline = _no_detail
        except Exception:
            headline = _no_detail
    tb = traceback.format_exc()
    lines = [ln for ln in tb.strip().splitlines() if ln.strip()]
    if len(lines) > 2:
        tail = "\n".join(lines[-14:])
        return f"{headline}\n\n—— 异常追踪（末尾）——\n{tail}"
    return headline


def _is_transient_llm_error(e: BaseException) -> bool:
    """识别可自动重试的瞬时网络/网关错误。"""
    text = f"{type(e).__name__}: {e}".lower()
    # 上下文超长、限流等不应在应用层反复重试，避免请求雪崩与资源耗尽
    if any(
        k in text
        for k in (
            "context length",
            "maximum context",
            "too many tokens",
            "token limit",
            "maximum length",
            "rate limit",
            "too many requests",
            "429",
        )
    ):
        return False
    keywords = (
        "apiconnectionerror",
        "readtimeout",
        "connecttimeout",
        "connection reset",
        "connection aborted",
        "temporarily unavailable",
        "remoteprotocolerror",
        "winerror 10054",
        "ssl eof",
        "timed out",
        "timeout",
        "502",
        "503",
        "504",
    )
    return any(k in text for k in keywords)


def _review_transient_retry_attempts() -> int:
    """DeepSeek 等云端接口：应用层少次重试，主要依赖客户端内建重试；避免叠峰。"""
    if (getattr(settings, "provider", "") or "").strip().lower() == "deepseek":
        return 2
    return 3


def _call_with_transient_retry(fn, *, attempts: int = 3, base_sleep: float = 1.2):
    """调用 fn；若为瞬时错误则做指数退避重试。"""
    last_err = None
    for i in range(max(1, attempts)):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if i >= attempts - 1 or not _is_transient_llm_error(e):
                raise
            time.sleep(base_sleep * (2 ** i))
    if last_err is not None:
        raise last_err


def _render_loading_overlay():
    """在蒙层上显示 loading：全屏遮罩 + 加载文案与转圈，用纯 CSS 在约 1.5s 后淡出并不可点击（不依赖 JS，避免被过滤）。"""
    overlay_html = """
    <style>
    @keyframes st-loading-spin { 0%{transform:rotate(0deg)} 100%{transform:rotate(360deg)} }
    @keyframes st-loading-fadeout {
      0%, 70% { opacity:1; pointer-events:auto; }
      100% { opacity:0; pointer-events:none; }
    }
    #st-loading-overlay {
      position:fixed; top:0; left:0; width:100%; height:100%;
      background:rgba(240,240,240,0.92);
      z-index:999999;
      display:flex;
      flex-direction:column;
      align-items:center;
      justify-content:center;
      font-family:inherit; font-size:1.1rem; color:#333;
      animation: st-loading-fadeout 1.5s ease-out forwards;
    }
    #st-loading-overlay .spinner {
      width:44px; height:44px;
      border:3px solid #e0e0e0;
      border-top:3px solid #1f77b4;
      border-radius:50%;
      animation: st-loading-spin 0.8s linear infinite;
    }
    #st-loading-overlay p { margin-top:14px; }
    </style>
    <div id="st-loading-overlay">
      <div class="spinner"></div>
      <p>⏳ 页面加载中…</p>
    </div>
    """
    st.markdown(overlay_html, unsafe_allow_html=True)


# 文件分类：界面显示 <-> 数据库/代码
CATEGORY_LABELS = {"regulation": "法规文件", "program": "程序文件", "project_case": "项目案例文件", "glossary": "词条"}
CATEGORY_VALUES = {"法规文件": "regulation", "程序文件": "program", "项目案例文件": "project_case", "词条": "glossary"}

_TRAIN_CHUNKED_JOB_KEY = "_train_embed_chunked_job"


def _effective_embed_batch_size(total: int, batch_size: int = 12) -> int:
    threshold = getattr(settings, "embedding_large_file_threshold", 60)
    max_batch = getattr(settings, "embedding_large_file_batch_size", 12)
    return min(batch_size, max_batch) if total > threshold else batch_size


def _plain_specs_to_lc_documents(specs: list) -> list:
    return [
        LCDocument(page_content=(p.get("page_content") or ""), metadata=dict(p.get("metadata") or {}))
        for p in specs
    ]


def _lc_documents_to_plain_specs(chunks: list) -> list:
    return [
        {"page_content": d.page_content or "", "metadata": dict(d.metadata or {})}
        for d in chunks
    ]


def _clear_train_chunked_job(job: Optional[dict] = None) -> None:
    j = job if job is not None else st.session_state.get(_TRAIN_CHUNKED_JOB_KEY)
    if not j:
        st.session_state.pop(_TRAIN_CHUNKED_JOB_KEY, None)
        return
    p = j.get("pickle_path")
    if p:
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass
    st.session_state.pop(_TRAIN_CHUNKED_JOB_KEY, None)


def _train_embed_resume_progress_for_ui(queue_idx: int, display_name: str) -> tuple:
    """若当前文件正在分段向量化，返回 (0~1 的已写入比例, 说明文案)。供在 init_agent 之前渲染进度条。"""
    job = st.session_state.get(_TRAIN_CHUNKED_JOB_KEY)
    if not job:
        return 0.0, ""
    if job.get("queue_idx") != queue_idx or job.get("display_name") != display_name:
        return 0.0, ""
    tot = int(job.get("total") or 0)
    done = int(job.get("next_i") or 0)
    if tot <= 0:
        return 0.0, ""
    frac = min(1.0, max(0.0, done / tot))
    cap = (
        f"📊 **本文件**向量化进度：{done}/{tot} 块（分段刷新页面中；请勿长时间将本标签页置于后台）"
    )
    return frac, cap


def _train_single_file(
    agent,
    file_path,
    file_name,
    status_box,
    embed_bar,
    log_lines,
    category: str = "regulation",
    case_id=None,
    queue_idx: Optional[int] = None,
):
    """
    训练单个文件。通过 st.empty() 实时刷新状态。
    分块数较多时按多轮 rerun 分段向量化，降低长时间阻塞导致的 WebSocket 断连（表现为训练「自动取消」）。
    log_lines: list, 累积日志，每次追加后刷新显示。
    返回 (成功?, 块数, 耗时)
    """
    t0 = time.time()
    st_chunk_th = max(1, int(getattr(settings, "embedding_streamlit_chunk_threshold", 48)))
    batches_per_rerun = max(1, int(getattr(settings, "embedding_streamlit_batches_per_rerun", 1)))

    job = st.session_state.get(_TRAIN_CHUNKED_JOB_KEY)
    if job and queue_idx is not None:
        if job.get("queue_idx") != queue_idx or job.get("display_name") != file_name:
            _clear_train_chunked_job(job)

    job = st.session_state.get(_TRAIN_CHUNKED_JOB_KEY)
    if (
        job
        and queue_idx is not None
        and job.get("queue_idx") == queue_idx
        and job.get("display_name") == file_name
        and (job.get("collection") or "") == (agent.collection_name or "")
    ):
        pickle_path = job.get("pickle_path")
        if not pickle_path or not Path(pickle_path).is_file():
            _clear_train_chunked_job(job)
            status_box.error(f"[{file_name}] 分段训练状态丢失，请重新训练该文件。")
            log_lines.append(f"- :x: **{file_name}** 分段状态丢失，已中止")
            return False, 0, time.time() - t0
        try:
            with open(pickle_path, "rb") as f:
                specs = pickle.load(f)
            chunks = _plain_specs_to_lc_documents(specs)
        except Exception as e:
            _clear_train_chunked_job(job)
            err_msg = str(e)
            log_lines.append(f"- :x: **{file_name}** 恢复分段训练失败：{err_msg}")
            status_box.error(f"恢复失败：{err_msg}")
            return False, 0, time.time() - t0

        total = len(chunks)
        eff_bs = int(job.get("effective_batch_size") or _effective_embed_batch_size(total, 12))
        next_i = int(job.get("next_i") or 0)
        t_start = float(job.get("t0") or t0)
        embed_bar.progress(min(1.0, max(0.0, next_i / total)) if total else 0.0)

        batches_this_run = 0
        try:
            while next_i < total and batches_this_run < batches_per_rerun:
                batch = chunks[next_i : next_i + eff_bs]
                if not batch:
                    break
                batch_start = next_i
                _add_batch_with_retry(agent.kb.vectorstore, batch)
                append_knowledge_docs(
                    agent.collection_name, file_name, batch_start, batch, category=category
                )
                next_i += len(batch)
                batches_this_run += 1
                frac = min(1.0, max(0.0, next_i / total))
                embed_bar.progress(frac)
                pct = int(frac * 100)
                status_box.info(
                    f"🔄 [{file_name}] 向量化 {next_i}/{total} 块 ({pct}%)（分段刷新防页面断连）"
                )
        except Exception as e:
            tb = traceback.format_exc()
            err_msg = str(e)
            _clear_train_chunked_job(job)
            log_lines.append(f"- :x: **{file_name}** 入库失败：{err_msg}")
            status_box.error(f"入库失败：{err_msg}")
            add_operation_log(
                op_type="train_error",
                collection=agent.collection_name,
                file_name=file_name,
                source=str(file_path),
                extra={"error": err_msg, "traceback": tb, "stage": "embed_chunked", "category": category},
                model_info=get_current_model_info(),
            )
            return False, 0, time.time() - t_start

        if next_i >= total:
            _clear_train_chunked_job(job)
            elapsed = time.time() - t_start
            embed_bar.progress(1.0)
            log_lines.append(
                f"- :white_check_mark: **{file_name}** — {total} 块, {_format_time(elapsed)}"
            )
            status_box.success(f"[{file_name}] 完成! {total} 块, {_format_time(elapsed)}")
            add_operation_log(
                op_type="train",
                collection=agent.collection_name,
                file_name=file_name,
                source=str(file_path),
                extra={"chunks": total, "duration_sec": elapsed, "category": category, "streamlit_chunked": True},
                model_info=get_current_model_info(),
            )
            return True, total, elapsed

        job["next_i"] = next_i
        st.session_state[_TRAIN_CHUNKED_JOB_KEY] = job
        try:
            st.session_state["train_queue_log"] = list(log_lines)
        except Exception:
            pass
        time.sleep(0.12)
        st.experimental_rerun()

    status_box.info(f"📂 [{file_name}] 正在加载文件...")
    try:
        docs = load_single_file(file_path)
    except Exception as e:
        tb = traceback.format_exc()
        err_msg = str(e)
        log_lines.append(f"- :x: **{file_name}** 加载失败：{err_msg}")
        status_box.error(f"加载失败：{err_msg}")
        add_operation_log(
            op_type="train_error",
            collection=agent.collection_name,
            file_name=file_name,
            source=str(file_path),
            extra={"error": err_msg, "traceback": tb, "stage": "load", "category": category},
            model_info=get_current_model_info(),
        )
        return False, 0, time.time() - t0

    status_box.info(f"✂ [{file_name}] 分块中... ({len(docs)} 页/段)")
    chunks = split_documents(docs)

    if not chunks:
        log_lines.append(f"- :warning: **{file_name}** 内容为空，跳过")
        status_box.warning(f"[{file_name}] 内容为空")
        return True, 0, time.time() - t0

    total = len(chunks)

    if queue_idx is not None and total > st_chunk_th:
        annotate_main_knowledge_documents(chunks, file_name, category=category, case_id=case_id)
        fd, pickle_path = tempfile.mkstemp(suffix=".pkl", prefix="aicheckword_chunked_embed_")
        os.close(fd)
        try:
            with open(pickle_path, "wb") as f:
                pickle.dump(_lc_documents_to_plain_specs(chunks), f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            try:
                Path(pickle_path).unlink(missing_ok=True)
            except Exception:
                pass
            err_msg = str(e)
            log_lines.append(f"- :x: **{file_name}** 无法启动分段训练：{err_msg}")
            status_box.error(err_msg)
            return False, 0, time.time() - t0

        st.session_state[_TRAIN_CHUNKED_JOB_KEY] = {
            "pickle_path": pickle_path,
            "queue_idx": queue_idx,
            "display_name": file_name,
            "collection": agent.collection_name,
            "category": category,
            "case_id": case_id,
            "next_i": 0,
            "total": total,
            "effective_batch_size": _effective_embed_batch_size(total, 12),
            "batches_per_rerun": batches_per_rerun,
            "t0": time.time(),
        }
        embed_bar.progress(0.0)
        status_box.info(
            f"📎 [{file_name}] 共 {total} 块，将分段向量化（每步刷新页面），请勿切换至后台过久…"
        )
        try:
            st.session_state["train_queue_log"] = list(log_lines)
        except Exception:
            pass
        time.sleep(0.12)
        st.experimental_rerun()

    embed_bar.progress(0.0)

    def on_batch_done(done, total_n):
        frac = min(1.0, max(0.0, (done / total_n) if total_n else 0.0))
        embed_bar.progress(frac)
        pct = int(frac * 100)
        status_box.info(
            f"🔄 [{file_name}] 向量化 {done}/{total_n} 块 ({pct}%)"
        )

    try:
        # 不要用 st.spinner 包住整段向量化：块内进度条/状态往往要等 spinner 结束才一并刷新，表现为「看不到进度」
        status_box.info(f"🔄 [{file_name}] 正在向量化 {len(chunks)} 块…")
        add_progress = agent.kb.add_documents_with_progress
        sig = inspect.signature(add_progress)
        kwargs = dict(documents=chunks, batch_size=12, callback=on_batch_done, file_name=file_name, category=category)
        if "case_id" in sig.parameters:
            kwargs["case_id"] = case_id
        count = add_progress(**kwargs)
    except Exception as e:
        tb = traceback.format_exc()
        err_msg = str(e)
        log_lines.append(f"- :x: **{file_name}** 入库失败：{err_msg}")
        status_box.error(f"入库失败：{err_msg}")
        add_operation_log(
            op_type="train_error",
            collection=agent.collection_name,
            file_name=file_name,
            source=str(file_path),
            extra={"error": err_msg, "traceback": tb, "stage": "embed", "category": category},
            model_info=get_current_model_info(),
        )
        return False, 0, time.time() - t0

    elapsed = time.time() - t0
    embed_bar.progress(1.0)
    log_lines.append(
        f"- :white_check_mark: **{file_name}** — {count} 块, {_format_time(elapsed)}"
    )
    status_box.success(f"[{file_name}] 完成! {count} 块, {_format_time(elapsed)}")
    add_operation_log(
        op_type="train",
        collection=agent.collection_name,
        file_name=file_name,
        source=str(file_path),
        extra={"chunks": count, "duration_sec": elapsed, "category": category},
        model_info=get_current_model_info(),
    )
    return True, count, elapsed


def render_step1_page():
    """第一步：训练法规/程序/案例 + 生成审核点"""
    st.header("① 法规训练 & 生成审核点")
    st.markdown(
        "**第一步**：上传法规、标准、程序文件、项目案例，训练法规知识库。"
        "训练完成后，可让 AI 基于知识库**自动生成审核点清单**，"
        "也可上传已有的基础审核点文档让 AI 优化。"
    )

    if not _require_provider():
        return

    # 训练成功后的弹窗提示（不自动消失，用户点击「确定」后再关闭）
    if st.session_state.get("train_show_success") and st.session_state.get("train_success_info"):
        info = st.session_state["train_success_info"]
        success_count = info.get("success_count", 0)
        total_files = info.get("total_files", 0)
        total_chunks = info.get("total_chunks", 0)
        total_time = info.get("total_time", 0)
        fail_count = info.get("fail_count", 0)
        src = info.get("source", "upload")
        src_label = "从目录训练" if src == "directory" else "上传训练"
        with st.container():
            st.success(
                f"✅ **训练完成**（{src_label}）\n\n"
                f"成功 **{success_count}/{total_files}** 个文件，共入库 **{total_chunks}** 块，"
                f"耗时 **{_format_time(total_time)}**。"
                + (f" 失败 {fail_count} 个。" if fail_count else "")
            )
            if fail_count == 0:
                st.balloons()
            if st.button("确定", key="train_success_confirm"):
                st.session_state.pop("train_show_success", None)
                st.session_state.pop("train_success_info", None)
                st.experimental_rerun()
        st.markdown("---")

    # ── 覆盖/跳过 action handler（在 tabs 之前，确保无论用户在哪个 tab 都能处理） ──
    _pending_action = None
    if st.session_state.get("train_do_overwrite") and st.session_state.get("train_pending_items"):
        _pending_action = "overwrite"
    elif st.session_state.get("train_do_skip") and st.session_state.get("train_pending_items"):
        _pending_action = "skip"
    if _pending_action:
        _ha_items = st.session_state["train_pending_items"]
        _ha_collection = st.session_state.get("collection_name", "regulations")
        _ha_category = st.session_state.get("train_pending_category", "regulation")
        _ha_case_id = st.session_state.get("train_pending_case_id")
        _ha_source = st.session_state.get("train_pending_source", "upload")
        _ha_dir_path = st.session_state.get("train_pending_dir_path", "")
        if _pending_action == "overwrite":
            _ha_existing = set(get_existing_file_names(_ha_collection, category=_ha_category, case_id=_ha_case_id))
            _ha_dup_set = set(dn for (_, dn, _) in _ha_items if dn in _ha_existing)
            _ha_queue_items = _ha_items
        else:
            _ha_dup_set = set()
            _ha_pending_dups = st.session_state.get("train_pending_duplicates") or set()
            _ha_queue_items = [(p, dn, ar) for (p, dn, ar) in _ha_items if dn not in _ha_pending_dups]
        _ha_cleanup_keys = (
            "train_do_overwrite", "train_do_skip", "train_pending_items", "train_pending_temp_dirs",
            "train_pending_category", "train_pending_category_label", "train_pending_duplicates",
            "train_pending_case_id", "train_pending_source", "train_pending_dir_path",
        )
        for key in _ha_cleanup_keys:
            st.session_state.pop(key, None)
        if not _ha_queue_items:
            st.warning("\u5168\u90e8\u6587\u4ef6\u5747\u91cd\u540d\uff0c\u5df2\u8df3\u8fc7\uff0c\u65e0\u6587\u4ef6\u9700\u8981\u8bad\u7ec3\u3002")
        else:
            _clear_train_chunked_job()
            st.session_state["train_queue"] = _ha_queue_items
            st.session_state["train_queue_index"] = 0
            st.session_state["train_queue_temp_dirs"] = st.session_state.get("train_pending_temp_dirs", [])
            st.session_state["train_queue_category"] = _ha_category
            st.session_state["train_queue_category_label"] = st.session_state.get("train_pending_category_label", "\u6cd5\u89c4\u6587\u4ef6")
            st.session_state["train_queue_dup_set"] = _ha_dup_set
            st.session_state["train_queue_log"] = []
            st.session_state["train_queue_success"] = 0
            st.session_state["train_queue_fail"] = 0
            st.session_state["train_queue_chunks"] = 0
            st.session_state["train_queue_start_time"] = time.time()
            st.session_state["train_queue_source"] = _ha_source
            if _ha_dir_path:
                st.session_state["train_queue_dir_path"] = _ha_dir_path
            if _ha_case_id is not None:
                st.session_state["train_queue_case_id"] = _ha_case_id
            st.experimental_rerun()

    def _step1_tab_upload():
        if (
            st.session_state.get("train_queue")
            and st.session_state.get("train_queue_source") == "directory"
            and st.session_state.get("train_queue_index", 0) < len(st.session_state["train_queue"])
        ):
            st.info("当前正在从目录训练，请点击上方 **「从目录训练」** 标签页查看进度。")
        file_category_label = st.selectbox(
            "文件分类",
            ["法规文件", "程序文件", "项目案例文件", "词条"],
            key="train_upload_category",
            help="选择本批上传文件的类型，用于知识库分类与统计",
        )
        # 项目案例文件：标注案例元数据（过往项目经验），入通用知识库供新项目复用（用缓存避免切换选项时卡顿）
        if file_category_label == "项目案例文件":
            collection = st.session_state.get("collection_name", "regulations")
            dims = _cached_dimension_options()
            countries = dims.get("registration_countries", ["中国", "美国", "欧盟"]) or ["中国", "美国", "欧盟"]
            forms = dims.get("project_forms", ["Web", "APP", "PC"]) or ["Web", "APP", "PC"]
            cases = _cached_list_project_cases(collection)
            # 「其他国家/语言版本」须在 selectbox 渲染前写入 session_state，否则仅按钮内赋值 + rerun 可能不切换选项
            _pending_var_u = st.session_state.pop("_pending_variant_upload_case_id", None)
            if _pending_var_u is not None:
                st.session_state["train_upload_case_sel"] = "➕ 新建案例"
                st.session_state["train_copy_from_case_id"] = _pending_var_u
            case_options = ["➕ 新建案例"] + [_format_case_option(c) for c in cases]
            sel_case = st.selectbox("选择过往项目案例", case_options, key="train_upload_case_sel",
                                    help="项目案例是过往已成功注册的项目经验，训练后可供类似新项目审核时参考。")
            if sel_case == "➕ 新建案例":
                st.caption("新建案例：描述这批文件所属的过往项目，训练到通用知识库；第三步审核时按产品名称与适用范围匹配启用（支持中英文匹配）。")
                # 复制自现有案例（快捷创建其他注册国家/语言版本）
                case_options_copy = [_format_case_option(c) for c in cases]
                copy_default_idx = 0
                if st.session_state.get("train_copy_from_case_id"):
                    try:
                        cid = st.session_state["train_copy_from_case_id"]
                        for i, c in enumerate(cases):
                            if c.get("id") == cid:
                                copy_default_idx = i + 1  # 0 为「不复制」
                                break
                        st.session_state.pop("train_copy_from_case_id", None)
                    except Exception:
                        pass
                copy_from_sel = st.selectbox(
                    "复制自现有案例（可选）",
                    ["不复制"] + case_options_copy,
                    index=min(copy_default_idx, len(case_options_copy)),
                    key="train_copy_from_sel",
                    help="选择后将预填案例信息，仅改注册国家、案例文档语言即可创建该案例的其他国家/语言版本；同一项目的案例会通过 project_key 关联。",
                )
                copy_case = None
                if copy_from_sel and copy_from_sel != "不复制" and copy_from_sel in case_options_copy:
                    copy_case = cases[case_options_copy.index(copy_from_sel)]
                if copy_case:
                    st.caption("已从现有案例预填，请修改「注册国家」「案例文档语言」等以区分为新版本。")
                _def = lambda k, d: (copy_case.get(k) or d) if copy_case else d
                _idx = lambda opts, val: opts.index(val) if val in opts else 0
                p_case_name = st.text_input("案例名称", value=_def("case_name", ""), placeholder="例如：某某血糖仪二类注册案例", key="train_case_name")
                p_case_name_en = st.text_input("案例名称（英文）", value=_def("case_name_en", ""), placeholder="e.g. XXX Blood Glucose Meter Class II", key="train_case_name_en")
                p_product = st.text_input("产品名称", value=_def("product_name", ""), placeholder="该过往项目的产品名称", key="train_case_product")
                p_product_en = st.text_input("产品名称（英文）", value=_def("product_name_en", ""), placeholder="Product name in English", key="train_case_product_en")
                _doc_lang_val = _def("document_language", "zh")
                _doc_lang_label = DOC_LANG_VALUE_TO_LABEL.get(_doc_lang_val, "中文版")
                p_doc_lang = st.selectbox("案例文档语言", DOC_LANG_OPTIONS, index=_idx(DOC_LANG_OPTIONS, _doc_lang_label), key="train_case_doc_lang", help="本批训练文档主要为中文、英文或中英文；与生成审核点、文档审核的文档语言选项保持一致")
                p_country = st.selectbox("注册国家", countries, index=_idx(countries, _def("registration_country", countries[0])), key="train_case_country")
                p_country_en = st.text_input("注册国家（英文）", value=_def("registration_country_en", ""), placeholder="e.g. China, USA", key="train_case_country_en")
                p_type = st.selectbox("注册类别", REGISTRATION_TYPES, index=_idx(REGISTRATION_TYPES, _def("registration_type", REGISTRATION_TYPES[0])), key="train_case_type")
                p_comp = st.selectbox("注册组成", REGISTRATION_COMPONENTS, index=_idx(REGISTRATION_COMPONENTS, _def("registration_component", REGISTRATION_COMPONENTS[0])), key="train_case_comp")
                p_form = st.selectbox("项目形态", forms, index=_idx(forms, _def("project_form", forms[0])), key="train_case_form")
                p_scope = st.text_area("产品适用范围（可选）", value=_def("scope_of_application", ""), placeholder="该过往项目的适用范围；第三步审核时可据此匹配", height=80, key="train_case_scope")
            else:
                try:
                    _idx = lambda opts, val: opts.index(val) if val in opts else 0
                    idx = case_options.index(sel_case)
                    case = cases[idx - 1]
                    _lang_label = DOC_LANG_VALUE_TO_LABEL.get(case.get("document_language") or "", "不指定")
                    st.caption(
                        f"将训练到案例：**{case.get('case_name')}**"
                        f"（文档语言：{_lang_label}；产品名称：{case.get('product_name') or '—'}，"
                        f"注册国家：{case.get('registration_country')}，"
                        f"注册类别：{case.get('registration_type')}）"
                    )
                    _cid = case.get("id")
                    with st.expander("✏️ 编辑此案例（中/英文）", expanded=False):
                        e_case_name = st.text_input("案例名称", value=case.get("case_name") or "", key=f"edit_case_name_{_cid}")
                        e_case_name_en = st.text_input("案例名称（英文）", value=case.get("case_name_en") or "", key=f"edit_case_name_en_{_cid}")
                        e_product = st.text_input("产品名称", value=case.get("product_name") or "", key=f"edit_case_product_{_cid}")
                        e_product_en = st.text_input("产品名称（英文）", value=case.get("product_name_en") or "", key=f"edit_case_product_en_{_cid}")
                        _country_val = (case.get("registration_country") or "").strip() or (countries[0] if countries else "")
                        _country_idx = _idx(countries, _country_val)
                        e_country = st.selectbox("注册国家", countries, index=min(_country_idx, len(countries) - 1) if countries else 0, key=f"edit_case_country_{_cid}")
                        e_country_en = st.text_input("注册国家（英文）", value=case.get("registration_country_en") or "", placeholder="e.g. China, USA", key=f"edit_case_country_en_{_cid}")
                        _type_val = (case.get("registration_type") or "").strip() or (REGISTRATION_TYPES[0] if REGISTRATION_TYPES else "")
                        _type_idx = _idx(REGISTRATION_TYPES, _type_val)
                        e_type = st.selectbox("注册类别", REGISTRATION_TYPES, index=min(_type_idx, len(REGISTRATION_TYPES) - 1) if REGISTRATION_TYPES else 0, key=f"edit_case_type_{_cid}")
                        _comp_val = (case.get("registration_component") or "").strip() or (REGISTRATION_COMPONENTS[0] if REGISTRATION_COMPONENTS else "")
                        _comp_idx = _idx(REGISTRATION_COMPONENTS, _comp_val)
                        e_comp = st.selectbox("注册组成", REGISTRATION_COMPONENTS, index=min(_comp_idx, len(REGISTRATION_COMPONENTS) - 1) if REGISTRATION_COMPONENTS else 0, key=f"edit_case_comp_{_cid}")
                        _form_val = (case.get("project_form") or "").strip() or (forms[0] if forms else "")
                        _form_idx = _idx(forms, _form_val)
                        e_form = st.selectbox("项目形态", forms, index=min(_form_idx, len(forms) - 1) if forms else 0, key=f"edit_case_form_{_cid}")
                        _doc_lang_val = case.get("document_language") or ""
                        _doc_lang_label = DOC_LANG_VALUE_TO_LABEL.get(_doc_lang_val, "不指定")
                        _doc_lang_idx = DOC_LANG_OPTIONS.index(_doc_lang_label) if _doc_lang_label in DOC_LANG_OPTIONS else 0
                        e_doc_lang = st.selectbox("案例文档语言", DOC_LANG_OPTIONS, index=_doc_lang_idx, key=f"edit_case_doc_lang_{_cid}")
                        e_scope = st.text_area("产品适用范围", value=case.get("scope_of_application") or "", height=60, key=f"edit_case_scope_{_cid}")
                        other_cases_upload = [c for c in cases if c.get("id") != case.get("id")]
                        link_options_upload = ["不关联（独立）"] + [_format_case_option(c) for c in other_cases_upload]
                        _link_idx_upload = 0
                        if case.get("project_key"):
                            for i, c in enumerate(other_cases_upload):
                                if str(c.get("id")) == str(case.get("project_key")) or (c.get("project_key") and str(c.get("project_key")) == str(case.get("project_key"))):
                                    _link_idx_upload = i + 1
                                    break
                        e_link_upload = st.selectbox("关联到同一项目", link_options_upload, index=min(_link_idx_upload, len(link_options_upload) - 1), key=f"edit_case_link_upload_{_cid}", help="与所选案例归为同一项目（多国家/多语言版本）；不关联则本案例独立。")
                        if st.button("保存案例", key=f"edit_case_save_{_cid}"):
                            project_key_new = ""
                            if e_link_upload and e_link_upload != "不关联（独立）" and e_link_upload in link_options_upload:
                                idx_link = link_options_upload.index(e_link_upload)
                                if idx_link > 0:
                                    linked = other_cases_upload[idx_link - 1]
                                    project_key_new = (linked.get("project_key") or "").strip() or str(linked.get("id", ""))
                            update_project_case(
                                case["id"],
                                case_name=e_case_name.strip() or None,
                                case_name_en=e_case_name_en.strip() or None,
                                product_name=e_product.strip() or None,
                                product_name_en=e_product_en.strip() or None,
                                registration_country=e_country.strip() or None,
                                registration_country_en=e_country_en.strip() or None,
                                registration_type=e_type,
                                registration_component=e_comp,
                                project_form=e_form,
                                document_language=DOC_LANG_LABEL_TO_VALUE.get(e_doc_lang, ""),
                                scope_of_application=e_scope.strip() or None,
                                project_key=project_key_new,
                            )
                            st.success("已保存")
                            st.experimental_rerun()
                    _case_files = get_project_case_file_names(collection, case["id"])
                    if _case_files:
                        _preview = ", ".join(_case_files[:5]) + ("..." if len(_case_files) > 5 else "")
                        st.caption(f"📁 **已入库文件**：共 **{len(_case_files)}** 个 — {_preview}")
                    else:
                        st.caption("📁 **已入库文件**：暂无")
                    if st.button("📋 创建本案例的其他国家/语言版本", key="btn_variant_upload", help="切换到新建案例并预填本案例信息，仅改注册国家、文档语言即可创建新版本"):
                        st.session_state["_pending_variant_upload_case_id"] = case["id"]
                        _streamlit_rerun()
                    _pending_del_u = st.session_state.get("train_del_case_upload_id")
                    if _pending_del_u == case["id"]:
                        _nf = len(_case_files or [])
                        st.warning(
                            f"**确认删除模板案例？** 当前案例下共有 **{_nf}** 个已入库文件；"
                            f"确认后将**永久删除**案例记录，并清空知识库中该案例关联的全部文档块与向量，**不可恢复**。"
                        )
                        if _case_files:
                            _pv = ", ".join(_case_files[:15]) + ("…" if len(_case_files) > 15 else "")
                            st.caption(f"文件清单（节选）：{_pv}")
                        _c1, _c2 = st.columns(2)
                        with _c1:
                            if st.button("✅ 确认删除", key=f"confirm_del_case_u_{case['id']}"):
                                try:
                                    from src.core.knowledge_base import KnowledgeBase

                                    KnowledgeBase(collection).delete_documents_by_case_id(int(case["id"]))
                                except Exception as _ex:
                                    st.error(f"清理知识库失败：{_ex}")
                                else:
                                    delete_project_case(int(case["id"]))
                                    st.session_state.pop("train_del_case_upload_id", None)
                                    _cached_list_project_cases.clear()
                                    st.success("已删除案例及其全部入库文件")
                                    _streamlit_rerun()
                        with _c2:
                            if st.button("取消", key=f"cancel_del_case_u_{case['id']}"):
                                st.session_state.pop("train_del_case_upload_id", None)
                                _streamlit_rerun()
                    elif st.button("🗑️ 删除此案例", key="del_case_upload"):
                        st.session_state["train_del_case_upload_id"] = case["id"]
                        _streamlit_rerun()
                except Exception:
                    pass
        uploaded_files = st.file_uploader(
            "选择训练文件（支持单个文档或 .zip / .tar / .tar.gz 压缩包，压缩包将自动解压后导入）",
            type=["pdf", "docx", "doc", "xlsx", "xls", "txt", "md", "zip", "tar", "gz", "tgz"],
            accept_multiple_files=True,
            key="train_uploader",
            help="PDF 优先读取文本层；若为扫描件/图片版且文本极少，将自动尝试 AI OCR 识别后再入库（兜底最多前 30 页）。单文件最多处理前 500 页。",
        )

        # 有待处理的重复文件时（上传来源），在稳定位置渲染覆盖/跳过按钮
        if (
            st.session_state.get("train_pending_items")
            and st.session_state.get("train_pending_source") == "upload"
            and not st.session_state.get("train_do_overwrite")
            and not st.session_state.get("train_do_skip")
            and not st.session_state.get("train_queue")
        ):
            _dup_names = st.session_state.get("train_pending_duplicates") or set()
            if _dup_names:
                st.warning("**\u4e0e\u6570\u636e\u5e93\u5bf9\u6bd4**\uff1a\u4ee5\u4e0b\u6587\u4ef6\u540d\u5728\u77e5\u8bc6\u5e93\uff08\u6570\u636e\u5e93\uff09\u4e2d\u5df2\u5b58\u5728\uff0c\u8bf7\u9009\u62e9\u8986\u76d6\u6216\u8df3\u8fc7\u3002")
                _dup_list = sorted(_dup_names) if isinstance(_dup_names, set) else list(_dup_names)
                st.caption(f"\u91cd\u540d\u6587\u4ef6\uff1a{', '.join(_dup_list[:10])}{'...' if len(_dup_list) > 10 else ''}")
                _dc1, _dc2 = st.columns(2)
                with _dc1:
                    if st.button("\u2705 \u8986\u76d6\uff1a\u7528\u6700\u65b0\u6587\u4ef6\u8986\u76d6\u65e7\u5185\u5bb9\u5e76\u8bad\u7ec3", key="train_confirm_overwrite"):
                        st.session_state["train_do_overwrite"] = True
                        st.experimental_rerun()
                with _dc2:
                    if st.button("\u23ed\ufe0f \u4e0d\u8986\u76d6\uff1a\u8df3\u8fc7\u91cd\u540d\u6587\u4ef6\uff0c\u4ec5\u8bad\u7ec3\u5176\u4f59\u6587\u4ef6", key="train_skip_duplicates"):
                        st.session_state["train_do_skip"] = True
                        st.experimental_rerun()

        # 每轮只处理一个文件；仅在上传来源时在此 tab 显示进度（目录训练进度在 tab2 显示）
        if (
            st.session_state.get("train_queue")
            and st.session_state.get("train_queue_index", 0) < len(st.session_state["train_queue"])
            and st.session_state.get("train_queue_source") != "directory"
        ):
            queue = st.session_state["train_queue"]
            idx = st.session_state["train_queue_index"]
            total_files = len(queue)
            path, display_name, is_from_archive = queue[idx]
            category = st.session_state.get("train_queue_category", "regulation")
            dup_set = st.session_state.get("train_queue_dup_set") or set()
            log_lines = list(st.session_state.get("train_queue_log", []))

            # 先输出提示，再加载 agent（首次加载较慢，用 spinner 包住避免误以为死机）
            st.info(f"🔄 **正在训练第 {idx + 1}/{total_files} 个文件**，请勿关闭页面。")
            _emb_frac0, _emb_cap0 = _train_embed_resume_progress_for_ui(idx, display_name)
            if _emb_cap0:
                st.caption(_emb_cap0)
            try:
                with st.spinner("正在加载模型与知识库（首次约 10–30 秒，请稍候）…"):
                    agent = init_agent()
            except Exception as e0:
                log_lines.append(f"- :x: **{display_name}** 加载模型/知识库失败，已跳过本文件：{e0}")
                st.session_state["train_queue_fail"] = st.session_state.get("train_queue_fail", 0) + 1
                st.session_state["train_queue_index"] = idx + 1
                st.session_state["train_queue_log"] = log_lines
                st.warning(f"本文件加载失败已跳过，继续训练其余文件。错误：{e0}")
                st.caption("如持续失败，请检查模型服务是否正常、网络配置是否正确。")
                st.experimental_rerun()

            overall_bar = st.progress((idx) / max(total_files, 1))
            overall_text = st.empty()
            status_box = st.empty()
            # 须用 st.progress(0) 而非 st.empty()：在部分 Streamlit 版本上对 empty 调用 .progress() 会导致嵌入进度条不显示
            embed_bar = st.progress(min(1.0, max(0.0, _emb_frac0)))
            log_display = st.empty()
            overall_text.info(
                f"总进度 {idx + 1}/{total_files} | 成功 {st.session_state.get('train_queue_success', 0)} | "
                f"失败 {st.session_state.get('train_queue_fail', 0)} | 已入库 {st.session_state.get('train_queue_chunks', 0)} 块"
            )
            log_display.markdown("\n".join(log_lines))

            try:
                if display_name in dup_set:
                    try:
                        agent.kb.delete_documents_by_file_name(display_name, case_id=st.session_state.get("train_queue_case_id") if category == "project_case" else None)
                        log_lines.append(f"- 🔄 **{display_name}** 已覆盖旧内容，正在重新训练…")
                    except Exception as e:
                        log_lines.append(f"- ⚠️ **{display_name}** 覆盖前清理失败：{e}")

                ok, chunks, elapsed = _train_single_file(
                    agent,
                    path,
                    display_name,
                    status_box,
                    embed_bar,
                    log_lines,
                    category=category,
                    case_id=st.session_state.get("train_queue_case_id"),
                    queue_idx=idx,
                )
                log_display.markdown("\n".join(log_lines))
                if ok:
                    st.session_state["train_queue_success"] = st.session_state.get("train_queue_success", 0) + 1
                    st.session_state["train_queue_chunks"] = st.session_state.get("train_queue_chunks", 0) + chunks
                    train_case_id = st.session_state.get("train_queue_case_id")
                    if train_case_id is not None:
                        try:
                            update_knowledge_docs_case_id(agent.collection_name, display_name, train_case_id)
                        except Exception:
                            pass
                else:
                    st.session_state["train_queue_fail"] = st.session_state.get("train_queue_fail", 0) + 1
            except Exception as e:
                log_lines.append(f"- :x: **{display_name}** 异常：{e}")
                st.session_state["train_queue_fail"] = st.session_state.get("train_queue_fail", 0) + 1
                try:
                    log_display.markdown("\n".join(log_lines))
                except Exception:
                    pass
                try:
                    add_operation_log(
                        op_type="train_error",
                        collection=agent.collection_name,
                        file_name=display_name,
                        source=str(path),
                        extra={"error": str(e), "traceback": traceback.format_exc(), "stage": "train", "category": category},
                        model_info=get_current_model_info(),
                    )
                except Exception:
                    pass

            # 仅在上传训练时删除临时文件；从目录训练时绝不删除本地文件
            if st.session_state.get("train_queue_source") == "upload" and not is_from_archive:
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception:
                    pass

            st.session_state["train_queue_index"] = idx + 1
            st.session_state["train_queue_log"] = log_lines

            if st.session_state["train_queue_index"] >= len(queue):
                total_time = time.time() - st.session_state.get("train_queue_start_time", time.time())
                success_count = st.session_state.get("train_queue_success", 0)
                fail_count = st.session_state.get("train_queue_fail", 0)
                total_chunks = st.session_state.get("train_queue_chunks", 0)
                for d in st.session_state.get("train_queue_temp_dirs", []):
                    shutil.rmtree(d, ignore_errors=True)
                cat_label = st.session_state.get("train_queue_category_label", "法规文件")
                src = st.session_state.get("train_queue_source", "upload")
                extra = {
                    "total_files": total_files,
                    "success_count": success_count,
                    "fail_count": fail_count,
                    "total_chunks": total_chunks,
                    "duration_sec": round(total_time, 2),
                    "category": category,
                    "category_label": cat_label,
                    "overwrite_duplicates": bool(dup_set),
                }
                completed_case_id = st.session_state.get("train_queue_case_id")
                if completed_case_id:
                    extra["case_id"] = completed_case_id
                if src == "directory":
                    extra["dir_path"] = st.session_state.get("train_queue_dir_path", "")
                try:
                    add_operation_log(
                        OP_TYPE_TRAIN_BATCH,
                        agent.collection_name,
                        "",
                        source=src,
                        extra=extra,
                        model_info=get_current_model_info(),
                    )
                except Exception:
                    pass
                for key in list(st.session_state.keys()):
                    if key.startswith("train_queue"):
                        st.session_state.pop(key, None)
                st.session_state["train_show_success"] = True
                st.session_state["train_success_info"] = {
                    "success_count": success_count,
                    "total_files": total_files,
                    "total_chunks": total_chunks,
                    "total_time": total_time,
                    "fail_count": fail_count,
                    "source": src,
                }
                st.experimental_rerun()
            else:
                time.sleep(0.3)
                st.experimental_rerun()

        if uploaded_files and st.button("🚀 开始训练", key="train_btn"):
            category = CATEGORY_VALUES.get(file_category_label, "regulation")
            collection = st.session_state.get("collection_name", "regulations")
            train_case_id = None
            if file_category_label == "项目案例文件":
                sel_case = st.session_state.get("train_upload_case_sel", "")
                if sel_case == "➕ 新建案例":
                    case_name = (st.session_state.get("train_case_name") or "").strip()
                    if not case_name:
                        st.warning("请填写案例名称后再开始训练。")
                        st.stop()
                    dims = _cached_dimension_options()
                    forms_list = dims.get("project_forms", ["Web", "APP", "PC"]) or ["Web", "APP", "PC"]
                    _doc_lang = DOC_LANG_LABEL_TO_VALUE.get(st.session_state.get("train_case_doc_lang") or "中文版", "zh")
                    copy_from_sel = st.session_state.get("train_copy_from_sel", "不复制")
                    copy_case = None
                    if copy_from_sel and copy_from_sel != "不复制":
                        cases_list = _cached_list_project_cases(collection)
                        opts = [_format_case_option(c) for c in cases_list]
                        if copy_from_sel in opts:
                            copy_case = cases_list[opts.index(copy_from_sel)]
                    project_key = ""
                    if copy_case:
                        project_key = (copy_case.get("project_key") or "").strip() or str(copy_case.get("id", ""))
                    train_case_id = create_project_case(
                        collection,
                        case_name,
                        product_name=(st.session_state.get("train_case_product") or "").strip(),
                        registration_country=st.session_state.get("train_case_country") or "",
                        registration_type=st.session_state.get("train_case_type") or "",
                        registration_component=st.session_state.get("train_case_comp") or "",
                        project_form=st.session_state.get("train_case_form") or "",
                        scope_of_application=(st.session_state.get("train_case_scope") or "").strip(),
                        case_name_en=(st.session_state.get("train_case_name_en") or "").strip(),
                        product_name_en=(st.session_state.get("train_case_product_en") or "").strip(),
                        registration_country_en=(st.session_state.get("train_case_country_en") or "").strip(),
                        document_language=_doc_lang,
                        project_key=project_key,
                    )
                    st.success(f"已创建案例「{case_name}」，本批文件将训练到通用知识库并关联此案例。")
                else:
                    cases_list = _cached_list_project_cases(collection)
                    case_opts = ["➕ 新建案例"] + [_format_case_option(c) for c in cases_list]
                    try:
                        idx = case_opts.index(sel_case)
                        train_case_id = cases_list[idx - 1]["id"]
                    except Exception:
                        st.warning("请选择有效案例。")
                        st.stop()

            prep_msg = st.empty()
            prep_msg.info("⏳ 正在准备文件列表（保存并解压上传文件），请勿关闭页面…")
            try:
                items, temp_dirs = _expand_uploads(uploaded_files)
            finally:
                prep_msg.empty()
            if not items:
                st.warning("没有可训练的文件。")
                return

            existing = set(get_existing_file_names(
                    collection,
                    category=category if category == "project_case" else None,
                    case_id=train_case_id if (category == "project_case" and train_case_id is not None) else None,
                ))
            duplicates = [dn for (_, dn, _) in items if dn in existing]

            if duplicates:
                st.session_state["train_pending_items"] = items
                st.session_state["train_pending_temp_dirs"] = temp_dirs
                st.session_state["train_pending_category"] = category
                st.session_state["train_pending_category_label"] = file_category_label
                st.session_state["train_pending_duplicates"] = set(duplicates)
                st.session_state["train_pending_source"] = "upload"
                if train_case_id is not None:
                    st.session_state["train_pending_case_id"] = train_case_id
                st.experimental_rerun()
                return

            # 无重复时也走“每轮一个文件”队列，避免文档多时中途超时中断
            _clear_train_chunked_job()
            st.session_state["train_queue"] = items
            st.session_state["train_queue_index"] = 0
            st.session_state["train_queue_temp_dirs"] = temp_dirs
            st.session_state["train_queue_category"] = category
            st.session_state["train_queue_category_label"] = file_category_label
            st.session_state["train_queue_dup_set"] = set()
            st.session_state["train_queue_log"] = []
            st.session_state["train_queue_success"] = 0
            st.session_state["train_queue_fail"] = 0
            st.session_state["train_queue_chunks"] = 0
            st.session_state["train_queue_start_time"] = time.time()
            st.session_state["train_queue_source"] = "upload"
            if train_case_id is not None:
                st.session_state["train_queue_case_id"] = train_case_id
            st.success(f"已就绪，共 {len(items)} 个文件，即将开始训练…")
            time.sleep(0.8)
            st.experimental_rerun()

    def _step1_tab_directory():
        # 目录训练时在此 tab 显示进度，不在「上传文件训练」页显示
        if (
            st.session_state.get("train_queue")
            and st.session_state.get("train_queue_source") == "directory"
            and st.session_state.get("train_queue_index", 0) < len(st.session_state["train_queue"])
        ):
            queue = st.session_state["train_queue"]
            idx = st.session_state["train_queue_index"]
            total_files = len(queue)
            path, display_name, is_from_archive = queue[idx]
            category = st.session_state.get("train_queue_category", "regulation")
            train_queue_case_id_dir = st.session_state.get("train_queue_case_id")
            dup_set = st.session_state.get("train_queue_dup_set") or set()
            log_lines = list(st.session_state.get("train_queue_log", []))

            st.info(f"🔄 **正在训练第 {idx + 1}/{total_files} 个文件**（从目录），请勿关闭页面。")
            _emb_frac0d, _emb_cap0d = _train_embed_resume_progress_for_ui(idx, display_name)
            if _emb_cap0d:
                st.caption(_emb_cap0d)
            try:
                with st.spinner("正在加载模型与知识库（首次约 10–30 秒，请稍候）…"):
                    agent = init_agent()
            except Exception as e0:
                log_lines.append(f"- :x: **{display_name}** 加载模型/知识库失败，已跳过本文件：{e0}")
                st.session_state["train_queue_fail"] = st.session_state.get("train_queue_fail", 0) + 1
                st.session_state["train_queue_index"] = idx + 1
                st.session_state["train_queue_log"] = log_lines
                st.warning(f"本文件加载失败已跳过，继续训练其余文件。错误：{e0}")
                st.caption("如持续失败，请检查模型服务是否正常、网络配置是否正确。")
                st.experimental_rerun()

            overall_bar = st.progress((idx) / max(total_files, 1))
            overall_text = st.empty()
            status_box = st.empty()
            embed_bar = st.progress(min(1.0, max(0.0, _emb_frac0d)))
            log_display = st.empty()
            overall_text.info(
                f"总进度 {idx + 1}/{total_files} | 成功 {st.session_state.get('train_queue_success', 0)} | "
                f"失败 {st.session_state.get('train_queue_fail', 0)} | 已入库 {st.session_state.get('train_queue_chunks', 0)} 块"
            )
            log_display.markdown("\n".join(log_lines))

            try:
                if display_name in dup_set:
                    try:
                        agent.kb.delete_documents_by_file_name(display_name, case_id=train_queue_case_id_dir if category == "project_case" else None)
                        log_lines.append(f"- 🔄 **{display_name}** 已覆盖旧内容，正在重新训练…")
                    except Exception as e:
                        log_lines.append(f"- ⚠️ **{display_name}** 覆盖前清理失败：{e}")
                ok, chunks, elapsed = _train_single_file(
                    agent,
                    path,
                    display_name,
                    status_box,
                    embed_bar,
                    log_lines,
                    category=category,
                    case_id=train_queue_case_id_dir,
                    queue_idx=idx,
                )
                log_display.markdown("\n".join(log_lines))
                if ok:
                    st.session_state["train_queue_success"] = st.session_state.get("train_queue_success", 0) + 1
                    st.session_state["train_queue_chunks"] = st.session_state.get("train_queue_chunks", 0) + chunks
                    if train_queue_case_id_dir is not None:
                        try:
                            update_knowledge_docs_case_id(agent.collection_name, display_name, train_queue_case_id_dir)
                        except Exception:
                            pass
                else:
                    st.session_state["train_queue_fail"] = st.session_state.get("train_queue_fail", 0) + 1
            except Exception as e:
                log_lines.append(f"- :x: **{display_name}** 异常：{e}")
                st.session_state["train_queue_fail"] = st.session_state.get("train_queue_fail", 0) + 1
                try:
                    log_display.markdown("\n".join(log_lines))
                except Exception:
                    pass
                try:
                    add_operation_log(
                        op_type="train_error",
                        collection=agent.collection_name,
                        file_name=display_name,
                        source=str(path),
                        extra={"error": str(e), "traceback": traceback.format_exc(), "stage": "train", "category": category},
                        model_info=get_current_model_info(),
                    )
                except Exception:
                    pass
            # 目录训练不删除本地文件，此处不执行 unlink
            st.session_state["train_queue_index"] = idx + 1
            st.session_state["train_queue_log"] = log_lines

            if st.session_state["train_queue_index"] >= len(queue):
                total_time = time.time() - st.session_state.get("train_queue_start_time", time.time())
                success_count = st.session_state.get("train_queue_success", 0)
                fail_count = st.session_state.get("train_queue_fail", 0)
                total_chunks = st.session_state.get("train_queue_chunks", 0)
                cat_label = st.session_state.get("train_queue_category_label", "法规文件")
                src = st.session_state.get("train_queue_source", "upload")
                extra = {
                        "total_files": total_files,
                        "success_count": success_count,
                        "fail_count": fail_count,
                        "total_chunks": total_chunks,
                        "duration_sec": round(total_time, 2),
                    "category": category,
                    "category_label": cat_label,
                    "overwrite_duplicates": bool(dup_set),
                    "dir_path": st.session_state.get("train_queue_dir_path", ""),
                }
                completed_case_id_dir = st.session_state.get("train_queue_case_id")
                if completed_case_id_dir:
                    extra["case_id"] = completed_case_id_dir
                try:
                    add_operation_log(
                        OP_TYPE_TRAIN_BATCH,
                        agent.collection_name,
                        "",
                        source=src,
                        extra=extra,
                    model_info=get_current_model_info(),
                )
                except Exception:
                    pass
                for key in list(st.session_state.keys()):
                    if key.startswith("train_queue"):
                        st.session_state.pop(key, None)
                st.session_state["train_show_success"] = True
                st.session_state["train_success_info"] = {
                    "success_count": success_count,
                    "total_files": total_files,
                    "total_chunks": total_chunks,
                    "total_time": total_time,
                    "fail_count": fail_count,
                    "source": src,
                }
                st.experimental_rerun()
            else:
                time.sleep(0.3)
                st.experimental_rerun()

        # 有待处理的重复文件时（目录来源），在稳定位置渲染覆盖/跳过按钮
        if (
            st.session_state.get("train_pending_items")
            and st.session_state.get("train_pending_source") == "directory"
            and not st.session_state.get("train_do_overwrite")
            and not st.session_state.get("train_do_skip")
            and not st.session_state.get("train_queue")
        ):
            _dup_names_dir = st.session_state.get("train_pending_duplicates") or set()
            if _dup_names_dir:
                st.warning("**\u4e0e\u6570\u636e\u5e93\u5bf9\u6bd4**\uff1a\u4ee5\u4e0b\u6587\u4ef6\u540d\u5728\u77e5\u8bc6\u5e93\uff08\u6570\u636e\u5e93\uff09\u4e2d\u5df2\u5b58\u5728\uff0c\u8bf7\u9009\u62e9\u8986\u76d6\u6216\u8df3\u8fc7\u3002")
                _dup_list_dir = sorted(_dup_names_dir) if isinstance(_dup_names_dir, set) else list(_dup_names_dir)
                st.caption(f"\u91cd\u540d\u6587\u4ef6\uff1a{', '.join(_dup_list_dir[:10])}{'...' if len(_dup_list_dir) > 10 else ''}")
                _ddc1, _ddc2 = st.columns(2)
                with _ddc1:
                    if st.button("\u2705 \u8986\u76d6\uff1a\u7528\u6700\u65b0\u6587\u4ef6\u8986\u76d6\u65e7\u5185\u5bb9\u5e76\u8bad\u7ec3", key="dir_confirm_overwrite"):
                        st.session_state["train_do_overwrite"] = True
                        st.experimental_rerun()
                with _ddc2:
                    if st.button("\u23ed\ufe0f \u4e0d\u8986\u76d6\uff1a\u8df3\u8fc7\u91cd\u540d\u6587\u4ef6\uff0c\u4ec5\u8bad\u7ec3\u5176\u4f59\u6587\u4ef6", key="dir_skip_duplicates"):
                        st.session_state["train_do_skip"] = True
                        st.experimental_rerun()

        file_category_label_dir = st.selectbox(
            "文件分类",
            ["法规文件", "程序文件", "项目案例文件", "词条"],
            key="train_dir_category",
            help="选择本目录下文件的类型",
        )
        train_dir_case_id = None
        if file_category_label_dir == "项目案例文件":
            collection_dir = st.session_state.get("collection_name", "regulations")
            dims_dir = _cached_dimension_options()
            countries_dir = dims_dir.get("registration_countries", ["中国", "美国", "欧盟"]) or ["中国", "美国", "欧盟"]
            forms_dir = dims_dir.get("project_forms", ["Web", "APP", "PC"]) or ["Web", "APP", "PC"]
            cases_dir = _cached_list_project_cases(collection_dir)
            _pending_var_d = st.session_state.pop("_pending_variant_dir_case_id", None)
            if _pending_var_d is not None:
                st.session_state["train_dir_case_sel"] = "➕ 新建案例"
                st.session_state["train_dir_copy_from_case_id"] = _pending_var_d
            case_options_dir = ["➕ 新建案例"] + [_format_case_option(c) for c in cases_dir]
            sel_case_dir = st.selectbox("选择过往项目案例", case_options_dir, key="train_dir_case_sel",
                                        help="项目案例为过往经验，训练到通用知识库；第三步审核时按产品名称与适用范围匹配。")
            if sel_case_dir == "➕ 新建案例":
                st.caption("新建案例：描述本目录文件所属的过往项目（支持中/英文）。")
                case_options_copy_dir = [_format_case_option(c) for c in cases_dir]
                copy_default_idx_dir = 0
                if st.session_state.get("train_dir_copy_from_case_id"):
                    try:
                        cid = st.session_state["train_dir_copy_from_case_id"]
                        for i, c in enumerate(cases_dir):
                            if c.get("id") == cid:
                                copy_default_idx_dir = i + 1
                                break
                        st.session_state.pop("train_dir_copy_from_case_id", None)
                    except Exception:
                        pass
                copy_from_sel_dir = st.selectbox(
                    "复制自现有案例（可选）",
                    ["不复制"] + case_options_copy_dir,
                    index=min(copy_default_idx_dir, len(case_options_copy_dir)),
                    key="train_dir_copy_from_sel",
                    help="选择后预填案例信息，可仅改注册国家、文档语言以创建其他国家/语言版本。",
                )
                copy_case_dir = None
                if copy_from_sel_dir and copy_from_sel_dir != "不复制" and copy_from_sel_dir in case_options_copy_dir:
                    copy_case_dir = cases_dir[case_options_copy_dir.index(copy_from_sel_dir)]
                if copy_case_dir:
                    st.caption("已从现有案例预填，请修改「注册国家」「案例文档语言」等以区分为新版本。")
                _def_dir = lambda k, d: (copy_case_dir.get(k) or d) if copy_case_dir else d
                _idx_dir = lambda opts, val: opts.index(val) if val in opts else 0
                p_case_name_dir = st.text_input("案例名称", value=_def_dir("case_name", ""), placeholder="例如：某某血糖仪二类注册案例", key="train_dir_case_name")
                p_case_name_en_dir = st.text_input("案例名称（英文）", value=_def_dir("case_name_en", ""), placeholder="e.g. XXX Blood Glucose Meter Class II", key="train_dir_case_name_en")
                p_product_dir = st.text_input("产品名称", value=_def_dir("product_name", ""), placeholder="该过往项目的产品名称", key="train_dir_case_product")
                p_product_en_dir = st.text_input("产品名称（英文）", value=_def_dir("product_name_en", ""), placeholder="Product name in English", key="train_dir_case_product_en")
                _doc_lang_val_dir = _def_dir("document_language", "zh")
                _doc_lang_label_dir = DOC_LANG_VALUE_TO_LABEL.get(_doc_lang_val_dir, "中文版")
                p_doc_lang_dir = st.selectbox("案例文档语言", DOC_LANG_OPTIONS, index=_idx_dir(DOC_LANG_OPTIONS, _doc_lang_label_dir), key="train_dir_case_doc_lang", help="本目录下文档主要为中文、英文或中英文；与全系统文档语言选项保持一致")
                p_country_dir = st.selectbox("注册国家", countries_dir, index=_idx_dir(countries_dir, _def_dir("registration_country", countries_dir[0] if countries_dir else "")), key="train_dir_case_country")
                p_country_en_dir = st.text_input("注册国家（英文）", value=_def_dir("registration_country_en", ""), placeholder="e.g. China, USA", key="train_dir_case_country_en")
                p_type_dir = st.selectbox("注册类别", REGISTRATION_TYPES, index=_idx_dir(REGISTRATION_TYPES, _def_dir("registration_type", REGISTRATION_TYPES[0])), key="train_dir_case_type")
                p_comp_dir = st.selectbox("注册组成", REGISTRATION_COMPONENTS, index=_idx_dir(REGISTRATION_COMPONENTS, _def_dir("registration_component", REGISTRATION_COMPONENTS[0])), key="train_dir_case_comp")
                p_form_dir = st.selectbox("项目形态", forms_dir, index=_idx_dir(forms_dir, _def_dir("project_form", forms_dir[0] if forms_dir else "")), key="train_dir_case_form")
                p_scope_dir = st.text_area("产品适用范围（可选）", value=_def_dir("scope_of_application", ""), placeholder="该过往项目的适用范围", height=60, key="train_dir_case_scope")
            else:
                try:
                    _idx_dir = lambda opts, val: opts.index(val) if val in opts else 0
                    idx_dir = case_options_dir.index(sel_case_dir)
                    case_dir = cases_dir[idx_dir - 1]
                    train_dir_case_id = case_dir["id"]
                    st.caption(
                        f"将训练到案例：**{case_dir.get('case_name')}**"
                        f"（产品名称：{case_dir.get('product_name') or '—'}，"
                        f"注册国家：{case_dir.get('registration_country')}）"
                    )
                    _cid_dir = case_dir.get("id")
                    with st.expander("✏️ 编辑此案例 & 关联项目", expanded=False):
                        e_case_name_d = st.text_input("案例名称", value=case_dir.get("case_name") or "", key=f"edit_dir_case_name_{_cid_dir}")
                        e_case_name_en_d = st.text_input("案例名称（英文）", value=case_dir.get("case_name_en") or "", key=f"edit_dir_case_name_en_{_cid_dir}")
                        e_product_d = st.text_input("产品名称", value=case_dir.get("product_name") or "", key=f"edit_dir_case_product_{_cid_dir}")
                        e_product_en_d = st.text_input("产品名称（英文）", value=case_dir.get("product_name_en") or "", key=f"edit_dir_case_product_en_{_cid_dir}")
                        _country_val_d = (case_dir.get("registration_country") or "").strip() or (countries_dir[0] if countries_dir else "")
                        _country_idx_d = _idx_dir(countries_dir, _country_val_d)
                        e_country_d = st.selectbox("注册国家", countries_dir, index=min(_country_idx_d, len(countries_dir) - 1) if countries_dir else 0, key=f"edit_dir_case_country_{_cid_dir}")
                        e_country_en_d = st.text_input("注册国家（英文）", value=case_dir.get("registration_country_en") or "", placeholder="e.g. China, USA", key=f"edit_dir_case_country_en_{_cid_dir}")
                        _type_val_d = (case_dir.get("registration_type") or "").strip() or (REGISTRATION_TYPES[0] if REGISTRATION_TYPES else "")
                        _type_idx_d = _idx_dir(REGISTRATION_TYPES, _type_val_d)
                        e_type_d = st.selectbox("注册类别", REGISTRATION_TYPES, index=min(_type_idx_d, len(REGISTRATION_TYPES) - 1) if REGISTRATION_TYPES else 0, key=f"edit_dir_case_type_{_cid_dir}")
                        _comp_val_d = (case_dir.get("registration_component") or "").strip() or (REGISTRATION_COMPONENTS[0] if REGISTRATION_COMPONENTS else "")
                        _comp_idx_d = _idx_dir(REGISTRATION_COMPONENTS, _comp_val_d)
                        e_comp_d = st.selectbox("注册组成", REGISTRATION_COMPONENTS, index=min(_comp_idx_d, len(REGISTRATION_COMPONENTS) - 1) if REGISTRATION_COMPONENTS else 0, key=f"edit_dir_case_comp_{_cid_dir}")
                        _form_val_d = (case_dir.get("project_form") or "").strip() or (forms_dir[0] if forms_dir else "")
                        _form_idx_d = _idx_dir(forms_dir, _form_val_d)
                        e_form_d = st.selectbox("项目形态", forms_dir, index=min(_form_idx_d, len(forms_dir) - 1) if forms_dir else 0, key=f"edit_dir_case_form_{_cid_dir}")
                        _doc_lang_d = case_dir.get("document_language") or ""
                        _doc_lang_label_d = DOC_LANG_VALUE_TO_LABEL.get(_doc_lang_d, "不指定")
                        _doc_lang_idx_d = DOC_LANG_OPTIONS.index(_doc_lang_label_d) if _doc_lang_label_d in DOC_LANG_OPTIONS else 0
                        e_doc_lang_d = st.selectbox("案例文档语言", DOC_LANG_OPTIONS, index=_doc_lang_idx_d, key=f"edit_dir_case_doc_lang_{_cid_dir}")
                        e_scope_d = st.text_area("产品适用范围", value=case_dir.get("scope_of_application") or "", height=60, key=f"edit_dir_case_scope_{_cid_dir}")
                        other_cases_dir = [c for c in cases_dir if c.get("id") != case_dir.get("id")]
                        link_options_dir = ["不关联（独立）"] + [_format_case_option(c) for c in other_cases_dir]
                        _link_idx_dir = 0
                        if case_dir.get("project_key"):
                            for i, c in enumerate(other_cases_dir):
                                if str(c.get("id")) == str(case_dir.get("project_key")) or (c.get("project_key") and str(c.get("project_key")) == str(case_dir.get("project_key"))):
                                    _link_idx_dir = i + 1
                                    break
                        e_link_dir = st.selectbox("关联到同一项目", link_options_dir, index=min(_link_idx_dir, len(link_options_dir) - 1), key=f"edit_dir_case_link_{_cid_dir}", help="与所选案例归为同一项目（多国家/多语言版本）。")
                        if st.button("保存案例", key=f"edit_dir_case_save_{_cid_dir}"):
                            project_key_d = ""
                            if e_link_dir and e_link_dir != "不关联（独立）" and e_link_dir in link_options_dir:
                                idx_ld = link_options_dir.index(e_link_dir)
                                if idx_ld > 0:
                                    linked_d = other_cases_dir[idx_ld - 1]
                                    project_key_d = (linked_d.get("project_key") or "").strip() or str(linked_d.get("id", ""))
                            update_project_case(
                                case_dir["id"],
                                case_name=e_case_name_d.strip() or None,
                                case_name_en=e_case_name_en_d.strip() or None,
                                product_name=e_product_d.strip() or None,
                                product_name_en=e_product_en_d.strip() or None,
                                registration_country=e_country_d.strip() or None,
                                registration_country_en=e_country_en_d.strip() or None,
                                registration_type=e_type_d,
                                registration_component=e_comp_d,
                                project_form=e_form_d,
                                document_language=DOC_LANG_LABEL_TO_VALUE.get(e_doc_lang_d, ""),
                                scope_of_application=e_scope_d.strip() or None,
                                project_key=project_key_d,
                            )
                            st.success("已保存")
                            st.experimental_rerun()
                    _case_files_dir = get_project_case_file_names(collection_dir, case_dir["id"])
                    if _case_files_dir:
                        _preview_dir = ", ".join(_case_files_dir[:5]) + ("..." if len(_case_files_dir) > 5 else "")
                        st.caption(f"📁 **已入库文件**：共 **{len(_case_files_dir)}** 个 — {_preview_dir}")
                    else:
                        st.caption("📁 **已入库文件**：暂无")
                    if st.button("📋 创建本案例的其他国家/语言版本", key="btn_variant_dir", help="切换到新建案例并预填本案例信息，仅改注册国家、文档语言即可创建新版本"):
                        st.session_state["_pending_variant_dir_case_id"] = case_dir["id"]
                        _streamlit_rerun()
                    _pending_del_d = st.session_state.get("train_del_case_dir_id")
                    if _pending_del_d == case_dir["id"]:
                        _nf_d = len(_case_files_dir or [])
                        st.warning(
                            f"**确认删除模板案例？** 当前案例下共有 **{_nf_d}** 个已入库文件；"
                            f"确认后将**永久删除**案例记录，并清空知识库中该案例关联的全部文档块与向量，**不可恢复**。"
                        )
                        if _case_files_dir:
                            _pv_d = ", ".join(_case_files_dir[:15]) + ("…" if len(_case_files_dir) > 15 else "")
                            st.caption(f"文件清单（节选）：{_pv_d}")
                        _d1, _d2 = st.columns(2)
                        with _d1:
                            if st.button("✅ 确认删除", key=f"confirm_del_case_d_{case_dir['id']}"):
                                try:
                                    from src.core.knowledge_base import KnowledgeBase

                                    KnowledgeBase(collection_dir).delete_documents_by_case_id(int(case_dir["id"]))
                                except Exception as _exd:
                                    st.error(f"清理知识库失败：{_exd}")
                                else:
                                    delete_project_case(int(case_dir["id"]))
                                    st.session_state.pop("train_del_case_dir_id", None)
                                    _cached_list_project_cases.clear()
                                    st.success("已删除案例及其全部入库文件")
                                    _streamlit_rerun()
                        with _d2:
                            if st.button("取消", key=f"cancel_del_case_d_{case_dir['id']}"):
                                st.session_state.pop("train_del_case_dir_id", None)
                                _streamlit_rerun()
                    elif st.button("🗑️ 删除此案例", key="del_case_dir"):
                        st.session_state["train_del_case_dir_id"] = case_dir["id"]
                        _streamlit_rerun()
                except Exception:
                    pass
        dir_path = st.text_input(
            "输入目录路径",
            value=str(settings.training_path),
            help="服务器上的目录路径，将递归加载所有支持格式的文件",
        )
        if st.button("🚀 从目录训练", key="train_dir_btn"):
            category_dir = CATEGORY_VALUES.get(file_category_label_dir, "regulation")
            if not Path(dir_path).exists():
                st.error(f"目录不存在：{dir_path}")
                return

            train_case_id_dir = None
            if file_category_label_dir == "项目案例文件":
                sel_case_dir = st.session_state.get("train_dir_case_sel", "")
                if sel_case_dir == "➕ 新建案例":
                    case_name_dir = (st.session_state.get("train_dir_case_name") or "").strip()
                    if not case_name_dir:
                        st.warning("请填写案例名称后再从目录训练。")
                        st.stop()
                    dims_dir = _cached_dimension_options()
                    forms_list_dir = dims_dir.get("project_forms", ["Web", "APP", "PC"]) or ["Web", "APP", "PC"]
                    collection_dir = st.session_state.get("collection_name", "regulations")
                    _doc_lang_dir = DOC_LANG_LABEL_TO_VALUE.get(st.session_state.get("train_dir_case_doc_lang") or "中文版", "zh")
                    copy_from_sel_dir = st.session_state.get("train_dir_copy_from_sel", "不复制")
                    copy_case_dir = None
                    if copy_from_sel_dir and copy_from_sel_dir != "不复制":
                        cases_list_dir = _cached_list_project_cases(collection_dir)
                        opts_dir = [_format_case_option(c) for c in cases_list_dir]
                        if copy_from_sel_dir in opts_dir:
                            copy_case_dir = cases_list_dir[opts_dir.index(copy_from_sel_dir)]
                    project_key_dir = ""
                    if copy_case_dir:
                        project_key_dir = (copy_case_dir.get("project_key") or "").strip() or str(copy_case_dir.get("id", ""))
                    train_case_id_dir = create_project_case(
                        collection_dir,
                        case_name_dir,
                        product_name=(st.session_state.get("train_dir_case_product") or "").strip(),
                        registration_country=st.session_state.get("train_dir_case_country") or "",
                        registration_type=st.session_state.get("train_dir_case_type") or "",
                        registration_component=st.session_state.get("train_dir_case_comp") or "",
                        project_form=st.session_state.get("train_dir_case_form") or "",
                        scope_of_application=(st.session_state.get("train_dir_case_scope") or "").strip(),
                        case_name_en=(st.session_state.get("train_dir_case_name_en") or "").strip(),
                        product_name_en=(st.session_state.get("train_dir_case_product_en") or "").strip(),
                        registration_country_en=(st.session_state.get("train_dir_case_country_en") or "").strip(),
                        document_language=_doc_lang_dir,
                        project_key=project_key_dir,
                    )
                    st.success(f"已创建案例「{case_name_dir}」，本目录文件将训练到通用知识库并关联此案例。")
                else:
                    cases_list_dir = _cached_list_project_cases(st.session_state.get("collection_name", "regulations"))
                    case_opts_dir = ["➕ 新建案例"] + [_format_case_option(c) for c in cases_list_dir]
                    try:
                        idx_dir = case_opts_dir.index(sel_case_dir)
                        train_case_id_dir = cases_list_dir[idx_dir - 1]["id"]
                    except Exception:
                        st.warning("请选择有效案例。")
                        st.stop()

            scan_status = st.empty()
            scan_status.info("🔍 正在扫描目录中的文件...")
            files = _scan_directory_files(dir_path)

            if not files:
                scan_status.warning("⚠️ 目录中没有找到支持格式的文件")
                return

            scan_status.success(f"扫描完成，发现 {len(files)} 个文件。")
            items = [(str(fp), fp.name, False) for fp in files]
            collection = st.session_state.get("collection_name", "regulations")
            # 与数据库（knowledge_docs）对比，取当前已入库文件名
            existing = set(get_existing_file_names(
                    collection,
                    category=category_dir if category_dir == "project_case" else None,
                    case_id=train_case_id_dir if (category_dir == "project_case" and train_case_id_dir is not None) else None,
                ))
            duplicates = [dn for (_, dn, _) in items if dn in existing]
            if duplicates:
                st.session_state["train_pending_items"] = items
                st.session_state["train_pending_temp_dirs"] = []
                st.session_state["train_pending_category"] = category_dir
                st.session_state["train_pending_category_label"] = file_category_label_dir
                st.session_state["train_pending_duplicates"] = set(duplicates)
                st.session_state["train_pending_source"] = "directory"
                st.session_state["train_pending_dir_path"] = dir_path
                if train_case_id_dir is not None:
                    st.session_state["train_pending_case_id"] = train_case_id_dir
                st.experimental_rerun()
                return
            # 无重名，直接入队
            _clear_train_chunked_job()
            st.session_state["train_queue"] = items
            st.session_state["train_queue_index"] = 0
            st.session_state["train_queue_temp_dirs"] = []
            st.session_state["train_queue_category"] = category_dir
            st.session_state["train_queue_category_label"] = file_category_label_dir
            st.session_state["train_queue_dup_set"] = set()
            st.session_state["train_queue_log"] = []
            st.session_state["train_queue_success"] = 0
            st.session_state["train_queue_fail"] = 0
            st.session_state["train_queue_chunks"] = 0
            st.session_state["train_queue_start_time"] = time.time()
            st.session_state["train_queue_source"] = "directory"
            st.session_state["train_queue_dir_path"] = dir_path
            if train_case_id_dir is not None:
                st.session_state["train_queue_case_id"] = train_case_id_dir
            st.experimental_rerun()

    def _step1_tab_generate():
        st.markdown(
            "基于第一步训练的法规知识库，让 AI 自动生成结构化审核点清单。\n"
            "也可以上传已有的**基础审核点文档**，AI 会在此基础上优化补充。"
        )

        collection = st.session_state.get("collection_name", "regulations")
        _kb_stats = _cached_knowledge_stats(collection)
        reg_count = _kb_stats.get("total_chunks", 0)
        if reg_count == 0:
            st.warning("⚠️ 法规知识库为空，请先在「上传文件训练」或「从目录训练」中导入法规/程序文件。")

        # 生成审核点提示词配置：默认填充已有（数据库或内置），支持修改；仅本次生效，也可保存为默认
        with st.expander("📝 生成审核点提示词（可修改，生成时使用当前内容；可保存为默认）", expanded=False):
            gen_default = _get_prompt_default("checklist_generate_prompt")
            opt_default = _get_prompt_default("checklist_optimize_prompt")
            gen_current = get_prompt_by_key("checklist_generate_prompt")
            opt_current = get_prompt_by_key("checklist_optimize_prompt")
            gen_fill = (gen_current or gen_default) if gen_current else gen_default
            opt_fill = (opt_current or opt_default) if opt_current else opt_default
            st.caption("留空则使用内置默认。占位符：生成用 {context}、{base_checklist_section}；优化用 {context}、{base_checklist}。")
            gen_prompt_edit = st.text_area(
                "生成审核点（全新生成）",
                value=gen_fill,
                height=220,
                key="gen_checklist_prompt_tab3",
            )
            opt_prompt_edit = st.text_area(
                "优化审核点（在已有清单基础上优化）",
                value=opt_fill,
                height=220,
                key="opt_checklist_prompt_tab3",
            )
            if st.button("将当前提示词保存为默认", key="save_checklist_prompts_tab3"):
                try:
                    update_prompt_by_key("checklist_generate_prompt", gen_prompt_edit.strip() or None)
                    update_prompt_by_key("checklist_optimize_prompt", opt_prompt_edit.strip() or None)
                    st.success("已保存为默认，下次生成及「提示词配置」页将使用当前内容。")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"保存失败：{e}")

        base_file = st.file_uploader(
            "（可选）上传基础审核点文档，AI 将在此基础上优化（支持 PDF/Word/Excel/TXT/Markdown）",
            type=["pdf", "docx", "doc", "xlsx", "xls", "txt", "md"],
            key="base_checklist_uploader",
        )
        gen_cl_doc_lang = st.selectbox(
            "审核点适用文档语言（项目案例相关）",
            DOC_LANG_OPTIONS,
            key="gen_checklist_doc_lang",
            help="仅影响项目案例相关审核点的生成语言。法规/程序知识库生成的审核点始终为所有语言通用。选「不指定」即为通用。",
        )
        _gen_dims = _cached_dimension_options()
        _gen_countries = _gen_dims.get("registration_countries", ["\u4e2d\u56fd", "\u7f8e\u56fd", "\u6b27\u76df"]) or ["\u4e2d\u56fd", "\u7f8e\u56fd", "\u6b27\u76df"]
        gen_sel_countries = st.multiselect(
            "\u6ce8\u518c\u56fd\u5bb6\uff08\u53ef\u9009\uff0c\u751f\u6210\u5ba1\u6838\u70b9\u65f6\u6309\u56fd\u5bb6\u2192\u6cd5\u89c4\u5173\u952e\u8bcd\u6269\u5c55\u68c0\u7d22\uff0c\u5982 CE\u2192MDR\uff09",
            _gen_countries,
            default=[],
            key="gen_checklist_countries",
            help="\u9009\u62e9\u6ce8\u518c\u56fd\u5bb6\u540e\uff0c\u751f\u6210\u5ba1\u6838\u70b9\u65f6\u4f1a\u81ea\u52a8\u6309\u300c\u7ef4\u5ea6\u9009\u9879\u914d\u7f6e\u300d\u4e2d\u7684\u56fd\u5bb6\u2192\u6cd5\u89c4\u5173\u952e\u8bcd\u6620\u5c04\u6269\u5c55\u68c0\u7d22\u77e5\u8bc6\u5e93\uff0c\u4e0d\u533a\u5206\u5927\u5c0f\u5199\u3002",
        )
        gen_reg_type = st.selectbox(
            "适用注册类别（可选）",
            ["不指定"] + REGISTRATION_TYPES,
            key="gen_checklist_reg_type",
            help="选择后生成审核点时会注入审核尺度提示（严格程度 Ⅲ>Ⅱb>Ⅱa>Ⅱ>Ι），便于清单更贴合二类/三类项目。",
        )
        # 显示知识库规模与预估审核点数量
        if reg_count > 0:
            from src.core.checklist_generator import estimate_checklist_scale
            _scale = estimate_checklist_scale(
                total_files=_kb_stats.get("total_files", 0),
                total_chunks=_kb_stats.get("total_chunks", 0),
            )
            st.info(
                f"📊 知识库规模：**{_kb_stats.get('total_files', 0)}** 个文件 / **{reg_count}** 块"
                f"（{_scale['scale_label']}）— "
                f"预计生成 **≥ {_scale['target_points']}** 个审核点"
                f"（检索参考 {_scale['max_context_docs']} 条）。"
                f"训练更多法规/程序/案例文件可提升审核点数量与覆盖面。"
            )

        _default_cl_name = f"审核点清单-{collection}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        checklist_name = st.text_input("审核点清单名称", value=_default_cl_name, key="cl_name", help="默认带当前时间，多次生成不会覆盖历史记录")

        if st.button("🤖 生成审核点清单", key="gen_checklist_btn"):
            if reg_count == 0:
                st.error("请先训练法规知识库再生成审核点。")
            else:
                agent = init_agent()
                base_text = None
                base_file_name = ""
                if base_file is not None:
                    base_file_name = base_file.name
                    try:
                        tmp_path = _save_uploaded_file(base_file)
                        docs = load_single_file(tmp_path)
                        base_text = "\n".join(d.page_content for d in docs)
                        Path(tmp_path).unlink(missing_ok=True)
                    except Exception as e:
                        st.warning(f"基础文档加载失败({e})，将全新生成。")
                        base_text = None

                gen_status = st.empty()
                gen_progress_bar = st.empty()
                gen_batch_info = st.empty()
                t0 = time.time()
                try:
                    _gen_doc_lang = st.session_state.get("gen_checklist_doc_lang") or "不指定"
                    _gen_doc_lang_val = DOC_LANG_LABEL_TO_VALUE.get(_gen_doc_lang, "")

                    def _gen_progress(batch_idx, total, msg):
                        if total > 1:
                            gen_progress_bar.progress(min(batch_idx / total, 1.0), text=f"批次进度 {batch_idx}/{total}")
                            gen_batch_info.info(f"🤖 {msg}")
                        else:
                            gen_status.info(f"🤖 {msg}")

                    with st.spinner("AI 正在分析法规知识库并生成审核点清单，请耐心等待…"):
                        gen_status.info("🤖 正在调用 AI 生成审核点清单...")
                        checklist = agent.generate_checklist(
                            base_checklist=base_text,
                            provider=st.session_state.get("current_provider"),
                            generate_prompt_override=gen_prompt_edit.strip() or None,
                            optimize_prompt_override=opt_prompt_edit.strip() or None,
                            document_language=_gen_doc_lang_val or None,
                            kb_stats=_kb_stats,
                            registration_countries=gen_sel_countries or None,
                            registration_type=gen_reg_type if (gen_reg_type and gen_reg_type != "不指定") else None,
                            progress_callback=_gen_progress,  # 分批进度
                        )
                    gen_progress_bar.empty()
                    gen_batch_info.empty()
                    elapsed = time.time() - t0
                    mi = get_current_model_info()

                    cl_id = save_audit_checklist(
                        collection=agent.collection_name,
                        name=checklist_name,
                        checklist=checklist,
                        base_file=base_file_name,
                        model_info=mi,
                        status="draft",
                        document_language=_gen_doc_lang_val,
                    )
                    add_operation_log(
                        op_type=OP_TYPE_GENERATE_CHECKLIST,
                        collection=agent.collection_name,
                        file_name=checklist_name,
                        source=base_file_name or "auto_generate",
                        extra={
                            "checklist_id": cl_id,
                            "total_points": len(checklist),
                            "has_base": base_text is not None,
                            "duration_sec": round(elapsed, 2),
                        },
                        model_info=mi,
                    )
                    gen_status.success(
                        f"✅ 生成完成！共 {len(checklist)} 个审核点，耗时 {_format_time(elapsed)}。"
                        f"已保存为「{checklist_name}」(ID:{cl_id})。请到「② 审核点管理」页面查看和训练。"
                    )

                    _render_checklist_preview(checklist)

                except Exception as e:
                    err_msg = str(e)
                    gen_status.error(f"❌ 生成失败：{err_msg}")
                    if "TIMEOUT" in err_msg.upper():
                        st.warning(
                            "⏱️ AI 处理超时（已自动分批，但单批仍超时），建议：\n"
                            "1. **切换至 OpenAI / Ollama 模式**：在设置中修改 provider，大模型直接调用无代理超时限制\n"
                            "2. **检查网络连接**：Cursor Agent 需要稳定的网络连接\n"
                            "3. **稍后重试**：服务端可能暂时繁忙"
                        )
                    st.code(traceback.format_exc(), language="text")

    _step1_tab_labels = ["📤 上传文件训练", "📂 从目录训练", "📝 生成审核点"]
    _st_run_tabs_or_pick(
        _step1_tab_labels,
        radio_label="第一步子功能",
        session_key="step1_tabs",
        tab_bodies=[_step1_tab_upload, _step1_tab_directory, _step1_tab_generate],
    )


def _render_checklist_preview(checklist: list, flat: bool = False, key_suffix: str = ""):
    """预览审核点清单。flat=True 时用 markdown 展示，避免嵌套 expander 报错"""
    if not checklist:
        st.info("清单为空。")
        return

    severity_icons = {"high": "🔴", "medium": "🟡", "low": "🔵", "info": "ℹ️"}
    counts = {}
    for p in checklist:
        s = p.get("severity", "info")
        counts[s] = counts.get(s, 0) + 1

    if flat:
        # 在列/expander 内不能嵌套 columns，用单行文案展示统计
        st.caption(
            f"总审核点 **{len(checklist)}** · "
            f"🔴 高 {counts.get('high', 0)} · 🟡 中 {counts.get('medium', 0)} · "
            f"🔵 低 {counts.get('low', 0)} · ℹ️ 提示 {counts.get('info', 0)}"
        )
    else:
        cols = st.columns(5)
        cols[0].metric("总审核点", len(checklist))
        cols[1].metric("🔴 高", counts.get("high", 0))
        cols[2].metric("🟡 中", counts.get("medium", 0))
        cols[3].metric("🔵 低", counts.get("low", 0))
        cols[4].metric("ℹ️ 提示", counts.get("info", 0))

    for idx, p in enumerate(checklist):
        sev = p.get("severity", "info")
        icon = severity_icons.get(sev, "ℹ️")
        header = f"{icon} {p.get('id', '')} | {p.get('name', '')} [{p.get('category', '')}]"
        body = (
            f"**描述：** {p.get('description', '')}\n\n"
            f"**法规依据：** {p.get('regulation_ref', '')}\n\n"
            f"**检查方法：** {p.get('check_method', '')}\n\n"
            f"**严重程度：** `{sev}`"
        )
        docs = p.get("applicable_docs", [])
        if docs:
            body += f"\n\n**适用文档：** {', '.join(docs)}"
        if flat:
            st.markdown(f"### {header}")
            if len(body) > 600:
                st.text_area("", value=body, height=min(280, max(100, len(body) // 5)), disabled=True, key=f"cl_preview_{key_suffix}_{idx}")
            else:
                st.markdown(body)
            st.markdown("---")
        else:
            with st.expander(header, expanded=False):
                if len(body) > 600:
                    st.text_area("", value=body, height=min(280, max(100, len(body) // 5)), disabled=True, key=f"cl_preview_{key_suffix}_{idx}")
                else:
                    st.markdown(body)

    cl_json = json.dumps(checklist, ensure_ascii=False, indent=2)
    st.download_button(
        "📥 下载审核点清单 (JSON)",
        data=cl_json,
        file_name="audit_checklist.json",
        mime="application/json",
        key=f"dl_checklist_preview_{key_suffix}" if key_suffix else "dl_checklist_preview",
    )


def _load_default_checklist_有源软件二类():
    """加载内置通用审核点（有源医疗器械软件二类，含网络接口与体系）。返回 list 或 None。"""
    path = _root / "config" / "default_checklist_有源软件二类.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else None
    except Exception:
        return None


def render_step2_page():
    """第二步：审核点管理与训练"""
    st.header("② 审核点管理 & 训练")
    st.markdown(
        "**第二步**：查看、编辑第一步生成的审核点清单，确认后将其训练到审核点知识库。\n"
        "也可以直接上传审核点 JSON 文件进行导入；**医疗器械软件通用要求**建议使用下方「内置通用审核点」加载并训练，无需放在自定义审核要求中。"
    )

    if not _require_provider():
        return

    agent = init_agent()
    collection = st.session_state.get("collection_name", "regulations")

    def _step2_tab_manage():
        # 审核点训练重名确认：若用户点了「训练此清单」且名称已存在，在此确认覆盖或跳过
        if st.session_state.get("train_cl_confirm_id") is not None:
            confirm_id = st.session_state["train_cl_confirm_id"]
            confirm_name = st.session_state.get("train_cl_confirm_name", "")
            confirm_data = st.session_state.get("train_cl_confirm_data", [])
            st.warning(f"**与数据库对比**：清单「{confirm_name}」已在审核点知识库（数据库）中存在，请选择覆盖或跳过。")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ 覆盖：用当前清单覆盖并训练", key="cl_confirm_overwrite"):
                    try:
                        agent.checkpoint_kb.delete_documents_by_file_name(confirm_name)
                    except Exception:
                        pass
                    _train_checklist(agent, confirm_data, confirm_id, confirm_name)
                    for k in ("train_cl_confirm_id", "train_cl_confirm_name", "train_cl_confirm_data"):
                        st.session_state.pop(k, None)
                    st.experimental_rerun()
            with c2:
                if st.button("⏭️ 不覆盖：跳过，不训练", key="cl_confirm_skip"):
                    for k in ("train_cl_confirm_id", "train_cl_confirm_name", "train_cl_confirm_data"):
                        st.session_state.pop(k, None)
                    st.info("已跳过该清单。")
                    st.experimental_rerun()

        default_cl = _load_default_checklist_有源软件二类()
        if default_cl is not None:
            with st.expander("📌 内置通用审核点（有源医疗器械软件二类，含网络接口与体系）", expanded=False):
                st.caption(
                    "以下为医疗器械软件通用要求，建议训练为审核点知识库后使用；「自定义审核要求」仅用于公司/项目特有补充（如内部规范、本批审核重点）。"
                )
                st.markdown(f"共 **{len(default_cl)}** 条审核点，覆盖：注册资料一致性、软件描述与生命周期、HTTP/MQTT/蓝牙接口及体系文件、技术要求与检验、说明书与标签等。")
                col_load, col_train = st.columns(2)
                with col_load:
                    if st.button("📥 加载为清单（可编辑后再训练）", key="load_default_cl"):
                        name = f"有源软件二类-通用审核点-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        cl_id = save_audit_checklist(
                            collection=collection,
                            name=name,
                            checklist=default_cl,
                            base_file="内置",
                            model_info="",
                            status="draft",
                        )
                        st.success(f"已保存为「{name}」(ID:{cl_id})，请在下方列表中训练。")
                        st.experimental_rerun()
                with col_train:
                    if st.button("📥 加载并直接训练", key="load_and_train_default_cl"):
                        default_name = "有源软件二类-通用审核点"
                        # 与数据库（checkpoint_docs）对比
                        existing_cp = set(get_existing_checkpoint_file_names(collection))
                        if default_name in existing_cp:
                            st.session_state["default_cl_confirm"] = True
                            st.experimental_rerun()
                        else:
                            with st.spinner("正在将内置通用审核点训练到审核点知识库…"):
                                _train_checklist(agent, default_cl, None, default_name)
                            st.success("内置通用审核点已训练入库；建议再保存为清单以便后续查看：点击「加载为清单」即可。")
                            st.experimental_rerun()
                if st.session_state.get("default_cl_confirm"):
                    default_name = "有源软件二类-通用审核点"
                    st.warning("**与数据库对比**：该清单名称已在审核点知识库（数据库）中存在，请选择覆盖或跳过。")
                    dc1, dc2 = st.columns(2)
                    with dc1:
                        if st.button("✅ 覆盖并训练", key="default_cl_overwrite"):
                            try:
                                agent.checkpoint_kb.delete_documents_by_file_name(default_name)
                            except Exception:
                                pass
                            with st.spinner("正在训练…"):
                                _train_checklist(agent, default_cl, None, default_name)
                            st.session_state.pop("default_cl_confirm", None)
                            st.success("已覆盖并训练完成。")
                            st.experimental_rerun()
                    with dc2:
                        if st.button("⏭️ 跳过", key="default_cl_skip"):
                            st.session_state.pop("default_cl_confirm", None)
                            st.info("已跳过。")
                            st.experimental_rerun()

        checklists = get_audit_checklists(collection=collection, limit=50)

        if not checklists:
            st.info(
                "暂无审核点清单。请先到「① 法规训练 & 生成审核点」页面生成，"
                "或在「导入审核点」标签页上传。"
            )
        else:
            # 批量训练入库：操作栏
            st.caption("勾选下方清单后可**批量训练入库**（建议选择未训练或需更新的清单）。")
            _c1, _c2, _c3, _ = st.columns([1, 1, 2, 4])
            with _c1:
                if st.button("全选", key="batch_sel_all"):
                    for r in checklists:
                        st.session_state[f"sel_cl_{r['id']}"] = True
                    st.experimental_rerun()
            with _c2:
                if st.button("取消全选", key="batch_sel_none"):
                    for r in checklists:
                        st.session_state[f"sel_cl_{r['id']}"] = False
                    st.experimental_rerun()
            with _c3:
                if st.button("🚀 批量训练所选清单入库", key="batch_train_btn"):
                    selected_ids = [r["id"] for r in checklists if st.session_state.get(f"sel_cl_{r['id']}", False)]
                    if not selected_ids:
                        st.warning("请先勾选要训练的清单。")
                    else:
                        _batch_train_checklists(agent, collection, selected_ids)
                        st.experimental_rerun()
            _n = sum(1 for r in checklists if st.session_state.get(f"sel_cl_{r['id']}", False))
            if _n > 0:
                st.caption(f"已选 **{_n}** 条清单")

            for rec in checklists:
                cl_id = rec.get("id")
                cl_name = rec.get("name", "")
                cl_status = rec.get("status", "draft")
                total = rec.get("total_points", 0)
                ts = rec.get("created_at", "")
                mi = rec.get("model_info", "")
                status_icon = "✅" if cl_status == "trained" else "📝"
                _cl_lang = rec.get("document_language") or ""
                _cl_lang_label = DOC_LANG_VALUE_TO_LABEL.get(_cl_lang, "")
                # 仅当明确指定了适用语言时展示（不展示「适用不指定」）
                if _cl_lang and _cl_lang in ("zh", "en", "both") and _cl_lang_label:
                    label = f"{status_icon} {cl_name} | {total}个审核点 | {cl_status} | {ts} | 适用{_cl_lang_label}"
                else:
                    label = f"{status_icon} {cl_name} | {total}个审核点 | {cl_status} | {ts}"
                if mi:
                    label += f" | {mi}"

                with st.expander(label, expanded=False):
                    st.checkbox("选", key=f"sel_cl_{cl_id}", help="勾选后可批量训练入库")
                    cl_data = rec.get("checklist", [])
                    _render_checklist_preview(cl_data, flat=True, key_suffix=str(cl_id))

                    col_train, col_edit, col_del = st.columns(3)
                    with col_train:
                        if cl_status == "trained":
                            st.success("已训练入库")
                        if st.button("🚀 训练此清单", key=f"train_cl_{cl_id}"):
                            existing_cp = set(get_existing_checkpoint_file_names(collection))
                            if cl_name and cl_name in existing_cp:
                                st.session_state["train_cl_confirm_id"] = cl_id
                                st.session_state["train_cl_confirm_name"] = cl_name
                                st.session_state["train_cl_confirm_data"] = cl_data
                                st.experimental_rerun()
                            else:
                                _train_checklist(agent, cl_data, cl_id, cl_name)

                    with col_edit:
                        st.download_button(
                            "📥 导出 JSON",
                            data=json.dumps(cl_data, ensure_ascii=False, indent=2),
                            file_name=f"checklist_{cl_id}.json",
                            mime="application/json",
                            key=f"dl_cl_{cl_id}",
                        )

                    with col_del:
                        if st.button("🗑️ 删除", key=f"del_cl_{cl_id}"):
                            delete_audit_checklist(cl_id)
                            st.success(f"已删除清单「{cl_name}」")
                            st.experimental_rerun()

    def _step2_tab_import():
        st.markdown("上传审核点 JSON 文件（格式同生成的清单 JSON）或纯文本审核点文档。")

        import_file = st.file_uploader(
            "选择审核点文件",
            type=["json", "txt", "md", "docx", "pdf"],
            key="import_checklist_uploader",
        )
        import_doc_lang = st.selectbox(
            "适用文档语言（项目案例相关）",
            DOC_LANG_OPTIONS,
            key="import_checklist_doc_lang",
            help="仅影响项目案例相关审核点的导入/解析语言。法规/程序审核点始终为所有语言通用。选「不指定」即为通用。",
        )
        import_name = st.text_input("清单名称", value=f"导入的审核点-{datetime.now().strftime('%Y%m%d_%H%M%S')}", key="import_cl_name", help="默认带当前时间，多次导入不会覆盖")

        if import_file and st.button("📥 导入", key="import_cl_btn"):
            try:
                if import_file.name.endswith(".json"):
                    raw = import_file.read().decode("utf-8")
                    cl_data = json.loads(raw)
                    if not isinstance(cl_data, list):
                        st.error("JSON 文件必须是数组格式。")
                        return
                else:
                    tmp_path = _save_uploaded_file(import_file)
                    docs = load_single_file(tmp_path)
                    text = "\n".join(d.page_content for d in docs)
                    Path(tmp_path).unlink(missing_ok=True)
                    st.info("非 JSON 文件将尝试让 AI 解析为审核点清单...")
                    _import_doc_lang_val = DOC_LANG_LABEL_TO_VALUE.get(st.session_state.get("import_checklist_doc_lang") or "不指定", "")
                    with st.spinner("AI 正在解析文档为审核点清单..."):
                        cl_data = agent.generate_checklist(
                            base_checklist=text,
                            provider=st.session_state.get("current_provider"),
                            document_language=_import_doc_lang_val or None,
                        )

                mi = get_current_model_info()
                _import_doc_lang_save = DOC_LANG_LABEL_TO_VALUE.get(st.session_state.get("import_checklist_doc_lang") or "不指定", "")
                cl_id = save_audit_checklist(
                    collection=collection,
                    name=import_name,
                    checklist=cl_data,
                    base_file=import_file.name,
                    model_info=mi,
                    status="draft",
                    document_language=_import_doc_lang_save,
                )
                st.success(f"✅ 已导入 {len(cl_data)} 个审核点，清单ID: {cl_id}")
                _render_checklist_preview(cl_data)
            except Exception as e:
                st.error(f"导入失败：{e}")
                st.code(traceback.format_exc(), language="text")

    def _step2_tab_projects():
        _render_projects_tab(agent, collection)

    _step2_tab_labels = ["📋 审核点清单管理", "📤 导入审核点", "📁 项目与专属资料"]
    _st_run_tabs_or_pick(
        _step2_tab_labels,
        radio_label="第二步子功能",
        session_key="step2_tabs",
        tab_bodies=[_step2_tab_manage, _step2_tab_import, _step2_tab_projects],
    )


def _render_projects_tab(agent, collection: str):
    """项目列表 + 新建/编辑/删除 + 上传项目专属资料（用缓存避免切换选项时卡顿）"""
    _pending_msg = st.session_state.pop("projects_success_message", None)
    if _pending_msg:
        st.success(_pending_msg)

    dims = _cached_dimension_options()
    countries = dims.get("registration_countries", ["中国", "美国", "欧盟"])
    forms = dims.get("project_forms", ["Web", "APP", "PC"])

    projects = _cached_list_projects(collection)
    project_options = ["➕ 新建项目"] + [f"{p['name']} (ID:{p['id']})" for p in projects]

    sel = st.selectbox("选择项目", project_options, key="proj_sel")
    if sel == "➕ 新建项目":
        st.subheader("新建项目")
        with st.form("new_project_form"):
            p_name = st.text_input("项目名称", key="np_name")
            p_code = st.text_input("项目编号（可选）", placeholder="例如：OXGWIS（用于文件名等前缀替换）", key="np_project_code")
            p_name_en = st.text_input("项目名称（英文）", placeholder="Project name in English", key="np_name_en")
            p_product = st.text_input("产品名称（可选）", placeholder="与项目名称一并加入审核点、一致性核对", key="np_product")
            p_product_en = st.text_input("产品名称（英文）", placeholder="Product name in English", key="np_product_en")
            p_model = st.text_input("型号（可选，Model）", placeholder="中英文均可；字段名称不区分大小写，取值区分大小写、精确匹配（含空格）", key="np_model")
            p_model_en = st.text_input("型号（英文，可选）", placeholder="Model in English", key="np_model_en")
            p_country = st.selectbox("注册国家", countries, key="np_country")
            p_country_en = st.text_input("注册国家（英文）", placeholder="e.g. China, USA", key="np_country_en")
            p_type = st.selectbox("注册类别", REGISTRATION_TYPES, key="np_type")
            p_comp = st.selectbox("注册组成", REGISTRATION_COMPONENTS, key="np_comp")
            p_form = st.selectbox("项目形态", forms, key="np_form")
            p_scope = st.text_area("产品适用范围（可选）", placeholder="审核时要求文档描述内容不超出此范围", height=80, key="np_scope")
            if st.form_submit_button("创建"):
                if p_name.strip():
                    pid = create_project(
                        collection, p_name.strip(), p_country, p_type, p_comp, p_form,
                        scope_of_application=(p_scope or "").strip(),
                        product_name=(p_product or "").strip(),
                        name_en=(p_name_en or "").strip(),
                        product_name_en=(p_product_en or "").strip(),
                        registration_country_en=(p_country_en or "").strip(),
                        model=(p_model or "").strip(),
                        model_en=(p_model_en or "").strip(),
                        project_code=(p_code or "").strip(),
                    )
                    st.session_state["projects_success_message"] = f"✅ 已创建项目「{p_name.strip()}」，ID: {pid}"
                    st.experimental_rerun()
                else:
                    st.warning("请填写项目名称")
    else:
        # 解析选中的项目 ID
        try:
            idx = project_options.index(sel)
            proj = projects[idx - 1]
            pid = proj["id"]
        except Exception:
            st.warning("请选择有效项目")
            return

        st.subheader(f"项目：{proj['name']}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("注册国家", proj.get("registration_country", ""))
        c2.metric("注册类别", proj.get("registration_type", ""))
        c3.metric("注册组成", proj.get("registration_component", ""))
        c4.metric("项目形态", proj.get("project_form", ""))
        if proj.get("product_name") or proj.get("product_name_en"):
            st.caption(f"产品名称：{proj.get('product_name') or '—'} / {proj.get('product_name_en') or '—'}（已加入审核点/一致性核对）")
        if proj.get("model") or proj.get("model_en"):
            st.caption(f"型号（Model）：{proj.get('model') or '—'} / {proj.get('model_en') or '—'}（字段名称不区分大小写，取值区分大小写、精确匹配含空格）")
        if proj.get("project_code"):
            st.caption(f"项目编号：{proj.get('project_code') or '—'}")

        _idx_proj = lambda opts, val: opts.index(val) if (val and val in opts) else 0
        with st.expander("✏️ 编辑项目（支持中/英文）", expanded=False):
            with st.form(f"edit_proj_{pid}"):
                e_name = st.text_input("项目名称", value=proj.get("name") or "", key=f"ep_name_{pid}")
                e_code = st.text_input("项目编号（可选）", value=proj.get("project_code") or "", placeholder="例如：OXGWIS（用于文件名等前缀替换）", key=f"ep_project_code_{pid}")
                e_name_en = st.text_input("项目名称（英文）", value=proj.get("name_en") or "", placeholder="Project name in English", key=f"ep_name_en_{pid}")
                e_product = st.text_input("产品名称（可选）", value=proj.get("product_name") or "", placeholder="与项目名称一并加入审核点、一致性核对", key=f"ep_product_{pid}")
                e_product_en = st.text_input("产品名称（英文）", value=proj.get("product_name_en") or "", placeholder="Product name in English", key=f"ep_product_en_{pid}")
                e_model = st.text_input("型号（可选，Model）", value=proj.get("model") or "", placeholder="字段名称不区分大小写，取值区分大小写、精确匹配（含空格）", key=f"ep_model_{pid}")
                e_model_en = st.text_input("型号（英文，可选）", value=proj.get("model_en") or "", placeholder="Model in English", key=f"ep_model_en_{pid}")
                _ec_idx = min(_idx_proj(countries, proj.get("registration_country") or ""), len(countries) - 1) if countries else 0
                e_country = st.selectbox("注册国家", countries, index=_ec_idx, key=f"ep_country_{pid}")
                e_country_en = st.text_input("注册国家（英文）", value=proj.get("registration_country_en") or "", placeholder="e.g. China", key=f"ep_country_en_{pid}")
                _et_idx = min(_idx_proj(REGISTRATION_TYPES, proj.get("registration_type") or ""), len(REGISTRATION_TYPES) - 1) if REGISTRATION_TYPES else 0
                e_type = st.selectbox("注册类别", REGISTRATION_TYPES, index=_et_idx, key=f"ep_type_{pid}")
                _ecomp_idx = min(_idx_proj(REGISTRATION_COMPONENTS, proj.get("registration_component") or ""), len(REGISTRATION_COMPONENTS) - 1) if REGISTRATION_COMPONENTS else 0
                e_comp = st.selectbox("注册组成", REGISTRATION_COMPONENTS, index=_ecomp_idx, key=f"ep_comp_{pid}")
                _ef_idx = min(_idx_proj(forms, proj.get("project_form") or ""), len(forms) - 1) if forms else 0
                e_form = st.selectbox("项目形态", forms, index=_ef_idx, key=f"ep_form_{pid}")
                e_scope = st.text_area("产品适用范围", value=proj.get("scope_of_application") or "", placeholder="审核时要求文档描述内容不超出此范围", height=80, key=f"ep_scope_{pid}")
                if st.form_submit_button("保存"):
                    update_project(
                        pid,
                        name=e_name,
                        registration_country=e_country,
                        registration_type=e_type,
                        registration_component=e_comp,
                        project_form=e_form,
                        scope_of_application=e_scope.strip() if e_scope else "",
                        product_name=e_product.strip() if e_product else "",
                        project_code=e_code.strip() if e_code else "",
                        name_en=e_name_en.strip() if e_name_en else "",
                        product_name_en=e_product_en.strip() if e_product_en else "",
                        registration_country_en=e_country_en.strip() if e_country_en else "",
                        model=e_model.strip() if e_model else "",
                        model_en=e_model_en.strip() if e_model_en else "",
                    )
                    st.session_state["projects_success_message"] = "✅ 项目已更新"
                    st.experimental_rerun()
        if st.button("🗑️ 删除项目", key=f"del_proj_{pid}"):
            try:
                agent.get_project_kb(pid).clear()
            except Exception:
                pass
            delete_project(pid)
            st.session_state["projects_success_message"] = "✅ 项目已删除"
            st.experimental_rerun()

        try:
            pstats = get_project_knowledge_stats(pid)
            st.caption(f"项目专属资料：{pstats.get('total_files', 0)} 个文件 / {pstats.get('total_chunks', 0)} 块")
        except Exception:
            pass

        # 识别系统功能（安装包 / URL），审核时与文档一致性核对
        with st.expander("🔧 识别系统功能（与文档一致性核对）", expanded=False):
            st.caption("根据安装包或系统 URL 识别实际功能，审核时将核对待审文档中的功能描述是否与之一致。")
            proj_detail = get_project(pid) or {}
            sys_func = (proj_detail.get("system_functionality_text") or "").strip()
            sys_src = (proj_detail.get("system_functionality_source") or "").strip()
            if sys_func:
                st.markdown("**当前已识别的系统功能**" + (f"（来源：{sys_src}）" if sys_src else "") + "：")
                st.text_area("系统功能描述", value=sys_func, height=120, disabled=True, key=f"sys_func_display_{pid}")
            else:
                st.info("尚未识别系统功能，可通过下方「导入安装包」或「输入 URL」识别。")

            st.markdown("**方式一：导入安装包识别**")
            pkg_file = st.file_uploader(
                "选择安装包或压缩包（支持 .zip / .tar / .gz / .apk / .exe / .msi 等）",
                type=["zip", "tar", "gz", "tgz", "apk", "exe", "msi"],
                accept_multiple_files=False,
                key=f"pkg_upload_{pid}",
            )
            if pkg_file and st.button("识别系统功能（安装包）", key=f"identify_pkg_{pid}"):
                with st.spinner("正在解析安装包并调用 AI 识别系统功能…"):
                    try:
                        tmp_path = _save_uploaded_file(pkg_file)
                        result = agent.identify_system_functionality_from_package(
                            pid, tmp_path, st.session_state.get("current_provider")
                        )
                        Path(tmp_path).unlink(missing_ok=True)
                        if result and not result.startswith("文件不存在") and not result.startswith("未能识别"):
                            st.success("已识别并保存系统功能，审核时将用于与文档一致性核对。")
                            st.experimental_rerun()
                        else:
                            st.info("识别完成，但未得到有效的系统功能描述。请尝试补充说明文档或使用「输入 URL」方式录入。")
                            if result:
                                st.caption(result)
                    except Exception as e:
                        st.error(f"识别失败：{e}")
                        st.code(traceback.format_exc(), language="text")

            st.markdown("**方式二：输入 URL 识别**")
            url_addr = st.text_input("系统/页面 URL", placeholder="https://...", key=f"sys_url_{pid}")
            url_user = st.text_input("账号（可选，用于 Basic 认证）", placeholder="留空则无需登录", key=f"sys_user_{pid}")
            url_pass = st.text_input("密码（可选）", type="password", key=f"sys_pass_{pid}")
            url_captcha = st.text_input("验证码（若页面弹出验证码时填写）", placeholder="留空则不需要", key=f"sys_captcha_{pid}")
            if st.button("识别系统功能（URL）", key=f"identify_url_{pid}"):
                if not (url_addr and url_addr.strip()):
                    st.warning("请填写 URL。")
                else:
                    with st.spinner("正在请求页面并调用 AI 识别系统功能…"):
                        try:
                            result = agent.identify_system_functionality_from_url(
                                pid,
                                url_addr.strip(),
                                username=url_user.strip() if url_user else "",
                                password=url_pass.strip() if url_pass else "",
                                provider=st.session_state.get("current_provider"),
                                captcha=url_captcha.strip() if url_captcha else "",
                            )
                            if result and not result.startswith("未填写") and not result.startswith("请求失败") and not result.startswith("未能识别") and not result.startswith("CAPTCHA_REQUIRED"):
                                st.success("已识别并保存系统功能，审核时将用于与文档一致性核对。")
                                st.experimental_rerun()
                            elif result and result.startswith("CAPTCHA_REQUIRED"):
                                st.warning("检测到页面需要验证码，请在上方「验证码」输入框中填写验证码后再次点击「识别系统功能（URL）」重试。")
                                snippet = result.split(":", 1)[-1].strip()[:200] if ":" in result else ""
                                if snippet:
                                    st.caption(snippet)
                            else:
                                st.info("识别完成，但未得到有效的系统功能描述。若页面需验证码，请在上方填写验证码后重试。")
                                if result:
                                    st.caption(result)
                        except Exception as e:
                            st.error(f"识别失败：{e}")
                            st.code(traceback.format_exc(), language="text")

        st.markdown("**上传项目专属资料**（技术要求、说明书等，将用于按项目审核时检索）")
        proj_files = st.file_uploader(
            "选择文件",
            type=["pdf", "docx", "doc", "xlsx", "xls", "txt", "md", "zip", "tar", "gz", "tgz"],
            accept_multiple_files=True,
            key=f"proj_upload_{pid}",
        )
        # 执行“覆盖”或“不覆盖”后的训练（从 session 取 items）
        if (st.session_state.get("proj_do_overwrite") or st.session_state.get("proj_do_skip")) and st.session_state.get("proj_pending_pid") == pid:
            items = list(st.session_state.get("proj_pending_items", []))
            temp_dirs = list(st.session_state.get("proj_pending_temp_dirs", []))
            dup_set = set(st.session_state.get("proj_pending_duplicates", [])) if st.session_state.get("proj_do_overwrite") else set()
            if st.session_state.get("proj_do_skip"):
                items = [(p, dn, ar) for (p, dn, ar) in items if dn not in st.session_state.get("proj_pending_duplicates", set())]
            for k in ("proj_do_overwrite", "proj_do_skip", "proj_pending_pid", "proj_pending_items", "proj_pending_temp_dirs", "proj_pending_duplicates"):
                st.session_state.pop(k, None)
            if not items:
                st.warning("全部文件均重名已跳过，无文件需要训练。")
            else:
                status_box = st.empty()
                bar = st.empty()
                log_display = st.empty()
                log_lines = []
                total = len(items)
                success_count = 0
                fail_count = 0
                try:
                    with st.spinner("⏳ 项目专属资料训练中，请勿切换页面…"):
                        for idx, (path, display_name, is_from_archive) in enumerate(items):
                            pct = int((idx + 1) / total * 100)
                            bar.progress(min(pct, 99))
                            status_box.info(f"📂 [{idx+1}/{total}] 正在处理 {display_name}…")
                            def _cb(done, tot):
                                p = int((idx + done / max(tot, 1)) / total * 100)
                                bar.progress(min(p, 99))
                                status_box.info(f"📂 [{idx+1}/{total}] {display_name} — 向量化 {done}/{tot}")
                            def _on_load(msg):
                                status_box.info(f"📂 [{idx+1}/{total}] {display_name} — {msg}")
                            try:
                                if display_name in dup_set:
                                    try:
                                        agent.get_project_kb(pid).delete_documents_by_file_name(display_name)
                                    except Exception:
                                        pass
                                agent.train_project_docs(pid, path, file_name=display_name, callback=_cb, on_loading=_on_load)
                                log_lines.append(f"- ✅ {display_name}")
                                success_count += 1
                                add_operation_log(OP_TYPE_TRAIN_PROJECT, collection=collection, file_name=display_name, source=str(path), extra={"project_id": pid, "project_name": proj.get("name")}, model_info=get_current_model_info())
                            except Exception as e:
                                fail_count += 1
                                log_lines.append(f"- ❌ {display_name}: {e}")
                                add_operation_log(OP_TYPE_TRAIN_PROJECT_ERROR, collection=collection, file_name=display_name, source=str(path), extra={"project_id": pid, "project_name": proj.get("name"), "error": str(e), "traceback": traceback.format_exc(), "stage": "single_file", "success_so_far": success_count, "fail_count": fail_count, "total": total}, model_info=get_current_model_info())
                            finally:
                                if not is_from_archive:
                                    Path(path).unlink(missing_ok=True)
                            log_display.markdown("\n".join(log_lines))
                    bar.progress(100)
                    status_box.success(f"✅ 项目专属资料训练完成，成功 {success_count}/{total}" + (f"，失败 {fail_count} 个" if fail_count else ""))
                    if success_count > 0:
                        try:
                            basic_info = agent.extract_and_save_project_basic_info(pid, st.session_state.get("current_provider"))
                            if basic_info:
                                log_lines.append("- 📋 已从项目资料中提取基本信息并写入知识库，审核时将用于一致性核对。")
                                log_display.markdown("\n".join(log_lines))
                        except Exception as _e:
                            log_lines.append(f"- ⚠️ 提取项目基本信息时出错：{_e}")
                except Exception as e:
                    status_box.error(f"训练中断: {e}")
                    add_operation_log(OP_TYPE_TRAIN_PROJECT_ERROR, collection=collection, file_name="", source="batch", extra={"project_id": pid, "project_name": proj.get("name"), "error": str(e), "traceback": traceback.format_exc(), "stage": "interrupted", "success_count": success_count, "fail_count": fail_count, "total": total, "reason": "训练被中断或自动取消"}, model_info=get_current_model_info())
                finally:
                    for d in temp_dirs:
                        shutil.rmtree(d, ignore_errors=True)
                st.experimental_rerun()
        # 项目训练重名：若已存在同名文件，先确认覆盖或跳过（与数据库 project_knowledge_docs 对比）
        elif st.session_state.get("proj_pending_pid") == pid and st.session_state.get("proj_pending_items"):
            items_pending = st.session_state["proj_pending_items"]
            # 再次与数据库对比，确保重名列表为最新
            existing_proj_now = set(get_existing_project_file_names(pid))
            dup_set = set(dn for (_, dn, _) in items_pending if dn in existing_proj_now)
            st.session_state["proj_pending_duplicates"] = dup_set
            st.warning("**与数据库对比**：以下文件名在本项目知识库（数据库）中已存在，请选择覆盖或跳过。")
            st.caption(f"重名文件：{', '.join(list(dup_set)[:10])}{'...' if len(dup_set) > 10 else ''}")
            pc1, pc2 = st.columns(2)
            with pc1:
                if st.button("✅ 覆盖：用最新文件覆盖并训练", key=f"proj_overwrite_{pid}"):
                    st.session_state["proj_do_overwrite"] = True
                    st.experimental_rerun()
            with pc2:
                if st.button("⏭️ 不覆盖：跳过重名，仅训练其余文件", key=f"proj_skip_{pid}"):
                    st.session_state["proj_do_skip"] = True
                    st.experimental_rerun()
        elif proj_files and st.button("🚀 训练到本项目", key=f"proj_train_btn_{pid}"):
            items, temp_dirs = _expand_uploads(proj_files)
            if not items:
                st.warning("没有可训练的文件")
                return
            # 与数据库（project_knowledge_docs）对比，取当前该项目已入库文件名
            existing_proj = set(get_existing_project_file_names(pid))
            duplicates = [dn for (_, dn, _) in items if dn in existing_proj]
            if duplicates:
                st.session_state["proj_pending_pid"] = pid
                st.session_state["proj_pending_items"] = items
                st.session_state["proj_pending_temp_dirs"] = temp_dirs
                st.session_state["proj_pending_duplicates"] = set(duplicates)
                st.experimental_rerun()
            status_box = st.empty()
            bar = st.empty()
            log_display = st.empty()
            log_lines = []
            total = len(items)
            success_count = 0
            fail_count = 0
            try:
                with st.spinner("⏳ 项目专属资料训练中，请勿切换页面…"):
                    for idx, (path, display_name, is_from_archive) in enumerate(items):
                        pct = int((idx + 1) / total * 100)
                        bar.progress(min(pct, 99))
                        status_box.info(f"📂 [{idx+1}/{total}] 正在处理 {display_name}…")

                        def _cb(done, tot):
                            p = int((idx + done / max(tot, 1)) / total * 100)
                            bar.progress(min(p, 99))
                            status_box.info(f"📂 [{idx+1}/{total}] {display_name} — 向量化 {done}/{tot}")

                        def _on_load(msg):
                            status_box.info(f"📂 [{idx+1}/{total}] {display_name} — {msg}")

                        try:
                            agent.train_project_docs(
                                pid, path,
                                file_name=display_name,
                                callback=_cb,
                                on_loading=_on_load,
                            )
                            log_lines.append(f"- ✅ {display_name}")
                            success_count += 1
                            add_operation_log(
                                OP_TYPE_TRAIN_PROJECT,
                                collection=collection,
                                file_name=display_name,
                                source=str(path),
                                extra={"project_id": pid, "project_name": proj.get("name")},
                                model_info=get_current_model_info(),
                            )
                        except Exception as e:
                            fail_count += 1
                            tb = traceback.format_exc()
                            log_lines.append(f"- ❌ {display_name}: {e}")
                            status_box.error(f"失败: {display_name} — {e}")
                            add_operation_log(
                                OP_TYPE_TRAIN_PROJECT_ERROR,
                                collection=collection,
                                file_name=display_name,
                                source=str(path),
                                extra={
                                    "project_id": pid,
                                    "project_name": proj.get("name"),
                                    "error": str(e),
                                    "traceback": tb,
                                    "stage": "single_file",
                                    "success_so_far": success_count,
                                    "fail_count": fail_count,
                                    "total": total,
                                },
                                model_info=get_current_model_info(),
                            )
                        finally:
                            if not is_from_archive:
                                Path(path).unlink(missing_ok=True)
                        log_display.markdown("\n".join(log_lines))

                bar.progress(100)
                status_box.success(f"✅ 项目专属资料训练完成，成功 {success_count}/{total}" + (f"，失败 {fail_count} 个" if fail_count else ""))
                if success_count > 0:
                    try:
                        basic_info = agent.extract_and_save_project_basic_info(pid, st.session_state.get("current_provider"))
                        if basic_info:
                            log_lines.append("- 📋 已从项目资料中提取基本信息并写入知识库，审核时将用于一致性核对。")
                            log_display.markdown("\n".join(log_lines))
                    except Exception as _e:
                        log_lines.append(f"- ⚠️ 提取项目基本信息时出错：{_e}")
                        log_display.markdown("\n".join(log_lines))
            except Exception as e:
                tb = traceback.format_exc()
                status_box.error(f"训练中断: {e}")
                log_lines.append(f"- ⚠️ 训练中断: {e}")
                add_operation_log(
                    OP_TYPE_TRAIN_PROJECT_ERROR,
                    collection=collection,
                    file_name="",
                    source="batch",
                    extra={
                        "project_id": pid,
                        "project_name": proj.get("name"),
                        "error": str(e),
                        "traceback": tb,
                        "stage": "interrupted",
                        "success_count": success_count,
                        "fail_count": fail_count,
                        "total": total,
                        "reason": "训练被中断或自动取消",
                    },
                    model_info=get_current_model_info(),
                )
            finally:
                for d in temp_dirs:
                    shutil.rmtree(d, ignore_errors=True)
            for line in log_lines:
                st.markdown(line)
            st.experimental_rerun()


def _batch_train_checklists(agent, collection: str, selected_ids: list):
    """将多条审核点清单批量训练到审核点知识库；重名视为已存在则跳过。"""
    existing_cp = set(get_existing_checkpoint_file_names(collection))
    trained = []
    skipped = []
    for i, cl_id in enumerate(selected_ids):
        rec = get_audit_checklist_by_id(cl_id)
        if not rec:
            continue
        cl_name = rec.get("name", "")
        cl_data = rec.get("checklist", [])
        if not cl_data:
            skipped.append((cl_name or f"ID:{cl_id}", "空清单"))
            continue
        if cl_name and cl_name in existing_cp:
            skipped.append((cl_name, "已存在，跳过"))
            continue
        try:
            with st.spinner(f"正在训练 ({i + 1}/{len(selected_ids)}): {cl_name}"):
                count = agent.train_checklist(cl_data, callback=None, file_name=cl_name)
                update_audit_checklist(cl_id, cl_data, status="trained")
                mi = get_current_model_info()
                add_operation_log(
                    op_type=OP_TYPE_TRAIN_CHECKLIST,
                    collection=collection,
                    file_name=cl_name,
                    source=f"checklist_id:{cl_id}",
                    extra={"checklist_id": cl_id, "total_points": len(cl_data), "chunks": count, "batch": True},
                    model_info=mi,
                )
                existing_cp.add(cl_name)
                trained.append((cl_name, count))
        except Exception as e:
            skipped.append((cl_name, str(e)))
    trained_count = len(trained)
    skip_count = len(skipped)
    if trained_count:
        st.success(f"批量训练完成：**{trained_count}** 个清单已入库" + (f"，{skip_count} 个跳过。" if skip_count else "。"))
    if skipped:
        for name, reason in skipped:
            st.caption(f"跳过「{name}」：{reason}")


def _train_checklist(agent, checklist: list, cl_id: int, cl_name: str):
    """将审核点清单训练到审核点知识库"""
    if not checklist:
        st.warning("清单为空，无法训练。")
        return

    status_box = st.empty()
    bar = st.empty()
    bar.progress(0)
    t0 = time.time()

    def on_batch(done, total):
        pct = int(done / total * 100)
        bar.progress(pct)
        status_box.info(f"🔄 训练审核点 {done}/{total} ({pct}%)")

    try:
        with st.spinner(f"正在将 {len(checklist)} 个审核点训练到审核点知识库..."):
            count = agent.train_checklist(checklist, callback=on_batch, file_name=cl_name)
        elapsed = time.time() - t0
        bar.progress(100)
        status_box.success(f"✅ 训练完成！{count} 个文档块已入库，耗时 {_format_time(elapsed)}")

        if cl_id is not None:
            update_audit_checklist(cl_id, checklist, status="trained")

        mi = get_current_model_info()
        add_operation_log(
            op_type=OP_TYPE_TRAIN_CHECKLIST,
            collection=agent.collection_name,
            file_name=cl_name,
            source=f"checklist_id:{cl_id}" if cl_id is not None else "内置通用审核点",
            extra={"checklist_id": cl_id, "total_points": len(checklist), "chunks": count, "duration_sec": round(elapsed, 2)},
            model_info=mi,
        )
    except Exception as e:
        status_box.error(f"❌ 训练失败：{e}")
        st.code(traceback.format_exc(), language="text")


def render_step3_page():
    """第三步：文档审核"""
    _ap_t0 = time.perf_counter()
    st.header("③ 文档审核")
    st.markdown(
        "**第三步**：上传待审核的注册文档，AI 将根据**审核点知识库**（第二步训练结果）及可选的**项目专属要求**识别审核点。"
    )
    if "review_reports" not in st.session_state or not st.session_state.get("review_reports"):
        st.caption("💡 完成一次审核后，本页下方将出现 **「📋 审核报告」**，按「批次→文档→问题」表格展示，并可按文档或按批次导出待办 CSV。")

    if not _require_provider():
        return

    collection = st.session_state.get("collection_name", "regulations")

    # 先渲染审核模式，避免切换时重复执行 init_agent（按项目审核用缓存拉取项目列表，减轻卡顿）
    project_id = None
    review_context = None
    review_mode = _st_radio_compat("审核模式", ["仅通用审核", "按项目审核"], key="review_mode")
    if review_mode == "按项目审核":
        projects = _cached_list_projects(collection)
        if not projects:
            st.info("请先在「② 审核点管理 & 训练」→「项目与专属资料」中创建项目。")
        else:
            dims = _cached_dimension_options()
            countries = dims.get("registration_countries", ["中国", "美国", "欧盟"]) or ["中国", "美国", "欧盟"]
            forms = dims.get("project_forms", ["Web", "APP", "PC"]) or ["Web", "APP", "PC"]
            proj_names = [p["name"] for p in projects]
            selected_name = st.selectbox("选择项目", proj_names, key="review_proj_name")
            proj = next((p for p in projects if p["name"] == selected_name), None)
            if proj:
                project_id = proj["id"]
                st.caption("审核维度（可多选、可临时修改，用于识别适用的法规/程序/项目案例）")
                c1, c2 = st.columns(2)
                with c1:
                    _country = proj.get("registration_country")
                    _type = proj.get("registration_type")
                    sel_countries = st.multiselect("注册国家", countries, default=[_country] if _country in countries else [], key="rev_countries")
                    sel_types = st.multiselect("适用注册类别", REGISTRATION_TYPES, default=[_type] if _type in REGISTRATION_TYPES else [], key="rev_types", help="与项目注册类别一致；按项目审核时将按此自动匹配审核点（仅使用适用该类别的审核点）。")
                with c2:
                    _comp = proj.get("registration_component")
                    _form = proj.get("project_form")
                    sel_components = st.multiselect("注册组成", REGISTRATION_COMPONENTS, default=[_comp] if _comp in REGISTRATION_COMPONENTS else [], key="rev_components")
                    sel_forms = st.multiselect("项目形态", forms, default=[_form] if _form in forms else [], key="rev_forms")
                review_context = {
                    "registration_country": sel_countries or [_country or ""],
                    "registration_type": sel_types or [_type or ""],
                    "registration_component": sel_components or [_comp or ""],
                    "project_form": sel_forms or [_form or ""],
                    "_project_name": selected_name,
                    "_product_name": proj.get("product_name") or "",
                    "_model": proj.get("model") or "",
                    "_model_en": proj.get("model_en") or "",
                }
    # 自动匹配过往项目案例（与「按项目审核」独立；用缓存避免切换选项时卡顿）
    cases = _cached_list_project_cases(collection)
    auto_match_case = False
    if cases:
        auto_match_case = st.checkbox(
            "自动匹配过往项目案例（产品名称+适用范围优先，否则按维度匹配）",
            value=True,
            key="review_auto_match_case",
            help="第一步训练的过往项目案例，审核时按产品名称/适用范围或注册维度自动匹配。匹配到的案例元数据会注入审核上下文，提升审核针对性。",
        )
        if auto_match_case:
            st.caption(f"已有 **{len(cases)}** 个过往项目案例；开始审核时将自动用首份文档匹配。")
            st.selectbox(
                "项目案例审核语言",
                DOC_LANG_OPTIONS,
                key="review_doc_lang",
                help="仅在匹配到项目案例时生效：按所选语言规范审核。法规/程序知识库审核始终为所有语言通用，无需选择。",
            )

    # 在需要 agent 时再初始化（避免切换模式时重复初始化导致卡顿）
    # 用缓存统计判断审核点库是否为空，避免为纯前端操作初始化 Agent（init_agent 延后到点击「开始审核」时）
    cp_count = _cached_checkpoint_stats(collection).get("document_count", 0)
    if cp_count == 0:
        st.warning("⚠️ 审核点知识库为空，请先完成「① 法规训练 & 生成审核点」和「② 审核点管理 & 训练」。")

    st.markdown("---")
    # 文档审核提示词：默认填充已有（数据库或内置），支持修改；审核时使用当前内容，可保存为默认
    with st.expander("📝 文档审核提示词（可修改，审核时使用当前内容；可保存为默认）", expanded=False):
        sys_default = _get_prompt_default("review_system_prompt")
        usr_default = _get_prompt_default("review_user_prompt")
        review_extra_fill = get_review_extra_instructions() or ""
        review_sys_fill = (get_review_system_prompt() or sys_default) if get_review_system_prompt() else sys_default
        review_usr_fill = (get_review_user_prompt() or usr_default) if get_review_user_prompt() else usr_default
        st.caption("留空则使用内置默认。用户提示词占位符：{context}、{file_name}、{document_content}。")
        review_extra_edit = st.text_area(
            "自定义审核要求（追加到审核上下文）",
            value=review_extra_fill,
            height=100,
            placeholder="例如：重点核对软件版本号、产品名称与技术要求一致…",
            key="review_extra_edit_step3",
        )
        review_sys_edit = st.text_area(
            "审核系统提示词（System Prompt）",
            value=review_sys_fill,
            height=180,
            placeholder="留空则使用内置默认",
            key="review_sys_edit_step3",
        )
        review_usr_edit = st.text_area(
            "审核用户提示词模板（User Prompt）",
            value=review_usr_fill,
            height=200,
            placeholder="留空则使用内置默认",
            key="review_usr_edit_step3",
        )
        if st.button("将当前提示词保存为默认", key="save_review_prompts_step3"):
            try:
                update_review_extra_instructions(review_extra_edit.strip() or "")
                update_review_prompts(
                    system_prompt=review_sys_edit.strip() or None,
                    user_prompt=review_usr_edit.strip() or None,
                )
                st.success("已保存为默认，下次审核及「提示词配置」页将使用当前内容。")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"保存失败：{e}")

    def _step3_tab_upload_review():
        nonlocal review_context
        _retry_ok = st.session_state.pop("review_retry_ok_msg", None)
        if _retry_ok:
            st.success(_retry_ok)
        _retry_partial = st.session_state.pop("review_retry_partial_msg", None)
        if _retry_partial:
            st.warning(_retry_partial)
        _batch_status = st.session_state.pop("review_batch_status_msg", None)
        if _batch_status:
            st.info(_batch_status)
        with st.expander("🔗 金山文档在线审核（粘贴链接即可审核，无需下载到本地）", expanded=False):
            from src.core.kdocs_client import has_api_credentials as _kdocs_has_api
            if _kdocs_has_api():
                st.caption("已配置金山文档开放平台 API，将通过平台接口提取正文。")
            else:
                st.caption("未配置开放平台 API — 将**直接下载文件并在本地解析**，支持 docx / pdf / xlsx / txt 等格式。只需填写下载地址即可审核。")
            kdocs_download_url = st.text_input(
                "文档下载地址（必填）",
                placeholder="粘贴文档的直接下载链接",
                key="kdocs_download_url",
                help="在金山文档中选择「分享」→ 复制「下载链接」，或右键文件 → 复制下载地址。系统将自动下载并解析文档内容用于审核。",
            )
            kdocs_view_url = st.text_input(
                "在线查看链接（选填，便于对照）",
                placeholder="https://www.kdocs.cn/l/...（选填，用于在页面直接打开文档对照）",
                key="kdocs_view_url",
                help="填写后可在审核结果下方直接打开在线文档对照查看。",
            )
            kdocs_filename = st.text_input("文档名称（含扩展名，选填）", value="", placeholder="如 产品说明书.docx（不填则自动识别）", key="kdocs_filename")
            kdocs_ref_files = st.file_uploader(
                "参考文件（可选：本次审核将按参考文件要求对照核查）",
                type=["pdf", "docx", "doc", "xlsx", "xls", "txt", "md", "zip", "tar", "gz", "tgz"],
                accept_multiple_files=True,
                key="kdocs_review_reference_uploader",
                help="可上传程序文件、技术要求、规范、追溯制度等作为本次审核的对照依据。系统会摘录其内容并注入审核上下文。",
            )
            if st.button("🚀 开始审核", key="kdocs_review_btn"):
                download_url = (kdocs_download_url or "").strip()
                view_url = (kdocs_view_url or "").strip()
                fn = (kdocs_filename or "").strip()
                if not download_url:
                    st.warning("请填写「文档下载地址」。")
                else:
                    try:
                        from src.core.kdocs_client import fetch_plaintext_from_url
                        with st.spinner("正在下载并解析文档内容…"):
                            text = fetch_plaintext_from_url(download_url, fn)
                        if not (text or "").strip():
                            st.warning("解析到的正文为空，未执行 AI 审核。请确认下载地址正确且文档非空。")
                        else:
                            if not fn:
                                from src.core.kdocs_client import _guess_filename_from_url
                                fn = _guess_filename_from_url(download_url, "document")
                            agent = init_agent()
                            ctx = dict(review_context or {})
                            ctx["document_language"] = DOC_LANG_LABEL_TO_VALUE.get(st.session_state.get("review_doc_lang") or "不指定", "")
                            # 参考文件：注入审核上下文
                            try:
                                if kdocs_ref_files:
                                    _ritems, _rdirs = _expand_uploads(kdocs_ref_files)
                                    _ref_blocks = []
                                    _ref_cap = 18000
                                    _used = 0
                                    for _rp, _rdn, _rarch in _ritems:
                                        try:
                                            _docs = load_single_file(_rp)
                                            _txt = "\n\n".join(getattr(d, "page_content", str(d)) for d in _docs) if _docs else ""
                                        except Exception:
                                            _txt = ""
                                        if not _txt.strip():
                                            continue
                                        _seg = f"【参考文件：{_rdn}】\n{_txt.strip()}"
                                        _ref_blocks.append(_seg)
                                        _used += len(_seg)
                                        if _used >= _ref_cap:
                                            break
                                    if _ref_blocks:
                                        ctx["reference_docs_excerpt"] = ("\n\n---\n\n".join(_ref_blocks))[:_ref_cap]
                                    # 清理参考文件临时路径
                                    for _p, _dn, _arch in _ritems:
                                        if not _arch:
                                            Path(_p).unlink(missing_ok=True)
                                    for _d in _rdirs:
                                        shutil.rmtree(_d, ignore_errors=True)
                            except Exception:
                                pass
                            with st.spinner("AI 正在审核文档…"):
                                report = agent.review_text(text, file_name=fn, review_context=ctx)
                            report["original_filename"] = fn
                            if view_url:
                                report["_kdocs_view_url"] = view_url
                            if download_url:
                                report["_kdocs_download_url"] = download_url
                            _inject_review_meta(report)
                            try:
                                save_audit_report(
                                    agent.collection_name if agent else "",
                                    report,
                                    model_info=get_current_model_info() or "",
                                )
                                _invalidate_audit_history_cache()
                            except Exception:
                                pass
                            # 自动沉淀：同报告内重复错误追加到 skills/rules
                            _auto_append_repeated_errors_to_skills_rules(report)
                            st.session_state.review_reports = [report]
                            st.session_state.review_kdocs_view_url = view_url or None
                            st.success("审核完成！结果已返回系统，请在下方查看。")
                            st.experimental_rerun()
                    except Exception as e:
                        st.error(f"下载或审核失败：{e}")
        review_files = st.file_uploader(
            "选择待审核文件（可多选多个文件，批量审核；支持 PDF/Word/Excel/TXT 等，也支持 .zip / .tar / .tar.gz 压缩包，压缩包将解压后逐个审核）",
            type=["pdf", "docx", "doc", "xlsx", "xls", "txt", "md", "zip", "tar", "gz", "tgz"],
            accept_multiple_files=True,
            key="review_uploader",
        )
        # 文件夹（服务端路径）输入：浏览器“上传文件夹”在 Streamlit 中不稳定，这里用路径扫描保证可用
        with st.expander("📁 通过文件夹路径加载待审核文件（可选）", expanded=False):
            st.caption("适用场景：文件在运行本工具的机器上（服务器/本机）。会递归扫描支持格式；路径中含「废弃」的文件/目录会自动跳过。")
            _dir = st.text_input("待审核文件夹路径", value="", placeholder=r"例如：D:\docs\to_review", key="review_dir_path")
            _dir_items = st.session_state.setdefault("review_dir_items", [])
            cda, cdb = st.columns(2)
            with cda:
                if st.button("添加该文件夹", key="review_dir_add_btn"):
                    try:
                        if _dir.strip():
                            fps = _scan_directory_files(_dir.strip())
                        else:
                            fps = []
                        if fps:
                            for fp in fps:
                                _dir_items.append((str(fp), fp.name, False))
                            # 去重（按 path）
                            seen = set()
                            ded = []
                            for p, dn, arch in _dir_items:
                                if p in seen:
                                    continue
                                seen.add(p)
                                ded.append((p, dn, arch))
                            st.session_state["review_dir_items"] = ded
                            st.success(f"已添加 {len(fps)} 个文件。")
                        else:
                            st.warning("未扫描到可审核的文件。")
                    except Exception as _e:
                        st.error(f"扫描失败：{_e}")
            with cdb:
                if st.button("清空文件夹列表", key="review_dir_clear_btn"):
                    st.session_state["review_dir_items"] = []
                    st.success("已清空。")
            if st.session_state.get("review_dir_items"):
                st.caption(f"文件夹已加入 **{len(st.session_state['review_dir_items'])}** 个文件；开始审核时会与上方上传文件合并。")
        ref_files = st.file_uploader(
            "参考文件（可选：本次审核将按参考文件要求对照核查）",
            type=["pdf", "docx", "doc", "xlsx", "xls", "txt", "md", "zip", "tar", "gz", "tgz"],
            accept_multiple_files=True,
            key="review_reference_uploader",
            help="可上传程序文件、技术要求、规范、追溯制度等作为本次审核的对照依据。系统会摘录其内容并注入审核上下文。",
        )
        with st.expander("📁 通过文件夹路径加载参考文件（可选）", expanded=False):
            st.caption("同上：仅当参考文件位于运行本工具的机器上时可用。")
            _rdir = st.text_input("参考文件夹路径", value="", placeholder=r"例如：D:\docs\refs", key="review_ref_dir_path")
            _rdir_items = st.session_state.setdefault("review_ref_dir_items", [])
            cra, crb = st.columns(2)
            with cra:
                if st.button("添加参考文件夹", key="review_ref_dir_add_btn"):
                    try:
                        if _rdir.strip():
                            fps = _scan_directory_files(_rdir.strip())
                        else:
                            fps = []
                        if fps:
                            for fp in fps:
                                _rdir_items.append((str(fp), fp.name, False))
                            seen = set()
                            ded = []
                            for p, dn, arch in _rdir_items:
                                if p in seen:
                                    continue
                                seen.add(p)
                                ded.append((p, dn, arch))
                            st.session_state["review_ref_dir_items"] = ded
                            st.success(f"已添加 {len(fps)} 个参考文件。")
                        else:
                            st.warning("未扫描到可用参考文件。")
                    except Exception as _e:
                        st.error(f"扫描失败：{_e}")
            with crb:
                if st.button("清空参考文件夹列表", key="review_ref_dir_clear_btn"):
                    st.session_state["review_ref_dir_items"] = []
                    st.success("已清空。")
            if st.session_state.get("review_ref_dir_items"):
                st.caption(f"参考文件夹已加入 **{len(st.session_state['review_ref_dir_items'])}** 个文件；开始审核时会与上方上传参考文件合并。")
        _has_review_inputs = bool(review_files) or bool(st.session_state.get("review_dir_items"))
        _force_ocr_refresh = False
        if _has_review_inputs:
            n_upload = len(review_files)
            n_dir = len(st.session_state.get("review_dir_items") or [])
            st.caption(f"已选择 **{n_upload}** 个上传文件，文件夹列表 **{n_dir}** 个文件；点击下方按钮将合并批量审核。")
            try:
                _sel_names = [str(getattr(f, "name", "") or "").strip() for f in (review_files or [])]
                _sel_names.extend([str(dn or "").strip() for (_p, dn, _a) in (st.session_state.get("review_dir_items") or [])])
                _sel_names = [x for x in _sel_names if x and x.lower().endswith(".pdf")]
                _cached_names = set(list_ocr_cache_file_names(limit=5000))
                _dup_ocr = sorted({x for x in _sel_names if x in _cached_names})
            except Exception:
                _dup_ocr = []
            if _dup_ocr:
                st.warning(
                    "检测到以下 PDF 在 OCR 缓存中已存在同名结果："
                    + "、".join(_dup_ocr[:8])
                    + ("…" if len(_dup_ocr) > 8 else "")
                )
                _force_ocr_refresh = st.checkbox(
                    "同名文件重新 OCR 并覆盖缓存（默认关闭：直接复用缓存，节省 token）",
                    value=False,
                    key="review_force_ocr_refresh",
                )
            else:
                st.caption("未命中同名 OCR 缓存：本次如遇扫描版 PDF 将执行 OCR，并自动写入缓存。")
            do_multi_doc = st.checkbox(
                "进行多文档一致性与模板风格审核（2 个及以上文件时）",
                value=True,
                key="do_multi_doc_consistency",
                help="额外生成「多文档一致性与模板风格」报告：产品信息、术语、版本日期、模板风格等跨文档一致性。**不含** REQ/风险/CS/测试编号等追溯链专项（请用下方「跨文档可追溯性审核」按钮）。",
            )
            only_multi_doc = st.checkbox(
                "仅进行多文档一致性与模板风格审核（不跑单文档审核，需 2 个及以上文件）",
                value=False,
                key="only_multi_doc_review",
                help="只跑多文档一致性/模板风格，不逐份单文档审核。追溯编号跨文档对齐请另点「跨文档可追溯性审核」。",
            )
            if len(review_files) >= 2:
                st.caption(
                    "单文档审核时模型**看不到同批其他文件**，可能误报「未上传其他文档、无法做跨文档追溯」。"
                    "专项会重点核对：**软件需求规范里追溯表所填 CS 编号**与**风险分析文档**是否一一对应且含义一致；**《软件可追溯性分析》矩阵**与需求/风险/设计/测试文档的**全链路追溯关系**是否一致（不跑单文档审核）。"
                )
                if st.button(
                    "🔗 仅执行：跨文档可追溯性审核（本批全部文件 + 知识库制度）",
                    key="review_traceability_cross_btn",
                    help="核对 SRS 追溯表中的 CS 与风险分析、可追溯性分析矩阵与各文档编号是否闭环一致，以及 REQ/测试等编号跨文档对齐。",
                ):
                    items_tr: list = []
                    temp_dirs_tr: list = []
                    try:
                        items_tr, temp_dirs_tr = _expand_uploads(review_files)
                        if len(items_tr) < 2:
                            st.warning("需要至少 2 个可解析的文件。")
                        else:
                            agent = init_agent()
                            _tr_ctx = dict(review_context) if review_context else {}
                            _doc_lang_sel = st.session_state.get("review_doc_lang") or "不指定"
                            _tr_ctx["document_language"] = DOC_LANG_LABEL_TO_VALUE.get(_doc_lang_sel, "")
                            _tr_ctx["current_provider"] = (
                                st.session_state.get("current_provider") or settings.provider or ""
                            ).strip().lower()
                            _tr_ctx["_force_ocr_refresh"] = bool(_force_ocr_refresh)
                            if project_id:
                                _tr_ctx["_filter_by_registration_type"] = True
                            _rbid_tr = datetime.now().strftime("%Y%m%d_%H%M%S") + "-" + uuid.uuid4().hex[:8]
                            _tr_ctx["review_batch_id"] = _rbid_tr
                            st.session_state["review_batch_id"] = _rbid_tr
                            _tr_ctx["_batch_review"] = True
                            if (_tr_ctx.get("current_provider") or "").strip().lower() == "deepseek":
                                _tr_ctx["_skip_llm_summary"] = True
                            _trace_fn = getattr(agent, "review_traceability_cross_document", None)
                            if not callable(_trace_fn):
                                st.error("当前运行环境不支持跨文档可追溯性审核，请确认已更新代码后重启应用。")
                            else:
                                _flat = [(p, dn) for (p, dn, _) in items_tr]
                                with st.spinner("跨文档可追溯性审核中（读取文件 + 知识库检索 + 模型分析）…"):
                                    tr_rep = _trace_fn(_flat, project_id=project_id, review_context=_tr_ctx)
                                tr_rep["original_filename"] = tr_rep.get("file_name") or "跨文档可追溯性审核"
                                tr_rep["related_doc_names"] = [dn for (_, dn, _) in items_tr]
                                _inject_review_meta(tr_rep, "batch_traceability")
                                _prev_rep = list(st.session_state.get("review_reports") or [])
                                _prev_rep.append(tr_rep)
                                st.session_state.review_reports = _prev_rep
                                try:
                                    save_audit_report(
                                        agent.collection_name,
                                        tr_rep,
                                        model_info=get_current_model_info() or "",
                                    )
                                    _invalidate_audit_history_cache()
                                except Exception as _save_tr:
                                    st.warning(f"跨文档可追溯性报告已生成，但写入历史失败：{_save_tr}")
                                st.success(
                                    f"已完成跨文档可追溯性审核（{tr_rep.get('total_points', 0)} 条审核点），"
                                    "报告已追加到下方列表并尽量写入历史。"
                                )
                    finally:
                        for _p, _dn, _arch in items_tr:
                            if not _arch:
                                Path(_p).unlink(missing_ok=True)
                        for _d in temp_dirs_tr:
                            shutil.rmtree(_d, ignore_errors=True)
                    _streamlit_rerun()
        if _has_review_inputs and st.button("🔍 开始批量审核" if (len(review_files) + len(st.session_state.get("review_dir_items") or [])) > 1 else "🔍 开始审核", key="review_btn"):
            items, temp_dirs = _expand_uploads(review_files or [])
            # 合并文件夹扫描结果（不会进入 temp_dirs 管理）
            try:
                for p, dn, arch in (st.session_state.get("review_dir_items") or []):
                    items.append((p, dn, arch))
            except Exception:
                pass

            if not items:
                st.warning("没有可审核的文件。")
                return
            if st.session_state.get("only_multi_doc_review") and len(items) < 2:
                st.warning("仅多文档一致性与模板风格审核需至少 2 个文件，请取消勾选或增加文件。")
                return

            agent = init_agent()
            # 自动匹配过往项目案例：如勾选了自动匹配，用首份文档内容匹配案例元数据
            matched_case = None
            if st.session_state.get("review_auto_match_case"):
                first_path = items[0][0]
                try:
                    docs = load_single_file(
                        first_path,
                        force_ocr_refresh=bool(_force_ocr_refresh),
                        ocr_cache_file_name=(items[0][1] or Path(first_path).name),
                    )
                    doc_text = "\n\n".join(getattr(d, "page_content", str(d)) for d in docs) if docs else ""
                except Exception:
                    doc_text = ""
                dim_filters = review_context if review_context else {}
                matched_case = _match_project_case_for_review(collection, doc_text, dim_filters)
                if matched_case:
                    st.info(
                        f"已匹配到过往项目案例：**{matched_case.get('case_name')}**"
                        f"（产品名称：{matched_case.get('product_name') or '—'}）"
                        f"，案例经验将纳入审核上下文。"
                    )
                    # 将案例元数据注入 review_context
                    if review_context is None:
                        review_context = {}
                    _case_lang = matched_case.get("document_language") or ""
                    _case_lang_label = DOC_LANG_VALUE_TO_LABEL.get(_case_lang, "不指定")
                    case_ctx = (
                        f"\n\n【过往项目案例参考】\n"
                        f"案例文档语言：{_case_lang_label}\n"
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
                    # 文档内容完整性：从历史案例库提取章节结构，供审核时对比待审文档是否缺章
                    try:
                        case_chunks = get_knowledge_docs_by_case_id(collection, matched_case["id"], limit=500)
                        if case_chunks:
                            outline = extract_section_outline_from_texts([c.get("content") or "" for c in case_chunks])
                            if outline.strip():
                                case_ctx += "\n【历史案例文档章节参考】\n以下为案例库中该案例文档的章节结构，请据此检查待审文档是否具备应有章节；缺失的章节须作为「文档内容完整性」审核点列出，并指明应补充的章节名称或位置。\n\n" + outline.strip() + "\n"
                                # P-4.6（不改顺序）：章节参考段落后强化逐条对照要求
                                case_ctx += (
                                    "\n**完整性审核执行要求**：请按上表章节**逐条**在待审文档中查找对应或等价章节；"
                                    "每发现一处缺失或标题/层级明显不一致，单列一条审核点，不得合并为多处以「若干章节缺失」一笔带过。"
                                    "location 须写清缺失章节名称或待审文档中应出现的位置。\n"
                                )
                    except Exception:
                        pass
                    case_ctx += "\n请参考上述案例经验审核当前文档，如有类似问题请重点关注。"
                    review_context["case_context_text"] = case_ctx
                else:
                    st.caption("未匹配到过往项目案例，将按通用知识库审核。")

            # 待审文档语言：不指定/中文版/英文版/中英文，传入审核上下文（与全系统 DOC_LANG_OPTIONS 一致）
            _doc_lang_sel = st.session_state.get("review_doc_lang") or "不指定"
            if review_context is None:
                review_context = {}
            review_context["document_language"] = DOC_LANG_LABEL_TO_VALUE.get(_doc_lang_sel, "")
            # 与侧栏一致，供审核内核（如长文档分块并发策略）识别实际提供方
            review_context["current_provider"] = (
                st.session_state.get("current_provider") or settings.provider or ""
            ).strip().lower()
            review_context["_force_ocr_refresh"] = bool(_force_ocr_refresh)
            # 按项目审核时自动按项目适用注册类别匹配审核点；通用审核不区分类别、使用全部审核点
            if project_id:
                review_context["_filter_by_registration_type"] = True

            # 参考文件：注入审核上下文（对所有本批文件生效）
            try:
                _ref_all = list(ref_files or [])
                # 文件夹参考文件也合并进来（转为伪 UploadedFile items 逻辑：直接复用 items 列表）
                _ritems, _rdirs = _expand_uploads(_ref_all) if _ref_all else ([], [])
                try:
                    for p, dn, arch in (st.session_state.get("review_ref_dir_items") or []):
                        _ritems.append((p, dn, arch))
                except Exception:
                    pass
                if _ritems:
                    _ref_blocks = []
                    _ref_cap = 22000
                    _used = 0
                    for _rp, _rdn, _rarch in _ritems:
                        try:
                            _docs = load_single_file(
                                _rp,
                                force_ocr_refresh=bool(_force_ocr_refresh),
                                ocr_cache_file_name=(_rdn or Path(_rp).name),
                            )
                            _txt = "\n\n".join(getattr(d, "page_content", str(d)) for d in _docs) if _docs else ""
                        except Exception:
                            _txt = ""
                        if not _txt.strip():
                            continue
                        _seg = f"【参考文件：{_rdn}】\n{_txt.strip()}"
                        _ref_blocks.append(_seg)
                        _used += len(_seg)
                        if _used >= _ref_cap:
                            break
                    if _ref_blocks:
                        review_context["reference_docs_excerpt"] = ("\n\n---\n\n".join(_ref_blocks))[:_ref_cap]
                    # 清理参考文件临时路径
                    for _p, _dn, _arch in _ritems:
                        if not _arch:
                            Path(_p).unlink(missing_ok=True)
                    for _d in _rdirs:
                        shutil.rmtree(_d, ignore_errors=True)
            except Exception:
                pass

            total_files = len(items)
            if total_files > 1:
                review_context["_batch_review"] = True
                # 本批唯一 ID：单文档历史记录与批次汇总记录共用，便于在历史列表中归类
                _rbid = datetime.now().strftime("%Y%m%d_%H%M%S") + "-" + uuid.uuid4().hex[:8]
                review_context["review_batch_id"] = _rbid
                st.session_state["review_batch_id"] = _rbid
            st.session_state.review_project_id = project_id
            st.session_state.review_context = review_context
            st.session_state._review_mode_snapshot = review_mode
            st.caption(
                "💡 批量审核为**逐文件排队**执行；任意模型在文件之间会有短暂间隔（可在 `.env` 设置 `REVIEW_BATCH_INTER_DOC_SLEEP_SEC`，DeepSeek 另见 `REVIEW_DEEPSEEK_INTER_DOC_SLEEP_SEC`），以减轻限流与断连。"
                " 请**保持本页与浏览器标签打开**直至结束；自托管 Streamlit 若仍中途断开，请调大服务器/反向代理读写超时。"
                " 已完成的单文件会写入「历史报告」且**显示为上传文件名**；建议每批不超过 10 个文件。"
            )
            review_bar = st.progress(0)
            review_status = st.empty()
            review_status.info(f"准备审核 {total_files} 个文件...")
            review_current = st.empty()
            review_panel = st.empty()
            log_display = st.empty()

            log_lines = []
            all_reports = []
            failed_items = []
            failed_errors = []  # 每个失败文件对应的错误信息，便于日志与界面展示
            t_start = time.time()
            batch_interrupted = False
            batch_error = None
            multi_doc_error_shown = False  # 仅多文档分支内已展示错误时不再用通用“未生成报告”覆盖
            batch_aggregate_saved = False

            def _render_review_panel(done: int, total: int, success: int, failed: int, start_ts: float, current_name: str = ""):
                elapsed = max(0.0, time.time() - start_ts)
                avg = (elapsed / done) if done > 0 else 0.0
                remain = max(0, total - done)
                eta = (avg * remain) if done > 0 else 0.0
                cur = current_name or "等待中"
                review_panel.markdown(
                    "\n".join([
                        "#### 📊 批量审核进度面板",
                        f"- 当前：`{cur}`",
                        f"- 进度：`{done}/{total}`（成功 `{success}` / 失败 `{failed}`）",
                        f"- 已用时：`{_format_time(elapsed)}`",
                        f"- 预计剩余：`{_format_time(eta) if done > 0 else '计算中…'}`",
                    ])
                )

            _render_review_panel(0, total_files, 0, 0, t_start, "准备中")

            try:
                if st.session_state.get("only_multi_doc_review") and len(items) >= 2:
                    review_status.info("仅进行多文档一致性与模板风格审核…")
                    multi_doc_fn = getattr(agent, "review_multi_document_consistency", None)
                    if callable(multi_doc_fn):
                        try:
                            _ctx = dict(review_context) if review_context else {}
                            _ctx["current_provider"] = st.session_state.get("current_provider") or settings.provider
                            _ctx["_batch_review"] = True
                            if (_ctx.get("current_provider") or "").strip().lower() == "deepseek":
                                _ctx["_skip_llm_summary"] = True
                            with st.spinner("正在调用多文档一致性审核接口（读取文件并请求 AI）…"):
                                consistency_report = multi_doc_fn(
                                    [(path, display_name) for (path, display_name, _) in items],
                                    project_id=project_id,
                                    review_context=_ctx,
                                )
                            consistency_report["original_filename"] = "多文档一致性与模板风格审核"
                            consistency_report["related_doc_names"] = [display_name for (_, display_name, _) in items]
                            consistency_report["file_name"] = consistency_report.get("file_name") or "多文档一致性与模板风格审核"
                            _inject_review_meta(consistency_report, "batch_multi_doc")
                            all_reports.append(consistency_report)
                            st.session_state.review_reports = all_reports
                            try:
                                save_audit_report(
                                    agent.collection_name,
                                    consistency_report,
                                    model_info=get_current_model_info() or "",
                                )
                                _invalidate_audit_history_cache()
                            except Exception as save_err:
                                st.warning(f"多文档一致性报告已生成，但写入历史审核报告失败，请稍后在「历史审核报告」中确认是否可见：{save_err}")
                            review_status.success("多文档一致性与模板风格审核完成。")
                        except Exception as ex:
                            multi_doc_error_shown = True
                            review_status.error(f"多文档一致性与模板风格审核失败：{ex}")
                            st.caption("若为接口/网络错误，请检查模型配置与 API；若为解析错误，请查看下方详情。")
                            with st.expander("错误详情", expanded=True):
                                st.code(traceback.format_exc(), language="text")
                    else:
                        multi_doc_error_shown = True  # 避免收尾时用「未生成报告」覆盖本提示
                        review_status.warning(
                            "当前运行环境中未找到多文档一致性审核功能（与是否上传压缩包无关，压缩包内多文件已正常识别）。"
                            "请重启应用后重试，或确认已部署包含该功能的最新代码。"
                        )
                else:
                    for idx, (path, display_name, is_from_archive) in enumerate(items):
                        try:
                            # 批量虽为逐份串行，DeepSeek 等仍可能在连续请求下更易触发网关抖动，份间轻微间隔
                            # 注意：此处任何异常都按“单文件失败”处理，避免中断整批审核。
                            # 批量排队：任意提供方份间小间隔 + DeepSeek 额外间隔（取较大者），降低叠峰断连/自动中断
                            if idx > 0 and total_files > 1:
                                try:
                                    _u = float(getattr(settings, "review_batch_inter_doc_sleep_sec", 0.0) or 0.0)
                                except Exception:
                                    _u = 0.0
                                _u = max(0.0, _u)
                                _gap = _u
                                if (review_context.get("current_provider") or "") == "deepseek":
                                    try:
                                        _d = float(getattr(settings, "review_deepseek_inter_doc_sleep_sec", 0.0) or 0.0)
                                    except Exception:
                                        _d = 1.0
                                    if _d <= 0:
                                        _d = 1.0
                                    _gap = max(_gap, _d)
                                if _gap > 0:
                                    time.sleep(_gap)
                            pct = int(idx / total_files * 100)
                            review_bar.progress(pct)
                            review_status.info(
                                f"审核进度 {idx+1}/{total_files} | "
                                f"已完成 {len(all_reports)} 个 | "
                                f"耗时 {_format_time(time.time() - t_start)}"
                            )
                            review_current.info(f"正在审核 [{display_name}]...")
                            _render_review_panel(
                                idx,
                                total_files,
                                len(all_reports),
                                len(failed_items),
                                t_start,
                                display_name,
                            )

                            t0 = time.time()
                            mi = get_current_model_info()
                            with st.spinner(f"AI 正在审核 [{display_name}]，请耐心等待..."):
                                report = _call_with_transient_retry(
                                    lambda p=path, dn=display_name: agent.review(
                                        p,
                                        project_id=project_id,
                                        review_context=review_context,
                                        system_prompt=review_sys_edit.strip() or None,
                                        user_prompt=review_usr_edit.strip() or None,
                                        extra_instructions=review_extra_edit.strip() or None,
                                        display_file_name=dn,
                                    ),
                                    attempts=_review_transient_retry_attempts(),
                                )
                            if (review_context.get("current_provider") or "") == "deepseek":
                                gc.collect()
                            elapsed = time.time() - t0
                            report["original_filename"] = display_name
                            report["_original_path"] = path
                            _inject_review_meta(report, "batch_member" if total_files > 1 else None)
                            all_reports.append(report)
                            # 单文件成功后立即落库，避免批量中断/取消或最终不足 2 份成功时丢失历史报告。
                            # 多文档一致性报告后续仍可单独保存为批次汇总，不影响这里的单文件归档。
                            try:
                                save_audit_report(agent.collection_name, report, model_info=mi)
                                _invalidate_audit_history_cache()
                            except Exception:
                                pass
                            n_points = report.get("total_points", 0)
                            add_operation_log(
                                op_type=OP_TYPE_REVIEW,
                                collection=agent.collection_name,
                                file_name=display_name,
                                source=str(path),
                                extra={
                                    "total_points": n_points,
                                    "duration_sec": round(elapsed, 2),
                                    **({"review_batch_id": review_context.get("review_batch_id")} if total_files > 1 and review_context.get("review_batch_id") else {}),
                                },
                                model_info=mi,
                            )
                            log_lines.append(
                                f"- :white_check_mark: **{display_name}** — {n_points} 个审核点, {_format_time(elapsed)}"
                            )
                            _render_review_panel(
                                idx + 1,
                                total_files,
                                len(all_reports),
                                len(failed_items),
                                t_start,
                                display_name,
                            )
                        except Exception as e:
                            err_str = _format_review_exception(e)
                            failed_items.append((path, display_name, is_from_archive))
                            failed_errors.append(err_str)
                            log_lines.append(f"- :x: **{display_name}** 失败：{err_str[:300]}{'…' if len(err_str) > 300 else ''}")
                            add_operation_log(
                                op_type="review_error",
                                collection=agent.collection_name,
                                file_name=display_name,
                                source=str(path),
                                extra={"error": err_str, "traceback": traceback.format_exc()},
                                model_info=get_current_model_info(),
                            )
                            _render_review_panel(
                                idx + 1,
                                total_files,
                                len(all_reports),
                                len(failed_items),
                                t_start,
                                display_name,
                            )
                            continue
                        log_display.markdown("\n".join(log_lines))

            except Exception as e:
                batch_interrupted = True
                batch_error = str(e)
                log_lines.append(f"- ⚠️ 批量审核中断：{e}")
                log_display.markdown("\n".join(log_lines))
                add_operation_log(
                    op_type=OP_TYPE_REVIEW_BATCH,
                    collection=agent.collection_name,
                    file_name="",
                    source="upload",
                    extra={
                        "total_files": total_files,
                        "success_count": len(all_reports),
                        "fail_count": total_files - len(all_reports),
                        "interrupted": True,
                        "reason": str(e),
                        "first_error": batch_error,
                        "traceback": traceback.format_exc(),
                        "total_audit_points": sum(r.get("total_points", 0) for r in all_reports),
                        "duration_sec": round(time.time() - t_start, 2),
                        **({"review_batch_id": review_context.get("review_batch_id")} if review_context.get("review_batch_id") else {}),
                    },
                    model_info=get_current_model_info(),
                )

                review_bar.progress(100)
                total_time = time.time() - t_start
                review_current.empty()
                review_status.warning(
                    f"审核过程中断：{batch_error}。已完成 {len(all_reports)}/{total_files} 个文件，耗时 {_format_time(total_time)}。请到「历史报告」查看已审核结果，剩余文件可分批重新上传。"
                )
                _render_review_panel(total_files, total_files, len(all_reports), len(failed_items), t_start, "已中断")
                if all_reports:
                    st.session_state.review_reports = all_reports
                    try:
                        _bid = (review_context or {}).get("review_batch_id") or ""
                        _partial_batch = {
                            "file_name": (f"批量审核·批次汇总(中断)·{_bid}" if _bid else "批量审核（部分完成）")[:512],
                            "original_filename": (f"批量审核·批次汇总(中断)·{_bid}" if _bid else "批量审核（部分完成）")[:512],
                            "batch": True,
                            "reports": list(all_reports),
                            "total_points": sum(r.get("total_points", 0) for r in all_reports),
                            "high_count": sum(r.get("high_count", 0) for r in all_reports),
                            "medium_count": sum(r.get("medium_count", 0) for r in all_reports),
                            "low_count": sum(r.get("low_count", 0) for r in all_reports),
                            "info_count": sum(r.get("info_count", 0) for r in all_reports),
                            "summary": (
                                f"本批次中断（批次ID：{_bid}），已将已完成的 {len(all_reports)}/{total_files} 份单文档审核整合为一条批次记录；"
                                f"各单文档记录仍可在历史中按同一批次ID筛选识别。"
                                if _bid
                                else f"本批次中断，已完成 {len(all_reports)}/{total_files} 份报告。"
                            ),
                        }
                        _inject_review_meta(_partial_batch, "batch_aggregated")
                        _partial_batch_id = save_audit_report(
                            agent.collection_name,
                            _partial_batch,
                            model_info=get_current_model_info() or "",
                        )
                        _invalidate_audit_history_cache()
                        batch_aggregate_saved = True
                        _bid = (review_context or {}).get("review_batch_id") or "未分配"
                        st.session_state["review_batch_status_msg"] = (
                            f"本批已自动中断：成功 {len(all_reports)} / {total_files}，失败 {total_files - len(all_reports)}。"
                            f" 已将成功项整合为批次报告（ID:{_partial_batch_id}，批次号：{_bid}）；可在下方点击“重新审核失败项”。"
                        )
                    except Exception as _save_err:
                        st.caption(f"已完成的报告写入历史失败：{_save_err}")
            else:
                review_bar.progress(100)
                total_time = time.time() - t_start
                review_current.empty()
                _render_review_panel(total_files, total_files, len(all_reports), len(failed_items), t_start, "已完成")
                only_multi = st.session_state.get("only_multi_doc_review") and len(items) >= 2
                if only_multi:
                    if all_reports:
                        review_status.success(
                            f"多文档一致性与模板风格审核已完成（基于 {total_files} 个文件），耗时 {_format_time(total_time)}。"
                        )
                    elif not multi_doc_error_shown:
                        review_status.warning("多文档一致性审核未生成报告，请查看上方错误或提示。")
                else:
                    review_status.success(
                        f"全部完成! 审核 {len(all_reports)}/{total_files} 个文件, "
                        f"耗时 {_format_time(total_time)}"
                    )

                if len(all_reports) >= 2 and st.session_state.get("do_multi_doc_consistency", True):
                    multi_doc_fn = getattr(agent, "review_multi_document_consistency", None)
                    if callable(multi_doc_fn):
                        try:
                            _ctx = dict(review_context) if review_context else {}
                            _ctx["current_provider"] = st.session_state.get("current_provider") or settings.provider
                            _ctx["_batch_review"] = True
                            if (_ctx.get("current_provider") or "").strip().lower() == "deepseek":
                                _ctx["_skip_llm_summary"] = True
                            review_status.info("正在进行多文档一致性与模板风格审核…")
                            with st.spinner("正在调用多文档一致性审核接口（请求 AI）…"):
                                consistency_report = multi_doc_fn(
                                    [(path, display_name) for (path, display_name, _) in items],
                                    project_id=project_id,
                                    review_context=_ctx,
                                )
                            consistency_report["original_filename"] = "多文档一致性与模板风格审核"
                            consistency_report["related_doc_names"] = [display_name for (_, display_name, _) in items]
                            consistency_report["file_name"] = consistency_report.get("file_name") or "多文档一致性与模板风格审核"
                            _inject_review_meta(consistency_report, "batch_multi_doc")
                            all_reports.append(consistency_report)
                            try:
                                _bid = (review_context or {}).get("review_batch_id") or ""
                                batch_report = {
                                    "file_name": (f"批量审核·批次汇总(含多文档)·{_bid}" if _bid else "批量审核（含多文档一致性）")[:512],
                                    "original_filename": (f"批量审核·批次汇总(含多文档)·{_bid}" if _bid else "批量审核（含多文档一致性）")[:512],
                                    "batch": True,
                                    "reports": list(all_reports),
                                    "total_points": sum(r.get("total_points", 0) for r in all_reports),
                                    "high_count": sum(r.get("high_count", 0) for r in all_reports),
                                    "medium_count": sum(r.get("medium_count", 0) for r in all_reports),
                                    "low_count": sum(r.get("low_count", 0) for r in all_reports),
                                    "info_count": sum(r.get("info_count", 0) for r in all_reports),
                                    "summary": (
                                        f"本批次汇总（批次ID：{_bid}），共 {len(all_reports)} 份子报告（含多文档一致性审核）。"
                                        if _bid
                                        else f"本批次共 {len(all_reports)} 份报告（含多文档一致性审核）。"
                                    ),
                                }
                                _inject_review_meta(batch_report, "batch_aggregated")
                                save_audit_report(
                                    agent.collection_name,
                                    batch_report,
                                    model_info=get_current_model_info() or "",
                                )
                                _invalidate_audit_history_cache()
                                batch_aggregate_saved = True
                            except Exception as save_err:
                                st.warning(f"批量报告写入历史失败：{save_err}")
                            log_lines.append(
                                f"- :white_check_mark: **多文档一致性与模板风格审核** — {consistency_report.get('total_points', 0)} 个审核点"
                            )
                            log_display.markdown("\n".join(log_lines))
                            review_status.success(
                                f"全部完成! 审核 {len(all_reports)}/{total_files} 个文件（含多文档一致性）, "
                                f"耗时 {_format_time(time.time() - t_start)}"
                            )
                        except Exception as ex:
                            log_lines.append(f"- :x: **多文档一致性与模板风格审核** 失败：{ex}")
                            log_display.markdown("\n".join(log_lines))
                            if all_reports:
                                try:
                                    _bid = (review_context or {}).get("review_batch_id") or ""
                                    _batch_no_consistency = {
                                        "file_name": (f"批量审核·批次汇总(多文档未完成)·{_bid}" if _bid else "批量审核（多文档一致性未完成）")[:512],
                                        "original_filename": (f"批量审核·批次汇总(多文档未完成)·{_bid}" if _bid else "批量审核（多文档一致性未完成）")[:512],
                                        "batch": True,
                                        "reports": list(all_reports),
                                        "total_points": sum(r.get("total_points", 0) for r in all_reports),
                                        "high_count": sum(r.get("high_count", 0) for r in all_reports),
                                        "medium_count": sum(r.get("medium_count", 0) for r in all_reports),
                                        "low_count": sum(r.get("low_count", 0) for r in all_reports),
                                        "info_count": sum(r.get("info_count", 0) for r in all_reports),
                                        "summary": (
                                            f"本批次汇总（批次ID：{_bid}），共 {len(all_reports)} 份单文档报告；多文档一致性审核未完成。"
                                            if _bid
                                            else f"本批次共 {len(all_reports)} 份单文档报告；多文档一致性审核未完成。"
                                        ),
                                    }
                                    _inject_review_meta(_batch_no_consistency, "batch_aggregated")
                                    save_audit_report(agent.collection_name, _batch_no_consistency, model_info=get_current_model_info() or "")
                                    _invalidate_audit_history_cache()
                                    batch_aggregate_saved = True
                                except Exception as _save_err:
                                    st.warning(f"本批报告写入历史失败：{_save_err}")
                    else:
                        log_lines.append("- :x: **多文档一致性与模板风格审核** 当前版本不支持，已跳过。")
                        log_display.markdown("\n".join(log_lines))
                        if all_reports:
                            try:
                                _bid = (review_context or {}).get("review_batch_id") or ""
                                _batch_skip_multi = {
                                    "file_name": (f"批量审核·批次汇总·{_bid}" if _bid else "批量审核（含多文档一致性）")[:512],
                                    "original_filename": (f"批量审核·批次汇总·{_bid}" if _bid else "批量审核（含多文档一致性）")[:512],
                                    "batch": True,
                                    "reports": list(all_reports),
                                    "total_points": sum(r.get("total_points", 0) for r in all_reports),
                                    "high_count": sum(r.get("high_count", 0) for r in all_reports),
                                    "medium_count": sum(r.get("medium_count", 0) for r in all_reports),
                                    "low_count": sum(r.get("low_count", 0) for r in all_reports),
                                    "info_count": sum(r.get("info_count", 0) for r in all_reports),
                                    "summary": (
                                        f"本批次汇总（批次ID：{_bid}），共 {len(all_reports)} 份单文档报告（未执行多文档一致性）。"
                                        if _bid
                                        else f"本批次共 {len(all_reports)} 份报告（多文档一致性未执行）。"
                                    ),
                                }
                                _inject_review_meta(_batch_skip_multi, "batch_aggregated")
                                save_audit_report(agent.collection_name, _batch_skip_multi, model_info=get_current_model_info() or "")
                                _invalidate_audit_history_cache()
                                batch_aggregate_saved = True
                            except Exception as _save_err:
                                st.warning(f"本批报告写入历史失败：{_save_err}")

                # 未勾选「多文档一致性」时：仍将本批已完成的单文档审核整合为一条批次汇总记录（与逐文件记录并存）
                if (
                    total_files > 1
                    and not st.session_state.get("only_multi_doc_review")
                    and all_reports
                    and not st.session_state.get("do_multi_doc_consistency", True)
                ):
                    try:
                        _bid = (review_context or {}).get("review_batch_id") or ""
                        _batch_singles_only = {
                            "file_name": (f"批量审核·批次汇总·{_bid}" if _bid else "批量审核（批次汇总）")[:512],
                            "original_filename": (f"批量审核·批次汇总·{_bid}" if _bid else "批量审核（批次汇总）")[:512],
                            "batch": True,
                            "reports": list(all_reports),
                            "total_points": sum(r.get("total_points", 0) for r in all_reports),
                            "high_count": sum(r.get("high_count", 0) for r in all_reports),
                            "medium_count": sum(r.get("medium_count", 0) for r in all_reports),
                            "low_count": sum(r.get("low_count", 0) for r in all_reports),
                            "info_count": sum(r.get("info_count", 0) for r in all_reports),
                            "summary": (
                                f"本批次汇总（批次ID：{_bid}），共 {len(all_reports)} 份单文档报告（未执行多文档一致性审核）。"
                                if _bid
                                else f"本批次共 {len(all_reports)} 份单文档报告（未执行多文档一致性审核）。"
                            ),
                        }
                        _inject_review_meta(_batch_singles_only, "batch_aggregated")
                        save_audit_report(
                            agent.collection_name,
                            _batch_singles_only,
                            model_info=get_current_model_info() or "",
                        )
                        _invalidate_audit_history_cache()
                        batch_aggregate_saved = True
                    except Exception as _save_err:
                        st.warning(f"批次汇总报告写入历史失败：{_save_err}")

                # 若本批存在失败项且尚未生成批次汇总，自动补一条“部分完成”汇总，便于定位与重试
                if failed_items and all_reports and not batch_aggregate_saved:
                    try:
                        _bid = (review_context or {}).get("review_batch_id") or ""
                        _auto_partial = {
                            "file_name": (f"批量审核·批次汇总(部分完成)·{_bid}" if _bid else "批量审核（部分完成）")[:512],
                            "original_filename": (f"批量审核·批次汇总(部分完成)·{_bid}" if _bid else "批量审核（部分完成）")[:512],
                            "batch": True,
                            "reports": list(all_reports),
                            "total_points": sum(r.get("total_points", 0) for r in all_reports),
                            "high_count": sum(r.get("high_count", 0) for r in all_reports),
                            "medium_count": sum(r.get("medium_count", 0) for r in all_reports),
                            "low_count": sum(r.get("low_count", 0) for r in all_reports),
                            "info_count": sum(r.get("info_count", 0) for r in all_reports),
                            "summary": (
                                f"本批次部分完成（批次ID：{_bid}），成功 {len(all_reports)} / {total_files}；其余文件可重试。"
                                if _bid else f"本批次部分完成，成功 {len(all_reports)} / {total_files}；其余文件可重试。"
                            ),
                        }
                        _inject_review_meta(_auto_partial, "batch_aggregated")
                        _auto_partial_id = save_audit_report(
                            agent.collection_name,
                            _auto_partial,
                            model_info=get_current_model_info() or "",
                        )
                        _invalidate_audit_history_cache()
                        batch_aggregate_saved = True
                        st.session_state["review_batch_status_msg"] = (
                            f"本批部分完成：成功 {len(all_reports)} / {total_files}，失败 {len(failed_items)}。"
                            f" 已生成批次汇总（ID:{_auto_partial_id}）；可在下方点击“重新审核失败项”。"
                        )
                    except Exception as _save_err:
                        st.warning(f"自动生成部分完成批次汇总失败：{_save_err}")

                if all_reports:
                    st.session_state.review_reports = all_reports
                total_points = sum(r.get("total_points", 0) for r in all_reports)
                batch_extra = {
                        "total_files": total_files,
                        "success_count": len(all_reports),
                        "fail_count": total_files - len(all_reports),
                        "total_audit_points": total_points,
                        "duration_sec": round(total_time, 2),
                }
                if review_context.get("review_batch_id"):
                    batch_extra["review_batch_id"] = review_context["review_batch_id"]
                if failed_errors:
                    batch_extra["first_error"] = failed_errors[0]
                    batch_extra["errors"] = failed_errors
                add_operation_log(
                    op_type=OP_TYPE_REVIEW_BATCH,
                    collection=agent.collection_name,
                    file_name="",
                    source="upload",
                    extra=batch_extra,
                    model_info=get_current_model_info(),
                )
            finally:
                st.session_state.pop("review_batch_id", None)
                failed_set = {(p, d, a) for (p, d, a) in failed_items}
                for path, _display_name, is_from_archive in items:
                    if not is_from_archive and (path, _display_name, is_from_archive) not in failed_set:
                        Path(path).unlink(missing_ok=True)
                if failed_items:
                    st.session_state.review_failed_items = failed_items
                    st.session_state.review_failed_errors = list(failed_errors)
                    st.session_state.review_failed_temp_dirs = temp_dirs
                    st.session_state.review_success_reports = list(all_reports)
                    if "review_batch_status_msg" not in st.session_state:
                        _bid = (review_context or {}).get("review_batch_id") or "未分配"
                        st.session_state["review_batch_status_msg"] = (
                            f"本批未全部完成（批次号：{_bid}）：成功 {len(all_reports)} / {total_files}，失败 {len(failed_items)}。"
                            " 下方可查看失败原因并一键重试失败项。"
                        )
                else:
                    for d in temp_dirs:
                        shutil.rmtree(d, ignore_errors=True)

    if st.session_state.get("review_failed_items"):
        failed_list = st.session_state.review_failed_items
        failed_errs = st.session_state.get("review_failed_errors") or []
        st.warning(
            f"本批次有 **{len(failed_list)}** 个文件审核失败，可点击下方按钮仅对失败项重新审核（与已成功报告合并为同一批）。"
            " 重新审核时页面会显示进度，单份可能耗时较长，请勿重复点击。"
        )
        with st.expander("查看失败原因（便于排查接口/模型配置）", expanded=True):
            for i, (_path, dn, _) in enumerate(failed_list):
                err = failed_errs[i] if i < len(failed_errs) else ""
                err = (err or "").strip() or "（未记录到详细原因，可重新审核或重新上传该文件）"
                st.markdown(f"**{i + 1}. {dn}**")
                st.code(err[:12000], language="text")
        if st.button("🔄 重新审核失败项", key="retry_failed_review_btn"):
            agent = init_agent()
            project_id = st.session_state.get("review_project_id")
            review_context = dict(st.session_state.get("review_context") or {})
            review_sys_edit = st.session_state.get("review_sys_edit_step3", "") or st.session_state.get("review_sys_edit", "")
            review_usr_edit = st.session_state.get("review_usr_edit_step3", "") or st.session_state.get("review_usr_edit", "")
            review_extra_edit = st.session_state.get("review_extra_edit_step3", "") or st.session_state.get("review_extra_edit", "")
            merged = list(st.session_state.get("review_success_reports", []))
            _dirs = list(st.session_state.get("review_failed_temp_dirs", []))
            still_failed = []
            still_errors = []
            n_fail = len(failed_list)
            retry_bar = st.progress(0)
            retry_status = st.empty()
            retry_panel = st.empty()
            retry_t0 = time.time()

            def _render_retry_panel(done: int, total: int, failed_now: int, start_ts: float, current_name: str = ""):
                elapsed = max(0.0, time.time() - start_ts)
                avg = (elapsed / done) if done > 0 else 0.0
                remain = max(0, total - done)
                eta = (avg * remain) if done > 0 else 0.0
                cur = current_name or "等待中"
                retry_panel.markdown(
                    "\n".join([
                        "#### 🔁 失败项重试进度",
                        f"- 当前：`{cur}`",
                        f"- 进度：`{done}/{total}`",
                        f"- 仍失败：`{failed_now}`",
                        f"- 已用时：`{_format_time(elapsed)}`",
                        f"- 预计剩余：`{_format_time(eta) if done > 0 else '计算中…'}`",
                    ])
                )

            _render_retry_panel(0, n_fail, 0, retry_t0, "准备中")
            for idx, (path, display_name, is_from_archive) in enumerate(failed_list):
                retry_bar.progress(min(int((idx / max(n_fail, 1)) * 99), 99))
                retry_status.info(f"正在重新审核 **{idx + 1}/{n_fail}**：{display_name} …（请稍候，勿关闭页面）")
                _render_retry_panel(idx, n_fail, len(still_failed), retry_t0, display_name)
                if not Path(path).exists():
                    still_failed.append((path, display_name, is_from_archive))
                    still_errors.append(
                        f"临时文件已不存在（可能会话过期或已被清理），请在本页重新上传 **{display_name}** 后再审核。"
                    )
                    continue
                try:
                    t0 = time.time()
                    with st.spinner(f"AI 正在重新审核 [{display_name}]，请耐心等待…"):
                        report = _call_with_transient_retry(
                            lambda p=path, dn=display_name: agent.review(
                                p,
                                project_id=project_id,
                                review_context=review_context,
                                system_prompt=review_sys_edit.strip() or None,
                                user_prompt=review_usr_edit.strip() or None,
                                extra_instructions=review_extra_edit.strip() or None,
                                display_file_name=dn,
                            ),
                            attempts=_review_transient_retry_attempts(),
                        )
                    if (review_context.get("current_provider") or "") == "deepseek":
                        gc.collect()
                    report["original_filename"] = display_name
                    report["file_name"] = display_name
                    report["_original_path"] = path
                    _inject_review_meta(report)
                    try:
                        save_audit_report(agent.collection_name, report, model_info=get_current_model_info() or "")
                        _invalidate_audit_history_cache()
                    except Exception:
                        pass
                    merged.append(report)
                    mi = get_current_model_info()
                    add_operation_log(
                        op_type=OP_TYPE_REVIEW,
                        collection=agent.collection_name,
                        file_name=display_name,
                        source=str(path),
                        extra={
                            "total_points": report.get("total_points", 0),
                            "duration_sec": round(time.time() - t0, 2),
                            "retry_failed_batch": True,
                        },
                        model_info=mi,
                    )
                except Exception as e:
                    err_str = _format_review_exception(e)
                    still_failed.append((path, display_name, is_from_archive))
                    still_errors.append(err_str)
                    add_operation_log(
                        op_type="review_error",
                        collection=agent.collection_name,
                        file_name=display_name,
                        source=str(path),
                        extra={"error": err_str, "traceback": traceback.format_exc(), "retry_failed_batch": True},
                        model_info=get_current_model_info(),
                    )
                _render_retry_panel(idx + 1, n_fail, len(still_failed), retry_t0, display_name)
            retry_bar.progress(100)
            retry_status.empty()
            _render_retry_panel(n_fail, n_fail, len(still_failed), retry_t0, "重试完成")
            st.session_state.review_reports = merged
            if still_failed:
                st.session_state.review_failed_items = still_failed
                st.session_state.review_failed_errors = still_errors
                st.session_state.review_success_reports = merged
                st.session_state.review_failed_temp_dirs = _dirs
                st.session_state["review_retry_partial_msg"] = (
                    f"重新审核后仍有 **{len(still_failed)}** 个文件失败，**{n_fail - len(still_failed)}** 个已成功合并。"
                )
                st.experimental_rerun()
            else:
                for k in ("review_failed_items", "review_failed_errors", "review_failed_temp_dirs", "review_success_reports", "review_retry_partial_msg"):
                    st.session_state.pop(k, None)
                for d in _dirs:
                    shutil.rmtree(d, ignore_errors=True)
                st.session_state["review_retry_ok_msg"] = f"已对 **{n_fail}** 个失败项重新审核并合并到本批报告。"
                st.experimental_rerun()

    def _step3_tab_text_review():
        review_text_doc_lang = st.selectbox(
            "项目案例审核语言",
            DOC_LANG_OPTIONS,
            key="review_text_doc_lang",
            help="仅在审核上下文包含项目案例时生效：按所选语言规范审核。法规/程序知识库审核始终为所有语言通用。",
        )
        review_text = st.text_area(
            "输入待审核文本",
            height=300,
            placeholder="粘贴文档内容到这里...",
        )
        text_file_name = st.text_input("文件名（可选）", value="直接输入")
        text_ref_files = st.file_uploader(
            "参考文件（可选：本次审核将按参考文件要求对照核查）",
            type=["pdf", "docx", "doc", "xlsx", "xls", "txt", "md", "zip", "tar", "gz", "tgz"],
            accept_multiple_files=True,
            key="review_text_reference_uploader",
            help="可上传程序文件、技术要求、规范、追溯制度等作为本次审核的对照依据。系统会摘录其内容并注入审核上下文。",
        )

        if review_text and st.button("🔍 审核文本", key="review_text_btn"):
            agent = init_agent()
            text_status = st.empty()
            text_status.info(f"🔍 正在审核文本（{len(review_text)} 字）...")
            t0 = time.time()
            try:
                mi = get_current_model_info()
                _text_ctx = dict(review_context) if review_context else {}
                _tl = st.session_state.get("review_text_doc_lang") or "不指定"
                _text_ctx["document_language"] = DOC_LANG_LABEL_TO_VALUE.get(_tl, "")
                # 参考文件：注入审核上下文
                try:
                    if text_ref_files:
                        _ritems, _rdirs = _expand_uploads(text_ref_files)
                        _ref_blocks = []
                        _ref_cap = 18000
                        _used = 0
                        for _rp, _rdn, _rarch in _ritems:
                            try:
                                _docs = load_single_file(_rp)
                                _txt = "\n\n".join(getattr(d, "page_content", str(d)) for d in _docs) if _docs else ""
                            except Exception:
                                _txt = ""
                            if not _txt.strip():
                                continue
                            _seg = f"【参考文件：{_rdn}】\n{_txt.strip()}"
                            _ref_blocks.append(_seg)
                            _used += len(_seg)
                            if _used >= _ref_cap:
                                break
                        if _ref_blocks:
                            _text_ctx["reference_docs_excerpt"] = ("\n\n---\n\n".join(_ref_blocks))[:_ref_cap]
                        for _p, _dn, _arch in _ritems:
                            if not _arch:
                                Path(_p).unlink(missing_ok=True)
                        for _d in _rdirs:
                            shutil.rmtree(_d, ignore_errors=True)
                except Exception:
                    pass
                with st.spinner("AI 正在审核文本，请耐心等待..."):
                    report = _call_with_transient_retry(
                        lambda: agent.review_text(
                            review_text,
                            text_file_name,
                            project_id=project_id,
                            review_context=_text_ctx,
                            system_prompt=review_sys_edit.strip() or None,
                            user_prompt=review_usr_edit.strip() or None,
                            extra_instructions=review_extra_edit.strip() or None,
                        ),
                        attempts=_review_transient_retry_attempts(),
                    )
                elapsed = time.time() - t0
                _inject_review_meta(report)
                try:
                    save_audit_report(agent.collection_name, report, model_info=mi)
                    _invalidate_audit_history_cache()
                except Exception:
                    pass
                n_points = report.get("total_points", 0)
                add_operation_log(
                    op_type=OP_TYPE_REVIEW_TEXT,
                    collection=agent.collection_name,
                    file_name=text_file_name,
                    source="text_input",
                    extra={"total_points": n_points, "duration_sec": round(elapsed, 2), "text_length": len(review_text)},
                    model_info=mi,
                )
                st.session_state.review_reports = [report]
                text_status.success(
                    f"✅ 审核完成！发现 {n_points} 个审核点，耗时 {_format_time(elapsed)}"
                )
            except Exception as e:
                text_status.error(f"❌ 审核失败：{e}")
                add_operation_log(
                    op_type="review_text_error",
                    collection=agent.collection_name,
                    file_name=text_file_name,
                    source="text_input",
                    extra={"error": str(e)},
                    model_info=get_current_model_info(),
                )

    _step3_review_tab_labels = ["📤 上传文件审核", "📝 文本审核"]
    _st_run_tabs_or_pick(
        _step3_review_tab_labels,
        radio_label="第三步审核方式",
        session_key="step3_review_tabs",
        tab_bodies=[_step3_tab_upload_review, _step3_tab_text_review],
    )

    if "review_reports" in st.session_state and st.session_state.review_reports:
        st.markdown("---")
        reports_list = st.session_state.review_reports
        kdocs_url = st.session_state.get("review_kdocs_view_url") or (reports_list[0].get("_kdocs_view_url") if reports_list else None)
        if kdocs_url:
            from html import escape
            safe_url = escape(kdocs_url, quote=True)
            st.markdown(
                f'<a href="{safe_url}" target="_blank" rel="noopener noreferrer" style="font-size:1rem;">📎 在金山文档中直接打开（不下载到本地）</a>',
                unsafe_allow_html=True,
            )
            st.caption("审核问题点已返回系统，见下方报告；可打开在线文档对照修改。")
            st.markdown("---")
        render_reports(reports_list)

    # 历史审核报告
    st.markdown("---")
    st.subheader("📂 历史审核报告")
    st.caption(
        "提示：Streamlit 在切换下拉、修改输入时会重跑页面；已关闭「fastReruns」并固定报告编辑框高度，减轻误触刷新感。"
        " 长时间编辑时请避免同时打开多个本应用标签页。"
    )

    if st.session_state.get("_hist_editing_id") is not None:
        st.session_state["_step3_show_history_panel"] = True

    if not st.session_state.get("_step3_show_history_panel", False):
        st.caption(
            "为加快本页响应，**历史列表与整合报告**默认未加载；仅审核当前文档时可不展开。"
            " 需要查看/编辑历史、整合多份报告时再点下方按钮。"
        )
        if st.button("📂 加载历史审核与整合功能", key="_step3_open_history_panel"):
            st.session_state["_step3_show_history_panel"] = True
            _streamlit_rerun()
        if audit_perf_enabled():
            audit_perf_log("render_step3_page", (time.perf_counter() - _ap_t0) * 1000.0, "history_panel_collapsed")
        return

    st.caption("首次展开本区域会查询 MySQL 并渲染整合区；若长时间白屏请检查数据库连接或查看终端报错。")
    _hclose, _ = st.columns([1, 3])
    with _hclose:
        if st.session_state.get("_hist_editing_id") is None:
            if st.button("收起历史区域（加快后续操作）", key="_step3_close_history_panel"):
                st.session_state["_step3_show_history_panel"] = False
                _streamlit_rerun()
        else:
            st.caption("正在编辑历史报告时请先在上方「关闭编辑区」，再收起本区域。")

    with st.expander("📋 整合报告（将同一文件或所选多份报告的问题点合并为一份完整报告）", expanded=False):
        collection = st.session_state.get("collection_name", "regulations")
        st.caption("多次审核同一文件时，各次报告问题点可能不一致。可在此按「文件名」或「勾选多份报告」整合为一份去重后的完整报告。")
        mode = _st_radio_compat(
            "整合方式",
            ["按文件名整合（同一文件的所有历史报告）", "选择多份报告整合"],
            key="merge_report_mode",
        )
        merged_report = None
        if mode == "按文件名整合（同一文件的所有历史报告）":
            file_names = get_audit_report_file_names(collection)
            if not file_names:
                st.info("当前暂无历史报告，无法按文件名整合。")
            else:
                sel_name = st.selectbox("选择文件名", file_names, key="merge_by_file_sel")
                if st.button("生成整合报告", key="merge_by_file_btn"):
                    recs = get_audit_reports_by_file_name(collection, sel_name)
                    if len(recs) < 2:
                        st.warning("该文件仅有 0 或 1 份报告，无需整合；请选择有多份报告的文件。")
                    else:
                        merged_report = _merge_audit_reports_into_one(recs, merged_file_name=f"整合报告：{sel_name}")
                        st.session_state.merged_report_result = merged_report
        else:
            history = _get_cached_audit_reports(collection)
            if not history:
                st.info("当前暂无历史报告。")
            else:
                options = [f"{r.get('created_at','')} | {r.get('file_name','')} | {r.get('total_points',0)}个点 (ID:{r.get('id')})" for r in history]
                selected = st.multiselect("选择要整合的报告（可多选）", options, key="merge_select_reports")
                if st.button("将所选报告整合为一份", key="merge_selected_btn"):
                    if len(selected) < 2:
                        st.warning("请至少选择 2 份报告进行整合。")
                    else:
                        id_map = {f"{r.get('created_at','')} | {r.get('file_name','')} | {r.get('total_points',0)}个点 (ID:{r.get('id')})": r for r in history}
                        recs = [id_map[o] for o in selected if o in id_map]
                        merged_report = _merge_audit_reports_into_one(recs, merged_file_name="整合报告（多份）")
                        st.session_state.merged_report_result = merged_report
        if st.session_state.get("merged_report_result"):
            merged_report = st.session_state.merged_report_result
            if not merged_report.get("related_doc_names"):
                merged_report["related_doc_names"] = [merged_report.get("original_filename", merged_report.get("file_name", "整合报告"))]
            st.success(f"已生成整合报告：共 **{merged_report.get('total_points', 0)}** 个不重复问题点。")
            # 与主报告一致：表格摘要 + 下拉只展开一条审核点，避免 _render_multi_doc_report 对每条都建 text_area 导致极慢
            _render_reports_table_layout(
                [merged_report],
                base_key_prefix="merged_r",
                history_id=0,
                parent_batch_report=None,
                key_suffix="_merged_edit",
                allow_nested_expander=True,
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button("保存为历史报告", key="save_merged_report_btn"):
                    try:
                        if not merged_report.get("_review_meta"):
                            _inject_review_meta(merged_report)
                        save_audit_report(collection, merged_report, model_info=get_current_model_info())
                        st.session_state.pop("merged_report_result", None)
                        _invalidate_audit_history_cache()
                        st.success("已保存，可在上方历史报告中查看。")
                        _streamlit_rerun()
                    except Exception as e:
                        st.error(f"保存失败：{e}")
            with c2:
                if st.button("清除本次整合结果", key="clear_merged_btn"):
                    st.session_state.pop("merged_report_result", None)
                    st.experimental_rerun()
            st.markdown("**📥 下载整合报告**")
            fmt = st.selectbox("格式", ["Excel", "PDF", "Word", "HTML", "JSON", "Markdown"], key="merged_dl_fmt")
            report_to_dl = [merged_report]
            if fmt == "Excel":
                try:
                    st.download_button("下载", data=report_to_excel(report_to_dl), file_name="audit_report_merged.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_merged_xlsx")
                except Exception as e:
                    st.caption(f"Excel 生成失败: {e}")
            elif fmt == "PDF":
                try:
                    st.download_button("下载", data=report_to_pdf(report_to_dl), file_name="audit_report_merged.pdf", mime="application/pdf", key="dl_merged_pdf")
                except Exception as e:
                    st.caption(f"PDF 生成失败: {e}")
            elif fmt == "JSON":
                dl_data = json.dumps(report_to_dl, ensure_ascii=False, indent=2)
                st.download_button("下载", data=dl_data, file_name="audit_report_merged.json", mime="application/json", key="dl_merged_json")
            elif fmt == "HTML":
                st.download_button("下载", data=report_to_html(report_to_dl), file_name="audit_report_merged.html", mime="text/html", key="dl_merged_html")
            elif fmt == "Word":
                try:
                    st.download_button("下载", data=report_to_docx(report_to_dl), file_name="audit_report_merged.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", key="dl_merged_docx")
                except Exception as e:
                    st.caption(f"Word 生成失败: {e}")
            else:
                md = []
                for r in report_to_dl:
                    md.append(f"# 审核报告：{r.get('original_filename', r.get('file_name',''))}\n\n{r.get('summary','')}\n\n")
                    for i, p in enumerate(r.get("audit_points", []), 1):
                        md.append(f"## {i}. [{p.get('severity')}] {p.get('category','')}\n\n位置：{p.get('location','')}\n\n{p.get('description','')}\n\n**建议：** {p.get('suggestion','')}\n\n")
                st.download_button("下载", data="\n".join(md), file_name="audit_report_merged.md", mime="text/markdown", key="dl_merged_md")

    try:
        collection = st.session_state.get("collection_name", "regulations")
        try:
            history = _get_cached_audit_reports(collection)
        except Exception as _hist_load_ex:
            st.error(f"历史列表加载失败：{_hist_load_ex}")
            history = []
        _open_hist_err = st.session_state.pop("_hist_open_error", None)
        if _open_hist_err:
            st.error(_open_hist_err)
        hed = st.session_state.get("_hist_editing_id")
        er = st.session_state.get("_hist_editing_report")

        if hed is not None and er is not None:
            st.markdown("##### 📂 历史报告编辑区（仅加载当前这一条）")
            h1, h2 = st.columns([1, 3])
            with h1:
                if st.button("✖ 关闭编辑区", key="_hist_close_editor"):
                    _hid = st.session_state.get("_hist_editing_id")
                    st.session_state.pop("_hist_editing_id", None)
                    st.session_state.pop("_hist_editing_report", None)
                    st.session_state.pop("_hist_pending_corrections", None)
                    st.session_state.pop("_hist_draft_dirty", None)
                    if _hid is not None:
                        st.session_state.pop(f"_hist_dl_blob_{_hid}", None)
                    _invalidate_audit_history_cache()
                    _streamlit_rerun()
            with h2:
                st.caption(
                    f"报告 ID **{hed}**：审核点以**整表**展示与编辑，点 **✏编号** 展开纠正。"
                    f" 本地批量模式（默认开）下改完点顶部「一次性保存」写库；关闭则每表下可「保存本表到数据库」。"
                )
            _render_open_history_editor_body()
        else:
            r1, r2 = st.columns([1, 4])
            with r1:
                if st.button("🔄 刷新列表", key="_hist_refresh_list"):
                    _invalidate_audit_history_cache()
                    _streamlit_rerun()
            if not history:
                st.info("暂无历史审核报告。")
            else:
                st.caption(
                    f"共 **{len(history)}** 条；列表仅占用一个下拉框，**打开**后再加载该条全文，避免每次翻页都渲染全部报告。"
                )

                def _hist_row_label(i: int) -> str:
                    r = history[i]
                    rpt = r.get("report") or {}
                    _hm = _extract_history_meta(rpt)
                    _tag = f"[{_hm.get('audit_type', '通用审核')}] " if _hm else ""
                    bid = (
                        (_hm.get("review_batch_id") or rpt.get("review_batch_id") or "") or ""
                    ).strip()
                    rk = (rpt.get("review_batch_record_kind") or "").strip()
                    batch_part = ""
                    if bid:
                        if rk == "batch_aggregated" or rpt.get("batch"):
                            batch_part = f"[批次汇总·{bid}] "
                        elif rk == "batch_multi_doc":
                            batch_part = f"[多文档一致性·{bid}] "
                        elif rk == "batch_traceability":
                            batch_part = f"[跨文档可追溯·{bid}] "
                        elif rk == "batch_member":
                            batch_part = f"[批次单文·{bid}] "
                        else:
                            batch_part = f"[批次·{bid}] "
                    mid = r.get("model_info", "") or ""
                    suf = f" | {mid}" if mid else ""
                    return f"{r.get('created_at', '')} | {_tag}{batch_part}{r.get('file_name', '')} | {r.get('total_points', 0)}点 (ID:{r.get('id')}){suf}"

                st.selectbox(
                    "选择历史报告",
                    options=list(range(len(history))),
                    format_func=_hist_row_label,
                    key="_hist_pick_idx",
                )
                _pick_i = int(st.session_state.get("_hist_pick_idx", 0) or 0)
                if 0 <= _pick_i < len(history):
                    _rid = int(history[_pick_i].get("id") or 0)
                    if _rid > 0:
                        _lite_url = _fastapi_report_edit_url(_rid)
                        st.markdown(
                            f'<p><b>轻量编辑</b>（新窗口，免 Streamlit 整页重载）：'
                            f'<a href="{html.escape(_lite_url, quote=True)}" target="_blank" rel="noopener noreferrer">'
                            f"{html.escape(_lite_url, quote=False)}</a></p>",
                            unsafe_allow_html=True,
                        )
                        st.caption("若链接打不开，请设置环境变量 AICHECKWORD_API_PUBLIC_BASE 为浏览器可访问的 API 根地址（如 http://服务器IP:8000）。")
                # on_click 在本轮 rerun 的脚本主体之前执行，打开后同一次运行即可进入编辑区分支，
                # 避免先渲染列表再 st.rerun 与 st.fragment 二次重绘导致的「加载完又收起」。
                st.button(
                    "📂 打开选中报告",
                    key="_hist_open_sel",
                    on_click=_hist_open_selected_callback,
                )
    except Exception as e:
        st.warning(f"加载历史报告失败：{e}")

    if audit_perf_enabled():
        audit_perf_log("render_step3_page", (time.perf_counter() - _ap_t0) * 1000.0, "full")


_SEVERITY_ICONS = {"high": "🔴", "medium": "🟡", "low": "🔵", "info": "ℹ️"}
_SEVERITY_LABELS = {"high": "高", "medium": "中", "low": "低", "info": "提示"}


def _build_review_meta_dict() -> dict:
    """从当前 session_state 构建包含全部审核上下文的元信息 dict，用于持久化到报告/数据库。"""
    mode = st.session_state.get("_review_mode_snapshot") or st.session_state.get("review_mode", "仅通用审核")
    ctx = st.session_state.get("review_context") or {}
    is_project = mode == "按项目审核" or bool(ctx.get("_project_name"))
    meta: dict = {"audit_type": "项目审核" if is_project else "通用审核"}
    meta["project_name"] = ctx.get("_project_name", "") if is_project else ""
    meta["product_name"] = ctx.get("_product_name", "") if is_project else ""
    meta["model"] = ctx.get("_model") or ctx.get("model", "") if is_project else ""
    meta["model_en"] = ctx.get("_model_en") or ctx.get("model_en", "") if is_project else ""
    countries = ctx.get("registration_country") or []
    meta["registration_country"] = (
        "、".join(c for c in countries if c) if isinstance(countries, list) else str(countries)
    )
    reg_types = ctx.get("registration_type") or []
    meta["registration_type"] = (
        "、".join(t for t in reg_types if t) if isinstance(reg_types, list) else str(reg_types)
    )
    reg_comps = ctx.get("registration_component") or []
    meta["registration_component"] = (
        "、".join(c for c in reg_comps if c) if isinstance(reg_comps, list) else str(reg_comps)
    )
    proj_forms = ctx.get("project_form") or []
    meta["project_form"] = (
        "、".join(f for f in proj_forms if f) if isinstance(proj_forms, list) else str(proj_forms)
    )
    meta["document_language"] = ctx.get("document_language", "")
    meta["collection_name"] = st.session_state.get("collection_name", "")
    meta["review_mode"] = mode
    pid = st.session_state.get("review_project_id")
    if pid:
        meta["project_id"] = pid
    _rbid = (ctx.get("review_batch_id") or st.session_state.get("review_batch_id") or "").strip()
    if _rbid:
        meta["review_batch_id"] = _rbid
    return meta


def _inject_review_meta(report: dict, batch_record_kind: Optional[str] = None):
    """将当前审核模式/项目信息注入报告 dict，便于后续展示与持久化。
    batch_record_kind：批量审核时区分记录类型（batch_member / batch_aggregated / batch_multi_doc / batch_traceability）。"""
    report["_review_meta"] = _build_review_meta_dict()
    meta = report.get("_review_meta") or {}
    _bid = (meta.get("review_batch_id") or "").strip()
    if _bid:
        report["review_batch_id"] = _bid
    if batch_record_kind:
        report["review_batch_record_kind"] = batch_record_kind


def _post_audit_dim_first_choice(raw) -> str:
    """_review_meta 中维度可能为「中国、美国」式拼接；下拉框取第一个可选项。"""
    s = str(raw or "").strip()
    if not s:
        return ""
    for sep in ("、", ",", ";", "；"):
        if sep in s:
            return s.split(sep)[0].strip()
    return s


def _post_audit_form_meta_defaults(report: dict) -> dict:
    """合并报告内 _review_meta 与当前「③ 文档审核」侧栏 review_context，避免从历史交接后栏位全空。"""
    base = report.get("_review_meta") if isinstance(report.get("_review_meta"), dict) else {}
    if (not base) and report.get("batch") and isinstance(report.get("reports"), list):
        for sub in report.get("reports") or []:
            if isinstance(sub, dict) and isinstance(sub.get("_review_meta"), dict) and sub.get("_review_meta"):
                base = dict(sub["_review_meta"])
                break
    out: dict = {}
    for k, v in dict(base).items():
        if v is None:
            out[k] = ""
        elif isinstance(v, (dict, list)):
            out[k] = v
        else:
            out[k] = str(v)
    ctx = st.session_state.get("review_context") or {}

    def _from_ctx_list(meta_key: str, ctx_key: str) -> str:
        raw = ctx.get(ctx_key)
        if isinstance(raw, list):
            return "、".join(str(x).strip() for x in raw if str(x).strip())
        return str(raw or "").strip()

    pairs = (
        ("registration_country", "registration_country"),
        ("registration_type", "registration_type"),
        ("registration_component", "registration_component"),
        ("project_form", "project_form"),
    )
    for mk, ck in pairs:
        if not str(out.get(mk) or "").strip():
            out[mk] = _from_ctx_list(mk, ck)

    if not str(out.get("document_language") or "").strip():
        out["document_language"] = str(ctx.get("document_language") or "").strip()

    for mk in ("project_name", "product_name", "model", "model_en"):
        if not str(out.get(mk) or "").strip() and ctx.get(mk) is not None:
            out[mk] = str(ctx.get(mk) or "").strip()

    if not str(out.get("project_id") or "").strip():
        pid = st.session_state.get("review_project_id")
        if pid:
            out["project_id"] = pid

    return out


ACTION_OPTIONS = ["立即修改", "延期修改", "无需修改"]

def _get_multi_doc_default_action(severity: str) -> str:
    """多文档审核点按严重程度的默认处理状态，可配置。"""
    defaults = st.session_state.get("multi_doc_default_action") or {
        "high": "立即修改",
        "medium": "立即修改",
        "low": "延期修改",
        "info": "无需修改",
    }
    return defaults.get(severity, "无需修改")


def _build_post_audit_handoff(report: dict, *, source_report_id: int = 0) -> dict:
    """将审核报告转换为“审核后修改”交接结构；支持单份报告或 batch+reports 批次汇总（合并子报告审核点）。"""
    rep = dict(report or {})
    if source_report_id and not rep.get("id"):
        rep["id"] = int(source_report_id)
    payload = build_immediate_audit_remediation_by_target(
        rep,
        get_default_action=_get_multi_doc_default_action,
    )
    return {
        "version": 1,
        "source_report_id": int(source_report_id or rep.get("id") or 0),
        "source_file_name": rep.get("original_filename") or rep.get("file_name") or "",
        "report": rep,
        "points_by_target": payload.get("points_by_target") or {},
        "text_by_target": payload.get("text_by_target") or {},
        "all_points": payload.get("all_points") or [],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _coerce_modify_docs_list(raw) -> list:
    """将审核点 modify_docs 规范为字符串列表（支持 tuple、JSON 数组字符串、单字符串）。"""
    if raw is None:
        return []
    if isinstance(raw, tuple):
        raw = list(raw)
    if isinstance(raw, list):
        out = []
        for x in raw:
            if x is None:
                continue
            s = str(x).strip()
            if s:
                out.append(s)
        return out
    if isinstance(raw, str):
        t = raw.strip()
        if t.startswith("["):
            try:
                parsed = json.loads(t)
                if isinstance(parsed, (list, tuple)):
                    return _coerce_modify_docs_list(parsed)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        return [t] if t else []
    s = str(raw).strip()
    return [s] if s else []


def _unique_doc_display_names(doc_names: list) -> list:
    """related_doc_names 去重保序，避免重复 option 导致 multiselect 选中异常。"""
    seen = set()
    out: list = []
    for d in doc_names or []:
        if d is None:
            continue
        s = str(d).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _report_related_doc_names(report: Optional[dict], *, reports: Optional[list] = None, parent_batch_report: Optional[dict] = None) -> list:
    """为详情/纠正页兜底获取关联文档名：优先报告自身 related_doc_names；否则从父批量报告或同批 reports 推断。"""
    if isinstance(report, dict):
        rdn = report.get("related_doc_names")
        if isinstance(rdn, list) and _unique_doc_display_names(rdn):
            return _unique_doc_display_names(rdn)
    if isinstance(parent_batch_report, dict):
        prdn = parent_batch_report.get("related_doc_names")
        if isinstance(prdn, list) and _unique_doc_display_names(prdn):
            return _unique_doc_display_names(prdn)
    if isinstance(reports, list) and reports:
        inferred: list = []
        for r in reports:
            if not isinstance(r, dict):
                continue
            nm = (r.get("original_filename") or r.get("file_name") or "").strip()
            if nm:
                inferred.append(nm)
        u = _unique_doc_display_names(inferred)
        if u:
            return u
    # 最后兜底：至少放当前报告名，避免 options 为空导致无法下拉
    if isinstance(report, dict):
        nm0 = (report.get("original_filename") or report.get("file_name") or "").strip()
        if nm0:
            return [nm0]
    return []


def _point_modify_docs_merged_list(point: Optional[dict]) -> list:
    """合并审核点里可能分散在多个键下的「需修改文档」（如 modify_docs、需修改文档）。"""
    if not isinstance(point, dict):
        return []
    merged: list = []
    for k in ("modify_docs", "需修改文档", "modify_documents"):
        if k not in point:
            continue
        merged.extend(_coerce_modify_docs_list(point.get(k)))
    seen = set()
    out: list = []
    for x in merged:
        x = (x or "").strip()
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _effective_point_modify_docs_list(
    point: Optional[dict],
    related_doc_names: Optional[list] = None,
    file_name_hint: str = "",
) -> list:
    """历史表/导出用：优先合并 modify_docs 等键；仍为空时用本份报告的关联文档名或文件名推断（单文档常见）。"""
    md = _point_modify_docs_merged_list(point)
    if md:
        return md
    rel = related_doc_names if related_doc_names is not None else []
    u = _unique_doc_display_names(rel)
    if len(u) == 1:
        return list(u)
    if len(u) > 1:
        # 多文档：modify_docs 为空时与 UI 下拉一致，用位置/描述等推断涉及文档；推断不出则等同「关联文档全集」（与 multiselect 默认全选一致）
        return _point_modify_docs_dropdown_options(point or {}, rel)
    fn = (file_name_hint or "").strip()
    if fn and fn != "未知":
        return [fn]
    return []


def _doc_file_stem_cf(path: str) -> str:
    """文件名（不含路径）去扩展名后小写，用于「说明书」与「说明书.docx」对齐。"""
    base = (path or "").replace("\\", "/").split("/")[-1].strip()
    if not base:
        return ""
    b = base.casefold()
    return b.rsplit(".", 1)[0] if "." in b else b


def _strip_doc_display_wrappers(segment: str) -> str:
    """去掉显示名两侧的书名号、引号等（如 related_doc_names 为「xxx.docx」而 modify_docs 为 xxx.docx）。"""
    s = (segment or "").strip()
    for _ in range(4):
        changed = False
        if s.startswith(("\u300c", "\u300a", "\u201c", "\u2018", '"', "'")):
            s = s[1:].lstrip()
            changed = True
        if s.endswith(("\u300d", "\u300b", "\u201d", "\u2019", '"', "'")):
            s = s[:-1].rstrip()
            changed = True
        if not changed:
            break
    return s.strip()


def _match_doc_name_option(option: str, stored: str) -> bool:
    """option 为 related_doc_names 中一项；stored 为 modify_docs 中一项（可能为路径、仅文件名或简称）。"""
    o = (option or "").replace("\\", "/").strip()
    s = (stored or "").replace("\\", "/").strip()
    if not o or not s:
        return False
    ob = _strip_doc_display_wrappers(o.split("/")[-1])
    sb = _strip_doc_display_wrappers(s.split("/")[-1])
    ol, sl = ob.casefold(), sb.casefold()
    if (
        o == s
        or o.endswith("/" + s)
        or s.endswith("/" + o)
        or ol == sl
        or o.endswith("/" + sb)
        or s.endswith("/" + ob)
    ):
        return True
    stem_o = _doc_file_stem_cf(ob)
    stem_s = _doc_file_stem_cf(sb)
    if stem_o and stem_s and stem_o == stem_s:
        return True
    # 简称包含于全名（如「风险管理」↔「医疗器械风险管理报告.docx」），避免多选空白
    if len(sl) >= 4 and sl in ol:
        return True
    if len(ol) >= 4 and ol in sl:
        return True
    return False


def _point_modify_docs_dropdown_options(point: Optional[dict], related_doc_names: Optional[list]) -> list:
    """本审核点「需修改文档」下拉的选项：从位置/描述/建议等推断涉及的文档（related_doc_names 的子集）；推断不出则退回整份关联文档列表。"""
    if not isinstance(point, dict):
        point = {}
    rel = _unique_doc_display_names(related_doc_names or [])
    if not rel:
        # 某些报告/历史/纠正路径里 related_doc_names 可能缺失；至少用已存 modify_docs 兜底，避免 options 为空
        return _unique_doc_display_names(_point_modify_docs_merged_list(point))
    import re

    parts = [
        str(point.get("location") or ""),
        str(point.get("description") or ""),
        str(point.get("suggestion") or ""),
        str(point.get("category") or ""),
    ]
    blob = " ".join(parts)
    blob_cf = blob.casefold()
    involved: list = []
    seen = set()

    def _add(nm: str) -> None:
        s = (nm or "").strip()
        if not s or s in seen:
            return
        involved.append(s)
        seen.add(s)

    for d in rel:
        d_strip = str(d).strip()
        if not d_strip:
            continue
        base = Path(d_strip.replace("\\", "/")).name
        base_cf = base.casefold()
        if d_strip in blob:
            _add(d_strip)
        elif base in blob:
            _add(d_strip)
        elif len(base_cf) >= 4 and base_cf in blob_cf:
            _add(d_strip)

    for m in re.finditer(r"\[([^\]\n]{2,300}?)\]", blob):
        inner = (m.group(1) or "").strip()
        if not inner:
            continue
        inner_base = Path(inner.replace("\\", "/")).name
        inner_cf = inner_base.casefold()
        for d in rel:
            ds = str(d).strip()
            if not ds:
                continue
            db = Path(ds.replace("\\", "/")).name
            dbcf = db.casefold()
            if inner == ds or inner_cf == dbcf or inner_base in ds or db in inner:
                _add(ds)

    for sd in _point_modify_docs_merged_list(point):
        for d in rel:
            ds = str(d).strip()
            if ds and _match_doc_name_option(ds, sd):
                _add(ds)
                break

    if involved:
        return _unique_doc_display_names(involved)
    return list(rel)


def _default_modify_docs_selection(doc_names: list, raw_modify_docs) -> list:
    """计算「需修改的文档」多选应对齐的选项（返回值均为 doc_names 中原字符串）。

    设计取舍：
    - 审核结果里的「需修改文档」下拉应避免出现“有选项可选、但默认勾选为空”导致用户误以为
      “没有需要修改的文档”的情况。
    - 若审核点未显式给出 modify_docs：
        - 单文档（names 仅 1 个）→ 默认选中该唯一项；
        - 多文档 → 默认全选关联文档（related_doc_names），由用户按需取消即可。
    - 若审核点给出了 modify_docs 但与 options 对不上：回退到与“未给出”相同的默认策略。
    """
    names = [str(d).strip() for d in (doc_names or []) if d is not None and str(d).strip()]
    stored = _coerce_modify_docs_list(raw_modify_docs)
    if not names:
        return []
    if not stored:
        return list(names)
    matched = [d for d in names if any(_match_doc_name_option(d, c) for c in stored)]
    return matched if matched else list(names)


def _modify_docs_options_and_picked(doc_names: list, raw_modify_docs) -> Tuple[list, list]:
    """构建 multiselect 的 options 与默认选中项。

    Streamlit 会丢弃不在 options 里的选中值；若只传 related_doc_names，而 modify_docs 里是模型写的别名/未入库名，
    界面会变成空白。故把「与任何文档名都对不上」的已存项追加进 options，并按 coerced 映射回 options 中的规范字符串。
    """
    names = [str(d).strip() for d in (doc_names or []) if d is not None and str(d).strip()]
    coerced = _coerce_modify_docs_list(raw_modify_docs)
    extra: list = []
    seen_name = set(names)
    for c in coerced:
        if c in seen_name:
            continue
        if any(_match_doc_name_option(n, c) for n in names):
            continue
        if c not in extra:
            extra.append(c)
    options = names + extra

    def _picked() -> list:
        if not coerced:
            return _default_modify_docs_selection(names, [])
        out: list = []
        seen = set()
        for c in coerced:
            hit = None
            if c in options:
                hit = c
            else:
                for opt in options:
                    if opt in seen:
                        continue
                    if _match_doc_name_option(opt, c):
                        hit = opt
                        break
            if hit is not None and hit not in seen:
                out.append(hit)
                seen.add(hit)
        if out:
            return out
        return [x for x in coerced if x in options]

    return options, _picked()


def _caption_stored_modify_docs(point: dict) -> None:
    """多选框偶发空白时，仍展示审核点 JSON 里已存的文档列表（只读）。"""
    md = _point_modify_docs_merged_list(point)
    if md:
        st.caption("📎 **本审核点已记录的需修改文档：** " + "、".join(md))


def _multiselect_modify_docs(label: str, doc_names: list, raw_modify_docs, *, key: str) -> list:
    """需修改文档多选：路径/文件名对齐；当审核点数据变化时刷新 session（避免 key 已存在时 default= 永远不生效）。

    当 related_doc_names 暂时为空时，必须返回已存储的 modify_docs，不能返回 []，否则每轮脚本会把报告里的列表写空。
    """
    names = _unique_doc_display_names(doc_names)
    coerced = _coerce_modify_docs_list(raw_modify_docs)
    if not names:
        # 极端兜底：报告未提供 related_doc_names 时，至少用已存 modify_docs / 当前 session 值作为 options
        # 否则会出现「已有 chips，但下拉 No results」的空选项状态
        fallback_opts = _unique_doc_display_names(coerced)
        if not fallback_opts:
            _cur = st.session_state.get(key)
            if isinstance(_cur, tuple):
                _cur = list(_cur)
            if isinstance(_cur, list):
                fallback_opts = _unique_doc_display_names(_cur)
        if not fallback_opts:
            return coerced
        if key not in st.session_state:
            st.session_state[key] = list(fallback_opts)
        return st.multiselect(label, options=fallback_opts, key=key)
    options, default_docs = _modify_docs_options_and_picked(names, raw_modify_docs)
    # 额外兜底：把当前会话里已选值也塞进 options，避免出现「已有 chips 但下拉 No results」
    _cur_for_opts = st.session_state.get(key)
    if isinstance(_cur_for_opts, tuple):
        _cur_for_opts = list(_cur_for_opts)
    if isinstance(_cur_for_opts, list) and _cur_for_opts:
        for x in _unique_doc_display_names(_cur_for_opts):
            if x and x not in options:
                options.append(x)
    if not options:
        # 额外兜底：极端情况下 options 仍可能为空，强制退回 names/coerced，避免出现「chips 存在但下拉 No results」
        options = list(names) if names else _unique_doc_display_names(coerced)
        default_docs = _default_modify_docs_selection(options, coerced) if options else list(coerced)
    opt_set = set(options)

    # 重要：不要同时使用 default= 与 session_state 赋值，否则 Streamlit 会报错
    # 仅在「审核点切换/选项变化/源数据变化」时初始化默认；允许用户手动清空而不被强制回填
    sig = (tuple(options), tuple(coerced))
    sk = f"{key}__modify_docs_sig"
    if key not in st.session_state or st.session_state.get(sk) != sig:
        st.session_state[key] = list(default_docs)
        st.session_state[sk] = sig
    else:
        _cur = st.session_state.get(key)
        if isinstance(_cur, tuple):
            _cur = list(_cur)
        if isinstance(_cur, list):
            # options 变化后只剔除无效项，不要在用户清空时强制回填
            valid = [x for x in _cur if x in opt_set]
            if valid != _cur:
                st.session_state[key] = valid
        elif default_docs:
            st.session_state[key] = list(default_docs)

    try:
        _cur2 = st.session_state.get(key)
        if isinstance(_cur2, tuple):
            _cur2 = list(_cur2)
        if isinstance(_cur2, list) and options and len(_cur2) >= len(options):
            st.caption("提示：下拉中不显示已选项；取消勾选请点击上方标签右侧「×」。")
    except Exception:
        pass

    return st.multiselect(label, options=options, key=key)


def _audit_point_modify_docs_key(report_id: int, point_idx: int, scope_key: str, extra_suffix: str = "") -> str:
    """同一条审核点在「普通编辑」与「纠正表单」中共用 multiselect 的 session key。

    否则纠正用 `cf_*_modify_docs` 会从 point 初始化出有值，而详情里用 `*_detail_*_docs` 仍是空的旧状态。
    """
    sk = str(scope_key).replace(" ", "_")[:200]
    return f"_apt_md_{int(report_id)}_{int(point_idx)}_{sk}{extra_suffix}"


def _render_multi_doc_report(report: dict, r_idx: int, reports: list, key_prefix: str = "", history_id: int = 0, parent_batch_report: dict = None):
    """通用审核报告渲染：分区域展示；可编辑内容、需修改文档多选、处理状态、纠正、一键待办。history_id>0 时显示纠正此审核点。parent_batch_report 为批量报告时纠正会更新整份报告。"""
    file_name = report.get("original_filename", report.get("file_name", "审核报告"))
    doc_names = _report_related_doc_names(report, reports=reports, parent_batch_report=parent_batch_report)
    points = report.get("audit_points") or []
    pk = key_prefix or "r"

    # ─── 区域一：报告概览 ───
    st.markdown("### 一、报告概览")
    st.markdown(f"**📄 {file_name}**")
    cols = st.columns(4)
    cols[0].metric("🔴 高风险", report.get("high_count", 0))
    cols[1].metric("🟡 中风险", report.get("medium_count", 0))
    cols[2].metric("🔵 低风险", report.get("low_count", 0))
    cols[3].metric("ℹ️ 提示", report.get("info_count", 0))
    if report.get("summary"):
        _markdown_text_with_hover_title("总结：", report["summary"], max_preview=220)
    st.markdown("---")

    # ─── 区域二：默认状态配置 ───
    st.markdown("### 二、审核点默认状态配置")
    if "multi_doc_default_action" not in st.session_state:
        st.session_state.multi_doc_default_action = {
            "high": "立即修改", "medium": "立即修改", "low": "延期修改", "info": "无需修改",
        }
    cfg = st.session_state.multi_doc_default_action.copy()
    c1, c2, c3 = st.columns(3)
    with c1:
        cfg["high"] = st.selectbox("高风险/中风险默认", ACTION_OPTIONS, index=ACTION_OPTIONS.index(cfg.get("high", "立即修改")), key=f"{pk}_cfg_high")
        cfg["medium"] = cfg["high"]
    with c2:
        cfg["low"] = st.selectbox("低风险默认", ACTION_OPTIONS, index=ACTION_OPTIONS.index(cfg.get("low", "延期修改")), key=f"{pk}_cfg_low")
    with c3:
        cfg["info"] = st.selectbox("提示默认", ACTION_OPTIONS, index=ACTION_OPTIONS.index(cfg.get("info", "无需修改")), key=f"{pk}_cfg_info")
    st.session_state.multi_doc_default_action = cfg
    st.caption("新建审核点将按严重程度应用上述默认。")
    st.markdown("---")

    # ─── 区域三：审核点列表 ───
    st.markdown("### 三、审核点列表")

    for i, point in enumerate(points):
        severity = point.get("severity", "info")
        icon = _SEVERITY_ICONS.get(severity, "ℹ️")
        st.markdown(f"**{icon} 审核点 {i + 1}：{point.get('category', '未分类')}**")
        one_based = i + 1
        in_correction = history_id and st.session_state.get(f"editing_{history_id}_{one_based}")

        _md_unify = _audit_point_modify_docs_key(int(history_id or 0), i, f"{pk}_{r_idx}", "")
        if in_correction:
            # 纠正模式：仅显示纠正表单（一组可编辑 + 保存纠正 / 取消纠正）；批量报告时传入父报告以便保存整份
            _render_correction_form(
                report, history_id, i, point,
                parent_batch_report=parent_batch_report,
                sub_report_index=r_idx if parent_batch_report else None,
                modify_docs_widget_key=_md_unify,
            )
        else:
            # 正常：可编辑描述/建议、需修改文档、处理状态
            key_desc = f"{pk}_desc_{r_idx}_{i}"
            key_sug = f"{pk}_sug_{r_idx}_{i}"
            key_loc = f"{pk}_loc_{r_idx}_{i}"
            st.caption(f"**类别（只读）** {point.get('category') or '—'}")
            new_loc = st.text_input("位置", value=point.get("location", "") or "", key=key_loc)
            st.caption(f"**法规依据（只读）** {point.get('regulation_ref') or '—'}")
            # 高度固定：避免随字数变化导致 text_area 重建、整页重跑后焦点丢失（像「自动刷新」）
            new_desc = st.text_area("问题描述", value=point.get("description", ""), height=120, key=key_desc)
            new_sug = st.text_area("修改建议（请写明需修改哪份或哪几份文档）", value=point.get("suggestion", ""), height=120, key=key_sug)

            _opt_md = _point_modify_docs_dropdown_options(point, doc_names)
            _caption_stored_modify_docs(point)
            new_modify_docs = _multiselect_modify_docs(
                "需修改的文档", _opt_md, _point_modify_docs_merged_list(point), key=_md_unify
            )

            key_action = f"{pk}_action_{r_idx}_{i}"
            current_action = point.get("action") or _get_multi_doc_default_action(severity)
            if current_action not in ACTION_OPTIONS:
                current_action = _get_multi_doc_default_action(severity)
            idx_opt = ACTION_OPTIONS.index(current_action) if current_action in ACTION_OPTIONS else 0
            new_action = st.selectbox("处理状态", ACTION_OPTIONS, index=idx_opt, key=key_action)

            if r_idx < len(reports) and i < len(reports[r_idx].get("audit_points", [])):
                reports[r_idx]["audit_points"][i]["location"] = new_loc
                reports[r_idx]["audit_points"][i]["description"] = new_desc
                reports[r_idx]["audit_points"][i]["suggestion"] = new_sug
                reports[r_idx]["audit_points"][i]["modify_docs"] = new_modify_docs
                reports[r_idx]["audit_points"][i]["action"] = new_action

            if history_id:
                btn_key = f"{pk}_correct_{history_id}_{i}"
                if st.button(f"✏️ 纠正此审核点", key=btn_key):
                    st.session_state[f"editing_{history_id}_{one_based}"] = True
                    st.experimental_rerun()

        st.markdown("---")

    # ─── 区域四：待办任务 ───
    st.markdown("### 四、待办任务（仅含「立即修改」项）")
    report_updated = reports[r_idx] if r_idx < len(reports) else report
    points_updated = report_updated.get("audit_points") or []
    immediate = [p for p in points_updated if (p.get("action") or _get_multi_doc_default_action(p.get("severity", "info"))) == "立即修改"]
    if immediate:
        todo_lines = []
        for idx, p in enumerate(immediate, 1):
            docs = p.get("modify_docs") or []
            # 多文档时以「文档名称1，文档名称2，文档名称3」形式显示（路径只取最后一段，逗号隔开）
            doc_displays = [(d.replace("\\", "/").split("/")[-1] if d else "") for d in docs]
            docs_str = "，".join(d for d in doc_displays if d) or "见修改建议"
            todo_lines.append(f"**{idx}. 涉及文档：{docs_str}**")
            todo_lines.append(f"   [{p.get('category', '')}] {p.get('description', '')[:120]}{'…' if len(p.get('description','')) > 120 else ''}")
            todo_lines.append(f"   修改建议：{p.get('suggestion', '')[:150]}{'…' if len(p.get('suggestion','')) > 150 else ''}")
            todo_lines.append("")
        todo_text = "\n".join(todo_lines)
        st.text_area("待办列表", value=todo_text, height=280, key=f"{pk}_todo_preview", disabled=True)
        _default_act = st.session_state.get("multi_doc_default_action") or {
            "high": "立即修改", "medium": "立即修改", "low": "延期修改", "info": "无需修改",
        }
        def _get_action(s):
            return _default_act.get((s or "info").lower(), "无需修改")
        _rmeta = report_updated.get("_review_meta") or {}
        todo_cols = st.columns(5)
        with todo_cols[0]:
            st.download_button("📥 待办（文本）", data=todo_text, file_name="audit_todo.txt", mime="text/plain", key=f"{pk}_todo_dl")
        with todo_cols[1]:
            try:
                csv_bytes = report_todo_to_csv(
                    [report_updated],
                    only_immediate=True,
                    get_default_action=_get_action,
                    project_name=_rmeta.get("project_name", ""),
                    product=_rmeta.get("product_name", ""),
                    country=_rmeta.get("registration_country", ""),
                )
                st.download_button(
                    "📥 待办（CSV）",
                    data=csv_bytes,
                    file_name="待办导入_审核待办.csv",
                    mime="text/csv; charset=utf-8",
                    key=f"{pk}_todo_csv_dl",
                    help="与待办导入模板同格式，可直接导入制氧机等管理系统",
                )
            except Exception as _e:
                st.caption(f"CSV 生成失败: {_e}")
        with todo_cols[2]:
            try:
                pdf_bytes = report_todo_to_pdf(
                    [report_updated],
                    only_immediate=True,
                    get_default_action=_get_action,
                    project_name=_rmeta.get("project_name", ""),
                    product=_rmeta.get("product_name", ""),
                    country=_rmeta.get("registration_country", ""),
                )
                st.download_button("📥 待办（PDF）", data=pdf_bytes, file_name="audit_todo.pdf", mime="application/pdf", key=f"{pk}_todo_pdf_dl")
            except Exception as _e:
                st.caption(f"PDF 生成失败: {_e}")
        with todo_cols[3]:
            try:
                docx_bytes = report_todo_to_docx(
                    [report_updated],
                    only_immediate=True,
                    get_default_action=_get_action,
                    project_name=_rmeta.get("project_name", ""),
                    product=_rmeta.get("product_name", ""),
                    country=_rmeta.get("registration_country", ""),
                )
                st.download_button("📥 待办（Word）", data=docx_bytes, file_name="audit_todo.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", key=f"{pk}_todo_docx_dl")
            except Exception as _e:
                st.caption(f"Word 生成失败: {_e}")
        with todo_cols[4]:
            try:
                xlsx_bytes = report_todo_to_excel(
                    [report_updated],
                    only_immediate=True,
                    get_default_action=_get_action,
                    project_name=_rmeta.get("project_name", ""),
                    product=_rmeta.get("product_name", ""),
                    country=_rmeta.get("registration_country", ""),
                )
                st.download_button("📥 待办（Excel）", data=xlsx_bytes, file_name="audit_todo.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"{pk}_todo_xlsx_dl")
            except Exception as _e:
                st.caption(f"Excel 生成失败: {_e}")
    else:
        st.caption("暂无标记为「立即修改」的审核点；可在上方将处理状态设为「立即修改」后刷新。")


def _render_report_flat(report: dict, history_id: int = 0):
    """平铺渲染单份报告（不使用 expander，适合嵌套在 expander 内的场景）
    history_id > 0 时显示纠正按钮。"""
    file_name = report.get("original_filename", report.get("file_name", "未知"))
    st.markdown(f"**📄 {file_name}**")
    cols = st.columns(4)
    cols[0].metric("🔴 高风险", report.get("high_count", 0))
    cols[1].metric("🟡 中风险", report.get("medium_count", 0))
    cols[2].metric("🔵 低风险", report.get("low_count", 0))
    cols[3].metric("ℹ️ 提示", report.get("info_count", 0))

    if report.get("summary"):
        _markdown_text_with_hover_title("总结：", report["summary"], max_preview=220)
    st.markdown("---")

    for i, point in enumerate(report.get("audit_points", []), 1):
        severity = point.get("severity", "info")
        icon = _SEVERITY_ICONS.get(severity, "ℹ️")
        st.markdown(f"**{icon} 审核点 {i}：{point.get('category', '未分类')}**")
        c1, c2 = st.columns(2)
        c1.markdown(f"严重程度：`{severity}`")
        c2.markdown(f"位置：{point.get('location', '未知')}")
        st.markdown(f"问题描述：{point.get('description', '')}")
        st.markdown(f"法规依据：{point.get('regulation_ref', '')}")
        st.markdown(f"修改建议：{point.get('suggestion', '')}")

        if history_id:
            btn_key = f"correct_{history_id}_{i}"
            if st.button(f"✏️ 纠正此审核点", key=btn_key):
                st.session_state[f"editing_{history_id}_{i}"] = True

            if st.session_state.get(f"editing_{history_id}_{i}"):
                _md_flat = _audit_point_modify_docs_key(int(history_id), i - 1, f"flat_{history_id}", "")
                _render_correction_form(report, history_id, i - 1, point, modify_docs_widget_key=_md_flat)

        st.markdown("---")


def _normalize_text_for_dedup(s: str) -> str:
    """去空格、标点，用于去重比较。"""
    if not s:
        return ""
    import re
    s = re.sub(r"\s+", "", str(s))
    s = re.sub(r"[，。、；：！？\"'\s\u3000]", "", s)
    return s.strip()


def _ngrams(s: str, n: int = 2) -> set:
    """字符级 n-gram 集合，用于中文相似度。"""
    s = _normalize_text_for_dedup(s)
    if len(s) < n:
        return {s} if s else set()
    return set(s[i : i + n] for i in range(len(s) - n + 1))


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _is_same_issue_point(p1: dict, p2: dict, desc_jaccard_threshold: float = 0.42) -> bool:
    """
    判断两个审核点是否为同一问题或意思相同仅表述不同。
    规则：位置相关（相同或一方包含另一方）+ 描述/建议语义相似（n-gram Jaccard 或包含关系）。
    """
    loc1 = _normalize_text_for_dedup(p1.get("location") or "")
    loc2 = _normalize_text_for_dedup(p2.get("location") or "")
    desc1 = (p1.get("description") or "").strip()
    desc2 = (p2.get("description") or "").strip()
    sug1 = (p1.get("suggestion") or "").strip()
    sug2 = (p2.get("suggestion") or "").strip()
    # 位置：完全一致，或较短者被较长者包含（同一处）
    loc_ok = loc1 == loc2 or (len(loc1) >= 2 and len(loc2) >= 2 and (loc1 in loc2 or loc2 in loc1))
    if not loc_ok and loc1 and loc2:
        # 允许位置部分重叠（如「第3章」与「第三章」）
        g1, g2 = _ngrams(loc1, 2), _ngrams(loc2, 2)
        loc_ok = _jaccard(g1, g2) >= 0.5
    if not loc_ok:
        return False
    # 描述相似：n-gram Jaccard 或短描述被长描述包含
    d1, d2 = _ngrams(desc1, 2), _ngrams(desc2, 2)
    desc_sim = _jaccard(d1, d2)
    desc_contain = (len(desc1) >= 5 and len(desc2) >= 5 and (desc1 in desc2 or desc2 in desc1))
    if desc_sim >= desc_jaccard_threshold or desc_contain:
        return True
    # 建议相似也视为同一问题
    s1, s2 = _ngrams(sug1, 2), _ngrams(sug2, 2)
    sug_sim = _jaccard(s1, s2)
    sug_contain = (len(sug1) >= 5 and len(sug2) >= 5 and (sug1 in sug2 or sug2 in sug1))
    if sug_sim >= desc_jaccard_threshold or sug_contain:
        return True
    # 描述+建议合并做一次整体相似度
    combined1 = _ngrams(desc1 + sug1, 2)
    combined2 = _ngrams(desc2 + sug2, 2)
    return _jaccard(combined1, combined2) >= desc_jaccard_threshold


def _merge_audit_reports_into_one(reports: list, merged_file_name: str = "") -> dict:
    """
    将多份审核报告的审核点合并为一份，做去重处理：
    - 同一问题点（位置+描述一致）只保留一条；
    - 意思相同仅表述不同的问题点视为同一条，只保留一条（保留严重程度更高的）。
    按严重程度排序输出。
    """
    severity_order = {"high": 0, "medium": 1, "low": 2, "info": 3}
    all_points = []
    for rec in reports:
        rpt = rec.get("report") if isinstance(rec.get("report"), dict) else {}
        if not rpt:
            try:
                rpt = json.loads(rec.get("report_json") or "{}")
            except Exception:
                continue
        for p in rpt.get("audit_points", []):
            all_points.append(dict(p))
    # 去重：遍历每条，若与已选列表中某条为同一/同义问题，则保留严重度更高的
    merged = []
    for p in all_points:
        found_dup = False
        for i, existing in enumerate(merged):
            if _is_same_issue_point(p, existing):
                found_dup = True
                # 若新点严重度更高，用新点替换已选中的
                if severity_order.get(p.get("severity", "info"), 99) < severity_order.get(
                    existing.get("severity", "info"), 99
                ):
                    merged[i] = p
                break
        if not found_dup:
            merged.append(p)
    merged.sort(key=lambda x: (severity_order.get(x.get("severity", "info"), 99), x.get("category", "")))
    high = sum(1 for x in merged if x.get("severity") == "high")
    medium = sum(1 for x in merged if x.get("severity") == "medium")
    low = sum(1 for x in merged if x.get("severity") == "low")
    info = sum(1 for x in merged if x.get("severity") == "info")
    name = merged_file_name or "整合报告"
    # 整合报告的可选文档列表：来自被整合的各报告文件名，供「需修改的文档」多选
    related_doc_names = []
    seen = set()
    for rec in reports:
        rpt = rec.get("report") if isinstance(rec.get("report"), dict) else {}
        if not rpt:
            try:
                rpt = json.loads(rec.get("report_json") or "{}")
            except Exception:
                continue
        fn = rpt.get("original_filename") or rpt.get("file_name") or ""
        if fn and fn not in seen:
            seen.add(fn)
            related_doc_names.append(fn)
    return {
        "file_name": name,
        "original_filename": name,
        "audit_points": merged,
        "total_points": len(merged),
        "high_count": high,
        "medium_count": medium,
        "low_count": low,
        "info_count": info,
        "related_doc_names": related_doc_names,
        "summary": f"整合自 {len(reports)} 份审核报告，共 {len(merged)} 个不重复问题点（已对同一问题及表述不同但意思相同项去重）。",
    }


def _extract_history_meta(rpt: dict) -> dict:
    """从历史报告中提取审核元数据（兼容批量和单份）。"""
    if not rpt:
        return {}
    meta = rpt.get("_review_meta")
    if meta:
        return meta
    if rpt.get("batch") and rpt.get("reports"):
        for sub in rpt["reports"]:
            m = sub.get("_review_meta")
            if m:
                return m
    return {}


def _render_history_meta_header(rpt: dict, report_id: int):
    """在历史报告头部渲染审核类型和项目信息（单行）。"""
    meta = _extract_history_meta(rpt)
    if not meta:
        return
    audit_type = meta.get("audit_type", "通用审核")
    parts = [f"**审核类型：** {audit_type}"]
    if audit_type == "项目审核":
        if meta.get("project_name"):
            parts.append(f"**项目：** {meta['project_name']}")
        if meta.get("product_name"):
            parts.append(f"**产品：** {meta['product_name']}")
        if meta.get("registration_country"):
            parts.append(f"**国家：** {meta['registration_country']}")
    st.markdown("　|　".join(parts))
    bid = (meta.get("review_batch_id") or rpt.get("review_batch_id") or "").strip()
    rk = (rpt.get("review_batch_record_kind") or "").strip()
    if bid:
        _kind_map = {
            "batch_aggregated": "批次汇总（多份整合）",
            "batch_member": "批次内单文档",
            "batch_multi_doc": "多文档一致性（同属该批次）",
            "batch_traceability": "跨文档可追溯性审核（同属该批次）",
        }
        st.caption(f"归属批次：`{bid}` · {_kind_map.get(rk, '批次相关')}")
    st.markdown("---")


_HISTORY_DL_HEAVY_FORMATS = frozenset({"Excel", "PDF", "Word", "HTML"})


def _build_history_report_download_payload(reports: list, fmt: str) -> tuple:
    """生成历史报告下载二进制/文本及 mime、扩展名。失败时抛出异常。"""
    if fmt == "Excel":
        data = report_to_excel(reports)
        return (
            data,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "xlsx",
        )
    if fmt == "JSON":
        return (
            json.dumps(reports, ensure_ascii=False, indent=2).encode("utf-8"),
            "application/json",
            "json",
        )
    if fmt == "HTML":
        return (report_to_html(reports).encode("utf-8"), "text/html", "html")
    if fmt == "PDF":
        return (report_to_pdf(reports), "application/pdf", "pdf")
    if fmt == "Word":
        return (
            report_to_docx(reports),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "docx",
        )
    # Markdown
    md_lines = []
    for r in reports:
        fn = r.get("original_filename", r.get("file_name", ""))
        md_lines.append(f"# 审核报告：{fn}\n")
        md_lines.append(f"**总结：** {r.get('summary', '')}\n")
        md_lines.append("| 高风险 | 中风险 | 低风险 | 提示 |")
        md_lines.append("|--------|--------|--------|------|")
        md_lines.append(
            f"| {r.get('high_count', 0)} | {r.get('medium_count', 0)} "
            f"| {r.get('low_count', 0)} | {r.get('info_count', 0)} |\n"
        )
        for i, point in enumerate(r.get("audit_points", []), 1):
            sev_l = {"high": "高", "medium": "中", "low": "低", "info": "提示"}.get(
                (point.get("severity") or "").lower(), point.get("severity", "")
            )
            md_lines.append(f"## [{sev_l}] 审核点 {i}：{point.get('category', '')}")
            md_lines.append(f"- **位置：** {point.get('location', '')}")
            md_lines.append(f"- **描述：** {point.get('description', '')}")
            md_lines.append(f"- **法规依据：** {point.get('regulation_ref', '')}")
            md_lines.append(f"- **建议：** {point.get('suggestion', '')}")
            _docs = point.get("modify_docs") or []
            if _docs:
                md_lines.append(f"- **需修改文档：** {'、'.join(d for d in _docs if d)}")
            _action = point.get("action") or ""
            if _action:
                md_lines.append(f"- **处理状态：** {_action}")
            md_lines.append("")
    text = "\n".join(md_lines)
    return (text.encode("utf-8"), "text/markdown", "md")


def _render_history_report_download(report_id: int, report: dict, display_name: str):
    """历史报告区域：格式下拉 + 下载按钮。batch 报告导出其 reports 列表。

    重格式（Excel/PDF/Word/HTML）改为点击「生成」后再下载，避免每次 rerun 都全量导出导致长时间阻塞甚至白屏。
    """
    st.markdown("**📥 下载此报告**")
    safe_name = "".join(c for c in (display_name or "report") if c.isalnum() or c in "._- ").strip() or "report"
    safe_name = safe_name[:64]

    # 默认 JSON，避免一打开历史报告就同步跑 Excel（大报告会卡死/白屏）
    format_options = ["JSON", "Markdown", "HTML", "Excel", "PDF", "Word"]
    idx = st.selectbox(
        "选择下载格式",
        format_options,
        key=f"hist_fmt_{report_id}",
    )
    if report.get("batch") and report.get("reports"):
        reports = report["reports"]
    else:
        reports = [report]

    cache_key = f"_hist_dl_blob_{report_id}"
    cache = st.session_state.get(cache_key)
    if cache and cache.get("fmt") != idx:
        st.session_state.pop(cache_key, None)
        cache = None

    if idx in _HISTORY_DL_HEAVY_FORMATS:
        if cache and cache.get("fmt") == idx:
            data, mime, ext = cache["data"], cache["mime"], cache["ext"]
        else:
            st.caption("Excel / PDF / Word / HTML 生成较慢，请先点击下方按钮生成，再下载（避免页面长时间无响应）。")
            if st.button("⏳ 生成下载文件", key=f"hist_gen_{report_id}"):
                try:
                    data, mime, ext = _build_history_report_download_payload(reports, idx)
                    st.session_state[cache_key] = {"fmt": idx, "data": data, "mime": mime, "ext": ext}
                    _streamlit_rerun()
                except Exception as e:
                    st.error(f"生成失败：{e}")
            return
    else:
        try:
            data, mime, ext = _build_history_report_download_payload(reports, idx)
        except Exception as e:
            st.error(f"生成失败：{e}")
            return

    st.download_button(
        f"📥 下载 {idx}",
        data=data,
        file_name=f"audit_report_{safe_name}.{ext}",
        mime=mime,
        key=f"hist_dl_{report_id}",
    )


def _render_open_history_editor_body():
    """历史报告编辑区主体（表格、审核点、下载）。从 session_state 读取。

    注意：此处不使用 st.fragment。fragment 在「打开报告」后易触发额外局部重跑，表现为内容加载完又被收起。
    """
    hed = st.session_state.get("_hist_editing_id")
    er = st.session_state.get("_hist_editing_report")
    if hed is None or er is None:
        return
    if "_hist_local_draft_mode" not in st.session_state:
        st.session_state["_hist_local_draft_mode"] = True
    st.checkbox(
        "本地批量模式：表格与纠正先写内存，最后点「一次性保存」入库并刷新。",
        key="_hist_local_draft_mode",
    )
    _hu = _fastapi_report_edit_url(int(hed))
    st.markdown(
        f'<p><b>轻量编辑（新窗口）</b>：<a href="{html.escape(_hu, quote=True)}" target="_blank" rel="noopener noreferrer">'
        f"{html.escape(_hu, quote=False)}</a></p>",
        unsafe_allow_html=True,
    )
    local_draft = bool(st.session_state.get("_hist_local_draft_mode", True))
    n_pending = len(st.session_state.get("_hist_pending_corrections") or [])
    dirty = bool(st.session_state.get("_hist_draft_dirty"))
    if local_draft:
        tb1, tb2, tb3 = st.columns([2, 1, 1])
        with tb1:
            st.caption(
                f"草稿状态：{'有未入库的编辑' if dirty else '无普通编辑未入库'}；"
                f"排队中的纠正记录 **{n_pending}** 条（保存时一并写入）。"
            )
        with tb2:
            if st.button(
                "💾 一次性保存到数据库",
                key="_hist_save_all_db",
                disabled=not (dirty or n_pending > 0),
                help="写入 MySQL 并从库刷新本条报告；纠正勾选「入知识库」时在保存时触发（异步线程）",
            ):
                ok, err = _hist_save_all_history_to_db(int(hed), er)
                if ok:
                    st.success("已保存并从数据库重新加载。")
                    _streamlit_rerun()
                else:
                    st.error(f"保存失败：{err}")
        with tb3:
            if st.button("↺ 放弃草稿并从数据库重新加载", key="_hist_reload_from_db"):
                row = get_audit_report_by_id(int(hed))
                if row and isinstance(row.get("report"), dict):
                    st.session_state["_hist_editing_report"] = copy.deepcopy(row["report"])
                st.session_state["_hist_pending_corrections"] = []
                st.session_state["_hist_draft_dirty"] = False
                st.session_state["_hist_widget_gen"] = int(st.session_state.get("_hist_widget_gen", 0)) + 1
                st.info("已从数据库重新加载；未保存的本地修改已丢弃。")
                _streamlit_rerun()
    fname = er.get("file_name") or er.get("original_filename") or "report"
    _render_history_meta_header(er, hed)
    _wg = int(st.session_state.get("_hist_widget_gen", 0))
    _ksuf = f"_hist_{hed}_g{_wg}"
    try:
        if er.get("batch") and er.get("reports"):
            _render_reports_table_layout(
                er["reports"],
                base_key_prefix=f"hist_{hed}_r",
                history_id=hed,
                parent_batch_report=er,
                key_suffix=_ksuf,
                allow_nested_expander=False,
                history_local_draft=local_draft,
            )
        else:
            if not er.get("related_doc_names"):
                er["related_doc_names"] = [er.get("original_filename", er.get("file_name", fname or "未知"))]
            _render_reports_table_layout(
                [er],
                base_key_prefix=f"hist_{hed}",
                history_id=hed,
                parent_batch_report=None,
                key_suffix=_ksuf,
                allow_nested_expander=False,
                history_local_draft=local_draft,
            )
    except Exception as ex:
        st.error(f"历史报告渲染失败：{ex}")
        try:
            st.exception(ex)
        except Exception:
            st.text(traceback.format_exc())
        return
    _render_history_report_download(hed, er, fname or "report")


def _aggregate_batch_report_totals(parent: dict) -> None:
    """根据子报告汇总批量报告的 total_points 与各严重程度计数。"""
    from src.core.audit_report_utils import aggregate_batch_report_totals

    aggregate_batch_report_totals(parent)


def _render_correction_form(
    report: dict,
    report_id: int,
    point_idx: int,
    point: dict,
    parent_batch_report: dict = None,
    sub_report_index: int = None,
    close_state_keys: list = None,
    local_draft: bool = False,
    modify_docs_widget_key: str = None,
):
    """纠正表单：可编辑描述、建议、需修改的文档、处理状态等，保存纠正 / 取消纠正。批量报告时传入 parent_batch_report 与 sub_report_index 以正确写回整份报告。
    close_state_keys：保存/取消后额外从 session_state 中 pop 的 key（如历史报告详情内的纠正开关）。
    local_draft=True：只改内存并排队纠正记录，由历史区「一次性保存」写库。
    modify_docs_widget_key：与「普通审核点编辑」共用 multiselect 的 session key。"""
    prefix = f"cf_{report_id}_{point_idx}_{sub_report_index if sub_report_index is not None else 'x'}"
    _md_key = modify_docs_widget_key if modify_docs_widget_key else f"{prefix}_modify_docs"
    doc_names = _report_related_doc_names(report, parent_batch_report=parent_batch_report)
    opt_modify_docs = _point_modify_docs_dropdown_options(point, doc_names)

    _kind_default = 0
    if point.get("correction_kind") == "false_positive" or point.get("false_positive_reason"):
        _kind_default = 1
    elif point.get("deprecated"):
        _kind_default = 2
    corr_kind = _st_radio_compat(
        "纠正方式",
        ["修订本条内容", "标记为误报", "弃用本条"],
        index=min(_kind_default, 2),
        key=f"{prefix}_kind",
    )

    if corr_kind == "修订本条内容":
        st.caption(f"**类别（只读）** {point.get('category') or '—'}")
        st.caption(f"**法规依据（只读）** {point.get('regulation_ref') or '—'}")
        new_desc = st.text_area("问题描述", value=point.get("description", ""), key=f"{prefix}_desc")
        _sev_list = ["high", "medium", "low", "info"]
        _sv = (point.get("severity") or "info").lower()
        _sev_i = _sev_list.index(_sv) if _sv in _sev_list else 3
        new_sev = st.selectbox("严重程度", _sev_list, index=_sev_i, key=f"{prefix}_sev")
        new_sug = st.text_area("修改建议", value=point.get("suggestion", ""), key=f"{prefix}_sug")
        if opt_modify_docs:
            _caption_stored_modify_docs(point)
            new_modify_docs = _multiselect_modify_docs(
                "需修改的文档", opt_modify_docs, _point_modify_docs_merged_list(point), key=_md_key
            )
        else:
            new_modify_docs = _point_modify_docs_merged_list(point)
        current_action = point.get("action") or _get_multi_doc_default_action(point.get("severity", "info"))
        if current_action not in ACTION_OPTIONS:
            current_action = ACTION_OPTIONS[0]
        new_action = st.selectbox("处理状态", ACTION_OPTIONS, index=ACTION_OPTIONS.index(current_action), key=f"{prefix}_action")
    elif corr_kind == "标记为误报":
        st.caption("误报：本条不再作为有效审核问题，处理状态将设为「无需修改」。")
        fp_reason = st.text_area("误报原因（必填）", value=point.get("false_positive_reason") or "", key=f"{prefix}_fp_reason", placeholder="例如：与法规/产品实际不符、模型误判等")
    else:
        st.caption("弃用：本条从有效问题清单中移除（不计入统计）；可选在下方新增一条您认为正确的审核结论。")
        dep_note = st.text_area("弃用说明（可选）", value=point.get("deprecation_note") or "", key=f"{prefix}_dep_note")
        add_replace = st.checkbox("同时新增一条正确的审核点（插入本条之后）", value=False, key=f"{prefix}_add_replace")
        if add_replace:
            st.markdown("**新增审核点**")
            n_cat = st.text_input("类别", value=point.get("category") or "一致性", key=f"{prefix}_n_cat")
            n_loc = st.text_input("位置", value="", key=f"{prefix}_n_loc", placeholder="文档中的章节或位置")
            n_desc = st.text_area("问题描述（正确结论）", value="", key=f"{prefix}_n_desc", height=80)
            n_ref = st.text_area("法规依据", value="", key=f"{prefix}_n_ref", height=60)
            n_sug = st.text_area("修改建议", value="", key=f"{prefix}_n_sug", height=80)
            n_sev = st.selectbox("严重程度", ["high", "medium", "low", "info"], index=2, key=f"{prefix}_n_sev")
            if doc_names:
                n_docs = st.multiselect("需修改的文档", options=doc_names, default=list(doc_names), key=f"{prefix}_n_docs")
            else:
                n_docs = []
            n_action = st.selectbox("处理状态", ACTION_OPTIONS, index=0, key=f"{prefix}_n_action")

    feed_kb = st.checkbox("将纠正内容写入知识库（下次审核可参考）", value=True, key=f"{prefix}_feed")

    def _persist(corrected_point: dict, original_snap: dict):
        collection = st.session_state.get("collection_name", "regulations")
        save_audit_correction(
            report_id=report_id,
            point_index=point_idx,
            collection=collection,
            file_name=report.get("file_name", ""),
            original=original_snap,
            corrected=corrected_point,
            fed_to_kb=feed_kb,
        )
        if feed_kb:
            _feed_correction_to_kb(collection, report.get("file_name", ""), corrected_point)
        add_operation_log(
            op_type=OP_TYPE_CORRECTION,
            collection=collection,
            file_name=report.get("file_name", ""),
            extra={"report_id": report_id, "point_index": point_idx, "fed_to_kb": feed_kb, "corrected": corrected_point},
            model_info=get_current_model_info(),
        )

    col_save, col_cancel, _ = st.columns([1, 1, 2])
    with col_save:
        if st.button("💾 保存纠正", key=f"{prefix}_save"):
            collection = st.session_state.get("collection_name", "regulations")
            original_snap = dict(point)
            try:
                points = report.get("audit_points", [])
                if not (0 <= point_idx < len(points)):
                    st.error("审核点索引无效")
                else:
                    if corr_kind == "标记为误报":
                        reason = (st.session_state.get(f"{prefix}_fp_reason") or "").strip()
                        if not reason:
                            st.error("请填写误报原因")
                        else:
                            corrected_point = dict(point)
                            corrected_point["correction_kind"] = "false_positive"
                            corrected_point["false_positive_reason"] = reason
                            corrected_point["action"] = "无需修改"
                            corrected_point["suggestion"] = (point.get("suggestion") or "").strip() + (f"\n\n【误报说明】{reason}" if reason else "")
                            points[point_idx] = corrected_point
                            _recount_severity(report)
                            if local_draft:
                                _hist_append_pending_correction(
                                    report_id=report_id,
                                    point_index=point_idx,
                                    report=report,
                                    original=original_snap,
                                    corrected=corrected_point,
                                    feed_kb=feed_kb,
                                )
                                if parent_batch_report is not None and sub_report_index is not None:
                                    _aggregate_batch_report_totals(parent_batch_report)
                                st.success(
                                    "已写入本地草稿（误报已排队）。保存时将写入数据库"
                                    + ("与知识库。" if feed_kb else "。")
                                )
                            else:
                                _persist(corrected_point, original_snap)
                                if parent_batch_report is not None and sub_report_index is not None:
                                    _aggregate_batch_report_totals(parent_batch_report)
                                    _timed_update_audit_report(report_id, parent_batch_report)
                                else:
                                    _timed_update_audit_report(report_id, report)
                                st.success("已标记为误报并保存。" + (" 已写入知识库。" if feed_kb else ""))
                                _invalidate_audit_history_cache()
                            st.session_state[f"editing_{report_id}_{point_idx + 1}"] = False
                            for _k in close_state_keys or []:
                                st.session_state.pop(_k, None)
                            _streamlit_rerun()
                    elif corr_kind == "弃用本条":
                        dep_note_val = (st.session_state.get(f"{prefix}_dep_note") or "").strip()
                        add_rep = st.session_state.get(f"{prefix}_add_replace", False)
                        base = dict(point)
                        base["deprecated"] = True
                        base["correction_kind"] = "deprecated"
                        base["deprecation_note"] = dep_note_val
                        base["action"] = "无需修改"
                        new_inserted = None
                        if add_rep:
                            n_cat_v = (st.session_state.get(f"{prefix}_n_cat") or "").strip() or "一致性"
                            n_loc_v = (st.session_state.get(f"{prefix}_n_loc") or "").strip()
                            n_desc_v = (st.session_state.get(f"{prefix}_n_desc") or "").strip()
                            n_ref_v = (st.session_state.get(f"{prefix}_n_ref") or "").strip()
                            n_sug_v = (st.session_state.get(f"{prefix}_n_sug") or "").strip()
                            n_sev_v = st.session_state.get(f"{prefix}_n_sev") or "low"
                            n_docs_v = st.session_state.get(f"{prefix}_n_docs") if doc_names else (point.get("modify_docs") or [])
                            if not isinstance(n_docs_v, list):
                                n_docs_v = list(doc_names) if doc_names else []
                            n_action_v = st.session_state.get(f"{prefix}_n_action") or "立即修改"
                            if not n_desc_v.strip():
                                st.error("新增审核点请填写「问题描述」")
                            else:
                                new_inserted = {
                                    "category": n_cat_v,
                                    "location": n_loc_v,
                                    "description": n_desc_v,
                                    "regulation_ref": n_ref_v,
                                    "suggestion": n_sug_v,
                                    "severity": n_sev_v,
                                    "modify_docs": n_docs_v,
                                    "action": n_action_v,
                                    "correction_kind": "user_replacement",
                                    "replaces_deprecated_index": point_idx,
                                }
                        if add_rep and not new_inserted:
                            pass
                        else:
                            points[point_idx] = base
                            corrected_for_log = dict(base)
                            if new_inserted:
                                points.insert(point_idx + 1, new_inserted)
                                corrected_for_log["replacement_point_added"] = new_inserted
                            _recount_severity(report)
                            if local_draft:
                                _hist_append_pending_correction(
                                    report_id=report_id,
                                    point_index=point_idx,
                                    report=report,
                                    original=original_snap,
                                    corrected=corrected_for_log,
                                    feed_kb=feed_kb,
                                )
                                if parent_batch_report is not None and sub_report_index is not None:
                                    _aggregate_batch_report_totals(parent_batch_report)
                                st.success(
                                    "已写入本地草稿（弃用/替换已排队）。保存时将写入数据库"
                                    + ("与知识库。" if feed_kb else "。")
                                )
                            else:
                                _persist(corrected_for_log, original_snap)
                                if parent_batch_report is not None and sub_report_index is not None:
                                    _aggregate_batch_report_totals(parent_batch_report)
                                    _timed_update_audit_report(report_id, parent_batch_report)
                                else:
                                    _timed_update_audit_report(report_id, report)
                                st.success("已弃用本条" + ("，并已插入新审核点。" if new_inserted else "。") + (" 已写入知识库。" if feed_kb else ""))
                                _invalidate_audit_history_cache()
                            st.session_state[f"editing_{report_id}_{point_idx + 1}"] = False
                            for _k in close_state_keys or []:
                                st.session_state.pop(_k, None)
                            _streamlit_rerun()
                    else:
                        corrected_point = dict(point)
                        corrected_point["description"] = st.session_state.get(f"{prefix}_desc", point.get("description", ""))
                        corrected_point["severity"] = st.session_state.get(f"{prefix}_sev", point.get("severity", "info"))
                        corrected_point["suggestion"] = st.session_state.get(f"{prefix}_sug", point.get("suggestion", ""))
                        corrected_point["modify_docs"] = (
                            st.session_state.get(_md_key)
                            if doc_names
                            else (point.get("modify_docs") or [])
                        )
                        corrected_point["action"] = st.session_state.get(f"{prefix}_action", ACTION_OPTIONS[0])
                        corrected_point.pop("correction_kind", None)
                        corrected_point.pop("false_positive_reason", None)
                        corrected_point.pop("deprecated", None)
                        points[point_idx] = corrected_point
                        _recount_severity(report)
                        if local_draft:
                            _hist_append_pending_correction(
                                report_id=report_id,
                                point_index=point_idx,
                                report=report,
                                original=original_snap,
                                corrected=corrected_point,
                                feed_kb=feed_kb,
                            )
                            if parent_batch_report is not None and sub_report_index is not None:
                                _aggregate_batch_report_totals(parent_batch_report)
                            st.success(
                                "已写入本地草稿（修订已排队）。保存时将写入数据库"
                                + ("与知识库。" if feed_kb else "。")
                            )
                        else:
                            _persist(corrected_point, original_snap)
                            if parent_batch_report is not None and sub_report_index is not None:
                                _aggregate_batch_report_totals(parent_batch_report)
                                _timed_update_audit_report(report_id, parent_batch_report)
                            else:
                                _timed_update_audit_report(report_id, report)
                            st.success("纠正已保存！" + ("已写入知识库。" if feed_kb else ""))
                            _invalidate_audit_history_cache()
                        st.session_state[f"editing_{report_id}_{point_idx + 1}"] = False
                        for _k in close_state_keys or []:
                            st.session_state.pop(_k, None)
                        _streamlit_rerun()
            except Exception as e:
                st.error(f"保存纠正失败：{e}")
    with col_cancel:
        if st.button("取消纠正", key=f"{prefix}_cancel"):
            st.session_state[f"editing_{report_id}_{point_idx + 1}"] = False
            for _k in close_state_keys or []:
                st.session_state.pop(_k, None)
            _streamlit_rerun()


def _markdown_text_with_hover_title(label: str, text: str, max_preview: int = 200) -> None:
    """长文本用浏览器原生 title 悬停显示全文（cursor:help）。"""
    t = (text or "").strip()
    if not t:
        return
    lab = html.escape(label, quote=False)
    if len(t) <= max_preview:
        st.markdown(f"**{lab}** {html.escape(t, quote=False)}")
        return
    preview = t[:max_preview] + "…"
    title_attr = html.escape(t.replace("\r", " ")[:12000], quote=True)
    prev_esc = html.escape(preview, quote=False)
    st.markdown(
        f'<p><strong>{lab}</strong> '
        f'<span style="border-bottom:1px dotted #888;cursor:help" title="{title_attr}">{prev_esc}</span></p>',
        unsafe_allow_html=True,
    )


def _streamlit_rerun():
    """统一 rerun，避免 experimental_rerun 在部分版本上卡住或行为异常。"""
    fn = getattr(st, "rerun", None)
    if callable(fn):
        fn()
    else:
        st.experimental_rerun()


def _auto_append_repeated_errors_to_skills_rules(report: dict) -> None:
    """
    将“同一份审核报告内重复出现的错误”自动沉淀到 skills/rules（追加去重）。
    目的：减少后续生成/审核中的重复错误与重复点位输出。
    """
    try:
        points = report.get("audit_points") or []
        if not points:
            return
        import re as _re

        def _norm(s: str) -> str:
            t = (s or "").strip().lower()
            t = _re.sub(r"\s+", " ", t)
            return t

        counts = {}
        examples = {}
        for p in points:
            cat = _norm(p.get("category") or "")
            desc = _norm(p.get("description") or "")
            if not desc:
                continue
            k = (cat, desc)
            counts[k] = counts.get(k, 0) + 1
            if k not in examples:
                examples[k] = {
                    "category": (p.get("category") or "").strip(),
                    "description": (p.get("description") or "").strip(),
                    "suggestion": (p.get("suggestion") or "").strip(),
                    "regulation_ref": (p.get("regulation_ref") or "").strip(),
                }

        repeated = [(k, c) for k, c in counts.items() if c >= 2]
        if not repeated:
            return

        # 避免一次报告重复写入多次（页面 rerun 会触发）
        ss = st.session_state.setdefault("_auto_sr_added_keys", set())
        new_items = []
        for (cat, desc), c in sorted(repeated, key=lambda x: -x[1]):
            key = f"{cat}|{desc}"
            if key in ss:
                continue
            ss.add(key)
            new_items.append((examples[(cat, desc)], c))

        if not new_items:
            return

        lines = []
        for ex, c in new_items[:12]:
            desc0 = ex.get("description") or ""
            sug0 = ex.get("suggestion") or ""
            reg0 = ex.get("regulation_ref") or ""
            cat0 = ex.get("category") or ""
            lines.append(f"- 【重复出现 {c} 次】{cat0}：{desc0}")
            if sug0:
                lines.append(f"  - 建议：{sug0}")
            if reg0:
                lines.append(f"  - 依据：{reg0}")
        block = "\n".join(lines).strip()
        if not block:
            return

        skills_patch_text = (
            "### FILE: .cursor/skills/rewrite-to-pass-audit-no-ai-tone/SKILL.md\n\n"
            "## 自动沉淀：重复审核错误（系统追加）\n"
            "以下条目来自近期审核报告中“同类问题重复出现”的自动汇总，用于后续生成/改写时优先修正。\n\n"
            f"{block}\n"
        )
        rules_patch_text = (
            "### FILE: .cursor/rules/document-authoring-and-audit.mdc\n\n"
            "## 自动沉淀：重复审核错误（系统追加）\n"
            "当同类问题在同一份报告内多次出现时，后续生成/改写应优先在文档中定位并一次性修正，避免重复遗漏。\n\n"
            f"{block}\n"
        )

        apply_patch_updates(
            skills_patch_text=skills_patch_text,
            rules_patch_text=rules_patch_text,
            workspace_root=_root,
        )
    except Exception:
        # 自动沉淀不应阻塞主流程
        return


def _invalidate_audit_history_cache():
    """写入历史报告/纠正后调用，下次再读列表时重新查库。"""
    st.session_state["_audit_hist_reload"] = True
    try:
        invalidate_audit_reports_list_cache()
    except Exception:
        pass


def _timed_update_audit_report(report_id: int, report: dict) -> None:
    """update_audit_report 包装：开启 audit_perf_log 时输出 MySQL 写入耗时。"""
    t0 = time.perf_counter()
    update_audit_report(report_id, report)
    if audit_perf_enabled():
        audit_perf_log("update_audit_report", (time.perf_counter() - t0) * 1000.0, f"id={report_id}")


def _hist_append_pending_correction(
    *,
    report_id: int,
    point_index: int,
    report: dict,
    original: dict,
    corrected: dict,
    feed_kb: bool,
) -> None:
    """本地批量模式：纠正先记入队列，一次性保存时再写 audit_corrections + 可选向量库。"""
    collection = st.session_state.get("collection_name", "regulations")
    fn = report.get("file_name", "") or report.get("original_filename", "")
    lst = st.session_state.setdefault("_hist_pending_corrections", [])
    lst.append(
        {
            "report_id": report_id,
            "point_index": point_index,
            "collection": collection,
            "file_name": fn,
            "original": original,
            "corrected": corrected,
            "feed_kb": bool(feed_kb),
        }
    )
    st.session_state["_hist_draft_dirty"] = True


def _hist_save_all_history_to_db(hed: int, er: dict) -> tuple[bool, str]:
    """刷出待写入的纠正记录 + 整包更新 audit_reports，再从库拉取一次替换内存草稿。"""
    try:
        pending = list(st.session_state.get("_hist_pending_corrections") or [])
        dirty = bool(st.session_state.get("_hist_draft_dirty"))
        for item in pending:
            save_audit_correction(
                report_id=item["report_id"],
                point_index=item["point_index"],
                collection=item.get("collection") or st.session_state.get("collection_name", "regulations"),
                file_name=item.get("file_name", ""),
                original=item["original"],
                corrected=item["corrected"],
                fed_to_kb=bool(item.get("feed_kb")),
            )
            if item.get("feed_kb"):
                _feed_correction_to_kb(
                    item.get("collection") or st.session_state.get("collection_name", "regulations"),
                    item.get("file_name", ""),
                    item["corrected"],
                )
            add_operation_log(
                op_type=OP_TYPE_CORRECTION,
                collection=item.get("collection") or st.session_state.get("collection_name", "regulations"),
                file_name=item.get("file_name", ""),
                extra={
                    "report_id": item["report_id"],
                    "point_index": item["point_index"],
                    "fed_to_kb": item.get("feed_kb"),
                    "corrected": item["corrected"],
                },
                model_info=get_current_model_info(),
            )
        st.session_state["_hist_pending_corrections"] = []
        if pending or dirty:
            _timed_update_audit_report(hed, er)
        row = get_audit_report_by_id(hed)
        if row and isinstance(row.get("report"), dict):
            st.session_state["_hist_editing_report"] = copy.deepcopy(row["report"])
        st.session_state["_hist_draft_dirty"] = False
        st.session_state["_hist_widget_gen"] = int(st.session_state.get("_hist_widget_gen", 0)) + 1
        _invalidate_audit_history_cache()
        return True, ""
    except Exception as ex:
        return False, str(ex)


def _get_cached_audit_reports(collection: str, limit: int = 100) -> list:
    """同一次浏览复用列表，避免每次整页重跑都查库；写库后需 _invalidate_audit_history_cache。"""
    snap_k = "_audit_hist_snap_v1"
    if st.session_state.pop("_audit_hist_reload", None):
        st.session_state.pop(snap_k, None)
    snap = st.session_state.get(snap_k)
    if not snap or snap.get("c") != collection:
        st.session_state[snap_k] = {
            "c": collection,
            "rows": get_audit_reports(collection=collection, limit=limit),
        }
    return st.session_state[snap_k]["rows"]


def _hist_open_selected_callback():
    """「打开选中报告」的 on_click：在本轮脚本主体执行前写入 _hist_editing_*，避免先画列表再 rerun/fragment 导致界面像自动收起。"""
    st.session_state.pop("_hist_open_error", None)
    collection = st.session_state.get("collection_name", "regulations")
    history = _get_cached_audit_reports(collection)
    if not history:
        st.session_state["_hist_open_error"] = "暂无历史审核报告。"
        return
    sel_i = st.session_state.get("_hist_pick_idx", 0)
    try:
        sel_i = int(sel_i)
    except (TypeError, ValueError):
        sel_i = 0
    if sel_i < 0 or sel_i >= len(history):
        st.session_state["_hist_open_error"] = "当前选择无效，请重新选择列表中的报告。"
        return
    rec = history[sel_i]
    rpt = rec.get("report")
    if not rpt and rec.get("report_json"):
        try:
            rpt = json.loads(rec["report_json"])
        except Exception:
            rpt = {}
    if not rpt:
        st.session_state["_hist_open_error"] = "该记录没有可展示的报告正文。"
        return
    try:
        st.session_state["_hist_editing_id"] = rec["id"]
        st.session_state["_hist_editing_report"] = copy.deepcopy(rpt)
        st.session_state["_step3_show_history_panel"] = True
        st.session_state["_hist_pending_corrections"] = []
        st.session_state["_hist_draft_dirty"] = False
        st.session_state["_hist_widget_gen"] = int(st.session_state.get("_hist_widget_gen", 0)) + 1
    except Exception as ex:
        st.session_state["_hist_open_error"] = f"加载报告到编辑区失败：{ex}"


def _point_row_label(p: dict, j: int) -> str:
    pre = "〔弃〕" if p.get("deprecated") else ("〔误〕" if p.get("correction_kind") == "false_positive" or p.get("false_positive_reason") else "")
    cat = (p.get("category") or "")[:24]
    d = (p.get("description") or "").replace("\n", " ")[:42]
    return f"{j + 1}. {pre}{cat} — {d}{'…' if len(p.get('description') or '') > 42 else ''}"


# 历史报告表格编辑：分页行数（减轻单次 rerun 体量）
HIST_EDIT_PAGE_SIZE = 20


def _fastapi_report_edit_url(report_id: int) -> str:
    """浏览器可打开的 FastAPI 轻量编辑页。优先环境变量 AICHECKWORD_API_PUBLIC_BASE（如 http://10.x.x.x:8000）。"""
    base = (os.environ.get("AICHECKWORD_API_PUBLIC_BASE") or "").strip().rstrip("/")
    if not base:
        port = int(getattr(settings, "api_port", 8000) or 8000)
        base = f"http://127.0.0.1:{port}"
    return f"{base}/api/reports/{int(report_id)}/edit"


def _st_try_data_editor(df: pd.DataFrame, *, column_config=None, **kwargs) -> Optional[pd.DataFrame]:
    """兼容旧版 Streamlit：优先 data_editor，其次 experimental_data_editor；均不可用时返回 None。

    PyArrow 对 object 列内混有 int/str 等会抛错（如 Expected bytes, got int）；调用方应保证 df 列类型一致，
    此处再兜底捕获异常以便走分页表单回退。

    部分浏览器+Streamlit 版本在 data_editor 全宽布局下，WebSocket 增量更新时前端 resize 监听会报
    Cannot read properties of undefined (reading 'firstElementChild')；可设环境变量
    AICHECKWORD_DISABLE_DATA_EDITOR=1 完全禁用表格编辑器，走分页表单。
    """
    _dd = (os.environ.get("AICHECKWORD_DISABLE_DATA_EDITOR") or "").strip().lower()
    if _dd in ("1", "true", "yes", "on"):
        return None
    safe_kw = {}
    for k in ("key", "hide_index", "num_rows", "use_container_width"):
        if k in kwargs:
            safe_kw[k] = kwargs[k]
    de = getattr(st, "data_editor", None)
    if callable(de):
        kw = dict(safe_kw)
        if column_config is not None:
            kw["column_config"] = column_config
        try:
            return de(df, **kw)
        except TypeError:
            kw.pop("column_config", None)
            try:
                return de(df, **kw)
            except Exception:
                pass
        except Exception:
            pass
    ex = getattr(st, "experimental_data_editor", None)
    if callable(ex):
        try:
            return ex(df, key=kwargs.get("key"), use_container_width=kwargs.get("use_container_width", False))
        except TypeError:
            try:
                return ex(df)
            except Exception:
                pass
        except Exception:
            pass
    return None


def _hist_cell_display_str(v) -> str:
    """历史审核点表格展示：统一为 str，避免 DataFrame object 列混 int 导致 st.data_editor / PyArrow 转换失败。"""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, (dict, list)):
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return str(v)
    return str(v)


def _hist_opt_display_str(v) -> str:
    """与 _hist_cell_display_str 相同，但显式区分「缺省」与「可序列化的空/零」。"""
    if v is None:
        return ""
    return _hist_cell_display_str(v)


def _history_point_row_dict(
    p: dict,
    seq_display: int,
    *,
    related_doc_names: Optional[list] = None,
    file_name_hint: str = "",
) -> dict:
    raw_sev = p.get("severity")
    if raw_sev is None or raw_sev == "":
        sev = "info"
    else:
        sev = str(raw_sev).lower()
    if sev not in ("high", "medium", "low", "info"):
        sev = "info"
    md_list = _effective_point_modify_docs_list(p, related_doc_names, file_name_hint)
    md_s = "，".join(_hist_cell_display_str(x) for x in md_list) if md_list else ""
    if p.get("deprecated"):
        tag = "弃用"
    elif p.get("correction_kind") == "false_positive" or p.get("false_positive_reason"):
        tag = "误报"
    else:
        tag = "正常"
    act = p.get("action")
    if act is None or act == "":
        act = _get_multi_doc_default_action(sev)
    else:
        act = str(act)
    if act not in ACTION_OPTIONS:
        act = ACTION_OPTIONS[0]
    return {
        # 用 str 避免 NumberColumn + PyArrow 将第 0 列推断为 object 混 int 时报 Expected bytes, got int
        "序号": str(int(seq_display)),
        "状态": tag,
        "级别": sev,
        "类别": _hist_opt_display_str(p.get("category")),
        "位置": _hist_opt_display_str(p.get("location")),
        "问题描述": _hist_opt_display_str(p.get("description")),
        "法规依据": _hist_opt_display_str(p.get("regulation_ref")),
        "修改建议": _hist_opt_display_str(p.get("suggestion")),
        "需修改文档": md_s,
        "处理状态": act,
    }


def _history_df_sanitize_for_editor(df: pd.DataFrame) -> pd.DataFrame:
    """传给 st.data_editor 前将每列强制为纯 str，杜绝 object 列内混 int/NaN 等导致 PyArrow 转换失败。"""

    def _cell(x):
        if x is None:
            return ""
        try:
            if pd.isna(x):
                return ""
        except (TypeError, ValueError):
            pass
        return _hist_cell_display_str(x)

    if df is None or df.empty:
        return df
    out = df.copy()
    for c in out.columns:
        out[c] = out[c].map(_cell)
    return out


def _history_points_to_df(
    points: list,
    *,
    related_doc_names: Optional[list] = None,
    file_name_hint: str = "",
) -> pd.DataFrame:
    """历史报告全表：转为 DataFrame。"""
    rows = []
    for j, p in enumerate(points):
        if not isinstance(p, dict):
            p = {}
        rows.append(
            _history_point_row_dict(
                p, j + 1, related_doc_names=related_doc_names, file_name_hint=file_name_hint
            )
        )
    return _history_df_sanitize_for_editor(pd.DataFrame(rows))


def _history_points_to_df_slice(
    points: list,
    start: int,
    end: int,
    *,
    related_doc_names: Optional[list] = None,
    file_name_hint: str = "",
) -> pd.DataFrame:
    end = min(int(end), len(points))
    start = max(0, int(start))
    rows = []
    for gi in range(start, end):
        p = points[gi]
        if not isinstance(p, dict):
            p = {}
        rows.append(
            _history_point_row_dict(
                p, gi + 1, related_doc_names=related_doc_names, file_name_hint=file_name_hint
            )
        )
    return _history_df_sanitize_for_editor(pd.DataFrame(rows))


def _apply_history_df_to_points_slice(edited_df: pd.DataFrame, points: list, start: int) -> None:
    """把当前页的表格编辑结果写回 audit_points（原地修改）。"""
    if edited_df is None or edited_df.empty or not points:
        return
    sev_opts = {"high", "medium", "low", "info"}
    for pos, (_, row) in enumerate(edited_df.iterrows()):
        gi = start + pos
        if gi >= len(points):
            break
        p = points[gi]
        sev = str(row.get("级别") or "info").lower()
        if sev not in sev_opts:
            sev = "info"
        p["severity"] = sev
        # 类别、法规依据表格列为只读，写回时不从 data_editor 覆盖（保持原审核点数据）
        p["location"] = str(row.get("位置") or "")
        p["description"] = str(row.get("问题描述") or "")
        p["suggestion"] = str(row.get("修改建议") or "")
        md_s = str(row.get("需修改文档") or "")
        parts = []
        if "，" in md_s or "," in md_s:
            parts = [x.strip() for x in md_s.replace("，", ",").split(",") if x.strip()]
        elif md_s.strip():
            parts = [md_s.strip()]
        else:
            parts = []
        p["modify_docs"] = parts
        act = str(row.get("处理状态") or ACTION_OPTIONS[0])
        if act not in ACTION_OPTIONS:
            act = ACTION_OPTIONS[0]
        p["action"] = act


def _render_history_fallback_slice(
    points: list,
    start: int,
    end: int,
    pk: str,
    key_suffix: str,
    history_id: int,
    history_local_draft: bool,
    *,
    related_doc_names: Optional[list] = None,
    file_name_hint: str = "",
) -> None:
    """无 st.data_editor 时的分页表单回退。"""
    if not st.session_state.get("_hist_data_editor_fallback_info"):
        st.info(
            "当前 Streamlit 无 `st.data_editor`（版本过旧）。已使用**分页表单**；若镜像允许，可执行 `pip install -U streamlit` 尽量升到新版本。"
            " 跨网访问 FastAPI 时请设置环境变量 **AICHECKWORD_API_PUBLIC_BASE**（如 `http://服务器IP:8000`）。"
        )
        st.session_state["_hist_data_editor_fallback_info"] = True
    sev_opts = ["high", "medium", "low", "info"]
    end = min(end, len(points))
    before = copy.deepcopy([points[i] for i in range(start, end)])
    for gi in range(start, end):
        p = points[gi]
        pfx = f"hfb_{pk}_{gi}{key_suffix}"
        sv = (p.get("severity") or "info").lower()
        if sv not in sev_opts:
            sv = "info"
        if f"{pfx}_sev" not in st.session_state:
            st.session_state[f"{pfx}_sev"] = sv
        act = p.get("action") or _get_multi_doc_default_action(sv)
        if act not in ACTION_OPTIONS:
            act = ACTION_OPTIONS[0]
        if f"{pfx}_act" not in st.session_state:
            st.session_state[f"{pfx}_act"] = act
        for fld, sk, default in (
            ("description", "desc", ""),
            ("suggestion", "sug", ""),
            ("location", "loc", ""),
        ):
            k = f"{pfx}_{sk}"
            if k not in st.session_state:
                st.session_state[k] = p.get(fld, default) or ""
        md_eff = _effective_point_modify_docs_list(p, related_doc_names, file_name_hint)
        md_init = "，".join(md_eff)
        md_sig_k = f"{pfx}__md_src_sig"
        if st.session_state.get(md_sig_k) != tuple(md_eff):
            st.session_state[f"{pfx}_md"] = md_init
            st.session_state[md_sig_k] = tuple(md_eff)
        st.markdown(f"**第 {gi + 1} 条** · {(p.get('category') or '未分类')[:36]}")
        c1, c2 = st.columns(2)
        with c1:
            st.selectbox("级别", sev_opts, key=f"{pfx}_sev")
        with c2:
            st.selectbox("处理状态", ACTION_OPTIONS, key=f"{pfx}_act")
        st.text_area("问题描述", key=f"{pfx}_desc", height=70)
        st.text_area("修改建议", key=f"{pfx}_sug", height=70)
        st.caption(f"**类别（只读）** {p.get('category') or '—'}")
        st.text_input("位置", key=f"{pfx}_loc")
        st.caption(f"**法规依据（只读）** {p.get('regulation_ref') or '—'}")
        if md_eff:
            st.caption("📎 **本条已记录的需修改文档（与下方输入框同步）：** " + "、".join(md_eff))
        st.text_input("需修改文档（逗号分隔）", key=f"{pfx}_md")
    for gi in range(start, end):
        p = points[gi]
        pfx = f"hfb_{pk}_{gi}{key_suffix}"
        p["severity"] = st.session_state.get(f"{pfx}_sev", p.get("severity", "info"))
        p["action"] = st.session_state.get(f"{pfx}_act", p.get("action", ACTION_OPTIONS[0]))
        p["description"] = st.session_state.get(f"{pfx}_desc", "")
        p["suggestion"] = st.session_state.get(f"{pfx}_sug", "")
        p["location"] = st.session_state.get(f"{pfx}_loc", "")
        md_s = st.session_state.get(f"{pfx}_md", "")
        parts = [x.strip() for x in str(md_s).replace("，", ",").split(",") if x.strip()]
        p["modify_docs"] = parts
    after = [points[i] for i in range(start, end)]
    try:
        if (
            history_local_draft
            and history_id
            and json.dumps(before, ensure_ascii=False, sort_keys=True, default=str)
            != json.dumps(after, ensure_ascii=False, sort_keys=True, default=str)
        ):
            st.session_state["_hist_draft_dirty"] = True
    except Exception:
        st.session_state["_hist_draft_dirty"] = True


def _render_history_doc_points_table(
    report: dict,
    r_idx: int,
    reports: list,
    pk: str,
    key_suffix: str,
    history_id: int,
    parent_batch_report: Optional[dict],
    history_local_draft: bool,
    file_name: str,
) -> None:
    """历史报告：分页可编辑表（或旧版回退表单）+ 当前页「纠正」按钮。"""
    points = report.get("audit_points") or []
    if not points:
        st.caption("本文档无审核点。")
        return

    if not report.get("related_doc_names"):
        report["related_doc_names"] = [
            report.get("original_filename", report.get("file_name", file_name or "未知"))
        ]
    _hist_rel = report.get("related_doc_names") or []
    _hist_fn = (report.get("original_filename") or report.get("file_name") or file_name or "").strip()

    focus_k = f"_hcf_{history_id}_{pk}{key_suffix}"
    n = len(points)
    ps = HIST_EDIT_PAGE_SIZE
    n_pages = max(1, (n + ps - 1) // ps)
    pg_key = f"_hist_pg_{pk}{key_suffix}"
    page = st.selectbox(
        f"审核点分页 · {file_name[:32]}",
        options=list(range(n_pages)),
        format_func=lambda i: f"第 {i + 1}/{n_pages} 页（第 {i * ps + 1}–{min((i + 1) * ps, n)} 条，共 {n} 条）",
        key=pg_key,
    )
    try:
        page = int(page)
    except (TypeError, ValueError):
        page = 0
    page = max(0, min(page, n_pages - 1))
    start = page * ps
    end = min(start + ps, n)

    df0 = _history_points_to_df_slice(
        points, start, end, related_doc_names=_hist_rel, file_name_hint=_hist_fn
    )
    ed_key = f"hist_ed_{pk}_p{page}{key_suffix}"
    edited_df: Optional[pd.DataFrame] = None
    col_cfg = None
    try:
        from streamlit.column_config import SelectboxColumn, TextColumn

        col_cfg = {
            "序号": TextColumn("序号", width="small", disabled=True),
            "状态": TextColumn("状态", width="small", disabled=True),
            "级别": SelectboxColumn("级别", options=["high", "medium", "low", "info"], required=True),
            "类别": TextColumn("类别", width="small", disabled=True),
            "位置": TextColumn("位置", width="medium"),
            "问题描述": TextColumn("问题描述", width="large"),
            "法规依据": TextColumn("法规依据", width="medium", disabled=True),
            "修改建议": TextColumn("修改建议", width="large"),
            "需修改文档": TextColumn("需修改文档（逗号/中文逗号分隔）", width="medium"),
            "处理状态": SelectboxColumn("处理状态", options=list(ACTION_OPTIONS), required=True),
        }
    except Exception:
        col_cfg = None

    def _hist_apply_editor_or_fallback():
        nonlocal edited_df
        edited_df = _st_try_data_editor(
            df0,
            column_config=col_cfg,
            key=ed_key,
            hide_index=True,
            num_rows="fixed",
            # False：减轻前端 addResizeListener/滚动与 WebSocket 重绘时的 firstElementChild 报错（全宽表格更易触发）
            use_container_width=False,
        )
        if edited_df is not None and not isinstance(edited_df, pd.DataFrame):
            try:
                edited_df = pd.DataFrame(edited_df)
            except Exception:
                edited_df = None
        if edited_df is None:
            _render_history_fallback_slice(
                points,
                start,
                end,
                pk,
                key_suffix,
                history_id,
                history_local_draft,
                related_doc_names=_hist_rel,
                file_name_hint=_hist_fn,
            )
            _recount_severity(report)
            if parent_batch_report is not None:
                _aggregate_batch_report_totals(parent_batch_report)
        else:
            if not isinstance(edited_df, pd.DataFrame):
                edited_df = df0
            if history_local_draft and history_id:
                try:
                    if df0.shape == edited_df.shape and df0.to_json() != edited_df.to_json():
                        st.session_state["_hist_draft_dirty"] = True
                except Exception:
                    st.session_state["_hist_draft_dirty"] = True
            _apply_history_df_to_points_slice(edited_df, points, start)
            _recount_severity(report)
            if parent_batch_report is not None:
                _aggregate_batch_report_totals(parent_batch_report)

    try:
        _hist_apply_editor_or_fallback()
    except Exception:
        logging.getLogger(__name__).exception("历史审核点 data_editor 异常，已回退分页表单")
        edited_df = None
        _render_history_fallback_slice(
            points,
            start,
            end,
            pk,
            key_suffix,
            history_id,
            history_local_draft,
            related_doc_names=_hist_rel,
            file_name_hint=_hist_fn,
        )
        _recount_severity(report)
        if parent_batch_report is not None:
            _aggregate_batch_report_totals(parent_batch_report)

    st.caption("当前页内点 **✏编号** 打开纠正表单；本地批量模式下改完点顶部「一次性保存到数据库」。")
    step = 14
    for r0 in range(start, end, step):
        chunk = list(range(r0, min(r0 + step, end)))
        bcols = st.columns(len(chunk))
        for bi, j in enumerate(chunk):
            p = points[j]
            if not isinstance(p, dict):
                p = {}
            tag = ""
            if p.get("deprecated"):
                tag = "·弃"
            elif p.get("correction_kind") == "false_positive" or p.get("false_positive_reason"):
                tag = "·误"
            help_txt = (
                _hist_opt_display_str(p.get("category"))
                + " | "
                + _hist_opt_display_str(p.get("description")).replace("\n", " ")
            )[:200]
            if bcols[bi].button(f"✏{j + 1}{tag}", key=f"hbtn_corr_{pk}_{j}{key_suffix}", help=help_txt or None):
                st.session_state[focus_k] = j

    if history_id and not history_local_draft:
        if st.button("💾 保存本表到数据库", key=f"hist_tbl_save_{pk}{key_suffix}"):
            try:
                root = parent_batch_report if parent_batch_report is not None else report
                _timed_update_audit_report(history_id, root)
                _invalidate_audit_history_cache()
                st.success("已保存。")
                _streamlit_rerun()
            except Exception as ex:
                st.error(f"保存失败：{ex}")

    fj = st.session_state.get(focus_k)
    if fj is not None and isinstance(fj, int) and 0 <= fj < len(points):
        streamlit_divider()
        h1, h2 = st.columns([4, 1])
        with h1:
            st.markdown(f"**纠正 · 第 {fj + 1} 条**（{file_name[:60]}）")
        with h2:
            if st.button("关闭纠正面板", key=f"hbtn_corr_close_{pk}{key_suffix}"):
                st.session_state.pop(focus_k, None)
                _streamlit_rerun()
        _md_hist = _audit_point_modify_docs_key(int(history_id), int(fj), str(pk), str(key_suffix or ""))
        _render_correction_form(
            report,
            history_id,
            fj,
            points[fj],
            parent_batch_report=parent_batch_report,
            sub_report_index=r_idx if parent_batch_report is not None else None,
            close_state_keys=[focus_k],
            local_draft=bool(history_id and history_local_draft),
            modify_docs_widget_key=_md_hist,
        )


def _recount_severity(report: dict):
    """根据审核点重新统计各严重程度计数（弃用 deprecated 的点不计入）"""
    from src.core.audit_report_utils import recount_severity

    recount_severity(report)


def _feed_correction_to_kb_impl(collection: str, file_name: str, corrected_point: dict) -> None:
    """同步写入反馈向量库（在后台线程中调用，勿使用 Streamlit session_state）。"""
    from src.core.langchain_compat import Document
    from src.core.agent import ReviewAgent

    is_fp = corrected_point.get("correction_kind") == "false_positive" or bool(
        (corrected_point.get("false_positive_reason") or "").strip()
    )
    is_dep = bool(corrected_point.get("deprecated"))
    repl = corrected_point.get("replacement_point_added")

    if is_fp:
        feedback_kind = "false_positive"
        content = (
            f"[用户反馈·误报] 入库分类：{feedback_kind}（非审核点清单原文）\n"
            f"关联文件：{file_name}\n"
            f"人工标记为误报。原因：{corrected_point.get('false_positive_reason', '')}\n"
            f"原审核类别：{corrected_point.get('category', '')}\n"
            f"原问题描述：{corrected_point.get('description', '')}\n"
            f"原位置：{corrected_point.get('location', '')}\n"
            f"原法规依据：{corrected_point.get('regulation_ref', '')}\n"
            f"说明：若待审文档与上述原审核点在语义上等价，**不得**再输出该审核点。"
        )
    elif is_dep:
        feedback_kind = "deprecated_with_replacement" if repl else "deprecated"
        content = (
            f"[用户反馈·弃用审核点] 入库分类：{feedback_kind}（非审核点清单原文）\n"
            f"关联文件：{file_name}\n"
            f"弃用说明：{corrected_point.get('deprecation_note', '')}\n"
            f"原类别/描述摘要：{corrected_point.get('category', '')} / {corrected_point.get('description', '')}\n"
        )
        if isinstance(repl, dict):
            content += (
                f"\n同时新增替代审核点：类别={repl.get('category', '')}；描述={repl.get('description', '')}；"
                f"建议={repl.get('suggestion', '')}"
            )
    else:
        feedback_kind = "revision"
        content = (
            f"[用户反馈·修订后结论] 入库分类：{feedback_kind}（非审核点清单原文）\n"
            f"关联文件：{file_name}\n"
            f"类别：{corrected_point.get('category', '')}\n"
            f"严重程度：{corrected_point.get('severity', '')}\n"
            f"问题描述（修订后）：{corrected_point.get('description', '')}\n"
            f"法规依据：{corrected_point.get('regulation_ref', '')}\n"
            f"修改建议：{corrected_point.get('suggestion', '')}"
        )
    loc = corrected_point.get("location") or ""
    if loc:
        content += f"\n位置/涉及文档：{loc}"
    modify_docs = corrected_point.get("modify_docs")
    if isinstance(modify_docs, list) and modify_docs:
        content += f"\n需修改的文档：{', '.join(modify_docs)}"

    safe_fn = (file_name or "doc").replace("\\", "/").split("/")[-1][:120]
    logical_name = f"user_fb_{feedback_kind}__{safe_fn}"

    doc = Document(
        page_content=content,
        metadata={
            "kb_entry_class": "user_audit_feedback",
            "feedback_kind": feedback_kind,
            "type": "audit_user_feedback",
            "origin_file_name": file_name or "",
            "collection_tag": collection or "",
        },
    )
    agent = ReviewAgent(collection or "regulations")
    agent.checkpoint_feedback_kb.add_documents(
        [doc],
        file_name=logical_name,
        category="audit_user_feedback",
    )


def _feed_correction_to_kb(collection: str, file_name: str, corrected_point: dict):
    """将误报/弃用/修订写入独立「用户审核反馈」向量库。默认后台线程执行，避免阻塞 Streamlit UI。"""
    import threading

    def _run():
        t0 = time.perf_counter()
        try:
            _feed_correction_to_kb_impl(collection, file_name, corrected_point)
            if audit_perf_enabled():
                audit_perf_log("feed_correction_kb_thread", (time.perf_counter() - t0) * 1000.0, "")
        except Exception:
            logging.getLogger(__name__).exception("后台写入审核反馈向量库失败")

    if getattr(settings, "async_correction_kb_feed", True):
        threading.Thread(target=_run, daemon=True).start()
        return
    _run()


def _render_reports_table_layout(
    reports: list,
    base_key_prefix: str,
    history_id: int = 0,
    parent_batch_report: dict = None,
    key_suffix: str = "",
    allow_nested_expander: bool = True,
    history_local_draft: bool = False,
):
    """按批次→文档→问题层级渲染报告内容（表格+待办），供当前会话与历史报告共用。
    allow_nested_expander: False 时不在文档块使用 expander（用于已在 expander 内的历史报告，避免 Streamlit 禁止嵌套 expander）。
    history_local_draft: 历史报告且开启本地批量模式时，单条保存不写库，改由顶部「一次性保存」。
    """
    if not reports:
        return
    for r in reports:
        try:
            sanitize_audit_report_dict(r, db_file_name=str((r.get("file_name") or "")).strip())
        except Exception:
            pass
    meta = {}
    for r in reports:
        m = r.get("_review_meta") or {}
        if m:
            meta = m
            break
    audit_type = meta.get("audit_type", "通用审核")

    overview_cols = st.columns([1, 1, 1, 1])
    with overview_cols[0]:
        st.metric("审核类型", audit_type)
    with overview_cols[1]:
        st.metric("文档数", len(reports))
    with overview_cols[2]:
        total_pts = sum(r.get("total_points", 0) for r in reports)
        st.metric("总问题点", total_pts)
    with overview_cols[3]:
        total_high = sum(r.get("high_count", 0) for r in reports)
        st.metric("高风险", total_high)

    if audit_type == "项目审核":
        info_parts = []
        if meta.get("project_name"):
            info_parts.append(f"**项目名称：** {meta['project_name']}")
        if meta.get("product_name"):
            info_parts.append(f"**产品名称：** {meta['product_name']}")
        if meta.get("model") or meta.get("model_en"):
            info_parts.append(f"**型号（Model）：** {(meta.get('model') or '') or '—'} / {(meta.get('model_en') or '') or '—'}")
        if meta.get("registration_country"):
            info_parts.append(f"**注册国家：** {meta['registration_country']}")
        if info_parts:
            st.info("　|　".join(info_parts))

    st.markdown("### 文档审核汇总")
    summary_rows = []
    for r_idx, r in enumerate(reports):
        fn = r.get("original_filename", r.get("file_name", "未知"))
        summary_rows.append({
            # 全用 str/int 分列一致：勿把「表头行+数据行」塞进 st.table(list)，否则第 0 列会 str/int 混用触发 PyArrow
            # Expected bytes, got int / Conversion failed for column 0 with type object
            "序号": str(r_idx + 1),
            "文件名": _hist_cell_display_str(fn),
            "🔴高": int(r.get("high_count", 0) or 0),
            "🟡中": int(r.get("medium_count", 0) or 0),
            "🔵低": int(r.get("low_count", 0) or 0),
            "ℹ️提示": int(r.get("info_count", 0) or 0),
            "总计": int(r.get("total_points", 0) or 0),
        })
    if summary_rows:
        _sum_headers = ["序号", "文件名", "🔴高", "🟡中", "🔵低", "ℹ️提示", "总计"]

        def _sum_cell(v) -> str:
            s = str(v).replace("\n", " ").replace("\r", " ")
            return s.replace("|", "｜")

        _sep = "| " + " | ".join(["---"] * len(_sum_headers)) + " |"
        _hdr = "| " + " | ".join(_sum_cell(h) for h in _sum_headers) + " |"
        _lines = [_hdr, _sep]
        for _sr in summary_rows:
            _lines.append("| " + " | ".join(_sum_cell(_sr.get(h, "")) for h in _sum_headers) + " |")
        st.markdown("\n".join(_lines))

    all_immediate = []
    for r_idx, r in enumerate(reports):
        fn = r.get("original_filename", r.get("file_name", "未知"))
        for p in r.get("audit_points") or []:
            if p.get("deprecated"):
                continue
            action = p.get("action") or _get_multi_doc_default_action(p.get("severity", "info"))
            if action == "立即修改":
                all_immediate.append((fn, p, r))

    if all_immediate:
        st.markdown(f"### 待办任务（共 {len(all_immediate)} 项「立即修改」）")
        _da = st.session_state.get("multi_doc_default_action") or {
            "high": "立即修改", "medium": "立即修改", "low": "延期修改", "info": "无需修改",
        }
        _get_action = lambda s: _da.get((s or "info").lower(), "无需修改")
        _batch_ready = f"_batch_todo_heavy_ready{key_suffix}"
        batch_todo_cols = st.columns(5)
        with batch_todo_cols[0]:
            csv_bytes = report_todo_to_csv(
                reports,
                only_immediate=True,
                get_default_action=_get_action,
                project_name=meta.get("project_name", ""),
                product=meta.get("product_name", ""),
                country=meta.get("registration_country", ""),
            )
            st.download_button(
                f"📥 全部待办 CSV（{len(all_immediate)} 项）",
                data=csv_bytes,
                file_name="待办导入_审核待办.csv",
                mime="text/csv; charset=utf-8",
                key=f"batch_todo_csv_dl{key_suffix}",
                help="导出本批次全部「立即修改」审核点为待办 CSV",
            )
        if st.session_state.get(_batch_ready):
            with batch_todo_cols[1]:
                try:
                    pdf_bytes = report_todo_to_pdf(
                        reports,
                        only_immediate=True,
                        get_default_action=_get_action,
                        project_name=meta.get("project_name", ""),
                        product=meta.get("product_name", ""),
                        country=meta.get("registration_country", ""),
                    )
                    st.download_button(f"📥 全部待办 PDF", data=pdf_bytes, file_name="audit_todo_batch.pdf", mime="application/pdf", key=f"batch_todo_pdf_dl{key_suffix}")
                except Exception as e:
                    st.caption(f"PDF 失败: {e}")
            with batch_todo_cols[2]:
                try:
                    docx_bytes = report_todo_to_docx(
                        reports,
                        only_immediate=True,
                        get_default_action=_get_action,
                        project_name=meta.get("project_name", ""),
                        product=meta.get("product_name", ""),
                        country=meta.get("registration_country", ""),
                    )
                    st.download_button("📥 全部待办 Word", data=docx_bytes, file_name="audit_todo_batch.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", key=f"batch_todo_docx_dl{key_suffix}")
                except Exception as e:
                    st.caption(f"Word 失败: {e}")
            with batch_todo_cols[3]:
                try:
                    xlsx_bytes = report_todo_to_excel(
                        reports,
                        only_immediate=True,
                        get_default_action=_get_action,
                        project_name=meta.get("project_name", ""),
                        product=meta.get("product_name", ""),
                        country=meta.get("registration_country", ""),
                    )
                    st.download_button("📥 全部待办 Excel", data=xlsx_bytes, file_name="audit_todo_batch.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"batch_todo_xlsx_dl{key_suffix}")
                except Exception as e:
                    st.caption(f"Excel 失败: {e}")
            with batch_todo_cols[4]:
                todo_lines = []
                for idx, (fn, p, _r) in enumerate(all_immediate, 1):
                    todo_lines.append(f"{idx}. [{fn}] {p.get('category', '')}：{p.get('description', '')[:100]}")
                    todo_lines.append(f"   修改建议：{p.get('suggestion', '')[:120]}")
                    todo_lines.append("")
                st.download_button(
                    "📥 全部待办（文本）",
                    data="\n".join(todo_lines),
                    file_name="audit_todo_batch.txt",
                    mime="text/plain",
                    key=f"batch_todo_txt_dl{key_suffix}",
                )
            if st.button("收起 PDF/Word/Excel 导出", key=f"_batch_todo_hide{key_suffix}"):
                st.session_state.pop(_batch_ready, None)
                _streamlit_rerun()
        else:
            with batch_todo_cols[1]:
                st.caption("PDF/Word/Excel 生成较慢")
            if st.button("📥 加载本批 PDF/Word/Excel/文本导出", key=f"_batch_todo_show{key_suffix}", help="点击后再生成，避免每次打开报告都卡顿"):
                st.session_state[_batch_ready] = True
                _streamlit_rerun()

    st.markdown("### 审核点详情")
    for r_idx, report in enumerate(reports):
        if not report.get("related_doc_names"):
            report["related_doc_names"] = [report.get("original_filename", report.get("file_name", "未知"))]
        file_name = report.get("original_filename", report.get("file_name", "未知"))
        pk = f"{base_key_prefix}{r_idx}"
        points = report.get("audit_points") or []

        # 标题勿含「问题点数」等会随保存而变的文案，否则 Streamlit 会把 expander 视为新组件、展开状态丢失（像自动收起）
        _exp_label = f"📄 文档 {r_idx + 1}：{file_name}"
        if allow_nested_expander:
            doc_container = st.expander(
                _exp_label,
                expanded=(len(reports) == 1),
            )
        else:
            st.markdown("---")
            st.markdown(
                f"#### 📄 {file_name}（{report.get('total_points', 0)} 个问题点："
                f"🔴{report.get('high_count',0)} 🟡{report.get('medium_count',0)} 🔵{report.get('low_count',0)}）"
            )
            doc_container = st.container()

        with doc_container:
            if allow_nested_expander:
                st.caption(
                    f"共 {report.get('total_points', 0)} 个问题点 · 🔴{report.get('high_count', 0)} "
                    f"🟡{report.get('medium_count', 0)} 🔵{report.get('low_count', 0)} "
                    f"ℹ️{report.get('info_count', 0)}"
                )
            if points:
                if history_id:
                    _render_history_doc_points_table(
                        report,
                        r_idx,
                        reports,
                        pk,
                        key_suffix,
                        history_id,
                        parent_batch_report,
                        history_local_draft,
                        file_name,
                    )
                else:
                    rows_h = [
                        "<table style='width:100%;border-collapse:collapse;font-size:0.92rem'>",
                        "<thead><tr><th>#</th><th>级别</th><th>类别</th><th>摘要（悬停看全文）</th></tr></thead><tbody>",
                    ]
                    for j, p in enumerate(points):
                        sev = p.get("severity", "info")
                        icon = _SEVERITY_ICONS.get(sev, "ℹ️")
                        lab = _SEVERITY_LABELS.get(sev, sev)
                        full = (p.get("description") or "").replace("\n", " ").replace("\r", " ").strip()
                        if p.get("deprecated"):
                            full = "〔弃用〕" + full
                        elif p.get("correction_kind") == "false_positive" or p.get("false_positive_reason"):
                            full = "〔误报〕" + full
                        short_txt = full[:52] + ("…" if len(full) > 52 else "")
                        short_esc = html.escape(short_txt, quote=False)
                        title_attr = html.escape(full[:12000], quote=True)
                        sev_cell = html.escape(f"{icon}{lab}", quote=False)
                        cat_cell = html.escape((p.get("category") or "")[:26], quote=False)
                        rows_h.append(
                            f"<tr><td>{j + 1}</td><td>{sev_cell}</td><td>{cat_cell}</td>"
                            f'<td><span style="border-bottom:1px dotted #888;cursor:help" title="{title_attr}">{short_esc}</span></td></tr>'
                        )
                    rows_h.append("</tbody></table>")
                    st.markdown("\n".join(rows_h), unsafe_allow_html=True)
                    sel_pt = st.selectbox(
                        f"选择审核点（仅展开这一条）· {file_name[:40]}",
                        options=list(range(len(points))),
                        format_func=lambda j: _point_row_label(points[j], j),
                        key=f"apt_sel_{pk}{key_suffix}",
                    )
                    _render_point_detail_inline(
                        report,
                        r_idx,
                        reports,
                        sel_pt,
                        points[sel_pt],
                        pk=pk,
                        key_suffix=key_suffix,
                        history_id=history_id,
                        parent_batch_report=parent_batch_report,
                        detail_scope_suffix=key_suffix,
                        history_local_draft=history_local_draft,
                    )

            # 本文档待办导出（默认仅 CSV；PDF/Word/Excel 按需加载，避免每行详情都触发整页重算卡顿）
            doc_immediate = [
                p for p in points
                if not p.get("deprecated")
                and (p.get("action") or _get_multi_doc_default_action(p.get("severity", "info"))) == "立即修改"
            ]
            if doc_immediate:
                st.markdown(f"**📋 本文档待办（{len(doc_immediate)} 项）**")
                _da2 = st.session_state.get("multi_doc_default_action") or {
                    "high": "立即修改", "medium": "立即修改", "low": "延期修改", "info": "无需修改",
                }
                _get_action_doc = lambda s: _da2.get((s or "info").lower(), "无需修改")
                _doc_heavy = f"_doc_todo_heavy_{pk}{key_suffix}"
                c_doc1, c_doc2 = st.columns(2)
                with c_doc1:
                    single_csv = report_todo_to_csv(
                        [report],
                        only_immediate=True,
                        get_default_action=_get_action_doc,
                        project_name=meta.get("project_name", ""),
                        product=meta.get("product_name", ""),
                        country=meta.get("registration_country", ""),
                    )
                    st.download_button(
                        "📥 本文档待办 CSV",
                        data=single_csv,
                        file_name=f"待办_{file_name}.csv",
                        mime="text/csv; charset=utf-8",
                        key=f"doc_todo_csv_{pk}{key_suffix}",
                    )
                with c_doc2:
                    if st.button(
                        "➡️ 传到文档生成修改（审核后修改）",
                        key=f"handoff_post_audit_{pk}{key_suffix}",
                    ):
                        handoff = _build_post_audit_handoff(
                            report,
                            source_report_id=int(history_id or 0),
                        )
                        st.session_state["_post_audit_handoff"] = handoff
                        # 不可在顶部 radio 已绑定后再写 main_function_nav_page，改用下一轮脚本开头消费
                        st.session_state["_pending_main_function_nav_page"] = "✍️ 文档初稿生成"
                        st.session_state["_draft_page_mode_force"] = "审核后修改"
                        _streamlit_rerun()
                if st.session_state.get(_doc_heavy):
                    c2, c3, c4, c5 = st.columns(4)
                    with c2:
                        try:
                            single_pdf = report_todo_to_pdf(
                                [report],
                                only_immediate=True,
                                get_default_action=_get_action_doc,
                                project_name=meta.get("project_name", ""),
                                product=meta.get("product_name", ""),
                                country=meta.get("registration_country", ""),
                            )
                            st.download_button("📥 PDF", data=single_pdf, file_name=f"待办_{file_name}.pdf", mime="application/pdf", key=f"doc_todo_pdf_{pk}{key_suffix}")
                        except Exception:
                            st.caption("PDF 失败")
                    with c3:
                        try:
                            single_docx = report_todo_to_docx(
                                [report],
                                only_immediate=True,
                                get_default_action=_get_action_doc,
                                project_name=meta.get("project_name", ""),
                                product=meta.get("product_name", ""),
                                country=meta.get("registration_country", ""),
                            )
                            st.download_button("📥 Word", data=single_docx, file_name=f"待办_{file_name}.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", key=f"doc_todo_docx_{pk}{key_suffix}")
                        except Exception:
                            st.caption("Word 失败")
                    with c4:
                        try:
                            single_xlsx = report_todo_to_excel(
                                [report],
                                only_immediate=True,
                                get_default_action=_get_action_doc,
                                project_name=meta.get("project_name", ""),
                                product=meta.get("product_name", ""),
                                country=meta.get("registration_country", ""),
                            )
                            st.download_button("📥 Excel", data=single_xlsx, file_name=f"待办_{file_name}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"doc_todo_xlsx_{pk}{key_suffix}")
                        except Exception:
                            st.caption("Excel 失败")
                    with c5:
                        single_lines = []
                        for idx, p in enumerate(doc_immediate, 1):
                            single_lines.append(f"{idx}. {p.get('category', '')}：{p.get('description', '')[:100]}")
                            single_lines.append(f"   修改建议：{p.get('suggestion', '')[:120]}")
                            single_lines.append("")
                        st.download_button(
                            "📥 文本",
                            data="\n".join(single_lines),
                            file_name=f"待办_{file_name}.txt",
                            mime="text/plain",
                            key=f"doc_todo_txt_{pk}{key_suffix}",
                        )
                    if st.button("收起慢速导出", key=f"_doc_todo_hide_{pk}{key_suffix}"):
                        st.session_state.pop(_doc_heavy, None)
                        _streamlit_rerun()
                else:
                    if st.button("加载本文档 PDF/Word/Excel（较慢）", key=f"_doc_todo_show_{pk}{key_suffix}"):
                        st.session_state[_doc_heavy] = True
                        _streamlit_rerun()


def _render_point_detail_inline(
    report: dict, r_idx: int, reports: list,
    point_idx: int, point: dict,
    pk: str, key_suffix: str,
    history_id: int = 0,
    parent_batch_report: dict = None,
    detail_scope_suffix: str = "",
    history_local_draft: bool = False,
):
    """在审核点行下方展开的详情/编辑面板。同一时间仅展开一条详情（按 detail_scope_suffix 区分），关闭用 _streamlit_rerun。"""
    sev = point.get("severity", "info")
    icon = _SEVERITY_ICONS.get(sev, "ℹ️")
    _detail_scope = detail_scope_suffix if detail_scope_suffix is not None else ""
    detail_key = f"_show_detail_{pk}_{point_idx}"
    file_name = report.get("original_filename", report.get("file_name", "未知"))
    doc_names = _report_related_doc_names(report, reports=reports, parent_batch_report=parent_batch_report)
    opt_modify_docs = _point_modify_docs_dropdown_options(point, doc_names)
    hist_corr_key = f"_hist_corr_{history_id}_{pk}_{point_idx}{key_suffix}" if history_id else ""

    with st.container():
        _md_unify_detail = _audit_point_modify_docs_key(int(history_id or 0), int(point_idx), pk, key_suffix or "")
        st.markdown(f"---\n**{icon} 审核点 {point_idx + 1} 详情 — {point.get('category', '未分类')}**")

        if history_id and hist_corr_key and st.session_state.get(hist_corr_key):
            st.info("**纠正模式**：修改后点「保存纠正」将更新本条历史报告，并可勾选写入知识库供后续审核参考。")
            _render_correction_form(
                report,
                history_id,
                point_idx,
                point,
                parent_batch_report=parent_batch_report,
                sub_report_index=r_idx if parent_batch_report is not None else None,
                close_state_keys=[hist_corr_key],
                local_draft=bool(history_id and history_local_draft),
                modify_docs_widget_key=_md_unify_detail,
            )
            if st.button("↩ 返回普通编辑", key=f"{pk}_detail_{point_idx}{key_suffix}_back_corr"):
                st.session_state.pop(hist_corr_key, None)
                _streamlit_rerun()
            return

        pfx = f"{pk}_detail_{point_idx}{key_suffix}"
        _sev_list = ["high", "medium", "low", "info"]
        _sv0 = (point.get("severity") or "info").lower()
        _sev_i0 = _sev_list.index(_sv0) if _sv0 in _sev_list else 3
        c_sev, c_loc = st.columns(2)
        with c_sev:
            try:
                new_sev = st.selectbox(
                    "严重程度",
                    _sev_list,
                    index=_sev_i0,
                    format_func=lambda x: _SEVERITY_LABELS.get(x, x),
                    key=f"{pfx}_sev",
                )
            except TypeError:
                new_sev = st.selectbox("严重程度", _sev_list, index=_sev_i0, key=f"{pfx}_sev")
        with c_loc:
            new_loc = st.text_input("位置", value=point.get("location", "") or "", key=f"{pfx}_loc")
        st.caption(f"**类别（只读）** {point.get('category') or '—'}")
        st.caption(f"**法规依据（只读）** {point.get('regulation_ref') or '—'}")
        new_desc = st.text_area("问题描述", value=point.get("description", ""), key=f"{pfx}_desc", height=100)
        new_sug = st.text_area("修改建议", value=point.get("suggestion", ""), key=f"{pfx}_sug", height=100)

        _caption_stored_modify_docs(point)
        new_modify_docs = _multiselect_modify_docs(
            "需修改的文档", opt_modify_docs, _point_modify_docs_merged_list(point), key=_md_unify_detail
        )

        current_action = point.get("action") or _get_multi_doc_default_action(new_sev)
        if current_action not in ACTION_OPTIONS:
            current_action = ACTION_OPTIONS[0]
        new_action = st.selectbox("处理状态", ACTION_OPTIONS, index=ACTION_OPTIONS.index(current_action), key=f"{pfx}_action")

        c_save, c_close = st.columns(2)
        with c_save:
            _save_label = "✅ 应用本条到草稿" if (history_id and history_local_draft) else "💾 保存修改"
            if st.button(_save_label, key=f"{pfx}_save"):
                if r_idx < len(reports) and point_idx < len(reports[r_idx].get("audit_points", [])):
                    _md_save = new_modify_docs
                    if not _md_save:
                        _fb = _point_modify_docs_merged_list(point)
                        if _fb:
                            _md_save = _fb
                    reports[r_idx]["audit_points"][point_idx]["severity"] = new_sev
                    reports[r_idx]["audit_points"][point_idx]["location"] = new_loc
                    reports[r_idx]["audit_points"][point_idx]["description"] = new_desc
                    reports[r_idx]["audit_points"][point_idx]["suggestion"] = new_sug
                    reports[r_idx]["audit_points"][point_idx]["modify_docs"] = _md_save
                    reports[r_idx]["audit_points"][point_idx]["action"] = new_action
                if history_id:
                    if history_local_draft:
                        tgt = reports[r_idx] if r_idx < len(reports) else report
                        _recount_severity(tgt)
                        if parent_batch_report is not None:
                            _aggregate_batch_report_totals(parent_batch_report)
                        st.session_state["_hist_draft_dirty"] = True
                        st.success("已写入本地草稿（尚未保存到数据库）；改完后请点上方「一次性保存到数据库」。")
                    else:
                        try:
                            updated_report = parent_batch_report if parent_batch_report else report
                            _timed_update_audit_report(history_id, updated_report)
                            _invalidate_audit_history_cache()
                            st.success("已保存到历史报告")
                            _streamlit_rerun()
                        except Exception as ex:
                            st.error(f"写入历史失败：{ex}")
                else:
                    st.success("已保存（当前会话，刷新页面会丢失未入库的修改）")
        with c_close:
            if st.button("✖ 关闭", key=f"{pfx}_close"):
                st.session_state.pop(detail_key, None)
                if hist_corr_key:
                    st.session_state.pop(hist_corr_key, None)
                st.session_state.pop(f"_rpt_detail_active{_detail_scope}", None)
                _streamlit_rerun()

        if history_id:
            st.markdown("---")
            if st.button("✏️ 纠正此审核点（写入历史 / 可选入知识库）", key=f"{pfx}_open_corr", help="与上方「保存修改」不同：会记录纠正日志，并可把经验写入审核点知识库"):
                st.session_state[hist_corr_key] = True
                _streamlit_rerun()

        st.markdown("---")


def render_reports(reports: list):
    """渲染审核报告（当前会话）：按批次→文档→问题层级表格展示，支持单条/批次待办。"""
    st.subheader("📋 审核报告")
    st.caption(
        "按 **批次 → 文档 → 问题** 层级展示；用下拉 **仅展开一条** 审核点编辑，减轻页面控件数量。"
        " 导出 **Excel/PDF/Word/HTML** 默认不自动生成（避免每次重跑页面都卡顿），需要时在下载区点击加载。"
    )

    _render_reports_table_layout(reports, "r", history_id=0, parent_batch_report=None, key_suffix="")
    _render_download_buttons(reports)


def render_draft_page():
    """文档初稿生成：复用项目案例模板，按输入文档补齐系统功能差异，并写入 projects/project_cases 与 knowledge base。"""
    st.header("✍️ 文档初稿生成")

    if not _require_provider():
        return

    import shutil
    import tempfile
    import re

    collection = st.session_state.get("collection_name", "regulations")
    projects = _cached_list_projects(collection)

    def _safe_out_path(*, base_path: str, out_path: Path) -> Path:
        """
        避免导出路径与 base 文件路径相同导致 SameFileError。
        若冲突则在文件名中追加时间戳+短随机串。
        """
        try:
            bp = Path(base_path).resolve()
            op = out_path.resolve()
            if str(bp).lower() == str(op).lower():
                stem = out_path.stem
                suf = out_path.suffix
                tag = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
                return out_path.with_name(f"{stem}_{tag}{suf}")
        except Exception:
            pass
        return out_path

    def _draft_show_patch_skipped_errors(patch_report, *, key_prefix: str = "") -> None:
        """
        就地 patch 导出后展示 skipped / errors。
        注意：Streamlit 不允许 expander 嵌套，因此这里用 checkbox 做“默认收起、点击展开”。
        """
        if not isinstance(patch_report, dict):
            return
        applied = patch_report.get("applied") or []
        changes = patch_report.get("changes") or []
        skipped = patch_report.get("skipped") or []
        errors = patch_report.get("errors") or []

        # 基础文件未发生任何实际修改：给出明确提示
        try:
            _no_change = (len(applied) == 0) and (len(changes) == 0)
        except Exception:
            _no_change = False
        if _no_change and (isinstance(skipped, list) and skipped) and not (isinstance(errors, list) and errors):
            st.info("基础文件未做任何修改：本次 patch 的操作均未命中/被跳过（可展开 skipped 查看原因）。")
        elif _no_change and not (isinstance(errors, list) and errors):
            st.info("基础文件未做任何修改：本次未产生可执行的修改操作。")

        # 概览行（高信噪）
        if any([applied, changes, skipped, errors]):
            try:
                st.caption(
                    f"patch 概览：applied={len(applied)} | changes={len(changes)} | skipped={len(skipped)} | errors={len(errors)}"
                )
            except Exception:
                pass

        if isinstance(skipped, list) and skipped:
            if st.checkbox(
                f"展开 skipped（未命中/跳过，共 {len(skipped)} 条）",
                value=False,
                key=f"{key_prefix}show_skipped",
            ):
                try:
                    st.code(json.dumps(skipped, ensure_ascii=False, indent=2), language="json")
                except Exception:
                    st.json(skipped)

        if isinstance(errors, list) and errors:
            if st.checkbox(
                f"展开 errors（执行错误，共 {len(errors)} 条）",
                value=False,
                key=f"{key_prefix}show_errors",
            ):
                try:
                    st.code(json.dumps(errors, ensure_ascii=False, indent=2), language="json")
                except Exception:
                    st.json(errors)

    def _render_draft_history(*, post_audit_only: bool = False) -> None:
        st.markdown("---")
        _hk = "postaudit_hist" if post_audit_only else "draft_hist"
        if post_audit_only:
            st.subheader("📜 审核后修改历史记录（可下载）")
            st.caption(
                "仅展示本流程写入库的记录（`source=post_audit` 或 `extra.post_audit`），与初稿生成历史分开展示。"
            )
        else:
            st.subheader("📜 初稿生成历史记录（可下载 / 可重新生成）")
            st.caption("不含审核后修改批次；审核后修改请在「审核后修改」模式查看专属历史。")
        _notice = st.session_state.get("_draft_regen_notice")
        if _notice and (not post_audit_only):
            st.info(_notice)

        _fetch_cap = 500

        def _rec_is_post_audit(r: dict) -> bool:
            ex = r.get("extra") if isinstance(r.get("extra"), dict) else {}
            return bool(ex.get("post_audit")) or str(r.get("source") or "").strip() == "post_audit"

        try:
            _raw_logs = get_operation_logs(
                op_type="draft_generate", collection=collection, limit=_fetch_cap
            )
        except Exception:
            _raw_logs = []
        if post_audit_only:
            draft_logs = [r for r in (_raw_logs or []) if _rec_is_post_audit(r)][:100]
        else:
            draft_logs = [r for r in (_raw_logs or []) if not _rec_is_post_audit(r)][:100]
        if not draft_logs:
            st.caption("暂无审核后修改记录。" if post_audit_only else "暂无初稿生成记录。")
            return

        def _draft_hist_file_merge_key(nm: str) -> str:
            """同一目标文件：out_map 用展示名，磁盘名常为 projectId_展示名，用于合并去重。"""
            s = (nm or "").strip()
            base = Path(s).name if s else ""
            m = re.match(r"^(\d+)_(.+)$", base)
            return m.group(2) if m else base

        def _merge_draft_batch_extra(recs: list) -> dict:
            """同一 batch_id 的多条操作日志合并为一条展示用的 extra（路径以后写为准）。"""
            recs_sorted = sorted(recs, key=lambda r: str(r.get("created_at") or ""))
            last_ex = dict(recs_sorted[-1].get("extra") or {})
            out_files: list = []
            seen_of = set()
            out_map: dict = {}
            summ_by_key: dict = {}
            pv_merge: dict = {}
            for r in recs_sorted:
                ex = r.get("extra") or {}
                pv = ex.get("postaudit_preview_text_by_target") or {}
                if isinstance(pv, dict):
                    for k, v in pv.items():
                        kk = str(k or "").strip()
                        if not kk or not isinstance(v, str):
                            continue
                        vv = v.strip()
                        if not vv:
                            continue
                        old = pv_merge.get(kk) or ""
                        if len(vv) > len(old):
                            pv_merge[kk] = vv
                for of in ex.get("out_files") or []:
                    if not isinstance(of, str):
                        continue
                    s = of.strip()
                    if not s or s.lower() in seen_of:
                        continue
                    seen_of.add(s.lower())
                    out_files.append(s)
                otm = ex.get("out_files_by_target") or {}
                if isinstance(otm, dict):
                    for k, v in otm.items():
                        kk = str(k or "").strip()
                        if not kk:
                            continue
                        if isinstance(v, str) and v.strip():
                            out_map[kk] = v.strip()
                for ent in ex.get("per_file_patch_summaries") or []:
                    if not isinstance(ent, dict):
                        continue
                    fk = _draft_hist_file_merge_key(str(ent.get("file_name") or ""))
                    if not fk:
                        continue
                    if fk not in summ_by_key:
                        summ_by_key[fk] = dict(ent)
                    else:
                        summ_by_key[fk].update({k: v for k, v in ent.items() if v not in (None, "", [])})
            last_ex["out_files"] = out_files
            last_ex["out_files_by_target"] = out_map
            last_ex["per_file_patch_summaries"] = list(summ_by_key.values())
            last_ex["postaudit_preview_text_by_target"] = pv_merge
            last_ex["generated_file_names"] = list(out_map.keys()) if out_map else (last_ex.get("generated_file_names") or [])
            if len(recs_sorted) > 1:
                tail = f"（已合并同批次 {len(recs_sorted)} 条进度记录）"
                last_ex["summary"] = str(last_ex.get("summary") or "").strip() + tail
            return last_ex

        def _draft_history_display_groups(logs: list) -> list:
            """按 batch_id 分组；无 batch_id 的每条单独成组。返回 [{display_rec, batch_id, merged_n}]。"""
            by_bid: dict = {}
            singles: list = []
            for rec in logs or []:
                if not isinstance(rec, dict):
                    continue
                bid = str((rec.get("extra") or {}).get("batch_id") or "").strip()
                if bid:
                    by_bid.setdefault(bid, []).append(rec)
                else:
                    singles.append(rec)
            groups = []
            for bid, recs in by_bid.items():
                if not recs:
                    continue
                recs_sorted = sorted(recs, key=lambda r: str(r.get("created_at") or ""))
                newest = dict(recs_sorted[-1])
                newest["extra"] = _merge_draft_batch_extra(recs_sorted)
                groups.append({"display_rec": newest, "batch_id": bid, "merged_n": len(recs)})
            for rec in singles:
                groups.append({"display_rec": dict(rec), "batch_id": "", "merged_n": 1})
            groups.sort(key=lambda g: str(g["display_rec"].get("created_at") or ""), reverse=True)
            return groups

        _hist_groups = _draft_history_display_groups(draft_logs)
        st.caption(
            f"共 **{len(_hist_groups)}** 条（同批次已合并）；在下拉框选择一条后下方展示详情，交互与「③ 文档审核」历史列表一致。"
        )

        def _hist_draft_row_label(i: int) -> str:
            g0 = _hist_groups[int(i)]
            r0 = g0["display_rec"]
            ex0 = r0.get("extra") or {}
            fns = ex0.get("generated_file_names") or list((ex0.get("out_files_by_target") or {}).keys())
            fn_line = "、".join(fns) if fns else "（无）"
            if len(fn_line) > 120:
                fn_line = fn_line[:117] + "…"
            bid = (g0.get("batch_id") or "").strip()
            bid_part = f"[批次·{bid}] " if bid else ""
            post_part = "" if post_audit_only else ("[审核后修改] " if ex0.get("post_audit") else "")
            merge_part = f" | 合并×{g0.get('merged_n')}" if int(g0.get("merged_n") or 1) > 1 else ""
            return f"{r0.get('created_at', '')} | {post_part}{bid_part}project_id={ex0.get('project_id', '')} | {fn_line}{merge_part}"

        _pick_g = st.selectbox(
            "选择历史记录" if post_audit_only else "选择历史生成记录",
            options=list(range(len(_hist_groups))),
            format_func=_hist_draft_row_label,
            key=f"{_hk}_group_pick",
        )
        _grp = _hist_groups[int(_pick_g)]
        rec = _grp["display_rec"]
        idx = int(_pick_g)

        extra = rec.get("extra") or {}
        out_files = extra.get("out_files") or []
        fns = extra.get("generated_file_names") or list((extra.get("out_files_by_target") or {}).keys())
        fn_line = "、".join(fns) if fns else "（无）"
        # 标题中展示全部文件名（过长截断），避免只看到第一个文件名
        _title_fn = fn_line if len(fn_line) <= 160 else (fn_line[:157] + "…")
        _bid = (extra.get("batch_id") or "").strip()
        _bid_part = f" | 批次={_bid}" if _bid else ""
        title = f"{rec.get('created_at','')}{_bid_part} | project_id={extra.get('project_id','')} | 生成文件：{_title_fn}"
        st.markdown("---")
        st.markdown(f"##### {title}")
        st.caption(extra.get("summary") or "")
        if _bid:
            st.caption(f"批次ID：{_bid}")
        st.caption(f"生成文件名：{fn_line}")
        mid = (rec.get("model_info") or "").strip()
        if mid:
            st.caption(f"调用模型：{mid}")
        if post_audit_only:
            _pv_map = extra.get("postaudit_preview_text_by_target") or {}
            if not isinstance(_pv_map, dict):
                _pv_map = {}
            _otm_pv = extra.get("out_files_by_target") or {}
            if isinstance(_otm_pv, dict):
                for _dk, _dv in _otm_pv.items():
                    _ds = str(_dv or "").strip()
                    if not _ds.lower().endswith(".postaudit_preview.txt"):
                        continue
                    try:
                        _pr = _resolve_draft_artifact_path(_ds)
                        if _pr.is_file():
                            _disk_txt = _pr.read_text(encoding="utf-8", errors="replace")
                            _kk = str(_dk or "").strip() or Path(_ds).stem
                            _old = _pv_map.get(_kk) or ""
                            if len(_disk_txt) > len(_old):
                                _pv_map[_kk] = _disk_txt
                    except Exception:
                        pass
            _pv_keys = [str(k) for k in _pv_map.keys() if str(k).strip()]
            if _pv_keys:
                st.markdown("##### 生成文本预览（历史内嵌，刷新后仍可查看）")
                _pv_i = st.selectbox(
                    "预览目标文件",
                    options=list(range(len(_pv_keys))),
                    format_func=lambda i: _pv_keys[int(i)],
                    key=f"{_hk}_pv_pick_{rec.get('id')}_{idx}",
                )
                _pv_k = _pv_keys[int(_pv_i)]
                _pv_val = str(_pv_map.get(_pv_k) or "")
                st.text_area(
                    "预览内容",
                    value=_pv_val[:50000],
                    height=280,
                    key=f"{_hk}_pv_txt_{rec.get('id')}_{idx}_{_pv_i}",
                )
                if len(_pv_val) > 50000:
                    st.caption("内容过长已截断；请下载落盘产物或原始输出文件查看全文。")
        cols = st.columns([1, 3])
        with cols[0]:
            if post_audit_only:
                st.caption(
                    "审核后修改历史仅支持下载产物；不复用「初稿重新生成」链路（避免丢失审核待落实上下文）。"
                    " 请在本页重新加载交接并执行。"
                )
            elif st.button("重新生成", key=f"{_hk}_regen_{rec.get('id')}_{idx}"):
                # Streamlit 点击按钮本身就会触发 rerun；这里仅写入 session_state 作为“指令”。
                st.session_state["draft_regen_extra"] = dict(extra)
                st.session_state["_draft_regen_notice"] = (
                    f"已触发历史记录重新生成：{rec.get('created_at','')} | project_id={extra.get('project_id','')}"
                )
        with cols[1]:
            # 下载：恢复为“下拉选择文件”，并在文件名后标记是否产生修改；同时区分输出文件与 JSON 产物
            # 修改状态：以 patch.report.json 的 changes 数为准（比 out_file 映射更可靠）
            _changes_by_report: dict = {}
            _changes_by_out: dict = {}
            # 兜底：从历史记录的 per_file_patch_summaries 读取（即使 report 文件不可读也能标记）
            try:
                _summ = extra.get("per_file_patch_summaries") or []
                if isinstance(_summ, list) and _summ:
                    for ent in _summ:
                        if not isinstance(ent, dict):
                            continue
                        rp0 = (ent.get("patch_report_path") or "").strip()
                        op0 = (ent.get("out_file") or "").strip()
                        pc0 = ent.get("patch_counts") if isinstance(ent.get("patch_counts"), dict) else {}
                        try:
                            n0 = int(pc0.get("changes") or 0) if isinstance(pc0, dict) else 0
                        except Exception:
                            n0 = 0
                        if rp0 and rp0 not in _changes_by_report:
                            _changes_by_report[rp0] = n0
                            try:
                                _changes_by_report[Path(rp0).name] = n0
                            except Exception:
                                pass
                        if op0 and op0 not in _changes_by_out:
                            _changes_by_out[op0] = n0
                            try:
                                _changes_by_out[Path(op0).name] = n0
                            except Exception:
                                pass
            except Exception:
                pass
            try:
                _rep_paths = [
                    str(x).strip()
                    for x in (out_files or [])
                    if isinstance(x, str) and str(x).strip().endswith(".patch.report.json")
                ]
                for rp in _rep_paths:
                    try:
                        rp_path = _resolve_draft_artifact_path(rp)
                        if not rp_path.is_file():
                            continue
                        obj = json.loads(rp_path.read_text(encoding="utf-8"))
                        if not isinstance(obj, dict):
                            continue
                        ch = obj.get("changes") or []
                        n_ch = len(ch) if isinstance(ch, list) else 0
                        _changes_by_report[rp] = n_ch
                        try:
                            _changes_by_report[Path(rp).name] = n_ch
                        except Exception:
                            pass
                        try:
                            out_guess = str(rp)[: -len(".patch.report.json")]
                            _changes_by_out[out_guess] = n_ch
                            _changes_by_out[Path(out_guess).name] = n_ch
                        except Exception:
                            pass
                    except Exception:
                        continue
            except Exception:
                pass

            # 1) 输出文件（原格式文件）
            _out_map = extra.get("out_files_by_target") or {}
            _out_paths: list = []
            if isinstance(_out_map, dict) and _out_map:
                for _k, _v in _out_map.items():
                    if isinstance(_v, str) and _v.strip():
                        _out_paths.append(str(_v).strip())
            if not _out_paths:
                _out_paths = [
                    str(x).strip()
                    for x in (out_files or [])
                    if isinstance(x, str)
                    and str(x).strip()
                    and (not str(x).endswith(".patch.json"))
                    and (not str(x).endswith(".patch.report.json"))
                    and (not str(x).endswith(".zip"))
                ]
            _out_paths = list(dict.fromkeys(_out_paths))
            if _out_paths:
                def _fmt_out(i: int) -> str:
                    p = _out_paths[int(i)]
                    n_ch = _changes_by_out.get(p) or _changes_by_out.get(Path(p).name)
                    if isinstance(n_ch, int):
                        tag = "（未修改）" if n_ch <= 0 else f"（已修改，changes={n_ch}）"
                    else:
                        tag = ""
                    return f"{Path(p).name}{tag}"

                _picked_out = st.selectbox(
                    "下载原格式文件（输出）",
                    options=list(range(len(_out_paths))),
                    format_func=_fmt_out,
                    index=0,
                    key=f"{_hk}_out_pick_{rec.get('id')}_{idx}",
                )
                _picked_out_path = _out_paths[int(_picked_out)]
                _draft_download_button(
                    label=f"下载：{Path(_picked_out_path).name}",
                    raw_path=_picked_out_path,
                    mime="application/octet-stream",
                    key=f"{_hk}_out_btn_{rec.get('id')}_{idx}_{Path(_picked_out_path).name}",
                )

            # 2) JSON 产物（patch / patch.report）
            _json_paths = [
                str(x).strip()
                for x in (out_files or [])
                if isinstance(x, str)
                and str(x).strip()
                and (str(x).endswith(".patch.json") or str(x).endswith(".patch.report.json"))
            ]
            _json_paths = list(dict.fromkeys(_json_paths))
            if _json_paths:
                def _fmt_json(i: int) -> str:
                    p = _json_paths[int(i)]
                    n_ch = _changes_by_report.get(p)
                    if n_ch is None:
                        try:
                            n_ch = _changes_by_report.get(Path(p).name)
                        except Exception:
                            n_ch = None
                    if p.endswith(".patch.report.json") and isinstance(n_ch, int):
                        return f"{Path(p).name}（changes={int(n_ch)}）"
                    return Path(p).name

                _picked_json = st.selectbox(
                    "下载 JSON（patch / report）",
                    options=list(range(len(_json_paths))),
                    format_func=_fmt_json,
                    index=0,
                    key=f"{_hk}_json_pick_{rec.get('id')}_{idx}",
                )
                _picked_json_path = _json_paths[int(_picked_json)]
                _draft_download_button(
                    label=f"下载：{Path(_picked_json_path).name}",
                    raw_path=_picked_json_path,
                    mime="application/json",
                    key=f"{_hk}_json_btn_{rec.get('id')}_{idx}_{Path(_picked_json_path).name}",
                )

            # 3) 打包下载：输出文件/JSON/全部 三种 zip
            def _zip_download(paths: list, *, label: str, tag: str) -> None:
                try:
                    _zip_members = []
                    for of in paths or []:
                        rpz = _resolve_draft_artifact_path(str(of))
                        if rpz.is_file():
                            _zip_members.append(rpz)
                    if len(_zip_members) < 2:
                        return
                    zdir = settings.uploads_path / "draft_outputs" / "_zip_exports"
                    zdir.mkdir(parents=True, exist_ok=True)
                    _zip_ns = "postaudit" if post_audit_only else "draft"
                    zname = f"{_zip_ns}_batch_{rec.get('id')}_{idx}_{tag}.zip"
                    zpath = zdir / zname
                    _reuse = False
                    try:
                        if zpath.is_file() and (time.time() - zpath.stat().st_mtime) < 600:
                            _reuse = True
                    except Exception:
                        _reuse = False
                    if not _reuse:
                        with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                            for rpz in _zip_members:
                                zf.write(rpz, arcname=rpz.name)
                    _draft_download_button(
                        label=f"{label}（{len(_zip_members)} 个文件，ZIP）",
                        raw_path=str(zpath),
                        mime="application/zip",
                        key=f"{_hk}_zip_{rec.get('id')}_{idx}_{tag}",
                    )
                except Exception:
                    return

            _zip_download(_out_paths, label="打包下载输出文件", tag="outputs")
            _zip_download(_json_paths, label="打包下载 JSON", tag="json")
            # 全部：包含输出文件 + JSON（如果存在）
            _zip_download((_out_paths or []) + (_json_paths or []), label="打包下载全部产物", tag="all")

        # 修改日志：从落盘的 patch.report.json 分项展示（每文件一条，避免大批次只显示前几条）
        # 目标：修改日志下拉列表与“下载产物”一致；即使某文件未产出 patch.report，也要在列表中可选并提示原因。
        rep_candidates = [x for x in out_files if isinstance(x, str) and x.endswith(".patch.report.json")]
        _summ = extra.get("per_file_patch_summaries") or []
        _entries: list = []

        # 1) 先以 out_files_by_target 的「展示名」为主生成条目（避免 8_xxx.docx 与 002_xxx.docx 重复两条）
        _out_map = extra.get("out_files_by_target") or {}
        _out_paths = []
        if isinstance(_out_map, dict) and _out_map:
            for disp_name, opath in _out_map.items():
                if not isinstance(opath, str) or not opath.strip():
                    continue
                opath = opath.strip()
                _out_paths.append(opath)
                try:
                    rp_guess = str(Path(opath).with_suffix(Path(opath).suffix + ".patch.report.json"))
                except Exception:
                    rp_guess = ""
                rp = rp_guess if (rp_guess and (rp_guess in (out_files or []) or rp_guess in rep_candidates)) else ""
                fn_disp = str(disp_name).strip() or Path(opath).name
                _entries.append({"file_name": fn_disp, "out_file": opath, "patch_report_path": rp})
        # 兜底：从 out_files 中挑出“非 patch/json 的主产物”
        if not _out_paths:
            for of in out_files or []:
                if not isinstance(of, str):
                    continue
                s = of.strip()
                if not s:
                    continue
                if s.endswith(".patch.json") or s.endswith(".patch.report.json"):
                    continue
                _out_paths.append(s)
        _out_paths = list(dict.fromkeys(_out_paths))  # 去重保序

        if not _entries:
            for opath in _out_paths:
                try:
                    rp_guess = str(Path(opath).with_suffix(Path(opath).suffix + ".patch.report.json"))
                except Exception:
                    rp_guess = ""
                rp = rp_guess if (rp_guess and (rp_guess in (out_files or []) or rp_guess in rep_candidates)) else ""
                _entries.append({"file_name": Path(opath).name, "out_file": opath, "patch_report_path": rp})

        # 2) 补充 per_file_patch_summaries（含 patch_counts 等信息），按展示名归并
        if isinstance(_summ, list) and _summ:
            for ent in _summ:
                if not isinstance(ent, dict):
                    continue
                rp0 = (ent.get("patch_report_path") or "").strip()
                if not rp0:
                    continue
                fn0 = (
                    (ent.get("file_name") or "").strip()
                    or Path(str(rp0)).name
                )
                mk0 = _draft_hist_file_merge_key(fn0)
                merged = False
                for ee in _entries:
                    if _draft_hist_file_merge_key(str(ee.get("file_name") or "")) == mk0:
                        ee.update(ent)
                        merged = True
                        break
                if not merged:
                    _entries.append(ent)

        if _entries:
            st.markdown(
                f"**就地修改执行报告**（共 **{len(_entries)}** 份输出；按文件**直接展示**修改日志 JSON，无需下拉切换）"
            )
            _show_all_ch = st.checkbox(
                "展开显示全部 changes（所有文件；可能较长）",
                value=False,
                key=f"{_hk}_pr_g{idx}_show_all_changes",
            )

            for j, ent in enumerate(_entries):
                fn_head = (
                    (ent.get("file_name") or Path(str(ent.get("patch_report_path") or "")).name).strip()
                )
                pc0 = ent.get("patch_counts") if isinstance(ent.get("patch_counts"), dict) else {}
                _sub = fn_head
                if pc0:
                    _sub += (
                        f" | applied={pc0.get('applied', '?')} changes={pc0.get('changes', '?')} "
                        f"skipped={pc0.get('skipped', '?')} errors={pc0.get('errors', '?')}"
                    )
                if not (ent.get("patch_report_path") or "").strip():
                    _sub += " | 无 patch.report（可能未启用就地修改或 patch 生成失败降级）"
                st.markdown(f"##### {_sub}")

                rp = (ent.get("patch_report_path") or "").strip()
                if rp:
                    try:
                        rp_path = _resolve_draft_artifact_path(str(rp))
                        if not rp_path.is_file():
                            st.caption("报告文件不存在或路径已失效。")
                        else:
                            data = rp_path.read_text(encoding="utf-8")
                            obj = json.loads(data)
                            if not isinstance(obj, dict):
                                st.caption("报告 JSON 格式异常。")
                            else:
                                changes = obj.get("changes") or []
                                skipped = obj.get("skipped") or []
                                errors = obj.get("errors") or []
                                _draft_show_patch_skipped_errors(
                                    obj, key_prefix=f"{_hk}_pr_{rec.get('id')}_{j}_"
                                )
                                if isinstance(changes, list) and changes:
                                    st.caption(
                                        f"修改日志 changes（共 {len(changes)} 条，JSON；默认每文件前 80 条）"
                                    )
                                    _cap = len(changes) if _show_all_ch else min(80, len(changes))
                                    st.code(
                                        json.dumps(changes[:_cap], ensure_ascii=False, indent=2),
                                        language="json",
                                    )
                                    if not _show_all_ch and len(changes) > 80:
                                        st.caption(
                                            "changes 过长已截断；勾选本节上方「展开显示全部」或下载 patch.report.json。"
                                        )
                                elif isinstance(skipped, list) and skipped:
                                    st.caption(
                                        f"changes=0，skipped（截断前 80 条，共 {len(skipped)} 条）"
                                    )
                                    st.code(
                                        json.dumps(skipped[:80], ensure_ascii=False, indent=2),
                                        language="json",
                                    )
                                    if len(skipped) > 80:
                                        st.caption(
                                            "skipped 过长已截断；请下载 patch.report.json 查看完整内容。"
                                        )
                                elif isinstance(errors, list) and errors:
                                    st.caption("errors")
                                    st.code(
                                        json.dumps(errors[:80], ensure_ascii=False, indent=2),
                                        language="json",
                                    )
                                try:
                                    _draft_download_button(
                                        label=f"下载报告：{Path(rp).name}",
                                        raw_path=str(rp_path),
                                        mime="application/json",
                                        key=f"{_hk}_pr_dl_{rec.get('id')}_{j}_{Path(rp).name}",
                                    )
                                except Exception:
                                    pass
                    except Exception as _he:
                        st.caption(f"读取失败：{_he}")
                else:
                    st.caption(
                        "该文件未生成 patch.report.json，因此没有可展示的修改日志。可在下载产物中查看输出文件内容。"
                    )

    def _render_post_audit_mode() -> None:
        st.markdown("---")
        st.subheader("🩺 审核后修改")
        st.caption("将审核报告中「立即修改」审核点按文件映射到生成流程；支持批量或单选后再修改。")

        handoff = st.session_state.get("_post_audit_handoff")
        if not isinstance(handoff, dict):
            handoff = {}
        if handoff:
            st.caption(
                f"已加载交接：{handoff.get('source_file_name') or '未知来源'} "
                f"| 审核点 {len(handoff.get('all_points') or [])} 条"
            )
            _rp_sum = handoff.get("report") if isinstance(handoff.get("report"), dict) else {}
            if _rp_sum.get("batch") and isinstance(_rp_sum.get("reports"), list):
                st.caption(
                    f"批次汇总：已合并 **{len(_rp_sum.get('reports') or [])}** 份子报告（含多文档一致性等）的「立即修改」审核点。"
                )
            c1, c2 = st.columns(2)
            with c1:
                if st.button("清除当前交接", key="postaudit_clear_handoff"):
                    st.session_state.pop("_post_audit_handoff", None)
                    _streamlit_rerun()
            with c2:
                st.caption(f"交接时间：{handoff.get('created_at') or '-'}")
        else:
            st.info("暂无来自审核页的交接数据。可在「③ 文档审核」中点击“传到文档生成修改”。")
            try:
                recent = get_audit_reports(collection=collection, limit=30)
            except Exception:
                recent = []
            if recent:
                pick = st.selectbox(
                    "或从历史报告加载",
                    options=list(range(len(recent))),
                    format_func=lambda i: f"{recent[i].get('created_at','')} | {(recent[i].get('file_name') or '')[:50]}",
                    key="postaudit_hist_pick",
                )
                if st.button("加载该历史报告", key="postaudit_hist_load"):
                    rec = dict(recent[int(pick)] or {})
                    try:
                        rec["report_json"] = json.loads(rec.get("report_json") or "{}")
                    except Exception:
                        rec["report_json"] = {}
                    report_obj = rec.get("report_json") if isinstance(rec.get("report_json"), dict) else {}
                    report_obj["id"] = int(rec.get("id") or 0)
                    report_obj.setdefault("file_name", rec.get("file_name") or "")
                    st.session_state["_post_audit_handoff"] = _build_post_audit_handoff(
                        report_obj,
                        source_report_id=int(rec.get("id") or 0),
                    )
                    _streamlit_rerun()
            _render_draft_history(post_audit_only=True)
            return

        report_obj = handoff.get("report") if isinstance(handoff.get("report"), dict) else {}
        if not report_obj:
            st.warning("交接数据缺少报告内容。请回到审核页重新传递。")
            _render_draft_history(post_audit_only=True)
            return

        points_all = build_immediate_audit_point_records(
            report_obj,
            get_default_action=_get_multi_doc_default_action,
        )
        if not points_all:
            st.warning("当前报告中没有「立即修改」审核点。")
            _render_draft_history(post_audit_only=True)
            return
        point_lookup = {str(p.get("audit_point_ref")): p for p in points_all}

        targets = list((handoff.get("points_by_target") or {}).keys()) if isinstance(handoff.get("points_by_target"), dict) else []
        if not targets:
            _tmp = build_immediate_audit_remediation_by_target(
                report_obj,
                get_default_action=_get_multi_doc_default_action,
            )
            targets = list((_tmp.get("points_by_target") or {}).keys())
        targets = [str(x).strip() for x in targets if str(x).strip()]
        targets = list(dict.fromkeys(targets))
        if not targets:
            st.warning("未解析到需修改文档目标。")
            _render_draft_history(post_audit_only=True)
            return

        selected_targets = st.multiselect(
            "选择本次需要修改的目标文件（批量）",
            options=targets,
            default=targets,
            key="postaudit_targets",
        )
        if not selected_targets:
            st.info("请至少选择一个目标文件。")
            _render_draft_history(post_audit_only=True)
            return

        st.markdown("#### 选择审核点（支持单个或多个）")
        st.caption(
            "仅传一条：在某个目标文件下的多选框中取消全选，只勾选需要的一条审核点即可；可多目标分别精简后再执行。"
        )
        selected_refs: set = set()
        for t in selected_targets:
            rows = [p for p in points_all if t in (p.get("targets") or [])]
            if not rows:
                st.caption(f"{t}：无可选审核点。")
                continue
            opts = [str(r.get("audit_point_ref") or "") for r in rows if str(r.get("audit_point_ref") or "").strip()]
            labels = {}
            for r in rows:
                rr = str(r.get("audit_point_ref") or "")
                labels[rr] = (
                    f"{rr} | {r.get('location','')[:26]} | {r.get('description','')[:42]}"
                )
            picked = st.multiselect(
                f"{t}（默认全选）",
                options=opts,
                default=opts,
                format_func=lambda x: labels.get(x, x),
                key=f"postaudit_pick_{abs(hash(t))}",
            )
            selected_refs.update(str(x).strip() for x in picked if str(x).strip())
        if not selected_refs:
            st.info("请至少选择一条审核点。")
            _render_draft_history(post_audit_only=True)
            return

        selected_payload = build_immediate_audit_remediation_by_target(
            report_obj,
            get_default_action=_get_multi_doc_default_action,
            selected_refs=selected_refs,
        )
        text_by_target_all = selected_payload.get("text_by_target") or {}
        text_by_target = {
            k: v for k, v in text_by_target_all.items()
            if k in selected_targets and str(v or "").strip()
        }
        if not text_by_target:
            st.warning("选中审核点未形成有效目标文本。")
            _render_draft_history(post_audit_only=True)
            return

        preview_target = st.selectbox(
            "预览某目标文件的审核待落实清单",
            options=list(text_by_target.keys()),
            key="postaudit_preview_target",
        )
        st.text_area(
            "待落实清单预览",
            value=text_by_target.get(preview_target, ""),
            height=220,
            key="postaudit_preview_text",
        )

        def _postaudit_target_key(nm: str) -> str:
            s = str(nm or "").strip()
            if not s:
                return ""
            base = Path(s).name
            m = re.match(r"^\d+_(.+)$", base)
            return (m.group(1) if m else base).strip().lower()

        skip_tpl = st.checkbox(
            "不使用案例模板（不读取案例库文档作结构/风格模板；仅用基础上传+参考+审核待落实文本）",
            value=False,
            key="postaudit_skip_case_template",
        )

        cases = _cached_list_project_cases(collection)
        if not cases:
            if not skip_tpl:
                st.warning("当前知识库下没有项目案例，无法生成。请先上传案例，或勾选「不使用案例模板」。")
                _render_draft_history(post_audit_only=True)
                return
            base_case_id = 0
            case_file_names = []
        else:
            if skip_tpl:
                base_case_id = int(cases[0].get("id") or 0)
                case_file_names = []
                st.caption(
                    "不使用案例模板：已跳过案例库模板文档内容；项目维度占位仍使用知识库中的首个案例记录。"
                )
            else:
                case_labels = [f"ID:{int(c.get('id'))} | {_format_case_option(c)}" for c in cases]
                case_idx = st.selectbox(
                    "模板案例（用于保持章节/编号风格）",
                    options=list(range(len(cases))),
                    format_func=lambda i: case_labels[int(i)],
                    key="postaudit_base_case_idx",
                )
                base_case_id = int(cases[int(case_idx)].get("id"))
                case_file_names = get_project_case_file_names(collection, base_case_id) or []

        case_name_by_key = {}
        for _cf in case_file_names:
            _k = _postaudit_target_key(_cf)
            if _k and _k not in case_name_by_key:
                case_name_by_key[_k] = _cf

        target_name_resolved: Dict[str, str] = {}
        unresolved_targets: List[str] = []
        for _t in text_by_target.keys():
            _k = _postaudit_target_key(_t)
            _resolved = ""
            if _t in case_file_names:
                _resolved = _t
            elif _k and _k in case_name_by_key:
                _resolved = case_name_by_key[_k]
            if _resolved:
                target_name_resolved[_t] = _resolved
            else:
                unresolved_targets.append(_t)
        if unresolved_targets:
            if skip_tpl:
                st.caption(
                    "已勾选不使用案例模板：审核点目标名未对齐到案例库文件（属预期）；将以基础上传与模糊匹配绑定 Base。"
                )
            else:
                st.info(
                    "以下目标文件未在模板案例中直接命中，将尝试按上传基础文件名继续生成："
                    + "、".join(unresolved_targets[:8])
                )

        remediation_for_generate: Dict[str, str] = {}
        for _src_t, _txt in text_by_target.items():
            _dst_t = target_name_resolved.get(_src_t, _src_t)
            if _dst_t in remediation_for_generate:
                remediation_for_generate[_dst_t] = (
                    remediation_for_generate[_dst_t].rstrip() + "\n\n" + str(_txt or "").strip()
                ).strip()
            else:
                remediation_for_generate[_dst_t] = str(_txt or "").strip()

        st.markdown("#### 上传基础文件与参考文件")
        st.caption(
            "基础文件请使用用户侧展示文件名；与审核点目标名尽量一致。"
            " 若仅略有差异，系统会在归一化后做模糊匹配（相似度≥95%视为同一文件）。"
        )
        base_uploads = st.file_uploader(
            "基础文件（可多选）",
            accept_multiple_files=True,
            key="postaudit_base_uploads",
        )
        ref_uploads = st.file_uploader(
            "参考文件（可选，多选）",
            accept_multiple_files=True,
            key="postaudit_ref_uploads",
        )
        base_name_map = {}
        for uf in (base_uploads or []):
            n = str(getattr(uf, "name", "") or "").strip()
            if n:
                base_name_map[n] = uf
        upload_by_key = {}
        for _n, _uf in base_name_map.items():
            _k = _postaudit_target_key(_n)
            if _k and _k not in upload_by_key:
                upload_by_key[_k] = (_n, _uf)
        base_pick_for_target = {}
        src_candidates_by_dst: Dict[str, List[str]] = {}
        for _src_t in text_by_target.keys():
            _dst_t = target_name_resolved.get(_src_t, _src_t)
            src_candidates_by_dst.setdefault(_dst_t, [])
            if _src_t not in src_candidates_by_dst[_dst_t]:
                src_candidates_by_dst[_dst_t].append(_src_t)
        for _dst_t in remediation_for_generate.keys():
            _pick = None
            if _dst_t in base_name_map:
                _pick = (_dst_t, base_name_map[_dst_t])
            else:
                _k = _postaudit_target_key(_dst_t)
                if _k and _k in upload_by_key:
                    _pick = upload_by_key[_k]
            if _pick is None:
                for _src_t in (src_candidates_by_dst.get(_dst_t) or [_dst_t]):
                    _k2 = _postaudit_target_key(_src_t)
                    if _k2 and _k2 in upload_by_key:
                        _pick = upload_by_key[_k2]
                        break
            if _pick is not None:
                base_pick_for_target[_dst_t] = _pick
        _fuzzy_min = 0.95
        _upload_names = list(base_name_map.keys())
        for _dst_t in list(remediation_for_generate.keys()):
            if _dst_t in base_pick_for_target:
                continue
            if not _upload_names:
                break
            _best_u = None
            _best_r = 0.0
            _dn = _postaudit_target_key(_dst_t)
            _dst_raw = str(_dst_t or "").strip().lower()
            for _u in _upload_names:
                _un = _postaudit_target_key(_u)
                _u_raw = str(_u or "").strip().lower()
                _r = 0.0
                if _dn and _un:
                    _r = max(_r, float(SequenceMatcher(None, _dn, _un).ratio()))
                if _dst_raw and _u_raw:
                    _r = max(_r, float(SequenceMatcher(None, _dst_raw, _u_raw).ratio()))
                if _r > _best_r:
                    _best_r = _r
                    _best_u = _u
            if _best_u is not None and _best_r >= _fuzzy_min:
                base_pick_for_target[_dst_t] = (_best_u, base_name_map[_best_u])
        missing = [t for t in remediation_for_generate.keys() if t not in base_pick_for_target]
        st.caption(
            f"目标文件 {len(remediation_for_generate)} 个；匹配到基础文件 {len(base_pick_for_target)} 个。"
            + (f" 未匹配 {len(missing)} 个。" if missing else "")
        )
        if missing:
            st.warning(
                "以下目标未匹配到基础文件："
                + "、".join(missing[:8])
                + "。仍可执行（将按非就地方式生成文本），匹配到基础文件的目标会优先走就地修改。"
            )

        rmeta = _post_audit_form_meta_defaults(report_obj)

        base_lang_val = str(rmeta.get("document_language") or "").strip()
        base_lang_label = DOC_LANG_VALUE_TO_LABEL.get(base_lang_val, "不指定")
        if base_lang_label not in DOC_LANG_OPTIONS:
            base_lang_label = "不指定"
        _pdl_idx = (
            DOC_LANG_OPTIONS.index(base_lang_label)
            if base_lang_label in DOC_LANG_OPTIONS
            else 0
        )
        doc_lang = DOC_LANG_LABEL_TO_VALUE.get(
            st.selectbox(
                "文档语言",
                DOC_LANG_OPTIONS,
                index=_pdl_idx,
                key="postaudit_doc_lang",
            ),
            base_lang_val,
        )

        dims_pa = _cached_dimension_options()
        countries_pa = dims_pa.get("registration_countries", ["中国", "美国", "欧盟"]) or [
            "中国",
            "美国",
            "欧盟",
        ]
        forms_pa = dims_pa.get("project_forms", ["Web", "APP", "PC"]) or ["Web", "APP", "PC"]

        _cdef = _post_audit_dim_first_choice(rmeta.get("registration_country")) or (
            countries_pa[0] if countries_pa else ""
        )
        reg_country = st.selectbox(
            "注册国家",
            countries_pa,
            index=countries_pa.index(_cdef) if _cdef in countries_pa else 0,
            key="postaudit_reg_country",
        )

        _tdef = _post_audit_dim_first_choice(rmeta.get("registration_type")) or (
            REGISTRATION_TYPES[0] if REGISTRATION_TYPES else ""
        )
        reg_type = st.selectbox(
            "注册类别",
            REGISTRATION_TYPES,
            index=REGISTRATION_TYPES.index(_tdef) if _tdef in REGISTRATION_TYPES else 0,
            key="postaudit_reg_type",
        )

        _rcdef = _post_audit_dim_first_choice(rmeta.get("registration_component")) or (
            REGISTRATION_COMPONENTS[0] if REGISTRATION_COMPONENTS else ""
        )
        reg_comp = st.selectbox(
            "注册组成",
            REGISTRATION_COMPONENTS,
            index=REGISTRATION_COMPONENTS.index(_rcdef) if _rcdef in REGISTRATION_COMPONENTS else 0,
            key="postaudit_reg_comp",
        )

        _pfdef = _post_audit_dim_first_choice(rmeta.get("project_form")) or (forms_pa[0] if forms_pa else "")
        proj_form = st.selectbox(
            "项目形态",
            forms_pa,
            index=forms_pa.index(_pfdef) if _pfdef in forms_pa else 0,
            key="postaudit_proj_form",
        )
        inplace_mode = st.checkbox("就地修改（推荐）", value=True, key="postaudit_inplace")
        if inplace_mode and (not base_pick_for_target):
            st.warning(
                "已勾选「就地修改」，但当前 0 个目标匹配到上传的基础文件："
                "无法对 Base 做就地 patch，模型会按非就地方式生成文本，可能多耗 token。"
                "是否继续由你自行判断；仍要执行可直接点下方「开始审核后修改」。"
                "若需要 patch，请先补传或对齐文件名后再跑。"
            )

        if st.button("🚀 开始审核后修改", key="postaudit_run"):
            try:
                generator = DocumentDraftGenerator(collection)
                with tempfile.TemporaryDirectory(prefix="aicheckword_postaudit_") as td:
                    td_path = Path(td)
                    existing_base_files = {}
                    for fn, (_orig_name, uf) in base_pick_for_target.items():
                        p = td_path / fn
                        p.write_bytes(uf.getvalue())
                        existing_base_files[fn] = str(p)
                    input_files = []
                    for uf in (ref_uploads or []):
                        n = str(getattr(uf, "name", "") or "").strip() or f"ref_{uuid.uuid4().hex[:8]}.txt"
                        p = td_path / n
                        p.write_bytes(uf.getvalue())
                        input_files.append((str(p), n))
                    project_id_val = 0
                    try:
                        project_id_val = int(rmeta.get("project_id") or 0)
                    except Exception:
                        project_id_val = 0
                    _progress_msg = st.empty()
                    _supports_progress_text = {"ok": True}
                    try:
                        _progress_bar = st.progress(0, text="审核后修改：准备开始…")
                    except TypeError:
                        _supports_progress_text["ok"] = False
                        _progress_bar = st.progress(0)
                        _progress_msg.caption("审核后修改：准备开始…")

                    def _on_postaudit_progress(msg: str, frac: float) -> None:
                        _f = max(0.0, min(1.0, float(frac)))
                        _pct = int(_f * 100)
                        _txt = f"审核后修改：{msg or ''}（{_pct}%）"
                        try:
                            if _supports_progress_text["ok"]:
                                _progress_bar.progress(_f, text=_txt)
                            else:
                                _progress_bar.progress(_f)
                                _progress_msg.caption(_txt)
                        except TypeError:
                            _supports_progress_text["ok"] = False
                            _progress_bar.progress(_f)
                            _progress_msg.caption(_txt)

                    with st.spinner("正在执行审核后修改…"):
                        res = generator.generate(
                            base_case_id=base_case_id,
                            template_file_names=list(remediation_for_generate.keys()),
                            project_id=(project_id_val or None),
                            existing_base_files=existing_base_files,
                            input_files=input_files,
                            document_language=doc_lang,
                            registration_country=reg_country,
                            registration_type=reg_type,
                            registration_component=reg_comp,
                            project_form=proj_form,
                            project_name=str(rmeta.get("project_name") or ""),
                            product_name=str(rmeta.get("product_name") or ""),
                            model=str(rmeta.get("model") or ""),
                            persist_project_fields=False,
                            skills_patch_text="",
                            rules_patch_text="",
                            provider=st.session_state.get("current_provider"),
                            inplace_patch=bool(inplace_mode),
                            save_as_case=False,
                            draft_strategy="change",
                            author_role="",
                            audit_remediation_by_target=remediation_for_generate,
                            progress_cb=_on_postaudit_progress,
                            skip_case_template_text=bool(skip_tpl),
                        )
                    _on_postaudit_progress("完成", 1.0)

                    out_prefix = f"postaudit_{res.project_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
                    postaudit_job_id = uuid.uuid4().hex[:12]
                    out_files_for_log: list = []
                    out_files_by_target: dict = {}
                    per_file_patch_summaries: list = []
                    last_items: list = []
                    _artifact_notes: list = []
                    _track_changes_pa = bool(st.session_state.get("draft_docx_track_changes", True))
                    _is_en_postaudit = str(doc_lang or "").strip().lower().startswith("en")
                    for fn, txt in (res.generated_files or {}).items():
                        st.markdown("---")
                        st.markdown(f"#### 文件：{fn}")
                        downloads: list = []
                        patch_path_str = ""
                        rep_path_str = ""
                        patch_report_obj = None
                        base_path = existing_base_files.get(fn)
                        patch_json = (getattr(res, "generated_patches", {}) or {}).get(fn) if inplace_mode else None
                        if base_path and inplace_mode and (patch_json or "").strip():
                            base_suffix = Path(base_path).suffix.lower()
                            out_name = Path(fn).stem + (base_suffix or Path(fn).suffix or ".docx")
                            drafts_dir = settings.uploads_path / "draft_outputs"
                            drafts_dir.mkdir(parents=True, exist_ok=True)
                            out_path = _safe_out_path(base_path=base_path, out_path=drafts_dir / f"{out_prefix}_{out_name}")
                            report_obj_patch = None
                            saved = None
                            meta_pa = {
                                "project_id": res.project_id,
                                "project_case_id": getattr(res, "project_case_id", None),
                                "base_file": Path(base_path).name,
                                "change_summary": (
                                    "Post-audit remediation (in-place patch)"
                                    if _is_en_postaudit and inplace_mode
                                    else (
                                        "审核后修改（就地 patch）"
                                        if inplace_mode
                                        else "审核后修改"
                                    )
                                ),
                                "generated_by": (
                                    "aicheckword post-audit revise"
                                    if _is_en_postaudit
                                    else "aicheckword 审核后修改"
                                ),
                            }
                            if base_suffix == ".docx":
                                from src.core.draft_export import export_docx_inplace_patch

                                saved, report_obj_patch = export_docx_inplace_patch(
                                    base_file_path=base_path,
                                    out_path=str(out_path),
                                    patch_json=patch_json,
                                    meta=meta_pa,
                                    track_changes=_track_changes_pa,
                                )
                            elif base_suffix in (".xlsx", ".xls"):
                                from src.core.draft_export import export_xlsx_inplace_patch

                                saved, report_obj_patch = export_xlsx_inplace_patch(
                                    base_file_path=base_path,
                                    out_path=str(out_path),
                                    patch_json=patch_json,
                                    meta=meta_pa,
                                )
                            else:
                                saved = export_like_base(base_path=base_path, out_path=str(out_path), generated_text=txt)
                            patch_report_obj = report_obj_patch
                            if saved and inplace_mode and base_suffix == ".docx" and (patch_json or "").strip():
                                patch_path = Path(saved).with_suffix(Path(saved).suffix + ".patch.json")
                                patch_path.write_text(patch_json, encoding="utf-8")
                                rep_path = Path(saved).with_suffix(Path(saved).suffix + ".patch.report.json")
                                rep_path.write_text(
                                    json.dumps(report_obj_patch, ensure_ascii=False, indent=2),
                                    encoding="utf-8",
                                )
                                patch_path_str = str(patch_path)
                                rep_path_str = str(rep_path)
                                out_files_for_log.append(patch_path_str)
                                out_files_for_log.append(rep_path_str)
                                downloads.extend([saved, patch_path_str, rep_path_str])
                                try:
                                    per_file_patch_summaries.append(
                                        {
                                            "file_name": fn,
                                            "out_file": str(saved),
                                            "base_file": str(base_path),
                                            "suffix": base_suffix,
                                            "patch_report_path": rep_path_str,
                                            "patch_json_path": patch_path_str,
                                            "patch_counts": {
                                                "applied": len((report_obj_patch or {}).get("applied") or [])
                                                if isinstance(report_obj_patch, dict)
                                                else None,
                                                "changes": len((report_obj_patch or {}).get("changes") or [])
                                                if isinstance(report_obj_patch, dict)
                                                else None,
                                                "skipped": len((report_obj_patch or {}).get("skipped") or [])
                                                if isinstance(report_obj_patch, dict)
                                                else None,
                                                "errors": len((report_obj_patch or {}).get("errors") or [])
                                                if isinstance(report_obj_patch, dict)
                                                else None,
                                            },
                                            "no_change": (
                                                isinstance(report_obj_patch, dict)
                                                and len((report_obj_patch.get("applied") or [])) == 0
                                                and len((report_obj_patch.get("changes") or [])) == 0
                                            ),
                                        }
                                    )
                                except Exception:
                                    pass
                            elif saved and inplace_mode and base_suffix in (".xlsx", ".xls") and (patch_json or "").strip():
                                patch_path = Path(saved).with_suffix(Path(saved).suffix + ".patch.json")
                                patch_path.write_text(patch_json, encoding="utf-8")
                                rep_path = Path(saved).with_suffix(Path(saved).suffix + ".patch.report.json")
                                rep_path.write_text(
                                    json.dumps(report_obj_patch, ensure_ascii=False, indent=2),
                                    encoding="utf-8",
                                )
                                patch_path_str = str(patch_path)
                                rep_path_str = str(rep_path)
                                out_files_for_log.append(patch_path_str)
                                out_files_for_log.append(rep_path_str)
                                downloads.extend([saved, patch_path_str, rep_path_str])
                                try:
                                    per_file_patch_summaries.append(
                                        {
                                            "file_name": fn,
                                            "out_file": str(saved),
                                            "base_file": str(base_path),
                                            "suffix": base_suffix,
                                            "patch_report_path": rep_path_str,
                                            "patch_json_path": patch_path_str,
                                            "patch_counts": {
                                                "applied": len((report_obj_patch or {}).get("applied") or [])
                                                if isinstance(report_obj_patch, dict)
                                                else None,
                                                "changes": len((report_obj_patch or {}).get("changes") or [])
                                                if isinstance(report_obj_patch, dict)
                                                else None,
                                                "skipped": len((report_obj_patch or {}).get("skipped") or [])
                                                if isinstance(report_obj_patch, dict)
                                                else None,
                                                "errors": len((report_obj_patch or {}).get("errors") or [])
                                                if isinstance(report_obj_patch, dict)
                                                else None,
                                            },
                                            "no_change": (
                                                isinstance(report_obj_patch, dict)
                                                and len((report_obj_patch.get("applied") or [])) == 0
                                                and len((report_obj_patch.get("changes") or [])) == 0
                                            ),
                                        }
                                    )
                                except Exception:
                                    pass
                            elif saved:
                                downloads.append(saved)
                            if saved:
                                out_files_by_target[fn] = str(saved)
                                out_files_for_log.append(str(saved))
                            if report_obj_patch:
                                _draft_show_patch_skipped_errors(report_obj_patch, key_prefix=f"postaudit_{fn}_")
                                changes = report_obj_patch.get("changes") or []
                                if isinstance(changes, list) and changes:
                                    st.caption(f"修改日志 changes（共 {len(changes)} 条，JSON）")
                                    _show_all_pa = st.checkbox(
                                        "展开显示全部 changes",
                                        value=False,
                                        key=f"postaudit_{fn}_show_all_changes",
                                    )
                                    _cap2 = len(changes) if _show_all_pa else min(80, len(changes))
                                    st.code(
                                        json.dumps(changes[:_cap2], ensure_ascii=False, indent=2),
                                        language="json",
                                    )
                        else:
                            st.text_area(
                                "生成文本预览",
                                value=str(txt or "")[:50000],
                                height=240,
                                key=f"postaudit_preview_{fn}",
                            )
                            _why: list = []
                            if not base_path:
                                _why.append("未匹配到本地上传的基础文件（文件名需与目标模板名一致）")
                            elif not inplace_mode:
                                _why.append("未勾选就地修改（仅预览文本；已尝试将预览落盘为 .txt 便于下载）")
                            elif not (patch_json or "").strip():
                                _why.append("就地修改已开启但本文件 patch 为空（模型未返回可执行修改块）")
                            if _why:
                                _artifact_notes.append("「" + str(fn) + "」" + "；".join(_why))
                            try:
                                drafts_dir = settings.uploads_path / "draft_outputs"
                                drafts_dir.mkdir(parents=True, exist_ok=True)
                                _stem = Path(str(fn or "output")).name or "output"
                                _stem = re.sub(r'[<>:"/\\\\|?*]', "_", _stem).strip() or "output"
                                _txt_name = f"{out_prefix}_{_stem}.postaudit_preview.txt"
                                _txt_path = drafts_dir / _txt_name
                                _txt_path.write_text(str(txt or ""), encoding="utf-8")
                                _saved_txt = str(_txt_path.resolve())
                                downloads.append(_saved_txt)
                                out_files_by_target[str(fn)] = _saved_txt
                                out_files_for_log.append(_saved_txt)
                                st.caption(f"已保存预览文本：{Path(_saved_txt).name}（可在下方「下载产物」或历史中下载）")
                            except Exception as _txe:
                                st.caption(f"预览文本落盘失败：{_txe}")
                        _dls = [str(x) for x in downloads if str(x).strip()]
                        if _dls:
                            _dl_labels = [Path(p).name for p in _dls]
                            _picked = st.selectbox(
                                "下载产物",
                                options=list(range(len(_dls))),
                                format_func=lambda i: _dl_labels[i],
                                index=0,
                                key=f"postaudit_cur_dl_pick_{fn}_{postaudit_job_id}",
                            )
                            _picked_path = _dls[int(_picked)]
                            _mime = (
                                "application/json"
                                if _picked_path.endswith(".json")
                                else (
                                    "text/plain"
                                    if _picked_path.endswith(".txt")
                                    else "application/octet-stream"
                                )
                            )
                            _draft_download_button(
                                label=f"下载：{Path(_picked_path).name}",
                                raw_path=str(_picked_path),
                                mime=_mime,
                                key=f"postaudit_cur_dl_btn_{fn}_{postaudit_job_id}_{Path(_picked_path).name}",
                            )
                        last_items.append(
                            {
                                "file_name": fn,
                                "downloads": downloads,
                                "patch_json_path": patch_path_str or "",
                                "patch_report_path": rep_path_str or "",
                                "per_file_skills_rules": None,
                            }
                        )

                    if not (res.generated_files or {}):
                        _artifact_notes.insert(
                            0,
                            "生成器返回的 generated_files 为空（未产出任何目标文本），请检查模型/模板案例/审核待落实文本是否过长或被截断。",
                        )
                    if out_files_by_target:
                        st.success("审核后修改已完成；可下载产物已落盘并写入历史记录。")
                    else:
                        st.warning(
                            "本次审核后修改未产生任何可下载的落盘路径（历史里可能显示「生成文件：无」）。"
                            + (" 详情：" + "；".join(_artifact_notes[:8]) if _artifact_notes else "")
                        )

                    _sum_extra = (
                        f"审核后修改：共 {len(list((out_files_by_target or {}).keys()))} 个目标文件；"
                        f"{len(per_file_patch_summaries)} 份含就地修改执行报告（*.patch.report.json）。"
                    )
                    if _artifact_notes:
                        _sum_extra += " 说明：" + " | ".join(_artifact_notes[:6])
                        if len(_artifact_notes) > 6:
                            _sum_extra += " …"

                _pv_log: dict = {}
                try:
                    for _fn0, _txt0 in (res.generated_files or {}).items():
                        _s0 = str(_txt0 or "").strip()
                        if _s0:
                            _pv_log[str(_fn0)] = _s0[:16000]
                except Exception:
                    _pv_log = {}

                try:
                    add_operation_log(
                        op_type="draft_generate",
                        collection=collection,
                        file_name="draft_outputs",
                        source="post_audit",
                        extra={
                            "batch_id": postaudit_job_id,
                            "post_audit": True,
                            "source_audit_report_id": int((handoff or {}).get("source_report_id") or 0),
                            "base_case_id": base_case_id,
                            "template_file_names": list(remediation_for_generate.keys()),
                            "project_id": res.project_id,
                            "project_case_id": getattr(res, "project_case_id", None),
                            "save_as_case": False,
                            "inplace_patch": bool(inplace_mode),
                            "draft_strategy": "change",
                            "author_role": "",
                            "document_language": doc_lang,
                            "registration_country": reg_country,
                            "registration_type": reg_type,
                            "registration_component": reg_comp,
                            "project_form": proj_form,
                            "project_name": str(rmeta.get("project_name") or ""),
                            "product_name": str(rmeta.get("product_name") or ""),
                            "model": str(rmeta.get("model") or ""),
                            "out_files": out_files_for_log,
                            "out_files_by_target": out_files_by_target,
                            "generated_file_names": list((out_files_by_target or {}).keys()),
                            "per_file_patch_summaries": per_file_patch_summaries,
                            "artifact_notes": list(_artifact_notes[:30]),
                            "generated_file_keys": list((res.generated_files or {}).keys()),
                            "postaudit_preview_text_by_target": _pv_log,
                            "skip_case_template": bool(skip_tpl),
                            "summary": _sum_extra,
                        },
                        model_info=get_current_model_info(),
                    )
                except Exception as _postaudit_log_err:
                    st.warning(
                        f"审核后修改已完成，但写入操作记录失败（历史列表可能缺失本条）：{_postaudit_log_err}"
                    )
                else:
                    _invalidate_operation_logs_cache()
                try:
                    _items_pa = list(last_items or [])
                    _expand_pa = True if len(_items_pa) <= 3 else False
                    st.session_state["_draft_last_result"] = {
                        "summary": (
                            f"审核后修改 | project_id={res.project_id}"
                            + (
                                f", source_audit_report_id={(handoff or {}).get('source_report_id') or ''}"
                                if (handoff or {}).get("source_report_id")
                                else ""
                            )
                        ).strip(),
                        "batch_id": postaudit_job_id,
                        "expand": _expand_pa,
                        "items": _items_pa,
                    }
                except Exception:
                    pass
            except Exception as e:
                st.error(f"审核后修改失败：{e}")
                st.code(repr(e), language="text")

        _render_draft_history(post_audit_only=True)

    _force_mode = st.session_state.pop("_draft_page_mode_force", None)
    _mode_opts = ["文档初稿生成", "审核后修改"]
    if _force_mode in _mode_opts:
        st.session_state["draft_page_mode_radio"] = _force_mode
    draft_page_mode = _st_radio_compat(
        "本页模式",
        _mode_opts,
        key="draft_page_mode_radio",
        horizontal=True,
    )
    if draft_page_mode == "审核后修改":
        _render_post_audit_mode()
        return

    st.markdown(
        "依据训练知识库中的“项目案例文档”生成新项目初稿："
        "以模板案例的章节/编号/法规部分为底，"
        "仅在系统功能相关章节按输入文档自动替换/补全。"
    )

    with st.expander("1) Skills / Rules 增量补丁（可选）", expanded=False):
        st.caption("文件块格式：每个块以 `### FILE: <相对路径> [@replace]` 开头；下一个 `### FILE:` 之前为内容。")
        st.caption("默认合并去重（按段落）；可用 `@replace` 覆盖同一路径文件。")
        skills_patch_text = st.text_area(
            "skills_patch_text",
            height=160,
            placeholder="### FILE: .cursor/skills/project-grounded-doc-writing/SKILL.md\n（粘贴新增/替换内容）",
            key="draft_skills_patch",
        )
        rules_patch_text = st.text_area(
            "rules_patch_text",
            height=160,
            placeholder="### FILE: .cursor/rules/document-authoring-and-audit.mdc\n（粘贴新增/替换内容）",
            key="draft_rules_patch",
        )

    st.markdown("---")
    st.subheader("🛠️ 生成设置")
    draft_strategy_label = st.selectbox(
        "生成策略（新项目复用 vs 注册变更）",
        [
            "注册变更：对照参考在基础文件上自动识别新增/细化/删除（保留版式与未涉及原文）",
            "新项目复用：按参考文件全量更新内容（保留格式章节不变）",
        ],
        index=0,
        key="draft_strategy",
        help="注册变更：以基础文件为底稿，对照参考逐项判断需增补、改写或删除的内容，并生成可定位修改；不以“改动越少越好”为目标，但避免无关的整篇替换。新项目复用：可变内容整体对齐参考，避免只改一小段仍残留旧项目表述。",
    )
    draft_strategy = "change" if draft_strategy_label.startswith("注册变更") else "reuse"

    _draft_author_role_labels = [
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
    ]
    _draft_author_role_keys = list(_DRAFT_AUTHOR_ROLE_KEYS)

    cases = _cached_list_project_cases(collection)
    if not cases:
        st.warning("当前知识库下没有项目案例。请先在「① 法规训练 & 生成审核点」→ 上传项目案例。")
        _render_draft_history(post_audit_only=False)
        return

    # 模板选择
    case_labels = []
    case_by_idx = []
    for c in cases:
        cid = int(c.get("id"))
        case_labels.append(f"ID:{cid} | {_format_case_option(c)}")
        case_by_idx.append((cid, c))
    base_case_idx = st.selectbox(
        "2) 选择模板项目案例（用于生成格式/法规部分）",
        list(range(len(case_by_idx))),
        format_func=lambda i: case_labels[int(i)],
        key="draft_base_case_idx",
    )
    base_case_id, base_case = case_by_idx[int(base_case_idx)]
    # 切换模板案例时：清空上一次目标文件选择，避免误用“留空=全部”导致默认跑全量
    try:
        _last_case_id = st.session_state.get("_draft_last_base_case_id")
        if _last_case_id != base_case_id:
            st.session_state["_draft_last_base_case_id"] = base_case_id
            st.session_state["draft_template_files"] = []
            # 同时清掉“生成全部文件”的确认（按 case 绑定）
            st.session_state.pop(f"draft_select_all_{_last_case_id}", None)
    except Exception:
        pass
    # 控件值统一由 session_state 驱动：避免 default 与 session_state 同时赋值触发 Streamlit 警告
    st.session_state.setdefault("draft_template_files", [])

    # 案例文件名列表：支持搜索下拉选择（可多选）
    try:
        case_file_names = get_project_case_file_names(collection, base_case_id) or []
    except Exception:
        case_file_names = []
    if case_file_names:
        selected_template_files = st.multiselect(
            "2.1) 选择要生成的文件名称（可搜索，多选；留空=生成该案例下全部文件）",
            options=case_file_names,
            key="draft_template_files",
        )
        # 重要：切换模板/清空多选后，不应默认“全选生成”。需显式确认。
        _select_all = False
        if not selected_template_files:
            _select_all = st.checkbox(
                "生成该案例下全部文件（需显式确认）",
                value=False,
                key=f"draft_select_all_{base_case_id}",
                help="避免切换模板后误触发全量生成。若不勾选，请在上方多选中选择要生成的文件。",
            )
        effective_template_files = list(selected_template_files) if selected_template_files else (list(case_file_names) if _select_all else [])
    else:
        selected_template_files = []
        effective_template_files = []
        st.caption("该案例在知识库中尚未关联到具体文件名（可能未完成训练或 case_id 未写入）。将尝试生成全部可用内容。")

    # 编写人员身份：随「模板案例 + 待生成文件名 + 案例内项目形态/注册类别」变化自动匹配默认项，用户可改
    try:
        _role_sig = (
            f"{base_case_id}|{','.join(sorted(effective_template_files or []))}|"
            f"{(base_case.get('project_form') or '').strip()}|{(base_case.get('registration_type') or '').strip()}"
        )
        if st.session_state.get("_draft_author_role_sig") != _role_sig:
            st.session_state["_draft_author_role_sig"] = _role_sig
            st.session_state["draft_author_role_idx"] = _infer_draft_author_role_idx(
                list(effective_template_files or []),
                registration_type=(base_case.get("registration_type") or ""),
                project_form=(base_case.get("project_form") or ""),
            )
    except Exception:
        st.session_state.setdefault("draft_author_role_idx", 0)

    _draft_ar_i = st.selectbox(
        "编写人员身份（默认按文件名与软件法规相关项目信息智能匹配，可改）",
        list(range(len(_draft_author_role_labels))),
        format_func=lambda i: _draft_author_role_labels[int(i)],
        key="draft_author_role_idx",
        help="默认根据当前待生成文件名关键词（如测试/风险/SRS/架构/说明书等）及模板案例上的注册类别、项目形态推断；更改文件选择或模板案例后会重新匹配。可手动覆盖。",
    )
    draft_author_role = _draft_author_role_keys[int(_draft_ar_i)]
    if draft_author_role == "qa":
        st.caption(
            "测试工程师：参考相对基底的新增/变更需求须在既有测试用例表或测试章节中生成对应用例覆盖（常见为一条需求对应多条用例），并保持可追溯；不新增顶层章节标题。"
        )
    author_role_map = {}

    # 多文件：允许为每个目标文件选择不同身份（同一套参考文件，共同属于同一 project_id 文件集）
    _targets_for_role = list(effective_template_files or [])
    if _targets_for_role:
        with st.expander("按目标文件设置编写人员身份（可选）", expanded=False):
            st.caption("默认按每个目标文件名智能匹配（同上方逻辑）；你也可以逐个手动覆盖。未单独设置时仍使用上方的默认身份。")
            import hashlib as _hashlib

            for _fn in _targets_for_role:
                if _fn not in (case_file_names or []):
                    continue
                _hk = _hashlib.md5((_fn + "role").encode("utf-8")).hexdigest()[:12]
                # 每个目标文件：随「模板案例 + 文件名 + 案例内项目形态/注册类别」变化自动匹配默认项
                try:
                    _sig_pf = (
                        f"{base_case_id}|{_fn}|"
                        f"{(base_case.get('project_form') or '').strip()}|{(base_case.get('registration_type') or '').strip()}"
                    )
                    _sig_key = f"_draft_role_pf_sig_{_hk}"
                    if st.session_state.get(_sig_key) != _sig_pf:
                        st.session_state[_sig_key] = _sig_pf
                        st.session_state[f"draft_role_pf_{_hk}"] = _infer_draft_author_role_idx(
                            [_fn],
                            registration_type=(base_case.get("registration_type") or ""),
                            project_form=(base_case.get("project_form") or ""),
                        )
                except Exception:
                    st.session_state.setdefault(f"draft_role_pf_{_hk}", 0)
                _i = st.selectbox(
                    f"{_fn} → 编写人员身份",
                    list(range(len(_draft_author_role_labels))),
                    format_func=lambda i: _draft_author_role_labels[int(i)],
                    key=f"draft_role_pf_{_hk}",
                )
                rk = _draft_author_role_keys[int(_i)] if int(_i) < len(_draft_author_role_keys) else ""
                if rk:
                    author_role_map[_fn] = rk

    _cfg_for_per_file = list(effective_template_files or [])
    if case_file_names and _cfg_for_per_file:
        with st.expander("2.2) 按生成文件配置 Skills / Rules（可选，保存到数据库）", expanded=False):
            st.caption(
                "为每个「待生成文件名」单独配置；绑定当前 **2）模板案例** 与 **知识库**。"
                "生成时与「1) 全局 Skills/Rules 增量补丁」叠加注入提示词。"
                "此处保存**仅写入数据库**，不会自动修改仓库里的 `.cursor/skills` / `.cursor/rules` 文件（除非你在 1) 中使用 `### FILE:` 块显式落盘）。"
            )
            import hashlib as _hashlib

            for _fn in _cfg_for_per_file:
                if _fn not in (case_file_names or []):
                    continue
                st.markdown(f"**{_fn}**")
                _row_pf = None
                try:
                    _row_pf = get_draft_file_skills_rules(collection, base_case_id, _fn)
                except Exception:
                    _row_pf = None
                _sk0 = (_row_pf or {}).get("skills_patch") or ""
                _ru0 = (_row_pf or {}).get("rules_patch") or ""
                _hk = _hashlib.md5(_fn.encode("utf-8")).hexdigest()[:12]
                _psk = st.text_area(
                    "本文件 skills（可写要点说明，或 `### FILE:` 块；仅注入本文件生成提示词）",
                    value=_sk0,
                    height=120,
                    key=f"draft_pfs_sk_{base_case_id}_{_hk}",
                )
                _pru = st.text_area(
                    "本文件 rules",
                    value=_ru0,
                    height=120,
                    key=f"draft_pfs_ru_{base_case_id}_{_hk}",
                )
                _bc1, _bc2 = st.columns(2)
                with _bc1:
                    if st.button("保存本文件配置", key=f"draft_pfs_save_{base_case_id}_{_hk}"):
                        try:
                            upsert_draft_file_skills_rules(collection, int(base_case_id), _fn, _psk, _pru)
                            st.success(f"已保存：{_fn}")
                            _streamlit_rerun()
                        except Exception as _e:
                            st.error(f"保存失败：{_e}")
                with _bc2:
                    if st.button("清除本文件配置", key=f"draft_pfs_del_{base_case_id}_{_hk}"):
                        try:
                            delete_draft_file_skills_rules(collection, int(base_case_id), _fn)
                            st.session_state.pop(f"draft_pfs_sk_{base_case_id}_{_hk}", None)
                            st.session_state.pop(f"draft_pfs_ru_{base_case_id}_{_hk}", None)
                            st.success("已清除")
                            _streamlit_rerun()
                        except Exception as _e:
                            st.error(f"清除失败：{_e}")
                st.markdown("---")

    # 语言
    base_lang_val = (base_case.get("document_language") or "").strip()
    base_lang_label = DOC_LANG_VALUE_TO_LABEL.get(base_lang_val, "不指定")
    doc_lang_val = DOC_LANG_LABEL_TO_VALUE.get(
        st.selectbox(
            "3) 生成文档语言",
            DOC_LANG_OPTIONS,
            index=DOC_LANG_OPTIONS.index(base_lang_label) if base_lang_label in DOC_LANG_OPTIONS else 0,
            key="draft_document_lang",
        ),
        base_lang_val,
    )

    dims = _cached_dimension_options()
    countries = dims.get("registration_countries", ["中国", "美国", "欧盟"]) or ["中国", "美国", "欧盟"]
    forms = dims.get("project_forms", ["Web", "APP", "PC"]) or ["Web", "APP", "PC"]

    registration_country_default = base_case.get("registration_country") or (countries[0] if countries else "")
    registration_country = st.selectbox(
        "4) 注册国家",
        countries,
        index=countries.index(registration_country_default) if registration_country_default in countries else 0,
        key="draft_reg_country",
    )

    registration_type_default = base_case.get("registration_type") or (REGISTRATION_TYPES[0] if REGISTRATION_TYPES else "")
    registration_type = st.selectbox(
        "5) 注册类别",
        REGISTRATION_TYPES,
        index=REGISTRATION_TYPES.index(registration_type_default) if registration_type_default in REGISTRATION_TYPES else 0,
        key="draft_reg_type",
    )

    registration_component_default = base_case.get("registration_component") or (REGISTRATION_COMPONENTS[0] if REGISTRATION_COMPONENTS else "")
    registration_component = st.selectbox(
        "6) 注册组成",
        REGISTRATION_COMPONENTS,
        index=REGISTRATION_COMPONENTS.index(registration_component_default) if registration_component_default in REGISTRATION_COMPONENTS else 0,
        key="draft_reg_component",
    )

    project_form_default = base_case.get("project_form") or (forms[0] if forms else "")
    project_form = st.selectbox(
        "7) 项目形态",
        forms,
        index=forms.index(project_form_default) if project_form_default in forms else 0,
        key="draft_project_form",
    )

    st.caption(f"适用范围（scope_of_application）：{(base_case.get('scope_of_application') or '').strip() or '（模板为空/将按生成过程保持空或尽量保持）'}")

    input_files = st.file_uploader(
        "8) 上传输入/参考文件（用于提炼系统功能与基本信息，同时作为生成参考；支持 Word/Excel/PDF，多文件）",
        type=["docx", "doc", "pdf", "xlsx", "xls", "txt", "md", "zip", "tar", "gz", "tgz"],
        accept_multiple_files=True,
        key="draft_input_upload",
    )
    with st.expander("📁 通过文件夹路径加载输入/参考文件（可选）", expanded=False):
        st.caption("适用场景：输入/参考文件位于运行本工具的机器上（服务器/本机）。会递归扫描支持格式；路径中含「废弃」的文件/目录会自动跳过。")
        _idir = st.text_input("输入/参考文件夹路径", value="", placeholder=r"例如：D:\docs\inputs", key="draft_input_dir_path")
        st.session_state.setdefault("draft_input_dir_items", [])
        da, db = st.columns(2)
        with da:
            if st.button("添加输入文件夹", key="draft_input_dir_add_btn"):
                try:
                    fps = _scan_directory_files(_idir.strip()) if _idir.strip() else []
                    if fps:
                        for fp in fps:
                            if str(fp) not in [x[0] for x in st.session_state["draft_input_dir_items"]]:
                                st.session_state["draft_input_dir_items"].append((str(fp), fp.name, False))
                        st.success(f"已添加 {len(fps)} 个文件。")
                    else:
                        st.warning("未扫描到可用输入/参考文件。")
                except Exception as _e:
                    st.error(f"扫描失败：{_e}")
        with db:
            if st.button("清空输入文件夹列表", key="draft_input_dir_clear_btn"):
                st.session_state["draft_input_dir_items"] = []
                st.success("已清空。")
        if st.session_state.get("draft_input_dir_items"):
            st.caption(f"输入文件夹已加入 **{len(st.session_state['draft_input_dir_items'])}** 个文件；生成时会与上方上传文件合并。")

    # 可选：上传已有文件作为基底（继续编写）
    st.markdown("---")
    st.subheader("📎 上传已有文件（多文件）")
    st.caption(
        "基础文件（Base）：作为对应目标文档的基底修改；可多份上传。"
        "开启「自动分配」后，由 AI 判断每个目标模板使用哪份 Base、以及各参考文件内容主要写入哪份文档；"
        "关闭则需手动将每份 Base 绑定到目标文件名。"
    )

    base_target_options = effective_template_files or []
    if not base_target_options and case_file_names:
        base_target_options = case_file_names[:]

    base_files = st.file_uploader(
        "基础文件（Base，可多选）",
        type=["docx", "doc", "pdf", "xlsx", "xls", "txt", "md", "zip", "tar", "gz", "tgz"],
        accept_multiple_files=True,
        key="draft_existing_base_files",
    )
    with st.expander("📁 通过文件夹路径加载基础文件（Base，可选）", expanded=False):
        st.caption("适用场景：基础文件位于运行本工具的机器上（服务器/本机）。会递归扫描支持格式；路径中含「废弃」的文件/目录会自动跳过。")
        _bdir = st.text_input("基础文件夹路径", value="", placeholder=r"例如：D:\docs\bases", key="draft_base_dir_path")
        st.session_state.setdefault("draft_base_dir_items", [])
        ba, bb = st.columns(2)
        with ba:
            if st.button("添加基础文件夹", key="draft_base_dir_add_btn"):
                try:
                    fps = _scan_directory_files(_bdir.strip()) if _bdir.strip() else []
                    # 仅保留 base 支持格式
                    fps = [fp for fp in fps if fp.suffix.lower() in LOADER_MAP]
                    if fps:
                        for fp in fps:
                            if str(fp) not in [x[0] for x in st.session_state["draft_base_dir_items"]]:
                                st.session_state["draft_base_dir_items"].append((str(fp), fp.name, False))
                        st.success(f"已添加 {len(fps)} 个基础文件。")
                    else:
                        st.warning("未扫描到可用基础文件。")
                except Exception as _e:
                    st.error(f"扫描失败：{_e}")
        with bb:
            if st.button("清空基础文件夹列表", key="draft_base_dir_clear_btn"):
                st.session_state["draft_base_dir_items"] = []
                st.success("已清空。")
        if st.session_state.get("draft_base_dir_items"):
            st.caption(f"基础文件夹已加入 **{len(st.session_state['draft_base_dir_items'])}** 个文件；生成时会与上方上传基础文件合并。")
    multi_base_auto = st.checkbox(
        "多份基础/多份参考时由 AI 自动分配（推荐：自动匹配改哪几份 Base、参考内容如何拆分）",
        value=True,
        key="draft_multi_base_auto",
        help="开启：无需逐份绑定；系统会为每个待生成目标选择基础文件并拆分参考摘要。关闭：使用下方手动绑定。",
    )
    docx_track_changes = st.checkbox(
        "就地修改导出 Word 时使用修订标记（插入/删除，便于在 Word 中审阅修订）",
        value=True,
        key="draft_docx_track_changes",
        help="仅对就地 patch 且输出为 .docx 时生效；表格内替换等仍可能为直接改写。",
    )
    inplace_mode = st.checkbox(
        "✅ 就地修改（保留基础文件格式，推荐用于注册递交版式）",
        value=True,
        key="draft_inplace_mode",
        help="开启后：若目标绑定了 Base（Word/Excel），AI 输出可执行 patch，在原文档上按锚点执行替换/插入/删除，尽量保留样式与结构；修改范围由“参考 vs 基础”的差异决定，而非刻意压少改动量。",
    )

    base_bindings = {}
    if base_files and not multi_base_auto:
        if not base_target_options:
            st.warning("请先在上方选择要生成的目标文件名（2.1 多选），再绑定基础文件。")
        else:
            st.markdown("**基础文件绑定（Base → 目标文件名）**")
            for i, bf in enumerate(base_files):
                # 尝试按文件名自动匹配（精确匹配），失败则默认第一个
                default_target = base_target_options[0]
                if bf and bf.name in base_target_options:
                    default_target = bf.name
                sel = st.selectbox(
                    f"基础文件：{bf.name} 绑定到目标文件名",
                    options=base_target_options,
                    index=base_target_options.index(default_target) if default_target in base_target_options else 0,
                    key=f"draft_base_bind_{i}_{bf.name}",
                )
                if sel:
                    if sel in base_bindings:
                        st.warning(f"目标「{sel}」已绑定过基础文件，将被本次「{bf.name}」覆盖（以后者为准）。")
                    base_bindings[sel] = bf
    elif base_files and multi_base_auto:
        st.caption(f"已上传 {len(base_files)} 份基础文件，将走自动分配（无需逐项绑定）。")

    # 目标文件名自动匹配：按基础文件名尝试在当前模板案例文件名列表中选中（不覆盖用户已手动选择）
    try:
        _base_name_candidates = []
        if base_files:
            _base_name_candidates.extend([bf.name for bf in base_files if bf and bf.name])
        try:
            for _p, _dn, _arch in (st.session_state.get("draft_base_dir_items") or []):
                if _dn:
                    _base_name_candidates.append(str(_dn))
        except Exception:
            pass
        _base_name_candidates = [x for x in _base_name_candidates if (x or "").strip()]
        if case_file_names and _base_name_candidates and not selected_template_files:
            _matched = [n for n in _base_name_candidates if n in case_file_names]
            if _matched:
                # 仅当用户未选择且未确认“生成全部”时自动填充
                try:
                    if not st.session_state.get(f"draft_select_all_{base_case_id}"):
                        st.session_state["draft_template_files"] = list(dict.fromkeys(_matched))
                except Exception:
                    pass
    except Exception:
        pass

    st.markdown("---")
    st.subheader("🧾 项目选择（可选：用已有项目 / 新建项目）")
    proj_mode = st.radio(
        "项目模式",
        ["使用已有项目（不新建）", "新建项目"],
        index=0 if projects else 1,
        key="draft_project_mode",
        horizontal=True,
    )
    _draft_mode_track = "_draft_proj_mode_track"
    if st.session_state.get(_draft_mode_track) != proj_mode:
        st.session_state["draft_persist_project_fields"] = False if proj_mode.startswith("使用已有") else True
        st.session_state[_draft_mode_track] = proj_mode
    elif "draft_persist_project_fields" not in st.session_state:
        st.session_state["draft_persist_project_fields"] = False if proj_mode.startswith("使用已有") else True
    persist_project_fields = st.checkbox(
        "将本页填写的项目字段写回项目（保持与②数据互通）",
        key="draft_persist_project_fields",
    )

    selected_project_id = None
    selected_project = None
    if proj_mode.startswith("使用已有"):
        if not projects:
            st.warning("当前知识库下暂无项目，请先在「② 项目与专属资料」创建项目，或切换为「新建项目」。")
        else:
            proj_names = [p["name"] for p in projects]
            sel_name = st.selectbox("选择已有项目", proj_names, key="draft_existing_proj_name")
            selected_project = next((p for p in projects if p.get("name") == sel_name), None)
            if selected_project:
                selected_project_id = int(selected_project["id"])
    st.caption("项目字段与「② 项目与专属资料」一致；本处可临时修改用于本次生成。")

    # 案例库写入策略：已有项目=默认不写入案例库，避免把“待做项目”混入“已完成案例项目”
    save_as_case_default = False if proj_mode.startswith("使用已有") else True
    save_as_case = st.checkbox(
        "将本次生成结果写入案例库（project_cases）",
        value=save_as_case_default,
        key="draft_save_as_case",
        help="建议：仅当该项目文档已达到可作为模板复用的“完成态”时再勾选；日常编写/迭代请勿写入案例库。",
    )

    # 字段默认值：已有项目优先，否则空/模板范围
    _d = selected_project or {}
    p_name = st.text_input("项目名称", value=_d.get("name", "") or "", key="draft_proj_name")
    p_code = st.text_input("项目编号（可选）", value=_d.get("project_code", "") or "", placeholder="例如：OXGWIS（用于文件名等前缀替换）", key="draft_proj_code")
    p_name_en = st.text_input("项目名称（英文）", value=_d.get("name_en", "") or "", placeholder="Project name in English", key="draft_proj_name_en")
    p_product = st.text_input("产品名称（可选）", value=_d.get("product_name", "") or "", placeholder="与项目名称一并加入审核点、一致性核对", key="draft_proj_product")
    p_product_en = st.text_input("产品名称（英文）", value=_d.get("product_name_en", "") or "", placeholder="Product name in English", key="draft_proj_product_en")
    p_model = st.text_input("型号（可选，Model）", value=_d.get("model", "") or "", placeholder="中英文均可；字段名称不区分大小写，取值区分大小写、精确匹配（含空格）", key="draft_proj_model")
    p_model_en = st.text_input("型号（英文，可选）", value=_d.get("model_en", "") or "", placeholder="Model in English", key="draft_proj_model_en")
    p_country_en = st.text_input("注册国家（英文）", value=_d.get("registration_country_en", "") or "", placeholder="e.g. China, USA", key="draft_proj_country_en")
    p_scope_default = (_d.get("scope_of_application") or "").strip() if _d else (base_case.get("scope_of_application") or "").strip()
    p_scope = st.text_area(
        "产品适用范围（可选）",
        value=p_scope_default,
        placeholder="审核时要求文档描述内容不超出此范围；默认带入模板案例或已有项目的适用范围，可修改",
        height=80,
        key="draft_proj_scope",
    )

    st.markdown("---")
    new_case_name = st.text_input("新案例名称（可选，留空则用模板案例名或提取结果拼接）", value="", key="draft_new_case_name")
    project_key = st.text_input("project_key（可选，留空沿用模板）", value="", key="draft_project_key")

    def _replace_prefix_for_key(name: str, new_code: str) -> str:
        fn = (name or "").strip()
        code = (new_code or "").strip()
        if not fn or not code:
            return fn
        m = re.match(r"^([A-Za-z0-9]+)-([A-Za-z]{2,10}-\d{3}.*)$", fn)
        if not m:
            return fn
        return f"{code}-{m.group(2)}"

    st.markdown("---")
    start = st.button("🚀 开始生成文档初稿（写入 DB 与 knowledge base）", key="draft_run")
    generator = DocumentDraftGenerator(collection)

    # ─────────────────────────────────────────────────────────────
    # 后台生成任务：避免页面控件变化导致任务被 rerun 中断
    # ─────────────────────────────────────────────────────────────
    def _get_draft_executor():
        import concurrent.futures

        ex = st.session_state.get("_draft_executor")
        if ex is None:
            ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            st.session_state["_draft_executor"] = ex
        return ex

    def _draft_job_state():
        st.session_state.setdefault(
            "_draft_job",
            {
                "running": False,
                "done": False,
                "error": "",
                "progress": 0.0,
                "progress_msg": "",
                "future": None,
                "tmp_dir": "",
                "result_payload": None,
            },
        )
        return st.session_state["_draft_job"]

    def _submit_draft_job(payload: dict) -> None:
        import concurrent.futures
        import threading
        import traceback

        job = _draft_job_state()
        if job.get("running"):
            return
        lock = threading.Lock()
        job["lock"] = lock
        job["running"] = True
        job["done"] = False
        job["error"] = ""
        job["progress"] = 0.0
        job["progress_msg"] = "准备开始…"
        job["result_payload"] = None
        job["traceback"] = ""
        job["job_id"] = uuid.uuid4().hex[:12]
        job["started_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 不在「仅提交」时写库：避免与「任务完成」draft_generate_job 重复刷屏；完成时会写一条含 batch_id 的记录。

        def _run() -> dict:
            # 在后台线程运行：不要调用 streamlit UI API
            def _on_progress(msg: str, frac: float) -> None:
                try:
                    f = float(frac)
                except Exception:
                    f = 0.0
                f = max(0.0, min(1.0, f))
                with lock:
                    job["progress"] = f
                    job["progress_msg"] = str(msg or "")
                # 同步打印到服务启动控制台，便于观察后台任务日志（不依赖 Streamlit UI）
                try:
                    ts = datetime.now().strftime("%H:%M:%S")
                    _m = (str(msg or "")).strip()
                    if _m:
                        print(f"[draft-job {job.get('job_id')}] {ts} {int(f*100):>3}% {_m}", flush=True)
                except Exception:
                    pass

            try:
                print(f"[draft-job {job.get('job_id')}] started payload_keys={sorted(list(payload.keys()))}", flush=True)
            except Exception:
                pass
            res = generator.generate(**payload, progress_cb=_on_progress)
            try:
                print(f"[draft-job {job.get('job_id')}] finished", flush=True)
            except Exception:
                pass

            # 仅返回必要信息，供主线程渲染下载/历史
            return {"res": res, "payload": payload}

        ex = _get_draft_executor()
        fut = ex.submit(_run)
        job["future"] = fut

    # 若已有后台任务：展示进度，并在完成后回填结果
    job = _draft_job_state()
    fut = job.get("future")
    if fut is not None and job.get("running"):
        import concurrent.futures

        with st.expander("⏳ 正在生成（后台任务）", expanded=True):
            try:
                p = float(job.get("progress") or 0.0)
            except Exception:
                p = 0.0
            st.progress(max(0.0, min(1.0, p)))
            st.caption(job.get("progress_msg") or "处理中…")
            if st.button("取消本次生成（尽力取消）", key="draft_cancel_job"):
                try:
                    # 取消：无论 future 是否已开始执行，都先让 UI 停止“正在生成”与自动刷新；
                    # 若 future 尚未开始执行，cancel 会成功；若已开始则无法立即停止，但不应让页面一直显示 running。
                    _cancelled = False
                    try:
                        _cancelled = bool(fut.cancel())
                    except Exception:
                        _cancelled = False
                    job["error"] = (
                        "已取消本次生成。"
                        if _cancelled
                        else "已请求取消（若任务已开始执行可能无法立即停止）；已停止前端刷新。"
                    )
                    job["running"] = False
                    job["future"] = None
                except Exception:
                    pass

        done = False
        try:
            done = fut.done()
        except Exception:
            done = False
        if done:
            try:
                out = fut.result()
                job["result_payload"] = out
                job["done"] = True
            except Exception as e:
                job["error"] = str(e)
                try:
                    job["traceback"] = traceback.format_exc()
                except Exception:
                    job["traceback"] = ""
            finally:
                job["running"] = False
                job["future"] = None

            # 写入操作记录：后台任务结束（成功/失败/取消）
            try:
                _gen_n = None
                try:
                    if job.get("done") and job.get("result_payload"):
                        _res0 = (job.get("result_payload") or {}).get("res")
                        if _res0 is not None and getattr(_res0, "generated_files", None) is not None:
                            _gen_n = len(_res0.generated_files or {})
                except Exception:
                    _gen_n = None
                add_operation_log(
                    op_type="draft_generate_job",
                    collection=collection,
                    file_name="",
                    source="draft_page_done",
                    extra={
                        "batch_id": job.get("job_id"),
                        "job_id": job.get("job_id"),
                        "started_at": job.get("started_at"),
                        "done": bool(job.get("done")),
                        "error": (job.get("error") or "").strip() or None,
                        "traceback": (job.get("traceback") or "").strip() or None,
                        "progress_msg": job.get("progress_msg"),
                        "generated_file_count": _gen_n,
                    },
                    model_info=get_current_model_info(),
                )
            except Exception:
                pass

        # 自动刷新页面以更新进度/完成状态
        # 若已出现错误或已请求取消，则停止自动刷新，避免“失败/取消仍显示正在生成”
        if job.get("running") and not (job.get("error") or "").strip():
            try:
                import time as _time

                _time.sleep(0.25)
            except Exception:
                pass
            _streamlit_rerun()

    # 保留本次生成结果：避免下载/页面重跑后“页面被清空看不到结果”
    last = st.session_state.get("_draft_last_result")
    if last:
        # 默认展开：勾选控件会触发 rerun，若折叠用户会误以为“内容消失”
        with st.expander(
            "✅ 上一次生成结果（保留展示，不会因下载清空）",
            expanded=bool(last.get("expand", True)),
        ):
            st.caption(last.get("summary") or "")
            if (last.get("batch_id") or "").strip():
                st.caption(f"批次ID：{last.get('batch_id')}")
            if st.button("清空上一次结果", key="draft_clear_last"):
                st.session_state.pop("_draft_last_result", None)
                _streamlit_rerun()
            # Streamlit 禁止 expander 嵌套：此处在外层「上一次生成结果」内，不得再包 st.expander
            for idx, it in enumerate(last.get("items", []) or []):
                fn = it.get("file_name") or ""
                if idx:
                    st.divider()
                st.markdown(f"##### 文件：{fn}")
                _pfsr = it.get("per_file_skills_rules")
                if _pfsr is True:
                    st.caption("单文件 Skills/Rules：已从数据库注入本次提示词（优先生效）。")
                elif _pfsr is False:
                    st.caption("单文件 Skills/Rules：未检测到该文件名的数据库配置，或内容为空。")

                # 修改日志：优先从 patch.report.json 展示（避免把超大 changes 塞进 session_state 导致前端不稳）
                _rep = (it.get("patch_report_path") or "").strip()
                _rep_path = _resolve_draft_artifact_path(_rep) if _rep else Path("")
                if _rep and _rep_path.is_file():
                    try:
                        obj = json.loads(_rep_path.read_text(encoding="utf-8"))
                        if isinstance(obj, dict):
                            _draft_show_patch_skipped_errors(obj, key_prefix=f"draft_last_{fn}_")
                            _chg = obj.get("changes") or []
                            if isinstance(_chg, list) and _chg:
                                st.caption(f"修改日志 changes（共 {len(_chg)} 条，JSON）")
                                _show_last = st.checkbox(
                                    "展开显示全部 changes",
                                    value=False,
                                    key=f"draft_last_show_changes_all_{fn}",
                                )
                                _cap3 = len(_chg) if _show_last else min(80, len(_chg))
                                st.code(
                                    json.dumps(_chg[:_cap3], ensure_ascii=False, indent=2),
                                    language="json",
                                )
                    except Exception:
                        st.caption("patch.report.json 解析失败，可直接下载查看。")

                _dls = [str(x) for x in (it.get("downloads", []) or []) if str(x).strip()]
                if _dls:
                    _dl_labels = [Path(p).name for p in _dls]
                    _picked = st.selectbox(
                        "下载产物",
                        options=list(range(len(_dls))),
                        format_func=lambda i: _dl_labels[i],
                        index=0,
                        key=f"draft_last_dl_pick_{fn}",
                    )
                    _picked_path = _dls[int(_picked)]
                    _draft_download_button(
                        label=f"下载：{Path(_picked_path).name}",
                        raw_path=_picked_path,
                        mime="application/octet-stream",
                        key=f"draft_last_dl_btn_{fn}_{Path(_picked_path).name}",
                    )
            st.markdown("---")

    # 若来自历史记录“重新生成”，允许不上传输入文件：复用项目中已保存的 basic_info/system_functionality
    regen_extra = st.session_state.pop("draft_regen_extra", None)
    if regen_extra:
        try:
            st.session_state["_draft_regen_notice"] = "正在执行历史记录重新生成…（进度见下方进度条）"
            out_map = regen_extra.get("out_files_by_target") or {}
            out_files_hint = regen_extra.get("out_files") or []
            # out_map: target_file_name -> existing artifact path (used as Base)
            existing_base_files_regen = {k: v for k, v in out_map.items() if k and v}
            # 重新生成必须复用原记录选项：
            # - 优先用 extra 中显式保存的 inplace_patch/inplace_mode
            # - 若历史记录未保存该字段，但原记录产物中存在 patch.json / patch.report.json，则推断为就地修改
            inplace_mode_regen = bool(regen_extra.get("inplace_patch") or regen_extra.get("inplace_mode") or False)
            if not inplace_mode_regen:
                try:
                    _has_patch = any(
                        isinstance(x, str) and (x.endswith(".patch.json") or x.endswith(".patch.report.json"))
                        for x in (out_files_hint or [])
                    )
                    inplace_mode_regen = bool(_has_patch)
                except Exception:
                    pass
            st.markdown("### 生成进度（历史记录重新生成）")
            prog = st.progress(0.0)
            status_box = st.empty()
            status_box.caption("准备开始…")

            def _on_progress_regen(msg: str, frac: float) -> None:
                try:
                    f = float(frac)
                except Exception:
                    f = 0.0
                f = max(0.0, min(1.0, f))
                prog.progress(f)
                status_box.caption(msg or "处理中…")

            with st.spinner("正在按历史记录重新生成…"):
                res = generator.generate(
                    base_case_id=int(regen_extra.get("base_case_id") or base_case_id),
                    template_file_names=(regen_extra.get("template_file_names") or None),
                    project_id=int(regen_extra.get("project_id")) if regen_extra.get("project_id") else None,
                    existing_base_files=existing_base_files_regen or None,
                    input_files=[],
                    document_language=str(regen_extra.get("document_language") or doc_lang_val),
                    registration_country=str(regen_extra.get("registration_country") or registration_country),
                    registration_type=str(regen_extra.get("registration_type") or registration_type),
                    registration_component=str(regen_extra.get("registration_component") or registration_component),
                    project_form=str(regen_extra.get("project_form") or project_form),
                    project_name=str(regen_extra.get("project_name") or ""),
                    project_code=str(regen_extra.get("project_code") or ""),
                    project_name_en=str(regen_extra.get("project_name_en") or ""),
                    product_name=str(regen_extra.get("product_name") or ""),
                    product_name_en=str(regen_extra.get("product_name_en") or ""),
                    model=str(regen_extra.get("model") or ""),
                    model_en=str(regen_extra.get("model_en") or ""),
                    registration_country_en=str(regen_extra.get("registration_country_en") or ""),
                    scope_of_application_override=str(regen_extra.get("scope_of_application") or "") or None,
                    persist_project_fields=False,
                    new_case_name=str(regen_extra.get("new_case_name") or ""),
                    project_key=str(regen_extra.get("project_key") or ""),
                    skills_patch_text="",
                    rules_patch_text="",
                    provider=st.session_state.get("current_provider"),
                    inplace_patch=bool(inplace_mode_regen),
                    # 历史记录“重新生成”属于复跑，不应污染案例库；也不应新建 project_case
                    save_as_case=False,
                    progress_cb=_on_progress_regen,
                    draft_strategy=str(regen_extra.get("draft_strategy") or "change"),
                    author_role=str(
                        regen_extra.get("author_role")
                        or _draft_author_role_keys[
                            max(
                                0,
                                min(
                                    int(st.session_state.get("draft_author_role_idx", 0) or 0),
                                    len(_draft_author_role_keys) - 1,
                                ),
                            )
                        ]
                    ),
                    author_role_map=(regen_extra.get("author_role_map") or None),
                )

            # 重新生成也落盘导出，并写入历史记录：便于刷新后查看/下载
            out_files_for_log = []
            out_files_by_target = {}
            last_items = []
            change_logs_by_file = {}
            patch_skipped_by_file = {}
            patch_errors_by_file = {}
            per_file_patch_summaries_regen: list = []
            out_prefix = f"{res.project_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
            _pf_map = getattr(res, "per_file_skills_rules_applied", None) or {}
            for fn, txt in (res.generated_files or {}).items():
                downloads = []
                change_log_obj = None
                patch_report_obj = None

                base_path = existing_base_files_regen.get(fn) if isinstance(existing_base_files_regen, dict) else None
                if base_path:
                    base_suffix = Path(base_path).suffix.lower()
                    # 输出文件后缀必须与基础文件一致（Excel Base 就导出 Excel；避免 .xlsx.docx 这类错误）
                    out_name = fn
                    if base_suffix:
                        out_name = Path(out_name).stem + base_suffix
                    drafts_dir = settings.uploads_path / "draft_outputs"
                    drafts_dir.mkdir(parents=True, exist_ok=True)
                    out_path = drafts_dir / f"{out_prefix}_{out_name}"
                    _is_en_doc = str(regen_extra.get("document_language") or doc_lang_val or "").strip().lower().startswith("en")
                    meta = {
                        "project_id": res.project_id,
                        "project_case_id": None,
                        "base_file": Path(base_path).name,
                        "change_summary": (
                            "Regenerated from history (identify additions/refinements/deletions from references)"
                            if _is_en_doc and inplace_mode_regen
                            else ("Regenerated from history" if _is_en_doc else (
                                "按历史记录重新生成（基于基础文件按规则补写/修订；对照参考识别新增·细化·删除）" if inplace_mode_regen else "按历史记录重新生成（基于基础文件按规则补写/修订）"
                            ))
                        ),
                        "generated_by": ("aicheckword draft generator (regenerated)" if _is_en_doc else "aicheckword 文档初稿生成（历史记录重新生成）"),
                    }
                    saved = None
                    patch_json = (getattr(res, "generated_patches", {}) or {}).get(fn) if inplace_mode_regen else None
                    if inplace_mode_regen and base_suffix == ".docx" and (patch_json or "").strip():
                        from src.core.draft_export import export_docx_inplace_patch

                        saved, patch_report = export_docx_inplace_patch(
                            base_file_path=base_path,
                            out_path=str(out_path),
                            patch_json=patch_json,
                            meta=meta,
                            track_changes=bool(st.session_state.get("draft_docx_track_changes", True)),
                        )
                        patch_report_obj = patch_report
                        change_log_obj = patch_report.get("changes") if isinstance(patch_report, dict) else None
                        patch_path = Path(saved).with_suffix(Path(saved).suffix + ".patch.json")
                        patch_path.write_text(patch_json, encoding="utf-8")
                        rep_path = Path(saved).with_suffix(Path(saved).suffix + ".patch.report.json")
                        rep_path.write_text(json.dumps(patch_report, ensure_ascii=False, indent=2), encoding="utf-8")
                        out_files_for_log.append(str(patch_path))
                        out_files_for_log.append(str(rep_path))
                        downloads.extend([saved, str(patch_path), str(rep_path)])
                        try:
                            per_file_patch_summaries_regen.append(
                                {
                                    "file_name": fn,
                                    "out_file": str(saved),
                                    "base_file": str(base_path),
                                    "suffix": base_suffix,
                                    "patch_report_path": str(rep_path),
                                    "patch_json_path": str(patch_path),
                                    "patch_counts": {
                                        "applied": len((patch_report or {}).get("applied") or [])
                                        if isinstance(patch_report, dict)
                                        else None,
                                        "changes": len((patch_report or {}).get("changes") or [])
                                        if isinstance(patch_report, dict)
                                        else None,
                                        "skipped": len((patch_report or {}).get("skipped") or [])
                                        if isinstance(patch_report, dict)
                                        else None,
                                        "errors": len((patch_report or {}).get("errors") or [])
                                        if isinstance(patch_report, dict)
                                        else None,
                                    },
                                    "no_change": (
                                        isinstance(patch_report, dict)
                                        and len((patch_report.get("applied") or [])) == 0
                                        and len((patch_report.get("changes") or [])) == 0
                                    ),
                                }
                            )
                        except Exception:
                            pass
                    elif inplace_mode_regen and base_suffix in (".xlsx", ".xls") and (patch_json or "").strip():
                        from src.core.draft_export import export_xlsx_inplace_patch

                        saved, patch_report = export_xlsx_inplace_patch(
                            base_file_path=base_path,
                            out_path=str(out_path),
                            patch_json=patch_json,
                            meta=meta,
                        )
                        patch_report_obj = patch_report
                        change_log_obj = patch_report.get("changes") if isinstance(patch_report, dict) else None
                        patch_path = Path(saved).with_suffix(Path(saved).suffix + ".patch.json")
                        patch_path.write_text(patch_json, encoding="utf-8")
                        rep_path = Path(saved).with_suffix(Path(saved).suffix + ".patch.report.json")
                        rep_path.write_text(json.dumps(patch_report, ensure_ascii=False, indent=2), encoding="utf-8")
                        out_files_for_log.append(str(patch_path))
                        out_files_for_log.append(str(rep_path))
                        downloads.extend([saved, str(patch_path), str(rep_path)])
                        try:
                            per_file_patch_summaries_regen.append(
                                {
                                    "file_name": fn,
                                    "out_file": str(saved),
                                    "base_file": str(base_path),
                                    "suffix": base_suffix,
                                    "patch_report_path": str(rep_path),
                                    "patch_json_path": str(patch_path),
                                    "patch_counts": {
                                        "applied": len((patch_report or {}).get("applied") or [])
                                        if isinstance(patch_report, dict)
                                        else None,
                                        "changes": len((patch_report or {}).get("changes") or [])
                                        if isinstance(patch_report, dict)
                                        else None,
                                        "skipped": len((patch_report or {}).get("skipped") or [])
                                        if isinstance(patch_report, dict)
                                        else None,
                                        "errors": len((patch_report or {}).get("errors") or [])
                                        if isinstance(patch_report, dict)
                                        else None,
                                    },
                                    "no_change": (
                                        isinstance(patch_report, dict)
                                        and len((patch_report.get("applied") or [])) == 0
                                        and len((patch_report.get("changes") or [])) == 0
                                    ),
                                }
                            )
                        except Exception:
                            pass
                    else:
                        saved = export_like_base(
                            base_file_path=base_path,
                            out_path=str(out_path),
                            title=fn,
                            content_text=txt or "",
                            meta=meta,
                        )
                        downloads.append(saved)
                    out_files_for_log.append(saved)
                    out_files_by_target[fn] = saved
                    if isinstance(patch_report_obj, dict):
                        _sk = patch_report_obj.get("skipped")
                        if isinstance(_sk, list) and _sk:
                            patch_skipped_by_file[fn] = _sk
                        _er = patch_report_obj.get("errors")
                        if isinstance(_er, list) and _er:
                            patch_errors_by_file[fn] = _er
                else:
                    drafts_dir = settings.uploads_path / "draft_outputs"
                    drafts_dir.mkdir(parents=True, exist_ok=True)
                    txt_path = drafts_dir / f"{out_prefix}_{fn}.draft.txt"
                    txt_path.write_text(txt or "", encoding="utf-8")
                    out_files_for_log.append(str(txt_path))
                    out_files_by_target[fn] = str(txt_path)
                    downloads.append(str(txt_path))

                if change_log_obj:
                    change_logs_by_file[fn] = change_log_obj
                last_items.append(
                    {
                        "file_name": fn,
                        "downloads": downloads,
                        "change_log": change_log_obj,
                        "patch_skipped": patch_report_obj.get("skipped")
                        if isinstance(patch_report_obj, dict)
                        else None,
                        "patch_errors": patch_report_obj.get("errors")
                        if isinstance(patch_report_obj, dict)
                        else None,
                        "per_file_skills_rules": _pf_map.get(fn),
                    }
                )

            try:
                add_operation_log(
                    op_type="draft_generate",
                    collection=collection,
                    file_name="draft_outputs",
                    source="draft_page_regen",
                    extra={
                        "base_case_id": int(regen_extra.get("base_case_id") or base_case_id),
                        "template_file_names": (regen_extra.get("template_file_names") or None),
                        "project_id": res.project_id,
                        "project_case_id": None,
                        "save_as_case": False,
                        "inplace_patch": bool(inplace_mode_regen),
                        "document_language": str(regen_extra.get("document_language") or doc_lang_val),
                        "registration_country": str(regen_extra.get("registration_country") or registration_country),
                        "registration_type": str(regen_extra.get("registration_type") or registration_type),
                        "registration_component": str(regen_extra.get("registration_component") or registration_component),
                        "project_form": str(regen_extra.get("project_form") or project_form),
                        "project_name": str(regen_extra.get("project_name") or ""),
                        "project_code": str(regen_extra.get("project_code") or ""),
                        "project_name_en": str(regen_extra.get("project_name_en") or ""),
                        "product_name": str(regen_extra.get("product_name") or ""),
                        "product_name_en": str(regen_extra.get("product_name_en") or ""),
                        "model": str(regen_extra.get("model") or ""),
                        "model_en": str(regen_extra.get("model_en") or ""),
                        "registration_country_en": str(regen_extra.get("registration_country_en") or ""),
                        "scope_of_application": str(regen_extra.get("scope_of_application") or ""),
                        "new_case_name": str(regen_extra.get("new_case_name") or ""),
                        "project_key": str(regen_extra.get("project_key") or ""),
                        "out_files": out_files_for_log,
                        "out_files_by_target": out_files_by_target,
                        "generated_file_names": list((out_files_by_target or {}).keys()),
                        "per_file_patch_summaries": per_file_patch_summaries_regen,
                        "change_logs_by_file": change_logs_by_file,
                        "patch_skipped_by_file": patch_skipped_by_file,
                        "patch_errors_by_file": patch_errors_by_file,
                        "summary": (
                            f"按历史记录重新生成：共 {len(list((out_files_by_target or {}).keys()))} 个目标文件；"
                            f"其中 {len(per_file_patch_summaries_regen)} 份含就地修改执行报告，请在历史记录下按文件展开查看。"
                        ),
                    },
                    model_info=get_current_model_info(),
                )
            except Exception:
                pass
            else:
                _invalidate_operation_logs_cache()

            try:
                for _it in last_items:
                    _psk = _it.get("patch_skipped")
                    if isinstance(_psk, list) and _psk:
                        st.caption(f"「{_it.get('file_name') or ''}」未命中/跳过的 patch（skipped，共 {len(_psk)} 条）")
                        try:
                            st.code(json.dumps(_psk, ensure_ascii=False, indent=2), language="json")
                        except Exception:
                            st.json(_psk)
                    _perr = _it.get("patch_errors")
                    if isinstance(_perr, list) and _perr:
                        st.caption(f"「{_it.get('file_name') or ''}」patch 执行错误（errors）")
                        try:
                            st.code(json.dumps(_perr, ensure_ascii=False, indent=2), language="json")
                        except Exception:
                            st.json(_perr)
            except Exception:
                pass

            try:
                st.session_state["_draft_last_result"] = {
                    "summary": f"project_id={res.project_id}（历史记录重新生成）",
                    "items": last_items,
                }
            except Exception:
                pass

            st.success(f"✅ 重新生成完成：project_id={res.project_id}（未写入案例库）")
            st.session_state.pop("_draft_regen_notice", None)
        except Exception as _re:
            st.error(f"重新生成失败：{_re}")
            st.session_state["_draft_regen_notice"] = f"重新生成失败：{_re}"
        # 不要 return：继续渲染历史记录区域，便于立即查看/下载

    # 生成进度、生成结果预览：放在“开始生成按钮”下方
    if start and (not job.get("running")):
        _dir_inputs = st.session_state.get("draft_input_dir_items") or []
        if not input_files and not _dir_inputs:
            st.warning("请先上传输入/参考文件，或通过文件夹路径添加输入文件。")
        else:
            # 允许目标文件名为空：将走“仅基于基础文件+参考文件”的生成模式（不强制依赖模板文件名）
            tmp_dir = Path(tempfile.mkdtemp(prefix="aicheckword_draft_in_ui_"))
            input_paths = []
            existing_base_files = {}
            expand_temp_dirs: list = []
            try:
                # 上传输入/参考文件（自动解压压缩包，保留子目录内文档）
                if input_files:
                    _in_items, _in_tmp_dirs = _expand_uploads(input_files)
                    expand_temp_dirs.extend(_in_tmp_dirs or [])
                    for _p, _dn, _arch in (_in_items or []):
                        if not _p:
                            continue
                        try:
                            if not Path(_p).is_file():
                                continue
                        except Exception:
                            continue
                        # 展示名：必须保留“压缩包名/子目录/文件名”，用于多份参考/多份 Base 的路由区分
                        _name = str((_dn or Path(_p).name) or Path(_p).name)
                        input_paths.append((str(_p), str(_name)))
                # 文件夹输入/参考文件（服务端路径）
                try:
                    for p, dn, _arch in (_dir_inputs or []):
                        if p and Path(p).is_file():
                            input_paths.append((str(p), str(dn or Path(p).name)))
                except Exception:
                    pass

                # 保存“已有文件”到临时目录并建立映射：template_file_name -> path（手动绑定时）
                base_files_manifest: list = []
                # 1) 自动分配：上传的 base_files + 文件夹 base_files 全部进入 manifest
                #    压缩包会被解压，子目录内的每个文档都作为独立 Base 进入 manifest
                if base_files and multi_base_auto:
                    _bf_items, _bf_tmp_dirs = _expand_uploads(base_files)
                    expand_temp_dirs.extend(_bf_tmp_dirs or [])
                    for _p, _dn, _arch in (_bf_items or []):
                        if not _p:
                            continue
                        try:
                            if not Path(_p).is_file():
                                continue
                        except Exception:
                            continue
                        # 仅保留受支持的文档类型作为 Base，避免 .zip/.rar 等误作为基础文件
                        if Path(_p).suffix.lower() not in LOADER_MAP:
                            continue
                        # 展示名：保留“压缩包名/子目录/文件名”，让文件夹名参与多基础自动路由
                        _name = str((_dn or Path(_p).name) or Path(_p).name)
                        base_files_manifest.append((str(_p), str(_name)))
                try:
                    _dir_bases = st.session_state.get("draft_base_dir_items") or []
                    for p, dn, _arch in (_dir_bases or []):
                        if p and Path(p).is_file() and Path(p).suffix.lower() in LOADER_MAP:
                            base_files_manifest.append((str(p), str(dn or Path(p).name)))
                except Exception:
                    pass

                # 2) 手动绑定：仅处理上传的 base_files（文件夹 base 不参与手动绑定，避免无 UI 绑定关系）
                if base_bindings and not multi_base_auto:
                    for target_name, bf in base_bindings.items():
                        if not bf or not target_name:
                            continue
                        _tmp_list, _tmp_dirs = _expand_uploads([bf])
                        expand_temp_dirs.extend(_tmp_dirs or [])
                        _doc_items = [
                            (p, dn, _arch) for (p, dn, _arch) in (_tmp_list or [])
                            if p and Path(p).is_file() and Path(p).suffix.lower() in LOADER_MAP
                        ]
                        if not _doc_items:
                            continue
                        # 绑定一份 Base 到目标文件名；若压缩包内多个文档，取首个并提醒
                        _p0, _dn0, _arch0 = _doc_items[0]
                        existing_base_files[str(target_name)] = str(_p0)
                        if len(_doc_items) > 1:
                            try:
                                st.warning(
                                    f"绑定到「{target_name}」的基础文件为压缩包（{bf.name}），内含多个文档；"
                                    f"本次仅使用首个：{Path(_p0).name}。如需分别绑定，请改为逐份上传。"
                                )
                            except Exception:
                                pass
                # 修复：若填写了项目编号，生成文件名会替换前缀；这里同时补一份“替换后 file_name”映射
                if existing_base_files and (p_code or "").strip():
                    extra_map = {}
                    for k, v in existing_base_files.items():
                        kk = _replace_prefix_for_key(k, (p_code or "").strip())
                        if kk and kk not in existing_base_files:
                            extra_map[kk] = v
                    existing_base_files.update(extra_map)

                st.markdown("### 生成进度")
                prog = st.progress(0.0)
                status_box = st.empty()
                status_box.caption("准备开始…")
                status_box.caption("已提交后台任务，进度见上方「正在生成（后台任务）」区域。")

                # 将临时目录记录到 job，便于失败时清理；实际清理在任务完成后进行
                job["tmp_dir"] = str(tmp_dir)

                _submit_draft_job(
                    {
                        "base_case_id": base_case_id,
                        # 目标文件为空时：显式传空列表触发“base-only”模式；否则按选择生成
                        "template_file_names": (effective_template_files if effective_template_files else []),
                        "project_id": selected_project_id if proj_mode.startswith("使用已有") else None,
                        "existing_base_files": existing_base_files or None,
                        "base_files_manifest": base_files_manifest or None,
                        "multi_base_auto_route": bool(base_files_manifest),
                        "input_files": input_paths,
                        "document_language": doc_lang_val,
                        "registration_country": registration_country,
                        "registration_type": registration_type,
                        "registration_component": registration_component,
                        "project_form": project_form,
                        "project_name": p_name.strip() or "",
                        "project_code": p_code.strip() or "",
                        "project_name_en": p_name_en.strip() or "",
                        "product_name": p_product.strip() or "",
                        "product_name_en": p_product_en.strip() or "",
                        "model": p_model.strip() or "",
                        "model_en": p_model_en.strip() or "",
                        "registration_country_en": p_country_en.strip() or "",
                        "scope_of_application_override": p_scope.strip() if p_scope is not None else None,
                        "persist_project_fields": bool(persist_project_fields),
                        "new_case_name": new_case_name.strip() or "",
                        "project_key": project_key.strip() or "",
                        "skills_patch_text": skills_patch_text,
                        "rules_patch_text": rules_patch_text,
                        "provider": st.session_state.get("current_provider"),
                        "inplace_patch": bool(inplace_mode),
                        "save_as_case": bool(save_as_case),
                        "draft_strategy": draft_strategy,
                        "author_role": draft_author_role,
                        "author_role_map": author_role_map or None,
                    }
                )

                _streamlit_rerun()

            except Exception as e:
                st.error(f"提交生成任务失败：{e}")
                st.code(repr(e), language="text")
                try:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                except Exception:
                    pass

    # 若后台任务完成：将结果渲染为“本次生成结果”
    if (not job.get("running")) and job.get("done") and job.get("result_payload"):
        try:
            res = (job.get("result_payload") or {}).get("res")
            if res is None:
                raise RuntimeError("后台任务未返回结果对象")

            st.markdown("### 生成结果预览与下载")
            _rp = getattr(res, "draft_routing_plan", None)
            if _rp:
                with st.expander("本次多基础/多参考路由（AI）", expanded=False):
                    try:
                        st.json(_rp)
                    except Exception:
                        st.write(str(_rp))
            out_files_for_log = []
            out_files_by_target = {}
            last_items = []
            change_logs_by_file = {}
            patch_skipped_by_file = {}
            patch_errors_by_file = {}
            per_file_patch_summaries: list = []
            out_prefix = str(res.project_case_id or res.project_id)
            _pf_map = getattr(res, "per_file_skills_rules_applied", None) or {}
            _base_map = getattr(res, "per_file_base_path", None) or {}
            for fn, txt in (res.generated_files or {}).items():
                with st.expander(f"文件：{fn}", expanded=False):
                    _ap = _pf_map.get(fn)
                    if _ap is True:
                        st.caption("单文件 Skills/Rules：已从数据库注入本次提示词（优先生效）。")
                    elif _ap is False:
                        st.caption("单文件 Skills/Rules：未检测到该输出文件名的数据库配置，或内容为空。")
                    st.text_area("预览（可复制）", value=txt[:50000], height=240)
                    downloads = []
                    change_log_obj = None
                    patch_report_obj = None
                    patch_path_str = ""
                    rep_path_str = ""

                    # 若该文件有绑定 Base，则输出同格式文件（并写入修订记录）；多基础路由时优先 per_file_base_path
                    base_path = _base_map.get(fn) if isinstance(_base_map, dict) else None
                    if not base_path and isinstance(existing_base_files, dict):
                        base_path = existing_base_files.get(fn)
                    if base_path:
                        base_suffix = Path(base_path).suffix.lower()
                        # 输出文件后缀必须与基础文件一致（Excel Base 就导出 Excel；避免 .xlsx.docx 这类错误）
                        out_name = fn
                        if base_suffix:
                            out_name = Path(out_name).stem + base_suffix
                        drafts_dir = settings.uploads_path / "draft_outputs"
                        drafts_dir.mkdir(parents=True, exist_ok=True)
                        out_path = drafts_dir / f"{out_prefix}_{out_name}"
                        out_path = _safe_out_path(base_path=base_path, out_path=out_path)
                        _is_en_doc = str(doc_lang_val or "").strip().lower().startswith("en")
                        meta = {
                            "project_id": res.project_id,
                            "project_case_id": res.project_case_id,
                            "base_file": Path(base_path).name,
                            "change_summary": (
                                "Update based on base document (identify additions/refinements/deletions from references)"
                                if _is_en_doc and inplace_mode
                                else ("Update based on base document" if _is_en_doc else (
                                    "基于基础文件按规则补写/修订生成（对照参考识别新增·细化·删除）" if inplace_mode else "基于基础文件按规则补写/修订生成"
                                ))
                            ),
                            "generated_by": ("aicheckword draft generator" if _is_en_doc else "aicheckword 文档初稿生成"),
                        }
                        saved = None
                        patch_json = (getattr(res, "generated_patches", {}) or {}).get(fn) if inplace_mode else None
                        if inplace_mode and base_suffix == ".docx" and (patch_json or "").strip():
                            from src.core.draft_export import export_docx_inplace_patch

                            saved, patch_report = export_docx_inplace_patch(
                                base_file_path=base_path,
                                out_path=str(out_path),
                                patch_json=patch_json,
                                meta=meta,
                                track_changes=bool(docx_track_changes),
                            )
                            patch_report_obj = patch_report
                            change_log_obj = patch_report.get("changes") if isinstance(patch_report, dict) else None
                            if change_log_obj:
                                st.caption(f"修改日志 changes（共 {len(change_log_obj)} 条，JSON）")
                                _show_now = st.checkbox(
                                    "展开显示全部 changes",
                                    value=False,
                                    key=f"draft_now_show_all_{fn}",
                                )
                                _cap4 = len(change_log_obj) if _show_now else min(80, len(change_log_obj))
                                st.code(
                                    json.dumps(change_log_obj[:_cap4], ensure_ascii=False, indent=2),
                                    language="json",
                                )
                            # 同步保存 patch 与执行报告，便于审计/复盘
                            patch_path = Path(saved).with_suffix(Path(saved).suffix + ".patch.json")
                            patch_path.write_text(patch_json, encoding="utf-8")
                            rep_path = Path(saved).with_suffix(Path(saved).suffix + ".patch.report.json")
                            rep_path.write_text(json.dumps(patch_report, ensure_ascii=False, indent=2), encoding="utf-8")
                            patch_path_str = str(patch_path)
                            rep_path_str = str(rep_path)
                            out_files_for_log.append(str(patch_path))
                            out_files_for_log.append(str(rep_path))
                            downloads.extend([saved, str(patch_path), str(rep_path)])
                            try:
                                per_file_patch_summaries.append(
                                    {
                                        "file_name": fn,
                                        "out_file": str(saved),
                                        "base_file": str(base_path),
                                        "suffix": base_suffix,
                                        "patch_report_path": rep_path_str,
                                        "patch_json_path": patch_path_str,
                                        "patch_counts": {
                                            "applied": len((patch_report or {}).get("applied") or [])
                                            if isinstance(patch_report, dict)
                                            else None,
                                            "changes": len((patch_report or {}).get("changes") or [])
                                            if isinstance(patch_report, dict)
                                            else None,
                                            "skipped": len((patch_report or {}).get("skipped") or [])
                                            if isinstance(patch_report, dict)
                                            else None,
                                            "errors": len((patch_report or {}).get("errors") or [])
                                            if isinstance(patch_report, dict)
                                            else None,
                                        },
                                        "no_change": (
                                            isinstance(patch_report, dict)
                                            and len((patch_report.get("applied") or [])) == 0
                                            and len((patch_report.get("changes") or [])) == 0
                                        ),
                                    }
                                )
                            except Exception:
                                pass
                        elif inplace_mode and base_suffix in (".xlsx", ".xls") and (patch_json or "").strip():
                            from src.core.draft_export import export_xlsx_inplace_patch

                            saved, patch_report = export_xlsx_inplace_patch(
                                base_file_path=base_path,
                                out_path=str(out_path),
                                patch_json=patch_json,
                                meta=meta,
                            )
                            patch_report_obj = patch_report
                            change_log_obj = patch_report.get("changes") if isinstance(patch_report, dict) else None
                            if change_log_obj:
                                st.caption(f"修改日志 changes（共 {len(change_log_obj)} 条，JSON）")
                                _show_now_x = st.checkbox(
                                    "展开显示全部 changes",
                                    value=False,
                                    key=f"draft_now_show_all_xlsx_{fn}",
                                )
                                _cap5 = len(change_log_obj) if _show_now_x else min(80, len(change_log_obj))
                                st.code(
                                    json.dumps(change_log_obj[:_cap5], ensure_ascii=False, indent=2),
                                    language="json",
                                )
                            patch_path = Path(saved).with_suffix(Path(saved).suffix + ".patch.json")
                            patch_path.write_text(patch_json, encoding="utf-8")
                            rep_path = Path(saved).with_suffix(Path(saved).suffix + ".patch.report.json")
                            rep_path.write_text(json.dumps(patch_report, ensure_ascii=False, indent=2), encoding="utf-8")
                            patch_path_str = str(patch_path)
                            rep_path_str = str(rep_path)
                            out_files_for_log.append(str(patch_path))
                            out_files_for_log.append(str(rep_path))
                            downloads.extend([saved, str(patch_path), str(rep_path)])
                            try:
                                per_file_patch_summaries.append(
                                    {
                                        "file_name": fn,
                                        "out_file": str(saved),
                                        "base_file": str(base_path),
                                        "suffix": base_suffix,
                                        "patch_report_path": rep_path_str,
                                        "patch_json_path": patch_path_str,
                                        "patch_counts": {
                                            "applied": len((patch_report or {}).get("applied") or [])
                                            if isinstance(patch_report, dict)
                                            else None,
                                            "changes": len((patch_report or {}).get("changes") or [])
                                            if isinstance(patch_report, dict)
                                            else None,
                                            "skipped": len((patch_report or {}).get("skipped") or [])
                                            if isinstance(patch_report, dict)
                                            else None,
                                            "errors": len((patch_report or {}).get("errors") or [])
                                            if isinstance(patch_report, dict)
                                            else None,
                                        },
                                        "no_change": (
                                            isinstance(patch_report, dict)
                                            and len((patch_report.get("applied") or [])) == 0
                                            and len((patch_report.get("changes") or [])) == 0
                                        ),
                                    }
                                )
                            except Exception:
                                pass
                        else:
                            saved = export_like_base(
                                base_file_path=base_path,
                                out_path=str(out_path),
                                title=fn,
                                content_text=txt or "",
                                meta=meta,
                            )
                            downloads.append(saved)

                        out_files_for_log.append(saved)
                        out_files_by_target[fn] = saved
                        _draft_show_patch_skipped_errors(
                            patch_report_obj,
                            key_prefix=f"draft_patch_{res.project_case_id}_{fn}_",
                        )
                        if isinstance(patch_report_obj, dict):
                            _sk = patch_report_obj.get("skipped")
                            if isinstance(_sk, list) and _sk:
                                patch_skipped_by_file[fn] = _sk
                            _er = patch_report_obj.get("errors")
                            if isinstance(_er, list) and _er:
                                patch_errors_by_file[fn] = _er
                    else:
                        # 非 Base：也落盘 txt，便于历史记录可见/可下载
                        drafts_dir = settings.uploads_path / "draft_outputs"
                        drafts_dir.mkdir(parents=True, exist_ok=True)
                        txt_path = drafts_dir / f"{res.project_case_id}_{fn}.draft.txt"
                        txt_path.write_text(txt or "", encoding="utf-8")
                        out_files_for_log.append(str(txt_path))
                        out_files_by_target[fn] = str(txt_path)
                        downloads.append(str(txt_path))
                    # 下载入口：下拉选择一个产物 + 单按钮（避免平铺刷屏）
                    _dls = [str(x) for x in (downloads or []) if str(x).strip()]
                    if _dls:
                        _dl_labels = [Path(p).name for p in _dls]
                        _picked = st.selectbox(
                            "下载产物",
                            options=list(range(len(_dls))),
                            format_func=lambda i: _dl_labels[i],
                            index=0,
                            key=f"draft_cur_dl_pick_{fn}_{res.project_case_id}",
                        )
                        _picked_path = _dls[int(_picked)]
                        _mime = (
                            "application/json"
                            if _picked_path.endswith(".json")
                            else ("text/plain" if _picked_path.endswith(".txt") else "application/octet-stream")
                        )
                        _draft_download_button(
                            label=f"下载：{Path(_picked_path).name}",
                            raw_path=str(_picked_path),
                            mime=_mime,
                            key=f"draft_cur_dl_btn_{fn}_{res.project_case_id}_{Path(_picked_path).name}",
                        )

                    last_items.append(
                        {
                            "file_name": fn,
                            "downloads": downloads,
                            "patch_json_path": patch_path_str or "",
                            "patch_report_path": rep_path_str or "",
                            "per_file_skills_rules": _pf_map.get(fn),
                        }
                    )
                    # 注意：不要把超大 changes/skipped/errors 塞进 session_state/DB extra；历史页可从 patch.report.json 落盘文件读取

                # 写入生成记录（用于历史查看/下载/重新生成）
                try:
                    add_operation_log(
                        op_type="draft_generate",
                        collection=collection,
                        file_name="draft_outputs",
                        source="draft_page",
                        extra={
                            "batch_id": job.get("job_id"),
                            "base_case_id": base_case_id,
                            "template_file_names": selected_template_files or None,
                            "project_id": res.project_id,
                            "project_case_id": res.project_case_id,
                            "save_as_case": bool(save_as_case),
                            "inplace_patch": bool(inplace_mode),
                            "draft_strategy": draft_strategy,
                            "author_role": draft_author_role,
                            "author_role_map": author_role_map or {},
                            "document_language": doc_lang_val,
                            "registration_country": registration_country,
                            "registration_type": registration_type,
                            "registration_component": registration_component,
                            "project_form": project_form,
                            "project_name": p_name.strip() or "",
                            "project_code": p_code.strip() or "",
                            "project_name_en": p_name_en.strip() or "",
                            "product_name": p_product.strip() or "",
                            "product_name_en": p_product_en.strip() or "",
                            "model": p_model.strip() or "",
                            "model_en": p_model_en.strip() or "",
                            "registration_country_en": p_country_en.strip() or "",
                            "scope_of_application": (p_scope.strip() if p_scope is not None else ""),
                            "new_case_name": new_case_name.strip() or "",
                            "project_key": project_key.strip() or "",
                            "out_files": out_files_for_log,
                            "out_files_by_target": out_files_by_target,
                            "generated_file_names": list((out_files_by_target or {}).keys()),
                            "per_file_patch_summaries": per_file_patch_summaries,
                            # 大体积的 changes/skipped/errors 不写入 DB extra，避免历史页渲染和前端传输不稳定；
                            # 需要审计时可从 per_file_patch_summaries / *.patch.report.json 读取
                            "summary": (
                                f"本批共 {len(list((out_files_by_target or {}).keys()))} 个目标文件；"
                                f"其中 {len(per_file_patch_summaries)} 份含就地修改执行报告（*.patch.report.json），"
                                "请在下方按文件展开查看或下载报告。"
                            ),
                        },
                        model_info=get_current_model_info(),
                    )
                except Exception:
                    pass
                else:
                    _invalidate_operation_logs_cache()

                # 缓存本次结果（仅存轻量信息），避免页面重跑后丢失下载与日志展示
                try:
                    _items0 = list(last_items or [])
                    _expand_last = True if len(_items0) <= 3 else False
                    st.session_state["_draft_last_result"] = {
                        "summary": f"project_id={res.project_id}, project_case_id={res.project_case_id or ''}".strip(),
                        "batch_id": job.get("job_id") or "",
                        "expand": _expand_last,
                        "items": _items0,
                    }
                except Exception:
                    pass

            st.success(
                f"✅ 生成完成：project_id={res.project_id}"
                + (f", project_case_id={res.project_case_id}" if res.project_case_id else "（未写入案例库）")
            )
        except Exception as e:
            st.error(f"生成失败：{e}")
            st.code(repr(e), language="text")
        finally:
            # 清理后台任务临时目录
            try:
                _td = (job.get("tmp_dir") or "").strip()
                if _td:
                    shutil.rmtree(_td, ignore_errors=True)
            except Exception:
                pass
            job["tmp_dir"] = ""
            # 防止重复渲染：重置 done 标记，但保留 last_result 与历史记录
            job["done"] = False
            job["result_payload"] = None

    if (job.get("error") or "").strip() and (not job.get("running")):
        st.error(f"生成任务失败/取消：{job.get('error')}")

    # 历史生成记录：放在“生成设置/生成结果”之后
    _render_draft_history(post_audit_only=False)


def _reports_markdown_for_download(reports: list) -> str:
    """生成当前会话报告 Markdown 文本（轻量，供下载按钮使用）。"""
    md_lines = []
    for report in reports:
        file_name = report.get("original_filename", report.get("file_name", ""))
        md_lines.append(f"# 审核报告：{file_name}\n")
        md_lines.append(f"**总结：** {report.get('summary', '')}\n")
        md_lines.append("| 高风险 | 中风险 | 低风险 | 提示 |")
        md_lines.append("|--------|--------|--------|------|")
        md_lines.append(
            f"| {report.get('high_count', 0)} | {report.get('medium_count', 0)} "
            f"| {report.get('low_count', 0)} | {report.get('info_count', 0)} |\n"
        )
        for i, point in enumerate(report.get("audit_points", []), 1):
            sev_label = {"high": "高", "medium": "中", "low": "低", "info": "提示"}.get(
                (point.get("severity") or "").lower(), point.get("severity", "")
            )
            md_lines.append(f"## [{sev_label}] 审核点 {i}：{point.get('category', '')}")
            md_lines.append(f"- **位置：** {point.get('location', '')}")
            md_lines.append(f"- **描述：** {point.get('description', '')}")
            md_lines.append(f"- **法规依据：** {point.get('regulation_ref', '')}")
            md_lines.append(f"- **建议：** {point.get('suggestion', '')}")
            _docs = point.get("modify_docs") or []
            if _docs:
                md_lines.append(f"- **需修改文档：** {'、'.join(d for d in _docs if d)}")
            _action = point.get("action") or ""
            if _action:
                md_lines.append(f"- **处理状态：** {_action}")
            md_lines.append("")
    return "\n".join(md_lines)


def _render_download_buttons(reports: list, key_suffix: str = ""):
    """渲染报告下载按钮组。Excel/PDF/Word/HTML/带批注 Word 默认不生成，避免 Streamlit 每次重跑都全量导出导致编辑卡顿。"""
    ks = key_suffix or "_root"
    ready_key = f"_sess_audit_dl_heavy_{ks}"

    has_single_docx = (
        len(reports) == 1
        and reports[0].get("_original_path")
        and Path(reports[0]["_original_path"]).suffix.lower() == ".docx"
    )
    has_kdocs_docx = (
        not has_single_docx
        and len(reports) == 1
        and reports[0].get("_kdocs_download_url")
        and (reports[0].get("file_name") or "").lower().endswith(".docx")
    )

    if not st.session_state.get(ready_key):
        st.markdown("**📥 下载**")
        st.caption(
            "默认仅 **JSON / Markdown**（即时生成）。**Excel、PDF、Word、HTML** 与 **带批注 Word** 较慢，点按钮后再生成，避免浏览报告时页面反复卡住。"
        )
        q1, q2, q3 = st.columns([1, 1, 2])
        with q1:
            st.download_button(
                "📥 JSON",
                data=json.dumps(reports, ensure_ascii=False, indent=2),
                file_name="audit_report.json",
                mime="application/json",
                key=f"dl_json_light_{ks}",
            )
        with q2:
            st.download_button(
                "📥 Markdown",
                data=_reports_markdown_for_download(reports),
                file_name="audit_report.md",
                mime="text/markdown",
                key=f"dl_md_light_{ks}",
            )
        with q3:
            if st.button("📥 加载全部格式下载（Excel / PDF / Word / HTML 等，较慢）", key=f"_sess_audit_dl_show_{ks}"):
                st.session_state[ready_key] = True
                _streamlit_rerun()
        return

    extra = 1 if (has_single_docx or has_kdocs_docx) else 0
    cols = st.columns(6 + extra)

    with cols[0]:
        try:
            xlsx_bytes = report_to_excel(reports)
            st.download_button(
                "📥 Excel", data=xlsx_bytes,
                file_name="audit_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"dl_xlsx_{ks}",
            )
        except Exception as e:
            st.caption(f"Excel 生成失败: {e}")
    with cols[1]:
        try:
            pdf_bytes = report_to_pdf(reports)
            st.download_button(
                "📥 PDF", data=pdf_bytes,
                file_name="audit_report.pdf", mime="application/pdf",
                key=f"dl_pdf_{ks}",
            )
        except Exception as e:
            st.caption(f"PDF 生成失败: {e}")
    with cols[2]:
        try:
            docx_bytes = report_to_docx(reports)
            st.download_button(
                "📥 Word", data=docx_bytes,
                file_name="audit_report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key=f"dl_docx_{ks}",
            )
        except Exception as e:
            st.caption(f"Word 生成失败: {e}")
    with cols[3]:
        html_data = report_to_html(reports)
        st.download_button(
            "📥 HTML", data=html_data,
            file_name="audit_report.html", mime="text/html",
            key=f"dl_html_{ks}",
        )
    with cols[4]:
        json_data = json.dumps(reports, ensure_ascii=False, indent=2)
        st.download_button(
            "📥 JSON", data=json_data,
            file_name="audit_report.json", mime="application/json",
            key=f"dl_json_{ks}",
        )
    with cols[5]:
        st.download_button(
            "📥 Markdown",
            data=_reports_markdown_for_download(reports),
            file_name="audit_report.md",
            mime="text/markdown",
            key=f"dl_md_{ks}",
        )
    if has_single_docx:
        with cols[6]:
            try:
                docx_with_comments = report_to_docx_with_comments(
                    reports[0]["_original_path"], reports[0], author="审核"
                )
                st.download_button(
                    "📥 带批注 Word",
                    data=docx_with_comments,
                    file_name="audit_with_comments.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key=f"dl_docx_comments_{ks}",
                    help="在原文档对应位置插入批注，可上传回金山文档或本地 Word 查看",
                )
            except Exception as e:
                st.caption(f"带批注 Word 生成失败: {e}")
    elif has_kdocs_docx:
        with cols[6]:
            try:
                from src.core.kdocs_client import download_file_from_url
                _dl_url = reports[0]["_kdocs_download_url"]
                with st.spinner("正在下载原文件并生成批注..."):
                    raw_bytes = download_file_from_url(_dl_url)
                    docx_with_comments = report_to_docx_with_comments(
                        raw_bytes, reports[0], author="审核"
                    )
                st.download_button(
                    "📥 带批注 Word",
                    data=docx_with_comments,
                    file_name="audit_with_comments.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key=f"dl_kdocs_comments_{ks}",
                    help="下载金山文档原文件并在对应位置插入批注，可上传回金山文档替换原文件",
                )
            except Exception as e:
                st.caption(f"带批注 Word 生成失败: {e}")

    if st.button("收起慢速格式（恢复仅 JSON/Markdown，加快翻页）", key=f"_sess_audit_dl_hide_{ks}"):
        st.session_state.pop(ready_key, None)
        _streamlit_rerun()


def render_knowledge_page():
    """知识库查询页面"""
    st.header("🔎 知识库查询")
    st.markdown("查询已训练的知识库内容，验证法规/标准或审核点是否已正确入库。")

    if not _require_provider():
        return

    agent = init_agent()
    collection = st.session_state.get("collection_name", "regulations")

    def _kb_tab_search():
        kb_target = _st_radio_compat(
            "查询目标",
            ["法规知识库（第一步）", "审核点知识库（第二步）"],
            key="kb_query_target",
        )
        use_checkpoints = "审核点" in kb_target

        query = st.text_input("输入查询内容", placeholder="例如：产品注册需要哪些资料？")
        top_k = st.slider("返回结果数", 1, 20, 5)

        if query and st.button("🔍 查询", key="search_btn"):
            with st.spinner("正在检索..."):
                results = agent.search_knowledge(query, top_k=top_k, use_checkpoints=use_checkpoints)

            if not results:
                st.warning("未找到相关内容，请确认知识库已训练。")
            else:
                for i, result in enumerate(results, 1):
                    with st.expander(f"📄 结果 {i} — 来源：{result['source']}", expanded=(i <= 3)):
                        st.markdown(result["content"])

    def _kb_tab_browse():
        from src.core.db import get_knowledge_docs as _get_kd
        st.markdown("浏览所有已训练入库的文档块（数据存储于 MySQL，可按类型筛选）。")
        try:
            by_cat_browse = get_knowledge_stats_by_category(collection)
            c1, c2 = st.columns(2)
            c1.metric("总文档块数", by_cat_browse.get("total_chunks", 0))
            c2.metric("涉及训练文件数", by_cat_browse.get("total_files", 0))
            bc = by_cat_browse.get("by_category") or {}
            st.caption(
                f"法规: {bc.get('regulation', {}).get('files', 0)} 文件 / {bc.get('regulation', {}).get('chunks', 0)} 块 | "
                f"程序: {bc.get('program', {}).get('files', 0)} / {bc.get('program', {}).get('chunks', 0)} 块 | "
                f"项目案例: {bc.get('project_case', {}).get('files', 0)} / {bc.get('project_case', {}).get('chunks', 0)} 块"
            )
        except Exception:
            pass

        browse_category = st.selectbox(
            "按类型筛选",
            ["全部", "法规文件", "程序文件", "项目案例文件", "词条"],
            key="kb_browse_category",
        )
        browse_cat_value = CATEGORY_VALUES.get(browse_category) if browse_category != "全部" else None

        browse_limit = st.selectbox("显示条数", [20, 50, 100, 200], index=0, key="kb_browse_limit")
        try:
            kb_rows = _get_kd(collection=collection, category=browse_cat_value, limit=browse_limit)
            if kb_rows:
                for row in kb_rows:
                    cat_label = CATEGORY_LABELS.get(row.get("category"), row.get("category") or "未分类")
                    label = f"[{cat_label}] [{row.get('file_name', '')}] 块 #{row.get('chunk_index', 0)} ({row.get('created_at', '')})"
                    with st.expander(label, expanded=False):
                        full_content = row.get("content", "") or ""
                        st.text_area("内容（可滚动查看全文）", value=full_content, height=min(400, max(120, len(full_content) // 25)), disabled=True, key=f"kb_content_{row.get('id')}_{row.get('chunk_index')}")
            else:
                st.info("当前知识库尚无文档记录，或该类型下无数据。")
        except Exception as e:
            st.warning(f"加载知识库文档失败：{e}")

    _kb_tab_labels = ["🔍 语义检索", "📂 知识库文档浏览"]
    _st_run_tabs_or_pick(
        _kb_tab_labels,
        radio_label="知识库查询子功能",
        session_key="kb_page_tabs",
        tab_bodies=[_kb_tab_search, _kb_tab_browse],
    )


def _parse_translation_correction_manual_rules(raw: str):
    """校正手工规则：每行「错误译文|正确译文|中文词条」，第三列可空（仅替换不写库）。"""
    rules = []
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 2 and parts[0] and parts[1]:
            rules.append((parts[0], parts[1], parts[2] if len(parts) > 2 else ""))
    return rules


def render_translation_page():
    """文档翻译页面：单文件/文件夹/zip，仅中文→英文，保持格式与结构，可选知识库参考。"""
    st.header("🌐 文档翻译")
    st.markdown(
        "面向 FDA 医疗器械认证的文档翻译：仅将中文逐句译为英文，保持原格式与目录结构，不覆盖原稿。"
        "支持 .docx / .txt / .xlsx；可上传单文件、多文件或 zip，也可填写本地路径。"
    )
    st.caption("翻译结果需经人工审校后用于正式提交。支持：中文→英文/德文，英文/德文→中文。")

    if not _require_provider():
        return

    # 历史翻译结果：查看与下载
    with st.expander("📥 历史翻译结果", expanded=False):
        try:
            trans_logs = get_operation_logs(op_type=OP_TYPE_TRANSLATION, limit=50)
            if not trans_logs:
                st.caption("暂无历史翻译记录。")
            else:
                for rec in trans_logs:
                    extra = rec.get("extra") or {}
                    out_paths = extra.get("out_paths") or []
                    created = rec.get("created_at")
                    if hasattr(created, "strftime"):
                        created = created.strftime("%Y-%m-%d %H:%M")
                    else:
                        created = str(created)[:16]
                    st.markdown(f"**{created}** — {rec.get('file_name') or '翻译'}（共 {len(out_paths)} 个文件）")
                    if rec.get("source"):
                        st.caption(f"来源：{rec['source'][:80]}{'…' if len(rec.get('source','')) > 80 else ''}")
                    for idx, p in enumerate(out_paths):
                        p_path = Path(p)
                        if p_path.is_file():
                            try:
                                data = p_path.read_bytes()
                                st.download_button(
                                    f"📥 {p_path.name}",
                                    data=data,
                                    file_name=p_path.name,
                                    mime="application/octet-stream",
                                    key=f"hist_dl_{rec.get('id', id(rec))}_{idx}_{p_path.name}",
                                )
                            except Exception:
                                st.caption(f"无法读取：{p}")
                        else:
                            st.caption(f"已不存在：{p}")
                    st.markdown("---")
        except Exception as e:
            st.caption(f"加载历史记录失败：{e}")

    trans_cfg = get_translation_config()
    target_lang_options = [("英文", "en"), ("德文", "de"), ("中文", "zh")]
    target_lang_label = st.selectbox(
        "目标语言",
        [x[0] for x in target_lang_options],
        index=next((i for i, (_, v) in enumerate(target_lang_options) if v == trans_cfg.get("target_lang", "en")), 0),
        key="translation_target_lang",
    )
    target_lang = next(v for lbl, v in target_lang_options if lbl == target_lang_label)

    with st.expander("公司信息翻译配置（可选）", expanded=False):
        st.caption("翻译时遇到公司名称、地址、联系人、电话等将优先使用下列译法，保证全文一致。留空则不强制。")
        cc = trans_cfg.get("company_config") or {}
        c_company = st.text_input("公司名称（目标语言）", value=cc.get("company_name", "") or "", key="trans_company_name")
        c_address = st.text_input("地址（目标语言）", value=cc.get("address", "") or "", key="trans_address")
        c_contact = st.text_input("联系人（目标语言）", value=cc.get("contact", "") or "", key="trans_contact")
        c_phone = st.text_input("电话", value=cc.get("phone", "") or "", key="trans_phone")
        c_fax = st.text_input("传真", value=cc.get("fax", "") or "", key="trans_fax")
        c_email = st.text_input("邮箱", value=cc.get("email", "") or "", key="trans_email")
        if st.button("保存公司信息配置", key="trans_save_company"):
            save_translation_config(target_lang=target_lang, company_config={
                "company_name": c_company.strip(),
                "address": c_address.strip(),
                "contact": c_contact.strip(),
                "phone": c_phone.strip(),
                "fax": c_fax.strip(),
                "email": c_email.strip(),
            })
            st.success("已保存，翻译时将使用上述译法。")

    company_overrides = {k: v for k, v in {
        "company_name": c_company.strip(),
        "address": c_address.strip(),
        "contact": c_contact.strip(),
        "phone": c_phone.strip(),
        "fax": c_fax.strip(),
        "email": c_email.strip(),
    }.items() if v}
    if not company_overrides and (trans_cfg.get("company_config") or {}):
        company_overrides = trans_cfg["company_config"]
    company_overrides = company_overrides or None

    collection = st.session_state.get("collection_name", "regulations")
    st.caption(f"当前知识库：**{collection}**（与第一步训练、第三步文档审核相同）。")
    use_kb = st.checkbox("使用知识库（词条/法规/案例）作为术语与风格参考", value=True, key="translation_use_kb")
    col_name = collection if use_kb else None

    trans_proj_mode = _st_radio_compat(
        "翻译上下文（与第三步「文档审核」一致，用于知识库检索与术语对齐）",
        ["通用（仅按当前知识库检索）", "按项目（选用与审核相同的项目与注册维度）"],
        key="translation_proj_mode",
    )
    kb_query_extra = ""
    trans_project_id = None
    if trans_proj_mode.startswith("按项目"):
        projects = _cached_list_projects(collection)
        if not projects:
            st.warning("当前知识库下暂无项目。请先在「② 审核点管理」→「项目与专属资料」创建项目，或改用「通用」。")
        else:
            dims = _cached_dimension_options()
            countries = dims.get("registration_countries", ["中国", "美国", "欧盟"]) or ["中国", "美国", "欧盟"]
            forms = dims.get("project_forms", ["Web", "APP", "PC"]) or ["Web", "APP", "PC"]
            proj_names = [p["name"] for p in projects]
            selected_name = st.selectbox(
                "选择翻译所属项目",
                proj_names,
                key="translation_proj_name",
                help="与第三步「按项目审核」使用同一项目列表；检索词条/法规/案例时会带上项目与产品信息。",
            )
            proj = next((p for p in projects if p["name"] == selected_name), None)
            if proj:
                trans_project_id = proj.get("id")
                _country = proj.get("registration_country")
                _type = proj.get("registration_type")
                _comp = proj.get("registration_component")
                _form = proj.get("project_form")
                tc1, tc2 = st.columns(2)
                with tc1:
                    sel_tr_c = st.multiselect(
                        "注册国家", countries,
                        default=[_country] if _country in countries else [],
                        key="trans_dim_countries",
                    )
                    sel_tr_t = st.multiselect(
                        "适用注册类别", REGISTRATION_TYPES,
                        default=[_type] if _type in REGISTRATION_TYPES else [],
                        key="trans_dim_types",
                    )
                with tc2:
                    sel_tr_comp = st.multiselect(
                        "注册组成", REGISTRATION_COMPONENTS,
                        default=[_comp] if _comp in REGISTRATION_COMPONENTS else [],
                        key="trans_dim_components",
                    )
                    sel_tr_f = st.multiselect(
                        "项目形态", forms,
                        default=[_form] if _form in forms else [],
                        key="trans_dim_forms",
                    )
                hint_parts = [
                    proj.get("name") or "",
                    proj.get("product_name") or "",
                    proj.get("name_en") or "",
                    proj.get("product_name_en") or "",
                    proj.get("model") or "",
                    proj.get("model_en") or "",
                    (proj.get("scope_of_application") or "")[:500],
                    " ".join(sel_tr_c or []),
                    " ".join(sel_tr_t or []),
                    " ".join(sel_tr_comp or []),
                    " ".join(sel_tr_f or []),
                ]
                kb_query_extra = " ".join(p for p in hint_parts if p).strip()
    st.caption("同一批翻译任务内，相同原文只调用一次模型并复用译法，多文件与表格内术语更易保持一致。")

    st.markdown("---")
    st.subheader("🛠️ 翻译校正（已翻译文件）")
    st.caption("自动修复：同词异译、截断词、数值表达模式（如 10-2 → 10^-2）等问题。")
    st.caption(
        "手工替换（可选）：每行「错误译文|正确译文|中文词条」；第三列可空则只替换、不写知识库。"
        "示例：`through|pass|通过`。校正语言为英文/德文时，替换词会按原文词的大小写自动调整（如 Through→Pass、THROUGH→PASS）。"
    )
    corr_manual = st.text_area(
        "手工替换与词条",
        height=120,
        placeholder="through|pass|通过",
        key="corr_manual_rules",
    )
    corr_save_glossary = st.checkbox(
        "将含中文词条的行写入当前知识库「词条」分类",
        value=True,
        key="corr_save_glossary",
    )
    corr_input_mode = _st_radio_compat("校正输入方式", ["上传文件", "本地路径"], key="corr_input_mode")
    corr_input_path = None
    corr_uploaded = []
    if corr_input_mode == "上传文件":
        corr_uploaded = st.file_uploader(
            "选择已翻译文件或 .zip",
            type=["docx", "txt", "xlsx", "zip"],
            accept_multiple_files=True,
            key="corr_upload",
        )
    else:
        corr_input_path = st.text_input(
            "校正文件/目录路径",
            placeholder="例如：C:\\docs\\translated.docx 或 D:\\translated_docs",
            key="corr_path",
        ).strip()
    corr_output_dir = st.text_input(
        "校正输出目录（可选，留空用默认）",
        placeholder="留空使用默认",
        key="corr_output_dir",
    ).strip() or None
    corr_lang = st.selectbox("校正语言", ["英文", "德文", "中文"], index=0, key="corr_lang")
    corr_lang_map = {"英文": "en", "德文": "de", "中文": "zh"}
    corr_use_kb = st.checkbox("校正补全时使用知识库（建议开启）", value=True, key="corr_use_kb")
    if st.button("开始校正", key="translation_correct_run"):
        from src.translation.correction import save_glossary_correction_entries
        from src.translation.pipeline import correct_path
        from src.translation.models import SUPPORTED_EXTENSIONS

        manual_rules = _parse_translation_correction_manual_rules(corr_manual)
        try:
            if corr_input_mode == "上传文件":
                import tempfile

                root = Path(tempfile.mkdtemp(prefix="aicheckword_correct_"))
                to_process = []
                for uf in corr_uploaded:
                    ext = Path(uf.name).suffix.lower()
                    if ext == ".zip":
                        zpath = root / uf.name
                        zpath.write_bytes(uf.getvalue())
                        to_process.append(str(zpath))
                    elif ext in SUPPORTED_EXTENSIONS:
                        fpath = root / uf.name
                        fpath.write_bytes(uf.getvalue())
                        to_process.append(str(fpath))
                if not to_process:
                    st.warning("请先上传要校正的文件。")
                else:
                    out_paths, summary = [], {}
                    with st.spinner("⏳ 正在校正，请稍候…"):
                        for inp in to_process:
                            outs, s = correct_path(
                                inp,
                                output_dir=(corr_output_dir or str(root / "out")),
                                target_lang=corr_lang_map[corr_lang],
                                collection_name=(collection if corr_use_kb else None),
                                use_kb=corr_use_kb,
                                provider=st.session_state.get("current_provider"),
                                manual_rules=manual_rules or None,
                            )
                            out_paths.extend(outs)
                            for k, v in (s or {}).items():
                                summary[k] = summary.get(k, 0) + (v or 0)
                    if out_paths:
                        st.success(f"✅ 校正完成，共 **{len(out_paths)}** 个文件。")
                        st.caption(
                            f"修复统计：改动块 {summary.get('changed_blocks', 0)}；"
                            f"手工替换命中 {summary.get('manual_replaced', 0)}；"
                            f"同词统一 {summary.get('term_unified', 0)}；"
                            f"截断词修复 {summary.get('truncation_fixed', 0)}；"
                            f"数值表达修复 {summary.get('numeric_fixed', 0)}"
                        )
                        if corr_save_glossary and manual_rules:
                            zh_to_r = {}
                            for _w, r, zh in manual_rules:
                                z = (zh or "").strip()
                                if z:
                                    zh_to_r[z] = (r or "").strip()
                            gel = [(z, t) for z, t in zh_to_r.items() if t]
                            if gel:
                                try:
                                    n = save_glossary_correction_entries(
                                        collection, gel, corr_lang_map[corr_lang]
                                    )
                                    st.caption(f"词条知识库已写入 **{n}** 条（中文→目标语译法）。")
                                except Exception as ex:
                                    st.warning(f"写入词条知识库失败：{ex}")
                        for p in out_paths:
                            p_path = Path(p)
                            if p_path.is_file():
                                st.download_button(
                                    f"📥 下载校正结果 {p_path.name}",
                                    data=p_path.read_bytes(),
                                    file_name=p_path.name,
                                    mime="application/octet-stream",
                                    key=f"dl_corr_{p_path.name}_{id(p)}",
                                )
            else:
                if not corr_input_path:
                    st.warning("请填写校正路径。")
                else:
                    with st.spinner("⏳ 正在校正，请稍候…"):
                        out_paths, summary = correct_path(
                            corr_input_path,
                            output_dir=corr_output_dir,
                            target_lang=corr_lang_map[corr_lang],
                            collection_name=(collection if corr_use_kb else None),
                            use_kb=corr_use_kb,
                            provider=st.session_state.get("current_provider"),
                            manual_rules=manual_rules or None,
                        )
                    if out_paths:
                        st.success(f"✅ 校正完成，共 **{len(out_paths)}** 个文件。")
                        st.caption(
                            f"修复统计：改动块 {summary.get('changed_blocks', 0)}；"
                            f"手工替换命中 {summary.get('manual_replaced', 0)}；"
                            f"同词统一 {summary.get('term_unified', 0)}；"
                            f"截断词修复 {summary.get('truncation_fixed', 0)}；"
                            f"数值表达修复 {summary.get('numeric_fixed', 0)}"
                        )
                        if corr_save_glossary and manual_rules:
                            zh_to_r = {}
                            for _w, r, zh in manual_rules:
                                z = (zh or "").strip()
                                if z:
                                    zh_to_r[z] = (r or "").strip()
                            gel = [(z, t) for z, t in zh_to_r.items() if t]
                            if gel:
                                try:
                                    n = save_glossary_correction_entries(
                                        collection, gel, corr_lang_map[corr_lang]
                                    )
                                    st.caption(f"词条知识库已写入 **{n}** 条（中文→目标语译法）。")
                                except Exception as ex:
                                    st.warning(f"写入词条知识库失败：{ex}")
                        for p in out_paths:
                            st.code(p)
                    else:
                        st.warning("未找到可校正文件（支持 .docx / .txt / .xlsx）。")
        except Exception as e:
            st.error(f"❌ 校正失败：{e}")

    st.markdown("---")
    st.caption(
        "同一页里的两步：**上面**改已译稿，**下面**把待译原稿交给「开始翻译」生成新译文文件（不覆盖原稿）。"
    )

    input_mode = _st_radio_compat("待译原稿 — 输入方式", ["上传文件", "本地路径"], key="translation_input_mode")
    input_path = None
    uploaded_files = []

    if input_mode == "上传文件":
        uploaded_files = st.file_uploader(
            "选择文件或 .zip",
            type=["docx", "txt", "xlsx", "zip"],
            accept_multiple_files=True,
            key="translation_upload",
        )
        if not uploaded_files:
            st.info("请上传至少一个 .docx / .txt / .xlsx 或一个 .zip；若仅校正已译稿，可只使用上方「翻译校正」区。")
    else:
        input_path = st.text_input(
            "输入文件或文件夹路径",
            placeholder="例如：C:\\docs\\file.docx 或 D:\\project\\docs",
            key="translation_path",
        )
        if not (input_path and input_path.strip()):
            st.info("请填写要翻译的文件或文件夹路径；若仅校正已译稿，可只使用上方「翻译校正」区。")
            input_path = None
        else:
            input_path = input_path.strip()

    output_dir = st.text_input(
        "输出目录（可选，留空则单文件为同目录、目录/zip 为 xxx_translated）",
        placeholder="留空使用默认",
        key="translation_output_dir",
    )
    output_dir = output_dir.strip() or None

    if st.button("开始翻译", key="translation_run"):
        from src.translation.pipeline import translate_path
        from src.translation.models import SUPPORTED_EXTENSIONS

        try:
            if input_mode == "上传文件":
                import tempfile
                import shutil
                root = Path(tempfile.mkdtemp(prefix="aicheckword_translate_"))
                try:
                    to_process = []
                    for uf in uploaded_files or []:
                        ext = Path(uf.name).suffix.lower()
                        if ext == ".zip":
                            zpath = root / uf.name
                            zpath.write_bytes(uf.getvalue())
                            to_process.append(str(zpath))
                        elif ext in SUPPORTED_EXTENSIONS:
                            fpath = root / uf.name
                            fpath.write_bytes(uf.getvalue())
                            to_process.append(str(fpath))
                    if not to_process:
                        st.error("未包含可翻译文件（支持 .docx / .txt / .xlsx）。请先上传源稿，或改用本地路径。")
                    else:
                        out_paths = []
                        out_dir = output_dir or str(root / "out")
                        with st.spinner("⏳ 正在翻译，请勿关闭或刷新页面…"):
                            for idx, inp in enumerate(to_process):
                                if len(to_process) > 1:
                                    st.caption(f"正在处理第 {idx + 1}/{len(to_process)} 个输入…")
                                out_paths.extend(
                                    translate_path(
                                        inp,
                                        output_dir=out_dir,
                                        collection_name=col_name,
                                        use_kb=use_kb,
                                        provider=st.session_state.get("current_provider"),
                                        target_lang=target_lang,
                                        company_overrides=company_overrides,
                                        kb_query_extra=kb_query_extra or None,
                                    )
                                )
                        if out_paths:
                            st.success(f"✅ 翻译完成，共 **{len(out_paths)}** 个文件。请点击下方按钮下载（刷新页面后需重新翻译）。")
                            if any(str(p).lower().endswith(".docx") for p in out_paths):
                                st.info("📌 Word 文档中的图片、字体与版式已按原稿保留；图片未参与翻译。")
                            for p in out_paths:
                                p_path = Path(p)
                                if p_path.is_file():
                                    data = p_path.read_bytes()
                                    st.download_button(
                                        f"📥 下载 {p_path.name}",
                                        data=data,
                                        file_name=p_path.name,
                                        mime="application/octet-stream",
                                        key=f"dl_trans_{p_path.name}_{id(p)}",
                                    )
                            add_operation_log(
                                op_type=OP_TYPE_TRANSLATION,
                                collection=collection,
                                file_name=f"上传 {len(out_paths)} 个文件",
                                source="上传文件",
                                extra={
                                    "count": len(out_paths),
                                    "out_paths": out_paths[:20],
                                    "use_kb": use_kb,
                                    "project_id": trans_project_id,
                                    "translation_context": "project" if trans_proj_mode.startswith("按项目") else "general",
                                },
                                model_info=get_current_model_info(),
                            )
                        else:
                            st.warning("未生成任何翻译文件，请检查输入。")
                finally:
                    try:
                        shutil.rmtree(root, ignore_errors=True)
                    except Exception:
                        pass
            else:
                if not input_path:
                    st.warning("请填写要翻译的文件或文件夹路径，或改用上传文件。")
                else:
                    with st.spinner("⏳ 正在翻译，请勿关闭或刷新页面…"):
                        out_paths = translate_path(
                            input_path,
                            output_dir=output_dir,
                            collection_name=col_name,
                            use_kb=use_kb,
                            provider=st.session_state.get("current_provider"),
                            target_lang=target_lang,
                            company_overrides=company_overrides,
                            kb_query_extra=kb_query_extra or None,
                        )
                    if out_paths:
                        st.success(f"✅ 翻译完成，共 **{len(out_paths)}** 个文件，已保存到以下路径：")
                        if any(str(p).lower().endswith(".docx") for p in out_paths):
                            st.info("📌 Word 文档中的图片、字体与版式已按原稿保留；图片未参与翻译。")
                        for p in out_paths:
                            st.code(p)
                        add_operation_log(
                            op_type=OP_TYPE_TRANSLATION,
                            collection=collection,
                            file_name=Path(input_path).name if input_path else "",
                            source=input_path or "",
                            extra={
                                "count": len(out_paths),
                                "out_paths": out_paths[:20],
                                "use_kb": use_kb,
                                "project_id": trans_project_id,
                                "translation_context": "project" if trans_proj_mode.startswith("按项目") else "general",
                            },
                            model_info=get_current_model_info(),
                        )
                    else:
                        st.warning("未找到可翻译文件（支持 .docx / .txt / .xlsx）。")
        except FileNotFoundError as e:
            st.error(f"❌ 路径不存在：{e}")
            add_operation_log(
                op_type=OP_TYPE_TRANSLATION_ERROR,
                collection=collection,
                file_name=Path(input_path).name if input_mode != "上传文件" and input_path else "上传",
                source=input_path or "上传文件",
                extra={"error": str(e)},
                model_info=get_current_model_info(),
            )
        except Exception as e:
            import traceback
            st.error(f"❌ 翻译失败：{e}")
            st.code(traceback.format_exc(), language="text")
            add_operation_log(
                op_type=OP_TYPE_TRANSLATION_ERROR,
                collection=collection,
                file_name=Path(input_path).name if input_mode != "上传文件" and input_path else "上传",
                source=input_path or "上传文件",
                extra={"error": str(e), "traceback": traceback.format_exc()},
                model_info=get_current_model_info(),
            )

def _op_type_label(op_type):
    """操作类型中文标签"""
    labels = {
        OP_TYPE_TRAIN_BATCH: "📚 法规训练批次",
        OP_TYPE_TRAIN: "📄 单文件训练",
        "train_error": "❌ 训练失败",
        OP_TYPE_GENERATE_CHECKLIST: "📝 生成审核点",
        OP_TYPE_TRAIN_CHECKLIST: "🚀 审核点训练",
        OP_TYPE_TRAIN_PROJECT: "📁 项目专属训练",
        OP_TYPE_TRAIN_PROJECT_ERROR: "❌ 项目专属训练失败/中断",
        OP_TYPE_REVIEW_BATCH: "🔍 审核批次",
        OP_TYPE_REVIEW: "✅ 单文件审核",
        OP_TYPE_REVIEW_TEXT: "✅ 文本审核",
        "review_error": "❌ 审核失败",
        "review_text_error": "❌ 文本审核失败",
        OP_TYPE_CORRECTION: "✏️ 审核纠正",
        OP_TYPE_TRANSLATION: "🌐 文档翻译",
        OP_TYPE_TRANSLATION_ERROR: "❌ 文档翻译失败",
        # 文档生成（初稿/就地修改）
        "draft_generate_job": "🧾 文档生成任务",
        "draft_generate": "🧾 文档生成",
        "draft_export": "📦 文档导出（同格式/就地修改）",
        # 与 draft_generate 同表存储，extra.post_audit 或 source=post_audit 区分
        "__post_audit_display__": "🩺 审核后修改",
    }
    return labels.get(op_type, op_type)


def render_operations_page():
    """操作记录页面：查看导入/训练/审核等日志"""
    st.header("📋 操作记录")
    st.markdown("查看每次导入、训练、审核的批次汇总与明细，支持按类型和知识库筛选。")

    collection = st.session_state.get("collection_name", "regulations")
    st.subheader("📊 训练统计（按文件类型）")
    try:
        by_cat = _cached_knowledge_stats_by_category(collection)
        total_f = by_cat.get("total_files", 0)
        total_c = by_cat.get("total_chunks", 0)
        bc = by_cat.get("by_category") or {}
        st.markdown(f"**当前知识库「{collection}」** — 共 **{total_f}** 个文件、**{total_c}** 块。")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            r = bc.get("regulation", {})
            st.metric("法规文件", f"{r.get('files', 0)} 文件 / {r.get('chunks', 0)} 块")
        with col2:
            p = bc.get("program", {})
            st.metric("程序文件", f"{p.get('files', 0)} 文件 / {p.get('chunks', 0)} 块")
        with col3:
            pc = bc.get("project_case", {})
            st.metric("项目案例文件", f"{pc.get('files', 0)} 文件 / {pc.get('chunks', 0)} 块")
        with col4:
            gl = bc.get("glossary", {})
            st.metric("词条", f"{gl.get('files', 0)} 文件 / {gl.get('chunks', 0)} 块")
    except Exception as e:
        st.caption(f"统计加载失败：{e}")

    st.markdown("---")

    summary = get_operation_summary()
    c1, c2, c3 = st.columns(3)
    c1.metric("训练批次数", summary["total_train_batches"])
    c2.metric("审核批次数", summary["total_review_batches"])
    c3.metric("今日操作数", summary["today_operations"])

    st.markdown("---")

    col_filter, col_limit = st.columns(2)
    with col_filter:
        op_type_filter = st.selectbox(
            "操作类型",
            [
                "全部",
                "法规训练批次",
                "生成审核点",
                "审核点训练",
                "项目专属训练",
                "项目专属训练失败/中断",
                "审核批次",
                "单文件训练",
                "单文件审核",
                "文本审核",
                "审核纠正",
                "文档翻译",
                "文档翻译失败",
                "文档生成",
                "审核后修改",
                "文档导出",
                "文档生成任务",
                "训练失败",
                "审核失败",
            ],
            key="op_type_filter",
        )
    with col_limit:
        limit = st.selectbox("显示条数", [50, 100, 200, 500], index=1, key="op_limit")

    type_map = {
        "全部": None,
        "法规训练批次": OP_TYPE_TRAIN_BATCH,
        "生成审核点": OP_TYPE_GENERATE_CHECKLIST,
        "审核点训练": OP_TYPE_TRAIN_CHECKLIST,
        "项目专属训练": OP_TYPE_TRAIN_PROJECT,
        "项目专属训练失败/中断": OP_TYPE_TRAIN_PROJECT_ERROR,
        "审核批次": OP_TYPE_REVIEW_BATCH,
        "单文件训练": OP_TYPE_TRAIN,
        "单文件审核": OP_TYPE_REVIEW,
        "文本审核": OP_TYPE_REVIEW_TEXT,
        "审核纠正": OP_TYPE_CORRECTION,
        "文档翻译": OP_TYPE_TRANSLATION,
        "文档翻译失败": OP_TYPE_TRANSLATION_ERROR,
        "文档生成": "draft_generate",
        "审核后修改": "__post_audit__",
        "文档导出": "draft_export",
        "文档生成任务": "draft_generate_job",
        "训练失败": "train_error",
        "审核失败": "review_error",
    }
    op_type = type_map.get(op_type_filter, None)

    only_current = st.checkbox("仅当前知识库", value=False, key="op_only_collection")
    collection_filter = st.session_state.get("collection_name", "regulations") if only_current else None
    st.caption(
        "「审核后修改」与「文档初稿生成」在库中同为 `draft_generate`，通过 `extra.post_audit` / `source=post_audit` 区分；"
        "写入后已自动失效操作记录缓存。"
    )

    if op_type == "__post_audit__":
        _fetch_n = max(int(limit or 50), 50) * 8
        _raw_pa = get_operation_logs(
            op_type="draft_generate", collection=collection_filter, limit=_fetch_n
        )
        logs = [
            r
            for r in (_raw_pa or [])
            if (r.get("extra") or {}).get("post_audit")
            or str(r.get("source") or "").strip() == "post_audit"
        ][: int(limit or 50)]
    else:
        logs = _cached_operation_logs(op_type, collection_filter, limit)

    if not logs:
        st.info("暂无操作记录，完成一次训练或审核后会自动记录。")
        return

    for rec in logs:
        extra = rec.get("extra") or {}
        op_label = _op_type_label(rec["op_type"])

        if rec["op_type"] == OP_TYPE_TRAIN_BATCH:
            cat_lbl = extra.get("category_label", "")
            cat_suffix = f" | {cat_lbl}" if cat_lbl else ""
            title = (
                f"{op_label} | 导入 {extra.get('total_files', 0)} 个文件，"
                f"成功 {extra.get('success_count', 0)}，失败 {extra.get('fail_count', 0)}，"
                f"入库 {extra.get('total_chunks', 0)} 块，耗时 {extra.get('duration_sec', 0):.1f}s{cat_suffix}"
            )
            detail = f"来源：{rec.get('source', '')} | 知识库：{rec.get('collection', '')}"
        elif rec["op_type"] == OP_TYPE_REVIEW_BATCH:
            title = (
                f"{op_label} | 审核 {extra.get('total_files', 0)} 个文件，"
                f"完成 {extra.get('success_count', 0)} 个，共 {extra.get('total_audit_points', 0)} 个审核点，"
                f"耗时 {extra.get('duration_sec', 0):.1f}s"
            )
            detail = f"来源：{rec.get('source', '')} | 知识库：{rec.get('collection', '')}"
            if extra.get("fail_count", 0) > 0 and extra.get("first_error"):
                detail += f" | 失败原因：{extra.get('first_error', '')[:200]}{'…' if len(extra.get('first_error', '')) > 200 else ''}"
        elif rec["op_type"] == OP_TYPE_TRAIN:
            cat_lbl = extra.get("category") and CATEGORY_LABELS.get(extra["category"], extra["category"]) or ""
            title = f"{op_label} | {rec.get('file_name', '')} | 入库 {extra.get('chunks', 0)} 块"
            if cat_lbl:
                title += f" | {cat_lbl}"
            detail = f"知识库：{rec.get('collection', '')} | 耗时 {extra.get('duration_sec', 0):.1f}s"
        elif rec["op_type"] == OP_TYPE_REVIEW:
            title = (
                f"{op_label} | {rec.get('file_name', '')} | "
                f"{extra.get('total_points', 0)} 个审核点, 耗时 {extra.get('duration_sec', 0):.1f}s"
            )
            detail = f"知识库：{rec.get('collection', '')}"
        elif rec["op_type"] == OP_TYPE_REVIEW_TEXT:
            title = (
                f"{op_label} | {rec.get('file_name', '')} | "
                f"{extra.get('total_points', 0)} 个审核点, {extra.get('text_length', 0)} 字"
            )
            detail = f"知识库：{rec.get('collection', '')} | 耗时 {extra.get('duration_sec', 0):.1f}s"
        elif rec["op_type"] == OP_TYPE_GENERATE_CHECKLIST:
            title = (
                f"{op_label} | {rec.get('file_name', '')} | "
                f"{extra.get('total_points', 0)} 个审核点, 耗时 {extra.get('duration_sec', 0):.1f}s"
            )
            detail = f"知识库：{rec.get('collection', '')} | 基础文件：{extra.get('has_base', False)}"
        elif rec["op_type"] == OP_TYPE_TRAIN_CHECKLIST:
            title = (
                f"{op_label} | {rec.get('file_name', '')} | "
                f"{extra.get('total_points', 0)} 个审核点 → {extra.get('chunks', 0)} 块, "
                f"耗时 {extra.get('duration_sec', 0):.1f}s"
            )
            detail = f"知识库：{rec.get('collection', '')} | 清单ID：{extra.get('checklist_id', '')}"
        elif rec["op_type"] == OP_TYPE_TRAIN_PROJECT:
            title = f"{op_label} | {rec.get('file_name', '')} | 项目：{extra.get('project_name', '')}"
            detail = f"知识库：{rec.get('collection', '')} | 项目ID：{extra.get('project_id', '')}"
        elif rec["op_type"] == OP_TYPE_TRAIN_PROJECT_ERROR:
            stage = extra.get("stage", "")
            if stage == "interrupted":
                title = f"{op_label} | 训练中断 | 项目：{extra.get('project_name', '')} | 成功 {extra.get('success_count', 0)}/{extra.get('total', 0)}"
                detail = f"原因：{extra.get('reason', extra.get('error', ''))} | 知识库：{rec.get('collection', '')}"
            else:
                title = f"{op_label} | {rec.get('file_name', '')} | 项目：{extra.get('project_name', '')}"
                detail = f"错误：{extra.get('error', '')} | 知识库：{rec.get('collection', '')} | 当时已成功 {extra.get('success_so_far', 0)} 个"
        elif rec["op_type"] == OP_TYPE_CORRECTION:
            title = f"{op_label} | {rec.get('file_name', '')} | 审核点 #{extra.get('point_index', 0)+1}"
            fed = "已回馈" if extra.get("fed_to_kb") else "未回馈"
            detail = f"知识库：{rec.get('collection', '')} | {fed}知识库"
        elif rec["op_type"] == OP_TYPE_TRANSLATION:
            title = f"{op_label} | {rec.get('file_name', '')} | 共 {extra.get('count', 0)} 个文件"
            detail = f"来源：{rec.get('source', '')} | 知识库：{rec.get('collection', '')}" + (" | 使用知识库参考" if extra.get("use_kb") else "")
        elif rec["op_type"] == OP_TYPE_TRANSLATION_ERROR:
            title = f"{op_label} | {rec.get('file_name', '')}"
            detail = f"错误：{extra.get('error', '')} | 来源：{rec.get('source', '')}"
        elif rec["op_type"] == "draft_generate_job":
            payload = extra.get("payload") or {}
            done = extra.get("done")
            err = (extra.get("error") or "").strip()
            status = "完成" if done else ("失败" if err else "已提交")
            base_case_id = payload.get("base_case_id", "")
            inplace = payload.get("inplace_patch")
            inplace_lbl = "就地修改" if inplace else "非就地"
            _gfc = extra.get("generated_file_count")
            _gfc_s = f" | 生成文件数={_gfc}" if _gfc is not None else ""
            title = f"{op_label} | {status} | base_case_id={base_case_id} | {inplace_lbl}{_gfc_s}"
            if err:
                detail = f"错误：{err[:200]}{'…' if len(err) > 200 else ''}"
            else:
                detail = f"job_id={extra.get('job_id','')} | started_at={extra.get('started_at','')}"
                if _gfc is not None:
                    detail += f" | 生成文件数={_gfc}"
        elif rec["op_type"] == "draft_generate":
            fns = extra.get("generated_file_names") or list((extra.get("out_files_by_target") or {}).keys())
            cnt = len([x for x in (fns or []) if str(x).strip()])
            project_id = extra.get("project_id", "")
            base_case_id = extra.get("base_case_id", "")
            inplace_lbl = "就地修改" if extra.get("inplace_patch") else "非就地"
            _is_post_audit = bool(extra.get("post_audit")) or str(rec.get("source") or "").strip() == "post_audit"
            _lbl = _op_type_label("__post_audit_display__") if _is_post_audit else op_label
            _sar = int(extra.get("source_audit_report_id") or 0)
            _sar_part = f" | 来源审核报告 id={_sar}" if _sar else ""
            title = f"{_lbl} | 生成 {cnt} 份 | project_id={project_id} | base_case_id={base_case_id} | {inplace_lbl}{_sar_part}"
            _sum = (extra.get("summary") or "").strip()
            detail = _sum if _sum else f"来源：{rec.get('source','')} | 知识库：{rec.get('collection','')}"
        elif rec["op_type"] == "draft_export":
            pc = (extra.get("patch_counts") or {}) if isinstance(extra.get("patch_counts"), dict) else {}
            applied = pc.get("applied")
            changes = pc.get("changes")
            skipped = pc.get("skipped")
            errors = pc.get("errors")
            suffix = extra.get("suffix", "")
            title = f"{op_label} | {rec.get('file_name','')} | {suffix}"
            parts = []
            if applied is not None:
                parts.append(f"applied={applied}")
            if changes is not None:
                parts.append(f"changes={changes}")
            if skipped is not None:
                parts.append(f"skipped={skipped}")
            if errors is not None:
                parts.append(f"errors={errors}")
            detail = (" | ".join(parts)) if parts else f"来源：{rec.get('source','')}"
        elif rec["op_type"] in ("train_error", "review_error", "review_text_error"):
            title = f"{op_label} | {rec.get('file_name', '')}"
            detail = f"错误：{extra.get('error', '')} | 知识库：{rec.get('collection', '')}"
        else:
            title = f"{op_label} | {rec.get('file_name', '')}"
            detail = rec.get("source", "")

        model_info = rec.get("model_info") or ""
        if model_info:
            detail = detail + " | **模型：** " + model_info

        with st.expander(f"**{rec.get('created_at', '')}** — {title}", expanded=False):
            st.caption(detail)
            if rec["op_type"] in ("train_error", "review_error", "review_text_error", OP_TYPE_TRAIN_PROJECT_ERROR, OP_TYPE_TRANSLATION_ERROR) and extra.get("traceback"):
                st.markdown("**堆栈日志：**")
                st.code(extra.get("traceback", ""), language="text")
            if extra:
                st.json(extra)

    st.markdown("---")
    st.caption("仅展示批次汇总与单条明细，按时间倒序。")


def _get_prompt_default(key: str) -> str:
    """按功能模块 key 返回内置默认提示词（用于展示与恢复默认）。"""
    if key == "checklist_generate_prompt":
        from src.core.checklist_generator import GENERATE_CHECKLIST_PROMPT
        return GENERATE_CHECKLIST_PROMPT
    if key == "checklist_optimize_prompt":
        from src.core.checklist_generator import OPTIMIZE_CHECKLIST_PROMPT
        return OPTIMIZE_CHECKLIST_PROMPT
    if key == "review_system_prompt":
        from src.core.reviewer import REVIEW_SYSTEM_PROMPT
        return REVIEW_SYSTEM_PROMPT
    if key == "review_user_prompt":
        from src.core.reviewer import REVIEW_USER_PROMPT
        return REVIEW_USER_PROMPT
    if key == "review_extra_instructions":
        return ""
    if key == "project_basic_info_prompt":
        return """从以下项目资料（如技术要求、说明书等）中，提取用于审核时与待审文档做一致性核对的基本信息。

## 项目资料摘要

{text}

## 要求

请仅输出以下项目的关键信息，每行一项，格式示例：
项目名称：xxx
产品名称：xxx
型号规格：xxx
注册单元名称：xxx
（若某类信息在资料中未出现可省略该行）

不要输出其他说明，仅输出上述格式的若干行。"""
    if key == "review_summary_prompt":
        from src.core.reviewer import SUMMARY_PROMPT
        return SUMMARY_PROMPT
    return ""


def render_prompts_page():
    """提示词配置页：分功能模块展示并支持配置、入库各模块传给 AI 的提示词。"""
    st.subheader("📝 提示词配置")
    st.caption("以下为系统中各功能模块调用 AI 时使用的提示词，可按模块编辑并保存到数据库；留空则使用内置默认。")

    # 分模块：标题 -> [(key, label, placeholders_hint), ...]
    modules = [
        (
            "生成审核点",
            [
                ("checklist_generate_prompt", "生成审核点（全新生成）", "占位符：{context}、{base_checklist_section}"),
                ("checklist_optimize_prompt", "优化审核点（在已有清单基础上优化）", "占位符：{context}、{base_checklist}"),
            ],
        ),
        (
            "文档审核",
            [
                ("review_system_prompt", "审核系统提示词（角色与能力）", "无占位符"),
                ("review_user_prompt", "审核用户提示词模板", "占位符：{context}、{file_name}、{document_content}"),
                ("review_extra_instructions", "自定义审核要求（追加到审核上下文）", "无占位符，自由文本"),
            ],
        ),
        (
            "项目基本信息提取",
            [
                ("project_basic_info_prompt", "从项目资料中提取基本信息", "占位符：{text}（项目资料摘要）"),
            ],
        ),
        (
            "审核总结",
            [
                ("review_summary_prompt", "审核报告总结", "占位符：{file_name}、{high}、{medium}、{low}、{info}、{details}"),
            ],
        ),
    ]

    for module_title, items in modules:
        with st.expander(f"**{module_title}**", expanded=True):
            for key, label, placeholders_hint in items:
                default_text = _get_prompt_default(key)
                current = get_prompt_by_key(key)
                display_value = current if current else ""
                status = "已配置" if current else "使用内置默认"
                st.caption(f"**{label}** — {placeholders_hint} · 当前：{status}")
                content = st.text_area(
                    label,
                    value=display_value,
                    height=min(400, 180 + (len(display_value or default_text) // 100) * 20),
                    placeholder="留空则使用内置默认" if default_text else "可选配置",
                    key=f"prompt_edit_{key}",
                )
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("保存到数据库", key=f"prompt_save_{key}"):
                        try:
                            update_prompt_by_key(key, content.strip() if content else None)
                            st.success("已保存，该模块将使用当前配置。")
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"保存失败：{e}")
                with col2:
                    if st.button("恢复默认", key=f"prompt_reset_{key}"):
                        try:
                            update_prompt_by_key(key, None)
                            st.success("已恢复为内置默认。")
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"恢复失败：{e}")
                st.markdown("---")


_STREAMLIT_RECOMMENDED = (1, 28, 0)


def _parse_streamlit_version_tuple() -> tuple:
    import re

    v = getattr(st, "__version__", "") or ""
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)", str(v).strip())
    if not m:
        return (0, 0, 0)
    return tuple(int(x) for x in m.groups())


def _maybe_warn_streamlit_version() -> None:
    """requirements.txt 要求 streamlit>=1.28；旧版仍能跑但功能/前端问题多，启动时提示一次。"""
    if st.session_state.get("_streamlit_version_warn_done"):
        return
    st.session_state["_streamlit_version_warn_done"] = True
    if _parse_streamlit_version_tuple() >= _STREAMLIT_RECOMMENDED:
        return
    ver = getattr(st, "__version__", "?")
    st.warning(
        f"当前环境 Streamlit **{ver}** 低于建议的 **{_STREAMLIT_RECOMMENDED[0]}.{_STREAMLIT_RECOMMENDED[1]}.{_STREAMLIT_RECOMMENDED[2]}**。"
        " 可能出现：无原生标签页、无 `st.data_editor`、控制台与 resize/滚动相关的前端报错等。"
        " **升级**：在可联网且 Python 足够新（建议 **3.10+**）的环境中执行 `pip install -U streamlit`。"
        " 若出现 **No matching distribution**，多为 **Python 版本过低** 或 **内网 PyPI 未同步**；请升级 Python 或换官方源/完整镜像。"
        " `requirements.txt` 中 Streamlit 未锁上限，以便旧环境仍能 `pip install -r requirements.txt`。"
    )


def main():
    st.set_page_config(
        page_title="注册文档审核工具",
        page_icon="📋",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _maybe_warn_streamlit_version()

    # 在绑定 main_function_nav_page 的 st.radio 之前应用跨页跳转（否则会 StreamlitAPIException）
    _pend_nav = st.session_state.pop("_pending_main_function_nav_page", None)
    if _pend_nav:
        st.session_state["main_function_nav_page"] = _pend_nav

    render_sidebar()

    st.title("📋 注册文档审核工具")

    page = _st_main_page_nav(
        [
            "① 法规训练 & 生成审核点",
            "② 审核点管理 & 训练",
            "✍️ 文档初稿生成",
            "③ 文档审核",
            "🌐 文档翻译",
            "📝 提示词配置",
            "🔎 知识库查询",
            "📋 操作记录",
            "⚙️ 系统配置",
        ],
        state_key="main_function_nav_page",
        title="选择功能",
    )

    # 仅在切换顶部功能页时短暂显示蒙层；避免编辑/纠正审核点时因任意控件重跑而反复出现「加载中」
    _prev_nav = st.session_state.get("_main_nav_page")
    if _prev_nav is not None and _prev_nav != page:
        _render_loading_overlay()
    st.session_state["_main_nav_page"] = page

    # 离开系统配置页后清除标记，下次进入时从 settings（含侧栏/库已加载值）重新灌入表单
    if page != "⚙️ 系统配置":
        st.session_state["_sys_cfg_in_page"] = False

    if page == "① 法规训练 & 生成审核点":
        render_step1_page()
    elif page == "② 审核点管理 & 训练":
        render_step2_page()
    elif page == "③ 文档审核":
        render_step3_page()
    elif page == "✍️ 文档初稿生成":
        render_draft_page()
    elif page == "🌐 文档翻译":
        render_translation_page()
    elif page == "📝 提示词配置":
        render_prompts_page()
    elif page == "🔎 知识库查询":
        render_knowledge_page()
    elif page == "📋 操作记录":
        render_operations_page()
    elif page == "⚙️ 系统配置":
        render_system_config_page()


if __name__ == "__main__":
    main()
