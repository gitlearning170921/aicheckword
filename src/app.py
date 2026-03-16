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
import time
import tempfile
from datetime import datetime
from typing import Optional
import shutil
import traceback
import inspect
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

from config import settings
from src.core.agent import ReviewAgent
from src.core.document_loader import (
    load_single_file,
    split_documents,
    LOADER_MAP,
    is_archive,
    extract_archive,
    is_deprecated_path,
    extract_section_outline_from_texts,
)
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
    save_audit_correction,
    get_knowledge_stats,
    get_checkpoint_stats,
    get_knowledge_stats_by_category,
    clear_knowledge_docs,
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
    create_project_case,
    list_project_cases,
    get_project_case_file_names,
    delete_project_case,
    get_project_case,
    update_project_case,
    update_knowledge_docs_case_id,
    get_knowledge_docs_by_case_id,
    get_review_extra_instructions,
    update_review_extra_instructions,
    get_review_system_prompt,
    get_review_user_prompt,
    update_review_prompts,
    get_prompt_by_key,
    update_prompt_by_key,
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
)
from src.core.report_export import report_to_html, report_to_docx, report_to_pdf, report_to_excel, report_to_docx_with_comments, report_todo_to_csv, report_todo_to_pdf, report_todo_to_docx, report_todo_to_excel


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


@_make_ttl_cache(ttl_sec=5)
def _cached_knowledge_stats_by_category(collection: str):
    return get_knowledge_stats_by_category(collection)


@_make_ttl_cache(ttl_sec=8)
def _cached_operation_logs(op_type, collection_filter, limit: int):
    return get_operation_logs(op_type=op_type, collection=collection_filter, limit=limit)


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


def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.title("⚙️ 设置")

        # 首次从 DB 载入配置到 settings（仅一次）
        if not st.session_state.get("db_settings_loaded"):
            db_conf = load_app_settings()
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
            st.session_state["db_settings_loaded"] = True
            st.session_state["current_provider"] = (settings.provider or "ollama").strip().lower()

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

        # 下拉刚切换时立即 rerun，使当前 run 的 index 与选中一致，避免需点两次才看到切换
        if _cur != _provider:
            _rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
            if callable(_rerun):
                _rerun()

        if is_cursor:
            cursor_api_key = st.text_input(
                "Cursor API Key",
                value=settings.cursor_api_key,
                type="password",
                help="Cursor Dashboard → Integrations 创建",
            )
            cursor_repo = st.text_input(
                "GitHub 仓库地址",
                value=settings.cursor_repository,
                help="必填，如 https://github.com/your-org/your-repo（Agent 会基于该仓库运行）",
            )
            cursor_ref = st.text_input(
                "分支/标签",
                value=settings.cursor_ref,
                help="默认 main",
            )
            cursor_embed = st.selectbox(
                "向量化使用",
                ["ollama", "openai"],
                index=0 if (settings.cursor_embedding or "ollama").lower() == "ollama" else 1,
                help="知识库向量化仍需要 Ollama 或 OpenAI",
            )
            llm_model = settings.llm_model
            embed_model = settings.embedding_model
        elif is_ollama:
            ollama_url = st.text_input(
                "Ollama 地址",
                value=settings.ollama_base_url,
                help="默认 http://localhost:11434，通常不用改",
            )
            llm_model = st.text_input(
                "审核模型",
                value=settings.llm_model,
                help="推荐 qwen2.5（中文好）、llama3.1、mistral 等",
            )
            embed_model = st.text_input(
                "向量化模型",
                value=settings.embedding_model,
                help="推荐 nomic-embed-text、bge-m3 等",
            )
        elif is_openai_form:
            _default_base = settings.openai_base_url
            if _provider == "deepseek":
                _default_base = settings.deepseek_base_url or "https://api.deepseek.com/v1"
            elif _provider == "lingyi":
                _default_base = settings.lingyi_base_url or "https://api.lingyiwanwu.com/v1"
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
                value=_default_base,
                key=f"sidebar_base_url_{_provider}",
                help="DeepSeek 填 https://api.deepseek.com/v1（404 时请确认无误）；零一默认 https://api.lingyiwanwu.com/v1",
            )
            _model_help = "DeepSeek 填 deepseek-chat 或 deepseek-chat-v2 等（404 多为模型名或 Base URL 错误）；零一如 yi-large 等"
            llm_model = st.text_input(
                "审核模型",
                value=settings.llm_model or ("deepseek-chat" if _provider == "deepseek" else "yi-large"),
                key=f"sidebar_llm_model_{_provider}",
                help=_model_help,
            )
            _embed_help = "向量化仍用 Ollama 或 OpenAI；选 Cursor 时可在 Cursor 区块选择。"
            if _provider == "deepseek":
                _embed_help = "DeepSeek 不提供 embeddings 接口，向量化将使用 **Ollama**；请确保 Ollama 已启动并在此填写向量化模型（如 nomic-embed-text）。"
            elif _provider == "lingyi":
                _embed_help = "零一万物向量化将使用 **Ollama**；请确保 Ollama 已启动并在此填写向量化模型（如 nomic-embed-text）。"
            embed_model = st.text_input(
                "向量化模型",
                value=settings.embedding_model,
                key=f"sidebar_embed_model_{_provider}",
                help=_embed_help,
            )
        elif _provider == "gemini":
            st.caption("密钥请配置环境变量 GEMINI_API_KEY 或 GOOGLE_API_KEY（.env）。")
            llm_model = st.text_input("审核模型", value=settings.llm_model or "gemini-1.5-flash", help="如 gemini-1.5-flash、gemini-1.5-pro")
            embed_model = st.text_input("向量化模型", value=settings.embedding_model, help="向量建议仍用 Ollama 或 OpenAI Embedding")
        elif _provider == "tongyi":
            st.caption("密钥请配置 DASHSCOPE_API_KEY（.env 或系统环境变量）。")
            llm_model = st.text_input("审核模型", value=settings.llm_model or "qwen-plus", help="如 qwen-plus、qwen-max")
            embed_model = st.text_input("向量化模型", value=settings.embedding_model, help="通义向量可用 text-embedding-v3，需单独接 Embedding API；当前可先 Ollama")
        elif _provider == "baidu":
            st.caption("密钥请配置 QIANFAN_AK、QIANFAN_SK（.env）。")
            llm_model = st.text_input("审核模型", value=settings.llm_model or "ERNIE-Bot-4", help="千帆模型名")
            embed_model = st.text_input("向量化模型", value=settings.embedding_model, help="向量可继续用 Ollama/OpenAI")

        if st.button("💾 保存配置"):
            from config.cursor_overrides import _cursor_overrides as _co
            settings.provider = _provider
            settings.llm_model = llm_model
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
                if _provider == "deepseek":
                    settings.deepseek_api_key = api_key
                    settings.deepseek_base_url = base_url
                    settings.openai_api_key = api_key
                    settings.openai_base_url = base_url
                elif _provider == "lingyi":
                    settings.lingyi_api_key = api_key
                    settings.lingyi_base_url = base_url
                    settings.openai_api_key = api_key
                    settings.openai_base_url = base_url
                else:
                    settings.openai_api_key = api_key
                    settings.openai_base_url = base_url
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
            if "agent" in st.session_state:
                st.session_state.agent.reset_clients()
            st.success("配置已保存，下次打开自动生效。")

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
            by_cat = _cached_knowledge_stats_by_category(collection)
        except Exception:
            reg_stats = {}
            cp_stats = {}
            by_cat = {}

        st.caption("法规知识库（第一步）— 以数据库为准")
        st.metric("法规向量块数", reg_stats.get("total_chunks", 0))

        st.caption("审核点知识库（第二步）— 以数据库为准")
        st.metric("审核点向量块数", cp_stats.get("total_chunks", 0))

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


def _train_single_file(agent, file_path, file_name, status_box, embed_bar, log_lines, category: str = "regulation", case_id=None):
    """
    训练单个文件。通过 st.empty() 实时刷新状态。
    log_lines: list, 累积日志，每次追加后刷新显示。
    返回 (成功?, 块数, 耗时)
    """
    t0 = time.time()

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

    embed_bar.progress(0)

    def on_batch_done(done, total):
        pct = int(done / total * 100)
        embed_bar.progress(pct)
        status_box.info(
            f"🔄 [{file_name}] 向量化 {done}/{total} 块 ({pct}%)"
        )

    try:
        with st.spinner(f"[{file_name}] 正在向量化 {len(chunks)} 块..."):
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
    embed_bar.progress(100)
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

    tab1, tab2, tab3 = st.tabs(["📤 上传文件训练", "📂 从目录训练", "📝 生成审核点"])

    with tab1:
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
                    idx = case_options.index(sel_case)
                    case = cases[idx - 1]
                    _lang_label = DOC_LANG_VALUE_TO_LABEL.get(case.get("document_language") or "", "不指定")
                    st.caption(
                        f"将训练到案例：**{case.get('case_name')}**"
                        f"（文档语言：{_lang_label}；产品名称：{case.get('product_name') or '—'}，"
                        f"注册国家：{case.get('registration_country')}，"
                        f"注册类别：{case.get('registration_type')}）"
                    )
                    with st.expander("✏️ 编辑此案例（中/英文）", expanded=False):
                        e_case_name = st.text_input("案例名称", value=case.get("case_name") or "", key="edit_case_name")
                        e_case_name_en = st.text_input("案例名称（英文）", value=case.get("case_name_en") or "", key="edit_case_name_en")
                        e_product = st.text_input("产品名称", value=case.get("product_name") or "", key="edit_case_product")
                        e_product_en = st.text_input("产品名称（英文）", value=case.get("product_name_en") or "", key="edit_case_product_en")
                        e_country = st.text_input("注册国家", value=case.get("registration_country") or "", key="edit_case_country")
                        e_country_en = st.text_input("注册国家（英文）", value=case.get("registration_country_en") or "", key="edit_case_country_en")
                        _doc_lang_val = case.get("document_language") or ""
                        _doc_lang_label = DOC_LANG_VALUE_TO_LABEL.get(_doc_lang_val, "不指定")
                        _doc_lang_idx = DOC_LANG_OPTIONS.index(_doc_lang_label) if _doc_lang_label in DOC_LANG_OPTIONS else 0
                        e_doc_lang = st.selectbox("案例文档语言", DOC_LANG_OPTIONS, index=_doc_lang_idx, key="edit_case_doc_lang")
                        e_scope = st.text_area("产品适用范围", value=case.get("scope_of_application") or "", height=60, key="edit_case_scope")
                        other_cases_upload = [c for c in cases if c.get("id") != case.get("id")]
                        link_options_upload = ["不关联（独立）"] + [_format_case_option(c) for c in other_cases_upload]
                        _link_idx_upload = 0
                        if case.get("project_key"):
                            for i, c in enumerate(other_cases_upload):
                                if str(c.get("id")) == str(case.get("project_key")) or (c.get("project_key") and str(c.get("project_key")) == str(case.get("project_key"))):
                                    _link_idx_upload = i + 1
                                    break
                        e_link_upload = st.selectbox("关联到同一项目", link_options_upload, index=min(_link_idx_upload, len(link_options_upload) - 1), key="edit_case_link_upload", help="与所选案例归为同一项目（多国家/多语言版本）；不关联则本案例独立。")
                        if st.button("保存案例", key="edit_case_save"):
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
                        st.session_state["train_upload_case_sel"] = "➕ 新建案例"
                        st.session_state["train_copy_from_case_id"] = case["id"]
                        st.session_state["train_copy_from_sel"] = _format_case_option(case)
                        _rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
                        if callable(_rerun):
                            _rerun()
                    if st.button("🗑️ 删除此案例", key="del_case_upload"):
                        if _case_files:
                            st.warning("该案例下已有入库文件，不能删除。请先删除关联文件后再试。")
                        else:
                            delete_project_case(case["id"])
                            st.success("已删除案例")
                            st.experimental_rerun()
                except Exception:
                    pass
        uploaded_files = st.file_uploader(
            "选择训练文件（支持单个文档或 .zip / .tar / .tar.gz 压缩包，压缩包将自动解压后导入）",
            type=["pdf", "docx", "doc", "xlsx", "xls", "txt", "md", "zip", "tar", "gz", "tgz"],
            accept_multiple_files=True,
            key="train_uploader",
            help="PDF 仅支持文本型（可复制文字）；扫描件/纯图片 PDF 无法提取文字会报错或卡住，建议先 OCR 或换用文本型 PDF。单文件最多处理前 500 页。",
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
            embed_bar = st.empty()
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

                ok, chunks, elapsed = _train_single_file(agent, path, display_name, status_box, embed_bar, log_lines, category=category, case_id=st.session_state.get("train_queue_case_id"))
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

    with tab2:
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
            embed_bar = st.empty()
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
                ok, chunks, elapsed = _train_single_file(agent, path, display_name, status_box, embed_bar, log_lines, category=category, case_id=train_queue_case_id_dir)
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
                    idx_dir = case_options_dir.index(sel_case_dir)
                    case_dir = cases_dir[idx_dir - 1]
                    train_dir_case_id = case_dir["id"]
                    st.caption(
                        f"将训练到案例：**{case_dir.get('case_name')}**"
                        f"（产品名称：{case_dir.get('product_name') or '—'}，"
                        f"注册国家：{case_dir.get('registration_country')}）"
                    )
                    with st.expander("✏️ 编辑此案例 & 关联项目", expanded=False):
                        e_case_name_d = st.text_input("案例名称", value=case_dir.get("case_name") or "", key="edit_dir_case_name")
                        e_case_name_en_d = st.text_input("案例名称（英文）", value=case_dir.get("case_name_en") or "", key="edit_dir_case_name_en")
                        e_product_d = st.text_input("产品名称", value=case_dir.get("product_name") or "", key="edit_dir_case_product")
                        e_product_en_d = st.text_input("产品名称（英文）", value=case_dir.get("product_name_en") or "", key="edit_dir_case_product_en")
                        e_country_d = st.text_input("注册国家", value=case_dir.get("registration_country") or "", key="edit_dir_case_country")
                        e_country_en_d = st.text_input("注册国家（英文）", value=case_dir.get("registration_country_en") or "", key="edit_dir_case_country_en")
                        _doc_lang_d = case_dir.get("document_language") or ""
                        _doc_lang_label_d = DOC_LANG_VALUE_TO_LABEL.get(_doc_lang_d, "不指定")
                        _doc_lang_idx_d = DOC_LANG_OPTIONS.index(_doc_lang_label_d) if _doc_lang_label_d in DOC_LANG_OPTIONS else 0
                        e_doc_lang_d = st.selectbox("案例文档语言", DOC_LANG_OPTIONS, index=_doc_lang_idx_d, key="edit_dir_case_doc_lang")
                        e_scope_d = st.text_area("产品适用范围", value=case_dir.get("scope_of_application") or "", height=60, key="edit_dir_case_scope")
                        other_cases_dir = [c for c in cases_dir if c.get("id") != case_dir.get("id")]
                        link_options_dir = ["不关联（独立）"] + [_format_case_option(c) for c in other_cases_dir]
                        _link_idx_dir = 0
                        if case_dir.get("project_key"):
                            for i, c in enumerate(other_cases_dir):
                                if str(c.get("id")) == str(case_dir.get("project_key")) or (c.get("project_key") and str(c.get("project_key")) == str(case_dir.get("project_key"))):
                                    _link_idx_dir = i + 1
                                    break
                        e_link_dir = st.selectbox("关联到同一项目", link_options_dir, index=min(_link_idx_dir, len(link_options_dir) - 1), key="edit_dir_case_link", help="与所选案例归为同一项目（多国家/多语言版本）。")
                        if st.button("保存案例", key="edit_dir_case_save"):
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
                        st.session_state["train_dir_case_sel"] = "➕ 新建案例"
                        st.session_state["train_dir_copy_from_case_id"] = case_dir["id"]
                        st.session_state["train_dir_copy_from_sel"] = _format_case_option(case_dir)
                        _rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
                        if callable(_rerun):
                            _rerun()
                    if st.button("🗑️ 删除此案例", key="del_case_dir"):
                        if _case_files_dir:
                            st.warning("该案例下已有入库文件，不能删除。请先删除关联文件后再试。")
                        else:
                            delete_project_case(case_dir["id"])
                            st.success("已删除案例")
                            st.experimental_rerun()
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

    with tab3:
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

    tab_manage, tab_import, tab_projects = st.tabs(["📋 审核点清单管理", "📤 导入审核点", "📁 项目与专属资料"])

    with tab_manage:
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

    with tab_import:
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

    with tab_projects:
        _render_projects_tab(agent, collection)


def _render_projects_tab(agent, collection: str):
    """项目列表 + 新建/编辑/删除 + 上传项目专属资料（用缓存避免切换选项时卡顿）"""
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
                    )
                    st.success(f"已创建项目「{p_name}」，ID: {pid}")
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

        with st.expander("✏️ 编辑项目（支持中/英文）", expanded=False):
            with st.form(f"edit_proj_{pid}"):
                e_name = st.text_input("项目名称", value=proj.get("name", ""), key=f"ep_name_{pid}")
                e_name_en = st.text_input("项目名称（英文）", value=proj.get("name_en") or "", placeholder="Project name in English", key=f"ep_name_en_{pid}")
                e_product = st.text_input("产品名称（可选）", value=proj.get("product_name") or "", placeholder="与项目名称一并加入审核点、一致性核对", key=f"ep_product_{pid}")
                e_product_en = st.text_input("产品名称（英文）", value=proj.get("product_name_en") or "", placeholder="Product name in English", key=f"ep_product_en_{pid}")
                e_model = st.text_input("型号（可选，Model）", value=proj.get("model") or "", placeholder="字段名称不区分大小写，取值区分大小写、精确匹配（含空格）", key=f"ep_model_{pid}")
                e_model_en = st.text_input("型号（英文，可选）", value=proj.get("model_en") or "", placeholder="Model in English", key=f"ep_model_en_{pid}")
                e_country = st.selectbox("注册国家", countries, index=countries.index(proj["registration_country"]) if proj.get("registration_country") in countries else 0, key=f"ep_country_{pid}")
                e_country_en = st.text_input("注册国家（英文）", value=proj.get("registration_country_en") or "", placeholder="e.g. China", key=f"ep_country_en_{pid}")
                e_type = st.selectbox("注册类别", REGISTRATION_TYPES, index=REGISTRATION_TYPES.index(proj["registration_type"]) if proj.get("registration_type") in REGISTRATION_TYPES else 0, key=f"ep_type_{pid}")
                e_comp = st.selectbox("注册组成", REGISTRATION_COMPONENTS, index=REGISTRATION_COMPONENTS.index(proj["registration_component"]) if proj.get("registration_component") in REGISTRATION_COMPONENTS else 0, key=f"ep_comp_{pid}")
                e_form = st.selectbox("项目形态", forms, index=forms.index(proj["project_form"]) if proj.get("project_form") in forms else 0, key=f"ep_form_{pid}")
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
                        name_en=e_name_en.strip() if e_name_en else "",
                        product_name_en=e_product_en.strip() if e_product_en else "",
                        registration_country_en=e_country_en.strip() if e_country_en else "",
                        model=e_model.strip() if e_model else "",
                        model_en=e_model_en.strip() if e_model_en else "",
                    )
                    st.success("已更新")
                    st.experimental_rerun()
        if st.button("🗑️ 删除项目", key=f"del_proj_{pid}"):
            try:
                agent.get_project_kb(pid).clear()
            except Exception:
                pass
            delete_project(pid)
            st.success("已删除项目")
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
    review_mode = st.radio("审核模式", ["仅通用审核", "按项目审核"], horizontal=True, key="review_mode")
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

    tab1, tab2 = st.tabs(["📤 上传文件审核", "📝 文本审核"])

    with tab1:
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
                            except Exception:
                                pass
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
        if review_files:
            n_upload = len(review_files)
            st.caption(f"已选择 **{n_upload}** 个文件，点击下方按钮将批量审核。")
            do_multi_doc = st.checkbox(
                "进行多文档一致性与模板风格审核（2 个及以上文件时）",
                value=True,
                key="do_multi_doc_consistency",
                help="完成后将额外生成一份「多文档一致性与模板风格」报告，检查各文档间信息一致性、术语与格式风格是否统一。",
            )
            only_multi_doc = st.checkbox(
                "仅进行多文档一致性与模板风格审核（不跑单文档审核，需 2 个及以上文件）",
                value=False,
                key="only_multi_doc_review",
                help="勾选后点击下方按钮将只执行多文档一致性/模板风格审核，不逐份单文档审核。",
            )
        if review_files and st.button("🔍 开始批量审核" if len(review_files) > 1 else "🔍 开始审核", key="review_btn"):
            items, temp_dirs = _expand_uploads(review_files)

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
                    docs = load_single_file(first_path)
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
            # 按项目审核时自动按项目适用注册类别匹配审核点；通用审核不区分类别、使用全部审核点
            if project_id:
                review_context["_filter_by_registration_type"] = True

            total_files = len(items)
            st.session_state.review_project_id = project_id
            st.session_state.review_context = review_context
            st.session_state._review_mode_snapshot = review_mode
            st.caption("💡 文档较多时可能因超时中断，已完成的报告会保存到「历史报告」；建议每批不超过 10 个文件，可分批上传。")
            review_bar = st.progress(0)
            review_status = st.empty()
            review_status.info(f"准备审核 {total_files} 个文件...")
            review_current = st.empty()
            log_display = st.empty()

            log_lines = []
            all_reports = []
            failed_items = []
            failed_errors = []  # 每个失败文件对应的错误信息，便于日志与界面展示
            t_start = time.time()
            batch_interrupted = False
            batch_error = None
            multi_doc_error_shown = False  # 仅多文档分支内已展示错误时不再用通用“未生成报告”覆盖

            try:
                if st.session_state.get("only_multi_doc_review") and len(items) >= 2:
                    review_status.info("仅进行多文档一致性与模板风格审核…")
                    multi_doc_fn = getattr(agent, "review_multi_document_consistency", None)
                    if callable(multi_doc_fn):
                        try:
                            _ctx = dict(review_context) if review_context else {}
                            _ctx["current_provider"] = st.session_state.get("current_provider") or settings.provider
                            with st.spinner("正在调用多文档一致性审核接口（读取文件并请求 AI）…"):
                                consistency_report = multi_doc_fn(
                                    [(path, display_name) for (path, display_name, _) in items],
                                    project_id=project_id,
                                    review_context=_ctx,
                                )
                            consistency_report["original_filename"] = "多文档一致性与模板风格审核"
                            consistency_report["related_doc_names"] = [display_name for (_, display_name, _) in items]
                            consistency_report["file_name"] = consistency_report.get("file_name") or "多文档一致性与模板风格审核"
                            _inject_review_meta(consistency_report)
                            all_reports.append(consistency_report)
                            st.session_state.review_reports = all_reports
                            try:
                                save_audit_report(
                                    agent.collection_name,
                                    consistency_report,
                                    model_info=get_current_model_info() or "",
                                )
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
                        pct = int(idx / total_files * 100)
                        review_bar.progress(pct)
                        review_status.info(
                            f"审核进度 {idx+1}/{total_files} | "
                            f"已完成 {len(all_reports)} 个 | "
                            f"耗时 {_format_time(time.time() - t_start)}"
                        )
                        review_current.info(f"正在审核 [{display_name}]...")

                        try:
                            t0 = time.time()
                            mi = get_current_model_info()
                            with st.spinner(f"AI 正在审核 [{display_name}]，请耐心等待..."):
                                report = agent.review(
                                    path,
                                    project_id=project_id,
                                    review_context=review_context,
                                    system_prompt=review_sys_edit.strip() or None,
                                    user_prompt=review_usr_edit.strip() or None,
                                    extra_instructions=review_extra_edit.strip() or None,
                                )
                            elapsed = time.time() - t0
                            report["original_filename"] = display_name
                            report["_original_path"] = path
                            _inject_review_meta(report)
                            all_reports.append(report)
                            do_multi_later = len(items) >= 2 and st.session_state.get("do_multi_doc_consistency", True)
                            if not do_multi_later:
                                try:
                                    save_audit_report(agent.collection_name, report, model_info=mi)
                                except Exception:
                                    pass
                            n_points = report.get("total_points", 0)
                            add_operation_log(
                                op_type=OP_TYPE_REVIEW,
                                collection=agent.collection_name,
                                file_name=display_name,
                                source=str(path),
                                extra={"total_points": n_points, "duration_sec": round(elapsed, 2)},
                                model_info=mi,
                            )
                            log_lines.append(
                                f"- :white_check_mark: **{display_name}** — {n_points} 个审核点, {_format_time(elapsed)}"
                            )
                        except Exception as e:
                            err_str = str(e)
                            failed_items.append((path, display_name, is_from_archive))
                            failed_errors.append(err_str)
                            log_lines.append(f"- :x: **{display_name}** 失败：{err_str}")
                            add_operation_log(
                                op_type="review_error",
                                collection=agent.collection_name,
                                file_name=display_name,
                                source=str(path),
                                extra={"error": err_str, "traceback": traceback.format_exc()},
                                model_info=get_current_model_info(),
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
                    },
                    model_info=get_current_model_info(),
                )

                review_bar.progress(100)
                total_time = time.time() - t_start
                review_current.empty()
                review_status.warning(
                    f"审核过程中断：{batch_error}。已完成 {len(all_reports)}/{total_files} 个文件，耗时 {_format_time(total_time)}。请到「历史报告」查看已审核结果，剩余文件可分批重新上传。"
                )
                if all_reports:
                    st.session_state.review_reports = all_reports
                    try:
                        _batch_meta = _build_review_meta_dict()
                        _partial_batch = {
                            "file_name": "批量审核（部分完成）",
                            "original_filename": "批量审核（部分完成）",
                            "batch": True,
                            "reports": list(all_reports),
                            "total_points": sum(r.get("total_points", 0) for r in all_reports),
                            "high_count": sum(r.get("high_count", 0) for r in all_reports),
                            "medium_count": sum(r.get("medium_count", 0) for r in all_reports),
                            "low_count": sum(r.get("low_count", 0) for r in all_reports),
                            "info_count": sum(r.get("info_count", 0) for r in all_reports),
                            "summary": f"本批次中断，已完成 {len(all_reports)}/{total_files} 份报告。",
                            "_review_meta": _batch_meta,
                        }
                        save_audit_report(agent.collection_name, _partial_batch, model_info=get_current_model_info() or "")
                    except Exception as _save_err:
                        st.caption(f"已完成的报告写入历史失败：{_save_err}")
            else:
                review_bar.progress(100)
                total_time = time.time() - t_start
                review_current.empty()
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
                            _inject_review_meta(consistency_report)
                            all_reports.append(consistency_report)
                            try:
                                _batch_meta = _build_review_meta_dict()
                                batch_report = {
                                    "file_name": "批量审核（含多文档一致性）",
                                    "original_filename": "批量审核（含多文档一致性）",
                                    "batch": True,
                                    "reports": list(all_reports),
                                    "total_points": sum(r.get("total_points", 0) for r in all_reports),
                                    "high_count": sum(r.get("high_count", 0) for r in all_reports),
                                    "medium_count": sum(r.get("medium_count", 0) for r in all_reports),
                                    "low_count": sum(r.get("low_count", 0) for r in all_reports),
                                    "info_count": sum(r.get("info_count", 0) for r in all_reports),
                                    "summary": f"本批次共 {len(all_reports)} 份报告（含多文档一致性审核）。",
                                    "_review_meta": _batch_meta,
                                }
                                save_audit_report(
                                    agent.collection_name,
                                    batch_report,
                                    model_info=get_current_model_info() or "",
                                )
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
                                    _batch_meta = _build_review_meta_dict()
                                    _batch_no_consistency = {
                                        "file_name": "批量审核（多文档一致性未完成）",
                                        "original_filename": "批量审核（多文档一致性未完成）",
                                        "batch": True,
                                        "reports": list(all_reports),
                                        "total_points": sum(r.get("total_points", 0) for r in all_reports),
                                        "high_count": sum(r.get("high_count", 0) for r in all_reports),
                                        "medium_count": sum(r.get("medium_count", 0) for r in all_reports),
                                        "low_count": sum(r.get("low_count", 0) for r in all_reports),
                                        "info_count": sum(r.get("info_count", 0) for r in all_reports),
                                        "summary": f"本批次共 {len(all_reports)} 份单文档报告；多文档一致性审核未完成。",
                                        "_review_meta": _batch_meta,
                                    }
                                    save_audit_report(agent.collection_name, _batch_no_consistency, model_info=get_current_model_info() or "")
                                except Exception as _save_err:
                                    st.warning(f"本批报告写入历史失败：{_save_err}")
                    else:
                        log_lines.append("- :x: **多文档一致性与模板风格审核** 当前版本不支持，已跳过。")
                        log_display.markdown("\n".join(log_lines))
                        if all_reports:
                            try:
                                _batch_meta = _build_review_meta_dict()
                                _batch_skip_multi = {
                                    "file_name": "批量审核（含多文档一致性）",
                                    "original_filename": "批量审核（含多文档一致性）",
                                    "batch": True,
                                    "reports": list(all_reports),
                                    "total_points": sum(r.get("total_points", 0) for r in all_reports),
                                    "high_count": sum(r.get("high_count", 0) for r in all_reports),
                                    "medium_count": sum(r.get("medium_count", 0) for r in all_reports),
                                    "low_count": sum(r.get("low_count", 0) for r in all_reports),
                                    "info_count": sum(r.get("info_count", 0) for r in all_reports),
                                    "summary": f"本批次共 {len(all_reports)} 份报告（多文档一致性未执行）。",
                                    "_review_meta": _batch_meta,
                                }
                                save_audit_report(agent.collection_name, _batch_skip_multi, model_info=get_current_model_info() or "")
                            except Exception as _save_err:
                                st.warning(f"本批报告写入历史失败：{_save_err}")

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
                failed_set = {(p, d, a) for (p, d, a) in failed_items}
                for path, _display_name, is_from_archive in items:
                    if not is_from_archive and (path, _display_name, is_from_archive) not in failed_set:
                        Path(path).unlink(missing_ok=True)
                if failed_items:
                    st.session_state.review_failed_items = failed_items
                    st.session_state.review_failed_errors = list(failed_errors)
                    st.session_state.review_failed_temp_dirs = temp_dirs
                    st.session_state.review_success_reports = list(all_reports)
                else:
                    for d in temp_dirs:
                        shutil.rmtree(d, ignore_errors=True)

    if st.session_state.get("review_failed_items"):
        failed_list = st.session_state.review_failed_items
        failed_errs = st.session_state.get("review_failed_errors") or []
        st.warning(f"本批次有 **{len(failed_list)}** 个文件审核失败，可点击下方按钮仅对失败项重新审核（与已成功报告合并为同一批）。")
        if failed_errs:
            with st.expander("查看失败原因（便于排查接口/模型配置）", expanded=True):
                st.caption("首个失败原因：")
                st.code(failed_errs[0], language="text")
                if len(failed_errs) > 1:
                    st.caption("全部失败原因：")
                    for i, err in enumerate(failed_errs, 1):
                        st.text(f"{i}. {err[:500]}{'…' if len(err) > 500 else ''}")
        for _path, _dn, _ in failed_list:
            st.caption(f"• {_dn}")
        if st.button("🔄 重新审核失败项", key="retry_failed_review_btn"):
            agent = init_agent()
            project_id = st.session_state.get("review_project_id")
            review_context = st.session_state.get("review_context")
            review_sys_edit = st.session_state.get("review_sys_edit_step3", "") or st.session_state.get("review_sys_edit", "")
            review_usr_edit = st.session_state.get("review_usr_edit_step3", "") or st.session_state.get("review_usr_edit", "")
            review_extra_edit = st.session_state.get("review_extra_edit_step3", "") or st.session_state.get("review_extra_edit", "")
            merged = list(st.session_state.get("review_success_reports", []))
            for path, display_name, is_from_archive in failed_list:
                try:
                    report = agent.review(
                        path,
                        project_id=project_id,
                        review_context=review_context,
                        system_prompt=review_sys_edit.strip() or None,
                        user_prompt=review_usr_edit.strip() or None,
                        extra_instructions=review_extra_edit.strip() or None,
                    )
                    report["original_filename"] = display_name
                    _inject_review_meta(report)
                    try:
                        save_audit_report(agent.collection_name, report, model_info=get_current_model_info() or "")
                    except Exception:
                        pass
                    merged.append(report)
                except Exception:
                    pass
            st.session_state.review_reports = merged
            _dirs = list(st.session_state.get("review_failed_temp_dirs", []))
            for k in ("review_failed_items", "review_failed_errors", "review_failed_temp_dirs", "review_success_reports"):
                st.session_state.pop(k, None)
            for d in _dirs:
                shutil.rmtree(d, ignore_errors=True)
            st.success("已对失败项重新审核并合并到本批报告。")
            st.experimental_rerun()

    with tab2:
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
                with st.spinner("AI 正在审核文本，请耐心等待..."):
                    report = agent.review_text(
                        review_text,
                        text_file_name,
                        project_id=project_id,
                        review_context=_text_ctx,
                        system_prompt=review_sys_edit.strip() or None,
                        user_prompt=review_usr_edit.strip() or None,
                        extra_instructions=review_extra_edit.strip() or None,
                    )
                elapsed = time.time() - t0
                _inject_review_meta(report)
                try:
                    save_audit_report(agent.collection_name, report, model_info=mi)
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

    with st.expander("📋 整合报告（将同一文件或所选多份报告的问题点合并为一份完整报告）", expanded=False):
        collection = st.session_state.get("collection_name", "regulations")
        st.caption("多次审核同一文件时，各次报告问题点可能不一致。可在此按「文件名」或「勾选多份报告」整合为一份去重后的完整报告。")
        mode = st.radio("整合方式", ["按文件名整合（同一文件的所有历史报告）", "选择多份报告整合"], key="merge_report_mode", horizontal=True)
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
            history = get_audit_reports(collection=collection, limit=100)
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
            _render_multi_doc_report(merged_report, 0, [merged_report], key_prefix="merged")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("保存为历史报告", key="save_merged_report_btn"):
                    try:
                        if not merged_report.get("_review_meta"):
                            _inject_review_meta(merged_report)
                        save_audit_report(collection, merged_report, model_info=get_current_model_info())
                        st.session_state.pop("merged_report_result", None)
                        st.success("已保存，可在上方历史报告中查看。")
                        st.experimental_rerun()
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
        history = get_audit_reports(collection=collection, limit=100)
        if history:
            for rec in history:
                ts = rec.get("created_at", "")
                fname = rec.get("file_name", "")
                tp = rec.get("total_points", 0)
                mid = rec.get("model_info", "")
                rpt = rec.get("report", {})
                # 从报告中提取审核类型标签
                _h_meta = _extract_history_meta(rpt)
                _h_type_tag = f"[{_h_meta.get('audit_type', '通用审核')}] " if _h_meta else ""
                label = f"{ts} | {_h_type_tag}{fname} | {tp}个审核点"
                if mid:
                    label += f" | {mid}"
                with st.expander(label, expanded=False):
                    if rpt:
                        _render_history_meta_header(rpt, rec.get("id", 0))
                        if rpt.get("batch") and rpt.get("reports"):
                            _render_reports_table_layout(
                                rpt["reports"],
                                base_key_prefix=f"hist_{rec.get('id')}_r",
                                history_id=rec.get("id") or 0,
                                parent_batch_report=rpt,
                                key_suffix=f"_hist_{rec.get('id')}",
                                allow_nested_expander=False,
                            )
                        else:
                            if not rpt.get("related_doc_names"):
                                rpt["related_doc_names"] = [rpt.get("original_filename", rpt.get("file_name", fname or "未知"))]
                            _render_reports_table_layout(
                                [rpt],
                                base_key_prefix=f"hist_{rec.get('id')}",
                                history_id=rec.get("id") or 0,
                                parent_batch_report=None,
                                key_suffix=f"_hist_{rec.get('id')}",
                                allow_nested_expander=False,
                            )
                        _render_history_report_download(rec.get("id"), rpt, fname or "report")
                    else:
                        st.json(rec.get("report_json", "{}"))
        else:
            st.info("暂无历史审核报告。")
    except Exception as e:
        st.warning(f"加载历史报告失败：{e}")


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
    return meta


def _inject_review_meta(report: dict):
    """将当前审核模式/项目信息注入报告 dict，便于后续展示与持久化。"""
    report["_review_meta"] = _build_review_meta_dict()

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


def _render_multi_doc_report(report: dict, r_idx: int, reports: list, key_prefix: str = "", history_id: int = 0, parent_batch_report: dict = None):
    """通用审核报告渲染：分区域展示；可编辑内容、需修改文档多选、处理状态、纠正、一键待办。history_id>0 时显示纠正此审核点。parent_batch_report 为批量报告时纠正会更新整份报告。"""
    file_name = report.get("original_filename", report.get("file_name", "审核报告"))
    doc_names = report.get("related_doc_names") or []
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
        st.markdown(f"**总结：** {report['summary']}")
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

        if in_correction:
            # 纠正模式：仅显示纠正表单（一组可编辑 + 保存纠正 / 取消纠正）；批量报告时传入父报告以便保存整份
            _render_correction_form(
                report, history_id, i, point,
                parent_batch_report=parent_batch_report,
                sub_report_index=r_idx if parent_batch_report else None,
            )
        else:
            # 正常：可编辑描述/建议、需修改文档、处理状态
            current_docs = point.get("modify_docs") or []
            if not isinstance(current_docs, list):
                current_docs = []
            if not current_docs and doc_names:
                current_docs = list(doc_names)
            default_docs = [d for d in current_docs if d in doc_names]

            key_desc = f"{pk}_desc_{r_idx}_{i}"
            key_sug = f"{pk}_sug_{r_idx}_{i}"
            new_desc = st.text_area("问题描述", value=point.get("description", ""), height=min(120, 60 + len((point.get("description") or "")) // 20), key=key_desc)
            st.markdown(f"**位置：** {point.get('location', '未知')}")
            st.markdown(f"**法规依据：** {point.get('regulation_ref', '')}")
            new_sug = st.text_area("修改建议（请写明需修改哪份或哪几份文档）", value=point.get("suggestion", ""), height=min(120, 60 + len((point.get("suggestion") or "")) // 20), key=key_sug)

            key_docs = f"{pk}_modify_docs_{r_idx}_{i}"
            new_modify_docs = st.multiselect("需修改的文档", options=doc_names, default=default_docs, key=key_docs)

            key_action = f"{pk}_action_{r_idx}_{i}"
            current_action = point.get("action") or _get_multi_doc_default_action(severity)
            if current_action not in ACTION_OPTIONS:
                current_action = _get_multi_doc_default_action(severity)
            idx_opt = ACTION_OPTIONS.index(current_action) if current_action in ACTION_OPTIONS else 0
            new_action = st.selectbox("处理状态", ACTION_OPTIONS, index=idx_opt, key=key_action)

            if r_idx < len(reports) and i < len(reports[r_idx].get("audit_points", [])):
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
        st.text_area("待办列表", value=todo_text, height=min(300, 80 + len(immediate) * 80), key=f"{pk}_todo_preview", disabled=True)
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
        st.markdown(f"**总结：** {report['summary']}")
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
                _render_correction_form(report, history_id, i - 1, point)

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
    st.markdown("---")


def _render_history_report_download(report_id: int, report: dict, display_name: str):
    """历史报告区域：格式下拉 + 下载按钮。batch 报告导出其 reports 列表。"""
    st.markdown("**📥 下载此报告**")
    safe_name = "".join(c for c in (display_name or "report") if c.isalnum() or c in "._- ").strip() or "report"
    safe_name = safe_name[:64]

    format_options = ["Excel", "PDF", "Word", "HTML", "JSON", "Markdown"]
    idx = st.selectbox(
        "选择下载格式",
        format_options,
        key=f"hist_fmt_{report_id}",
    )
    if report.get("batch") and report.get("reports"):
        reports = report["reports"]
    else:
        reports = [report]

    if idx == "Excel":
        try:
            data = report_to_excel(reports)
            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ext = "xlsx"
        except Exception as e:
            st.caption(f"Excel 生成失败: {e}")
            return
    elif idx == "JSON":
        data = json.dumps(reports, ensure_ascii=False, indent=2)
        mime = "application/json"
        ext = "json"
    elif idx == "HTML":
        data = report_to_html(reports)
        mime = "text/html"
        ext = "html"
    elif idx == "PDF":
        try:
            data = report_to_pdf(reports)
            mime = "application/pdf"
            ext = "pdf"
        except Exception as e:
            st.caption(f"PDF 生成失败: {e}")
            return
    elif idx == "Word":
        try:
            data = report_to_docx(reports)
            mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ext = "docx"
        except Exception as e:
            st.caption(f"Word 生成失败: {e}")
            return
    else:
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
                sev_l = {"high": "高", "medium": "中", "low": "低", "info": "提示"}.get((point.get("severity") or "").lower(), point.get("severity", ""))
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
        data = "\n".join(md_lines)
        mime = "text/markdown"
        ext = "md"

    st.download_button(
        f"📥 下载 {idx}",
        data=data,
        file_name=f"audit_report_{safe_name}.{ext}",
        mime=mime,
        key=f"hist_dl_{report_id}",
    )


def _aggregate_batch_report_totals(parent: dict) -> None:
    """根据子报告汇总批量报告的 total_points 与各严重程度计数。"""
    reports = parent.get("reports") or []
    parent["total_points"] = sum(r.get("total_points", 0) for r in reports)
    parent["high_count"] = sum(r.get("high_count", 0) for r in reports)
    parent["medium_count"] = sum(r.get("medium_count", 0) for r in reports)
    parent["low_count"] = sum(r.get("low_count", 0) for r in reports)
    parent["info_count"] = sum(r.get("info_count", 0) for r in reports)


def _render_correction_form(report: dict, report_id: int, point_idx: int, point: dict, parent_batch_report: dict = None, sub_report_index: int = None):
    """纠正表单：可编辑描述、建议、需修改的文档、处理状态等，保存纠正 / 取消纠正。批量报告时传入 parent_batch_report 与 sub_report_index 以正确写回整份报告。"""
    prefix = f"cf_{report_id}_{point_idx}"
    doc_names = report.get("related_doc_names") or []
    current_docs = point.get("modify_docs") or []
    if not isinstance(current_docs, list):
        current_docs = []
    # 需修改的文档默认带入：已有则按名称/路径匹配，无则默认选报告全部文档
    def _match(d: str, c: str) -> bool:
        d, c = (d or "").replace("\\", "/").strip(), (c or "").replace("\\", "/").strip()
        return d == c or d.endswith("/" + c) or c.endswith("/" + d)
    if current_docs:
        default_docs = [d for d in doc_names if any(_match(d, c) for c in current_docs)]
    else:
        default_docs = []
    if not default_docs and doc_names:
        default_docs = list(doc_names)

    new_desc = st.text_area("问题描述", value=point.get("description", ""), key=f"{prefix}_desc")
    new_sev = st.selectbox("严重程度", ["high", "medium", "low", "info"],
                           index=["high", "medium", "low", "info"].index(point.get("severity", "info")),
                           key=f"{prefix}_sev")
    new_ref = st.text_area("法规依据", value=point.get("regulation_ref", ""), key=f"{prefix}_ref")
    new_sug = st.text_area("修改建议", value=point.get("suggestion", ""), key=f"{prefix}_sug")
    if doc_names:
        new_modify_docs = st.multiselect("需修改的文档", options=doc_names, default=default_docs, key=f"{prefix}_modify_docs")
    else:
        new_modify_docs = default_docs
    current_action = point.get("action") or _get_multi_doc_default_action(point.get("severity", "info"))
    if current_action not in ACTION_OPTIONS:
        current_action = ACTION_OPTIONS[0]
    new_action = st.selectbox("处理状态", ACTION_OPTIONS, index=ACTION_OPTIONS.index(current_action), key=f"{prefix}_action")
    feed_kb = st.checkbox("将纠正内容写入知识库（下次审核可参考）", value=True, key=f"{prefix}_feed")

    col_save, col_cancel, _ = st.columns([1, 1, 2])
    with col_save:
        if st.button("💾 保存纠正", key=f"{prefix}_save"):
            corrected_point = dict(point)
            corrected_point["description"] = new_desc
            corrected_point["severity"] = new_sev
            corrected_point["regulation_ref"] = new_ref
            corrected_point["suggestion"] = new_sug
            corrected_point["modify_docs"] = new_modify_docs if doc_names else (point.get("modify_docs") or [])
            corrected_point["action"] = new_action

            collection = st.session_state.get("collection_name", "regulations")
            try:
                save_audit_correction(
                    report_id=report_id,
                    point_index=point_idx,
                    collection=collection,
                    file_name=report.get("file_name", ""),
                    original=point,
                    corrected=corrected_point,
                    fed_to_kb=feed_kb,
                )
                points = report.get("audit_points", [])
                if 0 <= point_idx < len(points):
                    points[point_idx] = corrected_point
                    _recount_severity(report)
                    # 批量报告时更新整份报告 JSON，避免只写入子报告覆盖掉 batch 结构
                    if parent_batch_report is not None and sub_report_index is not None:
                        _aggregate_batch_report_totals(parent_batch_report)
                        update_audit_report(report_id, parent_batch_report)
                    else:
                        update_audit_report(report_id, report)

                if feed_kb:
                    _feed_correction_to_kb(collection, report.get("file_name", ""), corrected_point)

                add_operation_log(
                    op_type=OP_TYPE_CORRECTION,
                    collection=collection,
                    file_name=report.get("file_name", ""),
                    extra={"report_id": report_id, "point_index": point_idx,
                           "fed_to_kb": feed_kb, "corrected": corrected_point},
                    model_info=get_current_model_info(),
                )
                st.success("纠正已保存！" + ("已写入知识库。" if feed_kb else ""))
                st.session_state[f"editing_{report_id}_{point_idx + 1}"] = False
            except Exception as e:
                st.error(f"保存纠正失败：{e}")
    with col_cancel:
        if st.button("取消纠正", key=f"{prefix}_cancel"):
            st.session_state[f"editing_{report_id}_{point_idx + 1}"] = False
            st.experimental_rerun()


def _recount_severity(report: dict):
    """根据审核点重新统计各严重程度计数"""
    counts = {"high": 0, "medium": 0, "low": 0, "info": 0}
    for p in report.get("audit_points", []):
        s = p.get("severity", "info")
        counts[s] = counts.get(s, 0) + 1
    report["high_count"] = counts["high"]
    report["medium_count"] = counts["medium"]
    report["low_count"] = counts["low"]
    report["info_count"] = counts["info"]
    report["total_points"] = sum(counts.values())


def _feed_correction_to_kb(collection: str, file_name: str, corrected_point: dict):
    """将纠正内容作为知识文档写入审核点知识库（含多文档一致性：位置、需修改的文档等）。"""
    from langchain.schema import Document
    content = (
        f"[审核纠正经验] 文件：{file_name}\n"
        f"类别：{corrected_point.get('category', '')}\n"
        f"严重程度：{corrected_point.get('severity', '')}\n"
        f"问题描述：{corrected_point.get('description', '')}\n"
        f"法规依据：{corrected_point.get('regulation_ref', '')}\n"
        f"修改建议：{corrected_point.get('suggestion', '')}"
    )
    loc = corrected_point.get("location") or ""
    if loc:
        content += f"\n位置/涉及文档：{loc}"
    modify_docs = corrected_point.get("modify_docs")
    if isinstance(modify_docs, list) and modify_docs:
        content += f"\n需修改的文档：{', '.join(modify_docs)}"
    doc = Document(page_content=content, metadata={
        "source": f"correction:{file_name}",
        "type": "audit_correction",
    })
    try:
        agent = init_agent()
        agent.checkpoint_kb.add_documents([doc], file_name=f"[纠正]{file_name}")
    except Exception:
        pass


def _render_reports_table_layout(
    reports: list,
    base_key_prefix: str,
    history_id: int = 0,
    parent_batch_report: dict = None,
    key_suffix: str = "",
    allow_nested_expander: bool = True,
):
    """按批次→文档→问题层级渲染报告内容（表格+待办），供当前会话与历史报告共用。
    allow_nested_expander: False 时不在文档块使用 expander（用于已在 expander 内的历史报告，避免 Streamlit 禁止嵌套 expander）。
    """
    if not reports:
        return
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
            "序号": r_idx + 1,
            "文件名": fn,
            "🔴高": r.get("high_count", 0),
            "🟡中": r.get("medium_count", 0),
            "🔵低": r.get("low_count", 0),
            "ℹ️提示": r.get("info_count", 0),
            "总计": r.get("total_points", 0),
        })
    if summary_rows:
        headers = ["序号", "文件名", "🔴高", "🟡中", "🔵低", "ℹ️提示", "总计"]
        table_data = [headers] + [[row[h] for h in headers] for row in summary_rows]
        st.table(table_data)

    all_immediate = []
    for r_idx, r in enumerate(reports):
        fn = r.get("original_filename", r.get("file_name", "未知"))
        for p in r.get("audit_points") or []:
            action = p.get("action") or _get_multi_doc_default_action(p.get("severity", "info"))
            if action == "立即修改":
                all_immediate.append((fn, p, r))

    if all_immediate:
        st.markdown(f"### 待办任务（共 {len(all_immediate)} 项「立即修改」）")
        _da = st.session_state.get("multi_doc_default_action") or {
            "high": "立即修改", "medium": "立即修改", "low": "延期修改", "info": "无需修改",
        }
        _get_action = lambda s: _da.get((s or "info").lower(), "无需修改")
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
                st.caption(f"PDF 生成失败: {e}")
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
                st.caption(f"Word 生成失败: {e}")
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
                st.caption(f"Excel 生成失败: {e}")
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

    st.markdown("### 审核点详情")
    for r_idx, report in enumerate(reports):
        if not report.get("related_doc_names"):
            report["related_doc_names"] = [report.get("original_filename", report.get("file_name", "未知"))]
        file_name = report.get("original_filename", report.get("file_name", "未知"))
        pk = f"{base_key_prefix}{r_idx}"
        points = report.get("audit_points") or []

        if allow_nested_expander:
            doc_container = st.expander(
                f"📄 {file_name}（{report.get('total_points', 0)} 个问题点：🔴{report.get('high_count',0)} 🟡{report.get('medium_count',0)} 🔵{report.get('low_count',0)}）",
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
            if points:
                for i, p in enumerate(points):
                    sev = p.get("severity", "info")
                    icon = _SEVERITY_ICONS.get(sev, "ℹ️")
                    action = p.get("action") or _get_multi_doc_default_action(sev)
                    desc_short = (p.get("description") or "")[:60] + ("…" if len(p.get("description", "")) > 60 else "")
                    cols = st.columns([0.5, 0.8, 1, 3, 1.2])
                    cols[0].markdown(f"**{i+1}**")
                    cols[1].markdown(f"{icon} {_SEVERITY_LABELS.get(sev, sev)}")
                    cols[2].markdown(p.get("category", ""))
                    cols[3].markdown(desc_short)
                    btn_key = f"detail_{pk}_{i}{key_suffix}"
                    if cols[4].button("📋 详情", key=btn_key):
                        st.session_state[f"_show_detail_{pk}_{i}"] = True
                        st.experimental_rerun()

                    if st.session_state.get(f"_show_detail_{pk}_{i}"):
                        _render_point_detail_inline(
                            report, r_idx, reports, i, p,
                            pk=pk, key_suffix=key_suffix,
                            history_id=history_id,
                            parent_batch_report=parent_batch_report,
                        )

            # 本文档待办导出
            doc_immediate = [p for p in points if (p.get("action") or _get_multi_doc_default_action(p.get("severity", "info"))) == "立即修改"]
            if doc_immediate:
                st.markdown(f"**📋 本文档待办（{len(doc_immediate)} 项）**")
                _da2 = st.session_state.get("multi_doc_default_action") or {
                    "high": "立即修改", "medium": "立即修改", "low": "延期修改", "info": "无需修改",
                }
                _get_action_doc = lambda s: _da2.get((s or "info").lower(), "无需修改")
                c_doc1, c_doc2, c_doc3, c_doc4, c_doc5 = st.columns(5)
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
                    try:
                        single_pdf = report_todo_to_pdf(
                            [report],
                            only_immediate=True,
                            get_default_action=_get_action_doc,
                            project_name=meta.get("project_name", ""),
                            product=meta.get("product_name", ""),
                            country=meta.get("registration_country", ""),
                        )
                        st.download_button("📥 本文档待办 PDF", data=single_pdf, file_name=f"待办_{file_name}.pdf", mime="application/pdf", key=f"doc_todo_pdf_{pk}{key_suffix}")
                    except Exception:
                        st.caption("PDF 失败")
                with c_doc3:
                    try:
                        single_docx = report_todo_to_docx(
                            [report],
                            only_immediate=True,
                            get_default_action=_get_action_doc,
                            project_name=meta.get("project_name", ""),
                            product=meta.get("product_name", ""),
                            country=meta.get("registration_country", ""),
                        )
                        st.download_button("📥 本文档待办 Word", data=single_docx, file_name=f"待办_{file_name}.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", key=f"doc_todo_docx_{pk}{key_suffix}")
                    except Exception:
                        st.caption("Word 失败")
                with c_doc4:
                    try:
                        single_xlsx = report_todo_to_excel(
                            [report],
                            only_immediate=True,
                            get_default_action=_get_action_doc,
                            project_name=meta.get("project_name", ""),
                            product=meta.get("product_name", ""),
                            country=meta.get("registration_country", ""),
                        )
                        st.download_button("📥 本文档待办 Excel", data=single_xlsx, file_name=f"待办_{file_name}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"doc_todo_xlsx_{pk}{key_suffix}")
                    except Exception:
                        st.caption("Excel 失败")
                with c_doc5:
                    single_lines = []
                    for idx, p in enumerate(doc_immediate, 1):
                        single_lines.append(f"{idx}. {p.get('category', '')}：{p.get('description', '')[:100]}")
                        single_lines.append(f"   修改建议：{p.get('suggestion', '')[:120]}")
                        single_lines.append("")
                    st.download_button(
                        "📥 本文档待办（文本）",
                        data="\n".join(single_lines),
                        file_name=f"待办_{file_name}.txt",
                        mime="text/plain",
                        key=f"doc_todo_txt_{pk}{key_suffix}",
                    )


def _render_point_detail_inline(
    report: dict, r_idx: int, reports: list,
    point_idx: int, point: dict,
    pk: str, key_suffix: str,
    history_id: int = 0,
    parent_batch_report: dict = None,
):
    """在审核点行下方展开的详情/编辑面板（替代弹窗，兼容所有 Streamlit 版本）。"""
    sev = point.get("severity", "info")
    icon = _SEVERITY_ICONS.get(sev, "ℹ️")
    detail_key = f"_show_detail_{pk}_{point_idx}"
    file_name = report.get("original_filename", report.get("file_name", "未知"))
    doc_names = report.get("related_doc_names") or []

    with st.container():
        st.markdown(f"---\n**{icon} 审核点 {point_idx + 1} 详情 — {point.get('category', '未分类')}**")

        c_info1, c_info2 = st.columns(2)
        c_info1.markdown(f"**严重程度：** {_SEVERITY_LABELS.get(sev, sev)}")
        c_info2.markdown(f"**位置：** {point.get('location', '未知')}")
        st.markdown(f"**法规依据：** {point.get('regulation_ref', '—')}")

        pfx = f"{pk}_detail_{point_idx}{key_suffix}"
        new_desc = st.text_area("问题描述", value=point.get("description", ""), key=f"{pfx}_desc", height=100)
        new_sug = st.text_area("修改建议", value=point.get("suggestion", ""), key=f"{pfx}_sug", height=100)

        current_docs = point.get("modify_docs") or []
        if not isinstance(current_docs, list):
            current_docs = []
        if not current_docs and doc_names:
            current_docs = list(doc_names)
        default_docs = [d for d in current_docs if d in doc_names]
        new_modify_docs = st.multiselect("需修改的文档", options=doc_names, default=default_docs, key=f"{pfx}_docs")

        current_action = point.get("action") or _get_multi_doc_default_action(sev)
        if current_action not in ACTION_OPTIONS:
            current_action = ACTION_OPTIONS[0]
        new_action = st.selectbox("处理状态", ACTION_OPTIONS, index=ACTION_OPTIONS.index(current_action), key=f"{pfx}_action")

        c_save, c_close = st.columns(2)
        with c_save:
            if st.button("💾 保存修改", key=f"{pfx}_save"):
                if r_idx < len(reports) and point_idx < len(reports[r_idx].get("audit_points", [])):
                    reports[r_idx]["audit_points"][point_idx]["description"] = new_desc
                    reports[r_idx]["audit_points"][point_idx]["suggestion"] = new_sug
                    reports[r_idx]["audit_points"][point_idx]["modify_docs"] = new_modify_docs
                    reports[r_idx]["audit_points"][point_idx]["action"] = new_action
                st.success("已保存")
                if history_id:
                    try:
                        updated_report = parent_batch_report if parent_batch_report else report
                        from src.core.db import update_audit_report
                        update_audit_report(history_id, updated_report)
                    except Exception:
                        pass
        with c_close:
            if st.button("✖ 关闭", key=f"{pfx}_close"):
                st.session_state.pop(detail_key, None)
                st.experimental_rerun()

        st.markdown("---")


def render_reports(reports: list):
    """渲染审核报告（当前会话）：按批次→文档→问题层级表格展示，支持单条/批次待办。"""
    st.subheader("📋 审核报告")
    st.caption("按 **批次 → 文档 → 问题** 层级展示；可导出单文档待办或整批待办 CSV。")

    _render_reports_table_layout(reports, "r", history_id=0, parent_batch_report=None, key_suffix="")
    _render_download_buttons(reports)


def _render_download_buttons(reports: list, key_suffix: str = ""):
    """渲染报告下载按钮组"""
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
    extra = 1 if (has_single_docx or has_kdocs_docx) else 0
    cols = st.columns(6 + extra)
    ks = key_suffix

    with cols[0]:
        try:
            xlsx_bytes = report_to_excel(reports)
            st.download_button(
                "📥 Excel", data=xlsx_bytes,
                file_name="audit_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"dl_xlsx{ks}",
            )
        except Exception as e:
            st.caption(f"Excel 生成失败: {e}")
    with cols[1]:
        try:
            pdf_bytes = report_to_pdf(reports)
            st.download_button(
                "📥 PDF", data=pdf_bytes,
                file_name="audit_report.pdf", mime="application/pdf",
                key=f"dl_pdf{ks}",
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
                key=f"dl_docx{ks}",
            )
        except Exception as e:
            st.caption(f"Word 生成失败: {e}")
    with cols[3]:
        html_data = report_to_html(reports)
        st.download_button(
            "📥 HTML", data=html_data,
            file_name="audit_report.html", mime="text/html",
            key=f"dl_html{ks}",
        )
    with cols[4]:
        json_data = json.dumps(reports, ensure_ascii=False, indent=2)
        st.download_button(
            "📥 JSON", data=json_data,
            file_name="audit_report.json", mime="application/json",
            key=f"dl_json{ks}",
        )
    with cols[5]:
        md_lines = []
        for report in reports:
            file_name = report.get("original_filename", report.get("file_name", ""))
            md_lines.append(f"# 审核报告：{file_name}\n")
            md_lines.append(f"**总结：** {report.get('summary', '')}\n")
            md_lines.append(f"| 高风险 | 中风险 | 低风险 | 提示 |")
            md_lines.append(f"|--------|--------|--------|------|")
            md_lines.append(
                f"| {report.get('high_count', 0)} | {report.get('medium_count', 0)} "
                f"| {report.get('low_count', 0)} | {report.get('info_count', 0)} |\n"
            )
            for i, point in enumerate(report.get("audit_points", []), 1):
                sev_label = {"high": "高", "medium": "中", "low": "低", "info": "提示"}.get((point.get("severity") or "").lower(), point.get("severity", ""))
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

        st.download_button(
            "📥 Markdown", data="\n".join(md_lines),
            file_name="audit_report.md", mime="text/markdown",
            key=f"dl_md{ks}",
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
                    key=f"dl_docx_comments{ks}",
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
                    key=f"dl_kdocs_comments{ks}",
                    help="下载金山文档原文件并在对应位置插入批注，可上传回金山文档替换原文件",
                )
            except Exception as e:
                st.caption(f"带批注 Word 生成失败: {e}")


def render_knowledge_page():
    """知识库查询页面"""
    st.header("🔎 知识库查询")
    st.markdown("查询已训练的知识库内容，验证法规/标准或审核点是否已正确入库。")

    if not _require_provider():
        return

    agent = init_agent()
    collection = st.session_state.get("collection_name", "regulations")

    tab_search, tab_browse = st.tabs(["🔍 语义检索", "📂 知识库文档浏览"])

    with tab_search:
        kb_target = st.radio(
            "查询目标",
            ["法规知识库（第一步）", "审核点知识库（第二步）"],
            horizontal=True,
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

    with tab_browse:
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
            ["全部", "法规训练批次", "生成审核点", "审核点训练", "项目专属训练", "项目专属训练失败/中断", "审核批次",
             "单文件训练", "单文件审核", "文本审核", "审核纠正", "训练失败", "审核失败"],
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
        "训练失败": "train_error",
        "审核失败": "review_error",
    }
    op_type = type_map.get(op_type_filter, None)

    only_current = st.checkbox("仅当前知识库", value=False, key="op_only_collection")
    collection_filter = st.session_state.get("collection_name", "regulations") if only_current else None

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
            if rec["op_type"] in ("train_error", "review_error", "review_text_error", OP_TYPE_TRAIN_PROJECT_ERROR) and extra.get("traceback"):
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


def main():
    st.set_page_config(
        page_title="注册文档审核工具",
        page_icon="📋",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    render_sidebar()

    st.title("📋 注册文档审核工具")

    page = st.radio(
        "选择功能",
        [
            "① 法规训练 & 生成审核点",
            "② 审核点管理 & 训练",
            "③ 文档审核",
            "📝 提示词配置",
            "🔎 知识库查询",
            "📋 操作记录",
        ],
        horizontal=True,
    )

    # 在蒙层上显示 loading：全屏遮罩 + 加载文案与动画，内容就绪后自动消失（不占用页面内容区域）
    _render_loading_overlay()
    if page == "① 法规训练 & 生成审核点":
        render_step1_page()
    elif page == "② 审核点管理 & 训练":
        render_step2_page()
    elif page == "③ 文档审核":
        render_step3_page()
    elif page == "📝 提示词配置":
        render_prompts_page()
    elif page == "🔎 知识库查询":
        render_knowledge_page()
    elif page == "📋 操作记录":
        render_operations_page()


if __name__ == "__main__":
    main()
