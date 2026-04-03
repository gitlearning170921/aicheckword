"""
系统全量配置：表单编辑、入库 runtime_settings_json + 兼容列双写、导出 .env / JSON 导入。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Set, Tuple

import streamlit as st

from config import settings
from src.streamlit_compat import streamlit_rerun
from config.runtime_settings import (
    apply_runtime_config_dict,
    export_dotenv_lines,
    list_persistable_setting_keys,
    serialize_settings_to_flat_dict,
    sync_cursor_overrides_from_settings,
)
from src.core.db import persist_settings_dual_write

# ── 静态分组（非 AI Tab）的字段 ─────────────────────────────────────────

MYSQL_KEYS = [
    "mysql_host",
    "mysql_port",
    "mysql_database",
    "mysql_user",
    "mysql_password",
    "mysql_charset",
]

NETWORK_KEYS = [
    "llm_verify_ssl",
    "llm_trust_env",
]

CHROMA_CHUNK_KEYS = [
    "chroma_server_host",
    "chroma_server_port",
    "chroma_server_ssl",
    "chroma_server_headers_json",
    "chroma_persist_dir",
    "chunk_size",
    "chunk_overlap",
    "embedding_max_retries",
    "embedding_retry_delay_sec",
    "embedding_large_file_threshold",
    "embedding_large_file_batch_size",
    "embedding_streamlit_chunk_threshold",
    "embedding_streamlit_batches_per_rerun",
]

REVIEW_STABILITY_KEYS = [
    "review_llm_min_interval_sec",
    "review_deepseek_chroma_fetch_cap",
    "review_deepseek_target_cap",
    "review_batch_skip_llm_summary",
    "review_batch_inter_doc_sleep_sec",
    "review_deepseek_inter_doc_sleep_sec",
    "audit_perf_log",
    "async_correction_kb_feed",
]

API_DIR_KEYS = [
    "api_host",
    "api_port",
    "training_docs_dir",
    "uploads_dir",
    "db_path",
]

KDOCS_KEYS = ["kdocs_app_id", "kdocs_app_key"]

# 与侧栏 provider 值一致；返回当前提供方需要编辑的字段（不含 provider / 模型）
_PROVIDER_OPTION_VALUES = [
    "deepseek",
    "openai",
    "lingyi",
    "gemini",
    "tongyi",
    "baidu",
    "ollama",
    "cursor",
]

_VALID_PROVIDERS = frozenset(_PROVIDER_OPTION_VALUES)


def _coerce_provider_to_valid(raw: Any) -> str:
    """非法提供方与侧栏 selectbox 的 next(..., 0) 一致，落到列表首项（DeepSeek）。"""
    p = str(raw or "").strip().lower()
    if p in _VALID_PROVIDERS:
        return p
    return _PROVIDER_OPTION_VALUES[0]


def _provider_current_for_sys_cfg_form(wk: str) -> str:
    """系统配置页内提供方：仅本页控件 → settings（入库值）→ ollama。不读 current_provider，避免牵动侧栏。"""
    v = st.session_state.get(wk) or getattr(settings, "provider", None) or "ollama"
    return str(v).strip().lower()


def _show_llm_embedding_fields(provider: str) -> bool:
    """侧栏不展示对话/嵌入模型输入的提供方，本页也不展示（保存时保留 settings 中已有模型字段）。"""
    return _coerce_provider_to_valid(provider) != "cursor"


def _all_ai_related_setting_keys() -> Set[str]:
    """所有「按提供方切换」会展示的 AI 相关字段（用于切换时从 settings 刷新 session，避免 Streamlit 同位控件错位）。"""
    keys: Set[str] = set()
    for p in _PROVIDER_OPTION_VALUES:
        keys.update(_ai_keys_for_provider(p))
    keys.update({"llm_model", "embedding_model"})
    return keys


def _refresh_session_ai_fields_from_settings(provider: str) -> None:
    """切换提供方后：用当前内存 settings 覆盖 AI 区块相关 session_state，便于自动带出各服务已保存的 Key/URL。"""
    prov = _coerce_provider_to_valid(provider)
    for k in _all_ai_related_setting_keys():
        if k in ("llm_model", "embedding_model") and not _show_llm_embedding_fields(prov):
            continue
        st.session_state[_widget_key(k)] = getattr(settings, k)


def _maybe_rerun_on_sys_cfg_provider_change(wkp: str) -> None:
    """
    下拉切换提供方后：旧版 Streamlit 同位置控件易错位（出现 Cursor 选中却显示 DeepSeek 字段）；
    从 settings 刷新该服务相关字段并 rerun 一次。
    """
    prov_now = _coerce_provider_to_valid(st.session_state.get(wkp))
    snap = st.session_state.get("_sys_cfg_provider_form_snap")
    if snap is None:
        st.session_state["_sys_cfg_provider_form_snap"] = prov_now
        return
    if snap == prov_now:
        return
    _refresh_session_ai_fields_from_settings(prov_now)
    st.session_state["_sys_cfg_provider_form_snap"] = prov_now
    streamlit_rerun()


def _ai_keys_for_provider(provider: str) -> List[str]:
    p = _coerce_provider_to_valid(provider)
    if p == "openai":
        return ["openai_api_key", "openai_base_url"]
    if p == "deepseek":
        return ["deepseek_api_key", "deepseek_base_url", "ollama_base_url"]
    if p == "lingyi":
        return ["lingyi_api_key", "lingyi_base_url", "ollama_base_url"]
    if p == "gemini":
        return ["gemini_api_key", "google_api_key"]
    if p == "tongyi":
        return ["dashscope_api_key"]
    if p == "baidu":
        return ["qianfan_ak", "qianfan_sk"]
    if p == "ollama":
        return ["ollama_base_url"]
    if p == "cursor":
        return [
            "cursor_api_key",
            "cursor_api_base",
            "cursor_repository",
            "cursor_ref",
            "cursor_embedding",
            "ollama_base_url",
        ]
    return ["ollama_base_url"]


def _all_ai_tab_keys_union() -> Set[str]:
    s: Set[str] = {"provider", "llm_model", "embedding_model"}
    for pv in _PROVIDER_OPTION_VALUES:
        s.update(_ai_keys_for_provider(pv))
    return s


# 全量可持久化键：须与 Settings 一致（含未在表单展示的 cursor_* 兼容项，保存时与 llm_* 同步）
FLAT_KEYS: List[str] = list_persistable_setting_keys()

STATIC_FORM_KEYS: Set[str] = set(MYSQL_KEYS + NETWORK_KEYS + CHROMA_CHUNK_KEYS + REVIEW_STABILITY_KEYS + API_DIR_KEYS + KDOCS_KEYS)
_AI_TAB_KEYS: Set[str] = _all_ai_tab_keys_union()
_GROUPED = STATIC_FORM_KEYS | _AI_TAB_KEYS | {"cursor_verify_ssl", "cursor_trust_env"}
_all = set(FLAT_KEYS)
if _GROUPED != _all:
    missing = sorted(_all - _GROUPED)
    extra = sorted(_GROUPED - _all)
    raise RuntimeError(f"system_config_ui: keys mismatch missing={missing} extra={extra}")

SYSTEM_CONFIG_TAB_LABELS: List[str] = [
    "MySQL",
    "AI 与模型",
    "网络/SSL",
    "向量分块",
    "审核稳定",
    "API 目录",
    "金山文档",
]

FIELD_LABELS: Dict[str, str] = {
    "provider": "服务提供方",
    "mysql_host": "MySQL 主机",
    "mysql_port": "MySQL 端口",
    "mysql_database": "数据库名",
    "mysql_user": "数据库用户",
    "mysql_password": "数据库密码",
    "mysql_charset": "字符集",
    "openai_api_key": "OpenAI API Key",
    "openai_base_url": "OpenAI Base URL",
    "ollama_base_url": "Ollama 地址（本地嵌入/对话）",
    "deepseek_api_key": "DeepSeek API Key",
    "deepseek_base_url": "DeepSeek Base URL",
    "lingyi_api_key": "零一万物 API Key",
    "lingyi_base_url": "零一万物 Base URL",
    "gemini_api_key": "Gemini API Key",
    "google_api_key": "Google API Key（兼容）",
    "dashscope_api_key": "通义 DashScope Key",
    "qianfan_ak": "文心 AK",
    "qianfan_sk": "文心 SK",
    "cursor_api_key": "Cursor API Key",
    "cursor_api_base": "Cursor API Base",
    "cursor_repository": "GitHub 仓库（owner/repo）",
    "cursor_ref": "分支 / ref",
    "cursor_embedding": "Cursor 侧嵌入后端（如 ollama）",
    "llm_verify_ssl": "校验 HTTPS 证书（关闭=不校验，与侧栏一致）",
    "llm_trust_env": "使用系统代理（关闭=直连，与侧栏一致）",
    "cursor_verify_ssl": "cursor_verify_ssl（兼容项，保存时与上一项同步）",
    "cursor_trust_env": "cursor_trust_env（兼容项，保存时与上一项同步）",
    "llm_model": "对话模型名（须与当前提供方一致，如 qwen2.5 / gpt-4o / deepseek-chat）",
    "embedding_model": "嵌入模型名（各客户端需一致）",
    "chroma_server_host": "Chroma 服务地址（留空=本机目录）",
    "chroma_server_port": "Chroma HTTP 端口",
    "chroma_server_ssl": "HTTPS 连接 Chroma",
    "chroma_server_headers_json": "Chroma 请求头 JSON（可选）",
    "chroma_persist_dir": "本机 Chroma 目录（仅本地模式）",
    "chunk_size": "分块大小",
    "chunk_overlap": "分块重叠",
    "embedding_max_retries": "嵌入失败最大重试",
    "embedding_retry_delay_sec": "嵌入重试间隔（秒）",
    "embedding_large_file_threshold": "大文件块数阈值",
    "embedding_large_file_batch_size": "大文件每批块数",
    "embedding_streamlit_chunk_threshold": "Streamlit 分段训练分块阈值（超此块数则多轮刷新防断连）",
    "embedding_streamlit_batches_per_rerun": "每轮页面刷新嵌入批次数（默认 1 最稳）",
    "review_llm_min_interval_sec": "审核 LLM 最小间隔（秒）",
    "review_deepseek_chroma_fetch_cap": "DeepSeek Chroma 取回上限",
    "review_deepseek_target_cap": "DeepSeek 审核点参考条数上限",
    "review_batch_skip_llm_summary": "批量审核跳过 LLM 总结",
    "review_batch_inter_doc_sleep_sec": "批量审核文件间间隔·全模型（秒）",
    "review_deepseek_inter_doc_sleep_sec": "批量审核文件间间隔·DeepSeek 额外（秒）",
    "audit_perf_log": "审核报告性能分段日志（或环境变量 AUDIT_PERF_LOG=1）",
    "async_correction_kb_feed": "纠正写入知识库：后台线程异步嵌入（关闭则同步阻塞保存）",
    "api_host": "FastAPI 监听地址",
    "api_port": "FastAPI 端口",
    "training_docs_dir": "训练文档目录",
    "uploads_dir": "上传目录",
    "db_path": "db_path（兼容旧项）",
    "kdocs_app_id": "金山文档 AppId",
    "kdocs_app_key": "金山文档 AppKey",
}

# 显示名、值须与 app.py 侧栏 _provider_list 完全一致
_PROVIDER_OPTIONS: List[Tuple[str, str]] = [
    ("DeepSeek (OpenAI 兼容)", "deepseek"),
    ("OpenAI", "openai"),
    ("零一万物 (OpenAI 兼容)", "lingyi"),
    ("Google Gemini", "gemini"),
    ("阿里通义千问", "tongyi"),
    ("百度文心一言", "baidu"),
    ("Ollama (本地免费)", "ollama"),
    ("Cursor Agent (Cloud API)", "cursor"),
]


def _widget_key(name: str) -> str:
    return f"sys_cfg__{name}"


def _is_secret(name: str) -> bool:
    return name in {
        "mysql_password",
        "openai_api_key",
        "deepseek_api_key",
        "lingyi_api_key",
        "gemini_api_key",
        "google_api_key",
        "dashscope_api_key",
        "qianfan_sk",
        "cursor_api_key",
        "kdocs_app_key",
    }


def _push_settings_to_widgets() -> None:
    """用当前内存 settings 覆盖所有表单键（与侧栏/本机配置对齐）。"""
    for key in FLAT_KEYS:
        val = getattr(settings, key)
        if key == "provider":
            val = _coerce_provider_to_valid(val)
        st.session_state[_widget_key(key)] = val


def _ensure_defaults_from_settings() -> None:
    """仅补齐尚未出现在 session_state 的键。"""
    for key in FLAT_KEYS:
        wk = _widget_key(key)
        if wk not in st.session_state:
            val = getattr(settings, key)
            if key == "provider":
                val = _coerce_provider_to_valid(val)
            st.session_state[wk] = val


def _widget_effective_value(key: str, wk: str) -> Any:
    """控件展示用：优先 session_state，否则当前 settings（与侧栏/库一致）。"""
    if wk in st.session_state:
        return st.session_state[wk]
    return getattr(settings, key)


def _collect_from_widgets() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in FLAT_KEYS:
        wk = _widget_key(key)
        out[key] = st.session_state.get(wk, getattr(settings, key))
    out["provider"] = _coerce_provider_to_valid(out.get("provider"))
    st.session_state[_widget_key("provider")] = out["provider"]
    # Cursor 等：不在此编辑模型，避免误用其它服务的模型名；入库仍以当前 settings 为准
    if not _show_llm_embedding_fields(out["provider"]):
        out["llm_model"] = getattr(settings, "llm_model")
        out["embedding_model"] = getattr(settings, "embedding_model")
    # 兼容库列与旧逻辑：与侧栏一致，cursor_* 跟随 llm_*
    out["cursor_verify_ssl"] = bool(out.get("llm_verify_ssl", True))
    out["cursor_trust_env"] = bool(out.get("llm_trust_env", True))
    st.session_state[_widget_key("cursor_verify_ssl")] = out["cursor_verify_ssl"]
    st.session_state[_widget_key("cursor_trust_env")] = out["cursor_trust_env"]
    return out


def _render_field(key: str) -> None:
    label = FIELD_LABELS.get(key, key)
    field = settings.model_fields[key]
    ann = field.annotation
    wk = _widget_key(key)

    if key == "provider":
        vals = [v for _, v in _PROVIDER_OPTIONS]
        labels_map = {v: lab for lab, v in _PROVIDER_OPTIONS}
        cur_form = _provider_current_for_sys_cfg_form(wk)
        idx_default = next((i for i, v in enumerate(vals) if v == cur_form), 0)
        if wk not in st.session_state or st.session_state.get(wk) not in vals:
            st.session_state[wk] = vals[idx_default]
        # 勿与 key 同时强传 index：部分旧版 Streamlit 会每轮把选中项打回 index，导致切换无效/控件错位
        try:
            st.selectbox(
                label,
                vals,
                format_func=lambda x, m=labels_map: m.get(x, x),
                key=wk,
            )
        except TypeError:
            idx = vals.index(st.session_state[wk])
            st.selectbox(
                label,
                vals,
                index=min(idx, len(vals) - 1),
                format_func=lambda x, m=labels_map: m.get(x, x),
                key=wk,
            )
        return

    if ann is bool:
        bval = bool(_widget_effective_value(key, wk))
        try:
            st.checkbox(label, value=bval, key=wk)
        except TypeError:
            st.session_state[wk] = bval
            st.checkbox(label, key=wk)
        return

    if ann is int:
        ival = int(_widget_effective_value(key, wk))
        try:
            st.number_input(label, value=ival, step=1, key=wk)
        except TypeError:
            st.session_state[wk] = ival
            st.number_input(label, step=1, key=wk)
        return

    if ann is float:
        fval = float(_widget_effective_value(key, wk))
        try:
            st.number_input(label, value=fval, step=0.1, format="%.6f", key=wk)
        except TypeError:
            st.session_state[wk] = fval
            st.number_input(label, step=0.1, format="%.6f", key=wk)
        return

    if key == "chroma_server_headers_json":
        tval = str(_widget_effective_value(key, wk) or "")
        try:
            st.text_area(
                label,
                value=tval,
                height=100,
                key=wk,
                placeholder='例如 {"Authorization":"Bearer xxx"}',
            )
        except TypeError:
            st.session_state[wk] = tval
            st.text_area(label, height=100, key=wk, placeholder='例如 {"Authorization":"Bearer xxx"}')
        return
    if _is_secret(key):
        # Streamlit 的 type="password" 不回显已保存内容，首次打开会像「没加载」；本页改为明文便于核对入库值（请注意环境安全）
        if wk not in st.session_state:
            st.session_state[wk] = getattr(settings, key) or ""
        sval = str(_widget_effective_value(key, wk) or "")
        try:
            st.text_input(label, value=sval, key=wk)
        except TypeError:
            st.session_state[wk] = sval
            st.text_input(label, key=wk)
        return
    sval = _widget_effective_value(key, wk)
    if sval is None:
        sval = ""
    sval = str(sval)
    try:
        st.text_input(label, value=sval, key=wk)
    except TypeError:
        st.session_state[wk] = sval
        st.text_input(label, key=wk)


def _render_static_keys(title: str, keys: List[str]) -> None:
    st.markdown(f"**{title}**")
    for key in keys:
        _render_field(key)


def _render_ai_tab() -> None:
    st.markdown("**AI 服务与模型**")
    st.caption(
        "此处**仅编辑要入库的配置**，下拉切换提供方**不会**改变左侧边栏当前使用的服务；"
        "侧栏用于切换实际运行中的 AI。保存系统配置后，侧栏会与入库的提供方对齐。"
        "无法识别的提供方默认选中列表第一项（与侧栏规则相同）。"
    )
    wkp = _widget_key("provider")
    _render_field("provider")
    _maybe_rerun_on_sys_cfg_provider_change(wkp)
    prov = _coerce_provider_to_valid(st.session_state.get(wkp))
    st.markdown("**当前提供方的连接与密钥**")
    for key in _ai_keys_for_provider(prov):
        _render_field(key)
    if _show_llm_embedding_fields(prov):
        st.markdown("**模型**")
        _render_field("llm_model")
        _render_field("embedding_model")
    else:
        st.caption(
            "当前提供方与侧栏一致，**无需在此配置**对话/嵌入模型；保存时保留内存中已有模型字段，不会被本页清空。"
        )


def render_system_config_body() -> None:
    """渲染配置表单主体。"""
    st.caption(
        "保存后写入数据库 `runtime_settings_json`（全量）并同步兼容列。"
        "迁机时新电脑只需 .env 配好 MySQL，启动后会自动从库恢复其余项。"
    )
    st.warning(
        "修改 **MySQL 连接信息** 后需**重启应用**才会用新连接；当前会话仍使用启动时的连接。"
    )
    st.caption(
        "🔐 **API Key / 数据库密码** 等在本页以**明文**回显，便于确认已从配置或库中加载；请勿在不受信任环境中截图或投屏。"
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("从当前内存载入表单", key="sys_cfg_mem"):
            _push_settings_to_widgets()
            st.session_state.pop("_sys_cfg_provider_form_snap", None)
            st.session_state["current_provider"] = (settings.provider or "ollama").strip().lower()
            st.success("已从当前内存覆盖表单。")
            streamlit_rerun()
    with c2:
        if st.button("重置表单为内存值", key="sys_cfg_reset_widgets"):
            for k in FLAT_KEYS:
                st.session_state.pop(_widget_key(k), None)
            _push_settings_to_widgets()
            st.session_state.pop("_sys_cfg_provider_form_snap", None)
            streamlit_rerun()

    # 不用 st.tabs：优先横向 radio；无 horizontal / label_visibility 时不用纵向长列表，改横向按钮（与旧版 Streamlit 1.9 等兼容）
    _sec_key = "sys_cfg_section_radio"
    if _sec_key not in st.session_state:
        _leg = st.session_state.get("sys_cfg_section_choice")
        if _leg in SYSTEM_CONFIG_TAB_LABELS:
            st.session_state[_sec_key] = _leg
    section = None
    try:
        section = st.radio(
            "配置分区",
            SYSTEM_CONFIG_TAB_LABELS,
            horizontal=True,
            key=_sec_key,
            label_visibility="collapsed",
        )
    except TypeError:
        try:
            section = st.radio(
                "配置分区",
                SYSTEM_CONFIG_TAB_LABELS,
                horizontal=True,
                key=_sec_key,
            )
        except TypeError:
            section = None
    if section is None:
        st.caption("配置分区（当前 Streamlit 无横向单选，已用按钮行模拟）")
        if _sec_key not in st.session_state or st.session_state[_sec_key] not in SYSTEM_CONFIG_TAB_LABELS:
            st.session_state[_sec_key] = SYSTEM_CONFIG_TAB_LABELS[0]
        row1, row2 = SYSTEM_CONFIG_TAB_LABELS[:4], SYSTEM_CONFIG_TAB_LABELS[4:]
        c1 = st.columns(len(row1))
        for i, lab in enumerate(row1):
            with c1[i]:
                try:
                    hit = st.button(lab, key=f"sys_cfg_sec_btn_{i}", use_container_width=True)
                except TypeError:
                    hit = st.button(lab, key=f"sys_cfg_sec_btn_{i}")
                if hit:
                    st.session_state[_sec_key] = lab
        if row2:
            c2 = st.columns(len(row2))
            for j, lab in enumerate(row2):
                with c2[j]:
                    try:
                        hit = st.button(lab, key=f"sys_cfg_sec_btn_{j + len(row1)}", use_container_width=True)
                    except TypeError:
                        hit = st.button(lab, key=f"sys_cfg_sec_btn_{j + len(row1)}")
                    if hit:
                        st.session_state[_sec_key] = lab
        section = st.session_state[_sec_key]

    if section == "MySQL":
        _render_static_keys("MySQL（新机器请至少在 .env 中配置此项以首次连库）", MYSQL_KEYS)
    elif section == "AI 与模型":
        _render_ai_tab()
    elif section == "网络/SSL":
        _render_static_keys(
            "网络 / SSL / 代理（与侧栏「不校验 SSL」「不使用系统代理」一致；默认即当前运行配置）",
            NETWORK_KEYS,
        )
    elif section == "向量分块":
        _render_static_keys("向量库、分块与嵌入（默认即当前本机/库中的配置，可直接改）", CHROMA_CHUNK_KEYS)
    elif section == "审核稳定":
        _render_static_keys("审核稳定性（DeepSeek 等）", REVIEW_STABILITY_KEYS)
    elif section == "API 目录":
        _render_static_keys("API 服务与目录", API_DIR_KEYS)
    elif section == "金山文档":
        _render_static_keys("金山文档开放平台", KDOCS_KEYS)

    st.markdown("---")
    st.subheader("高级：JSON 导入 / 导出 .env")
    ij = st.text_area("粘贴 JSON（与库内 runtime_settings_json 同结构）", height=160, key="sys_cfg_json_import")
    ic1, ic2 = st.columns(2)
    with ic1:
        if st.button("应用 JSON 到内存与数据库", key="sys_cfg_apply_json"):
            try:
                data = json.loads(ij.strip() or "{}")
                if not isinstance(data, dict):
                    st.error("JSON 须为对象。")
                else:
                    n = apply_runtime_config_dict(data)
                    sync_cursor_overrides_from_settings()
                    persist_settings_dual_write()
                    _push_settings_to_widgets()
                    st.session_state.pop("_sys_cfg_provider_form_snap", None)
                    st.session_state["current_provider"] = (settings.provider or "ollama").strip().lower()
                    st.success(f"已应用 {n} 个字段并保存到数据库。")
                    streamlit_rerun()
            except json.JSONDecodeError as e:
                st.error(f"JSON 解析失败：{e}")
    with ic2:
        dot = export_dotenv_lines()
        st.download_button("下载 .env 片段", dot, file_name="aicheckword-export.env", mime="text/plain", key="sys_cfg_dl_env")

    st.markdown("---")
    b1, b2 = st.columns(2)
    with b1:
        if st.button("保存到数据库", key="sys_cfg_save_db"):
            data = _collect_from_widgets()
            n = apply_runtime_config_dict(data)
            sync_cursor_overrides_from_settings()
            persist_settings_dual_write()
            st.session_state.pop("_sys_cfg_provider_form_snap", None)
            st.session_state["current_provider"] = (settings.provider or "ollama").strip().lower()
            st.success(f"已保存（应用 {n} 项配置）。")
            streamlit_rerun()
    with b2:
        blob = json.dumps(serialize_settings_to_flat_dict(), ensure_ascii=False, indent=2)
        st.download_button(
            "下载当前配置 JSON",
            blob,
            file_name="aicheckword-runtime-settings.json",
            mime="application/json",
            key="sys_cfg_dl_json",
        )

    st.markdown("---")
    st.caption("调试：当前内存中的配置摘要。")
    with st.expander("查看序列化 JSON 预览"):
        st.code(json.dumps(serialize_settings_to_flat_dict(), ensure_ascii=False, indent=2), language="json")


def render_system_config_page() -> None:
    st.header("⚙️ 系统配置")
    # 每次从其他页进入本页时由 main() 将 _sys_cfg_in_page 置 False，此处把 settings 全量灌入表单
    if not st.session_state.get("_sys_cfg_in_page", False):
        _push_settings_to_widgets()
        st.session_state["_sys_cfg_in_page"] = True
        st.session_state.pop("_sys_cfg_provider_form_snap", None)
    _ensure_defaults_from_settings()
    render_system_config_body()
