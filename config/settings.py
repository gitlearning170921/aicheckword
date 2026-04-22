"""配置包：应用与 AI 服务配置"""
import json
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # AI 服务提供方:
    # ollama | openai | cursor | gemini | tongyi | baidu | lingyi | deepseek
    provider: str = "ollama"

    # Google Gemini（provider=gemini）
    gemini_api_key: str = ""  # 或 GOOGLE_API_KEY
    google_api_key: str = ""  # 兼容 GOOGLE_API_KEY

    # 阿里通义 DashScope（provider=tongyi）
    dashscope_api_key: str = ""

    # 百度文心千帆（provider=baidu）
    qianfan_ak: str = ""
    qianfan_sk: str = ""

    # 零一万物 OpenAI 兼容（provider=lingyi）
    lingyi_api_key: str = ""
    lingyi_base_url: str = "https://api.lingyiwanwu.com/v1"

    # DeepSeek OpenAI 兼容（provider=deepseek）
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"

    # OpenAI 配置 (provider=openai 或兼容服务)
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"

    # Ollama 配置 (provider=ollama 时使用, 本地运行无需 Key)
    ollama_base_url: str = "http://localhost:11434"

    # 所有 AI 服务通用：HTTP 请求行为（侧栏「不校验 SSL」「不使用系统代理」）
    llm_verify_ssl: bool = True  # False 时不校验 SSL（代理/证书异常时可勾选）
    llm_trust_env: bool = True  # False 时不使用系统代理，直连 API（代理导致 SSL EOF 时可勾选）

    # Cursor Cloud Agents API (provider=cursor 时使用)
    cursor_api_key: str = ""
    cursor_api_base: str = "https://api.cursor.com"
    cursor_repository: str = ""
    cursor_ref: str = "main"
    cursor_embedding: str = "ollama"
    cursor_verify_ssl: bool = True  # 已废弃，请用 llm_verify_ssl
    cursor_trust_env: bool = True  # 已废弃，请用 llm_trust_env

    # 模型配置
    llm_model: str = "qwen2.5"
    # 扫描件/图片型 PDF：AI OCR 多模态模型（可与 llm_model 不同；也可用系统配置页写入库覆盖）。留空则回退为 llm_model。
    pdf_ocr_llm_model: str = ""
    embedding_model: str = "nomic-embed-text"
    # 侧栏「按服务提供方」独立保存的模型等（JSON 字符串，键为 provider）；与全局 llm_model 等双写，切换提供方时各自恢复。
    provider_sidebar_presets: str = "{}"

    # 向量库与分块
    # 本地模式：数据在 chroma_persist_dir。多机共享：在一台机器上部署 Chroma Server，各客户端填写 chroma_server_host（非空则走 HTTP，忽略本地目录作为存储位置）。
    chroma_persist_dir: str = "./knowledge_store"
    chroma_server_host: str = ""  # 例：10.0.0.5 或 chroma.internal；留空=仅用本地 PersistentClient
    chroma_server_port: int = 8000
    chroma_server_ssl: bool = False  # True 时 HTTPS
    chroma_server_headers_json: str = ""  # 可选，JSON：{"Authorization":"Bearer ..."}，用于自建网关鉴权
    chunk_size: int = 1000
    chunk_overlap: int = 200
    # 嵌入请求：远程主机强制关闭连接时重试（如 WinError 10054）
    embedding_max_retries: int = 3
    embedding_retry_delay_sec: float = 2.0
    # 大文件训练：块数超过阈值时自动缩小每批数量，减少单次请求超时/断连
    embedding_large_file_threshold: int = 60
    embedding_large_file_batch_size: int = 12
    # 单文件分块数超过此阈值时，按多轮 Streamlit 重跑分段向量化，减轻长时间阻塞导致的浏览器 WebSocket 断开/「训练自动取消」
    embedding_streamlit_chunk_threshold: int = 48
    # 每一轮 Streamlit 脚本执行最多完成多少个「嵌入批次」（每批 chunk 数见 embedding_large_file_*）；默认 1 最稳
    embedding_streamlit_batches_per_rerun: int = 1

    # ─── 文档审核稳定性（DeepSeek 等云端 API，多文件时降低 Chroma/内存与重复 LLM 调用）───
    # 两次 LLM（Chat）请求最小间隔（秒）。仅对 provider=deepseek 生效；0=使用内置默认约 0.9；-1=关闭
    review_llm_min_interval_sec: float = 0.0
    # 单次审核从 Chroma 取回的候选条数上限（仅 deepseek 生效，减轻检索与拼接上下文体积）
    review_deepseek_chroma_fetch_cap: int = 96
    # 实际拼入提示的审核点参考条数上限（deepseek）
    review_deepseek_target_cap: int = 64
    # 批量审核（多文件）时跳过第二次「总结」模型调用，改用规则摘要（每文件少 1 次 API）
    review_batch_skip_llm_summary: bool = True
    # 批量审核时相邻两个文件之间的额外间隔（秒），仅 deepseek；0=使用内置默认 1.0
    review_deepseek_inter_doc_sleep_sec: float = 0.0
    # 批量审核（≥2 文件）时，任意提供方在「下一文件」前的最小间隔（秒），减轻网关限流与叠峰导致的断连；0=关闭
    review_batch_inter_doc_sleep_sec: float = 0.35

    # 审核报告：性能与体验（环境变量 AUDIT_PERF_LOG=1 可强制开启分段日志）
    audit_perf_log: bool = False
    # 纠正入库写入反馈向量库时是否在后台线程执行（True=先返回 UI，向量异步写入）
    async_correction_kb_feed: bool = True

    # API 服务
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # 文件目录
    training_docs_dir: str = "./training_docs"
    uploads_dir: str = "./uploads"

    # MySQL 数据库（所有配置与操作记录均存于此）
    mysql_host: str = "10.26.1.221"
    mysql_port: int = 13306
    mysql_database: str = "aicheckword"
    mysql_user: str = "root"
    mysql_password: str = "mysql170921"
    mysql_charset: str = "utf8mb4"

    # 兼容旧配置项（已废弃，仅保留避免报错）
    db_path: str = "./aicheckword.db"

    # 金山文档开放平台（审核时通过链接拉取正文；需在 https://developer.kdocs.cn 创建应用）
    kdocs_app_id: str = ""
    kdocs_app_key: str = ""

    # 兼容历史部署：优先读取 .env；若只有 .env.txt 也可自动加载
    model_config = {"env_file": (".env", ".env.txt"), "env_file_encoding": "utf-8"}

    @property
    def is_ollama(self) -> bool:
        return self.provider.lower() == "ollama"

    @property
    def is_cursor(self) -> bool:
        return self.provider.lower() == "cursor"

    @property
    def is_openai_compatible(self) -> bool:
        """是否使用 OpenAI 兼容 Chat 接口（含 DeepSeek / 零一万物）。"""
        p = self.provider.lower()
        return p in ("openai", "deepseek", "lingyi")

    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_persist_dir)

    @property
    def chroma_use_remote_server(self) -> bool:
        """是否使用远程 Chroma HTTP 服务（多机共享同一向量库）。"""
        return bool((self.chroma_server_host or "").strip())

    @property
    def training_path(self) -> Path:
        return Path(self.training_docs_dir)

    @property
    def uploads_path(self) -> Path:
        return Path(self.uploads_dir)

    @property
    def db_file(self) -> Path:
        return Path(self.db_path)


settings = Settings()


def pdf_ocr_llm_model_field_available() -> bool:
    """当前 Settings 模型是否定义 pdf_ocr_llm_model（旧部署/未合并代码时避免 Pydantic 读写崩溃）。"""
    return "pdf_ocr_llm_model" in Settings.model_fields


def get_pdf_ocr_llm_model() -> str:
    if not pdf_ocr_llm_model_field_available():
        return ""
    return str(settings.pdf_ocr_llm_model or "").strip()


def set_pdf_ocr_llm_model(value: str) -> None:
    if not pdf_ocr_llm_model_field_available():
        return
    settings.pdf_ocr_llm_model = str(value or "").strip()


def parse_provider_sidebar_presets() -> Dict[str, Any]:
    """解析 provider_sidebar_presets JSON；失败或非法时返回空 dict。"""
    raw = getattr(settings, "provider_sidebar_presets", None)
    s = (raw if isinstance(raw, str) else str(raw or "")).strip()
    if not s:
        return {}
    try:
        o = json.loads(s)
        return o if isinstance(o, dict) else {}
    except Exception:
        return {}


def dump_provider_sidebar_presets(data: Dict[str, Any]) -> str:
    return json.dumps(data or {}, ensure_ascii=False)


def get_provider_sidebar_slot(provider: str) -> Dict[str, Any]:
    p = (provider or "").strip().lower()
    slot = parse_provider_sidebar_presets().get(p)
    return dict(slot) if isinstance(slot, dict) else {}


def upsert_provider_sidebar_slot(provider: str, updates: Dict[str, Any]) -> None:
    """合并写入某一提供方的侧栏预设（内存 settings，入库依赖 persist_settings_dual_write / 全量 JSON）。"""
    p = (provider or "").strip().lower()
    d = parse_provider_sidebar_presets()
    cur = dict(d.get(p) or {})
    for k, v in updates.items():
        cur[k] = v
    d[p] = cur
    settings.provider_sidebar_presets = dump_provider_sidebar_presets(d)


def canonical_openai_form_base_url(provider: str) -> str:
    pv = (provider or "").strip().lower()
    if pv == "deepseek":
        return "https://api.deepseek.com/v1"
    if pv == "lingyi":
        return "https://api.lingyiwanwu.com/v1"
    return "https://api.openai.com/v1"


def sanitize_openai_form_base_url(provider: str, url: str) -> str:
    """OpenAI / DeepSeek / 零一侧栏 Base URL：与提供方主机明显不一致时回退官方默认，避免串列。"""
    u = (url or "").strip()
    pv = (provider or "").strip().lower()
    if not u:
        return canonical_openai_form_base_url(pv)
    try:
        host = (urlparse(u).hostname or "").lower()
    except Exception:
        return u
    if pv == "openai":
        if "deepseek" in host or "lingyiwanwu" in host:
            return canonical_openai_form_base_url("openai")
        return u
    if pv == "deepseek":
        if host.endswith("openai.com") and "azure" not in host:
            return canonical_openai_form_base_url("deepseek")
        return u
    if pv == "lingyi":
        if "deepseek" in host or (host.endswith("openai.com") and "azure" not in host):
            return canonical_openai_form_base_url("lingyi")
        return u
    return u


def openai_form_base_url_default_from_settings(provider: str) -> str:
    """从全局 settings 各列取该提供方默认 Base URL，并做 sanitize。"""
    pv = (provider or "").strip().lower()
    if pv == "deepseek":
        raw = (settings.deepseek_base_url or canonical_openai_form_base_url("deepseek")).strip()
        return sanitize_openai_form_base_url("deepseek", raw)
    if pv == "lingyi":
        raw = (settings.lingyi_base_url or canonical_openai_form_base_url("lingyi")).strip()
        return sanitize_openai_form_base_url("lingyi", raw)
    raw = (settings.openai_base_url or canonical_openai_form_base_url("openai")).strip()
    return sanitize_openai_form_base_url("openai", raw)


def repair_provider_sidebar_presets_urls() -> None:
    """修正 JSON 预设里 openai/deepseek/lingyi 槽位中串错的 base_url（如 OpenAI 槽存了 DeepSeek 地址）。"""
    d = parse_provider_sidebar_presets()
    if not d:
        return
    changed = False
    for p in ("openai", "deepseek", "lingyi"):
        slot = d.get(p)
        if not isinstance(slot, dict):
            continue
        bu = (slot.get("base_url") or "").strip()
        if not bu:
            continue
        fixed = sanitize_openai_form_base_url(p, bu)
        if fixed != bu:
            slot["base_url"] = fixed
            d[p] = slot
            changed = True
    if changed:
        settings.provider_sidebar_presets = dump_provider_sidebar_presets(d)
    # 与兼容列对齐：预设槽位纠正后写回全局，避免下次 save_app_settings 把旧列又写进库
    d2 = parse_provider_sidebar_presets()
    for p, attr in (
        ("openai", "openai_base_url"),
        ("deepseek", "deepseek_base_url"),
        ("lingyi", "lingyi_base_url"),
    ):
        slot = d2.get(p)
        if not isinstance(slot, dict):
            continue
        bu = (slot.get("base_url") or "").strip()
        if not bu:
            continue
        setattr(settings, attr, sanitize_openai_form_base_url(p, bu))


def maybe_seed_provider_sidebar_presets_from_legacy() -> None:
    """旧库无预设时：OpenAI/DeepSeek/零一各占一条且 base_url 来自各自全局列；其余提供方写当前 provider 一条。"""
    if parse_provider_sidebar_presets():
        return
    p0 = (settings.provider or "ollama").strip().lower()
    for p in ("openai", "deepseek", "lingyi"):
        sub: Dict[str, Any] = {"base_url": openai_form_base_url_default_from_settings(p)}
        if p == p0:
            sub["llm_model"] = (settings.llm_model or "").strip()
            sub["embedding_model"] = (settings.embedding_model or "").strip()
            if pdf_ocr_llm_model_field_available():
                sub["pdf_ocr_llm_model"] = (get_pdf_ocr_llm_model() or "").strip()
        upsert_provider_sidebar_slot(p, sub)
    if p0 == "ollama":
        slot_o: Dict[str, Any] = {
            "ollama_base_url": (settings.ollama_base_url or "http://localhost:11434").strip(),
            "llm_model": (settings.llm_model or "").strip(),
            "embedding_model": (settings.embedding_model or "").strip(),
        }
        if pdf_ocr_llm_model_field_available():
            slot_o["pdf_ocr_llm_model"] = (get_pdf_ocr_llm_model() or "").strip()
        upsert_provider_sidebar_slot("ollama", slot_o)
    elif p0 in ("gemini", "tongyi", "baidu"):
        slot_g: Dict[str, Any] = {
            "llm_model": (settings.llm_model or "").strip(),
            "embedding_model": (settings.embedding_model or "").strip(),
        }
        if pdf_ocr_llm_model_field_available():
            slot_g["pdf_ocr_llm_model"] = (get_pdf_ocr_llm_model() or "").strip()
        upsert_provider_sidebar_slot(p0, slot_g)
    elif p0 == "cursor":
        upsert_provider_sidebar_slot(
            "cursor",
            {
                "llm_model": (settings.llm_model or "").strip(),
                "embedding_model": (settings.embedding_model or "").strip(),
            },
        )
