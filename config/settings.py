"""配置包：应用与 AI 服务配置"""
from pydantic_settings import BaseSettings
from pathlib import Path


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
    embedding_model: str = "nomic-embed-text"

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

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

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
