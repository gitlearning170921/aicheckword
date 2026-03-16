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
    chroma_persist_dir: str = "./knowledge_store"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    # 嵌入请求：远程主机强制关闭连接时重试（如 WinError 10054）
    embedding_max_retries: int = 3
    embedding_retry_delay_sec: float = 2.0
    # 大文件训练：块数超过阈值时自动缩小每批数量，减少单次请求超时/断连
    embedding_large_file_threshold: int = 60
    embedding_large_file_batch_size: int = 12

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
    def training_path(self) -> Path:
        return Path(self.training_docs_dir)

    @property
    def uploads_path(self) -> Path:
        return Path(self.uploads_dir)

    @property
    def db_file(self) -> Path:
        return Path(self.db_path)


settings = Settings()
