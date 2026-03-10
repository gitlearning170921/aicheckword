"""配置包：应用与 AI 服务配置"""
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # AI 服务提供方: "ollama" (本地, 免费) 或 "openai" (需 API Key)
    provider: str = "ollama"

    # OpenAI 配置 (provider=openai 时使用)
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"

    # Ollama 配置 (provider=ollama 时使用, 本地运行无需 Key)
    ollama_base_url: str = "http://localhost:11434"

    # Cursor Cloud Agents API (provider=cursor 时使用)
    cursor_api_key: str = ""
    cursor_api_base: str = "https://api.cursor.com"
    cursor_repository: str = ""
    cursor_ref: str = "main"
    cursor_embedding: str = "ollama"
    cursor_verify_ssl: bool = True  # 代理/证书异常时可设为 False
    cursor_trust_env: bool = True  # False 时不使用系统代理，直连 API（代理导致 SSL EOF 时可设为 False）

    # 模型配置
    llm_model: str = "qwen2.5"
    embedding_model: str = "nomic-embed-text"

    # 向量库与分块
    chroma_persist_dir: str = "./knowledge_store"
    chunk_size: int = 1000
    chunk_overlap: int = 200

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
