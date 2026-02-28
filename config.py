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

    # 应用内部使用的 SQLite 数据库
    db_path: str = "./aicheckword.db"

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
