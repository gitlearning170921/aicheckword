"""知识库管理：基于 ChromaDB 的向量存储与检索，支持 Ollama (本地) 和 OpenAI 两种 Embedding"""

from pathlib import Path
from typing import List, Optional

import chromadb
from langchain.schema import Document
from langchain_community.vectorstores import Chroma

from config import settings
from core.document_loader import load_and_split, load_and_split_directory


def _create_embeddings():
    """根据 provider（及 cursor 时的 cursor_embedding）创建 Embedding 实例"""
    use_ollama = settings.is_ollama or (settings.is_cursor and (settings.cursor_embedding or "").lower() == "ollama")
    if use_ollama:
        from langchain_community.embeddings import OllamaEmbeddings
        return OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.ollama_base_url,
        )
    else:
        from langchain_openai import OpenAIEmbeddings
        if not settings.openai_api_key:
            raise RuntimeError("请配置 OpenAI API Key（用于向量化或 OpenAI 模式）")
        return OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
            openai_api_base=settings.openai_base_url,
        )


def _get_chroma_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(settings.chroma_path))


class KnowledgeBase:
    """知识库：负责文档入库、检索、管理。延迟初始化 Embeddings。"""

    def __init__(self, collection_name: str = "regulations"):
        self.collection_name = collection_name
        self._embeddings = None
        self._vectorstore: Optional[Chroma] = None

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = _create_embeddings()
        return self._embeddings

    @property
    def vectorstore(self) -> Chroma:
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(settings.chroma_path),
            )
        return self._vectorstore

    def reset_clients(self):
        self._embeddings = None
        self._vectorstore = None

    def add_documents(self, documents: List[Document], batch_size: int = 50) -> int:
        if not documents:
            return 0
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            self.vectorstore.add_documents(batch)
        return len(documents)

    def add_documents_with_progress(self, documents: List[Document], batch_size: int = 50, callback=None) -> int:
        if not documents:
            return 0
        total = len(documents)
        done = 0
        for i in range(0, total, batch_size):
            batch = documents[i:i + batch_size]
            self.vectorstore.add_documents(batch)
            done += len(batch)
            if callback:
                callback(done, total)
        return total

    def train_from_file(self, file_path) -> int:
        chunks = load_and_split(file_path)
        return self.add_documents(chunks)

    def train_from_directory(self, dir_path) -> int:
        chunks = load_and_split_directory(dir_path)
        return self.add_documents(chunks)

    def search(self, query: str, top_k: int = 10) -> List[Document]:
        return self.vectorstore.similarity_search(query, k=top_k)

    def search_with_scores(self, query: str, top_k: int = 10):
        return self.vectorstore.similarity_search_with_relevance_scores(query, k=top_k)

    def get_collection_stats(self) -> dict:
        client = _get_chroma_client()
        try:
            collection = client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "document_count": collection.count(),
            }
        except Exception:
            return {
                "collection_name": self.collection_name,
                "document_count": 0,
            }

    def clear(self):
        client = _get_chroma_client()
        try:
            client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._vectorstore = None

    def list_collections(self) -> List[str]:
        client = _get_chroma_client()
        return [c.name for c in client.list_collections()]
