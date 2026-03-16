"""知识库管理：基于 ChromaDB 的向量存储与检索，支持 Ollama (本地) 和 OpenAI 两种 Embedding
   同时将文档块持久化到 MySQL，重启服务不丢失记录。
"""

import time
from pathlib import Path
from typing import List, Optional

import chromadb
from langchain.schema import Document
from langchain_community.vectorstores import Chroma

from config import settings
from .document_loader import load_and_split, load_and_split_directory
from .db import (
    save_knowledge_docs,
    clear_knowledge_docs,
    save_project_knowledge_docs,
    clear_project_knowledge_docs,
    delete_project_knowledge_docs_by_file,
    save_checkpoint_docs,
    clear_checkpoint_docs,
    delete_checkpoint_docs_by_file,
    delete_knowledge_docs_by_file,
)


def _add_batch_with_retry(vectorstore: Chroma, batch: List[Document]) -> None:
    """单批向量化并入库，遇连接被关闭等错误时重试（如 WinError 10054 / httpcore.ReadError）。"""
    max_retries = max(1, getattr(settings, "embedding_max_retries", 3))
    delay = max(0.5, getattr(settings, "embedding_retry_delay_sec", 2.0))
    last_err = None
    for attempt in range(max_retries):
        try:
            vectorstore.add_documents(batch)
            return
        except (ConnectionError, OSError) as e:
            last_err = e
        except Exception as e:
            mod = getattr(type(e), "__module__", "") or ""
            retryable = (
                "10054" in str(e)
                or "ReadError" in type(e).__name__
                or "ConnectError" in type(e).__name__
                or "RemoteProtocolError" in type(e).__name__
                or mod.startswith("httpcore")
                or mod.startswith("httpx")
            )
            if retryable:
                last_err = e
            else:
                raise
        if attempt < max_retries - 1:
            time.sleep(delay * (attempt + 1))
    if last_err is not None:
        raise last_err


def _create_embeddings():
    # DeepSeek/零一 不提供 /v1/embeddings，若用其 base_url 会 404；统一用 Ollama 做向量化
    # 优先用 session 中的当前 provider（Streamlit 侧栏选择），否则用 settings.provider
    try:
        import streamlit as _st
        p = (_st.session_state.get("current_provider") or settings.provider or "").strip().lower()
    except Exception:
        p = (settings.provider or "").strip().lower()
    use_ollama = (
        settings.is_ollama
        or (settings.is_cursor and (settings.cursor_embedding or "").lower() == "ollama")
        or p in ("deepseek", "lingyi")
    )
    # 防御：若即将用 OpenAIEmbeddings 且 base_url 为 DeepSeek（会 404），则改用 Ollama；排除 Cursor 以不影响其「向量化=openai」的用法
    if not use_ollama and ("deepseek.com" in (settings.openai_base_url or "")) and p != "cursor":
        use_ollama = True
    if use_ollama:
        from langchain_ollama import OllamaEmbeddings
        from config.cursor_overrides import get_llm_verify_ssl, get_llm_trust_env
        client_kwargs = {"verify": get_llm_verify_ssl(), "trust_env": get_llm_trust_env()}
        return OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.ollama_base_url,
            client_kwargs=client_kwargs,
        )
    else:
        from langchain_openai import OpenAIEmbeddings
        from config.cursor_overrides import get_llm_verify_ssl, get_llm_trust_env
        import httpx
        if not settings.openai_api_key:
            raise RuntimeError("请配置 OpenAI API Key（用于向量化或 OpenAI 模式）")
        http_client = httpx.Client(verify=get_llm_verify_ssl(), trust_env=get_llm_trust_env())
        return OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
            openai_api_base=settings.openai_base_url,
            http_client=http_client,
        )


def _get_chroma_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(settings.chroma_path))


class KnowledgeBase:
    def __init__(
        self,
        collection_name: str = "regulations",
        project_id: Optional[int] = None,
        base_collection: str = "",
        is_checkpoint: bool = False,
    ):
        self.collection_name = collection_name
        self.project_id = project_id
        self.base_collection = base_collection or (
            collection_name.split("_project_")[0] if "_project_" in collection_name else collection_name
        )
        self.is_checkpoint = is_checkpoint
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

    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 50,
        file_name: str = "",
        category: str = "regulation",
        case_id: Optional[int] = None,
    ) -> int:
        if not documents:
            return 0
        for doc in documents:
            if not hasattr(doc, "metadata") or doc.metadata is None:
                doc.metadata = {}
            doc.metadata["source_file"] = file_name or ""
            if not self.project_id and not self.is_checkpoint:
                doc.metadata["category"] = category
                if category == "project_case" and case_id is not None:
                    doc.metadata["case_id"] = case_id
        total = len(documents)
        threshold = getattr(settings, "embedding_large_file_threshold", 60)
        max_batch = getattr(settings, "embedding_large_file_batch_size", 12)
        effective_batch_size = min(batch_size, max_batch) if total > threshold else batch_size
        for i in range(0, total, effective_batch_size):
            batch = documents[i:i + effective_batch_size]
            _add_batch_with_retry(self.vectorstore, batch)
        if self.project_id:
            save_project_knowledge_docs(self.project_id, self.base_collection, file_name, documents)
        elif self.is_checkpoint:
            save_checkpoint_docs(self.base_collection, file_name, documents)
        else:
            save_knowledge_docs(self.collection_name, file_name, documents, category=category)
        return len(documents)

    def add_documents_with_progress(
        self,
        documents: List[Document],
        batch_size: int = 50,
        callback=None,
        file_name: str = "",
        category: str = "regulation",
        case_id: Optional[int] = None,
    ) -> int:
        if not documents:
            return 0
        for doc in documents:
            if not hasattr(doc, "metadata") or doc.metadata is None:
                doc.metadata = {}
            doc.metadata["source_file"] = file_name or ""
            if not self.project_id and not self.is_checkpoint:
                doc.metadata["category"] = category
                if category == "project_case" and case_id is not None:
                    doc.metadata["case_id"] = case_id
        total = len(documents)
        done = 0
        threshold = getattr(settings, "embedding_large_file_threshold", 60)
        max_batch = getattr(settings, "embedding_large_file_batch_size", 12)
        effective_batch_size = min(batch_size, max_batch) if total > threshold else batch_size
        for i in range(0, total, effective_batch_size):
            batch = documents[i:i + effective_batch_size]
            _add_batch_with_retry(self.vectorstore, batch)
            done += len(batch)
            if callback:
                callback(done, total)
        if self.project_id:
            save_project_knowledge_docs(self.project_id, self.base_collection, file_name, documents)
        elif self.is_checkpoint:
            save_checkpoint_docs(self.base_collection, file_name, documents)
        else:
            save_knowledge_docs(self.collection_name, file_name, documents, category=category)
        return total

    def train_from_file(self, file_path, category: str = "regulation") -> int:
        chunks = load_and_split(file_path)
        return self.add_documents(chunks, file_name=str(file_path), category=category)

    def train_from_directory(self, dir_path, category: str = "regulation") -> int:
        chunks = load_and_split_directory(dir_path)
        return self.add_documents(chunks, file_name=str(dir_path), category=category)

    def search(self, query: str, top_k: int = 10) -> List[Document]:
        return self.vectorstore.similarity_search(query, k=top_k)

    def search_by_category(self, query: str, category: str, top_k: int = 10) -> List[Document]:
        """按分类检索（如 category='glossary' 检索词条）。仅对主知识库生效；项目库/审核点库无 category 过滤时退回普通检索。"""
        if self.project_id or self.is_checkpoint:
            return self.vectorstore.similarity_search(query, k=top_k)
        try:
            return self.vectorstore.similarity_search(query, k=top_k, filter={"category": category})
        except TypeError:
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
        try:
            if self.project_id:
                clear_project_knowledge_docs(self.project_id)
            elif self.is_checkpoint:
                clear_checkpoint_docs(self.base_collection)
            else:
                clear_knowledge_docs(self.collection_name)
        except Exception:
            pass

    def list_collections(self) -> List[str]:
        client = _get_chroma_client()
        collections = client.list_collections()
        if not collections:
            return []
        if isinstance(collections[0], str):
            return collections
        return [c.name for c in collections]

    def delete_documents_by_file_name(self, file_name: str, case_id: Optional[int] = None) -> None:
        """按文件名删除该知识库下对应文档的所有块（Chroma + MySQL），用于覆盖前清理。
        当 case_id 有值时仅删除该案例下的记录（项目案例覆盖时用）。"""
        try:
            if case_id is not None:
                self.vectorstore._collection.delete(where={"source_file": file_name, "case_id": case_id})
            else:
                self.vectorstore._collection.delete(where={"source_file": file_name})
        except Exception:
            pass
        if self.project_id:
            delete_project_knowledge_docs_by_file(self.project_id, file_name)
        elif self.is_checkpoint:
            delete_checkpoint_docs_by_file(self.base_collection, file_name)
        else:
            delete_knowledge_docs_by_file(self.collection_name, file_name, case_id=case_id)
