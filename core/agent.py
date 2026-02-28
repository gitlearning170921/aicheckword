"""Agent 封装：将审核能力封装为可复用的 Agent，供其他项目调用"""

import json
from pathlib import Path
from typing import List, Optional

from core.knowledge_base import KnowledgeBase
from core.reviewer import DocumentReviewer, AuditReport


class ReviewAgent:
    """
    注册文档审核 Agent

    延迟初始化：创建时不连接 OpenAI，只在真正需要时才建立连接。
    get_status / clear_knowledge 等不需要 API Key 即可使用。
    """

    def __init__(self, collection_name: str = "regulations"):
        self.collection_name = collection_name
        self.kb = KnowledgeBase(collection_name)
        self._reviewer: Optional[DocumentReviewer] = None

    @property
    def reviewer(self) -> DocumentReviewer:
        if self._reviewer is None:
            self._reviewer = DocumentReviewer(knowledge_base=self.kb)
        return self._reviewer

    def reset_clients(self):
        """API Key 变更时重置所有客户端"""
        self.kb.reset_clients()
        if self._reviewer is not None:
            self._reviewer.reset_client()
        self._reviewer = None

    def train(self, file_path: str) -> dict:
        path = Path(file_path)
        if path.is_dir():
            count = self.kb.train_from_directory(path)
            return {"status": "success", "chunks_added": count, "source": str(path)}
        elif path.is_file():
            count = self.kb.train_from_file(path)
            return {"status": "success", "chunks_added": count, "source": path.name}
        else:
            return {"status": "error", "message": f"路径不存在：{file_path}"}

    def train_batch(self, file_paths: List[str]) -> List[dict]:
        results = []
        for fp in file_paths:
            result = self.train(fp)
            results.append(result)
        return results

    def review(self, file_path: str) -> dict:
        report = self.reviewer.review_file(file_path)
        return report.to_dict()

    def review_text(self, text: str, file_name: str = "直接输入") -> dict:
        report = self.reviewer.review_text(text, file_name)
        return report.to_dict()

    def review_batch(self, file_paths: List[str]) -> List[dict]:
        reports = self.reviewer.review_multiple_files(file_paths)
        return [r.to_dict() for r in reports]

    def search_knowledge(self, query: str, top_k: int = 5) -> List[dict]:
        docs = self.kb.search(query, top_k=top_k)
        return [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source_file", "未知"),
                "metadata": doc.metadata,
            }
            for doc in docs
        ]

    def get_status(self) -> dict:
        """获取状态（不需要 API Key）"""
        stats = self.kb.get_collection_stats()
        return {
            "agent_name": "注册文档审核Agent",
            "collection_name": self.collection_name,
            "knowledge_base": stats,
            "capabilities": [
                "文档训练（PDF/Word/Excel/TXT/Markdown）",
                "单文件审核",
                "批量文件审核",
                "文本审核",
                "知识库查询",
            ],
        }

    def clear_knowledge(self) -> dict:
        """清空知识库（不需要 API Key）"""
        self.kb.clear()
        return {"status": "success", "message": "知识库已清空"}

    def export_report(self, report: dict, output_path: str) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return str(path)
