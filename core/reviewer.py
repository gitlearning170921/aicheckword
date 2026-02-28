"""AI 审核引擎：基于 RAG 的注册文档审核，支持 Ollama、OpenAI、Cursor Cloud Agents"""

import json
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field, asdict

from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate

from config import settings
from core.knowledge_base import KnowledgeBase
from core.document_loader import load_single_file

def _create_llm():
    """根据 provider 创建对应的 LLM 实例（仅 ollama/openai，cursor 不走 LLM）"""
    if settings.is_ollama:
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            temperature=0.1,
        )
    else:
        from langchain_openai import ChatOpenAI
        if not settings.openai_api_key:
            raise RuntimeError("OpenAI 模式下请先配置 API Key")
        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0.1,
        )


@dataclass
class AuditPoint:
    """单个审核点"""
    category: str
    severity: str
    location: str
    description: str
    regulation_ref: str
    suggestion: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AuditReport:
    """审核报告"""
    file_name: str
    total_points: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    info_count: int = 0
    audit_points: List[AuditPoint] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "file_name": self.file_name,
            "total_points": self.total_points,
            "high_count": self.high_count,
            "medium_count": self.medium_count,
            "low_count": self.low_count,
            "info_count": self.info_count,
            "summary": self.summary,
            "audit_points": [p.to_dict() for p in self.audit_points],
        }


REVIEW_SYSTEM_PROMPT = """你是一位资深的注册文档审核专家。你的职责是根据已知的法规、标准和项目文件，
对提交的注册文档进行严格审核，找出所有不符合要求的地方。

你具备以下能力：
1. 熟悉相关法规和标准要求
2. 能够识别文档中的合规性问题
3. 能够检查文档的完整性和一致性
4. 能够给出专业的修改建议

审核时请关注以下维度：
- **合规性**：是否符合相关法规、标准要求
- **完整性**：必要信息是否齐全，是否有遗漏
- **一致性**：文档内部数据和表述是否前后一致
- **准确性**：技术参数、数据引用是否准确
- **格式规范**：文档格式是否符合要求"""

REVIEW_USER_PROMPT = """请根据以下参考知识对待审核文档进行审核。

## 参考知识（法规/标准/项目文件）

{context}

## 待审核文档内容

文件名：{file_name}

{document_content}

## 审核要求

请逐项审核以上文档，输出所有审核发现。每个审核点必须严格按以下 JSON 格式输出：

```json
[
  {{
    "category": "审核类别（合规性/完整性/一致性/准确性/格式规范）",
    "severity": "严重程度（high/medium/low/info）",
    "location": "问题所在位置",
    "description": "问题详细描述",
    "regulation_ref": "对应的法规或标准条款引用",
    "suggestion": "具体修改建议"
  }}
]
```

请确保：
1. 仅输出 JSON 数组，不要输出其他内容
2. 至少检查合规性、完整性、一致性三个维度
3. 每个问题都要给出明确的法规依据和修改建议
4. 如果文档整体合规，也请输出 info 级别的确认信息"""

SUMMARY_PROMPT = """请根据以下审核发现，生成一段简洁的审核总结（200字以内）：

文件名：{file_name}
审核发现数量：高风险 {high} 个，中风险 {medium} 个，低风险 {low} 个，提示 {info} 个

审核详情：
{details}

请用中文输出总结。"""

# Cursor Agent 使用：单条任务提示（不修改仓库，仅输出 JSON）
CURSOR_REVIEW_TASK = """你是一位资深的注册文档审核专家。请根据下面的参考知识和待审核文档内容，仅在你的回复中输出一个 JSON 数组，不要修改任何代码或文件。

## 参考知识（法规/标准/项目文件）
{context}

## 待审核文档
文件名：{file_name}

{document_content}

## 输出要求
请逐项审核以上文档，仅输出一个 JSON 数组，格式如下，不要其他说明或 markdown 标记：
[{{"category":"合规性|完整性|一致性|准确性|格式规范","severity":"high|medium|low|info","location":"位置","description":"描述","regulation_ref":"法规引用","suggestion":"修改建议"}}]
"""

CURSOR_SUMMARY_TASK = """请根据以下审核发现，用中文生成一段简洁的审核总结（200字以内）。不要修改任何文件，仅输出总结文字。

文件名：{file_name}
审核发现：高风险 {high} 个，中风险 {medium} 个，低风险 {low} 个，提示 {info} 个

审核详情：
{details}
"""


class DocumentReviewer:
    """文档审核器，延迟初始化 LLM 客户端"""

    def __init__(
        self,
        knowledge_base: Optional[KnowledgeBase] = None,
        collection_name: str = "regulations",
    ):
        self.kb = knowledge_base or KnowledgeBase(collection_name)
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = _create_llm()
        return self._llm

    def reset_client(self):
        self._llm = None

    def _retrieve_context(self, document_text: str, top_k: int = 15) -> str:
        results = self.kb.search(document_text[:2000], top_k=top_k)
        if not results:
            return "（暂无相关参考知识，请根据通用法规标准进行审核）"

        context_parts = []
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source_file", "未知来源")
            context_parts.append(f"【参考{i}】来源：{source}\n{doc.page_content}")

        return "\n\n".join(context_parts)

    def _parse_audit_points(self, llm_response: str) -> List[AuditPoint]:
        text = llm_response.strip()

        json_start = text.find("[")
        json_end = text.rfind("]") + 1
        if json_start == -1 or json_end == 0:
            return [AuditPoint(
                category="解析错误",
                severity="info",
                location="N/A",
                description=f"LLM 响应无法解析为结构化数据：{text[:200]}",
                regulation_ref="N/A",
                suggestion="请重新审核",
            )]

        try:
            data = json.loads(text[json_start:json_end])
        except json.JSONDecodeError:
            return [AuditPoint(
                category="解析错误",
                severity="info",
                location="N/A",
                description=f"JSON 解析失败：{text[json_start:json_start+200]}",
                regulation_ref="N/A",
                suggestion="请重新审核",
            )]

        points = []
        for item in data:
            points.append(AuditPoint(
                category=item.get("category", "未分类"),
                severity=item.get("severity", "info"),
                location=item.get("location", "未知"),
                description=item.get("description", ""),
                regulation_ref=item.get("regulation_ref", ""),
                suggestion=item.get("suggestion", ""),
            ))
        return points

    def review_text(self, text: str, file_name: str = "未命名文档") -> AuditReport:
        context = self._retrieve_context(text)

        if settings.is_cursor:
            from core.cursor_agent import complete_task
            prompt_text = CURSOR_REVIEW_TASK.format(
                context=context,
                file_name=file_name,
                document_content=text,
            )
            response_content = complete_task(prompt_text)
            audit_points = self._parse_audit_points(response_content)
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", REVIEW_SYSTEM_PROMPT),
                ("human", REVIEW_USER_PROMPT),
            ])
            chain = prompt | self.llm
            response = chain.invoke({
                "context": context,
                "file_name": file_name,
                "document_content": text,
            })
            audit_points = self._parse_audit_points(response.content)

        report = AuditReport(file_name=file_name)
        report.audit_points = audit_points
        report.total_points = len(audit_points)
        report.high_count = sum(1 for p in audit_points if p.severity == "high")
        report.medium_count = sum(1 for p in audit_points if p.severity == "medium")
        report.low_count = sum(1 for p in audit_points if p.severity == "low")
        report.info_count = sum(1 for p in audit_points if p.severity == "info")

        report.summary = self._generate_summary(report)
        return report

    def review_file(self, file_path) -> AuditReport:
        path = Path(file_path)
        docs = load_single_file(path)
        full_text = "\n\n".join(doc.page_content for doc in docs)

        if len(full_text) > 30000:
            return self._review_long_document(full_text, path.name)

        return self.review_text(full_text, path.name)

    def _review_long_document(self, text: str, file_name: str) -> AuditReport:
        chunk_size = 25000
        overlap = 2000
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start = end - overlap

        all_points = []
        for i, chunk in enumerate(chunks):
            chunk_name = f"{file_name} (第{i+1}/{len(chunks)}段)"
            report = self.review_text(chunk, chunk_name)
            all_points.extend(report.audit_points)

        final_report = AuditReport(file_name=file_name)
        final_report.audit_points = all_points
        final_report.total_points = len(all_points)
        final_report.high_count = sum(1 for p in all_points if p.severity == "high")
        final_report.medium_count = sum(1 for p in all_points if p.severity == "medium")
        final_report.low_count = sum(1 for p in all_points if p.severity == "low")
        final_report.info_count = sum(1 for p in all_points if p.severity == "info")
        final_report.summary = self._generate_summary(final_report)
        return final_report

    def review_multiple_files(self, file_paths: List[str]) -> List[AuditReport]:
        reports = []
        for fp in file_paths:
            try:
                report = self.review_file(fp)
                reports.append(report)
            except Exception as e:
                error_report = AuditReport(file_name=str(fp))
                error_report.audit_points = [AuditPoint(
                    category="系统错误",
                    severity="high",
                    location=str(fp),
                    description=f"文件审核失败：{str(e)}",
                    regulation_ref="N/A",
                    suggestion="请检查文件格式是否正确",
                )]
                error_report.total_points = 1
                error_report.high_count = 1
                error_report.summary = f"文件审核失败：{str(e)}"
                reports.append(error_report)
        return reports

    def _generate_summary(self, report: AuditReport) -> str:
        if not report.audit_points:
            return "未发现审核问题。"

        details = "\n".join(
            f"- [{p.severity}] {p.category}: {p.description}"
            for p in report.audit_points[:10]
        )

        if settings.is_cursor:
            from core.cursor_agent import complete_task
            prompt_text = CURSOR_SUMMARY_TASK.format(
                file_name=report.file_name,
                high=report.high_count,
                medium=report.medium_count,
                low=report.low_count,
                info=report.info_count,
                details=details,
            )
            return complete_task(prompt_text).strip()

        prompt = ChatPromptTemplate.from_messages([
            ("human", SUMMARY_PROMPT),
        ])
        chain = prompt | self.llm
        response = chain.invoke({
            "file_name": report.file_name,
            "high": report.high_count,
            "medium": report.medium_count,
            "low": report.low_count,
            "info": report.info_count,
            "details": details,
        })
        return response.content
