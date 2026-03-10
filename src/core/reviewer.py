"""AI 审核引擎：基于 RAG 的注册文档审核，支持 Ollama、OpenAI、Cursor Cloud Agents"""

import json
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass, field, asdict

from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate

from config import settings
from .knowledge_base import KnowledgeBase
from .document_loader import load_single_file
from .db import get_dimension_options


def _create_llm():
    if settings.is_ollama:
        from langchain_ollama import ChatOllama
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
    category: str
    severity: str
    location: str
    description: str
    regulation_ref: str
    suggestion: str
    modify_docs: List[str] = field(default_factory=list)  # 多文档审核时：需修改的文档名称列表

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AuditReport:
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


REVIEW_SYSTEM_PROMPT = """你是一位资深的注册文档审核专家。你的职责是根据已训练的审核点清单（包括法规条款、检查方法和合规要求），
对提交的注册文档进行严格审核，逐一对照每个审核点，找出所有不符合要求的地方。

你具备以下能力：
1. 熟悉相关法规和标准要求及对应的审核点
2. 能够识别文档中的合规性问题
3. 能够检查文档的完整性和一致性
4. 能够给出专业的、可操作的修改建议

审核时请关注以下维度：
- **合规性**：是否符合相关法规、标准要求
- **完整性**：必要信息是否齐全，是否有遗漏
- **一致性**：文档内部数据和表述是否前后一致
- **准确性**：技术参数、数据引用是否准确
- **格式规范**：文档格式是否符合要求
- **翻译正确性**（仅英文文档）：是否符合目标国家/语言语法习惯、是否通顺、是否符合逻辑；须结合词条、法规与英文案例参考

**输出要求**：
- 每个问题必须指明**在文档中的具体位置**（如「第3章 适用范围」「第5页 技术指标」或引用有问题的那句话/段落），不得笼统写「文档中」「全文」。
- 每条**修改建议**必须可操作（如「将第2段中的“XXX”改为“YYY”」「在 4.1 节补充……」「删除第6页重复表述」），不得笼统写「请完善」「请补充」。
- 每个审核点只针对一处具体问题，避免一条里混入多个不相关问题。"""

REVIEW_USER_PROMPT = """请根据以下审核点知识对待审核文档进行审核。

## 审核点参考知识（审核点清单 / 法规 / 纠正经验）

{context}

## 待审核文档内容

文件名：{file_name}

{document_content}

## 审核要求

请逐项审核以上文档，输出所有审核发现。每个审核点必须严格按以下 JSON 格式输出：

```json
[
  {{
    "category": "审核类别（合规性/完整性/一致性/准确性/格式规范/翻译正确性，英文文档须含翻译正确性）",
    "severity": "严重程度（high/medium/low/info）",
    "location": "问题在文档中的具体位置（必须具体：如章节号、段落、页码，或引用有问题的那句话，不得写「文档中」「全文」）",
    "description": "问题详细描述",
    "regulation_ref": "对应的法规或标准条款引用",
    "suggestion": "具体可操作的修改建议（必须写明改哪里、改成什么，如「将第2段“XXX”改为“YYY”」，不得笼统写「请完善」）"
  }}
]
```

请确保：
1. 仅输出 JSON 数组，不要输出其他内容
2. **location** 必须指明具体位置（章节/页码/引用原文），不能笼统
3. **suggestion** 必须可操作（写明改哪里、改成什么），不能笼统
4. 至少检查合规性、完整性、一致性三个维度
5. 若上下文提供了【项目专属要求】或本项目名称，须核对待审文档中的项目名称、产品名称、型号规格等与之一致，不一致须作为一致性审核点列出
6. **若待审文档为英文**：须增加「翻译正确性」审核维度，包括：是否符合目标国家/语言语法习惯、是否通顺、是否符合逻辑；审核时请参考上下文中的词条、法规及英文案例，审核点类别可包含「翻译正确性」
7. 如果文档整体合规，也请输出 info 级别的确认信息"""

SUMMARY_PROMPT = """请根据以下审核发现，生成一段简洁的审核总结（200字以内）：

文件名：{file_name}
审核发现数量：高风险 {high} 个，中风险 {medium} 个，低风险 {low} 个，提示 {info} 个

审核详情：
{details}

请用中文输出总结。"""

CURSOR_REVIEW_TASK = """你是一位资深的注册文档审核专家。请根据下面的审核点参考知识和待审核文档内容，仅在你的回复中输出一个 JSON 数组，不要修改任何代码或文件。

## 审核点参考知识
{context}

## 待审核文档
文件名：{file_name}

{document_content}

## 输出要求
1. 仅输出一个 JSON 数组，格式如下，不要其他说明或 markdown 标记：
[{{"category":"合规性|完整性|一致性|准确性|格式规范|翻译正确性（英文文档须含此项）","severity":"high|medium|low|info","location":"在文档中的具体位置（必须具体：章节/页码或引用有问题的那句话，不得写「文档中」「全文」）","description":"问题描述","regulation_ref":"法规引用","suggestion":"可操作的修改建议（必须写明改哪里、改成什么，不得笼统写「请完善」）"}}]
2. **location** 必须指明具体位置（如「第3章」「第5页」「第2段中“……”」），不能笼统。
3. **suggestion** 必须可操作（如「将……改为……」「在 4.1 节补充……」），不能笼统。
4. 若审核点参考知识中包含【项目专属要求】或本项目名称，须核对待审文档中的项目名称、产品名称、型号规格等与之一致，不一致须作为一致性审核点列出。
"""

CURSOR_SUMMARY_TASK = """请根据以下审核发现，用中文生成一段简洁的审核总结（200字以内）。不要修改任何文件，仅输出总结文字。

文件名：{file_name}
审核发现：高风险 {high} 个，中风险 {medium} 个，低风险 {low} 个，提示 {info} 个

审核详情：
{details}
"""

# 多文档一致性与模板风格审核：跨文档信息一致性、模板/风格一致性
MULTI_DOC_CONSISTENCY_PROMPT = """你是一位资深的注册文档审核专家。当前已对多份注册文档完成单文档审核，现需进行**跨文档**的补充审核，重点检查：

## 一、信息一致性
- 项目名称、产品名称、型号规格、注册单元名称在各文档中是否完全一致（含全称/简称、空格、标点）。
- 关键技术指标、性能参数、适用范围、禁忌症等在各文档中是否一致。
- 日期、版本号、引用标准等是否一致。

## 二、模板与风格一致性
- 标题层级（如 1 / 1.1 / 1.1.1）、章节编号方式是否统一。
- 术语用词是否统一（如同一概念在不同文档中是否用同一表述）。
- 格式风格（单位、数字与单位间空格、列表格式等）是否统一。

## 本批文档列表与摘要（下方标题即为各文档名称，输出时请直接使用这些名称，勿用「文档1」「文档2」）
{docs_summary}

## 审核要求
1. **location**：必须用**各文档的真实名称**说明涉及哪些文档或位置（如「[说明书] 与 [技术要求] 中产品名称不一致」），不要写「文档1」「文档2」。
2. **suggestion**：修改建议必须写明**需修改哪一份或哪几份文档**及具体改法（如「在 [说明书] 与 [用户手册] 中统一将“XXX”改为“YYY”」）。
3. 每个审核点增加 **modify_docs** 字段：需修改的文档名称数组，如 ["说明书", "技术要求"]。

请仅输出一个 JSON 数组，不要其他说明或 markdown 标记：
[{{"category":"一致性|格式规范","severity":"high|medium|low|info","location":"用文档名称说明涉及哪些文档或位置，勿用文档1/文档2","description":"问题描述","regulation_ref":"相关法规或标准（可选）","suggestion":"具体可操作的修改建议，并写明需修改哪份或哪几份文档","modify_docs":["文档名称1","文档名称2"]}}]
若各文档间信息与风格均一致，可输出一条 info 级别确认；若存在不一致或风格不统一，逐条列出。"""


class DocumentReviewer:
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

    def _retrieve_context_by_country_keywords(
        self, registration_countries, top_k_per_keyword: int = 5
    ) -> str:
        """按页面选中的注册国家，用「国家→额外关键词」从知识库扩展检索该国家相关法规，扩大审核面。关键词匹配不区分大小写。"""
        if not registration_countries or not self.kb:
            return ""
        countries = (
            registration_countries
            if isinstance(registration_countries, (list, tuple))
            else [registration_countries]
        )
        dims = get_dimension_options()
        country_extra_keywords = dims.get("country_extra_keywords") or {}
        if not isinstance(country_extra_keywords, dict):
            return ""
        key_lower_to_keywords = {
            str(k).strip().lower(): (v if isinstance(v, list) else [v])
            for k, v in country_extra_keywords.items()
            if k
        }
        extra_terms = []
        for c in countries:
            c = (c or "").strip()
            if not c:
                continue
            kws = key_lower_to_keywords.get(c.lower())
            if kws:
                extra_terms.extend(kw for kw in kws if (kw or "").strip())
        if not extra_terms:
            return ""
        seen = set()
        all_docs = []
        for term in extra_terms:
            try:
                results = self.kb.search(term, top_k=top_k_per_keyword)
                for doc in results:
                    key = doc.page_content[:100]
                    if key not in seen:
                        seen.add(key)
                        all_docs.append(doc)
            except Exception:
                continue
        if not all_docs:
            return ""
        parts = [
            "【按注册国家扩展检索的法规/审核点参考】",
            "以下为根据所选注册国家配置的额外关键词（如 CE→MDR）从知识库检索到的相关内容，用于扩大审核面。",
        ]
        for i, doc in enumerate(all_docs[:15], 1):
            source = doc.metadata.get("source_file", "未知来源")
            parts.append(f"【扩展参考{i}】来源：{source}\n{doc.page_content}")
        return "\n\n" + "\n\n".join(parts)

    def _retrieve_glossary_for_translation(self, document_text: str, top_k: int = 8) -> str:
        """检索词条（glossary）内容，供英文文档翻译正确性审核参考。"""
        if not self.kb:
            return ""
        try:
            results = self.kb.search_by_category(document_text[:2000], "glossary", top_k=top_k)
        except Exception:
            return ""
        if not results:
            return ""
        parts = [
            "【翻译正确性审核参考·词条】",
            "以下为知识库中的词条内容，供英文文档的术语、语法与表述审核参考。法规及英文案例见上方审核点参考知识。",
        ]
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source_file", "未知来源")
            parts.append(f"【词条参考{i}】来源：{source}\n{doc.page_content}")
        return "\n\n" + "\n\n".join(parts)

    def _country_audit_scope_hint(self, registration_countries) -> str:
        """根据页面选中的注册国家，生成「扩大审核面」提示：要求结合该国常见法规审核，即知识库未收录也需补充性审核。"""
        if not registration_countries:
            return ""
        countries = (
            registration_countries
            if isinstance(registration_countries, (list, tuple))
            else [registration_countries]
        )
        countries = [str(c).strip() for c in countries if (c or "").strip()]
        if not countries:
            return ""
        dims = get_dimension_options()
        country_extra_keywords = dims.get("country_extra_keywords") or {}
        if not isinstance(country_extra_keywords, dict):
            country_extra_keywords = {}
        key_lower_to_keywords = {
            str(k).strip().lower(): (v if isinstance(v, list) else [v])
            for k, v in country_extra_keywords.items()
            if k
        }
        # 所选国家对应的法规/标准关键词（供模型参照，即使未在知识库中命中）
        keywords_per_country = []
        for c in countries:
            kws = key_lower_to_keywords.get(c.lower())
            if kws:
                kws = [kw for kw in kws if (kw or "").strip()]
                if kws:
                    keywords_per_country.append(f"{c}：{'、'.join(kws)}")
        lines = [
            "【按注册国家扩大审核面】",
            f"本次注册国家以页面选择为准：{'、'.join(countries)}。",
            "除上述知识库检索到的内容外，请结合该国家/地区通常适用的法规与标准进行补充性审核；"
            "即使用户知识库中未收录某法规全文，也应根据该国家一般性要求对文档的合规性、完整性等提出审核意见，扩大审核面。",
        ]
        if keywords_per_country:
            lines.append("以下为各国家/地区常涉及的法规或标准关键词（即未在知识库中检索到，也请结合常识参照）：" + "；".join(keywords_per_country) + "。")
        return "\n\n" + "\n".join(lines)

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
            modify_docs = item.get("modify_docs")
            if not isinstance(modify_docs, list):
                modify_docs = []
            points.append(AuditPoint(
                category=item.get("category", "未分类"),
                severity=item.get("severity", "info"),
                location=item.get("location", "未知"),
                description=item.get("description", ""),
                regulation_ref=item.get("regulation_ref", ""),
                suggestion=item.get("suggestion", ""),
                modify_docs=modify_docs,
            ))
        return points

    def review_text(
        self,
        text: str,
        file_name: str = "未命名文档",
        review_context: Optional[dict] = None,
        project_context_text: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
    ) -> AuditReport:
        context = self._retrieve_context(text)
        # 按页面选中的注册国家，用「国家→额外关键词」扩展检索知识库中该国家相关法规，扩大审核面
        if review_context:
            reg_countries = review_context.get("registration_country")
            if reg_countries:
                extra = self._retrieve_context_by_country_keywords(reg_countries)
                if extra:
                    context += extra
                # 要求结合该国常见法规补充审核，即知识库中未收录的法规也纳入审核面
                scope_hint = self._country_audit_scope_hint(reg_countries)
                if scope_hint:
                    context += "\n\n" + scope_hint
            parts = []
            if review_context.get("project_name") or review_context.get("project_name_en"):
                v = review_context.get("project_name") or ""
                v_en = review_context.get("project_name_en") or ""
                parts.append(f"项目名称：{v if isinstance(v, str) else '、'.join(v)}" + (f" / {v_en}" if v_en else ""))
            if review_context.get("product_name") or review_context.get("product_name_en"):
                v = review_context.get("product_name") or ""
                v_en = review_context.get("product_name_en") or ""
                parts.append(f"产品名称：{v if isinstance(v, str) else '、'.join(v)}" + (f" / {v_en}" if v_en else ""))
            if review_context.get("registration_country") or review_context.get("registration_country_en"):
                v = review_context.get("registration_country") or ""
                v_en = review_context.get("registration_country_en") or ""
                parts.append(f"注册国家：{v if isinstance(v, str) else '、'.join(v)}" + (f" / {v_en}" if v_en else ""))
            if review_context.get("registration_type"):
                v = review_context["registration_type"]
                parts.append(f"注册类别：{v if isinstance(v, str) else '、'.join(v)}")
            if review_context.get("registration_component"):
                v = review_context["registration_component"]
                parts.append(f"注册组成：{v if isinstance(v, str) else '、'.join(v)}")
            if review_context.get("project_form"):
                v = review_context["project_form"]
                parts.append(f"项目形态：{v if isinstance(v, str) else '、'.join(v)}")
            if parts:
                context += "\n\n【本次审核维度】\n" + "；".join(parts) + "。请根据上述维度识别适用的法规、程序与项目案例要求。"
            # 待审文档语言：仅当有项目案例上下文时生效；法规/程序审核本身为所有语言通用
            doc_lang = review_context.get("document_language") or ""
            has_case = bool(review_context.get("case_context_text"))
            if doc_lang and has_case:
                if doc_lang == "zh":
                    context += "\n\n【待审文档语言】本次待审文档为**中文版**，请按中文注册文档规范与一致性要求进行审核；术语、格式、表述须符合中文注册文档习惯。"
                elif doc_lang == "en":
                    context += "\n\n【待审文档语言】本次待审文档为**英文版**，请按英文注册文档规范与一致性要求进行审核；术语、格式、表述须符合英文注册文档习惯。须增加「翻译正确性」审核维度（国家/语言语法习惯、是否通顺、是否符合逻辑），并参考下方词条及上方法规、英文案例。"
                    context += self._retrieve_glossary_for_translation(text)
                elif doc_lang == "both":
                    context += "\n\n【待审文档语言】本次待审文档可能为**中文版或英文版**（中英文混合或分批），请按中英文注册文档规范与一致性要求进行审核；术语、格式、表述须兼顾中英文，关键信息中英文一致时须同时核对。"
            elif doc_lang == "en":
                # 无项目案例时仍对英文文档做翻译正确性审核，参考词条与法规
                context += "\n\n【待审文档语言】本次待审文档为**英文版**，请按英文注册文档规范审核，并增加「翻译正确性」维度（国家/语言语法习惯、通顺性、逻辑性），参考下方词条及上方法规与英文案例。"
                context += self._retrieve_glossary_for_translation(text)
            if review_context.get("project_name") or review_context.get("project_name_en") or review_context.get("product_name") or review_context.get("product_name_en"):
                context += " 待审文档中出现的项目名称、产品名称、型号等（含中英文）须与上述项目名称、产品名称及下方【项目专属要求】中的对应信息保持一致，不一致须作为审核点（一致性）列出。"
        if review_context and review_context.get("basic_info_text"):
            context += "\n\n【项目基本信息（已入库，须与待审文档一致）】\n" + (review_context.get("basic_info_text") or "")
            context += "\n\n待审文档中的项目名称、产品名称、型号规格、注册单元名称等须与上述基本信息一致；若不一致须作为审核点（一致性）列出。"
        if review_context and review_context.get("scope_of_application"):
            context += "\n\n【产品适用范围】\n" + (review_context.get("scope_of_application") or "")
            context += "\n\n**范围约束**：所有文档描述的内容（功能、适应症、适用人群、使用场景等）不能超过上述适用范围。若文档中出现超出适用范围的功能描述、适应症、适用人群或使用场景，须作为审核点（类别可为「合规性」或「一致性」）明确列出，并给出修改建议。"
        if review_context and review_context.get("registration_type") and "二类" in (review_context.get("registration_type") or ""):
            context += "\n\n【二类医疗器械风险要求】\n本产品注册类别为二类，风险级别不能超过中等风险。文档中的风险描述、适应症、禁忌症、预期用途等不得超出中等风险范围。若文档中出现高于中等风险的表述（如高风险适应症、超出二类范围的用途）或与二类风险等级不符的内容，须作为审核点（类别为「合规性」）明确列出，并给出修改建议。"
        if review_context and review_context.get("system_functionality_text"):
            context += "\n\n【系统功能描述（来自安装包或 URL 识别，须与待审文档一致）】\n" + (review_context.get("system_functionality_text") or "")
            context += "\n\n请核对待审文档中的功能描述、界面说明、操作流程、模块列表等是否与上述系统功能一致；若不一致须作为审核点（一致性）明确列出，并给出修改建议。"
        if project_context_text:
            context += "\n\n【项目专属要求（技术要求、说明书等，以下为项目资料中须与待审文档保持一致的内容）】\n" + project_context_text
            context += "\n\n**一致性要求**：待审文档中的**项目名称、产品名称、型号规格、注册单元名称**等基本信息须与上述项目专属资料中出现的对应信息保持一致；若不一致须作为审核点（类别为「一致性」）明确列出，并给出修改建议。"
        if review_context and review_context.get("case_context_text"):
            context += review_context["case_context_text"]
        if review_context and review_context.get("extra_instructions"):
            context += "\n\n【自定义审核要求（请严格遵守）】\n" + (review_context.get("extra_instructions") or "")

        sys_prompt = system_prompt if (system_prompt and system_prompt.strip()) else REVIEW_SYSTEM_PROMPT
        usr_prompt = user_prompt if (user_prompt and user_prompt.strip()) else REVIEW_USER_PROMPT

        if settings.is_cursor:
            from .cursor_agent import complete_task
            template = CURSOR_REVIEW_TASK if not (user_prompt and user_prompt.strip()) else usr_prompt
            prompt_text = template.format(
                context=context,
                file_name=file_name,
                document_content=text,
            )
            response_content = complete_task(prompt_text)
            audit_points = self._parse_audit_points(response_content)
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", sys_prompt),
                ("human", usr_prompt),
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

    def review_file(
        self,
        file_path,
        review_context: Optional[dict] = None,
        project_context_text: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
    ) -> AuditReport:
        path = Path(file_path)
        docs = load_single_file(path)
        full_text = "\n\n".join(doc.page_content for doc in docs)

        if len(full_text) > 30000:
            return self._review_long_document(full_text, path.name, review_context, project_context_text, system_prompt, user_prompt)

        return self.review_text(full_text, path.name, review_context, project_context_text, system_prompt, user_prompt)

    def _review_long_document(
        self,
        text: str,
        file_name: str,
        review_context: Optional[dict] = None,
        project_context_text: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
    ) -> AuditReport:
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
            report = self.review_text(chunk, chunk_name, review_context, project_context_text, system_prompt, user_prompt)
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

    def review_multiple_files(
        self,
        file_paths: List[str],
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
    ) -> List[AuditReport]:
        reports = []
        for fp in file_paths:
            try:
                report = self.review_file(fp, system_prompt=system_prompt, user_prompt=user_prompt)
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

    def review_multi_document_consistency(
        self,
        doc_list: List[Tuple[str, str]],
        review_context: Optional[dict] = None,
        max_chars_per_doc: int = 4000,
    ) -> AuditReport:
        """
        多文档一致性与模板风格审核。doc_list = [(display_name, text), ...]。
        返回一份虚拟报告「多文档一致性与模板风格审核」，审核点均为跨文档问题。
        """
        if not doc_list or len(doc_list) < 2:
            report = AuditReport(file_name="多文档一致性与模板风格审核")
            report.summary = "不足两份文档，跳过多文档一致性审核。"
            return report

        parts = []
        for name, text in doc_list:
            t = (text or "").strip()[:max_chars_per_doc]
            parts.append(f"### {name}\n{t}")
        docs_summary = "\n\n".join(parts)

        extra = ""
        if review_context:
            if review_context.get("project_name") or review_context.get("product_name"):
                extra = "\n\n【约定】项目名称、产品名称等须在各文档中与上述一致；若某文档与另一文档或与约定不一致，须作为一致性审核点列出。"
            if review_context.get("basic_info_text"):
                extra += "\n\n【项目基本信息】\n" + (review_context.get("basic_info_text") or "")
            # 待审文档语言：仅当有项目案例上下文时生效；法规/程序审核为所有语言通用
            doc_lang = review_context.get("document_language") or ""
            has_case = bool(review_context.get("case_context_text"))
            if doc_lang and has_case:
                if doc_lang == "zh":
                    extra += "\n\n【待审文档语言】本批文档为**中文版**，请按中文注册文档规范检查各文档间一致性及术语、格式统一。"
                elif doc_lang == "en":
                    extra += "\n\n【待审文档语言】本批文档为**英文版**，请按英文注册文档规范检查各文档间一致性及术语、格式统一。"
                elif doc_lang == "both":
                    extra += "\n\n【待审文档语言】本批文档可能含**中文版与英文版**，请按中英文注册文档规范检查各文档间一致性，并兼顾中英文术语、格式与表述统一。"

        prompt_text = MULTI_DOC_CONSISTENCY_PROMPT.format(docs_summary=docs_summary) + extra

        try:
            if settings.is_cursor:
                from .cursor_agent import complete_task
                # 多文档一致性审核内容多，总超时与单次读超时需更长，避免 ReadTimeout
                response_content = complete_task(
                    prompt_text,
                    poll_interval=2.0,
                    timeout=600,
                )
                response_content = (response_content or "").strip()
            else:
                prompt = ChatPromptTemplate.from_messages([("human", prompt_text)])
                chain = prompt | self.llm
                response = chain.invoke({})
                response_content = (getattr(response, "content", None) or str(response) or "").strip()
            audit_points = self._parse_audit_points(response_content or "[]")
        except Exception as e:
            raise RuntimeError(f"多文档一致性审核接口调用失败：{e}") from e

        report = AuditReport(file_name="多文档一致性与模板风格审核")
        report.audit_points = audit_points
        report.total_points = len(audit_points)
        report.high_count = sum(1 for p in audit_points if p.severity == "high")
        report.medium_count = sum(1 for p in audit_points if p.severity == "medium")
        report.low_count = sum(1 for p in audit_points if p.severity == "low")
        report.info_count = sum(1 for p in audit_points if p.severity == "info")
        report.summary = self._generate_summary(report) if audit_points else "各文档间信息与风格一致性已检查；未发现不一致项。"
        return report

    def _generate_summary(self, report: AuditReport) -> str:
        if not report.audit_points:
            return "未发现审核问题。"

        details = "\n".join(
            f"- [{p.severity}] {p.category}: {p.description}"
            for p in report.audit_points[:10]
        )
        inv = {
            "file_name": report.file_name,
            "high": report.high_count,
            "medium": report.medium_count,
            "low": report.low_count,
            "info": report.info_count,
            "details": details,
        }
        from .db import get_prompt_by_key
        summary_tpl = (get_prompt_by_key("review_summary_prompt") or "").strip() or None
        if not summary_tpl:
            summary_tpl = CURSOR_SUMMARY_TASK if settings.is_cursor else SUMMARY_PROMPT

        if settings.is_cursor:
            from .cursor_agent import complete_task
            prompt_text = summary_tpl.format(**inv)
            return complete_task(prompt_text).strip()

        prompt = ChatPromptTemplate.from_messages([("human", summary_tpl)])
        chain = prompt | self.llm
        response = chain.invoke(inv)
        return response.content
