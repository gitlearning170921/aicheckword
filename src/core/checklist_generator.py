"""审核点生成器：从法规/程序/案例知识库中，生成结构化的审核点清单"""

import json
import re
from typing import List, Dict, Any, Optional

from .langchain_compat import Document, ChatPromptTemplate

from config import settings
from .db import get_prompt_by_key, update_prompt_by_key, get_dimension_options

# 生成/优化提示词末尾统一追加，保证自定义提示词也能输出可解析的 JSON 数组
OUTPUT_FORMAT_REMINDER = """

【重要】请仅输出一个 JSON 数组，不要输出任何前言、总结或 markdown。不要用 ``` 包裹。直接以 [ 开头、以 ] 结尾。格式示例：[{"id":"CP-001","category":"合规性","name":"...","description":"...","regulation_ref":"...","check_method":"...","severity":"medium","applicable_docs":["说明书"]}, ...]"""


def _document_language_hint(document_language: Optional[str]) -> str:
    """根据适用文档语言返回追加到生成/优化提示词中的说明。"""
    if not document_language or document_language not in ("zh", "en", "both"):
        return ""
    if document_language == "zh":
        return "【审核点适用文档语言】本审核点清单适用于**中文版**注册文档。请用中文撰写每条审核点的 name、description、check_method 及 regulation_ref 等，表述符合中文注册文档习惯。"
    if document_language == "en":
        return "【审核点适用文档语言】本审核点清单适用于**英文版**注册文档。请用英文撰写每条审核点的 name、description、check_method 及 regulation_ref 等，表述符合英文注册文档习惯。"
    return "【审核点适用文档语言】本审核点清单同时适用于**中文版与英文版**注册文档。请用中文撰写审核点（name、description、check_method 等），必要时在关键术语后标注英文对照，便于中英文文档审核时共用。"


def _registration_strictness_hint(registration_type: Optional[str]) -> str:
    """根据注册类别返回审核尺度说明。严格程度：Ⅲ > Ⅱb > Ⅱa > Ⅱ > Ι。"""
    if not registration_type or not str(registration_type).strip():
        return ""
    rt = str(registration_type).strip()
    if "三类" in rt or "Ⅲ" in rt:
        return "【审核尺度】本清单适用于**三类**医疗器械，审核尺度最严：须按三类风险与证据要求逐条核查，不得放宽。"
    if "Ⅱb" in rt or "Ⅱb" in rt:
        return "【审核尺度】本清单适用于**二类Ⅱb**，审核尺度严于二类/Ⅱa：须按Ⅱb风险与证据要求核查，严于一类、二类及Ⅱa。"
    if "Ⅱa" in rt:
        return "【审核尺度】本清单适用于**二类Ⅱa**，审核尺度严于二类/一类：须按Ⅱa风险与证据要求核查，严于一类及普通二类。"
    if "二类" in rt or "Ⅱ" in rt:
        return "【审核尺度】本清单适用于**二类**医疗器械，审核尺度严于一类：风险级别不超过中等风险，须按二类要求核查。"
    if "一类" in rt or "Ι" in rt or "Ⅰ" in rt:
        return "【审核尺度】本清单适用于**一类**医疗器械，按一类风险与证据要求核查；若后续用于二类/三类项目，审核时须提高尺度。"
    return "【审核尺度】注册类别审核严格程度为：Ⅲ > Ⅱb > Ⅱa > Ⅱ > Ι；请按当前适用类别采用对应尺度。"


def _optimize_prompt_format(template: str) -> str:
    """优化提示词格式：末尾若无输出格式说明则补全，并去掉尾部多余空行。"""
    if not template or not template.strip():
        return template
    s = template.rstrip()
    if OUTPUT_FORMAT_REMINDER.strip() in s or "只输出一个 JSON 数组" in s:
        return s
    return s + OUTPUT_FORMAT_REMINDER


def _safe_format_prompt(template: str, context: str, base_checklist_section: str = "", base_checklist: str = "") -> str:
    """安全替换占位符，避免用户自定义提示词中缺少或多了占位符导致 KeyError。若模板已含输出格式说明则不重复追加。"""
    placeholders = set(re.findall(r"\{(\w+)\}", template))
    values = {
        "context": context,
        "base_checklist_section": base_checklist_section,
        "base_checklist": base_checklist,
    }
    kwargs = {k: values.get(k, "") for k in placeholders}
    try:
        result = template.format(**kwargs)
    except KeyError:
        result = template.replace("{context}", context).replace("{base_checklist_section}", base_checklist_section).replace("{base_checklist}", base_checklist)
    if OUTPUT_FORMAT_REMINDER.strip() not in result and "只输出一个 JSON 数组" not in result:
        result = result + OUTPUT_FORMAT_REMINDER
    return result


def _create_llm():
    """根据当前配置创建 LLM，仅用于非 Cursor 模式（Cursor 在 generate_checklist 内用 complete_task 处理）"""
    if settings.is_cursor:
        raise RuntimeError("Cursor 模式下审核点生成由 complete_task 处理，不应调用 _create_llm")
    from .llm_factory import create_chat_llm
    return create_chat_llm(temperature=0.2)


GENERATE_CHECKLIST_PROMPT = """【输出要求】你只能输出一个 JSON 数组，以 [ 开头、以 ] 结尾。不要输出任何解释、前言、markdown 或代码块标记（不要用 ```）。

你是注册文档审核专家。根据下方「参考知识」生成一份结构化审核点清单，每条审核点具体、可落地，便于后续审核时在文档中精确定位并给出可操作建议。

## 参考知识

{context}

{base_checklist_section}

## 要求

1. 覆盖合规性、完整性、一致性、准确性、格式规范五个维度。
2. **完整性**维度须包含「文档内容完整性」审核点：依据历史项目案例库中的章节结构，检查待审文档是否包含所有应有章节；缺失的章节须在审核时指明并给出补充建议。若参考知识中包含项目案例，生成时须包含此项。
3. 每个审核点有明确法规依据；description 与 check_method 要具体，能指导审核时写出具体 location 和 suggestion。
4. **审核点须精准有效、避免冗余重复**：每条审核点应对应 distinct 的法规条款或检查维度，不与已有/同批其他审核点语义重复；数量与参考知识丰富程度成正比，但以质量为先、宁缺毋滥。
5. 每个对象必须包含：id、category、name、description、regulation_ref、check_method、severity、applicable_docs。id 按 CP-001、CP-002 递增。severity 取 high/medium/low/info。
6. **覆盖面须全面**：参考知识中涉及的法规条款、维度与典型问题均应有所对应审核点，不遗漏重要条款；审核点数量须与参考知识的丰富程度成正比，参考知识越多则审核点越多越细致，但不得为凑数而重复或泛化。
7. **编审批与时间线（优先）**：须包含或强化针对「编制/审核/批准职责与签批」「文档内日期、版本、修订履历与时间顺序」的审核点；description 与 check_method 须指导审核员在签批页、修订记录、封面与正文日期等处逐项核对，发现矛盾即列为一致性或格式规范问题。
8. **岗位与人员（全文，优先）**：须包含针对**文档中出现的所有岗位及人员**（含编审批签批栏及正文各章节）的审核点：**与岗位名称对应职责**（人员与其岗位名称、该岗位定义职责是否匹配）、**花名册符合性**（是否与组织花名册或受控人员清单一致）；check_method 须指导审核员对全文出现的岗位、人员逐项核对职责与花名册。
9. **医疗器械软件生命周期—时间与阶段（优先）**：须包含或强化针对「文档中出现的时间与阶段」的审核点，包括：同一份文件内多个时间点（日期、版本、修订履历、各章节时间）是否合理、顺序是否一致；多份文档之间（如需求—设计—开发—测试—发布等阶段）时间与阶段是否合理、是否存在阶段倒置或跨文档时间矛盾。description 与 check_method 须指导审核员按软件生命周期逐项核对单文件内时间线与跨文档阶段时间线，此项与编审批/时间线并列优先生成。
10. **设备编号与设备设施清单、程序文件（优先，质量体系）**：须包含或强化审核点：（1）文档中出现的**设备/工装/设施编号**（仪器编号、资产编号等）与**设备设施清单、台账、校准/验证设备列表**等**逐项一致**，无遗漏、无矛盾、无清单外未说明编号；（2）设备编号的**命名与编码规则**与受控**程序文件**（如设备编号管理、标识与可追溯性等，以参考知识中实际文件名为准）**一致**。description 与 check_method 须指导审核员提取编号、对照清单与程序条款逐条核对。
11. **需求—开发—测试—风险可追溯性（医疗器械软件，优先）**：须包含或强化审核点，覆盖：**用户需求 ID / 软件需求 ID / 风险 ID / 风险分析文档中的 CS 编号（或组织规定的危害-控制编号）** 等在《软件需求规范》《风险管理/风险分析》《软件可追溯性分析报告》及设计/开发/测试文档之间的**一致性**与**可追溯闭环**。check_method 须指导审核员对照各文档中的表格与正文，核对（1）需求规范中引用的风险 ID、用户需求 ID 与风险分析中对应编号及措施描述是否一致；（2）可追溯性分析报告中的追溯关系与需求/设计/测试文档中同一 ID 的实际内容是否一致；（3）是否存在断链、有号无文、同一 ID 指向矛盾。若参考知识中含组织**专门针对可追溯性的受控程序文件**，须另含或并入审核点：核对项目文档中的**ID 规则、矩阵/报告必备要素、记录与签批**等是否符合该程序规定。applicable_docs 可包含「软件需求规范」「风险管理报告」「软件可追溯性分析报告」「设计开发文档」「测试计划」「测试报告」「可追溯性程序文件」等。
12. **欧盟/美国注册文档语言（优先）**：须包含或强化审核点：当注册国家为**欧盟或美国**时，核查待审文档是否为英文，或是否提供受控且可追溯的英文版本/翻译件；若仅中文且无英文受控版本，或中英文版本关键术语（产品名称、型号、适用范围、风险术语、接口术语等）不一致，须判定为合规性/一致性问题。check_method 须指导审核员核对文档语言属性、版本关系、关键术语对照表与受控记录。applicable_docs 可包含「软件需求规范」「软件描述文档」「风险管理报告」「软件可追溯性分析报告」「说明书」「标签」等。

直接输出 JSON 数组，例如：[{"id":"CP-001","category":"合规性","name":"...","description":"...","regulation_ref":"...","check_method":"...","severity":"medium","applicable_docs":["说明书"]},...]"""

OPTIMIZE_CHECKLIST_PROMPT = """【输出要求】你只能输出一个 JSON 数组，以 [ 开头、以 ] 结尾。不要输出任何解释、前言、markdown 或代码块标记（不要用 ```）。

你是注册文档审核专家。根据下方「参考知识」和「现有基础审核点清单」，优化并补充该清单，使每条审核点具体、可落地，便于审核时在文档中精确定位并给出可操作建议。

## 参考知识

{context}

## 现有基础审核点清单

{base_checklist}

## 要求

1. 补充遗漏的审核点。
2. **完整性**维度须包含「文档内容完整性」审核点（若现有清单中尚无）：依据历史项目案例库中的章节结构，检查待审文档是否包含所有应有章节；缺失的章节须在审核时指明并给出补充建议。
3. 完善 description 与 check_method：明确在文档哪类位置查找、如何定位，便于审核输出具体 location 和可操作 suggestion。
4. **审核点须精准有效、避免冗余重复**：不添加与现有审核点语义重复的条目；完善时以质量为先、宁缺毋滥。
5. 完善法规引用与严重程度。
6. 覆盖五个维度：合规性、完整性、一致性、准确性、格式规范。
7. **编审批与时间线（优先）**：若现有清单中缺少针对「编制/审核/批准与签批」「文档日期/版本/修订履历时间线」的审核点，须补充；若已有则完善 check_method，要求审核时在签批栏、修订页、封面与正文日期等处逐项对照。
8. **岗位与人员（全文，优先）**：若现有清单中缺少针对**文档中出现的所有岗位及人员**（含编审批及正文各章节）的审核点，须补充：与岗位名称对应职责、花名册符合性；若已有则完善 check_method，要求对全文出现的岗位、人员逐项核对职责与花名册。
9. **医疗器械软件生命周期—时间与阶段（优先）**：若现有清单中缺少针对「文档时间与阶段」的审核点，须补充：同一文件内多时间点是否合理、多份文档间阶段与时间是否合理（如需求—设计—测试—发布）；若已有则完善 check_method，要求按软件生命周期核对单文件时间线与跨文档阶段时间线。
10. **设备编号与设备设施清单、程序文件（优先，质量体系）**：若现有清单中缺少针对「设备编号与清单一致性」「编号规则与程序文件一致性」的审核点，须补充；若已有则完善 check_method：要求审核员提取文内设备/工装/设施编号，与设备设施清单（或台账）逐项对照，并对照程序文件中编号规则条款核查格式与前缀/分段含义是否一致。
11. **需求—开发—测试—风险可追溯性**：若现有清单缺少针对「需求 ID / 风险 ID / CS 编号与可追溯性分析报告、需求规范、测试文档一致性」的审核点，须补充；若已有则完善 description 与 check_method，明确要求交叉核对多文档中**同一追溯 ID** 的目标内容是否一致、追溯链是否闭环。若参考知识含**可追溯性管理程序**等受控文件，须补充或完善「文档与程序符合性」：按程序条款核对追溯标识、矩阵栏目、更新与批准等是否落实。
12. **欧盟/美国注册文档语言**：若现有清单缺少针对「注册国家为欧盟/美国时文档英文要求」的审核点，须补充；若已有则完善 check_method：核对是否存在英文受控版本、翻译受控记录，以及中英文关键术语与核心声明的一致性。

每个审核点必须是 JSON 对象，且包含字段：id、category、name、description、regulation_ref、check_method、severity、applicable_docs。id 按 CP-001、CP-002 递增。severity 取 high/medium/low/info。applicable_docs 为字符串数组。

直接输出 JSON 数组，例如：[{"id":"CP-001","category":"合规性","name":"示例","description":"...","regulation_ref":"...","check_method":"...","severity":"medium","applicable_docs":["说明书"]},{"id":"CP-002",...}]"""


def estimate_checklist_scale(total_files: int = 0, total_chunks: int = 0) -> dict:
    """根据知识库规模估算检索参数与目标审核点数量。

    比例逻辑（与知识库规模正相关，无固定上限）:
    - 基础量 + 每文件约 5 个审核点 + 每约 25 块额外 1 个（细节补充）
    - 下限 15，上限定为 500，随知识库增大而增大
    - 检索 top_k 与上下文条数同步放大，保证 AI 有足够参考知识

    返回 dict:
      target_points: 建议生成的审核点数量
      top_k: 检索每个 query 的 top_k 总量
      max_context_docs: 最终取多少条参考知识作为上下文
      scale_label: 规模描述（小/中/大/超大/超大规模）
    """
    base = 15
    from_files = max(0, total_files) * 5
    from_chunks = max(0, total_chunks) // 25
    target = max(base, from_files + from_chunks)
    target = min(target, 500)

    if total_files <= 3 and total_chunks <= 60:
        top_k, max_ctx, label = 20, 20, "小"
    elif total_files <= 10 and total_chunks <= 200:
        top_k, max_ctx, label = 40, 40, "中"
    elif total_files <= 30 and total_chunks <= 600:
        top_k, max_ctx, label = 80, 80, "大"
    elif total_files <= 80 and total_chunks <= 1500:
        top_k, max_ctx, label = 150, 150, "超大"
    else:
        top_k = min(200, 20 + total_files * 2 + total_chunks // 50)
        max_ctx = top_k
        label = "超大规模"

    return {
        "target_points": target,
        "top_k": top_k,
        "max_context_docs": max_ctx,
        "scale_label": label,
    }


class ChecklistGenerator:
    """从法规/程序知识库生成审核点清单"""

    def __init__(self, knowledge_base=None):
        self.kb = knowledge_base
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = _create_llm()
        return self._llm

    def reset_client(self):
        self._llm = None

    _MAX_BATCH_CHARS = 45000

    def _retrieve_all_docs(
        self,
        query_hints: List[str] = None,
        top_k: int = 20,
        max_docs: int = 20,
        extra_query_terms: List[str] = None,
    ) -> list:
        """从知识库检索去重后的所有相关 Document 列表（不拼装文本）。"""
        if not self.kb:
            return []

        queries = query_hints or [
            "注册文档审核要求",
            "法规标准要求",
            "质量管理体系文件要求",
            "产品注册申报资料要求",
            "设计开发文档要求",
            "注册文档完整性检查",
            "产品技术要求与标准",
            "注册文档格式与一致性规范",
        ]

        all_docs = []
        seen = set()
        per_query_k = max(4, top_k // len(queries))
        for q in queries:
            try:
                results = self.kb.search(q, top_k=per_query_k)
                for doc in results:
                    key = doc.page_content[:100]
                    if key not in seen:
                        seen.add(key)
                        all_docs.append(doc)
            except Exception:
                continue
        for term in extra_query_terms or []:
            term = (term or "").strip()
            if not term:
                continue
            try:
                results = self.kb.search(term, top_k=min(per_query_k, 8))
                for doc in results:
                    key = doc.page_content[:100]
                    if key not in seen:
                        seen.add(key)
                        all_docs.append(doc)
            except Exception:
                continue
        return all_docs[:max_docs]

    def _docs_to_batched_contexts(self, docs: list, max_batch_chars: int = 0) -> List[str]:
        """将 Document 列表按字符上限拆分为多批上下文文本。每批是一个可直接嵌入 prompt 的字符串。"""
        if not docs:
            return ["（知识库中暂无相关内容，请先训练法规/程序文件）"]
        limit = max_batch_chars if max_batch_chars > 0 else self._MAX_BATCH_CHARS
        batches: List[str] = []
        current_parts: List[str] = []
        current_chars = 0
        idx = 0
        for doc in docs:
            source = doc.metadata.get("source_file", "未知来源")
            chunk_text = doc.page_content
            if len(chunk_text) > 1500:
                chunk_text = chunk_text[:1500] + "…（已截断）"
            idx += 1
            entry = f"【参考{idx}】来源：{source}\n{chunk_text}"
            if current_chars + len(entry) > limit and current_parts:
                batches.append("\n\n".join(current_parts))
                current_parts = []
                current_chars = 0
            current_parts.append(entry)
            current_chars += len(entry)
        if current_parts:
            batches.append("\n\n".join(current_parts))
        return batches

    def _gather_context(
        self,
        query_hints: List[str] = None,
        top_k: int = 20,
        max_docs: int = 20,
        extra_query_terms: List[str] = None,
    ) -> str:
        """兼容旧调用：返回单批上下文（取第一批）。"""
        docs = self._retrieve_all_docs(query_hints, top_k, max_docs, extra_query_terms)
        batches = self._docs_to_batched_contexts(docs)
        return batches[0] if batches else "（暂无知识库内容）"

    def generate_checklist(
        self,
        base_checklist: Optional[str] = None,
        query_hints: List[str] = None,
        provider: Optional[str] = None,
        generate_prompt_override: Optional[str] = None,
        optimize_prompt_override: Optional[str] = None,
        document_language: Optional[str] = None,
        kb_stats: Optional[Dict[str, Any]] = None,
        registration_countries: Optional[List[str]] = None,
        registration_type: Optional[str] = None,
        progress_callback=None,
    ) -> List[Dict[str, Any]]:
        """
        生成审核点清单。知识库较大时自动分批调用 LLM 再合并去重。
        progress_callback: 可选回调 fn(batch_index, total_batches, message)，用于 UI 显示进度。
        """
        stats = kb_stats or {}
        scale = estimate_checklist_scale(
            total_files=stats.get("total_files", 0),
            total_chunks=stats.get("total_chunks", 0),
        )
        extra_query_terms = []
        if registration_countries:
            dims = get_dimension_options()
            country_extra_keywords = dims.get("country_extra_keywords") or {}
            if isinstance(country_extra_keywords, dict):
                key_lower_to_keywords = {str(k).strip().lower(): (v if isinstance(v, list) else [v]) for k, v in country_extra_keywords.items() if k}
                for c in registration_countries:
                    c = (c or "").strip()
                    if not c:
                        continue
                    kws = key_lower_to_keywords.get(c.lower())
                    if kws:
                        extra_query_terms.extend(kw for kw in kws if (kw or "").strip())

        all_docs = self._retrieve_all_docs(
            query_hints,
            top_k=scale["top_k"],
            max_docs=scale["max_context_docs"],
            extra_query_terms=extra_query_terms or None,
        )
        context_batches = self._docs_to_batched_contexts(all_docs)
        total_batches = len(context_batches)

        target_points = scale["target_points"]
        doc_lang_hint = _document_language_hint(document_language)
        reg_strict_hint = _registration_strictness_hint(registration_type)
        use_cursor = (provider or getattr(settings, "provider", "") or "").strip().lower() == "cursor"
        gen_raw = get_prompt_by_key("checklist_generate_prompt")
        opt_raw = get_prompt_by_key("checklist_optimize_prompt")
        gen_prompt = (generate_prompt_override or (gen_raw or "").strip() or GENERATE_CHECKLIST_PROMPT).strip()
        opt_prompt = (optimize_prompt_override or (opt_raw or "").strip() or OPTIMIZE_CHECKLIST_PROMPT).strip()
        gen_prompt = _optimize_prompt_format(gen_prompt)
        opt_prompt = _optimize_prompt_format(opt_prompt)
        if not generate_prompt_override and gen_raw and (gen_raw or "").strip():
            if gen_prompt != (gen_raw or "").strip():
                update_prompt_by_key("checklist_generate_prompt", gen_prompt)
        if not optimize_prompt_override and opt_raw and (opt_raw or "").strip():
            if opt_prompt != (opt_raw or "").strip():
                update_prompt_by_key("checklist_optimize_prompt", opt_prompt)

        if total_batches <= 1:
            target_per_batch = target_points
        else:
            target_per_batch = max(15, target_points // total_batches + 5)

        all_points: List[Dict[str, Any]] = []

        for batch_idx, context in enumerate(context_batches):
            if progress_callback:
                progress_callback(batch_idx, total_batches, f"正在处理第 {batch_idx + 1}/{total_batches} 批知识…")

            batch_scale_hint = (
                f"\n\n【审核点数量要求】当前为第 {batch_idx + 1}/{total_batches} 批参考知识"
                f"（知识库规模{scale['scale_label']}：{stats.get('total_files', 0)}个文件 / {stats.get('total_chunks', 0)}块），"
                f"请根据本批参考知识生成**约 {target_per_batch} 个**审核点，"
                f"覆盖本批中涉及的不同法规条款与检查维度；每条须精准、可落地，避免与已有或同批其他审核点语义重复。"
            )
            if total_batches > 1 and all_points:
                existing_names = [p.get("name", "") for p in all_points[:30]]
                dedup_hint = (
                    "\n\n【去重提示】前几批已生成的审核点名称（部分）：" + "、".join(existing_names[:20])
                    + "\n请避免重复上述已有的审核点，专注于本批参考知识中未被覆盖的新内容。"
                )
                batch_scale_hint += dedup_hint

            if use_cursor:
                from .cursor_agent import complete_task
                if base_checklist and base_checklist.strip():
                    prompt_text = _safe_format_prompt(opt_prompt, context, base_checklist=base_checklist)
                else:
                    prompt_text = _safe_format_prompt(gen_prompt, context, base_checklist_section="")
                if doc_lang_hint:
                    prompt_text = prompt_text.rstrip() + "\n\n" + doc_lang_hint
                if reg_strict_hint:
                    prompt_text = prompt_text.rstrip() + "\n\n" + reg_strict_hint
                prompt_text = prompt_text.rstrip() + batch_scale_hint
                response_text = complete_task(prompt_text)
                batch_points = self._parse_checklist(response_text)
            else:
                if base_checklist and base_checklist.strip():
                    prompt_template = _safe_format_prompt(opt_prompt, context, base_checklist=base_checklist)
                else:
                    prompt_template = _safe_format_prompt(gen_prompt, context, base_checklist_section="")
                if doc_lang_hint:
                    prompt_template = prompt_template.rstrip() + "\n\n" + doc_lang_hint
                if reg_strict_hint:
                    prompt_template = prompt_template.rstrip() + "\n\n" + reg_strict_hint
                prompt_template = prompt_template.rstrip() + batch_scale_hint
                p = (provider or settings.provider or "").strip().lower()
                if p in ("openai", "deepseek", "lingyi", "ollama"):
                    from .llm_factory import invoke_chat_direct
                    response_text = invoke_chat_direct(prompt_template, temperature=0.2).strip()
                    batch_points = self._parse_checklist(response_text)
                else:
                    prompt = ChatPromptTemplate.from_messages([("human", prompt_template)])
                    chain = prompt | self.llm
                    response = chain.invoke({})
                    batch_points = self._parse_checklist(response.content)

            all_points.extend(batch_points)

        if progress_callback:
            progress_callback(total_batches, total_batches, "正在合并去重…")

        merged = self._merge_and_dedup(all_points)
        # 为每条审核点设置适用注册类别：传入则仅适用该类别，不传则覆盖所有类别
        reg_types = [registration_type] if registration_type and str(registration_type).strip() else []
        for p in merged:
            p["applicable_registration_types"] = p.get("applicable_registration_types") or reg_types
        return merged

    @staticmethod
    def _merge_and_dedup(points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """合并多批审核点，按 name 去重并重新编号。"""
        seen_names = set()
        unique = []
        for p in points:
            if p.get("id") == "CP-ERR":
                continue
            name_key = (p.get("name") or "").strip().lower()
            if name_key and name_key in seen_names:
                continue
            if name_key:
                seen_names.add(name_key)
            unique.append(p)
        for i, p in enumerate(unique, 1):
            p["id"] = f"CP-{i:03d}"
        return unique

    def _parse_json_array_with_auto_fix(self, raw: str):
        """自动格式转换后解析 JSON 数组，失败返回 None。"""
        if not raw or not raw.strip():
            return None
        s = raw.strip()
        # 1) 直接解析
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        # 2) 去掉尾部逗号
        s = re.sub(r",\s*]", "]", s)
        s = re.sub(r",\s*}", "}", s)
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        # 3) 无引号 key 转成 "key":（仅对常见字段名）
        for key in ("id", "category", "name", "description", "regulation_ref", "check_method", "severity", "applicable_docs"):
            s = re.sub(r"\b" + key + r"\s*:", f'"{key}":', s)
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        # 4) 单引号字符串转双引号（仅替换键名与简单值，避免破坏内容里的引号）
        def _replace_simple_single_quotes(txt: str) -> str:
            out = []
            i = 0
            while i < len(txt):
                if txt[i] == "'" and (i == 0 or txt[i - 1] not in "\\"):
                    j = i + 1
                    while j < len(txt):
                        if txt[j] == "\\":
                            j += 2
                            continue
                        if txt[j] == "'":
                            break
                        j += 1
                    if j < len(txt):
                        inner = txt[i + 1 : j]
                        if '"' not in inner and "\n" not in inner and len(inner) < 80:
                            out.append('"' + inner.replace('\\', "\\\\").replace('"', '\\"') + '"')
                        else:
                            out.append(txt[i : j + 1])
                        i = j + 1
                        continue
                out.append(txt[i])
                i += 1
            return "".join(out)
        try:
            s2 = _replace_simple_single_quotes(s)
            return json.loads(s2)
        except (json.JSONDecodeError, Exception):
            pass
        # 5) 提取多个 {...} 拼成数组再解析
        objs = []
        depth = 0
        start = -1
        for i, c in enumerate(s):
            if c == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0 and start != -1:
                    objs.append(s[start : i + 1])
        if objs:
            fixed = "[" + ",".join(objs) + "]"
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass
        # 6) 整段为单个 {...} 时包成数组
        s_stripped = s.strip()
        if s_stripped.startswith("{") and s_stripped.endswith("}"):
            try:
                one = json.loads(s_stripped)
                if isinstance(one, dict):
                    return [one]
            except json.JSONDecodeError:
                pass
        return None

    def _extract_first_json_array(self, text: str) -> Optional[str]:
        """从文本中提取第一个完整的 JSON 数组（按方括号配对），忽略字符串内的 [ ]。"""
        if not text:
            return None
        i = 0
        n = len(text)
        while i < n:
            if text[i] == "[":
                depth = 1
                j = i + 1
                in_str = False
                escape = False
                quote = None
                while j < n and depth > 0:
                    c = text[j]
                    if escape:
                        escape = False
                        j += 1
                        continue
                    if c == "\\" and in_str:
                        escape = True
                        j += 1
                        continue
                    if not in_str and (c == '"' or c == "'"):
                        in_str = True
                        quote = c
                        j += 1
                        continue
                    if in_str and c == quote:
                        in_str = False
                        j += 1
                        continue
                    if not in_str:
                        if c == "[":
                            depth += 1
                        elif c == "]":
                            depth -= 1
                            if depth == 0:
                                return text[i : j + 1]
                    j += 1
                break
            elif text[i] in ('"', "'"):
                quote = text[i]
                j = i + 1
                while j < n:
                    if text[j] == "\\":
                        j += 2
                        continue
                    if text[j] == quote:
                        break
                    j += 1
                i = j + 1
            else:
                i += 1
        return None

    def _parse_checklist(self, llm_response: str) -> List[Dict[str, Any]]:
        text = (llm_response or "").strip()
        # 1) 优先从 markdown 代码块中取内容
        if "```json" in text:
            start_m = text.find("```json")
            end_m = text.find("```", start_m + 7)
            if end_m != -1:
                text = text[start_m + 7 : end_m].strip()
        elif "```" in text:
            start_m = text.find("```")
            end_m = text.find("```", start_m + 3)
            if end_m != -1:
                text = text[start_m + 3 : end_m].strip()
        # 2) 提取第一个完整的 [...] 数组（配对），避免混入后文
        json_slice = self._extract_first_json_array(text)
        if not json_slice:
            fallback = text.find("["), text.rfind("]")
            if fallback[0] != -1 and fallback[1] >= fallback[0]:
                json_slice = text[fallback[0] : fallback[1] + 1]
        if not json_slice:
            hint = "请确保提示词末尾有「仅输出一个 JSON 数组」的说明，且模型实际返回了包含 [...] 的内容。"
            return [{"id": "CP-ERR", "category": "解析错误", "name": "LLM输出无法解析",
                     "description": (text[:400] if text else "LLM 未返回有效内容。") + " " + hint, "regulation_ref": "", "check_method": "",
                     "severity": "info", "applicable_docs": []}]
        data = self._parse_json_array_with_auto_fix(json_slice)
        if data is None:
            return [{"id": "CP-ERR", "category": "解析错误", "name": "JSON解析失败",
                     "description": json_slice[:500] + ("..." if len(json_slice) > 500 else ""), "regulation_ref": "", "check_method": "",
                     "severity": "info", "applicable_docs": []}]

        if not isinstance(data, list):
            return [{"id": "CP-ERR", "category": "解析错误", "name": "LLM输出格式错误",
                     "description": "应输出 JSON 数组，当前为 " + type(data).__name__, "regulation_ref": "", "check_method": "",
                     "severity": "info", "applicable_docs": []}]
        result = []
        for i, item in enumerate(data):
            point = {
                "id": item.get("id", f"CP-{i+1:03d}"),
                "category": item.get("category", "未分类"),
                "name": item.get("name", ""),
                "description": item.get("description", ""),
                "regulation_ref": item.get("regulation_ref", ""),
                "check_method": item.get("check_method", ""),
                "severity": item.get("severity", "medium"),
                "applicable_docs": item.get("applicable_docs", []),
                "applicable_registration_types": item.get("applicable_registration_types", []),
            }
            result.append(point)
        return result

    def checklist_to_documents(self, checklist: List[Dict[str, Any]]) -> List[Document]:
        """将审核点清单转换为 LangChain Document 列表，供后续向量化训练。metadata 中含 applicable_registration_types 供按项目审核时过滤。"""
        docs = []
        for point in checklist:
            app_reg = point.get("applicable_registration_types") or []
            app_reg_str = "all" if not app_reg else ",".join(str(x).strip() for x in app_reg if (x or "").strip())
            content = (
                f"审核点编号：{point.get('id', '')}\n"
                f"审核类别：{point.get('category', '')}\n"
                f"审核点名称：{point.get('name', '')}\n"
                f"详细描述：{point.get('description', '')}\n"
                f"法规依据：{point.get('regulation_ref', '')}\n"
                f"检查方法：{point.get('check_method', '')}\n"
                f"严重程度：{point.get('severity', '')}\n"
                f"适用文档：{', '.join(point.get('applicable_docs', []))}"
            )
            doc = Document(
                page_content=content,
                metadata={
                    "source_file": f"审核点清单:{point.get('id', '')}",
                    "type": "audit_checklist_point",
                    "kb_entry_class": "checklist_trained",
                    "point_id": point.get("id", ""),
                    "category": point.get("category", ""),
                    "severity": point.get("severity", ""),
                    "applicable_registration_types": app_reg_str,
                },
            )
            docs.append(doc)
        return docs
