"""
文档初稿生成器（项目案例模板复用 + 输入文档差异填充）。

目标：
1) 生成新的 `projects` 记录，并填充：
   - basic_info_text（用于一致性核对）
   - system_functionality_text（用于与待审文档系统功能一致性核对）
   - scope_of_application（沿用模板/案例边界）
2) 生成新的 `project_cases` 记录，并将生成后的“项目案例文档文本”训练进 knowledge base
   （category=project_case + case_id 关联），以便后续审核能匹配到对应 case。
3) 以训练知识库中的项目案例文档文本为模板：
   - 只替换/更新“系统功能相关部分”
   - 基本信息字段（项目名/产品名/型号/注册单元名）与新项目一致
   - 法规/编号/追溯 ID 等按模板保留（避免编号误写/编号幻觉）
"""

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

from config import settings
from .cursor_skills_rules_updater import apply_patch_updates
from .document_loader import (
    Document,
    split_documents,
    load_single_file,
    extract_section_outline_from_texts,
)
from .knowledge_base import KnowledgeBase
from .llm_factory import invoke_chat_direct
from .system_functionality import identify_system_functionality_with_llm
from .cursor_agent import complete_task
from .agent import ReviewAgent
from .db import (
    create_project,
    update_project,
    update_project_basic_info,
    update_project_system_functionality,
    get_project,
    create_project_case,
    get_project_case,
    get_project_case_file_names,
    get_knowledge_docs_by_case_id,
    get_knowledge_docs_by_case_id_and_file_name,
    get_draft_file_skills_rules,
    list_project_cases,
)
DOC_DRAFT_GEN_TEMPERATURE = 0.2


@dataclass
class GeneratedCaseDocs:
    project_id: int
    project_case_id: Optional[int]
    generated_files: Dict[str, str]  # file_name -> full_text
    generated_patches: Dict[str, str]  # file_name -> patch_json (for inplace edit; optional)
    # 输出文件名（含项目编号前缀替换后）-> 是否已从 draft_file_skills_rules 注入本文件专用块
    per_file_skills_rules_applied: Dict[str, bool] = field(default_factory=dict)
    # 输出文件名 -> 本次实际使用的基础文件磁盘路径（多 Base 自动分配时用）
    per_file_base_path: Dict[str, str] = field(default_factory=dict)
    # 多基础/多参考自动路由结果（可选，供 UI 展示）
    draft_routing_plan: Optional[Dict[str, Any]] = None


_BASIC_INFO_LINE_RE = re.compile(r"^\s*(项目名称|产品名称|型号规格|注册单元名称)\s*[:：]\s*(.+?)\s*$")


_FILE_PROJECT_CODE_RE = re.compile(r"^(?P<prefix>[A-Za-z0-9]+)-(?P<rest>[A-Za-z]{2,10}-\d{3}.*)$")


def _replace_file_project_code_prefix(file_name: str, new_project_code: str) -> str:
    """
    将类似 `OXGWIS-STR-001 系统测试报告` 替换为 `${new_project_code}-STR-001 系统测试报告`。
    仅当文件名形如：<字母数字前缀>-<文档代码-三位序号...> 时替换，避免误替换普通连字符名称。
    """
    fn = (file_name or "").strip()
    code = (new_project_code or "").strip()
    if not fn or not code:
        return fn
    m = _FILE_PROJECT_CODE_RE.match(fn)
    if not m:
        return fn
    return f"{code}-{m.group('rest')}"


def _smart_excerpt_from_docs(
    docs: List[Document],
    *,
    file_name: str,
    max_chars: int = 18000,
) -> str:
    """
    从 docs 中构建“更稳的参考摘录”，避免只取开头导致遗漏（PDF/长文档的关键信息常在后半段）。
    策略：
    - 优先保留开头与结尾；
    - 额外按关键词抓取若干片段（带少量上下文）；
    - 最终整体截断到 max_chars。
    """
    if not docs:
        return ""

    parts: List[str] = []
    pages: List[Tuple[int, str]] = []
    for d in docs:
        txt = (getattr(d, "page_content", "") or "").strip()
        if not txt:
            continue
        meta = getattr(d, "metadata", {}) or {}
        p = meta.get("page")
        try:
            p = int(p) if p is not None else None
        except Exception:
            p = None
        pages.append((p or -1, txt))

    if not pages:
        return ""

    full = "\n\n".join(t for _, t in pages).strip()
    if len(full) <= max_chars:
        return full

    # 关键词：按文件名与常见“权限/日志/审计”域增强命中
    base_kws = [
        "权限", "角色", "用户", "账号", "登录", "认证", "授权", "口令", "密码",
        "日志", "审计", "追踪", "Audit", "audit", "log", "logging",
        "事件", "操作", "记录", "保留", "留存", "查询", "导出",
        "失败", "告警", "异常", "超时", "锁定",
        "interface", "API", "endpoint", "字段", "参数", "数据字典",
    ]
    fn = (file_name or "").lower()
    if "权限" in file_name or "permission" in fn or "auth" in fn:
        base_kws.extend(["权限矩阵", "最小权限", "RBAC", "访问控制", "Access Control"])
    if "日志" in file_name or "log" in fn:
        base_kws.extend(["日志级别", "INFO", "WARN", "ERROR", "审计日志", "操作日志"])
    kws = [k for k in base_kws if k]

    # 1) 头尾保留
    head = full[: min(6500, len(full))]
    tail = full[max(0, len(full) - 4500) :]
    parts.append("【摘录-开头】\n" + head)
    parts.append("【摘录-结尾】\n" + tail)

    # 2) 关键词片段：从全文中抓取若干窗口
    window = 220
    max_hits = 18
    seen_spans: List[Tuple[int, int]] = []
    hits: List[str] = []
    low = full.lower()
    for kw in kws:
        k = kw.lower()
        if not k or k not in low:
            continue
        start = 0
        while True:
            idx = low.find(k, start)
            if idx < 0:
                break
            a = max(0, idx - window)
            b = min(len(full), idx + len(k) + window)
            # 去重：与已收集片段重叠则跳过
            overlapped = False
            for sa, sb in seen_spans:
                if not (b <= sa or a >= sb):
                    overlapped = True
                    break
            if not overlapped:
                seen_spans.append((a, b))
                snippet = full[a:b].strip()
                if snippet:
                    hits.append(f"…{snippet}…")
            start = idx + len(k)
            if len(hits) >= max_hits:
                break
        if len(hits) >= max_hits:
            break
    if hits:
        parts.append("【摘录-关键词命中片段（用于补齐长文档后半段信息）】\n" + "\n\n".join(hits))

    merged = "\n\n".join(p for p in parts if p.strip()).strip()
    if len(merged) > max_chars:
        merged = merged[:max_chars] + "\n【提示】该输入/参考文件内容已按单文件上限截断（已做头尾+关键词片段补偿）。"
    return merged


def _parse_basic_info_lines(basic_info_text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not basic_info_text:
        return out
    for line in (basic_info_text or "").splitlines():
        m = _BASIC_INFO_LINE_RE.match(line.strip())
        if not m:
            continue
        k = m.group(1).strip()
        v = (m.group(2) or "").strip()
        if v:
            out[k] = v
    return out


def _collect_cursor_skills_rules_for_prompt(workspace_root: Optional[Path] = None) -> str:
    """
    将 `.cursor/skills/**/SKILL.md` 与 `.cursor/rules/*.mdc` 的内容拼进 prompt。
    """
    root = workspace_root or Path(__file__).resolve().parents[2]
    skills_root = root / ".cursor" / "skills"
    rules_root = root / ".cursor" / "rules"

    parts: List[str] = []
    if skills_root.exists():
        skill_files = sorted(skills_root.glob("**/SKILL.md"), key=lambda p: str(p))
        for sf in skill_files:
            try:
                rel = sf.relative_to(root).as_posix()
            except Exception:
                rel = str(sf)
            txt = sf.read_text(encoding="utf-8", errors="ignore").strip()
            if txt:
                parts.append(f"【Skill 文件：{rel}】\n{txt}\n")

    if rules_root.exists():
        rule_files = sorted(rules_root.glob("*.mdc"), key=lambda p: str(p))
        for rf in rule_files:
            try:
                rel = rf.relative_to(root).as_posix()
            except Exception:
                rel = str(rf)
            txt = rf.read_text(encoding="utf-8", errors="ignore").strip()
            if txt:
                parts.append(f"【Rule 文件：{rel}】\n{txt}\n")

    # 防御：如果 skills/rules 特别多，给个硬截断，避免 prompt 直接爆上下文。
    raw = "\n".join(parts).strip()
    max_chars = 25000
    if len(raw) > max_chars:
        raw = raw[:max_chars] + "\n【提示】skills/rules 内容已截断以适配上下文，请确保仍遵循硬约束。"
    return raw


def _names_match_file(a: str, b: str) -> bool:
    a = (a or "").strip()
    b = (b or "").strip()
    if not a or not b:
        return False
    if a == b:
        return True
    try:
        return Path(a).name == Path(b).name
    except Exception:
        return False


def _filter_combined_input_by_reference_names(combined_input_text: str, reference_files: Optional[List[str]]) -> str:
    """
    按参考文件显示名过滤 combined_input_text 中的「【输入/参考文件：…】」块。
    reference_files 为 None 表示不过滤（使用全部参考）；为空列表表示不使用任何参考摘要。
    """
    if reference_files is None:
        return combined_input_text
    if not reference_files:
        return "【提示】本目标按路由未分配参考文件摘要；仅使用基础文件与模板知识库内容。\n"
    allowed = [x.strip() for x in reference_files if (x or "").strip()]
    if not allowed:
        return "【提示】本目标按路由未分配参考文件摘要；仅使用基础文件与模板知识库内容。\n"
    parts = re.split(r"(?=【输入/参考文件：)", combined_input_text or "")
    kept: List[str] = []
    for part in parts:
        if not part.strip():
            continue
        m = re.match(r"【输入/参考文件：([^】]+)】", part)
        if not m:
            kept.append(part)
            continue
        title = (m.group(1) or "").strip()
        if any(_names_match_file(title, r) for r in allowed):
            kept.append(part)
    if not kept:
        return "【提示】路由指定的参考文件名未在摘要中匹配到任何块；请检查文件名是否一致。\n"
    return "\n".join(kept).strip()


def _outline_snippet(file_path: str, max_chars: int = 2800) -> str:
    try:
        docs = load_single_file(file_path)
        blob = "\n\n".join((d.page_content or "") for d in docs if (d.page_content or "").strip()).strip()
        if len(blob) > max_chars:
            blob = blob[:max_chars] + "\n…"
        ol = extract_section_outline_from_texts([blob], max_sections=40)
        return ((ol or "") + "\n\n" + blob)[: max_chars + 800]
    except Exception as e:
        return f"（摘录失败：{e}）"


def _fallback_multi_base_route(
    chosen: List[str],
    base_manifest: List[Tuple[str, str]],
    ref_names: List[str],
) -> Dict[str, Any]:
    """LLM 失败时：若仅一份 Base，则所有目标共用；若多份 Base，则按文件名与模板名子串粗匹配，否则回退第一份。"""
    if not chosen:
        return {"assignments": []}
    if not base_manifest:
        return {
            "assignments": [
                {
                    "template_file": t,
                    "base_file": None,
                    "_base_path": None,
                    "reference_files": None,
                    "reason": "无基础文件",
                }
                for t in chosen
            ]
        }
    assigns = []
    for t in chosen:
        tl = (t or "").lower()
        picked = base_manifest[0]
        if len(base_manifest) > 1:
            for bp, bn in base_manifest:
                bl = (bn or "").lower()
                if tl and bl and (Path(bn).stem.lower() in tl or Path(t).stem.lower() in bl):
                    picked = (bp, bn)
                    break
            else:
                picked = base_manifest[0]
        assigns.append(
            {
                "template_file": t,
                "base_file": picked[1],
                "_base_path": picked[0],
                "reference_files": None,
                "reason": "路由降级：按文件名粗匹配或默认第一份基础文件；全量参考。",
            }
        )
    return {"assignments": assigns}


def _parse_json_object_from_llm(raw: str) -> Optional[Dict[str, Any]]:
    t = (raw or "").strip()
    if not t:
        return None
    if "```" in t:
        m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", t)
        if m:
            t = m.group(1)
    i = t.find("{")
    j = t.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(t[i : j + 1])
    except Exception:
        return None


def _normalize_route_assignments(
    plan: Dict[str, Any],
    *,
    chosen: List[str],
    base_manifest: List[Tuple[str, str]],
    ref_all_names: List[str],
) -> Dict[str, Any]:
    """校验并补全 assignments；将 base_file 名称解析为 path（写入 _base_path）。"""
    name_to_base = {bn: bp for bp, bn in base_manifest}
    by_t: Dict[str, Dict[str, Any]] = {}
    for a in plan.get("assignments") or []:
        if not isinstance(a, dict):
            continue
        tf = (a.get("template_file") or "").strip()
        if not tf:
            continue
        bf_name = a.get("base_file")
        if bf_name is not None and not isinstance(bf_name, str):
            bf_name = str(bf_name) if bf_name else None
        if isinstance(bf_name, str) and not bf_name.strip():
            bf_name = None
        rf = a.get("reference_files")
        if rf is None:
            rf_names: Optional[List[str]] = None
        elif isinstance(rf, list):
            rf_names = [str(x).strip() for x in rf if str(x).strip()]
        else:
            rf_names = None
        bp = name_to_base.get(bf_name) if bf_name else None
        if bf_name and bf_name not in name_to_base:
            # 尝试按 basename 对齐
            for k, v in name_to_base.items():
                if _names_match_file(k, bf_name):
                    bf_name = k
                    bp = v
                    break
        by_t[tf] = {
            "template_file": tf,
            "base_file": bf_name,
            "_base_path": bp,
            "reference_files": rf_names,
            "reason": (a.get("reason") or "").strip(),
        }
    out_list: List[Dict[str, Any]] = []
    for tf in chosen:
        if tf in by_t:
            out_list.append(by_t[tf])
            continue
        # 单条补全
        fb = _fallback_multi_base_route([tf], base_manifest, ref_all_names)
        out_list.extend(fb.get("assignments") or [])
    return {"assignments": out_list}


def _plan_multi_base_route_llm(
    *,
    chosen: List[str],
    base_manifest: List[Tuple[str, str]],
    ref_manifest: List[Tuple[str, str]],
    provider: Optional[str],
) -> Dict[str, Any]:
    ref_all_names = [rn for _, rn in ref_manifest]
    if not chosen:
        return {"assignments": []}
    if not base_manifest:
        return _fallback_multi_base_route(chosen, [], ref_all_names)

    bases_payload = [{"name": bn, "outline_excerpt": _outline_snippet(bp, max_chars=2400)} for bp, bn in base_manifest]
    refs_payload = [{"name": rn, "outline_excerpt": _outline_snippet(rp, max_chars=2000)} for rp, rn in ref_manifest]

    prompt = (
        "你是文档生成任务编排器。只输出一个 JSON 对象，不要 markdown 围栏，不要解释。\n\n"
        "【待生成的目标文件 template_file（必须与下列字符串完全一致）】\n"
        + json.dumps(chosen, ensure_ascii=False)
        + "\n\n【基础文件（name 必须与 assignments.base_file 一致）】\n"
        + json.dumps(bases_payload, ensure_ascii=False)
        + "\n\n【参考文件（name 必须与 assignments.reference_files 数组项一致）】\n"
        + json.dumps(refs_payload, ensure_ascii=False)
        + "\n\n【规则】\n"
        "1) 为每个 template_file 指定 base_file：填基础文件 name；若该目标不需要上传的基础文档则填 null。\n"
        "2) reference_files：本次生成该目标时应使用的参考文件 name 列表；可将不同参考拆到不同目标；无关的参考不要塞给错误目标。\n"
        "3) reference_files 填 null 表示使用全部参考；填 [] 表示不使用参考摘要（仅基础+模板）。\n"
        "4) 必须覆盖每一个 template_file。\n\n"
        '【输出格式】{"assignments":[{"template_file":"...","base_file":"...或null",'
        '"reference_files":["..."]或null,"reason":"..."}]}\n'
    )

    try:
        raw = invoke_chat_direct(prompt, temperature=0.15, provider=provider)
        parsed = _parse_json_object_from_llm(raw or "")
        if not parsed or not isinstance(parsed.get("assignments"), list):
            raise ValueError("invalid routing json")
        return _normalize_route_assignments(parsed, chosen=chosen, base_manifest=base_manifest, ref_all_names=ref_all_names)
    except Exception:
        return _normalize_route_assignments(
            {"assignments": []},
            chosen=chosen,
            base_manifest=base_manifest,
            ref_all_names=ref_all_names,
        )


# 编写人员身份 → 注入提示词（与 skills/rules、生成策略并行；不覆盖编号逐字、页码、范围等硬约束）
_AUTHOR_ROLE_PROMPTS: Dict[str, str] = {
    "": "（未指定编写身份：按受控技术文档常规主笔口径输出；保持可审计、可核对、少口号。）",
    "pm": (
        "【编写身份：产品经理】\n"
        "- 侧重：用户需求、使用场景、预期用途、范围边界、与临床/用户价值相关的表述；需求可验证、可测试。\n"
        "- 表述：面向“要做什么、为谁解决什么问题”，避免实现细节堆砌；与 SRS/需求类章节语气一致。\n"
        "- 避免：编造法规结论；无依据的功能承诺；擅自新增适应症或超出参考/基底范围的能力。\n"
    ),
    "pjm": (
        "【编写身份：项目经理】\n"
        "- 侧重：交付范围、里程碑、角色职责、接口与依赖、假设与约束、变更可追溯性（在文档允许处）。\n"
        "- 表述：可执行、可跟踪；责任边界清晰但不虚构具体日期。\n"
        "- 避免：代替研发写实现细节；代替注册下法规结论。\n"
    ),
    "rm": (
        "【编写身份：风险经理】\n"
        "- 侧重：危害识别、风险控制措施、残余风险、验证与记录需求；与风险分析/风险管理类表述一致。\n"
        "- 表述：强调“原因—措施—验证—可追溯”；措施与编号（如 CS/风险需求）须与来源一致，不得改写编号。\n"
        "- 避免：无依据的风险等级；把推测写成既成事实。\n"
    ),
    "rdm": (
        "【编写身份：研发经理】\n"
        "- 侧重：架构与模块边界、接口、数据流、非功能需求（性能/可靠性/安全实现侧）、开发与维护约束。\n"
        "- 表述：工程化、可落地；与实现相关的需求应可测试、可定位到模块/接口层级（在文档结构允许内）。\n"
        "- 避免：替代产品经理写用户故事口号；替代注册写递交策略。\n"
    ),
    "ui": (
        "【编写身份：UI 设计师】\n"
        "- 侧重：界面结构、交互流程、可用性、错误提示与状态反馈、与权限/角色相关的界面呈现（在文档允许处）。\n"
        "- 表述：以用户任务路径描述为主，可与字段/界面元素名称对齐参考材料。\n"
        "- 避免：臆造像素级视觉规范；无参考时不要用具体色号/字体号充数。\n"
    ),
    "qa": (
        "【编写身份：测试工程师】\n"
        "- 侧重：验证与确认（V&V）口径；可测试性、验收准则、边界与异常、回归范围；在需求/设计变更语境下，**必须维护需求—测试的可追溯关系**。\n"
        "- **强制（参考相对基底有新增或实质性变更的软件需求/功能点时）**：不得在测试类章节/用例表中**零更新**。须在基底已有的「系统测试/确认测试/验证」等**对应章节或表格**中，为每条受影响需求补充或修订**一条或多条**测试用例（常见为**一对多**：一条需求对应多条用例以覆盖正向/边界/异常等）。\n"
        "- 表述：条件—步骤—期望结果；用例须能回溯到所覆盖的需求表述或需求标识（若表中有 REQ/URS/CS 等列则逐字对齐，无编号则用可引用的原文短语关联，禁止编造追溯号）。\n"
        "- 避免：代替研发写实现代码或详细设计；**不要**凭空发明用例编号格式——应沿用模板/相邻行已有编号规则，无法确定则 TBD 或留空待分配。\n"
    ),
    "cm": (
        "【编写身份：配置管理员】\n"
        "- 侧重：配置项标识、基线、变更受控、发布与构建标识、环境/版本一致性（在文档允许处）。\n"
        "- 表述：强调可追溯与唯一标识；不编造配置项编号，沿用模板或写 TBD。\n"
        "- 避免：把配置管理写成营销话术。\n"
    ),
    "ra": (
        "【编写身份：注册工程师】\n"
        "- 侧重：法规与递交语境一致性、适用范围、同路径参照的克制表述、标签与说明书相关边界（若文档涉及）。\n"
        "- 表述：与注册国家/类别/组成对齐；不确定处明确“需补充证据/受控翻译件”。\n"
        "- 避免：无来源的法规条款号或结论；英文文档不要夹带中文审阅口吻。\n"
    ),
    "prod": (
        "【编写身份：生产专员】\n"
        "- 侧重：生产与制造相关约束、批次/序列追溯（若适用）、现场部署与运维在制造端的输入（在文档允许处）。\n"
        "- 表述：与工艺、检验、放行、记录保存相关的可执行要求；不扩展研发功能范围。\n"
        "- 避免：编造产线设备编号或台账项。\n"
    ),
}


def _build_author_role_block(role_key: str) -> str:
    k = (role_key or "").strip().lower()
    if k not in _AUTHOR_ROLE_PROMPTS:
        k = ""
    return _AUTHOR_ROLE_PROMPTS[k]


def _build_qa_extra_block(role_key: str) -> str:
    """编写身份为测试工程师时追加：需求变更必须驱动测试用例更新（法规/工程可追溯性）。"""
    if (role_key or "").strip().lower() != "qa":
        return ""
    return (
        "【测试工程师专条（本任务强制，与上方编写身份一并生效）】\n"
        "- 依据软件工程实践及医疗器械独立软件**可追溯性、验证与确认**的一般要求：凡在 INPUT_DOCS_EXCERPT / INPUT_SYSTEM_FUNCTIONALITY 中相对 EXISTING_DRAFT_TEXT "
        "体现为**新增、细化或实质性变更**且属于**应验证的软件需求/功能**的条目，必须在**同一份基底文档内**已有的测试相关落点（如测试用例表、测试计划/方案中的用例清单、确认测试列表等——以基底实际标题/表格为准）生成**对应测试覆盖**。\n"
        "- **需求—用例关系**：默认按 **一对多** 设计（一条需求至少一条用例，通常多条以覆盖主流程/边界/异常/回归要点）；若表格含「需求编号/REQ」与「用例编号/TC」列，应保持行级可核对关系。\n"
        "- **落点约束**：优先使用已有表格追加行（`insert_table_row_after_contains`）或更新单元格（`replace_table_cell_contains`）；若仅有正文型「测试/验证」小节且无表，可在该小节内用 `insert_paragraph_after_contains` 插入结构化用例段落（**不得**为此新建文档顶层章节标题）。\n"
        "- **禁止**：在参考已引入新需求或变更要点时，仍以「减少改动」为由让测试章节/用例表保持与变更前完全一致。\n"
    )


# 医疗器械软件：器械监管语境 + 软件生命周期语境（默认注入生成提示词，与仓库 rules/skills 一致）
_MDSW_FRAMEWORK_BLOCK = (
    "【医疗器械软件文档双重要求（本任务默认适用）】\n"
    "- 须**同时**对齐：**医疗器械监管语境**（适用范围、预期用途、风险管理证据、说明书/标签相关表述、质量管理体系对软件相关活动的受控要求等——以项目维度与输入可支撑者为准）与"
    " **软件工程生命周期语境**（需求、设计、实现、验证与确认、配置与变更、发布与维护、网络安全与数据、现成软件/SOUP 等——**仅写本次文档类型与参考文件实际涉及项**）。\n"
    "- **可追溯与验证**：在结构允许范围内，需求—设计—测试—风险 应可核对；不得为表面完整而编造追溯编号。**法规/标准引用**（如 YY/T 0664、IEC 62304、ISO 14971、ISO 13485、技术审查指导原则等）"
    " 仅作框架提示；**具体条款号、路径结论**须来自 INPUT/知识库中**可核对**的表述，禁止虚构。\n"
)


_DOC_DRAFT_PROMPT_TEMPLATE = """你将生成"项目注册文档初稿"，用于后续审核。

【硬约束（来自 skills/rules）——必须逐条遵守】
{cursor_skills_rules}

{strategy_block}

{author_role_block}
{qa_extra_block}
{mdsw_framework_block}

【任务要求】
1. 你必须以下面的"基底文本（TEMPLATE_TEXT）"为格式基础：保留其标题/章节层级/表格行列/编号与表头字段，尽可能原样保留与本次修改无关的部分。
   - 若提供了 EXISTING_DRAFT_TEXT：则 **EXISTING_DRAFT_TEXT 为唯一基底**（优先保留已有内容与结构），TEMPLATE_REFERENCE_TEXT 仅用于对齐章节/格式与补缺。
2. 默认允许修改项：
   - 基本信息字段：项目名称、产品名称、型号规格、注册单元名称（出现处需与 NEW_BASIC_INFO 一致）
   - 系统功能相关部分：与"系统功能/软件功能/模块列表/界面说明/操作流程/功能说明/权限管理/审计追踪/日志/数据备份"相关的章节或表格
   - 若基底文本中存在明显缺失章节（对照 TEMPLATE_REFERENCE_TEXT 可识别）：允许补充缺失章节，但不得新增项目范围外内容与任何虚构编号。
3. **重要——本文件专用 Skills 扩展修改范围**：若上方硬约束中包含「本文件专用 Skills」，则其列举的每一条操作指令（如"变更记录追加""目录更新""预期用途替换""版本号升级"等）均属于本次允许修改的范围，你必须逐条执行，不得因默认限制而跳过。本文件专用 Skills 的优先级高于第 2 条默认限制。
4. **重要——充分利用参考文件内容**：INPUT_DOCS_EXCERPT 中的每份参考文件都包含本项目的实际业务信息（如权限列表、日志规则、功能描述、操作流程等）。你必须仔细阅读每份参考文件的内容，将其中与基底文档相关章节对应的信息逐项更新到对应位置，不能只更新部分要点而遗漏其余。
5. **重要——全自动适配“落点”（不要反问用户放哪）**：
   - 你必须根据「BASE_OUTLINE（基底章节纲要）」自动判断应落入的章节位置。
   - 对“权限管理”“审计追踪/日志”等内容：优先落入基底中已存在的「网络安全需求/安全需求/访问控制/审计/日志」等相关章节；若不存在同名章节，则落入最接近的安全相关章节（如“网络安全需求”上级章节）或模板中对应章节。
   - 在目标章节内：你必须对比参考与基底，**自动识别**差异类型（新增 / 细化或替换 / 删除或废止表述），再按下面处理：
     - 若基底已覆盖参考要点且表述一致：**不要重复堆砌**；无需为“多改一点”而改写。
     - 若部分覆盖、表述不完整或与参考不一致：在原条目/段落上**细化、补缺、替换或统一术语**，使与参考对齐。
     - 若参考有要求而基底完全缺失：在该章节**新增**条目或段落（沿用该章节编号规则；无法确定编号则用 TBD，禁止编造追溯编号）。
     - 若参考表明某条应删除、合并或不再适用：在允许范围内**删除或改写**对应内容（就地 patch 时用可审计的删除/替换类操作）。
   - 新增需求时的编号规则：严格沿用该章节现有编号/格式（例如 REQ-xxx、SRS-xxx、NCS-xxx 等）。若无法从基底中确定编号生成规则，则不要凭空造编号；可用“（TBD 编号）”占位或沿用模板同位置格式。
6. **重要——同一文档内一致性同步（必须自洽）**：
   - 当你对“模块/功能项/系统组成/术语命名”做了新增、删减、合并或重命名时，必须在同一文档内同步更新所有相关表达，避免只改一处导致不一致。
   - 你必须快速回扫基底文本中与该信息相关的其它位置，并同步更新。至少包含：
     - 正文段落中的模块列举句/系统组成描述（如 “The system consists of … modules”）
     - 模块清单表/功能表（Module、Function description 等表格行/单元格）
     - 图相关说明：图题/图注/图下说明/与图紧邻的段落描述（如逻辑结构/架构图下面的模块说明）
     - 交叉引用处：如“见图/见表”附近的文字需与图/表一致（不要虚构图号/表号）
   - 若模块信息出现在“图片内部”（流程图/结构图文字）而无法直接修改图片内容：不得忽略一致性。应在图片下方或紧邻处新增“图示模块说明/模块清单”段落，把模块名称与正文/表格保持一致；必要时在待办处提示需要更换/更新图片，但不得编造页码。
   - 若同一关键词在多个位置出现且都应一致：应一并更新所有相关位置；若锚点多处出现可用 require_unique=false（就地 patch 模式）或在全文输出中保持一致替换（非 patch 模式）。
7. **重要——同一项目多份文档互相关联**：
   - 若本次生成包含多份目标文档（例如 SRS/风险分析/追溯/软件描述等），同一项目的核心信息必须保持一致：项目名称/产品名称/型号、系统模块命名、术语缩写、日志/权限等关键规则的表述口径。
   - 发生冲突时：以 INPUT_DOCS_EXCERPT / INPUT_SYSTEM_FUNCTIONALITY 中的“项目实际资料”为准，并在允许修改范围内同步修正其它文档表述，避免“某文档新增了模块但另一个文档仍旧模块清单缺失/不一致”。
8. 法规/编号/追溯 ID：除非模板本身对应位置必须变化，否则保持与模板一致；任何出现的 URS/REQ/CS/CE/HC 等编号必须逐字复制，不得改写/补零/拼接。
9. 若输入文档未提供足够证据：请不要编造；在允许修改的位置保留模板原文或写入 TBD（不得虚构编号与证据来源）。
10. 输出要求：只输出"替换后的完整文档文本"，不要输出 JSON、不要输出解释、不要输出 markdown 代码块标记（不要用 ```）。
11. **语言一致性（必须遵守）**：若【文档语言】为英文（English / en / 英文版），则你输出的全文（包括新增段落、表格内容、图注/说明、修订/变更描述等）必须为英文；即使 INPUT_DOCS_EXCERPT 为中文，也只能将其信息翻译/转写为英文后写入，不得夹带中文。

【项目约束（用于防越界）】
- 注册国家：{registration_country}
- 注册类别：{registration_type}
- 注册组成：{registration_component}
- 项目形态：{project_form}
- 适用范围（scope_of_application）：{scope_of_application}
- 文档语言：{document_language}

【NEW_BASIC_INFO（从输入文档提取/或按下述值）】
{new_basic_info}

【INPUT_SYSTEM_FUNCTIONALITY（来自输入文档提炼，供替换系统功能部分）】
{input_system_functionality}

【INPUT_DOCS_EXCERPT（参考文件内容摘录——你必须逐份阅读并将相关要点更新到基底文档对应章节；不得发散到项目范围之外）】
{input_docs_excerpt}

【BASE_OUTLINE（基底章节纲要——用于自动判断“放在哪”）】
{base_outline}

【EXISTING_DRAFT_TEXT（可选：用户上传的已有文件正文；若非空则它是唯一基底）】
{existing_draft_text}

【模板文档（必须遵循其格式/编号/非系统功能内容）】
TEMPLATE_FILE_NAME: {template_file_name}
TEMPLATE_TEXT:
{template_text}

【TEMPLATE_REFERENCE_TEXT（可选：案例知识库同名文件的参考文本，用于对齐章节与补缺；不得机械复制无关内容）】
{template_reference_text}
"""

_DOC_INPLACE_PATCH_PROMPT_TEMPLATE = """你将对"基础文档（EXISTING_DRAFT_TEXT）"进行**就地修改指令**生成，用于后续由程序在 docx/xlsx 内执行，必须可审计、可定位；修改项由**参考文件 vs 基础文档**的差异决定，需逐项覆盖新增、细化（替换/补缺）、删除等，而非刻意减少改动条数。

【硬约束（来自 skills/rules）——必须逐条遵守】
{cursor_skills_rules}

{strategy_block}

{author_role_block}
{qa_extra_block}
{mdsw_framework_block}

【任务目标】
- 你需要根据 INPUT_SYSTEM_FUNCTIONALITY 与 INPUT_DOCS_EXCERPT，检查 EXISTING_DRAFT_TEXT 中与"系统功能/软件功能/模块列表/界面说明/操作流程/功能说明/基本信息字段/权限/审计/日志"等相关的内容，判断哪些地方需要修改。
- 你不能依赖"固定章节标题"。必须使用"锚点文本/表格表头+行关键字"等方式自动定位。
- 你必须避免破坏格式：只输出可执行的 patch + 修改后的 UPDATED_TEXT（用于知识库入库与人工预览）。
- 禁止新增新的章节/标题：不得新增 Heading/标题段落。若需补充参考文件中的新信息，应在**已有章节内容中**插入新段落，或在**已有表格**中插入新行。
 - **同一文档内一致性同步（必须自洽）**：对某处插入/替换/删除后，你必须回扫 EXISTING_DRAFT_TEXT 中与该信息相关的其它位置，并对这些位置生成对应 operation，同步保持一致。至少覆盖：正文模块列举句、模块清单表/功能表（Module 列）、图题/图注/图下说明/与图紧邻段落、交叉引用处。
 - 若模块信息主要体现在“图片内部”（结构图/流程图文字），你无法直接修改图片：不得忽略一致性。必须在图片下方或紧邻处插入“图示模块说明/模块清单”段落（insert_paragraph_after_contains），使模块列表与正文/表格一致；不得虚构图号/表号/页码。
 - **同一项目多份文档互相关联**：若本次涉及多份目标文件，你在任意一份里新增/删除了模块或规则，必须在其它目标文件中对应位置也生成 operation，使多个文档口径一致、互相引用不冲突。
 - **语言一致性（必须遵守）**：若【文档语言】为英文（English / en / 英文版），则 PATCH_JSON 中写入的所有 new_text 必须为英文；UPDATED_TEXT 也必须为英文；不得夹带中文。

【全自动适配“落点”（不要反问用户放哪）】
- 你必须根据「BASE_OUTLINE（基底章节纲要）」判断应落入的章节位置，并在 EXISTING_DRAFT_TEXT 中找到可锚定的段落/表格进行插入/替换。
- 对“权限管理”“审计追踪/日志”等内容：优先在“网络安全需求/安全需求/访问控制/审计/日志”相关段落附近插入或补全。
- 你必须对比参考与基础，**自动识别**需 **新增 / 细化或替换 / 删除** 的位置，再生成 operation：
  - 若基础已覆盖参考要点且表述一致：不必重复插入；无需为“少下几条指令”而省略必要核对。
  - 若部分覆盖或与参考不一致：对原条目或单元格做**补缺、替换、细化**，使与参考一致。
  - 若参考有要求而基础完全缺失：用 insert 类 operation **新增**段落或表格行（不得编造编号；格式沿用相邻条目，无法确定则用 TBD）。
  - 若参考表明条目应移除或废止：在可锚定前提下使用 **delete** 或替换为符合参考的表述。

【多份参考文档（重要）】
- INPUT_DOCS_EXCERPT 中若出现多个「【输入/参考文件：…】」块，表示用户上传了多份独立资料（例如「权限」与「日志」各一份）。你必须**分别阅读每一块**，凡是**在基础文档中存在对应可改写的段落/表格**的要点，都应尽量**单独一条 operation**，不要合并成一条"只改一处"。
- 若某份参考中的要点在 EXISTING_DRAFT_TEXT 里**没有对应章节/句子**可锚定，则**不要**为凑数而虚构 anchor；此类遗漏可在 UPDATED_TEXT 末尾用简短「待补充说明」段落列出（仍不得编造编号）。
- 若参考文件要求的是“新增信息”，而基础文档对应章节已存在标题/小节且需要在原章节中补充内容：
  - 优先使用 `insert_paragraph_after_contains`：以该章节内已有段落或小节标题（EXISTING_DRAFT_TEXT 中原文）作为 anchor，把新内容插入其后（不得新增新标题）。
  - 若需要补充到原表格中（新增行）：可使用 `insert_table_row_after_contains`，以表头或某行关键单元格原文作为 anchor，在其后插入新行；new_text 用制表符 `\\t` 分隔每列（列数需与原表格一致或更少），且**至少有一列含非空白字符**；禁止只输出一串 `\\t`（分列后全空会被视为无效、不插入行）。
  - **测试用例/追溯表（程序侧已处理，模型只需给行内容）**：首列用例编号形如 `GN3-21`、`GN3-250`（前缀-数字）时，导出程序会扫描**整张表**内同前缀的最大编号；**新插入行**的首列会自动设为「最大编号 +1」（例如在已有最大 `GN3-250` 时新行为 `GN3-251`），避免与已有编号冲突。若首列编号**已在表中出现**：程序会比较**整行文本相似度**——达到或超过 90% 则**原地更新**该行，低于 90% 则视为不同用例并**插入新行且首列用最大编号+1**。你仍可用 `insert_table_row_after_contains` 定位锚点行，new_text 首列可写意图编号或占位，程序会按规则调整。
- 若目标内容属于「变更记录/Change record/Revision history」等**表格**：必须优先用表格类 operation（`replace_table_cell_contains` / `insert_table_row_after_contains`）在对应单元格/行内填写；不要把变更记录写成普通段落。
- 禁止为“少写几条 operation”而把多份参考里**本应分处落点**的信息硬塞进同一条 `replace_paragraph_contains`，除非基础文档里客观只有一处可承载全部信息。

【允许修改项】
1) 默认范围：
   - 基本信息字段：项目名称、产品名称、型号规格、注册单元名称
   - 系统功能相关部分：与系统功能相关的章节或表格（只更新这些部分）
   - 法规/编号/追溯 ID：保持不变（除非基底本身在允许修改项内且必须变化）
2) **本文件专用 Skills 扩展**：若上方硬约束中包含「本文件专用 Skills」，则其列举的每一条操作指令（如"变更记录追加""目录更新""预期用途替换""版本号升级"等）均属于本次允许修改的范围，你必须逐条生成对应 operation，不得因默认限制而跳过。
3) **充分利用参考文件内容**：INPUT_DOCS_EXCERPT 中的每份参考文件都包含本项目的实际业务信息（如权限列表、日志规则、功能描述等）。你必须仔细阅读每份参考文件的内容，将其中与基础文档相关章节对应的信息逐项生成 operation 更新到对应位置，不能只更新部分要点而遗漏其余。

【输出格式（必须严格遵守）】
你必须只输出以下两段，顺序固定，不能输出解释，不能输出 markdown 代码块（不要用 ```）：

### PATCH_JSON
<一个 JSON 对象>

### UPDATED_TEXT
<替换后的完整文档文本>

【PATCH_JSON 约束】
- 顶层字段必须包含：operations（数组）
- 每个 operation 必须包含：
  - type: 只能是 "replace_paragraph_contains" 或 "replace_table_cell_contains" 或 "insert_paragraph_after_contains" 或 "insert_table_row_after_contains" 或 "delete_paragraph_contains"
  - anchor: 用于定位的锚点短语（必须来自 EXISTING_DRAFT_TEXT 的原文片段，尽量短且唯一）
  - new_text: 替换后的文本（可多行）
  - require_unique: true/false（默认 true；若锚点可能多处出现且你能容忍多处替换才设 false）
 - 当你需要同步修改同一表述在多处出现的位置：应优先为每处落点生成单独 operation；若确实是同一锚点多处都应替换且内容一致，可设置 require_unique=false 允许多处命中替换。
- **图示一致性（可选但推荐）**：若你新增/删减/重命名了模块，且基础文档存在“Logical structure/结构图”等图示说明段落（或图题/图注），你可以额外生成一个 operation 用于替换图示（程序会生成正式风格的框图并插入/替换图片）：  
  - type: "replace_diagram_after_paragraph_contains"  
  - anchor: 图题/图注/或“Logical structure”等附近段落中可唯一定位的原文短语  
  - new_text: 以逗号分隔的模块名称列表（按文档一致命名）  
  - require_unique: 通常 true  
- 你不得输出"整篇重写"类型操作；但在**多份参考文档**且**基础文档多处可对应**时，operations 应**覆盖多条**（可定位前提下），不要故意压成一条。

【项目约束（用于防越界）】
- 注册国家：{registration_country}
- 注册类别：{registration_type}
- 注册组成：{registration_component}
- 项目形态：{project_form}
- 适用范围（scope_of_application）：{scope_of_application}
- 文档语言：{document_language}

【NEW_BASIC_INFO（从输入文档提取/或按下述值）】
{new_basic_info}

【INPUT_SYSTEM_FUNCTIONALITY（来自输入文档提炼，供替换系统功能部分）】
{input_system_functionality}

【INPUT_DOCS_EXCERPT（参考文件内容摘录——你必须逐份阅读、逐个要点在基础文档中定位并生成 operation；不得发散到项目范围之外）】
{input_docs_excerpt}

【BASE_OUTLINE（基底章节纲要——用于自动判断“放在哪”）】
{base_outline}

【EXISTING_DRAFT_TEXT（基础文档正文；你生成的 patch 必须能在此文本中找到 anchor）】
{existing_draft_text}

【模板文档（仅供对齐章节/缺失补齐的参考；不得覆盖基底结构）】
TEMPLATE_FILE_NAME: {template_file_name}
TEMPLATE_TEXT:
{template_text}

【TEMPLATE_REFERENCE_TEXT（可选：案例知识库同名文件参考文本）】
{template_reference_text}
"""


def _parse_patch_and_updated_text(raw: str) -> Tuple[str, str]:
    txt = (raw or "").strip()
    if not txt:
        return "", ""
    m = re.search(r"###\s*PATCH_JSON\s*(\{[\s\S]*?\})\s*###\s*UPDATED_TEXT\s*([\s\S]*)\s*$", txt)
    if not m:
        return "", txt
    return (m.group(1) or "").strip(), (m.group(2) or "").strip()


class DocumentDraftGenerator:
    def __init__(self, collection: str):
        self.collection = collection
        self.agent = ReviewAgent(collection)
        self._tc_id_rules_cache: Dict[str, List[Dict[str, Any]]] = {}

    def _extract_tc_id_rules_from_kb(
        self,
        *,
        template_file_name: str,
        provider: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """从知识库中提取适用于当前目标文件的测试用例编号规则（tc_id_rules）。

        依据：用户已将《软件可追溯性分析管理制度》训练入库，且其中“编号规则”章节对不同文件类型有不同规定。
        输出：draft_export.py 可消费的 tc_id_rules（list of {name,regex,prefix_group,number_group,render}）。
        """
        key = (template_file_name or "").strip()
        if not key:
            return []
        if key in self._tc_id_rules_cache:
            return self._tc_id_rules_cache[key]

        # 1) 检索知识库：优先命中文件名 + 编号规则章节
        kb = getattr(self.agent, "kb", None)
        if not kb:
            self._tc_id_rules_cache[key] = []
            return []

        queries = [
            "软件可追溯性分析管理制度 编号规则 测试用例 编号",
            "可追溯性 分析 管理 制度 编号规则 用例编号",
            "编号规则 测试用例 规则 递增 前缀",
            f"软件可追溯性分析管理制度 编号规则 {Path(key).suffix}",
            f"编号规则 {Path(key).name}",
        ]
        docs: List[Document] = []
        seen = set()
        for q in queries:
            try:
                res = kb.search(q, top_k=8)
            except Exception:
                res = []
            for d in res or []:
                src = (getattr(d, "metadata", {}) or {}).get("source_file") or ""
                txt = (getattr(d, "page_content", "") or "")
                k2 = (src, txt[:120])
                if k2 in seen:
                    continue
                seen.add(k2)
                # 尽量优先保留制度文件来源；若 source_file 不含名称也保留，交给模型判别
                docs.append(d)
            if len(docs) >= 18:
                break
        if not docs:
            self._tc_id_rules_cache[key] = []
            return []

        # 2) 组装上下文（控制长度，避免挤掉主生成内容）
        blocks: List[str] = []
        total = 0
        cap = 9000
        for d in docs:
            src = (getattr(d, "metadata", {}) or {}).get("source_file") or "未知来源"
            txt = (getattr(d, "page_content", "") or "").strip()
            if not txt:
                continue
            seg = f"【来源：{src}】\n{txt}"
            blocks.append(seg)
            total += len(seg)
            if total >= cap:
                break
        kb_ctx = "\n\n---\n\n".join(blocks)[:cap]

        def _tc_rules_from_policy_37(text: str, file_name: str) -> List[Dict[str, Any]]:
            """从《软件可追溯性分析管理制度》3.7 编号规则（或等价文字）中确定性构造规则。"""
            raw = (text or "")
            if not raw.strip():
                return []
            # 命中“3.7 编号规则”或任一关键小节，认为可用
            if not (
                ("3.7" in raw and "编号规则" in raw)
                or ("系统测试用例ID编号规则" in raw)
                or ("单元测试用例ID编号规则" in raw)
                or ("用户测试用例ID编号规则" in raw)
            ):
                return []

            fn = (file_name or "").lower()
            want = "all"
            if any(k in fn for k in ["system test", "system test case", "appendix 4", "stc", "gn"]):
                want = "gn"
            elif any(k in fn for k in ["unit test", "unit test case", "utc", "ut"]):
                want = "ut"
            elif any(k in fn for k in ["user test", "confirmation", "ct"]):
                want = "ct"

            # 规则按制度内容构造：prefix_group 捕获前缀（含模块号/单元名），number_group 捕获流水号
            all_rules: List[Dict[str, Any]] = [
                # 单元测试用例：UTn-XXX，从001开始（补零到3位，>999自然扩展为4位及以上）
                {
                    "name": "unit_test_case_UTn",
                    "regex": r"^(UT\d+)-(\d+)$",
                    "prefix_group": 1,
                    "number_group": 2,
                    "render": "{prefix}-{num:03d}",
                },
                # 系统测试用例：GNn-X，从1开始（不强制补零）
                {
                    "name": "system_test_case_GNn",
                    "regex": r"^(GN\d+)-(\d+)$",
                    "prefix_group": 1,
                    "number_group": 2,
                    "render": "{prefix}-{num}",
                },
                # 用户测试用例：CTn-XXX，从001开始（补零到3位，>999自然扩展）
                {
                    "name": "user_test_case_CTn",
                    "regex": r"^(CT\d+)-(\d+)$",
                    "prefix_group": 1,
                    "number_group": 2,
                    "render": "{prefix}-{num:03d}",
                },
                # 用户需求：URS_XXX（制度示例含 URS001 / URS_0001 等，兼容下划线）
                {
                    "name": "user_req_URS",
                    "regex": r"^(URS)_?(\d+)$",
                    "prefix_group": 1,
                    "number_group": 2,
                    "render": "{prefix}{num:03d}",
                },
                # 软件需求：SRS_XXX（示例含 SRS_001 / SRS_0001）
                {
                    "name": "sw_req_SRS",
                    "regex": r"^(SRS)_?(\d+)$",
                    "prefix_group": 1,
                    "number_group": 2,
                    "render": "{prefix}_{num:03d}",
                },
                # 详细设计：DS-A_XXX（A 为软件单元名称）
                {
                    "name": "detail_design_DS_unit",
                    "regex": r"^(DS-[A-Za-z0-9]+)_(\d+)$",
                    "prefix_group": 1,
                    "number_group": 2,
                    "render": "{prefix}_{num:03d}",
                },
                # 风险分析措施：CSXX / CSWA01（将 WA 作为前缀一部分）
                {
                    "name": "risk_control_CS_num",
                    "regex": r"^(CS)(\d+)$",
                    "prefix_group": 1,
                    "number_group": 2,
                    "render": "{prefix}{num:02d}",
                },
                {
                    "name": "risk_control_CSWA",
                    "regex": r"^(CSWA)(\d+)$",
                    "prefix_group": 1,
                    "number_group": 2,
                    "render": "{prefix}{num:02d}",
                },
            ]

            # 目标文件优先：测试用例表只用对应规则，其他文件返回全量以兜底
            if want == "ut":
                return [all_rules[0]]
            if want == "gn":
                return [all_rules[1]]
            if want == "ct":
                return [all_rules[2]]
            return all_rules

        # 2.5) 若命中制度 3.7 编号规则：用确定性规则（不走 LLM 抽取）
        try:
            det = _tc_rules_from_policy_37(kb_ctx, key)
            if det:
                self._tc_id_rules_cache[key] = det
                return det
        except Exception:
            pass

        # 3) 用 LLM 抽取结构化规则（只抽取，不编造）
        prompt = (
            "你将从制度/程序摘录中提取“测试用例编号规则（tc_id_rules）”，用于文档就地 patch 的表格行插入。\n"
            "目标文件名：{file_name}\n\n"
            "【制度/程序摘录（来自知识库检索）】\n"
            "{kb_ctx}\n\n"
            "【输出要求（必须严格）】\n"
            "仅输出一个 JSON 对象，不要解释，不要 markdown，不要代码块标记。\n"
            "JSON 格式：\n"
            "{\n"
            '  "tc_id_rules": [\n'
            "    {\n"
            '      "name": "规则名（简短）",\n'
            '      "regex": "用于匹配首列用例编号的正则（必须能捕获 prefix 与 number）",\n'
            '      "prefix_group": 1,\n'
            '      "number_group": 2,\n'
            '      "render": "用 Python format 生成新编号的模板（可补零），可用 {prefix} 与 {num}，例如 {prefix}-{num:03d}"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "【硬约束】\n"
            "- 只基于摘录中“编号规则”章节能明确支持的内容提取；摘录未提供则输出空数组 tc_id_rules=[]。\n"
            "- 若制度对不同文件类型有不同规则：请选择最适用于目标文件名/文件类型的一组规则；不确定则输出空数组。\n"
        ).format(file_name=key, kb_ctx=kb_ctx)

        try:
            use_cursor = settings.is_cursor or (provider or "").strip().lower() == "cursor"
            raw = (complete_task(prompt) if use_cursor else invoke_chat_direct(prompt, temperature=0.1, provider=provider)) or ""
            obj = json.loads((raw or "").strip())
            rules = obj.get("tc_id_rules") if isinstance(obj, dict) else None
            if not isinstance(rules, list):
                rules = []
            # 轻量校验：regex 必须存在
            cleaned: List[Dict[str, Any]] = []
            for r in rules:
                if not isinstance(r, dict):
                    continue
                rx = str(r.get("regex") or "").strip()
                if not rx:
                    continue
                cleaned.append(
                    {
                        "name": str(r.get("name") or "tc_rule").strip() or "tc_rule",
                        "regex": rx,
                        "prefix_group": int(r.get("prefix_group") or 1),
                        "number_group": int(r.get("number_group") or 2),
                        "render": str(r.get("render") or "{prefix}-{num}").strip() or "{prefix}-{num}",
                    }
                )
            self._tc_id_rules_cache[key] = cleaned
            return cleaned
        except Exception:
            self._tc_id_rules_cache[key] = []
            return []

    def _gen_one_file(
        self,
        *,
        base_case_id: int,
        template_file_name: str,
        template_text: str,
        existing_draft_text: str = "",
        template_reference_text: str = "",
        registration_country: str,
        registration_type: str,
        registration_component: str,
        project_form: str,
        scope_of_application: str,
        document_language: str,
        new_basic_info: str,
        input_system_functionality: str,
        input_docs_excerpt: str,
        draft_strategy: str = "change",
        author_role: str = "",
        inplace_patch: bool = False,
        workspace_root: Optional[Path] = None,
        provider: Optional[str] = None,
    ) -> Tuple[str, bool]:
        try:
            _row_pf = get_draft_file_skills_rules(self.collection, int(base_case_id), template_file_name)
        except Exception:
            _row_pf = None
        _esk = ((_row_pf or {}).get("skills_patch") or "").strip()
        _eru = ((_row_pf or {}).get("rules_patch") or "").strip()
        had_per_file = bool(_esk or _eru)

        extra_block = ""
        if _esk or _eru:
            extra_block = (
                "=== 本文件专用 Skills（按「模板案例+文件名」保存的配置）===\n"
                + "你必须逐条执行以下 Skills 指令，每一条都代表用户明确要求的修改操作：\n"
                + (_esk or "（无）")
                + "\n\n=== 本文件专用 Rules（按「模板案例+文件名」保存的配置）===\n"
                + "以下 Rules 是本文件的硬性约束，你的输出必须遵守：\n"
                + (_eru or "（无）")
            )
        # 单文件块过长时单独截断，避免挤掉仓库 skills/rules 或反过来
        _MAX_PER_FILE_BLOCK = 12000
        if len(extra_block) > _MAX_PER_FILE_BLOCK:
            extra_block = extra_block[:_MAX_PER_FILE_BLOCK] + "\n【提示】本文件专用 skills/rules 过长已截断。"

        base = _collect_cursor_skills_rules_for_prompt(workspace_root=workspace_root)
        _MAX_TOTAL = 28000
        _SEP = "\n\n---\n\n"
        if extra_block:
            reserved = len(extra_block) + len(_SEP)
            base_max = max(6000, _MAX_TOTAL - reserved)
            if len(base) > base_max:
                base = base[:base_max] + "\n【提示】仓库 skills/rules 已截断以保留本文件专用配置。"
            cursor_sr = extra_block + _SEP + base
        else:
            cursor_sr = base
            if len(cursor_sr) > _MAX_TOTAL:
                cursor_sr = cursor_sr[:_MAX_TOTAL] + "\n【提示】skills/rules 总长度已截断。"
        if len(cursor_sr) > _MAX_TOTAL:
            cursor_sr = cursor_sr[:_MAX_TOTAL] + "\n【提示】skills/rules 总长度已截断。"

        tmpl = _DOC_INPLACE_PATCH_PROMPT_TEMPLATE if inplace_patch else _DOC_DRAFT_PROMPT_TEMPLATE
        base_blob = (existing_draft_text or "").strip() or (template_text or "").strip()
        try:
            base_outline = extract_section_outline_from_texts([base_blob], max_sections=90).strip()
        except Exception:
            base_outline = ""
        if not base_outline:
            base_outline = "（未能提取到章节纲要；请根据基底文本内容自行定位最接近的章节/段落。）"
        strat = (draft_strategy or "").strip().lower()
        if strat not in ("change", "reuse"):
            strat = "change"
        strategy_block = (
            "【生成策略】\n"
            + (
                "- change（注册变更）：以基础文件为当前递交底稿，对照参考（INPUT_DOCS_EXCERPT、INPUT_SYSTEM_FUNCTIONALITY 等）**自动识别**需 **新增、细化（补缺/替换）、删除** 的部位并执行；已与参考一致且无冲突的表述可保留；冲突或过时处以参考为准。修改须可定位、可审计，避免无依据的全文替换。\n"
                if strat == "change"
                else "- reuse（新项目复用模板）：保持格式/章节/编号风格不变，但对可变内容按参考文件做全量更新；避免只改一小段导致内容仍是旧项目。\n"
            )
        )
        author_role_block = _build_author_role_block(author_role)
        qa_extra_block = _build_qa_extra_block(author_role)
        prompt = tmpl.format(
            cursor_skills_rules=cursor_sr,
            strategy_block=strategy_block,
            author_role_block=author_role_block,
            qa_extra_block=qa_extra_block,
            mdsw_framework_block=_MDSW_FRAMEWORK_BLOCK,
            registration_country=registration_country,
            registration_type=registration_type,
            registration_component=registration_component,
            project_form=project_form,
            scope_of_application=scope_of_application or "",
            document_language=document_language or "",
            new_basic_info=new_basic_info or "",
            input_system_functionality=input_system_functionality or "",
            input_docs_excerpt=input_docs_excerpt or "",
            base_outline=base_outline,
            existing_draft_text=(existing_draft_text or "").strip()[:40000],
            template_file_name=template_file_name,
            template_text=template_text[:22000],
            template_reference_text=(template_reference_text or "").strip()[:12000],
        )

        if settings.is_cursor or (provider or "").strip().lower() == "cursor":
            # Cursor 模式下 skills/rules 也会参与；但此处仍把其内容注入 prompt，确保非 Cursor/或截断时依旧遵循硬约束。
            return ((complete_task(prompt) or "").strip(), had_per_file)

        return (invoke_chat_direct(prompt, temperature=DOC_DRAFT_GEN_TEMPERATURE, provider=provider).strip(), had_per_file)

    def generate(
        self,
        *,
        base_case_id: int,
        template_file_names: Optional[List[str]] = None,
        project_id: Optional[int] = None,
        existing_base_files: Optional[Dict[str, str]] = None,
        input_files: List[Tuple[str, str]],
        document_language: str,
        # 新项目维度（法规/边界保持一致时可直接等同模板）
        registration_country: str,
        registration_type: str,
        registration_component: str,
        project_form: str,
        # 与「② 项目与专属资料」一致的项目字段（用户填写优先；AI 提取仅补空）
        project_name: str = "",
        project_code: str = "",
        project_name_en: str = "",
        product_name: str = "",
        product_name_en: str = "",
        model: str = "",
        model_en: str = "",
        registration_country_en: str = "",
        scope_of_application_override: Optional[str] = None,
        persist_project_fields: bool = True,
        # 案例/项目元信息（允许留空，后续按提取结果补齐）
        new_case_name: str = "",
        project_key: str = "",
        # 基于你的输入：skills/rules 增量补丁文本（按文件块）
        skills_patch_text: str = "",
        rules_patch_text: str = "",
        # 模型 provider（不传则用 settings.provider；仅影响非 cursor 模式）
        provider: Optional[str] = None,
        # 就地修改（Base 存在时输出 patch JSON + UPDATED_TEXT）
        inplace_patch: bool = False,
        # 是否将本次生成结果写入“案例库(project_cases)”：用于“已完成案例项目”
        # 选择已有项目做日常编写时，建议 False（仅写入项目文档，不污染案例库）。
        save_as_case: bool = True,
        # 进度回调：用于 UI 显示当前阶段/文件
        progress_cb: Optional[Callable[[str, float], None]] = None,
        # 模板 case 文档读取上限（拼接模板文本用；避免超长上下文）
        base_case_limit_chunks: int = 2000,
        # 多份基础文件：(临时路径, 原始文件名)；与 multi_base_auto_route 配合，由 AI 为每个目标模板分配 Base 与参考子集
        base_files_manifest: Optional[List[Tuple[str, str]]] = None,
        multi_base_auto_route: bool = False,
        draft_strategy: str = "change",
        author_role: str = "",
        author_role_map: Optional[Dict[str, str]] = None,
    ) -> GeneratedCaseDocs:
        def _progress(msg: str, frac: float) -> None:
            try:
                if progress_cb:
                    progress_cb(str(msg or ""), float(frac))
            except Exception:
                pass

        _progress("准备开始…", 0.01)

        # 1) 每次任务开始前：更新 skills/rules（本地合并 + 去重）
        if (skills_patch_text or "").strip() or (rules_patch_text or "").strip():
            _progress("更新 skills/rules…", 0.03)
            apply_patch_updates(
                skills_patch_text=skills_patch_text,
                rules_patch_text=rules_patch_text,
                workspace_root=Path(__file__).resolve().parents[2],
            )

        base_case = get_project_case(base_case_id)
        if not base_case:
            raise RuntimeError(f"找不到 base_case_id={base_case_id}")

        scope_of_application = (
            (scope_of_application_override or "").strip()
            if scope_of_application_override is not None
            else (base_case.get("scope_of_application") or "").strip()
        )

        # 2) 保存输入文件到临时目录（用于 load_and_split/训练）
        with tempfile.TemporaryDirectory(prefix="aicheckword_draft_") as td:
            temp_dir = Path(td)
            saved_inputs: List[Tuple[str, str]] = []
            _progress("准备输入文件…", 0.06)
            for src_path, src_name in input_files:
                p = Path(src_path)
                if not p.exists():
                    raise FileNotFoundError(f"输入文件不存在：{src_path}")
                # 复制一份到临时目录，避免后续路径/权限影响训练或转换
                dst = temp_dir / (src_name or p.name)
                dst.write_bytes(p.read_bytes())
                saved_inputs.append((str(dst), dst.name))

            # 3) 选择已有项目或新建项目
            proj_existing = None
            if project_id:
                proj_existing = get_project(int(project_id))
                if not proj_existing:
                    raise RuntimeError(f"找不到 project_id={project_id}")
            if not project_id:
                # 新建：按用户填写优先；若为空则用模板占位，后续 AI 提取仅补空
                placeholder_project_name = (project_name or "").strip() or (base_case.get("case_name") or base_case.get("product_name") or "新项目")
                placeholder_product_name = (product_name or "").strip() or (base_case.get("product_name") or "").strip()
                placeholder_model = (model or "").strip()
                project_id = create_project(
                    collection=self.collection,
                    name=placeholder_project_name,
                    registration_country=registration_country or (base_case.get("registration_country") or ""),
                    registration_type=registration_type or (base_case.get("registration_type") or ""),
                    registration_component=registration_component or (base_case.get("registration_component") or ""),
                    project_form=project_form or (base_case.get("project_form") or ""),
                    scope_of_application=scope_of_application,
                    product_name=placeholder_product_name,
                    name_en=(project_name_en or "").strip(),
                    product_name_en=(product_name_en or "").strip(),
                    registration_country_en=(registration_country_en or "").strip(),
                    model=placeholder_model,
                    model_en=(model_en or "").strip(),
                    project_code=(project_code or "").strip(),
                )
            else:
                # 已有项目：可选写回字段（保持与②一致的数据互通）
                if persist_project_fields:
                    update_project(
                        project_id=int(project_id),
                        name=(project_name or "").strip() or None,
                        product_name=(product_name or "").strip() or None,
                        model=(model or "").strip() or None,
                        registration_country=registration_country or None,
                        registration_type=registration_type or None,
                        registration_component=registration_component or None,
                        project_form=project_form or None,
                        scope_of_application=scope_of_application or None,
                        name_en=(project_name_en or "").strip() or None,
                        product_name_en=(product_name_en or "").strip() or None,
                        registration_country_en=(registration_country_en or "").strip() or None,
                        model_en=(model_en or "").strip() or None,
                        project_code=(project_code or "").strip() or None,
                    )

            # 拉取项目最新信息（用于“空输入重跑”场景复用已保存的 basic_info/system_functionality）
            proj_latest = get_project(int(project_id)) or {}

            # 4) 训练输入文件到项目专属向量库，用于提取基本信息
            if saved_inputs:
                _progress(f"训练输入文件（{len(saved_inputs)} 个）…", 0.12)
            for fp, fn in saved_inputs:
                self.agent.train_project_docs(int(project_id), fp, file_name=fn)

            if saved_inputs:
                _progress("提取项目基本信息…", 0.22)
                extracted_basic_info_text = self.agent.extract_and_save_project_basic_info(int(project_id), provider=provider)
            else:
                # 允许“历史记录重新生成”不上传输入文件：复用项目中已保存的 basic_info_text
                extracted_basic_info_text = (proj_latest.get("basic_info_text") or "").strip()

            parsed_basic = _parse_basic_info_lines(extracted_basic_info_text)
            # 输入提取字段：项目名称/产品名称/型号规格/注册单元名称
            extracted_project_name = parsed_basic.get("项目名称") or parsed_basic.get("项目名称".strip()) or ""
            extracted_product_name = parsed_basic.get("产品名称") or ""
            extracted_model = parsed_basic.get("型号规格") or ""

            # 更新 projects 主字段：用户填写优先；AI 提取仅补空（避免覆盖用户录入）
            _name_final = (project_name or "").strip() or extracted_project_name
            _prod_final = (product_name or "").strip() or extracted_product_name
            _model_final = (model or "").strip() or extracted_model
            if persist_project_fields:
                update_project(
                    project_id=int(project_id),
                    name=_name_final or None,
                    product_name=_prod_final or None,
                    model=_model_final or None,
                    registration_country=registration_country or None,
                    registration_type=registration_type or None,
                    registration_component=registration_component or None,
                    project_form=project_form or None,
                    scope_of_application=scope_of_application or None,
                    name_en=(project_name_en or "").strip() if (project_name_en or "").strip() else None,
                    product_name_en=(product_name_en or "").strip() if (product_name_en or "").strip() else None,
                    registration_country_en=(registration_country_en or "").strip() if (registration_country_en or "").strip() else None,
                    model_en=(model_en or "").strip() if (model_en or "").strip() else None,
                    project_code=(project_code or "").strip() if (project_code or "").strip() else None,
                )

            # 5) 提取/更新系统功能描述（供项目审核一致性核对）
            #    这里直接从输入文档“可读文本”提炼，不依赖 project_knowledge_text（避免训练/提取耦合）
            input_docs_texts: List[str] = []
            for fp, _fn in saved_inputs:
                try:
                    docs = load_single_file(fp)
                    # 防止长 PDF/长 Word 仅取开头导致关键信息遗漏：构建“头尾+关键词命中”的稳健摘录
                    per_file_cap = 18000
                    _txt = _smart_excerpt_from_docs(docs, file_name=Path(fp).name, max_chars=per_file_cap)
                    input_docs_texts.append(
                        f"【输入/参考文件：{Path(fp).name}】\n{_txt}"
                    )
                except Exception:
                    # 训练阶段通常可通过；此处失败就降级：把文件名占位
                    input_docs_texts.append(f"【输入文档加载失败/仅文件名占位：{Path(fp).name}】")
            combined_input_text = "\n\n".join(t for t in input_docs_texts if t.strip())

            if saved_inputs and combined_input_text.strip():
                _progress("提取系统功能描述…", 0.30)
                sys_fun_text = identify_system_functionality_with_llm(
                    raw_content=combined_input_text,
                    source_hint="输入的 Word/Excel/PDF 文档",
                    provider=provider,
                )
            else:
                # 空输入重跑：复用项目中已保存的 system_functionality_text
                sys_fun_text = (proj_latest.get("system_functionality_text") or "").strip()

            # 系统功能写回项目（用于后续按项目审核一致性核对）
            update_project_system_functionality(
                project_id=int(project_id),
                system_functionality_text=sys_fun_text,
                system_functionality_source="input_docs",
            )

            # 6) 是否写入“案例库(project_cases)”
            #    - True：创建 project_case，并将生成文档作为 case 文档入库（category=project_case + case_id）
            #    - False：不创建 project_case，仅将生成文档作为项目文档入库（category=project_doc，不带 case_id）
            project_case_id: Optional[int] = None
            if save_as_case:
                _progress("创建案例库记录（project_cases）…", 0.38)
                parsed_product_name = _prod_final or (base_case.get("product_name") or "").strip()
                parsed_registration_country = registration_country or (base_case.get("registration_country") or "").strip()

                resolved_case_name = (new_case_name or "").strip()
                if not resolved_case_name:
                    resolved_case_name = (base_case.get("case_name") or "").strip() or parsed_product_name or "新项目案例"

                project_case_id = create_project_case(
                    collection=self.collection,
                    case_name=resolved_case_name,
                    product_name=parsed_product_name,
                    registration_country=parsed_registration_country,
                    registration_type=registration_type or (base_case.get("registration_type") or ""),
                    registration_component=registration_component or (base_case.get("registration_component") or ""),
                    project_form=project_form or (base_case.get("project_form") or ""),
                    scope_of_application=scope_of_application,
                    document_language=document_language or (base_case.get("document_language") or "zh"),
                    project_key=project_key or (base_case.get("project_key") or ""),
                )

            # 7) 确定本次要生成的模板文件名列表（支持下拉搜索选择）
            all_case_files = get_project_case_file_names(self.collection, base_case_id) or []
            chosen = [x for x in (template_file_names or []) if (x or "").strip()]
            if not chosen:
                chosen = list(all_case_files)
            # 防御：只生成该 case 下存在的文件
            chosen = [x for x in chosen if x in all_case_files]
            if not chosen:
                raise RuntimeError(f"base_case_id={base_case_id} 在 knowledge base 中无可用 project_case 文件名")

            # 性能优化：开启“就地修改(inplace_patch)”且用户提供了 Base 绑定时，
            # 仅对“有 Base 的目标文件”进行生成/patch，避免对未绑定 Base 的文件逐个调用大模型导致极慢。
            if inplace_patch and existing_base_files:
                base_keys = {k for k in (existing_base_files or {}).keys() if (k or "").strip()}
                chosen_with_base = [x for x in chosen if x in base_keys]
                if chosen_with_base:
                    chosen = chosen_with_base

            # 输入摘要（避免 prompt 爆上下文）
            _names_hint = ""
            if saved_inputs:
                _nl = [Path(x[0]).name for x in saved_inputs if x and x[0]]
                if _nl:
                    _names_hint = f"【输入/参考文件清单】共 {len(_nl)} 个：{', '.join(_nl)}\n\n"
            input_excerpt = (_names_hint + combined_input_text.strip()).strip()
            # 多文件场景下允许更大的摘要上限（已经做了 per_file_cap），避免遗漏关键差异
            if len(input_excerpt) > 42000:
                input_excerpt = input_excerpt[:42000] + "\n【提示】输入/参考文件摘要已截断。"

            # basic_info_text 作为“新项目基本信息”
            new_basic_info_for_prompt = extracted_basic_info_text.strip() or ""

            # 多基础 / 多参考：为每个目标模板分配基础文件路径与参考文件子集
            routing_plan: Optional[Dict[str, Any]] = None
            route_map: Dict[str, Dict[str, Any]] = {}
            if multi_base_auto_route and (base_files_manifest or []):
                _progress("规划多基础文件与参考文件路由（AI）…", 0.42)
                routing_plan = _plan_multi_base_route_llm(
                    chosen=chosen,
                    base_manifest=list(base_files_manifest or []),
                    ref_manifest=list(saved_inputs),
                    provider=provider,
                )
                for a in routing_plan.get("assignments") or []:
                    if not isinstance(a, dict):
                        continue
                    tf = (a.get("template_file") or "").strip()
                    if not tf:
                        continue
                    route_map[tf] = {
                        "base_path": a.get("_base_path"),
                        "reference_files": a.get("reference_files"),
                    }
            else:
                if existing_base_files:
                    for tf in chosen:
                        pth = existing_base_files.get(tf)
                        if pth:
                            route_map[tf] = {"base_path": pth, "reference_files": None}

            # 性能优化：就地修改且已得到路由时，仅生成“有基础路径”的目标
            if inplace_patch and route_map:
                _wk = [x for x in chosen if (route_map.get(x) or {}).get("base_path")]
                if _wk:
                    chosen = _wk

            if not chosen:
                raise RuntimeError(
                    "没有可生成的目标文件：就地修改需要可用的基础文件。"
                    "请上传基础文件并开启「自动分配」，或关闭自动分配后在下方将 Base 绑定到目标文件名。"
                )

            generated: Dict[str, str] = {}
            patches: Dict[str, str] = {}
            per_file_skills_rules_applied: Dict[str, bool] = {}
            per_file_base_path: Dict[str, str] = {}
            total = max(len(chosen), 1)
            for template_file_name in chosen:
                i = len(generated) + 1
                _progress(f"生成中（{i}/{total}）：{template_file_name}", 0.45 + 0.45 * (i - 1) / total)
                rows = get_knowledge_docs_by_case_id_and_file_name(
                    collection=self.collection,
                    case_id=base_case_id,
                    file_name=template_file_name,
                    limit=base_case_limit_chunks,
                )
                if not rows:
                    # 跳过空文件（一般不应发生）
                    continue
                template_text = "\n\n".join((r.get("content") or "").strip() for r in rows if (r.get("content") or "").strip()).strip()
                if not template_text:
                    continue

                # 可选：用户上传已有文件作为基底（继续编写）；多基础时由 route_map 指定路径
                rte = route_map.get(template_file_name) or {}
                _base_path = (rte.get("base_path") or "").strip()
                if not _base_path and existing_base_files and template_file_name in existing_base_files:
                    _base_path = (existing_base_files.get(template_file_name) or "").strip()
                existing_text = ""
                if _base_path:
                    try:
                        _p = Path(_base_path)
                        if _p.exists():
                            docs0 = load_single_file(str(_p))
                            existing_text = "\n\n".join((d.page_content or "") for d in docs0 if (d.page_content or "").strip()).strip()
                    except Exception:
                        existing_text = ""

                _ref_names = rte.get("reference_files")
                if _ref_names is None:
                    input_excerpt_this = input_excerpt
                else:
                    _nl = [x for x in (_ref_names or []) if (x or "").strip()]
                    _hint = (
                        f"【输入/参考文件清单（本目标路由子集）】共 {len(_nl)} 个：{', '.join(_nl)}\n\n"
                        if _nl
                        else "【输入/参考文件清单（本目标路由子集）】无\n\n"
                    )
                    input_excerpt_this = (_hint + _filter_combined_input_by_reference_names(combined_input_text, _ref_names)).strip()
                    if len(input_excerpt_this) > 42000:
                        input_excerpt_this = input_excerpt_this[:42000] + "\n【提示】输入/参考文件摘要已截断。"

                # 生成单文件
                raw_out, had_per_file_sr = self._gen_one_file(
                    base_case_id=int(base_case_id),
                    template_file_name=template_file_name,
                    template_text=(existing_text or template_text),
                    existing_draft_text=existing_text or "",
                    template_reference_text=(template_text if existing_text else ""),
                    registration_country=registration_country or "",
                    registration_type=registration_type or "",
                    registration_component=registration_component or "",
                    project_form=project_form or "",
                    scope_of_application=scope_of_application or "",
                    document_language=document_language or "",
                    new_basic_info=new_basic_info_for_prompt,
                    input_system_functionality=sys_fun_text,
                    input_docs_excerpt=input_excerpt_this,
                    draft_strategy=draft_strategy,
                    author_role=((author_role_map or {}).get(template_file_name) or author_role or ""),
                    inplace_patch=bool(inplace_patch and bool(existing_text)),
                    workspace_root=Path(__file__).resolve().parents[2],
                    provider=provider,
                )
                if not (raw_out or "").strip():
                    raise RuntimeError(f"生成失败：{template_file_name} 返回空文本")

                patch_json, new_text = _parse_patch_and_updated_text(raw_out)
                if bool(inplace_patch and bool(existing_text)):
                    # patch 模式下必须同时有 patch_json 与 updated_text；否则降级为全文
                    if patch_json.strip():
                        # 从知识库中的《软件可追溯性分析管理制度》编号规则章节提取 tc_id_rules，并注入到 patch 顶层
                        try:
                            p_obj = json.loads(patch_json)
                            if isinstance(p_obj, dict) and "tc_id_rules" not in p_obj:
                                tc_rules = self._extract_tc_id_rules_from_kb(
                                    template_file_name=template_file_name, provider=provider
                                )
                                if tc_rules:
                                    p_obj["tc_id_rules"] = tc_rules
                                    patch_json = json.dumps(p_obj, ensure_ascii=False)
                        except Exception:
                            pass
                        patches[template_file_name] = patch_json.strip()
                if not (new_text or "").strip():
                    raise RuntimeError(f"生成失败：{template_file_name} 返回空正文")

                # 生成文件名：沿用模板原名；若填写了项目编号则替换前缀
                out_file_name = _replace_file_project_code_prefix(template_file_name, (project_code or "").strip())
                _out_key = out_file_name or template_file_name
                per_file_skills_rules_applied[_out_key] = bool(had_per_file_sr)
                if _base_path:
                    per_file_base_path[_out_key] = _base_path
                generated[_out_key] = new_text
                # patch 的 key 也按 out_file_name 同步（便于导出阶段按生成文件名取 patch）
                if patches.get(template_file_name):
                    patches[out_file_name or template_file_name] = patches.pop(template_file_name)

                # 8) 将生成结果入库：
                #    - save_as_case=True：category=project_case + case_id
                #    - save_as_case=False：category=project_doc（不带 case_id）
                #    用 split_documents 再分块写入 knowledge_docs，便于后续检索/审核 outline 提取。
                kb = KnowledgeBase(self.collection)

                doc_obj = Document(page_content=new_text, metadata={
                    "source_file": out_file_name or template_file_name,
                    "file_type": Path(template_file_name).suffix.lower() or "",
                })
                chunks = split_documents([doc_obj])
                if save_as_case and project_case_id:
                    kb.add_documents(
                        chunks,
                        file_name=out_file_name or template_file_name,
                        category="project_case",
                        case_id=project_case_id,
                    )
                else:
                    kb.add_documents(
                        chunks,
                        file_name=out_file_name or template_file_name,
                        category="project_doc",
                        case_id=None,
                    )

            _progress("完成", 1.0)
            return GeneratedCaseDocs(
                project_id=int(project_id),
                project_case_id=project_case_id,
                generated_files=generated,
                generated_patches=patches,
                per_file_skills_rules_applied=per_file_skills_rules_applied,
                per_file_base_path=per_file_base_path,
                draft_routing_plan=routing_plan,
            )

