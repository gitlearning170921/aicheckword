import hashlib
import json
import math
import random
import re
import threading
from difflib import SequenceMatcher
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

from config import settings
from src.core.agent import ReviewAgent
from src.core.llm_factory import invoke_chat_direct

from .models import EXAM_TRACKS, QUESTION_TYPES
from . import repository as repo


TRACK_HINTS = {
    "cn": "中国医疗器械法规、注册与质量体系要求",
    "iso13485": "ISO 13485 质量管理体系要求与实施",
    "mdsap": "MDSAP 审核程序与多国监管要求",
}


_OBJECTIVE_QUESTION_TYPES = frozenset(("single_choice", "multiple_choice", "true_false"))

# 选项文案前常见的「A. / B、」等前缀（落库前剥离，避免与前端自动编号重复）
_OPTION_LETTER_PREFIX_RE = re.compile(
    r"^\s*([A-H])(?:[\.\、:：\)\）\]\s]+)\s*",
    flags=re.IGNORECASE,
)


def _strip_option_letter_prefix(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    m = _OPTION_LETTER_PREFIX_RE.match(s)
    if m:
        rest = s[m.end() :].strip()
        if rest:
            return rest
    return s


def _single_choice_correct_index(answer: Any, opts_raw: List[str], n: int) -> int:
    if n <= 0:
        return 0
    a = answer
    if isinstance(a, list) and a:
        a = a[0]
    s = str(a).strip()
    if not s:
        return 0
    su = s[:1].upper()
    if len(su) == 1 and "A" <= su <= "Z":
        idx = ord(su) - ord("A")
        if 0 <= idx < n:
            return idx
    sc = _strip_option_letter_prefix(s).strip().lower()
    for i in range(n):
        if _strip_option_letter_prefix(str(opts_raw[i] or "").strip()).strip().lower() == sc:
            return i
    return 0


def _shuffle_objective_options_if_applicable(q: Dict[str, Any]) -> Dict[str, Any]:
    """单选/多选：打乱选项顺序并把答案改写为字母，避免模型总把正确项放在固定位置。"""
    qt = str(q.get("question_type") or "")
    if qt not in ("single_choice", "multiple_choice"):
        return q
    opts_raw = q.get("options") or []
    if not isinstance(opts_raw, list) or len(opts_raw) < 2:
        return q
    texts = [_strip_option_letter_prefix(str(x or "").strip()) for x in opts_raw]
    if any(not t for t in texts):
        return q
    n = len(texts)
    if qt == "single_choice":
        ci = _single_choice_correct_index(q.get("answer"), opts_raw, n)
        order = list(range(n))
        random.shuffle(order)
        new_texts = [texts[i] for i in order]
        new_ci = order.index(ci)
        out = dict(q)
        out["options"] = new_texts
        out["answer"] = chr(ord("A") + new_ci)
        return out
    ans = q.get("answer")
    if not isinstance(ans, list):
        ans = [ans] if ans is not None else []
    correct: set[int] = set()
    for a in ans:
        ci = _single_choice_correct_index(a, opts_raw, n)
        if 0 <= ci < n:
            correct.add(ci)
    if not correct:
        correct.add(_single_choice_correct_index(ans[0] if ans else "A", opts_raw, n))
    order = list(range(n))
    random.shuffle(order)
    new_texts = [texts[i] for i in order]
    new_letters = sorted({chr(ord("A") + order.index(i)) for i in correct})
    out = dict(q)
    out["options"] = new_texts
    out["answer"] = new_letters
    return out


# 单次提交只启动一次主观题异步阅卷线程
_submit_grade_lock = threading.Lock()
_submit_grade_inflight: set[int] = set()


def _norm_json_text(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    if s.startswith("json"):
        s = s[4:].strip()
    return s


def _hash_text(*parts: str) -> str:
    raw = "|".join((x or "").strip() for x in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _make_scope_hash(exam_track: str, category: str, difficulty: str, question_type: str) -> str:
    return _hash_text(exam_track, category, difficulty, question_type)[:32]


def _safe_question_type(v: str) -> str:
    x = (v or "").strip().lower()
    return x if x in QUESTION_TYPES else "single_choice"


def _safe_difficulty(v: str) -> str:
    x = (v or "").strip().lower()
    return x if x in ("easy", "medium", "hard") else "medium"


def _split_int_by_weights(n: int, weights: List[float]) -> List[int]:
    """将整数 n 按权重比例拆分为若干非负整数，且总和严格等于 n（最大余额法）。"""
    if n <= 0:
        return [0] * len(weights)
    if not weights:
        return []
    ws = [max(0.0, float(w)) for w in weights]
    s = sum(ws) or 1.0
    raw = [n * w / s for w in ws]
    floors = [int(x) for x in raw]
    rem = n - sum(floors)
    order = sorted(range(len(ws)), key=lambda i: raw[i] - floors[i], reverse=True)
    for k in range(rem):
        floors[order[k % len(order)]] += 1
    return floors


def _difficulty_question_type_plan(difficulty: str, question_count: int) -> List[tuple[str, int]]:
    """考试/练习套题：按难度确定客观题与主观题占比（与产品约定一致）。"""
    n = max(1, int(question_count))
    d = _safe_difficulty(difficulty)
    if d == "easy":
        types = ["single_choice", "true_false"]
        w = [0.5, 0.5]
    elif d == "medium":
        types = ["single_choice", "multiple_choice", "true_false"]
        w = [0.4, 0.3, 0.3]
    else:
        types = ["single_choice", "multiple_choice", "true_false", "case_analysis"]
        w = [0.3, 0.2, 0.3, 0.2]
    counts = _split_int_by_weights(n, w)
    return [(types[i], counts[i]) for i in range(len(types)) if counts[i] > 0]


def _ingest_knowledge_scope_plan(target_count: int) -> List[tuple[str, str, int]]:
    """AI 录题：知识来源占比 — 项目案例 30%、审核点 30%、法规标准 20%、程序文件 20%。"""
    n = max(1, int(target_count))
    keys = ["project_case", "audit_checkpoint", "regulation", "program"]
    labels = ["项目案例", "审核点", "法规标准", "程序文件"]
    weights = [0.3, 0.3, 0.2, 0.2]
    counts = _split_int_by_weights(n, weights)
    return [(keys[i], labels[i], counts[i]) for i in range(len(keys)) if counts[i] > 0]


def _ingest_question_type_plan(segment_count: int) -> List[tuple[str, int]]:
    """单段录题内题型占比：单选 30%、多选 10%、判断 10%、主观案例分析 50%。"""
    c = max(0, int(segment_count))
    if c <= 0:
        return []
    types = ["single_choice", "multiple_choice", "true_false", "case_analysis"]
    w = [0.3, 0.1, 0.1, 0.5]
    counts = _split_int_by_weights(c, w)
    return [(types[i], counts[i]) for i in range(len(types)) if counts[i] > 0]


def _rows_to_evidence(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    evidence: List[Dict[str, Any]] = []
    for r in rows:
        content = str(r.get("content") or "").strip()
        if not content:
            continue
        meta = r.get("metadata") or {}
        source_file = (
            str(meta.get("source_file") or "").strip()
            or str(meta.get("title") or "").strip()
            or str(r.get("source") or "").strip()
        )
        evidence.append({"content_snippet": content[:700], "source_file": source_file})
    return evidence


# 法规类未训练主库时：命题素材来源标签（与真实上传文件名区分）
_OPEN_REGULATION_EVIDENCE_SOURCE = "通用法规知识（大模型摘要·非用户向量库）"


def _regulation_open_evidence_via_llm(exam_track: str, need: int) -> List[Dict[str, Any]]:
    """本地 regulation 类向量不足时，用模型生成通用监管要点摘录；不冒充具体上传文件或条款号。"""
    need = max(1, min(int(need), 20))
    track_name = EXAM_TRACKS.get(exam_track, exam_track)
    track_hint = TRACK_HINTS.get(exam_track, "")
    prompt = f"""
你是医疗器械法规教研助手。用户本地向量库可能**未导入法规原文**，需要为考试命题准备若干条**一般性、可公开核对方向的表述**作为素材摘录（不等同于引用具体成文法条）。

体考类型：{track_name}（{track_hint}）

硬性要求：
1) 只输出 JSON，不要其它文字。
2) 格式：{{"snippets":[{{"content_snippet":"...","angle":"..."}}]}}
3) 输出 {need} 条 snippets；每条 content_snippet 180～420 字；angle 为该条角度标签（10～30 字）。
4) 内容围绕分类、临床评价、风险管理、质量管理体系、上市后监督、软件生命周期、网络安全与数据保护等**通用监管关注点**；**禁止**编造具体条款号、公告号、标准号、页码或「某具体文件名」。
5) 不要出现 tmp 临时文件名。

JSON:""".strip()
    try:
        prov = (settings.quiz_provider or settings.provider or "").strip().lower()
        model = (settings.quiz_llm_model or settings.llm_model or "").strip()
        temp = float(getattr(settings, "quiz_temperature", 0.2) or 0.2)
        txt = invoke_chat_direct(prompt, temperature=temp, provider=prov, model=model)
        data = json.loads(_norm_json_text(txt))
        arr = data.get("snippets") if isinstance(data, dict) else None
        if not isinstance(arr, list):
            arr = []
        out: List[Dict[str, Any]] = []
        for x in arr:
            if not isinstance(x, dict):
                continue
            sn = str(x.get("content_snippet") or "").strip()
            if not sn:
                continue
            out.append({"content_snippet": sn[:700], "source_file": _OPEN_REGULATION_EVIDENCE_SOURCE})
            if len(out) >= need:
                break
        if len(out) < need:
            filler = (
                f"[{track_name}] 医疗器械软件/独立软件注册与变更常见关注点包括：预期用途与适用范围一致性、"
                "风险管理闭环、验证与确认证据、说明书与标签符合性、网络安全与数据保护（如适用）。"
            )
            while len(out) < need:
                out.append({"content_snippet": filler[:420], "source_file": _OPEN_REGULATION_EVIDENCE_SOURCE})
        return out[:need]
    except Exception:
        fb = (
            f"{track_name}：关注产品风险分类、设计开发控制、软件生命周期与发布管理、"
            "与临床/使用场景相关的安全有效性证据组织方式。"
        )
        return [{"content_snippet": fb[:500], "source_file": _OPEN_REGULATION_EVIDENCE_SOURCE} for _ in range(need)]


def _extract_evidence_scoped(agent: ReviewAgent, exam_track: str, scope_key: str, top_k: int) -> List[Dict[str, Any]]:
    """按知识来源维度检索命题素材（项目案例 / 审核点 / 法规标准 / 程序文件）。

    法规标准：优先主库 category=regulation；若不足或未训练，则用大模型生成「通用法规要点」摘录补足（source_file
    固定为「通用法规知识（大模型摘要·非用户向量库）」，与真实上传文件区分）。
    """
    base_q = f"{EXAM_TRACKS.get(exam_track, exam_track)} {TRACK_HINTS.get(exam_track, '')} 典型考点 命题依据"
    tk = max(1, int(top_k))

    if scope_key == "regulation":
        docs: List[Any] = []
        try:
            docs = agent.kb.search_by_category(base_q + " 法规 标准 技术要求", "regulation", top_k=tk)
        except Exception:
            docs = []
        rows: List[Dict[str, Any]] = []
        for doc in docs:
            md = getattr(doc, "metadata", None) or {}
            rows.append(
                {
                    "content": getattr(doc, "page_content", "") or "",
                    "source": md.get("source_file") or "",
                    "metadata": md if isinstance(md, dict) else {},
                }
            )
        evidence = _rows_to_evidence(rows)
        if len(evidence) < tk:
            evidence.extend(_regulation_open_evidence_via_llm(exam_track, tk - len(evidence)))
        return evidence[:tk]

    docs: List[Any] = []
    try:
        if scope_key == "audit_checkpoint":
            docs = agent.checkpoint_kb.search(base_q + " 审核点 检查表 符合性", top_k=tk)
        elif scope_key == "project_case":
            docs = agent.kb.search_by_category(base_q + " 项目案例 注册资料", "project_case", top_k=tk)
        elif scope_key == "program":
            docs = agent.kb.search_by_category(base_q + " 程序文件 SOP 规程", "program", top_k=tk)
        else:
            docs = agent.kb.search(base_q, top_k=tk)
    except Exception:
        docs = []
    rows2: List[Dict[str, Any]] = []
    for doc in docs:
        md = getattr(doc, "metadata", None) or {}
        rows2.append(
            {
                "content": getattr(doc, "page_content", "") or "",
                "source": md.get("source_file") or "",
                "metadata": md if isinstance(md, dict) else {},
            }
        )
    return _rows_to_evidence(rows2)


def _ensure_question_shape(question: Dict[str, Any], fallback_category: str = "") -> Dict[str, Any]:
    q_type = _safe_question_type(str(question.get("question_type") or "single_choice"))
    opts = question.get("options") or []
    if not isinstance(opts, list):
        opts = []
    answer = question.get("answer")
    if q_type == "single_choice" and isinstance(answer, list) and answer:
        answer = answer[0]
    if q_type == "true_false":
        if isinstance(answer, str):
            answer = answer.strip().lower() in ("true", "1", "yes", "对", "正确")
        else:
            answer = bool(answer)
        opts = ["正确", "错误"]
    if q_type == "multiple_choice" and not isinstance(answer, list):
        answer = [answer] if answer is not None else []
    if q_type == "case_analysis":
        # 案例分析题：不应有选项；答案为参考要点文本（学生端文本作答）
        opts = []
        if isinstance(answer, list):
            answer = "\n".join([str(x).strip() for x in answer if str(x).strip()])[:1200] or ""
        if isinstance(answer, (dict, int, float, bool)):
            answer = str(answer)
        answer = str(answer or "").strip()
        if not answer or len(answer) <= 1 or answer.upper() in ("A", "B", "C", "D", "E", "F"):
            answer = "参考作答要点：结论 + 依据文件名 + 与摘录内容的对应关系。"
    if q_type != "case_analysis" and not opts:
        opts = ["A", "B", "C", "D"]
    if q_type == "multiple_choice":
        # 多选题：必须有多个正确项（>=2）。若不足，则兜底补足，避免生成“伪多选”。
        arr = answer if isinstance(answer, list) else []
        cleaned: List[str] = []
        for x in arr:
            s = str(x or "").strip().upper()
            if not s:
                continue
            if "," in s or "、" in s or "，" in s:
                parts = re.split(r"[，,、\s]+", s)
                for p in parts:
                    p2 = str(p or "").strip().upper()
                    if p2:
                        cleaned.append(p2)
            else:
                cleaned.append(s)
        uniq: List[str] = []
        for s in cleaned:
            if len(s) == 1 and "A" <= s <= "Z" and s not in uniq:
                uniq.append(s)
        max_opt = len(opts) if isinstance(opts, list) else 0
        if max_opt > 0:
            uniq = [x for x in uniq if (ord(x) - ord("A")) < max_opt]
        if len(uniq) < 2:
            cand = [chr(ord("A") + i) for i in range(max(2, min(6, max_opt or 4)))]
            for c in cand:
                if c not in uniq:
                    uniq.append(c)
                if len(uniq) >= 2:
                    break
        answer = uniq
    return {
        "question_type": q_type,
        "stem": str(question.get("stem") or "").strip(),
        "options": opts,
        "answer": answer,
        "explanation": str(question.get("explanation") or "").strip(),
        "category": str(question.get("category") or fallback_category or "").strip(),
        "difficulty": _safe_difficulty(str(question.get("difficulty") or "medium")),
        "evidence": question.get("evidence") if isinstance(question.get("evidence"), list) else [],
    }


def _extract_evidence(agent: ReviewAgent, exam_track: str, top_k: int = 8) -> List[Dict[str, Any]]:
    query = f"{EXAM_TRACKS.get(exam_track, exam_track)} {TRACK_HINTS.get(exam_track, '')} 典型考题要点"
    try:
        rows = agent.search_knowledge(query, top_k=top_k, use_checkpoints=True)
    except Exception:
        rows = []
    evidence = []
    for r in rows:
        content = str(r.get("content") or "").strip()
        if not content:
            continue
        meta = r.get("metadata") or {}
        # 统一给前端/LLM 的“可定位来源文件名”：优先 source_file，其次 title，再次 source
        source_file = (
            str(meta.get("source_file") or "").strip()
            or str(meta.get("title") or "").strip()
            or str(r.get("source") or "").strip()
        )
        evidence.append(
            {
                "content_snippet": content[:700],
                "source_file": source_file,
            }
        )
    return evidence


def _fallback_questions(
    exam_track: str,
    category: str,
    count: int,
    evidence: List[Dict[str, Any]],
    *,
    question_type: str = "true_false",
    difficulty: str = "medium",
) -> List[Dict[str, Any]]:
    qt = _safe_question_type(question_type)
    diff = _safe_difficulty(difficulty)
    out: List[Dict[str, Any]] = []
    for i in range(count):
        ev = evidence[i % len(evidence)] if evidence else {"content_snippet": "知识库命中不足", "source_file": ""}
        src = str(ev.get("source_file") or "").strip()
        src_text = f"《{src}》" if src else "（来源文件未标注）"
        snip = str(ev.get("content_snippet", "") or "")[:160]
        if qt == "single_choice":
            stem = (
                f"[{EXAM_TRACKS.get(exam_track, exam_track)}] 依据 {src_text}，下列哪项最恰当？\n{snip}"
            )
            opt_sets = (
                (
                    "与摘录一致，可作为当前结论的支持性表述",
                    "与摘录部分一致，但不足以单独支撑结论",
                    "与摘录存在冲突，需要回到原始资料核对",
                    "摘录信息不足，无法判断与结论的关系",
                ),
                (
                    "摘录支持将风险控制措施限定在所述范围内",
                    "摘录支持扩大适用范围至未提及的产品类别",
                    "摘录仅涉及质量管理体系，与产品安全无直接关联",
                    "摘录与题干结论属于不同监管环节，不宜直接引用",
                ),
                (
                    "在现有控制下残余风险可接受，且与摘录表述一致",
                    "残余风险需追加控制措施，摘录未给出充分依据",
                    "摘录未讨论残余风险，结论应另找证据支持",
                    "摘录与风险结论同向，但缺少量化或验证信息",
                ),
            )
            oi = i % len(opt_sets)
            out.append(
                {
                    "question_type": "single_choice",
                    "stem": stem,
                    "options": list(opt_sets[oi]),
                    "answer": "A",
                    "explanation": f"依据：{src_text}；摘录：{snip}",
                    "category": category or exam_track,
                    "difficulty": diff,
                    "evidence": [ev],
                }
            )
        elif qt == "multiple_choice":
            stem = (
                f"[{EXAM_TRACKS.get(exam_track, exam_track)}] 依据 {src_text}，下列哪些说法成立（多选）？\n{snip}"
            )
            opt_texts = [
                "摘录可作为题干结论的前提依据之一",
                "摘录与题干结论在监管要求层面相容",
                "摘录与题干结论无关且不宜作为依据",
                "摘录否定题干结论，应仅以摘录推翻题干而不复核上下文",
            ]
            out.append(
                {
                    "question_type": "multiple_choice",
                    "stem": stem,
                    "options": opt_texts,
                    "answer": ["A", "B"],
                    "explanation": f"依据：{src_text}；摘录：{snip}",
                    "category": category or exam_track,
                    "difficulty": diff,
                    "evidence": [ev],
                }
            )
        elif qt == "case_analysis":
            stem = (
                f"[{EXAM_TRACKS.get(exam_track, exam_track)}] 案例分析：结合 {src_text} 的摘录，说明应关注的合规要点与理由。\n{snip}"
            )
            out.append(
                {
                    "question_type": "case_analysis",
                    "stem": stem,
                    "options": [],
                    "answer": "请作答：结论 + 依据文件名 + 与摘录的对应关系。",
                    "explanation": f"依据：{src_text}；摘录：{snip}",
                    "category": category or exam_track,
                    "difficulty": diff,
                    "evidence": [ev],
                }
            )
        else:
            stem = (
                f"[{EXAM_TRACKS.get(exam_track, exam_track)}] "
                f"根据 {src_text} 的内容判断下述说法是否正确：{snip[:120]}"
            )
            out.append(
                {
                    "question_type": "true_false",
                    "stem": stem,
                    "options": ["正确", "错误"],
                    "answer": True,
                    "explanation": f"依据：{src_text}；摘录：{snip[:180]}",
                    "category": category or exam_track,
                    "difficulty": diff,
                    "evidence": [ev],
                }
            )
    return out


def _generate_questions_by_ai(
    *,
    exam_track: str,
    category: str,
    difficulty: str,
    question_type: str,
    count: int,
    evidence: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    min_plausible = 0 if count <= 1 else max(1, (count * 4 + 9) // 10)
    prompt = f"""
你是医疗器械法规考试命题助手。请只输出 JSON，不要额外说明。

要求：
1) 生成 {count} 道题，体考类型：{EXAM_TRACKS.get(exam_track, exam_track)}（{TRACK_HINTS.get(exam_track, '')}）
2) 题型：{question_type}
3) 难度：{difficulty}
4) 分类：{category or exam_track}
5) 必须依据 evidence，不得编造法规条款编号/章节号/文件名。
6) 每题 explanation 必须明确写出“依据的来源文件名”，格式示例：依据：《文件名》；……。禁止出现“根据审核点XXX”这类不可定位表述。
7) evidence[].source_file 必须填写为具体可读的文件名（程序文件/法规文件/项目案例文件名）；如果无法确定，填空字符串，并在 explanation 中写“来源文件未标注”。不得使用 tmp 临时文件名。
8) JSON 格式：{{"questions":[{{"question_type":"single_choice|multiple_choice|true_false|case_analysis","stem":"...","options":[...],"answer":...,"explanation":"...","category":"...","difficulty":"easy|medium|hard","evidence":[{{"content_snippet":"...","source_file":"..."}}]}}]}}
9) 单选/多选：`options` 为**纯陈述文本数组**（2～6 项），**不要**在每项前加 `A./B.` 等字母前缀（系统会统一编号）。单选 `answer` 为单个大写字母（如 `C`）；多选 `answer` 为字母数组（如 `["A","D"]`）。
9.1) 多选题（multiple_choice）必须至少有 **2 个**正确选项（`answer` 数组长度 ≥ 2），不得只给一个正确项。
9.2) 案例分析题（case_analysis）必须满足：`options` 为空数组；`answer` 为**参考作答要点文本**（不是 A/B/C/D），学生端会用文本框输入。
10) 干扰项质量（本批共 {count} 题）：其中至少 **{min_plausible}** 题须标为 `medium` 或 `hard`，且这些题的**错误选项**须与正确选项在句式长度、术语层级上**尽量平行**，呈现**易混淆结论**；**禁止**整批题都靠「仅…」「只需要…」「不需要…」「绝不是…」「与审核无关」等一眼可排除的标语式否定句凑满四个选项。
11) 约 **60%** 题目可为 `easy`：允许轻度否定或排除语气，但**不要**让四个错误项共用同一种开头模板；**不得**出现「明显三项都错、只剩一项像真命题」的凑数结构。
12) 正确答案在命制时不要刻意总落在同一字母位；落库前系统会打乱选项顺序并重写 `answer`，你仍须保证**每个错误选项本身像合理结论**而非「反着说就对了」的口号。
13) 同批各题题干须围绕不同角度设问，避免只改一两处用词、结构高度雷同的「换皮题」；系统还会与历史题库做相似度过滤，雷同过多会被丢弃。

evidence:
{json.dumps(evidence, ensure_ascii=False)}
""".strip()
    prov = (settings.quiz_provider or settings.provider or "").strip().lower()
    model = (settings.quiz_llm_model or settings.llm_model or "").strip()
    temp = float(getattr(settings, "quiz_temperature", 0.2) or 0.2)
    txt = invoke_chat_direct(prompt, temperature=temp, provider=prov, model=model)
    data = json.loads(_norm_json_text(txt))
    arr = data.get("questions") if isinstance(data, dict) else []
    if not isinstance(arr, list):
        return []
    return [_ensure_question_shape(x, fallback_category=category or exam_track) for x in arr][:count]


def _normalize_stem_for_dedupe(stem: str) -> str:
    s = re.sub(r"\s+", "", (stem or "").strip().lower())
    s = re.sub(r"^(\[[^\]]+\])+", "", s)
    return s[:900]


def _stem_similar(stem_a: str, stem_b: str, threshold: float = 0.82) -> bool:
    na = _normalize_stem_for_dedupe(stem_a)
    nb = _normalize_stem_for_dedupe(stem_b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    return SequenceMatcher(None, na, nb).ratio() >= threshold


def _set_diversity_signature(q: Dict[str, Any]) -> str:
    qt = str(q.get("question_type") or "").strip().lower()
    kh = str(q.get("knowledge_scope_hash") or "").strip()[:120]
    cat = str(q.get("category") or "").strip()[:96]
    return f"{qt}|{kh}|{cat}"


def _dedupe_questions_for_ingest(
    questions: List[Dict[str, Any]],
    *,
    prior_stems: List[str],
    max_similar_frac: float = 0.2,
) -> List[Dict[str, Any]]:
    """AI 录题：与历史题干近似重复的题控制在约 max_similar_frac（默认 20%）以内。"""
    if not questions:
        return []
    pool = list(questions)
    random.shuffle(pool)
    quota = max(0, int(math.ceil(len(pool) * max_similar_frac)))
    sim_used = 0
    acc: List[Dict[str, Any]] = []
    stems_in_acc: List[str] = []
    for q in pool:
        st = str(q.get("stem") or "")
        hit_prior = any(_stem_similar(st, ps) for ps in prior_stems if (ps or "").strip())
        hit_acc = any(_stem_similar(st, s) for s in stems_in_acc)
        if not hit_prior and not hit_acc:
            acc.append(q)
            stems_in_acc.append(st)
            continue
        if sim_used < quota:
            acc.append(q)
            stems_in_acc.append(st)
            sim_used += 1
    return acc


def _pick_cached_questions(
    *,
    collection: str,
    exam_track: str,
    category: Optional[str],
    difficulty: str,
    question_type: str,
    scope_hash: str,
    count: int,
    exclude_question_ids: Optional[List[int]] = None,
    sig_counts: Dict[str, int],
    question_count_total: int,
) -> List[Dict[str, Any]]:
    cat_f = (category or "").strip() or None
    excl = list(exclude_question_ids or [])
    ex_set = {int(x) for x in excl if int(x) > 0}
    if count <= 0:
        return []
    max_rep = max(1, (max(1, int(question_count_total)) + 9) // 10)

    def _narrow_pool() -> List[Dict[str, Any]]:
        return repo.list_bank_questions(
            collection=collection,
            exam_track=exam_track,
            knowledge_scope_hash=scope_hash,
            category=cat_f,
            difficulty=difficulty,
            question_type=question_type,
            limit=max(count * 5, 24),
            exclude_question_ids=sorted(ex_set),
        )

    pool = repo.list_bank_questions(
        collection=collection,
        exam_track=exam_track,
        knowledge_scope_hash=None,
        category=cat_f,
        difficulty=difficulty,
        question_type=question_type,
        limit=min(220, max(count * 14, 48)),
        exclude_question_ids=sorted(ex_set),
    )
    random.shuffle(pool)
    out: List[Dict[str, Any]] = []

    for cand in pool:
        if len(out) >= count:
            break
        qid = int(cand.get("id") or 0)
        if not qid or qid in ex_set:
            continue
        sig = _set_diversity_signature(cand)
        if sig_counts.get(sig, 0) >= max_rep:
            continue
        out.append(cand)
        ex_set.add(qid)
        sig_counts[sig] = sig_counts.get(sig, 0) + 1

    if len(out) < count:
        for cand in pool:
            if len(out) >= count:
                break
            qid = int(cand.get("id") or 0)
            if not qid or qid in ex_set:
                continue
            out.append(cand)
            ex_set.add(qid)
            sig = _set_diversity_signature(cand)
            sig_counts[sig] = sig_counts.get(sig, 0) + 1

    if len(out) < count:
        for cand in _narrow_pool():
            if len(out) >= count:
                break
            qid = int(cand.get("id") or 0)
            if not qid or qid in ex_set:
                continue
            sig = _set_diversity_signature(cand)
            if sig_counts.get(sig, 0) >= max_rep:
                continue
            out.append(cand)
            ex_set.add(qid)
            sig_counts[sig] = sig_counts.get(sig, 0) + 1

    if len(out) < count:
        for cand in _narrow_pool():
            if len(out) >= count:
                break
            qid = int(cand.get("id") or 0)
            if not qid or qid in ex_set:
                continue
            out.append(cand)
            ex_set.add(qid)
            sig = _set_diversity_signature(cand)
            sig_counts[sig] = sig_counts.get(sig, 0) + 1

    return out[:count]


def _save_questions_to_bank(
    *,
    collection: str,
    exam_track: str,
    category: str,
    difficulty: str,
    question_type: str,
    scope_hash: str,
    questions: List[Dict[str, Any]],
    origin: str,
    created_by: str,
) -> List[Dict[str, Any]]:
    out = []
    for q in questions:
        q = _ensure_question_shape(q, fallback_category=category)
        q = _shuffle_objective_options_if_applicable(q)
        q_hash = _hash_text(q["stem"], json.dumps(q["options"], ensure_ascii=False), json.dumps(q["answer"], ensure_ascii=False))
        qid = repo.create_question(
            collection=collection,
            exam_track=exam_track,
            question_hash=q_hash,
            question_type=q["question_type"],
            difficulty=q["difficulty"],
            category=q.get("category") or category,
            knowledge_scope_hash=scope_hash,
            stem=q["stem"],
            options=q["options"],
            answer=q["answer"],
            explanation=q.get("explanation") or "",
            evidence=q.get("evidence") or [],
            origin=origin,
            created_by=created_by,
        )
        repo.upsert_question_bank(
            collection=collection,
            exam_track=exam_track,
            category=q.get("category") or category,
            question_type=q["question_type"],
            difficulty=q["difficulty"],
            knowledge_scope_hash=scope_hash,
            question_id=qid,
            quality_score=70.0 if q.get("evidence") else 50.0,
        )
        q["id"] = qid
        out.append(q)
    return out


def _letter_choice_index(raw: Any) -> Optional[int]:
    if raw is None or isinstance(raw, (dict, list)):
        return None
    s = str(raw).strip().upper()
    if len(s) != 1 or s < "A" or s > "Z":
        return None
    return ord(s) - ord("A")


def _resolve_choice_letter_to_option_value(value: Any, options: Optional[List[Any]]) -> Any:
    """单选/多选/判断：若作答为 A–Z 且题干有 options，则解析为对应选项原文再比较。"""
    if not isinstance(options, list) or not options:
        return value
    ix = _letter_choice_index(value)
    if ix is None or ix >= len(options):
        return value
    return options[ix]


def _norm_single_choice_cmp_key(value: Any, options: Optional[List[Any]]) -> str:
    v = _resolve_choice_letter_to_option_value(value, options)
    if v is None:
        return ""
    return str(v).strip().lower()


def _norm_multiple_choice_set(raw: Any, options: Optional[List[Any]]) -> set[str]:
    if raw is None:
        return set()
    items = raw if isinstance(raw, list) else [raw]
    out: set[str] = set()
    for x in items:
        v = _resolve_choice_letter_to_option_value(x, options)
        out.add(str(v).strip().lower() if v is not None else "")
    return {x for x in out if x}


def _true_false_to_bool(v: Any) -> bool:
    """判断题：bool / 数字 / 常见中英文真假字面量 → bool（禁止对任意字符串用 Python bool()）。"""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v) != 0.0
    if isinstance(v, str):
        t = v.strip().lower()
        if t in ("false", "0", "no", "n", "f", "wrong", "错误", "错", "否", "不正确", "不对"):
            return False
        if t in ("true", "1", "yes", "y", "t", "对", "正确", "是", "√"):
            return True
        return False
    return False


def _score_objective_answer(
    question_type: str,
    answer: Any,
    user_answer: Any,
    options: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    opts = options if isinstance(options, list) else None

    if question_type == "single_choice":
        ok = _norm_single_choice_cmp_key(answer, opts) == _norm_single_choice_cmp_key(user_answer, opts)
        return {"is_correct": ok, "score": 1.0 if ok else 0.0}
    if question_type == "true_false":
        aa = _resolve_choice_letter_to_option_value(answer, opts)
        ua = _resolve_choice_letter_to_option_value(user_answer, opts)
        ok = _true_false_to_bool(aa) == _true_false_to_bool(ua)
        return {"is_correct": ok, "score": 1.0 if ok else 0.0}
    if question_type == "multiple_choice":
        as_set = _norm_multiple_choice_set(answer, opts)
        us_set = _norm_multiple_choice_set(user_answer, opts)
        ok = as_set == us_set and len(as_set) > 0
        return {"is_correct": ok, "score": 1.0 if ok else 0.0}
    return {"is_correct": False, "score": 0.0}


def generate_set(
    *,
    collection: str,
    exam_track: str,
    set_type: str,
    created_by: str,
    title: str = "",
    category: str = "",
    difficulty: str = "medium",
    question_type: str = "single_choice",
    question_count: int = 20,
    status: str = "draft",
) -> Dict[str, Any]:
    """组卷：题型占比仅由 difficulty 决定；question_type 参数保留兼容，不参与组卷。"""
    if exam_track not in EXAM_TRACKS:
        raise ValueError(f"不支持的 exam_track: {exam_track}")
    question_count = max(1, int(question_count))
    difficulty = _safe_difficulty(difficulty)
    bank_cat = (category or "").strip() or None
    hash_cat = bank_cat or exam_track
    plan = _difficulty_question_type_plan(difficulty, question_count)
    mix_map = {qt: n for qt, n in plan}
    set_cfg = {
        "set_config_hash": _hash_text(exam_track, hash_cat or "", difficulty, json.dumps(mix_map, sort_keys=True), str(question_count))[:32],
        "question_type_mix": mix_map,
        "question_type": "mixed",
        "difficulty": difficulty,
        "question_count": question_count,
        "category": category or exam_track,
        "bank_category_filter": bank_cat,
    }
    selected_all: List[Dict[str, Any]] = []
    seen_ids: set[int] = set()
    sig_counts: Dict[str, int] = {}
    max_sig_rep = max(1, (question_count + 9) // 10)
    from_cache_total = 0
    generated_total = 0
    agent = ReviewAgent(collection)
    practice_wrong: Dict[str, deque] = {}
    practice_unpr: Dict[str, deque] = {}
    uid_pr = ""
    if set_type == "practice":
        uid_pr = (created_by or "").strip()
        if uid_pr:
            for row in repo.list_wrong_questions_for_student(collection=collection, user_id=uid_pr, limit=260):
                tr = str(row.get("exam_track") or "").strip()
                if tr != exam_track:
                    continue
                qt0 = _safe_question_type(str(row.get("question_type") or "single_choice"))
                try:
                    qid0 = int(row.get("question_id") or 0)
                except (TypeError, ValueError):
                    qid0 = 0
                if qid0 <= 0:
                    continue
                practice_wrong.setdefault(qt0, deque()).append(qid0)
            for row in repo.list_unpracticed_questions_for_student(
                collection=collection, user_id=uid_pr, exam_track=exam_track, limit=520
            ):
                qt0 = _safe_question_type(str(row.get("question_type") or "single_choice"))
                try:
                    qid0 = int(row.get("question_id") or 0)
                except (TypeError, ValueError):
                    qid0 = 0
                if qid0 <= 0:
                    continue
                practice_unpr.setdefault(qt0, deque()).append(qid0)
    for qtype, cnt in plan:
        if cnt <= 0:
            continue
        qtype = _safe_question_type(qtype)
        scope_hash = _make_scope_hash(exam_track, hash_cat, difficulty, qtype)
        selected: List[Dict[str, Any]] = []
        if uid_pr:
            prioritized_ids: List[int] = []
            wq = practice_wrong.get(qtype)
            uq = practice_unpr.get(qtype)
            if wq:
                while len(prioritized_ids) < cnt and wq:
                    qid_x = wq.popleft()
                    if qid_x in seen_ids:
                        continue
                    prioritized_ids.append(qid_x)
            if uq:
                while len(prioritized_ids) < cnt and uq:
                    qid_x = uq.popleft()
                    if qid_x in seen_ids:
                        continue
                    prioritized_ids.append(qid_x)
            if prioritized_ids:
                loaded = repo.list_questions_by_ids(collection=collection, question_ids=prioritized_ids)
                by_lid = {int(x["id"]): x for x in loaded if x.get("id") is not None}
                for qid_y in prioritized_ids:
                    if len(selected) >= cnt:
                        break
                    obj = by_lid.get(int(qid_y))
                    if obj:
                        sig0 = _set_diversity_signature(obj)
                        if sig_counts.get(sig0, 0) >= max_sig_rep:
                            continue
                        selected.append(obj)
                        sig_counts[sig0] = sig_counts.get(sig0, 0) + 1
                for x in selected:
                    if x.get("id"):
                        seen_ids.add(int(x["id"]))
        short_pick = cnt - len(selected)
        cached = _pick_cached_questions(
            collection=collection,
            exam_track=exam_track,
            category=bank_cat,
            difficulty=difficulty,
            question_type=qtype,
            scope_hash=scope_hash,
            count=short_pick,
            exclude_question_ids=sorted(seen_ids),
            sig_counts=sig_counts,
            question_count_total=question_count,
        )
        selected.extend(cached)
        from_cache_total += len(cached)
        for x in selected:
            if x.get("id"):
                seen_ids.add(int(x["id"]))
        short = cnt - len(selected)
        if short > 0:
            evidence = _extract_evidence(agent, exam_track, top_k=max(8, short))
            stem_cat = (category or "").strip() or exam_track
            try:
                generated = _generate_questions_by_ai(
                    exam_track=exam_track,
                    category=stem_cat,
                    difficulty=difficulty,
                    question_type=qtype,
                    count=short,
                    evidence=evidence,
                )
            except Exception:
                generated = _fallback_questions(
                    exam_track,
                    stem_cat,
                    short,
                    evidence,
                    question_type=qtype,
                    difficulty=difficulty,
                )
            saved = _save_questions_to_bank(
                collection=collection,
                exam_track=exam_track,
                category=stem_cat,
                difficulty=difficulty,
                question_type=qtype,
                scope_hash=scope_hash,
                questions=generated,
                origin=("exam_teacher_generated" if set_type == "exam" else "practice_runtime_generated"),
                created_by=created_by,
            )
            generated_total += len(saved)
            for x in saved:
                if isinstance(x, dict):
                    sg = _set_diversity_signature(x)
                    sig_counts[sg] = sig_counts.get(sg, 0) + 1
            selected.extend(saved)
            for x in saved:
                if x.get("id"):
                    seen_ids.add(int(x["id"]))
        selected_all.extend(selected[:cnt])
    random.shuffle(selected_all)
    selected_all = selected_all[:question_count]
    repo.touch_bank_questions([int(x.get("id")) for x in selected_all if x.get("id")])
    set_id = repo.create_set(
        collection=collection,
        set_type=set_type,
        exam_track=exam_track,
        title=title or f"{EXAM_TRACKS.get(exam_track, exam_track)}-{set_type}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        set_config=set_cfg,
        status=status,
        created_by=created_by,
        items=[(int(x["id"]), 1.0) for x in selected_all if x.get("id")],
    )
    out = repo.load_set(set_id) or {"id": set_id, "items": []}
    out["from_cache_count"] = from_cache_total
    out["generated_count"] = generated_total
    return out


def ingest_bank_by_ai(
    *,
    collection: str,
    exam_track: str,
    target_count: int,
    created_by: str,
    review_mode: str = "auto_apply",
    category: str = "",
    difficulty: str = "medium",
    question_type: str = "single_choice",
    set_title: str = "",
) -> Dict[str, Any]:
    target_count = max(1, int(target_count))
    job_id = repo.create_ingest_job(
        collection=collection,
        exam_track=exam_track,
        target_count=target_count,
        review_mode=review_mode,
        created_by=created_by,
    )
    title = (set_title or "").strip() or (
        f"{EXAM_TRACKS.get(exam_track, exam_track)}-AI录题-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    )
    difficulty = _safe_difficulty(difficulty)
    try:
        draft_set_id = repo.create_set(
            collection=collection,
            set_type="bank_ingest",
            exam_track=exam_track,
            title=title,
            set_config={
                "source": "bank_ingest_by_ai",
                "ingest_job_id": job_id,
                "ingest_knowledge_mix": "project_case30_audit30_regulation20_program20",
                "ingest_type_mix": "single30_multi10_tf10_case50",
                "difficulty": difficulty,
            },
            status="draft",
            created_by=created_by,
            items=[],
        )
    except Exception as e:
        repo.update_ingest_job(job_id, generated_count=0, status="failed", message=f"create_set: {e}")
        raise
    repo.update_ingest_job(
        job_id,
        generated_count=0,
        status="running",
        message="starting",
        set_id=draft_set_id,
    )

    def _run():
        try:
            agent = ReviewAgent(collection)
            hash_cat = exam_track
            all_saved: List[Dict[str, Any]] = []
            bank_stems_for_dedupe = repo.list_recent_question_stems(
                collection=collection, exam_track=exam_track, limit=260
            )
            for scope_key, cat_label, seg_n in _ingest_knowledge_scope_plan(target_count):
                if seg_n <= 0:
                    continue
                for qt, qc in _ingest_question_type_plan(seg_n):
                    if qc <= 0:
                        continue
                    qt = _safe_question_type(qt)
                    scope_hash = _make_scope_hash(exam_track, hash_cat, difficulty, qt)
                    evidence = _extract_evidence_scoped(agent, exam_track, scope_key, top_k=max(8, qc))
                    gen_n = qc + max(2, (qc + 3) // 4)
                    try:
                        generated = _generate_questions_by_ai(
                            exam_track=exam_track,
                            category=cat_label,
                            difficulty=difficulty,
                            question_type=qt,
                            count=gen_n,
                            evidence=evidence,
                        )
                    except Exception:
                        generated = _fallback_questions(
                            exam_track,
                            cat_label,
                            gen_n,
                            evidence,
                            question_type=qt,
                            difficulty=difficulty,
                        )
                    prior_stems = [str(x.get("stem") or "") for x in all_saved]
                    prior_stems.extend(bank_stems_for_dedupe)
                    generated = _dedupe_questions_for_ingest(
                        generated, prior_stems=prior_stems, max_similar_frac=0.2
                    )
                    generated = generated[:qc]
                    if len(generated) < qc:
                        need = qc - len(generated)
                        fb = _fallback_questions(
                            exam_track,
                            cat_label,
                            need,
                            evidence,
                            question_type=qt,
                            difficulty=difficulty,
                        )
                        more_prior = prior_stems + [str(x.get("stem") or "") for x in generated]
                        fb = _dedupe_questions_for_ingest(fb, prior_stems=more_prior, max_similar_frac=0.2)
                        generated.extend(fb[:need])
                        generated = generated[:qc]
                    saved = _save_questions_to_bank(
                        collection=collection,
                        exam_track=exam_track,
                        category=cat_label,
                        difficulty=difficulty,
                        question_type=qt,
                        scope_hash=scope_hash,
                        questions=generated,
                        origin="teacher_bulk_ingest",
                        created_by=created_by,
                    )
                    all_saved.extend(saved)
            if not all_saved:
                raise ValueError("no questions saved")
            repo.add_set_items(
                draft_set_id,
                [(int(x["id"]), 1.0) for x in all_saved if x.get("id")],
                replace=True,
            )
            repo.update_ingest_job(
                job_id,
                generated_count=len(all_saved),
                status="done",
                message="ok",
                set_id=draft_set_id,
            )
        except Exception as e:
            repo.update_ingest_job(
                job_id,
                generated_count=0,
                status="failed",
                message=str(e),
                set_id=draft_set_id,
            )

    th = threading.Thread(target=_run, name=f"quiz_ingest_{job_id}", daemon=True)
    th.start()
    return {"job_id": job_id, "set_id": draft_set_id, "status": "running"}


def get_ingest_job(job_id: int) -> Dict[str, Any]:
    row = repo.get_ingest_job(job_id)
    if not row:
        raise ValueError("job not found")
    d = dict(row)
    rid = d.get("id")
    if rid is not None:
        d.setdefault("job_id", int(rid))
    sid = d.get("set_id")
    if sid is not None:
        d["set_id"] = int(sid)
    return d


def set_ingest_job_set_id(job_id: int, set_id: int) -> Dict[str, Any]:
    row = repo.get_ingest_job(job_id)
    if not row:
        raise ValueError("job not found")
    set_id = int(set_id)
    if set_id < 1:
        raise ValueError("set_id invalid")
    # 保留原有状态与计数，仅回写 set_id
    repo.update_ingest_job(
        int(job_id),
        generated_count=int(row.get("generated_count") or 0),
        status=str(row.get("status") or "unknown"),
        message=str(row.get("message") or ""),
        set_id=set_id,
    )
    return {"job_id": int(job_id), "set_id": set_id, "ok": True}


def start_attempt(*, collection: str, set_id: int, user_id: str, mode: str) -> Dict[str, Any]:
    aid = repo.create_attempt(collection=collection, set_id=set_id, user_id=user_id, mode=mode)
    return {"attempt_id": aid, "set_id": set_id, "mode": mode}


def _row_needs_async_subjective_llm(
    *, collection: str, attempt_id: int, paper_id: Optional[int], r: Dict[str, Any]
) -> bool:
    """无 cache 规则且非客观题时改走 LLM → 提交阶段异步处理。"""
    del attempt_id  # 预留与 attempt 级策略
    qid = int(r["question_id"])
    rule = repo.get_grading_rule(collection=collection, paper_id=paper_id, question_id=qid, version=1)
    if rule:
        return False
    qt = str(r.get("question_type") or "").strip()
    return qt not in _OBJECTIVE_QUESTION_TYPES


def _float01(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _percent_score(correct: int, total: int) -> float:
    if total <= 0:
        return 0.0
    v = (float(correct) / float(total)) * 100.0
    if v < 0:
        return 0.0
    if v > 100.0:
        return 100.0
    return v


def summarize_attempt_metrics_from_db(*, attempt_id: int) -> Dict[str, Any]:
    """从 quiz_answers 聚合分数/对错数（阅卷全部完成后调用）。

    约定：对外分数使用 0~100 百分制，保证与前端「>=80 通过」一致。
    """
    rows = repo.list_attempt_answers_with_questions(int(attempt_id))
    corr = wrong = 0
    for r in rows:
        ic = r.get("is_correct")
        if ic is True or ic == 1:
            corr += 1
        elif ic is False or ic == 0:
            wrong += 1
    total_q = len(rows)
    score = _percent_score(corr, total_q)
    return {
        "score": score,
        "total_score": 100.0 if total_q > 0 else 0.0,
        "graded_count": total_q,
        "correct_count": corr,
        "wrong_count": wrong,
    }


def finalize_attempt_aggregate_and_grade_complete(*, attempt_id: int) -> Dict[str, Any]:
    """主观/客观已全部写入各行后汇总并置 state=graded。"""
    metrics = summarize_attempt_metrics_from_db(attempt_id=int(attempt_id))
    payload = dict(metrics)
    payload["grading_status"] = "complete"
    repo.finalize_attempt(int(attempt_id), payload, state="graded")
    return {
        **metrics,
        "attempt_id": int(attempt_id),
        "mode": "auto",
        "grading_status": "complete",
    }


def _grade_single_subjective_with_llm(
    *, collection: str, attempt_id: int, paper_id: Optional[int], r: Dict[str, Any]
) -> None:
    qid = int(r["question_id"])
    prompt = f"""
请按 0~1 评分并返回 JSON: {{"score":0.0,"comment":"..."}}。
题干: {r.get('stem') or ''}
标准答案要点: {json.dumps(r.get('answer'), ensure_ascii=False)}
学生答案: {json.dumps(r.get('user_answer'), ensure_ascii=False)}
""".strip()
    try:
        prov = (settings.quiz_provider or settings.provider or "").strip().lower()
        model = (settings.quiz_llm_model or settings.llm_model or "").strip()
        temp = float(getattr(settings, "quiz_temperature", 0.2) or 0.2)
        raw = invoke_chat_direct(prompt, temperature=temp, provider=prov, model=model)
        data = json.loads(_norm_json_text(raw))
        score = max(0.0, min(1.0, float(data.get("score") or 0.0)))
        comment = str(data.get("comment") or "ai_score")
    except Exception:
        score = 0.0
        comment = "ai_score_failed"
    ev = {"is_correct": score >= 0.6}
    repo.upsert_grading_rule(
        collection=collection,
        paper_id=paper_id,
        question_id=qid,
        version=1,
        answer_key=r.get("answer"),
        rubric={"source": "auto_async"},
        updated_by="system",
    )
    repo.update_answer_grade(
        answer_id=int(r["answer_id"]),
        auto_score=float(score),
        final_score=float(score),
        is_correct=bool(ev["is_correct"]),
        teacher_comment=comment,
        graded_by_cache=False,
    )


def run_async_subjective_grading(*, collection: str, attempt_id: int, paper_id: Optional[int]) -> None:
    """线程入口：对已入库的主观题逐一 LLM 判分，最后再汇总总分。"""
    try:
        rows = repo.list_attempt_answers_with_questions(int(attempt_id))
        subj = []
        for r in rows:
            if _row_needs_async_subjective_llm(collection=collection, attempt_id=int(attempt_id), paper_id=paper_id, r=r):
                subj.append(r)
        for r in subj:
            try:
                _grade_single_subjective_with_llm(
                    collection=collection, attempt_id=int(attempt_id), paper_id=paper_id, r=dict(r)
                )
            except Exception:
                repo.update_answer_grade(
                    answer_id=int(r["answer_id"]),
                    auto_score=0.0,
                    final_score=0.0,
                    is_correct=False,
                    teacher_comment="ai_score_failed",
                    graded_by_cache=False,
                )
        finalize_attempt_aggregate_and_grade_complete(attempt_id=int(attempt_id))
    except Exception as e:
        try:
            metrics = summarize_attempt_metrics_from_db(attempt_id=int(attempt_id))
            payload = dict(metrics)
            payload["grading_status"] = "failed"
            payload["error"] = str(e)[:800]
            payload["message"] = "主观题异步阅卷异常"
            repo.finalize_attempt(int(attempt_id), payload, state="graded")
        except Exception:
            repo.finalize_attempt(
                int(attempt_id),
                {"grading_status": "failed", "error": str(e)[:800], "message": "主观题异步阅卷异常"},
                state="graded",
            )


def _enqueue_async_subjective_if_needed(*, attempt_id: int, collection: str, paper_id: Optional[int]) -> None:
    aid = int(attempt_id)
    with _submit_grade_lock:
        if aid in _submit_grade_inflight:
            return
        _submit_grade_inflight.add(aid)

    def _runner():
        try:
            run_async_subjective_grading(collection=collection, attempt_id=aid, paper_id=paper_id)
        finally:
            with _submit_grade_lock:
                _submit_grade_inflight.discard(aid)

    th = threading.Thread(target=_runner, name=f"quiz_subjective_grade_{aid}", daemon=True)
    th.start()


def get_attempt_grading_status(*, attempt_id: int) -> Dict[str, Any]:
    row = repo.get_attempt_by_id(int(attempt_id))
    if not row:
        raise ValueError(f"attempt not found: {attempt_id}")
    st = str(row.get("state") or "").strip()
    sj = {}
    raw_s = row.get("score_json")
    try:
        sj = json.loads(raw_s) if isinstance(raw_s, str) else (raw_s or {})
        if sj is None or not isinstance(sj, dict):
            sj = {}
    except Exception:
        sj = {}
    gstat = str(sj.get("grading_status") or "").strip()
    failed = gstat == "failed"
    # state=grading：主观题仍在阅卷；state=graded：最终分数已汇总（含同步失败兜底）
    ready = st == "graded"
    if not gstat:
        gstat = "complete" if ready else ("pending" if st == "grading" else "")
    msg = str(sj.get("message") or "").strip()
    if st == "grading" and not failed:
        msg = msg or "阅卷中"
    elif failed:
        msg = msg or "主观题阅卷失败"
    out: Dict[str, Any] = {
        "attempt_id": int(attempt_id),
        "state": st,
        "grading_status": gstat,
        "ready": ready,
        "subjective_pending": sj.get("subjective_pending"),
        "grading_message": msg,
        "score_json_meta": sj,
    }
    if failed:
        out["error"] = sj.get("error")
    if ready:
        m = summarize_attempt_metrics_from_db(attempt_id=int(attempt_id))
        sc = float(m.get("score") or 0.0)
        ts = float(m.get("total_score") or 0.0)
        out.update(
            {
                "total_score": ts,
                "score": sc,
                "graded_count": int(m.get("graded_count") or 0),
                "correct_count": int(m.get("correct_count") or 0),
                "wrong_count": int(m.get("wrong_count") or 0),
            }
        )
    return out


def submit_answers(*, attempt_id: int, answers: List[Dict[str, Any]]) -> Dict[str, Any]:
    repo.save_attempt_answers(attempt_id, answers)
    return {"attempt_id": attempt_id, "saved": len(answers)}


def submit_answers_and_grade(*, attempt_id: int, answers: List[Dict[str, Any]], collection: Optional[str] = None) -> Dict[str, Any]:
    """落库作答后客观题立即判分；主观题异步模型阅卷，就绪后再汇总总分。"""
    repo.save_attempt_answers(attempt_id, answers)
    row = repo.get_attempt_by_id(attempt_id)
    if not row:
        raise ValueError(f"attempt not found: {attempt_id}")
    coll = (collection or row.get("collection") or "").strip() or "regulations"
    g = auto_grade_attempt(collection=coll, attempt_id=int(attempt_id), paper_id=None)
    saved = len([a for a in answers if isinstance(a, dict)])
    out = dict(g)
    out["attempt_id"] = int(attempt_id)
    out["saved"] = saved
    if g.get("grading_status") == "pending":
        out["score"] = None
        out["total_score"] = None
    else:
        out["score"] = g.get("score")
        out["total_score"] = g.get("total_score")
    return out


def is_exam_attempt(*, attempt_id: int) -> bool:
    """用于网关下线判断：exam 链路已迁移到 aiword，本接口仅识别 attempt 的 mode。"""
    try:
        row = repo.get_attempt_by_id(int(attempt_id))
    except Exception:
        row = None
    if not row or not isinstance(row, dict):
        return False
    mode = str(row.get("mode") or "").strip().lower()
    return mode == "exam"


# -----------------------------
# 整卷主观判分 Job（供 aiword 本地考试调用）
# -----------------------------
_paper_grade_lock = threading.Lock()
_paper_grade_jobs: dict[str, dict[str, Any]] = {}


def _evidence_for_subjective(agent: ReviewAgent, exam_track: str, stem: str, top_k: int = 6) -> List[Dict[str, Any]]:
    q = f"{EXAM_TRACKS.get(exam_track, exam_track)} {stem}".strip()
    try:
        rows = agent.search_knowledge(q, top_k=int(top_k), use_checkpoints=True)
    except Exception:
        rows = []
    out: List[Dict[str, Any]] = []
    for r in rows:
        content = str(r.get("content") or "").strip()
        if not content:
            continue
        meta = r.get("metadata") or {}
        src = str(meta.get("source_file") or meta.get("title") or r.get("source") or "").strip()
        out.append({"source_file": src, "snippet": content[:500]})
        if len(out) >= int(top_k):
            break
    return out


def _grade_subjective_question_llm(
    *,
    exam_track: str,
    stem: str,
    user_answer: Any,
    evidence: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """返回 {score(0~1), reason, recommendation, evidence_used[]}；证据只要求文件名定位。"""
    ev_lines = "\n".join(
        [f"- {str(e.get('source_file') or '').strip()}: {str(e.get('snippet') or '')[:240]}" for e in (evidence or [])]
    )
    prompt = f"""
你是医疗器械注册资料相关考试的阅卷老师。请对“学生答案”按 0~1 评分并返回 JSON：
{{
  "score": 0.0,
  "reason": "...",
  "recommendation": "...",
  "evidence_used": [{{"source_file":"...","snippet":"..."}}]
}}

体考类型: {EXAM_TRACKS.get(exam_track, exam_track)}
题干: {stem}
学生答案: {json.dumps(user_answer, ensure_ascii=False)}

可用证据（仅供参考，优先引用其中内容）：\n{ev_lines}

要求：
- score 为 0~1 浮点数
- evidence_used 至少返回 1 条（仅需要 source_file 文件名；snippet 可截断）
- 不要输出除 JSON 外的任何文本
""".strip()
    try:
        prov = (settings.quiz_provider or settings.provider or "").strip().lower()
        model = (settings.quiz_llm_model or settings.llm_model or "").strip()
        temp = float(getattr(settings, "quiz_temperature", 0.2) or 0.2)
        raw = invoke_chat_direct(prompt, temperature=temp, provider=prov, model=model)
        data = json.loads(_norm_json_text(raw))
        score = max(0.0, min(1.0, float(data.get("score") or 0.0)))
        reason = str(data.get("reason") or "").strip()[:2000]
        reco = str(data.get("recommendation") or "").strip()[:2000]
        used = data.get("evidence_used") if isinstance(data.get("evidence_used"), list) else []
        used2: List[Dict[str, Any]] = []
        for u in used:
            if not isinstance(u, dict):
                continue
            sf = str(u.get("source_file") or "").strip()
            sn = str(u.get("snippet") or "").strip()
            if not sf:
                continue
            used2.append({"source_file": sf, "snippet": sn[:500]})
            if len(used2) >= 6:
                break
        if not used2 and evidence:
            used2 = [{"source_file": str(evidence[0].get("source_file") or "").strip(), "snippet": str(evidence[0].get("snippet") or "")[:500]}]
        return {"score": score, "reason": reason, "recommendation": reco, "evidence_used": used2}
    except Exception:
        # 失败兜底：保留证据文件名，便于审计与排查
        used_fallback = []
        for e in (evidence or [])[:1]:
            sf = str(e.get("source_file") or "").strip()
            if sf:
                used_fallback.append({"source_file": sf, "snippet": str(e.get("snippet") or "")[:300]})
        return {"score": 0.0, "reason": "ai_score_failed", "recommendation": "", "evidence_used": used_fallback}


def start_paper_grading_job(
    *,
    collection: str,
    exam_track: str,
    attempt_id: str,
    items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    job_id = uuid.uuid4().hex
    now_ts = datetime.utcnow().isoformat()
    with _paper_grade_lock:
        _paper_grade_jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "created_at": now_ts,
            "updated_at": now_ts,
            "attempt_id": attempt_id,
            "items": [],
            "error": None,
        }

    def _runner():
        with _paper_grade_lock:
            if job_id in _paper_grade_jobs:
                _paper_grade_jobs[job_id]["status"] = "running"
                _paper_grade_jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()
        try:
            agent = ReviewAgent(collection)
            out_items: List[Dict[str, Any]] = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                qid = str(it.get("question_id") or it.get("questionId") or "").strip()
                stem = str(it.get("stem") or "").strip()
                ua = it.get("user_answer")
                evidence = _evidence_for_subjective(agent, exam_track, stem, top_k=6)
                graded = _grade_subjective_question_llm(exam_track=exam_track, stem=stem, user_answer=ua, evidence=evidence)
                out_items.append(
                    {
                        "question_id": qid,
                        "score": graded.get("score"),
                        "reason": graded.get("reason"),
                        "recommendation": graded.get("recommendation"),
                        "evidence_used": graded.get("evidence_used") or [],
                    }
                )
            with _paper_grade_lock:
                if job_id in _paper_grade_jobs:
                    _paper_grade_jobs[job_id]["status"] = "done"
                    _paper_grade_jobs[job_id]["items"] = out_items
                    _paper_grade_jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()
        except Exception as e:
            with _paper_grade_lock:
                if job_id in _paper_grade_jobs:
                    _paper_grade_jobs[job_id]["status"] = "failed"
                    _paper_grade_jobs[job_id]["error"] = str(e)[:800]
                    _paper_grade_jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()

    threading.Thread(target=_runner, name=f"quiz_paper_grade_{job_id[:8]}", daemon=True).start()
    return {"job_id": job_id, "status": "pending", "attempt_id": attempt_id}


def get_paper_grading_job(*, job_id: str) -> Dict[str, Any]:
    jid = str(job_id or "").strip()
    if not jid:
        raise ValueError("job_id required")
    with _paper_grade_lock:
        job = _paper_grade_jobs.get(jid)
        if not job:
            raise ValueError(f"job not found: {jid}")
        return dict(job)


def get_attempt_answers_with_questions(*, attempt_id: int) -> Dict[str, Any]:
    rows = repo.list_attempt_answers_with_questions(attempt_id)
    # 直接返回题目+作答，用于 aiword 详情展示（未必已自动判分）
    return {"attempt_id": int(attempt_id), "items": rows}


def student_wrongbook(*, collection: str, user_id: str, limit: int = 80) -> Dict[str, Any]:
    rows = repo.list_wrong_questions_for_student(collection=collection, user_id=user_id, limit=limit)
    return {"collection": collection, "user_id": user_id, "count": len(rows), "items": rows}


def student_unpracticed_bank(*, collection: str, user_id: str, exam_track: str = "", limit: int = 100) -> Dict[str, Any]:
    total = repo.count_unpracticed_questions_for_student(
        collection=collection, user_id=user_id, exam_track=exam_track
    )
    rows = repo.list_unpracticed_questions_for_student(
        collection=collection, user_id=user_id, exam_track=exam_track, limit=limit
    )
    return {
        "collection": collection,
        "user_id": user_id,
        "exam_track": exam_track,
        "total_count": int(total),
        "count": len(rows),
        "items": rows,
    }


def grade_attempt_by_cache(*, collection: str, attempt_id: int, paper_id: Optional[int] = None) -> Dict[str, Any]:
    rows = repo.list_attempt_answers_with_questions(attempt_id)
    hit = 0
    correct_total = 0
    wrong_total = 0
    for r in rows:
        qid = int(r["question_id"])
        rule = repo.get_grading_rule(collection=collection, paper_id=paper_id, question_id=qid, version=1)
        if not rule:
            continue
        ev = _score_objective_answer(
            r["question_type"], rule.get("answer_key"), r.get("user_answer"), r.get("options")
        )
        score = float(ev["score"])
        hit += 1
        if bool(ev["is_correct"]):
            correct_total += 1
        else:
            wrong_total += 1
        repo.update_answer_grade(
            answer_id=int(r["answer_id"]),
            auto_score=score,
            final_score=score,
            is_correct=bool(ev["is_correct"]),
            teacher_comment="cache_rule",
            graded_by_cache=True,
        )
        repo.log_grading_cache_hit(attempt_id, qid, "rule", 1.0)
    repo.finalize_attempt(
        attempt_id,
        {
            "score": _percent_score(correct_total, hit),
            "total_score": 100.0 if hit > 0 else 0.0,
            "graded_count": hit,
            "correct_count": correct_total,
            "wrong_count": wrong_total,
        },
        state="graded",
    )
    return {
        "attempt_id": attempt_id,
        "score": _percent_score(correct_total, hit),
        "total_score": 100.0 if hit > 0 else 0.0,
        "graded_count": hit,
        "correct_count": correct_total,
        "wrong_count": wrong_total,
        "mode": "cache",
    }


def auto_grade_attempt(*, collection: str, attempt_id: int, paper_id: Optional[int] = None) -> Dict[str, Any]:
    """客观题/缓存规则题立即打分；主观题（无缓存规则的非客观类型）不写 LLM，仅标记阅卷中队列。"""
    rows = repo.list_attempt_answers_with_questions(attempt_id)
    graded = 0
    correct_total = 0
    wrong_total = 0
    pending_rows: List[Dict[str, Any]] = []

    for r in rows:
        qid = int(r["question_id"])
        rule = repo.get_grading_rule(collection=collection, paper_id=paper_id, question_id=qid, version=1)
        if rule:
            ev = _score_objective_answer(
                r["question_type"], rule.get("answer_key"), r.get("user_answer"), r.get("options")
            )
            score = float(ev["score"])
            repo.log_grading_cache_hit(attempt_id, qid, "rule", 1.0)
            graded_by_cache = True
            comment = "cache_rule"
        else:
            if r["question_type"] in ("single_choice", "multiple_choice", "true_false"):
                ev = _score_objective_answer(
                    r["question_type"], r.get("answer"), r.get("user_answer"), r.get("options")
                )
                score = float(ev["score"])
                comment = "direct_answer"
                repo.upsert_grading_rule(
                    collection=collection,
                    paper_id=paper_id,
                    question_id=qid,
                    version=1,
                    answer_key=r.get("answer"),
                    rubric={"source": "auto"},
                    updated_by="system",
                )
                graded_by_cache = False
            else:
                pending_rows.append(r)
                repo.update_answer_grade(
                    answer_id=int(r["answer_id"]),
                    auto_score=0.0,
                    final_score=0.0,
                    is_correct=False,
                    teacher_comment="pending_subjective_grading",
                    graded_by_cache=False,
                )
                continue

        graded += 1
        if bool(ev["is_correct"]):
            correct_total += 1
        else:
            wrong_total += 1
        repo.update_answer_grade(
            answer_id=int(r["answer_id"]),
            auto_score=score,
            final_score=score,
            is_correct=bool(ev["is_correct"]),
            teacher_comment=comment,
            graded_by_cache=graded_by_cache,
        )

    if not pending_rows:
        sc = _percent_score(correct_total, graded)
        repo.finalize_attempt(
            attempt_id,
            {
                "score": sc,
                "total_score": 100.0 if graded > 0 else 0.0,
                "graded_count": graded,
                "correct_count": correct_total,
                "wrong_count": wrong_total,
                "grading_status": "complete",
            },
            state="graded",
        )
        return {
            "attempt_id": attempt_id,
            "score": sc,
            "total_score": 100.0 if graded > 0 else 0.0,
            "graded_count": graded,
            "correct_count": correct_total,
            "wrong_count": wrong_total,
            "grading_status": "complete",
            "mode": "auto",
        }

    repo.finalize_attempt(
        attempt_id,
        {
            "grading_status": "pending",
            "message": "主观题阅卷中",
            "subjective_pending": len(pending_rows),
        },
        state="grading",
    )
    _enqueue_async_subjective_if_needed(attempt_id=int(attempt_id), collection=collection, paper_id=paper_id)
    return {
        "attempt_id": attempt_id,
        "grading_status": "pending",
        "message": "阅卷中",
        "subjective_pending": len(pending_rows),
        "total_score": None,
        "graded_count": None,
        "correct_count": None,
        "wrong_count": None,
        "mode": "auto",
    }


def upsert_grading_rule(
    *,
    collection: str,
    paper_id: Optional[int],
    question_id: int,
    answer_key: Any,
    rubric: Any,
    updated_by: str,
) -> Dict[str, Any]:
    repo.upsert_grading_rule(
        collection=collection,
        paper_id=paper_id,
        question_id=question_id,
        version=1,
        answer_key=answer_key,
        rubric=rubric,
        updated_by=updated_by,
    )
    return {"ok": True}


def publish_set(set_id: int) -> Dict[str, Any]:
    repo.publish_set(set_id)
    return {"set_id": set_id, "id": set_id, "status": "published"}


def review_set_by_ai(set_id: int) -> Dict[str, Any]:
    """同步执行套题 AI 复审（供异步任务线程调用；可能较慢）。"""
    root = repo.load_set(set_id)
    if not root:
        raise ValueError("set not found")
    # 当前版本用轻量复核：随机抽样并更新时间戳；后续可替换为逐题 AI 复核
    sampled = random.sample(root["items"], min(3, len(root["items"]))) if root["items"] else []
    return {"set_id": set_id, "id": set_id, "checked_items": len(sampled), "status": "reviewed"}


def start_review_set_by_ai_job(*, collection: str, set_id: int, created_by: str) -> Dict[str, Any]:
    """异步启动复审：立即返回 job_id，后台线程执行 review_set_by_ai。"""
    job_id = repo.create_review_job(collection=collection, set_id=int(set_id), created_by=created_by or "")
    repo.update_review_job(job_id, status="running", message="starting")

    def _run():
        try:
            out = review_set_by_ai(int(set_id))
            repo.update_review_job(job_id, status="done", message="ok", result=out)
        except Exception as e:
            repo.update_review_job(job_id, status="failed", message=str(e)[:2000], result={"error": str(e)})

    threading.Thread(target=_run, name=f"quiz_review_{job_id}", daemon=True).start()
    return {"job_id": job_id, "set_id": int(set_id), "status": "running"}


def fetch_review_job(job_id: int) -> Dict[str, Any]:
    row = repo.get_review_job(job_id)
    if not row:
        raise ValueError("job not found")
    d = dict(row)
    rid = d.get("id")
    if rid is not None:
        d.setdefault("job_id", int(rid))
    sid = d.get("set_id")
    if sid is not None:
        d["set_id"] = int(sid)
    return d


def get_tracks_inventory(collection: str) -> List[Dict[str, Any]]:
    rows = repo.get_bank_tracks_inventory(collection)
    out = []
    for r in rows:
        t = str(r.get("exam_track") or "")
        out.append({"exam_track": t, "label": EXAM_TRACKS.get(t, t), "total": int(r.get("total") or 0)})
    return out


def get_overview_stats(collection: str) -> Dict[str, Any]:
    return repo.get_overview_stats(collection)


def get_stats_options(collection: str) -> Dict[str, Any]:
    return repo.get_stats_options(collection)


def list_sets(
    *,
    collection: str,
    set_type: str = "",
    exam_track: str = "",
    status: str = "",
    q: str = "",
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    items, total = repo.list_sets(
        collection=collection,
        set_type=set_type,
        exam_track=exam_track,
        status=status,
        q=q,
        limit=limit,
        offset=offset,
    )
    return {"items": items, "total": int(total), "limit": int(limit), "offset": int(offset)}


def delete_set(*, set_id: int) -> Dict[str, Any]:
    repo.delete_set(int(set_id))
    return {"set_id": int(set_id), "deleted": True}


def get_set(*, set_id: int) -> Dict[str, Any]:
    root = repo.load_set(int(set_id))
    if not root:
        raise ValueError("set not found")
    return root


def admin_list_bank_questions(
    *,
    collection: str,
    exam_track: str = "",
    q: str = "",
    category: str = "",
    question_type: str = "",
    difficulty: str = "",
    is_active: Optional[bool] = True,
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    total = repo.admin_count_bank_questions(
        collection=collection,
        exam_track=exam_track or None,
        q=q,
        category=category,
        question_type=question_type,
        difficulty=difficulty,
        is_active=is_active,
    )
    items = repo.admin_list_bank_questions(
        collection=collection,
        exam_track=exam_track or None,
        q=q,
        category=category,
        question_type=question_type,
        difficulty=difficulty,
        is_active=is_active,
        limit=limit,
        offset=offset,
    )
    return {"items": items, "total": int(total), "limit": int(limit), "offset": int(offset)}


def admin_patch_bank_question(
    *,
    collection: str,
    question_id: int,
    stem: Optional[str] = None,
    options: Optional[List[str]] = None,
    answer_present: bool = False,
    answer: Any = None,
    explanation: Optional[str] = None,
    evidence: Optional[List[Dict[str, Any]]] = None,
    status: Optional[str] = None,
    exam_track: Optional[str] = None,
    category: Optional[str] = None,
    question_type: Optional[str] = None,
    difficulty: Optional[str] = None,
    is_active: Optional[bool] = None,
) -> Dict[str, Any]:
    repo.admin_update_question(
        collection=collection,
        question_id=int(question_id),
        stem=stem,
        options=options,
        explanation=explanation,
        evidence=evidence,
        status=status,
    )
    if answer_present:
        repo.admin_update_question_answer(collection=collection, question_id=int(question_id), answer=answer)
    repo.admin_update_bank_fields(
        collection=collection,
        question_id=int(question_id),
        exam_track=exam_track,
        category=category,
        question_type=question_type,
        difficulty=difficulty,
        is_active=is_active,
    )
    return {"question_id": int(question_id), "ok": True}


def admin_delete_bank_question(*, collection: str, question_id: int) -> Dict[str, Any]:
    repo.admin_deactivate_question(collection=collection, question_id=int(question_id))
    return {"question_id": int(question_id), "deleted": True}

