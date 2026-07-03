import hashlib
import json
import logging
import time
import uuid
import math
import random
import re
import threading
from difflib import SequenceMatcher
from collections import deque
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from config import settings
from src.core.agent import ReviewAgent
from src.core.llm_factory import invoke_chat_direct

from .models import EXAM_CATEGORIES, EXAM_TRACKS, QUESTION_TYPES
from . import repository as repo

logger = logging.getLogger(__name__)

TRACK_HINTS = {
    "cn": "中国医疗器械法规、注册与质量体系要求",
    "iso13485": "ISO 13485 质量管理体系要求与实施",
    "mdsap": "MDSAP 审核程序与多国监管要求",
}

# 练习身份（与 aicheckword 初稿 author_role keys 对齐）→ 项目案例文件名关键词范围。
# pm=产品经理（需求/SRS 等产品文档）；pjm=项目经理（项目计划 + 研发/设计/验证等项目交付文件，按约定覆盖研发链路）。
_AUTHOR_ROLE_FILE_KEYWORDS: Dict[str, List[str]] = {
    "pm": [
        "需求",
        "requirement",
        "urs",
        "srs",
        "产品",
        "product",
        "规格",
        "specification",
        "用户故事",
        "功能需求",
        "非功能",
        "适用范围",
        "预期用途",
        "prd",
    ],
    "pjm": [
        "项目计划",
        "project plan",
        "项目",
        "project",
        "里程碑",
        "进度",
        "wbs",
        "计划",
        "变更",
        "需求",
        "requirement",
        "srs",
        "设计",
        "验证",
        "测试",
        "研发",
        "开发",
        "架构",
        "代码",
        "风险",
        "risk",
        "追溯",
        "trace",
        "版本",
        "配置",
        "sdd",
        "详细设计",
        "交付",
        "资源",
        "干系人",
        "结项",
        "立项",
        "sdp",
        "软件计划",
        "项目报告",
        # 体考/法规：项目交付与 design control 核查口径
        "设计开发",
        "设计控制",
        "核查指南",
        "法规",
        "指导原则",
        "可追溯",
        "软件生存周期",
        "生命周期",
        "确认",
    ],
    "rm": [
        "风险",
        "hazard",
        "fmea",
        "风险分析",
        "风险控制",
        "残余风险",
        # 体考/法规：风险管理与受益-风险
        "14971",
        "iso 14971",
        "风险管理",
        "风险可接受",
        "受益",
        "安全",
        "核查指南",
        "法规",
        "指导原则",
        "不良事件",
        "临床",
        "危害",
    ],
    "rdm": [
        "研发",
        "开发",
        "设计",
        "架构",
        "sdd",
        "设计说明",
        "需求",
        "代码",
        "验证",
        # 体考/法规：设计开发 / SaMD 生命周期
        "设计开发",
        "软件生存周期",
        "软件生命周期",
        "医疗器械软件",
        "独立软件",
        "现成软件",
        "soup",
        "核查指南",
        "法规",
        "指导原则",
        "确认",
        "可追溯",
        "设计变更",
        "设计控制",
    ],
    "ui": ["ui", "界面", "交互", "可用性", "视觉", "原型"],
    "qa": [
        "测试",
        "test",
        "验证",
        "v&v",
        "缺陷",
        "验证报告",
        "测试用例",
        "确认",
        "单元测试",
        "集成测试",
        "系统测试",
        # 体考/法规：V&V 与体系核查
        "设计验证",
        "设计确认",
        "确认与验证",
        "核查指南",
        "法规",
        "指导原则",
        "审核",
        "可追溯",
        "医疗器械软件",
        "独立软件",
    ],
    "cm": [
        "配置管理计划",
        "配置状态报告",
        "配置审计",
        "配置基线",
        "配置控制",
        "基线管理",
        "配置",
        "cm",
        "baseline",
        "版本",
        "发布",
        "变更",
        "追溯",
        "配置管理",
    ],
    "ra": [
        "注册", "法规", "标准", "指导原则", "合规", "申报",
        "条例", "办法", "通告", "公告", "mdr", "ivdr", "510k",
        "分类", "界定", "审查", "审评", "受理", "ce", "临床评价",
        "同品种", "比对", "13485", "iso", "现场核查", "监督检查", "质量体系",
    ],
    "prod": [
        "生产",
        "工艺",
        "制造",
        "gmp",
        "批记录",
        "检验",
        "放行",
        "sop",
        "批生产",
        "物料",
        "生产工艺",
        "发布",
        "软件发布",
        "生产质量管理",
        "生产质量管理规范",
        "委托生产",
        "受托生产",
        "批放行",
        "制造过程",
        "生产过程",
        "生产环境",
        "生产现场",
        "独立软件",
        "转换",
        # 体考/法规：生产与上市放行
        "体考",
        "现场核查",
        "核查指南",
        "法规",
        "指导原则",
        "监督管理",
        "质量管理规范",
        "上市",
    ],
}

COMMON_AUTHOR_ROLE_KEY = "common"
COMMON_AUTHOR_ROLE_LABEL = "通用（各身份必考）"
_FOCUS_EXAM_ROLES = {"pjm", "rdm", "rm", "qa", "prod", "cm"}
_LEADERSHIP_CROSS_ROLES = {"pjm", "rdm"}


def _load_role_focus_config() -> Dict[str, Dict[str, Any]]:
    try:
        from src.core.quiz.role_focus_config import ROLE_FOCUS_CONFIG

        if isinstance(ROLE_FOCUS_CONFIG, dict):
            return ROLE_FOCUS_CONFIG
    except Exception:
        pass
    return {}


_ROLE_FOCUS_CONFIG: Dict[str, Dict[str, Any]] = _load_role_focus_config()


def _role_common_label_empty() -> bool:
    try:
        from src.core.quiz.role_focus_config import COMMON_ROLE_LABEL_EMPTY

        return bool(COMMON_ROLE_LABEL_EMPTY)
    except Exception:
        return True


_COMMON_ROLE_LABEL_EMPTY = _role_common_label_empty()


def _cfg_list(role: str, field: str) -> List[str]:
    cfg = _ROLE_FOCUS_CONFIG.get(str(role).strip().lower()) or {}
    vals = cfg.get(field)
    if not isinstance(vals, list):
        return []
    return [str(x).strip() for x in vals if str(x).strip()]


# 体考关注度权重（老师端不勾身份时按此自动分配命题岗位；pjm/rdm 更高）
_ROLE_EXAM_WEIGHT: Dict[str, float] = {}
for _r, _c in (_ROLE_FOCUS_CONFIG or {}).items():
    try:
        _ROLE_EXAM_WEIGHT[str(_r).strip().lower()] = float((_c or {}).get("exam_weight") or 0.0)
    except (TypeError, ValueError):
        _ROLE_EXAM_WEIGHT[str(_r).strip().lower()] = 0.0

# 岗位命题口吻/侧重 与 必考文件（来自配置，供提示语使用）
_ROLE_PROMPT_EMPHASIS: Dict[str, str] = {
    str(_r).strip().lower(): str((_c or {}).get("prompt_emphasis") or "").strip()
    for _r, _c in (_ROLE_FOCUS_CONFIG or {}).items()
}
_ROLE_MUST_FILES: Dict[str, List[str]] = {
    str(_r).strip().lower(): _cfg_list(_r, "must_files") for _r in (_ROLE_FOCUS_CONFIG or {}).keys()
}
# 易混淆需排除词（仅命中这些词、且无上下文时不计入该岗位）
_ROLE_EXCLUDE_WORDS: Dict[str, set[str]] = {
    str(_r).strip().lower(): {w.lower() for w in _cfg_list(_r, "exclude_words")}
    for _r in (_ROLE_FOCUS_CONFIG or {}).keys()
}

_PROD_WEAK_KEYWORDS = _ROLE_EXCLUDE_WORDS.get("prod") or {"发布", "软件发布", "上市"}
_PROD_CONTEXT_HINTS = set(_cfg_list("prod", "must_hints")) | set(_cfg_list("prod", "topics")) | {
    "生产",
    "放行",
    "批放行",
    "批记录",
    "gmp",
    "生产质量管理",
    "生产质量管理规范",
    "委托生产",
    "受托生产",
}
_CM_WEAK_KEYWORDS = _ROLE_EXCLUDE_WORDS.get("cm") or {"版本", "发布", "变更", "追溯", "baseline"}
_CM_CONTEXT_HINTS = set(_cfg_list("cm", "must_hints")) | {
    "配置管理",
    "配置管理计划",
    "配置状态报告",
    "配置审计",
    "配置项",
    "配置基线",
    "基线管理",
    "配置控制",
}

# 将配置中的别名并入岗位识别关键词（去重，配置为增量补充，不破坏内置）
for _r, _c in (_ROLE_FOCUS_CONFIG or {}).items():
    _rk = str(_r).strip().lower()
    if not _rk:
        continue
    _base = [str(x).strip().lower() for x in (_AUTHOR_ROLE_FILE_KEYWORDS.get(_rk) or []) if str(x).strip()]
    _extra = [str(x).strip().lower() for x in (_c or {}).get("aliases", []) if str(x).strip()]
    _extra += [str(x).strip().lower() for x in (_c or {}).get("must_hints", []) if str(x).strip()]
    _merged: List[str] = []
    _seen_kw: set[str] = set()
    for _kw in _base + _extra:
        if _kw and _kw not in _seen_kw:
            _seen_kw.add(_kw)
            _merged.append(_kw)
    if _merged:
        _AUTHOR_ROLE_FILE_KEYWORDS[_rk] = _merged
_REGULATORY_HINTS = {
    "法规",
    "条例",
    "指导原则",
    "核查指南",
    "现场核查",
    "监督管理",
    "标准",
    "gmp",
    "质量管理规范",
}
_ROLE_REGULATORY_HINTS: Dict[str, set[str]] = {
    "pjm": {"设计控制", "设计开发", "软件生存周期", "可追溯", "核查指南", "法规", "指导原则"},
    "rdm": {"软件生命周期", "设计开发", "现成软件", "soup", "核查指南", "法规", "指导原则"},
    "rm": {"14971", "风险管理", "残余风险", "可接受", "核查指南", "法规", "指导原则"},
    "qa": {"设计验证", "设计确认", "确认与验证", "测试", "核查指南", "法规", "指导原则"},
    "prod": {"gmp", "生产质量管理规范", "批放行", "放行", "核查指南", "法规", "指导原则", "转换"},
}
# 必考硬性关键词 / 必考主题：优先取配置（role_focus_config.py），配置缺项时用内置默认兜底
_ROLE_MUST_HINTS_DEFAULT: Dict[str, set[str]] = {
    "cm": {"配置管理计划", "配置状态报告", "配置审计", "配置基线", "基线管理", "配置控制", "配置项", "版本控制", "变更控制"},
    "prod": {"gmp", "生产质量管理规范", "批放行", "生产工艺", "批记录", "生产工艺规程", "洁净", "灭菌", "工艺验证"},
    "pjm": {"项目计划", "里程碑", "进度", "wbs", "设计控制", "设计开发", "资源", "干系人", "结项", "立项"},
    "rdm": {"详细设计", "架构设计", "软件生命周期", "软件生存周期", "现成软件", "soup", "设计开发", "设计变更", "代码"},
    "rm": {"风险管理计划", "fmea", "风险分析", "风险控制", "残余风险", "受益风险", "iso 14971", "风险可接受准则"},
    "qa": {"测试用例", "测试方案", "验证方案", "确认方案", "设计验证", "设计确认", "缺陷管理", "单元测试", "集成测试", "系统测试"},
    "ra": {"注册申报", "临床评价", "同品种", "分类界定", "现场核查", "质量体系", "iso 13485", "注册变更"},
    "pm": {"用户需求", "urs", "srs", "产品需求", "功能需求", "非功能需求", "预期用途", "适用范围"},
    "ui": {"界面设计", "交互设计", "可用性", "人因", "原型", "用户体验"},
}
_ROLE_MUST_HINTS: Dict[str, set[str]] = {}
for _r in set(list(_ROLE_MUST_HINTS_DEFAULT.keys()) + list((_ROLE_FOCUS_CONFIG or {}).keys())):
    _cfg_must = {w for w in _cfg_list(_r, "must_hints")}
    _ROLE_MUST_HINTS[_r] = _cfg_must or set(_ROLE_MUST_HINTS_DEFAULT.get(_r) or set())

_ROLE_FOCUS_TOPICS: Dict[str, List[str]] = {}
for _r in (_ROLE_FOCUS_CONFIG or {}).keys():
    _tp = _cfg_list(_r, "topics")
    if _tp:
        _ROLE_FOCUS_TOPICS[str(_r).strip().lower()] = _tp


_ROLE_COVERAGE_CACHE: Dict[str, tuple[float, Dict[str, Any]]] = {}
_ROLE_COVERAGE_CACHE_TTL_SEC = 300.0


class QuizRequestError(Exception):
    """组卷/录题等入口参数不合法（应由 API 层映射为 HTTP 400）。"""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.args[0] if self.args else "QuizRequestError"


def require_project_case_quiz(*, collection: str, exam_category: str, project_case_id: Any) -> Optional[int]:
    """考试类型为 project_case 时校验并返回案例 id；其它类型返回 None。"""
    ec = _normalize_exam_category(exam_category)
    if ec != "project_case":
        return None
    if project_case_id is None or str(project_case_id).strip() == "":
        raise QuizRequestError("项目案例考试必须提供 project_case_id（请选择已训练入库的项目案例）")
    try:
        pid = int(project_case_id)
    except (TypeError, ValueError):
        raise QuizRequestError("project_case_id 无效") from None
    if pid <= 0:
        raise QuizRequestError("project_case_id 无效")
    from src.core.db import get_project_case, get_project_case_file_names

    row = get_project_case(pid)
    if not row:
        raise QuizRequestError("项目案例不存在")
    coll = str(row.get("collection") or "").strip()
    if coll != str(collection or "").strip():
        raise QuizRequestError("项目案例不属于当前知识库 collection")
    names = get_project_case_file_names(collection, pid) or []
    if not names:
        raise QuizRequestError("该案例尚未训练入库项目案例知识库，无法组卷/录题")
    return pid


def list_ready_project_cases_for_quiz(*, collection: str) -> List[Dict[str, Any]]:
    """与已训练 project_case 知识库一致的案例列表（供考试中心下拉）。"""
    from src.core.db import get_project_case_file_names, list_project_cases

    out: List[Dict[str, Any]] = []
    for row in list_project_cases(collection) or []:
        if not isinstance(row, dict):
            continue
        try:
            cid = int(row.get("id") or 0)
        except (TypeError, ValueError):
            cid = 0
        if cid <= 0:
            continue
        names = get_project_case_file_names(collection, cid) or []
        if not names:
            continue
        out.append(
            {
                "id": cid,
                "case_name": str(row.get("case_name") or ""),
                "product_name": str(row.get("product_name") or ""),
                "registration_country": str(row.get("registration_country") or ""),
                "registration_type": str(row.get("registration_type") or ""),
                "registration_component": str(row.get("registration_component") or ""),
                "project_form": str(row.get("project_form") or ""),
            }
        )
    return out


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


def _normalize_exam_category(v: Any) -> str:
    s = str(v or "").strip().lower()
    if s in ("new_standard", "new_standard_release", "newstd", "新标", "新标发布"):
        return "new_standard"
    if s in (
        "project_case",
        "projectcase",
        "case_audit",
        "caseaudit",
        "项目案例",
        "案例考试",
        "项目案例考试",
    ):
        return "project_case"
    return "daily"


def _normalize_author_roles(author_roles: Any) -> List[str]:
    try:
        from src.core.draft_integration_ui_meta import DRAFT_AUTHOR_ROLE_KEYS

        allowed = {str(x).strip().lower() for x in (DRAFT_AUTHOR_ROLE_KEYS or []) if str(x).strip()}
    except Exception:
        allowed = {str(k).strip().lower() for k in (_AUTHOR_ROLE_FILE_KEYWORDS or {}).keys() if str(k).strip()}
    raw_list: List[str] = []
    if isinstance(author_roles, list):
        raw_list = [str(x).strip().lower() for x in author_roles]
    elif author_roles is not None:
        raw = str(author_roles).strip()
        if raw:
            raw_list = [seg.strip().lower() for seg in raw.split(",")]
    out: List[str] = []
    seen: set[str] = set()
    for role in raw_list:
        if not role or role in seen:
            continue
        if role not in allowed:
            continue
        seen.add(role)
        out.append(role)
    return out


def _role_file_keyword_scope(author_roles: List[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for role in author_roles or []:
        kws = [str(x).strip().lower() for x in (_AUTHOR_ROLE_FILE_KEYWORDS.get(role) or []) if str(x).strip()]
        if kws:
            out[role] = kws
    return out


def _all_author_role_keyword_scope() -> Dict[str, List[str]]:
    return _role_file_keyword_scope(list(_AUTHOR_ROLE_FILE_KEYWORDS.keys()))


def _question_search_text(question: Dict[str, Any]) -> str:
    parts: List[str] = [
        str(question.get("stem") or ""),
        str(question.get("explanation") or ""),
        str(question.get("category") or ""),
    ]
    ev = question.get("evidence")
    if isinstance(ev, list):
        for item in ev:
            if not isinstance(item, dict):
                continue
            parts.append(str(item.get("source_file") or ""))
            parts.append(str(item.get("file_name") or ""))
            parts.append(str(item.get("content_snippet") or item.get("content") or ""))
            md = item.get("metadata")
            if isinstance(md, dict):
                parts.append(str(md.get("source_file") or ""))
                parts.append(str(md.get("category") or ""))
                parts.append(str(md.get("case_id") or ""))
    elif question.get("evidence_json") is not None:
        raw_ej = question.get("evidence_json")
        try:
            parsed = json.loads(raw_ej) if isinstance(raw_ej, str) else raw_ej
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        parts.append(str(item.get("source_file") or ""))
                        parts.append(str(item.get("file_name") or ""))
                        parts.append(str(item.get("content_snippet") or item.get("content") or ""))
            else:
                parts.append(str(raw_ej))
        except Exception:
            parts.append(str(raw_ej))
    return " ".join(parts).lower()


def _question_source_files(question: Dict[str, Any]) -> List[str]:
    ev = question.get("evidence")
    if not isinstance(ev, list):
        return []
    out: List[str] = []
    for item in ev:
        if not isinstance(item, dict):
            continue
        sf = str(item.get("source_file") or "").strip()
        if sf:
            out.append(sf)
    return out


_AUDIT_CHECKPOINT_FIELD_NAMES: tuple[str, ...] = (
    "审核点编号",
    "审核类别",
    "审核点名称",
    "详细描述",
    "法规依据",
    "检查方法",
    "严重程度",
    "适用文档",
)


def _is_audit_checkpoint_text(text: str) -> bool:
    s = str(text or "")
    return "审核点编号" in s or "审核点清单:" in s


def _parse_audit_checkpoint_fields(text: str) -> Dict[str, str]:
    """从审核点题干/摘录解析结构化字段（保留详细描述整段，不在分号处截断）。"""
    s = str(text or "").strip()
    out: Dict[str, str] = {}
    if not s:
        return out
    pattern = r"(审核点编号|审核类别|审核点名称|详细描述|法规依据|检查方法|严重程度|适用文档)[：:]"
    parts = re.split(pattern, s)
    if len(parts) > 2:
        i = 1
        while i + 1 < len(parts):
            key = str(parts[i] or "").strip()
            val = str(parts[i + 1] or "").strip()
            if key and val:
                prev = str(out.get(key) or "").strip()
                if not prev or len(val) > len(prev):
                    out[key] = val
            i += 2
    for line in re.split(r"[\r\n]+", s):
        line = line.strip()
        if not line:
            continue
        for fn in _AUDIT_CHECKPOINT_FIELD_NAMES:
            if line.startswith(fn + "：") or line.startswith(fn + ":"):
                val = line.split("：", 1)[-1].split(":", 1)[-1].strip()
                if val:
                    prev = str(out.get(fn) or "").strip()
                    if not prev or len(val) > len(prev):
                        out[fn] = val
    return out


def _audit_checkpoint_fields_from_stem_and_evidence(
    stem: str,
    evidence: Optional[List[Any]] = None,
) -> Dict[str, str]:
    fields = _parse_audit_checkpoint_fields(stem)
    if not isinstance(evidence, list):
        return fields
    for ev in evidence:
        if not isinstance(ev, dict):
            continue
        sn = str(ev.get("content_snippet") or ev.get("content") or "").strip()
        if not sn or not _is_audit_checkpoint_text(sn):
            continue
        for k, v in _parse_audit_checkpoint_fields(sn).items():
            if not v:
                continue
            prev = str(fields.get(k) or "").strip()
            if not prev or len(v) > len(prev):
                fields[k] = v
    return fields


def _format_checklist_point_text(point: Dict[str, Any]) -> str:
    if not isinstance(point, dict):
        return ""
    docs = point.get("applicable_docs") or []
    if isinstance(docs, list):
        docs_str = ", ".join(str(x).strip() for x in docs if str(x).strip())
    else:
        docs_str = str(docs or "").strip()
    return (
        f"审核点编号：{point.get('id', '')}\n"
        f"审核类别：{point.get('category', '')}\n"
        f"审核点名称：{point.get('name', '')}\n"
        f"详细描述：{point.get('description', '')}\n"
        f"法规依据：{point.get('regulation_ref', '')}\n"
        f"检查方法：{point.get('check_method', '')}\n"
        f"严重程度：{point.get('severity', '')}\n"
        f"适用文档：{docs_str}"
    ).strip()


def _format_audit_checkpoint_true_false_stem(
    stem: str,
    evidence: Optional[List[Any]] = None,
) -> str:
    """审核点判断题：学生端仅展示待判断陈述（不含编号/法规/检查方法等字段）。"""
    fields = _audit_checkpoint_fields_from_stem_and_evidence(stem, evidence)
    if not fields.get("详细描述") and not fields.get("审核点编号"):
        return str(stem or "").strip()
    desc = str(fields.get("详细描述") or "").strip()
    if not desc:
        name = str(fields.get("审核点名称") or "").strip()
        method = str(fields.get("检查方法") or "").strip()
        if name and method:
            desc = f"{name}。检查要求：{method}"
        elif name:
            desc = name
        elif method:
            desc = method
    if not desc:
        return str(stem or "").strip()
    return f"判断下列陈述是否正确：\n{desc}".strip()


def _match_role_keywords_in_blob(
    blob: str,
    sfiles: List[str],
    role_keywords: Dict[str, List[str]],
) -> set[str]:
    hits: set[str] = set()
    if not role_keywords:
        return hits
    for role, kws in role_keywords.items():
        if not kws:
            continue
        kws_set = {str(x).strip().lower() for x in kws if str(x).strip()}
        if not kws_set:
            continue
        matched = False
        for kw in kws_set:
            if kw and kw in blob:
                matched = True
                break
        if not matched:
            for sf in sfiles:
                if _text_hits_any_keyword(sf, list(kws_set)):
                    matched = True
                    break
        if role == "prod" and matched:
            weak_hit = any(kw in _PROD_WEAK_KEYWORDS for kw in kws_set if kw in blob) or any(
                _text_hits_any_keyword(sf, list(_PROD_WEAK_KEYWORDS)) for sf in sfiles
            )
            if weak_hit:
                has_prod_context = any(ctx in blob for ctx in _PROD_CONTEXT_HINTS) or any(
                    _text_hits_any_keyword(sf, list(_PROD_CONTEXT_HINTS)) for sf in sfiles
                )
                has_strong = any((kw in blob) and (kw not in _PROD_WEAK_KEYWORDS) for kw in kws_set)
                if not has_strong and not has_prod_context:
                    matched = False
        if role == "cm" and matched:
            # 「发布/版本/变更」在多类题中都高频；配置管理员要求明确配置管理语境。
            weak_hit = any(kw in _CM_WEAK_KEYWORDS for kw in kws_set if kw in blob) or any(
                _text_hits_any_keyword(sf, list(_CM_WEAK_KEYWORDS)) for sf in sfiles
            )
            if weak_hit:
                has_cm_context = any(ctx in blob for ctx in _CM_CONTEXT_HINTS) or any(
                    _text_hits_any_keyword(sf, list(_CM_CONTEXT_HINTS)) for sf in sfiles
                )
                has_strong = any((kw in blob) and (kw not in _CM_WEAK_KEYWORDS) for kw in kws_set)
                if not has_strong and not has_cm_context:
                    matched = False
        if role == "qa" and matched:
            # 生产/配置语境下的「检验」不等同于测试工程师职责
            if any(x in blob for x in ("批记录", "批生产", "生产工艺", "gmp", "生产现场", "物料")):
                if not any(x in blob for x in ("测试方案", "测试用例", "验证方案", "确认方案", "v&v", "软件测试")):
                    matched = False
        if matched and role not in ("prod", "cm"):
            # 通用「易混淆排除词」门控：若仅因排除词命中、且无必考主题/关键词上下文，则不计入该岗位
            excl = _ROLE_EXCLUDE_WORDS.get(role) or set()
            if excl:
                weak_hit = any((kw in excl) and (kw in blob) for kw in kws_set) or any(
                    _text_hits_any_keyword(sf, list(excl)) for sf in sfiles
                )
                if weak_hit:
                    ctx_terms = {str(x).lower() for x in (_ROLE_MUST_HINTS.get(role) or set())}
                    ctx_terms |= {str(x).lower() for x in (_ROLE_FOCUS_TOPICS.get(role) or [])}
                    has_ctx = any(ct and ct in blob for ct in ctx_terms) or any(
                        _text_hits_any_keyword(sf, list(ctx_terms)) for sf in sfiles
                    )
                    has_strong = any((kw in blob) and (kw not in excl) for kw in kws_set)
                    if not has_strong and not has_ctx:
                        matched = False
        if matched:
            hits.add(role)
    return hits


def _infer_audit_checkpoint_role_hits(question: Dict[str, Any]) -> set[str]:
    blob = _question_search_text(question)
    if not _is_audit_checkpoint_text(blob):
        return set()
    fields = _audit_checkpoint_fields_from_stem_and_evidence(
        blob,
        _item_evidence_for_display(question),
    )
    name = str(fields.get("审核点名称") or "").strip()
    desc = str(fields.get("详细描述") or "").strip()
    docs = str(fields.get("适用文档") or "").strip()
    check_method = str(fields.get("检查方法") or "").strip()
    primary = f"{name} {desc} {docs} {check_method}".strip().lower()
    if not primary:
        return set()
    sfiles = _question_source_files(question)
    sfiles.extend([str(fields.get("审核点编号") or "").strip(), docs])
    hits = _match_role_keywords_in_blob(primary, [x for x in sfiles if x], _all_author_role_keyword_scope())
    reg = str(fields.get("法规依据") or "").strip().lower()
    if reg:
        ra_scope = {"ra": list(_AUTHOR_ROLE_FILE_KEYWORDS.get("ra") or [])}
        hits.update(_match_role_keywords_in_blob(reg, [], ra_scope))
    if any(x in name for x in ("配置管理", "配置变更", "配置基线", "配置状态", "配置审计", "配置控制")):
        hits.add("cm")
    if any(x in name for x in ("批生产", "批记录", "生产现场", "生产工艺", "gmp", "放行")):
        hits.add("prod")
    return hits


def _text_hits_any_keyword(text: str, keywords: List[str]) -> bool:
    low = str(text or "").strip().lower()
    if not low:
        return False
    return any(kw in low for kw in (keywords or []))


def _question_role_hits(question: Dict[str, Any], role_keywords: Dict[str, List[str]]) -> set[str]:
    return _question_role_hits_extended(question, role_keywords)


def _question_role_hits_extended(question: Dict[str, Any], role_keywords: Dict[str, List[str]]) -> set[str]:
    if not role_keywords:
        return set()
    blob = _question_search_text(question)
    if _is_audit_checkpoint_text(blob):
        # 审核点题仅用结构化字段推断，避免题干「审核点编号」等误命中测试工程师「审核」等泛词
        return _infer_audit_checkpoint_role_hits(question) & set(role_keywords.keys())
    sfiles = _question_source_files(question)
    return _match_role_keywords_in_blob(blob, sfiles, role_keywords)


def _question_hits_unselected_focus_roles(
    question: Dict[str, Any],
    selected_role_keywords: Dict[str, List[str]],
) -> set[str]:
    selected = {str(k).strip().lower() for k in selected_role_keywords.keys() if str(k).strip()}
    if not selected:
        return set()
    all_hits = _question_role_hits_extended(question, _all_author_role_keyword_scope())
    return {r for r in all_hits if r in _FOCUS_EXAM_ROLES and r not in selected}


def _question_focus_primary_for_selection(
    question: Dict[str, Any],
    selected_role_keywords: Dict[str, List[str]],
) -> bool:
    """优先选择命中所选重点身份、且不强命中未选重点身份的题。"""
    if not selected_role_keywords:
        return True
    sel_hits = _question_role_hits_extended(question, selected_role_keywords)
    if not sel_hits:
        return _question_is_universal_common(question)
    return not _question_hits_unselected_focus_roles(question, selected_role_keywords)


def _selection_has_leadership_cross_need(selected_role_keywords: Dict[str, List[str]]) -> bool:
    keys = {str(k).strip().lower() for k in selected_role_keywords.keys() if str(k).strip()}
    return any(k in _LEADERSHIP_CROSS_ROLES for k in keys)


def _question_is_leadership_cross_candidate(
    question: Dict[str, Any],
    selected_role_keywords: Dict[str, List[str]],
) -> bool:
    if not _selection_has_leadership_cross_need(selected_role_keywords):
        return False
    if _question_role_hits_extended(question, selected_role_keywords):
        return False
    if _question_is_universal_common(question):
        return False
    return bool(_question_hits_unselected_focus_roles(question, selected_role_keywords))


def _question_has_regulatory_context(question: Dict[str, Any], extra_hints: Optional[set[str]] = None) -> bool:
    blob = _question_search_text(question)
    hints = set(_REGULATORY_HINTS)
    if extra_hints:
        hints.update({str(x).strip().lower() for x in extra_hints if str(x).strip()})
    return any(h in blob for h in hints if h)


def _question_hits_role_regulatory(
    question: Dict[str, Any],
    role: str,
    role_keywords: Dict[str, List[str]],
) -> bool:
    if role not in _question_role_hits_extended(question, role_keywords):
        return False
    role_hints = _ROLE_REGULATORY_HINTS.get(str(role).strip().lower()) or set()
    return _question_has_regulatory_context(question, role_hints)


def _question_hits_role_must_hints(question: Dict[str, Any], role: str) -> bool:
    hints = _ROLE_MUST_HINTS.get(str(role).strip().lower()) or set()
    if not hints:
        return False
    blob = _question_search_text(question)
    return any(h in blob for h in hints if h)


def _question_is_universal_common(
    question: Dict[str, Any],
    all_role_keywords: Optional[Dict[str, List[str]]] = None,
) -> bool:
    """与任一编写身份关键词均不匹配 → 各身份通用必考基线题。"""
    scope = all_role_keywords or _all_author_role_keyword_scope()
    if _is_audit_checkpoint_text(_question_search_text(question)):
        return not bool(_infer_audit_checkpoint_role_hits(question))
    return not _question_role_hits_extended(question, scope)


_REGULATORY_COMMON_MARKERS: tuple[str, ...] = (
    "[国内体考]",
    "[欧盟体考]",
    "[美国体考]",
    "[澳大利亚体考]",
    "[加拿大体考]",
    "体考",
    "法规",
    "标准",
    "指导原则",
    "核查指南",
    "13485",
    "14971",
    "条例",
    "办法",
    "通告",
)


def _question_is_regulatory_common_baseline(
    question: Dict[str, Any],
    selected_role_keywords: Optional[Dict[str, List[str]]] = None,
) -> bool:
    """法规/体考类各身份共用考点（可与某一身份专属计数并存）。"""
    blob = _question_search_text(question)
    if _is_audit_checkpoint_text(blob):
        return not bool(_infer_audit_checkpoint_role_hits(question))
    if not _question_has_regulatory_context(question):
        return False
    low = blob.lower()
    if not any(m.lower() in low or m in blob for m in _REGULATORY_COMMON_MARKERS):
        return False
    if selected_role_keywords:
        hits = _question_role_hits_extended(question, selected_role_keywords)
        if len(hits) >= 2:
            return True
        if any(m in blob for m in ("[国内体考]", "[欧盟体考]", "[美国体考]", "体考")):
            return True
        if not hits:
            return True
    return True


def _question_counts_as_common_baseline(
    question: Dict[str, Any],
    selected_role_keywords: Optional[Dict[str, List[str]]] = None,
    all_role_keywords: Optional[Dict[str, List[str]]] = None,
) -> bool:
    scope = all_role_keywords or _all_author_role_keyword_scope()
    if _question_is_universal_common(question, scope):
        return True
    return _question_is_regulatory_common_baseline(question, selected_role_keywords)


def _question_eligible_for_selected_roles(
    question: Dict[str, Any],
    selected_role_keywords: Dict[str, List[str]],
) -> bool:
    """出题池：命中所选身份之一，或为通用基线题（不含其它身份专属）。"""
    if not selected_role_keywords:
        return True
    if _question_role_hits_extended(question, selected_role_keywords):
        return True
    if _is_audit_checkpoint_text(_question_search_text(question)):
        return False
    qt = _safe_question_type(str(question.get("question_type") or ""))
    if qt == "case_analysis":
        off = _question_hits_unselected_focus_roles(question, selected_role_keywords)
        if off:
            return False
    return _question_is_universal_common(question)


def _question_matches_role_keywords(
    question: Dict[str, Any],
    role_keywords: Dict[str, List[str]],
    *,
    strict: bool = False,
) -> bool:
    if _question_eligible_for_selected_roles(question, role_keywords):
        return True
    if strict:
        return False
    # 项目经理/研发经理：低权重纳入其它重点身份题，帮助覆盖全局内容。
    return _question_is_leadership_cross_candidate(question, role_keywords)


def _build_role_coverage_summary(questions: List[Dict[str, Any]], role_keywords: Dict[str, List[str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {k: 0 for k in role_keywords.keys()}
    all_scope = _all_author_role_keyword_scope()
    common = 0
    for q in questions or []:
        hits = _question_role_hits_extended(q, role_keywords)
        if hits:
            for role in hits:
                out[role] = int(out.get(role, 0)) + 1
        if _question_counts_as_common_baseline(q, role_keywords, all_scope):
            common += 1
    out[COMMON_AUTHOR_ROLE_KEY] = common
    return out


def _allocate_role_and_common_questions(
    questions: List[Dict[str, Any]],
    role_keywords: Dict[str, List[str]],
    question_count: int,
) -> List[Dict[str, Any]]:
    """balanced_union：保证各所选身份 + 通用基线题均有机会进入最终题量。"""
    if not questions or not role_keywords:
        return questions[:question_count]
    all_scope = _all_author_role_keyword_scope()
    leadership_mode = _selection_has_leadership_cross_need(role_keywords)
    eligible = [q for q in questions if _question_matches_role_keywords(q, role_keywords, strict=False)]
    if not eligible:
        return questions[:question_count]

    by_role_preferred: Dict[str, List[Dict[str, Any]]] = {r: [] for r in role_keywords.keys()}
    by_role_relaxed: Dict[str, List[Dict[str, Any]]] = {r: [] for r in role_keywords.keys()}
    common_preferred: List[Dict[str, Any]] = []
    common_relaxed: List[Dict[str, Any]] = []
    cross_preferred: List[Dict[str, Any]] = []
    cross_relaxed: List[Dict[str, Any]] = []
    for q in eligible:
        hits = _question_role_hits_extended(q, role_keywords)
        is_pref = _question_focus_primary_for_selection(q, role_keywords)
        if hits:
            for r in hits:
                if is_pref:
                    by_role_preferred.setdefault(r, []).append(q)
                else:
                    by_role_relaxed.setdefault(r, []).append(q)
        elif _question_counts_as_common_baseline(q, role_keywords, all_scope):
            if is_pref:
                common_preferred.append(q)
            else:
                common_relaxed.append(q)
        elif leadership_mode and _question_is_leadership_cross_candidate(q, role_keywords):
            if is_pref:
                cross_preferred.append(q)
            else:
                cross_relaxed.append(q)

    picked: List[Dict[str, Any]] = []
    used_ids: set[int] = set()

    def _take(q: Dict[str, Any]) -> bool:
        qid = int(q.get("id") or 0)
        if not qid or qid in used_ids:
            return False
        used_ids.add(qid)
        picked.append(q)
        return True

    roles = list(role_keywords.keys())
    min_role_hits = 2 if question_count >= 18 else 1
    max_common = max(1, question_count // 6)
    if len(roles) == 1:
        max_common = max(0, min(max_common, max(1, question_count // 12)))
    min_reg_hits = 1 if question_count >= 8 else 0
    for role in roles:
        role_preferred = by_role_preferred.get(role, [])
        role_relaxed = by_role_relaxed.get(role, [])
        must_hits_min = 0
        if role in _ROLE_MUST_HINTS:
            must_hits_min = 2 if (len(roles) == 1 and question_count >= 18) else 1
        if must_hits_min > 0:
            must_pool = [q for q in (role_preferred + role_relaxed) if _question_hits_role_must_hints(q, role)]
            for q in must_pool:
                if sum(1 for p in picked if _question_hits_role_must_hints(p, role)) >= must_hits_min:
                    break
                _take(q)
        reg_pool = [q for q in role_preferred if _question_hits_role_regulatory(q, role, role_keywords)]
        if not reg_pool:
            reg_pool = [q for q in role_relaxed if _question_hits_role_regulatory(q, role, role_keywords)]
        for q in reg_pool:
            if sum(1 for p in picked if _question_hits_role_regulatory(p, role, role_keywords)) >= min_reg_hits:
                break
            _take(q)
        base_pool = role_preferred + ([] if role == "prod" else role_relaxed)
        for q in base_pool:
            if sum(1 for p in picked if role in _question_role_hits_extended(p, role_keywords)) >= min_role_hits:
                break
            _take(q)

    for q in (common_preferred + common_relaxed):
        if sum(1 for p in picked if _question_counts_as_common_baseline(p, role_keywords, all_scope)) >= max_common:
            break
        _take(q)

    source_pool = (
        [q for r in roles for q in by_role_preferred.get(r, [])]
        + common_preferred
        + common_relaxed
        + [q for r in roles for q in by_role_relaxed.get(r, [])]
    )
    for q in source_pool:
        if len(picked) >= question_count:
            break
        _take(q)

    if leadership_mode and len(picked) < question_count:
        cross_pool = cross_preferred + cross_relaxed
        cross_cap = max(1, question_count // 5)
        cross_ids = {int(x.get("id") or 0) for x in cross_pool if int(x.get("id") or 0) > 0}
        for q in cross_pool:
            if len(picked) >= question_count:
                break
            picked_cross = sum(1 for p in picked if int(p.get("id") or 0) in cross_ids)
            enough_primary = len(picked) >= max(1, question_count - cross_cap)
            if picked_cross >= cross_cap and enough_primary:
                continue
            _take(q)

    return picked[:question_count]


def _promote_role_coverage_questions(
    questions: List[Dict[str, Any]],
    role_keywords: Dict[str, List[str]],
    question_count: int = 0,
) -> List[Dict[str, Any]]:
    """balanced_union 走分配器；否则将各身份命中题前置。"""
    if question_count > 0:
        return _allocate_role_and_common_questions(questions, role_keywords, question_count)
    if not questions or not role_keywords:
        return questions
    picked: List[Dict[str, Any]] = []
    used_idx: set[int] = set()
    for role in role_keywords.keys():
        for idx, q in enumerate(questions):
            if idx in used_idx:
                continue
            if role in _question_role_hits(q, role_keywords):
                picked.append(q)
                used_idx.add(idx)
                break
    for idx, q in enumerate(questions):
        if idx not in used_idx:
            picked.append(q)
    return picked


def _author_role_label_map() -> Dict[str, str]:
    try:
        from src.core.draft_integration_ui_meta import DRAFT_AUTHOR_ROLE_KEYS, DRAFT_AUTHOR_ROLE_LABELS

        return {
            str(k).strip().lower(): str(lbl).strip()
            for k, lbl in zip(DRAFT_AUTHOR_ROLE_KEYS, DRAFT_AUTHOR_ROLE_LABELS)
            if str(k).strip()
        }
    except Exception:
        return {}


def _attach_author_roles_to_set_items(
    items: List[Dict[str, Any]],
    role_keywords: Optional[Dict[str, List[str]]] = None,
    *,
    strict_role_filter: bool = False,
) -> None:
    """为套题内每题标注命中的身份（供前端展示「适用于 xx 身份」）。

    - 传入所选 role_keywords 时：仅按所选身份标注。
    - 不传/为空时：按**全部身份**标注（老师不勾身份也让每题关联到岗位，可关联多个）。
    - 通用题（不命中任何身份）默认标签为空（由 role_focus_config.COMMON_ROLE_LABEL_EMPTY 控制）。
    """
    del strict_role_filter
    if not items:
        return
    scope = role_keywords or _all_author_role_keyword_scope()
    labels = _author_role_label_map()
    labels[COMMON_AUTHOR_ROLE_KEY] = COMMON_AUTHOR_ROLE_LABEL
    all_scope = _all_author_role_keyword_scope()
    for it in items:
        if not isinstance(it, dict):
            continue
        q = it.get("question") if isinstance(it.get("question"), dict) else it
        hits = sorted(_question_role_hits_extended(q, scope))
        if hits:
            role_labels = [labels.get(h) or h for h in hits]
        elif not _COMMON_ROLE_LABEL_EMPTY and _question_is_universal_common(q, all_scope):
            hits = [COMMON_AUTHOR_ROLE_KEY]
            role_labels = [COMMON_AUTHOR_ROLE_LABEL]
        else:
            role_labels = []
            hits = []
        it["author_role_hits"] = hits
        it["author_role_labels"] = role_labels
        if q is not it:
            q["author_role_hits"] = hits
            q["author_role_labels"] = role_labels


def _make_scope_hash(
    exam_track: str,
    category: str,
    difficulty: str,
    question_type: str,
    exam_category: str = "daily",
    project_case_id: Optional[int] = None,
) -> str:
    ec = _normalize_exam_category(exam_category)
    pc = ""
    if ec == "project_case" and project_case_id is not None:
        pc = f"|pcid={int(project_case_id)}"
    return _hash_text(exam_track, category, difficulty, question_type, ec, pc)[:32]


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


def _ingest_knowledge_scope_plan(
    target_count: int,
    weights: Optional[List[float]] = None,
) -> List[tuple[str, str, int]]:
    """AI 录题：知识来源占比（默认 项目案例 30%、审核点 30%、法规标准 20%、程序文件 20%）。"""
    n = max(1, int(target_count))
    keys = ["project_case", "audit_checkpoint", "regulation", "program"]
    labels = ["项目案例", "审核点", "法规标准", "程序文件"]
    wdef = [0.3, 0.3, 0.2, 0.2]
    w = list(weights) if weights is not None and len(weights) == 4 else list(wdef)
    w = [max(0.0, float(x)) for x in w]
    s = sum(w)
    if s <= 0:
        w = list(wdef)
        s = sum(w)
    w = [x / s for x in w]
    counts = _split_int_by_weights(n, w)
    return [(keys[i], labels[i], counts[i]) for i in range(len(keys)) if counts[i] > 0]


def _ingest_question_type_plan(
    segment_count: int,
    weights: Optional[List[float]] = None,
) -> List[tuple[str, int]]:
    """单段录题内题型占比（默认 单选 30%、多选 10%、判断 10%、主观案例分析 50%）。"""
    c = max(0, int(segment_count))
    if c <= 0:
        return []
    types = ["single_choice", "multiple_choice", "true_false", "case_analysis"]
    wdef = [0.3, 0.1, 0.1, 0.5]
    w = list(weights) if weights is not None and len(weights) == 4 else list(wdef)
    w = [max(0.0, float(x)) for x in w]
    s = sum(w)
    if s <= 0:
        w = list(wdef)
        s = sum(w)
    w = [x / s for x in w]
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


def _regulation_open_evidence_via_llm(exam_track: str, need: int, exam_category: str = "daily") -> List[Dict[str, Any]]:
    """本地 regulation 类向量不足时，用模型生成通用监管要点摘录；不冒充具体上传文件或条款号。"""
    need = max(1, min(int(need), 20))
    track_name = EXAM_TRACKS.get(exam_track, exam_track)
    track_hint = TRACK_HINTS.get(exam_track, "")
    ec = _normalize_exam_category(exam_category)
    scene = EXAM_CATEGORIES.get(ec, ec)
    focus_extra = ""
    if ec == "new_standard":
        focus_extra = (
            "\n本次为「新标发布」导向：请围绕**可能的新版/修订/替代关系、实施过渡期、对软件生命周期与注册变更的影响方向**组织素材，"
            "强调「需以主管部门/标准化组织正式发布为准」；仍**禁止**编造具体文号、条款号、实施日期。"
        )
    prompt = f"""
你是医疗器械法规教研助手。用户本地向量库可能**未导入法规原文**，需要为考试命题准备若干条**一般性、可公开核对方向的表述**作为素材摘录（不等同于引用具体成文法条）。

体考类型：{track_name}（{track_hint}）
考试类型：{scene}{focus_extra}

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


def _extract_evidence_scoped(
    agent: ReviewAgent,
    exam_track: str,
    scope_key: str,
    top_k: int,
    exam_category: str = "daily",
    project_case_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """按知识来源维度检索命题素材（项目案例 / 审核点 / 法规标准 / 程序文件）。

    法规标准：优先主库 category=regulation；若不足或未训练，则用大模型生成「通用法规要点」摘录补足（source_file
    固定为「通用法规知识（大模型摘要·非用户向量库）」，与真实上传文件区分）。
    """
    ec = _normalize_exam_category(exam_category)
    rev_q = " 修订 新版本 替代 废止 过渡期 实施日期 专标 指南 变化点" if ec == "new_standard" else ""
    base_q = f"{EXAM_TRACKS.get(exam_track, exam_track)} {TRACK_HINTS.get(exam_track, '')} 典型考点 命题依据{rev_q}"
    tk = max(1, int(top_k))

    if scope_key == "regulation":
        docs: List[Any] = []
        try:
            docs = agent.kb.search_by_category(base_q + " 法规 标准 技术要求 指南", "regulation", top_k=tk)
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
            evidence.extend(_regulation_open_evidence_via_llm(exam_track, tk - len(evidence), exam_category))
        return evidence[:tk]

    docs: List[Any] = []
    try:
        tail = " 对照表" if ec == "new_standard" else ""
        if scope_key == "audit_checkpoint":
            docs = agent.checkpoint_kb.search(base_q + tail + " 审核点 检查表 符合性", top_k=tk)
        elif scope_key == "project_case":
            # 与组卷/练习补题共用同一取证实现，避免两处检索词或过滤逻辑漂移
            cid = int(project_case_id) if project_case_id is not None else None
            if cid is not None:
                return _extract_evidence_project_case(
                    agent, exam_track, top_k=tk, exam_category=ec, project_case_id=cid
                )
            docs = []
        elif scope_key == "program":
            docs = agent.kb.search_by_category(base_q + tail + " 程序文件 SOP 规程", "program", top_k=tk)
        else:
            docs = agent.kb.search(base_q + tail, top_k=tk)
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


def _trim_embedded_evidence_from_stem(stem: str, evidence: List[Any]) -> str:
    """若题干中粘入了与 evidence[].content_snippet 相同的大段原文（任意题型），截断到材料出现之前。

    材料应放在 `evidence_json` 的摘录字段中，**不是**选项或判断句本身；此处防止模型/历史数据把摘录糊进 `stem`。
    若截断后会导致判断题缺少「待判断陈述」，则保留原题干，改由题干补全逻辑处理。
    """
    s = str(stem or "").strip()
    if not s or not isinstance(evidence, list):
        return s
    cut_at: Optional[int] = None
    for ev in evidence:
        if not isinstance(ev, dict):
            continue
        sn = str(ev.get("content_snippet") or ev.get("content") or "").strip()
        if len(sn) < 80:
            continue
        needles: List[str] = [sn]
        if len(sn) > 200:
            needles.append(sn[:360])
        if len(sn) > 130:
            needles.append(sn[:220])
        if len(sn) > 95:
            needles.append(sn[:120])
        for needle in needles:
            if len(needle) < 72:
                continue
            pos = s.find(needle)
            # pos=0 表示整段 stem 即材料，不截断；否则去掉从首次命中起的重复粘贴
            if pos > 0:
                if cut_at is None or pos < cut_at:
                    cut_at = pos
                break
    if cut_at is not None:
        out = s[:cut_at].rstrip()
        tail = s[cut_at:].strip()
        if len(tail) >= 80 and len(out) >= 6:
            if re.search(r"判断下列(?:陈述|说法)是否正确：\s*$", out):
                return s
            return out
    return s


_STUDENT_STEM_META_LITERALS: tuple[str, ...] = (
    "（具体摘录在本题 evidence 中，勿在题干中整段复述。）",
    "（摘录见本题 evidence。）",
    "（更长原文见本题 evidence。）",
    "（背景摘录在本题 evidence 中；勿把整段原文当作题干复述抄写。）",
    "（完整上下文见本题 evidence）",
)


def _strip_student_stem_meta(stem: str) -> str:
    """去掉仅面向录题/模型的 evidence 提示语，避免泄露到学生端。"""
    s = str(stem or "").strip()
    if not s:
        return s
    for lit in _STUDENT_STEM_META_LITERALS:
        s = s.replace(lit, "")
    s = re.sub(r"（[^）]{0,160}evidence[^）]{0,160}）", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()


def _format_evidence_excerpt_for_student(evidence: List[Any], *, max_chars: int = 900) -> str:
    """内部用：从 evidence 取材料文本，不直接下发给学生。"""
    parts: List[str] = []
    remain = max(120, int(max_chars))
    for ev in evidence or []:
        if not isinstance(ev, dict):
            continue
        sn = str(ev.get("content_snippet") or ev.get("content") or "").strip()
        if not sn:
            continue
        src = str(ev.get("source_file") or ev.get("file_name") or "").strip()
        chunk = sn[:remain]
        if len(sn) > remain:
            chunk += "…"
        if src:
            parts.append(f"【{src}】\n{chunk}")
        else:
            parts.append(chunk)
        remain = max(0, remain - len(chunk))
        if remain <= 0:
            break
    return "\n\n".join(parts).strip()


def _first_claim_from_excerpt(excerpt: str, *, max_len: int = 80) -> str:
    txt = str(excerpt or "").strip()
    if not txt:
        return ""
    txt = re.sub(r"^【[^】]+】\s*", "", txt)
    txt = re.sub(r"\s+", " ", txt)
    for seg in re.split(r"[。\n；;]", txt):
        seg = seg.strip(" ，,、")
        if not seg:
            continue
        sub_parts = [x.strip(" ，,、") for x in re.split(r"[，,、]", seg) if x.strip(" ，,、")]
        for part in sub_parts:
            if len(part) < 10:
                continue
            return part[:max_len] + ("…" if len(part) > max_len else "")
        if len(seg) >= 10:
            return seg[:max_len] + ("…" if len(seg) > max_len else "")
    return txt[:max_len] + ("…" if len(txt) > max_len else "")


def _summarize_statement_text(text: str, *, max_sentences: int = 2, max_chars: int = 100) -> str:
    raw = re.sub(r"\s+", " ", str(text or "")).strip()
    if not raw:
        return ""
    chunks = [x.strip(" ，,、；;。") for x in re.split(r"[。；;！？!?]", raw) if x.strip(" ，,、；;。")]
    if not chunks:
        chunks = [x.strip(" ，,、；;。") for x in re.split(r"[，,、]", raw) if x.strip(" ，,、；;。")]
    short = []
    for seg in chunks:
        if not seg:
            continue
        short.append(seg[: max_chars // max_sentences] if len(seg) > (max_chars // max_sentences) else seg)
        if len(short) >= max_sentences:
            break
    if not short:
        short = [raw[:max_chars]]
    out = "；".join(short)
    return out[:max_chars] + ("…" if len(out) > max_chars else "")


def _collect_open_book_refs(item: Dict[str, Any]) -> List[Dict[str, str]]:
    refs: List[Dict[str, str]] = []
    seen: set[str] = set()
    stem = str(item.get("stem") or "")
    for sf in _question_source_files(item):
        if not sf or sf in seen:
            continue
        if "审核点清单" in sf or sf.startswith("审核点"):
            seen.add(sf)
            refs.append({"source_file": sf, "kind": "audit_checklist", "title": sf})
    for m in re.finditer(r"《([^》]{4,260})》", stem):
        fn = str(m.group(1) or "").strip()
        if fn and fn not in seen:
            seen.add(fn)
            refs.append({"source_file": fn, "kind": "document", "title": fn})
    for m in re.finditer(r"(审核点清单-[\w.\-]+)", stem):
        fn = str(m.group(1) or "").strip()
        if fn and fn not in seen:
            seen.add(fn)
            refs.append({"source_file": fn, "kind": "audit_checklist", "title": fn})
    return refs


def _item_evidence_for_display(item: Dict[str, Any]) -> List[Any]:
    """从题目项或其嵌套 question 中取出 evidence 列表（含 evidence_json）。"""
    for holder in (item, item.get("question") if isinstance(item.get("question"), dict) else None):
        if not isinstance(holder, dict):
            continue
        ev = holder.get("evidence")
        if isinstance(ev, list) and ev:
            return ev
        ej = holder.get("evidence_json")
        if ej:
            try:
                parsed = json.loads(ej) if isinstance(ej, str) else ej
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
    return []


def _question_has_audit_checkpoint_context(
    item: Dict[str, Any],
    stem: str,
    evidence: Optional[List[Any]] = None,
) -> bool:
    """题干/evidence/已缓存 stem_full 任一含审核点结构即视为审核点判断题。"""
    if _is_audit_checkpoint_text(stem):
        return True
    existing = str(item.get("stem_full") or item.get("stemFull") or "").strip()
    if _is_audit_checkpoint_text(existing):
        return True
    if not isinstance(evidence, list):
        return False
    for ev in evidence:
        if not isinstance(ev, dict):
            continue
        sn = str(ev.get("content_snippet") or ev.get("content") or "").strip()
        if _is_audit_checkpoint_text(sn):
            return True
    return False


def _audit_checkpoint_raw_stem_for_format(
    item: Dict[str, Any],
    stem: str,
    evidence: Optional[List[Any]] = None,
) -> str:
    """取含完整字段的审核点原文（优先 evidence，其次 stem_full，最后当前 stem）。"""
    if _is_audit_checkpoint_text(stem):
        return stem
    if isinstance(evidence, list):
        best = ""
        for ev in evidence:
            if not isinstance(ev, dict):
                continue
            sn = str(ev.get("content_snippet") or ev.get("content") or "").strip()
            if _is_audit_checkpoint_text(sn) and len(sn) > len(best):
                best = sn
        if best:
            return best
    cached = str(item.get("stem_full") or item.get("stemFull") or "").strip()
    if _is_audit_checkpoint_text(cached):
        return cached
    return stem


def _strip_broken_open_book_html(stem: str) -> str:
    """清理题干中误写入的前端开卷链接 HTML 片段（历史脏数据）。"""
    s = str(stem or "")
    s = re.sub(r'" title="开卷查阅：点击展开全文">', "", s)
    s = re.sub(r'\s*data-open-book-file="[^"]*"', "", s, flags=re.I)
    s = re.sub(r'\s*class="[^"]*exam-open-book-link[^"]*"', "", s, flags=re.I)
    s = re.sub(
        r'<button[^>]*exam-open-book-link[^>]*>(.*?)</button>',
        lambda m: re.sub(r"</?[^>]+>", "", m.group(1)).strip("《》"),
        s,
        flags=re.I | re.S,
    )
    s = re.sub(r"</?button[^>]*>", "", s, flags=re.I)
    return s.strip()


def _sanitize_question_stem_for_display(item: Dict[str, Any]) -> None:
    """学生端展示：仅保留/补全题干，不单独输出 reference_excerpt。"""
    ev = _item_evidence_for_display(item)
    qt = _safe_question_type(str(item.get("question_type") or "single_choice"))
    stem = _strip_broken_open_book_html(str(item.get("stem") or "").strip())
    if not stem and isinstance(item.get("question"), dict):
        stem = _strip_broken_open_book_html(str(item["question"].get("stem") or "").strip())
    is_audit = _is_audit_checkpoint_text(stem)
    if not is_audit:
        stem = _trim_embedded_evidence_from_stem(stem, ev)
    stem = _strip_student_stem_meta(stem)
    excerpt = _format_evidence_excerpt_for_student(ev)
    if qt == "true_false" and _question_has_audit_checkpoint_context(item, stem, ev):
        raw_stem = _audit_checkpoint_raw_stem_for_format(item, stem, ev)
        item["stem"] = _format_audit_checkpoint_true_false_stem(raw_stem, ev)
        item.pop("stem_full", None)
        item.pop("stemFull", None)
        item.pop("stem_has_full", None)
        item.pop("stemHasFull", None)
        item.pop("reference_excerpt", None)
        item.pop("referenceExcerpt", None)
        return
    if qt == "true_false":
        m = re.search(r"(判断下列(?:陈述|说法)是否正确：)(.*)$", stem, flags=re.S)
        if m:
            prefix = m.group(1).strip()
            tail_raw = m.group(2).strip()
            if _question_has_audit_checkpoint_context(item, tail_raw, ev):
                raw_stem = _audit_checkpoint_raw_stem_for_format(item, tail_raw, ev)
                item["stem"] = _format_audit_checkpoint_true_false_stem(raw_stem, ev)
                item.pop("stem_full", None)
                item.pop("stemFull", None)
                item.pop("stem_has_full", None)
                item.pop("stemHasFull", None)
            else:
                tail = _summarize_statement_text(tail_raw, max_sentences=3, max_chars=420)
                if not tail:
                    tail = _first_claim_from_excerpt(excerpt, max_len=320)
                if not tail:
                    tail = "（题目材料缺失，请联系管理员补充题库。）"
                stem = prefix + "\n" + tail
                item["stem"] = stem.strip()
                item.pop("stem_full", None)
                item.pop("stemFull", None)
    if qt in ("single_choice", "multiple_choice") and stem and not re.search(r"[？?]\s*$", stem):
        if re.search(r"(下列哪项|下列哪些|最恰当|哪些说法)", stem):
            stem = stem.rstrip("。．.") + "？"
    if "stem" not in item or not str(item.get("stem") or "").strip():
        item["stem"] = stem.strip()
    item.pop("reference_excerpt", None)
    item.pop("referenceExcerpt", None)


def _redact_student_quiz_item(item: Dict[str, Any]) -> None:
    """练习/考试下发：仅题干+选项，不含 evidence/答案/解析；保留开卷查阅引用。"""
    if not isinstance(item, dict):
        return
    refs = _collect_open_book_refs(item)
    qobj = item.get("question") if isinstance(item.get("question"), dict) else None
    if qobj is not None:
        refs.extend(_collect_open_book_refs(qobj))
    if refs:
        dedup: List[Dict[str, str]] = []
        seen: set[str] = set()
        for r in refs:
            sf = str(r.get("source_file") or "").strip()
            if not sf or sf in seen:
                continue
            seen.add(sf)
            dedup.append(r)
        item["open_book_refs"] = dedup
        item["openBookRefs"] = dedup
    _sanitize_question_stem_for_display(item)
    if qobj is not None:
        _sanitize_question_stem_for_display(qobj)
        if refs:
            qobj["open_book_refs"] = item.get("open_book_refs")
            qobj["openBookRefs"] = item.get("openBookRefs")
    for key in (
        "evidence",
        "evidence_json",
        "reference_excerpt",
        "referenceExcerpt",
        "explanation",
        "answer",
        "answer_json",
        "stem_full",
        "stemFull",
        "stem_has_full",
        "stemHasFull",
    ):
        item.pop(key, None)
        if qobj is not None:
            qobj.pop(key, None)


def _prepare_student_facing_question(item: Dict[str, Any]) -> None:
    """兼容旧调用：等同 _sanitize_question_stem_for_display。"""
    _sanitize_question_stem_for_display(item)


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
    ev_out = question.get("evidence") if isinstance(question.get("evidence"), list) else []
    stem_out = str(question.get("stem") or "").strip()
    stem_out = _trim_embedded_evidence_from_stem(stem_out, ev_out)
    return {
        "question_type": q_type,
        "stem": stem_out,
        "options": opts,
        "answer": answer,
        "explanation": str(question.get("explanation") or "").strip(),
        "category": str(question.get("category") or fallback_category or "").strip(),
        "difficulty": _safe_difficulty(str(question.get("difficulty") or "medium")),
        "evidence": ev_out,
    }


def _extract_evidence(agent: ReviewAgent, exam_track: str, top_k: int = 8, exam_category: str = "daily") -> List[Dict[str, Any]]:
    ec = _normalize_exam_category(exam_category)
    tail = " 修订 新版本 替代 废止 过渡期 专标 指南 变化点" if ec == "new_standard" else ""
    query = f"{EXAM_TRACKS.get(exam_track, exam_track)} {TRACK_HINTS.get(exam_track, '')} 典型考题要点{tail}"
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


def _role_focus_meta(role_keywords: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """为所选身份汇总命题聚焦信息：标签、必考主题、必考关键词、必考文件、命题侧重。"""
    label_map = _author_role_label_map()
    out: List[Dict[str, Any]] = []
    for role in (role_keywords or {}).keys():
        r = str(role).strip().lower()
        if not r:
            continue
        must = sorted(_ROLE_MUST_HINTS.get(r) or set())
        topics = list(_ROLE_FOCUS_TOPICS.get(r) or [])
        kws = [str(x).strip() for x in (_AUTHOR_ROLE_FILE_KEYWORDS.get(r) or []) if str(x).strip()]
        out.append(
            {
                "role": r,
                "label": str(label_map.get(r) or r),
                "must_hints": must,
                "topics": topics,
                "keywords": kws,
                "must_files": list(_ROLE_MUST_FILES.get(r) or []),
                "emphasis": str(_ROLE_PROMPT_EMPHASIS.get(r) or "").strip(),
            }
        )
    return out


def _role_focus_prompt_block(role_focus: Optional[List[Dict[str, Any]]]) -> str:
    """把岗位聚焦信息拼成命题提示块，指导 AI 覆盖各岗位必考主题与命题侧重。"""
    if not role_focus:
        return ""
    lines: List[str] = [
        "\n【本套题按岗位身份定向命题】所选身份及其必考主题/必考文件/命题侧重如下，"
        "请让题目**尽量均衡覆盖各身份的必考主题**，每个身份至少 1～2 题落到其必考主题上；"
        "题目要能关联到对应身份（可同时关联多个身份）；仍须严格依据 evidence，不得脱离摘录编造。",
    ]
    for rf in role_focus:
        topics = "；".join(rf.get("topics") or []) or "（按该岗位职责命题）"
        must = "、".join(rf.get("must_hints") or [])
        files = "、".join(rf.get("must_files") or [])
        emphasis = str(rf.get("emphasis") or "").strip()
        seg = f"- {rf.get('label')}：{topics}"
        if files:
            seg += f"；必考文件：{files}"
        if must:
            seg += f"；关键词：{must}"
        if emphasis:
            seg += f"；命题侧重：{emphasis}"
        lines.append(seg)
    return "\n".join(lines)


def _default_exam_weighted_scope(top_n: int = 5) -> Dict[str, List[str]]:
    """老师端不勾身份时：按体考关注度权重取前若干重点岗位，构造定向取材用的 role scope。"""
    weighted = [(r, w) for r, w in (_ROLE_EXAM_WEIGHT or {}).items() if w > 0]
    if not weighted:
        weighted = [(r, 1.0) for r in _FOCUS_EXAM_ROLES]
    weighted.sort(key=lambda x: (-x[1], x[0]))
    picked = [r for r, _w in weighted[: max(1, top_n)]]
    return _role_file_keyword_scope(picked)


def _default_exam_role_focus(top_n: int = 5) -> List[Dict[str, Any]]:
    return _role_focus_meta(_default_exam_weighted_scope(top_n=top_n))


def _extract_evidence_for_roles(
    agent: ReviewAgent,
    exam_track: str,
    top_k: int,
    exam_category: str,
    role_keywords: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """按所选身份**逐一定向检索**命题素材，保证各岗位必考主题（如配置管理计划/配置状态报告）被覆盖。

    策略：对每个所选身份，先用「必考主题 + 必考关键词」做定向检索，再用岗位强关键词补充；
    命中该岗位关键词的摘录打上 `_role` 标签，最后按岗位轮询取材，避免某一身份被淹没。
    """
    tk = max(1, int(top_k))
    ec = _normalize_exam_category(exam_category)
    tail = " 修订 新版本 替代 废止 过渡期 专标 指南 变化点" if ec == "new_standard" else ""
    track_head = f"{EXAM_TRACKS.get(exam_track, exam_track)} {TRACK_HINTS.get(exam_track, '')}".strip()

    role_focus = _role_focus_meta(role_keywords or {})
    per_role_evidence: Dict[str, List[Dict[str, Any]]] = {}
    seen_snips: set[str] = set()

    def _keep_from_docs(rows_or_docs: List[Any], role: str, role_kws: List[str], target: int) -> None:
        bucket = per_role_evidence.setdefault(role, [])
        for r in rows_or_docs or []:
            if len(bucket) >= target:
                break
            content = str(r.get("content") or "").strip()
            if not content:
                continue
            meta = r.get("metadata") or {}
            source_file = (
                str(meta.get("source_file") or "").strip()
                or str(meta.get("title") or "").strip()
                or str(r.get("source") or "").strip()
            )
            blob = f"{source_file} {content}"
            if role_kws and not _text_hits_any_keyword(blob, role_kws):
                continue
            key = content[:120]
            if key in seen_snips:
                continue
            seen_snips.add(key)
            bucket.append({"content_snippet": content[:700], "source_file": source_file, "_role": role})

    # 每个岗位期望取材量（至少 2，或按题量均分）
    per_role_target = max(2, (tk + max(1, len(role_focus)) - 1) // max(1, len(role_focus)))
    for rf in role_focus:
        role = rf["role"]
        role_kws = [str(x).strip().lower() for x in (rf.get("keywords") or []) if str(x).strip()]
        must = list(rf.get("must_hints") or [])
        topics = list(rf.get("topics") or [])
        # 1) 必考主题 / 必考关键词定向检索（保证 cm 的配置管理计划、配置状态报告等被命中）
        focus_terms = must + topics
        queries: List[str] = []
        if focus_terms:
            queries.append(f"{track_head} {' '.join(focus_terms[:8])} 核查 要点{tail}".strip())
        # 2) 岗位强关键词补充
        if role_kws:
            queries.append(f"{track_head} {' '.join(role_kws[:10])} GMP 法规 核查 指南 典型考题{tail}".strip())
        for q in queries:
            if len(per_role_evidence.get(role, [])) >= per_role_target:
                break
            try:
                rows = agent.search_knowledge(q, top_k=max(per_role_target * 3, 10), use_checkpoints=True)
            except Exception:
                rows = []
            _keep_from_docs(rows, role, role_kws, per_role_target)

    # 按岗位轮询汇总，均衡覆盖
    evidence: List[Dict[str, Any]] = []
    idx = 0
    while len(evidence) < tk:
        progressed = False
        for rf in role_focus:
            bucket = per_role_evidence.get(rf["role"], [])
            if idx < len(bucket):
                evidence.append(bucket[idx])
                progressed = True
                if len(evidence) >= tk:
                    break
        if not progressed:
            break
        idx += 1

    if evidence:
        return evidence[:tk]
    return _extract_evidence(agent, exam_track, top_k=tk, exam_category=exam_category)


def _extract_evidence_project_case(
    agent: ReviewAgent,
    exam_track: str,
    top_k: int,
    exam_category: str,
    project_case_id: int,
    role_keywords: Optional[Dict[str, List[str]]] = None,
) -> List[Dict[str, Any]]:
    """项目案例考试：仅从所选案例的 project_case 向量取证（不混入法规 open LLM 摘要）。"""
    ec = _normalize_exam_category(exam_category)
    rev_q = " 修订 新版本 替代 废止 过渡期 实施日期 专标 指南 变化点" if ec == "new_standard" else ""
    base_q = f"{EXAM_TRACKS.get(exam_track, exam_track)} {TRACK_HINTS.get(exam_track, '')} 典型考点 命题依据{rev_q}"
    tail = " 对照表" if ec == "new_standard" else ""
    q = (
        base_q
        + tail
        + " 项目案例 注册资料 设计开发 生产 记录 偏差 变更 风险管理 现场核查 GMP 软件生命周期"
    )
    tk = max(1, int(top_k))
    docs: List[Any] = []
    try:
        docs = agent.kb.search_by_category(q, "project_case", top_k=tk, case_id=int(project_case_id))
    except Exception:
        docs = []
    role_keywords = role_keywords or {}
    if role_keywords:
        all_scope = _all_author_role_keyword_scope()
        union_selected: List[str] = []
        union_all: List[str] = []
        for kws in role_keywords.values():
            union_selected.extend(kws or [])
        for kws in all_scope.values():
            union_all.extend(kws or [])
        union_selected = list(dict.fromkeys([x for x in union_selected if x]))
        union_all = list(dict.fromkeys([x for x in union_all if x]))
        if union_selected:
            kept: List[Any] = []
            for doc in docs:
                md = getattr(doc, "metadata", None) or {}
                blob = " ".join(
                    [
                        str(md.get("source_file") or ""),
                        str(getattr(doc, "page_content", "") or ""),
                    ]
                )
                if _text_hits_any_keyword(blob, union_selected):
                    kept.append(doc)
                elif union_all and not _text_hits_any_keyword(blob, union_all):
                    kept.append(doc)
            if kept:
                docs = kept
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
    return _rows_to_evidence(rows)


def _fallback_questions(
    exam_track: str,
    category: str,
    count: int,
    evidence: List[Dict[str, Any]],
    *,
    question_type: str = "true_false",
    difficulty: str = "medium",
    exam_category: str = "daily",
) -> List[Dict[str, Any]]:
    qt = _safe_question_type(question_type)
    diff = _safe_difficulty(difficulty)
    ec_fb = _normalize_exam_category(exam_category)
    out: List[Dict[str, Any]] = []
    for i in range(count):
        ev = evidence[i % len(evidence)] if evidence else {"content_snippet": "知识库命中不足", "source_file": ""}
        src = str(ev.get("source_file") or "").strip()
        src_text = f"《{src}》" if src else "（来源文件未标注）"
        snip = str(ev.get("content_snippet", "") or "")[:160]
        if qt == "single_choice":
            stem = f"[{EXAM_TRACKS.get(exam_track, exam_track)}] 依据 {src_text} 的入库摘录，下列哪项最恰当？"
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
            stem = f"[{EXAM_TRACKS.get(exam_track, exam_track)}] 依据 {src_text} 的入库摘录，下列哪些说法成立（多选）？"
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
            prefix = "【本案项目案例资料】 " if ec_fb == "project_case" else ""
            stem = (
                f"{prefix}[{EXAM_TRACKS.get(exam_track, exam_track)}] 案例分析：请仅结合 {src_text} 中与本案相关的入库摘录作答，"
                f"说明核查员可能追问的焦点及你方应如何举证、补证。"
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
            raw_snip = str(ev.get("content_snippet", "") or "").strip()
            short_claim = (raw_snip[:150] + ("…" if len(raw_snip) > 150 else "")).strip() or "（完整上下文见本题 evidence）"
            stem = f"[{EXAM_TRACKS.get(exam_track, exam_track)}] 依据 {src_text}，判断下列陈述是否正确：{short_claim}"
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
    exam_category: str = "daily",
    project_case_id: Optional[int] = None,
    role_focus: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    min_plausible = 0 if count <= 1 else max(1, (count * 4 + 9) // 10)
    ec = _normalize_exam_category(exam_category)
    scene_line = f"考试类型：{EXAM_CATEGORIES.get(ec, ec)}。"
    new_std_rules = ""
    if ec == "new_standard":
        new_std_rules = (
            "\n14) 本批为「新标发布」：题干与选项应引导考生识别**修订/替代/适用范围变化/对软件文档与验证的影响**等；"
            "仍不得编造具体文号与实施日期；explanation 中须提示「以官方发布文本为准」。"
        )
    project_case_anchor = ""
    if ec == "project_case" and project_case_id:
        from src.core.db import get_project_case

        prow = get_project_case(int(project_case_id)) or {}
        cn = str(prow.get("case_name") or "").strip()
        pn = str(prow.get("product_name") or "").strip()
        rc = str(prow.get("registration_country") or "").strip()
        rt = str(prow.get("registration_type") or "").strip()
        rcomp = str(prow.get("registration_component") or "").strip()
        pform = str(prow.get("project_form") or "").strip()
        project_case_anchor = (
            f"\n【本套题锁定的项目案例】case_id={int(project_case_id)}；"
            f"案例名：{cn or '—'}；产品：{pn or '—'}；注册国家/地区：{rc or '—'}；注册类别：{rt or '—'}；"
            f"结构组成：{rcomp or '—'}；项目形态：{pform or '—'}。"
            "命题时仅可将上述字段用于**与 evidence 摘录一致**的表述；不得编造上述字段未给出的注册号、版本、日期、批号等。"
        )
    project_case_rules = ""
    if ec == "project_case":
        project_case_rules = (
            "\n14) 本批为「项目案例」：**全部题型**（单选/多选/判断/案例分析）均须**严格依据 evidence 中本项目案例已入库资料摘录**命题，"
            "题干与选项/判断陈述中的事实、功能描述、记录名称、流程节点等应能在摘录中找到依据或合理概括，**禁止**写成与本案资料无关的泛化法规背诵题。"
            "\n15) 模拟核查口吻：你扮演**现场核查/体考考官**（面向研发、生产、质量体系人员），可结合 GMP、现场核查指南等**通用核查关注点**发问，"
            "但必须**落脚到 evidence 中的具体文档与摘录内容**；不得编造具体条款号、公告号、标准号、页码或 evidence 未出现的文件名。"
            "\n16) **案例分析题（case_analysis）硬性要求**："
            "题干必须呈现**基于摘录的具体情境或矛盾点**（例如记录缺失、版本不一致、验证范围与声称功能不符等），不得出「泛泛谈质量管理体系」且与摘录脱钩的题；"
            "`answer` 参考要点须按「摘录事实 → 核查风险/追问点 → 建议补充的证据或记录（对应 source_file）」组织，**至少 2 条**且每条能指回 evidence；"
            "禁止「教材式」或与本案无关的标准答案套话。"
            f"{project_case_anchor}"
        )
    role_focus_block = _role_focus_prompt_block(role_focus)
    prompt = f"""
你是医疗器械法规考试命题助手。请只输出 JSON，不要额外说明。

要求：
1) 生成 {count} 道题，体考类型：{EXAM_TRACKS.get(exam_track, exam_track)}（{TRACK_HINTS.get(exam_track, '')}）；{scene_line}
2) 题型：{question_type}
3) 难度：{difficulty}
4) 分类：{category or exam_track}
5) 必须依据 evidence，不得编造法规条款编号/章节号/文件名。
5a) **全部题型（单选/多选/判断/案例分析）**：`stem` **禁止**整段粘贴下方 evidence 中的原文摘录、表格转写或 OCR 残留标记；题干只用简短设问+必要概括；完整摘录**仅**写入该题 `evidence[].content_snippet`（与其它题型一致）。
6) 每题 explanation 必须明确写出“依据的来源文件名”，格式示例：依据：《文件名》；……。禁止出现“根据审核点XXX”这类不可定位表述。
7) evidence[].source_file 必须填写为具体可读的文件名（程序文件/法规文件/项目案例文件名）；如果无法确定，填空字符串，并在 explanation 中写“来源文件未标注”。不得使用 tmp 临时文件名。
8) JSON 格式：{{"questions":[{{"question_type":"single_choice|multiple_choice|true_false|case_analysis","stem":"...","options":[...],"answer":...,"explanation":"...","category":"...","difficulty":"easy|medium|hard","evidence":[{{"content_snippet":"...","source_file":"..."}}]}}]}}
9) 单选/多选：`options` 为**纯陈述文本数组**（2～6 项），**不要**在每项前加 `A./B.` 等字母前缀（系统会统一编号）。单选 `answer` 为单个大写字母（如 `C`）；多选 `answer` 为字母数组（如 `["A","D"]`）。
9.1) 多选题（multiple_choice）必须至少有 **2 个**正确选项（`answer` 数组长度 ≥ 2），不得只给一个正确项。
9.2) 案例分析题（case_analysis）必须满足：`options` 为空数组；`answer` 为**参考作答要点文本**（不是 A/B/C/D），学生端会用文本框输入。
9.2.1) 若当前为「项目案例」考试类型：案例分析题还须满足第 16) 条对情境与答案结构的全部要求。
9.2.2) **案例分析**在遵守第 5a) 条前提下：`stem` 只写**设问与简短情境**（建议 ≤ 500 汉字或等效），用 1～3 句概括矛盾点；**禁止**在 `stem` 中整段粘贴 `evidence[].content_snippet`。
9.2.3) 需要考生引用的具体事实，在题干中**概括**并指向文件名；**背景原文只放在**各题 `evidence[].content_snippet`；`answer` 为参考要点，**禁止**将 `answer` 全文写入 `stem`。
10) 干扰项质量（本批共 {count} 题）：其中至少 **{min_plausible}** 题须标为 `medium` 或 `hard`，且这些题的**错误选项**须与正确选项在句式长度、术语层级上**尽量平行**，呈现**易混淆结论**；**禁止**整批题都靠「仅…」「只需要…」「不需要…」「绝不是…」「与审核无关」等一眼可排除的标语式否定句凑满四个选项。
11) 约 **60%** 题目可为 `easy`：允许轻度否定或排除语气，但**不要**让四个错误项共用同一种开头模板；**不得**出现「明显三项都错、只剩一项像真命题」的凑数结构。
12) 正确答案在命制时不要刻意总落在同一字母位；落库前系统会打乱选项顺序并重写 `answer`，你仍须保证**每个错误选项本身像合理结论**而非「反着说就对了」的口号。
13) 同批各题题干须围绕不同角度设问，避免只改一两处用词、结构高度雷同的「换皮题」；系统还会与历史题库做相似度过滤，雷同过多会被丢弃。
{new_std_rules}{project_case_rules}{role_focus_block}

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
    max_similar_frac: float = 0.1,
) -> List[Dict[str, Any]]:
    """AI 录题：与历史题干近似重复的题控制在约 max_similar_frac（默认 10%）以内。"""
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
    role_keywords: Optional[Dict[str, List[str]]] = None,
    shuffle_seed: str = "",
) -> List[Dict[str, Any]]:
    cat_f = (category or "").strip() or None
    excl = list(exclude_question_ids or [])
    ex_set = {int(x) for x in excl if int(x) > 0}
    if count <= 0:
        return []
    max_rep = max(1, (max(1, int(question_count_total)) + 9) // 10)
    role_filter = bool(role_keywords)

    def _eligible(cand: Dict[str, Any]) -> bool:
        if not role_filter:
            return True
        return _question_eligible_for_selected_roles(cand, role_keywords or {})

    def _fetch_pool(*, limit: int, offset: int = 0) -> List[Dict[str, Any]]:
        return repo.list_bank_questions(
            collection=collection,
            exam_track=exam_track,
            knowledge_scope_hash=None,
            category=cat_f,
            difficulty=difficulty,
            question_type=question_type,
            limit=limit,
            offset=offset,
            exclude_question_ids=sorted(ex_set),
        )

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

    if role_filter:
        pool_limit = min(280, max(count * 8, 64))
        max_scan_rows = 900
        max_extra_pages = 4
    else:
        pool_limit = min(220, max(count * 14, 48))
        max_scan_rows = 0
        max_extra_pages = 8
    pool = _fetch_pool(limit=pool_limit, offset=0)
    rows_scanned = len(pool)
    if shuffle_seed:
        rng = random.Random(int(_hash_text(shuffle_seed, str(pool_limit), "0")[:12], 16))
        rng.shuffle(pool)
    else:
        random.shuffle(pool)
    out: List[Dict[str, Any]] = []

    def _try_append(cand: Dict[str, Any], *, enforce_sig: bool) -> bool:
        qid = int(cand.get("id") or 0)
        if not qid or qid in ex_set or not _eligible(cand):
            return False
        sig = _set_diversity_signature(cand)
        if enforce_sig and sig_counts.get(sig, 0) >= max_rep:
            return False
        out.append(cand)
        ex_set.add(qid)
        sig_counts[sig] = sig_counts.get(sig, 0) + 1
        return True

    for cand in pool:
        if len(out) >= count:
            break
        _try_append(cand, enforce_sig=True)

    if len(out) < count:
        for cand in pool:
            if len(out) >= count:
                break
            qid = int(cand.get("id") or 0)
            if not qid or qid in ex_set or not _eligible(cand):
                continue
            out.append(cand)
            ex_set.add(qid)
            sig = _set_diversity_signature(cand)
            sig_counts[sig] = sig_counts.get(sig, 0) + 1

    if len(out) < count:
        for cand in _narrow_pool():
            if len(out) >= count:
                break
            _try_append(cand, enforce_sig=True)

    if len(out) < count and role_filter:
        offset = len(pool)
        page_limit = 160
        for _page in range(max_extra_pages):
            if len(out) >= count:
                break
            if max_scan_rows > 0 and rows_scanned >= max_scan_rows:
                break
            batch = _fetch_pool(limit=page_limit, offset=offset)
            if not batch:
                break
            rows_scanned += len(batch)
            if shuffle_seed:
                rng.shuffle(batch)
            else:
                random.shuffle(batch)
            for cand in batch:
                if len(out) >= count:
                    break
                _try_append(cand, enforce_sig=True)
            offset += len(batch)
            if len(batch) < page_limit:
                break

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
    exam_category: str = "daily",
    project_case_id: Optional[Any] = None,
    author_roles: Optional[List[str]] = None,
    author_role_coverage: str = "balanced_union",
) -> Dict[str, Any]:
    """组卷：题型占比仅由 difficulty 决定；question_type 参数保留兼容，不参与组卷。"""
    if exam_track not in EXAM_TRACKS:
        raise ValueError(f"不支持的 exam_track: {exam_track}")
    ec = _normalize_exam_category(exam_category)
    pc_id = require_project_case_quiz(collection=collection, exam_category=exam_category, project_case_id=project_case_id)
    roles = _normalize_author_roles(author_roles or [])
    role_keywords = _role_file_keyword_scope(roles)
    strict_role_filter = bool(roles) and not _selection_has_leadership_cross_need(role_keywords)
    role_shuffle_key = ",".join(sorted(roles)) if roles else ""
    coverage_mode = "balanced_union" if str(author_role_coverage or "").strip().lower() == "balanced_union" else "union"
    question_count = max(1, int(question_count))
    difficulty = _safe_difficulty(difficulty)
    bank_cat = (category or "").strip() or None
    hash_cat = bank_cat or exam_track
    plan = _difficulty_question_type_plan(difficulty, question_count)
    mix_map = {qt: n for qt, n in plan}
    set_cfg = {
        "set_config_hash": _hash_text(
            exam_track,
            hash_cat or "",
            difficulty,
            json.dumps(mix_map, sort_keys=True),
            str(question_count),
            ec,
            str(pc_id or ""),
            ",".join(sorted(roles)),
        )[:32],
        "question_type_mix": mix_map,
        "question_type": "mixed",
        "difficulty": difficulty,
        "question_count": question_count,
        "category": category or exam_track,
        "bank_category_filter": bank_cat,
        "exam_category": ec,
        "examCategory": ec,
        "author_roles": roles,
        "author_role_coverage": coverage_mode,
    }
    if pc_id is not None:
        set_cfg["project_case_id"] = int(pc_id)
        set_cfg["projectCaseId"] = int(pc_id)
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
    if set_type == "practice" and not role_keywords:
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
        scope_hash = _make_scope_hash(exam_track, hash_cat, difficulty, qtype, ec, project_case_id=pc_id)
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
                        if not _question_matches_role_keywords(obj, role_keywords, strict=strict_role_filter):
                            continue
                        sig0 = _set_diversity_signature(obj)
                        if sig_counts.get(sig0, 0) >= max_sig_rep:
                            continue
                        selected.append(obj)
                        sig_counts[sig0] = sig_counts.get(sig0, 0) + 1
                for x in selected:
                    if x.get("id"):
                        seen_ids.add(int(x["id"]))
        short_pick = cnt - len(selected)
        cache_pick_n = short_pick
        if role_keywords:
            cache_pick_n = min(max(short_pick * 10, short_pick + 12), 400)
        cached = _pick_cached_questions(
            collection=collection,
            exam_track=exam_track,
            category=bank_cat,
            difficulty=difficulty,
            question_type=qtype,
            scope_hash=scope_hash,
            count=cache_pick_n,
            exclude_question_ids=sorted(seen_ids),
            sig_counts=sig_counts,
            question_count_total=question_count,
            role_keywords=role_keywords or None,
            shuffle_seed=role_shuffle_key,
        )
        if role_keywords:
            cached = [q for q in cached if _question_matches_role_keywords(q, role_keywords, strict=strict_role_filter)]
        cached = cached[:short_pick]
        selected.extend(cached)
        from_cache_total += len(cached)
        for x in selected:
            if x.get("id"):
                seen_ids.add(int(x["id"]))
        short = cnt - len(selected)
        if short > 0:
            if ec == "project_case" and pc_id is not None:
                evidence = _extract_evidence_project_case(
                    agent,
                    exam_track,
                    top_k=max(8, short),
                    exam_category=ec,
                    project_case_id=int(pc_id),
                    role_keywords=role_keywords if role_keywords else None,
                )
            elif role_keywords:
                evidence = _extract_evidence_for_roles(
                    agent,
                    exam_track,
                    top_k=max(12, short),
                    exam_category=ec,
                    role_keywords=role_keywords,
                )
            else:
                # 未勾选身份：按体考关注度权重定向取材，使生成题库自然偏向重点岗位
                evidence = _extract_evidence_for_roles(
                    agent,
                    exam_track,
                    top_k=max(12, short),
                    exam_category=ec,
                    role_keywords=_default_exam_weighted_scope(),
                )
            if role_keywords and ec == "project_case" and pc_id is not None and not evidence:
                evidence = _extract_evidence_project_case(
                    agent,
                    exam_track,
                    top_k=max(12, short),
                    exam_category=ec,
                    project_case_id=int(pc_id),
                    role_keywords=None,
                )
            stem_cat = (category or "").strip() or exam_track
            role_focus_meta = _role_focus_meta(role_keywords) if role_keywords else _default_exam_role_focus()
            use_ai_gap = short > 6 and not (role_keywords and short <= 8)
            if use_ai_gap:
                try:
                    generated = _generate_questions_by_ai(
                        exam_track=exam_track,
                        category=stem_cat,
                        difficulty=difficulty,
                        question_type=qtype,
                        count=short,
                        evidence=evidence,
                        exam_category=ec,
                        project_case_id=int(pc_id) if ec == "project_case" and pc_id is not None else None,
                        role_focus=role_focus_meta,
                    )
                except Exception:
                    generated = _fallback_questions(
                        exam_track,
                        stem_cat,
                        short,
                        evidence,
                        question_type=qtype,
                        difficulty=difficulty,
                        exam_category=ec,
                    )
            else:
                generated = _fallback_questions(
                    exam_track,
                    stem_cat,
                    short,
                    evidence,
                    question_type=qtype,
                    difficulty=difficulty,
                    exam_category=ec,
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
            if role_keywords:
                saved_matched = [q for q in saved if _question_matches_role_keywords(q, role_keywords, strict=strict_role_filter)]
                remain = short - len(saved_matched)
                if remain > 0:
                    fb_extra = _fallback_questions(
                        exam_track,
                        stem_cat,
                        remain,
                        evidence,
                        question_type=qtype,
                        difficulty=difficulty,
                        exam_category=ec,
                    )
                    saved_matched.extend(
                        _save_questions_to_bank(
                            collection=collection,
                            exam_track=exam_track,
                            category=stem_cat,
                            difficulty=difficulty,
                            question_type=qtype,
                            scope_hash=scope_hash,
                            questions=fb_extra,
                            origin=("exam_teacher_generated" if set_type == "exam" else "practice_runtime_generated"),
                            created_by=created_by,
                        )
                    )
                saved = saved_matched
            for x in saved:
                if isinstance(x, dict):
                    sg = _set_diversity_signature(x)
                    sig_counts[sg] = sig_counts.get(sg, 0) + 1
            selected.extend(saved)
            for x in saved:
                if x.get("id"):
                    seen_ids.add(int(x["id"]))
        selected_all.extend(selected[:cnt])
    if len(selected_all) < question_count:
        used_fill = {int(q.get("id") or 0) for q in selected_all if int(q.get("id") or 0) > 0}
        for qtype, cnt in plan:
            if len(selected_all) >= question_count:
                break
            have = sum(
                1
                for q in selected_all
                if _safe_question_type(str(q.get("question_type") or "")) == qtype
            )
            need_t = cnt - have
            if need_t <= 0:
                continue
            extras = _pick_cached_questions(
                collection=collection,
                exam_track=exam_track,
                category=bank_cat,
                difficulty=difficulty,
                question_type=qtype,
                scope_hash=_make_scope_hash(exam_track, hash_cat, difficulty, qtype, ec, project_case_id=pc_id),
                count=need_t,
                exclude_question_ids=sorted(used_fill),
                sig_counts=sig_counts,
                question_count_total=question_count,
                role_keywords=role_keywords or None,
                shuffle_seed=role_shuffle_key,
            )
            for q in extras:
                qid = int(q.get("id") or 0)
                if not qid or qid in used_fill:
                    continue
                selected_all.append(q)
                used_fill.add(qid)
                from_cache_total += 1
                if len(selected_all) >= question_count:
                    break
    pre_alloc = list(selected_all)
    random.shuffle(selected_all)
    if role_keywords and coverage_mode == "balanced_union":
        allocated = _promote_role_coverage_questions(selected_all, role_keywords, question_count)
        if len(allocated) < question_count:
            used_ids = {int(q.get("id") or 0) for q in allocated if int(q.get("id") or 0) > 0}
            for q in pre_alloc:
                if len(allocated) >= question_count:
                    break
                qid = int(q.get("id") or 0)
                if qid and qid not in used_ids:
                    allocated.append(q)
                    used_ids.add(qid)
        selected_all = allocated
    elif role_keywords:
        eligible_only = [q for q in selected_all if _question_eligible_for_selected_roles(q, role_keywords)]
        if eligible_only:
            selected_all = eligible_only
    selected_all = selected_all[:question_count]
    if len(selected_all) < question_count:
        stem_cat = (category or "").strip() or exam_track
        if role_keywords:
            gap_evidence = _extract_evidence_for_roles(
                agent,
                exam_track,
                top_k=12,
                exam_category=ec,
                role_keywords=role_keywords,
            )
        else:
            gap_evidence = _extract_evidence_for_roles(
                agent,
                exam_track,
                top_k=12,
                exam_category=ec,
                role_keywords=_default_exam_weighted_scope(),
            )
        for qtype, cnt in plan:
            if len(selected_all) >= question_count:
                break
            have = sum(
                1
                for q in selected_all
                if _safe_question_type(str(q.get("question_type") or "")) == qtype
            )
            need_t = min(cnt - have, question_count - len(selected_all))
            if need_t <= 0:
                continue
            fb = _fallback_questions(
                exam_track,
                stem_cat,
                need_t,
                gap_evidence,
                question_type=qtype,
                difficulty=difficulty,
                exam_category=ec,
            )
            saved_gap = _save_questions_to_bank(
                collection=collection,
                exam_track=exam_track,
                category=stem_cat,
                difficulty=difficulty,
                question_type=qtype,
                scope_hash=_make_scope_hash(exam_track, hash_cat, difficulty, qtype, ec, project_case_id=pc_id),
                questions=fb,
                origin=("exam_teacher_generated" if set_type == "exam" else "practice_runtime_generated"),
                created_by=created_by,
            )
            generated_total += len(saved_gap)
            for q in saved_gap:
                qid = int(q.get("id") or 0)
                if not qid or qid in seen_ids:
                    continue
                selected_all.append(q)
                seen_ids.add(qid)
                if len(selected_all) >= question_count:
                    break
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
    if role_keywords:
        _attach_author_roles_to_set_items(
            out.get("items") or [],
            role_keywords,
            strict_role_filter=strict_role_filter,
        )
        cov = _build_role_coverage_summary(selected_all, role_keywords)
        out["author_roles"] = roles
        label_map = _author_role_label_map()
        out["author_role_labels"] = {r: label_map.get(r) or r for r in roles}
        out["role_coverage_summary"] = cov
        missing = [r for r in roles if int(cov.get(r, 0)) <= 0]
        if missing and coverage_mode == "balanced_union":
            missing_labels = [label_map.get(r) or r for r in missing]
            out["coverage_warning"] = (
                f"以下身份专属题覆盖偏少：{', '.join(missing_labels)}。"
                f"通用基线题 {int(cov.get(COMMON_AUTHOR_ROLE_KEY, 0))} 道；"
                "请增加题量、调整身份组合或补充录题。"
            )
    else:
        # 未勾选身份：仍为每题标注其关联岗位（可多个），通用题标签为空
        _attach_author_roles_to_set_items(out.get("items") or [])
    if set_type in ("practice", "exam"):
        for it in out.get("items") or []:
            if isinstance(it, dict):
                _redact_student_quiz_item(it)
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
    exam_category: str = "daily",
    ingest_knowledge_weights: Optional[List[float]] = None,
    ingest_question_type_weights: Optional[List[float]] = None,
    max_similar_frac: Optional[float] = None,
    project_case_id: Optional[Any] = None,
) -> Dict[str, Any]:
    target_count = max(1, int(target_count))
    ec = _normalize_exam_category(exam_category)
    pc_id = require_project_case_quiz(collection=collection, exam_category=exam_category, project_case_id=project_case_id)
    kw = ingest_knowledge_weights if ingest_knowledge_weights is not None else None
    qw = ingest_question_type_weights if ingest_question_type_weights is not None else None
    msf = float(max_similar_frac) if max_similar_frac is not None else 0.1
    if msf < 0.0:
        msf = 0.0
    if msf > 0.5:
        msf = 0.5
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
    mix_label = "project_case100" if ec == "project_case" else "project_case30_audit30_regulation20_program20"
    draft_cfg: Dict[str, Any] = {
        "source": "bank_ingest_by_ai",
        "ingest_job_id": job_id,
        "ingest_knowledge_weights": kw,
        "ingest_question_type_weights": qw,
        "ingest_max_similar_frac": msf,
        "ingest_knowledge_mix": mix_label,
        "ingest_type_mix": "single30_multi10_tf10_case50",
        "difficulty": difficulty,
        "exam_category": ec,
        "examCategory": ec,
    }
    if pc_id is not None:
        draft_cfg["project_case_id"] = int(pc_id)
        draft_cfg["projectCaseId"] = int(pc_id)
    try:
        draft_set_id = repo.create_set(
            collection=collection,
            set_type="bank_ingest",
            exam_track=exam_track,
            title=title,
            set_config=draft_cfg,
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
            if ec == "project_case":
                scope_plan: List[tuple[str, str, int]] = [("project_case", "项目案例", target_count)]
            else:
                scope_plan = _ingest_knowledge_scope_plan(target_count, weights=kw)
            for scope_key, cat_label, seg_n in scope_plan:
                if seg_n <= 0:
                    continue
                for qt, qc in _ingest_question_type_plan(seg_n, weights=qw):
                    if qc <= 0:
                        continue
                    qt = _safe_question_type(qt)
                    scope_hash = _make_scope_hash(exam_track, hash_cat, difficulty, qt, ec, project_case_id=pc_id)
                    gen_n = qc + max(2, (qc + 3) // 4)
                    # 项目案例录题：与「来一套」补题同源取证，并加大 top_k（录题无缓存、全靠当次素材，过少易泛化）
                    if ec == "project_case" and pc_id is not None:
                        tk_pc = max(16, qc, gen_n, qc * 2 + 4)
                        tk_pc = min(tk_pc, 120)
                        evidence = _extract_evidence_project_case(
                            agent,
                            exam_track,
                            top_k=tk_pc,
                            exam_category=ec,
                            project_case_id=int(pc_id),
                        )
                    else:
                        evidence = _extract_evidence_scoped(
                            agent,
                            exam_track,
                            scope_key,
                            top_k=max(8, qc),
                            exam_category=ec,
                            project_case_id=pc_id,
                        )
                    try:
                        generated = _generate_questions_by_ai(
                            exam_track=exam_track,
                            category=cat_label,
                            difficulty=difficulty,
                            question_type=qt,
                            count=gen_n,
                            evidence=evidence,
                            exam_category=ec,
                            project_case_id=int(pc_id) if ec == "project_case" and pc_id is not None else None,
                        )
                    except Exception:
                        generated = _fallback_questions(
                            exam_track,
                            cat_label,
                            gen_n,
                            evidence,
                            question_type=qt,
                            difficulty=difficulty,
                            exam_category=ec,
                        )
                    prior_stems = [str(x.get("stem") or "") for x in all_saved]
                    prior_stems.extend(bank_stems_for_dedupe)
                    generated = _dedupe_questions_for_ingest(
                        generated, prior_stems=prior_stems, max_similar_frac=msf
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
                            exam_category=ec,
                        )
                        more_prior = prior_stems + [str(x.get("stem") or "") for x in generated]
                        fb = _dedupe_questions_for_ingest(fb, prior_stems=more_prior, max_similar_frac=msf)
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
    ua = r.get("user_answer")
    ua_text = _subjective_user_answer_text(ua)
    if not ua_text:
        repo.update_answer_grade(
            answer_id=int(r["answer_id"]),
            auto_score=0.0,
            final_score=0.0,
            is_correct=False,
            teacher_comment="未作答或答案为空",
            graded_by_cache=False,
        )
        return
    if len(ua_text) < 8:
        repo.update_answer_grade(
            answer_id=int(r["answer_id"]),
            auto_score=0.0,
            final_score=0.0,
            is_correct=False,
            teacher_comment="答案过短",
            graded_by_cache=False,
        )
        return
    prompt = f"""
请按 0~1 评分并返回 JSON: {{"score":0.0,"comment":"..."}}。
题干: {r.get('stem') or ''}
标准答案要点: {json.dumps(r.get('answer'), ensure_ascii=False)}
学生答案: {json.dumps(ua, ensure_ascii=False)}

规则：未作答=0；明显不全或缺少关键要点不得超过0.35；仅部分正确0.35~0.65；基本完整才可用0.65以上。
""".strip()
    try:
        prov = (settings.quiz_provider or settings.provider or "").strip().lower()
        model = (settings.quiz_llm_model or settings.llm_model or "").strip()
        temp = float(getattr(settings, "quiz_temperature", 0.2) or 0.2)
        raw = invoke_chat_direct(prompt, temperature=temp, provider=prov, model=model)
        data = json.loads(_norm_json_text(raw))
        score = max(0.0, min(1.0, float(data.get("score") or 0.0)))
        score = _cap_subjective_score_by_answer_length(score, ua_text)
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


def _subjective_user_answer_text(user_answer: Any) -> str:
    if user_answer is None:
        return ""
    if isinstance(user_answer, str):
        return user_answer.strip()
    if isinstance(user_answer, (int, float, bool)):
        return str(user_answer).strip()
    if isinstance(user_answer, dict):
        for key in ("value", "text", "answer", "content"):
            if key in user_answer and user_answer[key] is not None:
                s = str(user_answer[key]).strip()
                if s:
                    return s
        parts = [str(v).strip() for v in user_answer.values() if v is not None and str(v).strip()]
        return " ".join(parts).strip()
    if isinstance(user_answer, list):
        parts = [str(x).strip() for x in user_answer if x is not None and str(x).strip()]
        return " ".join(parts).strip()
    return str(user_answer).strip()


def _cap_subjective_score_by_answer_length(score: float, ua_text: str) -> float:
    sc = max(0.0, min(1.0, float(score)))
    txt = str(ua_text or "").strip()
    if not txt:
        return 0.0
    n = len(txt)
    if n < 8:
        return 0.0
    if n < 30:
        return min(sc, 0.15)
    if n < 80:
        return min(sc, 0.35)
    if n < 150:
        return min(sc, 0.65)
    return sc


def _grade_subjective_question_llm(
    *,
    exam_track: str,
    stem: str,
    user_answer: Any,
    evidence: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """返回 {score(0~1), reason, recommendation, evidence_used[]}；证据只要求文件名定位。"""
    ua_text = _subjective_user_answer_text(user_answer)
    if not ua_text:
        used_fallback = []
        for e in (evidence or [])[:1]:
            sf = str(e.get("source_file") or "").strip()
            if sf:
                used_fallback.append({"source_file": sf, "snippet": str(e.get("snippet") or "")[:300]})
        return {
            "score": 0.0,
            "reason": "未作答或答案为空",
            "recommendation": "请按题干要点完整作答后再提交。",
            "evidence_used": used_fallback,
        }
    if len(ua_text) < 8:
        return {
            "score": 0.0,
            "reason": "答案过短，无法视为有效作答",
            "recommendation": "请补充关键要点与依据。",
            "evidence_used": [],
        }
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
- 未作答、空白或与题干无关：score 必须为 0
- 明显不全、缺少多数关键要点：score 不得超过 0.35
- 仅部分正确：score 宜在 0.35~0.65
- 基本完整且要点正确：score 宜在 0.65~0.85；仅当几乎全面正确才可用 0.9 以上
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
        score = _cap_subjective_score_by_answer_length(score, ua_text)
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


def _resolve_audit_checklist_open_book(*, collection: str, source_file: str) -> Dict[str, Any]:
    """从 audit_checklists / checkpoint 向量库解析审核点清单开卷内容。"""
    from src.core import db

    sf = str(source_file or "").strip()
    if not sf:
        return {"content": "", "title": "", "kind": "audit_checklist", "chunk_count": 0}

    point_id = ""
    if sf.startswith("审核点清单:"):
        point_id = sf.split(":", 1)[-1].strip()

    row = db.get_audit_checklist_by_name(sf, collection=collection or None)
    if not row:
        for cand in db.search_audit_checklists_by_name(sf, collection=collection or None, limit=3):
            row = cand
            break

    if row:
        checklist = row.get("checklist") or []
        title = str(row.get("name") or sf).strip()
        if point_id:
            for point in checklist:
                if str(point.get("id") or "").strip() == point_id:
                    content = _format_checklist_point_text(point)
                    return {
                        "content": content,
                        "title": f"{title} · {point_id}",
                        "kind": "audit_checkpoint_point",
                        "chunk_count": 1,
                    }
        parts = [_format_checklist_point_text(p) for p in checklist if isinstance(p, dict)]
        content = "\n\n---\n\n".join(x for x in parts if x).strip()
        return {
            "content": content,
            "title": title,
            "kind": "audit_checklist",
            "chunk_count": len(parts),
        }

    if point_id:
        found = db.find_audit_checkpoint_in_checklists(point_id, collection=collection or None, limit=50)
        if found and isinstance(found.get("point"), dict):
            content = _format_checklist_point_text(found["point"])
            cl_name = str(found.get("checklist_name") or "").strip()
            title = f"{cl_name} · {point_id}" if cl_name else sf
            return {
                "content": content,
                "title": title,
                "kind": "audit_checkpoint_point",
                "chunk_count": 1,
            }
        try:
            agent = ReviewAgent(collection)
            docs = agent.checkpoint_kb.search(f"审核点编号：{point_id}", top_k=8)
            for doc in docs:
                md = getattr(doc, "metadata", None) or {}
                if str(md.get("point_id") or "").strip() == point_id:
                    content = str(getattr(doc, "page_content", "") or "").strip()
                    if content:
                        return {
                            "content": content,
                            "title": sf,
                            "kind": "audit_checkpoint_point",
                            "chunk_count": 1,
                        }
        except Exception:
            pass

    return {"content": "", "title": sf, "kind": "audit_checklist", "chunk_count": 0}


def open_book_reference(*, collection: str, source_file: str) -> Dict[str, Any]:
    """开卷查阅：按来源文件名返回知识库全文摘录（审核点清单 / 项目文档等）。"""
    from src.core import db

    sf = str(source_file or "").strip()
    if not sf:
        raise ValueError("缺少 source_file")

    if "审核点清单" in sf or sf.startswith("审核点"):
        audit_res = _resolve_audit_checklist_open_book(collection=collection, source_file=sf)
        if str(audit_res.get("content") or "").strip():
            return {
                "source_file": sf,
                "title": audit_res.get("title") or sf,
                "content": audit_res["content"],
                "kind": audit_res.get("kind") or "audit_checklist",
                "chunk_count": int(audit_res.get("chunk_count") or 0),
                "open_book": True,
            }

    def _norm_name(x: str) -> str:
        t = str(x or "").strip()
        t = t.replace("《", "").replace("》", "").replace("\\", "/")
        if "/" in t:
            t = t.rsplit("/", 1)[-1]
        return t.strip()

    sf_norm = _norm_name(sf)
    sf_stem = sf_norm.rsplit(".", 1)[0] if "." in sf_norm else sf_norm
    name_candidates = [x for x in {sf, sf_norm, sf_stem} if x]

    rows: List[Dict[str, Any]] = []
    for cand in name_candidates:
        rows = db.get_knowledge_docs(collection=collection, file_name=cand, limit=80)
        if rows:
            break
    if not rows:
        db.init_db()
        conn = db._get_conn()  # noqa: SLF001
        try:
            with conn.cursor() as cur:
                like_tail = sf_norm[-64:] if len(sf_norm) > 64 else sf_norm
                like_stem = sf_stem[-64:] if len(sf_stem) > 64 else sf_stem
                cur.execute(
                    """
                    SELECT file_name, content, category, metadata_json
                    FROM knowledge_docs
                    WHERE collection=%s
                      AND (
                        file_name IN (%s, %s, %s)
                        OR file_name LIKE %s
                        OR file_name LIKE %s
                        OR content LIKE %s
                        OR metadata_json LIKE %s
                        OR metadata_json LIKE %s
                      )
                    ORDER BY id DESC
                    LIMIT 120
                    """,
                    (
                        collection,
                        sf,
                        sf_norm,
                        sf_stem,
                        f"%{like_tail}",
                        f"%{like_stem}%",
                        f"%{sf_stem[:120]}%",
                        f"%{sf_norm[:120]}%",
                        f"%{sf[:120]}%",
                    ),
                )
                rows = [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()
    chunks: List[str] = []
    for r in rows or []:
        c = str(r.get("content") or "").strip()
        if c:
            chunks.append(c)
    content = "\n\n".join(chunks).strip()
    kind = "audit_checklist" if ("审核点清单" in sf or sf.startswith("审核点")) else "document"
    return {
        "source_file": sf,
        "title": sf,
        "content": content,
        "kind": kind,
        "chunk_count": len(chunks),
        "open_book": True,
    }


def bank_author_role_coverage(
    *,
    collection: str,
    exam_track: str = "",
    author_roles: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """题库身份覆盖：与练习组卷一致（所选身份专属 + 通用基线并集）。"""
    roles = _normalize_author_roles(author_roles or [])
    cache_key = f"{collection}|{exam_track or ''}|{','.join(sorted(roles))}"
    now = float(time.time())
    cached = _ROLE_COVERAGE_CACHE.get(cache_key)
    if cached and now - float(cached[0]) <= _ROLE_COVERAGE_CACHE_TTL_SEC:
        return dict(cached[1])

    label_map = _author_role_label_map()
    label_map[COMMON_AUTHOR_ROLE_KEY] = COMMON_AUTHOR_ROLE_LABEL
    selected_scope = _role_file_keyword_scope(roles)
    all_scope = _all_author_role_keyword_scope()

    items: List[Dict[str, Any]] = []
    offset = 0
    try:
        for _ in range(15):
            batch = repo.admin_list_bank_questions_role_coverage_batch(
                collection=collection,
                exam_track=exam_track or None,
                limit=200,
                offset=offset,
            )
            if not batch:
                break
            items.extend(batch)
            offset += len(batch)
            if len(batch) < 200:
                break
    except Exception as exc:
        logger.exception("bank_author_role_coverage list questions failed collection=%s track=%s", collection, exam_track)
        raise RuntimeError(f"题库身份覆盖统计失败：{exc}") from exc

    role_counts: Dict[str, int] = {r: 0 for r in roles}
    leadership_mode = _selection_has_leadership_cross_need(selected_scope)
    diagnostics = {
        "eligible_total": 0,
        "selected_primary_count": 0,
        "selected_cross_focus_count": 0,
        "leadership_cross_pool_count": 0,
        "common_count": 0,
    }
    common_count = 0
    seen: set[int] = set()
    for q in items:
        qid = int(q.get("id") or 0)
        if not qid or qid in seen:
            continue
        seen.add(qid)
        if leadership_mode and _question_is_leadership_cross_candidate(q, selected_scope):
            diagnostics["leadership_cross_pool_count"] = int(diagnostics["leadership_cross_pool_count"]) + 1
        if not _question_eligible_for_selected_roles(q, selected_scope):
            continue
        diagnostics["eligible_total"] = int(diagnostics["eligible_total"]) + 1
        hits = _question_role_hits_extended(q, selected_scope)
        if hits:
            if _question_hits_unselected_focus_roles(q, selected_scope):
                diagnostics["selected_cross_focus_count"] = int(diagnostics["selected_cross_focus_count"]) + 1
            else:
                diagnostics["selected_primary_count"] = int(diagnostics["selected_primary_count"]) + 1
            for r in hits:
                role_counts[r] = int(role_counts.get(r, 0)) + 1
        elif _question_counts_as_common_baseline(q, selected_scope, all_scope):
            common_count += 1
            diagnostics["common_count"] = int(diagnostics["common_count"]) + 1

    role_checks: List[Dict[str, Any]] = []
    for role in roles:
        kws = [str(x).strip() for x in (_AUTHOR_ROLE_FILE_KEYWORDS.get(role) or []) if str(x).strip()]
        hit = int(role_counts.get(role, 0))
        role_checks.append(
            {
                "role": role,
                "label": str(label_map.get(role) or role),
                "keywords": kws,
                "keyword": kws[0] if kws else "",
                "hit_count": hit,
                "is_met": hit > 0,
            }
        )
    role_checks.append(
        {
            "role": COMMON_AUTHOR_ROLE_KEY,
            "label": COMMON_AUTHOR_ROLE_LABEL,
            "keywords": [],
            "keyword": "",
            "hit_count": int(common_count),
            "is_met": common_count > 0,
        }
    )
    met = len([x for x in role_checks if x.get("is_met") is True and x.get("role") != COMMON_AUTHOR_ROLE_KEY])
    rate = (met / len(roles)) if roles else 1.0
    out = {
        "role_checks": role_checks,
        "role_coverage_rate": round(float(rate), 4),
        "selected_author_roles": roles,
        "common_pool_count": int(common_count),
        "role_diagnostics": diagnostics,
    }
    _ROLE_COVERAGE_CACHE[cache_key] = (now, out)
    return out


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


def regulatory_updates_hint(*, exam_track: str, as_of: Optional[str] = None, since: Optional[str] = None) -> Dict[str, Any]:
    """面向「新标发布」备考：由模型归纳**可能需关注的监管/标准动态方向**（不等同于官方发布清单）。

    输出仅供内部培训选题线索；实施与注册决策必须以主管部门与标准发布机构正式文本为准。
    """
    tr = str(exam_track or "").strip().lower()
    if tr not in EXAM_TRACKS:
        raise ValueError(f"不支持的 exam_track: {exam_track}")
    as_of_d = date.today().isoformat()
    if as_of:
        try:
            as_of_d = str(date.fromisoformat(str(as_of).strip()[:10]))
        except Exception:
            pass
    since_d = (date.fromisoformat(as_of_d) - timedelta(days=365)).isoformat()
    if since:
        try:
            since_d = str(date.fromisoformat(str(since).strip()[:10]))
        except Exception:
            pass
    track_name = EXAM_TRACKS.get(tr, tr)
    track_hint = TRACK_HINTS.get(tr, "")
    prompt = f"""
你是医疗器械软件（SaMD/独立软件）合规信息助理。
时间窗（必须遵守）：**自 {since_d} 起至 {as_of_d}（含）止，约近 12 个月**；请优先围绕该时间窗内**可能已公开讨论/征求意见/换版预告/过渡期安排**等方向组织要点（仍不得编造具体文号、公告号与实施日期）。
体考类型：{track_name}（{track_hint}）

任务：列出培训/考试命题时**值得优先核对**的「通用法规、标准、指南、专标」**更新与修订方向**（不写死具体文号与实施日期，避免幻觉）。

硬性要求：
1) 只输出 JSON，不要其它文字。
2) 格式：{{
  "since": "{since_d}",
  "as_of": "{as_of_d}",
  "exam_track": "{tr}",
  "disclaimer": "本结果由模型归纳，可能不完整或滞后；**不得以本 JSON 替代官方发布**。命题请围绕“变化影响与组织应对”而非虚构条款。",
  "checklist": [
    {{"domain":"监管/标准/指南/专标之一","what_to_watch":"要关注什么变化","why_for_software":"与医疗器械软件相关的典型影响方向","how_to_verify":"建议的官方核对渠道类型（如主管部门法规库、标准委公开文本）"}}
  ],
  "suggested_question_angles": ["用于录题/出题的角度1","角度2","角度3"]
}}
3) checklist 4～8 条；每条应能回答「与上一稳定版本相比，组织要核对什么」；表述克制，避免断言「已发布」「已实施」除非属于常识性周期描述且不绑定具体文号。

JSON:""".strip()
    try:
        prov = (settings.quiz_provider or settings.provider or "").strip().lower()
        model = (settings.quiz_llm_model or settings.llm_model or "").strip()
        temp = float(getattr(settings, "quiz_temperature", 0.2) or 0.2)
        logger.info(
            "regulatory_updates_hint: calling LLM (invoke_chat_direct) exam_track=%s since=%s as_of=%s provider=%r model=%r",
            tr,
            since_d,
            as_of_d,
            prov or "",
            model or "",
        )
        txt = invoke_chat_direct(prompt, temperature=temp, provider=prov, model=model)
        data = json.loads(_norm_json_text(txt))
        if not isinstance(data, dict):
            data = {}
        data.setdefault("since", since_d)
        data.setdefault("as_of", as_of_d)
        data.setdefault("exam_track", tr)
        data.setdefault(
            "disclaimer",
            "本结果由模型归纳，可能不完整或滞后；不得以本输出替代官方发布。",
        )
        n_chk = len(data.get("checklist") or []) if isinstance(data.get("checklist"), list) else 0
        logger.info(
            "regulatory_updates_hint: LLM returned JSON ok exam_track=%s checklist_items=%s",
            tr,
            n_chk,
        )
        return data
    except Exception as exc:
        logger.warning(
            "regulatory_updates_hint: LLM or parse failed, using static fallback exam_track=%s err=%s",
            tr,
            str(exc)[:500],
        )
        return {
            "since": since_d,
            "as_of": as_of_d,
            "exam_track": tr,
            "disclaimer": "模型不可用或解析失败；请直接查阅主管部门法规库与标准发布机构公开文本。",
            "error": str(exc),
            "checklist": [
                {
                    "domain": "软件生命周期与配置管理",
                    "what_to_watch": "设计开发/变更控制/发布与可追溯性要求是否有更新导向",
                    "why_for_software": "独立软件版本迭代频繁，需对齐变更证据与风险管理更新",
                    "how_to_verify": "对照最新质量管理体系与软件相关指导原则公开文本",
                },
                {
                    "domain": "网络安全与数据保护",
                    "what_to_watch": "数据出境、日志留存、脆弱性管理等要求是否有强化趋势",
                    "why_for_software": "联网与远程功能影响安全证据组织方式",
                    "how_to_verify": "查阅网络安全相关法规与配套指南的公开版本说明",
                },
            ],
            "suggested_question_angles": ["修订前后差异对验证范围的影响", "适用范围变化对说明书与标签的影响", "过渡期内的证据组织策略（不绑定具体日期）"],
        }

