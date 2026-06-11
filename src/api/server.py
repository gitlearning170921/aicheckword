"""FastAPI 服务：将 Agent 能力暴露为 REST API，供其他项目调用"""

import sys
import os
import logging
import re
import time
import uuid
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Disable Chroma/PostHog telemetry early (avoid noisy errors on some envs)
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["POSTHOG_DISABLED"] = "true"
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb").setLevel(logging.WARNING)


class _QuietDraftJobStatusPollAccessFilter(logging.Filter):
    """屏蔽 aiword 等对初稿任务状态的轮询行，避免 uvicorn access 刷屏（保留 POST 创建与 download）。"""

    _needle = "/api/integration/draft/jobs/"

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        if self._needle not in msg:
            return True
        u = msg.upper()
        if "GET " not in u and '"GET ' not in msg:
            return True
        if "/DOWNLOAD" in u:
            return True
        return False


import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import json

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Header, Request
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import (
    get_swagger_ui_oauth2_redirect_html,
    swagger_ui_default_parameters,
)
from fastapi.responses import HTMLResponse
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from config import settings
from src.core.agent import ReviewAgent
from src.core.db import (
    get_dimension_options,
    REGISTRATION_TYPES,
    REGISTRATION_COMPONENTS,
    list_projects,
    list_project_cases,
    list_companies,
    create_project_case,
    get_project_case,
    upsert_company_mapping,
    delete_company_by_aiword_id,
)
from src.core.project_option_label import format_project_option_label
from src.core.document_loader import load_single_file, split_documents
from config.runtime_settings import apply_runtime_config_dict, sync_cursor_overrides_from_settings
from src.core.db import load_app_settings
from src.core.quiz import service as quiz_service
from src.core.quiz.models import EXAM_TRACKS
from src.core.quiz.service import QuizRequestError
from src.api.audit_integration import router as audit_integration_router
from src.api.draft_integration import router as draft_integration_router
from src.api.integration_common import router as integration_common_router
from src.api.translation_integration import router as translation_integration_router

app = FastAPI(
    title="注册文档审核 Agent API",
    description="基于 RAG 的注册文档审核服务，支持训练知识库和自动审核文档",
    version="1.0.0",
    # 关闭默认 docs，改为下方自定义路由（避免默认 jsdelivr 在部分网络环境下加载失败）
    docs_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(draft_integration_router)
app.include_router(audit_integration_router)
app.include_router(translation_integration_router)
app.include_router(integration_common_router)

_agents: Dict[str, ReviewAgent] = {}


def _collection_by_company_header(company_id: str) -> str:
    cid = str(company_id or "").strip()
    if not cid:
        return ""
    try:
        for row in list_companies() or []:
            rid = str(row.get("id") or "").strip()
            aid = str(row.get("aiword_company_id") or "").strip()
            if cid in (rid, aid):
                return str(row.get("knowledge_collection") or "").strip() or "regulations"
    except Exception:
        return ""
    return ""


def _resolve_request_collection(req: Request, provided: Optional[str]) -> str:
    h = str(req.headers.get("X-Aiword-Company-Id") or "").strip()
    if h:
        by_header = _collection_by_company_header(h)
        if by_header:
            return by_header
    p = str(provided or "").strip()
    return p or "regulations"


def _configure_api_service_console_timestamps() -> None:
    """为 uvicorn 控制台日志增加本地时间（兼容 `python -m uvicorn ...` 等未传 log_config 的启动方式）。"""
    datefmt = "%Y-%m-%d %H:%M:%S"
    try:
        from uvicorn.logging import AccessFormatter, DefaultFormatter

        default_fmt = DefaultFormatter(
            fmt="%(asctime)s | %(levelprefix)s %(message)s",
            datefmt=datefmt,
            use_colors=None,
        )
        access_fmt = AccessFormatter(
            fmt='%(asctime)s | %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            datefmt=datefmt,
            use_colors=None,
        )
    except Exception:
        default_fmt = access_fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt=datefmt,
        )
    for h in logging.getLogger("uvicorn.access").handlers:
        h.setFormatter(access_fmt)
        if not any(
            type(f).__name__ == "_QuietDraftJobStatusPollAccessFilter"
            for f in getattr(h, "filters", ())
        ):
            h.addFilter(_QuietDraftJobStatusPollAccessFilter())
    for name in ("uvicorn.error", "uvicorn"):
        for h in logging.getLogger(name).handlers:
            h.setFormatter(default_fmt)


def _build_uvicorn_log_config_with_time() -> dict:
    """供 `start_server()` 传入 uvicorn：在默认格式前增加 asctime。"""
    import copy

    from uvicorn.config import LOGGING_CONFIG

    cfg = copy.deepcopy(LOGGING_CONFIG)
    datefmt = "%Y-%m-%d %H:%M:%S"
    cfg.setdefault("filters", {})
    cfg["filters"]["quiet_draft_job_status_poll"] = {
        "()": "src.api.server._QuietDraftJobStatusPollAccessFilter",
    }
    acc_handler = cfg.get("handlers", {}).get("access")
    if isinstance(acc_handler, dict):
        prev = acc_handler.get("filters")
        if isinstance(prev, list):
            names = [x for x in prev if x != "quiet_draft_job_status_poll"]
            names.append("quiet_draft_job_status_poll")
            acc_handler["filters"] = names
        elif prev:
            acc_handler["filters"] = [str(prev), "quiet_draft_job_status_poll"]
        else:
            acc_handler["filters"] = ["quiet_draft_job_status_poll"]
    for key in ("default", "access"):
        spec = cfg.get("formatters", {}).get(key)
        if isinstance(spec, dict):
            prev = str(spec.get("fmt") or "")
            if "%(asctime)s" not in prev:
                spec["fmt"] = "%(asctime)s | " + prev
            spec["datefmt"] = datefmt
    return cfg


@app.on_event("startup")
def _load_runtime_settings_from_db_on_startup() -> None:
    """
    API 服务启动时从数据库恢复 runtime_settings_json（与 Streamlit Web UI 一致）。
    否则系统配置页保存的 quiz_provider/quiz_llm_model 等不会影响 /quiz/* 调用。
    """
    _configure_api_service_console_timestamps()
    try:
        db_conf = load_app_settings()
        if not db_conf:
            return
        raw_json = db_conf.get("runtime_settings_json")
        if not raw_json:
            return
        try:
            parsed = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
        except Exception:
            return
        if isinstance(parsed, dict) and parsed:
            apply_runtime_config_dict(parsed)
            sync_cursor_overrides_from_settings()
    except Exception:
        # 启动阶段不阻断 API；保持 .env 默认配置继续运行
        return

# Swagger UI 静态资源：多 CDN 回退（unpkg/jsdelivr 在部分网络下不可达会导致 /docs 空白）
_SWAGGER_UI_CSS_URLS = [
    "https://unpkg.com/swagger-ui-dist@5/swagger-ui.css",
    "https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.17.14/swagger-ui.css",
    "https://cdn.bootcdn.net/ajax/libs/swagger-ui/5.17.14/swagger-ui.css",
]
_SWAGGER_UI_BUNDLE_URLS = [
    "https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js",
    "https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.17.14/swagger-ui-bundle.js",
    "https://cdn.bootcdn.net/ajax/libs/swagger-ui/5.17.14/swagger-ui-bundle.js",
]


def _swagger_ui_html_with_cdn_fallback(
    *,
    openapi_url: str,
    title: str,
    oauth2_redirect_url: Optional[str] = None,
) -> HTMLResponse:
    params = swagger_ui_default_parameters.copy()
    cfg_lines = [f"url: {json.dumps(openapi_url)}"]
    for key, value in params.items():
        cfg_lines.append(f"{json.dumps(key)}: {json.dumps(jsonable_encoder(value))}")
    if oauth2_redirect_url:
        cfg_lines.append(
            "oauth2RedirectUrl: window.location.origin + " + json.dumps(oauth2_redirect_url)
        )
    cfg_lines.append(
        "presets: [\n"
        "        SwaggerUIBundle.presets.apis,\n"
        "        SwaggerUIBundle.SwaggerUIStandalonePreset\n"
        "    ]"
    )
    cfg_inner = ",\n        ".join(cfg_lines)

    css_json = json.dumps(_SWAGGER_UI_CSS_URLS)
    js_json = json.dumps(_SWAGGER_UI_BUNDLE_URLS)
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
</head>
<body>
    <div id="swagger-ui"></div>
    <script>
    (function () {{
        // Polyfill for older browser engines (e.g., old Chromium/WebView)
        // Swagger UI 5.x may call Object.hasOwn which doesn't exist everywhere.
        if (typeof Object.hasOwn !== "function") {{
            Object.hasOwn = function (obj, prop) {{
                return Object.prototype.hasOwnProperty.call(obj, prop);
            }};
        }}
        function loadCss(urls, idx) {{
            if (idx >= urls.length) return;
            var l = document.createElement("link");
            l.rel = "stylesheet";
            l.type = "text/css";
            l.href = urls[idx];
            l.onerror = function () {{ loadCss(urls, idx + 1); }};
            document.head.appendChild(l);
        }}
        function fail(msg) {{
            document.getElementById("swagger-ui").innerHTML =
                '<div style="padding:1rem;font-family:sans-serif">' + msg + "</div>";
        }}
        function loadBundle(urls, idx) {{
            if (idx >= urls.length) {{
                fail("Swagger UI 脚本无法加载，请检查网络或对 unpkg / Cloudflare / BootCDN 的访问是否被拦截。");
                return;
            }}
            var s = document.createElement("script");
            s.src = urls[idx];
            s.onload = function () {{ initSwagger(); }};
            s.onerror = function () {{ loadBundle(urls, idx + 1); }};
            document.body.appendChild(s);
        }}
        function initSwagger() {{
            try {{
                const ui = SwaggerUIBundle({{
        {cfg_inner}
    }});
            }} catch (e) {{
                fail("Swagger UI 初始化失败: " + (e && e.message ? e.message : String(e)));
            }}
        }}
        loadCss({css_json}, 0);
        loadBundle({js_json}, 0);
    }})();
    </script>
</body>
</html>"""
    return HTMLResponse(html)


def get_agent(collection: str = "regulations") -> ReviewAgent:
    if collection not in _agents:
        _agents[collection] = ReviewAgent(collection)
    return _agents[collection]


class TextReviewRequest(BaseModel):
    text: str
    file_name: str = "直接输入"
    collection: str = "regulations"
    project_id: Optional[int] = None
    review_context: Optional[Dict[str, Any]] = None
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    extra_instructions: Optional[str] = None
    # 便捷直传（与页面维度一致，优先合并到 review_context）
    project_name: Optional[str] = None
    project_name_en: Optional[str] = None
    product_name: Optional[str] = None
    product_name_en: Optional[str] = None
    model: Optional[str] = None
    model_en: Optional[str] = None
    registration_country: Optional[str] = None
    registration_country_en: Optional[str] = None
    registration_type: Optional[str] = None
    registration_component: Optional[str] = None
    project_form: Optional[str] = None
    document_language: Optional[str] = None
    scope_of_application: Optional[str] = None
    basic_info_text: Optional[str] = None
    system_functionality_text: Optional[str] = None
    current_provider: Optional[str] = None


class KnowledgeQueryRequest(BaseModel):
    # 自由文本（必填）
    query: str
    # 必传筛选维度（与页面一致）
    registration_country: str
    registration_type: str
    registration_component: str
    project_form: str
    document_language: str  # zh / en / both
    # 可选补充维度
    project_name: Optional[str] = None
    project_name_en: Optional[str] = None
    product_name: Optional[str] = None
    product_name_en: Optional[str] = None
    model: Optional[str] = None
    model_en: Optional[str] = None
    registration_country_en: Optional[str] = None
    # 项目案例维度（可选；用于前端下拉）
    case_name: Optional[str] = None
    case_country: Optional[str] = None
    case_type: Optional[str] = None
    # 仍保留 collection 供多租户场景，不对外强调
    collection: str = "regulations"


class CompanySyncRequest(BaseModel):
    aiword_company_id: str = Field(..., description="aiword organizations.id")
    name: str = Field(..., description="公司名称")
    slug: str = Field(..., description="公司 slug")
    knowledge_collection: str = Field(..., description="知识库 collection")
    is_active: bool = True
    is_default: bool = False


class CollectionRequest(BaseModel):
    collection: str = "regulations"


class ChatReplyOptions(BaseModel):
    domain: str = "system_record_writing"
    knowledge_category: str = "program"
    top_k: int = Field(default=0, ge=0, le=30)
    min_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    max_reply_chars: int = Field(default=280, ge=80, le=600)
    max_detail_chars: int = Field(default=0, ge=0, le=6000)


class ChatReplyRequest(BaseModel):
    query: str
    group_id: str = ""
    message_id: str = ""
    trigger_type: str = "at_or_keyword"
    context: Optional[Dict[str, Any]] = None
    options: Optional[ChatReplyOptions] = None
    collection: str = "regulations"
    current_provider: Optional[str] = None


class ChatFeedbackRequest(BaseModel):
    request_id: str
    group_id: str = ""
    message_id: str = ""
    feedback_type: str = ""
    operator: str = ""
    corrected_answer: str = ""


class QuizGenerateSetRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    collection: str = "regulations"
    exam_track: str = "cn"
    # daily=日常考试；new_standard=新标发布（与体考类型正交）
    exam_category: str = Field(
        default="daily",
        validation_alias=AliasChoices("exam_category", "examCategory"),
    )
    title: str = ""
    category: str = ""
    difficulty: str = "medium"
    question_type: str = "single_choice"
    question_count: int = 20
    created_by: str = "system"
    # exam_category=project_case 时必填：project_cases.id
    project_case_id: Optional[int] = Field(default=None, alias="projectCaseId")


class QuizPracticeSetRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    collection: str = "regulations"
    exam_track: str = "cn"
    exam_category: str = Field(
        default="daily",
        validation_alias=AliasChoices("exam_category", "examCategory"),
    )
    category: str = ""
    difficulty: str = "medium"
    question_type: str = "single_choice"
    question_count: int = 20
    user_id: str = ""
    project_case_id: Optional[int] = Field(default=None, alias="projectCaseId")


class QuizIngestByAIRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    collection: str = "regulations"
    exam_track: str = "cn"
    exam_category: str = Field(
        default="daily",
        validation_alias=AliasChoices("exam_category", "examCategory"),
    )
    target_count: int = 50
    review_mode: str = "auto_apply"
    category: str = ""
    difficulty: str = "medium"
    question_type: str = "single_choice"
    created_by: str = "system"
    set_title: str = ""
    project_case_id: Optional[int] = Field(default=None, alias="projectCaseId")
    # 由 aiword 从 app_configs 注入；缺省则 aicheckword 内用产品默认占比
    ingest_knowledge_weights: Optional[List[float]] = None
    ingest_question_type_weights: Optional[List[float]] = None
    max_similar_frac: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class QuizRegulatoryHintRequest(BaseModel):
    exam_track: str = "cn"
    as_of: str = ""
    # 时间窗起点（YYYY-MM-DD）；缺省由服务端按 as_of 回推约 365 天
    since: str = ""


class QuizIngestJobSetIdRequest(BaseModel):
    set_id: int


class QuizBankQuestionPatchRequest(BaseModel):
    # quiz_questions 可编辑字段
    stem: Optional[str] = None
    options: Optional[List[str]] = None
    # Pydantic 无法区分 answer=None 与未传，因此用 answer_present 显式声明是否更新
    answer_present: bool = False
    answer: Any = None
    explanation: Optional[str] = None
    evidence: Optional[List[Dict[str, Any]]] = None
    status: Optional[str] = None  # active / inactive
    # quiz_question_bank 可编辑字段（对该 question_id 在本 collection 下的所有 bank 行生效）
    exam_track: Optional[str] = None
    category: Optional[str] = None
    question_type: Optional[str] = None
    difficulty: Optional[str] = None
    is_active: Optional[bool] = None


class QuizAttemptStartRequest(BaseModel):
    collection: str = "regulations"
    user_id: str = ""
    mode: str = "practice"


class QuizSubmitAnswersRequest(BaseModel):
    """考试交卷 submit；collection 可由调用方传入，省略则从 attempt 行读取。"""
    model_config = ConfigDict(extra="ignore")
    answers: List[Dict[str, Any]] = Field(default_factory=list)
    collection: Optional[str] = None


class QuizPracticeSubmitRequest(BaseModel):
    """练习提交：attempt_id 放在 JSON body（与 aiword 一致）；也兼容 camelCase attemptId。"""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    collection: str = "regulations"
    user_id: str = Field(default="", alias="userId")
    set_id: Optional[int] = Field(default=None, ge=1, alias="setId")
    attempt_id: Optional[int] = Field(default=None, ge=1, alias="attemptId")
    answers: List[Dict[str, Any]] = Field(default_factory=list)

    @field_validator("set_id", "attempt_id", mode="before")
    @classmethod
    def _blank_str_to_none(cls, v: Any) -> Any:
        # 兼容上游/前端传空串："" 不应触发 int_parsing，应按未传处理
        if v is None:
            return None
        if isinstance(v, str) and not v.strip():
            return None
        return v


class QuizUpsertGradingRuleRequest(BaseModel):
    collection: str = "regulations"
    paper_id: Optional[int] = None
    question_id: int
    answer_key: Any
    rubric: Optional[Dict[str, Any]] = None
    updated_by: str = "system"


def _merge_review_context(
    base: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Optional[Dict[str, Any]]:
    ctx = dict(base or {})
    for k, v in kwargs.items():
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        ctx[k] = v
    return ctx or None


def _extract_first_json_object(text: str) -> Dict[str, Any]:
    s = (text or "").strip()
    if not s:
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _chat_bearer_token(authorization: Optional[str]) -> str:
    raw = (authorization or "").strip()
    if not raw:
        return ""
    if raw.lower().startswith("bearer "):
        return raw[7:].strip()
    return raw


def _ensure_chat_api_authorized(authorization: Optional[str]) -> None:
    expected = (getattr(settings, "chat_api_auth_token", "") or "").strip()
    # 未配置 token 时放行，便于本地联调
    if not expected:
        return
    got = _chat_bearer_token(authorization)
    if not got or got != expected:
        raise HTTPException(status_code=401, detail="unauthorized")


def _chat_normalize_answer_text(text: Any) -> str:
    """LLM 有时输出字面量 \\n，统一为换行便于展示。"""
    s = str(text or "").strip()
    if "\\n" in s:
        s = s.replace("\\r\\n", "\n").replace("\\n", "\n")
    return s


def _chat_as_str_list(val: Any, *, limit: int = 8) -> List[str]:
    out: List[str] = []
    if isinstance(val, list):
        for x in val:
            s = str(x or "").strip()
            if s and s not in out:
                out.append(s)
    elif isinstance(val, str) and val.strip():
        for part in re.split(r"[,，;；\n]", val):
            s = part.strip()
            if s and s not in out:
                out.append(s)
    return out[:limit]


def _chat_retrieval_plan(intent: Dict[str, Any], query: str) -> Dict[str, Any]:
    """由意图分析产出检索计划（查询扩展 + 主题词），通用适配各业务域，不单靠仓库等硬编码。"""
    q = str(intent.get("normalized_question") or query or "").strip() or (query or "").strip()
    search_queries = _chat_as_str_list(intent.get("search_queries"), limit=6)
    if not search_queries:
        search_queries = [
            q,
            f"{q} 程序文件 管理制度 体系运行记录",
            f"{q} SOP 规程 记录 填写要求",
        ]
    boost_terms = _chat_as_str_list(intent.get("topic_keywords") or intent.get("relevant_topics"), limit=10)
    penalty_terms = _chat_as_str_list(intent.get("avoid_topics") or intent.get("irrelevant_topics"), limit=8)
    return {
        "search_queries": search_queries,
        "boost_terms": boost_terms,
        "penalty_terms": penalty_terms,
        "normalized_question": q,
    }


def _chat_ref_domain_score(query: str, ref: Dict[str, Any], ctx: Dict[str, Any]) -> float:
    """向量分 + 业务域关键词加减分（文件名与片段正文均参与）。"""
    base = float(ref.get("score") or 0.0)
    blob = f"{ref.get('file_name') or ''}\n{(ref.get('content') or '')[:1200]}"
    for term in ctx.get("boost_terms") or []:
        if term and term in blob:
            base += 0.14
    for term in ctx.get("penalty_terms") or []:
        if term and term in blob:
            base -= 0.42
    q = (query or "").strip()
    if q and q[:8] in blob:
        base += 0.05
    return base


def _chat_retrieve_program_scored_refs(
    agent: Any,
    query: str,
    *,
    allowed_category: str,
    top_k: int,
    plan: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """程序文件检索：多查询扩展 + category 过滤 + 主题词加减分（计划由 LLM 意图分析生成）。"""
    ctx = plan if isinstance(plan, dict) else _chat_retrieval_plan({}, query)
    pool: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def _push(doc: Any, score: float) -> None:
        md = getattr(doc, "metadata", {}) or {}
        if str(md.get("category") or "").strip() != allowed_category:
            return
        content = (getattr(doc, "page_content", "") or "").strip()
        file_name = str(md.get("source_file") or "未知")
        key = f"{file_name}\n{content[:200]}"
        if key in seen:
            return
        seen.add(key)
        pool.append(
            {
                "content": content,
                "score": float(score) if score is not None else 0.0,
                "file_name": file_name,
                "category": str(md.get("category") or ""),
            }
        )

    over_k = max(top_k * 6, 24)
    for sq in ctx.get("search_queries") or [query]:
        sq = (sq or "").strip()
        if not sq:
            continue
        try:
            for pair in agent.kb.search_with_scores(sq, top_k=over_k) or []:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    _push(pair[0], pair[1])
        except Exception:
            pass
        try:
            for doc in agent.kb.search_by_category(sq, category=allowed_category, top_k=over_k) or []:
                _push(doc, 0.55)
        except Exception:
            pass

    pool.sort(key=lambda r: _chat_ref_domain_score(query, r, ctx), reverse=True)
    return pool[:top_k]


def _chat_rerank_refs_with_llm(
    query: str,
    candidates: List[Dict[str, Any]],
    *,
    top_k: int,
    current_provider: Optional[str] = None,
    client_llm: Optional[Any] = None,
    header_provider: str = "",
) -> tuple[List[Dict[str, Any]], float, str]:
    """用 LLM 从候选片段中挑选与用户问题真正相关的程序文件依据（通用，非关键词表）。"""
    if not candidates:
        return [], 0.0, "无候选片段"
    if len(candidates) <= top_k:
        ctx = {"boost_terms": [], "penalty_terms": []}
        score = _chat_ref_domain_score(query, candidates[0], ctx) if candidates else 0.0
        return candidates, score, "候选较少，跳过重排"

    lines: List[str] = []
    for i, r in enumerate(candidates[:14], start=1):
        snippet = (r.get("content") or "")[:320].replace("\n", " ")
        lines.append(f"[{i}] 文件: {r.get('file_name') or '未知'}\n片段: {snippet}")
    prompt = (
        "你是知识库检索质检员。用户要问的是医疗器械质量管理体系「运行记录怎么写」。\n"
        f"用户问题：{query}\n\n"
        "下列是从程序文件知识库召回的候选片段。请判断哪些片段**确实**能用来回答该问题"
        "（制度要求、记录项目、填写要点等），与问题业务域无关的（如问仓库管理却只有软件确认）不要选。\n\n"
        + "\n\n".join(lines)
        + "\n\n仅输出 JSON：\n"
        "{\n"
        '  "selected": [1, 3],\n'
        '  "scores": {"1": 0.92, "2": 0.05},\n'
        '  "reason": "一句话说明选择依据"\n'
        "}\n"
        "selected 为相关条目编号（至少 0 个，最多 "
        f"{top_k}"
        " 个）；scores 为各编号 0~1 相关度。"
    )
    out = _extract_first_json_object(
        _invoke_chat_llm_text(
            prompt,
            current_provider=current_provider,
            client_llm=client_llm,
            header_provider=header_provider,
        ).text
    )
    selected_raw = out.get("selected") if isinstance(out, dict) else []
    scores_map = out.get("scores") if isinstance(out, dict) else {}
    reason = str(out.get("reason") or "").strip() if isinstance(out, dict) else ""

    selected_idx: List[int] = []
    if isinstance(selected_raw, list):
        for x in selected_raw:
            try:
                n = int(x)
                if 1 <= n <= len(candidates):
                    selected_idx.append(n)
            except (TypeError, ValueError):
                continue
    selected_idx = list(dict.fromkeys(selected_idx))[:top_k]

    if not selected_idx:
        return [], 0.0, reason or "LLM 未选中任何相关程序文件片段"

    picked: List[Dict[str, Any]] = []
    rel_scores: List[float] = []
    for n in selected_idx:
        ref = dict(candidates[n - 1])
        key = str(n)
        try:
            rel = float((scores_map or {}).get(key, scores_map.get(str(n), 0.0)) or 0.0)
        except (TypeError, ValueError):
            rel = 0.0
        ref["relevance_score"] = max(0.0, min(1.0, rel))
        picked.append(ref)
        if rel > 0:
            rel_scores.append(rel)
    top_rel = max(rel_scores) if rel_scores else 0.72
    return picked, top_rel, reason or "已按语义重排"


_CHAT_INVOKABLE_PROVIDERS = frozenset({"openai", "deepseek", "lingyi", "ollama", "tongyi", "cursor"})
_CHAT_FALLBACK_PROVIDER = "deepseek"


@dataclass
class _ChatLlmInvokeResult:
    text: str
    provider_used: str
    fallback_note: Optional[str] = None


def _resolve_chat_provider(
    raw: Optional[str],
    *,
    header_provider: str = "",
) -> tuple[str, Optional[str]]:
    """钉钉/联调聊天：解析实际 provider；Cursor 可用但较慢，不自动回退。"""
    p = (raw or header_provider or getattr(settings, "provider", "") or "deepseek").strip().lower()
    if not p:
        return "deepseek", None
    if p in _CHAT_INVOKABLE_PROVIDERS:
        return p, None
    fb = (getattr(settings, "provider", "") or "deepseek").strip().lower()
    if fb not in _CHAT_INVOKABLE_PROVIDERS:
        fb = "deepseek"
    return fb, f"不支持的 provider={p}，已改用 {fb}"


def _invoke_chat_llm_text(
    prompt_text: str,
    current_provider: Optional[str] = None,
    *,
    client_llm: Optional[Any] = None,
    header_provider: str = "",
) -> _ChatLlmInvokeResult:
    from src.core.llm_factory import ClientLlmConfig, invoke_chat_direct

    provider, _ = _resolve_chat_provider(current_provider, header_provider=header_provider)
    cl = client_llm if isinstance(client_llm, ClientLlmConfig) else None
    if provider == "cursor":
        from src.core.cursor_agent import complete_task

        return _ChatLlmInvokeResult(
            text=complete_task(prompt_text, client_llm=cl if cl and cl.has_any() else None),
            provider_used="cursor",
        )

    def _call(prov: str, use_cl: Optional[ClientLlmConfig]) -> str:
        return invoke_chat_direct(
            prompt_text,
            temperature=0.1,
            provider=prov,
            model=None,
            client_llm=use_cl if use_cl and use_cl.has_any() else None,
        )

    try:
        return _ChatLlmInvokeResult(
            text=_call(provider, cl),
            provider_used=provider,
        )
    except Exception as primary_err:
        if provider == _CHAT_FALLBACK_PROVIDER:
            raise
        err_short = str(primary_err).strip().replace("\n", " ")[:200]
        try:
            return _ChatLlmInvokeResult(
                text=_call(_CHAT_FALLBACK_PROVIDER, None),
                provider_used=_CHAT_FALLBACK_PROVIDER,
                fallback_note=f"{provider} 调用失败，已回退 DeepSeek：{err_short}",
            )
        except Exception as fallback_err:
            raise RuntimeError(
                f"{provider} 调用失败（{err_short}），DeepSeek 回退也失败（{fallback_err}）"
            ) from fallback_err


def _classify_chat_intent(
    query: str,
    current_provider: Optional[str] = None,
    *,
    client_llm: Optional[Any] = None,
    header_provider: str = "",
) -> Dict[str, Any]:
    prompt = (
        "你是医疗器械体系运行记录问答的前置分析器。仅输出 JSON。\n"
        "用户问题：\n"
        f"{query}\n\n"
        "判断用户意图，并**同时**给出用于检索程序文件知识库的查询计划（须贴合用户业务域，"
        "如问仓库管理员应检索仓库/仓储/物料管理制度，而非无关的软件确认程序）。\n"
        "返回：\n"
        "{\n"
        '  "is_system_record_writing": true/false,\n'
        '  "is_date_or_personnel_question": true/false,\n'
        '  "normalized_question": "归一化后的完整问题",\n'
        '  "search_queries": ["检索句1", "检索句2", "检索句3"],\n'
        '  "topic_keywords": ["相关主题词1", "相关主题词2"],\n'
        '  "avoid_topics": ["应排除的无关主题1"],\n'
        '  "reason": "一句话原因"\n'
        "}\n"
        "规则：\n"
        "1) 只有「体系运行记录怎么写/怎么填/记录项怎么写」相关才算 is_system_record_writing=true。\n"
        "2) 涉及日期/人员/签字/审核/批准等即 is_date_or_personnel_question=true。\n"
        "3) search_queries 写 3～5 条中文检索句，包含可能的制度名称、业务域、记录类型；"
        "不要编造不存在的文件编号。\n"
        "4) topic_keywords / avoid_topics 用于过滤明显跑题的召回结果。\n"
    )
    out = _extract_first_json_object(
        _invoke_chat_llm_text(
            prompt,
            current_provider=current_provider,
            client_llm=client_llm,
            header_provider=header_provider,
        ).text
    )
    if out:
        return out
    q = (query or "").strip()
    low = q.lower()
    date_or_personnel = any(x in low for x in ("日期", "什么时候", "人员", "谁签", "谁审核", "谁批准", "写谁"))
    system_record = any(x in low for x in ("体系", "记录", "仓库", "设备", "电脑", "人事", "培训", "采购", "台账"))
    return {
        "is_system_record_writing": bool(system_record),
        "is_date_or_personnel_question": bool(date_or_personnel),
        "normalized_question": q,
        "search_queries": [q, f"{q} 程序文件 管理制度 体系运行记录"],
        "topic_keywords": [],
        "avoid_topics": [],
        "reason": "fallback",
    }


@app.get("/")
def root():
    return {"service": "注册文档审核 Agent", "version": "1.0.0", "status": "running"}


@app.get("/health")
def health():
    """兼容上游健康检查：aiword 默认探测 /health。"""
    return {"status": "ok", "service": "aicheckword"}


@app.get("/docs", include_in_schema=False)
def custom_swagger_ui_html():
    return _swagger_ui_html_with_cdn_fallback(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
    )


@app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
def swagger_ui_redirect():
    return get_swagger_ui_oauth2_redirect_html()


@app.get("/status")
def agent_status(req: Request, collection: str = "regulations"):
    agent = get_agent(_resolve_request_collection(req, collection))
    return agent.get_status()


@app.post("/admin/companies/sync")
def admin_companies_sync(body: CompanySyncRequest):
    try:
        row = upsert_company_mapping(
            aiword_company_id=body.aiword_company_id,
            name=body.name,
            slug=body.slug,
            knowledge_collection=body.knowledge_collection,
            is_active=bool(body.is_active),
            is_default=bool(body.is_default),
        )
        return {"ok": True, "data": row}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.delete("/admin/companies/sync/{aiword_company_id}")
def admin_companies_sync_delete(aiword_company_id: str):
    try:
        ok = delete_company_by_aiword_id(aiword_company_id)
        return {"ok": True, "data": {"deleted": bool(ok)}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/train/upload")
async def train_upload(
    req: Request,
    files: List[UploadFile] = File(...),
    collection: str = Form("regulations"),
    category: str = Form("regulation"),
):
    collection = _resolve_request_collection(req, collection)
    agent = get_agent(collection)
    results = []

    for file in files:
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            result = agent.train(tmp_path, category=category)
            result["original_filename"] = file.filename
            results.append(result)
        except Exception as e:
            results.append({
                "status": "error",
                "original_filename": file.filename,
                "message": str(e),
            })
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    total_chunks = sum(r.get("chunks_added", 0) for r in results)
    return {
        "status": "success",
        "files_processed": len(results),
        "total_chunks_added": total_chunks,
        "details": results,
    }


@app.post("/train/directory")
def train_directory(
    req: Request,
    dir_path: str = Form(...),
    collection: str = Form("regulations"),
    category: str = Form("regulation"),
):
    if not Path(dir_path).exists():
        raise HTTPException(status_code=404, detail=f"目录不存在：{dir_path}")
    collection = _resolve_request_collection(req, collection)
    agent = get_agent(collection)
    result = agent.train(dir_path, category=category)
    return result


@app.post("/train/project-cases/create")
def train_project_case_create(
    req: Request,
    collection: str = Form("regulations"),
    case_name: str = Form(...),
    product_name: str = Form(""),
    registration_country: str = Form(""),
    registration_type: str = Form(""),
    registration_component: str = Form(""),
    project_form: str = Form(""),
    scope_of_application: str = Form(""),
    case_name_en: str = Form(""),
    product_name_en: str = Form(""),
    registration_country_en: str = Form(""),
    document_language: str = Form("zh"),
    project_key: str = Form(""),
):
    collection = _resolve_request_collection(req, collection)
    nm = str(case_name or "").strip()
    if not nm:
        raise HTTPException(status_code=400, detail="case_name 不能为空")
    try:
        case_id = create_project_case(
            collection=collection,
            case_name=nm,
            product_name=product_name or "",
            registration_country=registration_country or "",
            registration_type=registration_type or "",
            registration_component=registration_component or "",
            project_form=project_form or "",
            scope_of_application=scope_of_application or "",
            case_name_en=case_name_en or "",
            product_name_en=product_name_en or "",
            registration_country_en=registration_country_en or "",
            document_language=document_language or "zh",
            project_key=project_key or "",
        )
        row = get_project_case(int(case_id))
        return {"ok": True, "data": {"case_id": int(case_id), "case": row or {}}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/train/project-cases/upload")
async def train_project_case_upload(
    req: Request,
    files: List[UploadFile] = File(...),
    collection: str = Form("regulations"),
    case_id: int = Form(...),
):
    collection = _resolve_request_collection(req, collection)
    case = get_project_case(int(case_id))
    if not case or str(case.get("collection") or "").strip() != collection:
        raise HTTPException(status_code=404, detail="case_id 不存在或不属于当前 collection")
    agent = get_agent(collection)
    results = []
    for f in files:
        suffix = Path(str(f.filename or "")).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            raw = await f.read()
            tmp.write(raw)
            tmp_path = tmp.name
        try:
            docs = load_single_file(tmp_path)
            chunks = split_documents(docs)
            n = agent.kb.add_documents(
                chunks,
                file_name=str(f.filename or Path(tmp_path).name),
                category="project_case",
                case_id=int(case_id),
            )
            results.append(
                {
                    "status": "success",
                    "original_filename": str(f.filename or ""),
                    "chunks_added": int(n or 0),
                }
            )
        except Exception as e:
            results.append(
                {
                    "status": "error",
                    "original_filename": str(f.filename or ""),
                    "message": str(e),
                }
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    return {
        "ok": True,
        "data": {
            "collection": collection,
            "case_id": int(case_id),
            "files_processed": len(results),
            "total_chunks_added": sum(int(x.get("chunks_added") or 0) for x in results),
            "details": results,
        },
    }


@app.post("/review/upload")
async def review_upload(
    req: Request,
    files: List[UploadFile] = File(...),
    collection: str = Form("regulations"),
    project_id: Optional[int] = Form(None),
    review_context_json: str = Form(""),
    system_prompt: str = Form(""),
    user_prompt: str = Form(""),
    extra_instructions: str = Form(""),
    project_name: str = Form(""),
    project_name_en: str = Form(""),
    product_name: str = Form(""),
    product_name_en: str = Form(""),
    model: str = Form(""),
    model_en: str = Form(""),
    registration_country: str = Form(""),
    registration_country_en: str = Form(""),
    registration_type: str = Form(""),
    registration_component: str = Form(""),
    project_form: str = Form(""),
    document_language: str = Form(""),
    scope_of_application: str = Form(""),
    basic_info_text: str = Form(""),
    system_functionality_text: str = Form(""),
    current_provider: str = Form(""),
):
    collection = _resolve_request_collection(req, collection)
    agent = get_agent(collection)
    reports = []
    parsed_ctx: Dict[str, Any] = {}
    if (review_context_json or "").strip():
        import json
        try:
            parsed_ctx = json.loads(review_context_json)
            if not isinstance(parsed_ctx, dict):
                raise ValueError("review_context_json 必须是 JSON 对象")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"review_context_json 解析失败：{e}")
    merged_ctx = _merge_review_context(
        parsed_ctx,
        project_name=project_name,
        project_name_en=project_name_en,
        product_name=product_name,
        product_name_en=product_name_en,
        model=model,
        model_en=model_en,
        registration_country=registration_country,
        registration_country_en=registration_country_en,
        registration_type=registration_type,
        registration_component=registration_component,
        project_form=project_form,
        document_language=document_language,
        scope_of_application=scope_of_application,
        basic_info_text=basic_info_text,
        system_functionality_text=system_functionality_text,
        current_provider=current_provider,
    )

    for file in files:
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            report = agent.review(
                tmp_path,
                project_id=project_id,
                review_context=merged_ctx,
                system_prompt=(system_prompt or None),
                user_prompt=(user_prompt or None),
                extra_instructions=(extra_instructions or None),
                display_file_name=file.filename,
            )
            report["original_filename"] = file.filename
            reports.append(report)
        except Exception as e:
            reports.append({
                "file_name": file.filename,
                "error": str(e),
            })
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    return {"reports": reports, "total_files": len(reports)}


@app.post("/review/text")
def review_text(req: Request, request: TextReviewRequest):
    request.collection = _resolve_request_collection(req, request.collection)
    agent = get_agent(request.collection)
    merged_ctx = _merge_review_context(
        request.review_context,
        project_name=request.project_name,
        project_name_en=request.project_name_en,
        product_name=request.product_name,
        product_name_en=request.product_name_en,
        model=request.model,
        model_en=request.model_en,
        registration_country=request.registration_country,
        registration_country_en=request.registration_country_en,
        registration_type=request.registration_type,
        registration_component=request.registration_component,
        project_form=request.project_form,
        document_language=request.document_language,
        scope_of_application=request.scope_of_application,
        basic_info_text=request.basic_info_text,
        system_functionality_text=request.system_functionality_text,
        current_provider=request.current_provider,
    )
    report = agent.review_text(
        request.text,
        request.file_name,
        project_id=request.project_id,
        review_context=merged_ctx,
        system_prompt=request.system_prompt,
        user_prompt=request.user_prompt,
        extra_instructions=request.extra_instructions,
    )
    return report


@app.post("/knowledge/search")
def search_knowledge(req: Request, request: KnowledgeQueryRequest):
    request.collection = _resolve_request_collection(req, request.collection)
    agent = get_agent(request.collection)
    # 仅按指定维度查询；以维度组合为检索种子，再按 metadata 二次过滤
    extra_terms = []
    for x in (
        request.project_name,
        request.project_name_en,
        request.product_name,
        request.product_name_en,
        request.model,
        request.model_en,
        request.registration_country,
        request.registration_country_en,
        request.registration_type,
        request.registration_component,
        request.project_form,
        request.document_language,
        request.case_name,
        request.case_country,
        request.case_type,
    ):
        if isinstance(x, str) and x.strip():
            extra_terms.append(x.strip())
    q = (request.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="query 不能为空")
    required_terms = [
        request.registration_country.strip(),
        request.registration_type.strip(),
        request.registration_component.strip(),
        request.project_form.strip(),
        request.document_language.strip(),
    ]
    if any(not x for x in required_terms):
        raise HTTPException(
            status_code=400,
            detail="registration_country/registration_type/registration_component/project_form/document_language 为必填且不能为空",
        )
    merged_query = "\n".join([q] + required_terms + extra_terms)
    results = agent.search_knowledge(
        merged_query,
        top_k=30,
        use_checkpoints=False,
    )

    def _eq(a: Any, b: str) -> bool:
        if a is None:
            return False
        return str(a).strip().lower() == str(b).strip().lower()

    filtered = []
    for r in results:
        meta = r.get("metadata") or {}
        # metadata 无该字段时不做强拒绝，避免历史数据缺元信息导致全空；有字段时必须匹配
        checks = []
        if meta.get("registration_country") is not None:
            checks.append(_eq(meta.get("registration_country"), request.registration_country))
        if meta.get("registration_type") is not None:
            checks.append(_eq(meta.get("registration_type"), request.registration_type))
        if meta.get("registration_component") is not None:
            checks.append(_eq(meta.get("registration_component"), request.registration_component))
        if meta.get("project_form") is not None:
            checks.append(_eq(meta.get("project_form"), request.project_form))
        if meta.get("document_language") is not None:
            checks.append(_eq(meta.get("document_language"), request.document_language))
        if meta.get("project_name") is not None and request.project_name:
            checks.append(_eq(meta.get("project_name"), request.project_name))
        if meta.get("project_name_en") is not None and request.project_name_en:
            checks.append(_eq(meta.get("project_name_en"), request.project_name_en))
        if meta.get("product_name") is not None and request.product_name:
            checks.append(_eq(meta.get("product_name"), request.product_name))
        if meta.get("product_name_en") is not None and request.product_name_en:
            checks.append(_eq(meta.get("product_name_en"), request.product_name_en))
        if meta.get("model") is not None and request.model:
            checks.append(_eq(meta.get("model"), request.model))
        if meta.get("model_en") is not None and request.model_en:
            checks.append(_eq(meta.get("model_en"), request.model_en))
        if meta.get("case_name") is not None and request.case_name:
            checks.append(_eq(meta.get("case_name"), request.case_name))
        if meta.get("case_country") is not None and request.case_country:
            checks.append(_eq(meta.get("case_country"), request.case_country))
        if meta.get("case_type") is not None and request.case_type:
            checks.append(_eq(meta.get("case_type"), request.case_type))
        if all(checks) if checks else True:
            filtered.append(r)

    return {
        "conditions": {
            "query": q,
            "registration_country": request.registration_country,
            "registration_type": request.registration_type,
            "registration_component": request.registration_component,
            "project_form": request.project_form,
            "document_language": request.document_language,
            "project_name": request.project_name,
            "project_name_en": request.project_name_en,
            "product_name": request.product_name,
            "product_name_en": request.product_name_en,
            "model": request.model,
            "model_en": request.model_en,
            "case_name": request.case_name,
            "case_country": request.case_country,
            "case_type": request.case_type,
        },
        "results": filtered,
        "total": len(filtered),
    }


@app.get("/knowledge/search/options")
def knowledge_search_options(req: Request, collection: str = Query("regulations", description="知识库名称")):
    """返回查询参数可选值（与页面一致）"""
    collection = _resolve_request_collection(req, collection)
    def _uniq(values: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for v in values or []:
            s = str(v or "").strip()
            if not s:
                continue
            k = s.casefold()
            if k in seen:
                continue
            seen.add(k)
            out.append(s)
        return out

    dims = get_dimension_options()
    reg_country = _uniq(dims.get("registration_countries") or ["中国", "美国", "欧盟"])
    project_forms = _uniq(dims.get("project_forms") or ["Web", "APP", "PC"])
    project_name_opts: List[str] = []
    project_name_en_opts: List[str] = []
    product_name_opts: List[str] = []
    product_name_en_opts: List[str] = []
    model_opts: List[str] = []
    model_en_opts: List[str] = []
    project_list: List[Dict[str, str]] = []
    try:
        _proj_seen = set()
        for p in list_projects(collection) or []:
            if not isinstance(p, dict):
                continue
            name = str(p.get("name") or "").strip()
            name_en = str(p.get("name_en") or "").strip()
            product = str(p.get("product_name") or "").strip()
            product_en = str(p.get("product_name_en") or "").strip()
            model = str(p.get("model") or "").strip()
            model_en = str(p.get("model_en") or "").strip()
            p_country = str(p.get("registration_country") or "").strip()
            p_type = str(p.get("registration_type") or "").strip()
            p_component = str(p.get("registration_component") or "").strip()
            p_form = str(p.get("project_form") or "").strip()

            project_name_opts.append(name)
            project_name_en_opts.append(name_en)
            product_name_opts.append(product)
            product_name_en_opts.append(product_en)
            model_opts.append(model)
            model_en_opts.append(model_en)

            sk = (
                name.casefold(),
                p_country.casefold(),
                p_type.casefold(),
                p_component.casefold(),
                p_form.casefold(),
            )
            if any(sk) and sk not in _proj_seen:
                _proj_seen.add(sk)
                project_list.append(
                    {
                        "project_name": name,
                        "project_name_en": name_en,
                        "product_name": product,
                        "product_name_en": product_en,
                        "model": model,
                        "model_en": model_en,
                        "registration_country": p_country,
                        "registration_type": p_type,
                        "registration_component": p_component,
                        "project_form": p_form,
                    }
                )
    except Exception:
        pass
    project_name_opts = _uniq(project_name_opts)
    project_name_en_opts = _uniq(project_name_en_opts)
    product_name_opts = _uniq(product_name_opts)
    product_name_en_opts = _uniq(product_name_en_opts)
    model_opts = _uniq(model_opts)
    model_en_opts = _uniq(model_en_opts)

    case_name_opts: List[str] = []
    case_country_opts: List[str] = []
    case_type_opts: List[str] = []
    try:
        for c in list_project_cases(collection) or []:
            if not isinstance(c, dict):
                continue
            case_name_opts.append(str(c.get("case_name") or ""))
            case_country_opts.append(str(c.get("registration_country") or ""))
            case_type_opts.append(str(c.get("registration_type") or ""))
    except Exception:
        # 选项接口应尽量可用：案例读取失败时只返回基础维度
        pass
    case_name_opts = _uniq(case_name_opts)
    case_country_opts = _uniq(case_country_opts)
    case_type_opts = _uniq(case_type_opts)
    return {
        # 兼容旧调用：仍保留顶层键
        "registration_country": reg_country,
        "registration_type": REGISTRATION_TYPES,
        "registration_component": REGISTRATION_COMPONENTS,
        "project_form": project_forms,
        "document_language": ["zh", "en", "both"],
        "project_list": project_list,
        "project_name_options": project_name_opts,
        "project_name_en_options": project_name_en_opts,
        "product_name_options": product_name_opts,
        "product_name_en_options": product_name_en_opts,
        "model_options": model_opts,
        "model_en_options": model_en_opts,
        "project_name": {"type": "string", "required": False, "description": "可选，自由文本"},
        "project_name_en": {"type": "string", "required": False, "description": "可选，自由文本"},
        "product_name": {"type": "string", "required": False, "description": "可选，自由文本"},
        "product_name_en": {"type": "string", "required": False, "description": "可选，自由文本"},
        "model": {"type": "string", "required": False, "description": "可选，自由文本"},
        "model_en": {"type": "string", "required": False, "description": "可选，自由文本"},
        "case_name": case_name_opts,
        "case_country": case_country_opts,
        "case_type": case_type_opts,
        # 推荐新调用：统一字段定义，一次取全参数规则
        "fields": {
            "query": {"type": "string", "required": True, "description": "自由文本检索词"},
            "registration_country": {"type": "enum", "required": True, "options": reg_country},
            "registration_type": {"type": "enum", "required": True, "options": REGISTRATION_TYPES},
            "registration_component": {"type": "enum", "required": True, "options": REGISTRATION_COMPONENTS},
            "project_form": {"type": "enum", "required": True, "options": project_forms},
            "document_language": {"type": "enum", "required": True, "options": ["zh", "en", "both"]},
            "project_name": {"type": "enum", "required": False, "options": project_name_opts},
            "project_name_en": {"type": "enum", "required": False, "options": project_name_en_opts},
            "product_name": {"type": "enum", "required": False, "options": product_name_opts},
            "product_name_en": {"type": "enum", "required": False, "options": product_name_en_opts},
            "model": {"type": "enum", "required": False, "options": model_opts},
            "model_en": {"type": "enum", "required": False, "options": model_en_opts},
            "case_name": {"type": "enum", "required": False, "options": case_name_opts},
            "case_country": {"type": "enum", "required": False, "options": case_country_opts},
            "case_type": {"type": "enum", "required": False, "options": case_type_opts},
        },
    }


@app.post("/knowledge/clear")
def clear_knowledge(req: Request, request: CollectionRequest):
    request.collection = _resolve_request_collection(req, request.collection)
    agent = get_agent(request.collection)
    return agent.clear_knowledge()


@app.post("/api/chat/reply/generate")
def chat_reply_generate(
    body: ChatReplyRequest,
    http_request: Request,
    authorization: Optional[str] = Header(default=None),
    x_request_id: Optional[str] = Header(default=None),
):
    from .draft_integration import _header_provider, _parse_client_llm

    _ensure_chat_api_authorized(authorization)
    started = time.time()
    request_id = (x_request_id or "").strip() or f"chat-{uuid.uuid4().hex[:12]}"
    query = (body.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query 不能为空")

    client_llm = _parse_client_llm(http_request)
    hp = _header_provider(http_request)
    eff_provider, provider_note = _resolve_chat_provider(
        body.current_provider, header_provider=hp
    )

    opts = body.options or ChatReplyOptions()
    allowed_domain = (getattr(settings, "chat_allowed_domain", "system_record_writing") or "system_record_writing").strip()
    allowed_category = (getattr(settings, "chat_allowed_knowledge_category", "program") or "program").strip()
    requested_domain = (opts.domain or "").strip() or allowed_domain
    requested_category = (opts.knowledge_category or "").strip() or allowed_category
    if requested_domain != allowed_domain or requested_category != allowed_category:
        return {
            "request_id": request_id,
            "need_human": True,
            "answer": "",
            "confidence": 0.0,
            "reason": "首版仅支持体系运行记录写作，且仅基于程序文件知识库。",
            "references": [],
            "model_used": (getattr(settings, "llm_model", "") or "").strip(),
            "latency_ms": int((time.time() - started) * 1000),
        }

    intent = _classify_chat_intent(
        query,
        current_provider=eff_provider,
        client_llm=client_llm if client_llm.has_any() else None,
        header_provider=hp,
    )
    if not bool(intent.get("is_system_record_writing")):
        return {
            "request_id": request_id,
            "need_human": True,
            "answer": "",
            "confidence": 0.0,
            "reason": "问题不在首版体系记录填写范围内。",
            "references": [],
            "model_used": (getattr(settings, "llm_model", "") or "").strip(),
            "latency_ms": int((time.time() - started) * 1000),
        }
    if bool(intent.get("is_date_or_personnel_question")):
        return {
            "request_id": request_id,
            "need_human": True,
            "answer": "",
            "confidence": 0.0,
            "reason": "首版不自动回答日期与人员填写问题。",
            "references": [],
            "model_used": (getattr(settings, "llm_model", "") or "").strip(),
            "latency_ms": int((time.time() - started) * 1000),
        }

    body.collection = _resolve_request_collection(http_request, body.collection or "regulations")
    agent = get_agent(body.collection or "regulations")
    top_k = int(opts.top_k or 0) or int(getattr(settings, "chat_default_top_k", 6) or 6)
    top_k = max(2, min(20, top_k))
    min_similarity = float(opts.min_similarity or 0.0) or float(getattr(settings, "chat_min_similarity", 0.55) or 0.55)
    conf_threshold = float(getattr(settings, "chat_confidence_threshold", 0.65) or 0.65)
    max_summary_chars = max(80, min(600, int(opts.max_reply_chars or 280)))
    max_detail_chars = int(opts.max_detail_chars or 0) or int(
        getattr(settings, "chat_max_detail_chars", 2400) or 2400
    )
    max_detail_chars = max(200, min(6000, max_detail_chars))

    retrieval_plan = _chat_retrieval_plan(intent, query)
    search_query = str(retrieval_plan.get("normalized_question") or query).strip() or query
    recall_k = max(top_k * 3, 12)
    candidates = _chat_retrieve_program_scored_refs(
        agent,
        search_query,
        allowed_category=allowed_category,
        top_k=recall_k,
        plan=retrieval_plan,
    )
    if len(candidates) < 2:
        return {
            "request_id": request_id,
            "need_human": True,
            "answer": "",
            "confidence": 0.0,
            "reason": "未检索到足够的程序文件依据。",
            "references": [],
            "model_used": (getattr(settings, "llm_model", "") or "").strip(),
            "latency_ms": int((time.time() - started) * 1000),
        }
    scored_refs, top_score, rerank_reason = _chat_rerank_refs_with_llm(
        search_query,
        candidates,
        top_k=top_k,
        current_provider=eff_provider,
        client_llm=client_llm if client_llm.has_any() else None,
        header_provider=hp,
    )
    if len(scored_refs) < 2:
        return {
            "request_id": request_id,
            "need_human": True,
            "answer": "",
            "confidence": 0.0,
            "reason": rerank_reason or "召回片段与问题业务域不匹配，建议人工确认。",
            "references": [],
            "model_used": (getattr(settings, "llm_model", "") or "").strip(),
            "latency_ms": int((time.time() - started) * 1000),
        }
    if top_score < min_similarity:
        return {
            "request_id": request_id,
            "need_human": True,
            "answer": "",
            "confidence": top_score,
            "reason": rerank_reason or "程序文件匹配度不足，建议人工确认。",
            "references": [],
            "model_used": (getattr(settings, "llm_model", "") or "").strip(),
            "latency_ms": int((time.time() - started) * 1000),
        }

    refs_prompt = []
    for idx, r in enumerate(scored_refs[:top_k], start=1):
        snippet = (r["content"] or "")[:400]
        refs_prompt.append(
            f"[{idx}] 文件: {r['file_name']}\n分类: {r['category']}\n片段: {snippet}"
        )
    answer_prompt = (
        "你是医疗器械质量管理体系记录填写助手。\n"
        "任务：根据“程序文件”片段回答用户“怎么写记录”的问题。\n"
        "限制：\n"
        "1) 仅可依据给定程序文件片段回答；无法确认就 need_human=true。\n"
        "2) 仅使用与用户业务域一致的片段；已预筛选，若仍不对则 need_human=true。\n"
        "3) 首版不回答日期写法和人员填写问题。\n"
        "4) 不输出法规结论，不编造不存在的字段。\n"
        "5) 输出简洁、可执行，列清记录项与依据来源文件。\n\n"
        f"用户问题：{query}\n\n"
        "程序文件片段：\n"
        + "\n\n".join(refs_prompt)
        + "\n\n仅输出 JSON：\n"
        "{\n"
        '  "need_human": false,\n'
        '  "answer_summary": "2～4 句简化总结（群消息用，列关键记录项）",\n'
        '  "answer_detail": "详细版：分条说明怎么写、依据哪份程序文件（可用换行）",\n'
        '  "answer": "与 answer_summary 相同（兼容）",\n'
        '  "confidence": 0.0,\n'
        '  "reason": "一句话原因"\n'
        "}"
    )
    answer_invoke = _invoke_chat_llm_text(
        answer_prompt,
        current_provider=eff_provider,
        client_llm=client_llm if client_llm.has_any() else None,
        header_provider=hp,
    )
    llm_out = _extract_first_json_object(answer_invoke.text)
    llm_provider_used = answer_invoke.provider_used
    if answer_invoke.fallback_note:
        provider_note = (
            f"{provider_note}；{answer_invoke.fallback_note}"
            if provider_note
            else answer_invoke.fallback_note
        )
    answer_summary = _chat_normalize_answer_text(
        llm_out.get("answer_summary") or llm_out.get("answer") or ""
    )[:max_summary_chars]
    answer_detail = _chat_normalize_answer_text(
        llm_out.get("answer_detail") or llm_out.get("answer") or answer_summary
    )[:max_detail_chars]
    answer = answer_summary
    try:
        confidence = float(llm_out.get("confidence", 0.0) or 0.0)
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    need_human = bool(llm_out.get("need_human"))
    if not answer_summary and not answer_detail:
        need_human = True
    if confidence < conf_threshold:
        need_human = True

    references = []
    for r in scored_refs[: min(3, len(scored_refs))]:
        references.append(
            {
                "source": "knowledge_base",
                "file_name": r.get("file_name") or "未知",
                "category": r.get("category") or allowed_category,
                "snippet": (r.get("content") or "")[:120],
            }
        )
    from src.core.llm_factory import resolve_model_for_provider

    cl_for_model = (
        client_llm
        if client_llm.has_any() and llm_provider_used == eff_provider
        else None
    )
    model_label = resolve_model_for_provider(llm_provider_used, client_llm=cl_for_model)
    out_body: Dict[str, Any] = {
        "request_id": request_id,
        "need_human": need_human,
        "answer": "" if need_human else answer_summary,
        "answer_summary": "" if need_human else answer_summary,
        "answer_detail": "" if need_human else answer_detail,
        "confidence": confidence,
        "reason": str(llm_out.get("reason") or ""),
        "references": references,
        "model_used": f"{llm_provider_used}:{model_label}",
        "effective_provider": eff_provider,
        "llm_provider_used": llm_provider_used,
        "latency_ms": int((time.time() - started) * 1000),
    }
    if provider_note:
        out_body["provider_note"] = provider_note
    return out_body


@app.post("/api/chat/feedback")
def chat_feedback(
    request: ChatFeedbackRequest,
    authorization: Optional[str] = Header(default=None),
):
    _ensure_chat_api_authorized(authorization)
    return {
        "ok": True,
        "request_id": request.request_id,
        "accepted": True,
        "message": "feedback received",
    }


@app.post("/quiz/sets/generate")
def quiz_generate_set(http_req: Request, request: QuizGenerateSetRequest):
    if request.exam_track not in EXAM_TRACKS:
        raise HTTPException(status_code=400, detail=f"exam_track 不支持: {request.exam_track}")
    try:
        request.collection = _resolve_request_collection(http_req, request.collection)
        data = quiz_service.generate_set(
            collection=request.collection,
            exam_track=request.exam_track,
            set_type="exam",
            created_by=request.created_by or "teacher",
            title=request.title,
            category=request.category,
            difficulty=request.difficulty,
            question_type=request.question_type,
            question_count=request.question_count,
            status="draft",
            exam_category=request.exam_category,
            project_case_id=request.project_case_id,
        )
        return {"ok": True, "data": data}
    except QuizRequestError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/quiz/practice/generate-set")
def quiz_generate_practice_set(http_req: Request, request: QuizPracticeSetRequest):
    if request.exam_track not in EXAM_TRACKS:
        raise HTTPException(status_code=400, detail=f"exam_track 不支持: {request.exam_track}")
    try:
        request.collection = _resolve_request_collection(http_req, request.collection)
        data = quiz_service.generate_set(
            collection=request.collection,
            exam_track=request.exam_track,
            set_type="practice",
            created_by=request.user_id or "student",
            title=f"{EXAM_TRACKS.get(request.exam_track)}练习套题",
            category=request.category,
            difficulty=request.difficulty,
            question_type=request.question_type,
            question_count=request.question_count,
            status="published",
            exam_category=request.exam_category,
            project_case_id=request.project_case_id,
        )
        sid = int(data.get("id") or data.get("set_id") or 0) if isinstance(data, dict) else 0
        if sid > 0 and isinstance(data, dict):
            att = quiz_service.start_attempt(
                collection=request.collection,
                set_id=sid,
                user_id=request.user_id or "",
                mode="practice",
            )
            aid = int(att.get("attempt_id") or 0) if isinstance(att, dict) else 0
            if aid > 0:
                data["attempt_id"] = aid
                data["practice_session_id"] = str(aid)
                data["session_id"] = str(aid)
                data["practiceSessionId"] = str(aid)
                data["sessionId"] = str(aid)
        return {"ok": True, "data": data}
    except QuizRequestError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/quiz/bank/ingest-by-ai")
def quiz_ingest_bank_by_ai(http_req: Request, request: QuizIngestByAIRequest):
    if request.exam_track not in EXAM_TRACKS:
        raise HTTPException(status_code=400, detail=f"exam_track 不支持: {request.exam_track}")
    try:
        request.collection = _resolve_request_collection(http_req, request.collection)
        data = quiz_service.ingest_bank_by_ai(
            collection=request.collection,
            exam_track=request.exam_track,
            target_count=request.target_count,
            created_by=request.created_by,
            review_mode=request.review_mode,
            category=request.category,
            difficulty=request.difficulty,
            question_type=request.question_type,
            set_title=request.set_title,
            exam_category=request.exam_category,
            ingest_knowledge_weights=request.ingest_knowledge_weights,
            ingest_question_type_weights=request.ingest_question_type_weights,
            max_similar_frac=request.max_similar_frac,
            project_case_id=request.project_case_id,
        )
        return {"ok": True, "data": data}
    except QuizRequestError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/quiz/tools/project-cases")
def quiz_tools_project_cases(req: Request, collection: str = Query("regulations", description="知识库 collection")):
    """已训练入库（knowledge_docs 有 project_case 块）的项目案例列表，供考试中心下拉。"""
    try:
        collection = _resolve_request_collection(req, collection)
        rows = quiz_service.list_ready_project_cases_for_quiz(collection=collection)
        return {"ok": True, "data": {"cases": rows, "collection": collection}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/quiz/tools/regulatory-updates-hint")
def quiz_regulatory_updates_hint(request: QuizRegulatoryHintRequest):
    """「新标发布」备考：按体考类型给出需关注的法规/标准/指南更新方向（模型归纳，非官方清单）。"""
    if request.exam_track not in EXAM_TRACKS:
        raise HTTPException(status_code=400, detail=f"exam_track 不支持: {request.exam_track}")
    try:
        data = quiz_service.regulatory_updates_hint(
            exam_track=request.exam_track,
            as_of=request.as_of or None,
            since=request.since or None,
        )
        return {"ok": True, "data": data}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/quiz/bank/ingest-jobs/{job_id}")
def quiz_get_ingest_job(job_id: int):
    try:
        return {"ok": True, "data": quiz_service.get_ingest_job(job_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/quiz/bank/ingest-jobs/{job_id}/set-id")
def quiz_set_ingest_job_set_id(job_id: int, request: QuizIngestJobSetIdRequest):
    try:
        return {"ok": True, "data": quiz_service.set_ingest_job_set_id(job_id, request.set_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/quiz/sets/{set_id}/publish")
def quiz_publish_set(set_id: int):
    try:
        return {"ok": True, "data": quiz_service.publish_set(set_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/quiz/sets/{set_id}/review-by-ai")
def quiz_review_set_by_ai(
    req: Request,
    set_id: int,
    collection: str = Query("regulations"),
    created_by: str = Query("system"),
):
    try:
        collection = _resolve_request_collection(req, collection)
        return {"ok": True, "data": quiz_service.start_review_set_by_ai_job(collection=collection, set_id=set_id, created_by=created_by)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/quiz/sets/review-jobs/{job_id}")
def quiz_review_job_get(job_id: int):
    try:
        return {"ok": True, "data": quiz_service.fetch_review_job(job_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/quiz/sets")
def quiz_sets_list(
    req: Request,
    collection: str = Query("regulations"),
    set_type: str = Query(""),
    exam_track: str = Query(""),
    status: str = Query(""),
    q: str = Query(""),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    try:
        collection = _resolve_request_collection(req, collection)
        return {
            "ok": True,
            "data": quiz_service.list_sets(
                collection=collection,
                set_type=set_type,
                exam_track=exam_track,
                status=status,
                q=q,
                limit=limit,
                offset=offset,
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/quiz/sets/{set_id}")
def quiz_sets_get(set_id: int):
    try:
        return {"ok": True, "data": quiz_service.get_set(set_id=set_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/quiz/sets/{set_id}")
def quiz_sets_delete(set_id: int):
    try:
        return {"ok": True, "data": quiz_service.delete_set(set_id=set_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/quiz/exams/{set_id}/start", include_in_schema=False)
def quiz_start_exam(set_id: int, request: QuizAttemptStartRequest):
    raise HTTPException(status_code=404, detail="已迁移到 aiword 本地考试：请改用 aiword /api/exam-center/student/exams/start-local")


@app.post("/quiz/practice/submit")
def quiz_submit_practice(
    req: Request,
    body: QuizPracticeSubmitRequest,
    attempt_id: Optional[int] = Query(default=None, ge=1, description="兼容旧调用：也可仅 query 传 attempt_id"),
):
    body.collection = _resolve_request_collection(req, body.collection)
    aid = attempt_id or body.attempt_id
    # 兼容 aiword 仅传 set_id 的场景：自动创建一次 practice attempt，再提交答案
    if not aid and body.set_id:
        created = quiz_service.start_attempt(
            collection=body.collection,
            set_id=int(body.set_id),
            user_id=(body.user_id or ""),
            mode="practice",
        )
        aid = int((created or {}).get("attempt_id") or 0) or None
    if not aid:
        raise HTTPException(
            status_code=422,
            detail="attempt_id 必填：请放在 JSON body（attempt_id 或 attemptId）或 query 参数 attempt_id；或传 set_id 让后端自动创建练习会话",
        )
    try:
        data = quiz_service.submit_answers_and_grade(
            attempt_id=int(aid),
            answers=body.answers or [],
            collection=(body.collection or "").strip() or None,
        )
        return {"ok": True, "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/quiz/exams/{attempt_id}/submit", include_in_schema=False)
def quiz_submit_exam(attempt_id: int, request: QuizSubmitAnswersRequest):
    raise HTTPException(status_code=404, detail="已迁移到 aiword 本地考试：请改用 aiword /api/exam-center/student/exams/submit-local")


@app.post("/quiz/attempts/{attempt_id}/grade-by-cache")
def quiz_grade_by_cache(req: Request, attempt_id: int, collection: str = Query("regulations"), paper_id: Optional[int] = Query(None)):
    if quiz_service.is_exam_attempt(attempt_id=attempt_id):
        raise HTTPException(status_code=404, detail="exam attempt 已迁移到 aiword 本地考试，当前接口不再提供")
    try:
        collection = _resolve_request_collection(req, collection)
        data = quiz_service.grade_attempt_by_cache(collection=collection, attempt_id=attempt_id, paper_id=paper_id)
        return {"ok": True, "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/quiz/attempts/{attempt_id}/auto-grade")
def quiz_auto_grade(req: Request, attempt_id: int, collection: str = Query("regulations"), paper_id: Optional[int] = Query(None)):
    if quiz_service.is_exam_attempt(attempt_id=attempt_id):
        raise HTTPException(status_code=404, detail="exam attempt 已迁移到 aiword 本地考试，当前接口不再提供")
    try:
        collection = _resolve_request_collection(req, collection)
        data = quiz_service.auto_grade_attempt(collection=collection, attempt_id=attempt_id, paper_id=paper_id)
        return {"ok": True, "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/quiz/attempts/{attempt_id}/grading-status")
def quiz_attempt_grading_status(attempt_id: int):
    """异步主观题阅卷进度：阅卷中 grading / 完成后 state=graded 且含总分。"""
    if quiz_service.is_exam_attempt(attempt_id=attempt_id):
        raise HTTPException(status_code=404, detail="exam attempt 已迁移到 aiword 本地考试，当前接口不再提供")
    try:
        return {"ok": True, "data": quiz_service.get_attempt_grading_status(attempt_id=attempt_id)}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/quiz/attempts/{attempt_id}/answers")
def quiz_attempt_answers(req: Request, attempt_id: int, collection: str = Query("regulations")):
    """
    返回 attempt 的作答明细（题干/选项/标准答案/学生答案等），用于 aiword 的考试/练习详情展示。

    说明：
    - is_correct 可能为空/不准确（若未触发自动判分）；前端可优先展示“标准答案 vs 学生答案”。
    - collection 参数仅用于兼容未来按 collection 分库的扩展；当前实现从题库表聚合即可。
    """
    if quiz_service.is_exam_attempt(attempt_id=attempt_id):
        raise HTTPException(status_code=404, detail="exam attempt 已迁移到 aiword 本地考试，当前接口不再提供")
    try:
        collection = _resolve_request_collection(req, collection)
        _ = collection  # 预留参数，避免上层固定传参导致 422
        return {"ok": True, "data": quiz_service.get_attempt_answers_with_questions(attempt_id=attempt_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/quiz/grading-rules/upsert")
def quiz_upsert_grading_rules(req: Request, request: QuizUpsertGradingRuleRequest):
    try:
        request.collection = _resolve_request_collection(req, request.collection)
        data = quiz_service.upsert_grading_rule(
            collection=request.collection,
            paper_id=request.paper_id,
            question_id=request.question_id,
            answer_key=request.answer_key,
            rubric=request.rubric or {},
            updated_by=request.updated_by,
        )
        return {"ok": True, "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class QuizPaperGradeByAIItem(BaseModel):
    question_id: str = Field(..., description="题目ID")
    question_type: str = Field(default="", description="题型（可空）")
    stem: str = Field(default="", description="题干")
    options: List[Any] = Field(default_factory=list, description="选项（可空）")
    user_answer: Any = Field(default=None, description="学生答案")


class QuizPaperGradeByAIRequest(BaseModel):
    collection: str = Field(default="regulations")
    exam_track: str = Field(default="cn")
    attempt_id: str = Field(..., description="aiword 本地 attempt_id（用于回传）")
    assignment_id: str = Field(default="", description="可选：aiword assignment_id")
    items: List[QuizPaperGradeByAIItem] = Field(default_factory=list)


@app.post("/quiz/grading/paper-by-ai")
def quiz_grade_paper_by_ai(req: Request, request: QuizPaperGradeByAIRequest):
    """整卷主观判分：创建异步 job，返回 job_id。证据定位仅需文件名。"""
    if request.exam_track not in EXAM_TRACKS:
        raise HTTPException(status_code=400, detail=f"exam_track 不支持: {request.exam_track}")
    if not request.items:
        raise HTTPException(status_code=422, detail="items 不能为空")
    try:
        request.collection = _resolve_request_collection(req, request.collection)
        data = quiz_service.start_paper_grading_job(
            collection=request.collection,
            exam_track=request.exam_track,
            attempt_id=request.attempt_id,
            items=[dict(x) for x in request.items],
        )
        return {"ok": True, "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/quiz/grading/jobs/{job_id}")
def quiz_grade_paper_job(job_id: str):
    """查询整卷主观判分 job 状态与结果。"""
    try:
        data = quiz_service.get_paper_grading_job(job_id=str(job_id))
        return {"ok": True, "data": data}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/quiz/bank/tracks")
def quiz_bank_tracks(req: Request, collection: str = "regulations"):
    try:
        collection = _resolve_request_collection(req, collection)
        return {"ok": True, "data": quiz_service.get_tracks_inventory(collection)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/quiz/wrongbook")
def quiz_wrongbook_get(
    req: Request,
    collection: str = Query("regulations"),
    user_id: str = Query("", description="学生 user_id，由网关注入"),
    limit: int = Query(80, ge=1, le=200),
):
    try:
        collection = _resolve_request_collection(req, collection)
        return {"ok": True, "data": quiz_service.student_wrongbook(collection=collection, user_id=user_id, limit=limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/quiz/student/assignments", include_in_schema=False)
def quiz_student_assignments_compat():
    """兼容 aiword 多路径探测：考试任务由 aiword 管理；返回空列表（HTTP 200），避免日志误报 404。"""
    return {"ok": True, "data": {"assignments": []}}


@app.get("/quiz/me/assignments", include_in_schema=False)
def quiz_me_assignments_compat():
    """同上：/me 别名。"""
    return {"ok": True, "data": {"assignments": []}}


@app.get("/quiz/student/exams", include_in_schema=False)
def quiz_student_exams_compat():
    """兼容 aiword 探测的旧路径；考试列表由 aiword 本地提供。"""
    return {"ok": True, "data": {"assignments": []}}


@app.get("/quiz/student/unpracticed-bank")
def quiz_student_unpracticed_bank(
    req: Request,
    collection: str = Query("regulations"),
    user_id: str = Query("", description="学生 user_id，由网关注入"),
    exam_track: str = Query(""),
    limit: int = Query(100, ge=0, le=300),
):
    try:
        collection = _resolve_request_collection(req, collection)
        return {
            "ok": True,
            "data": quiz_service.student_unpracticed_bank(
                collection=collection, user_id=user_id, exam_track=exam_track, limit=limit
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/quiz/bank/questions")
def quiz_bank_questions_list(
    req: Request,
    collection: str = Query("regulations"),
    exam_track: str = Query(""),
    q: str = Query(""),
    category: str = Query(""),
    question_type: str = Query(""),
    difficulty: str = Query(""),
    is_active: Optional[bool] = Query(True),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    try:
        collection = _resolve_request_collection(req, collection)
        data = quiz_service.admin_list_bank_questions(
            collection=collection,
            exam_track=exam_track,
            q=q,
            category=category,
            question_type=question_type,
            difficulty=difficulty,
            is_active=is_active,
            limit=limit,
            offset=offset,
        )
        return {"ok": True, "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/quiz/bank/questions/{question_id}")
def quiz_bank_question_patch(
    req: Request,
    question_id: int,
    request: QuizBankQuestionPatchRequest,
    collection: str = Query("regulations"),
):
    try:
        collection = _resolve_request_collection(req, collection)
        data = quiz_service.admin_patch_bank_question(
            collection=collection,
            question_id=int(question_id),
            stem=request.stem,
            options=request.options,
            answer_present=bool(request.answer_present),
            answer=request.answer,
            explanation=request.explanation,
            evidence=request.evidence,
            status=request.status,
            exam_track=request.exam_track,
            category=request.category,
            question_type=request.question_type,
            difficulty=request.difficulty,
            is_active=request.is_active,
        )
        return {"ok": True, "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/quiz/bank/questions/{question_id}")
def quiz_bank_question_delete(req: Request, question_id: int, collection: str = Query("regulations")):
    try:
        collection = _resolve_request_collection(req, collection)
        data = quiz_service.admin_delete_bank_question(collection=collection, question_id=int(question_id))
        return {"ok": True, "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/quiz/stats/overview", include_in_schema=False)
def quiz_stats_overview(collection: str = "regulations"):
    raise HTTPException(status_code=404, detail="统计已迁移到 aiword：请改用 /api/exam-center/stats/overview")


@app.get("/quiz/stats/options", include_in_schema=False)
def quiz_stats_options(collection: str = "regulations"):
    """统计端下拉：学生列表、考试任务（assignment_id）列表，来源为 quiz_attempts 聚合。"""
    raise HTTPException(status_code=404, detail="统计已迁移到 aiword：请改用 /api/exam-center/stats/options")


@app.get("/quiz/stats/student/{student_id}", include_in_schema=False)
def quiz_stats_student(student_id: str, collection: str = "regulations"):
    raise HTTPException(status_code=404, detail="统计已迁移到 aiword：请改用 /api/exam-center/stats/student/{student_id}")


@app.get("/quiz/stats/exam/{assignment_id}", include_in_schema=False)
def quiz_stats_exam(assignment_id: int, collection: str = "regulations"):
    raise HTTPException(status_code=404, detail="统计已迁移到 aiword：请改用 /api/exam-center/stats/exam/{assignment_id}")


@app.get("/quiz/config/effective")
def quiz_effective_config():
    """调试接口：返回当前 API 进程内生效的刷题 AI 配置（便于确认 DB 配置是否已加载）。"""
    try:
        return {
            "ok": True,
            "data": {
                "quiz_provider": getattr(settings, "quiz_provider", "") or "",
                "quiz_llm_model": getattr(settings, "quiz_llm_model", "") or "",
                "quiz_temperature": float(getattr(settings, "quiz_temperature", 0.2) or 0.2),
                "fallback_provider": (settings.provider or ""),
                "fallback_llm_model": (settings.llm_model or ""),
                "embedding_model": (settings.embedding_model or ""),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/checklist/generate")
def generate_checklist(
    req: Request,
    collection: str = Form("regulations"),
    base_checklist: Optional[str] = Form(None),
):
    collection = _resolve_request_collection(req, collection)
    agent = get_agent(collection)
    checklist = agent.generate_checklist(base_checklist=base_checklist)
    return {"checklist": checklist, "total_points": len(checklist)}


@app.post("/checklist/train")
def train_checklist(
    req: Request,
    collection: str = Form("regulations"),
    checklist_json: str = Form(...),
):
    import json
    try:
        checklist = json.loads(checklist_json)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"JSON 解析失败：{e}")
    collection = _resolve_request_collection(req, collection)
    agent = get_agent(collection)
    count = agent.train_checklist(checklist)
    return {"status": "success", "chunks_added": count, "total_points": len(checklist)}


@app.get("/knowledge/collections")
def list_collections():
    agent = get_agent()
    collections = agent.kb.list_collections()
    return {"collections": collections}


# ── 集成 API（供 AISystem 网关调用） ──


class IntegrationKdocsReviewRequest(BaseModel):
    """通过金山文档链接进行审核（核心集成接口）"""
    kdocs_url: str
    kdocs_download_url: str = ""
    file_name: str = ""
    collection: str = "regulations"
    project_id: Optional[int] = None
    aiword_task_id: str = ""
    integration_task_id: str = ""


class AuditReportPointPatchRequest(BaseModel):
    """PATCH 审核点时仅提交需要修改的字段；未出现的键表示不修改。"""

    description: Optional[str] = None
    suggestion: Optional[str] = None
    action: Optional[str] = None
    modify_docs: Optional[List[str]] = None
    severity: Optional[str] = None
    regulation_ref: Optional[str] = None
    location: Optional[str] = None
    category: Optional[str] = None
    save_correction: Optional[bool] = None
    correction_kind: Optional[str] = None
    feed_to_kb: bool = True
    false_positive_reason: Optional[str] = None
    deprecation_note: Optional[str] = None
    replacement_point: Optional["AuditReportPointReplacementRequest"] = None
    collection: Optional[str] = None


class AuditReportPointReplacementRequest(BaseModel):
    """弃用时可选插入的替代审核点。"""

    category: Optional[str] = None
    location: Optional[str] = None
    description: str = ""
    regulation_ref: Optional[str] = None
    suggestion: Optional[str] = None
    severity: Optional[str] = "low"
    modify_docs: Optional[List[str]] = None
    action: Optional[str] = "立即修改"


class AuditReportPointCorrectionRequest(BaseModel):
    """POST 纠正：误报 / 弃用 / 修订，并可写入反馈向量库。"""

    correction_kind: str = "revision"
    feed_to_kb: bool = True
    description: Optional[str] = None
    suggestion: Optional[str] = None
    action: Optional[str] = None
    modify_docs: Optional[List[str]] = None
    severity: Optional[str] = None
    regulation_ref: Optional[str] = None
    location: Optional[str] = None
    category: Optional[str] = None
    false_positive_reason: Optional[str] = None
    deprecation_note: Optional[str] = None
    replacement_point: Optional[AuditReportPointReplacementRequest] = None
    collection: Optional[str] = None


class IntegrationRecordRequest(BaseModel):
    """记录审核结果（供跟踪追溯）"""
    reviewResult: dict = {}
    aiwordTaskId: str = ""
    integrationTaskId: str = ""


@app.post("/api/integration/review-kdocs")
def integration_review_kdocs(http_req: Request, req: IntegrationKdocsReviewRequest):
    """
    核心集成接口：通过金山文档链接审核文档。
    1. 用 kdocs_download_url 通过金山文档开放平台提取纯文本
    2. 使用指定 collection + project_id 进行 AI 审核
    3. 生成带批注的 Word 文档（如果是 .docx）
    4. 返回审核结果 + 问题摘要
    """
    kdocs_url = (req.kdocs_url or "").strip()
    download_url = (req.kdocs_download_url or "").strip()
    fn = (req.file_name or "").strip() or "文档.docx"
    collection = _resolve_request_collection(http_req, (req.collection or "regulations").strip())

    if not kdocs_url and not download_url:
        raise HTTPException(status_code=400, detail="请提供金山文档链接（kdocs_url 或 kdocs_download_url）")

    actual_download_url = download_url or kdocs_url

    from src.core.kdocs_client import fetch_plaintext_from_url
    try:
        text = fetch_plaintext_from_url(actual_download_url, fn)
    except Exception as e:
        return {
            "status": "error",
            "message": f"金山文档内容提取失败: {e}",
            "aiword_task_id": req.aiword_task_id,
        }

    if not (text or "").strip():
        return {
            "status": "error",
            "message": "金山文档内容为空，无法审核",
            "aiword_task_id": req.aiword_task_id,
        }

    agent = get_agent(collection)

    review_context = None
    if req.project_id:
        review_context = agent._build_review_context(req.project_id)

    report = agent.review_text(
        text,
        file_name=fn,
        project_id=req.project_id,
        review_context=review_context,
    )

    report["original_filename"] = fn
    report["_kdocs_view_url"] = kdocs_url
    report["_kdocs_download_url"] = download_url
    report["aiword_task_id"] = req.aiword_task_id
    report["integration_task_id"] = req.integration_task_id

    has_comments_docx = False
    if fn.lower().endswith((".docx", ".doc")) and download_url:
        try:
            from src.core.kdocs_client import download_file_from_url
            from src.core.report_export import report_to_docx_with_comments
            raw_bytes = download_file_from_url(download_url)
            docx_with_comments = report_to_docx_with_comments(raw_bytes, report, author="AI审核")
            has_comments_docx = True
            report["_has_comments_docx"] = True
        except Exception:
            pass

    from src.core.db import save_audit_report, get_current_model_info
    mi = get_current_model_info()
    report_id = save_audit_report(collection, report, model_info=mi)
    report["_report_id"] = report_id

    points = report.get("audit_points", [])
    high_points = [p for p in points if (p.get("severity") or "").lower() == "high"]
    medium_points = [p for p in points if (p.get("severity") or "").lower() == "medium"]

    issue_summary_parts = []
    for p in high_points + medium_points:
        loc = (p.get("location") or "").strip()
        desc = (p.get("description") or "").strip()
        sev = (p.get("severity") or "").strip()
        line = f"[{sev}]"
        if loc:
            line += f" {loc}："
        line += f" {desc}"
        issue_summary_parts.append(line)

    issue_summary = ""
    if issue_summary_parts:
        issue_summary = "审核发现以下问题：\n" + "\n".join(issue_summary_parts)
        issue_summary += "\n详情请查看金山文档批注"

    has_issues = len(high_points) > 0 or len(medium_points) > 0
    audit_status = "审核不通过待修改" if has_issues else "审核通过"
    total = len(points)
    summary = f"审核完成：共 {total} 个审核点（高风险 {len(high_points)}，中风险 {len(medium_points)}）"

    return {
        "status": "completed",
        "auditStatus": audit_status,
        "summary": summary,
        "issueSummary": issue_summary,
        "hasIssues": has_issues,
        "totalPoints": total,
        "highCount": len(high_points),
        "mediumCount": len(medium_points),
        "reportId": report_id,
        "hasCommentDocx": has_comments_docx,
        "report": report,
        "aiwordTaskId": req.aiword_task_id,
        "integrationTaskId": req.integration_task_id,
    }


@app.get("/api/reports/{report_id}")
def api_get_audit_report(report_id: int):
    """按主键读取历史审核报告（含完整 report JSON）。"""
    from src.core.db import get_audit_report_by_id

    row = get_audit_report_by_id(report_id)
    if not row:
        raise HTTPException(status_code=404, detail="report not found")
    out = {k: v for k, v in row.items() if k != "report_json"}
    out["report"] = row.get("report") or {}
    return jsonable_encoder(out)


_CORRECTION_ONLY_PATCH_KEYS = frozenset(
    {
        "save_correction",
        "correction_kind",
        "feed_to_kb",
        "false_positive_reason",
        "deprecation_note",
        "replacement_point",
        "collection",
        "upload_id",
    }
)


def _patch_is_correction_request(patch: Dict[str, Any]) -> bool:
    return bool(patch.get("save_correction")) or bool(
        str(patch.get("correction_kind") or "").strip()
    )


def _correction_request_from_patch(patch: Dict[str, Any]) -> AuditReportPointCorrectionRequest:
    fields = set(AuditReportPointCorrectionRequest.model_fields.keys())
    data = {k: v for k, v in patch.items() if k in fields}
    if not str(data.get("correction_kind") or "").strip():
        data["correction_kind"] = "revision"
    return AuditReportPointCorrectionRequest(**data)


def _handle_audit_point_correction(
    http_req: Request,
    report_id: int,
    point_index: int,
    body: AuditReportPointCorrectionRequest,
    sub_report_index: int,
):
    """纠正审核点（误报/弃用/修订），写入 audit_corrections 并可选入库反馈向量库。"""
    from src.core.audit_correction import apply_audit_point_correction, persist_audit_point_correction
    from src.core.audit_report_utils import get_target_report_for_points
    from src.core.db import get_audit_report_by_id

    kind = (body.correction_kind or "revision").strip().lower()
    if kind not in ("revision", "false_positive", "deprecated"):
        raise HTTPException(status_code=400, detail="correction_kind must be revision, false_positive, or deprecated")

    row = get_audit_report_by_id(report_id)
    if not row:
        raise HTTPException(status_code=404, detail="report not found")
    root = row.get("report") or {}
    target_before = get_target_report_for_points(root, sub_report_index)
    points_before = target_before.get("audit_points") or []
    if point_index < 0 or point_index >= len(points_before):
        raise HTTPException(status_code=404, detail="point index out of range")

    original_snap = dict(points_before[point_index])
    repl = body.replacement_point.model_dump() if body.replacement_point else None

    try:
        target, corrected_for_log, inserted = apply_audit_point_correction(
            root,
            point_index,
            correction_kind=kind,
            sub_report_index=sub_report_index,
            feed_to_kb=body.feed_to_kb,
            description=body.description,
            suggestion=body.suggestion,
            action=body.action,
            modify_docs=body.modify_docs,
            severity=body.severity,
            regulation_ref=body.regulation_ref,
            location=body.location,
            category=body.category,
            false_positive_reason=body.false_positive_reason,
            deprecation_note=body.deprecation_note,
            replacement_point=repl,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    collection = _resolve_request_collection(
        http_req,
        (body.collection or row.get("collection") or "").strip() or "regulations",
    )
    file_name = (
        target.get("original_filename")
        or target.get("file_name")
        or row.get("file_name")
        or ""
    )
    persist_audit_point_correction(
        report_id,
        point_index,
        collection=collection,
        file_name=file_name,
        root_report=root,
        original_snap=original_snap,
        corrected_for_log=corrected_for_log,
        feed_to_kb=body.feed_to_kb,
    )

    points = target.get("audit_points") or []
    out_point = points[point_index] if point_index < len(points) else corrected_for_log
    return jsonable_encoder(
        {
            "ok": True,
            "report_id": report_id,
            "point_index": point_index,
            "sub_report_index": sub_report_index,
            "correction_kind": kind,
            "fed_to_kb": body.feed_to_kb,
            "collection": collection,
            "point": out_point,
            "inserted_point": inserted,
            "total_points": target.get("total_points"),
            "high_count": target.get("high_count"),
            "medium_count": target.get("medium_count"),
            "low_count": target.get("low_count"),
            "info_count": target.get("info_count"),
        }
    )


@app.patch("/api/reports/{report_id}/points/{point_index}")
def api_patch_audit_report_point(
    http_req: Request,
    report_id: int,
    point_index: int,
    body: AuditReportPointPatchRequest,
    sub_report_index: int = Query(0, ge=0),
):
    """按报告 ID 与审核点下标局部更新并写回 MySQL（批量报告用 sub_report_index 指定子报告）。"""
    from src.core.audit_report_utils import (
        aggregate_batch_report_totals,
        apply_point_field_updates,
        get_target_report_for_points,
        recount_severity,
    )
    from src.core.db import get_audit_report_by_id, update_audit_report

    patch = body.model_dump(exclude_unset=True)
    if not patch:
        raise HTTPException(status_code=400, detail="no fields to patch")
    if _patch_is_correction_request(patch):
        return _handle_audit_point_correction(
            http_req,
            report_id,
            point_index,
            _correction_request_from_patch(patch),
            sub_report_index,
        )

    row = get_audit_report_by_id(report_id)
    if not row:
        raise HTTPException(status_code=404, detail="report not found")
    root = row.get("report") or {}
    target = get_target_report_for_points(root, sub_report_index)
    points = target.get("audit_points") or []
    if point_index < 0 or point_index >= len(points):
        raise HTTPException(status_code=404, detail="point index out of range")
    plain_patch = {
        k: v for k, v in patch.items() if k not in _CORRECTION_ONLY_PATCH_KEYS
    }
    if not plain_patch:
        raise HTTPException(status_code=400, detail="no fields to patch")
    apply_point_field_updates(points[point_index], **plain_patch)
    recount_severity(target)
    if root.get("batch") and root.get("reports"):
        aggregate_batch_report_totals(root)
    update_audit_report(report_id, root)
    return jsonable_encoder(
        {
            "ok": True,
            "report_id": report_id,
            "point_index": point_index,
            "sub_report_index": sub_report_index,
            "point": points[point_index],
            "total_points": target.get("total_points"),
            "high_count": target.get("high_count"),
            "medium_count": target.get("medium_count"),
            "low_count": target.get("low_count"),
            "info_count": target.get("info_count"),
        }
    )


@app.post("/api/reports/{report_id}/points/{point_index}/correction")
def api_correct_audit_report_point(
    http_req: Request,
    report_id: int,
    point_index: int,
    body: AuditReportPointCorrectionRequest,
    sub_report_index: int = Query(0, ge=0),
):
    """纠正审核点（POST 兼容入口；推荐 PATCH + save_correction）。"""
    return _handle_audit_point_correction(
        http_req, report_id, point_index, body, sub_report_index
    )


@app.get("/api/reports/{report_id}/edit", response_class=HTMLResponse, include_in_schema=False)
def api_audit_report_light_editor(report_id: int):
    """极简 HTML 编辑页：按点加载与 PATCH 保存，避免 Streamlit 整页重跑。"""
    rid = int(report_id)
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>审核报告 #{rid} 点编辑</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 1rem; max-width: 52rem; }}
    label {{ display: block; margin-top: 0.6rem; font-weight: 600; }}
    textarea, input[type=text], select {{ width: 100%; box-sizing: border-box; }}
    textarea {{ min-height: 5rem; }}
    .row {{ display: flex; gap: 0.75rem; flex-wrap: wrap; align-items: center; margin-top: 0.5rem; }}
    .msg {{ margin-top: 0.75rem; padding: 0.5rem; border-radius: 4px; }}
    .ok {{ background: #e8f5e9; }}
    .err {{ background: #ffebee; }}
    code {{ font-size: 0.9em; }}
  </style>
</head>
<body>
  <h1>审核报告 <code>#{rid}</code></h1>
  <p>批量报告可改 URL 参数 <code>?sub_report_index=1</code>（从 0 起）。</p>
  <div class="row">
    <label>子报告 sub_report_index <input type="number" id="subIdx" value="0" min="0" style="width:6rem"/></label>
    <button type="button" id="btnReload">重新加载</button>
  </div>
  <label>审核点</label>
  <select id="ptSel"></select>
  <label>description</label>
  <textarea id="f_desc"></textarea>
  <label>suggestion</label>
  <textarea id="f_sug"></textarea>
  <label>action</label>
  <input type="text" id="f_action"/>
  <label>modify_docs（每行一条）</label>
  <textarea id="f_md" placeholder="每行一条"></textarea>
  <div class="row">
    <button type="button" id="btnSave">PATCH 保存</button>
    <span id="status"></span>
  </div>
  <div id="msg"></div>
<script>
(function () {{
  const reportId = {rid};
  const subInput = document.getElementById('subIdx');
  const ptSel = document.getElementById('ptSel');
  const fDesc = document.getElementById('f_desc');
  const fSug = document.getElementById('f_sug');
  const fAction = document.getElementById('f_action');
  const fMd = document.getElementById('f_md');
  const msg = document.getElementById('msg');
  let reportPayload = null;
  let points = [];

  function subReportIndex() {{
    const n = parseInt(subInput.value || '0', 10);
    return isNaN(n) || n < 0 ? 0 : n;
  }}

  function setMsg(text, ok) {{
    msg.textContent = text || '';
    msg.className = 'msg ' + (ok ? 'ok' : 'err');
  }}

  function pickTargetReport(root) {{
    const idx = subReportIndex();
    if (root.batch && root.reports && root.reports.length) {{
      const i = Math.min(idx, root.reports.length - 1);
      return root.reports[i];
    }}
    return root;
  }}

  function syncFormFromPoint(i) {{
    const p = points[i] || {{}};
    fDesc.value = p.description != null ? String(p.description) : '';
    fSug.value = p.suggestion != null ? String(p.suggestion) : '';
    fAction.value = p.action != null ? String(p.action) : '';
    const md = p.modify_docs;
    fMd.value = Array.isArray(md) ? md.join('\\n') : (md != null ? String(md) : '');
  }}

  async function loadReport() {{
    setMsg('', true);
    try {{
      const r = await fetch('/api/reports/' + reportId);
      if (!r.ok) throw new Error(await r.text());
      const data = await r.json();
      reportPayload = data.report || {{}};
      const t = pickTargetReport(reportPayload);
      points = t.audit_points || [];
      ptSel.innerHTML = '';
      points.forEach((_, i) => {{
        const o = document.createElement('option');
        o.value = String(i);
        o.textContent = '点 #' + i + (points[i].severity ? ' [' + points[i].severity + ']' : '');
        ptSel.appendChild(o);
      }});
      if (points.length) {{
        ptSel.value = '0';
        syncFormFromPoint(0);
      }} else {{
        fDesc.value = fSug.value = fAction.value = fMd.value = '';
      }}
      setMsg('已加载 ' + points.length + ' 个审核点', true);
    }} catch (e) {{
      setMsg('加载失败: ' + e, false);
    }}
  }}

  function modifyDocsValue() {{
    return fMd.value.split('\\n').map(s => s.trim()).filter(Boolean);
  }}

  async function savePoint() {{
    const i = parseInt(ptSel.value, 10);
    if (isNaN(i)) return;
    setMsg('', true);
    const body = {{
      description: fDesc.value,
      suggestion: fSug.value,
      action: fAction.value,
      modify_docs: modifyDocsValue(),
    }};
    const url = '/api/reports/' + reportId + '/points/' + i + '?sub_report_index=' + subReportIndex();
    try {{
      const r = await fetch(url, {{
        method: 'PATCH',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify(body),
      }});
      const txt = await r.text();
      if (!r.ok) throw new Error(txt);
      const data = JSON.parse(txt);
      points[i] = data.point;
      setMsg('已保存', true);
    }} catch (e) {{
      setMsg('保存失败: ' + e, false);
    }}
  }}

  const qs = new URLSearchParams(window.location.search);
  const s0 = qs.get('sub_report_index');
  if (s0 != null && s0 !== '') subInput.value = s0;

  ptSel.addEventListener('change', () => syncFormFromPoint(parseInt(ptSel.value, 10)));
  document.getElementById('btnReload').addEventListener('click', loadReport);
  document.getElementById('btnSave').addEventListener('click', savePoint);
  loadReport();
}})();
</script>
</body>
</html>"""
    return HTMLResponse(html)


@app.get("/api/integration/projects")
def integration_list_projects(collection: str = "regulations"):
    """获取 aicheckword 中的审核项目列表（供 AISystem 选择）"""
    from src.core.db import list_projects
    projects = list_projects(collection)
    return {
        "projects": [
            {
                "id": p["id"],
                "name": p.get("name", ""),
                "collection": p.get("collection", ""),
                "registrationCountry": p.get("registration_country", ""),
                "registrationType": p.get("registration_type", ""),
                "registrationComponent": p.get("registration_component", ""),
                "projectForm": p.get("project_form", ""),
                "productName": p.get("product_name", ""),
                "label": format_project_option_label(p),
            }
            for p in projects
        ]
    }


@app.post("/api/integration/record-review")
def integration_record_review(request: IntegrationRecordRequest):
    """记录从 AISystem 回传的审核结果（便于在 aicheckword 中追溯）"""
    return {
        "status": "recorded",
        "aiwordTaskId": request.aiwordTaskId,
        "integrationTaskId": request.integrationTaskId,
    }


@app.get("/api/integration/health")
def integration_health():
    return {"status": "ok", "service": "aicheckword"}


def _expose_only_public_schema_routes() -> None:
    """
    仅在 Swagger/OpenAPI 中公开指定接口。
    说明：不影响接口实际可调用性，只控制文档可见性。
    """
    public_paths = {"/knowledge/search", "/knowledge/search/options"}
    for route in app.routes:
        path = getattr(route, "path", "")
        if path in public_paths or path.startswith("/quiz/"):
            route.include_in_schema = True
        else:
            route.include_in_schema = False


_expose_only_public_schema_routes()


def start_server():
    import uvicorn

    try:
        log_cfg = _build_uvicorn_log_config_with_time()
    except Exception:
        log_cfg = None
    kwargs = {"app": app, "host": settings.api_host, "port": settings.api_port}
    if log_cfg is not None:
        kwargs["log_config"] = log_cfg
    uvicorn.run(**kwargs)


if __name__ == "__main__":
    start_server()
