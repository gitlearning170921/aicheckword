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
logger = logging.getLogger(__name__)


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
    create_project,
    create_project_case,
    get_project_case,
    upsert_company_mapping,
    delete_company_by_aiword_id,
    CompanyMappingConflictError,
)
from src.core.project_option_label import format_project_option_label
from src.core.document_loader import (
    _build_docx_headers_footers_plaintext,
    _build_docx_textbox_plaintext,
    load_single_file,
    split_documents,
)
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


def _configure_application_logging() -> None:
    """让 src.* 业务日志输出到 API 进程控制台（与 uvicorn 同一窗口）。"""
    level_name = (os.environ.get("AICHECKWORD_LOG_LEVEL") or "INFO").strip().upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        level = logging.INFO
    datefmt = "%Y-%m-%d %H:%M:%S"
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt=datefmt,
    )
    root = logging.getLogger()
    root.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        root.addHandler(handler)
    for name in ("src", "src.core", "src.core.quiz", "src.core.quiz.service"):
        logging.getLogger(name).setLevel(level)


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
    _configure_application_logging()
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


class DocumentControlValidateRequest(BaseModel):
    collection: str = "regulations"
    document_number: str = ""
    doc_type_code: str = ""
    title: str = ""
    project_code: str = ""


class DocumentControlParseRulesRequest(BaseModel):
    collection: str = "regulations"
    query: str = "文件控制程序 编号规则"
    text: str = ""


class DocumentControlTranslateTitleRequest(BaseModel):
    collection: str = "regulations"
    text: str = ""


class DocumentControlReleaseDateSuggestRequest(BaseModel):
    collection: str = "regulations"
    productName: str = ""
    fromVersion: str = ""
    toVersion: str = ""
    intermediateVersions: list[str] = Field(default_factory=list)
    targetVersion: Optional[str] = None
    registrationCountry: str = ""


class IntegrationCreateProjectBody(BaseModel):
    """创建 aicheckword 专属项目（与 Streamlit「项目与专属资料」新建表单一致）。"""

    collection: str = "regulations"
    name: str = Field(..., description="项目名称")
    registration_country: str = Field(..., description="注册国家")
    registration_type: str = Field(..., description="注册类别")
    registration_component: str = Field(..., description="注册组成")
    project_form: str = Field(..., description="项目形态")
    scope_of_application: str = ""
    product_name: str = ""
    name_en: str = ""
    product_name_en: str = ""
    registration_country_en: str = ""
    model: str = ""
    model_en: str = ""
    project_code: str = ""


class ChatReplyOptions(BaseModel):
    domain: str = "system_record_writing"
    knowledge_category: str = "program"
    top_k: int = Field(default=0, ge=0, le=30)
    min_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    max_reply_chars: int = Field(default=280, ge=80, le=8000)
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
    author_roles: List[str] = Field(default_factory=list, validation_alias=AliasChoices("author_roles", "authorRoles"))
    author_role_coverage: str = Field(
        default="balanced_union",
        validation_alias=AliasChoices("author_role_coverage", "authorRoleCoverage"),
    )


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
    author_roles: List[str] = Field(default_factory=list, validation_alias=AliasChoices("author_roles", "authorRoles"))
    author_role_coverage: str = Field(
        default="balanced_union",
        validation_alias=AliasChoices("author_role_coverage", "authorRoleCoverage"),
    )


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
    author_roles_present: bool = False
    author_roles: Optional[List[str]] = None


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


_DOCNO_RE = re.compile(r"([A-Z0-9]{2,12}-[A-Z0-9]{2,12}-\d{2,6})", re.I)
_VERSION_RE = re.compile(r"\b([A-Z]\/\d{1,2}|V(?:ER)?\.?\s*\d+(?:\.\d+)*)\b", re.I)
_TYPE_RE = re.compile(r"-([A-Z]{2,10})-\d{2,6}\b")


def _normalize_document_number_local(value: str) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    text = re.sub(r"[\s_]+", "-", text)
    text = text.replace("—", "-").replace("–", "-").replace("－", "-")
    text = re.sub(r"-{2,}", "-", text)
    return text.strip("-")


def _extract_document_control_meta(file_name: str, text_blob: str) -> Dict[str, Any]:
    source = "\n".join([str(file_name or ""), str(text_blob or "")])
    m_doc = _DOCNO_RE.search(source)
    raw_no = (m_doc.group(1) if m_doc else "").strip()
    doc_no = _normalize_document_number_local(raw_no)
    m_ver = _VERSION_RE.search(source)
    version = (m_ver.group(1) if m_ver else "").strip().upper()
    doc_type = ""
    if doc_no:
        tm = _TYPE_RE.search(doc_no)
        doc_type = (tm.group(1) if tm else "").strip().upper()
    confidence = 0.45
    if doc_no:
        confidence = 0.85
        if doc_no in _normalize_document_number_local(file_name):
            confidence = 0.95
    return {
        "document_number": doc_no,
        "version": version or "",
        "doc_type": doc_type or "",
        "extract_confidence": round(confidence, 2),
    }


_KNOWN_DOC_TYPE_CODES = (
    "SOP",
    "DHF",
    "QR",
    "SMP",
    "WI",
    "FR",
    "QP",
    "URS",
    "SRS",
    "IFU",
    "DMR",
    "PAP",
    "CAPA",
    "FMEA",
    "FRM",
    "REC",
    "CE",
    "CS",
)


def _rule_candidate_item(doc_type: str, *, name: str = "") -> Dict[str, Any]:
    typ = re.sub(r"[^A-Z0-9]", "", (doc_type or "").upper())
    if len(typ) < 2 or len(typ) > 12:
        return {}
    label = (name or "").strip() or f"{typ} 文件编号规则"
    return {
        "name": label,
        "docTypeCode": typ,
        "patternRegex": rf"^([A-Z0-9]{{2,12}})-{typ}-(\d{{2,6}})$",
        "renderTemplate": "{prefix}-{type}-{seq:03d}",
        "prefixSource": "from_project_code",
        "seqStart": 1,
        "seqPad": 3,
    }


def _append_rule_candidate(candidates: List[Dict[str, Any]], seen: set[str], doc_type: str, *, name: str = "") -> None:
    item = _rule_candidate_item(doc_type, name=name)
    typ = item.get("docTypeCode") if item else ""
    if not typ or typ in seen:
        return
    seen.add(typ)
    candidates.append(item)


def _parse_rule_candidates_from_text(text: str) -> List[Dict[str, Any]]:
    blob = str(text or "")
    if not blob.strip():
        return []
    candidates: List[Dict[str, Any]] = []
    seen: set[str] = set()

    # 三段式示例：PAPUWIS-DHF-001 / PROJECT-SOP-012
    for m in re.finditer(r"\b([A-Z0-9]{2,12})-([A-Z][A-Z0-9]{1,11})-(\d{2,6})\b", blob, re.I):
        _append_rule_candidate(candidates, seen, m.group(2))

    # 四段式示例：QR-SMP7.3-03-01（取首段类型码；与 PREFIX-TYPE-SEQ 区分）
    for m in re.finditer(
        r"\b([A-Z]{2,8})-(?:[A-Z]+\d[\w.-]*|[A-Z0-9]+(?:\.\d+)+)-\d{2,4}-\d{2,4}\b",
        blob,
        re.I,
    ):
        _append_rule_candidate(candidates, seen, m.group(1))

    # 中文条款：类型代号 / 文件类别
    for m in re.finditer(
        r"(?:类型代号|文件类别|类别代码|文件类型|代号)[：:\s]*([A-Z]{2,12})",
        blob,
        re.I,
    ):
        _append_rule_candidate(candidates, seen, m.group(1))

    # 常见缩写（须在编号/文件语境附近）
    ctx = blob
    for typ in _KNOWN_DOC_TYPE_CODES:
        if re.search(rf"(?:编号|文件|受控|记录|程序).{{0,40}}\b{typ}\b|\b{typ}\b.{{0,40}}(?:编号|文件|受控)", ctx, re.I):
            _append_rule_candidate(candidates, seen, typ)

    # 旧版兼容：TYPE-001 / TYPE 001
    for m in re.finditer(r"\b([A-Z]{2,10})[-_ ](\d{2,6})\b", blob, re.I):
        _append_rule_candidate(candidates, seen, m.group(1))

    return candidates[:12]


def _numbering_rule_ref_from_hit(row: Dict[str, Any]) -> Dict[str, Any]:
    md = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    content = str(row.get("content") or "").strip()
    file_name = str(
        row.get("file_name")
        or row.get("source")
        or md.get("source_file")
        or ""
    ).strip()
    return {
        "file_name": file_name,
        "score": row.get("score"),
        "snippet": content[:240],
        "content": content,
    }


def _gather_numbering_rule_kb_context(collection: str, query: str) -> tuple[str, List[Dict[str, Any]]]:
    """向量检索 + 关键词库检索，尽量召回《文件控制程序》编号章节。"""
    coll = (collection or "regulations").strip() or "regulations"
    base_query = (query or "文件控制程序 编号规则").strip() or "文件控制程序 编号规则"
    queries = [
        base_query,
        "文件控制程序 文件编号 格式",
        "受控文件 编号规则 类型代号",
        "文件控制程序",
    ]
    chunks: List[str] = []
    refs: List[Dict[str, Any]] = []
    seen_snip: set[str] = set()

    def _append_ref(row: Dict[str, Any]) -> None:
        ref = _numbering_rule_ref_from_hit(row)
        content = (ref.get("content") or "").strip()
        if not content:
            return
        key = f"{ref.get('file_name') or ''}|{content[:160]}"
        if key in seen_snip:
            return
        seen_snip.add(key)
        chunks.append(content)
        refs.append({k: ref[k] for k in ("file_name", "score", "snippet") if k in ref})

    try:
        agent = get_agent(coll)
    except Exception:
        agent = None

    if agent is not None:
        for q in queries:
            try:
                hits = agent.search_knowledge(q, top_k=8, use_checkpoints=False) or []
            except Exception:
                hits = []
            for row in hits:
                _append_ref(row)

    keyword_sets = [
        (["文件控制程序", "编号"], "program"),
        (["文件控制", "编号规则"], "program"),
        (["受控文件", "编号"], None),
        (["文件编号", "规则"], None),
    ]
    for keywords, category in keyword_sets:
        try:
            from src.core.db import search_knowledge_docs_by_content_keywords

            rows = search_knowledge_docs_by_content_keywords(
                collection=coll,
                keywords=keywords,
                category=category,
                limit=40,
            )
        except Exception:
            rows = []
        for row in rows or []:
            fn = str(row.get("file_name") or "").strip()
            if category == "program" or "文件控制" in fn or "编号" in fn:
                _append_ref(row)

    text_blob = "\n\n".join(chunks)
    return text_blob, refs


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


# 近义词组：仅当用户问题里已出现组内词时才做替换扩展；不含过宽单字词（如单独「文件」）
_CHAT_SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    ("更新", "复审", "修订", "换版", "再评审", "改版"),
    ("程序文件", "受控文件", "体系文件", "管理文件"),
    ("文件控制", "受控文件", "文件管理"),
    ("规定", "要求", "制度"),
    ("周期", "期限", "频次", "有效期", "保留期"),
    ("记录", "台账", "日志", "表单"),
    ("哪些", "何种", "什么", "有哪些"),
    ("两年", "2年", "二年", "每两年", "每2年", "每二年"),
)
_CHAT_SYNONYM_STOP_ALTS = frozenset({"文件", "规定", "要求", "一次", "1次", "一回", "标准", "制度"})


def _chat_is_list_or_policy_question(query: str) -> bool:
    """清单/周期类规定问答：需要跨多份程序文件召回。"""
    q = (query or "").strip()
    if any(x in q for x in ("哪些", "什么文件", "哪些文件", "哪些记录", "清单", "列出", "有哪些", "包括")):
        return True
    if re.search(r"\d+\s*年", q) and any(x in q for x in ("更新", "复审", "修订", "换版", "周期", "规定", "受控")):
        return True
    return False


def _chat_build_synonym_queries(query: str, base_queries: List[str], *, limit: int = 10) -> List[str]:
    """基于问题已有表述做近义替换/改写，生成多条检索句。"""
    q = (query or "").strip()
    out: List[str] = []

    def _add(s: str) -> None:
        s = (s or "").strip()
        if s and s not in out:
            out.append(s)

    for b in base_queries:
        _add(b)
    _add(q)
    if not q:
        return out[:limit]

    for group in _CHAT_SYNONYM_GROUPS:
        hits = [term for term in group if term in q]
        if not hits:
            continue
        for hit in hits:
            for alt in group:
                if alt == hit or alt in _CHAT_SYNONYM_STOP_ALTS or len(alt) < 2:
                    continue
                _add(q.replace(hit, alt, 1))
                if len(out) >= limit:
                    return out[:limit]

    m_year = re.search(r"(\d+)\s*年", q)
    if m_year:
        n = m_year.group(1)
        for alt in (f"每{n}年", f"{n}年一次", f"{n}年1次", f"至少每{n}年"):
            _add(re.sub(r"\d+\s*年", alt, q, count=1))

    if "程序文件" in q or "规定" in q:
        core = re.sub(r"^(根据|按|依照|按照)", "", q).strip()
        if core and core != q:
            _add(core)

    return out[:limit]


def _chat_extract_year_num(query: str) -> Optional[str]:
    q = (query or "").strip()
    m = re.search(r"(\d+)\s*年", q)
    if m:
        return m.group(1)
    if re.search(r"每?\s*两\s*年|每?\s*二\s*年|每?\s*2\s*年", q):
        return "2"
    return None


def _chat_year_period_db_terms(year_num: Optional[str]) -> List[str]:
    """与用户在 knowledge_docs 中手工 LIKE 的年限近义词对齐。"""
    if not year_num:
        return []
    n = str(year_num).strip()
    if not n.isdigit():
        return []
    terms = [
        f"{n}年",
        f"每{n}年",
        f"至少每{n}年",
        f"{n}年1次",
        f"{n}年一次",
        f"每{n}年1次",
        f"每{n}年一次",
    ]
    if n == "2":
        terms.extend(
            [
                "两年",
                "二年",
                "每两年",
                "每二年",
                "每2年",
                "至少每两年",
                "至少每2年",
                "24个月",
                "24 个月",
                "2 年",
                "每 2 年",
                "两 年",
                "每 两 年",
            ]
        )
    dedup: List[str] = []
    for t in terms:
        t = (t or "").strip()
        if len(t) >= 2 and t not in dedup:
            dedup.append(t)
    return dedup


def _chat_count_keyword_hits(content: str, terms: List[str]) -> int:
    text = (content or "").replace("\r", "")
    compact = text.replace(" ", "")
    hits = 0
    for t in terms or []:
        t = (t or "").strip()
        if len(t) < 2:
            continue
        if t in text or t.replace(" ", "") in compact:
            hits += 1
    return hits


def _chat_content_has_year_period(text: str, year_num: Optional[str]) -> bool:
    if not text:
        return False
    blob = (text or "").replace("\r", "")
    if year_num:
        if any(t in blob for t in _chat_year_period_db_terms(year_num)):
            return True
        if year_num == "2" and re.search(r"每?\s*2\s*年|每?\s*两\s*年|每?\s*二\s*年", blob):
            return True
        return False
    return bool(re.search(r"\d+\s*年|两\s*年|二\s*年", blob))


def _chat_ref_content_relevance(query: str, ref: Dict[str, Any]) -> float:
    """片段正文与问题的语义贴合度（0~1），用于过滤跑题召回。"""
    content = (ref.get("content") or "").replace("\r", "")
    fn = str(ref.get("file_name") or "")
    blob = f"{fn}\n{content}"
    list_q = _chat_is_list_or_policy_question(query)
    year_num = _chat_extract_year_num(query)
    period_terms = ("更新", "复审", "修订", "换版", "再评审", "有效期", "保留期")

    score = 0.0
    hints = [h for h in _chat_rule_hint_terms(query) if len(h) >= 2]
    hint_hits = sum(1 for h in hints if h in blob)
    score += min(0.35, hint_hits * 0.07)

    has_period = any(t in blob for t in period_terms)
    has_year = _chat_content_has_year_period(blob, year_num)

    if list_q and year_num:
        if has_period and has_year:
            score += 0.5
        elif has_period:
            score += 0.12
        else:
            score -= 0.25
        if not has_year:
            score -= 0.15
    elif list_q:
        if has_period:
            score += 0.28
        if has_year:
            score += 0.18
    else:
        if has_period or has_year:
            score += 0.08

    vec = float(ref.get("relevance_score") or ref.get("score") or 0.0)
    if 0.0 < vec <= 1.0:
        score += min(0.2, vec * 0.22)
    elif vec > 1.0:
        score += 0.08

    llm_rel = float(ref.get("relevance_score") or 0.0)
    if llm_rel >= 0.55:
        score += 0.12

    return max(0.0, min(1.0, score))


def _chat_filter_relevant_refs(
    refs: List[Dict[str, Any]],
    query: str,
    *,
    min_score: float = 0.38,
    list_or_policy: bool = False,
) -> List[Dict[str, Any]]:
    threshold = min_score + (0.06 if list_or_policy else 0.0)
    scored: List[tuple[float, Dict[str, Any]]] = []
    for r in refs:
        rel = _chat_ref_content_relevance(query, r)
        r2 = dict(r)
        r2["content_relevance"] = rel
        if rel >= threshold:
            scored.append((rel, r2))
    scored.sort(key=lambda x: (-x[0], -float(x[1].get("relevance_score") or x[1].get("score") or 0.0)))
    return [r for _, r in scored]


def _chat_supplement_policy_queries(query: str) -> List[str]:
    """周期/清单类问题的补充检索句（锚定年限+复审，提高漏召回文件的命中率）。"""
    if not _chat_is_list_or_policy_question(query):
        return []
    out: List[str] = []
    year_num = _chat_extract_year_num(query)
    if year_num:
        for yt in _chat_year_period_db_terms(year_num)[:6]:
            out.append(f"{yt} 复审 更新 受控文件")
        out.append(f"受控文件 {year_num}年 复审 修订")
    out.extend(["受控文件 复审周期 更新周期", "文件控制 复审 修订周期"])
    dedup: List[str] = []
    for s in out:
        s = s.strip()
        if s and s not in dedup:
            dedup.append(s)
    return dedup[:5]


def _chat_db_keyword_terms(query: str) -> List[str]:
    """用于 knowledge_docs LIKE 检索的关键词（与用户手工查库近义词口径一致）。"""
    q = (query or "").strip()
    if not q:
        return []
    terms: List[str] = []

    def _add(t: str) -> None:
        t = (t or "").strip()
        if len(t) >= 2 and t not in terms:
            terms.append(t)

    year_num = _chat_extract_year_num(q)
    for yt in _chat_year_period_db_terms(year_num):
        _add(yt)
    for group in _CHAT_SYNONYM_GROUPS:
        for term in group:
            if term in q:
                _add(term)
    for h in _chat_rule_hint_terms(q):
        if len(h) >= 2:
            _add(h)
    return terms[:18]


def _chat_keyword_search_program_refs(
    collection: str,
    query: str,
    *,
    allowed_category: str,
    limit: int = 80,
) -> List[Dict[str, Any]]:
    """MySQL knowledge_docs 关键词检索，补齐向量漏召回的 program 块。"""
    terms = _chat_db_keyword_terms(query)
    if not terms or not (collection or "").strip():
        return []
    try:
        from src.core.db import search_knowledge_docs_by_content_keywords

        rows = search_knowledge_docs_by_content_keywords(
            collection=collection.strip(),
            keywords=terms,
            category=allowed_category,
            limit=limit,
        )
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    for row in rows or []:
        content = (row.get("content") or "").strip()
        if not content:
            continue
        fn = str(row.get("file_name") or "未知")
        kw_hits = _chat_count_keyword_hits(content, terms)
        ref = {
            "content": content,
            "file_name": fn,
            "category": str(row.get("category") or allowed_category),
            "score": 0.75 + min(0.15, kw_hits * 0.04),
            "source": "keyword_db",
            "chunk_index": row.get("chunk_index"),
            "db_id": row.get("id"),
            "keyword_hits": kw_hits,
        }
        ref["content_relevance"] = _chat_ref_content_relevance(query, ref)
        out.append(ref)
    return out


def _chat_merge_ref_pools(*pools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for pool in pools:
        for r in pool or []:
            key = f"{r.get('file_name')}\n{(r.get('content') or '')[:240]}"
            if key not in order:
                order.append(key)
            old = merged.get(key)
            if not old or float(r.get("score") or 0) > float(old.get("score") or 0):
                merged[key] = dict(r)
    return [merged[k] for k in order if k in merged]


def _chat_ref_should_keep_for_policy(ref: Dict[str, Any], query: str, terms: List[str]) -> bool:
    content = ref.get("content") or ""
    year_num = _chat_extract_year_num(query)
    kw_hits = int(ref.get("keyword_hits") or _chat_count_keyword_hits(content, terms))
    cr = float(ref.get("content_relevance") or _chat_ref_content_relevance(query, ref))
    if ref.get("source") == "keyword_db" and kw_hits >= 1:
        return True
    if kw_hits >= 2:
        return True
    if kw_hits >= 1 and _chat_content_has_year_period(content, year_num):
        return True
    if cr >= 0.32:
        return True
    return False


def _chat_ref_match_score(ref: Dict[str, Any], query: str, terms: Optional[List[str]] = None) -> float:
    """统一匹配度（越高越靠前）。"""
    tms = terms if terms is not None else _chat_db_keyword_terms(query)
    content = ref.get("content") or ""
    kw = int(ref.get("keyword_hits") or _chat_count_keyword_hits(content, tms))
    cr = float(ref.get("content_relevance") or _chat_ref_content_relevance(query, ref))
    vec = float(ref.get("score") or 0.0)
    if vec > 1.0:
        vec = min(0.35, vec * 0.08)
    llm = float(ref.get("relevance_score") or 0.0)
    return round(kw * 0.32 + cr * 0.48 + vec * 0.12 + llm * 0.08, 4)


def _chat_finalize_policy_refs(
    refs: List[Dict[str, Any]],
    query: str,
) -> List[Dict[str, Any]]:
    """周期/清单类：保留全部关键词命中块，并计算匹配度。"""
    terms = _chat_db_keyword_terms(query)
    kept: List[Dict[str, Any]] = []
    for r in refs or []:
        r2 = dict(r)
        r2["content_relevance"] = float(
            r2.get("content_relevance") or _chat_ref_content_relevance(query, r2)
        )
        r2["keyword_hits"] = int(
            r2.get("keyword_hits") or _chat_count_keyword_hits(r2.get("content") or "", terms)
        )
        if _chat_ref_should_keep_for_policy(r2, query, terms):
            r2["match_score"] = _chat_ref_match_score(r2, query, terms)
            kept.append(r2)
    kept.sort(
        key=lambda x: (
            -float(x.get("match_score") or 0.0),
            -int(x.get("keyword_hits") or 0),
            -float(x.get("content_relevance") or 0.0),
        )
    )
    return kept


def _chat_simplify_match_text(text: str, *, max_len: int = 56) -> str:
    s = re.sub(r"\s+", " ", (text or "").replace("…", "")).strip()
    if len(s) > max_len:
        return s[: max_len - 1].rstrip() + "…"
    return s


def _chat_split_into_sentences(text: str) -> List[str]:
    """将正文切分为句子列表（保留句末标点）。"""
    t = (text or "").replace("\r", "")
    if not t.strip():
        return []
    parts = re.split(r"([。；;!?？\n])", t)
    sents: List[str] = []
    buf = ""
    for p in parts:
        if not p:
            continue
        buf += p
        if p in "。；;!?？\n":
            s = re.sub(r"\s+", " ", buf).strip()
            if s:
                sents.append(s)
            buf = ""
    if buf.strip():
        sents.append(re.sub(r"\s+", " ", buf).strip())
    return sents


def _chat_sentence_seems_incomplete(sent: str, keyword: str = "") -> bool:
    """判断单句是否过短/缺主语，简化版需补前后文。"""
    s = re.sub(r"\s+", " ", (sent or "").strip())
    if not s:
        return True
    if len(s) < 20:
        return True
    subject_hints = (
        "文件",
        "记录",
        "程序",
        "受控",
        "应",
        "须",
        "应当",
        "必须",
        "管理",
        "体系",
        "复审",
        "更新",
        "修订",
        "换版",
        "规定",
        "要求",
    )
    has_subject = any(h in s for h in subject_hints)
    if len(s) < 40 and not has_subject:
        return True
    if re.search(r"(一次|每两|每\d+年|至少每|每\d+个月)[。；;]?$", s) and not has_subject:
        return True
    if keyword and len(s) <= len(keyword) + 8:
        return True
    return False


def _chat_find_sentence_index(sentences: List[str], full_text: str, pos: int, term_end: int) -> int:
    if not sentences:
        return 0
    hit = max(pos, 0)
    end_hit = max(term_end, hit)
    cum = 0
    for i, s in enumerate(sentences):
        found = full_text.find(s, cum)
        if found < 0:
            found = cum
        seg_end = found + len(s)
        if found <= hit <= seg_end or found <= end_hit <= seg_end:
            return i
        cum = max(cum, seg_end)
    return 0


def _chat_expand_summary_sentence(
    text: str,
    pos: int,
    term_end: int,
    keyword: str = "",
    *,
    max_sentences: int = 5,
    max_chars: int = 320,
) -> str:
    """简易版：核心句不完整时，向前后扩展若干句直至语义相对完整。"""
    core = _chat_extract_sentence_containing(text, pos, term_end)
    if not _chat_sentence_seems_incomplete(core, keyword):
        return core
    full = (text or "").replace("\r", "")
    sentences = _chat_split_into_sentences(full)
    if len(sentences) <= 1:
        para = _chat_extract_paragraph_containing(full, pos)
        para_one = re.sub(r"\s+", " ", para).strip()
        if para_one and len(para_one) > len(core):
            return para_one[:max_chars].rstrip() + ("…" if len(para_one) > max_chars else "")
        return core
    idx = _chat_find_sentence_index(sentences, full, pos, term_end)
    lo = hi = idx
    parts = [sentences[idx]]
    while _chat_sentence_seems_incomplete(" ".join(parts), keyword) and (hi - lo + 1) < max_sentences:
        grew = False
        if lo > 0:
            lo -= 1
            parts.insert(0, sentences[lo])
            grew = True
        if _chat_sentence_seems_incomplete(" ".join(parts), keyword) and hi < len(sentences) - 1:
            hi += 1
            parts.append(sentences[hi])
            grew = True
        if len(" ".join(parts)) >= max_chars:
            break
        if not grew:
            break
    merged = " ".join(parts).strip()
    if _chat_sentence_seems_incomplete(merged, keyword):
        para = re.sub(r"\s+", " ", _chat_extract_paragraph_containing(full, pos)).strip()
        if para and len(para) > len(merged):
            merged = para
    if len(merged) > max_chars:
        merged = merged[: max_chars - 1].rstrip() + "…"
    return merged or core


def _chat_extract_sentence_containing(text: str, pos: int, term_end: int) -> str:
    """提取包含关键词的整句话（以。；;!?？或换行分句）。"""
    t = (text or "").replace("\r", "")
    if not t or pos < 0:
        return t.strip()
    boundaries = [0]
    for i, c in enumerate(t):
        if c in "。；;!?？":
            boundaries.append(i + 1)
        elif c == "\n":
            boundaries.append(i + 1)
    boundaries.append(len(t))
    uniq: List[int] = []
    for b in boundaries:
        if not uniq or uniq[-1] != b:
            uniq.append(b)
    hit = max(pos, 0)
    for j in range(len(uniq) - 1):
        s, e = uniq[j], uniq[j + 1]
        if s <= hit < e or (s <= term_end <= e):
            sent = t[s:e].strip()
            if sent:
                return re.sub(r"\s+", " ", sent)
    line_start = t.rfind("\n", 0, hit) + 1
    line_end = t.find("\n", term_end)
    if line_end < 0:
        line_end = len(t)
    return re.sub(r"\s+", " ", t[line_start:line_end].strip())


def _chat_extract_paragraph_containing(text: str, pos: int) -> str:
    """提取包含关键词的整段（以空行或连续非空行组成段落）。"""
    t = (text or "").replace("\r", "")
    if not t or pos < 0:
        return t.strip()
    line_starts: List[tuple[int, str]] = []
    idx = 0
    for line in t.split("\n"):
        line_starts.append((idx, line))
        idx += len(line) + 1
    target = 0
    for i, (st, ln) in enumerate(line_starts):
        en = st + len(ln)
        if st <= pos <= en:
            target = i
            break
    lines = [ln for _, ln in line_starts]
    start = target
    while start > 0 and lines[start - 1].strip():
        start -= 1
    end = target
    while end < len(lines) - 1 and lines[end + 1].strip():
        end += 1
    para = "\n".join(lines[start : end + 1]).strip()
    return para or t.strip()


def _chat_extract_match_spans(content: str, query: str) -> Dict[str, Any]:
    """定位最佳关键词，并提取对应句子（简易版）与段落（详细版）。"""
    text = (content or "").replace("\r", "")
    terms = _chat_db_keyword_terms(query)
    best_pos = -1
    best_term = ""
    best_rank = -1
    year_num = _chat_extract_year_num(query)
    compact = text.replace(" ", "")
    for term in sorted(terms, key=len, reverse=True):
        start = 0
        while True:
            pos = text.find(term, start)
            if pos < 0:
                break
            rank = len(term) * 10
            window = text[max(0, pos - 40): min(len(text), pos + len(term) + 40)]
            if _chat_content_has_year_period(window, year_num):
                rank += 30
            if any(x in window for x in ("更新", "复审", "修订", "换版", "周期")):
                rank += 15
            if rank > best_rank:
                best_rank = rank
                best_pos = pos
                best_term = term
            start = pos + max(1, len(term))
    term_end = best_pos + len(best_term) if best_pos >= 0 and best_term else best_pos
    if best_pos < 0:
        sent, _ = _chat_pick_rule_sentences(text, query, max_chars=200)
        return {
            "match_keyword": "",
            "match_sentence": sent,
            "match_paragraph": sent,
            "match_keywords": [],
            "full_content": text,
        }
    sentence = _chat_expand_summary_sentence(text, best_pos, term_end, best_term)
    paragraph = _chat_extract_paragraph_containing(text, best_pos)
    matched = [t for t in terms if t in text or t.replace(" ", "") in compact]
    return {
        "match_keyword": best_term,
        "match_sentence": sentence,
        "match_paragraph": paragraph,
        "match_keywords": matched[:6],
        "full_content": text,
    }


def _chat_extract_match_excerpt(content: str, query: str, *, context: int = 20) -> Dict[str, Any]:
    """兼容旧字段名：excerpt=段落。"""
    spans = _chat_extract_match_spans(content, query)
    return {
        "excerpt": spans.get("match_paragraph") or spans.get("match_sentence") or "",
        "match_keyword": spans.get("match_keyword") or "",
        "match_keywords": spans.get("match_keywords") or [],
        "match_sentence": spans.get("match_sentence") or "",
        "match_paragraph": spans.get("match_paragraph") or "",
        "full_content": spans.get("full_content") or content,
    }


def _chat_build_policy_file_items(
    refs: List[Dict[str, Any]],
    query: str,
    *,
    context: int = 20,
) -> List[Dict[str, Any]]:
    """每份文件保留匹配度最高的一条，按 match_score 倒序。"""
    by_file: Dict[str, Dict[str, Any]] = {}
    for r in refs or []:
        fn = str(r.get("file_name") or "未知")
        old = by_file.get(fn)
        if not old or float(r.get("match_score") or 0.0) > float(old.get("match_score") or 0.0):
            by_file[fn] = r
    ordered = sorted(
        by_file.values(),
        key=lambda x: (
            -float(x.get("match_score") or 0.0),
            -int(x.get("keyword_hits") or 0),
        ),
    )
    items: List[Dict[str, Any]] = []
    for i, r in enumerate(ordered, start=1):
        content = r.get("content") or ""
        match_info = _chat_extract_match_spans(content, query)
        kw = match_info.get("match_keyword") or ""
        sentence = (match_info.get("match_sentence") or "").strip()
        paragraph = (match_info.get("match_paragraph") or sentence).strip()
        items.append(
            {
                "index": i,
                "file_name": r.get("file_name") or "未知",
                "chunk_index": r.get("chunk_index"),
                "category": r.get("category") or "program",
                "match_score": float(r.get("match_score") or 0.0),
                "keyword_hits": int(r.get("keyword_hits") or 0),
                "match_keyword": kw,
                "match_keywords": match_info.get("match_keywords") or [],
                "match_sentence": sentence,
                "match_paragraph": paragraph,
                "summary_snippet": sentence,
                "matched_excerpt": paragraph,
                "full_content": match_info.get("full_content") or content,
                "source": r.get("source") or "knowledge_base",
            }
        )
    return items


def _chat_build_detail_items(
    refs: List[Dict[str, Any]],
    query: str,
    *,
    context: int = 20,
) -> List[Dict[str, Any]]:
    """兼容旧调用：委托给按文件聚合的实现。"""
    return _chat_build_policy_file_items(refs, query, context=context)


def _chat_compose_policy_answer_from_items(
    detail_items: List[Dict[str, Any]],
    query: str,
    *,
    max_summary_chars: int,
) -> tuple[str, str, int]:
    if not detail_items:
        return "", "", 0
    n_files = len(detail_items)
    header = f"共匹配 {n_files} 份文件："
    summary_lines = [header]

    for it in detail_items:
        idx = it.get("index") or 0
        fn = str(it.get("file_name") or "未知").strip()
        kw = str(it.get("match_keyword") or "").strip()
        sent = str(it.get("match_sentence") or it.get("summary_snippet") or "").strip()
        if fn and kw and sent:
            summary_lines.append(f"{idx}. {fn} · 「{kw}」：{sent}")
        elif fn and sent:
            summary_lines.append(f"{idx}. {fn} · {sent}")
        elif fn and kw:
            summary_lines.append(f"{idx}. {fn} · 「{kw}」")
        elif kw and sent:
            summary_lines.append(f"{idx}. 「{kw}」：{sent}")
        elif kw:
            summary_lines.append(f"{idx}. 「{kw}」")
        elif sent:
            summary_lines.append(f"{idx}. {sent}")

    summary = "\n".join(summary_lines)
    if len(summary) > max_summary_chars:
        summary = summary[: max_summary_chars - 1].rstrip() + "…"
    detail_hint = f"共 {n_files} 份文件；详细版展示含关键词的整段内容，可点「查看更多」查看全文。"
    return summary, detail_hint, n_files


def _chat_diversify_refs_by_file(
    refs: List[Dict[str, Any]],
    *,
    max_total: int,
    max_per_file: int = 1,
) -> List[Dict[str, Any]]:
    """清单类问题：优先覆盖不同来源文件，避免只保留单份程序文件片段。"""
    if not refs or max_total <= 0:
        return []
    per_file: Dict[str, List[Dict[str, Any]]] = {}
    for r in refs:
        fn = str(r.get("file_name") or "未知")
        per_file.setdefault(fn, []).append(r)
    picked: List[Dict[str, Any]] = []
    seen_keys: set[str] = set()
    # 第一轮：每个文件先取一条
    for fn, items in per_file.items():
        if len(picked) >= max_total:
            break
        for r in items[:max_per_file]:
            key = f"{fn}\n{(r.get('content') or '')[:120]}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            picked.append(r)
            break
    # 第二轮：若未满，再按原顺序补入
    if len(picked) < max_total:
        for r in refs:
            if len(picked) >= max_total:
                break
            fn = str(r.get("file_name") or "未知")
            key = f"{fn}\n{(r.get('content') or '')[:120]}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            picked.append(r)
    return picked


def _chat_group_refs_by_file(refs: List[Dict[str, Any]]) -> List[tuple[str, List[Dict[str, Any]]]]:
    """按来源文件分组并保持首次出现顺序。"""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for r in refs:
        fn = str(r.get("file_name") or "未知").strip() or "未知"
        if fn not in grouped:
            grouped[fn] = []
            order.append(fn)
        grouped[fn].append(r)
    return [(fn, grouped[fn]) for fn in order]


def _chat_rule_hint_terms(query: str) -> List[str]:
    q = (query or "").strip()
    terms: List[str] = []
    for group in _CHAT_SYNONYM_GROUPS:
        for term in group:
            if term in q and term not in terms:
                terms.append(term)
    if re.search(r"\d+\s*年", q):
        for t in ("年", "更新", "复审", "修订", "换版", "周期", "有效期", "保留"):
            if t not in terms:
                terms.append(t)
    if not terms:
        terms = ["规定", "要求", "应", "须", "至少", "每"]
    return terms


def _chat_pick_rule_sentences(content: str, query: str, *, max_chars: int = 260) -> tuple[str, int]:
    """从片段中抽取与问题最相关的规定句，返回 (文本, 相关度分)。"""
    text = (content or "").replace("\r", "").strip()
    if not text:
        return "（片段中未提取到明确条文，请查阅原文）", 0
    parts = [p.strip() for p in re.split(r"[\n；;。!?？]", text) if p.strip()]
    if not parts:
        parts = [text]
    hints = _chat_rule_hint_terms(query)
    year_num = _chat_extract_year_num(query)
    scored: List[tuple[int, str]] = []
    for part in parts:
        score = sum(1 for h in hints if len(h) >= 2 and h in part)
        if _chat_content_has_year_period(part, year_num):
            score += 4
        if any(x in part for x in ("更新", "复审", "修订", "换版", "周期")):
            score += 2
        scored.append((score, part))
    scored.sort(key=lambda x: (-x[0], -len(x[1])))
    picked = scored[0][1] if scored else text
    best_score = scored[0][0] if scored else 0
    if best_score <= 0 and len(text) <= max_chars:
        picked = text
    picked = re.sub(r"\s+", " ", picked).strip()
    if len(picked) > max_chars:
        picked = picked[: max_chars - 1].rstrip() + "…"
    return picked, best_score


def _chat_compose_multi_file_policy_answer(
    scored_refs: List[Dict[str, Any]],
    query: str,
    *,
    max_summary_chars: int,
    max_detail_chars: int,
    min_rule_score: int = 2,
) -> tuple[str, str, int]:
    """清单/周期类：仅汇总通过正文相关度校验的文件。"""
    filtered = _chat_filter_relevant_refs(scored_refs, query, min_score=0.38, list_or_policy=True)
    groups = _chat_group_refs_by_file(filtered)
    if not groups:
        return "", "", 0

    n_files = len(groups)
    header = f"根据程序文件检索，共 {n_files} 份文件与问题相关："
    per_summary = max(48, min(140, (max_summary_chars - len(header) - 10) // max(n_files, 1)))
    per_detail = max(120, min(520, (max_detail_chars - 80) // max(n_files, 1)))

    summary_lines: List[str] = [header]
    detail_blocks: List[str] = []
    included = 0

    for idx, (fn, items) in enumerate(groups, start=1):
        best = max(
            items,
            key=lambda x: (
                float(x.get("content_relevance") or 0.0),
                float(x.get("relevance_score") or x.get("score") or 0.0),
            ),
        )
        rule, rule_score = _chat_pick_rule_sentences(best.get("content") or "", query, max_chars=per_detail)
        if rule_score < min_rule_score and float(best.get("content_relevance") or 0.0) < 0.5:
            continue
        short_rule, _ = _chat_pick_rule_sentences(best.get("content") or "", query, max_chars=per_summary)
        included += 1
        summary_lines.append(f"{included}. {fn}：{short_rule}")
        detail_blocks.append(f"{included}. 《{fn}》\n{rule}")

    if included == 0:
        return "", "", 0
    if included != n_files:
        summary_lines[0] = f"根据程序文件检索，共 {included} 份文件与问题相关："

    summary = "\n".join(summary_lines)
    detail = summary_lines[0] + "\n\n" + "\n\n".join(detail_blocks)
    if len(summary) > max_summary_chars:
        summary = summary[: max_summary_chars - 1].rstrip() + "…"
    if len(detail) > max_detail_chars:
        detail = detail[: max_detail_chars - 1].rstrip() + "…"
    return summary, detail, included


def _chat_merge_llm_with_ref_coverage(
    llm_summary: str,
    llm_detail: str,
    scored_refs: List[Dict[str, Any]],
    query: str,
    *,
    max_summary_chars: int,
    max_detail_chars: int,
) -> tuple[str, str]:
    """LLM 漏写文件时，仅用高相关检索结果补全。"""
    det = _chat_normalize_answer_text(llm_detail)
    filtered = _chat_filter_relevant_refs(scored_refs, query, min_score=0.42, list_or_policy=True)
    groups = _chat_group_refs_by_file(filtered)
    missing: List[tuple[str, str]] = []
    for fn, items in groups:
        if fn and fn in det:
            continue
        best = max(items, key=lambda x: float(x.get("content_relevance") or 0.0))
        rule, rule_score = _chat_pick_rule_sentences(best.get("content") or "", query, max_chars=260)
        if rule_score < 2:
            continue
        missing.append((fn, rule))
    if not missing:
        return (
            _chat_normalize_answer_text(llm_summary)[:max_summary_chars],
            det[:max_detail_chars],
        )
    append_lines = [f"《{fn}》\n{rule}" for fn, rule in missing]
    merged_detail = det.rstrip() + "\n\n【补全未写入的文件】\n\n" + "\n\n".join(append_lines)
    fallback_summary, fallback_detail, _ = _chat_compose_multi_file_policy_answer(
        filtered,
        query,
        max_summary_chars=max_summary_chars,
        max_detail_chars=max_detail_chars,
    )
    if len(fallback_detail) <= max_detail_chars:
        return fallback_summary[:max_summary_chars], fallback_detail
    return fallback_summary[:max_summary_chars], merged_detail[:max_detail_chars]


def _chat_retrieval_plan(intent: Dict[str, Any], query: str) -> Dict[str, Any]:
    """由意图分析产出检索计划（查询扩展 + 主题词），通用适配各业务域。"""
    q = str(intent.get("normalized_question") or query or "").strip() or (query or "").strip()
    llm_queries = _chat_as_str_list(intent.get("search_queries"), limit=6)
    search_queries = _chat_build_synonym_queries(q, llm_queries, limit=8)
    search_queries.extend(_chat_supplement_policy_queries(q))
    dedup_q: List[str] = []
    for s in search_queries:
        s = (s or "").strip()
        if s and s not in dedup_q:
            dedup_q.append(s)
    search_queries = dedup_q[:12]
    boost_terms = _chat_as_str_list(intent.get("topic_keywords") or intent.get("relevant_topics"), limit=12)
    penalty_terms = _chat_as_str_list(intent.get("avoid_topics") or intent.get("irrelevant_topics"), limit=8)
    return {
        "search_queries": search_queries,
        "boost_terms": boost_terms,
        "penalty_terms": penalty_terms,
        "normalized_question": q,
        "is_list_or_policy": _chat_is_list_or_policy_question(q),
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
    recall_limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """程序文件检索：多查询近义扩展 + category 过滤 + 主题词加减分。"""
    ctx = plan if isinstance(plan, dict) else _chat_retrieval_plan({}, query)
    pool: List[Dict[str, Any]] = []
    seen: set[str] = set()
    list_q = bool(ctx.get("is_list_or_policy")) or _chat_is_list_or_policy_question(query)
    pool_cap = recall_limit or (max(top_k * 3, 30) if list_q else max(top_k * 2, 20))

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

    per_query_k = max(top_k * 4, 20) if list_q else max(top_k * 3, 16)
    for sq in ctx.get("search_queries") or [query]:
        sq = (sq or "").strip()
        if not sq:
            continue
        try:
            for pair in agent.kb.search_with_scores(sq, top_k=per_query_k) or []:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    _push(pair[0], pair[1])
        except Exception:
            pass
        try:
            for doc in agent.kb.search_by_category(sq, category=allowed_category, top_k=per_query_k) or []:
                content = (getattr(doc, "page_content", "") or "").strip()
                md = getattr(doc, "metadata", {}) or {}
                stub = {
                    "content": content,
                    "file_name": str(md.get("source_file") or "未知"),
                    "score": 0.45,
                }
                cr = _chat_ref_content_relevance(query, stub)
                if cr >= 0.28:
                    _push(doc, 0.42 + cr * 0.35)
        except Exception:
            pass
        if len(pool) >= pool_cap * 2:
            break

    pool.sort(
        key=lambda r: (
            _chat_ref_content_relevance(query, r),
            _chat_ref_domain_score(query, r, ctx),
        ),
        reverse=True,
    )
    return pool[:pool_cap]


def _chat_rerank_refs_with_llm(
    query: str,
    candidates: List[Dict[str, Any]],
    *,
    top_k: int,
    current_provider: Optional[str] = None,
    client_llm: Optional[Any] = None,
    header_provider: str = "",
    list_or_policy: bool = False,
) -> tuple[List[Dict[str, Any]], float, str]:
    """用 LLM 从候选片段中挑选与用户问题真正相关的程序文件依据。"""
    if not candidates:
        return [], 0.0, "无候选片段"
    pick_k = min(max(top_k, 6 if list_or_policy else top_k), 10)
    preview_n = min(20 if list_or_policy else 14, len(candidates))
    prefiltered = _chat_filter_relevant_refs(
        candidates[:preview_n + 8],
        query,
        min_score=0.28,
        list_or_policy=list_or_policy,
    )
    rerank_pool = prefiltered[:preview_n] if prefiltered else candidates[:preview_n]

    if len(rerank_pool) <= pick_k and not list_or_policy:
        ctx = {"boost_terms": [], "penalty_terms": []}
        score = _chat_ref_content_relevance(query, rerank_pool[0]) if rerank_pool else 0.0
        return rerank_pool, score, "候选较少，跳过重排"

    lines: List[str] = []
    for i, r in enumerate(rerank_pool, start=1):
        snippet = (r.get("content") or "")[:320].replace("\n", " ")
        lines.append(f"[{i}] 文件: {r.get('file_name') or '未知'}\n片段: {snippet}")
    task_hint = (
        "用户可能在问「程序文件中的规定」（如更新/复审周期、哪些受控文件需几年更新一次等）。"
        "仅选片段中**明确写出**与问题一致的复审/更新/修订年限或周期要求的条目；"
        "仅泛泛提到「文件」「记录」但未写周期年限的不要选。"
        "在相关前提下尽量覆盖不同来源文件（同一文件最多 1 条）。"
        if list_or_policy
        else "请选出能回答该问题的片段（制度要求、记录项目、填写要点或文件控制规定等）。"
    )
    prompt = (
        "你是知识库检索质检员。用户要问的是医疗器械质量管理体系程序文件相关问题。\n"
        f"用户问题：{query}\n\n"
        f"{task_hint}\n"
        "与问题无关的片段（业务域明显不符）不要选。\n\n"
        + "\n\n".join(lines)
        + "\n\n仅输出 JSON：\n"
        "{\n"
        '  "selected": [1, 3, 5],\n'
        '  "scores": {"1": 0.92, "2": 0.05},\n'
        '  "reason": "一句话说明选择依据"\n'
        "}\n"
        "selected 为相关条目编号（至少 0 个，最多 "
        f"{pick_k}"
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
                if 1 <= n <= len(rerank_pool):
                    selected_idx.append(n)
            except (TypeError, ValueError):
                continue
    selected_idx = list(dict.fromkeys(selected_idx))[:pick_k]

    min_llm_rel = 0.48 if list_or_policy else 0.42

    if not selected_idx:
        fallback = _chat_filter_relevant_refs(
            rerank_pool,
            query,
            min_score=0.42,
            list_or_policy=list_or_policy,
        )
        if fallback:
            fb = _chat_diversify_refs_by_file(fallback, max_total=pick_k, max_per_file=1)
            score = float(fb[0].get("content_relevance") or _chat_ref_content_relevance(query, fb[0]))
            return fb, score, reason or "LLM 未选中，已按正文相关度回退"
        return [], 0.0, reason or "LLM 未选中任何相关程序文件片段"

    picked: List[Dict[str, Any]] = []
    rel_scores: List[float] = []
    for n in selected_idx:
        ref = dict(rerank_pool[n - 1])
        key = str(n)
        try:
            rel = float((scores_map or {}).get(key, scores_map.get(str(n), 0.0)) or 0.0)
        except (TypeError, ValueError):
            rel = 0.0
        content_rel = float(ref.get("content_relevance") or _chat_ref_content_relevance(query, ref))
        if rel < min_llm_rel and content_rel < 0.45:
            continue
        ref["relevance_score"] = max(0.0, min(1.0, rel))
        ref["content_relevance"] = content_rel
        picked.append(ref)
        if rel > 0:
            rel_scores.append(rel)
        else:
            rel_scores.append(content_rel)

    if not picked:
        fallback = _chat_filter_relevant_refs(
            rerank_pool,
            query,
            min_score=0.44,
            list_or_policy=list_or_policy,
        )
        if fallback:
            fb = _chat_diversify_refs_by_file(fallback, max_total=pick_k, max_per_file=1)
            score = float(fb[0].get("content_relevance") or 0.0)
            return fb, score, reason or "选中片段相关度不足，已按正文过滤回退"
        return [], 0.0, reason or "选中片段与问题不匹配"

    if list_or_policy:
        picked = _chat_diversify_refs_by_file(picked, max_total=pick_k, max_per_file=1)
    top_rel = max(rel_scores) if rel_scores else float(picked[0].get("content_relevance") or 0.0)
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


def _chat_heuristic_program_scope(query: str) -> bool:
    """规则兜底：问题明显与程序文件/体系记录相关时，不因意图 LLM 误判直接拒答。"""
    q = (query or "").strip()
    if not q:
        return False
    keys = (
        "程序文件",
        "体系",
        "运行记录",
        "受控",
        "文件控制",
        "更新",
        "复审",
        "修订",
        "换版",
        "周期",
        "哪些文件",
        "哪些记录",
        "什么文件",
        "多长时间",
        "有效期",
        "保留",
        "SOP",
        "规程",
        "台账",
        "怎么写",
        "如何写",
        "填写",
        "规定",
        "依据",
    )
    return any(k in q for k in keys)


def _chat_restricted_date_personnel(intent: Dict[str, Any], query: str) -> bool:
    """仅拦截「记录里日期/签字人员怎么填」，不拦截更新周期、复审年限等政策问法。"""
    if not bool(intent.get("is_date_or_personnel_question")):
        return False
    q = (query or "").strip()
    if re.search(r"\d+\s*年", q) and any(
        x in q for x in ("更新", "复审", "修订", "换版", "周期", "有效期", "保留", "受控")
    ):
        return False
    if any(x in q for x in ("哪些文件", "哪些记录", "什么文件", "哪些需要", "程序文件规定", "规定")):
        return False
    return True


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
        "1) is_system_record_writing=true 包括：①体系运行记录怎么写/怎么填；"
        "②程序文件中可检索的规定性问答（如哪些文件需更新/复审、更新周期几年、受控范围等）。"
        "纯法规注册结论、与程序文件无关的 IT/行政杂问为 false。\n"
        "2) is_date_or_personnel_question=true 仅当问「某条记录日期填什么、签字人员写谁」；"
        "问「文件几年更新/复审」不算人员日期填写。\n"
        "3) search_queries 写 5～8 条中文检索句：对用户问题做**近义改写**（如 更新↔复审↔修订、2年↔两年↔每两年、"
        "程序文件↔受控文件），换表述不换含义；可含可能的制度名称、记录类型，勿编造文件编号。\n"
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
    system_record = _chat_heuristic_program_scope(q)
    if re.search(r"\d+\s*年", q) and any(x in q for x in ("更新", "复审", "修订", "周期")):
        date_or_personnel = False
    fallback_queries = _chat_build_synonym_queries(q, [q], limit=8)
    return {
        "is_system_record_writing": bool(system_record),
        "is_date_or_personnel_question": bool(date_or_personnel),
        "normalized_question": q,
        "search_queries": fallback_queries,
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
    import os

    ver = (os.environ.get("AICHECKWORD_APP_VERSION") or os.environ.get("APP_VERSION") or "").strip()
    payload = {"status": "ok", "service": "aicheckword"}
    if ver:
        payload["version"] = ver
    return payload


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
    except CompanyMappingConflictError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
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


@app.post("/api/integration/document-control/extract-batch")
async def integration_document_control_extract_batch(
    req: Request,
    files: List[UploadFile] = File(...),
    collection: str = Form("regulations"),
):
    collection = _resolve_request_collection(req, collection)
    items: List[Dict[str, Any]] = []
    for file in files or []:
        suffix = Path(file.filename or "").suffix or ".tmp"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)
        try:
            text_parts: List[str] = []
            if tmp_path.suffix.lower() == ".docx":
                text_parts.append(_build_docx_headers_footers_plaintext(tmp_path))
                text_parts.append(_build_docx_textbox_plaintext(tmp_path))
            try:
                docs = split_documents(load_single_file(tmp_path))
                if docs:
                    text_parts.append("\n".join((d.page_content or "") for d in docs[:12]))
            except Exception:
                pass
            merged = "\n".join(p for p in text_parts if p).strip()
            meta = _extract_document_control_meta(file.filename or "", merged)
            title = Path(file.filename or "").stem
            items.append(
                {
                    "original_filename": file.filename or "",
                    "document_number": meta["document_number"],
                    "version": meta["version"],
                    "title": title,
                    "doc_type": meta["doc_type"],
                    "extract_confidence": meta["extract_confidence"],
                    "status": "new",
                }
            )
        except Exception as e:
            items.append(
                {
                    "original_filename": file.filename or "",
                    "status": "error",
                    "message": str(e),
                }
            )
        finally:
            tmp_path.unlink(missing_ok=True)
    return {"ok": True, "collection": collection, "items": items}


@app.post("/api/integration/document-control/validate-allocation")
def integration_document_control_validate_allocation(
    req: Request,
    body: DocumentControlValidateRequest,
):
    body.collection = _resolve_request_collection(req, body.collection or "regulations")
    number = _normalize_document_number_local(body.document_number or "")
    warnings: List[str] = []
    blocked: List[str] = []
    if not number:
        blocked.append("文件编号不能为空")
    elif not _DOCNO_RE.search(number):
        blocked.append("文件编号不符合“前缀-类型-流水号”格式")
    if body.doc_type_code and number and f"-{body.doc_type_code.strip().upper()}-" not in number:
        warnings.append("编号中的类型码与当前文档类型不一致，请人工确认")
    if body.project_code and number and not number.startswith(_normalize_document_number_local(body.project_code)):
        warnings.append("编号前缀与项目编号不一致，请确认是否为跨项目文档")
    kb_hits = []
    try:
        agent = get_agent(body.collection)
        kb_hits = agent.search_knowledge(
            f"文件控制程序 编号规则 {body.doc_type_code or ''} {body.title or ''}",
            top_k=5,
            use_checkpoints=False,
        ) or []
    except Exception:
        kb_hits = []
    if not kb_hits:
        warnings.append("未检索到明确的知识库规则摘录，已降级为规则引擎校验")
    status = "ok"
    if blocked:
        status = "blocked"
    elif warnings:
        status = "warnings"
    refs = []
    for row in kb_hits[:3]:
        refs.append(
            {
                "file_name": row.get("file_name") or "",
                "score": row.get("score"),
                "snippet": str(row.get("content") or "")[:240],
            }
        )
    return {
        "ok": status != "blocked",
        "status": status,
        "warnings": warnings,
        "blocked": blocked,
        "knowledge_refs": refs,
    }


@app.post("/api/integration/document-control/parse-numbering-rules")
def integration_document_control_parse_numbering_rules(
    req: Request,
    body: DocumentControlParseRulesRequest,
):
    body.collection = _resolve_request_collection(req, body.collection or "regulations")
    text_blob = (body.text or "").strip()
    refs: List[Dict[str, Any]] = []
    if not text_blob:
        text_blob, refs = _gather_numbering_rule_kb_context(body.collection, body.query)
    candidates = _parse_rule_candidates_from_text(text_blob)
    from src.core.document_control_rules import merge_kb_rules_with_fallback

    source_file = ""
    if refs:
        source_file = str((refs[0] or {}).get("file_name") or "").strip()
    rules = merge_kb_rules_with_fallback(text_blob, source_file=source_file)
    if not rules and candidates:
        rules = [
            {
                "docTypeCode": c.get("docTypeCode"),
                "name": c.get("name"),
                "renderTemplate": c.get("renderTemplate"),
                "prefixSource": c.get("prefixSource"),
                "seqStart": c.get("seqStart"),
                "seqPad": c.get("seqPad"),
                "example": c.get("example") or "",
                "autoAllocatable": True,
                "needsProjectCode": (c.get("prefixSource") or "") == "from_project_code",
                "kbRuleExcerpt": c.get("name") or "",
            }
            for c in candidates
        ]
    ref_items = refs[:8]
    if not ref_items and text_blob.strip():
        ref_items = [{"file_name": "", "snippet": text_blob[:240]}]
    message = f"知识库检索到 {len(refs)} 条相关片段"
    if rules:
        auto_n = sum(1 for r in rules if r.get("autoAllocatable"))
        message += f"，已解析 {len(rules)} 类文件编号规则（{auto_n} 类可自动取号）"
    elif refs:
        message += "，已加载《文件控制程序》缺省规则结构"
        rules = merge_kb_rules_with_fallback("", source_file=source_file)
    else:
        message += "；未检索到《文件控制程序》，请确认知识库已训练且 collection 与当前公司一致"
    return {
        "ok": True,
        "collection": body.collection,
        "rules": rules,
        "candidates": candidates,
        "references": ref_items,
        "referenceCount": len(refs),
        "candidateCount": len(candidates),
        "ruleCount": len(rules),
        "sourceFile": source_file,
        "message": message,
    }


@app.post("/api/integration/document-control/release-date-suggest")
def integration_document_control_release_date_suggest(
    req: Request,
    body: DocumentControlReleaseDateSuggestRequest,
):
    """AI 联网检索各版本公开发布时间候选（供 aiword 版本任务清单页调用）。"""
    body.collection = _resolve_request_collection(req, body.collection or "regulations")
    from_version = (body.fromVersion or "").strip()
    to_version = (body.toVersion or "").strip()
    if not from_version or not to_version:
        raise HTTPException(status_code=400, detail="fromVersion 与 toVersion 必填")
    from src.core.release_date_search import suggest_release_dates
    from .draft_integration import _header_provider, _parse_client_llm

    client_llm = _parse_client_llm(req)
    provider = _header_provider(req)
    cl = client_llm if client_llm.has_any() else None

    def llm_text_fn(prompt: str) -> str:
        result = _invoke_chat_llm_text(
            prompt,
            header_provider=provider,
            client_llm=cl,
        )
        return result.text or ""

    try:
        result = suggest_release_dates(
            product_name=(body.productName or "").strip(),
            from_version=from_version,
            to_version=to_version,
            intermediate_versions=[str(x or "").strip() for x in (body.intermediateVersions or []) if str(x or "").strip()],
            target_version=(body.targetVersion or "").strip() or None,
            registration_country=(body.registrationCountry or "").strip(),
            llm_text_fn=llm_text_fn,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "collection": body.collection, **result}


@app.post("/api/integration/document-control/release-date-suggest/diagnose")
def integration_document_control_release_date_suggest_diagnose(
    req: Request,
    body: DocumentControlReleaseDateSuggestRequest,
):
    """发布时间检索诊断：返回应用市场检索命中、解析器与日期抽取详情。"""
    body.collection = _resolve_request_collection(req, body.collection or "regulations")
    from_version = (body.fromVersion or "").strip()
    to_version = (body.toVersion or "").strip()
    if not from_version or not to_version:
        raise HTTPException(status_code=400, detail="fromVersion 与 toVersion 必填")
    from src.core.release_date_search import diagnose_release_dates
    from .draft_integration import _header_provider, _parse_client_llm

    client_llm = _parse_client_llm(req)
    provider = _header_provider(req)
    cl = client_llm if client_llm.has_any() else None

    def llm_text_fn(prompt: str) -> str:
        result = _invoke_chat_llm_text(
            prompt,
            header_provider=provider,
            client_llm=cl,
        )
        return result.text or ""

    try:
        result = diagnose_release_dates(
            product_name=(body.productName or "").strip(),
            from_version=from_version,
            to_version=to_version,
            intermediate_versions=[
                str(x or "").strip()
                for x in (body.intermediateVersions or [])
                if str(x or "").strip()
            ],
            target_version=(body.targetVersion or "").strip() or None,
            registration_country=(body.registrationCountry or "").strip(),
            llm_text_fn=llm_text_fn,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "collection": body.collection, **result}


@app.post("/api/integration/document-control/translate-title")
def integration_document_control_translate_title(
    req: Request,
    body: DocumentControlTranslateTitleRequest,
):
    body.collection = _resolve_request_collection(req, body.collection or "regulations")
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text 必填")
    from .draft_integration import _header_provider, _parse_client_llm

    client_llm = _parse_client_llm(req)
    provider = _header_provider(req)
    prompt = (
        "将以下医疗器械技术文件名称直译为简洁、专业的英文标题（Title Case）。"
        "须忠实于原意，不要套用固定缩写（如不要无故使用 SRS），不要添加原文没有的含义。"
        "仅输出一行英文标题，不要解释、不要引号、不要编号。\n\n"
        f"{text}"
    )
    try:
        result = _invoke_chat_llm_text(
            prompt,
            header_provider=provider,
            client_llm=client_llm if client_llm.has_any() else None,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"翻译失败：{exc}") from exc
    raw_en = (result.text or "").strip()
    lines = raw_en.splitlines()
    title_en = (lines[0] if lines else raw_en).strip().strip("\"'「」『』《》")
    if not title_en:
        raise HTTPException(status_code=502, detail="翻译结果为空")
    return {"ok": True, "titleEn": title_en[:255], "source": "translated"}


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
            "reason": "首版仅支持基于程序文件知识库的体系记录与程序规定问答。",
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
    if _chat_restricted_date_personnel(intent, query):
        return {
            "request_id": request_id,
            "need_human": True,
            "answer": "",
            "confidence": 0.0,
            "reason": "首版不自动回答记录中具体日期与人员填写问题。",
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
    max_summary_chars = max(80, min(8000, int(opts.max_reply_chars or 280)))
    max_detail_chars = int(opts.max_detail_chars or 0) or int(
        getattr(settings, "chat_max_detail_chars", 2400) or 2400
    )
    max_detail_chars = max(200, min(6000, max_detail_chars))

    retrieval_plan = _chat_retrieval_plan(intent, query)
    search_query = str(retrieval_plan.get("normalized_question") or query).strip() or query
    list_or_policy = bool(retrieval_plan.get("is_list_or_policy")) or _chat_is_list_or_policy_question(query)
    recall_k = max(top_k * (5 if list_or_policy else 3), 24 if list_or_policy else 16)
    candidates = _chat_retrieve_program_scored_refs(
        agent,
        search_query,
        allowed_category=allowed_category,
        top_k=top_k,
        plan=retrieval_plan,
        recall_limit=recall_k,
    )
    collection_name = str(body.collection or "regulations").strip() or "regulations"
    if list_or_policy:
        kw_refs = _chat_keyword_search_program_refs(
            collection_name,
            query,
            allowed_category=allowed_category,
            limit=80,
        )
        candidates = _chat_merge_ref_pools(candidates, kw_refs)

    in_scope = bool(intent.get("is_system_record_writing")) or _chat_heuristic_program_scope(query)
    if not in_scope and len(candidates) < 1:
        return {
            "request_id": request_id,
            "need_human": True,
            "answer": "",
            "confidence": 0.0,
            "reason": intent.get("reason") or "问题不在程序文件知识库问答范围内。",
            "references": [],
            "model_used": (getattr(settings, "llm_model", "") or "").strip(),
            "latency_ms": int((time.time() - started) * 1000),
        }
    if len(candidates) < 1:
        return {
            "request_id": request_id,
            "need_human": True,
            "answer": "",
            "confidence": 0.0,
            "reason": "未检索到程序文件依据，请确认知识库已导入 program 类文档。",
            "references": [],
            "model_used": (getattr(settings, "llm_model", "") or "").strip(),
            "latency_ms": int((time.time() - started) * 1000),
        }
    detail_items: List[Dict[str, Any]] = []
    if list_or_policy:
        detail_pool = _chat_finalize_policy_refs(candidates, query)
        detail_items = _chat_build_policy_file_items(detail_pool, query, context=20)
        scored_refs = detail_pool[: max(1, min(len(detail_pool), 20))]
        top_score = float(detail_items[0].get("match_score") or 0.0) if detail_items else 0.0
        rerank_reason = f"关键词库共匹配 {len(detail_items)} 份文件（{len(detail_pool)} 条片段）"
    else:
        rerank_k = min(max(top_k, 8 if list_or_policy else top_k), 10)
        scored_refs, top_score, rerank_reason = _chat_rerank_refs_with_llm(
            search_query,
            candidates,
            top_k=rerank_k,
            current_provider=eff_provider,
            client_llm=client_llm if client_llm.has_any() else None,
            header_provider=hp,
            list_or_policy=list_or_policy,
        )
        scored_refs = _chat_filter_relevant_refs(
            scored_refs,
            query,
            min_score=0.35,
            list_or_policy=False,
        )
    if len(scored_refs) < 1:
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
        content_top = float(scored_refs[0].get("content_relevance") or 0.0) if scored_refs else 0.0
        if content_top < min_similarity - 0.08:
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
        top_score = max(top_score, content_top)

    refs_prompt = []
    prompt_ref_n = len(scored_refs) if list_or_policy else min(top_k, len(scored_refs))
    for idx, r in enumerate(scored_refs[:prompt_ref_n], start=1):
        snippet = (r["content"] or "")[:480]
        refs_prompt.append(
            f"[{idx}] 文件: {r['file_name']}\n分类: {r['category']}\n片段: {snippet}"
        )

    n_ref_files = len(detail_items) if list_or_policy else len({str(r.get("file_name") or "") for r in scored_refs})
    if list_or_policy and detail_items:
        max_summary_chars = min(8000, max(max_summary_chars, 48 + n_ref_files * 88))

    llm_out: Dict[str, Any] = {}
    llm_provider_used = eff_provider
    if list_or_policy and detail_items:
        answer_summary, answer_detail, n_composed = _chat_compose_policy_answer_from_items(
            detail_items,
            query,
            max_summary_chars=max_summary_chars,
        )
        if n_composed >= 1:
            confidence = min(0.92, 0.62 + n_composed * 0.04)
            if confidence < conf_threshold:
                confidence = conf_threshold
            need_human = False
            llm_out = {
                "need_human": False,
                "confidence": confidence,
                "reason": (
                    f"已汇总 {n_composed} 份文件、{len(detail_items)} 条关键词匹配片段"
                ),
            }
        else:
            answer_summary, answer_detail = "", ""
            need_human = True
            llm_out = {
                "need_human": True,
                "confidence": 0.0,
                "reason": "未找到与关键词匹配的程序文件片段。",
            }
        answer = answer_summary
    elif list_or_policy:
        need_human = True
        answer_summary, answer_detail = "", ""
        answer = ""
        llm_out = {
            "need_human": True,
            "confidence": 0.0,
            "reason": rerank_reason or "未检索到匹配片段",
        }
    else:
        answer_prompt = (
            "你是医疗器械质量管理体系程序文件问答助手。\n"
            "任务：根据给定的「程序文件」片段回答用户问题。"
            "可能是「体系运行记录怎么写」，也可能是「程序文件中的规定"
            "（如哪些文件需更新/复审、更新周期、受控要求等）」。\n"
            "限制：\n"
            "1) 仅可依据给定程序文件片段回答；片段中无依据则 need_human=true。\n"
            "2) 不编造文件编号、周期或清单；可归纳多条片段，须注明来源文件名。\n"
            "3) 不回答「某条记录具体日期填哪天、签字人员写谁」类问题（need_human=true）。\n"
            "4) 不输出法规注册结论。\n"
            "5) answer_summary 简洁可发群；answer_detail 可列条目与依据。\n\n"
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
        answer_summary, answer_detail = _chat_merge_llm_with_ref_coverage(
            answer_summary,
            answer_detail,
            scored_refs,
            query,
            max_summary_chars=max_summary_chars,
            max_detail_chars=max_detail_chars,
        )
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
        answer = answer_summary

    answer_mode = "multi_file_compose" if list_or_policy else "llm"

    ref_cap = len(detail_items) if list_or_policy else min(3, len(scored_refs))
    if list_or_policy and detail_items:
        ref_sources = detail_items
    else:
        ref_sources = _chat_diversify_refs_by_file(scored_refs, max_total=ref_cap, max_per_file=1)
    references = []
    for r in ref_sources:
        if isinstance(r, dict) and r.get("matched_excerpt") is not None:
            references.append({"file_name": r.get("file_name") or "未知"})
        else:
            references.append({"file_name": r.get("file_name") or "未知"})
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
        "detail_items": detail_items if list_or_policy else [],
        "model_used": f"{llm_provider_used}:{model_label}",
        "effective_provider": eff_provider,
        "llm_provider_used": llm_provider_used,
        "answer_mode": answer_mode,
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
            author_roles=request.author_roles,
            author_role_coverage=request.author_role_coverage,
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
            author_roles=request.author_roles,
            author_role_coverage=request.author_role_coverage,
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


@app.get("/quiz/tools/open-book-reference")
def quiz_open_book_reference(
    req: Request,
    collection: str = Query("regulations"),
    source_file: str = Query(""),
    sourceFile: str = Query(""),
):
    try:
        collection = _resolve_request_collection(req, collection)
        sf = (source_file or sourceFile or "").strip()
        if not sf:
            raise HTTPException(status_code=400, detail="缺少 source_file")
        data = quiz_service.open_book_reference(collection=collection, source_file=sf)
        return {"ok": True, "data": data}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/quiz/bank/author-role-coverage")
def quiz_bank_author_role_coverage(
    req: Request,
    collection: str = Query("regulations"),
    exam_track: str = Query(""),
    author_roles: Optional[str] = Query(None),
    authorRoles: Optional[str] = Query(None),
):
    try:
        collection = _resolve_request_collection(req, collection)
        roles_raw: List[str] = []
        for key in ("author_roles", "authorRoles"):
            for item in req.query_params.getlist(key):
                for seg in str(item or "").split(","):
                    s = seg.strip().lower()
                    if s:
                        roles_raw.append(s)
        for blob in (author_roles, authorRoles):
            if blob is None:
                continue
            for seg in str(blob).split(","):
                s = seg.strip().lower()
                if s and s not in roles_raw:
                    roles_raw.append(s)
        seen_r: set[str] = set()
        roles_norm: List[str] = []
        for r in roles_raw:
            if r in seen_r:
                continue
            seen_r.add(r)
            roles_norm.append(r)
        data = quiz_service.bank_author_role_coverage(
            collection=collection,
            exam_track=exam_track,
            author_roles=roles_norm,
        )
        return {"ok": True, "data": data}
    except Exception as e:
        logger.exception(
            "quiz_bank_author_role_coverage failed collection=%s exam_track=%s roles=%s",
            collection,
            exam_track,
            author_roles or authorRoles,
        )
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
            author_roles_present=bool(request.author_roles_present),
            author_roles=request.author_roles,
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


@app.post("/api/integration/projects")
def integration_create_project(request: Request, body: IntegrationCreateProjectBody):
    """在 aicheckword 创建专属项目（供 aiword 初稿页从页面1 项目映射后写入）。"""
    from src.core.db import create_project

    collection = _resolve_request_collection(request, body.collection or "regulations")
    name = (body.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="项目名称不能为空")
    reg_country = (body.registration_country or "").strip()
    reg_type = (body.registration_type or "").strip()
    reg_comp = (body.registration_component or "").strip()
    proj_form = (body.project_form or "").strip()
    if not reg_country or not reg_type or not reg_comp or not proj_form:
        raise HTTPException(status_code=400, detail="注册国家/类别/组成/项目形态均为必填")
    if reg_type not in REGISTRATION_TYPES:
        raise HTTPException(status_code=400, detail=f"注册类别无效，允许：{', '.join(REGISTRATION_TYPES)}")
    if reg_comp not in REGISTRATION_COMPONENTS:
        raise HTTPException(
            status_code=400,
            detail=f"注册组成无效，允许：{', '.join(REGISTRATION_COMPONENTS)}",
        )
    dims = get_dimension_options() or {}
    forms = list(dims.get("project_forms") or ["Web", "APP", "PC"])
    if proj_form not in forms:
        raise HTTPException(status_code=400, detail=f"项目形态无效，允许：{', '.join(forms)}")
    countries = list(dims.get("registration_countries") or ["中国", "美国", "欧盟"])
    if reg_country not in countries:
        raise HTTPException(status_code=400, detail=f"注册国家无效，允许：{', '.join(countries)}")
    pid = create_project(
        collection,
        name,
        reg_country,
        reg_type,
        reg_comp,
        proj_form,
        scope_of_application=(body.scope_of_application or "").strip(),
        product_name=(body.product_name or "").strip(),
        name_en=(body.name_en or "").strip(),
        product_name_en=(body.product_name_en or "").strip(),
        registration_country_en=(body.registration_country_en or "").strip(),
        model=(body.model or "").strip(),
        model_en=(body.model_en or "").strip(),
        project_code=(body.project_code or "").strip(),
    )
    return {"ok": True, "projectId": int(pid), "id": int(pid)}


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
