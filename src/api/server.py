"""FastAPI 服务：将 Agent 能力暴露为 REST API，供其他项目调用"""

import sys
import os
import logging
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

import tempfile
from typing import Any, Dict, List, Optional

import json

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import (
    get_swagger_ui_oauth2_redirect_html,
    swagger_ui_default_parameters,
)
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from config import settings
from src.core.agent import ReviewAgent
from src.core.db import (
    get_dimension_options,
    REGISTRATION_TYPES,
    REGISTRATION_COMPONENTS,
    list_projects,
    list_project_cases,
)
from config.runtime_settings import apply_runtime_config_dict, sync_cursor_overrides_from_settings
from src.core.db import load_app_settings
from src.core.quiz import service as quiz_service
from src.core.quiz.models import EXAM_TRACKS

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

_agents: Dict[str, ReviewAgent] = {}


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
    for name in ("uvicorn.error", "uvicorn"):
        for h in logging.getLogger(name).handlers:
            h.setFormatter(default_fmt)


def _build_uvicorn_log_config_with_time() -> dict:
    """供 `start_server()` 传入 uvicorn：在默认格式前增加 asctime。"""
    import copy

    from uvicorn.config import LOGGING_CONFIG

    cfg = copy.deepcopy(LOGGING_CONFIG)
    datefmt = "%Y-%m-%d %H:%M:%S"
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


class CollectionRequest(BaseModel):
    collection: str = "regulations"


class QuizGenerateSetRequest(BaseModel):
    collection: str = "regulations"
    exam_track: str = "cn"
    title: str = ""
    category: str = ""
    difficulty: str = "medium"
    question_type: str = "single_choice"
    question_count: int = 20
    created_by: str = "system"


class QuizPracticeSetRequest(BaseModel):
    collection: str = "regulations"
    exam_track: str = "cn"
    category: str = ""
    difficulty: str = "medium"
    question_type: str = "single_choice"
    question_count: int = 20
    user_id: str = ""


class QuizIngestByAIRequest(BaseModel):
    collection: str = "regulations"
    exam_track: str = "cn"
    target_count: int = 50
    review_mode: str = "auto_apply"
    category: str = ""
    difficulty: str = "medium"
    question_type: str = "single_choice"
    created_by: str = "system"
    set_title: str = ""


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
def agent_status(collection: str = "regulations"):
    agent = get_agent(collection)
    return agent.get_status()


@app.post("/train/upload")
async def train_upload(
    files: List[UploadFile] = File(...),
    collection: str = Form("regulations"),
    category: str = Form("regulation"),
):
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
def train_directory(dir_path: str = Form(...), collection: str = Form("regulations"), category: str = Form("regulation")):
    if not Path(dir_path).exists():
        raise HTTPException(status_code=404, detail=f"目录不存在：{dir_path}")
    agent = get_agent(collection)
    result = agent.train(dir_path, category=category)
    return result


@app.post("/review/upload")
async def review_upload(
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
def review_text(request: TextReviewRequest):
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
def search_knowledge(request: KnowledgeQueryRequest):
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
def knowledge_search_options(collection: str = Query("regulations", description="知识库名称")):
    """返回查询参数可选值（与页面一致）"""
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
def clear_knowledge(request: CollectionRequest):
    agent = get_agent(request.collection)
    return agent.clear_knowledge()


@app.post("/quiz/sets/generate")
def quiz_generate_set(request: QuizGenerateSetRequest):
    if request.exam_track not in EXAM_TRACKS:
        raise HTTPException(status_code=400, detail=f"exam_track 不支持: {request.exam_track}")
    try:
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
        )
        return {"ok": True, "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/quiz/practice/generate-set")
def quiz_generate_practice_set(request: QuizPracticeSetRequest):
    if request.exam_track not in EXAM_TRACKS:
        raise HTTPException(status_code=400, detail=f"exam_track 不支持: {request.exam_track}")
    try:
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/quiz/bank/ingest-by-ai")
def quiz_ingest_bank_by_ai(request: QuizIngestByAIRequest):
    if request.exam_track not in EXAM_TRACKS:
        raise HTTPException(status_code=400, detail=f"exam_track 不支持: {request.exam_track}")
    try:
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
        )
        return {"ok": True, "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    set_id: int,
    collection: str = Query("regulations"),
    created_by: str = Query("system"),
):
    try:
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
    collection: str = Query("regulations"),
    set_type: str = Query(""),
    exam_track: str = Query(""),
    status: str = Query(""),
    q: str = Query(""),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    try:
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
    body: QuizPracticeSubmitRequest,
    attempt_id: Optional[int] = Query(default=None, ge=1, description="兼容旧调用：也可仅 query 传 attempt_id"),
):
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
def quiz_grade_by_cache(attempt_id: int, collection: str = Query("regulations"), paper_id: Optional[int] = Query(None)):
    if quiz_service.is_exam_attempt(attempt_id=attempt_id):
        raise HTTPException(status_code=404, detail="exam attempt 已迁移到 aiword 本地考试，当前接口不再提供")
    try:
        data = quiz_service.grade_attempt_by_cache(collection=collection, attempt_id=attempt_id, paper_id=paper_id)
        return {"ok": True, "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/quiz/attempts/{attempt_id}/auto-grade")
def quiz_auto_grade(attempt_id: int, collection: str = Query("regulations"), paper_id: Optional[int] = Query(None)):
    if quiz_service.is_exam_attempt(attempt_id=attempt_id):
        raise HTTPException(status_code=404, detail="exam attempt 已迁移到 aiword 本地考试，当前接口不再提供")
    try:
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
def quiz_attempt_answers(attempt_id: int, collection: str = Query("regulations")):
    """
    返回 attempt 的作答明细（题干/选项/标准答案/学生答案等），用于 aiword 的考试/练习详情展示。

    说明：
    - is_correct 可能为空/不准确（若未触发自动判分）；前端可优先展示“标准答案 vs 学生答案”。
    - collection 参数仅用于兼容未来按 collection 分库的扩展；当前实现从题库表聚合即可。
    """
    if quiz_service.is_exam_attempt(attempt_id=attempt_id):
        raise HTTPException(status_code=404, detail="exam attempt 已迁移到 aiword 本地考试，当前接口不再提供")
    try:
        _ = collection  # 预留参数，避免上层固定传参导致 422
        return {"ok": True, "data": quiz_service.get_attempt_answers_with_questions(attempt_id=attempt_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/quiz/grading-rules/upsert")
def quiz_upsert_grading_rules(request: QuizUpsertGradingRuleRequest):
    try:
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
def quiz_grade_paper_by_ai(request: QuizPaperGradeByAIRequest):
    """整卷主观判分：创建异步 job，返回 job_id。证据定位仅需文件名。"""
    if request.exam_track not in EXAM_TRACKS:
        raise HTTPException(status_code=400, detail=f"exam_track 不支持: {request.exam_track}")
    if not request.items:
        raise HTTPException(status_code=422, detail="items 不能为空")
    try:
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
def quiz_bank_tracks(collection: str = "regulations"):
    try:
        return {"ok": True, "data": quiz_service.get_tracks_inventory(collection)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/quiz/wrongbook")
def quiz_wrongbook_get(
    collection: str = Query("regulations"),
    user_id: str = Query("", description="学生 user_id，由网关注入"),
    limit: int = Query(80, ge=1, le=200),
):
    try:
        return {"ok": True, "data": quiz_service.student_wrongbook(collection=collection, user_id=user_id, limit=limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/quiz/student/unpracticed-bank")
def quiz_student_unpracticed_bank(
    collection: str = Query("regulations"),
    user_id: str = Query("", description="学生 user_id，由网关注入"),
    exam_track: str = Query(""),
    limit: int = Query(100, ge=0, le=300),
):
    try:
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
    question_id: int,
    request: QuizBankQuestionPatchRequest,
    collection: str = Query("regulations"),
):
    try:
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
def quiz_bank_question_delete(question_id: int, collection: str = Query("regulations")):
    try:
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
    collection: str = Form("regulations"),
    base_checklist: Optional[str] = Form(None),
):
    agent = get_agent(collection)
    checklist = agent.generate_checklist(base_checklist=base_checklist)
    return {"checklist": checklist, "total_points": len(checklist)}


@app.post("/checklist/train")
def train_checklist(
    collection: str = Form("regulations"),
    checklist_json: str = Form(...),
):
    import json
    try:
        checklist = json.loads(checklist_json)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"JSON 解析失败：{e}")
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


class IntegrationRecordRequest(BaseModel):
    """记录审核结果（供跟踪追溯）"""
    reviewResult: dict = {}
    aiwordTaskId: str = ""
    integrationTaskId: str = ""


@app.post("/api/integration/review-kdocs")
def integration_review_kdocs(req: IntegrationKdocsReviewRequest):
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
    collection = (req.collection or "regulations").strip()

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


@app.patch("/api/reports/{report_id}/points/{point_index}")
def api_patch_audit_report_point(
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

    row = get_audit_report_by_id(report_id)
    if not row:
        raise HTTPException(status_code=404, detail="report not found")
    root = row.get("report") or {}
    target = get_target_report_for_points(root, sub_report_index)
    points = target.get("audit_points") or []
    if point_index < 0 or point_index >= len(points):
        raise HTTPException(status_code=404, detail="point index out of range")
    patch = body.model_dump(exclude_unset=True)
    if not patch:
        raise HTTPException(status_code=400, detail="no fields to patch")
    apply_point_field_updates(points[point_index], **patch)
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
