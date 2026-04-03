"""FastAPI 服务：将 Agent 能力暴露为 REST API，供其他项目调用"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

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
from pydantic import BaseModel

from config import settings
from src.core.agent import ReviewAgent
from src.core.db import (
    get_dimension_options,
    REGISTRATION_TYPES,
    REGISTRATION_COMPONENTS,
)

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
    # 仍保留 collection 供多租户场景，不对外强调
    collection: str = "regulations"


class CollectionRequest(BaseModel):
    collection: str = "regulations"


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
        },
        "results": filtered,
        "total": len(filtered),
    }


@app.get("/knowledge/search/options")
def knowledge_search_options():
    """返回查询参数可选值（与页面一致）"""
    dims = get_dimension_options()
    reg_country = dims.get("registration_countries") or ["中国", "美国", "欧盟"]
    project_forms = dims.get("project_forms") or ["Web", "APP", "PC"]
    return {
        # 兼容旧调用：仍保留顶层键
        "registration_country": reg_country,
        "registration_type": REGISTRATION_TYPES,
        "registration_component": REGISTRATION_COMPONENTS,
        "project_form": project_forms,
        "document_language": ["zh", "en", "both"],
        "project_name": {"type": "string", "required": False, "description": "可选，自由文本"},
        "project_name_en": {"type": "string", "required": False, "description": "可选，自由文本"},
        "product_name": {"type": "string", "required": False, "description": "可选，自由文本"},
        "product_name_en": {"type": "string", "required": False, "description": "可选，自由文本"},
        "model": {"type": "string", "required": False, "description": "可选，自由文本"},
        "model_en": {"type": "string", "required": False, "description": "可选，自由文本"},
        # 推荐新调用：统一字段定义，一次取全参数规则
        "fields": {
            "query": {"type": "string", "required": True, "description": "自由文本检索词"},
            "registration_country": {"type": "enum", "required": True, "options": reg_country},
            "registration_type": {"type": "enum", "required": True, "options": REGISTRATION_TYPES},
            "registration_component": {"type": "enum", "required": True, "options": REGISTRATION_COMPONENTS},
            "project_form": {"type": "enum", "required": True, "options": project_forms},
            "document_language": {"type": "enum", "required": True, "options": ["zh", "en", "both"]},
            "project_name": {"type": "string", "required": False},
            "project_name_en": {"type": "string", "required": False},
            "product_name": {"type": "string", "required": False},
            "product_name_en": {"type": "string", "required": False},
            "model": {"type": "string", "required": False},
            "model_en": {"type": "string", "required": False},
        },
    }


@app.post("/knowledge/clear")
def clear_knowledge(request: CollectionRequest):
    agent = get_agent(request.collection)
    return agent.clear_knowledge()


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
        if path in public_paths:
            route.include_in_schema = True
        else:
            route.include_in_schema = False


_expose_only_public_schema_routes()


def start_server():
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    start_server()
