"""FastAPI 服务：将 Agent 能力暴露为 REST API，供其他项目调用"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import shutil
import tempfile
from typing import Dict, List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import settings
from src.core.agent import ReviewAgent

app = FastAPI(
    title="注册文档审核 Agent API",
    description="基于 RAG 的注册文档审核服务，支持训练知识库和自动审核文档",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_agents: Dict[str, ReviewAgent] = {}


def get_agent(collection: str = "regulations") -> ReviewAgent:
    if collection not in _agents:
        _agents[collection] = ReviewAgent(collection)
    return _agents[collection]


class TextReviewRequest(BaseModel):
    text: str
    file_name: str = "直接输入"
    collection: str = "regulations"


class KnowledgeQueryRequest(BaseModel):
    query: str
    top_k: int = 5
    collection: str = "regulations"


class CollectionRequest(BaseModel):
    collection: str = "regulations"


@app.get("/")
def root():
    return {"service": "注册文档审核 Agent", "version": "1.0.0", "status": "running"}


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
):
    agent = get_agent(collection)
    reports = []

    for file in files:
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            report = agent.review(tmp_path)
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
    report = agent.review_text(request.text, request.file_name)
    return report


@app.post("/knowledge/search")
def search_knowledge(request: KnowledgeQueryRequest):
    agent = get_agent(request.collection)
    results = agent.search_knowledge(request.query, request.top_k)
    return {"results": results}


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


def start_server():
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    start_server()
