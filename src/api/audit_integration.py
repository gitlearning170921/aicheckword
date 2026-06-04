"""
文档审核 — 集成 API（供 aiword 等调用）。

与 ``draft_integration`` 同构的"上游异步 job"集成模型：

- POST /api/integration/audit/jobs ：multipart，payload(JSON) + input_files；按 ``mode`` 路由
- GET  /api/integration/audit/jobs/{job_id} ：任务状态
- GET  /api/integration/audit/jobs/{job_id}/download ：成功后可下载 ZIP（reports/*.json + summary.json）
- GET  /api/integration/audit/reports/by-upload ：按 aiword 上传维度查最近一份审核报告（供"审核后修改"自动 fallback）

支持的 mode：
- single：对每个文件独立单文档审核（沿用 ``ReviewAgent.review``）
- multi：多文档一致性与模板风格审核（``ReviewAgent.review_multi_document_consistency``，至少 2 个文件）
- traceability：跨文档可追溯性专项审核（``ReviewAgent.review_traceability_cross_document``，至少 2 个文件）

LLM 凭据透传与 draft 集成完全一致：X-Client-Llm-* / X-Client-Cursor-*。
"""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
import traceback
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from config import settings
from src.api.draft_integration import (
    _expand_draft_upload_if_archive,
    _header_provider,
    _parse_client_llm,
    _safe_upload_name,
)
from src.core.agent import ReviewAgent
from src.core.audit_review_context import enrich_review_context_for_integration, peek_document_text
from src.core.db import (
    OP_TYPE_REVIEW,
    OP_TYPE_REVIEW_BATCH,
    OP_TYPE_REVIEW_ERROR,
    add_operation_log,
    get_audit_report_by_id,
    get_audit_reports_by_file_name,
    get_current_model_info,
    get_project,
    list_project_cases,
    save_audit_report,
)
from src.core.llm_factory import ClientLlmConfig
from src.core.operation_logs_invalidation import invalidate_operation_logs_cache


router = APIRouter(prefix="/api/integration/audit", tags=["integration-audit"])

_jobs_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}
_executor = ThreadPoolExecutor(max_workers=2)

_VALID_MODES = ("single", "multi", "traceability")
_SINGLE_MODE_MAX_FILES = 50


class AuditJobPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mode: str = Field(default="single", description="single | multi | traceability")
    collection: str = Field(default="regulations")
    project_id: Optional[int] = None
    document_language: str = ""
    registration_country: str = ""
    registration_country_en: str = ""
    registration_type: str = ""
    registration_component: str = ""
    project_form: str = ""
    project_name: str = ""
    project_name_en: str = ""
    product_name: str = ""
    product_name_en: str = ""
    model: str = ""
    model_en: str = ""
    scope_of_application: str = ""
    basic_info_text: str = ""
    system_functionality_text: str = ""
    provider: Optional[str] = None
    review_context: Optional[Dict[str, Any]] = None
    system_prompt: str = ""
    user_prompt: str = ""
    extra_instructions: str = ""
    # 与 Streamlit「自动匹配过往项目案例」一致，默认开启
    auto_match_case: bool = True
    # multipart 上传文件名 → aiword 端展示名（避免 tmp/临时路径名落库）
    display_name_map: Optional[Dict[str, str]] = None
    # aiword 维度：每个 multipart 上传名 → aiword UploadRecord.id，便于 by-upload 反查
    aiword_upload_id_map: Optional[Dict[str, str]] = None
    aiword_user_id: str = ""
    aiword_task_id: str = ""


def _merge_review_context_local(
    base: Optional[Dict[str, Any]],
    payload: AuditJobPayload,
    *,
    effective_provider: Optional[str],
) -> Dict[str, Any]:
    ctx: Dict[str, Any] = dict(base or {})
    for k, v in {
        "project_name": payload.project_name,
        "project_name_en": payload.project_name_en,
        "product_name": payload.product_name,
        "product_name_en": payload.product_name_en,
        "model": payload.model,
        "model_en": payload.model_en,
        "registration_country": payload.registration_country,
        "registration_country_en": payload.registration_country_en,
        "registration_type": payload.registration_type,
        "registration_component": payload.registration_component,
        "project_form": payload.project_form,
        "document_language": payload.document_language,
        "scope_of_application": payload.scope_of_application,
        "basic_info_text": payload.basic_info_text,
        "system_functionality_text": payload.system_functionality_text,
    }.items():
        if isinstance(v, str) and v.strip():
            ctx[k] = v.strip()
    if (effective_provider or "").strip():
        ctx["current_provider"] = (effective_provider or "").strip()
    return ctx


def _update_job(job_id: str, **kwargs: Any) -> None:
    with _jobs_lock:
        j = _jobs.get(job_id)
        if not j:
            return
        j.update(kwargs)


def _report_severity_summary(report: Dict[str, Any]) -> Dict[str, int]:
    out = {
        "total": int(report.get("total_points") or 0),
        "high": int(report.get("high_count") or 0),
        "medium": int(report.get("medium_count") or 0),
        "low": int(report.get("low_count") or 0),
        "info": int(report.get("info_count") or 0),
    }
    return out


def _persist_report_artifact(job_dir: Path, idx: int, report: Dict[str, Any]) -> Path:
    job_dir.mkdir(parents=True, exist_ok=True)
    sub = job_dir / "reports"
    sub.mkdir(parents=True, exist_ok=True)
    safe = (str(report.get("file_name") or "report") or "report").strip()
    safe = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in safe)[:60]
    out = sub / f"{idx:02d}_{safe or 'report'}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _zip_artifacts(job_dir: Path, files: List[Path], zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        seen: set[str] = set()
        for i, p in enumerate(files):
            if not p.is_file():
                continue
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            try:
                rel = p.relative_to(job_dir)
                arc = rel.as_posix()
            except ValueError:
                arc = f"{i:03d}_{p.name}"
            zf.write(p, arcname=arc)


def _run_audit_job(job_id: str) -> None:
    with _jobs_lock:
        spec = dict(_jobs.get(job_id) or {})
    if not spec:
        return
    payload: AuditJobPayload = spec["payload"]
    items: List[Dict[str, str]] = spec["items"]
    client_llm: Optional[ClientLlmConfig] = spec.get("client_llm")
    eff_provider = (spec.get("effective_provider") or "").strip() or None
    job_dir = Path(spec["job_dir"])

    def _progress(msg: str, frac: float) -> None:
        _update_job(
            job_id, progress=float(frac), message=str(msg or ""), status="running"
        )

    try:
        _update_job(job_id, status="running", progress=0.02, message="开始审核…")
        mode = (payload.mode or "single").lower().strip() or "single"
        review_ctx = _merge_review_context_local(
            payload.review_context, payload, effective_provider=eff_provider
        )
        _peek_path = items[0]["path"] if items else ""
        _peek_name = items[0].get("display_name") or "" if items else ""
        review_ctx = enrich_review_context_for_integration(
            review_ctx,
            collection=payload.collection,
            project_id=payload.project_id,
            doc_text_for_case_match=peek_document_text(
                _peek_path,
                display_name=_peek_name,
            )
            if _peek_path
            else "",
            auto_match_case=bool(getattr(payload, "auto_match_case", True)),
        )
        # 个人 Key 模式：把客户端 llm 凭据塞进上下文供底层 invoke_chat_direct 使用
        if client_llm and client_llm.has_any():
            review_ctx["_client_llm"] = {
                "api_key": client_llm.api_key,
                "base_url": client_llm.base_url,
                "model": client_llm.model,
                "personal_keys_only": bool(client_llm.personal_keys_only),
            }

        agent = ReviewAgent(payload.collection)
        reports: List[Dict[str, Any]] = []
        report_ids: List[int] = []
        reports_summary: List[Dict[str, Any]] = []
        failed_files: List[Dict[str, str]] = []
        report_artifacts: List[Path] = []
        mi = get_current_model_info()
        eff_mi = mi if not eff_provider else f"{mi} | job_provider={eff_provider}"

        if mode == "single":
            n = max(1, len(items))
            for i, it in enumerate(items):
                fp = it["path"]
                disp = it["display_name"]
                upload_id = it.get("aiword_upload_id") or ""
                _progress(f"审核中（{i + 1}/{n}）：{disp}", 0.05 + 0.85 * (i / n))
                try:
                    report = agent.review(
                        fp,
                        project_id=payload.project_id,
                        review_context=review_ctx,
                        system_prompt=(payload.system_prompt or None),
                        user_prompt=(payload.user_prompt or None),
                        extra_instructions=(payload.extra_instructions or None),
                        display_file_name=disp,
                    )
                    report["original_filename"] = disp
                    if upload_id:
                        report["aiword_upload_id"] = upload_id
                    try:
                        rid = save_audit_report(
                            payload.collection, report, model_info=eff_mi
                        )
                        if rid:
                            report["id"] = rid
                            report_ids.append(int(rid))
                    except Exception:
                        # 落库失败也不阻塞 job，仅记录在产物里
                        pass
                    reports.append(report)
                    reports_summary.append(
                        {
                            "file": disp,
                            "report_id": report.get("id"),
                            "aiword_upload_id": upload_id or None,
                            **_report_severity_summary(report),
                        }
                    )
                    report_artifacts.append(
                        _persist_report_artifact(job_dir, i, report)
                    )
                    add_operation_log(
                        op_type=OP_TYPE_REVIEW,
                        collection=payload.collection,
                        file_name=disp,
                        source="audit_integration_api",
                        extra={
                            "job_id": job_id,
                            "mode": mode,
                            "aiword_upload_id": upload_id or None,
                            "aiword_user_id": (payload.aiword_user_id or "").strip()
                            or None,
                            "aiword_task_id": (payload.aiword_task_id or "").strip()
                            or None,
                            "effective_provider": eff_provider,
                            "report_id": report.get("id"),
                            **_report_severity_summary(report),
                        },
                        model_info=eff_mi,
                    )
                except Exception as e:  # noqa: BLE001
                    failed_files.append({"file": disp, "error": str(e)})
                    reports.append({"file_name": disp, "error": str(e)})
                    reports_summary.append({"file": disp, "error": str(e)})
                    add_operation_log(
                        op_type=OP_TYPE_REVIEW_ERROR,
                        collection=payload.collection,
                        file_name=disp,
                        source="audit_integration_api_error",
                        extra={"job_id": job_id, "error": str(e)},
                        model_info=eff_mi,
                    )
        else:
            # multi / traceability：拼成 (path, display_name) 列表
            if len(items) < 2:
                raise RuntimeError("multi / traceability 模式至少需要 2 个文件")
            pairs = [(it["path"], it["display_name"]) for it in items]
            _progress(f"审核中（{mode}）：合并 {len(pairs)} 份文档", 0.2)
            if mode == "multi":
                report = agent.review_multi_document_consistency(
                    pairs,
                    project_id=payload.project_id,
                    review_context=review_ctx,
                )
            elif mode == "traceability":
                report = agent.review_traceability_cross_document(
                    pairs,
                    project_id=payload.project_id,
                    review_context=review_ctx,
                )
            else:
                raise RuntimeError(f"未知 mode: {mode}")
            try:
                rid = save_audit_report(payload.collection, report, model_info=eff_mi)
                if rid:
                    report["id"] = rid
                    report_ids.append(int(rid))
            except Exception:
                pass
            reports.append(report)
            reports_summary.append(
                {
                    "file": report.get("file_name") or mode,
                    "report_id": report.get("id"),
                    "related_files": [it["display_name"] for it in items],
                    **_report_severity_summary(report),
                }
            )
            report_artifacts.append(_persist_report_artifact(job_dir, 0, report))
            add_operation_log(
                op_type=OP_TYPE_REVIEW_BATCH,
                collection=payload.collection,
                file_name=str(report.get("file_name") or mode),
                source="audit_integration_api",
                extra={
                    "job_id": job_id,
                    "mode": mode,
                    "aiword_user_id": (payload.aiword_user_id or "").strip() or None,
                    "aiword_task_id": (payload.aiword_task_id or "").strip() or None,
                    "report_id": report.get("id"),
                    "effective_provider": eff_provider,
                    **_report_severity_summary(report),
                },
                model_info=eff_mi,
            )

        # 产物：summary.json + reports/*.json → 打 ZIP
        summary_obj: Dict[str, Any] = {
            "ok": True,
            "job_id": job_id,
            "mode": mode,
            "effective_provider": eff_provider,
            "collection": payload.collection,
            "report_ids": report_ids,
            "reports_summary": reports_summary,
            "failed_files": failed_files,
        }
        summary_path = job_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary_obj, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        zip_path = job_dir / "artifacts.zip"
        _zip_artifacts(job_dir, report_artifacts + [summary_path], zip_path)

        # 失败显式化（沿用 draft 经验：上游"语义异常"也算 failed）
        audit_failed = False
        finish_msg = "完成"
        finish_status = "succeeded"
        if mode == "single":
            success_n = len([r for r in reports if "error" not in r])
            if success_n == 0:
                audit_failed = True
        else:
            r = reports[0] if reports else {}
            if (
                not isinstance(r, dict)
                or r.get("error")
                or int(r.get("total_points") or 0) == 0
            ):
                audit_failed = True
        if audit_failed:
            finish_status = "failed"
            if mode == "single":
                _top = "；".join(
                    f"{x.get('file', '')}: {x.get('error', '')}"
                    for x in failed_files[:3]
                )
                finish_msg = (
                    f"失败：所有文件审核失败（共 {len(failed_files)} 条）。"
                    + (f" 主因：{_top}" if _top else "")
                )
            else:
                finish_msg = (
                    f"失败：{mode} 审核未产出有效审核点（或返回错误）。"
                )

        result_obj = {
            "mode": mode,
            "report_ids": report_ids,
            "reports_summary": reports_summary,
            "failed_files": failed_files,
            "audit_failed": bool(audit_failed),
            "zip_path": str(zip_path),
        }
        _update_job(
            job_id,
            status=finish_status,
            progress=1.0,
            message=finish_msg,
            error=(finish_msg if finish_status == "failed" else ""),
            result=result_obj,
            client_llm=None,
        )
        invalidate_operation_logs_cache()
    except Exception as e:  # noqa: BLE001
        pl = spec.get("payload")
        coll = getattr(pl, "collection", None) or "regulations"
        mi = get_current_model_info()
        add_operation_log(
            op_type=OP_TYPE_REVIEW_ERROR,
            collection=coll,
            file_name="",
            source="audit_integration_api_error",
            extra={
                "job_id": job_id,
                "done": False,
                "error": str(e),
                "aiword_user_id": (
                    getattr(pl, "aiword_user_id", "") or ""
                ).strip()
                or None,
                "aiword_task_id": (
                    getattr(pl, "aiword_task_id", "") or ""
                ).strip()
                or None,
            },
            model_info=mi,
        )
        invalidate_operation_logs_cache()
        _update_job(
            job_id,
            status="failed",
            progress=1.0,
            message="失败",
            error=str(e),
            traceback=traceback.format_exc(),
            client_llm=None,
        )


@router.post("/jobs")
async def audit_create_job(
    request: Request,
    payload: str = Form(..., description="JSON：AuditJobPayload"),
    input_files: Optional[List[UploadFile]] = File(None),
):
    input_files = input_files or []
    if not (payload or "").strip():
        raise HTTPException(status_code=400, detail="payload 不能为空")
    try:
        body = AuditJobPayload.model_validate(json.loads(payload))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"payload JSON 无效: {e}") from e

    try:
        from src.api.server import _resolve_request_collection
        body.collection = _resolve_request_collection(request, body.collection or "regulations")
    except Exception:
        pass

    mode = (body.mode or "single").lower().strip() or "single"
    if mode not in _VALID_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"mode 必须是 {list(_VALID_MODES)}，当前 {mode!r}",
        )
    if not input_files:
        raise HTTPException(status_code=400, detail="至少需要 1 个 input_files")

    client_llm = _parse_client_llm(request)
    if not bool(getattr(settings, "draft_interop_personal_keys_only", True)):
        client_llm.personal_keys_only = False
    hdr_prov = _header_provider(request)
    eff_provider = (
        hdr_prov or body.provider or settings.provider or ""
    ).strip() or None

    job_id = uuid.uuid4().hex[:16]
    root = settings.uploads_path / "audit_api_jobs" / job_id
    root.mkdir(parents=True, exist_ok=True)
    in_dir = root / "inputs"
    in_dir.mkdir(parents=True, exist_ok=True)

    display_map = {str(k): str(v) for k, v in (body.display_name_map or {}).items()}
    upload_id_map = {
        str(k): str(v) for k, v in (body.aiword_upload_id_map or {}).items()
    }
    items: List[Dict[str, str]] = []
    for i, uf in enumerate(input_files or []):
        raw_name = _safe_upload_name(uf.filename)
        dest = in_dir / raw_name
        dest.write_bytes(await uf.read())
        try:
            expanded = _expand_draft_upload_if_archive(
                dest, raw_name, in_dir, slot_tag=f"in{i}"
            )
        except Exception as e:
            shutil.rmtree(root, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(e)) from e
        for pth, disp_in_zip in expanded:
            # display_name 优先级：调用方显式映射 > 压缩包内相对路径 > 原始上传名
            disp = display_map.get(raw_name) or display_map.get(disp_in_zip) or disp_in_zip or raw_name
            upload_id = (
                upload_id_map.get(raw_name)
                or upload_id_map.get(disp_in_zip)
                or ""
            )
            items.append(
                {
                    "path": str(pth),
                    "display_name": disp,
                    "aiword_upload_id": upload_id,
                }
            )

    if mode == "single" and len(items) > _SINGLE_MODE_MAX_FILES:
        shutil.rmtree(root, ignore_errors=True)
        raise HTTPException(
            status_code=400,
            detail=(
                f"single 模式单次最多 {_SINGLE_MODE_MAX_FILES} 个文件，当前 {len(items)} 个；请分批"
            ),
        )
    if mode in ("multi", "traceability") and len(items) < 2:
        shutil.rmtree(root, ignore_errors=True)
        raise HTTPException(
            status_code=400,
            detail=f"{mode} 模式至少需要 2 个文件，当前 {len(items)} 个",
        )

    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "progress": 0.0,
            "message": "已排队",
            "error": "",
            "traceback": "",
            "payload": body,
            "items": items,
            "client_llm": client_llm if client_llm.has_any() else None,
            "effective_provider": eff_provider,
            "job_dir": str(root),
        }

    _executor.submit(_run_audit_job, job_id)
    return {"ok": True, "job_id": job_id, "status": "queued", "mode": mode}


@router.get("/jobs/{job_id}")
def audit_job_status(job_id: str):
    with _jobs_lock:
        j = _jobs.get(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="job not found")
    return {
        "ok": True,
        "job_id": job_id,
        "status": j.get("status"),
        "progress": j.get("progress"),
        "message": j.get("message"),
        "error": j.get("error") or None,
        "result": j.get("result"),
    }


@router.get("/jobs/{job_id}/download")
def audit_job_download(job_id: str):
    with _jobs_lock:
        j = _jobs.get(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="job not found")
    if j.get("status") not in ("succeeded", "failed"):
        raise HTTPException(status_code=400, detail="job not completed")
    res = j.get("result") or {}
    zp = (res.get("zip_path") or "").strip()
    if not zp or not Path(zp).is_file():
        raise HTTPException(status_code=404, detail="zip not found")
    return FileResponse(
        zp, filename=f"audit_{job_id}.zip", media_type="application/zip"
    )


@router.get("/reports/by-upload")
def audit_reports_by_upload(
    request: Request,
    aiword_upload_id: str = Query("", description="aiword UploadRecord.id"),
    file_name: str = Query("", description="展示名（fallback）"),
    collection: str = Query("regulations"),
    limit: int = Query(5, ge=1, le=20),
):
    try:
        from src.api.server import _resolve_request_collection
        collection = _resolve_request_collection(request, collection)
    except Exception:
        pass
    """按 aiword 上传维度查最近的审核报告。

    匹配优先级：
    1) report JSON 中 ``aiword_upload_id`` 等于 ``aiword_upload_id``
    2) 展示名等于 ``file_name``
    返回字段：list of {id, file_name, summary, total_points, high/medium/low/info_count, created_at, report}
    """
    if not (aiword_upload_id or "").strip() and not (file_name or "").strip():
        raise HTTPException(
            status_code=400,
            detail="aiword_upload_id 与 file_name 至少一个非空",
        )
    out: List[Dict[str, Any]] = []
    target_uid = (aiword_upload_id or "").strip()
    if target_uid:
        # 暂无单独索引：从近 800 条历史按 collection 反查
        rows = get_audit_reports_by_file_name(collection, file_name or "", limit=800)
        if not rows and (file_name or "").strip():
            rows = get_audit_reports_by_file_name(collection, file_name, limit=800)
        for row in rows or []:
            rep = row.get("report") if isinstance(row, dict) else None
            if not isinstance(rep, dict):
                continue
            ru = str(rep.get("aiword_upload_id") or "").strip()
            if ru and ru == target_uid:
                out.append(_compact_report_row(row))
                if len(out) >= limit:
                    break
    if not out and (file_name or "").strip():
        rows = get_audit_reports_by_file_name(collection, file_name, limit=limit)
        for row in rows or []:
            out.append(_compact_report_row(row))
            if len(out) >= limit:
                break
    return {"ok": True, "collection": collection, "items": out}


def _compact_report_row(row: Dict[str, Any]) -> Dict[str, Any]:
    rep = row.get("report") if isinstance(row, dict) else None
    if not isinstance(rep, dict):
        rep = {}
    return {
        "id": row.get("id"),
        "file_name": row.get("file_name"),
        "summary": (row.get("summary") or "")[:1000],
        "total_points": row.get("total_points"),
        "high_count": row.get("high_count"),
        "medium_count": row.get("medium_count"),
        "low_count": row.get("low_count"),
        "info_count": row.get("info_count"),
        "created_at": (
            str(row.get("created_at")) if row.get("created_at") is not None else None
        ),
        "report": rep,
    }


@router.get("/reports/{report_id}")
def audit_report_full(report_id: int):
    """读取完整审核报告（供 aiword「审核后修改」拉取 audit_remediation 用）。"""
    rec = get_audit_report_by_id(int(report_id))
    if not rec:
        raise HTTPException(status_code=404, detail="report not found")
    return {"ok": True, "data": _compact_report_row(rec)}


class ImmediateRemediationBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    report: Dict[str, Any] = Field(default_factory=dict)
    selected_refs: Optional[List[str]] = None


def _immediate_remediation_data(
    report: Dict[str, Any],
    *,
    selected_refs: Optional[set] = None,
) -> Dict[str, Any]:
    """与 Streamlit「审核后修改」同源：``build_immediate_audit_remediation_by_target``。"""
    from src.core.audit_handoff import build_immediate_audit_remediation_by_target

    if not isinstance(report, dict):
        report = {}
    payload = build_immediate_audit_remediation_by_target(
        report,
        selected_refs=selected_refs,
    )
    text_by_target = payload.get("text_by_target") or {}
    if not isinstance(text_by_target, dict):
        text_by_target = {}
    return {
        "text_by_target": text_by_target,
        "targets": list(text_by_target.keys()),
        "immediate_count": len(payload.get("all_points") or []),
        "all_points": payload.get("all_points") or [],
        "points_by_target": payload.get("points_by_target") or {},
    }


@router.get("/reports/{report_id}/immediate-remediation")
def audit_report_immediate_remediation(
    report_id: int,
    selected_refs: Optional[str] = Query(
        None, description="逗号分隔的 audit_point_ref，空则包含全部「立即修改」点"
    ),
):
    """按报告构建「立即修改」待落实文本（与 Streamlit 第三步→审核后修改一致）。"""
    rec = get_audit_report_by_id(int(report_id))
    if not rec:
        raise HTTPException(status_code=404, detail="report not found")
    rep = rec.get("report") if isinstance(rec, dict) else {}
    refs_set: Optional[set] = None
    if selected_refs:
        refs_set = {x.strip() for x in str(selected_refs).split(",") if x.strip()}
    data = _immediate_remediation_data(rep if isinstance(rep, dict) else {}, selected_refs=refs_set)
    if not data.get("text_by_target"):
        raise HTTPException(
            status_code=400,
            detail="当前报告没有可执行的「立即修改」审核点（高/中风险默认视为立即修改）",
        )
    return {"ok": True, "data": data}


@router.post("/reports/immediate-remediation")
def audit_report_immediate_remediation_from_body(body: ImmediateRemediationBody):
    """从完整 report JSON 构建 immediate-remediation（供 aiword 上传 report.json 兜底）。"""
    refs_set: Optional[set] = None
    if body.selected_refs:
        refs_set = {str(x).strip() for x in body.selected_refs if str(x).strip()}
    data = _immediate_remediation_data(body.report or {}, selected_refs=refs_set)
    if not data.get("text_by_target"):
        raise HTTPException(
            status_code=400,
            detail="当前报告没有可执行的「立即修改」审核点",
        )
    return {"ok": True, "data": data}


@router.get("/projects/{project_id}/review-fields")
def audit_project_review_fields(project_id: int):
    """按 project_id 返回审核/初稿 payload 常用项目字段（来自 aicheckword DB）。"""
    from src.core.integration_ui_meta import project_row_for_integration

    proj = get_project(int(project_id))
    if not proj:
        raise HTTPException(status_code=404, detail="project not found")
    return {"ok": True, "data": project_row_for_integration(proj)}


@router.get("/prompt-defaults")
def audit_prompt_defaults_api():
    """审核页默认 system/user/extra 提示词（与 Streamlit ③ 一致）。"""
    from src.core.integration_ui_meta import audit_prompt_defaults

    return {"ok": True, "data": audit_prompt_defaults()}


@router.get("/reports/{report_id}/post-audit-defaults")
def audit_post_audit_defaults(report_id: int, project_id: Optional[int] = Query(None)):
    """审核后修改表单维度默认值（报告 _review_meta + 可选 project_id 补全）。"""
    from src.core.integration_ui_meta import (
        merge_project_into_post_audit_meta,
        post_audit_form_meta_defaults_from_report,
    )

    rec = get_audit_report_by_id(int(report_id))
    if not rec:
        raise HTTPException(status_code=404, detail="report not found")
    rep = rec.get("report") if isinstance(rec, dict) else {}
    meta = post_audit_form_meta_defaults_from_report(rep if isinstance(rep, dict) else {})
    pid = project_id
    if not pid:
        try:
            pid = int(meta.get("project_id") or 0) or None
        except (TypeError, ValueError):
            pid = None
    if pid:
        proj = get_project(int(pid))
        meta = merge_project_into_post_audit_meta(meta, proj)
    return {"ok": True, "data": meta}


class PrepareAuditModifyDraftBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    report_id: Optional[int] = None
    report: Optional[Dict[str, Any]] = None
    template_file_name: str = ""
    base_file_name: str = ""
    selected_refs: Optional[List[str]] = None
    collection: str = "regulations"
    project_id: Optional[int] = None
    skip_case_template_text: bool = False
    docx_track_changes: bool = True
    provider: Optional[str] = None


@router.post("/audit-modify/prepare-draft-payload")
def prepare_audit_modify_draft_payload(request: Request, body: PrepareAuditModifyDraftBody):
    """
    一步生成 aiword 提交 ``/api/integration/draft/jobs`` 所需的 payload 片段（与 Streamlit 审核后修改对齐）。
    """
    try:
        from src.api.server import _resolve_request_collection
        body.collection = _resolve_request_collection(request, body.collection or "regulations")
    except Exception:
        pass

    from src.core.integration_ui_meta import (
        collapse_remediation_to_template,
        merge_project_into_post_audit_meta,
        post_audit_form_meta_defaults_from_report,
        project_row_for_integration,
    )

    rep: dict[str, Any] = {}
    if body.report_id:
        rec = get_audit_report_by_id(int(body.report_id))
        if not rec:
            raise HTTPException(status_code=404, detail="report not found")
        rep = rec.get("report") if isinstance(rec, dict) else {}
    elif isinstance(body.report, dict):
        rep = body.report
    else:
        raise HTTPException(status_code=400, detail="需要 report_id 或 report")

    refs_set: Optional[set] = None
    if body.selected_refs:
        refs_set = {str(x).strip() for x in body.selected_refs if str(x).strip()}
    imm = _immediate_remediation_data(rep, selected_refs=refs_set)
    text_by_target = imm.get("text_by_target") or {}
    if not text_by_target:
        raise HTTPException(status_code=400, detail="没有可执行的「立即修改」审核点")

    template_key = (body.template_file_name or "").strip()
    if not template_key:
        template_key = (body.base_file_name or "").strip() or next(iter(text_by_target.keys()), "")
    remediation = collapse_remediation_to_template(
        text_by_target,
        template_key,
        body.base_file_name or "",
    )
    if not remediation:
        raise HTTPException(status_code=400, detail="remediation 对齐失败")

    template_key = next(iter(remediation.keys()))
    meta = post_audit_form_meta_defaults_from_report(rep)
    pid = body.project_id
    if not pid:
        try:
            pid = int(meta.get("project_id") or 0) or None
        except (TypeError, ValueError):
            pid = None
    proj_fields: dict[str, Any] = {}
    if pid:
        proj_fields = project_row_for_integration(get_project(int(pid)) or {})

    base_case_id: Optional[int] = None
    if not body.skip_case_template_text:
        try:
            cases = list_project_cases((body.collection or "regulations").strip() or "regulations")
            if cases:
                base_case_id = int(cases[0].get("id") or 0) or None
        except Exception:
            base_case_id = None

    draft_payload: dict[str, Any] = {
        "collection": (body.collection or "regulations").strip() or "regulations",
        "template_file_names": [template_key],
        "audit_remediation_by_target": remediation,
        "base_files_by_target": {template_key: (body.base_file_name or template_key).strip()},
        "inplace_patch": True,
        "save_as_case": False,
        "draft_strategy": "change",
        "skip_case_template_text": bool(body.skip_case_template_text),
        "docx_track_changes": bool(body.docx_track_changes),
        "base_case_id": base_case_id,
        "provider": (body.provider or "").strip() or None,
    }
    for k, v in proj_fields.items():
        if v is not None and v != "" and k not in draft_payload:
            draft_payload[k] = v
    meta = merge_project_into_post_audit_meta(meta, get_project(int(pid)) if pid else None)
    for dim_k in (
        "document_language",
        "registration_country",
        "registration_type",
        "registration_component",
        "project_form",
        "project_name",
        "product_name",
        "model",
    ):
        if str(meta.get(dim_k) or "").strip() and not str(draft_payload.get(dim_k) or "").strip():
            draft_payload[dim_k] = meta.get(dim_k)

    return {
        "ok": True,
        "data": {
            "draftPayload": draft_payload,
            "remediation": remediation,
            "immediateCount": imm.get("immediate_count"),
            "targets": imm.get("targets"),
            "postAuditMeta": meta,
        },
    }
