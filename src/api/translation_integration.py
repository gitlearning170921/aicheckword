"""
文档翻译 — 集成 API（供 aiword 等调用）。

- POST /api/integration/translation/jobs ：multipart，payload(JSON) + input_files
- GET  /api/integration/translation/jobs/{job_id} ：任务状态
- GET  /api/integration/translation/jobs/{job_id}/download ：成功后下载 ZIP（含全部译文输出）

后台直接调用 ``src.translation.pipeline.translate_path``，并把所有输出文件打 ZIP。
LLM 凭据透传与 draft / audit 集成一致：X-Client-Llm-* / X-Client-Cursor-*。
"""

from __future__ import annotations

import json
import shutil
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
    _header_provider,
    _parse_client_llm,
    _safe_upload_name,
)
from src.core.db import (
    OP_TYPE_TRANSLATION,
    OP_TYPE_TRANSLATION_ERROR,
    add_operation_log,
    get_current_model_info,
    get_translation_config,
    list_project_cases,
    list_projects,
)
from src.core.llm_factory import ClientLlmConfig
from src.core.operation_logs_invalidation import invalidate_operation_logs_cache
from src.translation.models import SUPPORTED_EXTENSIONS
from src.translation.correction import save_glossary_correction_entries
from src.translation.pipeline import correct_path, translate_file


router = APIRouter(prefix="/api/integration/translation", tags=["integration-translation"])

_jobs_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}
_executor = ThreadPoolExecutor(max_workers=2)

_VALID_LANGS = ("en", "de", "zh")
_MAX_FILES_PER_JOB = 5
_MAX_CORRECT_FILES_PER_JOB = 10
_INVOKABLE_TRANSLATION_PROVIDERS = frozenset(
    {"openai", "deepseek", "lingyi", "ollama", "tongyi"}
)


def _resolve_translation_provider(raw: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """
    翻译走 invoke_chat_direct，不支持 Cursor Cloud Agents。
    返回 (实际 provider, 若发生回退则给用户看的说明)。
    """
    p = (raw or settings.provider or "deepseek").strip().lower()
    if not p:
        return "deepseek", None
    if p in _INVOKABLE_TRANSLATION_PROVIDERS:
        return p, None
    if p == "cursor":
        fb = (settings.provider or "deepseek").strip().lower()
        if fb not in _INVOKABLE_TRANSLATION_PROVIDERS:
            fb = "deepseek"
        return fb, f"翻译不支持 Cursor 代理模式，已自动改用 {fb}"
    fb = "deepseek"
    return fb, f"翻译不支持 provider={p}，已自动改用 {fb}"


class TranslationJobPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    target_lang: str = Field(default="en", description="en | de | zh")
    collection: str = Field(default="regulations")
    use_kb: bool = True
    provider: Optional[str] = None
    company_overrides: Optional[Dict[str, str]] = None
    kb_query_extra: str = ""
    # multipart 上传名 → 展示名
    display_name_map: Optional[Dict[str, str]] = None
    aiword_upload_id_map: Optional[Dict[str, str]] = None
    aiword_user_id: str = ""
    aiword_task_id: str = ""


class TranslationCorrectionJobPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    target_lang: str = Field(default="en", description="en | de | zh")
    collection: str = Field(default="regulations")
    use_kb: bool = True
    provider: Optional[str] = None
    kb_query_extra: str = ""
    manual_rules: Optional[List[Dict[str, str]]] = None
    save_glossary: bool = False
    display_name_map: Optional[Dict[str, str]] = None
    aiword_upload_id_map: Optional[Dict[str, str]] = None
    aiword_user_id: str = ""
    aiword_task_id: str = ""


@router.get("/meta")
def translation_meta(
    collection: str = Query("regulations"),
):
    """翻译页元数据：项目下拉 + 目标语言 + 公司信息配置（来源 app_settings.translation_company_config）。"""
    try:
        tcfg = get_translation_config() or {}
    except Exception:
        tcfg = {}
    try:
        projects = list_projects(collection) or []
    except Exception:
        projects = []
    try:
        cases = list_project_cases(collection) or []
    except Exception:
        cases = []
    return {
        "ok": True,
        "collection": collection,
        "target_lang_default": (tcfg.get("target_lang") or "en"),
        "company_config": (tcfg.get("company_config") if isinstance(tcfg.get("company_config"), dict) else {}) or {},
        "projects": projects,
        "cases": cases,
        "supported_target_langs": list(_VALID_LANGS),
    }


@router.get("/kb-query-extra")
def translation_kb_query_extra(
    project_id: int = Query(..., ge=1),
    registration_country: Optional[str] = Query(None),
    registration_type: Optional[str] = Query(None),
    registration_component: Optional[str] = Query(None),
    project_form: Optional[str] = Query(None),
):
    """按项目与维度拼接翻译知识库检索 hint（与 Streamlit 翻译页一致）。"""
    from src.core.integration_ui_meta import build_kb_query_extra_from_project

    countries = [x.strip() for x in (registration_country or "").split(",") if x.strip()]
    types = [x.strip() for x in (registration_type or "").split(",") if x.strip()]
    comps = [x.strip() for x in (registration_component or "").split(",") if x.strip()]
    forms = [x.strip() for x in (project_form or "").split(",") if x.strip()]
    text = build_kb_query_extra_from_project(
        int(project_id),
        registration_countries=countries or None,
        registration_types=types or None,
        registration_components=comps or None,
        project_forms=forms or None,
    )
    return {"ok": True, "data": {"kb_query_extra": text}}


def _update_job(job_id: str, **kwargs: Any) -> None:
    with _jobs_lock:
        j = _jobs.get(job_id)
        if not j:
            return
        j.update(kwargs)


def _zip_files(job_dir: Path, files: List[Path], zip_path: Path) -> None:
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


def _run_translation_job(job_id: str) -> None:
    with _jobs_lock:
        spec = dict(_jobs.get(job_id) or {})
    if not spec:
        return
    payload: TranslationJobPayload = spec["payload"]
    items: List[Dict[str, str]] = spec["items"]
    client_llm: Optional[ClientLlmConfig] = spec.get("client_llm")
    eff_provider = (spec.get("effective_provider") or "").strip() or None
    job_dir = Path(spec["job_dir"])
    out_dir = job_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _progress(msg: str, frac: float) -> None:
        _update_job(
            job_id, progress=float(frac), message=str(msg or ""), status="running"
        )

    try:
        start_msg = "开始翻译…"
        prov_note = (spec.get("provider_note") or "").strip()
        if prov_note:
            start_msg = prov_note + "；" + start_msg
        _update_job(job_id, status="running", progress=0.02, message=start_msg)
        target_lang = (payload.target_lang or "en").strip().lower()
        if target_lang not in _VALID_LANGS:
            target_lang = "en"
        company_overrides = payload.company_overrides or None
        kb_query_extra = (payload.kb_query_extra or "").strip() or None
        out_files: List[Path] = []
        failed_files: List[Dict[str, str]] = []
        shared_cache: Dict[Any, Any] = {}
        shared_glossary: Dict[Any, Any] = {}

        # 客户端 LLM 凭据：translation 当前直接走 invoke_chat_direct，provider 已是有效路由。
        # 这里把 provider 传下去；个人 Key 等通过环境/全局上下文（无侵入路径，避免破坏现有翻译流程）。
        n = max(1, len(items))
        for i, it in enumerate(items):
            fp = it["path"]
            disp = it["display_name"]
            _progress(f"翻译中（{i + 1}/{n}）：{disp}", 0.05 + 0.85 * (i / n))
            try:
                # 输出目录：保持与上传名同名子目录，避免重名冲突
                file_out_dir = out_dir / f"{i:02d}_{Path(disp).stem}"
                file_out_dir.mkdir(parents=True, exist_ok=True)
                rendered = translate_file(
                    fp,
                    output_dir=str(file_out_dir),
                    collection_name=payload.collection,
                    use_kb=bool(payload.use_kb),
                    provider=(eff_provider or None),
                    target_lang=target_lang,
                    company_overrides=company_overrides,
                    kb_query_extra=kb_query_extra,
                    translation_cache=shared_cache,
                    running_glossary=shared_glossary,
                )
                p = Path(rendered)
                if p.is_file():
                    out_files.append(p)
                add_operation_log(
                    op_type=OP_TYPE_TRANSLATION,
                    collection=payload.collection,
                    file_name=disp,
                    source="translation_integration_api",
                    extra={
                        "job_id": job_id,
                        "target_lang": target_lang,
                        "out_path": str(p),
                        "aiword_user_id": (payload.aiword_user_id or "").strip()
                        or None,
                        "aiword_task_id": (payload.aiword_task_id or "").strip()
                        or None,
                        "effective_provider": eff_provider,
                    },
                    model_info=get_current_model_info(),
                )
            except Exception as e:  # noqa: BLE001
                failed_files.append({"file": disp, "error": str(e)})
                add_operation_log(
                    op_type=OP_TYPE_TRANSLATION_ERROR,
                    collection=payload.collection,
                    file_name=disp,
                    source="translation_integration_api_error",
                    extra={"job_id": job_id, "error": str(e)},
                    model_info=get_current_model_info(),
                )

        # summary + ZIP
        summary_obj: Dict[str, Any] = {
            "ok": True,
            "job_id": job_id,
            "target_lang": target_lang,
            "effective_provider": eff_provider,
            "collection": payload.collection,
            "out_files": [str(p) for p in out_files],
            "out_file_names": [Path(p).name for p in out_files],
            "failed_files": failed_files,
        }
        summary_path = job_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary_obj, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        zip_path = job_dir / "artifacts.zip"
        _zip_files(job_dir, out_files + [summary_path], zip_path)

        translation_empty = len(out_files) == 0
        finish_status = "succeeded"
        finish_msg = "完成"
        if translation_empty:
            finish_status = "failed"
            _top = "；".join(
                f"{x.get('file', '')}: {x.get('error', '')}" for x in failed_files[:3]
            )
            finish_msg = (
                f"失败：所有文件翻译为空（失败 {len(failed_files)} 条）。"
                + (f" 主因：{_top}" if _top else "")
            )

        result_obj = {
            "target_lang": target_lang,
            "out_files": [Path(p).name for p in out_files],
            "failed_files": failed_files,
            "translation_empty": bool(translation_empty),
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
        add_operation_log(
            op_type=OP_TYPE_TRANSLATION_ERROR,
            collection=coll,
            file_name="",
            source="translation_integration_api_error",
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
            model_info=get_current_model_info(),
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


def _parse_manual_rules(
    raw_rules: Optional[List[Dict[str, str]]],
) -> List[tuple[str, str, str]]:
    out: List[tuple[str, str, str]] = []
    if not isinstance(raw_rules, list):
        return out
    for it in raw_rules:
        if not isinstance(it, dict):
            continue
        wrong = str(it.get("wrong") or "").strip()
        right = str(it.get("right") or "").strip()
        source = str(it.get("source_zh") or it.get("source") or "").strip()
        if not wrong or not right:
            continue
        out.append((wrong, right, source))
    return out


def _run_correction_job(job_id: str) -> None:
    with _jobs_lock:
        spec = dict(_jobs.get(job_id) or {})
    if not spec:
        return
    payload: TranslationCorrectionJobPayload = spec["payload"]
    items: List[Dict[str, str]] = spec["items"]
    eff_provider = (spec.get("effective_provider") or "").strip() or None
    job_dir = Path(spec["job_dir"])
    out_dir = job_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _progress(msg: str, frac: float) -> None:
        _update_job(
            job_id, progress=float(frac), message=str(msg or ""), status="running"
        )

    try:
        _update_job(job_id, status="running", progress=0.02, message="开始翻译校正…")
        target_lang = (payload.target_lang or "en").strip().lower()
        if target_lang not in _VALID_LANGS:
            target_lang = "en"
        kb_query_extra = (payload.kb_query_extra or "").strip() or None
        manual_rules = _parse_manual_rules(payload.manual_rules)
        out_files: List[Path] = []
        failed_files: List[Dict[str, str]] = []
        merged_stats: Dict[str, int] = {
            "total_blocks": 0,
            "changed_blocks": 0,
            "term_unified": 0,
            "truncation_fixed": 0,
            "numeric_fixed": 0,
            "manual_replaced": 0,
        }
        n = max(1, len(items))
        for i, it in enumerate(items):
            fp = it["path"]
            disp = it["display_name"]
            _progress(f"校正中（{i + 1}/{n}）：{disp}", 0.05 + 0.85 * (i / n))
            try:
                file_out_dir = out_dir / f"{i:02d}_{Path(disp).stem}"
                file_out_dir.mkdir(parents=True, exist_ok=True)
                rendered_paths, stats = correct_path(
                    fp,
                    output_dir=str(file_out_dir),
                    target_lang=target_lang,
                    collection_name=payload.collection,
                    use_kb=bool(payload.use_kb),
                    provider=(eff_provider or None),
                    kb_query_extra=kb_query_extra,
                    manual_rules=manual_rules,
                )
                for p in rendered_paths or []:
                    pp = Path(p)
                    if pp.is_file():
                        out_files.append(pp)
                if isinstance(stats, dict):
                    for k in merged_stats.keys():
                        try:
                            merged_stats[k] += int(stats.get(k) or 0)
                        except Exception:
                            pass
                add_operation_log(
                    op_type=OP_TYPE_TRANSLATION,
                    collection=payload.collection,
                    file_name=disp,
                    source="translation_correction_integration_api",
                    extra={
                        "job_id": job_id,
                        "target_lang": target_lang,
                        "aiword_user_id": (payload.aiword_user_id or "").strip()
                        or None,
                        "aiword_task_id": (payload.aiword_task_id or "").strip()
                        or None,
                        "effective_provider": eff_provider,
                    },
                    model_info=get_current_model_info(),
                )
            except Exception as e:  # noqa: BLE001
                failed_files.append({"file": disp, "error": str(e)})
                add_operation_log(
                    op_type=OP_TYPE_TRANSLATION_ERROR,
                    collection=payload.collection,
                    file_name=disp,
                    source="translation_correction_integration_api_error",
                    extra={"job_id": job_id, "error": str(e)},
                    model_info=get_current_model_info(),
                )

        glossary_saved = 0
        if bool(payload.save_glossary) and manual_rules:
            try:
                entries = [
                    (zh, right)
                    for _wrong, right, zh in manual_rules
                    if (zh or "").strip() and (right or "").strip()
                ]
                glossary_saved = save_glossary_correction_entries(
                    payload.collection, entries, target_lang
                )
            except Exception:
                glossary_saved = 0

        summary_obj: Dict[str, Any] = {
            "ok": True,
            "job_id": job_id,
            "target_lang": target_lang,
            "effective_provider": eff_provider,
            "collection": payload.collection,
            "out_files": [str(p) for p in out_files],
            "out_file_names": [Path(p).name for p in out_files],
            "failed_files": failed_files,
            "stats": merged_stats,
            "glossary_saved": glossary_saved,
        }
        summary_path = job_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary_obj, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        zip_path = job_dir / "artifacts.zip"
        _zip_files(job_dir, out_files + [summary_path], zip_path)

        correction_empty = len(out_files) == 0
        finish_status = "succeeded"
        finish_msg = "校正完成"
        if correction_empty:
            finish_status = "failed"
            _top = "；".join(
                f"{x.get('file', '')}: {x.get('error', '')}" for x in failed_files[:3]
            )
            finish_msg = (
                f"失败：所有文件校正为空（失败 {len(failed_files)} 条）。"
                + (f" 主因：{_top}" if _top else "")
            )

        result_obj = {
            "target_lang": target_lang,
            "out_files": [Path(p).name for p in out_files],
            "failed_files": failed_files,
            "correction_empty": bool(correction_empty),
            "zip_path": str(zip_path),
            "stats": merged_stats,
            "glossary_saved": glossary_saved,
        }
        _update_job(
            job_id,
            status=finish_status,
            progress=1.0,
            message=finish_msg,
            error=(finish_msg if finish_status == "failed" else ""),
            result=result_obj,
        )
        invalidate_operation_logs_cache()
    except Exception as e:  # noqa: BLE001
        _update_job(
            job_id,
            status="failed",
            progress=1.0,
            message="失败",
            error=str(e),
            traceback=traceback.format_exc(),
        )


@router.post("/jobs")
async def translation_create_job(
    request: Request,
    payload: str = Form(..., description="JSON：TranslationJobPayload"),
    input_files: Optional[List[UploadFile]] = File(None),
):
    input_files = input_files or []
    if not (payload or "").strip():
        raise HTTPException(status_code=400, detail="payload 不能为空")
    try:
        body = TranslationJobPayload.model_validate(json.loads(payload))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"payload JSON 无效: {e}") from e

    target_lang = (body.target_lang or "en").strip().lower()
    if target_lang not in _VALID_LANGS:
        raise HTTPException(
            status_code=400,
            detail=f"target_lang 必须是 {list(_VALID_LANGS)}",
        )
    if not input_files:
        raise HTTPException(status_code=400, detail="至少需要 1 个 input_files")
    if len(input_files) > _MAX_FILES_PER_JOB:
        raise HTTPException(
            status_code=400,
            detail=f"单次最多 {_MAX_FILES_PER_JOB} 个文件，当前 {len(input_files)} 个；请分批",
        )

    client_llm = _parse_client_llm(request)
    if not bool(getattr(settings, "draft_interop_personal_keys_only", True)):
        client_llm.personal_keys_only = False
    hdr_prov = _header_provider(request)
    raw_provider = (hdr_prov or body.provider or settings.provider or "").strip() or None
    eff_provider, provider_note = _resolve_translation_provider(raw_provider)

    job_id = uuid.uuid4().hex[:16]
    root = settings.uploads_path / "translation_api_jobs" / job_id
    root.mkdir(parents=True, exist_ok=True)
    in_dir = root / "inputs"
    in_dir.mkdir(parents=True, exist_ok=True)

    display_map = {str(k): str(v) for k, v in (body.display_name_map or {}).items()}
    items: List[Dict[str, str]] = []
    for uf in input_files or []:
        raw_name = _safe_upload_name(uf.filename)
        suffix = Path(raw_name).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            shutil.rmtree(root, ignore_errors=True)
            raise HTTPException(
                status_code=400,
                detail=(
                    f"不支持的翻译文件类型：{raw_name!r}，仅支持 "
                    f"{list(SUPPORTED_EXTENSIONS)}"
                ),
            )
        dest = in_dir / raw_name
        dest.write_bytes(await uf.read())
        disp = display_map.get(raw_name) or raw_name
        items.append({"path": str(dest), "display_name": disp})

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
            "provider_note": provider_note,
            "job_dir": str(root),
        }

    _executor.submit(_run_translation_job, job_id)
    out: Dict[str, Any] = {
        "ok": True,
        "job_id": job_id,
        "status": "queued",
        "target_lang": target_lang,
        "effective_provider": eff_provider,
    }
    if provider_note:
        out["provider_note"] = provider_note
    return out


@router.get("/jobs/{job_id}")
def translation_job_status(job_id: str):
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
def translation_job_download(job_id: str):
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
        zp, filename=f"translation_{job_id}.zip", media_type="application/zip"
    )


@router.post("/correct/jobs")
async def translation_correct_create_job(
    request: Request,
    payload: str = Form(..., description="JSON：TranslationCorrectionJobPayload"),
    input_files: Optional[List[UploadFile]] = File(None),
):
    input_files = input_files or []
    if not (payload or "").strip():
        raise HTTPException(status_code=400, detail="payload 不能为空")
    try:
        body = TranslationCorrectionJobPayload.model_validate(json.loads(payload))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"payload JSON 无效: {e}") from e

    target_lang = (body.target_lang or "en").strip().lower()
    if target_lang not in _VALID_LANGS:
        raise HTTPException(
            status_code=400,
            detail=f"target_lang 必须是 {list(_VALID_LANGS)}",
        )
    if not input_files:
        raise HTTPException(status_code=400, detail="至少需要 1 个 input_files")
    if len(input_files) > _MAX_CORRECT_FILES_PER_JOB:
        raise HTTPException(
            status_code=400,
            detail=(
                f"单次最多 {_MAX_CORRECT_FILES_PER_JOB} 个文件，"
                f"当前 {len(input_files)} 个；请分批"
            ),
        )

    hdr_prov = _header_provider(request)
    raw_provider = (hdr_prov or body.provider or settings.provider or "").strip() or None
    eff_provider, provider_note = _resolve_translation_provider(raw_provider)

    job_id = uuid.uuid4().hex[:16]
    root = settings.uploads_path / "translation_correct_api_jobs" / job_id
    root.mkdir(parents=True, exist_ok=True)
    in_dir = root / "inputs"
    in_dir.mkdir(parents=True, exist_ok=True)

    display_map = {str(k): str(v) for k, v in (body.display_name_map or {}).items()}
    items: List[Dict[str, str]] = []
    for uf in input_files or []:
        raw_name = _safe_upload_name(uf.filename)
        suffix = Path(raw_name).suffix.lower()
        if suffix not in set(SUPPORTED_EXTENSIONS).union({".zip"}):
            shutil.rmtree(root, ignore_errors=True)
            raise HTTPException(
                status_code=400,
                detail=(
                    f"不支持的校正文件类型：{raw_name!r}，仅支持 "
                    f"{list(SUPPORTED_EXTENSIONS) + ['.zip']}"
                ),
            )
        dest = in_dir / raw_name
        dest.write_bytes(await uf.read())
        disp = display_map.get(raw_name) or raw_name
        items.append({"path": str(dest), "display_name": disp})

    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "progress": 0.0,
            "message": "已排队",
            "error": "",
            "traceback": "",
            "payload": body,
            "items": items,
            "effective_provider": eff_provider,
            "provider_note": provider_note,
            "job_dir": str(root),
        }

    _executor.submit(_run_correction_job, job_id)
    out_corr: Dict[str, Any] = {
        "ok": True,
        "job_id": job_id,
        "status": "queued",
        "target_lang": target_lang,
        "effective_provider": eff_provider,
    }
    if provider_note:
        out_corr["provider_note"] = provider_note
    return out_corr


@router.get("/correct/jobs/{job_id}")
def translation_correct_job_status(job_id: str):
    return translation_job_status(job_id)


@router.get("/correct/jobs/{job_id}/download")
def translation_correct_job_download(job_id: str):
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
        zp, filename=f"translation_correct_{job_id}.zip", media_type="application/zip"
    )
