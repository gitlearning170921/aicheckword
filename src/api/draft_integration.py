"""
文档初稿生成 — 集成 API（供 aiword 等调用）。

与 aiword 初稿页的联调白名单与语义说明见仓库 ``docs/integration-draft-provider-status.md``。

- POST /api/integration/draft/jobs ：multipart，字段 payload（JSON 字符串）+ input_files + base_files（可选）
- GET  /api/integration/draft/jobs/{job_id} ：任务状态
- GET  /api/integration/draft/jobs/{job_id}/download ：成功后可下载 ZIP
- GET  /api/integration/draft/meta ：项目 / 案例 / 模板文件名
- GET  /api/integration/draft/interop-config ：联调策略（允许 provider、是否须个人 Key、备注），供 aiword 拉取与表单同步

LLM 凭据（禁止写入日志与 operation_logs）：HTTP Header
  X-Client-Llm-Api-Key, X-Client-Llm-Base-Url, X-Client-Llm-Model, X-Client-Llm-Provider（可选，缺省用 payload.provider 或服务端 settings）
  Cursor 另可选：X-Client-Cursor-Repository, X-Client-Cursor-Ref（与系统设置 ``cursor_*`` 合并，请求级优先）
  X-Client-Llm-Personal-Keys-Only：为 ``1``/``true`` 时，DeepSeek/OpenAI/零一/通义/Cursor 的 **API Key 仅**用 ``X-Client-Llm-Api-Key``，不回退 ``settings``（aiword 页面2个人配置）
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
from src.core.db import (
    add_operation_log,
    get_current_model_info,
    get_project_case_file_names,
    list_project_cases,
    list_projects,
)
from src.core.document_draft_generator import DocumentDraftGenerator
from src.core.document_loader import SUPPORTED_DOC_EXTENSIONS, extract_archive, is_archive
from src.core.draft_job_artifacts import export_draft_job_files
from src.core.llm_factory import ClientLlmConfig, merged_cursor_launch_params
from src.core.operation_logs_invalidation import invalidate_operation_logs_cache

router = APIRouter(prefix="/api/integration/draft", tags=["integration-draft"])

_jobs_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}
_executor = ThreadPoolExecutor(max_workers=2)

# 集成方可能上送的 provider id（小写）；用于 interop-config 文案与任务侧白名单
_INTEROP_PROVIDER_LABELS: Dict[str, str] = {
    "deepseek": "DeepSeek (OpenAI 兼容)",
    "openai": "OpenAI",
    "lingyi": "零一万物 (OpenAI 兼容)",
    "gemini": "Google Gemini",
    "tongyi": "阿里通义千问",
    "baidu": "百度文心一言",
    "ollama": "Ollama (本地免费)",
    "cursor": "Cursor Agent (Cloud API)",
}

_DEFAULT_INTEROP_DISPLAY_IDS: tuple[str, ...] = tuple(sorted(_INTEROP_PROVIDER_LABELS.keys()))


def _draft_provider_whitelist_for_jobs() -> Optional[frozenset[str]]:
    """若管理员配置非空，则初稿任务仅允许列出的 provider；None 表示不限制。"""
    raw = (getattr(settings, "draft_interop_allowed_providers", None) or "").strip()
    if not raw:
        return None
    s = {x.strip().lower() for x in raw.split(",") if x.strip()}
    return frozenset(s) if s else None


def _interop_config_dict() -> Dict[str, Any]:
    w = _draft_provider_whitelist_for_jobs()
    notes = (getattr(settings, "draft_interop_notes", None) or "").strip()
    pko = bool(getattr(settings, "draft_interop_personal_keys_only", True))
    if w is None:
        ids = list(_DEFAULT_INTEROP_DISPLAY_IDS)
    else:
        ids = sorted(w)
    providers: List[Dict[str, Any]] = [
        {
            "id": i,
            "label": _INTEROP_PROVIDER_LABELS.get(i, i),
            "requiresApiKey": True,
        }
        for i in ids
    ]
    return {
        "ok": True,
        "restrictProviders": w is not None,
        "allowedProviders": providers,
        "personalKeysOnly": pko,
        "adminNotes": notes,
    }


class DraftGeneratePayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    collection: str = Field(default="regulations")
    base_case_id: int = Field(default=0)
    template_file_names: Optional[List[str]] = None
    project_id: Optional[int] = None
    document_language: str = Field(default="zh")
    registration_country: str = ""
    registration_type: str = ""
    registration_component: str = ""
    project_form: str = ""
    project_name: str = ""
    project_code: str = ""
    project_name_en: str = ""
    product_name: str = ""
    product_name_en: str = ""
    model: str = ""
    model_en: str = ""
    registration_country_en: str = ""
    scope_of_application_override: Optional[str] = None
    persist_project_fields: bool = True
    new_case_name: str = ""
    project_key: str = ""
    skills_patch_text: str = ""
    rules_patch_text: str = ""
    provider: Optional[str] = None
    inplace_patch: bool = False
    save_as_case: bool = True
    base_case_limit_chunks: int = 2000
    multi_base_auto_route: bool = False
    draft_strategy: str = "change"
    author_role: str = ""
    author_role_map: Optional[Dict[str, str]] = None
    audit_remediation_by_target: Optional[Dict[str, str]] = None
    skip_case_template_text: bool = False
    # True（集成默认）：跳过 train_project_docs，避免每份输入在 Ollama 等 embed 上耗时；基本信息从本地摘录提取。
    # 若需在 aicheckword「项目专属资料」中保留本次输入的可检索向量，请显式传 false。
    skip_input_vector_training: bool = True
    docx_track_changes: bool = False
    # 模板目标文件名 -> 上传的基底文件原始文件名（须与 base_files 上传名一致）
    base_files_by_target: Optional[Dict[str, str]] = None
    aiword_user_id: str = ""
    aiword_task_id: str = ""


def _parse_client_llm(request: Request) -> ClientLlmConfig:
    h = request.headers
    api_key = (h.get("x-client-llm-api-key") or h.get("X-Client-Llm-Api-Key") or "").strip()
    base_url = (h.get("x-client-llm-base-url") or h.get("X-Client-Llm-Base-Url") or "").strip()
    model = (h.get("x-client-llm-model") or h.get("X-Client-Llm-Model") or "").strip()
    cursor_repository = (
        h.get("x-client-cursor-repository") or h.get("X-Client-Cursor-Repository") or ""
    ).strip()
    cursor_ref = (h.get("x-client-cursor-ref") or h.get("X-Client-Cursor-Ref") or "").strip()
    pk = (h.get("x-client-llm-personal-keys-only") or h.get("X-Client-Llm-Personal-Keys-Only") or "").strip().lower()
    personal_keys_only = pk in ("1", "true", "yes", "on")
    return ClientLlmConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        cursor_repository=cursor_repository,
        cursor_ref=cursor_ref,
        personal_keys_only=personal_keys_only,
    )


def _header_provider(request: Request) -> str:
    h = request.headers
    return (h.get("x-client-llm-provider") or h.get("X-Client-Llm-Provider") or "").strip()


def _safe_upload_name(name: Optional[str]) -> str:
    p = Path(name or "").name.strip()
    if not p or p in (".", ".."):
        return "unnamed.bin"
    return p


def _expand_draft_upload_if_archive(
    dest: Path, raw_name: str, dest_parent: Path, *, slot_tag: str
) -> List[tuple[str, str]]:
    """与主站 ``_expand_uploads`` 一致：压缩包解压为若干文档路径；非压缩包原样返回一条。

    解压出的文件复制到 ``dest_parent`` 下持久子目录，供后台线程读取（不可仅用 extract_archive 的临时目录）。
    """
    if not dest.is_file():
        raise ValueError(f"上传文件不存在：{dest}")
    if not is_archive(dest):
        return [(str(dest), raw_name)]
    temp_root_str, paths = extract_archive(dest)
    temp_root = Path(temp_root_str)
    try:
        stem = Path(raw_name).stem
        sub = dest_parent / f"_exp_{slot_tag}_{stem}"
        shutil.rmtree(sub, ignore_errors=True)
        sub.mkdir(parents=True, exist_ok=True)
        out: List[tuple[str, str]] = []
        for p in sorted(paths):
            if not p.is_file():
                continue
            if p.suffix.lower() not in SUPPORTED_DOC_EXTENSIONS:
                continue
            try:
                rel_part = p.relative_to(temp_root)
            except ValueError:
                rel_part = Path(p.name)
            target = sub / rel_part
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
            disp = f"{stem}/{rel_part.as_posix()}"
            out.append((str(target), disp))
        dest.unlink(missing_ok=True)
        if not out:
            raise ValueError(
                f"压缩包 {raw_name!r} 内未找到支持的文档类型；当前支持：{list(SUPPORTED_DOC_EXTENSIONS)}。"
                "请与 aicheckword 文档初稿页一致（压缩包内为 PDF/Word/Excel/TXT/Markdown 等）。"
            )
        return out
    finally:
        shutil.rmtree(str(temp_root), ignore_errors=True)


def _update_job(job_id: str, **kwargs: Any) -> None:
    with _jobs_lock:
        j = _jobs.get(job_id)
        if not j:
            return
        j.update(kwargs)


def _run_draft_job(job_id: str) -> None:
    with _jobs_lock:
        spec = dict(_jobs.get(job_id) or {})
    if not spec:
        return
    payload: DraftGeneratePayload = spec["payload"]
    input_paths: List[tuple] = spec["input_paths"]
    base_paths_map: Dict[str, str] = spec["base_paths_map"]
    client_llm: Optional[ClientLlmConfig] = spec.get("client_llm")
    eff_provider = (spec.get("effective_provider") or "").strip() or None

    def _progress(msg: str, frac: float) -> None:
        _update_job(job_id, progress=float(frac), message=str(msg or ""), status="running")

    try:
        _update_job(job_id, status="running", progress=0.02, message="开始生成…")
        base_manifest = None
        if payload.multi_base_auto_route and spec.get("base_name_to_path"):
            bmp = spec["base_name_to_path"]
            base_manifest = [(pth, nm) for nm, pth in sorted(bmp.items())]
        gen = DocumentDraftGenerator(payload.collection)
        res = gen.generate(
            base_case_id=int(payload.base_case_id),
            template_file_names=payload.template_file_names,
            project_id=payload.project_id,
            existing_base_files=base_paths_map or None,
            input_files=input_paths,
            document_language=payload.document_language,
            registration_country=payload.registration_country,
            registration_type=payload.registration_type,
            registration_component=payload.registration_component,
            project_form=payload.project_form,
            project_name=payload.project_name,
            project_code=payload.project_code,
            project_name_en=payload.project_name_en,
            product_name=payload.product_name,
            product_name_en=payload.product_name_en,
            model=payload.model,
            model_en=payload.model_en,
            registration_country_en=payload.registration_country_en,
            scope_of_application_override=payload.scope_of_application_override,
            persist_project_fields=payload.persist_project_fields,
            new_case_name=payload.new_case_name,
            project_key=payload.project_key,
            skills_patch_text=payload.skills_patch_text,
            rules_patch_text=payload.rules_patch_text,
            provider=eff_provider,
            inplace_patch=payload.inplace_patch,
            save_as_case=payload.save_as_case,
            progress_cb=_progress,
            base_case_limit_chunks=payload.base_case_limit_chunks,
            base_files_manifest=base_manifest,
            multi_base_auto_route=payload.multi_base_auto_route,
            draft_strategy=payload.draft_strategy,
            author_role=payload.author_role,
            author_role_map=payload.author_role_map,
            audit_remediation_by_target=payload.audit_remediation_by_target,
            skip_case_template_text=payload.skip_case_template_text,
            skip_input_vector_training=payload.skip_input_vector_training,
            client_llm=client_llm,
        )
        _progress("导出产物…", 0.95)
        out_log, out_by_target, summaries = export_draft_job_files(
            res=res,
            existing_base_files=base_paths_map or None,
            inplace_patch=payload.inplace_patch,
            document_language=payload.document_language,
            docx_track_changes=payload.docx_track_changes,
        )
        zip_path = Path(spec["job_dir"]) / "artifacts.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            seen = set()
            for p in out_log:
                pp = Path(p)
                if pp.is_file() and str(pp.resolve()) not in seen:
                    seen.add(str(pp.resolve()))
                    zf.write(pp, arcname=pp.name)
        mi = get_current_model_info()
        add_operation_log(
            op_type="draft_generate",
            collection=payload.collection,
            file_name="draft_outputs",
            source="draft_integration_api",
            extra={
                "batch_id": job_id,
                "job_id": job_id,
                "base_case_id": payload.base_case_id,
                "template_file_names": payload.template_file_names,
                "project_id": res.project_id,
                "project_case_id": res.project_case_id,
                "save_as_case": payload.save_as_case,
                "inplace_patch": payload.inplace_patch,
                "draft_strategy": payload.draft_strategy,
                "document_language": payload.document_language,
                "aiword_user_id": (payload.aiword_user_id or "").strip() or None,
                "aiword_task_id": (payload.aiword_task_id or "").strip() or None,
                "out_files": out_log,
                "out_files_by_target": out_by_target,
                "generated_file_names": list((out_by_target or {}).keys()),
                "per_file_patch_summaries": summaries,
            },
            model_info=mi,
        )
        add_operation_log(
            op_type="draft_generate_job",
            collection=payload.collection,
            file_name="",
            source="draft_integration_api_done",
            extra={
                "batch_id": job_id,
                "job_id": job_id,
                "done": True,
                "generated_file_count": len(res.generated_files or {}),
                "aiword_user_id": (payload.aiword_user_id or "").strip() or None,
                "aiword_task_id": (payload.aiword_task_id or "").strip() or None,
            },
            model_info=mi,
        )
        invalidate_operation_logs_cache()
        _update_job(
            job_id,
            status="succeeded",
            progress=1.0,
            message="完成",
            result={
                "project_id": res.project_id,
                "project_case_id": res.project_case_id,
                "generated_file_names": list((res.generated_files or {}).keys()),
                "zip_path": str(zip_path),
                "out_files_by_target": out_by_target,
            },
            client_llm=None,
        )
    except Exception as e:
        pl = spec.get("payload")
        coll = getattr(pl, "collection", None) or "regulations"
        mi = get_current_model_info()
        add_operation_log(
            op_type="draft_generate_job",
            collection=coll,
            file_name="",
            source="draft_integration_api_error",
            extra={
                "batch_id": job_id,
                "job_id": job_id,
                "done": False,
                "error": str(e),
                "aiword_user_id": (getattr(pl, "aiword_user_id", "") or "").strip() or None,
                "aiword_task_id": (getattr(pl, "aiword_task_id", "") or "").strip() or None,
            },
            model_info=mi,
        )
        invalidate_operation_logs_cache()
        _update_job(
            job_id,
            status="failed",
            message="失败",
            error=str(e),
            traceback=traceback.format_exc(),
            client_llm=None,
        )


@router.get("/meta")
def draft_meta(
    collection: str = Query("regulations"),
    base_case_id: Optional[int] = Query(None, description="若提供则返回该案例下模板文件名列表"),
):
    try:
        out: Dict[str, Any] = {
            "collection": collection,
            "projects": list_projects(collection) or [],
            "cases": list_project_cases(collection) or [],
        }
        if base_case_id is not None and int(base_case_id) > 0:
            out["template_file_names"] = get_project_case_file_names(collection, int(base_case_id)) or []
        return {"ok": True, "data": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/interop-config")
def draft_interop_config():
    """联调策略：与系统配置「初稿集成」入库字段一致；aiword 可定时或进入页面时拉取。"""
    return _interop_config_dict()


@router.post("/jobs")
async def draft_create_job(
    request: Request,
    payload: str = Form(..., description="JSON：DraftGeneratePayload"),
    input_files: Optional[List[UploadFile]] = File(None),
    base_files: Optional[List[UploadFile]] = File(None),
):
    input_files = input_files or []
    base_files = base_files or []
    if not (payload or "").strip():
        raise HTTPException(status_code=400, detail="payload 不能为空")
    try:
        body = DraftGeneratePayload.model_validate(json.loads(payload))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"payload JSON 无效: {e}") from e

    client_llm = _parse_client_llm(request)
    if not bool(getattr(settings, "draft_interop_personal_keys_only", True)):
        client_llm.personal_keys_only = False
    hdr_prov = _header_provider(request)
    eff_provider = (hdr_prov or body.provider or settings.provider or "").strip() or None
    wlist = _draft_provider_whitelist_for_jobs()
    if wlist is not None and eff_provider:
        ep = eff_provider.strip().lower()
        if ep and ep not in wlist:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"当前初稿联调配置不允许 provider={ep!r}，允许：{', '.join(sorted(wlist))}"
                ),
            )
    if client_llm.personal_keys_only:
        ep = (eff_provider or "").lower()
        if ep in ("deepseek", "openai", "lingyi", "tongyi", "cursor"):
            if not (client_llm.api_key or "").strip():
                raise HTTPException(
                    status_code=400,
                    detail="X-Client-Llm-Personal-Keys-Only 模式下必须提供 X-Client-Llm-Api-Key（不使用上游系统 Key）",
                )
    if (eff_provider or "").lower() == "cursor":
        cp = merged_cursor_launch_params(client_llm)
        if not (cp.get("api_key") or "").strip():
            raise HTTPException(
                status_code=400,
                detail="Cursor 初稿需要 Cursor API Key：请求头 X-Client-Llm-Api-Key"
                + (
                    "（个人模式已禁止回退系统 Key）"
                    if client_llm.personal_keys_only
                    else " 或 aicheckword 系统设置 cursor_api_key"
                ),
            )
        if not (cp.get("repository") or "").strip():
            raise HTTPException(
                status_code=400,
                detail="Cursor 初稿需要 GitHub 仓库：请求头 X-Client-Cursor-Repository 或系统设置 cursor_repository",
            )

    job_id = uuid.uuid4().hex[:16]
    root = settings.uploads_path / "draft_api_jobs" / job_id
    root.mkdir(parents=True, exist_ok=True)
    in_dir = root / "inputs"
    base_dir = root / "bases"
    in_dir.mkdir(parents=True, exist_ok=True)
    base_dir.mkdir(parents=True, exist_ok=True)

    input_paths: List[tuple] = []
    for i, uf in enumerate(input_files or []):
        raw_name = _safe_upload_name(uf.filename)
        dest = in_dir / raw_name
        dest.write_bytes(await uf.read())
        try:
            input_paths.extend(_expand_draft_upload_if_archive(dest, raw_name, in_dir, slot_tag=f"in{i}"))
        except Exception as e:
            shutil.rmtree(root, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(e)) from e

    base_name_to_path: Dict[str, str] = {}
    for i, uf in enumerate(base_files or []):
        raw_name = _safe_upload_name(uf.filename)
        dest = base_dir / raw_name
        dest.write_bytes(await uf.read())
        try:
            expanded = _expand_draft_upload_if_archive(dest, raw_name, base_dir, slot_tag=f"bf{i}")
        except Exception as e:
            shutil.rmtree(root, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(e)) from e
        # base_files_by_target 的值须能映射到上传名；压缩包另登记包内文件名 -> 路径便于按名匹配
        base_name_to_path[raw_name] = expanded[0][0]
        for pth, disp in expanded:
            leaf = Path(disp).name
            if leaf and leaf not in base_name_to_path:
                base_name_to_path[leaf] = pth

    base_paths_map: Dict[str, str] = {}
    if body.base_files_by_target:
        for target, upload_name in (body.base_files_by_target or {}).items():
            un = (upload_name or "").strip()
            if not un:
                continue
            pth = base_name_to_path.get(un)
            if not pth:
                shutil.rmtree(root, ignore_errors=True)
                raise HTTPException(
                    status_code=400,
                    detail=f"base_files_by_target 引用的文件未在上传中找到：{un}",
                )
            base_paths_map[(target or "").strip()] = pth

    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "progress": 0.0,
            "message": "已排队",
            "error": "",
            "traceback": "",
            "payload": body,
            "input_paths": input_paths,
            "base_paths_map": base_paths_map,
            "base_name_to_path": base_name_to_path,
            "client_llm": client_llm if client_llm.has_any() else None,
            "effective_provider": eff_provider,
            "job_dir": str(root),
        }

    _executor.submit(_run_draft_job, job_id)
    return {"ok": True, "job_id": job_id, "status": "queued"}


@router.get("/jobs/{job_id}")
def draft_job_status(job_id: str):
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
def draft_job_download(job_id: str):
    with _jobs_lock:
        j = _jobs.get(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="job not found")
    if j.get("status") != "succeeded":
        raise HTTPException(status_code=400, detail="job not completed")
    res = j.get("result") or {}
    zp = (res.get("zip_path") or "").strip()
    if not zp or not Path(zp).is_file():
        raise HTTPException(status_code=404, detail="zip not found")
    return FileResponse(zp, filename=f"draft_{job_id}.zip", media_type="application/zip")
