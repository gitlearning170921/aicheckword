"""
文档初稿生成 — 集成 API（供 aiword 等调用）。

与 aiword 初稿页的联调白名单与语义说明见仓库 ``docs/integration-draft-provider-status.md``。

- POST /api/integration/draft/jobs ：multipart，字段 payload（JSON 字符串）+ input_files + base_files（可选）
- GET  /api/integration/draft/jobs/{job_id} ：任务状态
- GET  /api/integration/draft/jobs/{job_id}/download ：成功后可下载 ZIP
- GET  /api/integration/draft/meta ：项目 / 案例 / 模板文件名（含 ``bootstrap`` UI 字段）
- GET  /api/integration/draft/page-bootstrap ：初稿页完整下拉/默认值（与 Streamlit 初稿页同源）
- GET  /api/integration/draft/suggest-author-role ：编写人员身份推断
- GET  /api/integration/draft/projects/{project_id}/draft-defaults ：选中项目时的维度默认值
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
    # 与 Streamlit 初稿页 checkbox 默认一致（value=True）；旧默认 False 会导致 aiword 未显式传参时不走就地 patch
    inplace_patch: bool = True
    save_as_case: bool = True
    base_case_limit_chunks: int = 2000
    multi_base_auto_route: bool = True
    draft_strategy: str = "change"
    author_role: str = ""
    author_role_map: Optional[Dict[str, str]] = None
    audit_remediation_by_target: Optional[Dict[str, str]] = None
    skip_case_template_text: bool = False
    # False：与 Streamlit 初稿一致，先 train_project_docs 再抽基本信息；aiword 若需提速可显式传 true。
    skip_input_vector_training: bool = False
    # 与 Streamlit 初稿页「Word 修订标记」默认勾选一致；aiword 审核后修改应显式传 true
    docx_track_changes: bool = True
    # 模板目标文件名 -> 上传的基底文件原始文件名（须与 base_files 上传名一致）
    base_files_by_target: Optional[Dict[str, str]] = None
    aiword_user_id: str = ""
    aiword_task_id: str = ""
    # 追加到每文件 LLM 提示词末尾（与仓库 skills/rules 无关）；建议写清须改章节/表等可执行指令
    user_prompt_append: str = ""
    # 参考文件在项目向量库中已存在时：skip=不重复向量化；overwrite=删除后重新向量化
    input_vector_on_duplicate: str = "skip"


class CheckInputVectorDuplicatesBody(BaseModel):
    project_id: int = Field(..., ge=1)
    file_names: List[str] = Field(default_factory=list)


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


def _integration_match_upload_key_for_template(
    template_file_name: str, base_name_to_path: Dict[str, str]
) -> Optional[str]:
    """未显式传 base_files_by_target 时，用「模板目标文件名」匹配 multipart 上传键或压缩包内叶子名。"""
    t = (template_file_name or "").strip()
    if not t or not base_name_to_path:
        return None
    leaf = Path(t).name
    keys = list(base_name_to_path.keys())

    def _pick(cands: List[str]) -> Optional[str]:
        if not cands:
            return None
        uniq = {base_name_to_path[k] for k in cands}
        if len(uniq) != 1:
            return None
        cands.sort(key=lambda s: (len(str(s)), str(s)))
        return str(cands[0])

    exact = [str(k) for k in keys if str(k) == t or str(k) == leaf]
    hit = _pick(exact)
    if hit:
        return hit
    tl, ll = t.lower(), leaf.lower()
    ci = [str(k) for k in keys if str(k).lower() == tl or Path(str(k)).name.lower() == ll]
    return _pick(ci)


def _integration_merge_base_files_by_target(
    body: DraftGeneratePayload,
    base_name_to_path: Dict[str, str],
) -> Dict[str, str]:
    """显式 base_files_by_target + 按模板文件名自动补全（与 aicheckword 本机「绑定 Base」语义对齐）。"""
    out: Dict[str, str] = {}
    if body.base_files_by_target:
        for tgt, un in body.base_files_by_target.items():
            tg = (tgt or "").strip()
            uv = (un or "").strip()
            if tg and uv:
                out[tg] = uv
    for tf in body.template_file_names or []:
        tf2 = (tf or "").strip()
        if not tf2:
            continue
        if tf2 in out and (out.get(tf2) or "").strip():
            continue
        m = _integration_match_upload_key_for_template(tf2, base_name_to_path)
        if m:
            out[tf2] = m
    return out


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


def _patch_skip_reason_histogram(summaries: List[dict]) -> Dict[str, int]:
    """从 *.patch.report.json 汇总跳过原因，便于判断「有 AI 输出但 Word 未改」。"""
    hist: Dict[str, int] = {}
    for s in summaries or []:
        if not isinstance(s, dict):
            continue
        rp = (s.get("patch_report_path") or "").strip()
        if not rp:
            continue
        try:
            p = Path(rp)
            if not p.is_file():
                continue
            report = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for sk in report.get("skipped") or []:
            if not isinstance(sk, dict):
                continue
            reason = (sk.get("reason") or "（未注明）").strip() or "（未注明）"
            hist[reason] = hist.get(reason, 0) + 1
    return hist


def _summaries_indicate_no_docx_change(summaries: List[dict]) -> bool:
    """存在 patch 产物但 applied/changes 均为 0 → 导出 docx 与基底实质相同。"""
    saw_patch = False
    for s in summaries or []:
        if not isinstance(s, dict):
            continue
        if not (s.get("patch_json_path") or s.get("patch_report_path")):
            continue
        saw_patch = True
        pc = s.get("patch_counts") if isinstance(s.get("patch_counts"), dict) else {}
        if int(pc.get("applied") or 0) > 0 or int(pc.get("changes") or 0) > 0:
            return False
    return saw_patch


def _integration_effective_draft_flags(
    payload: DraftGeneratePayload,
    *,
    base_paths_map: Dict[str, str],
    base_manifest: Optional[List[tuple[str, str]]],
) -> tuple[bool, bool]:
    """
    集成 API 与 Streamlit 初稿页对齐：
    - 有 Base（绑定或 manifest）时默认就地修改；
    - 有 base manifest 时默认多 Base 自动路由（与 app.py 提交 job 时 bool(manifest) 一致）。
    """
    has_bound_base = bool(base_paths_map)
    has_manifest = bool(base_manifest)
    inplace = bool(payload.inplace_patch)
    if (has_bound_base or has_manifest) and not inplace:
        inplace = True
    multi = bool(payload.multi_base_auto_route)
    if has_manifest and not multi:
        multi = True
    return inplace, multi


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
        # 与 aicheckword 本机一致：只要有上传的 Base，就提供 manifest，供「按名映射 / 未匹配 leftover」等分支使用。
        # 旧逻辑仅在 multi_base_auto_route 时构建，导致 aiword 等客户端未开自动路由时上传 Base 完全被忽略，
        # 就地修改无真实基底路径 → 参考写入不全、Word 表格 patch 锚点错位。
        base_manifest: Optional[List[tuple[str, str]]] = None
        bmp = spec.get("base_name_to_path") or {}
        if bmp:
            base_manifest = [(pth, nm) for nm, pth in sorted(bmp.items())]
        eff_inplace, eff_multi_route = _integration_effective_draft_flags(
            payload,
            base_paths_map=base_paths_map or {},
            base_manifest=base_manifest,
        )
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
            inplace_patch=eff_inplace,
            save_as_case=payload.save_as_case,
            progress_cb=_progress,
            base_case_limit_chunks=payload.base_case_limit_chunks,
            base_files_manifest=base_manifest,
            multi_base_auto_route=eff_multi_route,
            draft_strategy=payload.draft_strategy,
            author_role=payload.author_role,
            author_role_map=payload.author_role_map,
            audit_remediation_by_target=payload.audit_remediation_by_target,
            skip_case_template_text=payload.skip_case_template_text,
            skip_input_vector_training=payload.skip_input_vector_training,
            input_vector_on_duplicate=payload.input_vector_on_duplicate,
            client_llm=client_llm,
            user_prompt_append=payload.user_prompt_append,
        )
        _progress("导出产物…", 0.95)
        out_log, out_by_target, summaries = export_draft_job_files(
            res=res,
            existing_base_files=base_paths_map or None,
            inplace_patch=eff_inplace,
            document_language=payload.document_language,
            docx_track_changes=payload.docx_track_changes,
        )
        zip_path = Path(spec["job_dir"]) / "artifacts.zip"
        summary_path = Path(spec["job_dir"]) / "draft_artifacts_summary.json"
        try:
            _uplen = len((payload.user_prompt_append or "").strip())
            _skip_hist = _patch_skip_reason_histogram(summaries or [])
            _docx_unchanged = _summaries_indicate_no_docx_change(summaries or [])
            _warn = ""
            if _docx_unchanged:
                _top = sorted(_skip_hist.items(), key=lambda x: (-x[1], x[0]))[:5]
                _top_s = "；".join(f"{k}（{v}）" for k, v in _top) if _top else "见各 *.patch.report.json"
                _warn = (
                    "模型已输出 PATCH_JSON，但全部 operation 未写入 Word（applied=0）。"
                    "常见原因：anchor 与基底 docx 单元格原文不一致（勿用案例库模板措辞作锚点）。"
                    f"跳过原因统计：{_top_s}"
                )
            summary_obj: Dict[str, Any] = {
                "ok": True,
                "job_id": job_id,
                "effective_provider": (eff_provider or "").strip() or None,
                "inplace_patch": bool(eff_inplace),
                "inplace_patch_requested": bool(payload.inplace_patch),
                "multi_base_auto_route": bool(eff_multi_route),
                "multi_base_auto_route_requested": bool(payload.multi_base_auto_route),
                "docx_track_changes": bool(payload.docx_track_changes),
                "skip_input_vector_training": bool(payload.skip_input_vector_training),
                "input_vector_on_duplicate": str(payload.input_vector_on_duplicate or "skip"),
                "base_manifest_count": len(base_manifest or []),
                "base_bound_targets": list((base_paths_map or {}).keys())[:32],
                "generated_file_names": list((res.generated_files or {}).keys()),
                "per_file_patch_summaries": summaries or [],
                "patch_skip_reason_histogram": _skip_hist,
                "docx_unchanged": bool(_docx_unchanged),
                "warning": _warn or None,
                "user_prompt_append_chars": _uplen,
                "note": (
                    "就地修改且模型输出可解析的 PATCH_JSON 时，ZIP 内应有 *.patch.json 与 *.patch.report.json（修改日志）。"
                    "若未开启就地修改、或模型未输出 patch、或 patch 为空，则通常仅有主输出文件；此时以本 JSON 的 per_file_patch_summaries 为准。"
                    "skipped>0 且 applied=0 表示通常已调用 LLM，但锚点未命中基底 Word/表格。"
                ),
            }
            summary_path.write_text(
                json.dumps(summary_obj, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            summary_path = None

        paths_for_zip: List[str] = list(out_log)
        if summary_path and summary_path.is_file():
            paths_for_zip.append(str(summary_path))

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            seen_resolved: set[str] = set()
            for i, p in enumerate(paths_for_zip):
                pp = Path(p)
                if not pp.is_file():
                    continue
                key = str(pp.resolve())
                if key in seen_resolved:
                    continue
                seen_resolved.add(key)
                arc = f"{i:03d}_{pp.name}"
                zf.write(pp, arcname=arc)
        mi = get_current_model_info()
        _eff_mi = (eff_provider or "").strip() or mi
        if (eff_provider or "").strip():
            _eff_mi = f"{mi} | job_provider={eff_provider}"
        add_operation_log(
            op_type="draft_generate",
            collection=payload.collection,
            file_name="draft_outputs",
            source="draft_integration_api",
            extra={
                "batch_id": job_id,
                "job_id": job_id,
                "effective_provider": (eff_provider or "").strip() or None,
                "docx_unchanged": bool(_summaries_indicate_no_docx_change(summaries or [])),
                "patch_skip_reason_histogram": _patch_skip_reason_histogram(summaries or []),
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
                "user_prompt_append_chars": len((payload.user_prompt_append or "").strip()),
                "out_files": paths_for_zip,
                "out_files_by_target": out_by_target,
                "generated_file_names": list((out_by_target or {}).keys()),
                "per_file_patch_summaries": summaries,
            },
            model_info=_eff_mi,
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
                "docx_unchanged": bool(_summaries_indicate_no_docx_change(summaries or [])),
                "aiword_user_id": (payload.aiword_user_id or "").strip() or None,
                "aiword_task_id": (payload.aiword_task_id or "").strip() or None,
            },
            model_info=mi,
        )
        invalidate_operation_logs_cache()
        _docx_unchanged_final = _summaries_indicate_no_docx_change(summaries or [])
        _skip_hist_final = _patch_skip_reason_histogram(summaries or [])
        _finish_status = "succeeded"
        _finish_msg = "完成"
        if _docx_unchanged_final:
            _finish_status = "failed"
            _top = sorted(_skip_hist_final.items(), key=lambda x: (-x[1], x[0]))[:3]
            _ts = "；".join(f"{k}({v})" for k, v in _top) if _top else "见 patch.report"
            _finish_msg = (
                f"失败：patch 未写入 Word，文档与基底相同；已跳过 {sum(_skip_hist_final.values())} 条。"
                f" 主因：{_ts}"
            )
        _update_job(
            job_id,
            status=_finish_status,
            progress=1.0,
            message=_finish_msg,
            error=(_finish_msg if _finish_status == "failed" else ""),
            result={
                "project_id": res.project_id,
                "project_case_id": res.project_case_id,
                "generated_file_names": list((res.generated_files or {}).keys()),
                "zip_path": str(zip_path),
                "out_files_by_target": out_by_target,
                "docx_unchanged": bool(_docx_unchanged_final),
                "patch_skip_reason_histogram": _skip_hist_final,
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


@router.post("/check-input-vector-duplicates")
def check_input_vector_duplicates(body: CheckInputVectorDuplicatesBody):
    """按项目 ID 检测参考/输入文件名是否已在 project_knowledge_docs 中（用于集成端提交前确认）。"""
    from src.core.draft_input_vectorization import find_duplicate_input_file_names

    dups = find_duplicate_input_file_names(int(body.project_id), list(body.file_names or []))
    return {
        "ok": True,
        "project_id": int(body.project_id),
        "duplicates": dups,
        "has_duplicates": bool(dups),
    }


@router.get("/meta")
def draft_meta(
    request: Request,
    collection: str = Query("regulations"),
    base_case_id: Optional[int] = Query(None, description="若提供则返回该案例下模板文件名列表"),
):
    try:
        from src.api.server import _resolve_request_collection
        collection = _resolve_request_collection(request, collection)
        bcid: Optional[int] = None
        if base_case_id is not None and int(base_case_id) > 0:
            bcid = int(base_case_id)
        from src.core.draft_integration_ui_meta import build_draft_page_bootstrap

        bootstrap = build_draft_page_bootstrap(collection, base_case_id=bcid)
        out: Dict[str, Any] = {
            "collection": collection,
            "projects": bootstrap.get("projectsRaw") or list_projects(collection) or [],
            "cases": bootstrap.get("casesRaw") or list_project_cases(collection) or [],
            "bootstrap": {
                k: v
                for k, v in bootstrap.items()
                if k not in ("projectsRaw", "casesRaw")
            },
        }
        if bcid is not None:
            out["template_file_names"] = bootstrap.get("templateFileNames") or []
        return {"ok": True, "data": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/page-bootstrap")
def draft_page_bootstrap(
    request: Request,
    collection: str = Query("regulations"),
    base_case_id: Optional[int] = Query(None),
    templates: Optional[List[str]] = Query(
        None, description="用于 author_role 推断的模板文件名（可多传）"
    ),
):
    """初稿页完整 UI 元数据（aiword 薄代理透传，勿在 BFF 重复拼 author_role 等）。"""
    try:
        from src.api.server import _resolve_request_collection
        from src.core.draft_integration_ui_meta import build_draft_page_bootstrap

        collection = _resolve_request_collection(request, collection)
        bcid: Optional[int] = None
        if base_case_id is not None and int(base_case_id) > 0:
            bcid = int(base_case_id)
        tpl = [str(x).strip() for x in (templates or []) if str(x).strip()]
        data = build_draft_page_bootstrap(
            collection,
            base_case_id=bcid,
            template_file_names=tpl or None,
        )
        return {
            "ok": True,
            "data": {k: v for k, v in data.items() if k not in ("projectsRaw", "casesRaw")},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/suggest-author-role")
def draft_suggest_author_role(
    registration_type: str = Query(""),
    project_form: str = Query(""),
    templates: Optional[List[str]] = Query(None),
):
    from src.core.draft_integration_ui_meta import infer_draft_author_role_key

    names = [str(x).strip() for x in (templates or []) if str(x).strip()]
    key = infer_draft_author_role_key(
        names,
        registration_type=registration_type,
        project_form=project_form,
    )
    return {"ok": True, "authorRole": key}


@router.get("/projects/{project_id}/draft-defaults")
def draft_project_defaults(project_id: int):
    from src.core.draft_integration_ui_meta import project_draft_defaults

    try:
        pid = int(project_id)
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail="project_id 无效") from e
    if pid <= 0:
        raise HTTPException(status_code=400, detail="project_id 无效")
    row = project_draft_defaults(pid)
    if not row.get("project_id"):
        raise HTTPException(status_code=404, detail="项目不存在")
    return {"ok": True, "data": row}


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

    try:
        from src.api.server import _resolve_request_collection
        body.collection = _resolve_request_collection(request, body.collection or "regulations")
    except Exception:
        pass

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

    merged_bt = _integration_merge_base_files_by_target(body, base_name_to_path)
    base_paths_map: Dict[str, str] = {}
    for target, upload_name in merged_bt.items():
        un = (upload_name or "").strip()
        tg = (target or "").strip()
        if not un or not tg:
            continue
        pth = base_name_to_path.get(un)
        if not pth:
            shutil.rmtree(root, ignore_errors=True)
            raise HTTPException(
                status_code=400,
                detail=f"Base 绑定引用的上传名未找到：{un!r}（目标 {tg!r}）。请检查文件名与 multipart 一致。",
            )
        base_paths_map[tg] = pth

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
