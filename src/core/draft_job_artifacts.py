"""
将 DocumentDraftGenerator 产物落盘（与 Streamlit 初稿页逻辑对齐），供集成 API 打包下载。
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import settings

from .document_draft_generator import GeneratedCaseDocs
from .draft_export import (
    export_docx_inplace_patch,
    export_like_base,
    export_xlsx_inplace_patch,
    sniff_word_processing_suffix,
)


def _safe_out_path(*, base_path: str, out_path: Path) -> Path:
    try:
        bp = Path(base_path).resolve()
        op = out_path.resolve()
        if str(bp).lower() == str(op).lower():
            stem = out_path.stem
            suf = out_path.suffix
            tag = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
            return out_path.with_name(f"{stem}_{tag}{suf}")
    except Exception:
        pass
    return out_path


def _write_audit_coverage_sidecars(
    *,
    saved: str,
    file_name: str,
    patch_report: Any,
    out_files_for_log: List[str],
    downloads: List[str],
) -> Tuple[Optional[Dict[str, Any]], str, str]:
    """写入 audit_point_coverage.json 与 audit_modify.log.md，返回 (cov_obj, cov_path, log_path)。"""
    cov_obj = (
        (patch_report or {}).get("audit_point_coverage")
        if isinstance(patch_report, dict)
        else None
    )
    cov_path_str = ""
    log_md_str = ""
    if not isinstance(cov_obj, dict) or cov_obj.get("points") is None:
        return cov_obj if isinstance(cov_obj, dict) else None, "", ""
    try:
        from src.core.audit_handoff import format_audit_point_coverage_markdown

        cov_path = Path(saved).with_suffix(Path(saved).suffix + ".audit_point_coverage.json")
        cov_path.write_text(json.dumps(cov_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        log_path = Path(saved).with_suffix(Path(saved).suffix + ".audit_modify.log.md")
        log_path.write_text(
            format_audit_point_coverage_markdown(cov_obj, file_name=file_name),
            encoding="utf-8",
        )
        cov_path_str = str(cov_path)
        log_md_str = str(log_path)
        out_files_for_log.extend([cov_path_str, log_md_str])
        downloads.extend([cov_path_str, log_md_str])
    except Exception:
        pass
    return cov_obj, cov_path_str, log_md_str


def _patch_summary_entry(
    *,
    fn: str,
    saved: str,
    base_path: str,
    base_suffix: str,
    patch_report: Any,
    patch_path_str: str,
    rep_path_str: str,
    cov_obj: Optional[Dict[str, Any]],
    cov_path_str: str,
    log_md_str: str,
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "file_name": fn,
        "out_file": str(saved),
        "base_file": str(base_path),
        "suffix": base_suffix,
        "patch_report_path": rep_path_str,
        "patch_json_path": patch_path_str,
        "patch_counts": {
            "applied": len((patch_report or {}).get("applied") or [])
            if isinstance(patch_report, dict)
            else None,
            "changes": len((patch_report or {}).get("changes") or [])
            if isinstance(patch_report, dict)
            else None,
            "skipped": len((patch_report or {}).get("skipped") or [])
            if isinstance(patch_report, dict)
            else None,
            "errors": len((patch_report or {}).get("errors") or [])
            if isinstance(patch_report, dict)
            else None,
        },
        "no_change": (
            isinstance(patch_report, dict)
            and len((patch_report.get("applied") or [])) == 0
            and len((patch_report.get("changes") or [])) == 0
        ),
    }
    if cov_path_str:
        entry["audit_point_coverage_path"] = cov_path_str
    if log_md_str:
        entry["audit_modify_log_path"] = log_md_str
    if isinstance(cov_obj, dict):
        entry["audit_point_coverage"] = cov_obj
    return entry


def export_draft_job_files(
    *,
    res: GeneratedCaseDocs,
    existing_base_files: Optional[Dict[str, str]],
    inplace_patch: bool,
    document_language: str,
    docx_track_changes: bool,
    drafts_subdir: str = "draft_outputs",
    audit_immediate_points_by_target: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    collection: str = "",
    base_case_id: Optional[int] = None,
) -> Tuple[List[str], Dict[str, str], List[dict]]:
    """
    返回 (out_files_for_log, out_files_by_target, per_file_patch_summaries)
    """
    out_files_for_log: List[str] = []
    out_files_by_target: Dict[str, str] = {}
    per_file_patch_summaries: List[dict] = []

    _base_map = getattr(res, "per_file_base_path", None) or {}
    out_prefix = str(res.project_case_id or res.project_id)
    _is_en_doc = str(document_language or "").strip().lower().startswith("en")

    drafts_dir = settings.uploads_path / drafts_subdir
    drafts_dir.mkdir(parents=True, exist_ok=True)

    for fn, txt in (res.generated_files or {}).items():
        downloads: List[str] = []
        patch_path_str = ""
        rep_path_str = ""

        base_path = _base_map.get(fn) if isinstance(_base_map, dict) else None
        if not base_path and isinstance(existing_base_files, dict):
            base_path = existing_base_files.get(fn)
        if not base_path and int(base_case_id or 0) > 0:
            try:
                from src.core.case_template_files import resolve_case_template_file_path

                base_path = resolve_case_template_file_path(
                    collection=collection,
                    case_id=int(base_case_id),
                    file_name=fn,
                )
            except Exception:
                base_path = None
        if base_path:
            base_suffix = sniff_word_processing_suffix(base_path).lower()
            out_name = fn
            if base_suffix:
                out_name = Path(out_name).stem + base_suffix
            out_path = drafts_dir / f"{out_prefix}_{out_name}"
            out_path = _safe_out_path(base_path=base_path, out_path=out_path)
            meta = {
                "project_id": res.project_id,
                "project_case_id": res.project_case_id,
                "base_file": Path(base_path).name,
                "change_summary": (
                    "Update based on base document (identify additions/refinements/deletions from references)"
                    if _is_en_doc and inplace_patch
                    else (
                        "Update based on base document"
                        if _is_en_doc
                        else (
                            "基于基础文件按规则补写/修订生成（对照参考识别新增·细化·删除）"
                            if inplace_patch
                            else "基于基础文件按规则补写/修订生成"
                        )
                    )
                ),
                "generated_by": ("aicheckword draft generator" if _is_en_doc else "aicheckword 文档初稿生成"),
            }
            _imm_pts: List[Dict[str, Any]] = []
            if isinstance(audit_immediate_points_by_target, dict):
                raw_pts = audit_immediate_points_by_target.get(fn)
                if isinstance(raw_pts, list):
                    _imm_pts = [x for x in raw_pts if isinstance(x, dict)]
            if _imm_pts:
                meta["immediate_audit_points"] = _imm_pts
            saved = None
            patch_json = (getattr(res, "generated_patches", {}) or {}).get(fn) if inplace_patch else None
            if inplace_patch and base_suffix == ".docx" and (patch_json or "").strip():
                saved, patch_report = export_docx_inplace_patch(
                    base_file_path=base_path,
                    out_path=str(out_path),
                    patch_json=patch_json,
                    meta=meta,
                    track_changes=bool(docx_track_changes),
                )
                patch_path = Path(saved).with_suffix(Path(saved).suffix + ".patch.json")
                patch_path.write_text(patch_json, encoding="utf-8")
                rep_path = Path(saved).with_suffix(Path(saved).suffix + ".patch.report.json")
                rep_path.write_text(json.dumps(patch_report, ensure_ascii=False, indent=2), encoding="utf-8")
                patch_path_str = str(patch_path)
                rep_path_str = str(rep_path)
                out_files_for_log.extend([str(patch_path), str(rep_path)])
                downloads.extend([saved, str(patch_path), str(rep_path)])
                cov_obj, cov_path_str, log_md_str = _write_audit_coverage_sidecars(
                    saved=str(saved),
                    file_name=fn,
                    patch_report=patch_report,
                    out_files_for_log=out_files_for_log,
                    downloads=downloads,
                )
                try:
                    per_file_patch_summaries.append(
                        _patch_summary_entry(
                            fn=fn,
                            saved=str(saved),
                            base_path=str(base_path),
                            base_suffix=base_suffix,
                            patch_report=patch_report,
                            patch_path_str=patch_path_str,
                            rep_path_str=rep_path_str,
                            cov_obj=cov_obj if isinstance(cov_obj, dict) else None,
                            cov_path_str=cov_path_str,
                            log_md_str=log_md_str,
                        )
                    )
                except Exception:
                    pass
            elif inplace_patch and base_suffix in (".xlsx", ".xls") and (patch_json or "").strip():
                saved, patch_report = export_xlsx_inplace_patch(
                    base_file_path=base_path,
                    out_path=str(out_path),
                    patch_json=patch_json,
                    meta=meta,
                )
                patch_path = Path(saved).with_suffix(Path(saved).suffix + ".patch.json")
                patch_path.write_text(patch_json, encoding="utf-8")
                rep_path = Path(saved).with_suffix(Path(saved).suffix + ".patch.report.json")
                rep_path.write_text(json.dumps(patch_report, ensure_ascii=False, indent=2), encoding="utf-8")
                patch_path_str = str(patch_path)
                rep_path_str = str(rep_path)
                out_files_for_log.extend([str(patch_path), str(rep_path)])
                downloads.extend([saved, str(patch_path), str(rep_path)])
                cov_obj, cov_path_str, log_md_str = _write_audit_coverage_sidecars(
                    saved=str(saved),
                    file_name=fn,
                    patch_report=patch_report,
                    out_files_for_log=out_files_for_log,
                    downloads=downloads,
                )
                try:
                    per_file_patch_summaries.append(
                        _patch_summary_entry(
                            fn=fn,
                            saved=str(saved),
                            base_path=str(base_path),
                            base_suffix=base_suffix,
                            patch_report=patch_report,
                            patch_path_str=patch_path_str,
                            rep_path_str=rep_path_str,
                            cov_obj=cov_obj if isinstance(cov_obj, dict) else None,
                            cov_path_str=cov_path_str,
                            log_md_str=log_md_str,
                        )
                    )
                except Exception:
                    pass
            else:
                saved = export_like_base(
                    base_file_path=base_path,
                    out_path=str(out_path),
                    title=fn,
                    content_text=txt or "",
                    meta=meta,
                    append_generated_content=not bool(inplace_patch),
                )
                if inplace_patch and (txt or "").strip():
                    sidecar = Path(saved).with_suffix(
                        Path(saved).suffix + ".model-output.txt"
                    )
                    try:
                        sidecar.write_text(txt or "", encoding="utf-8")
                        downloads.append(str(sidecar))
                        out_files_for_log.append(str(sidecar))
                    except Exception:
                        pass
                downloads.append(saved)

            out_files_for_log.append(str(saved))
            out_files_by_target[fn] = str(saved)
        else:
            txt_path = drafts_dir / f"{res.project_case_id}_{Path(str(fn)).stem}.draft.txt"
            txt_path.write_text(txt or "", encoding="utf-8")
            out_files_for_log.append(str(txt_path))
            out_files_by_target[fn] = str(txt_path)

    return out_files_for_log, out_files_by_target, per_file_patch_summaries
