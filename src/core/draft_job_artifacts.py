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


def export_draft_job_files(
    *,
    res: GeneratedCaseDocs,
    existing_base_files: Optional[Dict[str, str]],
    inplace_patch: bool,
    document_language: str,
    docx_track_changes: bool,
    drafts_subdir: str = "draft_outputs",
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
        patch_report_obj = None

        base_path = _base_map.get(fn) if isinstance(_base_map, dict) else None
        if not base_path and isinstance(existing_base_files, dict):
            base_path = existing_base_files.get(fn)
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
                patch_report_obj = patch_report
                patch_path = Path(saved).with_suffix(Path(saved).suffix + ".patch.json")
                patch_path.write_text(patch_json, encoding="utf-8")
                rep_path = Path(saved).with_suffix(Path(saved).suffix + ".patch.report.json")
                rep_path.write_text(json.dumps(patch_report, ensure_ascii=False, indent=2), encoding="utf-8")
                patch_path_str = str(patch_path)
                rep_path_str = str(rep_path)
                out_files_for_log.extend([str(patch_path), str(rep_path)])
                downloads.extend([saved, str(patch_path), str(rep_path)])
                try:
                    per_file_patch_summaries.append(
                        {
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
                patch_report_obj = patch_report
                patch_path = Path(saved).with_suffix(Path(saved).suffix + ".patch.json")
                patch_path.write_text(patch_json, encoding="utf-8")
                rep_path = Path(saved).with_suffix(Path(saved).suffix + ".patch.report.json")
                rep_path.write_text(json.dumps(patch_report, ensure_ascii=False, indent=2), encoding="utf-8")
                patch_path_str = str(patch_path)
                rep_path_str = str(rep_path)
                out_files_for_log.extend([str(patch_path), str(rep_path)])
                downloads.extend([saved, str(patch_path), str(rep_path)])
                try:
                    per_file_patch_summaries.append(
                        {
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
                    )
                except Exception:
                    pass
            else:
                # 无 PATCH_JSON 时勿把 UPDATED_TEXT 整段追加进 docx（会破坏表格/版式）
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
