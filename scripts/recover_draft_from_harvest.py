#!/usr/bin/env python3
"""一次性：从 cursor harvest 文件恢复初稿产物，不调 Cursor API。"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import settings
from src.core.document_draft_generator import _parse_patch_and_updated_text
from src.core.draft_export import export_docx_inplace_patch


def _find_upstream_job_for_aiword_task(aiword_task_id: str) -> str | None:
    try:
        import pymysql
    except ImportError:
        return None
    try:
        conn = pymysql.connect(
            host=settings.mysql_host,
            port=int(settings.mysql_port),
            user=settings.mysql_user,
            password=settings.mysql_password,
            database=settings.mysql_database,
            charset=settings.mysql_charset,
            connect_timeout=8,
        )
    except Exception:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT extra_json FROM operation_logs
                WHERE extra_json LIKE %s
                ORDER BY id DESC LIMIT 20
                """,
                (f"%{aiword_task_id}%",),
            )
            for (extra_json,) in cur.fetchall():
                try:
                    obj = json.loads(extra_json or "{}")
                except Exception:
                    continue
                jid = str(obj.get("job_id") or obj.get("draft_job_id") or "").strip()
                if jid:
                    return jid
    finally:
        conn.close()
    return None


def recover(
    *,
    harvest_path: Path,
    base_docx: Path,
    upstream_job_id: str,
    display_name: str,
    aiword_task_id: str = "",
    document_language: str = "en",
    docx_track_changes: bool = False,
) -> dict:
    raw = harvest_path.read_text(encoding="utf-8")
    patch_json, updated_text = _parse_patch_and_updated_text(raw)
    if not patch_json.strip():
        raise SystemExit("harvest 中未解析到 PATCH_JSON")

    job_dir = settings.uploads_path / "draft_api_jobs" / upstream_job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    drafts_dir = settings.uploads_path / "draft_outputs"
    drafts_dir.mkdir(parents=True, exist_ok=True)

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"recovered_{tag}_{base_docx.name}"
    out_path = drafts_dir / out_name

    meta = {
        "recovered_from_harvest": str(harvest_path.name),
        "aiword_task_id": aiword_task_id,
        "upstream_job_id": upstream_job_id,
        "generated_by": "aicheckword draft recovery (no cursor api)",
        "change_summary": "Recovered from Cursor harvest after local TIMEOUT",
    }
    saved, patch_report = export_docx_inplace_patch(
        base_file_path=str(base_docx),
        out_path=str(out_path),
        patch_json=patch_json,
        meta=meta,
        track_changes=bool(docx_track_changes),
    )

    patch_path = Path(saved).with_suffix(Path(saved).suffix + ".patch.json")
    patch_path.write_text(patch_json, encoding="utf-8")
    rep_path = Path(saved).with_suffix(Path(saved).suffix + ".patch.report.json")
    rep_path.write_text(json.dumps(patch_report, ensure_ascii=False, indent=2), encoding="utf-8")

    if updated_text.strip():
        sidecar = Path(saved).with_suffix(Path(saved).suffix + ".model-output.txt")
        sidecar.write_text(updated_text, encoding="utf-8")

    applied = len((patch_report or {}).get("applied") or []) if isinstance(patch_report, dict) else 0
    errors = len((patch_report or {}).get("errors") or []) if isinstance(patch_report, dict) else 0
    skipped = len((patch_report or {}).get("skipped") or []) if isinstance(patch_report, dict) else 0

    summary = {
        "ok": True,
        "recovered": True,
        "job_id": upstream_job_id,
        "aiword_task_id": aiword_task_id,
        "file_name": display_name,
        "out_file": str(saved),
        "patch_json_path": str(patch_path),
        "patch_report_path": str(rep_path),
        "patch_counts": {"applied": applied, "errors": errors, "skipped": skipped},
        "docx_unchanged": applied == 0 and errors == 0,
    }

    summary_path = job_dir / "draft_artifacts_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    zip_path = job_dir / "artifacts.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in (saved, patch_path, rep_path):
            zf.write(p, arcname=Path(p).name)
        if updated_text.strip():
            sc = Path(saved).with_suffix(Path(saved).suffix + ".model-output.txt")
            if sc.is_file():
                zf.write(sc, arcname=sc.name)

    summary["artifacts_zip"] = str(zip_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _update_aiword_job(aiword_task_id: str, zip_path: Path, upstream_job_id: str, summary: dict) -> bool:
    try:
        import pymysql
    except ImportError:
        return False
    try:
        conn = pymysql.connect(
            host=settings.mysql_host,
            port=int(settings.mysql_port),
            user=settings.mysql_user,
            password=settings.mysql_password,
            database="aiword",
            charset=settings.mysql_charset,
            connect_timeout=8,
        )
    except Exception:
        return False
    try:
        applied = summary.get("patch_counts", {}).get("applied", 0)
        err = summary.get("patch_counts", {}).get("errors", 0)
        if summary.get("docx_unchanged"):
            status = "failed"
            msg = "PATCH 已生成但未写入文档（applied=0）"
        elif err:
            status = "failed"
            msg = f"恢复完成但有 {err} 条 patch 错误"
        else:
            status = "succeeded"
            msg = f"已从 harvest 恢复（applied={applied}）"

        out_zip = Path("outputs/draft_zips") / f"{aiword_task_id}.zip"
        out_zip.parent.mkdir(parents=True, exist_ok=True)
        out_zip.write_bytes(zip_path.read_bytes())

        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE draft_generation_jobs
                SET status=%s, progress=1.0, message=%s, error_summary=%s,
                    upstream_job_id=%s, local_zip_path=%s, updated_at=NOW()
                WHERE id=%s
                """,
                (
                    status,
                    msg[:4000],
                    "" if status == "succeeded" else msg[:4000],
                    upstream_job_id,
                    str(out_zip.resolve()),
                    aiword_task_id,
                ),
            )
        conn.commit()
        return True
    except Exception as e:
        print("aiword update failed:", e)
        return False
    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest", required=True, help="cursor harvest txt 路径")
    ap.add_argument("--base", required=True, help="基底 docx 路径")
    ap.add_argument("--upstream-job-id", default="", help="aicheckword draft job id")
    ap.add_argument("--aiword-task-id", default="70c8a5cb-1575-4288-924c-50970af1a4b8")
    ap.add_argument("--display-name", default="BCMAS-UDN-001 Instructions for Use")
    args = ap.parse_args()

    harvest_path = Path(args.harvest).resolve()
    base_docx = Path(args.base).resolve()
    if not harvest_path.is_file():
        raise SystemExit(f"harvest 不存在: {harvest_path}")
    if not base_docx.is_file():
        raise SystemExit(f"基底不存在: {base_docx}")

    upstream_job_id = (args.upstream_job_id or "").strip()
    if not upstream_job_id and args.aiword_task_id:
        upstream_job_id = _find_upstream_job_for_aiword_task(args.aiword_task_id) or ""
    if not upstream_job_id:
        upstream_job_id = uuid.uuid4().hex[:16]

    summary = recover(
        harvest_path=harvest_path,
        base_docx=base_docx,
        upstream_job_id=upstream_job_id,
        display_name=args.display_name,
        aiword_task_id=args.aiword_task_id,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    zip_path = Path(summary["artifacts_zip"])
    if args.aiword_task_id:
        ok = _update_aiword_job(args.aiword_task_id, zip_path, upstream_job_id, summary)
        print("aiword_job_updated:", ok)


if __name__ == "__main__":
    main()
