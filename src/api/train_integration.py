# -*- coding: utf-8 -*-
"""知识库异步训练任务（供 aiword 轮询进度，避免同步长请求断连）。

- POST /api/integration/train/jobs ：multipart，字段 collection/category/overwrite_mode + files
- GET  /api/integration/train/jobs/{job_id} ：状态与进度
"""
from __future__ import annotations

import shutil
import tempfile
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from src.core.agent import ReviewAgent
from src.core.db import get_existing_file_names

router = APIRouter(prefix="/api/integration/train", tags=["train-integration"])

_jobs_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="train_job")


def _resolve_collection(req: Request, collection: str) -> str:
    from src.api.server import _resolve_request_collection

    return _resolve_request_collection(req, collection or "regulations")


def _get_agent(collection: str) -> ReviewAgent:
    from src.api.server import get_agent

    return get_agent(collection)


def _update_job(job_id: str, **kwargs: Any) -> None:
    with _jobs_lock:
        j = _jobs.get(job_id)
        if not j:
            return
        j.update(kwargs)


def _run_train_job(job_id: str) -> None:
    with _jobs_lock:
        spec = dict(_jobs.get(job_id) or {})
    if not spec:
        return
    collection = str(spec.get("collection") or "regulations")
    category = str(spec.get("category") or "regulation")
    overwrite_mode = str(spec.get("overwrite_mode") or "overwrite").strip().lower()
    if overwrite_mode not in ("overwrite", "skip"):
        overwrite_mode = "overwrite"
    work_dir = Path(str(spec.get("work_dir") or ""))
    file_entries: List[Dict[str, str]] = list(spec.get("files") or [])
    total = max(len(file_entries), 1)
    details: List[Dict[str, Any]] = []
    total_chunks = 0

    try:
        _update_job(job_id, status="running", progress=0.02, message="开始训练…")
        agent = _get_agent(collection)
        existing = set(get_existing_file_names(collection, category=category) or [])

        for idx, entry in enumerate(file_entries):
            display_name = str(entry.get("display_name") or "").strip() or "upload.bin"
            path = Path(str(entry.get("path") or ""))
            frac_base = idx / total
            _update_job(
                job_id,
                progress=min(0.95, frac_base + 0.01),
                message=f"处理文件 {idx + 1}/{len(file_entries)}：{display_name}",
                current_file=display_name,
                files_done=idx,
                files_total=len(file_entries),
            )
            if not path.is_file():
                details.append(
                    {
                        "status": "error",
                        "original_filename": display_name,
                        "message": "临时文件不存在",
                    }
                )
                continue
            if display_name in existing and overwrite_mode == "skip":
                details.append(
                    {
                        "status": "skipped",
                        "original_filename": display_name,
                        "message": "已存在，按 skip 跳过",
                        "chunks_added": 0,
                    }
                )
                continue
            try:
                if display_name in existing and overwrite_mode == "overwrite":
                    try:
                        agent.kb.delete_documents_by_file_name(display_name)
                    except Exception:
                        pass
                    existing.discard(display_name)

                def _progress(done: int, tot: int) -> None:
                    sub = (done / tot) if tot else 1.0
                    _update_job(
                        job_id,
                        progress=min(0.95, frac_base + sub / total),
                        message=f"{display_name}：向量化 {done}/{tot}",
                        chunks_done=done,
                        chunks_total=tot,
                    )

                from src.core.document_loader import load_and_split

                chunks = load_and_split(str(path))
                n = agent.kb.add_documents_with_progress(
                    chunks,
                    batch_size=12,
                    callback=_progress,
                    file_name=display_name,
                    category=category,
                )
                total_chunks += int(n or 0)
                existing.add(display_name)
                details.append(
                    {
                        "status": "success",
                        "original_filename": display_name,
                        "chunks_added": int(n or 0),
                    }
                )
            except Exception as exc:
                details.append(
                    {
                        "status": "error",
                        "original_filename": display_name,
                        "message": str(exc) or type(exc).__name__,
                    }
                )

        ok_n = sum(1 for d in details if d.get("status") == "success")
        err_n = sum(1 for d in details if d.get("status") == "error")
        skip_n = sum(1 for d in details if d.get("status") == "skipped")
        _update_job(
            job_id,
            status="succeeded",
            progress=1.0,
            message=f"完成：成功 {ok_n}，跳过 {skip_n}，失败 {err_n}，新增块 {total_chunks}",
            files_done=len(file_entries),
            files_total=len(file_entries),
            result={
                "status": "success",
                "files_processed": len(details),
                "total_chunks_added": total_chunks,
                "success_count": ok_n,
                "error_count": err_n,
                "skipped_count": skip_n,
                "details": details,
            },
            error="",
        )
    except Exception as exc:
        _update_job(
            job_id,
            status="failed",
            progress=1.0,
            message="训练失败",
            error=str(exc) or type(exc).__name__,
            result={"details": details, "traceback": traceback.format_exc()[-2000:]},
        )
    finally:
        try:
            if work_dir.is_dir():
                shutil.rmtree(str(work_dir), ignore_errors=True)
        except Exception:
            pass


@router.post("/jobs")
async def create_train_job(
    req: Request,
    files: List[UploadFile] = File(...),
    collection: str = Form("regulations"),
    category: str = Form("regulation"),
    overwrite_mode: str = Form("overwrite"),
):
    collection = _resolve_collection(req, collection)
    category = (category or "regulation").strip() or "regulation"
    mode = (overwrite_mode or "overwrite").strip().lower()
    if mode not in ("overwrite", "skip"):
        mode = "overwrite"

    job_id = uuid.uuid4().hex[:16]
    work_dir = Path(tempfile.mkdtemp(prefix=f"train_job_{job_id}_"))
    file_entries: List[Dict[str, str]] = []
    try:
        for f in files or []:
            raw_name = str(f.filename or "upload.bin").strip() or "upload.bin"
            display_name = Path(raw_name).name
            suffix = Path(display_name).suffix or ".bin"
            dest = work_dir / f"{uuid.uuid4().hex[:8]}_{display_name}"
            content = await f.read()
            if not content:
                continue
            dest.write_bytes(content)
            file_entries.append({"path": str(dest), "display_name": display_name})
    except Exception:
        shutil.rmtree(str(work_dir), ignore_errors=True)
        raise

    if not file_entries:
        shutil.rmtree(str(work_dir), ignore_errors=True)
        raise HTTPException(status_code=400, detail="无有效训练文件")

    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "progress": 0.0,
            "message": "已排队",
            "error": "",
            "collection": collection,
            "category": category,
            "overwrite_mode": mode,
            "work_dir": str(work_dir),
            "files": file_entries,
            "files_done": 0,
            "files_total": len(file_entries),
            "current_file": "",
            "chunks_done": 0,
            "chunks_total": 0,
            "result": None,
        }
    _executor.submit(_run_train_job, job_id)
    return {
        "ok": True,
        "job_id": job_id,
        "status": "queued",
        "collection": collection,
        "category": category,
        "files_total": len(file_entries),
    }


@router.get("/jobs/{job_id}")
def train_job_status(job_id: str):
    with _jobs_lock:
        j = _jobs.get(job_id)
        if not j:
            raise HTTPException(status_code=404, detail="job 不存在或已过期")
        view = {
            "ok": True,
            "job_id": job_id,
            "status": j.get("status"),
            "progress": j.get("progress"),
            "message": j.get("message") or "",
            "error": j.get("error") or "",
            "collection": j.get("collection"),
            "category": j.get("category"),
            "files_done": j.get("files_done"),
            "files_total": j.get("files_total"),
            "current_file": j.get("current_file") or "",
            "chunks_done": j.get("chunks_done"),
            "chunks_total": j.get("chunks_total"),
            "result": j.get("result"),
        }
    return view
