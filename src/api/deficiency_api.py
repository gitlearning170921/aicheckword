# -*- coding: utf-8 -*-
"""发补记录 API（CRUD / 资产 / 训练）。"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from langchain_core.documents import Document
from pydantic import BaseModel

from config import settings
from src.core.deficiency_store import (
    add_deficiency_asset,
    archive_deficiency_record,
    create_deficiency_record,
    get_deficiency_record,
    list_deficiency_assets,
    list_deficiency_records,
    update_deficiency_record,
)
from src.core.document_loader import load_and_split

router = APIRouter(prefix="/api/deficiency", tags=["deficiency"])


def _resolve_collection(req: Request, collection: str) -> str:
    from src.api.server import _resolve_request_collection

    return _resolve_request_collection(req, collection or "regulations")


def _get_agent(collection: str):
    from src.api.server import get_agent

    return get_agent(collection)


class DeficiencyCreateBody(BaseModel):
    collection: str = "regulations"
    linked_company_project_id: str = ""
    linked_project_id: Optional[int] = None
    registration_country: str = ""
    registration_category: str = ""
    opinion_text: str = ""
    priority: str = "medium"
    remediation_plan: str = ""
    issued_on: str = ""
    remediation_status: str = "open"
    completed_on: Optional[str] = None
    deficiency_type: str = "registration_review"
    deficiency_source: str = ""


class DeficiencyUpdateBody(BaseModel):
    collection: str = "regulations"
    linked_company_project_id: Optional[str] = None
    linked_project_id: Optional[int] = None
    registration_country: Optional[str] = None
    registration_category: Optional[str] = None
    opinion_text: Optional[str] = None
    priority: Optional[str] = None
    remediation_plan: Optional[str] = None
    issued_on: Optional[str] = None
    remediation_status: Optional[str] = None
    completed_on: Optional[str] = None
    deficiency_type: Optional[str] = None
    deficiency_source: Optional[str] = None
    status: Optional[str] = None


@router.post("/records")
def create_record(req: Request, body: DeficiencyCreateBody):
    collection = _resolve_collection(req, body.collection)
    try:
        rid = create_deficiency_record({**body.model_dump(), "collection": collection})
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    row = get_deficiency_record(rid)
    return {"ok": True, "data": row}


@router.get("/records")
def list_records(
    req: Request,
    collection: str = Query("regulations"),
    remediation_status: str = Query(""),
    deficiency_type: str = Query(""),
    registration_country: str = Query(""),
    registration_category: str = Query(""),
    limit: int = Query(200),
):
    collection = _resolve_collection(req, collection)
    rows = list_deficiency_records(
        collection,
        remediation_status=remediation_status,
        deficiency_type=deficiency_type,
        registration_country=registration_country,
        registration_category=registration_category,
        limit=limit,
    )
    return {"ok": True, "data": rows, "collection": collection}


@router.get("/records/{record_id}")
def get_record(req: Request, record_id: int, collection: str = Query("regulations")):
    collection = _resolve_collection(req, collection)
    row = get_deficiency_record(int(record_id))
    if not row or str(row.get("collection") or "") != collection:
        raise HTTPException(status_code=404, detail="记录不存在")
    assets = list_deficiency_assets(int(record_id))
    return {"ok": True, "data": {**row, "assets": assets}}


@router.patch("/records/{record_id}")
def patch_record(req: Request, record_id: int, body: DeficiencyUpdateBody):
    collection = _resolve_collection(req, body.collection)
    row = get_deficiency_record(int(record_id))
    if not row or str(row.get("collection") or "") != collection:
        raise HTTPException(status_code=404, detail="记录不存在")
    payload = {k: v for k, v in body.model_dump().items() if k != "collection" and v is not None}
    try:
        update_deficiency_record(int(record_id), payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"ok": True, "data": get_deficiency_record(int(record_id))}


@router.delete("/records/{record_id}")
def delete_record(
    req: Request,
    record_id: int,
    collection: str = Query("regulations"),
):
    collection = _resolve_collection(req, collection)
    row = get_deficiency_record(int(record_id))
    if not row or str(row.get("collection") or "") != collection:
        raise HTTPException(status_code=404, detail="记录不存在")
    archive_deficiency_record(int(record_id))
    return {"ok": True}


@router.post("/records/{record_id}/assets")
async def upload_assets(
    req: Request,
    record_id: int,
    files: List[UploadFile] = File(...),
    collection: str = Form("regulations"),
    role: str = Form("before_doc"),
):
    collection = _resolve_collection(req, collection)
    row = get_deficiency_record(int(record_id))
    if not row or str(row.get("collection") or "") != collection:
        raise HTTPException(status_code=404, detail="记录不存在")
    role = (role or "before_doc").strip()
    root = Path(settings.uploads_path) / "deficiency" / collection / str(record_id)
    root.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files or []:
        display_name = Path(str(f.filename or "upload.bin")).name
        raw = await f.read()
        if not raw:
            continue
        dest = root / f"{uuid.uuid4().hex[:8]}_{display_name}"
        dest.write_bytes(raw)
        text_excerpt = ""
        try:
            chunks = load_and_split(str(dest))
            text_excerpt = "\n".join((c.page_content or "")[:2000] for c in (chunks or [])[:3])[:8000]
        except Exception:
            text_excerpt = ""
        aid = add_deficiency_asset(
            int(record_id),
            role=role,
            display_name=display_name,
            storage_path=str(dest),
            text_excerpt=text_excerpt,
        )
        saved.append({"id": aid, "display_name": display_name, "role": role})
    return {"ok": True, "data": {"assets": saved, "record": get_deficiency_record(int(record_id))}}


def _build_train_documents(row: Dict[str, Any], assets: List[Dict[str, Any]]) -> List[Document]:
    meta_base = {
        "category": "deficiency",
        "record_id": int(row["id"]),
        "issued_on": str(row.get("issued_on") or "")[:10],
        "registration_country": str(row.get("registration_country") or ""),
        "registration_category": str(row.get("registration_category") or ""),
        "deficiency_type": str(row.get("deficiency_type") or ""),
        "priority": str(row.get("priority") or ""),
        "remediation_status": str(row.get("remediation_status") or ""),
    }
    parts = [
        f"发补类型：{row.get('deficiency_type')}",
        f"发补来源：{row.get('deficiency_source') or '—'}",
        f"发补日期：{str(row.get('issued_on') or '')[:10]}",
        f"整改状态：{row.get('remediation_status')}",
        f"注册国家：{row.get('registration_country')}",
        f"注册类别：{row.get('registration_category')}",
        f"发补意见：\n{row.get('opinion_text') or ''}",
        f"整改方案：\n{row.get('remediation_plan') or ''}",
    ]
    docs = [
        Document(
            page_content="\n".join(parts),
            metadata={**meta_base, "source_file": f"deficiency_{row['id']}_core.txt", "role": "core"},
        )
    ]
    for a in assets or []:
        excerpt = str(a.get("text_excerpt") or "").strip()
        path = str(a.get("storage_path") or "").strip()
        display = str(a.get("display_name") or "asset").strip()
        role = str(a.get("role") or "")
        content = excerpt
        if path and Path(path).is_file():
            try:
                chunks = load_and_split(path)
                content = "\n".join((c.page_content or "") for c in (chunks or [])[:20])[:20000]
            except Exception:
                content = excerpt
        if not content.strip():
            continue
        docs.append(
            Document(
                page_content=f"[{role}] {display}\n{content}",
                metadata={**meta_base, "source_file": display, "role": role},
            )
        )
    return docs


@router.post("/records/{record_id}/train")
def train_record(req: Request, record_id: int, collection: str = Form("regulations")):
    collection = _resolve_collection(req, collection)
    row = get_deficiency_record(int(record_id))
    if not row or str(row.get("collection") or "") != collection:
        raise HTTPException(status_code=404, detail="记录不存在")
    assets = list_deficiency_assets(int(record_id))
    docs = _build_train_documents(row, assets)
    if not docs:
        raise HTTPException(status_code=400, detail="无可训练内容")
    agent = _get_agent(collection)
    file_key = f"deficiency_{record_id}"
    try:
        agent.kb.delete_documents_by_file_name(file_key)
    except Exception:
        pass
    # 也清理按资产名写入的旧块（尽力）
    for a in assets:
        try:
            agent.kb.delete_documents_by_file_name(str(a.get("display_name") or ""))
        except Exception:
            pass
    n = 0
    try:
        for doc in docs:
            fn = str((doc.metadata or {}).get("source_file") or file_key)
            n += int(
                agent.kb.add_documents(
                    [doc],
                    file_name=fn if fn != file_key else file_key,
                    category="deficiency",
                )
                or 0
            )
        update_deficiency_record(int(record_id), {"train_status": "trained"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"训练失败：{e}") from e
    return {
        "ok": True,
        "data": {
            "record_id": int(record_id),
            "chunks_added": n,
            "train_status": "trained",
            "record": get_deficiency_record(int(record_id)),
        },
    }
