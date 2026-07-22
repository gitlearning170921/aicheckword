# -*- coding: utf-8 -*-
"""发补记录持久化（MySQL）。"""
from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from src.core.db import _get_conn, init_db


PRIORITY_VALUES = ("high", "medium", "low")
STATUS_VALUES = ("open", "done")
TYPE_VALUES = ("registration_review", "type_testing")
TRAIN_STATUS_VALUES = ("not_trained", "trained", "stale")
ASSET_ROLES = ("before_doc", "after_doc", "opinion_file", "plan_file")


def ensure_deficiency_tables() -> None:
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS deficiency_records (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    collection VARCHAR(128) NOT NULL,
                    linked_company_project_id VARCHAR(64) DEFAULT NULL,
                    linked_project_id BIGINT DEFAULT NULL,
                    registration_country VARCHAR(128) NOT NULL DEFAULT '',
                    registration_category VARCHAR(128) NOT NULL DEFAULT '',
                    opinion_text MEDIUMTEXT,
                    priority VARCHAR(16) NOT NULL DEFAULT 'medium',
                    remediation_plan MEDIUMTEXT,
                    issued_on DATE NOT NULL,
                    remediation_status VARCHAR(16) NOT NULL DEFAULT 'open',
                    completed_on DATE DEFAULT NULL,
                    deficiency_type VARCHAR(64) NOT NULL DEFAULT 'registration_review',
                    deficiency_source VARCHAR(255) DEFAULT '',
                    train_status VARCHAR(32) NOT NULL DEFAULT 'not_trained',
                    status VARCHAR(16) NOT NULL DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_def_coll_country_cat (collection, registration_country, registration_category),
                    INDEX idx_def_issued (collection, issued_on),
                    INDEX idx_def_status (collection, status, remediation_status)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS deficiency_assets (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    record_id BIGINT NOT NULL,
                    role VARCHAR(32) NOT NULL,
                    display_name VARCHAR(512) NOT NULL,
                    storage_path VARCHAR(1024) DEFAULT '',
                    text_excerpt MEDIUMTEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_def_asset_record (record_id),
                    CONSTRAINT fk_def_asset_record FOREIGN KEY (record_id)
                        REFERENCES deficiency_records(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
        conn.commit()
    finally:
        conn.close()


def _parse_date(val: Any) -> Optional[date]:
    if val is None or val == "":
        return None
    if isinstance(val, date) and not isinstance(val, datetime):
        return val
    if isinstance(val, datetime):
        return val.date()
    s = str(val).strip()[:10]
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def _row_to_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    if not row:
        return {}
    out = dict(row)
    for k in ("issued_on", "completed_on", "created_at", "updated_at"):
        if out.get(k) is not None:
            out[k] = str(out[k])
    return out


def create_deficiency_record(data: Dict[str, Any]) -> int:
    ensure_deficiency_tables()
    issued = _parse_date(data.get("issued_on"))
    if not issued:
        raise ValueError("issued_on 必填且须为 YYYY-MM-DD")
    rem_status = str(data.get("remediation_status") or "open").strip() or "open"
    if rem_status not in STATUS_VALUES:
        rem_status = "open"
    completed = _parse_date(data.get("completed_on"))
    if rem_status == "done" and not completed:
        completed = date.today()
    if rem_status == "open":
        completed = None
    priority = str(data.get("priority") or "medium").strip() or "medium"
    if priority not in PRIORITY_VALUES:
        priority = "medium"
    dtype = str(data.get("deficiency_type") or "registration_review").strip() or "registration_review"
    if dtype not in TYPE_VALUES:
        dtype = "registration_review"
    country = str(data.get("registration_country") or "").strip()
    category = str(data.get("registration_category") or "").strip()
    if not country or not category:
        raise ValueError("registration_country 与 registration_category 必填")
    opinion = str(data.get("opinion_text") or "").strip()
    if not opinion:
        raise ValueError("opinion_text 必填")

    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO deficiency_records (
                    collection, linked_company_project_id, linked_project_id,
                    registration_country, registration_category,
                    opinion_text, priority, remediation_plan,
                    issued_on, remediation_status, completed_on,
                    deficiency_type, deficiency_source, train_status, status
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    str(data.get("collection") or "regulations").strip() or "regulations",
                    str(data.get("linked_company_project_id") or "").strip() or None,
                    int(data["linked_project_id"]) if data.get("linked_project_id") not in (None, "") else None,
                    country,
                    category,
                    opinion,
                    priority,
                    str(data.get("remediation_plan") or "").strip(),
                    issued,
                    rem_status,
                    completed,
                    dtype,
                    str(data.get("deficiency_source") or "").strip(),
                    "not_trained",
                    "active",
                ),
            )
            rid = int(cur.lastrowid)
        conn.commit()
        return rid
    finally:
        conn.close()


def update_deficiency_record(record_id: int, data: Dict[str, Any]) -> None:
    ensure_deficiency_tables()
    existing = get_deficiency_record(record_id)
    if not existing:
        raise ValueError("记录不存在")
    fields: Dict[str, Any] = {}
    for key in (
        "linked_company_project_id",
        "linked_project_id",
        "registration_country",
        "registration_category",
        "opinion_text",
        "priority",
        "remediation_plan",
        "deficiency_type",
        "deficiency_source",
        "train_status",
        "status",
    ):
        if key in data:
            fields[key] = data[key]
    if "issued_on" in data:
        issued = _parse_date(data.get("issued_on"))
        if not issued:
            raise ValueError("issued_on 无效")
        fields["issued_on"] = issued
    if "remediation_status" in data or "completed_on" in data:
        rem_status = str(
            data.get("remediation_status")
            if "remediation_status" in data
            else existing.get("remediation_status")
            or "open"
        ).strip() or "open"
        if rem_status not in STATUS_VALUES:
            rem_status = "open"
        fields["remediation_status"] = rem_status
        if rem_status == "done":
            completed = _parse_date(
                data.get("completed_on") if "completed_on" in data else existing.get("completed_on")
            )
            if not completed:
                completed = date.today()
            fields["completed_on"] = completed
        else:
            fields["completed_on"] = None
    if "priority" in fields and str(fields["priority"]) not in PRIORITY_VALUES:
        fields["priority"] = "medium"
    if "deficiency_type" in fields and str(fields["deficiency_type"]) not in TYPE_VALUES:
        fields["deficiency_type"] = "registration_review"
    if "train_status" in fields and str(fields["train_status"]) not in TRAIN_STATUS_VALUES:
        fields["train_status"] = "not_trained"
    # 内容变更后标记待重训
    content_keys = {"opinion_text", "remediation_plan", "registration_country", "registration_category"}
    if content_keys & set(fields.keys()) and "train_status" not in data:
        if str(existing.get("train_status") or "") == "trained":
            fields["train_status"] = "stale"

    if not fields:
        return
    cols = ", ".join(f"{k}=%s" for k in fields.keys())
    vals = list(fields.values())
    vals.append(int(record_id))
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"UPDATE deficiency_records SET {cols} WHERE id=%s", vals)
        conn.commit()
    finally:
        conn.close()


def get_deficiency_record(record_id: int) -> Optional[Dict[str, Any]]:
    ensure_deficiency_tables()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM deficiency_records WHERE id=%s", (int(record_id),))
            row = cur.fetchone()
            return _row_to_dict(row) if row else None
    finally:
        conn.close()


def list_deficiency_records(
    collection: str,
    *,
    remediation_status: str = "",
    deficiency_type: str = "",
    registration_country: str = "",
    registration_category: str = "",
    include_archived: bool = False,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    ensure_deficiency_tables()
    sql = "SELECT * FROM deficiency_records WHERE collection=%s"
    params: List[Any] = [collection]
    if not include_archived:
        sql += " AND status='active'"
    if remediation_status in STATUS_VALUES:
        sql += " AND remediation_status=%s"
        params.append(remediation_status)
    if deficiency_type in TYPE_VALUES:
        sql += " AND deficiency_type=%s"
        params.append(deficiency_type)
    if registration_country:
        sql += " AND registration_country=%s"
        params.append(registration_country)
    if registration_category:
        sql += " AND registration_category=%s"
        params.append(registration_category)
    sql += " ORDER BY issued_on DESC, id DESC LIMIT %s"
    params.append(max(1, min(int(limit or 200), 500)))
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return [_row_to_dict(r) for r in (cur.fetchall() or [])]
    finally:
        conn.close()


def list_injectable_deficiency_records(
    collection: str,
    *,
    registration_country: str,
    registration_category: str,
    as_of_date: date,
) -> List[Dict[str, Any]]:
    """注入用：active + 国家/类别一致 + issued_on <= as_of；含 open 与 done。"""
    ensure_deficiency_tables()
    country = (registration_country or "").strip()
    category = (registration_category or "").strip()
    if not country or not category:
        return []
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM deficiency_records
                WHERE collection=%s AND status='active'
                  AND registration_country=%s AND registration_category=%s
                  AND issued_on <= %s
                ORDER BY
                  CASE priority WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
                  issued_on DESC, id DESC
                LIMIT 50
                """,
                (collection, country, category, as_of_date),
            )
            return [_row_to_dict(r) for r in (cur.fetchall() or [])]
    finally:
        conn.close()


def add_deficiency_asset(
    record_id: int,
    *,
    role: str,
    display_name: str,
    storage_path: str = "",
    text_excerpt: str = "",
) -> int:
    ensure_deficiency_tables()
    role = (role or "").strip()
    if role not in ASSET_ROLES:
        raise ValueError(f"role 无效：{role}")
    name = (display_name or "").strip()
    if not name:
        raise ValueError("display_name 必填")
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO deficiency_assets (record_id, role, display_name, storage_path, text_excerpt)
                VALUES (%s,%s,%s,%s,%s)
                """,
                (int(record_id), role, name, storage_path or "", text_excerpt or ""),
            )
            aid = int(cur.lastrowid)
        conn.commit()
        # 文档变更 → stale
        update_deficiency_record(int(record_id), {"train_status": "stale"})
        return aid
    finally:
        conn.close()


def list_deficiency_assets(record_id: int) -> List[Dict[str, Any]]:
    ensure_deficiency_tables()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM deficiency_assets WHERE record_id=%s ORDER BY id ASC",
                (int(record_id),),
            )
            rows = cur.fetchall() or []
            out = []
            for r in rows:
                d = dict(r)
                if d.get("created_at") is not None:
                    d["created_at"] = str(d["created_at"])
                out.append(d)
            return out
    finally:
        conn.close()


def archive_deficiency_record(record_id: int) -> None:
    update_deficiency_record(int(record_id), {"status": "archived"})
