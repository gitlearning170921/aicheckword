# -*- coding: utf-8 -*-
"""发补记录持久化（MySQL）。"""
from __future__ import annotations

import json
import re
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
                    project_name VARCHAR(255) DEFAULT '',
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
                    import_batch_id VARCHAR(36) DEFAULT NULL,
                    excel_row_index INT DEFAULT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_def_coll_country_cat (collection, registration_country, registration_category),
                    INDEX idx_def_issued (collection, issued_on),
                    INDEX idx_def_status (collection, status, remediation_status)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            # 兼容旧表：补 project_name
            try:
                cur.execute(
                    "ALTER TABLE deficiency_records "
                    "ADD COLUMN project_name VARCHAR(255) DEFAULT '' AFTER linked_project_id"
                )
            except Exception:
                pass
            try:
                cur.execute(
                    "ALTER TABLE deficiency_records "
                    "ADD COLUMN import_batch_id VARCHAR(36) DEFAULT NULL AFTER status"
                )
            except Exception:
                pass
            try:
                cur.execute(
                    "ALTER TABLE deficiency_records "
                    "ADD COLUMN excel_row_index INT DEFAULT NULL AFTER import_batch_id"
                )
            except Exception:
                pass
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


def _parse_excel_row_index(val: Any) -> Optional[int]:
    """Excel 行号（>=1）；无效则 None（手工录入）。"""
    if val is None or val == "":
        return None
    try:
        n = int(val)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


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


def _normalize_record_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """校验并规范化单条发补字段，返回可入库字典；失败抛 ValueError。"""
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
    # 允许暂无国家/类别（未关联总览项目时可由 Excel 后补）；空值不参与下游注入匹配
    opinion = str(data.get("opinion_text") or "").strip()
    if not opinion:
        raise ValueError("opinion_text 必填")
    linked_pid = None
    if data.get("linked_project_id") not in (None, ""):
        linked_pid = int(data["linked_project_id"])
    project_name = str(data.get("project_name") or "").strip()
    return {
        "collection": str(data.get("collection") or "regulations").strip() or "regulations",
        "linked_company_project_id": str(data.get("linked_company_project_id") or "").strip() or None,
        "linked_project_id": linked_pid,
        "project_name": project_name,
        "registration_country": country,
        "registration_category": category,
        "opinion_text": opinion,
        "priority": priority,
        "remediation_plan": str(data.get("remediation_plan") or "").strip(),
        "issued_on": issued,
        "remediation_status": rem_status,
        "completed_on": completed,
        "deficiency_type": dtype,
        "deficiency_source": str(data.get("deficiency_source") or "").strip(),
        "train_status": "not_trained",
        "status": "active",
        "import_batch_id": str(data.get("import_batch_id") or "").strip() or None,
        "excel_row_index": _parse_excel_row_index(
            data.get("excel_row_index") if "excel_row_index" in data else data.get("_excel_row")
        ),
    }


def create_deficiency_record(data: Dict[str, Any]) -> int:
    ensure_deficiency_tables()
    row = _normalize_record_fields(data)
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO deficiency_records (
                    collection, linked_company_project_id, linked_project_id, project_name,
                    registration_country, registration_category,
                    opinion_text, priority, remediation_plan,
                    issued_on, remediation_status, completed_on,
                    deficiency_type, deficiency_source, train_status, status,
                    import_batch_id, excel_row_index
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    row["collection"],
                    row["linked_company_project_id"],
                    row["linked_project_id"],
                    row["project_name"],
                    row["registration_country"],
                    row["registration_category"],
                    row["opinion_text"],
                    row["priority"],
                    row["remediation_plan"],
                    row["issued_on"],
                    row["remediation_status"],
                    row["completed_on"],
                    row["deficiency_type"],
                    row["deficiency_source"],
                    row["train_status"],
                    row["status"],
                    row.get("import_batch_id"),
                    row.get("excel_row_index"),
                ),
            )
            rid = int(cur.lastrowid)
        conn.commit()
        return rid
    finally:
        conn.close()


def _compact_text(val: Any) -> str:
    return re.sub(r"\s+", "", str(val or "").strip().lower())


def _norm_opinion(val: Any) -> str:
    return re.sub(r"\s+", " ", str(val or "").strip().lower())


def deficiency_import_fingerprint(data: Dict[str, Any]) -> str:
    """导入幂等键：项目名 + 发补日期 + 类型 + 意见正文（跨次导入匹配更新用）。"""
    project = _compact_text(data.get("project_name") or data.get("linked_company_project_id"))
    issued = str(data.get("issued_on") or "")[:10]
    dtype = str(data.get("deficiency_type") or "registration_review").strip() or "registration_review"
    opinion = _norm_opinion(data.get("opinion_text"))
    return f"{project}|{issued}|{dtype}|{opinion}"


def _load_active_fingerprint_index(collection: str) -> Dict[str, int]:
    """collection 下 active 记录的 fingerprint -> 最新 id（同键多条取 id 最大）。"""
    ensure_deficiency_tables()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, project_name, linked_company_project_id, issued_on,
                       deficiency_type, opinion_text
                FROM deficiency_records
                WHERE collection=%s AND status='active'
                ORDER BY id ASC
                """,
                (str(collection or "regulations").strip() or "regulations",),
            )
            index: Dict[str, int] = {}
            for row in cur.fetchall() or []:
                d = dict(row) if not isinstance(row, dict) else row
                issued = d.get("issued_on")
                if hasattr(issued, "isoformat"):
                    issued = issued.isoformat()
                fp = deficiency_import_fingerprint(
                    {
                        "project_name": d.get("project_name"),
                        "linked_company_project_id": d.get("linked_company_project_id"),
                        "issued_on": issued,
                        "deficiency_type": d.get("deficiency_type"),
                        "opinion_text": d.get("opinion_text"),
                    }
                )
                if fp:
                    index[fp] = int(d["id"])
            return index
    finally:
        conn.close()


def create_deficiency_records_batch(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """批量入库；单条失败不影响其它。返回 created / failed 明细。"""
    ensure_deficiency_tables()
    created = 0
    failed: List[Dict[str, Any]] = []
    if not items:
        return {"created": 0, "failed": []}
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            sql = """
                INSERT INTO deficiency_records (
                    collection, linked_company_project_id, linked_project_id, project_name,
                    registration_country, registration_category,
                    opinion_text, priority, remediation_plan,
                    issued_on, remediation_status, completed_on,
                    deficiency_type, deficiency_source, train_status, status,
                    import_batch_id, excel_row_index
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """
            for idx, data in enumerate(items):
                excel_row = data.get("_excel_row")
                try:
                    row = _normalize_record_fields(data)
                    cur.execute(
                        sql,
                        (
                            row["collection"],
                            row["linked_company_project_id"],
                            row["linked_project_id"],
                            row["project_name"],
                            row["registration_country"],
                            row["registration_category"],
                            row["opinion_text"],
                            row["priority"],
                            row["remediation_plan"],
                            row["issued_on"],
                            row["remediation_status"],
                            row["completed_on"],
                            row["deficiency_type"],
                            row["deficiency_source"],
                            row["train_status"],
                            row["status"],
                    row.get("import_batch_id"),
                    row.get("excel_row_index"),
                        ),
                    )
                    created += 1
                except Exception as exc:
                    failed.append(
                        {
                            "index": idx,
                            "excelRow": excel_row,
                            "message": str(exc)[:300],
                        }
                    )
        conn.commit()
    finally:
        conn.close()
    return {"created": created, "failed": failed}



def find_duplicates_by_opinion(
    collection: str,
    opinion_text: str,
    *,
    exclude_id: Optional[int] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """按规范化意见正文查找 active 重复记录，并按 import_batch_id 汇总。"""
    ensure_deficiency_tables()
    opinion = _norm_opinion(opinion_text)
    if not opinion:
        return {"total": 0, "batches": [], "records": []}
    coll = str(collection or "regulations").strip() or "regulations"
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, project_name, linked_company_project_id, issued_on,
                       deficiency_type, opinion_text, import_batch_id, created_at
                FROM deficiency_records
                WHERE collection=%s AND status='active'
                ORDER BY id DESC
                LIMIT 5000
                """,
                (coll,),
            )
            rows = cur.fetchall() or []
    finally:
        conn.close()
    matched = []
    for r in rows:
        d = dict(r) if not isinstance(r, dict) else r
        if exclude_id is not None and int(d.get("id") or 0) == int(exclude_id):
            continue
        if _norm_opinion(d.get("opinion_text")) != opinion:
            continue
        issued = d.get("issued_on")
        if hasattr(issued, "isoformat"):
            issued = issued.isoformat()
        created = d.get("created_at")
        if hasattr(created, "isoformat"):
            created = created.isoformat()
        matched.append(
            {
                "id": int(d["id"]),
                "project_name": d.get("project_name") or "",
                "linked_company_project_id": d.get("linked_company_project_id") or "",
                "issued_on": str(issued or "")[:10],
                "deficiency_type": d.get("deficiency_type") or "",
                "import_batch_id": d.get("import_batch_id") or "",
                "created_at": str(created or ""),
            }
        )
        if len(matched) >= max(1, min(int(limit or 50), 200)):
            break
    batch_map: Dict[str, int] = {}
    for mrow in matched:
        key = str(mrow.get("import_batch_id") or "").strip() or "__manual__"
        batch_map[key] = batch_map.get(key, 0) + 1
    batches = []
    for bid, cnt in sorted(batch_map.items(), key=lambda x: (-x[1], x[0])):
        batches.append(
            {
                "importBatchId": "" if bid == "__manual__" else bid,
                "label": "手工录入/未标注批次" if bid == "__manual__" else ("导入批次 " + bid[:8] + "…"),
                "count": cnt,
            }
        )
    return {"total": len(matched), "batches": batches, "records": matched}



def upsert_deficiency_records_batch(
    items: List[Dict[str, Any]],
    *,
    match_index: Optional[Dict[str, int]] = None,
    grow_match_index: bool = False,
) -> Dict[str, Any]:
    """批量增量导入：指纹命中则更新，否则新增。

    match_index=None 时从库加载；传入时作为导入开始快照（同一次 Excel 分块共用）。
    grow_match_index=False（默认）时本批新增不进入判重索引，同文件相同键各插一条。
    """
    ensure_deficiency_tables()
    created = 0
    updated = 0
    failed: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []
    if not items:
        return {"created": 0, "updated": 0, "failed": [], "results": []}
    collection = str((items[0] or {}).get("collection") or "regulations").strip() or "regulations"
    if match_index is None:
        lookup: Dict[str, int] = _load_active_fingerprint_index(collection)
    else:
        lookup = {}
        for k, v in (match_index or {}).items():
            try:
                lookup[str(k)] = int(v)
            except (TypeError, ValueError):
                continue
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            insert_sql = """
                INSERT INTO deficiency_records (
                    collection, linked_company_project_id, linked_project_id, project_name,
                    registration_country, registration_category,
                    opinion_text, priority, remediation_plan,
                    issued_on, remediation_status, completed_on,
                    deficiency_type, deficiency_source, train_status, status,
                    import_batch_id, excel_row_index
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """
            update_sql = """
                UPDATE deficiency_records SET
                    linked_company_project_id=%s,
                    linked_project_id=%s,
                    project_name=%s,
                    registration_country=%s,
                    registration_category=%s,
                    opinion_text=%s,
                    priority=%s,
                    remediation_plan=%s,
                    issued_on=%s,
                    remediation_status=%s,
                    completed_on=%s,
                    deficiency_type=%s,
                    deficiency_source=%s,
                    train_status=CASE
                        WHEN train_status='trained' THEN 'stale'
                        ELSE train_status
                    END
                WHERE id=%s AND collection=%s AND status='active'
            """
            for idx, data in enumerate(items):
                excel_row = data.get("_excel_row")
                try:
                    row = _normalize_record_fields(data)
                    fp = deficiency_import_fingerprint(
                        {
                            "project_name": row.get("project_name"),
                            "linked_company_project_id": row.get("linked_company_project_id"),
                            "issued_on": row.get("issued_on"),
                            "deficiency_type": row.get("deficiency_type"),
                            "opinion_text": row.get("opinion_text"),
                        }
                    )
                    if not fp or fp.endswith("|||") or fp.count("|") < 3:
                        raise ValueError("无法生成导入指纹（项目/日期/意见不完整）")
                    existing_id = lookup.get(fp)
                    if existing_id:
                        cur.execute(
                            update_sql,
                            (
                                row["linked_company_project_id"],
                                row["linked_project_id"],
                                row["project_name"],
                                row["registration_country"],
                                row["registration_category"],
                                row["opinion_text"],
                                row["priority"],
                                row["remediation_plan"],
                                row["issued_on"],
                                row["remediation_status"],
                                row["completed_on"],
                                row["deficiency_type"],
                                row["deficiency_source"],
                                int(existing_id),
                                row["collection"],
                            ),
                        )
                        updated += 1
                        results.append(
                            {
                                "index": idx,
                                "action": "updated",
                                "id": int(existing_id),
                                "excelRow": excel_row,
                            }
                        )
                    else:
                        cur.execute(
                            insert_sql,
                            (
                                row["collection"],
                                row["linked_company_project_id"],
                                row["linked_project_id"],
                                row["project_name"],
                                row["registration_country"],
                                row["registration_category"],
                                row["opinion_text"],
                                row["priority"],
                                row["remediation_plan"],
                                row["issued_on"],
                                row["remediation_status"],
                                row["completed_on"],
                                row["deficiency_type"],
                                row["deficiency_source"],
                                row["train_status"],
                                row["status"],
                    row.get("import_batch_id"),
                    row.get("excel_row_index"),
                            ),
                        )
                        new_id = int(cur.lastrowid)
                        created += 1
                        if grow_match_index:
                            lookup[fp] = new_id
                        results.append(
                            {
                                "index": idx,
                                "action": "created",
                                "id": new_id,
                                "excelRow": excel_row,
                            }
                        )
                except Exception as exc:
                    failed.append(
                        {
                            "index": idx,
                            "excelRow": excel_row,
                            "message": str(exc)[:300],
                        }
                    )
        conn.commit()
    finally:
        conn.close()
    return {"created": created, "updated": updated, "failed": failed, "results": results}



def update_deficiency_record(record_id: int, data: Dict[str, Any]) -> None:
    ensure_deficiency_tables()
    existing = get_deficiency_record(record_id)
    if not existing:
        raise ValueError("记录不存在")
    fields: Dict[str, Any] = {}
    for key in (
        "linked_company_project_id",
        "linked_project_id",
        "project_name",
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
    # 对齐文控：有 Excel 行号的按导入顺序；手工录入排后
    sql += (
        " ORDER BY "
        "CASE WHEN excel_row_index IS NULL THEN 1 ELSE 0 END ASC, "
        "CASE WHEN excel_row_index IS NOT NULL THEN created_at END ASC, "
        "excel_row_index ASC, "
        "id ASC "
        "LIMIT %s"
    )
    params.append(max(1, min(int(limit or 200), 2000)))
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
